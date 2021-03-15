// inclusion guard
#ifndef _UPDATER_HPMC_CLUSTERS_IMPLICIT_
#define _UPDATER_HPMC_CLUSTERS_IMPLICIT_

/*! \file UpdaterBoxClusters.h
    \brief Declaration of UpdaterBoxClusters
*/

#include "UpdaterClusters.h"

#ifdef ENABLE_TBB
#include <tbb/parallel_for.h>
#endif

namespace hpmc
{

/*! A generic cluster move for integrators with implicit depletants.

    The algorithm has been simplified to not perform any detailed overlap
    checks with depletants, only circumsphere overlap checks. This choice does not affect
    correctness (only ergodicity). Therefore the cluster move should
    be combined with a local move, that is, IntegratorHPMCMono(Implicit).
*/

template< class Shape, class Integrator >
class UpdaterClustersImplicit : public UpdaterClusters<Shape>
    {
    public:
        //! Constructor
        /*! \param sysdef System definition
            \param mc_implicit Implicit depletants integrator
            \param seed PRNG seed
        */
        UpdaterClustersImplicit(std::shared_ptr<SystemDefinition> sysdef,
                        std::shared_ptr<Integrator> mc_implicit,
                        unsigned int seed)
            : UpdaterClusters<Shape>(sysdef, mc_implicit, seed), m_mc_implicit(mc_implicit)
            {
            this->m_exec_conf->msg->notice(5) << "Constructing UpdaterClustersImplicit" << std::endl;
            }

        //! Destructor
        virtual ~UpdaterClustersImplicit() {}

    protected:
        //! Get the interaction range
        virtual Scalar getNominalWidth()
            {
            Scalar nominal_width = m_mc_implicit->getMaxCoreDiameter();

            // access parameters
            auto& params = m_mc_implicit->getParams();

            if (m_mc_implicit->getDepletantDensity() > Scalar(0.0))
                {
                // add range of depletion interaction
                quat<Scalar> o;
                Shape tmp(o, params[m_mc_implicit->getDepletantType()]);
                nominal_width += tmp.getCircumsphereDiameter();
                }

            auto patch = m_mc_implicit->getPatchInteraction();
            if (patch) nominal_width = std::max(nominal_width, patch->getRCut());

            return nominal_width;
            }

        std::shared_ptr< Integrator> m_mc_implicit; //!< Implicit depletants integrator object

        //! Find interactions between particles due to overlap and depletion interaction
        /*! \param timestep Current time step
            \param pivot The current pivot point
            \param q The current line reflection axis
            \param line True if this is a line reflection
            \param map Map to lookup new tag from old tag
        */
        virtual void findInteractions(unsigned int timestep, vec3<Scalar> pivot, quat<Scalar> q, bool swap, bool line,
            const std::map<unsigned int, unsigned int>& map);

    };

template< class Shape, class Integrator >
void UpdaterClustersImplicit<Shape,Integrator>::findInteractions(unsigned int timestep, vec3<Scalar> pivot,
    quat<Scalar> q, bool swap, bool line, const std::map<unsigned int, unsigned int>& map)
    {
    // call base class method
    UpdaterClusters<Shape>::findInteractions(timestep, pivot, q, swap, line, map);

    if (this->m_prof) this->m_prof->push(this->m_exec_conf,"Interactions");

    // access parameters
    auto& params = m_mc_implicit->getParams();

    // Depletant diameter
    Scalar d_dep;
    unsigned int depletant_type = m_mc_implicit->getDepletantType();
        {
        // add range of depletion interaction
        quat<Scalar> o;
        Shape tmp(o, params[depletant_type]);
        d_dep = tmp.getCircumsphereDiameter();
        }

    // update the image list
    auto image_list = m_mc_implicit->updateImageList();

    // clear the local bond and rejection lists
    this->m_interact_old_old.clear();
    this->m_interact_new_old.clear();

    // cluster according to overlap of excluded volume shells
    // loop over local particles
    unsigned int nptl = this->m_pdata->getN();

    Index2D overlap_idx = m_mc_implicit->getOverlapIndexer();
    ArrayHandle<unsigned int> h_overlaps(m_mc_implicit->getInteractionMatrix(), access_location::host, access_mode::read);

    // access particle data
    ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(this->m_pdata->getOrientationArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(this->m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<int3> h_image(this->m_pdata->getImages(), access_location::host, access_mode::read);

    // test old configuration against itself

    #ifdef ENABLE_TBB
    tbb::parallel_for((unsigned int)0,this->m_n_particles_old, [&](unsigned int i)
    #else
    for (unsigned int i = 0; i < this->m_n_particles_old; ++i)
    #endif
        {
        unsigned int typ_i = __scalar_as_int(this->m_postype_backup[i].w);

        vec3<Scalar> pos_i(this->m_postype_backup[i]);
        quat<Scalar> orientation_i(this->m_orientation_backup[i]);

        Shape shape_i(orientation_i, params[typ_i]);
        Scalar r_excl_i = shape_i.getCircumsphereDiameter()/Scalar(2.0);

        detail::AABB aabb_local(vec3<Scalar>(0,0,0), Scalar(0.5)*shape_i.getCircumsphereDiameter()+d_dep);

        const unsigned int n_images = image_list.size();

        for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
            {
            vec3<Scalar> pos_i_image = pos_i + image_list[cur_image];

            detail::AABB aabb_i_image = aabb_local;
            aabb_i_image.translate(pos_i_image);

            // stackless search
            for (unsigned int cur_node_idx = 0; cur_node_idx < this->m_aabb_tree_old.getNumNodes(); cur_node_idx++)
                {
                if (detail::overlap(this->m_aabb_tree_old.getNodeAABB(cur_node_idx), aabb_i_image))
                    {
                    if (this->m_aabb_tree_old.isNodeLeaf(cur_node_idx))
                        {
                        for (unsigned int cur_p = 0; cur_p < this->m_aabb_tree_old.getNodeNumParticles(cur_node_idx); cur_p++)
                            {
                            // read in its position and orientation
                            unsigned int j = this->m_aabb_tree_old.getNodeParticle(cur_node_idx, cur_p);

                            if (this->m_tag_backup[i] == this->m_tag_backup[j] && cur_image == 0) continue;

                            // load the position and orientation of the j particle
                            vec3<Scalar> pos_j = vec3<Scalar>(this->m_postype_backup[j]);
                            unsigned int typ_j = __scalar_as_int(this->m_postype_backup[j].w);
                            Shape shape_j(quat<Scalar>(this->m_orientation_backup[j]), params[typ_j]);

                            // put particles in coordinate system of particle i
                            vec3<Scalar> r_ij = pos_j - pos_i_image;

                            // check for excluded volume sphere overlap
                            Scalar r_excl_j = shape_j.getCircumsphereDiameter()/Scalar(2.0);
                            Scalar RaRb = r_excl_i + r_excl_j + d_dep;
                            Scalar rsq_ij = dot(r_ij, r_ij);

                            if (h_overlaps.data[overlap_idx(typ_i,depletant_type)] &&
                                h_overlaps.data[overlap_idx(typ_j,depletant_type)] &&
                                rsq_ij <= RaRb*RaRb)
                                {
                                unsigned int new_tag_i;
                                    {
                                    auto it = map.find(this->m_tag_backup[i]);
                                    assert(it != map.end());
                                    new_tag_i = it->second;
                                    }
                                unsigned int new_tag_j;
                                    {
                                    auto it = map.find(this->m_tag_backup[j]);
                                    assert(it!=map.end());
                                    new_tag_j = it->second;
                                    }

                                this->m_interact_old_old.push_back(std::make_pair(new_tag_i,new_tag_j));

                                int3 delta_img = this->m_image_backup[i] - this->m_image_backup[j];
                                bool interacts_via_pbc = delta_img.x || delta_img.y || delta_img.z;
                                interacts_via_pbc |= cur_image != 0;

                                if (line && !swap && interacts_via_pbc)
                                    {
                                    // if interaction across PBC, reject cluster move
                                    this->m_local_reject.insert(new_tag_i);
                                    this->m_local_reject.insert(new_tag_j);
                                    }
                                } // end if overlap

                            } // end loop over AABB tree leaf
                        } // end is leaf
                    } // end if overlap
                else
                    {
                    // skip ahead
                    cur_node_idx += this->m_aabb_tree_old.getNodeSkip(cur_node_idx);
                    }

                } // end loop over nodes

            } // end loop over images

        } // end loop over old configuration
    #ifdef ENABLE_TBB
        );
    #endif

    // loop over new configuration
    #ifdef ENABLE_TBB
    tbb::parallel_for((unsigned int)0,nptl, [&](unsigned int i)
    #else
    for (unsigned int i = 0; i < nptl; ++i)
    #endif
        {
        unsigned int typ_i = __scalar_as_int(h_postype.data[i].w);

        vec3<Scalar> pos_i_new(h_postype.data[i]);
        quat<Scalar> orientation_i_new(h_orientation.data[i]);

        Shape shape_i(orientation_i_new, params[typ_i]);
        Scalar r_excl_i = shape_i.getCircumsphereDiameter()/Scalar(2.0);

        // All image boxes (including the primary)
        const unsigned int n_images = image_list.size();

        // check overlap of depletant-excluded volumes
        detail::AABB aabb_local(vec3<Scalar>(0,0,0), Scalar(0.5)*shape_i.getCircumsphereDiameter()+d_dep);

        // query new against old
        for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
            {
            vec3<Scalar> pos_i_image = pos_i_new + image_list[cur_image];

            detail::AABB aabb_i_image = aabb_local;
            aabb_i_image.translate(pos_i_image);

            // stackless search
            for (unsigned int cur_node_idx = 0; cur_node_idx < this->m_aabb_tree_old.getNumNodes(); cur_node_idx++)
                {
                if (detail::overlap(this->m_aabb_tree_old.getNodeAABB(cur_node_idx), aabb_i_image))
                    {
                    if (this->m_aabb_tree_old.isNodeLeaf(cur_node_idx))
                        {
                        for (unsigned int cur_p = 0; cur_p < this->m_aabb_tree_old.getNodeNumParticles(cur_node_idx); cur_p++)
                            {
                            // read in its position and orientation
                            unsigned int j = this->m_aabb_tree_old.getNodeParticle(cur_node_idx, cur_p);

                            unsigned int new_tag_j;
                                {
                                auto it = map.find(this->m_tag_backup[j]);
                                assert(it != map.end());
                                new_tag_j = it->second;
                                }

                            if (h_tag.data[i] == new_tag_j && cur_image == 0) continue;

                            vec3<Scalar> pos_j(this->m_postype_backup[j]);
                            unsigned int typ_j = __scalar_as_int(this->m_postype_backup[j].w);
                            Shape shape_j(quat<Scalar>(this->m_orientation_backup[j]), params[typ_j]);

                            // put particles in coordinate system of particle i
                            vec3<Scalar> r_ij = pos_j - pos_i_image;

                            // check for excluded volume sphere overlap
                            Scalar r_excl_j = shape_j.getCircumsphereDiameter()/Scalar(2.0);
                            Scalar RaRb = r_excl_i + r_excl_j + d_dep;
                            Scalar rsq_ij = dot(r_ij, r_ij);

                            if (h_overlaps.data[overlap_idx(typ_i,depletant_type)] &&
                                h_overlaps.data[overlap_idx(typ_j,depletant_type)] &&
                                rsq_ij <= RaRb*RaRb)
                                {
                                this->m_interact_new_old.push_back(std::make_pair(h_tag.data[i],new_tag_j));

                                int3 delta_img = h_image.data[i] - this->m_image_backup[j];
                                bool interacts_via_pbc = delta_img.x || delta_img.y || delta_img.z;
                                interacts_via_pbc |= cur_image != 0;

                                if (line && !swap && interacts_via_pbc)
                                    {
                                    // if interaction across PBC, reject cluster move
                                    this->m_local_reject.insert(h_tag.data[i]);
                                    this->m_local_reject.insert(new_tag_j);
                                    }
                                }
                            } // end loop over AABB tree leaf
                        } // end is leaf
                    } // end if overlap
                else
                    {
                    // skip ahead
                    cur_node_idx += this->m_aabb_tree_old.getNodeSkip(cur_node_idx);
                    }

                } // end loop over nodes

            } // end loop over images

        } // end loop over local particles
    #ifdef ENABLE_TBB
        );
    #endif

    // locality data in new configuration
    const detail::AABBTree& aabb_tree = m_mc_implicit->buildAABBTree();

    if (line && !swap)
        {
        // check if particles are interacting in the new configuration
        #ifdef ENABLE_TBB
        tbb::parallel_for((unsigned int)0,nptl, [&](unsigned int i)
        #else
        for (unsigned int i = 0; i < nptl; ++i)
        #endif
            {
            unsigned int typ_i = __scalar_as_int(h_postype.data[i].w);

            vec3<Scalar> pos_i_new(h_postype.data[i]);
            quat<Scalar> orientation_i_new(h_orientation.data[i]);

            Shape shape_i(orientation_i_new, params[typ_i]);
            Scalar r_excl_i = shape_i.getCircumsphereDiameter()/Scalar(2.0);

            // add depletant diameter to search radius
            detail::AABB aabb_i(pos_i_new, Scalar(0.5)*shape_i.getCircumsphereDiameter()+d_dep);

            // All image boxes (including the primary)
            const unsigned int n_images = image_list.size();

            // check against new AABB tree
            for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                {
                vec3<Scalar> pos_i_image = pos_i_new + image_list[cur_image];

                detail::AABB aabb_i_image = aabb_i;
                aabb_i_image.translate(image_list[cur_image]);

                // stackless search
                for (unsigned int cur_node_idx = 0; cur_node_idx < aabb_tree.getNumNodes(); cur_node_idx++)
                    {
                    if (detail::overlap(aabb_tree.getNodeAABB(cur_node_idx), aabb_i_image))
                        {
                        if (aabb_tree.isNodeLeaf(cur_node_idx))
                            {
                            for (unsigned int cur_p = 0; cur_p < aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                                {
                                // read in its position and orientation
                                unsigned int j = aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                                // no trivial bonds
                                if (h_tag.data[i] == h_tag.data[j] && cur_image == 0) continue;

                                // load the position and orientation of the j particle
                                vec3<Scalar> pos_j = vec3<Scalar>(h_postype.data[j]);
                                unsigned int typ_j = __scalar_as_int(h_postype.data[j].w);
                                Shape shape_j(quat<Scalar>(h_orientation.data[j]), params[typ_j]);

                                // put particles in coordinate system of particle i
                                vec3<Scalar> r_ij = pos_j - pos_i_image;

                                // check for circumsphere overlap
                                Scalar r_excl_j = shape_j.getCircumsphereDiameter()/Scalar(2.0);
                                Scalar RaRb = r_excl_i + r_excl_j + d_dep;
                                Scalar rsq_ij = dot(r_ij, r_ij);

                                if (h_overlaps.data[overlap_idx(typ_i,depletant_type)] &&
                                    h_overlaps.data[overlap_idx(typ_j,depletant_type)] &&
                                    rsq_ij <= RaRb*RaRb)
                                    {
                                    int3 delta_img = h_image.data[i] - h_image.data[j];
                                    bool interacts_via_pbc = delta_img.x || delta_img.y || delta_img.z;
                                    interacts_via_pbc |= cur_image != 0;

                                    if (interacts_via_pbc)
                                        {
                                        // add to list
                                        this->m_local_reject.insert(h_tag.data[i]);
                                        this->m_local_reject.insert(h_tag.data[j]);

                                        this->m_interact_new_new.insert(std::make_pair(h_tag.data[i],h_tag.data[j]));
                                        }
                                    } // end if overlap

                                } // end loop over AABB tree leaf
                            } // end is leaf
                        } // end if overlap
                    else
                        {
                        // skip ahead
                        cur_node_idx += aabb_tree.getNodeSkip(cur_node_idx);
                        }

                    } // end loop over nodes
                } // end loop over images
            } // end loop over local particles
        #ifdef ENABLE_TBB
            );
        #endif
        }

    if (this->m_prof) this->m_prof->pop(this->m_exec_conf);
    }



template < class Shape, class Integrator > void export_UpdaterClustersImplicit(pybind11::module& m, const std::string& name)
    {
    pybind11::class_< UpdaterClustersImplicit<Shape,Integrator>, std::shared_ptr< UpdaterClustersImplicit<Shape,Integrator> > >(m, name.c_str(), pybind11::base<UpdaterClusters<Shape> >())
          .def( pybind11::init< std::shared_ptr<SystemDefinition>,
                         std::shared_ptr< Integrator >,
                         unsigned int >())
    ;
    }

} // end namespace hpmc

#endif // _UPDATER_HPMC_CLUSTERS_IMPLICIT_
