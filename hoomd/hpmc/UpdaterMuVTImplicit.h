// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef __UPDATER_MUVT_IMPLICIT_H__
#define __UPDATER_MUVT_IMPLICIT_H__

#include "UpdaterMuVT.h"
#include "IntegratorHPMCMonoImplicit.h"
#include "ComputeFreeVolume.h"
#include "Moves.h"

#include <random>

#ifndef NVCC
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#endif

namespace hpmc
{

/*!
 * This class implements an Updater for simulations in the grand-canonical ensemble (mu-V-T)
 * with depletant cluster moves
 *
 * for pure GC moves see Vink and Horbach JCP 2004
 * Bolhuis Frenkel JCP 1994, Biben/Hansen J. Phys. Cond. Mat. 1996
 */
template<class Shape, class Integrator>
class UpdaterMuVTImplicit : public UpdaterMuVT<Shape>
    {
    public:
        //! Constructor
        UpdaterMuVTImplicit(std::shared_ptr<SystemDefinition> sysdef,
            std::shared_ptr<Integrator > mc_implicit,
            unsigned int seed,
            unsigned int npartition);

    protected:
        std::poisson_distribution<unsigned int> m_poisson;   //!< Poisson distribution
        std::shared_ptr<Integrator > m_mc_implicit;   //!< The associated implicit depletants integrator

        /*! Check for overlaps in the new configuration
         * \param timestep  time step
         * \param type Type of particle to test
         * \param pos Position of fictitious particle
         * \param orientation Orientation of particle
         * \param lnboltzmann Log of Boltzmann weight of insertion attempt (return value)
         * \returns True if boltzmann weight is non-zero
         */
        virtual bool tryInsertParticle(unsigned int timestep, unsigned int type, vec3<Scalar> pos,
            quat<Scalar> orientation, Scalar &lnboltzmann);

        /*! Remove particle and try to insert depletants
            \param timestep  time step
            \param tag Tag of particle being removed
            \param lnboltzmann Log of Boltzmann weight of insertion attempt (return value)
            \returns True if boltzmann weight is non-zero
         */
        virtual bool tryRemoveParticle(unsigned int timestep, unsigned int tag, Scalar &lnboltzmann);

        /*! Try switching particle type
         * \param timestep  time step
         * \param tag Tag of particle that is considered for switching types
         * \param newtype New type of particle
         * \param lnboltzmann Log of Boltzmann weight (return value)
         * \returns True if boltzmann weight is non-zero
         *
         * \note The method has to check that getNGlobal() > 0, otherwise tag is invalid
         */
        virtual bool trySwitchType(unsigned int timestep, unsigned int tag, unsigned newtype, Scalar &lnboltzmann);

        /*! Rescale box to new dimensions and scale particles
         * \param timestep current timestep
         * \param old_box the old BoxDim
         * \param new_box the new BoxDim
         * \param extra_ndof Extra degrees of freedom added (depletants)
         * \param lnboltzmann Exponent of Boltzmann factor (-deltaE)
         * \returns true if box resize could be performed
         */
        virtual bool boxResizeAndScale(unsigned int timestep, const BoxDim old_box, const BoxDim new_box,
            unsigned int &extra_ndof, Scalar &lnboltzmann);

        /*! Try inserting depletants into space created by changing a particle type
         * \param timestep  time step
         * \param n_insert Number of depletants to insert
         * \param delta Sphere diameter
         * \param tag Particle that is replaced
         * \param new_type New type of particle (ignored, if ignore==True)
         * \param n_trial Number of insertion trials per depletant
         * \param lnboltzmann Log of Boltzmann factor for insertion (return value)
         * \returns True if Boltzmann factor is non-zero
         */
        bool moveDepletantsInUpdatedRegion(unsigned int timestep, unsigned int n_insert, Scalar delta,
            unsigned int tag, unsigned int new_type, unsigned int n_trial, Scalar &lnboltzmann);

        /*! Insert depletants into such that they overlap with a particle of given tag
         * \param timestep time step
         * \param n_insert Number of depletants to insert
         * \param delta Sphere diameter
         * \param tag Tag of the particle depletants must overlap with
         * \param n_trial Number of insertion trials per depletant
         * \param lnboltzmann Log of Boltzmann factor for insertion (return value)
         * \param need_overlap_shape If true, successful insertions need to overlap with shape at old position
         * \returns True if Boltzmann factor is non-zero
         */
        bool moveDepletantsIntoOldPosition(unsigned int timestep, unsigned int n_insert, Scalar delta, unsigned int tag,
            unsigned int n_trial, Scalar &lnboltzmann, bool need_overlap_shape);

        /*! Insert depletants such that they overlap with a fictitious particle at a specified position
         * \param timestep time step
         * \param n_insert Number of depletants to insert
         * \param delta Sphere diameter
         * \param pos Position of inserted particle
         * \param orientation Orientationof inserted particle
         * \param type Type of inserted particle
         * \param n_trial Number of insertion trials per depletant
         * \param lnboltzmann Log of Boltzmann factor for insertion (return value)
         * \returns True if Boltzmann factor is non-zero
         */
        bool moveDepletantsIntoNewPosition(unsigned int timestep, unsigned int n_insert, Scalar delta, vec3<Scalar> pos, quat<Scalar> orientation,
            unsigned int type, unsigned int n_trial, Scalar &lnboltzmann);

        /*! Count overlapping depletants due to insertion of a fictitious particle
         * \param timestep time step
         * \param n_insert Number of depletants in circumsphere
         * \param delta Sphere diameter
         * \param pos Position of new particle
         * \param orientation Orientation of new particle
         * \param type Type of new particle (ignored, if ignore==True)
         * \param n_free Depletants that were free in old configuration
         * \returns Number of overlapping depletants
         */
        unsigned int countDepletantOverlapsInNewPosition(unsigned int timestep, unsigned int n_insert, Scalar delta,
            vec3<Scalar>pos, quat<Scalar> orientation, unsigned int type, unsigned int &n_free);

        /*! Count overlapping depletants in a sphere of diameter delta
         * \param timestep time step
         * \param n_insert Number of depletants in circumsphere
         * \param delta Sphere diameter
         * \param pos Center of sphere
         * \returns Number of overlapping depletants
         */
        unsigned int countDepletantOverlaps(unsigned int timestep, unsigned int n_insert, Scalar delta, vec3<Scalar>pos);


        //! Get the random number of depletants
        virtual unsigned int getNumDepletants(unsigned int timestep, Scalar V, bool local);

    };

//! Export the UpdaterMuVT class to python
/*! \param name Name of the class in the exported python module
    \tparam Shape An instantiation of UpdaterMuVTImplicit<Shape,Integrator> will be exported
*/
template < class Shape, class Integrator >
void export_UpdaterMuVTImplicit(pybind11::module& m, const std::string& name)
    {
    pybind11::class_< UpdaterMuVTImplicit<Shape, Integrator>, std::shared_ptr< UpdaterMuVTImplicit<Shape, Integrator> > >(m, name.c_str(),
          pybind11::base<UpdaterMuVT<Shape> >())
          .def(pybind11::init< std::shared_ptr<SystemDefinition>,
            std::shared_ptr< Integrator >, unsigned int, unsigned int>())
          ;
    }

/*! Constructor
    \param sysdef The system definition
    \param mc_implicit The HPMC integrator
    \param seed RNG seed
    \param npartition How many partitions to use in parallel for Gibbs ensemble (n=1 == grand canonical)
*/
template<class Shape, class Integrator>
UpdaterMuVTImplicit<Shape, Integrator>::UpdaterMuVTImplicit(std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<Integrator> mc_implicit,
    unsigned int seed,
    unsigned int npartition)
    : UpdaterMuVT<Shape>(sysdef, mc_implicit,seed,npartition), m_mc_implicit(mc_implicit)
    {
    }

template<class Shape, class Integrator>
bool UpdaterMuVTImplicit<Shape,Integrator>::tryInsertParticle(unsigned int timestep, unsigned int type, vec3<Scalar> pos, quat<Scalar> orientation, Scalar &lnboltzmann)
    {
    // check overlaps with colloid particles first
    lnboltzmann = Scalar(0.0);
    Scalar lnb(0.0);
    bool nonzero = UpdaterMuVT<Shape>::tryInsertParticle(timestep, type, pos, orientation, lnb);
    if (nonzero)
        {
        lnboltzmann += lnb;
        }

    // Depletant type
    unsigned int type_d = m_mc_implicit->getDepletantType();

    // Depletant and colloid diameter
    Scalar d_dep, d_colloid;
        {
        const std::vector<typename Shape::param_type, managed_allocator<typename Shape::param_type> > & params = this->m_mc->getParams();
        quat<Scalar> o;
        Shape tmp(o, params[type_d]);
        d_dep = tmp.getCircumsphereDiameter();
        Shape shape(o, params[type]);
        d_colloid = shape.getCircumsphereDiameter();
        }

    // test sphere diameter and volume
    Scalar delta = d_dep + d_colloid;
    Scalar V = Scalar(M_PI/6.0)*delta*delta*delta;


    unsigned int n_overlap = 0;
    #ifdef ENABLE_MPI
    // number of depletants to insert
    unsigned int n_insert = 0;

    if (this->m_gibbs)
        {
        // perform cluster move
        if (nonzero)
            {
            // generate random depletant number
            unsigned int n_dep = getNumDepletants(timestep, V, false);

            unsigned int tmp = 0;

            // count depletants overlapping with new config (but ignore overlap in old one)
            n_overlap = countDepletantOverlapsInNewPosition(timestep, n_dep, delta, pos, orientation, type, tmp);

            lnb = Scalar(0.0);

            // try inserting depletants in old configuration (compute configurational bias weight factor)
            if (moveDepletantsIntoNewPosition(timestep, n_overlap, delta, pos, orientation, type, m_mc_implicit->getNumTrials(), lnb))
                {
                lnboltzmann -= lnb;
                }
            else
                {
                nonzero = false;
                }

            }

        unsigned int other = this->m_gibbs_other;

        if (this->m_exec_conf->getRank() == 0)
            {
            MPI_Request req[2];
            MPI_Status status[2];
            MPI_Isend(&n_overlap, 1, MPI_UNSIGNED, other, 0, this->m_exec_conf->getHOOMDWorldMPICommunicator(), &req[0]);
            MPI_Irecv(&n_insert, 1, MPI_UNSIGNED, other, 0, this->m_exec_conf->getHOOMDWorldMPICommunicator(), &req[1]);
            MPI_Waitall(2, req, status);
            }
        if (this->m_comm)
            {
            bcast(n_insert, 0, this->m_exec_conf->getMPICommunicator());
            }

        // if we have to insert depletants in addition, reject
        if (n_insert)
            {
            nonzero = false;
            }
        }
    else
    #endif
        {
        if (nonzero)
            {
            // generate random depletant number
            unsigned int n_dep = getNumDepletants(timestep, V, false);

            // count depletants overlapping with new config (but ignore overlap in old one)
            unsigned int n_free;
            n_overlap = countDepletantOverlapsInNewPosition(timestep, n_dep, delta, pos, orientation, type, n_free);
            nonzero = !n_overlap;
            }
        }

    return nonzero;
    }

template<class Shape, class Integrator>
bool UpdaterMuVTImplicit<Shape,Integrator>::trySwitchType(unsigned int timestep, unsigned int tag, unsigned int new_type,
    Scalar &lnboltzmann)
    {
    // check overlaps with colloid particles first
    lnboltzmann = Scalar(0.0);
    Scalar lnb(0.0);
    bool nonzero = UpdaterMuVT<Shape>::trySwitchType(timestep, tag, new_type, lnb);

    if (nonzero)
        {
        lnboltzmann += lnb;
        }

    #ifdef ENABLE_MPI
    quat<Scalar> orientation(this->m_pdata->getOrientation(tag));

    // getPosition() takes into account grid shift, correct for that
    Scalar3 p = this->m_pdata->getPosition(tag)+this->m_pdata->getOrigin();
    int3 tmp = make_int3(0,0,0);
    this->m_pdata->getGlobalBox().wrap(p,tmp);
    vec3<Scalar> pos(p);

    // Depletant type
    unsigned int type_d = m_mc_implicit->getDepletantType();

    // old type
    unsigned int type = this->m_pdata->getType(tag);

    // Depletant and colloid diameter
    Scalar d_dep, d_colloid, d_colloid_old;
        {
        const std::vector<typename Shape::param_type, managed_allocator<typename Shape::param_type> >&  params = this->m_mc->getParams();
        quat<Scalar> o;
        Shape tmp(o, params[type_d]);
        d_dep = tmp.getCircumsphereDiameter();

        Shape shape(o, params[new_type]);
        d_colloid = shape.getCircumsphereDiameter();

        Shape shape_old(o, params[type]);
        d_colloid_old = shape_old.getCircumsphereDiameter();
        }

    // test sphere diameter and volume
    Scalar delta = d_dep + d_colloid;
    Scalar delta_old = d_dep + d_colloid_old;
    Scalar V = Scalar(M_PI/6.0)*delta*delta*delta;

    // generate random depletant number
    unsigned int n_dep = getNumDepletants(timestep, V, false);

    // count depletants overlapping with new config (but ignore overlaps with old one)
    unsigned int tmp_free = 0;
    unsigned int n_overlap = countDepletantOverlapsInNewPosition(timestep, n_dep, delta, pos, orientation, new_type, tmp_free);

    // reject if depletant overlap
    if (! this->m_gibbs && n_overlap)
        {
        // FIXME: need to apply GC acceptance criterion here for muVT
        nonzero = false;
        }

    // number of depletants to insert
    unsigned int n_insert = 0;

    if (this->m_gibbs)
        {
        if (nonzero)
            {
            lnb = Scalar(0.0);
            // compute configurational bias weight
            if (moveDepletantsIntoNewPosition(timestep, n_overlap, delta, pos, orientation, new_type, m_mc_implicit->getNumTrials(), lnb))
                {
                lnboltzmann -= lnb;
                }
            else
                {
                nonzero = false;
                }
            }
        unsigned int other = this->m_gibbs_other;

        if (this->m_exec_conf->getRank() == 0)
            {
            MPI_Request req[2];
            MPI_Status status[2];
            MPI_Isend(&n_overlap, 1, MPI_UNSIGNED, other, 0, this->m_exec_conf->getHOOMDWorldMPICommunicator(), &req[0]);
            MPI_Irecv(&n_insert, 1, MPI_UNSIGNED, other, 0, this->m_exec_conf->getHOOMDWorldMPICommunicator(), &req[1]);
            MPI_Waitall(2, req, status);
            }
        if (this->m_comm)
            {
            bcast(n_insert, 0, this->m_exec_conf->getMPICommunicator());
            }

        // try inserting depletants in new configuration
        if (moveDepletantsInUpdatedRegion(timestep, n_insert, delta_old, tag, new_type, m_mc_implicit->getNumTrials(), lnb))
            {
            lnboltzmann += lnb;
            }
        else
            {
            nonzero = false;
            }
        }
    #endif

    return nonzero;
    }

template<class Shape, class Integrator>
bool UpdaterMuVTImplicit<Shape,Integrator>::tryRemoveParticle(unsigned int timestep, unsigned int tag, Scalar &lnboltzmann)
    {
    // call parent class method
    lnboltzmann = Scalar(0.0);
    Scalar lnb(0.0);
    bool nonzero = UpdaterMuVT<Shape>::tryRemoveParticle(timestep, tag, lnb);

    if (nonzero)
        {
        lnboltzmann += lnb;
        }

    #ifdef ENABLE_MPI

    // number of depletants to insert
    unsigned int n_insert = 0;

    // zero overlapping depletants after removal
    unsigned int n_overlap = 0;

    if (this->m_gibbs)
        {
        unsigned int other = this->m_gibbs_other;

        if (this->m_exec_conf->getRank() == 0)
            {
            MPI_Request req[2];
            MPI_Status status[2];
            MPI_Isend(&n_overlap, 1, MPI_UNSIGNED, other, 0, this->m_exec_conf->getHOOMDWorldMPICommunicator(), &req[0]);
            MPI_Irecv(&n_insert, 1, MPI_UNSIGNED, other, 0, this->m_exec_conf->getHOOMDWorldMPICommunicator(), &req[1]);
            MPI_Waitall(2, req, status);
            }
        if (this->m_comm)
            {
            bcast(n_insert, 0, this->m_exec_conf->getMPICommunicator());
            }
        }
    #endif

    // only if the particle to be removed actually exists
    if (tag != UINT_MAX)
        {
        #ifdef ENABLE_MPI
        // old type
        unsigned int type = this->m_pdata->getType(tag);

        // Depletant type
        unsigned int type_d = m_mc_implicit->getDepletantType();

        // Depletant and colloid diameter
        Scalar d_dep, d_colloid_old;
            {
            const std::vector<typename Shape::param_type, managed_allocator<typename Shape::param_type> >& params = this->m_mc->getParams();
            quat<Scalar> o;
            Shape tmp(o, params[type_d]);
            d_dep = tmp.getCircumsphereDiameter();

            Shape shape_old(o, params[type]);
            d_colloid_old = shape_old.getCircumsphereDiameter();
            }

        if (this->m_gibbs)
            {
            // try inserting depletants in new configuration (where particle is removed)
            Scalar delta = d_dep + d_colloid_old;
            if (moveDepletantsIntoOldPosition(timestep, n_insert, delta, tag, m_mc_implicit->getNumTrials(), lnb, true))
                {
                lnboltzmann += lnb;
                }
            else
                {
                nonzero = false;
                }
            }
        else
        #endif
            {
            // just accept
            }
        } // end nglobal

    return nonzero;
    }

template<class Shape, class Integrator>
bool UpdaterMuVTImplicit<Shape,Integrator>::moveDepletantsInUpdatedRegion(unsigned int timestep, unsigned int n_insert,
    Scalar delta, unsigned int tag, unsigned int new_type, unsigned int n_trial, Scalar &lnboltzmann)
    {
    lnboltzmann = Scalar(0.0);

    unsigned int type_d = m_mc_implicit->getDepletantType();

    // getPosition() takes into account grid shift, correct for that
    Scalar3 p = this->m_pdata->getPosition(tag)+this->m_pdata->getOrigin();
    int3 tmp = make_int3(0,0,0);
    this->m_pdata->getGlobalBox().wrap(p,tmp);
    vec3<Scalar> pos(p);

    bool is_local = this->m_pdata->isParticleLocal(tag);

    // initialize another rng
    #ifdef ENABLE_MPI
    hoomd::detail::Saru rng(timestep, this->m_seed, this->m_exec_conf->getPartition() ^0x974762fa );
    #else
    hoomd::detail::Saru rng(timestep, this->m_seed, 0x974762fa );
    #endif

    // update the aabb tree
    const detail::AABBTree& aabb_tree = this->m_mc->buildAABBTree();

    // update the image list
    const std::vector<vec3<Scalar> >&image_list = this->m_mc->updateImageList();

    unsigned int zero = 0;

    if (is_local)
        {
        ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_orientation(this->m_pdata->getOrientationArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_tag(this->m_pdata->getTags(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_rtag(this->m_pdata->getRTags(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_overlaps(this->m_mc->getInteractionMatrix(), access_location::host, access_mode::read);

        const std::vector<typename Shape::param_type, managed_allocator<typename Shape::param_type> >& params = this->m_mc->getParams();

        const Index2D& overlap_idx = this->m_mc->getOverlapIndexer();

        // for every test depletant
        for (unsigned int k = 0; k < n_insert; ++k)
            {
            unsigned int n_success = 0;

            for (unsigned int itrial = 0; itrial < n_trial; ++itrial)
                {
                // draw a random vector in the excluded volume sphere of the particle to be inserted
                Scalar theta = rng.template s<Scalar>(Scalar(0.0),Scalar(2.0*M_PI));
                Scalar z = rng.template s<Scalar>(Scalar(-1.0),Scalar(1.0));

                // random normalized vector
                vec3<Scalar> n(fast::sqrt(Scalar(1.0)-z*z)*fast::cos(theta),fast::sqrt(Scalar(1.0)-z*z)*fast::sin(theta),z);

                // draw random radial coordinate in test sphere
                Scalar r3 = rng.template s<Scalar>();
                Scalar r = Scalar(0.5)*delta*powf(r3,1.0/3.0);

                // test depletant position
                vec3<Scalar> pos_test = pos+r*n;

                Shape shape_test(quat<Scalar>(), params[type_d]);
                if (shape_test.hasOrientation())
                    {
                    // if the depletant is anisotropic, generate orientation
                    shape_test.orientation = generateRandomOrientation(rng);
                    }

                bool overlap_old = false;

                detail::AABB aabb_test_local = shape_test.getAABB(vec3<Scalar>(0,0,0));

                unsigned int err_count = 0;
                // All image boxes (including the primary)
                const unsigned int n_images = image_list.size();
                for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                    {
                    vec3<Scalar> pos_test_image = pos_test + image_list[cur_image];
                    detail::AABB aabb = aabb_test_local;
                    aabb.translate(pos_test_image);

                    // stackless search
                    for (unsigned int cur_node_idx = 0; cur_node_idx < aabb_tree.getNumNodes(); cur_node_idx++)
                        {
                        if (detail::overlap(aabb_tree.getNodeAABB(cur_node_idx), aabb))
                            {
                            if (aabb_tree.isNodeLeaf(cur_node_idx))
                                {
                                for (unsigned int cur_p = 0; cur_p < aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                                    {
                                    // read in its position and orientation
                                    unsigned int j = aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                                    Scalar4 postype_j;
                                    Scalar4 orientation_j;

                                    // load the old position and orientation of the j particle
                                    postype_j = h_postype.data[j];
                                    orientation_j = h_orientation.data[j];

                                    if (h_tag.data[j] == tag)
                                        {
                                        // do not check against old particle configuration
                                        continue;
                                        }

                                    // put particles in coordinate system of particle i
                                    vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_test_image;

                                    unsigned int type = __scalar_as_int(postype_j.w);

                                    Shape shape_j(quat<Scalar>(orientation_j), params[type]);

                                    if (h_overlaps.data[overlap_idx(type,type_d)]
                                        && check_circumsphere_overlap(r_ij, shape_test, shape_j)
                                        && test_overlap(r_ij, shape_test, shape_j, err_count))
                                        {
                                        overlap_old = true;
                                        break;
                                        }
                                    }
                                }
                            }
                        else
                            {
                            // skip ahead
                            cur_node_idx += aabb_tree.getNodeSkip(cur_node_idx);
                            }

                        if (overlap_old)
                            break;
                        } // end loop over AABB nodes
                    if (overlap_old)
                        break;
                    } // end loop over images

                if (!overlap_old)
                    {
                    // resolve the updated particle tag
                    unsigned int j = h_rtag.data[tag];
                    assert(j < this->m_pdata->getN());

                    // load the old position and orientation of the updated particle
                    Scalar4 postype_j = h_postype.data[j];
                    Scalar4 orientation_j = h_orientation.data[j];

                    // see if it overlaps with depletant
                    for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                        {
                        vec3<Scalar> pos_test_image = pos_test + image_list[cur_image];
                        vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_test_image;

                        // old particle shape
                        unsigned int typ_j = __scalar_as_int(postype_j.w);
                        Shape shape_old(quat<Scalar>(orientation_j), params[typ_j]);

                        if (h_overlaps.data[overlap_idx(type_d, typ_j)]
                            && check_circumsphere_overlap(r_ij, shape_test, shape_old)
                            && test_overlap(r_ij, shape_test, shape_old, err_count))
                            {
                            // overlap with old particle configuration

                            // new particle shape
                            Shape shape_new(quat<Scalar>(orientation_j), params[new_type]);

                            if (!(h_overlaps.data[overlap_idx(type_d,new_type)]
                                && check_circumsphere_overlap(r_ij, shape_test, shape_new)
                                && test_overlap(r_ij, shape_test, shape_new, err_count)))
                                {
                                // no overlap wth new configuration
                                n_success++;
                                }
                            }
                        }
                    }
                } // end loop over insertion attempts

            if (n_success)
                {
                lnboltzmann += log((Scalar) n_success/(Scalar)n_trial);
                }
            else
                {
                zero = 1;
                }
            } // end loop over test depletants
        } // end is_local

    #ifdef ENABLE_MPI
    if (this->m_comm)
        {
        MPI_Allreduce(MPI_IN_PLACE, &lnboltzmann, 1, MPI_HOOMD_SCALAR, MPI_SUM, this->m_exec_conf->getMPICommunicator());
        MPI_Allreduce(MPI_IN_PLACE, &zero, 1, MPI_UNSIGNED, MPI_SUM, this->m_exec_conf->getMPICommunicator());
        }
    #endif

    return !zero;
    }

template<class Shape, class Integrator>
bool UpdaterMuVTImplicit<Shape,Integrator>::moveDepletantsIntoNewPosition(unsigned int timestep, unsigned int n_insert,
    Scalar delta, vec3<Scalar> pos, quat<Scalar> orientation, unsigned int type, unsigned int n_trial, Scalar &lnboltzmann)
    {
    lnboltzmann = Scalar(0.0);
    unsigned int zero = 0;

    unsigned int type_d = m_mc_implicit->getDepletantType();

    bool is_local = true;
    #ifdef ENABLE_MPI
    if (this->m_pdata->getDomainDecomposition())
        {
        const BoxDim& global_box = this->m_pdata->getGlobalBox();
        ArrayHandle<unsigned int> h_cart_ranks(this->m_pdata->getDomainDecomposition()->getCartRanks(), access_location::host, access_mode::read);
        is_local = this->m_exec_conf->getRank() == this->m_pdata->getDomainDecomposition()->placeParticle(global_box, vec_to_scalar3(pos), h_cart_ranks.data);
        }
    #endif

    // initialize another rng
    #ifdef ENABLE_MPI
    hoomd::detail::Saru rng(timestep, this->m_seed, this->m_exec_conf->getPartition() ^0x123b09af );
    #else
    hoomd::detail::Saru rng(timestep, this->m_seed, 0x123b09af );
    #endif

    // update the aabb tree
    const detail::AABBTree& aabb_tree = this->m_mc->buildAABBTree();

    // update the image list
    const std::vector<vec3<Scalar> >&image_list = this->m_mc->updateImageList();

    if (is_local)
        {
        ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_orientation(this->m_pdata->getOrientationArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_tag(this->m_pdata->getTags(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_overlaps(this->m_mc->getInteractionMatrix(), access_location::host, access_mode::read);

        const std::vector<typename Shape::param_type, managed_allocator<typename Shape::param_type> >& params = this->m_mc->getParams();

        const Index2D& overlap_idx = this->m_mc->getOverlapIndexer();

        // for every test depletant
        for (unsigned int k = 0; k < n_insert; ++k)
            {
            // Number of successfully reinsert depletants

            // we start with one because of super-detailed balance (we already inserted one overlapping depletant in the trial move)
            unsigned int n_success = 1;

            // Number of allowed insertion trials (those which overlap with colloid at old position)
            unsigned int n_overlap_shape = 1;

            for (unsigned int itrial = 0; itrial < n_trial; ++itrial)
                {
                // draw a random vector in the excluded volume sphere of the particle to be inserted
                Scalar theta = rng.template s<Scalar>(Scalar(0.0),Scalar(2.0*M_PI));
                Scalar z = rng.template s<Scalar>(Scalar(-1.0),Scalar(1.0));

                // random normalized vector
                vec3<Scalar> n(fast::sqrt(Scalar(1.0)-z*z)*fast::cos(theta),fast::sqrt(Scalar(1.0)-z*z)*fast::sin(theta),z);

                // draw random radial coordinate in test sphere
                Scalar r3 = rng.template s<Scalar>();
                Scalar r = Scalar(0.5)*delta*powf(r3,1.0/3.0);

                // test depletant position
                vec3<Scalar> pos_test = pos+r*n;

                Shape shape_test(quat<Scalar>(), params[type_d]);
                if (shape_test.hasOrientation())
                    {
                    // if the depletant is anisotropic, generate orientation
                    shape_test.orientation = generateRandomOrientation(rng);
                    }

                // check against overlap with old configuration
                bool overlap_old = false;

                detail::AABB aabb_test_local = shape_test.getAABB(vec3<Scalar>(0,0,0));

                unsigned int err_count = 0;
                // All image boxes (including the primary)
                const unsigned int n_images = image_list.size();
                for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                    {
                    vec3<Scalar> pos_test_image = pos_test + image_list[cur_image];
                    detail::AABB aabb = aabb_test_local;
                    aabb.translate(pos_test_image);

                    // stackless search
                    for (unsigned int cur_node_idx = 0; cur_node_idx < aabb_tree.getNumNodes(); cur_node_idx++)
                        {
                        if (detail::overlap(aabb_tree.getNodeAABB(cur_node_idx), aabb))
                            {
                            if (aabb_tree.isNodeLeaf(cur_node_idx))
                                {
                                for (unsigned int cur_p = 0; cur_p < aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                                    {
                                    // read in its position and orientation
                                    unsigned int j = aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                                    Scalar4 postype_j;
                                    Scalar4 orientation_j;

                                    // load the old position and orientation of the j particle
                                    postype_j = h_postype.data[j];
                                    orientation_j = h_orientation.data[j];

                                    // put particles in coordinate system of particle i
                                    vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_test_image;

                                    unsigned int typ_j = __scalar_as_int(postype_j.w);
                                    Shape shape_j(quat<Scalar>(orientation_j), params[typ_j]);

                                    if (h_overlaps.data[overlap_idx(type_d,typ_j)]
                                        && check_circumsphere_overlap(r_ij, shape_test, shape_j)
                                        && test_overlap(r_ij, shape_test, shape_j, err_count))
                                        {
                                        overlap_old = true;
                                        break;
                                        }
                                    }
                                }
                            }
                        else
                            {
                            // skip ahead
                            cur_node_idx += aabb_tree.getNodeSkip(cur_node_idx);
                            }
                        if (overlap_old)
                            break;
                        } // end loop over AABB nodes
                    if (overlap_old)
                        break;
                    } // end loop over images

                // checking the (0,0,0) image is sufficient
                Shape shape(orientation, params[type]);
                vec3<Scalar> r_ij = pos - pos_test;
                if (h_overlaps.data[overlap_idx(type, type_d)]
                    && check_circumsphere_overlap(r_ij, shape_test, shape)
                    && test_overlap(r_ij, shape_test, shape, err_count))
                    {
                    if (!overlap_old)
                        {
                        // insertion counts if it overlaps with inserted particle at new position, but not with other particles
                        n_success++;
                        }
                    n_overlap_shape++;
                    }
                } // end loop over insertion attempts

            if (n_success)
                {
                lnboltzmann += log((Scalar)n_success/(Scalar)n_overlap_shape);
                }
            else
                {
                zero = 1;
                }
            } // end loop over test depletants
        } // is_local

    #ifdef ENABLE_MPI
    if (this->m_comm)
        {
        MPI_Allreduce(MPI_IN_PLACE, &lnboltzmann, 1, MPI_HOOMD_SCALAR, MPI_SUM, this->m_exec_conf->getMPICommunicator());
        MPI_Allreduce(MPI_IN_PLACE, &zero, 1, MPI_UNSIGNED, MPI_SUM, this->m_exec_conf->getMPICommunicator());
        }
    #endif

    return !zero;
    }

template<class Shape, class Integrator>
bool UpdaterMuVTImplicit<Shape,Integrator>::moveDepletantsIntoOldPosition(unsigned int timestep, unsigned int n_insert,
    Scalar delta, unsigned int tag, unsigned int n_trial, Scalar &lnboltzmann, bool need_overlap_shape)
    {
    lnboltzmann = Scalar(0.0);

    unsigned int type_d = m_mc_implicit->getDepletantType();

    // getPosition() corrects for grid shift, add it back
    Scalar3 p = this->m_pdata->getPosition(tag)+this->m_pdata->getOrigin();
    int3 tmp = make_int3(0,0,0);
    this->m_pdata->getGlobalBox().wrap(p,tmp);
    vec3<Scalar> pos(p);

    bool is_local = this->m_pdata->isParticleLocal(tag);

    // initialize another rng
    #ifdef ENABLE_MPI
    hoomd::detail::Saru rng(timestep, this->m_seed, this->m_exec_conf->getPartition() ^0x64f123da );
    #else
    hoomd::detail::Saru rng(timestep, this->m_seed, 0x64f123da );
    #endif

    // update the aabb tree
    const detail::AABBTree& aabb_tree = this->m_mc->buildAABBTree();

    // update the image list
    const std::vector<vec3<Scalar> >&image_list = this->m_mc->updateImageList();

    unsigned int zero = 0;

    if (is_local)
        {
        ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_orientation(this->m_pdata->getOrientationArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_tag(this->m_pdata->getTags(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_rtag(this->m_pdata->getRTags(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_overlaps(this->m_mc->getInteractionMatrix(), access_location::host, access_mode::read);

        const std::vector<typename Shape::param_type, managed_allocator<typename Shape::param_type> > & params = this->m_mc->getParams();

        const Index2D & overlap_idx = this->m_mc->getOverlapIndexer();

        // for every test depletant
        for (unsigned int k = 0; k < n_insert; ++k)
            {
            // Number of successfully reinsert depletants
            unsigned int n_success = 0;

            // Number of allowed insertion trials (those which overlap with colloid at old position)
            unsigned int n_overlap_shape = 0;

            for (unsigned int itrial = 0; itrial < n_trial; ++itrial)
                {
                // draw a random vector in the excluded volume sphere of the particle to be inserted
                Scalar theta = rng.template s<Scalar>(Scalar(0.0),Scalar(2.0*M_PI));
                Scalar z = rng.template s<Scalar>(Scalar(-1.0),Scalar(1.0));

                // random normalized vector
                vec3<Scalar> n(fast::sqrt(Scalar(1.0)-z*z)*fast::cos(theta),fast::sqrt(Scalar(1.0)-z*z)*fast::sin(theta),z);

                // draw random radial coordinate in test sphere
                Scalar r3 = rng.template s<Scalar>();
                Scalar r = Scalar(0.5)*delta*powf(r3,1.0/3.0);

                // test depletant position
                vec3<Scalar> pos_test = pos+r*n;

                Shape shape_test(quat<Scalar>(), params[type_d]);
                if (shape_test.hasOrientation())
                    {
                    // if the depletant is anisotropic, generate orientation
                    shape_test.orientation = generateRandomOrientation(rng);
                    }

                bool overlap_old = false;
                bool overlap = false;

                detail::AABB aabb_test_local = shape_test.getAABB(vec3<Scalar>(0,0,0));

                unsigned int err_count = 0;
                // All image boxes (including the primary)
                const unsigned int n_images = image_list.size();
                for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                    {
                    vec3<Scalar> pos_test_image = pos_test + image_list[cur_image];
                    detail::AABB aabb = aabb_test_local;
                    aabb.translate(pos_test_image);

                    // stackless search
                    for (unsigned int cur_node_idx = 0; cur_node_idx < aabb_tree.getNumNodes(); cur_node_idx++)
                        {
                        if (detail::overlap(aabb_tree.getNodeAABB(cur_node_idx), aabb))
                            {
                            if (aabb_tree.isNodeLeaf(cur_node_idx))
                                {
                                for (unsigned int cur_p = 0; cur_p < aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                                    {
                                    // read in its position and orientation
                                    unsigned int j = aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                                    Scalar4 postype_j;
                                    Scalar4 orientation_j;

                                    // load the old position and orientation of the j particle
                                    postype_j = h_postype.data[j];
                                    orientation_j = h_orientation.data[j];

                                    if (h_tag.data[j] == tag)
                                        {
                                        // do not check against old particle configuration
                                        continue;
                                        }

                                    // put particles in coordinate system of particle i
                                    vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_test_image;

                                    unsigned int type = __scalar_as_int(postype_j.w);
                                    Shape shape_j(quat<Scalar>(orientation_j), params[type]);

                                    if (h_overlaps.data[overlap_idx(type_d, type)]
                                        && check_circumsphere_overlap(r_ij, shape_test, shape_j)
                                        && test_overlap(r_ij, shape_test, shape_j, err_count))
                                        {
                                        overlap_old = true;
                                        break;
                                        }
                                    }
                                }
                            }
                        else
                            {
                            // skip ahead
                            cur_node_idx += aabb_tree.getNodeSkip(cur_node_idx);
                            }

                        if (overlap_old)
                            break;
                        } // end loop over AABB nodes
                    if (overlap_old)
                        break;
                    } // end loop over images

                // resolve the updated particle tag
                unsigned int j = h_rtag.data[tag];
                assert(j < this->m_pdata->getN());

                // load the old position and orientation of the updated particle
                Scalar4 postype_j = h_postype.data[j];
                Scalar4 orientation_j = h_orientation.data[j];

                // see if it overlaps with depletant
                // only need to consider (0,0,0) image
                vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_test;

                unsigned int typ_j = __scalar_as_int(postype_j.w);
                Shape shape(quat<Scalar>(orientation_j), params[typ_j]);

                if (h_overlaps.data[overlap_idx(type_d, typ_j)]
                    && check_circumsphere_overlap(r_ij, shape_test, shape)
                    && test_overlap(r_ij, shape_test, shape, err_count))
                    {
                    overlap = true;
                    n_overlap_shape++;
                    }

                if (!overlap_old && (overlap || !need_overlap_shape))
                    {
                    // success if it overlaps with the particle identified by the tag
                    n_success++;
                    }
                } // end loop over insertion attempts

            if (n_success)
                {
                lnboltzmann += log((Scalar) n_success/(Scalar)n_overlap_shape);
                }
            else
                {
                zero = 1;
                }
            } // end loop over test depletants
        } // end is_local

    #ifdef ENABLE_MPI
    if (this->m_comm)
        {
        MPI_Allreduce(MPI_IN_PLACE, &lnboltzmann, 1, MPI_HOOMD_SCALAR, MPI_SUM, this->m_exec_conf->getMPICommunicator());
        MPI_Allreduce(MPI_IN_PLACE, &zero, 1, MPI_UNSIGNED, MPI_MAX, this->m_exec_conf->getMPICommunicator());
        }
    #endif

    return !zero;
    }

template<class Shape, class Integrator>
unsigned int UpdaterMuVTImplicit<Shape,Integrator>::countDepletantOverlapsInNewPosition(unsigned int timestep, unsigned int n_insert,
    Scalar delta, vec3<Scalar> pos, quat<Scalar> orientation, unsigned int type, unsigned int &n_free)
    {
    // number of depletants successfully inserted
    unsigned int n_overlap = 0;

    unsigned int type_d = m_mc_implicit->getDepletantType();

    bool is_local = true;
    #ifdef ENABLE_MPI
    if (this->m_pdata->getDomainDecomposition())
        {
        const BoxDim& global_box = this->m_pdata->getGlobalBox();
        ArrayHandle<unsigned int> h_cart_ranks(this->m_pdata->getDomainDecomposition()->getCartRanks(), access_location::host, access_mode::read);
        is_local = this->m_exec_conf->getRank() == this->m_pdata->getDomainDecomposition()->placeParticle(global_box, vec_to_scalar3(pos), h_cart_ranks.data);
        }
    #endif

    // initialize another rng
    #ifdef ENABLE_MPI
    hoomd::detail::Saru rng(timestep, this->m_seed, this->m_exec_conf->getPartition() ^0x1412459a );
    #else
    hoomd::detail::Saru rng(timestep, this->m_seed, 0x1412459a);
    #endif

    // update the aabb tree
    const detail::AABBTree& aabb_tree = this->m_mc->buildAABBTree();

    // update the image list
    const std::vector<vec3<Scalar> >&image_list = this->m_mc->updateImageList();

    n_free = 0;

    if (is_local)
        {
        ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_orientation(this->m_pdata->getOrientationArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_tag(this->m_pdata->getTags(), access_location::host, access_mode::read);

        const std::vector<typename Shape::param_type, managed_allocator<typename Shape::param_type> > & params = this->m_mc->getParams();

        ArrayHandle<unsigned int> h_overlaps(this->m_mc->getInteractionMatrix(), access_location::host, access_mode::read);
        const Index2D & overlap_idx = this->m_mc->getOverlapIndexer();

        // for every test depletant
        for (unsigned int k = 0; k < n_insert; ++k)
            {
            // draw a random vector in the excluded volume sphere of the particle to be inserted
            Scalar theta = rng.template s<Scalar>(Scalar(0.0),Scalar(2.0*M_PI));
            Scalar z = rng.template s<Scalar>(Scalar(-1.0),Scalar(1.0));

            // random normalized vector
            vec3<Scalar> n(fast::sqrt(Scalar(1.0)-z*z)*fast::cos(theta),fast::sqrt(Scalar(1.0)-z*z)*fast::sin(theta),z);

            // draw random radial coordinate in test sphere
            Scalar r3 = rng.template s<Scalar>();
            Scalar r = Scalar(0.5)*delta*powf(r3,1.0/3.0);

            // test depletant position
            vec3<Scalar> pos_test = pos+r*n;

            Shape shape_test(quat<Scalar>(), params[type_d]);
            if (shape_test.hasOrientation())
                {
                // if the depletant is anisotropic, generate orientation
                shape_test.orientation = generateRandomOrientation(rng);
                }

            // check against overlap with old configuration
            bool overlap_old = false;

            detail::AABB aabb_test_local = shape_test.getAABB(vec3<Scalar>(0,0,0));

            unsigned int err_count = 0;
            // All image boxes (including the primary)
            const unsigned int n_images = image_list.size();
            for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                {
                vec3<Scalar> pos_test_image = pos_test + image_list[cur_image];
                detail::AABB aabb = aabb_test_local;
                aabb.translate(pos_test_image);

                // stackless search
                for (unsigned int cur_node_idx = 0; cur_node_idx < aabb_tree.getNumNodes(); cur_node_idx++)
                    {
                    if (detail::overlap(aabb_tree.getNodeAABB(cur_node_idx), aabb))
                        {
                        if (aabb_tree.isNodeLeaf(cur_node_idx))
                            {
                            for (unsigned int cur_p = 0; cur_p < aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                                {
                                // read in its position and orientation
                                unsigned int j = aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                                Scalar4 postype_j;
                                Scalar4 orientation_j;

                                // load the old position and orientation of the j particle
                                postype_j = h_postype.data[j];
                                orientation_j = h_orientation.data[j];

                                // put particles in coordinate system of particle i
                                vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_test_image;

                                unsigned int typ_j = __scalar_as_int(postype_j.w);
                                Shape shape_j(quat<Scalar>(orientation_j), params[typ_j]);

                                if (h_overlaps.data[overlap_idx(type_d, typ_j)]
                                    && check_circumsphere_overlap(r_ij, shape_test, shape_j)
                                    && test_overlap(r_ij, shape_test, shape_j, err_count))
                                    {
                                    overlap_old = true;
                                    break;
                                    }
                                }
                            }
                        }
                    else
                        {
                        // skip ahead
                        cur_node_idx += aabb_tree.getNodeSkip(cur_node_idx);
                        }
                    if (overlap_old)
                        break;
                    } // end loop over AABB nodes
                if (overlap_old)
                    break;
                } // end loop over images

            if (! overlap_old)
                {
                n_free++;
                // see if it overlaps with inserted particle
                for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                    {
                    Shape shape(orientation, params[type]);

                    vec3<Scalar> pos_test_image = pos_test + image_list[cur_image];
                    vec3<Scalar> r_ij = pos - pos_test_image;
                    if (h_overlaps.data[overlap_idx(type_d, type)]
                        && check_circumsphere_overlap(r_ij, shape_test, shape)
                        && test_overlap(r_ij, shape_test, shape, err_count))
                        {
                        n_overlap++;
                        }
                    }
                }

            } // end loop over test depletants
        } // is_local

    #ifdef ENABLE_MPI
    if (this->m_comm)
        {
        MPI_Allreduce(MPI_IN_PLACE, &n_overlap, 1, MPI_UNSIGNED, MPI_SUM, this->m_exec_conf->getMPICommunicator());
        MPI_Allreduce(MPI_IN_PLACE, &n_free, 1, MPI_UNSIGNED, MPI_SUM, this->m_exec_conf->getMPICommunicator());
        }
    #endif

    return n_overlap;
    }

template<class Shape, class Integrator>
unsigned int UpdaterMuVTImplicit<Shape,Integrator>::countDepletantOverlaps(unsigned int timestep, unsigned int n_insert, Scalar delta, vec3<Scalar> pos)
    {
    // number of depletants successfully inserted
    unsigned int n_overlap = 0;

    unsigned int type_d = m_mc_implicit->getDepletantType();

    bool is_local = true;
    #ifdef ENABLE_MPI
    if (this->m_pdata->getDomainDecomposition())
        {
        const BoxDim& global_box = this->m_pdata->getGlobalBox();
        ArrayHandle<unsigned int> h_cart_ranks(this->m_pdata->getDomainDecomposition()->getCartRanks(), access_location::host, access_mode::read);
        is_local = this->m_exec_conf->getRank() == this->m_pdata->getDomainDecomposition()->placeParticle(global_box, vec_to_scalar3(pos), h_cart_ranks.data);
        }
    #endif

    // initialize another rng
    #ifdef ENABLE_MPI
    hoomd::detail::Saru rng(timestep, this->m_seed, this->m_exec_conf->getPartition() ^0x1412459a );
    #else
    hoomd::detail::Saru rng(timestep, this->m_seed, 0x1412459a);
    #endif

    // update the aabb tree
    const detail::AABBTree& aabb_tree = this->m_mc->buildAABBTree();

    // update the image list
    const std::vector<vec3<Scalar> >&image_list = this->m_mc->updateImageList();

    if (is_local)
        {
        ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_orientation(this->m_pdata->getOrientationArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_tag(this->m_pdata->getTags(), access_location::host, access_mode::read);

        const std::vector<typename Shape::param_type, managed_allocator<typename Shape::param_type> > & params = this->m_mc->getParams();

        ArrayHandle<unsigned int> h_overlaps(this->m_mc->getInteractionMatrix(), access_location::host, access_mode::read);
        const Index2D& overlap_idx = this->m_mc->getOverlapIndexer();

        // for every test depletant
        for (unsigned int k = 0; k < n_insert; ++k)
            {
            // draw a random vector in the excluded volume sphere of the particle to be inserted
            Scalar theta = rng.template s<Scalar>(Scalar(0.0),Scalar(2.0*M_PI));
            Scalar z = rng.template s<Scalar>(Scalar(-1.0),Scalar(1.0));

            // random normalized vector
            vec3<Scalar> n(fast::sqrt(Scalar(1.0)-z*z)*fast::cos(theta),fast::sqrt(Scalar(1.0)-z*z)*fast::sin(theta),z);

            // draw random radial coordinate in test sphere
            Scalar r3 = rng.template s<Scalar>();
            Scalar r = Scalar(0.5)*delta*powf(r3,1.0/3.0);

            // test depletant position
            vec3<Scalar> pos_test = pos+r*n;

            Shape shape_test(quat<Scalar>(), params[type_d]);
            if (shape_test.hasOrientation())
                {
                // if the depletant is anisotropic, generate orientation
                shape_test.orientation = generateRandomOrientation(rng);
                }

            // check against overlap with present configuration
            bool overlap = false;

            detail::AABB aabb_test_local = shape_test.getAABB(vec3<Scalar>(0,0,0));

            unsigned int err_count = 0;
            // All image boxes (including the primary)
            const unsigned int n_images = image_list.size();
            for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                {
                vec3<Scalar> pos_test_image = pos_test + image_list[cur_image];
                detail::AABB aabb = aabb_test_local;
                aabb.translate(pos_test_image);

                // stackless search
                for (unsigned int cur_node_idx = 0; cur_node_idx < aabb_tree.getNumNodes(); cur_node_idx++)
                    {
                    if (detail::overlap(aabb_tree.getNodeAABB(cur_node_idx), aabb))
                        {
                        if (aabb_tree.isNodeLeaf(cur_node_idx))
                            {
                            for (unsigned int cur_p = 0; cur_p < aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                                {
                                // read in its position and orientation
                                unsigned int j = aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                                Scalar4 postype_j;
                                Scalar4 orientation_j;

                                // load the old position and orientation of the j particle
                                postype_j = h_postype.data[j];
                                orientation_j = h_orientation.data[j];

                                // put particles in coordinate system of particle i
                                vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_test_image;

                                unsigned int typ_j = __scalar_as_int(postype_j.w);
                                Shape shape_j(quat<Scalar>(orientation_j), params[typ_j]);

                                if (h_overlaps.data[overlap_idx(typ_j, type_d)]
                                    && check_circumsphere_overlap(r_ij, shape_test, shape_j)
                                    && test_overlap(r_ij, shape_test, shape_j, err_count))
                                    {
                                    overlap = true;
                                    break;
                                    }
                                }
                            }
                        }
                    else
                        {
                        // skip ahead
                        cur_node_idx += aabb_tree.getNodeSkip(cur_node_idx);
                        }
                    if (overlap)
                        break;
                    } // end loop over AABB nodes
                if (overlap)
                    break;
                } // end loop over images

            if (overlap)
                {
                n_overlap++;
                }
            } // end loop over test depletants
        } // is_local

    #ifdef ENABLE_MPI
    if (this->m_comm)
        {
        MPI_Allreduce(MPI_IN_PLACE, &n_overlap, 1, MPI_UNSIGNED, MPI_SUM, this->m_exec_conf->getMPICommunicator());
        }
    #endif

    return n_overlap;
    }


//! Get a poisson-distributed number of depletants
template<class Shape, class Integrator>
unsigned int UpdaterMuVTImplicit<Shape,Integrator>::getNumDepletants(unsigned int timestep,  Scalar V, bool local)
    {
    // parameter for Poisson distribution
    Scalar lambda = this->m_mc_implicit->getDepletantDensity()*V;

    unsigned int n = 0;
    if (lambda>Scalar(0.0))
        {
        std::poisson_distribution<unsigned int> poisson =
            std::poisson_distribution<unsigned int>(lambda);

        // combine four seeds
        std::vector<unsigned int> seed_seq(4);
        seed_seq[0] = this->m_seed;
        seed_seq[1] = timestep;
        seed_seq[2] = local ? this->m_exec_conf->getRank() : 0;
        #ifdef ENABLE_MPI
        seed_seq[3] = this->m_exec_conf->getPartition();
        #else
        seed_seq[3] = 0;
        #endif
        std::seed_seq seed(seed_seq.begin(), seed_seq.end());

        // RNG for poisson distribution
        std::mt19937 rng_poisson(seed);

        n = poisson(rng_poisson);
        }
    return n;
    }

template<class Shape, class Integrator>
bool UpdaterMuVTImplicit<Shape,Integrator>::boxResizeAndScale(unsigned int timestep, const BoxDim old_box, const BoxDim new_box,
    unsigned int &extra_ndof, Scalar &lnboltzmann)
    {
    // call parent class method
    lnboltzmann = Scalar(0.0);
    bool result = UpdaterMuVT<Shape>::boxResizeAndScale(timestep, old_box, new_box, extra_ndof, lnboltzmann);

    if (result)
        {
        // update the aabb tree
        const detail::AABBTree& aabb_tree = this->m_mc->buildAABBTree();

        // update the image list
        const std::vector<vec3<Scalar> >&image_list = this->m_mc->updateImageList();

        if (this->m_prof) this->m_prof->push(this->m_exec_conf, "HPMC implicit volume move ");

        // access particle data and system box
        ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_orientation(this->m_pdata->getOrientationArray(), access_location::host, access_mode::read);

        // access parameters
        const std::vector<typename Shape::param_type, managed_allocator<typename Shape::param_type> > & params = this->m_mc->getParams();
        ArrayHandle<unsigned int> h_overlaps(this->m_mc->getInteractionMatrix(), access_location::host, access_mode::read);

        const Index2D& overlap_idx = this->m_mc->getOverlapIndexer();

        bool overlap = false;

        // get old local box
        BoxDim old_local_box = old_box;
        #ifdef ENABLE_MPI
        if (this->m_pdata->getDomainDecomposition())
            {
            old_local_box = this->m_pdata->getDomainDecomposition()->calculateLocalBox(old_box);
            }
        #endif

        // draw number from Poisson distribution (using old box)
        unsigned int n = getNumDepletants(timestep, old_local_box.getVolume(), true);

        // Depletant type
        unsigned int type_d = m_mc_implicit->getDepletantType();

        // place a cut-off on the result to avoid long-running loops
        unsigned int err_count = 0;

        // draw a random vector in the box
        #ifdef ENABLE_MPI
        hoomd::detail::Saru rng(this->m_seed, this->m_exec_conf->getNPartitions()*this->m_exec_conf->getRank()+this->m_exec_conf->getPartition(), timestep);
        #else
        hoomd::detail::Saru rng(this->m_seed, timestep);
        #endif

        uint3 dim = make_uint3(1,1,1);
        uint3 grid_pos = make_uint3(0,0,0);
        #ifdef ENABLE_MPI
        if (this->m_pdata->getDomainDecomposition())
            {
            Index3D didx = this->m_pdata->getDomainDecomposition()->getDomainIndexer();
            dim = make_uint3(didx.getW(), didx.getH(), didx.getD());
            grid_pos = this->m_pdata->getDomainDecomposition()->getGridPos();
            }
        #endif

        // for every test depletant
        for (unsigned int k = 0; k < n; ++k)
            {
            Scalar xrand = rng.template s<Scalar>();
            Scalar yrand = rng.template s<Scalar>();
            Scalar zrand = rng.template s<Scalar>();

            Scalar3 f_test = make_scalar3(xrand, yrand, zrand);
            f_test = (f_test + make_scalar3(grid_pos.x,grid_pos.y,grid_pos.z))/make_scalar3(dim.x,dim.y,dim.z);
            vec3<Scalar> pos_test = vec3<Scalar>(new_box.makeCoordinates(f_test));

            Shape shape_test(quat<Scalar>(), params[type_d]);
            if (shape_test.hasOrientation())
                {
                // if the depletant is anisotropic, generate orientation
                shape_test.orientation = generateRandomOrientation(rng);
                }

            // check against overlap in old box
            overlap=false;
            bool overlap_old = false;
            detail::AABB aabb_test_local = shape_test.getAABB(vec3<Scalar>(0,0,0));

            // All image boxes (including the primary)
            const unsigned int n_images = image_list.size();
            for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                {
                vec3<Scalar> pos_test_image = pos_test + image_list[cur_image];
                Scalar3 f = new_box.makeFraction(vec_to_scalar3(pos_test_image));
                vec3<Scalar> pos_test_image_old = vec3<Scalar>(old_box.makeCoordinates(f));

                // set up AABB in old coordinates
                detail::AABB aabb = aabb_test_local;
                aabb.translate(pos_test_image_old);

                // scale AABB to new coordinates (the AABB tree contains new coordinates)
                vec3<Scalar> lower, upper;
                lower = aabb.getLower();
                f = old_box.makeFraction(vec_to_scalar3(lower));
                lower = vec3<Scalar>(new_box.makeCoordinates(f));
                upper = aabb.getUpper();
                f = old_box.makeFraction(vec_to_scalar3(upper));
                upper = vec3<Scalar>(new_box.makeCoordinates(f));
                aabb = detail::AABB(lower,upper);

                // stackless search
                for (unsigned int cur_node_idx = 0; cur_node_idx < aabb_tree.getNumNodes(); cur_node_idx++)
                    {
                    if (detail::overlap(aabb_tree.getNodeAABB(cur_node_idx), aabb))
                        {
                        if (aabb_tree.isNodeLeaf(cur_node_idx))
                            {
                            for (unsigned int cur_p = 0; cur_p < aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                                {
                                // read in its position and orientation
                                unsigned int j = aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                                Scalar4 postype_j;
                                Scalar4 orientation_j;

                                // load the old position and orientation of the j particle
                                postype_j = h_postype.data[j];
                                orientation_j = h_orientation.data[j];

                                // compute the particle position scaled in the old box
                                f = new_box.makeFraction(make_scalar3(postype_j.x,postype_j.y,postype_j.z));
                                vec3<Scalar> pos_j_old(old_box.makeCoordinates(f));

                                // put particles in coordinate system of particle i
                                vec3<Scalar> r_ij = pos_j_old - pos_test_image_old;

                                unsigned int typ_j = __scalar_as_int(postype_j.w);
                                Shape shape_j(quat<Scalar>(orientation_j), params[typ_j]);

                                if (h_overlaps.data[overlap_idx(typ_j, type_d)]
                                    && check_circumsphere_overlap(r_ij, shape_test, shape_j)
                                    && test_overlap(r_ij, shape_test, shape_j, err_count))
                                    {
                                    overlap = true;

                                    // depletant is ignored for any overlap in the old configuration
                                    overlap_old = true;
                                    break;
                                    }
                                }
                            }
                        }
                    else
                        {
                        // skip ahead
                        cur_node_idx += aabb_tree.getNodeSkip(cur_node_idx);
                        }
                    if (overlap)
                        break;
                    }  // end loop over AABB nodes

                if (overlap)
                    break;

                } // end loop over images

            if (!overlap)
                {
                // depletant in free volume
                extra_ndof++;

                // check for overlap in new configuration

                // new depletant coordinates
                vec3<Scalar> pos_test(new_box.makeCoordinates(f_test));

                for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                    {
                    vec3<Scalar> pos_test_image = pos_test + image_list[cur_image];
                    detail::AABB aabb = aabb_test_local;
                    aabb.translate(pos_test_image);

                    // stackless search
                    for (unsigned int cur_node_idx = 0; cur_node_idx < aabb_tree.getNumNodes(); cur_node_idx++)
                       {
                       if (detail::overlap(aabb_tree.getNodeAABB(cur_node_idx), aabb))
                            {
                            if (aabb_tree.isNodeLeaf(cur_node_idx))
                                {
                                for (unsigned int cur_p = 0; cur_p < aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                                    {
                                    // read in its position and orientation
                                    unsigned int j = aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                                    Scalar4 postype_j;
                                    Scalar4 orientation_j;

                                    // load the new position and orientation of the j particle
                                    postype_j = h_postype.data[j];
                                    orientation_j = h_orientation.data[j];

                                    // put particles in coordinate system of particle i
                                    vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_test_image;

                                    unsigned int typ_j = __scalar_as_int(postype_j.w);
                                    Shape shape_j(quat<Scalar>(orientation_j), params[typ_j]);

                                    if (h_overlaps.data[overlap_idx(typ_j, type_d)]
                                         && check_circumsphere_overlap(r_ij, shape_test, shape_j)
                                         && test_overlap(r_ij, shape_test, shape_j, err_count))
                                         {
                                         overlap = true;
                                         break;
                                         }
                                     }
                                 }
                             }
                         else
                             {
                             // skip ahead
                             cur_node_idx += aabb_tree.getNodeSkip(cur_node_idx);
                             }

                         }  // end loop over AABB nodes

                    if (overlap)
                        break;
                    } // end loop over images
               } // end overlap check in new configuration

            if (overlap_old)
                {
                overlap = false;
                continue;
                }

            if (overlap)
                break;
            } // end loop over test depletants

        unsigned int overlap_count = overlap;

        #ifdef ENABLE_MPI
        if (this->m_comm)
            {
            MPI_Allreduce(MPI_IN_PLACE, &overlap_count, 1, MPI_UNSIGNED, MPI_SUM, this->m_exec_conf->getMPICommunicator());
            MPI_Allreduce(MPI_IN_PLACE, &extra_ndof, 1, MPI_UNSIGNED, MPI_SUM, this->m_exec_conf->getMPICommunicator());
            }
        #endif

        if (this->m_prof) this->m_prof->pop(this->m_exec_conf);

        result = !overlap_count;
        }
    return result;
    }
} // end namespace

#endif
