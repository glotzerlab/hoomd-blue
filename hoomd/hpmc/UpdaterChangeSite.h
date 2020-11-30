// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// **********************
// This is a simple example code written for no function purpose other then to demonstrate the steps needed to write a
// c++ source code plugin for HOOMD-Blue. This example includes an example Updater class, but it can just as easily be
// replaced with a ForceCompute, Integrator, or any other C++ code at all.

// inclusion guard
#ifndef _CHANGE_SITE_UPDATER_H_
#define _CHANGE_SITE_UPDATER_H_

/*! \file ExampleUpdater.h
    \brief Declaration of ExampleUpdater
*/

// First, hoomd.h should be included

#include "hoomd/Updater.h"
#include "ExternalFieldLattice.h"
#include "IntegratorHPMCMono.h"
#include "hoomd/AABBTree.h"
#include "hoomd/RNGIdentifiers.h"

#ifndef NVCC
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#endif

namespace hpmc {
// (if you really don't want to include the whole hoomd.h, you can include individual files IF AND ONLY IF
// hoomd_config.h is included first)
// For example:
//
// #include "hoomd/Updater.h"

// Second, we need to declare the class. One could just as easily use any class in HOOMD as a template here, there are
// no restrictions on what a template can do

//! A nonsense particle updater written to demonstrate how to write a plugin
/*! This updater simply sets all of the particle's velocities to 0 when update() is called.
*/
template<class Shape>
class ChangeSiteUpdater : public Updater
    {
    public:
        //! Constructor
        ChangeSiteUpdater( std::shared_ptr<SystemDefinition> sysdef,
                            std::shared_ptr<ExternalFieldLattice<Shape> > externalLattice,
                            pybind11::list lattice_vectors,
                            std::shared_ptr<IntegratorHPMCMono<Shape> > mc,
                            unsigned int seed
                          ) : Updater(sysdef), m_externalLattice(externalLattice), m_mc(mc),
                              m_update_order(seed+m_exec_conf->getRank(), m_pdata->getN()), m_seed(seed)
            {
                setReferences(lattice_vectors);
            }

        //! Take one timestep forward
        virtual void update(unsigned int timestep);

        void setReferences(const pybind11::list& lv)
            {
            std::vector<Scalar3> lattice_vectors;
            std::vector<Scalar> lvbuffer;
            #ifdef ENABLE_MPI
            unsigned int lvsz = 0;

            if ( this->m_exec_conf->isRoot() )
                {
                hpmc::python_list_to_vector_scalar3(lv, lattice_vectors,3);
                lvsz = lattice_vectors.size();
                }
            if( this->m_pdata->getDomainDecomposition())
                {
                if(lvsz)
                    {
                    lvbuffer.resize(3*lvsz, 0.0);
                    for(size_t i = 0; i < lvsz; i++)
                        {
                        lvbuffer[3*i] = lattice_vectors[i].x;
                        lvbuffer[3*i+1] = lattice_vectors[i].y;
                        lvbuffer[3*i+2] = lattice_vectors[i].z;
                        }
                    }
                MPI_Bcast(&lvsz, 1, MPI_UNSIGNED, 0, m_exec_conf->getMPICommunicator());
                if(lvsz)
                    {
                    if(!lvbuffer.size())
                        lvbuffer.resize(3*lvsz, 0.0);
                    MPI_Bcast(&lvbuffer.front(), 3*lvsz, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
                    if(!lattice_vectors.size())
                        {
                        lattice_vectors.resize(lvsz, make_scalar3(0.0, 0.0, 0.0));
                        for(size_t i = 0; i < lvsz; i++)
                            {
                            lattice_vectors[i].x = lvbuffer[3*i];
                            lattice_vectors[i].y = lvbuffer[3*i+1];
                            lattice_vectors[i].z = lvbuffer[3*i+2];
                            }
                        }
                    }
                }

            #else
            hpmc::python_list_to_vector_scalar3(lv, lattice_vectors, 3);
            #endif

            if( lattice_vectors.size() )
                m_latticeVector.setReferences(lattice_vectors.begin(), lattice_vectors.end(), m_pdata, m_exec_conf);

            }

    protected:
                std::shared_ptr<ExternalFieldLattice<Shape> > m_externalLattice;
                LatticeReferenceList<Scalar3> m_latticeVector;
                std::shared_ptr<IntegratorHPMCMono<Shape> > m_mc;
                detail::UpdateOrder m_update_order;         //!< Update order
                unsigned int m_seed;         
    };

template <class Shape>
void ChangeSiteUpdater<Shape>::update(unsigned int timestep)
    {

    const BoxDim& box = m_pdata->getBox();

    #ifdef ENABLE_MPI
    // compute the width of the active region
    Scalar3 npd = box.getNearestPlaneDistance();
    Scalar3 ghost_fraction = m_mc->getMaxCoreDiameter() / npd;
    #endif

    // Shuffle the order of particles for this step
    m_update_order.resize(m_pdata->getN());
    m_update_order.shuffle(timestep);

    // update the AABB Tree
    detail::AABBTree aabb_tree = m_mc->buildAABBTree();
    std::vector<vec3<Scalar> > image_list = m_mc->updateImageList();

    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_r0(m_externalLattice->getReferenceLatticePositions(), access_location::host, access_mode::readwrite);
	ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_overlaps(m_mc->getInteractionMatrix(), access_location::host, access_mode::read);

    Index2D overlap_idx =  m_mc->getOverlapIndexer();


    for (unsigned int cur_particle = 0; cur_particle < m_pdata->getN(); cur_particle++)
        {
        unsigned int i = m_update_order[cur_particle];

        // read in the current position and orientation
        Scalar4 postype_i = h_postype.data[i];
        Scalar4 orientation_i = h_orientation.data[i];
        vec3<Scalar> pos_i = vec3<Scalar>(postype_i);
        vec3<Scalar> ref_i = vec3<Scalar>(h_r0.data[i]);

        #ifdef ENABLE_MPI
        if (m_comm)
            {
            // only move particle if active
            if (!isActive(make_scalar3(postype_i.x, postype_i.y, postype_i.z), box, ghost_fraction))
                continue;
            }
        #endif

        // make a trial move for i
        hoomd::RandomGenerator rng_i(hoomd::RNGIdentifier::UpdaterChangeSite, m_seed, i, m_exec_conf->getRank(), timestep);
        unsigned int indx = int(hoomd::UniformDistribution<Scalar>(0,m_latticeVector.getSize())(rng_i)); 

        int typ_i = __scalar_as_int(postype_i.w);
        Shape shape_i(quat<Scalar>(orientation_i), m_mc->getParams()[typ_i]);

        vec3<Scalar> dr =  vec3<Scalar>( m_latticeVector.getReference(indx) );

        pos_i += dr;
        ref_i += dr;

        #ifdef ENABLE_MPI
        if (m_comm)
            {
            // check if particle has moved into the ghost layer, and skip if it is
            if (!isActive(vec_to_scalar3(pos_i), box, ghost_fraction))
                continue;
            }
        #endif

        bool overlap=false;
        OverlapReal r_cut_patch = 0;

        // subtract minimum AABB extent from search radius
        OverlapReal R_query = std::max(shape_i.getCircumsphereDiameter()/OverlapReal(2.0),
            r_cut_patch-m_mc->getMinCoreDiameter()/(OverlapReal)2.0);
        detail::AABB aabb_i_local = detail::AABB(vec3<Scalar>(0,0,0),R_query);

        // check for overlaps with neighboring particle's positions (also calculate the new energy)
        // All image boxes (including the primary)
        const unsigned int n_images = image_list.size();
      
        for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
            {
            vec3<Scalar> pos_i_image = pos_i + image_list[cur_image];
            detail::AABB aabb = aabb_i_local;
            aabb.translate(pos_i_image);

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

                            // handle j==i situations
                            if ( j != i )
                                {
                                // load the position and orientation of the j particle
                                postype_j = h_postype.data[j];
                                orientation_j = h_orientation.data[j];
                                }
                            else
                                {
                                if (cur_image == 0)
                                    {
                                    // in the first image, skip i == j
                                    continue;
                                    }
                                else
                                    {
                                    // If this is particle i and we are in an outside image, use the translated position and orientation
                                    postype_j = make_scalar4(pos_i.x, pos_i.y, pos_i.z, postype_i.w);
                                    orientation_j = quat_to_scalar4(shape_i.orientation);
                                    }
                                }

                            // put particles in coordinate system of particle i
                            vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_i_image;

                            unsigned int typ_j = __scalar_as_int(postype_j.w);
                            Shape shape_j(quat<Scalar>(orientation_j), m_mc->getParams()[typ_j]);
                            unsigned int err_count = 0;

                            if ( h_overlaps.data[overlap_idx(typ_i, typ_j)]
                                && check_circumsphere_overlap(r_ij, shape_i, shape_j)
                                && test_overlap(r_ij, shape_i, shape_j, err_count))
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
                }  // end loop over AABB nodes

            if (overlap)
                break;
            } // end loop over images

            // If no overlaps and Metropolis criterion is met, accept
            // trial move and update positions  and/or orientations.
            if (!overlap)
                {
                // update the position of the particle in the tree for future updates
                detail::AABB aabb = aabb_i_local;
                aabb.translate(pos_i);
                aabb_tree.update(i, aabb);

                h_postype.data[i] = make_scalar4(pos_i.x,pos_i.y,pos_i.z,postype_i.w);
                h_r0.data[i] = make_scalar3(ref_i.x,ref_i.y,ref_i.z);
                }
        }

        ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);
        // wrap particles back into box
        int3 dummy = make_int3(0,0,0);
        for (unsigned int i = 0; i < m_pdata->getN(); i++)
            {
            box.wrap(h_postype.data[i], h_image.data[i]);
            box.wrap(h_postype.data[i], dummy);
            }
    }

//! Export the ExampleUpdater class to python
template <class Shape>
void export_UpdaterChangeSite(pybind11::module& m, std::string name)
    {
    using pybind11::class_;
   pybind11::class_<ChangeSiteUpdater<Shape>, std::shared_ptr<ChangeSiteUpdater<Shape> > >(m, name.c_str(), pybind11::base<Updater>())
   .def(pybind11::init<     std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ExternalFieldLattice<Shape> >, 
                            pybind11::list,
                            std::shared_ptr<IntegratorHPMCMono<Shape> >, 
                            unsigned int>())
    ;
    }
}

#endif // _CHANGE_SITE_UPDATER_H_
