// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// **********************
// This is a simple example code written for no function purpose other then to demonstrate the steps needed to write a
// c++ source code plugin for HOOMD-Blue. This example includes an example Updater class, but it can just as easily be
// replaced with a ForceCompute, Integrator, or any other C++ code at all.

// inclusion guard
#ifndef _CHANGE_SITE_UPDATER_HYPERSPHERE_H_
#define _CHANGE_SITE_UPDATER_HYPERSPHERE_H_

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
class ChangeSiteUpdaterHypersphere : public Updater
    {
    public:
        //! Constructor
        ChangeSiteUpdaterHypersphere( std::shared_ptr<SystemDefinition> sysdef,
                            std::shared_ptr<ExternalFieldLatticeHypersphere<Shape> > externalLattice,
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
            std::vector<Scalar4> lattice_vectors;
            std::vector<Scalar> lvbuffer;
            #ifdef ENABLE_MPI
            unsigned int lvsz = 0;

            if ( this->m_exec_conf->isRoot() )
                {
                hpmc::python_list_to_vector_scalar4(lv, lattice_vectors);
                lvsz = lattice_vectors.size();
                }
            if( this->m_pdata->getDomainDecomposition())
                {
                if(lvsz)
                    {
                    lvbuffer.resize(4*lvsz, 0.0);
                    for(size_t i = 0; i < lvsz; i++)
                        {
                        lvbuffer[4*i] = lattice_vectors[i].x;
                        lvbuffer[4*i+1] = lattice_vectors[i].y;
                        lvbuffer[4*i+2] = lattice_vectors[i].z;
                        lvbuffer[4*i+3] = lattice_vectors[i].w;
                        }
                    }
                MPI_Bcast(&lvsz, 1, MPI_UNSIGNED, 0, m_exec_conf->getMPICommunicator());
                if(lvsz)
                    {
                    if(!lvbuffer.size())
                        lvbuffer.resize(4*lvsz, 0.0);
                    MPI_Bcast(&lvbuffer.front(), 4*lvsz, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
                    if(!lattice_vectors.size())
                        {
                        lattice_vectors.resize(lvsz, make_scalar4(0.0, 0.0, 0.0, 0.0));
                        for(size_t i = 0; i < lvsz; i++)
                            {
                            lattice_vectors[i].x = lvbuffer[4*i];
                            lattice_vectors[i].y = lvbuffer[4*i+1];
                            lattice_vectors[i].z = lvbuffer[4*i+2];
                            lattice_vectors[i].w = lvbuffer[4*i+3];
                            }
                        }
                    }
                }

            #else
            hpmc::python_list_to_vector_scalar4(lv, lattice_vectors);
            #endif

            if( lattice_vectors.size() )
                m_latticeVector.setReferences(lattice_vectors.begin(), lattice_vectors.end(), m_pdata, m_exec_conf);

            }

    protected:
                std::shared_ptr<ExternalFieldLatticeHypersphere<Shape> > m_externalLattice;
                LatticeReferenceList<Scalar4> m_latticeVector;
                std::shared_ptr<IntegratorHPMCMono<Shape> > m_mc;
                detail::UpdateOrder m_update_order;         //!< Update order
                unsigned int m_seed;         
    };

template <class Shape>
void ChangeSiteUpdaterHypersphere<Shape>::update(unsigned int timestep)
    {

    const Hypersphere& hypersphere = m_pdata->getHypersphere();

    // Shuffle the order of particles for this step
    m_update_order.resize(m_pdata->getN());

    // update the AABB Tree
    detail::AABBTree aabb_tree = m_mc->buildAABBTree();
    std::vector<vec3<Scalar> > image_list = m_mc->updateImageList();

    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
	ArrayHandle<Scalar4> h_quat_l(m_pdata->getLeftQuaternionArray(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_quat_r(m_pdata->getRightQuaternionArray(), access_location::host, access_mode::readwrite);

    ArrayHandle<Scalar4> h_ql0(m_externalLattice->getReferenceLatticeQuat_l(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_qr0(m_externalLattice->getReferenceLatticeQuat_r(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_overlaps(m_mc->getInteractionMatrix(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

    Index2D overlap_idx =  m_mc->getOverlapIndexer();


    for (unsigned int cur_particle = 0; cur_particle < m_pdata->getN(); cur_particle++)
        {
        unsigned int i = m_update_order[cur_particle];

        // read in the current position and orientation
        Scalar4 postype_i = h_postype.data[i];
        quat<Scalar> quat_l_i(h_quat_l.data[i]);
        quat<Scalar> quat_r_i(h_quat_r.data[i]);
        quat<Scalar> refql_i = quat<Scalar>(h_ql0.data[i]);
        quat<Scalar> refqr_i = quat<Scalar>(h_qr0.data[i]);

        // make a trial move for i
        hoomd::RandomGenerator rng_i(hoomd::RNGIdentifier::UpdaterChangeSite, m_seed, i, m_exec_conf->getRank(), timestep);
        unsigned int indx = int(hoomd::UniformDistribution<Scalar>(0,m_latticeVector.getSize())(rng_i)); 


        quat<Scalar> dql =  quat<Scalar>( m_latticeVector.getReference(indx) );
        quat<Scalar> dqr =  conj(dql);

        quat_l_i = quat_l_i*dql;
        quat_r_i = dqr*quat_r_i;
        refql_i = refql_i*dql;
        refqr_i = dqr*refqr_i;

        int typ_i = __scalar_as_int(postype_i.w);
        Shape shape_i(quat_l_i, quat_r_i, m_mc->getParams()[typ_i]);


        bool overlap=false;
        detail::AABB aabb_i = shape_i.getAABBHypersphere(hypersphere);

        // stackless search
        for (unsigned int cur_node_idx = 0; cur_node_idx < aabb_tree.getNumNodes(); cur_node_idx++)
            {
            if (detail::overlap(aabb_tree.getNodeAABB(cur_node_idx), aabb_i))
                {
                if (aabb_tree.isNodeLeaf(cur_node_idx))
                    {
                    for (unsigned int cur_p = 0; cur_p < aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                        {
                        // read in its position and orientation
                        unsigned int j = aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                        // handle j==i situations
                        if ( j != i )
                            {
                            if (test_self_overlap_hypersphere(shape_i, hypersphere))
                                {
                                overlap = true;
                                break;
                                }
                            else
                                continue;
                            }

                        Scalar4 postype_j = h_postype.data[j];
                        quat<Scalar> quat_l_j(h_quat_l.data[j]);
                        quat<Scalar> quat_r_j(h_quat_r.data[j]);

                        unsigned int typ_j = __scalar_as_int(postype_j.w);
                        Shape shape_j(quat_l_j, quat_r_j, m_mc->getParams()[typ_j]);

                        unsigned int err_count = 0;

                        if ( h_overlaps.data[overlap_idx(typ_i, typ_j)]
                            && check_circumsphere_overlap_hypersphere(shape_i, shape_j, hypersphere)
                            && test_overlap_hypersphere(shape_i, shape_j, hypersphere, err_count))
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


            // If no overlaps and Metropolis criterion is met, accept
            // trial move and update positions  and/or orientations.
            if (!overlap)
                {
                // update the position of the particle in the tree for future updates
                aabb_tree.update(i, aabb_i);

                h_quat_l.data[i] = quat_to_scalar4(shape_i.quat_l);
                h_quat_r.data[i] = quat_to_scalar4(shape_i.quat_r);
                h_ql0.data[i] = quat_to_scalar4(refql_i);
                h_qr0.data[i] = quat_to_scalar4(refqr_i);
                }
        }

    }

//! Export the ExampleUpdater class to python
template <class Shape>
void export_UpdaterChangeSiteHypersphere(pybind11::module& m, std::string name)
    {
    using pybind11::class_;
   pybind11::class_<ChangeSiteUpdaterHypersphere<Shape>, std::shared_ptr<ChangeSiteUpdaterHypersphere<Shape> > >(m, name.c_str(), pybind11::base<Updater>())
   .def(pybind11::init<     std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ExternalFieldLatticeHypersphere<Shape> >, 
                            pybind11::list,
                            std::shared_ptr<IntegratorHPMCMono<Shape> >, 
                            unsigned int>())
    ;
    }
}

#endif // _CHANGE_SITE_UPDATER_HYPERSPHERE_H_
