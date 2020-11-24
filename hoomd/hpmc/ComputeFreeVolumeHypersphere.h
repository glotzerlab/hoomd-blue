// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef __COMPUTE_FREE_VOLUME__HYPERSPHERE__H__
#define __COMPUTE_FREE_VOLUME__HYPERSPHERE__H__


#include "hoomd/Compute.h"
#include "hoomd/CellList.h"
#include "hoomd/Autotuner.h"

#include "HPMCPrecisionSetup.h"
#include "IntegratorHPMCMono.h"
#include "hoomd/RNGIdentifiers.h"


/*! \file ComputeFreeVolumeHypersphere.h
    \brief Defines the template class for an approximate free volume integration
    \note This header cannot be compiled by nvcc
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>


namespace hpmc
{

//! Template class for a free volume integration analyzer
/*!
    \ingroup hpmc_integrators
*/
template< class Shape >
class ComputeFreeVolumeHypersphere : public Compute
    {
    public:
        //! Construct the integrator
        ComputeFreeVolumeHypersphere(std::shared_ptr<SystemDefinition> sysdef,
                             std::shared_ptr<IntegratorHPMCMono<Shape> > mc,
                             std::shared_ptr<CellList> cl,
                             unsigned int seed,
                             std::string suffix);
        //! Destructor
        virtual ~ComputeFreeVolumeHypersphere() { };

        //! Set the number of MC samples to perform
        void setNumSamples(unsigned int n_sample)
            {
            m_n_sample = n_sample;
            }

        //! Set the type of depletant particle
        void setTestParticleType(unsigned int type)
            {
            assert(type < m_pdata->getNTypes());
            m_type = type;
            }

        /* \returns a list of provided quantities
        */
        std::vector< std::string > getProvidedLogQuantities()
            {
            std::vector< std::string> result;
            result.push_back("hpmc_free_volume"+m_suffix);

            return result;
            }

        //! Get the value of a logged quantity
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep);

        //! Return an estimate of the overlap volume
        virtual void computeFreeVolume(unsigned int timestep);

        //! Analyze the current configuration
        virtual void compute(unsigned int timestep);

    protected:
        std::shared_ptr<IntegratorHPMCMono<Shape> > m_mc;              //!< The parent integrator
        std::shared_ptr<CellList> m_cl;                        //!< The cell list

        unsigned int m_type;                                     //!< Type of depletant particle to generate
        unsigned int m_n_sample;                                 //!< Number of sampling depletants to generate
        unsigned int m_seed;                                     //!< The RNG seed
        const std::string m_suffix;                              //!< Log suffix

        GPUArray<unsigned int> m_n_overlap_all;                  //!< Number of overlap volume particles in box
    };


template< class Shape >
ComputeFreeVolumeHypersphere< Shape >::ComputeFreeVolumeHypersphere(std::shared_ptr<SystemDefinition> sysdef,
                                                    std::shared_ptr<IntegratorHPMCMono<Shape> > mc,
                                                    std::shared_ptr<CellList> cl,
                                                    unsigned int seed,
                                                    std::string suffix)
    : Compute(sysdef), m_mc(mc), m_cl(cl), m_type(0), m_n_sample(0), m_seed(seed), m_suffix(suffix)
    {
    this->m_exec_conf->msg->notice(5) << "Constructing ComputeFreeVolumeHypersphere" << std::endl;

    // broadcast the seed from rank 0 to all other ranks.
    #ifdef ENABLE_MPI
        if(this->m_pdata->getDomainDecomposition())
            bcast(m_seed, 0, this->m_exec_conf->getMPICommunicator());
    #endif

    this->m_cl->setRadius(1);
    this->m_cl->setComputeTDB(false);
    this->m_cl->setFlagType();
    this->m_cl->setComputeIdx(true);

    // allocate mem for overlap counts
    GPUArray<unsigned int> n_overlap_all(1,this->m_exec_conf);
    m_n_overlap_all.swap(n_overlap_all);
    }

template<class Shape>
void ComputeFreeVolumeHypersphere<Shape>::compute(unsigned int timestep)
    {
    if (!shouldCompute(timestep))
        return;

    // update ghost layers
    m_mc->communicate(false);

    this->computeFreeVolume(timestep);
    }

/*! \return the current free volume estimate by MC integration
*/
template<class Shape>
void ComputeFreeVolumeHypersphere<Shape>::computeFreeVolume(unsigned int timestep)
    {
    unsigned int overlap_count = 0;
    unsigned int err_count = 0;

    this->m_exec_conf->msg->notice(5) << "HPMC computing free volume " << timestep << std::endl;

    // update AABB tree
    const detail::AABBTree& aabb_tree = this->m_mc->buildAABBTree();

    // update the image list
    std::vector<vec3<Scalar> > image_list = this->m_mc->updateImageList();

    if (m_prof) m_prof->push("Free volume");

    // only check if AABB tree is populated
    if (m_pdata->getN() + m_pdata->getNGhosts())
        {
        // access particle data and system box
        ArrayHandle<Scalar4> h_quat_l(m_pdata->getLeftQuaternionArray(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_quat_r(m_pdata->getRightQuaternionArray(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
        const Hypersphere& hypersphere = m_pdata->getHypersphere();

        // access parameters and interaction matrix
        const std::vector<typename Shape::param_type, managed_allocator<typename Shape::param_type> > & params = m_mc->getParams();

        ArrayHandle<unsigned int> h_overlaps(m_mc->getInteractionMatrix(), access_location::host, access_mode::read);
        const Index2D& overlap_idx = m_mc->getOverlapIndexer();

        // generate n_sample random test depletants in the global box
        unsigned int n_sample = m_n_sample;

        #ifdef ENABLE_MPI
        n_sample /= this->m_exec_conf->getNRanks();
        #endif


        for (unsigned int i = 0; i < n_sample; i++)
            {
            // select a random particle coordinate in the box
            hoomd::RandomGenerator rng_i(hoomd::RNGIdentifier::ComputeFreeVolume, m_seed, m_exec_conf->getRank(), i, timestep);

            quat<Scalar> rand1 = generateRandomOrientation(rng_i);
            quat<Scalar> rand2 = generateRandomOrientation(rng_i);

            quat<Scalar> quat_l_i = rand1*rand2;
            quat<Scalar> quat_r_i = conj(rand2)*rand1;


            Shape shape_i(quat_l_i, quat_r_i, params[m_type]);

            // check for overlaps with neighboring particle's positions
            bool overlap=false;
            detail::AABB aabb_i_local = shape_i.getAABBHypersphere(hypersphere);

            // stackless search
            for (unsigned int cur_node_idx = 0; cur_node_idx < aabb_tree.getNumNodes(); cur_node_idx++)
                {
                if (detail::overlap(aabb_tree.getNodeAABB(cur_node_idx), aabb_i_local))
                    {
                    if (aabb_tree.isNodeLeaf(cur_node_idx))
                        {
                        for (unsigned int cur_p = 0; cur_p < aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                            {
                            // read in its position and orientation
                            unsigned int j = aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                             // handle j==i situations
                             if (j == i)
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
                            Shape shape_j(quat_l_j, quat_r_j, params[typ_j]);

                            if (h_overlaps.data[overlap_idx(m_type, typ_j)]
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

            if (overlap)
                {
                overlap_count++;
                }
            } // end loop through all particles

        } // end lexical scope

    #ifdef ENABLE_MPI
    if (m_comm)
        {
        MPI_Allreduce(MPI_IN_PLACE, &overlap_count, 1, MPI_UNSIGNED, MPI_SUM, m_exec_conf->getMPICommunicator());
        }
    #endif

    if (m_prof) m_prof->pop();

    ArrayHandle<unsigned int> h_n_overlap_all(m_n_overlap_all, access_location::host, access_mode::overwrite);
    *h_n_overlap_all.data = overlap_count;
    }

/*! \param quantity Name of the log quantity to get
    \param timestep Current time step of the simulation
    \return the requested log quantity.
*/
template<class Shape>
Scalar ComputeFreeVolumeHypersphere<Shape>::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == "hpmc_free_volume"+m_suffix)
        {
        // perform MC integration
        compute(timestep);

        // access counters
        ArrayHandle<unsigned int> h_n_overlap_all(m_n_overlap_all, access_location::host, access_mode::read);

        // generate n_sample random test depletants in the global box
        unsigned int n_sample = m_n_sample;

        #ifdef ENABLE_MPI
        // in MPI, for small n_sample we can encounter round-off issues
        unsigned int n_ranks = this->m_exec_conf->getNRanks();
        n_sample = (n_sample/n_ranks)*n_ranks;
        #endif


        // total free volume
        const Hypersphere& hypersphere = this->m_pdata->getHypersphere();
        Scalar V_free = (Scalar)(n_sample-*h_n_overlap_all.data)/(Scalar)n_sample*hypersphere.getVolume();

        return V_free;
        }
    throw std::runtime_error("Undefined log quantity");
    }

//! Export this hpmc analyzer to python
/*! \param name Name of the class in the exported python module
    \tparam Shape An instantiation of IntegratorHPMCMono<Shape> will be exported
*/
template < class Shape > void export_ComputeFreeVolumeHypersphere(pybind11::module& m, const std::string& name)
    {
     pybind11::class_<ComputeFreeVolumeHypersphere<Shape>, std::shared_ptr< ComputeFreeVolumeHypersphere<Shape> > >(m, name.c_str(), pybind11::base< Compute >())
              .def(pybind11::init< std::shared_ptr<SystemDefinition>,
                std::shared_ptr<IntegratorHPMCMono<Shape> >,
                std::shared_ptr<CellList>,
                unsigned int,
                std::string >())
        .def("setNumSamples", &ComputeFreeVolumeHypersphere<Shape>::setNumSamples)
        .def("setTestParticleType", &ComputeFreeVolumeHypersphere<Shape>::setTestParticleType)
        ;
    }

} // end namespace hpmc

#endif // __COMPUTE_FREE_VOLUME__HYPERSPHERE__H__
