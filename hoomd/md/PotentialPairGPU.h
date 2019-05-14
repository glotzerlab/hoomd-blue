// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#ifndef __POTENTIAL_PAIR_GPU_H__
#define __POTENTIAL_PAIR_GPU_H__

#ifdef ENABLE_CUDA

#include <memory>

#include "PotentialPair.h"
#include "PotentialPairGPU.cuh"

#include "hoomd/Autotuner.h"

/*! \file PotentialPairGPU.h
    \brief Defines the template class for standard pair potentials on the GPU
    \note This header cannot be compiled by nvcc
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>


//! Template class for computing pair potentials on the GPU
/*! Derived from PotentialPair, this class provides exactly the same interface for computing pair potentials and forces.
    In the same way as PotentialPair, this class serves as a shell dealing with all the details common to every pair
    potential calculation while the \a evaluator calculates V(r) in a generic way.

    Due to technical limitations, the instantiation of PotentialPairGPU cannot create a CUDA kernel automatically
    with the \a evaluator. Instead, a .cu file must be written that provides a driver function to call
    gpu_compute_pair_forces() instantiated with the same evaluator. (See PotentialPairLJGPU.cu and
    PotentialPairLJGPU.cuh for an example). That function is then passed into this class as another template parameter
    \a gpu_cgpf

    \tparam evaluator EvaluatorPair class used to evaluate V(r) and F(r)/r
    \tparam gpu_cgpf Driver function that calls gpu_compute_pair_forces<evaluator>()

    \sa export_PotentialPairGPU()
*/
template< class evaluator, cudaError_t gpu_cgpf(const pair_args_t& pair_args,
                                                const typename evaluator::param_type *d_params)>
class PotentialPairGPU : public PotentialPair<evaluator>
    {
    public:
        //! Construct the pair potential
        PotentialPairGPU(std::shared_ptr<SystemDefinition> sysdef,
                         std::shared_ptr<NeighborList> nlist,
                         const std::string& log_suffix="");
        //! Destructor
        virtual ~PotentialPairGPU() {}

        //! Set the number of threads per particle to execute on the GPU
        /*! \param threads_per_particl Number of threads per particle
            \a threads_per_particle must be a power of two and smaller than 32.
         */
        void setTuningParam(unsigned int param)
            {
            m_param = param;
            }

        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs

            Derived classes should override this to set the parameters of their autotuners.
        */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            PotentialPair<evaluator>::setAutotunerParams(enable, period);
            m_tuner->setPeriod(period);
            m_tuner->setEnabled(enable);
            }

    protected:
        std::unique_ptr<Autotuner> m_tuner;   //!< Autotuner for block size and threads per particle
        unsigned int m_param;                       //!< Kernel tuning parameter

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);

    };

template< class evaluator, cudaError_t gpu_cgpf(const pair_args_t& pair_args,
                                                const typename evaluator::param_type *d_params)>
PotentialPairGPU< evaluator, gpu_cgpf >::PotentialPairGPU(std::shared_ptr<SystemDefinition> sysdef,
                                                          std::shared_ptr<NeighborList> nlist, const std::string& log_suffix)
    : PotentialPair<evaluator>(sysdef, nlist, log_suffix), m_param(0)
    {
    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!this->m_exec_conf->isCUDAEnabled())
        {
        this->m_exec_conf->msg->error() << "Creating a PotentialPairGPU with no GPU in the execution configuration"
                  << std::endl;
        throw std::runtime_error("Error initializing PotentialPairGPU");
        }

    // initialize autotuner
    // the full block size and threads_per_particle matrix is searched,
    // encoded as block_size*10000 + threads_per_particle
    std::vector<unsigned int> valid_params;
    for (unsigned int block_size = 32; block_size <= 1024; block_size += 32)
        {
        for (auto s : Autotuner::getTppListPow2(this->m_exec_conf->dev_prop.warpSize))
            {
            valid_params.push_back(block_size*10000 + s);
            }
        }

    m_tuner.reset(new Autotuner(valid_params, 5, 100000, "pair_" + evaluator::getName(), this->m_exec_conf));
    #ifdef ENABLE_MPI
    // synchronize autotuner results across ranks
    m_tuner->setSync(bool(this->m_pdata->getDomainDecomposition()));
    #endif
    }

template< class evaluator, cudaError_t gpu_cgpf(const pair_args_t& pair_args,
                                                const typename evaluator::param_type *d_params)>
void PotentialPairGPU< evaluator, gpu_cgpf >::computeForces(unsigned int timestep)
    {
    this->m_nlist->compute(timestep);

    // start the profile
    if (this->m_prof) this->m_prof->push(this->m_exec_conf, this->m_prof_name);

    // The GPU implementation CANNOT handle a half neighborlist, error out now
    bool third_law = this->m_nlist->getStorageMode() == NeighborList::half;
    if (third_law)
        {
        this->m_exec_conf->msg->error() << "PotentialPairGPU cannot handle a half neighborlist"
                  << std::endl;
        throw std::runtime_error("Error computing forces in PotentialPairGPU");
        }

    // access the neighbor list
    ArrayHandle<unsigned int> d_n_neigh(this->m_nlist->getNNeighArray(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_nlist(this->m_nlist->getNListArray(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_head_list(this->m_nlist->getHeadList(), access_location::device, access_mode::read);

    // access the particle data
    ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_diameter(this->m_pdata->getDiameters(), access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_charge(this->m_pdata->getCharges(), access_location::device, access_mode::read);

    BoxDim box = this->m_pdata->getBox();

    // access parameters
    ArrayHandle<Scalar> d_ronsq(this->m_ronsq, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_rcutsq(this->m_rcutsq, access_location::device, access_mode::read);
    ArrayHandle<typename evaluator::param_type> d_params(this->m_params, access_location::device, access_mode::read);

    ArrayHandle<Scalar4> d_force(this->m_force, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_virial(this->m_virial, access_location::device, access_mode::readwrite);

    // access flags
    PDataFlags flags = this->m_pdata->getFlags();

    this->m_exec_conf->beginMultiGPU();

    if (! m_param) this->m_tuner->begin();
    unsigned int param = !m_param ?  this->m_tuner->getParam() : m_param;
    unsigned int block_size = param / 10000;
    unsigned int threads_per_particle = param % 10000;

    gpu_cgpf(pair_args_t(d_force.data,
                         d_virial.data,
                         this->m_virial.getPitch(),
                         this->m_pdata->getN(),
                         this->m_pdata->getMaxN(),
                         d_pos.data,
                         d_diameter.data,
                         d_charge.data,
                         box,
                         d_n_neigh.data,
                         d_nlist.data,
                         d_head_list.data,
                         d_rcutsq.data,
                         d_ronsq.data,
                         this->m_nlist->getNListArray().getPitch(),
                         this->m_pdata->getNTypes(),
                         block_size,
                         this->m_shift_mode,
                         flags[pdata_flag::pressure_tensor] || flags[pdata_flag::isotropic_virial],
                         threads_per_particle,
                         this->m_pdata->getGPUPartition()),
             d_params.data);

    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    if (!m_param) this->m_tuner->end();

    this->m_exec_conf->endMultiGPU();

    if (this->m_prof) this->m_prof->pop(this->m_exec_conf);
    }

//! Export this pair potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Class type to export. \b Must be an instantiated PotentialPairGPU class template.
    \tparam Base Base class of \a T. \b Must be PotentialPair<evaluator> with the same evaluator as used in \a T.
*/
template < class T, class Base > void export_PotentialPairGPU(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<T, std::shared_ptr<T> >(m, name.c_str(), pybind11::base<Base>())
        .def(pybind11::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>, const std::string& >())
        .def("setTuningParam",&T::setTuningParam)
    ;
    }

#endif // ENABLE_CUDA
#endif // __POTENTIAL_PAIR_GPU_H__
