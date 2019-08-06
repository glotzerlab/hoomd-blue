// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#ifndef __POTENTIAL_BOND_GPU_H__
#define __POTENTIAL_BOND_GPU_H__

#ifdef ENABLE_CUDA

#include "PotentialBond.h"
#include "PotentialBondGPU.cuh"
#include "hoomd/Autotuner.h"

/*! \file PotentialBondGPU.h
    \brief Defines the template class for standard bond potentials on the GPU
    \note This header cannot be compiled by nvcc
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

//! Template class for computing bond potentials on the GPU

/*!
    \tparam evaluator EvaluatorBond class used to evaluate V(r) and F(r)/r
    \tparam gpu_cgbf Driver function that calls gpu_compute_bond_forces<evaluator>()

    \sa export_PotentialBondGPU()
*/
template< class evaluator, cudaError_t gpu_cgbf(const bond_args_t& bond_args,
                                                const typename evaluator::param_type *d_params,
                                                unsigned int *d_flags) >
class PotentialBondGPU : public PotentialBond<evaluator>
    {
    public:
        //! Construct the bond potential
        PotentialBondGPU(std::shared_ptr<SystemDefinition> sysdef,
                         const std::string& log_suffix="");
        //! Destructor
        virtual ~PotentialBondGPU() {}

        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs
        */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            PotentialBond<evaluator>::setAutotunerParams(enable, period);
            m_tuner->setPeriod(period);
            m_tuner->setEnabled(enable);
            }

    protected:
        std::unique_ptr<Autotuner> m_tuner; //!< Autotuner for block size
        GPUArray<unsigned int> m_flags;       //!< Flags set during the kernel execution

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

template< class evaluator, cudaError_t gpu_cgbf(const bond_args_t& bond_args,
                                                const typename evaluator::param_type *d_params,
                                                unsigned int *d_flags) >
PotentialBondGPU< evaluator, gpu_cgbf >::PotentialBondGPU(std::shared_ptr<SystemDefinition> sysdef,
                                                          const std::string& log_suffix)
    : PotentialBond<evaluator>(sysdef, log_suffix)
    {
    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!this->m_exec_conf->isCUDAEnabled())
        {
        this->m_exec_conf->msg->error() << "Creating a PotentialBondGPU with no GPU in the execution configuration" << std::endl;
        throw std::runtime_error("Error initializing PotentialBondGPU");
        }

     // allocate and zero device memory
    GPUArray<typename evaluator::param_type> params(this->m_bond_data->getNTypes(), this->m_exec_conf);
    this->m_params.swap(params);

     // allocate flags storage on the GPU
    GPUArray<unsigned int> flags(1, this->m_exec_conf);
    m_flags.swap(flags);

    // reset flags
    ArrayHandle<unsigned int> h_flags(m_flags,access_location::host, access_mode::overwrite);
    h_flags.data[0] = 0;

    m_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "harmonic_bond", this->m_exec_conf));
    }

template< class evaluator, cudaError_t gpu_cgbf(const bond_args_t& bond_args,
                                                const typename evaluator::param_type *d_params,
                                                unsigned int *d_flags) >
void PotentialBondGPU< evaluator, gpu_cgbf >::computeForces(unsigned int timestep)
    {
    // start the profile
    if (this->m_prof) this->m_prof->push(this->m_exec_conf, this->m_prof_name);

    // access the particle data
    ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_diameter(this->m_pdata->getDiameters(), access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_charge(this->m_pdata->getCharges(), access_location::device, access_mode::read);

    // we are using the minimum image of the global box here
    // to ensure that ghosts are always correctly wrapped (even if a bond exceeds half the domain length)
    BoxDim box = this->m_pdata->getGlobalBox();

    // access parameters
    ArrayHandle<typename evaluator::param_type> d_params(this->m_params, access_location::device, access_mode::read);

    // access net force & virial
    ArrayHandle<Scalar4> d_force(this->m_force, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_virial(this->m_virial, access_location::device, access_mode::readwrite);

        {
        const GPUArray<typename BondData::members_t>& gpu_bond_list = this->m_bond_data->getGPUTable();
        const Index2D& gpu_table_indexer = this->m_bond_data->getGPUTableIndexer();

        ArrayHandle<typename BondData::members_t> d_gpu_bondlist(gpu_bond_list, access_location::device, access_mode::read);
        ArrayHandle<unsigned int > d_gpu_n_bonds(this->m_bond_data->getNGroupsArray(),
                                                 access_location::device, access_mode::read);

        // access the flags array for overwriting
        ArrayHandle<unsigned int> d_flags(m_flags, access_location::device, access_mode::readwrite);

        this->m_tuner->begin();
        gpu_cgbf(bond_args_t(d_force.data,
                             d_virial.data,
                             this->m_virial.getPitch(),
                             this->m_pdata->getN(),
                             this->m_pdata->getMaxN(),
                             d_pos.data,
                             d_charge.data,
                             d_diameter.data,
                             box,
                             d_gpu_bondlist.data,
                             gpu_table_indexer,
                             d_gpu_n_bonds.data,
                             this->m_bond_data->getNTypes(),
                             this->m_tuner->getParam()),
                 d_params.data,
                 d_flags.data);
        }

    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        {
        CHECK_CUDA_ERROR();

        // check the flags for any errors
        ArrayHandle<unsigned int> h_flags(m_flags, access_location::host, access_mode::read);

        if (h_flags.data[0] & 1)
            {
            this->m_exec_conf->msg->error() << "bond." << evaluator::getName() << ": bond out of bounds (" << h_flags.data[0] << ")" << std::endl << std::endl;
            throw std::runtime_error("Error in bond calculation");
            }
        }
    this->m_tuner->end();

    if (this->m_prof) this->m_prof->pop(this->m_exec_conf);
    }

//! Export this bond potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Class type to export. \b Must be an instantiated PotentialPairGPU class template.
    \tparam Base Base class of \a T. \b Must be PotentialPair<evaluator> with the same evaluator as used in \a T.
*/
template < class T, class Base > void export_PotentialBondGPU(pybind11::module& m, const std::string& name)
    {
     pybind11::class_<T, std::shared_ptr<T> >(m, name.c_str(), pybind11::base<Base>())
            .def(pybind11::init< std::shared_ptr<SystemDefinition>, const std::string& >())
            ;
    }

#endif // ENABLE_CUDA
#endif // __POTENTIAL_PAIR_GPU_H__
