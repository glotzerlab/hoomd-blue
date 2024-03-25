// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __POTENTIAL_SPECIAL_PAIR_GPU_H__
#define __POTENTIAL_SPECIAL_PAIR_GPU_H__

#ifdef ENABLE_HIP

#include "PotentialSpecialPair.h"
//! Use GPU functions for bonds
#include "PotentialBondGPU.cuh"
#include "hoomd/Autotuner.h"

/*! \file PotentialSpecialPairGPU.h
    \brief Defines the template class for special pair potentials on the GPU
    \note This header cannot be compiled by nvcc
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

namespace hoomd
    {
namespace md
    {
//! Template class for computing special pair potentials on the GPU
/*!
    \tparam evaluator EvaluatorSpecialPair class used to evaluate V(r) and F(r)/r

    \sa export_PotentialSpecialPairGPU()
*/
template<class evaluator> class PotentialSpecialPairGPU : public PotentialSpecialPair<evaluator>
    {
    public:
    //! Construct the special_pair potential
    PotentialSpecialPairGPU(std::shared_ptr<SystemDefinition> sysdef);
    //! Destructor
    virtual ~PotentialSpecialPairGPU() { }

    protected:
    std::shared_ptr<Autotuner<1>> m_tuner; //!< Autotuner for block size
    GPUArray<unsigned int> m_flags;        //!< Flags set during the kernel execution

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

template<class evaluator>
PotentialSpecialPairGPU<evaluator>::PotentialSpecialPairGPU(
    std::shared_ptr<SystemDefinition> sysdef)
    : PotentialSpecialPair<evaluator>(sysdef)
    {
    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!this->m_exec_conf->isCUDAEnabled())
        {
        this->m_exec_conf->msg->error()
            << "Creating a PotentialSpecialPairGPU with no GPU in the execution configuration"
            << std::endl;
        throw std::runtime_error("Error initializing PotentialSpecialPairGPU");
        }

    // allocate and zero device memory
    GPUArray<typename evaluator::param_type> params(this->m_pair_data->getNTypes(),
                                                    this->m_exec_conf);
    this->m_params.swap(params);

    // allocate flags storage on the GPU
    GPUArray<unsigned int> flags(1, this->m_exec_conf);
    m_flags.swap(flags);

    // reset flags
    ArrayHandle<unsigned int> h_flags(m_flags, access_location::host, access_mode::overwrite);
    h_flags.data[0] = 0;

    m_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf)},
                                   this->m_exec_conf,
                                   "special_pair_" + evaluator::getName()));
    this->m_autotuners.push_back(m_tuner);
    }

template<class evaluator> void PotentialSpecialPairGPU<evaluator>::computeForces(uint64_t timestep)
    {
    // access the particle data
    ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(),
                               access_location::device,
                               access_mode::read);
    ArrayHandle<Scalar> d_charge(this->m_pdata->getCharges(),
                                 access_location::device,
                                 access_mode::read);

    // we are using the minimum image of the global box here
    // to ensure that ghosts are always correctly wrapped (even if a bond exceeds half the domain
    // length)
    BoxDim box = this->m_pdata->getGlobalBox();

    // access parameters
    ArrayHandle<typename evaluator::param_type> d_params(this->m_params,
                                                         access_location::device,
                                                         access_mode::read);

    // access net force & virial
    ArrayHandle<Scalar4> d_force(this->m_force, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_virial(this->m_virial, access_location::device, access_mode::readwrite);

        {
        const GPUArray<typename PairData::members_t>& gpu_bond_list
            = this->m_pair_data->getGPUTable();
        const Index2D& gpu_table_indexer = this->m_pair_data->getGPUTableIndexer();

        ArrayHandle<typename PairData::members_t> d_gpu_bondlist(gpu_bond_list,
                                                                 access_location::device,
                                                                 access_mode::read);
        ArrayHandle<unsigned int> d_gpu_bond_pos_list(this->m_pair_data->getGPUPosTable(),
                                                      access_location::device,
                                                      access_mode::read);
        ArrayHandle<unsigned int> d_gpu_n_bonds(this->m_pair_data->getNGroupsArray(),
                                                access_location::device,
                                                access_mode::read);

        // access the flags array for overwriting
        ArrayHandle<unsigned int> d_flags(m_flags, access_location::device, access_mode::readwrite);

        m_tuner->begin();
        kernel::gpu_compute_bond_forces<evaluator, 2>(
            kernel::bond_args_t<2>(d_force.data,
                                   d_virial.data,
                                   this->m_virial.getPitch(),
                                   this->m_pdata->getN(),
                                   this->m_pdata->getMaxN(),
                                   d_pos.data,
                                   d_charge.data,
                                   box,
                                   d_gpu_bondlist.data,
                                   gpu_table_indexer,
                                   d_gpu_bond_pos_list.data,
                                   d_gpu_n_bonds.data,
                                   this->m_pair_data->getNTypes(),
                                   m_tuner->getParam()[0],
                                   this->m_exec_conf->dev_prop),
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
            this->m_exec_conf->msg->error()
                << "special_pair." << evaluator::getName() << ": pair out of bounds ("
                << h_flags.data[0] << ")" << std::endl
                << std::endl;
            throw std::runtime_error("Error in special_pair calculation");
            }
        }
    m_tuner->end();
    }

namespace detail
    {
//! Export this special pair potential to python
/*! \param name Name of the class in the exported python module
    \tparam T evaluator type to export.
*/
template<class T> void export_PotentialSpecialPairGPU(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<PotentialSpecialPairGPU<T>,
                     PotentialSpecialPair<T>,
                     std::shared_ptr<PotentialSpecialPairGPU<T>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif // ENABLE_HIP
#endif // __POTENTIAL_SPECIAL_PAIR_GPU_H__
