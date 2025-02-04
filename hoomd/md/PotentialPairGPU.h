// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __POTENTIAL_PAIR_GPU_H__
#define __POTENTIAL_PAIR_GPU_H__

#ifdef ENABLE_HIP

#include <memory>

#include "PotentialPair.h"
#include "PotentialPairGPU.cuh"

#include "hoomd/Autotuner.h"

/*! \file PotentialPairGPU.h
    \brief Defines the template class for standard pair potentials on the GPU
    \note This header cannot be compiled by nvcc
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {
//! Template class for computing pair potentials on the GPU
/*! Derived from PotentialPair, this class provides exactly the same interface for computing pair
   potentials and forces. In the same way as PotentialPair, this class serves as a shell dealing
   with all the details common to every pair potential calculation while the \a evaluator calculates
   V(r) in a generic way.

    Due to technical limitations, the instantiation of PotentialPairGPU cannot create a CUDA kernel
   automatically with the \a evaluator. Instead, a .cu file must be written that provides a driver
   function to call gpu_compute_pair_forces() instantiated with the same evaluator. (See
   PotentialPairLJGPU.cu and PotentialPairLJGPU.cuh for an example).

    \tparam evaluator EvaluatorPair class used to evaluate V(r) and F(r)/r

    \sa export_PotentialPairGPU()
*/
template<class evaluator> class PotentialPairGPU : public PotentialPair<evaluator>
    {
    public:
    //! Construct the pair potential
    PotentialPairGPU(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<NeighborList> nlist);
    //! Destructor
    virtual ~PotentialPairGPU() { }

    protected:
    std::shared_ptr<Autotuner<2>> m_tuner; //!< Autotuner for block size and threads per particle

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

template<class evaluator>
PotentialPairGPU<evaluator>::PotentialPairGPU(std::shared_ptr<SystemDefinition> sysdef,
                                              std::shared_ptr<NeighborList> nlist)
    : PotentialPair<evaluator>(sysdef, nlist)
    {
    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!this->m_exec_conf->isCUDAEnabled())
        {
        this->m_exec_conf->msg->error()
            << "Creating a PotentialPairGPU with no GPU in the execution configuration"
            << std::endl;
        throw std::runtime_error("Error initializing PotentialPairGPU");
        }

    // Initialize autotuner.
    m_tuner.reset(new Autotuner<2>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf),
                                    AutotunerBase::getTppListPow2(this->m_exec_conf)},
                                   this->m_exec_conf,
                                   "pair_" + evaluator::getName()));

    this->m_autotuners.push_back(m_tuner);

#ifdef ENABLE_MPI
    // synchronize autotuner results across ranks
    m_tuner->setSync(bool(this->m_pdata->getDomainDecomposition()));
#endif
    }

template<class evaluator> void PotentialPairGPU<evaluator>::computeForces(uint64_t timestep)
    {
    this->m_nlist->compute(timestep);

    // The GPU implementation CANNOT handle a half neighborlist, error out now
    bool third_law = this->m_nlist->getStorageMode() == NeighborList::half;
    if (third_law)
        {
        this->m_exec_conf->msg->error()
            << "PotentialPairGPU cannot handle a half neighborlist" << std::endl;
        throw std::runtime_error("Error computing forces in PotentialPairGPU");
        }

    // access the neighbor list
    ArrayHandle<unsigned int> d_n_neigh(this->m_nlist->getNNeighArray(),
                                        access_location::device,
                                        access_mode::read);
    ArrayHandle<unsigned int> d_nlist(this->m_nlist->getNListArray(),
                                      access_location::device,
                                      access_mode::read);
    ArrayHandle<size_t> d_head_list(this->m_nlist->getHeadList(),
                                    access_location::device,
                                    access_mode::read);

    // access the particle data
    ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(),
                               access_location::device,
                               access_mode::read);
    ArrayHandle<Scalar> d_charge(this->m_pdata->getCharges(),
                                 access_location::device,
                                 access_mode::read);

    BoxDim box = this->m_pdata->getBox();

    // access parameters
    ArrayHandle<Scalar> d_ronsq(this->m_ronsq, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_rcutsq(this->m_rcutsq, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_force(this->m_force, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_virial(this->m_virial, access_location::device, access_mode::readwrite);

    // access flags
    PDataFlags flags = this->m_pdata->getFlags();

    this->m_exec_conf->beginMultiGPU();

    m_tuner->begin();
    auto param = m_tuner->getParam();
    unsigned int block_size = param[0];
    unsigned int threads_per_particle = param[1];

    kernel::gpu_compute_pair_forces<evaluator>(
        kernel::pair_args_t(d_force.data,
                            d_virial.data,
                            this->m_virial.getPitch(),
                            this->m_pdata->getN(),
                            this->m_pdata->getMaxN(),
                            d_pos.data,
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
                            flags[pdata_flag::pressure_tensor],
                            threads_per_particle,
                            this->m_pdata->getGPUPartition(),
                            this->m_exec_conf->dev_prop),
        this->m_params.data());

    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    m_tuner->end();

    this->m_exec_conf->endMultiGPU();

    // energy and pressure corrections
    this->computeTailCorrection();
    }

namespace detail
    {
//! Export this pair potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Evaluator type to export.
*/
template<class T> void export_PotentialPairGPU(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<PotentialPairGPU<T>, PotentialPair<T>, std::shared_ptr<PotentialPairGPU<T>>>(
        m,
        name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif // ENABLE_HIP
#endif // __POTENTIAL_PAIR_GPU_H__
