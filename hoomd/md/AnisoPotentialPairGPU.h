// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __ANISO_POTENTIAL_PAIR_GPU_H__
#define __ANISO_POTENTIAL_PAIR_GPU_H__

#ifdef ENABLE_HIP

#include "AnisoPotentialPair.h"
#include "AnisoPotentialPairGPU.cuh"
#include "hoomd/Autotuner.h"

/*! \file AnisoPotentialPairGPU.h
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
//! Template class for computing anisotropic pair potentials on the GPU
/*! Derived from AnisoPotentialPair, this class provides exactly the same interface for computing
   anisotropic pair potentials, forces and torques.  In the same way as PotentialPair, this class
   serves as a shell dealing with all the details common to every pair potential calculation while
   te \a evaluator calculates \f$V(\vec r,\vec e_i, \vec e_j)\f$ in a generic way.

    \tparam evaluator EvaluatorPair class used to evaluate potential, force and torque.
    \sa export_AnisoPotentialPairGPU()
*/
template<class evaluator> class AnisoPotentialPairGPU : public AnisoPotentialPair<evaluator>
    {
    public:
    //! Construct the pair potential
    AnisoPotentialPairGPU(std::shared_ptr<SystemDefinition> sysdef,
                          std::shared_ptr<NeighborList> nlist);
    //! Destructor
    virtual ~AnisoPotentialPairGPU() { };

    virtual void
    setParams(unsigned int typ1, unsigned int typ2, const typename evaluator::param_type& param);

    virtual void setShape(unsigned int typ, const typename evaluator::shape_type& shape_param);

    protected:
    std::shared_ptr<Autotuner<2>> m_tuner; //!< Autotuner for block size and threads per particle

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

template<class evaluator>
AnisoPotentialPairGPU<evaluator>::AnisoPotentialPairGPU(std::shared_ptr<SystemDefinition> sysdef,
                                                        std::shared_ptr<NeighborList> nlist)
    : AnisoPotentialPair<evaluator>(sysdef, nlist)
    {
    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!this->m_exec_conf->isCUDAEnabled())
        {
        this->m_exec_conf->msg->error()
            << "ai_pair." << evaluator::getName()
            << ": Creating a AnisoPotentialPairGPU with no GPU in the execution configuration"
            << std::endl
            << std::endl;
        throw std::runtime_error("Error initializing AnisoPotentialPairGPU");
        }

    // Initialize autotuner that tunes block sizes and threads per particle.
    m_tuner.reset(new Autotuner<2>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf),
                                    AutotunerBase::getTppListPow2(this->m_exec_conf)},
                                   this->m_exec_conf,
                                   "aniso_pair_" + evaluator::getName()));
    this->m_autotuners.push_back(m_tuner);

#ifdef ENABLE_MPI
    // synchronize autotuner results across ranks
    m_tuner->setSync(bool(this->m_pdata->getDomainDecomposition()));
#endif
    }

template<class evaluator> void AnisoPotentialPairGPU<evaluator>::computeForces(uint64_t timestep)
    {
    this->m_nlist->compute(timestep);

    // The GPU implementation CANNOT handle a half neighborlist, error out now
    bool third_law = this->m_nlist->getStorageMode() == NeighborList::half;
    if (third_law)
        {
        this->m_exec_conf->msg->error()
            << "ai_pair." << evaluator::getName()
            << ": AnisoPotentialPairGPU cannot handle a half neighborlist" << std::endl
            << std::endl;
        throw std::runtime_error("Error computing forces in AnisoPotentialPairGPU");
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
    ArrayHandle<Scalar4> d_orientation(this->m_pdata->getOrientationArray(),
                                       access_location::device,
                                       access_mode::read);
    ArrayHandle<unsigned int> d_tag(this->m_pdata->getTags(),
                                    access_location::device,
                                    access_mode::read);

    BoxDim box = this->m_pdata->getBox();

    // access parameters
    ArrayHandle<Scalar> d_rcutsq(this->m_rcutsq, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_force(this->m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_torque(this->m_torque, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(this->m_virial, access_location::device, access_mode::overwrite);

    // access flags
    PDataFlags flags = this->m_pdata->getFlags();

    this->m_exec_conf->beginMultiGPU();

    this->m_tuner->begin();
    unsigned int block_size = this->m_tuner->getParam()[0];
    unsigned int threads_per_particle = this->m_tuner->getParam()[1];

    // On the first iteration, shape parameters are updated. For optimization,
    // could track this between calls to avoid extra copying.
    bool first = true;

    kernel::gpu_compute_pair_aniso_forces<evaluator>(
        kernel::a_pair_args_t(d_force.data,
                              d_torque.data,
                              d_virial.data,
                              this->m_virial.getPitch(),
                              this->m_pdata->getN(),
                              this->m_pdata->getMaxN(),
                              d_pos.data,
                              d_charge.data,
                              d_orientation.data,
                              d_tag.data,
                              box,
                              d_n_neigh.data,
                              d_nlist.data,
                              d_head_list.data,
                              d_rcutsq.data,
                              this->m_pdata->getNTypes(),
                              block_size,
                              this->m_shift_mode,
                              flags[pdata_flag::pressure_tensor],
                              threads_per_particle,
                              this->m_pdata->getGPUPartition(),
                              this->m_exec_conf->dev_prop,
                              first),
        this->m_params.data(),
        this->m_shape_params.data());

    this->m_tuner->end();

    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    this->m_exec_conf->endMultiGPU();
    }

template<class evaluator>
void AnisoPotentialPairGPU<evaluator>::setParams(unsigned int typ1,
                                                 unsigned int typ2,
                                                 const typename evaluator::param_type& param)
    {
    AnisoPotentialPair<evaluator>::setParams(typ1, typ2, param);
    this->m_params[this->m_typpair_idx(typ1, typ2)].set_memory_hint();
    this->m_params[this->m_typpair_idx(typ2, typ1)].set_memory_hint();
    }

template<class evaluator>
void AnisoPotentialPairGPU<evaluator>::setShape(unsigned int typ,
                                                const typename evaluator::shape_type& shape_param)
    {
    AnisoPotentialPair<evaluator>::setShape(typ, shape_param);
    this->m_shape_params[typ].set_memory_hint();
    }

namespace detail
    {
//! Export this pair potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Class type to export. \b Must be an instantiated AnisoPotentialPairGPU class template.
    \tparam Base Base class of \a T. \b Must be PotentialPair<evaluator> with the same evaluator as
   used in \a T.
*/
template<class T> void export_AnisoPotentialPairGPU(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<AnisoPotentialPairGPU<T>,
                     AnisoPotentialPair<T>,
                     std::shared_ptr<AnisoPotentialPairGPU<T>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif // ENABLE_HIP
#endif // __ANISO_POTENTIAL_PAIR_GPU_H__
