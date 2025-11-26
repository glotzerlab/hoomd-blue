// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __FRICTION_PAIR_GPU_H__
#define __FRICTION_PAIR_GPU_H__

#ifdef ENABLE_HIP

#include "FrictionPair.h"
#include "FrictionPairGPU.cuh"
#include "hoomd/Autotuner.h"

/*! \file FrictionPairGPU.h
    \brief Defines the template class for frictional pair interactions on the GPU
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
//! Template class for computing friction pair potentials on the GPU
/*! Derived from FrictionPair, this class provides exactly the same interface for computing
   friction pair interactions, forces and torques.  In the same way as FrictionPair, this class
   serves as a shell dealing with all the details common to every frictional contact calculation
   while te \a evaluator calculates \f$V(\vec r,\vec e_i, \vec e_j)\f$ in a generic way.

    \tparam evaluator EvaluatorPair class used to evaluate potential, force and torque.
    \sa export_FrictionPairGPU()
*/
template<class evaluator> class FrictionPairGPU : public FrictionPair<evaluator>
    {
    public:
    //! Construct the pair potential
    FrictionPairGPU(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<NeighborList> nlist);
    //! Destructor
    virtual ~FrictionPairGPU() { };

    virtual void
    setParams(unsigned int typ1, unsigned int typ2, const typename evaluator::param_type& param);

    protected:
    std::shared_ptr<Autotuner<2>> m_tuner; //!< Autotuner for block size and threads per particle

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

template<class evaluator>
FrictionPairGPU<evaluator>::FrictionPairGPU(std::shared_ptr<SystemDefinition> sysdef,
                                            std::shared_ptr<NeighborList> nlist)
    : FrictionPair<evaluator>(sysdef, nlist)
    {
    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!this->m_exec_conf->isCUDAEnabled())
        {
        this->m_exec_conf->msg->error()
            << "ai_pair." << evaluator::getName()
            << ": Creating a FrictionPairGPU with no GPU in the execution configuration"
            << std::endl
            << std::endl;
        throw std::runtime_error("Error initializing FrictionPairGPU");
        }

    // Initialize autotuner that tunes block sizes and threads per particle.
    m_tuner.reset(new Autotuner<2>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf),
                                    AutotunerBase::getTppListPow2(this->m_exec_conf)},
                                   this->m_exec_conf,
                                   "friction_pair_" + evaluator::getName()));
    this->m_autotuners.push_back(m_tuner);

#ifdef ENABLE_MPI
    // synchronize autotuner results across ranks
    m_tuner->setSync(bool(this->m_pdata->getDomainDecomposition()));
#endif
    }

template<class evaluator> void FrictionPairGPU<evaluator>::computeForces(uint64_t timestep)
    {
    this->m_nlist->compute(timestep);

    // The GPU implementation CANNOT handle a half neighborlist, error out now
    bool third_law = this->m_nlist->getStorageMode() == NeighborList::half;
    if (third_law)
        {
        this->m_exec_conf->msg->error()
            << "ai_pair." << evaluator::getName()
            << ": FrictionPairGPU cannot handle a half neighborlist" << std::endl
            << std::endl;
        throw std::runtime_error("Error computing forces in FrictionPairGPU");
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
    ArrayHandle<Scalar4> d_vel(this->m_pdata->getVelocities(),
                               access_location::device,
                               access_mode::read);
    ArrayHandle<Scalar> d_charge(this->m_pdata->getCharges(),
                                 access_location::device,
                                 access_mode::read);
    ArrayHandle<Scalar4> d_orientation(this->m_pdata->getOrientationArray(),
                                       access_location::device,
                                       access_mode::read);
    ArrayHandle<Scalar4> d_angmom(this->m_pdata->getAngularMomentumArray(),
                                  access_location::device,
                                  access_mode::read);
    ArrayHandle<Scalar> d_diameter(this->m_pdata->getDiameters(),
                                   access_location::device,
                                   access_mode::read);
    ArrayHandle<Scalar3> d_moment_inertia(this->m_pdata->getMomentsOfInertiaArray(),
                                          access_location::device,
                                          access_mode::read);
    ArrayHandle<unsigned int> d_tag(this->m_pdata->getTags(),
                                    access_location::device,
                                    access_mode::read);

    BoxDim box = this->m_pdata->getBox();
    unsigned int dim = this->m_sysdef->getNDimensions();

    // uint16_t seed = this->m_sysdef->getSeed();
    // Scalar deltaT = this->m_deltaT;

    // access parameters
    ArrayHandle<Scalar> d_rcutsq(this->m_rcutsq, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_force(this->m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_torque(this->m_torque, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(this->m_virial, access_location::device, access_mode::overwrite);

    // access flags
    PDataFlags flags = this->m_pdata->getFlags();

    this->m_exec_conf->setDevice();

    this->m_tuner->begin();
    unsigned int block_size = this->m_tuner->getParam()[0];
    unsigned int threads_per_particle = this->m_tuner->getParam()[1];

    kernel::gpu_compute_pair_friction_forces<evaluator>(
        kernel::a_pair_args_t(d_force.data,
                              d_torque.data,
                              d_virial.data,
                              this->m_virial.getPitch(),
                              this->m_pdata->getN(),
                              this->m_pdata->getMaxN(),
                              d_pos.data,
                              d_vel.data,
                              d_charge.data,
                              d_orientation.data,
                              d_angmom.data,
                              d_diameter.data,
                              d_moment_inertia.data,
                              d_tag.data,
                              box,
                              third_law,
                              dim,
                              this->m_sysdef->getSeed(),
                              timestep,
                              this->m_deltaT,
                              d_n_neigh.data,
                              d_nlist.data,
                              d_head_list.data,
                              d_rcutsq.data,
                              this->m_pdata->getNTypes(),
                              block_size,
                              flags[pdata_flag::pressure_tensor],
                              threads_per_particle,
                              this->m_exec_conf->dev_prop),
        this->m_params.data());

    this->m_tuner->end();

    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    }

template<class evaluator>
void FrictionPairGPU<evaluator>::setParams(unsigned int typ1,
                                           unsigned int typ2,
                                           const typename evaluator::param_type& param)
    {
    FrictionPair<evaluator>::setParams(typ1, typ2, param);
    this->m_params[this->m_typpair_idx(typ1, typ2)].set_memory_hint();
    this->m_params[this->m_typpair_idx(typ2, typ1)].set_memory_hint();
    }

namespace detail
    {
//! Export this pair potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Class type to export. \b Must be an instantiated FrictionPairGPU class template.
    \tparam Base Base class of \a T. \b Must be Pair<evaluator> with the same evaluator as
   used in \a T.
*/
template<class T> void export_FrictionPairGPU(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<FrictionPairGPU<T>, FrictionPair<T>, std::shared_ptr<FrictionPairGPU<T>>>(
        m,
        name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif // ENABLE_HIP
#endif // __FRICTION_PAIR_GPU_H__
