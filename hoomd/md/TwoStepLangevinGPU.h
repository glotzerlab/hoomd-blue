// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "TwoStepLangevin.h"
#include "hoomd/Autotuner.h"

#pragma once

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {
//! Implements Langevin dynamics on the GPU
/*! GPU accelerated version of TwoStepLangevin

    \ingroup updaters
*/
class PYBIND11_EXPORT TwoStepLangevinGPU : public TwoStepLangevin
    {
    public:
    //! Constructs the integration method and associates it with the system
    TwoStepLangevinGPU(std::shared_ptr<SystemDefinition> sysdef,
                       std::shared_ptr<ParticleGroup> group,
                       std::shared_ptr<Variant> T);
    virtual ~TwoStepLangevinGPU() {};

    //! Performs the first step of the integration
    virtual void integrateStepOne(uint64_t timestep);

    //! Performs the second step of the integration
    virtual void integrateStepTwo(uint64_t timestep);

    protected:
    GPUVector<Scalar> m_partial_sum1; //!< memory space for partial sum over bd energy transfers
    GPUArray<Scalar> m_sum;           //!< memory space for sum over bd energy transfers

    /// Autotuner for block size (step one kernel)
    std::shared_ptr<Autotuner<1>> m_tuner_one;

    /// Autotuner for block size (step two kernel)
    std::shared_ptr<Autotuner<1>> m_tuner_two;

    /// Autotuner for block size (angular step one kernel).
    std::shared_ptr<Autotuner<1>> m_tuner_angular_one;

    /// Autotuner for block size (angular step two kernel).
    std::shared_ptr<Autotuner<1>> m_tuner_angular_two;
    };

    } // end namespace md
    } // end namespace hoomd
