// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "TwoStepBD.h"

#pragma once

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {
//! Implements Brownian dynamics on the GPU
/*! GPU accelerated version of TwoStepBD

    \ingroup updaters
*/
class PYBIND11_EXPORT TwoStepBDGPU : public TwoStepBD
    {
    public:
    //! Constructs the integration method and associates it with the system
    TwoStepBDGPU(std::shared_ptr<SystemDefinition> sysdef,
                 std::shared_ptr<ParticleGroup> group,
                 std::shared_ptr<Variant> T,
                 bool noiseless_t,
                 bool noiseless_r);

    virtual ~TwoStepBDGPU() {};

    //! Performs the first step of the integration
    virtual void integrateStepOne(uint64_t timestep);

    //! Performs the second step of the integration
    virtual void integrateStepTwo(uint64_t timestep);

    protected:
    /// Autotuner for block size.
    std::shared_ptr<Autotuner<1>> m_tuner;
    };

    } // end namespace md
    } // end namespace hoomd
