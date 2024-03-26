// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "TwoStepLangevinBase.h"

#pragma once

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {
/** Integrates part of the system forward in two steps with Brownian dynamics

    Brownian dynamics modifies the Langevin equation by setting the acceleration term to 0 and
    assuming terminal velocity.
*/
class PYBIND11_EXPORT TwoStepBD : public TwoStepLangevinBase
    {
    public:
    /// Constructs the integration method and associates it with the system
    TwoStepBD(std::shared_ptr<SystemDefinition> sysdef,
              std::shared_ptr<ParticleGroup> group,
              std::shared_ptr<Variant> T,
              bool noiseless_t,
              bool noiseless_r);

    virtual ~TwoStepBD();

    /// Performs the second step of the integration
    virtual void integrateStepOne(uint64_t timestep);

    /// Performs the second step of the integration
    virtual void integrateStepTwo(uint64_t timestep);

    protected:
    bool m_noiseless_t;
    bool m_noiseless_r;
    };

    } // end namespace md
    } // end namespace hoomd
