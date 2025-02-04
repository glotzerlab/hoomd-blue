// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef HOOMD_TWOSTEPCONSTANTPRESSUREGPU_H
#define HOOMD_TWOSTEPCONSTANTPRESSUREGPU_H

#include "TwoStepConstantPressure.h"
namespace hoomd::md
    {
class TwoStepConstantPressureGPU : public TwoStepConstantPressure
    {
    public:
    TwoStepConstantPressureGPU(std::shared_ptr<SystemDefinition> sysdef,
                               std::shared_ptr<ParticleGroup> group,
                               std::shared_ptr<ComputeThermo> thermo_full_step,
                               Scalar tauS,
                               const std::vector<std::shared_ptr<Variant>>& S,
                               const std::string& couple,
                               const std::vector<bool>& flags,
                               std::shared_ptr<Thermostat> thermostat,
                               Scalar gamma);

    //! Performs the first step of the integration
    virtual void integrateStepOne(uint64_t timestep);

    //! Performs the second step of the integration
    virtual void integrateStepTwo(uint64_t timestep);

    protected:
    std::shared_ptr<Autotuner<1>> m_tuner_one;     //!< Autotuner for block size (step one kernel)
    std::shared_ptr<Autotuner<1>> m_tuner_two;     //!< Autotuner for block size (step two kernel)
    std::shared_ptr<Autotuner<1>> m_tuner_wrap;    //!< Autotuner for wrapping particle positions
    std::shared_ptr<Autotuner<1>> m_tuner_rescale; //!< Autotuner for thermostat rescaling
    std::shared_ptr<Autotuner<1>> m_tuner_angular_one; //!< Autotuner for angular step one
    std::shared_ptr<Autotuner<1>> m_tuner_angular_two; //!< Autotuner for angular step two
    };

    } // namespace hoomd::md
#endif // HOOMD_TWOSTEPCONSTANTPRESSUREGPU_H
