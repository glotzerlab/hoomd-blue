// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef HOOMD_TWOSTEPCONSTANTVOLUMEGPU_H
#define HOOMD_TWOSTEPCONSTANTVOLUMEGPU_H

#include "Thermostat.h"
#include "TwoStepConstantVolume.h"
#include <hoomd/Autotuner.h>

namespace hoomd::md
    {

/// Implement TwoStepConstantVolume on the GPU.
class PYBIND11_EXPORT TwoStepConstantVolumeGPU : public TwoStepConstantVolume
    {
    public:
    TwoStepConstantVolumeGPU(std::shared_ptr<SystemDefinition> sysdef,
                             std::shared_ptr<ParticleGroup> group,
                             std::shared_ptr<Thermostat> thermostat);

    virtual ~TwoStepConstantVolumeGPU() { }

    virtual void integrateStepOne(uint64_t timestep);

    virtual void integrateStepTwo(uint64_t timestep);

    protected:
    /// Autotuner for block size (step one kernel).
    std::shared_ptr<Autotuner<1>> m_tuner_one;

    /// Autotuner for block size (step two kernel).
    std::shared_ptr<Autotuner<1>> m_tuner_two;

    /// Autotuner_angular for block size (angular step one kernel).
    std::shared_ptr<Autotuner<1>> m_tuner_angular_one;

    /// Autotuner_angular for block size (angular step two kernel).
    std::shared_ptr<Autotuner<1>> m_tuner_angular_two;
    };
    } // namespace hoomd::md
#endif // HOOMD_TWOSTEPCONSTANTVOLUMEGPU_H
