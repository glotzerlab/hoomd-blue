//
// Created by girard01 on 10/27/22.
//

#ifndef HOOMD_TWOSTEPNPTMTTKBASEGPU_H
#define HOOMD_TWOSTEPNPTMTTKBASEGPU_H

#include "TwoStepNPTMTTKBase.h"
namespace hoomd::md
    {
class TwoStepNPTMTTKBaseGPU : public virtual TwoStepNPTMTTKBase
    {
    public:
    TwoStepNPTMTTKBaseGPU(std::shared_ptr<SystemDefinition> sysdef,
                      std::shared_ptr<ParticleGroup> group,
                      std::shared_ptr<ComputeThermo> thermo_half_step,
                      std::shared_ptr<ComputeThermo> thermo_full_step,
                          Scalar tauS,
                      std::shared_ptr<Variant> T,
                      const std::vector<std::shared_ptr<Variant>>& S,
                      const std::string& couple,
                      const std::vector<bool>& flags,
                      const bool nph = false);

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

    }
#endif // HOOMD_TWOSTEPNPTMTTKBASEGPU_H
