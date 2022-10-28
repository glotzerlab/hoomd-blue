//
// Created by girard01 on 10/28/22.
//

#ifndef HOOMD_TWOSTEPNPTLANGEVINPISTONGPU_H
#define HOOMD_TWOSTEPNPTLANGEVINPISTONGPU_H

#include "TwoStepNPTLangevinPiston.h"
#include "TwoStepNPTBaseGPU.h"

namespace hoomd::md{
class TwoStepNPTLangevinPistonGPU : public TwoStepNPTLangevinPiston, public TwoStepNPTBaseGPU
    {
    public:
    TwoStepNPTLangevinPistonGPU(std::shared_ptr<SystemDefinition> sysdef,
                                std::shared_ptr<ParticleGroup> group,
                                std::shared_ptr<ComputeThermo> thermo_half_step,
                                std::shared_ptr<ComputeThermo> thermo_full_step,
                                Scalar tau,
                                Scalar tauS,
                                std::shared_ptr<Variant> T,
                                const std::vector<std::shared_ptr<Variant>>& S,
                                const std::string& couple,
                                const std::vector<bool>& flags,
                                const bool nph = false) :
          TwoStepNPTBase(sysdef, group, thermo_half_step, thermo_full_step, T, S, couple, flags, nph),
          TwoStepNPTMTK(sysdef, group, thermo_half_step, thermo_full_step, tau, tauS, T, S, couple, flags, nph),
          TwoStepNPTLangevinPiston(sysdef, group, thermo_half_step, thermo_full_step, tau, tauS, T, S, couple, flags, nph),
          TwoStepNPTBaseGPU(sysdef, group, thermo_half_step, thermo_full_step, T, S, couple, flags, nph)
        {}
    };

    }

#endif // HOOMD_TWOSTEPNPTLANGEVINPISTONGPU_H
