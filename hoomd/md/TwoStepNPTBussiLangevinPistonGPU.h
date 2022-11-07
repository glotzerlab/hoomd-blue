//
// Created by girard01 on 11/2/22.
//

#ifndef HOOMD_TWOSTEPNPTBUSSILANGEVINPISTONGPU_H
#define HOOMD_TWOSTEPNPTBUSSILANGEVINPISTONGPU_H

#include "TwoStepNPTBussiLangevinPiston.h"
#include "TwoStepNPTMTTKBaseGPU.h"

namespace hoomd::md{
class TwoStepNPTBussiLangevinPistonGPU : public TwoStepNPTBussiLangevinPiston, public TwoStepNPTMTTKBaseGPU
    {
    public:
        TwoStepNPTBussiLangevinPistonGPU(std::shared_ptr<SystemDefinition> sysdef,
                                     std::shared_ptr<ParticleGroup> group,
                                     std::shared_ptr<ComputeThermo> thermo_half_step,
                                     std::shared_ptr<ComputeThermo> thermo_full_step,
                                     Scalar tauS,
                                     std::shared_ptr<Variant> T,
                                     const std::vector<std::shared_ptr<Variant>>& S,
                                     const std::string& couple,
                                     const std::vector<bool>& flags,
                                     const bool nph = false):
            TwoStepNPTMTTKBase(sysdef, group, thermo_half_step, thermo_full_step, tauS, T, S, couple, flags, nph),
            TwoStepNPTBussiLangevinPiston(sysdef, group, thermo_half_step, thermo_full_step, tauS, T, S, couple, flags, nph),
            TwoStepNPTMTTKBaseGPU(sysdef, group, thermo_half_step, thermo_full_step, tauS, T, S, couple, flags, nph)
            {}

    };
    }

#endif // HOOMD_TWOSTEPNPTBUSSILANGEVINPISTONGPU_H
