//
// Created by girard01 on 10/28/22.
//

#ifndef HOOMD_TWOSTEPNPTLANGEVINPISTON_H
#define HOOMD_TWOSTEPNPTLANGEVINPISTON_H

#include "TwoStepNPTMTK.h"

namespace hoomd::md
    {

class TwoStepNPTLangevinPiston : public virtual TwoStepNPTMTK
    {
    public:
    TwoStepNPTLangevinPiston(std::shared_ptr<SystemDefinition> sysdef,
                             std::shared_ptr<ParticleGroup> group,
                             std::shared_ptr<ComputeThermo> thermo_half_step,
                             std::shared_ptr<ComputeThermo> thermo_full_step,
                             Scalar tau,
                             Scalar tauS,
                             std::shared_ptr<Variant> T,
                             const std::vector<std::shared_ptr<Variant>>& S,
                             const std::string& couple,
                             const std::vector<bool>& flags,
                             const bool nph = false)
        : TwoStepNPTBase(sysdef, group, thermo_half_step, thermo_full_step, T, S, couple, flags, nph),
          TwoStepNPTMTK(sysdef, group, thermo_half_step, thermo_full_step, tau, tauS, T, S, couple, flags, nph){}
    protected:
    virtual void advanceBarostat(uint64_t);

    };
    }

#endif // HOOMD_TWOSTEPNPTLANGEVINPISTON_H
