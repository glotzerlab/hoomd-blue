//
// Created by girard01 on 10/25/22.
//

#ifndef HOOMD_TWOSTEPNVTSTOCHASTIC_H
#define HOOMD_TWOSTEPNVTSTOCHASTIC_H

#include "TwoStepNVTBase.h"
namespace hoomd::md
    {
class PYBIND11_EXPORT TwoStepNVTStochastic : public virtual TwoStepNVTBase
    {
    public:
    TwoStepNVTStochastic(std::shared_ptr<SystemDefinition> sysdef,
                         std::shared_ptr<ParticleGroup> group,
                         std::shared_ptr<ComputeThermo> thermo,
                         std::shared_ptr<Variant> T);

    virtual ~TwoStepNVTStochastic(){}

    virtual std::array<Scalar, 2> NVT_rescale_factor_one(uint64_t timestep);
    };

    }

#endif // HOOMD_TWOSTEPNVTSTOCHASTIC_H
