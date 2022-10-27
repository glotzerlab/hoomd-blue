//
// Created by girard01 on 10/25/22.
//

#ifndef HOOMD_TWOSTEPNVTSTOCHASTICGPU_H
#define HOOMD_TWOSTEPNVTSTOCHASTICGPU_H
#include "TwoStepNVTBaseGPU.h"
#include "TwoStepNVTStochastic.h"
namespace hoomd::md
    {
class PYBIND11_EXPORT TwoStepNVTStochasticGPU : public TwoStepNVTStochastic, public TwoStepNVTBaseGPU
    {
    public:
        TwoStepNVTStochasticGPU(std::shared_ptr<SystemDefinition> sysdef,
                            std::shared_ptr<ParticleGroup> group,
                            std::shared_ptr<ComputeThermo> thermo,
                            std::shared_ptr<Variant> T);

        virtual ~TwoStepNVTStochasticGPU(){}
    };
    }

#endif // HOOMD_TWOSTEPNVTSTOCHASTICGPU_H
