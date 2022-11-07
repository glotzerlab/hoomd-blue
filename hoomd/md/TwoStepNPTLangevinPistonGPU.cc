//
// Created by girard01 on 10/28/22.
//

#include "TwoStepNPTLangevinPistonGPU.h"

namespace hoomd::md::detail{
void export_TwoStepNPTLangevinPistonGPU(pybind11::module& m){
    pybind11::class_<TwoStepNPTLangevinPistonGPU, TwoStepNPTLangevinPiston,
                     TwoStepNPTMTTKBaseGPU, std::shared_ptr<TwoStepNPTLangevinPistonGPU>>(m, "TwoStepNPTLangevinPistonGPU")
    .def(pybind11::init<std::shared_ptr<SystemDefinition> ,
                            std::shared_ptr<ParticleGroup> ,
                            std::shared_ptr<ComputeThermo> ,
                            std::shared_ptr<ComputeThermo> ,
                            Scalar ,
                            Scalar ,
                            std::shared_ptr<Variant> ,
                            const std::vector<std::shared_ptr<Variant>>& ,
                            const std::string& ,
                            const std::vector<bool>& ,
                            const bool >());
    }
    }
