//
// Created by girard01 on 11/2/22.
//

#include "TwoStepNPTBussiLangevinPistonGPU.h"
namespace hoomd::md::detail{
void export_TwoStepNPTBussiLangevinPistonGPU(pybind11::module& m){
    pybind11::class_<TwoStepNPTBussiLangevinPistonGPU, TwoStepNPTBussiLangevinPiston, TwoStepNPTMTTKBaseGPU, std::shared_ptr<TwoStepNPTBussiLangevinPistonGPU>>
        (m, "TwoStepNPTBussiLangevinPistonGPU")
    .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                                std::shared_ptr<ParticleGroup>,
                                std::shared_ptr<ComputeThermo>,
                                std::shared_ptr<ComputeThermo> ,
                                Scalar ,
                                std::shared_ptr<Variant> ,
                                const std::vector<std::shared_ptr<Variant>>& ,
                                const std::string& ,
                                const std::vector<bool>& ,
                                const bool>());
    }
    }