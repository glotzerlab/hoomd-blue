//
// Created by girard01 on 10/25/22.
//

#include "TwoStepNVTStochasticGPU.h"
namespace hoomd::md{
TwoStepNVTStochasticGPU::TwoStepNVTStochasticGPU(std::shared_ptr<SystemDefinition> sysdef,
                                                 std::shared_ptr<ParticleGroup> group,
                                                 std::shared_ptr<ComputeThermo> thermo,
                                                 std::shared_ptr<Variant> T):
      TwoStepNVTBase(sysdef, group, thermo, T),
      TwoStepNVTStochastic(sysdef, group, thermo, T),
      TwoStepNVTBaseGPU(sysdef, group, thermo, T)
    {}

namespace detail{
void export_TwoStepNVTStochasticGPU(pybind11::module& m){
    pybind11::class_<TwoStepNVTStochasticGPU, TwoStepNVTStochastic, TwoStepNVTBaseGPU, std::shared_ptr<TwoStepNVTStochasticGPU>>(m, "TwoStepNVTStochasticGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>, std::shared_ptr<ComputeThermo>, std::shared_ptr<Variant>>());
    }
    }

    }
