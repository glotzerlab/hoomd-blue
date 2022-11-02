//
// Created by girard01 on 10/28/22.
//

#include "TwoStepNPTBussiLangevinPiston.h"
namespace hoomd::md{

TwoStepNPTBussiLangevinPiston::TwoStepNPTBussiLangevinPiston(std::shared_ptr<SystemDefinition> sysdef,
                                                             std::shared_ptr<ParticleGroup> group,
                                                             std::shared_ptr<ComputeThermo> thermo_half_step,
                                                             std::shared_ptr<ComputeThermo> thermo_full_step,
                                                             Scalar tauS,
                                                             std::shared_ptr<Variant> T,
                                                             const std::vector<std::shared_ptr<Variant>>& S,
                                                             const std::string& couple,
                                                             const std::vector<bool>& flags,
                                                             const bool nph) :
      TwoStepNPTMTTKBase(sysdef, group, thermo_half_step, thermo_full_step, tauS, T, S, couple, flags, nph)
    {}

namespace detail{
void export_TwoStepNPTBussiLangevinPiston(pybind11::module& m){
    pybind11::class_<TwoStepNPTBussiLangevinPiston, TwoStepNPTMTTKBase, std::shared_ptr<TwoStepNPTBussiLangevinPiston>>
        (m, "TwoStepNPTBussiLangevinPiston")
    .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                                std::shared_ptr<ParticleGroup>,
                                std::shared_ptr<ComputeThermo>,
                                std::shared_ptr<ComputeThermo>,
                                Scalar,
                                std::shared_ptr<Variant>,
                                const std::vector<std::shared_ptr<Variant>>,
                                const std::string&,
                                const std::vector<bool>&,
                                const bool>());
    }
    }

    }