//
// Created by girard01 on 10/25/22.
//

#include "TwoStepNVTStochastic.h"
#include <hoomd/RandomNumbers.h>
#include <hoomd/RNGIdentifiers.h>

namespace hoomd::md{
TwoStepNVTStochastic::TwoStepNVTStochastic(std::shared_ptr<SystemDefinition> sysdef,
                                           std::shared_ptr<ParticleGroup> group,
                                           std::shared_ptr<ComputeThermo> thermo,
                                           std::shared_ptr<Variant> T):
      TwoStepNVTBase(sysdef, group, thermo, T)
    {}

    std::array<Scalar, 2> TwoStepNVTStochastic::NVT_rescale_factor_one(uint64_t timestep){

        m_thermo->compute(timestep);

        const auto ntdof = m_thermo->getTranslationalDOF();
        const auto nrdof = m_thermo->getRotationalDOF();

        RandomGenerator rng(Seed(RNGIdentifier::StochasticVRescale, timestep,m_sysdef->getSeed()), 0);

        const auto set_T = (*m_T)(timestep);
        GammaDistribution<double> gamma_translation(ntdof / 2.0, set_T);
        GammaDistribution<double> gamma_rotation(nrdof / 2.0, set_T);

        return{
            std::sqrt(gamma_translation(rng) / m_thermo->getTranslationalKineticEnergy()),
            std::sqrt(gamma_rotation(rng) / m_thermo->getRotationalKineticEnergy())
            };
    }

    namespace detail{
        void export_TwoStepNVTStochastic(pybind11::module& m){
            pybind11::class_<TwoStepNVTStochastic, TwoStepNVTBase, std::shared_ptr<TwoStepNVTStochastic>>(m, "TwoStepNVTStochastic")
            .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>, std::shared_ptr<ComputeThermo>, std::shared_ptr<Variant>>());
            }
        }

    }