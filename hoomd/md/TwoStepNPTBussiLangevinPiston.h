//
// Created by girard01 on 10/28/22.
//

#ifndef HOOMD_TWOSTEPNPTBUSSILANGEVINPISTON_H
#define HOOMD_TWOSTEPNPTBUSSILANGEVINPISTON_H

#include "TwoStepNPTLangevinPiston.h"
#include <hoomd/RandomNumbers.h>
#include <hoomd/RNGIdentifiers.h>

namespace hoomd::md{
class TwoStepNPTBussiLangevinPiston : public virtual TwoStepNPTMTTKBase{
    public:

    TwoStepNPTBussiLangevinPiston(std::shared_ptr<SystemDefinition> sysdef,
                                  std::shared_ptr<ParticleGroup> group,
                                  std::shared_ptr<ComputeThermo> thermo_half_step,
                                  std::shared_ptr<ComputeThermo> thermo_full_step,
                                  Scalar tauS,
                                  std::shared_ptr<Variant> T,
                                  const std::vector<std::shared_ptr<Variant>>& S,
                                  const std::string& couple,
                                  const std::vector<bool>& flags,
                                  const bool nph = false);

    virtual std::array<Scalar, 2> NPT_thermo_rescale_factor_one(uint64_t timestep){

        std::array<Scalar, 2> thermo_values {Scalar(1.), Scalar(1.)};
        if(!m_nph)
            {
            m_thermo_half_step->compute(timestep);
            const auto ntdof = m_thermo_half_step->getTranslationalDOF();
            const auto nrdof = m_thermo_half_step->getRotationalDOF();

            RandomGenerator rng(
                Seed(RNGIdentifier::StochasticVRescale, timestep, m_sysdef->getSeed()),
                0);

            const auto set_T = (*m_T)(timestep);
            GammaDistribution<double> gamma_translation(ntdof / 2.0, set_T);
            GammaDistribution<double> gamma_rotation(nrdof / 2.0, set_T);

            thermo_values = {
                std::sqrt(gamma_translation(rng) / m_thermo_half_step->getTranslationalKineticEnergy()),
                std::sqrt(gamma_rotation(rng) / m_thermo_half_step->getRotationalKineticEnergy())
            };

            }
        const auto& mtk_terms = TwoStepNPTMTTKBase::NPT_thermo_rescale_factor_one(timestep);

        return{
            thermo_values[0] * mtk_terms[0], thermo_values[1] * mtk_terms[1]
        };

        }

    };
    }

#endif // HOOMD_TWOSTEPNPTBUSSILANGEVINPISTON_H
