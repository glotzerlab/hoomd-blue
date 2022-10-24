//
// Created by girard01 on 10/24/22.
//

#ifndef HOOMD_TWOSTEPNVTBASEGPU_H
#define HOOMD_TWOSTEPNVTBASEGPU_H

#include "TwoStepNVTBase.h"
#include <hoomd/Autotuner.h>

namespace hoomd::md
        {
    class PYBIND11_EXPORT TwoStepNVTBaseGPU : public virtual TwoStepNVTBase
        {
        public:
            TwoStepNVTBaseGPU(std::shared_ptr<SystemDefinition> sysdef,
                          std::shared_ptr<ParticleGroup> group,
                          std::shared_ptr<ComputeThermo> thermo,
                          std::shared_ptr<Variant> T);

            virtual ~TwoStepNVTBaseGPU(){}

            //! Performs the first step of the integration
            virtual void integrateStepOne(uint64_t timestep);

            //! Performs the second step of the integration
            virtual void integrateStepTwo(uint64_t timestep);

            protected:
            /// Autotuner for block size (step one kernel).
            std::shared_ptr<Autotuner<1>> m_tuner_one;

            /// Autotuner for block size (step two kernel).
            std::shared_ptr<Autotuner<1>> m_tuner_two;

            /// Autotuner_angular for block size (angular step one kernel).
            std::shared_ptr<Autotuner<1>> m_tuner_angular_one;

            /// Autotuner_angular for block size (angular step two kernel).
            std::shared_ptr<Autotuner<1>> m_tuner_angular_two;
        };
        }
#endif // HOOMD_TWOSTEPNVTBASEGPU_H
