//
// Created by girard01 on 10/21/22.
//

#ifndef HOOMD_TWOSTEPCONSTANTVOLUME_H
#define HOOMD_TWOSTEPCONSTANTVOLUME_H

#include "ComputeThermo.h"
#include "IntegrationMethodTwoStep.h"
#include "hoomd/Variant.h"
#include <pybind11/pybind11.h>
#include "Thermostat.h"
namespace hoomd::md
    {

class PYBIND11_EXPORT TwoStepConstantVolume : public IntegrationMethodTwoStep
    {
    public:
    TwoStepConstantVolume(std::shared_ptr<SystemDefinition> sysdef,
                          std::shared_ptr<ParticleGroup> group,
                          std::shared_ptr<ComputeThermo> thermo,
                          std::shared_ptr<Thermostat> thermostat) : IntegrationMethodTwoStep(sysdef, group),
                          m_thermo(thermo), m_thermostat(thermostat) {
        m_thermostat->attach(m_group, m_thermo, m_sysdef->getSeed()); // attach thermostat
    }

    virtual ~TwoStepConstantVolume() {}


    //! Performs the first step of the integration
    virtual void integrateStepOne(uint64_t timestep);
    //! Performs the second step of the integration
    virtual void integrateStepTwo(uint64_t timestep);

    void setThermostat(std::shared_ptr<Thermostat> thermostat){
        // attach before resetting m_thermostat in case it throws
        thermostat->attach(m_group, m_thermo, m_sysdef->getSeed());
        // detach old thermostat
        m_thermostat->detach();
        m_thermostat = thermostat;
    }


    protected:
    std::shared_ptr<ComputeThermo> m_thermo; //!< compute for thermodynamic quantities
    std::shared_ptr<Thermostat> m_thermostat;

    };

    } // namespace hoomd::md

#endif // HOOMD_TWOSTEPCONSTANTVOLUME_H
