// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: jproc

#include "AlchemostatTwoStep.h"
#include "hoomd/Variant.h"

#ifndef __TWO_STEP_NVE_ALCHEMO_H__
#define __TWO_STEP_NVE_ALCHEMO_H__

/*! \file TwoStepNVTAlchemy.h
    \brief Declares the TwoStepNVTAlchemy class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/nano-signal-slot/nano_signal_slot.hpp>
#include <pybind11/pybind11.h>

//! Integrates part of the system forward in two steps in the NVE ensemble
/*! Implements NVT digital alchemy integration through the IntegrationMethodTwoStep interface

    \ingroup updaters
*/
class PYBIND11_EXPORT TwoStepNVTAlchemy : public AlchemostatTwoStep
    {
    public:
    //! Constructs the integration method and associates it with the system
    TwoStepNVTAlchemy(std::shared_ptr<SystemDefinition> sysdef,
                      std::shared_ptr<Variant> T,
                      const std::string& suffix);
    virtual ~TwoStepNVTAlchemy();

    virtual void setQ(Scalar Q)
        {
        m_Q = Q;
        }

    virtual void setT(std::shared_ptr<Variant> T)
        {
        m_T = T;
        }


    //! Returns a list of log quantities this integrator calculates
    virtual std::vector<std::string> getProvidedLogQuantities();

    //! Returns logged values
    Scalar getLogValue(const std::string& quantity, uint64_t timestep, bool& my_quantity_flag);

    //! Performs the first step of the integration
    void integrateStepOne(uint64_t timestep) override;

    //! Performs the second step of the integration
    void integrateStepTwo(uint64_t timestep) override;

    private:
    Scalar m_Q;
    Scalar m_alchem_KE;
    std::shared_ptr<Variant> m_T;
    //! advance the thermostat
    /*!\param timestep The time step
     * \param broadcast True if we should broadcast the integrator variables via MPI
     */
    void advanceThermostat(uint64_t timestep, bool broadcast = true);
    };

//! Exports the TwoStepNVTAlchemy class to python
void export_TwoStepNVTAlchemy(pybind11::module& m);

#endif // #ifndef __TWO_STEP_NVE_ALCHEMO_H__
