// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include "AlchemostatTwoStep.h"
#include "hoomd/Variant.h"

/*! \file TwoStepNVTAlchemy.h
    \brief Declares the TwoStepNVTAlchemy class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/nano-signal-slot/nano_signal_slot.hpp>
#include <pybind11/pybind11.h>

namespace hoomd
    {

namespace md
    {

//! Integrates part of the system forward in two steps in the NVE ensemble
/*! Implements NVT digital alchemy integration through the IntegrationMethodTwoStep interface

    \ingroup updaters
*/
class TwoStepNVTAlchemy : public AlchemostatTwoStep
    {
    public:
    //! Constructs the integration method and associates it with the system
    TwoStepNVTAlchemy(std::shared_ptr<SystemDefinition> sysdef,
                      unsigned int alchemTimeFactor,
                      std::shared_ptr<Variant> T);
    virtual ~TwoStepNVTAlchemy();

    void setQ(Scalar Q)
        {
        m_Q = Q;
        }

    /// get the Q value
    Scalar getQ()
        {
        return m_Q;
        }

    void setT(std::shared_ptr<Variant> T)
        {
        m_T = T;
        }

    /// Get the current temperature variant
    std::shared_ptr<Variant> getT()
        {
        return m_T;
        }

    //! Performs the first step of the integration
    void integrateStepOne(uint64_t timestep) override;

    //! Performs the second step of the integration
    void integrateStepTwo(uint64_t timestep) override;

    protected:
    struct Thermostat
        {
        Scalar xi = 0;
        Scalar eta = 0;
        };

    Scalar m_Q;
    Scalar m_alchem_KE;
    std::shared_ptr<Variant> m_T;
    Thermostat m_thermostat; //!< Thermostat degrees of freedom

    //! advance the thermostat
    /*!\param timestep The time step
     * \param broadcast True if we should broadcast the integrator variables via MPI
     */
    void advanceThermostat(uint64_t timestep, bool broadcast = true);
    };

    } // end namespace md

    } // end namespace hoomd
