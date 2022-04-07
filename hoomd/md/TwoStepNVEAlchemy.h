// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include "AlchemostatTwoStep.h"

/*! \file TwoStepNVEAlchemy.h
    \brief Declares the TwoStepNVEAlchemy class
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
/*! Implements velocity-verlet NVE integration through the IntegrationMethodTwoStep interface

    \ingroup updaters
*/
class TwoStepNVEAlchemy : public AlchemostatTwoStep
    {
    public:
    //! Constructs the integration method and associates it with the system
    TwoStepNVEAlchemy(std::shared_ptr<SystemDefinition> sysdef, unsigned int alchemTimeFactor);
    virtual ~TwoStepNVEAlchemy();

    //! Performs the first step of the integration
    void integrateStepOne(uint64_t timestep) override;

    //! Performs the second step of the integration
    void integrateStepTwo(uint64_t timestep) override;

    static unsigned int getIntegraorNDOF()
        {
        return 0;
        }
    };

namespace detail
    {
//! Exports the TwoStepNVEAlchemy class to python
void export_TwoStepNVEAlchemy(pybind11::module& m);
    } // end namespace detail

    } // end namespace md

    } // end namespace hoomd
