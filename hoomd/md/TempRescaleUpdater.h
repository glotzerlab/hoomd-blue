// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file TempRescaleUpdater.h
    \brief Declares an updater that rescales velocities to achieve a set temperature
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include "hoomd/Updater.h"
#include "hoomd/ComputeThermo.h"
#include "hoomd/Variant.h"

#include <memory>

#include <vector>

#ifndef __TEMPRESCALEUPDATER_H__
#define __TEMPRESCALEUPDATER_H__

//! Updates particle velocities to set a temperature
/*! This updater computes the current temperature of the system and then scales the velocities in order to set the
    temperature.

    \ingroup updaters
*/
class PYBIND11_EXPORT TempRescaleUpdater : public Updater
    {
    public:
        //! Constructor
        TempRescaleUpdater(std::shared_ptr<SystemDefinition> sysdef,
                           std::shared_ptr<ComputeThermo> thermo,
                           std::shared_ptr<Variant> tset);

        //! Destructor
        ~TempRescaleUpdater();

        //! Take one timestep forward
        virtual void update(unsigned int timestep);

        //! Change the temperature set point
        void setT(std::shared_ptr<Variant> T);

    private:
        std::shared_ptr<ComputeThermo> m_thermo;  //!< Computes the temperature
        std::shared_ptr<Variant> m_tset;          //!< Temperature set point
    };

//! Export the TempRescaleUpdater to python
void export_TempRescaleUpdater(pybind11::module& m);

#endif
