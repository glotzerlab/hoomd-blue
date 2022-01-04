// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Maintainer: joaander
#include "Tuner.h"

/*! \file Updater.cc
    \brief Defines a base class for all tuners
*/

namespace hoomd
    {
/*! \param sysdef System this compute will act on. Must not be NULL.
    \post The Updater is constructed with the given particle data and a NULL profiler.
*/
Tuner::Tuner(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<Trigger> trigger)
    : Updater(sysdef), m_trigger(trigger)
    {
    }

namespace detail
    {
void export_Tuner(pybind11::module& m)
    {
    pybind11::class_<Tuner, Updater, std::shared_ptr<Tuner>>(m, "Tuner")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<Trigger>>())
        .def_property("trigger", &Tuner::getTrigger, &Tuner::setTrigger);
    }

    } // end namespace detail

    } // end namespace hoomd
