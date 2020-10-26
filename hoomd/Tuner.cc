// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: joaander
#include "Tuner.h"

namespace py = pybind11;

/*! \file Updater.cc
    \brief Defines a base class for all tuners
*/

/*! \param sysdef System this compute will act on. Must not be NULL.
    \post The Updater is constructed with the given particle data and a NULL profiler.
*/
Tuner::Tuner(std::shared_ptr<SystemDefinition> sysdef,
             std::shared_ptr<Trigger> trigger)
    : Updater(sysdef), m_trigger(trigger) {}


void export_Tuner(py::module& m)
    {
    py::class_<Tuner, Updater, std::shared_ptr<Tuner> >(m,"Tuner")
    .def(py::init< std::shared_ptr<SystemDefinition>,
                   std::shared_ptr<Trigger> >())
    .def_property("trigger", &Tuner::getTrigger, &Tuner::setTrigger)
    ;
    }
