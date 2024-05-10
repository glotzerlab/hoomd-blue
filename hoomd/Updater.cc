// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "Updater.h"

/*! \file Updater.cc
    \brief Defines a base class for all updaters
*/

namespace hoomd
    {
/*! \param sysdef System this compute will act on. Must not be NULL.
    \post The Updater is constructed with the given particle data.
*/
Updater::Updater(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<Trigger> trigger)
    : Action(sysdef), m_trigger(trigger)
    {
    // sanity check
    assert(m_sysdef);
    assert(m_pdata);
    }

namespace detail
    {
void export_Updater(pybind11::module& m)
    {
    pybind11::class_<Updater, Action, std::shared_ptr<Updater>>(m, "Updater")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<Trigger>>())
        .def("update", &Updater::update)
        .def("notifyDetach", &Updater::notifyDetach)
        .def_property("trigger", &Updater::getTrigger, &Updater::setTrigger);
    }

    } // end namespace detail

    } // end namespace hoomd
