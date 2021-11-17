// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: joaander
#include "Updater.h"

/*! \file Updater.cc
    \brief Defines a base class for all updaters
*/

namespace hoomd
    {
/*! \param sysdef System this compute will act on. Must not be NULL.
    \post The Updater is constructed with the given particle data and a NULL profiler.
*/
Updater::Updater(std::shared_ptr<SystemDefinition> sysdef)
    : m_sysdef(sysdef), m_pdata(m_sysdef->getParticleData()), m_exec_conf(m_pdata->getExecConf())
    {
    // sanity check
    assert(m_sysdef);
    assert(m_pdata);
    }

/*! It is useful for the user to know where computation time is spent, so all Updaters
    should profile themselves. This method sets the profiler for them to use.
    This method does not need to be called, as Updaters will not profile themselves
    on a NULL profiler
    \param prof Pointer to a profiler for the compute to use. Set to NULL
        (std::shared_ptr<Profiler>()) to stop the
        analyzer from profiling itself.
    \note Derived classes MUST check if m_prof is set before calling any profiler methods.
*/
void Updater::setProfiler(std::shared_ptr<Profiler> prof)
    {
    m_prof = prof;
    }

namespace detail
    {
void export_Updater(pybind11::module& m)
    {
    pybind11::class_<Updater, std::shared_ptr<Updater>>(m, "Updater")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
        .def("update", &Updater::update)
        .def("setProfiler", &Updater::setProfiler)
        .def("notifyDetach", &Updater::notifyDetach);
    }

    } // end namespace detail

    } // end namespace hoomd
