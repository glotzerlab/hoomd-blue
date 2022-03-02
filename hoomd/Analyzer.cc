// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file Analyzer.cc
    \brief Defines the base class Analyzer
*/

#include "Analyzer.h"

namespace hoomd
    {
/*! \param sysdef System definition this analyzer will act on. Must not be NULL.
    \post The Analyzer is constructed with the given particle data.
*/
Analyzer::Analyzer(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<Trigger> trigger)
    : m_sysdef(sysdef), m_pdata(m_sysdef->getParticleData()), m_exec_conf(m_pdata->getExecConf()),
      m_trigger(trigger)
    {
    // sanity check
    assert(m_sysdef);
    assert(m_pdata);
    }

namespace detail
    {
void export_Analyzer(pybind11::module& m)
    {
    pybind11::class_<Analyzer, std::shared_ptr<Analyzer>>(m, "Analyzer")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<Trigger>>())
        .def("analyze", &Analyzer::analyze)
        .def("notifyDetach", &Analyzer::notifyDetach)
        .def_property("trigger", &Analyzer::getTrigger, &Analyzer::setTrigger);
    }

    } // end namespace detail

    } // end namespace hoomd
