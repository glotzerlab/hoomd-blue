// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file Analyzer.cc
    \brief Defines the base class Analyzer
*/



#include "Analyzer.h"

namespace py = pybind11;

/*! \param sysdef System definition this analyzer will act on. Must not be NULL.
    \post The Analyzer is constructed with the given particle data and a NULL profiler.
*/
Analyzer::Analyzer(std::shared_ptr<SystemDefinition> sysdef) : m_sysdef(sysdef), m_pdata(m_sysdef->getParticleData()),
    m_exec_conf(m_pdata->getExecConf())
    {
    // sanity check
    assert(m_sysdef);
    assert(m_pdata);
    }

/*! It is useful for the user to know where computation time is spent, so all Analyzers
    should profile themselves. This method sets the profiler for them to use.
    This method does not need to be called, as Analyzers will not profile themselves
    on a NULL profiler
    \param prof Pointer to a profiler for the compute to use. Set to NULL
        (std::shared_ptr<Profiler>()) to stop the
        analyzer from profiling itself.
    \note Derived classes MUST check if m_prof is set before calling any profiler methods.
*/
void Analyzer::setProfiler(std::shared_ptr<Profiler> prof)
    {
    m_prof = prof;
    }

void export_Analyzer(py::module& m)
    {
    py::class_<Analyzer, std::shared_ptr<Analyzer>>(m,"Analyzer")
        .def(py::init< std::shared_ptr<SystemDefinition> >())
        .def("analyze", &Analyzer::analyze)
        .def("setProfiler", &Analyzer::setProfiler)
        ;
    }
