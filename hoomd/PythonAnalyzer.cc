// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "PythonAnalyzer.h"

#include <exception>
#include <string>

namespace hoomd
    {
PythonAnalyzer::PythonAnalyzer(std::shared_ptr<SystemDefinition> sysdef,
                               std::shared_ptr<Trigger> trigger,
                               pybind11::object analyzer)
    : Analyzer(sysdef, trigger)
    {
    setAnalyzer(analyzer);
    }

void PythonAnalyzer::analyze(uint64_t timestep)
    {
    Analyzer::analyze(timestep);
    m_analyzer.attr("act")(timestep);
    }

void PythonAnalyzer::setAnalyzer(pybind11::object analyzer)
    {
    m_analyzer = analyzer;
    auto flags = PDataFlags();
    for (auto flag : analyzer.attr("flags"))
        {
        flags.set(flag.cast<size_t>());
        }
    m_flags = flags;
    }

PDataFlags PythonAnalyzer::getRequestedPDataFlags()
    {
    return m_flags;
    }

namespace detail
    {
void export_PythonAnalyzer(pybind11::module& m)
    {
    pybind11::class_<PythonAnalyzer, Analyzer, std::shared_ptr<PythonAnalyzer>>(m, "PythonAnalyzer")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<Trigger>,
                            pybind11::object>());
    }

    } // end namespace detail

    } // end namespace hoomd
