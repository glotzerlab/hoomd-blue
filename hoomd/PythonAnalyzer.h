// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include <pybind11/pybind11.h>

#include "Analyzer.h"

namespace hoomd
    {
class PYBIND11_EXPORT PythonAnalyzer : public Analyzer
    {
    public:
    PythonAnalyzer(std::shared_ptr<SystemDefinition> sysdef,
                   std::shared_ptr<Trigger> trigger,
                   pybind11::object analyzer);

    void analyze(uint64_t timestep);

    PDataFlags getRequestedPDataFlags();

    void setAnalyzer(pybind11::object analyzer);

    pybind11::object getAnalyzer()
        {
        return m_analyzer;
        }

    protected:
    pybind11::object m_analyzer;
    PDataFlags m_flags;
    };

namespace detail
    {
void export_PythonAnalyzer(pybind11::module& m);

    } // end namespace detail

    } // end namespace hoomd
