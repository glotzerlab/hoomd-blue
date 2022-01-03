#pragma once

#include <pybind11/pybind11.h>

#include "Analyzer.h"

namespace hoomd
    {
class PYBIND11_EXPORT PythonAnalyzer : public Analyzer
    {
    public:
    PythonAnalyzer(std::shared_ptr<SystemDefinition> sysdef, pybind11::object analyzer);

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
