#include "PythonAnalyzer.h"

#include <exception>
#include <string>

PythonAnalyzer::PythonAnalyzer(std::shared_ptr<SystemDefinition> sysdef,
                               pybind11::object analyzer) : Analyzer(sysdef)
    {
    setAnalyzer(analyzer);
    }

void PythonAnalyzer::analyze(unsigned int timestep)
    {
    m_analyzer.attr("act")(timestep);
    }

void PythonAnalyzer::setAnalyzer(pybind11::object analyzer)
    {
    m_analyzer = analyzer;
    auto flags = PDataFlags();
    for (auto flag: analyzer.attr("flags"))
        {
        flags.set(flag.cast<size_t>());
        }
    m_flags = flags;
    }

PDataFlags PythonAnalyzer::getRequestedPDataFlags()
    {
    return m_flags;
    }

void export_PythonAnalyzer(pybind11::module& m)
    {
    pybind11::class_<PythonAnalyzer, Analyzer, std::shared_ptr<PythonAnalyzer>
                    >(m, "PythonAnalyzer")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            pybind11::object>())
        ;
    }
