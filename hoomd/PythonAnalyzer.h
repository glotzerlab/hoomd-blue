#include <pybind11/pybind11.h>
#include <memory>

#include "Analyzer.h"

class PYBIND11_EXPORT PythonAnalyzer : public Analyzer
{
    public:
        PythonAnalyzer(std::shared_ptr<SystemDefinition> sysdef,
                       pybind11::object analyzer);

        void analyze(unsigned int timestep);

        PDataFlags getRequestedPDataFlags();

        void setAnalyzer(pybind11::object analyzer);

        pybind11::object getAnalyzer() {return m_analyzer;}

    protected:
        pybind11::object m_analyzer;
        PDataFlags m_flags;
};

void export_PythonAnalyzer(pybind11::module& m);
