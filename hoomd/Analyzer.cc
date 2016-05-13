// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file Analyzer.cc
    \brief Defines the base class Analyzer
*/



#include "Analyzer.h"

#include <boost/python.hpp>
using namespace boost::python;

/*! \param sysdef System definition this analyzer will act on. Must not be NULL.
    \post The Analyzer is constructed with the given particle data and a NULL profiler.
*/
Analyzer::Analyzer(boost::shared_ptr<SystemDefinition> sysdef) : m_sysdef(sysdef), m_pdata(m_sysdef->getParticleData()),
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
        (boost::shared_ptr<Profiler>()) to stop the
        analyzer from profiling itself.
    \note Derived classes MUST check if m_prof is set before calling any profiler methods.
*/
void Analyzer::setProfiler(boost::shared_ptr<Profiler> prof)
    {
    m_prof = prof;
    }

//! Wrapper class to expose pure virtual method to python
class AnalyzerWrap: public Analyzer, public wrapper<Analyzer>
    {
    public:
        //! Forwards construction on to the base class
        /*! \param sysdef parameter to forward to the base class constructor
        */
        AnalyzerWrap(boost::shared_ptr<SystemDefinition> sysdef) : Analyzer(sysdef) { }

        //! Hanldes pure virtual Analyzer::analyze()
        /*! \param timestep parameter to forward to Analyzer::analyze()
        */
        void analyze(unsigned int timestep)
            {
            this->get_override("analyze")(timestep);
            }
    };

void export_Analyzer()
    {
    class_<AnalyzerWrap, boost::shared_ptr<AnalyzerWrap>, boost::noncopyable>
        ("Analyzer", init< boost::shared_ptr<SystemDefinition> >())
        .def("analyze", pure_virtual(&Analyzer::analyze))
        .def("setProfiler", &Analyzer::setProfiler)
        ;
    }
