/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: joaander

/*! \file Analyzer.cc
    \brief Defines the base class Analyzer
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "Analyzer.h"

/*! \param sysdef System definition this analyzer will act on. Must not be NULL.
    \post The Analyzer is constructed with the given particle data and a NULL profiler.
*/
Analyzer::Analyzer(boost::shared_ptr<SystemDefinition> sysdef) : m_sysdef(sysdef), m_pdata(m_sysdef->getParticleData())
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

#ifdef WIN32
#pragma warning( pop )
#endif

