/*
Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
Source Software License
Copyright (c) 2008 Ames Laboratory Iowa State University
All rights reserved.

Redistribution and use of HOOMD, in source and binary forms, with or
without modification, are permitted, provided that the following
conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names HOOMD's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$

/*! \file Analyzer.cc
	\brief Defines the base class Analyzer
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#ifdef USE_PYTHON
#include <boost/python.hpp>
using namespace boost::python;
#endif

#include "Analyzer.h"

/*! \param pdata Particle data this compute will act on. Must not be NULL.
	\post The Compute is constructed with the given particle data and a NULL profiler.
*/
Analyzer::Analyzer(boost::shared_ptr<ParticleData> pdata) : m_pdata(pdata)
	{
	// sanity check
	assert(pdata);
	assert(pdata->getN() > 0);
	}

/*! It is useful for the user to know where computation time is spent. All Analyzers
	should profile themselves. This method sets the profiler for them to use.
	This method does not need to be called, as Analyzers will not profile themselves
	on a NULL profiler
	\param prof Pointer to a profiler for the compute to use. Set to NULL to stop the 
		compute from profiling itself.
	\note Derived classes MUST check for m_prof != NULL before calling any profiler methods.
*/
void Analyzer::setProfiler(boost::shared_ptr<Profiler> prof)
	{
	m_prof = prof;
	}

#ifdef USE_PYTHON
//! Wrapper class to expose pure virtual method to python
class AnalyzerWrap: public Analyzer, public wrapper<Analyzer>
	{
	public:
		//! Forwards construction on to the base class
		/*! \param pdata parameter to forward to the base class constructor
		*/
		AnalyzerWrap(boost::shared_ptr<ParticleData> pdata) : Analyzer(pdata) { }
		
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
	class_<AnalyzerWrap, boost::shared_ptr<AnalyzerWrap>, boost::noncopyable>("Analyzer", init< boost::shared_ptr<ParticleData> >())
		.def("analyze", pure_virtual(&Analyzer::analyze))
		.def("setProfiler", &Analyzer::setProfiler)
		;
	}
#endif

#ifdef WIN32
#pragma warning( pop )
#endif
