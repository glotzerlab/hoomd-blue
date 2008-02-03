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

#include <iostream>

//! Name the unit test module
#define BOOST_TEST_MODULE SystemClassTest
#include "boost_utf_configure.h"

#include <math.h>
#include "System.h"

#include <stdexcept>
#include <string>
#include <iostream>

/*! \file system_test.cc
	\brief Unit tests for System
	\ingroup unit_tests
*/

using namespace std;

// we are going to need a dummy compute, a dummy analyzer and a dummy updater in order to test the system

//! Dummy analyzer for unit testing System
class DummyAnalyzer : public Analyzer
	{
	public:
		//! Constructs a named analyzer
		DummyAnalyzer(boost::shared_ptr<ParticleData> pdata, const string& name)
			: Analyzer(pdata), m_name(name)
			{
			}
	
		//! Just prints our name and the current time step
		void analyze(unsigned int timestep)
			{
			if (m_prof)
				m_prof->push(m_name);
			cout << m_name << ": " << timestep << endl;
			Sleep(50);
			if (m_prof)
				m_prof->pop();
			}
	private:
		string m_name;	//!< Name of the dummy
	};
	
//! Dummy updater for unit testing System
class DummyUpdater : public Integrator
	{
	// this derives from Integrator so the unit tests can use them in setIntegrator
	public:
		//! Constructs a named analyzer
		DummyUpdater(boost::shared_ptr<ParticleData> pdata, const string& name)
			: Integrator(pdata, 0.0), m_name(name)
			{
			}
	
		//! Just prints our name and the current time step
		void update(unsigned int timestep)
			{
			if (m_prof)
				m_prof->push(m_name);
			cout << m_name << ": " << timestep << endl;
			Sleep(80);
			if (m_prof)
				m_prof->pop();
			}
	private:
		string m_name;	//!< Name of the dummy
	};
	
//! Dummy updater for unit testing System
class DummyCompute : public Compute
	{
	public:
		//! Constructs a named analyzer
		DummyCompute(boost::shared_ptr<ParticleData> pdata, const string& name)
			: Compute(pdata), m_name(name)
			{
			}
	
		//! Just prints our name and the current time step
		void compute(unsigned int timestep)
			{
			if (m_prof)
				m_prof->push(m_name);
			cout << m_name << ": " << timestep << endl;
			Sleep(80);
			if (m_prof)
				m_prof->pop();
			}
	private:
		string m_name;	//!< Name of the dummy
	};
			
//! Tests the add, get, and set routines in System
BOOST_AUTO_TEST_CASE( getter_setter_tests )
	{
	boost::shared_ptr< ParticleData > pdata(new ParticleData(10, BoxDim(10)));
	
	cout << "Running getter_setter_tests, expect error messages to be printed to the screen as this tests error checking in System" << endl;
	
	// create two analyzers to test adding
	boost::shared_ptr< Analyzer > analyzer1(new DummyAnalyzer(pdata, "analyzer1"));
	boost::shared_ptr< Analyzer > analyzer2(new DummyAnalyzer(pdata, "analyzer2"));
	
	// add them both to a System
	System sys(pdata, 0);
	sys.addAnalyzer(analyzer1, "analyzer1", 10);
	sys.addAnalyzer(analyzer2, "analyzer2", 105);

	// test adding another of the same name
	bool except = false;
	try { sys.addAnalyzer(analyzer2, "analyzer1", 1); }
	catch (runtime_error) { except = true; }
	BOOST_CHECK(except);
	
	// check the get method
	BOOST_CHECK_EQUAL(sys.getAnalyzer("analyzer1"), analyzer1);
	BOOST_CHECK_EQUAL(sys.getAnalyzer("analyzer2"), analyzer2);
	except = false;
	try { sys.getAnalyzer("analyzer3"); }
	catch (runtime_error) { except = true; }
	BOOST_CHECK(except);
	
	// test the get and set period functions
	BOOST_CHECK_EQUAL(sys.getAnalyzerPeriod("analyzer2"), (unsigned int)105);
	BOOST_CHECK_EQUAL(sys.getAnalyzerPeriod("analyzer1"), (unsigned int)10);
	except = false;
	try { sys.getAnalyzerPeriod("analyzer3"); }
	catch (runtime_error) { except = true; }
	BOOST_CHECK(except);
	
	sys.setAnalyzerPeriod("analyzer1", 15);
	sys.setAnalyzerPeriod("analyzer2", 8);
	BOOST_CHECK_EQUAL(sys.getAnalyzerPeriod("analyzer2"), (unsigned int)8);
	BOOST_CHECK_EQUAL(sys.getAnalyzerPeriod("analyzer1"), (unsigned int)15);
	
	// remove the analyzers 1 by one and make sure they don't exist
	sys.removeAnalyzer("analyzer1");
	BOOST_CHECK_EQUAL(sys.getAnalyzer("analyzer2"), analyzer2);
	except = false;
	try { sys.getAnalyzer("analyzer1"); }
	catch (runtime_error) { except = true; }
	BOOST_CHECK(except);
	
	sys.removeAnalyzer("analyzer2");
	except = false;
	try { sys.getAnalyzer("analyzer2"); }
	catch (runtime_error) { except = true; }
	BOOST_CHECK(except);
	
	// ************ Updaters
	// create two updaters to test adding
	boost::shared_ptr< Updater > updater1(new DummyUpdater(pdata, "updater1"));
	boost::shared_ptr< Updater > updater2(new DummyUpdater(pdata, "updater2"));
	
	// add them both to a System
	sys.addUpdater(updater1, "updater1", 10);
	sys.addUpdater(updater2, "updater2", 105);

	// test adding another of the same name
	except = false;
	try { sys.addUpdater(updater2, "updater1", 1); }
	catch (runtime_error) { except = true; }
	BOOST_CHECK(except);
	
	// check the get method
	BOOST_CHECK_EQUAL(sys.getUpdater("updater1"), updater1);
	BOOST_CHECK_EQUAL(sys.getUpdater("updater2"), updater2);
	except = false;
	try { sys.getUpdater("updater3"); }
	catch (runtime_error) { except = true; }
	BOOST_CHECK(except);
	
	// test the get and set period functions
	BOOST_CHECK_EQUAL(sys.getUpdaterPeriod("updater2"), (unsigned int)105);
	BOOST_CHECK_EQUAL(sys.getUpdaterPeriod("updater1"), (unsigned int)10);
	except = false;
	try { sys.getUpdaterPeriod("updater3"); }
	catch (runtime_error) { except = true; }
	BOOST_CHECK(except);
	
	sys.setUpdaterPeriod("updater1", 15);
	sys.setUpdaterPeriod("updater2", 8);
	BOOST_CHECK_EQUAL(sys.getUpdaterPeriod("updater2"), (unsigned int)8);
	BOOST_CHECK_EQUAL(sys.getUpdaterPeriod("updater1"), (unsigned int)15);
	
	// remove the updaters 1 by one and make sure they don't exist
	sys.removeUpdater("updater1");
	BOOST_CHECK_EQUAL(sys.getUpdater("updater2"), updater2);
	except = false;
	try { sys.getUpdater("updater1"); }
	catch (runtime_error) { except = true; }
	BOOST_CHECK(except);
	
	sys.removeUpdater("updater2");
	except = false;
	try { sys.getUpdater("updater2"); }
	catch (runtime_error) { except = true; }
	BOOST_CHECK(except);
	
	// ********* Computes
	// create two updaters to test adding
	boost::shared_ptr< Compute > compute1(new DummyCompute(pdata, "compute1"));
	boost::shared_ptr< Compute > compute2(new DummyCompute(pdata, "compute2"));
	
	// add them both to a System
	sys.addCompute(compute1, "compute1");
	sys.addCompute(compute2, "compute2");

	// test adding another of the same name
	except = false;
	try { sys.addCompute(compute2, "compute1"); }
	catch (runtime_error) { except = true; }
	BOOST_CHECK(except);
	
	// check the get method
	BOOST_CHECK_EQUAL(sys.getCompute("compute1"), compute1);
	BOOST_CHECK_EQUAL(sys.getCompute("compute2"), compute2);
	except = false;
	try { sys.getCompute("compute3"); }
	catch (runtime_error) { except = true; }
	BOOST_CHECK(except);
	
	// remove the computes 1 by one and make sure they don't exist
	sys.removeCompute("compute1");
	BOOST_CHECK_EQUAL(sys.getCompute("compute2"), compute2);
	except = false;
	try { sys.getCompute("compute1"); }
	catch (runtime_error) { except = true; }
	BOOST_CHECK(except);
	
	sys.removeCompute("compute2");
	except = false;
	try { sys.getCompute("compute2"); }
	catch (runtime_error) { except = true; }
	BOOST_CHECK(except);
	
	// ************ Integrator
	boost::shared_ptr< Integrator > integrator1(new DummyUpdater(pdata, "integrator1"));
	boost::shared_ptr< Integrator > integrator2(new DummyUpdater(pdata, "integrator2"));

	sys.setIntegrator(integrator1);
	BOOST_CHECK_EQUAL(sys.getIntegrator(), integrator1);
	sys.setIntegrator(integrator2);
	BOOST_CHECK_EQUAL(sys.getIntegrator(), integrator2);
	}
	
//! Tests System::run(), System::enableProfiler(), and System::setStatsPeriod()
/*! Unfortunately, there is no way to automatically test these. A human will have to 
	observe the output and verify that it is correct
*/
BOOST_AUTO_TEST_CASE( run_tests )
	{
	// setup the system to run
	boost::shared_ptr< ParticleData > pdata(new ParticleData(10, BoxDim(10)));
		// create two analyzers to test adding
	boost::shared_ptr< Analyzer > analyzer1(new DummyAnalyzer(pdata, "analyzer1"));
	boost::shared_ptr< Analyzer > analyzer2(new DummyAnalyzer(pdata, "analyzer2"));
	
	// add them both to a System
	System sys(pdata, 0);
	sys.addAnalyzer(analyzer1, "analyzer1", 15);
	sys.addAnalyzer(analyzer2, "analyzer2", 20);
	
	// create two updaters to test adding
	boost::shared_ptr< Updater > updater1(new DummyUpdater(pdata, "updater1"));
	boost::shared_ptr< Updater > updater2(new DummyUpdater(pdata, "updater2"));
	
	// add them both to a System
	sys.addUpdater(updater1, "updater1", 5);
	sys.addUpdater(updater2, "updater2", 10);
	
	// create two updaters to test adding
	boost::shared_ptr< Compute > compute1(new DummyCompute(pdata, "compute1"));
	boost::shared_ptr< Compute > compute2(new DummyCompute(pdata, "compute2"));
	
	// add them both to a System
	sys.addCompute(compute1, "compute1");
	sys.addCompute(compute2, "compute2");
	
	// create an integrator and add it to the system
	boost::shared_ptr< Integrator > integrator(new DummyUpdater(pdata, "integrator"));
	sys.setIntegrator(integrator);
	
	cout << "First run: no profiling, statsPeriod = 10 s" << endl;
	sys.run(100);
	
	cout << "Second run: no prilfing, statsPeriod = 1.0 s" << endl;
	sys.setStatsPeriod(1);
	sys.run(100);
	
	cout << "Third run: profiling enabled" << endl;
	sys.enableProfiler(true);
	sys.run(100);
	}

