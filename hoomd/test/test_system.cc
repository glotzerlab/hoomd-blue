// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// this include is necessary to get MPI included before anything else to support intel MPI
#include "hoomd/ExecutionConfiguration.h"

#include <iostream>


#include "upp11_config.h"
HOOMD_UP_MAIN();


#include <math.h>
#include "hoomd/System.h"

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
        DummyAnalyzer(std::shared_ptr<SystemDefinition> sysdef, const string& name)
                : Analyzer(sysdef), m_name(name)
            {
            }

        //! Just prints our name and the current time step
        void analyze(unsigned int timestep)
            {
            if (m_prof)
                m_prof->push(m_name);
            cout << m_name << ": " << timestep << endl;
            Sleep(5);
            if (m_prof)
                m_prof->pop();
            }
    private:
        string m_name;  //!< Name of the dummy
    };

//! Dummy updater for unit testing System
class DummyUpdater : public Integrator
    {
        // this derives from Integrator so the unit tests can use them in setIntegrator
    public:
        //! Constructs a named analyzer
        DummyUpdater(std::shared_ptr<SystemDefinition> sysdef, const string& name)
                : Integrator(sysdef, 0.0), m_name(name)
            {
            }

        //! Just prints our name and the current time step
        void update(unsigned int timestep)
            {
            if (m_prof)
                m_prof->push(m_name);
            cout << m_name << ": " << timestep << endl;
            Sleep(8);
            if (m_prof)
                m_prof->pop();
            }
    private:
        string m_name;  //!< Name of the dummy
    };

//! Dummy updater for unit testing System
class DummyCompute : public Compute
    {
    public:
        //! Constructs a named analyzer
        DummyCompute(std::shared_ptr<SystemDefinition> sysdef, const string& name)
                : Compute(sysdef), m_name(name)
            {
            }

        //! Just prints our name and the current time step
        void compute(unsigned int timestep)
            {
            if (m_prof)
                m_prof->push(m_name);
            cout << m_name << ": " << timestep << endl;
            Sleep(8);
            if (m_prof)
                m_prof->pop();
            }
    private:
        string m_name;  //!< Name of the dummy
    };

//! Tests the add, get, and set routines in System
UP_TEST( getter_setter_tests )
    {
    std::shared_ptr< SystemDefinition > sysdef(new SystemDefinition(10, BoxDim(10)));

    cout << "Running getter_setter_tests, expect error messages to be printed to the screen as this tests error checking in System" << endl;

    // create two analyzers to test adding
    std::shared_ptr< Analyzer > analyzer1(new DummyAnalyzer(sysdef, "analyzer1"));
    std::shared_ptr< Analyzer > analyzer2(new DummyAnalyzer(sysdef, "analyzer2"));

    // add them both to a System
    System sys(sysdef, 0);
    sys.addAnalyzer(analyzer1, "analyzer1", 10, -1);
    sys.addAnalyzer(analyzer2, "analyzer2", 105, -1);

    // test adding another of the same name
    bool except = false;
    try
        {
        sys.addAnalyzer(analyzer2, "analyzer1", 1, -1);
        }
    catch (const std::runtime_error&)
        {
        except = true;
        }
    UP_ASSERT(except);

    // check the get method
    MY_ASSERT_EQUAL(sys.getAnalyzer("analyzer1"), analyzer1);
    MY_ASSERT_EQUAL(sys.getAnalyzer("analyzer2"), analyzer2);
    except = false;
    try
        {
        sys.getAnalyzer("analyzer3");
        }
    catch (const std::runtime_error&)
        {
        except = true;
        }
    UP_ASSERT(except);

    // test the get and set period functions
    UP_ASSERT_EQUAL(sys.getAnalyzerPeriod("analyzer2"), (unsigned int)105);
    UP_ASSERT_EQUAL(sys.getAnalyzerPeriod("analyzer1"), (unsigned int)10);
    except = false;
    try
        {
        sys.getAnalyzerPeriod("analyzer3");
        }
    catch (const std::runtime_error&)
        {
        except = true;
        }
    UP_ASSERT(except);

    sys.setAnalyzerPeriod("analyzer1", 15, -1);
    sys.setAnalyzerPeriod("analyzer2", 8, -1);
    UP_ASSERT_EQUAL(sys.getAnalyzerPeriod("analyzer2"), (unsigned int)8);
    UP_ASSERT_EQUAL(sys.getAnalyzerPeriod("analyzer1"), (unsigned int)15);

    // remove the analyzers 1 by one and make sure they don't exist
    sys.removeAnalyzer("analyzer1");
    MY_ASSERT_EQUAL(sys.getAnalyzer("analyzer2"), analyzer2);
    except = false;
    try
        {
        sys.getAnalyzer("analyzer1");
        }
    catch (const std::runtime_error&)
        {
        except = true;
        }
    UP_ASSERT(except);

    sys.removeAnalyzer("analyzer2");
    except = false;
    try
        {
        sys.getAnalyzer("analyzer2");
        }
    catch (const std::runtime_error&)
        {
        except = true;
        }
    UP_ASSERT(except);

    // ************ Updaters
    // create two updaters to test adding
    std::shared_ptr< Updater > updater1(new DummyUpdater(sysdef, "updater1"));
    std::shared_ptr< Updater > updater2(new DummyUpdater(sysdef, "updater2"));

    // add them both to a System
    sys.addUpdater(updater1, "updater1", 10, -1);
    sys.addUpdater(updater2, "updater2", 105, -1);

    // test adding another of the same name
    except = false;
    try
        {
        sys.addUpdater(updater2, "updater1", 1, -1);
        }
    catch (const std::runtime_error&)
        {
        except = true;
        }
    UP_ASSERT(except);

    // check the get method
    MY_ASSERT_EQUAL(sys.getUpdater("updater1"), updater1);
    MY_ASSERT_EQUAL(sys.getUpdater("updater2"), updater2);
    except = false;
    try
        {
        sys.getUpdater("updater3");
        }
    catch (const std::runtime_error&)
        {
        except = true;
        }
    UP_ASSERT(except);

    // test the get and set period functions
    UP_ASSERT_EQUAL(sys.getUpdaterPeriod("updater2"), (unsigned int)105);
    UP_ASSERT_EQUAL(sys.getUpdaterPeriod("updater1"), (unsigned int)10);
    except = false;
    try
        {
        sys.getUpdaterPeriod("updater3");
        }
    catch (const std::runtime_error&)
        {
        except = true;
        }
    UP_ASSERT(except);

    sys.setUpdaterPeriod("updater1", 15, -1);
    sys.setUpdaterPeriod("updater2", 8, -1);
    UP_ASSERT_EQUAL(sys.getUpdaterPeriod("updater2"), (unsigned int)8);
    UP_ASSERT_EQUAL(sys.getUpdaterPeriod("updater1"), (unsigned int)15);

    // remove the updaters 1 by one and make sure they don't exist
    sys.removeUpdater("updater1");
    MY_ASSERT_EQUAL(sys.getUpdater("updater2"), updater2);
    except = false;
    try
        {
        sys.getUpdater("updater1");
        }
    catch (const std::runtime_error&)
        {
        except = true;
        }
    UP_ASSERT(except);

    sys.removeUpdater("updater2");
    except = false;
    try
        {
        sys.getUpdater("updater2");
        }
    catch (const std::runtime_error&)
        {
        except = true;
        }
    UP_ASSERT(except);

    // ********* Computes
    // create two updaters to test adding
    std::shared_ptr< Compute > compute1(new DummyCompute(sysdef, "compute1"));
    std::shared_ptr< Compute > compute2(new DummyCompute(sysdef, "compute2"));

    // add them both to a System
    sys.addCompute(compute1, "compute1");
    sys.addCompute(compute2, "compute2");

    // test adding another of the same name
    except = false;
    try
        {
        sys.addCompute(compute2, "compute1");
        }
    catch (const std::runtime_error&)
        {
        except = true;
        }
    UP_ASSERT(except);

    // check the get method
    MY_ASSERT_EQUAL(sys.getCompute("compute1"), compute1);
    MY_ASSERT_EQUAL(sys.getCompute("compute2"), compute2);
    except = false;
    try
        {
        sys.getCompute("compute3");
        }
    catch (const std::runtime_error&)
        {
        except = true;
        }
    UP_ASSERT(except);

    // remove the computes 1 by one and make sure they don't exist
    sys.removeCompute("compute1");
    MY_ASSERT_EQUAL(sys.getCompute("compute2"), compute2);
    except = false;
    try
        {
        sys.getCompute("compute1");
        }
    catch (const std::runtime_error&)
        {
        except = true;
        }
    UP_ASSERT(except);

    sys.removeCompute("compute2");
    except = false;
    try
        {
        sys.getCompute("compute2");
        }
    catch (const std::runtime_error&)
        {
        except = true;
        }
    UP_ASSERT(except);

    // ************ Integrator
    std::shared_ptr< Integrator > integrator1(new DummyUpdater(sysdef, "integrator1"));
    std::shared_ptr< Integrator > integrator2(new DummyUpdater(sysdef, "integrator2"));

    sys.setIntegrator(integrator1);
    MY_ASSERT_EQUAL(sys.getIntegrator(), integrator1);
    sys.setIntegrator(integrator2);
    MY_ASSERT_EQUAL(sys.getIntegrator(), integrator2);
    }


// since there is no automatic verification, there is no reason to run this test all the time
// this test can be uncommented only when it needs to be checked by a person

//! Tests System::run(), System::enableProfiler(), and System::setStatsPeriod()
/*! Unfortunately, there is no way to automatically test these. A human will have to
    observe the output and verify that it is correct
*/
/*UP_TEST( run_tests )
    {
    Py_Initialize();
    #ifdef ENABLE_CUDA
    g_gpu_error_checking = true;
    #endif

    // setup the system to run
    std::shared_ptr< SystemDefinition > sysdef(new SystemDefinition(10, BoxDim(10)));
        // create two analyzers to test adding
    std::shared_ptr< Analyzer > analyzer1(new DummyAnalyzer(sysdef, "analyzer1"));
    std::shared_ptr< Analyzer > analyzer2(new DummyAnalyzer(sysdef, "analyzer2"));
    std::shared_ptr< Analyzer > analyzer3(new DummyAnalyzer(pdata, "analyzer3"));

    // add them both to a System
    System sys(sysdef, 0);
    sys.addAnalyzer(analyzer1, "analyzer1", 15, -1);
    sys.addAnalyzer(analyzer2, "analyzer2", 20, -1);
    sys.addAnalyzer(analyzer3, "analyzer3", 1, -1);

    sys.setAnalyzerPeriodVariable("analyzer3", eval("lambda n: n**2"));

    // create two updaters to test adding
    std::shared_ptr< Updater > updater1(new DummyUpdater(sysdef, "updater1"));
    std::shared_ptr< Updater > updater2(new DummyUpdater(sysdef, "updater2"));
    std::shared_ptr< Updater > updater3(new DummyUpdater(pdata, "updater3"));

    // add them both to a System
    sys.addUpdater(updater1, "updater1", 5, -1);
    sys.addUpdater(updater2, "updater2", 10, -1);
    sys.addUpdater(updater3, "updater3", 20, -1);

    sys.setUpdaterPeriodVariable("updater3", eval("lambda n: 0.5 * 10**n"));

    // create two updaters to test adding
    std::shared_ptr< Compute > compute1(new DummyCompute(sysdef, "compute1"));
    std::shared_ptr< Compute > compute2(new DummyCompute(sysdef, "compute2"));

    // add them both to a System
    sys.addCompute(compute1, "compute1");
    sys.addCompute(compute2, "compute2");

    // create an integrator and add it to the system
    std::shared_ptr< Integrator > integrator(new DummyUpdater(sysdef, "integrator"));
    sys.setIntegrator(integrator);

    cout << "First run: no profiling, statsPeriod = 10 s" << endl;
    sys.run(100);

    cout << "Second run: no profiling, statsPeriod = 1.0 s" << endl;
    sys.setStatsPeriod(1);
    sys.run(100);

    cout << "Third run: profiling enabled" << endl;
    sys.enableProfiler(true);
    sys.run(100);

    Py_Finalize();
    }*/
