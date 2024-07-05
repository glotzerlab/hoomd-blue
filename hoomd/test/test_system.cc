// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// this include is necessary to get MPI included before anything else to support intel MPI
#include "hoomd/ExecutionConfiguration.h"

#include <iostream>

#include "upp11_config.h"
HOOMD_UP_MAIN();

#include "hoomd/System.h"
#include <math.h>

#include <iostream>
#include <stdexcept>
#include <string>

/*! \file system_test.cc
    \brief Unit tests for System
    \ingroup unit_tests
*/

using namespace std;
using namespace hoomd;

// we are going to need a dummy compute, a dummy analyzer and a dummy updater in order to test the
// system

//! Dummy analyzer for unit testing System
class DummyAnalyzer : public Analyzer
    {
    public:
    //! Constructs a named analyzer
    DummyAnalyzer(std::shared_ptr<SystemDefinition> sysdef,
                  std::shared_ptr<Trigger> trigger,
                  const string& name)
        : Analyzer(sysdef, trigger), m_name(name)
        {
        }

    //! Just prints our name and the current time step
    void analyze(uint64_t timestep)
        {
        cout << m_name << ": " << timestep << endl;
        Sleep(5);
        }

    private:
    string m_name; //!< Name of the dummy
    };

//! Dummy updater for unit testing System
class DummyUpdater : public Updater
    {
    // this derives from Integrator so the unit tests can use them in setIntegrator
    public:
    //! Constructs a named analyzer
    DummyUpdater(std::shared_ptr<SystemDefinition> sysdef,
                 std::shared_ptr<Trigger> trigger,
                 const string& name)
        : Updater(sysdef, trigger), m_name(name)
        {
        }

    //! Just prints our name and the current time step
    void update(uint64_t timestep)
        {
        cout << m_name << ": " << timestep << endl;
        Sleep(8);
        }

    private:
    string m_name; //!< Name of the dummy
    };

//! Dummy integrator for unit testing System
class DummyIntegrator : public Integrator
    {
    // this derives from Integrator so the unit tests can use them in setIntegrator
    public:
    //! Constructs a named analyzer
    DummyIntegrator(std::shared_ptr<SystemDefinition> sysdef, const string& name)
        : Integrator(sysdef, 0), m_name(name)
        {
        }

    //! Just prints our name and the current time step
    void update(uint64_t timestep)
        {
        cout << m_name << ": " << timestep << endl;
        Sleep(8);
        }

    private:
    string m_name; //!< Name of the dummy
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
    void compute(uint64_t timestep)
        {
        cout << m_name << ": " << timestep << endl;
        Sleep(8);
        }

    private:
    string m_name; //!< Name of the dummy
    };

//! Tests the add, get, and set routines in System
UP_TEST(getter_setter_tests)
    {
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(10, BoxDim(10)));

    // create two analyzers to test adding
    auto analyzer1 = std::make_shared<DummyAnalyzer>(sysdef,
                                                     std::make_shared<PeriodicTrigger>(10),
                                                     "analyzer1");
    auto analyzer2 = std::make_shared<DummyAnalyzer>(sysdef,
                                                     std::make_shared<PeriodicTrigger>(105),
                                                     "analyzer2");

    // add them both to a System
    System sys(sysdef, 0);
    sys.getAnalyzers().push_back(analyzer1);
    sys.getAnalyzers().push_back(analyzer2);

    // check the get method
    MY_ASSERT_EQUAL(sys.getAnalyzers()[0], analyzer1);
    MY_ASSERT_EQUAL(sys.getAnalyzers()[1], analyzer2);

    // test the get and set period functions
    UP_ASSERT_EQUAL(std::dynamic_pointer_cast<PeriodicTrigger>(sys.getAnalyzers()[1]->getTrigger())
                        ->getPeriod(),
                    (uint64_t)105);
    UP_ASSERT_EQUAL(std::dynamic_pointer_cast<PeriodicTrigger>(sys.getAnalyzers()[0]->getTrigger())
                        ->getPeriod(),
                    (uint64_t)10);

    std::dynamic_pointer_cast<PeriodicTrigger>(sys.getAnalyzers()[0]->getTrigger())->setPeriod(15);
    std::dynamic_pointer_cast<PeriodicTrigger>(sys.getAnalyzers()[1]->getTrigger())->setPeriod(8);
    UP_ASSERT_EQUAL(std::dynamic_pointer_cast<PeriodicTrigger>(sys.getAnalyzers()[1]->getTrigger())
                        ->getPeriod(),
                    (uint64_t)8);
    UP_ASSERT_EQUAL(std::dynamic_pointer_cast<PeriodicTrigger>(sys.getAnalyzers()[0]->getTrigger())
                        ->getPeriod(),
                    (uint64_t)15);

    // ************ Updaters
    // create two updaters to test adding
    auto updater1
        = std::make_shared<DummyUpdater>(sysdef, std::make_shared<PeriodicTrigger>(10), "updater1");
    auto updater2 = std::make_shared<DummyUpdater>(sysdef,
                                                   std::make_shared<PeriodicTrigger>(105),
                                                   "updater2");

    // add them both to a System
    sys.getUpdaters().push_back(updater1);
    sys.getUpdaters().push_back(updater2);

    // check the get method
    MY_ASSERT_EQUAL(sys.getUpdaters()[0], updater1);
    MY_ASSERT_EQUAL(sys.getUpdaters()[1], updater2);

    // test the get and set period functions
    UP_ASSERT_EQUAL(
        std::dynamic_pointer_cast<PeriodicTrigger>(sys.getUpdaters()[1]->getTrigger())->getPeriod(),
        (uint64_t)105);
    UP_ASSERT_EQUAL(
        std::dynamic_pointer_cast<PeriodicTrigger>(sys.getUpdaters()[0]->getTrigger())->getPeriod(),
        (uint64_t)10);

    std::dynamic_pointer_cast<PeriodicTrigger>(sys.getUpdaters()[0]->getTrigger())->setPeriod(15);
    std::dynamic_pointer_cast<PeriodicTrigger>(sys.getUpdaters()[1]->getTrigger())->setPeriod(8);
    UP_ASSERT_EQUAL(
        std::dynamic_pointer_cast<PeriodicTrigger>(sys.getUpdaters()[1]->getTrigger())->getPeriod(),
        (uint64_t)8);
    UP_ASSERT_EQUAL(
        std::dynamic_pointer_cast<PeriodicTrigger>(sys.getUpdaters()[0]->getTrigger())->getPeriod(),
        (uint64_t)15);

    // ************ Integrator
    auto integrator1 = std::make_shared<DummyIntegrator>(sysdef, "integrator1");
    auto integrator2 = std::make_shared<DummyIntegrator>(sysdef, "integrator2");

    sys.setIntegrator(integrator1);
    MY_ASSERT_EQUAL(sys.getIntegrator(), integrator1);
    sys.setIntegrator(integrator2);
    MY_ASSERT_EQUAL(sys.getIntegrator(), integrator2);
    }

// since there is no automatic verification, there is no reason to run this test all the time
// this test can be uncommented only when it needs to be checked by a person

//! Tests System::run()
/*! Unfortunately, there is no way to automatically test these. A human will have to
    observe the output and verify that it is correct
*/
/*UP_TEST( run_tests )
    {
    Py_Initialize();
    #ifdef ENABLE_HIP
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

    sys.run(100);

    Py_Finalize();
    }*/
