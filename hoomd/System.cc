// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file System.cc
    \brief Defines the System class
*/


#include "System.h"
#include "SignalHandler.h"

#ifdef ENABLE_MPI
#include "Communicator.h"
#endif

// #include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include <stdexcept>
#include <time.h>

using namespace std;
namespace py = pybind11;

PyObject* walltimeLimitExceptionTypeObj = 0;

/*! \param sysdef SystemDefinition for the system to be simulated
    \param initial_tstep Initial time step of the simulation

    \post The System is constructed with no attached computes, updaters,
    analyzers or integrators. Profiling defaults to disabled and
    statistics are printed every 10 seconds.
*/
System::System(std::shared_ptr<SystemDefinition> sysdef, unsigned int initial_tstep)
        : m_sysdef(sysdef), m_start_tstep(initial_tstep), m_end_tstep(0), m_cur_tstep(initial_tstep), m_cur_tps(0),
        m_med_tps(0), m_last_status_time(0), m_last_status_tstep(initial_tstep), m_quiet_run(false),
        m_profile(false), m_stats_period(10)
    {
    // sanity check
    assert(m_sysdef);
    m_exec_conf = m_sysdef->getParticleData()->getExecConf();

    // initialize tps array
    m_tps_list.resize(0);
    #ifdef ENABLE_MPI
    // the initial time step is defined on the root processor
    if (m_sysdef->getParticleData()->getDomainDecomposition())
        {
        bcast(m_start_tstep, 0, m_exec_conf->getMPICommunicator());
        bcast(m_cur_tstep, 0, m_exec_conf->getMPICommunicator());
        bcast(m_last_status_tstep, 0, m_exec_conf->getMPICommunicator());
        }
    #endif
    }

/*! \param analyzer Shared pointer to the Analyzer to add
    \param name A unique name to identify the Analyzer by
    \param period Analyzer::analyze() will be called for every time step that is a multiple
    of \a period.
    \param phase Phase offset. A value of -1 sets no phase, updates start on the current step. A value of 0 or greater
                 sets the analyzer to run at steps where (step % (period + phase)) == 0.

    All Analyzers will be called, in the order that they are added, and with the specified
    \a period during time step calculations performed when run() is called. An analyzer
    can be prevented from running in future runs by removing it (removeAnalyzer()) before
    calling run()
*/
void System::addAnalyzer(std::shared_ptr<Analyzer> analyzer, const std::string& name, unsigned int period, int phase)
    {
    // sanity check
    assert(analyzer);
    assert(period != 0);

    // first check that the name is unique
    vector<analyzer_item>::iterator i;
    for (i = m_analyzers.begin(); i != m_analyzers.end(); ++i)
        {
        if (i->m_name == name)
            {
            m_exec_conf->msg->error() << "Analyzer " << name << " already exists" << endl;
            throw runtime_error("System: cannot add Analyzer");
            }
        }

    unsigned int start_step = m_cur_tstep;
    if (phase >= 0)
        {
        // determine next step that is in line with period + phase
        unsigned int multiple = m_cur_tstep / period + (m_cur_tstep % period != 0);
        start_step = multiple * period + phase;
        }

    // if we get here, we can add it
    m_analyzers.push_back(analyzer_item(analyzer, name, period, m_cur_tstep, start_step));
    }

/*! \param name Name of the Analyzer to find in m_analyzers
    \returns An iterator into m_analyzers of the found Analyzer
*/
std::vector<System::analyzer_item>::iterator System::findAnalyzerItem(const std::string &name)
    {
    // search for the analyzer
    vector<analyzer_item>::iterator i;
    for (i = m_analyzers.begin(); i != m_analyzers.end(); ++i)
        {
        if (i->m_name == name)
            {
            return i;
            }
        }

    m_exec_conf->msg->error() << "Analyzer " << name << " not found" << endl;
    throw runtime_error("System: cannot find Analyzer");
    // dummy return
    return m_analyzers.begin();
    }

/*! \param name Name of the Analyzer to be removed
    \sa addAnalyzer()
*/
void System::removeAnalyzer(const std::string& name)
    {
    vector<analyzer_item>::iterator i = findAnalyzerItem(name);
    m_analyzers.erase(i);
    }

/*! \param name Name of the Analyzer to retrieve
    \returns A shared pointer to the requested Analyzer
*/
std::shared_ptr<Analyzer> System::getAnalyzer(const std::string& name)
    {
    vector<System::analyzer_item>::iterator i = findAnalyzerItem(name);
    return i->m_analyzer;
    }

/*! \param name Name of the Analyzer to modify
    \param period New period to set
*/
void System::setAnalyzerPeriod(const std::string& name, unsigned int period, int phase)
    {
    // sanity check
    assert(period != 0);

    unsigned int start_step = m_cur_tstep;
    if (phase >= 0)
        {
        // determine next step that is in line with period + phase
        unsigned int multiple = m_cur_tstep / period + (m_cur_tstep % period != 0);
        start_step = multiple * period + phase;
        }

    vector<System::analyzer_item>::iterator i = findAnalyzerItem(name);
    i->setPeriod(period, start_step);
    }

/*! \param name Name of the Updater to modify
    \param update_func A python callable function taking one argument that returns an integer value of the next time step to analyze at
*/
void System::setAnalyzerPeriodVariable(const std::string& name, py::object update_func)
    {
    vector<System::analyzer_item>::iterator i = findAnalyzerItem(name);
    i->setVariablePeriod(update_func, m_cur_tstep);
    }


/*! \param name Name of the Analyzer to get the period of
    \returns Period of the Analyzer
*/
unsigned int System::getAnalyzerPeriod(const std::string& name)
    {
    vector<System::analyzer_item>::iterator i = findAnalyzerItem(name);
    return i->m_period;
    }


// -------------- Updater get/set methods
/*! \param name Name of the Updater to find in m_updaters
    \returns An iterator into m_updaters of the found Updater
*/
std::vector<System::updater_item>::iterator System::findUpdaterItem(const std::string &name)
    {
    // search for the analyzer
    vector<System::updater_item>::iterator i;
    for (i = m_updaters.begin(); i != m_updaters.end(); ++i)
        {
        if (i->m_name == name)
            {
            return i;
            }
        }

    m_exec_conf->msg->error() << "Updater " << name << " not found" << endl;
    throw runtime_error("System: cannot find Updater");
    // dummy return
    return m_updaters.begin();
    }


/*! \param updater Shared pointer to the Updater to add
    \param name A unique name to identify the Updater by
    \param period Updater::update() will be called for every time step that is a multiple
    of \a period.
    \param phase Phase offset. A value of -1 sets no phase, updates start on the current step. A value of 0 or greater
                 sets the analyzer to run at steps where (step % (period + phase)) == 0.

    All Updaters will be called, in the order that they are added, and with the specified
    \a period during time step calculations performed when run() is called. An updater
    can be prevented from running in future runs by removing it (removeUpdater()) before
    calling run()
*/
void System::addUpdater(std::shared_ptr<Updater> updater, const std::string& name, unsigned int period, int phase)
    {
    // sanity check
    assert(updater);

    if (period == 0)
        {
        m_exec_conf->msg->error() << "The period cannot be set to 0!" << endl;
        throw runtime_error("System: cannot add Updater");
        }

    // first check that the name is unique
    vector<updater_item>::iterator i;
    for (i = m_updaters.begin(); i != m_updaters.end(); ++i)
        {
        if (i->m_name == name)
            {
            m_exec_conf->msg->error() << "Updater " << name << " already exists" << endl;
            throw runtime_error("System: cannot add Updater");
            }
        }

    unsigned int start_step = m_cur_tstep;
    if (phase >= 0)
        {
        // determine next step that is in line with period + phase
        unsigned int multiple = m_cur_tstep / period + (m_cur_tstep % period != 0);
        start_step = multiple * period + phase;
        }

    // if we get here, we can add it
    m_updaters.push_back(updater_item(updater, name, period, m_cur_tstep, start_step));
    }

/*! \param name Name of the Updater to be removed
    \sa addUpdater()
*/
void System::removeUpdater(const std::string& name)
    {
    vector<updater_item>::iterator i = findUpdaterItem(name);
    m_updaters.erase(i);
    }

/*! \param name Name of the Updater to retrieve
    \returns A shared pointer to the requested Updater
*/
std::shared_ptr<Updater> System::getUpdater(const std::string& name)
    {
    vector<System::updater_item>::iterator i = findUpdaterItem(name);
    return i->m_updater;
    }

/*! \param name Name of the Updater to modify
    \param period New period to set
    \param phase New phase to set
*/
void System::setUpdaterPeriod(const std::string& name, unsigned int period, int phase)
    {
    // sanity check
    assert(period != 0);

    unsigned int start_step = m_cur_tstep;
    if (phase >= 0)
        {
        // determine next step that is in line with period + phase
        unsigned int multiple = m_cur_tstep / period + (m_cur_tstep % period != 0);
        start_step = multiple * period + phase;
        }

    vector<System::updater_item>::iterator i = findUpdaterItem(name);
    i->setPeriod(period, start_step);
    }

/*! \param name Name of the Updater to modify
    \param update_func A python callable function taking one argument that returns an integer value of the next time step to update at
*/
void System::setUpdaterPeriodVariable(const std::string& name, py::object update_func)
    {
    vector<System::updater_item>::iterator i = findUpdaterItem(name);
    i->setVariablePeriod(update_func, m_cur_tstep);
    }

/*! \param name Name of the Updater to get the period of
    \returns Period of the Updater
*/
unsigned int System::getUpdaterPeriod(const std::string& name)
    {
    vector<System::updater_item>::iterator i = findUpdaterItem(name);
    return i->m_period;
    }


// -------------- Compute get/set methods

/*! \param compute Shared pointer to the Compute to add
    \param name Unique name to assign to this Compute

    Computes are added to the System only as a convenience for naming,
    saving to restart files, and to activate profiling. They are never
    directly called by the system.
*/
void System::addCompute(std::shared_ptr<Compute> compute, const std::string& name)
    {
    // sanity check
    assert(compute);

    // check if the name is unique
    map< string, std::shared_ptr<Compute> >::iterator i = m_computes.find(name);
    if (i == m_computes.end())
        m_computes[name] = compute;
    else
        {
        m_exec_conf->msg->error() << "Compute " << name << " already exists" << endl;
        throw runtime_error("System: cannot add compute");
        }
    }

/*! \param compute Shared pointer to the Compute to add
    \param name Unique name to assign to this Compute

    Computes are added to the System only as a convenience for naming,
    saving to restart files, and to activate profiling. They are never
    directly called by the system. This method adds a compute, overwriting
    any existing compute by the same name.
*/
void System::overwriteCompute(std::shared_ptr<Compute> compute, const std::string& name)
    {
    // sanity check
    assert(compute);

    m_computes[name] = compute;
    }

/*! \param name Name of the Compute to remove
*/
void System::removeCompute(const std::string& name)
    {
    // see if the compute exists to be removed
    map< string, std::shared_ptr<Compute> >::iterator i = m_computes.find(name);
    if (i == m_computes.end())
        {
        m_exec_conf->msg->error() << "Compute " << name << " not found" << endl;
        throw runtime_error("System: cannot remove compute");
        }
    else
        m_computes.erase(i);
    }

/*! \param name Name of the compute to access
    \returns A shared pointer to the Compute as provided previously by addCompute()
*/
std::shared_ptr<Compute> System::getCompute(const std::string& name)
    {
    // see if the compute even exists first
    map< string, std::shared_ptr<Compute> >::iterator i = m_computes.find(name);
    if (i == m_computes.end())
        {
        m_exec_conf->msg->error() << "Compute " << name << " not found" << endl;
        throw runtime_error("System: cannot retrieve compute");
        return std::shared_ptr<Compute>();
        }
    else
        return m_computes[name];
    }

// -------------- Integrator methods

/*! \param integrator Updater to set as the Integrator for this System
*/
void System::setIntegrator(std::shared_ptr<Integrator> integrator)
    {
    m_integrator = integrator;
    }

/*! \returns A shared pointer to the Integrator for this System
*/
std::shared_ptr<Integrator> System::getIntegrator()
    {
    return m_integrator;
    }

#ifdef ENABLE_MPI
// -------------- Methods for communication
void System::setCommunicator(std::shared_ptr<Communicator> comm)
    {
    m_comm = comm;
    }
#endif

// -------------- Methods for running the simulation

/*! \param nsteps Number of simulation steps to run
    \param limit_hours Number of hours to run for (0.0 => infinity)
    \param cb_frequency Modulus of timestep number when to call the callback (0 = at end)
    \param callback Python function to be called periodically during run.
    \param limit_multiple Only allow \a limit_hours to break the simulation at steps that are a multiple of
           \a limit_multiple .

    During each simulation step, all added Analyzers and
    Updaters are called, then the Integrator to move the system
    forward one step in time. This is repeated \a nsteps times,
    or until a \a limit_hours hours have passed.

    run() can be called as many times as the user wishes:
    each time, it will continue at the time step where it left off.
*/

void System::run(unsigned int nsteps, unsigned int cb_frequency,
                 py::object callback, double limit_hours,
                 unsigned int limit_multiple)
    {
    // track if a wall clock timeout ended the run
    unsigned int timeout_end_run = 0;
    char *walltime_stop = getenv("HOOMD_WALLTIME_STOP");

    m_start_tstep = m_cur_tstep;
    m_end_tstep = m_cur_tstep + nsteps;

    // initialize the last status time
    int64_t initial_time = m_clk.getTime();
    m_last_status_time = initial_time;
    setupProfiling();

    // preset the flags before the run loop so that any analyzers/updaters run on step 0 have the info they need
    // but set the flags before prepRun, as prepRun may remove some flags that it cannot generate on the first step
    m_sysdef->getParticleData()->setFlags(determineFlags(m_cur_tstep));

#ifdef ENABLE_MPI
    if (m_comm)
        {
        //! Set communicator in all Updaters
        vector<updater_item>::iterator updater;
        for (updater =  m_updaters.begin(); updater != m_updaters.end(); ++updater)
            updater->m_updater->setCommunicator(m_comm);

        // Set communicator in all Computes
        map< string, std::shared_ptr<Compute> >::iterator compute;
        for (compute = m_computes.begin(); compute != m_computes.end(); ++compute)
            compute->second->setCommunicator(m_comm);

        // Set communicator in all Analyzers
        vector<analyzer_item>::iterator analyzer;
        for (analyzer =  m_analyzers.begin(); analyzer != m_analyzers.end(); ++analyzer)
            analyzer->m_analyzer->setCommunicator(m_comm);

        // Set communicator in Integrator
        if (m_integrator)
            m_integrator->setCommunicator(m_comm);
        }
#endif

    resetStats();

    #ifdef ENABLE_MPI
    if (m_comm)
        {
        // make sure we start off with a migration substep
        m_comm->forceMigrate();

        // communicate here, to run before the Logger
        m_comm->communicate(m_cur_tstep);
        }
    #endif

    // Prepare the run
    if (!m_integrator)
        {
        m_exec_conf->msg->warning() << "You are running without an integrator" << endl;
        }
    else
        {
        m_integrator->prepRun(m_cur_tstep);
        }

    // handle time steps
    for ( ; m_cur_tstep < m_end_tstep; m_cur_tstep++)
        {
        // check the clock and output a status line if needed
        uint64_t cur_time = m_clk.getTime();

        // check if the time limit has exceeded
        if (limit_hours != 0.0f)
            {
            if (m_cur_tstep % limit_multiple == 0)
                {
                int64_t time_limit = int64_t(limit_hours * 3600.0 * 1e9);
                if (int64_t(cur_time) - initial_time > time_limit)
                    timeout_end_run = 1;

                #ifdef ENABLE_MPI
                // if any processor wants to end the run, end it on all processors
                if (m_comm)
                    {
                    if (m_profiler) m_profiler->push("MPI sync");
                    MPI_Allreduce(MPI_IN_PLACE, &timeout_end_run, 1, MPI_INT, MPI_SUM, m_exec_conf->getMPICommunicator());
                    if (m_profiler) m_profiler->pop();
                    }
                #endif

                if (timeout_end_run)
                    {
                    m_exec_conf->msg->notice(2) << "Ending run at time step " << m_cur_tstep << " as " << limit_hours << " hours have passed" << endl;
                    break;
                    }
                }
            }

        // check if wall clock time limit has passed
        if (walltime_stop != NULL)
            {
            if (m_cur_tstep % limit_multiple == 0)
                {
                time_t end_time = atoi(walltime_stop);
                time_t predict_time = time(NULL);

                // predict when the next limit_multiple will be reached
                if (m_med_tps != Scalar(0))
                    predict_time += time_t(Scalar(limit_multiple) / m_med_tps);

                if (predict_time >= end_time)
                    timeout_end_run = 1;

                #ifdef ENABLE_MPI
                // if any processor wants to end the run, end it on all processors
                if (m_comm)
                    {
                    if (m_profiler) m_profiler->push("MPI sync");
                    MPI_Allreduce(MPI_IN_PLACE, &timeout_end_run, 1, MPI_INT, MPI_SUM, m_exec_conf->getMPICommunicator());
                    if (m_profiler) m_profiler->pop();
                    }
                #endif

                if (timeout_end_run)
                    {
                    m_exec_conf->msg->notice(2) << "Ending run before HOOMD_WALLTIME_STOP - current time step: " << m_cur_tstep << endl;
                    break;
                    }
                }
            }

        // execute python callback, if present and needed
        // a negative return value indicates immediate end of run.
        if (callback != py::none() && (cb_frequency > 0) && (m_cur_tstep % cb_frequency == 0))
            {
            py::object rv = callback(m_cur_tstep);
            if (rv != py::none())
                {
                int extracted_rv = py::cast<int>(rv);
                if (extracted_rv < 0)
                    {
                    m_exec_conf->msg->notice(2) << "End of run requested by python callback at step "
                         << m_cur_tstep << " / " << m_end_tstep << endl;
                    break;
                    }
                }
            }

        if (cur_time - m_last_status_time >= uint64_t(m_stats_period)*uint64_t(1000000000))
            {
            generateStatusLine();
            m_last_status_time = cur_time;
            m_last_status_tstep = m_cur_tstep;

            // check for any CUDA errors
            #ifdef ENABLE_CUDA
            if (m_exec_conf->isCUDAEnabled())
                {
                CHECK_CUDA_ERROR();
                }
            #endif
            }

        // execute analyzers
        vector<analyzer_item>::iterator analyzer;
        for (analyzer =  m_analyzers.begin(); analyzer != m_analyzers.end(); ++analyzer)
            {
            if (analyzer->shouldExecute(m_cur_tstep))
                analyzer->m_analyzer->analyze(m_cur_tstep);
            }

        // execute updaters
        vector<updater_item>::iterator updater;
        for (updater =  m_updaters.begin(); updater != m_updaters.end(); ++updater)
            {
            if (updater->shouldExecute(m_cur_tstep))
                updater->m_updater->update(m_cur_tstep);
            }

        // look ahead to the next time step and see which analyzers and updaters will be executed
        // or together all of their requested PDataFlags to determine the flags to set for this time step
        m_sysdef->getParticleData()->setFlags(determineFlags(m_cur_tstep+1));

        // execute the integrator
        if (m_integrator)
            m_integrator->update(m_cur_tstep);

        // quit if Ctrl-C was pressed
        if (g_sigint_recvd)
            {
            g_sigint_recvd = 0;
            return;
            }
        }

    // generate a final status line
    generateStatusLine();
    m_last_status_tstep = m_cur_tstep;

    // execute python callback, if present and needed
    if (callback != py::none() && (cb_frequency == 0))
        {
        callback(m_cur_tstep);
        }

    // calculate average TPS
    Scalar TPS = Scalar(m_cur_tstep - m_start_tstep) / Scalar(m_clk.getTime() - initial_time) * Scalar(1e9);

    m_last_TPS = TPS;

    #ifdef ENABLE_MPI
    // make sure all ranks return the same TPS
    if (m_comm)
        bcast(m_last_TPS, 0, m_exec_conf->getMPICommunicator());
    #endif

    if (!m_quiet_run)
        m_exec_conf->msg->notice(1) << "Average TPS: " << m_last_TPS << endl;

    // write out the profile data
    if (m_profiler)
        m_exec_conf->msg->notice(1) << *m_profiler;

    if (!m_quiet_run)
        printStats();

    // throw a WalltimeLimitReached exception if we timed out, but only if the user is using the HOOMD_WALLTIME_STOP feature
    if (timeout_end_run && walltime_stop != NULL)
        {
        PyErr_SetString(walltimeLimitExceptionTypeObj, "HOOMD_WALLTIME_STOP reached");
        throw py::error_already_set();
        }
    }

/*! \param enable Set to true to enable profiling during calls to run()
*/
void System::enableProfiler(bool enable)
    {
    m_profile = enable;
    }

/*! \param logger Logger to register computes and updaters with
    All computes and updaters registered with the system are also registered with the logger.
*/
void System::registerLogger(std::shared_ptr<Logger> logger)
    {
    // set the profiler on everything
    if (m_integrator)
        logger->registerUpdater(m_integrator);

    // updaters
    vector<updater_item>::iterator updater;
    for (updater = m_updaters.begin(); updater != m_updaters.end(); ++updater)
        logger->registerUpdater(updater->m_updater);

    // computes
    map< string, std::shared_ptr<Compute> >::iterator compute;
    for (compute = m_computes.begin(); compute != m_computes.end(); ++compute)
        logger->registerCompute(compute->second);
    }

/*! \param seconds Period between statistics output in seconds
*/
void System::setStatsPeriod(unsigned int seconds)
    {
    m_stats_period = seconds;
    }

/*! \param enable Enable/disable autotuning
    \param period period (approximate) in time steps when returning occurs
*/
void System::setAutotunerParams(bool enabled, unsigned int period)
    {
    // set the autotuner parameters on everything
    if (m_integrator)
        m_integrator->setAutotunerParams(enabled, period);

    // analyzers
    vector<analyzer_item>::iterator analyzer;
    for (analyzer = m_analyzers.begin(); analyzer != m_analyzers.end(); ++analyzer)
        analyzer->m_analyzer->setAutotunerParams(enabled, period);

    // updaters
    vector<updater_item>::iterator updater;
    for (updater = m_updaters.begin(); updater != m_updaters.end(); ++updater)
        updater->m_updater->setAutotunerParams(enabled, period);

    // computes
    map< string, std::shared_ptr<Compute> >::iterator compute;
    for (compute = m_computes.begin(); compute != m_computes.end(); ++compute)
        compute->second->setAutotunerParams(enabled, period);

    #ifdef ENABLE_MPI
    if (m_comm)
        m_comm->setAutotunerParams(enabled, period);
    #endif
    }

// --------- Steps in the simulation run implemented in helper functions

void System::setupProfiling()
    {
    if (m_profile)
        m_profiler = std::shared_ptr<Profiler>(new Profiler("Simulation"));
    else
        m_profiler = std::shared_ptr<Profiler>();

    // set the profiler on everything
    if (m_integrator)
        m_integrator->setProfiler(m_profiler);
    m_sysdef->getParticleData()->setProfiler(m_profiler);
    m_sysdef->getBondData()->setProfiler(m_profiler);
    m_sysdef->getPairData()->setProfiler(m_profiler);
    m_sysdef->getAngleData()->setProfiler(m_profiler);
    m_sysdef->getDihedralData()->setProfiler(m_profiler);
    m_sysdef->getImproperData()->setProfiler(m_profiler);
    m_sysdef->getConstraintData()->setProfiler(m_profiler);

    // analyzers
    vector<analyzer_item>::iterator analyzer;
    for (analyzer = m_analyzers.begin(); analyzer != m_analyzers.end(); ++analyzer)
        analyzer->m_analyzer->setProfiler(m_profiler);

    // updaters
    vector<updater_item>::iterator updater;
    for (updater = m_updaters.begin(); updater != m_updaters.end(); ++updater)
        updater->m_updater->setProfiler(m_profiler);

    // computes
    map< string, std::shared_ptr<Compute> >::iterator compute;
    for (compute = m_computes.begin(); compute != m_computes.end(); ++compute)
        compute->second->setProfiler(m_profiler);

#ifdef ENABLE_MPI
    // communicator
    if (m_comm)
        m_comm->setProfiler(m_profiler);
#endif
    }

void System::printStats()
    {
    m_exec_conf->msg->notice(1) << "---------" << endl;
    // print the stats for everything
    if (m_integrator)
        m_integrator->printStats();

    // analyzers
    vector<analyzer_item>::iterator analyzer;
    for (analyzer = m_analyzers.begin(); analyzer != m_analyzers.end(); ++analyzer)
      analyzer->m_analyzer->printStats();

    // updaters
    vector<updater_item>::iterator updater;
    for (updater = m_updaters.begin(); updater != m_updaters.end(); ++updater)
        updater->m_updater->printStats();

    // computes
    map< string, std::shared_ptr<Compute> >::iterator compute;
    for (compute = m_computes.begin(); compute != m_computes.end(); ++compute)
        compute->second->printStats();

    // output memory trace information
    if (m_exec_conf->getMemoryTracer())
        m_exec_conf->getMemoryTracer()->outputTraces(m_exec_conf->msg);
    }

void System::resetStats()
    {
    if (m_integrator)
        m_integrator->resetStats();

    // analyzers
    vector<analyzer_item>::iterator analyzer;
    for (analyzer = m_analyzers.begin(); analyzer != m_analyzers.end(); ++analyzer)
      analyzer->m_analyzer->resetStats();

    // updaters
    vector<updater_item>::iterator updater;
    for (updater = m_updaters.begin(); updater != m_updaters.end(); ++updater)
        updater->m_updater->resetStats();

    // computes
    map< string, std::shared_ptr<Compute> >::iterator compute;
    for (compute = m_computes.begin(); compute != m_computes.end(); ++compute)
        compute->second->resetStats();
    }

void System::generateStatusLine()
    {
    // a status line consists of
    // elapsed time
    // current timestep / end time step
    // time steps per second
    // ETA

    // elapsed time
    int64_t cur_time = m_clk.getTime();
    string t_elap = ClockSource::formatHMS(cur_time);

    // time steps per second
    Scalar TPS = Scalar(m_cur_tstep - m_last_status_tstep) / Scalar(cur_time - m_last_status_time) * Scalar(1e9);
    // put into the tps list
    size_t tps_size = m_tps_list.size();
    if ((unsigned int)tps_size < 10)
        {
        // add to list if list less than 10
        m_tps_list.push_back(TPS);
        }
    else
        {
        // remove the first item, add to the end
        m_tps_list.erase(m_tps_list.begin());
        m_tps_list.push_back(TPS);
        }
    tps_size = m_tps_list.size();
    std::vector<Scalar> l_tps_list = m_tps_list;
    std::sort(l_tps_list.begin(), l_tps_list.end());
    // not the "true" median calculation, but it doesn't really matter in this case
    Scalar median = l_tps_list[tps_size / 2];
    m_med_tps = median;
    m_cur_tps = TPS;

    // estimated time to go (base on current TPS)
    string ETA = ClockSource::formatHMS(int64_t((m_end_tstep - m_cur_tstep) / TPS * Scalar(1e9)));

    // write the line
    if (!m_quiet_run)
        {
        m_exec_conf->msg->notice(1) << "Time " << t_elap << " | Step " << m_cur_tstep << " / " << m_end_tstep << " | TPS " << TPS << " | ETA " << ETA << endl;
        }
    }

/*! \param tstep Time step for which to determine the flags

    The flags needed are determined by peeking to \a tstep and then using bitwise or to combine all of the flags from the
    analyzers and updaters that are to be executed on that step.
*/
PDataFlags System::determineFlags(unsigned int tstep)
    {
    PDataFlags flags(0);
    if (m_integrator)
        flags = m_integrator->getRequestedPDataFlags();

    vector<analyzer_item>::iterator analyzer;
    for (analyzer = m_analyzers.begin(); analyzer != m_analyzers.end(); ++analyzer)
        {
        if (analyzer->peekExecute(tstep))
            flags |= analyzer->m_analyzer->getRequestedPDataFlags();
        }

    vector<updater_item>::iterator updater;
    for (updater = m_updaters.begin(); updater != m_updaters.end(); ++updater)
        {
        if (updater->peekExecute(tstep))
            flags |= updater->m_updater->getRequestedPDataFlags();
        }

    return flags;
    }

//! Create a custom exception
PyObject* createExceptionClass(py::module& m, const char* name, PyObject* baseTypeObj = PyExc_Exception)
    {
    // http://stackoverflow.com/questions/9620268/boost-python-custom-exception-class, modified by jproc for pybind11

    using std::string;

    string scopeName = py::cast<string>(m.attr("__name__"));
    string qualifiedName0 = scopeName + "." + name;
    char* qualifiedName1 = const_cast<char*>(qualifiedName0.c_str());

    PyObject* typeObj = PyErr_NewException(qualifiedName1, baseTypeObj, 0);
    if(!typeObj) throw py::error_already_set();
    m.attr(name) = py::object(typeObj,true);
    return typeObj;
    }

void export_System(py::module& m)
    {
    walltimeLimitExceptionTypeObj = createExceptionClass(m,"WalltimeLimitReached");

    py::class_< System, std::shared_ptr<System> > (m,"System")
    .def(py::init< std::shared_ptr<SystemDefinition>, unsigned int >())
    .def("addAnalyzer", &System::addAnalyzer)
    .def("removeAnalyzer", &System::removeAnalyzer)
    .def("getAnalyzer", &System::getAnalyzer)
    .def("setAnalyzerPeriod", &System::setAnalyzerPeriod)
    .def("setAnalyzerPeriodVariable", &System::setAnalyzerPeriodVariable)
    .def("getAnalyzerPeriod", &System::getAnalyzerPeriod)

    .def("addUpdater", &System::addUpdater)
    .def("removeUpdater", &System::removeUpdater)
    .def("getUpdater", &System::getUpdater)
    .def("setUpdaterPeriod", &System::setUpdaterPeriod)
    .def("setUpdaterPeriodVariable", &System::setUpdaterPeriodVariable)
    .def("getUpdaterPeriod", &System::getUpdaterPeriod)

    .def("addCompute", &System::addCompute)
    .def("overwriteCompute", &System::overwriteCompute)
    .def("removeCompute", &System::removeCompute)
    .def("getCompute", &System::getCompute)

    .def("setIntegrator", &System::setIntegrator)
    .def("getIntegrator", &System::getIntegrator)

    .def("registerLogger", &System::registerLogger)
    .def("setStatsPeriod", &System::setStatsPeriod)
    .def("setAutotunerParams", &System::setAutotunerParams)
    .def("enableProfiler", &System::enableProfiler)
    .def("enableQuietRun", &System::enableQuietRun)
    .def("run", &System::run)

    .def("getLastTPS", &System::getLastTPS)
    .def("getCurrentTimeStep", &System::getCurrentTimeStep)
#ifdef ENABLE_MPI
    .def("setCommunicator", &System::setCommunicator)
    .def("getCommunicator", &System::getCommunicator)
#endif
    ;
    }
