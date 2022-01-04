// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file System.cc
    \brief Defines the System class
*/

#include "System.h"

#ifdef ENABLE_MPI
#include "Communicator.h"
#endif

// #include <pybind11/pybind11.h>
#include <pybind11/cast.h>
#include <pybind11/stl_bind.h>
#include <stdexcept>
#include <time.h>

using namespace std;

// the typedef works around an issue with older versions of the preprocessor
// specifically, gcc8
typedef std::pair<std::shared_ptr<hoomd::Analyzer>, std::shared_ptr<hoomd::Trigger>> _analyzer_pair;
PYBIND11_MAKE_OPAQUE(std::vector<_analyzer_pair>)
typedef std::pair<std::shared_ptr<hoomd::Updater>, std::shared_ptr<hoomd::Trigger>> _updater_pair;
PYBIND11_MAKE_OPAQUE(std::vector<_updater_pair>)
PYBIND11_MAKE_OPAQUE(std::vector<std::shared_ptr<hoomd::Tuner>>)

namespace hoomd
    {
/*! \param sysdef SystemDefinition for the system to be simulated
    \param initial_tstep Initial time step of the simulation

    \post The System is constructed with no attached computes, updaters,
    analyzers or integrators. Profiling defaults to disabled and
    statistics are printed every 10 seconds.
*/
System::System(std::shared_ptr<SystemDefinition> sysdef, uint64_t initial_tstep)
    : m_sysdef(sysdef), m_start_tstep(initial_tstep), m_end_tstep(0), m_cur_tstep(initial_tstep),
      m_profile(false)
    {
    // sanity check
    assert(m_sysdef);
    m_exec_conf = m_sysdef->getParticleData()->getExecConf();

#ifdef ENABLE_MPI
    // the initial time step is defined on the root processor
    if (m_sysdef->getParticleData()->getDomainDecomposition())
        {
        bcast(m_start_tstep, 0, m_exec_conf->getMPICommunicator());
        bcast(m_cur_tstep, 0, m_exec_conf->getMPICommunicator());
        }
#endif
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

void System::run(uint64_t nsteps, bool write_at_start)
    {
    m_start_tstep = m_cur_tstep;
    m_end_tstep = m_cur_tstep + nsteps;

    // initialize the last status time
    m_initial_time = m_clk.getTime();
    setupProfiling();

    // preset the flags before the run loop so that any analyzers/updaters run on step 0 have the
    // info they need but set the flags before prepRun, as prepRun may remove some flags that it
    // cannot generate on the first step
    m_sysdef->getParticleData()->setFlags(determineFlags(m_cur_tstep));

    resetStats();

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        // make sure we start off with a migration substep
        m_comm->forceMigrate();

        // communicate here
        m_comm->communicate(m_cur_tstep);
        }
#endif

    // Prepare the run
    if (m_integrator)
        {
        m_integrator->prepRun(m_cur_tstep);
        }

    // execute analyzers on initial step if requested
    if (write_at_start)
        {
        for (auto& analyzer_trigger_pair : m_analyzers)
            {
            if ((*analyzer_trigger_pair.second)(m_cur_tstep))
                analyzer_trigger_pair.first->analyze(m_cur_tstep);
            }
        }

    // run the steps
    for (uint64_t count = 0; count < nsteps; count++)
        {
        for (auto& tuner : m_tuners)
            {
            if ((*tuner->getTrigger())(m_cur_tstep))
                tuner->update(m_cur_tstep);
            }

        // execute updaters
        for (auto& updater_trigger_pair : m_updaters)
            {
            if ((*updater_trigger_pair.second)(m_cur_tstep))
                updater_trigger_pair.first->update(m_cur_tstep);
            }

        // look ahead to the next time step and see which analyzers and updaters will be executed
        // or together all of their requested PDataFlags to determine the flags to set for this time
        // step
        m_sysdef->getParticleData()->setFlags(determineFlags(m_cur_tstep + 1));

        // execute the integrator
        if (m_integrator)
            m_integrator->update(m_cur_tstep);

        m_cur_tstep++;

        // execute analyzers after incrementing the step counter
        for (auto& analyzer_trigger_pair : m_analyzers)
            {
            if ((*analyzer_trigger_pair.second)(m_cur_tstep))
                analyzer_trigger_pair.first->analyze(m_cur_tstep);
            }

        updateTPS();

        // propagate Python exceptions related to signals
        if (PyErr_CheckSignals() != 0)
            {
            throw pybind11::error_already_set();
            }
        }

#ifdef ENABLE_MPI
    // make sure all ranks return the same TPS after the run completes
    if (m_sysdef->isDomainDecomposed())
        {
        bcast(m_last_TPS, 0, m_exec_conf->getMPICommunicator());
        bcast(m_last_walltime, 0, m_exec_conf->getMPICommunicator());
        }
#endif
    }

void System::updateTPS()
    {
    m_last_walltime = double(m_clk.getTime() - m_initial_time) / double(1e9);

    // calculate average TPS
    m_last_TPS = double(m_cur_tstep - m_start_tstep) / m_last_walltime;
    }

/*! \param enable Set to true to enable profiling during calls to run()
 */
void System::enableProfiler(bool enable)
    {
    m_profile = enable;
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
    for (auto& analyzer_trigger_pair : m_analyzers)
        analyzer_trigger_pair.first->setAutotunerParams(enabled, period);

    // updaters
    for (auto& updater_trigger_pair : m_updaters)
        updater_trigger_pair.first->setAutotunerParams(enabled, period);

    // computes
    for (auto compute : m_computes)
        compute->setAutotunerParams(enabled, period);

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
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
    for (auto& analyzer_trigger_pair : m_analyzers)
        analyzer_trigger_pair.first->setProfiler(m_profiler);

    // updaters
    for (auto& updater_trigger_pair : m_updaters)
        {
        if (!updater_trigger_pair.first)
            throw runtime_error("Invalid updater_trigger_pair");
        updater_trigger_pair.first->setProfiler(m_profiler);
        }

    // computes
    for (auto compute : m_computes)
        compute->setProfiler(m_profiler);

#ifdef ENABLE_MPI
    // communicator
    if (m_sysdef->isDomainDecomposed())
        m_comm->setProfiler(m_profiler);
#endif
    }

void System::resetStats()
    {
    if (m_integrator)
        m_integrator->resetStats();

    // analyzers
    for (auto& analyzer_trigger_pair : m_analyzers)
        analyzer_trigger_pair.first->resetStats();

    // updaters
    for (auto& updater_trigger_pair : m_updaters)
        updater_trigger_pair.first->resetStats();

    // computes
    for (auto compute : m_computes)
        compute->resetStats();
    }

/*! \param tstep Time step for which to determine the flags

    The flags needed are determined by peeking to \a tstep and then using bitwise or to combine all
   of the flags from the analyzers and updaters that are to be executed on that step.
*/
PDataFlags System::determineFlags(uint64_t tstep)
    {
    PDataFlags flags = m_default_flags;
    if (m_integrator)
        flags |= m_integrator->getRequestedPDataFlags();

    for (auto& analyzer_trigger_pair : m_analyzers)
        {
        if ((*analyzer_trigger_pair.second)(tstep))
            flags |= analyzer_trigger_pair.first->getRequestedPDataFlags();
        }

    for (auto& updater_trigger_pair : m_updaters)
        {
        if ((*updater_trigger_pair.second)(tstep))
            flags |= updater_trigger_pair.first->getRequestedPDataFlags();
        }

    for (auto& tuner : m_tuners)
        {
        if ((*tuner->getTrigger())(tstep))
            flags |= tuner->getRequestedPDataFlags();
        }

    return flags;
    }

namespace detail
    {
void export_System(pybind11::module& m)
    {
    pybind11::bind_vector<
        std::vector<std::pair<std::shared_ptr<Analyzer>, std::shared_ptr<Trigger>>>>(
        m,
        "AnalyzerTriggerList");
    pybind11::bind_vector<
        std::vector<std::pair<std::shared_ptr<Updater>, std::shared_ptr<Trigger>>>>(
        m,
        "UpdaterTriggerList");
    pybind11::bind_vector<std::vector<std::shared_ptr<Tuner>>>(m, "TunerList");
    pybind11::bind_vector<std::vector<std::shared_ptr<Compute>>>(m, "ComputeList");

    pybind11::class_<System, std::shared_ptr<System>>(m, "System")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, uint64_t>())

        .def("setIntegrator", &System::setIntegrator)
        .def("getIntegrator", &System::getIntegrator)

        .def("setAutotunerParams", &System::setAutotunerParams)
        .def("enableProfiler", &System::enableProfiler)
        .def("run", &System::run)

        .def("getLastTPS", &System::getLastTPS)
        .def("getCurrentTimeStep", &System::getCurrentTimeStep)
        .def("setPressureFlag", &System::setPressureFlag)
        .def("getPressureFlag", &System::getPressureFlag)
        .def_property_readonly("walltime", &System::getCurrentWalltime)
        .def_property_readonly("final_timestep", &System::getEndStep)
        .def_property_readonly("analyzers", &System::getAnalyzers)
        .def_property_readonly("updaters", &System::getUpdaters)
        .def_property_readonly("tuners", &System::getTuners)
        .def_property_readonly("computes", &System::getComputes)
#ifdef ENABLE_MPI
        .def("setCommunicator", &System::setCommunicator)
#endif
        ;
    }

    } // end namespace detail

    } // end namespace hoomd
