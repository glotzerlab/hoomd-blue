// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file System.cc
    \brief Defines the System class
*/

#include "System.h"

#ifdef ENABLE_MPI
#include "Communicator.h"
#endif

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <stdexcept>
#include <time.h>

using namespace std;

// the typedef works around an issue with older versions of the preprocessor
// specifically, gcc8
PYBIND11_MAKE_OPAQUE(std::vector<std::shared_ptr<hoomd::Analyzer>>)
PYBIND11_MAKE_OPAQUE(std::vector<std::shared_ptr<hoomd::Updater>>)
PYBIND11_MAKE_OPAQUE(std::vector<std::shared_ptr<hoomd::Tuner>>)
PYBIND11_MAKE_OPAQUE(std::vector<std::shared_ptr<hoomd::ParticleGroup>>);

namespace hoomd
    {
/*! \param sysdef SystemDefinition for the system to be simulated
    \param initial_tstep Initial time step of the simulation

    \post The System is constructed with no attached computes, updaters,
    analyzers or integrators.
*/
System::System(std::shared_ptr<SystemDefinition> sysdef, uint64_t initial_tstep)
    : m_sysdef(sysdef), m_start_tstep(initial_tstep), m_end_tstep(0), m_cur_tstep(initial_tstep)
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
    m_last_walltime = 0.0;
    m_last_TPS = 0.0;

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

    if (m_update_group_dof_next_step)
        {
        updateGroupDOF();
        m_update_group_dof_next_step = false;
        }

    // Prepare the run
    if (m_integrator)
        {
        m_integrator->prepRun(m_cur_tstep);
        }

    // preset the flags before the run loop so that any analyzers/updaters run on step 0 have the
    // info they need but set the flags before prepRun, as prepRun may remove some flags that it
    // cannot generate on the first step
    m_sysdef->getParticleData()->setFlags(determineFlags(m_cur_tstep));

    // execute analyzers on initial step if requested
    if (write_at_start)
        {
        for (auto& analyzer : m_analyzers)
            {
            if ((*analyzer->getTrigger())(m_cur_tstep))
                analyzer->analyze(m_cur_tstep);
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
        for (auto& updater : m_updaters)
            {
            if ((*updater->getTrigger())(m_cur_tstep))
                {
                updater->update(m_cur_tstep);
                m_update_group_dof_next_step |= updater->mayChangeDegreesOfFreedom(m_cur_tstep);
                }
            }

        if (m_update_group_dof_next_step)
            {
            updateGroupDOF();
            m_update_group_dof_next_step = false;
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
        for (auto& analyzer : m_analyzers)
            {
            if ((*analyzer->getTrigger())(m_cur_tstep))
                analyzer->analyze(m_cur_tstep);
            }

        updateTPS();

        // propagate Python exceptions related to signals
        if (PyErr_CheckSignals() != 0)
            {
            throw pybind11::error_already_set();
            }
        }
    }

void System::updateTPS()
    {
    m_last_walltime = double(m_clk.getTime() - m_initial_time) / double(1e9);

    // calculate average TPS
    m_last_TPS = double(m_cur_tstep - m_start_tstep) / m_last_walltime;
    }

// --------- Steps in the simulation run implemented in helper functions

void System::resetStats()
    {
    if (m_integrator)
        m_integrator->resetStats();

    // analyzers
    for (auto& analyzer : m_analyzers)
        analyzer->resetStats();

    // updaters
    for (auto& updater : m_updaters)
        updater->resetStats();

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

    for (auto& analyzer : m_analyzers)
        {
        if ((*analyzer->getTrigger())(tstep))
            flags |= analyzer->getRequestedPDataFlags();
        }

    for (auto& updater : m_updaters)
        {
        if ((*updater->getTrigger())(tstep))
            flags |= updater->getRequestedPDataFlags();
        }

    for (auto& tuner : m_tuners)
        {
        if ((*tuner->getTrigger())(tstep))
            flags |= tuner->getRequestedPDataFlags();
        }

    return flags;
    }

/*! Apply the degrees of freedom given by the integrator to all groups in the cache.
 */
void System::updateGroupDOF()
    {
    for (auto group : m_group_cache)
        {
        if (m_integrator)
            {
            m_integrator->updateGroupDOF(group);
            }
        else
            {
            group->setTranslationalDOF(0);
            group->setRotationalDOF(0);
            }
        }
    }

namespace detail
    {
void export_System(pybind11::module& m)
    {
    pybind11::bind_vector<std::vector<std::shared_ptr<Analyzer>>>(m, "AnalyzerList");
    pybind11::bind_vector<std::vector<std::shared_ptr<Updater>>>(m, "UpdaterList");
    pybind11::bind_vector<std::vector<std::shared_ptr<Tuner>>>(m, "TunerList");
    pybind11::bind_vector<std::vector<std::shared_ptr<Compute>>>(m, "ComputeList");

    pybind11::class_<System, std::shared_ptr<System>>(m, "System")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, uint64_t>())

        .def("setIntegrator", &System::setIntegrator)
        .def("getIntegrator", &System::getIntegrator)

        .def("run", &System::run)

        .def("getLastTPS", &System::getLastTPS)
        .def("getCurrentTimeStep", &System::getCurrentTimeStep)
        .def("setPressureFlag", &System::setPressureFlag)
        .def("getPressureFlag", &System::getPressureFlag)
        .def_property_readonly("walltime", &System::getCurrentWalltime)
        .def_property_readonly("final_timestep", &System::getEndStep)
        .def_property_readonly("initial_timestep", &System::getStartStep)
        .def_property_readonly("analyzers", &System::getAnalyzers)
        .def_property_readonly("updaters", &System::getUpdaters)
        .def_property_readonly("tuners", &System::getTuners)
        .def_property_readonly("computes", &System::getComputes)
        .def_property_readonly("group_cache", &System::getGroupCache)
        .def("getGroupCache", &System::getGroupCache)
        .def("updateGroupDOFOnNextStep", &System::updateGroupDOFOnNextStep)
#ifdef ENABLE_MPI
        .def("setCommunicator", &System::setCommunicator)
#endif
        ;
    }

    } // end namespace detail

    } // end namespace hoomd
