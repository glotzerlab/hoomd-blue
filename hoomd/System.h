// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "Updater.h"
#include "Analyzer.h"
#include "Compute.h"
#include "Integrator.h"
#include "Logger.h"
#include "Trigger.h"

#include <string>
#include <vector>
#include <map>

#ifndef __SYSTEM_H__
#define __SYSTEM_H__

#ifdef ENABLE_MPI
//! Forward declarations
class Communicator;
#endif

/*! \file System.h
    \brief Declares the System class and associated helper classes
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

//! Ties Analyzers, Updaters, and Computes together to run a full MD simulation
/*! The System class is responsible for making all the time steps in an MD simulation.
    It brings Analyzers, Updaters, and Computes all in one place to implement the full
    simulation. Any number of Analyzers and Updaters can be added, but only one Integrator.

    Usage: Add the Analyzers and Updaters, along with an Integrator to the System.
    Then call run() with the desired number of time steps to execute.
    Any added Analyzers or Updaters can be removed as desired and run()
    can be called multiple times if a multiple stage simulation is needed.

    Calling run() will step forward the specified number of time steps.
    During each time step, the Analyzers added have their Analyzer::analyze()
    methods called first, in the order in which they were added. A period
    can be specified when adding the Analyzer so that it only runs every so
    often. Then, all Updaters have their Updater::update() methods called,
    in order and with a specified period as with the analyzers. Finally, the
    Integrator::update() method is called to advance the simulation forward
    one step and the process is repeated again.

    \note Adding/removing/accessing analyzers, updaters, and computes by name
    is meant to be a once per simulation operation. In other words, the accesses
    are not optimized.

    See \ref page_system_class_design for more info.

    \ingroup hoomd_lib
*/
class PYBIND11_EXPORT System
    {
    public:
        //! Constructor
        System(std::shared_ptr<SystemDefinition> sysdef, unsigned int initial_tstep);

        // -------------- Compute get/set methods

        //! Adds a Compute
        void addCompute(std::shared_ptr<Compute> compute, const std::string& name);

        //! Overwrites a Compute
        void overwriteCompute(std::shared_ptr<Compute> compute, const std::string& name);

        //! Removes a Compute
        void removeCompute(const std::string& name);

        //! Access a stored Compute by name
        std::shared_ptr<Compute> getCompute(const std::string& name);

        // -------------- Integrator methods

        //! Sets the current Integrator
        void setIntegrator(std::shared_ptr<Integrator> integrator);

        //! Gets the current Integrator
        std::shared_ptr<Integrator> getIntegrator();

#ifdef ENABLE_MPI
        // -------------- Methods for communication

        //! Sets the communicator
        void setCommunicator(std::shared_ptr<Communicator> comm);

        //! Returns the communicator
        std::shared_ptr<Communicator> getCommunicator()
            {
            return m_comm;
            }
#endif

        // -------------- Methods for running the simulation

        //! Runs the simulation for a number of time steps
        void run(unsigned int nsteps, unsigned int cb_frequency,
                 pybind11::object callback, double limit_hours=0.0f,
                 unsigned int limit_multiple=1);

        //! Configures profiling of runs
        void enableProfiler(bool enable);

        //! Toggle whether or not to print the status line and TPS for each run
        void enableQuietRun(bool enable)
            {
            m_quiet_run = enable;
            }

        //! Register logger
        void registerLogger(std::shared_ptr<Logger> logger);

        //! Sets the statistics period
        void setStatsPeriod(unsigned int seconds);

        //! Get the average TPS from the last run
        Scalar getLastTPS() const
            {
            return m_last_TPS;
            }

        //! Get the current time step
        unsigned int getCurrentTimeStep()
            {
            return m_cur_tstep;
            }

        // -------------- Misc methods

        //! Get the system definition
        std::shared_ptr<SystemDefinition> getSystemDefinition()
            {
            return m_sysdef;
            }

        //! Set autotuner parameters
        void setAutotunerParams(bool enable, unsigned int period);

        std::vector<std::pair<std::shared_ptr<Analyzer>, std::shared_ptr<Trigger> > >& getAnalyzers()
			{
			return m_analyzers;
			}

        std::vector<std::pair<std::shared_ptr<Updater>, std::shared_ptr<Trigger> > >& getUpdaters()
			{
			return m_updaters;
			}

    private:
        std::vector<std::pair<std::shared_ptr<Analyzer>,
                    std::shared_ptr<Trigger> > > m_analyzers; //!< List of analyzers belonging to this System

        std::vector<std::pair<std::shared_ptr<Updater>,
                    std::shared_ptr<Trigger> > > m_updaters; //!< List of updaters belonging to this System

        std::map< std::string, std::shared_ptr<Compute> > m_computes; //!< Named list of Computes belonging to this System

        std::shared_ptr<Integrator> m_integrator;     //!< Integrator that advances time in this System
        std::shared_ptr<SystemDefinition> m_sysdef;   //!< SystemDefinition for this System
        std::shared_ptr<Profiler> m_profiler;         //!< Profiler to profile runs

#ifdef ENABLE_MPI
        std::shared_ptr<Communicator> m_comm;         //!< Communicator to use
#endif
        unsigned int m_start_tstep;     //!< Initial time step of the current run
        unsigned int m_end_tstep;       //!< Final time step of the current run
        unsigned int m_cur_tstep;       //!< Current time step
        Scalar m_cur_tps;               //!< Current average TPS
        Scalar m_med_tps;               //!< Current median TPS
        std::vector<Scalar> m_tps_list; //!< vector containing the last 10 tps

        ClockSource m_clk;              //!< A clock counting time from the beginning of the run
        uint64_t m_last_status_time;    //!< Time (measured by m_clk) of the last time generateStatusLine() was called
        unsigned int m_last_status_tstep;   //!< Time step last time generateStatusLine() was called

        bool m_quiet_run;       //!< True to suppress the status line and TPS from being printed to stdout for each run
        bool m_profile;         //!< True if runs should be profiled
        unsigned int m_stats_period; //!< Number of seconds between statistics output lines

        // --------- Steps in the simulation run implemented in helper functions
        //! Sets up m_profiler and attaches/detaches to/from all computes, updaters, and analyzers
        void setupProfiling();

        //! Prints detailed statistics for all attached computes, updaters, and integrators
        void printStats();

        //! Resets stats for all contained classes
        void resetStats();

        //! Prints out a formatted status line
        void generateStatusLine();

        //! Get the flags needed for a particular step
        PDataFlags determineFlags(unsigned int tstep);

        Scalar m_last_TPS;  //!< Stores the average TPS from the last run
        std::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< Stored shared ptr to the execution configuration
    };

//! Exports the System class to python
void export_System(pybind11::module& m);

#endif
