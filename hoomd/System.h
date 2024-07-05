// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "Analyzer.h"
#include "Compute.h"
#include "Integrator.h"
#include "Tuner.h"
#include "Updater.h"

#include <map>
#include <string>
#include <vector>

#ifndef __SYSTEM_H__
#define __SYSTEM_H__

/*! \file System.h
    \brief Declares the System class and associated helper classes
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

namespace hoomd
    {
#ifdef ENABLE_MPI
//! Forward declarations
class Communicator;
#endif

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
    System(std::shared_ptr<SystemDefinition> sysdef, uint64_t initial_tstep);

    // -------------- Integrator methods

    //! Sets the current Integrator
    void setIntegrator(std::shared_ptr<Integrator> integrator);

    //! Gets the current Integrator
    std::shared_ptr<Integrator> getIntegrator();

#ifdef ENABLE_MPI
    // -------------- Methods for communication

    //! Sets the communicator
    void setCommunicator(std::shared_ptr<Communicator> comm);
#endif

    // -------------- Methods for running the simulation

    /** Run the simulation for a number of time steps.

        During the run, Simulation applies all of the Tuners, Updaters, the integrator,
        and Analyzers who's triggers evaluate true.

        @param nsteps Number of steps to advance the simulation
        @param write_at_start Set to true to evaluate writers before the
            loop
    */
    void run(uint64_t nsteps, bool write_at_start = false);

    //! Get the average TPS from the last run
    Scalar getLastTPS()
        {
#ifdef ENABLE_MPI
        // make sure all ranks return the same TPS
        if (m_sysdef->isDomainDecomposed())
            {
            bcast(m_last_TPS, 0, m_exec_conf->getMPICommunicator());
            }
#endif
        return m_last_TPS;
        }

    //! Get the current time step
    uint64_t getCurrentTimeStep()
        {
        return m_cur_tstep;
        }

    /// Get the current wall time
    double getCurrentWalltime()
        {
#ifdef ENABLE_MPI
        // make sure all ranks return the same walltime
        if (m_sysdef->isDomainDecomposed())
            {
            bcast(m_last_walltime, 0, m_exec_conf->getMPICommunicator());
            }
#endif
        return m_last_walltime;
        }

    /// Get the end time step
    uint64_t getEndStep()
        {
        return m_end_tstep;
        }

    /// Get the end time step
    uint64_t getStartStep()
        {
        return m_start_tstep;
        }

    // -------------- Misc methods

    //! Get the system definition
    std::shared_ptr<SystemDefinition> getSystemDefinition()
        {
        return m_sysdef;
        }

    std::vector<std::shared_ptr<Analyzer>>& getAnalyzers()
        {
        return m_analyzers;
        }

    std::vector<std::shared_ptr<Updater>>& getUpdaters()
        {
        return m_updaters;
        }

    std::vector<std::shared_ptr<Tuner>>& getTuners()
        {
        return m_tuners;
        }

    std::vector<std::shared_ptr<Compute>>& getComputes()
        {
        return m_computes;
        }

    /// Set pressure computation particle data flag
    void setPressureFlag(bool flag)
        {
        m_default_flags[pdata_flag::pressure_tensor] = flag;
        }

    /// Get the pressure computation particle data flag
    bool getPressureFlag()
        {
        return m_default_flags[pdata_flag::pressure_tensor];
        }

    /// Get the particle group cache.
    std::vector<std::shared_ptr<ParticleGroup>>& getGroupCache()
        {
        return m_group_cache;
        }

    /// Trigger an update of the group degrees of freedom.
    void updateGroupDOFOnNextStep()
        {
        m_update_group_dof_next_step = true;
        }

    private:
    /// Update the number of degrees of freedom in cached groups
    void updateGroupDOF();

    std::vector<std::shared_ptr<Analyzer>>
        m_analyzers; //!< List of analyzers belonging to this System

    std::vector<std::shared_ptr<Updater>> m_updaters; //!< List of updaters belonging to this System

    std::vector<std::shared_ptr<Tuner>> m_tuners; //!< List of tuners belonging to the System

    std::vector<std::shared_ptr<Compute>> m_computes; //!< list of Computes belonging to this System

    std::shared_ptr<Integrator> m_integrator;   //!< Integrator that advances time in this System
    std::shared_ptr<SystemDefinition> m_sysdef; //!< SystemDefinition for this System

#ifdef ENABLE_MPI
    /// The system's communicator.
    std::shared_ptr<Communicator> m_comm;
#endif

    uint64_t m_start_tstep; //!< Initial time step of the current run
    uint64_t m_end_tstep;   //!< Final time step of the current run
    uint64_t m_cur_tstep;   //!< Current time step

    ClockSource m_clk; //!< A clock counting time from the beginning of the run

    /// Particle data flags to always set
    PDataFlags m_default_flags;

    // --------- Steps in the simulation run implemented in helper functions
    //! Resets stats for all contained classes
    void resetStats();

    //! Get the flags needed for a particular step
    PDataFlags determineFlags(uint64_t tstep);

    /// Record the initial time of the last run
    int64_t m_initial_time = 0;

    /// Store the last recorded tPS
    double m_last_TPS = 0;

    /// Store the last recorded walltime
    double m_last_walltime = 0;

    /// Update the TPS average
    void updateTPS();

    std::shared_ptr<const ExecutionConfiguration>
        m_exec_conf; //!< Stored shared ptr to the execution configuration

    /// Cache of ParticleGroup objects
    std::vector<std::shared_ptr<ParticleGroup>> m_group_cache;

    /// Flag to trigger update of group degrees of freedom
    bool m_update_group_dof_next_step = false;
    };

namespace detail
    {
//! Exports the System class to python
void export_System(pybind11::module& m);

    } // end namespace detail

    } // end namespace hoomd

#endif
