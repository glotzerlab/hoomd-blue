// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "Updater.h"
#include "Analyzer.h"
#include "Compute.h"
#include "Integrator.h"
#include "Logger.h"

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

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

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

        // -------------- Analyzer get/set methods

        //! Adds an Analyzer
        void addAnalyzer(std::shared_ptr<Analyzer> analyzer, const std::string& name, unsigned int period, int phase);

        //! Removes an Analyzer
        void removeAnalyzer(const std::string& name);

        //! Access a stored Analyzer by name
        std::shared_ptr<Analyzer> getAnalyzer(const std::string& name);

        //! Change the period of an Analyzer
        void setAnalyzerPeriod(const std::string& name, unsigned int period, int phase);

        //! Change the period of an Analyzer to be variable
        void setAnalyzerPeriodVariable(const std::string& name, pybind11::object update_func);

        //! Get the period of an Analyzer
        unsigned int getAnalyzerPeriod(const std::string& name);

        // -------------- Updater get/set methods

        //! Adds an Updater
        void addUpdater(std::shared_ptr<Updater> updater, const std::string& name, unsigned int period, int phase);

        //! Removes an Updater
        void removeUpdater(const std::string& name);

        //! Access a stored Updater by name
        std::shared_ptr<Updater> getUpdater(const std::string& name);

        //! Change the period of an Updater
        void setUpdaterPeriod(const std::string& name, unsigned int period, int phase);

        //! Change the period of an Updater to be variable
        void setUpdaterPeriodVariable(const std::string& name, pybind11::object update_func);

        //! Get the period of on Updater
        unsigned int getUpdaterPeriod(const std::string& name);

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

    private:
        //! Holds an item in the list of analyzers
        struct analyzer_item
            {
            //! Constructor
            /*! \param analyzer the Analyzer shared pointer to store
                \param name user defined name of the analyzer
                \param period number of time steps between calls to Analyzer::analyze() for this analyzer
                \param created_tstep time step the analyzer was created on
                \param next_execute_tstep time step to first execute the analyzer
            */
            analyzer_item(std::shared_ptr<Analyzer> analyzer, const std::string& name, unsigned int period,
                          unsigned int created_tstep, unsigned int next_execute_tstep)
                    : m_analyzer(analyzer), m_name(name), m_period(period), m_created_tstep(created_tstep), m_next_execute_tstep(next_execute_tstep), m_is_variable_period(false), m_n(1)
                {
                }

            //! Test if this analyzer should be executed
            /*! \param tstep Current simulation step
                \returns true if the Analyzer should be executed this \a tstep
                \note This function maintains state and should only be called once per time step
            */
            bool shouldExecute(unsigned int tstep)
                {
                if (tstep == m_next_execute_tstep)
                    {
                    if (m_is_variable_period)
                        {
                        pybind11::object pynext = m_update_func(m_n);
                        int next = pybind11::cast<int>(pynext) + m_created_tstep;

                        if (next < 0)
                            {
                            m_analyzer->getExecConf()->msg->warning() << "Variable period returned a negative value. Increasing to 1 to prevent inconsistencies" << std::endl;
                            next = 1;
                            }

                        if ((unsigned int)next <= tstep)
                            {
                            m_analyzer->getExecConf()->msg->warning() << "Variable period returned a value equal to the current timestep. Increasing by 1 to prevent inconsistencies" << std::endl;
                            next = tstep+1;
                            }

                        m_next_execute_tstep = next;
                        m_n++;
                        }
                    else
                        {
                        m_next_execute_tstep += m_period;
                        }
                    return true;
                    }
                else
                    return false;
                }

            //! Peek if this analyzer will execute on the given step
            /*! \param tstep Requested simulation step
                \returns true if the Analyze will be executed on \a tstep

                peekExecute will return true for the same step that shouldExecute will. However, peekExecute does not
                update any internal state. It offers a way to peek and determine if a given step will be the very next
                step that the analyzer is to be called.
            */
            bool peekExecute(unsigned int tstep)
                {
                return (tstep == m_next_execute_tstep);
                }


            //! Changes the period
            /*! \param period New period to set
                \param tstep current time step
            */
            void setPeriod(unsigned int period, unsigned int tstep)
                {
                m_period = period;
                m_next_execute_tstep = tstep;
                m_is_variable_period = false;
                }

            //! Changes to a variable period
            /*! \param update_func A python callable function. \a update_func(n) should return a positive integer which is the time step to update at frame n
                \param tstep current time step

                \a n is initialized to 1 when the period func is changed. Each time a new output is made, \a period_func is evaluated to
                calculate the period to the next time step to make an output. \a n is then incremented by one.
            */
            void setVariablePeriod(pybind11::object update_func, unsigned int tstep)
                {
                m_update_func = update_func;
                m_next_execute_tstep = tstep;
                m_is_variable_period = true;
                }

            std::shared_ptr<Analyzer> m_analyzer; //!< The analyzer
            std::string m_name;                     //!< Its name
            unsigned int m_period;                  //!< The period between analyze() calls
            unsigned int m_created_tstep;           //!< The timestep when the analyzer was added
            unsigned int m_next_execute_tstep;      //!< The next time step we will execute on
            bool m_is_variable_period;              //!< True if the variable period should be used

            unsigned int m_n;                       //!< Current value of n for the variable period func
            pybind11::object m_update_func;    //!< Python lambda function to evaluate time steps to update at
            };

        std::vector<analyzer_item> m_analyzers; //!< List of analyzers belonging to this System

        //! Holds an item in the list of updaters
        struct updater_item
            {
            //! Constructor
            /*! \param updater the Updater shared pointer to store
                \param name user defined name of the updater
                \param period number of time steps between calls to Updater::update() for this updater
                \param created_tstep time step the analyzer was created on
                \param next_execute_tstep time step to first execute the analyzer
            */
            updater_item(std::shared_ptr<Updater> updater, const std::string& name, unsigned int period,
                         unsigned int created_tstep, unsigned int next_execute_tstep)
                    : m_updater(updater), m_name(name), m_period(period), m_created_tstep(created_tstep), m_next_execute_tstep(next_execute_tstep), m_is_variable_period(false), m_n(1)
                {
                }

            //! Test if this updater should be executed
            /*! \param tstep Current simulation step
                \returns true if the Updater should be executed this \a tstep
                \note This function maintains state and should only be called once per time step
            */
            bool shouldExecute(unsigned int tstep)
                {
                if (tstep == m_next_execute_tstep)
                    {
                    if (m_is_variable_period)
                        {
                        pybind11::object pynext = m_update_func(m_n);
                        int next = pybind11::cast<int>(pynext) + m_created_tstep;

                        if (next < 0)
                            {
                            m_updater->getExecConf()->msg->warning() << "Variable period returned a negative value. Increasing to 1 to prevent inconsistencies" << std::endl;
                            next = 1;
                            }

                        if ((unsigned int)next <= tstep)
                            {
                            m_updater->getExecConf()->msg->warning() << "Variable period returned a value equal to the current timestep. Increasing by 1 to prevent inconsistencies" << std::endl;
                            next = tstep+1;
                            }

                        m_next_execute_tstep = next;
                        m_n++;
                        }
                    else
                        {
                        m_next_execute_tstep += m_period;
                        }
                    return true;
                    }
                else
                    return false;
                }

            //! Peek if this updater will execute on the given step
            /*! \param tstep Requested simulation step
                \returns true if the Analyze will be executed on \a tstep

                peekExecute will return true for the same step that shouldExecute will. However, peekExecute does not
                update any internal state. It offers a way to peek and determine if a given step will be the very next
                step that the analyzer is to be called.
            */
            bool peekExecute(unsigned int tstep)
                {
                return (tstep == m_next_execute_tstep);
                }

            //! Changes the period
            /*! \param period New period to set
                \param tstep current time step
            */
            void setPeriod(unsigned int period, unsigned int tstep)
                {
                m_period = period;
                m_next_execute_tstep = tstep;
                m_is_variable_period = false;
                }

            //! Changes to a variable period
            /*! \param update_func A python callable function. \a update_func(n) should return a positive integer which is the time step to update at frame n
                \param tstep current time step

                \a n is initialized to 1 when the period func is changed. Each time a new output is made, \a period_func is evaluated to
                calculate the period to the next time step to make an output. \a n is then incremented by one.
            */
            void setVariablePeriod(pybind11::object update_func, unsigned int tstep)
                {
                m_update_func = update_func;
                m_next_execute_tstep = tstep;
                m_is_variable_period = true;
                }

            std::shared_ptr<Updater> m_updater;   //!< The analyzer
            std::string m_name;                     //!< Its name
            unsigned int m_period;                  //!< The period between analyze() calls
            unsigned int m_created_tstep;           //!< The timestep when the analyzer was added
            unsigned int m_next_execute_tstep;      //!< The next time step we will execute on
            bool m_is_variable_period;              //!< True if the variable period should be used

            unsigned int m_n;                       //!< Current value of n for the variable period func
            pybind11::object m_update_func;    //!< Python lambda function to evaluate time steps to update at
            };

        std::vector<updater_item> m_updaters;   //!< List of updaters belonging to this System

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

        // --------- Helper function for handling lists
        //! Search for an Analyzer by name
        std::vector<analyzer_item>::iterator findAnalyzerItem(const std::string &name);
        //! Search for an Updater by name
        std::vector<updater_item>::iterator findUpdaterItem(const std::string &name);

        Scalar m_last_TPS;  //!< Stores the average TPS from the last run
        std::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< Stored shared ptr to the execution configuration
    };

//! Exports the System class to python
void export_System(pybind11::module& m);

#endif
