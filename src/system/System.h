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

#include "Updater.h"
#include "Analyzer.h"
#include "Compute.h"
#include "Integrator.h"
#include "Logger.h"

#include <string>
#include <vector>
#include <map>

#include <boost/python.hpp>

#ifndef __SYSTEM_H__
#define __SYSTEM_H__

/*! \file System.h
	\brief Declares the System class and associated helper classes
*/

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
	in order and with a specified period as with the anlalyzers. Finally, the
	Integrator::update() method is called to advance the simulation forward 
	one step and the process is repeated again.
	
	\note Adding/removing/accessing analyzers, updaters, and computes by name
	is meant to be a once per simulation operation. In other words, the accesses
	are not optimized.
	
	See \ref page_system_class_design for more info.
	
	\ingroup hoomd_lib
*/
class System
	{
	public:
		//! Constructor
		System(boost::shared_ptr<SystemDefinition> sysdef, unsigned int initial_tstep);
		
		// -------------- Analyzer get/set methods
		
		//! Adds an Analyzer
		void addAnalyzer(boost::shared_ptr<Analyzer> analyzer, const std::string& name, unsigned int period);
		
		//! Removes an Analyzer
		void removeAnalyzer(const std::string& name);
		
		//! Access a stored Analyzer by name
		boost::shared_ptr<Analyzer> getAnalyzer(const std::string& name);
		
		//! Change the period of an Analyzer
		void setAnalyzerPeriod(const std::string& name, unsigned int period);
		
		//! Change the period of an Analyzer to be variable
		void setAnalyzerPeriodVariable(const std::string& name, boost::python::object update_func);
		
		//! Get the period of an Analyzer
		unsigned int getAnalyzerPeriod(const std::string& name);
		
		// -------------- Updater get/set methods
		
		//! Adds an Updater
		void addUpdater(boost::shared_ptr<Updater> updater, const std::string& name, unsigned int period);
		
		//! Removes an Updater
		void removeUpdater(const std::string& name);
		
		//! Access a stored Updater by name
		boost::shared_ptr<Updater> getUpdater(const std::string& name);
		
		//! Change the period of an Updater
		void setUpdaterPeriod(const std::string& name, unsigned int period);
		
		//! Change the period of an Updater to be variable
		void setUpdaterPeriodVariable(const std::string& name, boost::python::object update_func);
		
		//! Get the period of on Updater
		unsigned int getUpdaterPeriod(const std::string& name);
		
		// -------------- Compute get/set methods
		
		//! Adds a Compute
		void addCompute(boost::shared_ptr<Compute> compute, const std::string& name);
		
		//! Removes a Compute
		void removeCompute(const std::string& name);
		
		//! Access a stored Compute by name
		boost::shared_ptr<Compute> getCompute(const std::string& name);
		
		// -------------- Integrator methods
		
		//! Sets the current Integrator
		void setIntegrator(boost::shared_ptr<Integrator> integrator);
		
		//! Gets the current Integrator
		boost::shared_ptr<Integrator> getIntegrator();
		
		// -------------- Methods for running the simulation
		
		//! Runs the simulation for a number of time steps
		void run(unsigned int nsteps, double limit_hours=0.0f);
		
		//! Configures profiling of runs
		void enableProfiler(bool enable);
		
		//! Register logger
		void registerLogger(boost::shared_ptr<Logger> logger);
		
		//! Sets the statistics period
		void setStatsPeriod(unsigned int seconds);
		
		//! Get the average TPS from the last run
		Scalar getLastTPS()	const { return m_last_TPS; }
		
		//! Get the current time step
		unsigned int getCurrentTimeStep() { return m_cur_tstep; }
		
	private:
		//! Holds an item in the list of analyzers
		struct analyzer_item
			{
			//! Constructor
			/*! \param analyzer the Analyzer shared pointer to store
				\param name user defined name of the analyzer
				\param period number of time steps between calls to Analyzer::analyze() for this analyzer
				\param created_tstep time step the analyzer was created on
			*/
			analyzer_item(boost::shared_ptr<Analyzer> analyzer, const std::string& name, unsigned int period,
							unsigned int created_tstep)
				: m_analyzer(analyzer), m_name(name), m_period(period), m_created_tstep(created_tstep), m_next_execute_tstep(created_tstep), m_is_variable_period(false), m_n(1)
				{
				}
			
			//! Tets if this analyzer should be executed
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
						boost::python::object pynext = m_update_func(m_n);
						int next = (int)boost::python::extract<float>(pynext) + m_created_tstep;
						
						if (next < 0)
							{
							cout << "***Warning! Variable period returned a negative value. Increasing to 1 to prevent inconsistancies" << endl;
							next = 1;
							}
							
						if ((unsigned int)next <= tstep)
							{
							cout << "***Warning! Variable period returned a value equal to the current timestep. Increasing by 1 to prevent inconsistancies" << endl;
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
			void setVariablePeriod(boost::python::object update_func, unsigned int tstep)
				{
				m_update_func = update_func;
				m_next_execute_tstep = tstep;
				m_is_variable_period = true;
				}
				
			boost::shared_ptr<Analyzer> m_analyzer;	//!< The analyzer
			std::string m_name;						//!< Its name
			unsigned int m_period;					//!< The period between analyze() calls
			unsigned int m_created_tstep;			//!< The timestep when the analyzer was added
			unsigned int m_next_execute_tstep;		//!< The next time step we will execute on
			bool m_is_variable_period;				//!< True if the variable period should be used
			
			unsigned int m_n;						//!< Current value of n for the variable period func
			boost::python::object m_update_func;	//!< Python lambda function to evaluate time steps to update at
			};
			
		std::vector<analyzer_item> m_analyzers;	//!< List of analyzers belonging to this System
		
		//! Holds an item in the list of updaters
		struct updater_item
			{
			//! Constructor
			/*! \param updater the Updater shared pointer to store
				\param name user defined name of the updater
				\param period number of time steps between calls to Updater::update() for this updater
				\param created_tstep time step the analyzer was created on
			*/
			updater_item(boost::shared_ptr<Updater> updater, const std::string& name, unsigned int period,
							unsigned int created_tstep)
				: m_updater(updater), m_name(name), m_period(period), m_created_tstep(created_tstep), m_next_execute_tstep(created_tstep), m_is_variable_period(false), m_n(1)
				{
				}
			
			//! Tets if this updater should be executed
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
						boost::python::object pynext = m_update_func(m_n);
						int next = (int)boost::python::extract<float>(pynext) + m_created_tstep;
						
						if (next < 0)
							{
							cout << "***Warning! Variable period returned a negative value. Increasing to 1 to prevent inconsistancies" << endl;
							next = 1;
							}
							
						if ((unsigned int)next <= tstep)
							{
							cout << "***Warning! Variable period returned a value equal to the current timestep. Increasing by 1 to prevent inconsistancies" << endl;
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
			void setVariablePeriod(boost::python::object update_func, unsigned int tstep)
				{
				m_update_func = update_func;
				m_next_execute_tstep = tstep;
				m_is_variable_period = true;
				}
				
			boost::shared_ptr<Updater> m_updater;	//!< The analyzer
			std::string m_name;						//!< Its name
			unsigned int m_period;					//!< The period between analyze() calls
			unsigned int m_created_tstep;			//!< The timestep when the analyzer was added
			unsigned int m_next_execute_tstep;		//!< The next time step we will execute on
			bool m_is_variable_period;				//!< True if the variable period should be used
			
			unsigned int m_n;						//!< Current value of n for the variable period func
			boost::python::object m_update_func;	//!< Python lambda function to evaluate time steps to update at
			};
			
		std::vector<updater_item> m_updaters;	//!< List of updaters belonging to this System
		
		std::map< std::string, boost::shared_ptr<Compute> > m_computes;	//!< Named list of Computes belonging to this System
		
		boost::shared_ptr<Integrator> m_integrator;		//!< Integrator that advances time in this System
		boost::shared_ptr<SystemDefinition> m_sysdef;	//!< SystemDefinition for this System
		boost::shared_ptr<Profiler> m_profiler;			//!< Profiler to profile runs
		unsigned int m_start_tstep;		//!< Intial time step of the current run
		unsigned int m_end_tstep;		//!< Final time step of the current run
		unsigned int m_cur_tstep;		//!< Current time step
		
		ClockSource m_clk;				//!< A clock counting time from the beginning of the run
		uint64_t m_last_status_time;	//!< Time (measured by m_clk) of the last time generateStatusLine() was called
		unsigned int m_last_status_tstep;	//!< Time step last time generateStatusLine() was called
		
		bool m_profile;			//!< True if runs should be profiled
		unsigned int m_stats_period; //!< Number of seconds between statistics output lines
		
		// --------- Steps in the simulation run implemented in helper functions
		//! Sets up m_profiler and attaches/detaches to/from all computes, updaters, and analyzers
		void setupProfiling();

		//! Prints detailed statistics for all attached computes, updaters, and integrators
		void printStats();
		
		//! Prints out a formatted status line
		void generateStatusLine();
		
		// --------- Helper function for handling lists
		//! Search for an Analyzer by name
		std::vector<analyzer_item>::iterator findAnalyzerItem(const std::string &name);
		//! Search for an Updater by name
		std::vector<updater_item>::iterator findUpdaterItem(const std::string &name);

		Scalar m_last_TPS;	//!< Stores the average TPS from the last run
	};
	
//! Exports the System class to python
void export_System();
	
#endif
