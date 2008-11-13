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

#include <boost/shared_ptr.hpp>
#include <boost/utility.hpp>
#include <string>
#include <vector>

#include "ParticleData.h"
#include "Profiler.h"

#ifndef __COMPUTE_H__
#define __COMPUTE_H__

/*! \file Compute.h
	\brief Declares a base class for all computes
*/

/*! \ingroup hoomd_lib
	@{
*/

/*! \defgroup computes Computes
	\brief All classes that implement the Compute concept.
	\details See \ref page_dev_info for more information
*/

/*! @}
*/

//! Performs computations on ParticleData structures
/*! The Compute is an abstract concept that performs some kind of computation on the
	particles in a ParticleData structure. This computation is to be done by reading
	the particle data only, no writing. Computes will be used to generate neighbor lists,
	calculate forces, and calculate temperatures, just to name a few.
	
	For performance and simplicity, each compute is associated with a ParticleData
	on construction. ParticleData pointers are managed with reference counted boost::shared_ptr.
	Since each ParticleData cannot change size, this allows the Compute to preallocate 
	any data structures that it may need.
	
	Computes may be referenced more than once and may reference other computes. To prevent
	uneeded data from being calculated, the time step will be passed into the compute
	method so that it can skip caculations if they have already been done this timestep.
	For convenience, the base class will provide a shouldCompute() method that implements
	this behaviour. Derived classes can override if more complicated behavior is needed.

	See \ref page_dev_info for more information
	\ingroup computes
*/
class Compute : boost::noncopyable
	{
	public:
		//! Constructs the compute and associates it with the ParticleData
		Compute(boost::shared_ptr<ParticleData> pdata);
		virtual ~Compute() {};
		
		//! Abstract method that performs the computation
		/*! \param timestep Current time step
			Derived classes will implement this method to calculate their results*/
		virtual void compute(unsigned int timestep) = 0;
		
		//! Print some basic stats to stdout
		/*! Derived classes can optionally implement this function. A System will
			call all of the Compute's printStats functions at the end of a run 
			so the user can see useful information */
		virtual void printStats()
			{
			}

		//! Sets the profiler for the compute to use
		void setProfiler(boost::shared_ptr<Profiler> prof);
		
		//! Returns a list of log quantities this compute calculates
		/*! The base class implementation just returns an empty vector. Derived classes should override
			this behavior and return a list of quantities that they log.			

			See Logger for more information on what this is about.
		*/
		virtual std::vector< std::string > getProvidedLogQuantities() { return std::vector< std::string >(); }
		
		//! Calculates the requested log value and returns it
		/*! \param quantity Name of the log quantity to get
			\param timestep Current time step of the simulation
			
			The base class just returns 0. Derived classes should override this behavior and return
			the calculated value for the given quantity. Only quantities listed in
			the return value getProvidedLogQuantities() will be requested from 
			getLogValue().

			See Logger for more information on what this is about.
		*/
		virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep) { return Scalar(0.0); }
		
	protected:
		const boost::shared_ptr<ParticleData> m_pdata;		//!< The particle data this compute is associated with
		boost::shared_ptr<Profiler> m_prof;					//!< The profiler this compute is to use
		const ExecutionConfiguration& exec_conf;			//!< Cached reference to the execution configuration
		
		//! Simple method for testing if the computation should be run or not
		virtual bool shouldCompute(unsigned int timestep);		
	private:
		unsigned int m_last_computed;	//!< Stores the last timestep compute was called
		bool m_first_compute;			//!< true if compute has not yet been called
		
		//! The python export needs to be a friend to export shouldCompute()
		friend void export_Compute();
	};
	
//! Exports the Compute class to python
void export_Compute();
		
#endif

