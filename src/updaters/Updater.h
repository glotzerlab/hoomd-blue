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

#include "ParticleData.h"
#include "Profiler.h"

#ifndef __UPDATER_H__
#define __UPDATER_H__

/*! \file Updater.h
	\brief Declares a base class for all updaters
*/

/*! \ingroup hoomd_lib
	@{
*/

/*! \defgroup updaters Updaters
	\brief All classes that implement the Updater concept.
	\details See \ref page_dev_info for more information
*/

/*! @}
*/

//! Performs updates of ParticleData structures
/*! The Updater is an abstract concept that takes a particle data structure and changes it in some way.
	For example, an updater may make a verlet step and update the particle positions to the next timestep.
	Or, it may force a certain particle to be in a certain location. Or, it may sort the particle data
	so that the many Computes suffer far fewer cache misses. The possibilities are endless.
	
	The base class just defines an update method. Since updaters can reference Compute's, the timestep
	is passed in so that it can be forwarded on to the Compute. Of course, the timestep can also be used
	for time dependant updaters, such as a moving temperature set point. Of course, when an updater is changing
	particle positions/velocities etc... the line between when a timestep begins and ends blurs. See the System class
	for a clear definition.

	See \ref page_dev_info for more information
	
	\ingroup updaters
*/
class Updater : boost::noncopyable
	{
	public:
		//! Constructs the compute and associates it with the ParticleData
		Updater(boost::shared_ptr<ParticleData> pdata);
		virtual ~Updater() {};
		
		//! Abstract method that performs the update
		/*! Derived classes will implement this method to perform their specific update
			\param timestep Current time step of the simulation
		*/
		virtual void update(unsigned int timestep) = 0;
				
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
			
			The base class just returns 0. Derived classes should override this behavior and return
			the calculated value for the given quantity. Only quantities listed in
			the return value getProvidedLogQuantities() will be requested from 
			getLogValue().

			See Logger for more information on what this is about.
		*/
		virtual Scalar getLogValue(const std::string& quantity) { return Scalar(0.0); }
		
	protected:
		const boost::shared_ptr<ParticleData> m_pdata;	//!< The particle data this compute is associated with
		boost::shared_ptr<Profiler> m_prof;				//!< The profiler this compute is to use
		const ExecutionConfiguration& exec_conf;			//!< Cached reference to the execution configuration
		
		
	};
	
//! Export the Updater class to python
void export_Updater();
		
#endif

