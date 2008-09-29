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

// $Id: NVEUpdater.h 1085 2008-09-26 20:22:24Z phillicl $
// $URL: http://svn2.assembla.com/svn/hoomd/trunk/src/updaters/BD_NVTUpdater.h $

/*! \file NVEUpdater.h
	\brief Declares an updater that implements NVE dynamics
*/

#include "NVEUpdater.h"
#include "Integrator.h"
#include <StochasticForceCompute.h>
#include <vector>
#include <boost/shared_ptr.hpp>

#ifndef __BD_NVTUPDATER_H__
#define __BD_NVTUPDATER_H__

//! Updates particle positions and velocities
/*! This updater performes constant N, constant volume, constant T dynamics, by applying a stochastic heat bath on top of a NVE Simulation
    Particle positions and velocities are updated according to the velocity verlet algorithm. The forces that drive this motion are defined external to this class
	in ForceCompute. Any number of ForceComputes can be given, the resulting forces will be summed to produce a net force on 
	each particle.
	
	\ingroup updaters
*/
class BD_NVTUpdater : public NVEUpdater
	{
	public:
		//! Constructor
		BD_NVTUpdater(boost::shared_ptr<ParticleData> pdata, Scalar deltaT, Scalar Temp);
				
		
		//! Update the temperature
		/*! \param T New temperature to set
		*/
		virtual void setT(Scalar T);
		
		//! Returns a list of log quantities this compute calculates
		virtual std::vector< std::string > getProvidedLogQuantities(); 
		
		//! Calculates the requested log value and returns it
		virtual Scalar getLogValue(const std::string& quantity);
		
	protected:
    boost::shared_ptr<StochasticForceCompute> m_bdfc;	
	Scalar m_T;			//!< Temperature set point

	};
	
#ifdef USE_PYTHON
//! Exports the BD_NVTUpdater class to python
void export_BD_NVTUpdater();
#endif
	
#endif
