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
// Maintainer: joaander

/*! \file NPTUpdater.h
	\brief Declares the NPTUpdater class
*/

#include "Updater.h"
#include "Integrator.h"
#include "Variant.h"
#include <vector>
#include <boost/shared_ptr.hpp>

#ifndef __NPTUPDATER_H__
#define __NPTUPDATER_H__

//! NPT Integration via the Nose-Hoover thermostat and Anderson barostat
/*! This updater performes constant N, constant pressure, constant temperature (NVT) dynamics. Particle positions and velocities are 
	updated according to the Nose-Hoover/Anderson algorithm. The forces that drive this motion are defined external to this class
	in ForceCompute. Any number of ForceComputes can be given, the resulting forces will be summed to produce a net force on 
	each particle.
	
	\ingroup updaters
*/
class NPTUpdater : public Integrator
	{
	public:
		//! Constructor
	        NPTUpdater(boost::shared_ptr<ParticleData> pdata, Scalar deltaT, Scalar tau, Scalar tauP, boost::shared_ptr<Variant> T, boost::shared_ptr<Variant> P);
		
		//! Take one timestep forward
		virtual void update(unsigned int timestep);
		
		//! Update the temperature
		/*! \param T New temperature to set
		*/
		virtual void setT(boost::shared_ptr<Variant> T) { m_T = T; }
				
		//! Update the tau value
		/*! \param tau New time constant to set
		*/		
		virtual void setTau(Scalar tau) { m_tau = tau; }

		//! Update the pressure
		/*! \param P New pressure to set
		*/
		virtual void setP(boost::shared_ptr<Variant> P) { m_P = P; }

		//! Update the nuP value
		/*! \param tauP New pressure constant to set
		*/		
		virtual void setTauP(Scalar tauP) { m_tauP = tauP; }
		
		//! Calculates the requested log value and returns it
		virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep);

		//! Computes current pressure
		virtual Scalar computePressure(unsigned int timestep);
		
		//! Computes current temperature
		virtual Scalar computeTemperature(unsigned int timestep);

	protected:
		Scalar m_tau;					//!< tau value for Nose-Hoover
		Scalar m_tauP;					//!< tauP value for the barostat
		boost::shared_ptr<Variant> m_T;	//!< Temperature set point
		boost::shared_ptr<Variant> m_P;	//!< Pressure set point
		Scalar m_Xi;					//!< Friction coeff
		Scalar m_Eta;					//!< barostat friction
		bool m_accel_set;				//!< Flag to tell if we have set the accelleration yet
		Scalar m_curr_P;				//!< Current (instanteneous) pressure
		Scalar m_curr_T;				//!< Current (instanteneous)temperature
		Scalar m_V;						//!< Current volume
		Scalar m_Lx;					//!< Box length in x direction
		Scalar m_Ly;					//!< Box length in y direction
		Scalar m_Lz;					//!< Box length in z direction
		
	};
	
//! Exports the NPTUpdater class to python
void export_NPTUpdater();

#endif
