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

// $Id: BD_NVTUpdater.h 1085 2008-07-30 20:22:24Z joaander $
// $URL: http://svn2.assembla.com/svn/hoomd/trunk/src/updaters/BD_NVTUpdater.h $

/*! \file BD_NVTUpdater.h
	\brief Declares an updater that implements BD_NVT dynamics
*/

#include "Updater.h"
#include "NVEUpdater.h"
#include "StochasticForceCompute.h"
#ifdef ENABLE_CUDA
#include "StochasticForceComputeGPU.h"
#endif
#include <vector>
#include <boost/shared_ptr.hpp>

#ifndef __BD_NVTUPDATER_H__
#define __BD_NVTUPDATER_H__

//! Brownian dynamics integration of particles
/*! This updater performes constant N, constant volume, constant energy dynamics, with 
	random and drag forces applied (BD_NVT). Particle positions and velocities are updated 
	according to the velocity verlet algorithm. The forces that drive this motion are 
	defined external to this class in ForceCompute. Any number of ForceComputes can be given, 
	the resulting forces will be summed to produce a net force on each particle.
	
	BD_NVTUpdater internally creates it's own StochasticForceCompute to handle the calculation
	of the stochastic and drag forces needed for BD.
	
	\ingroup updaters
*/
class BD_NVTUpdater : public NVEUpdater
	{
	public:
		//! Constructor
		BD_NVTUpdater(boost::shared_ptr<ParticleData> pdata, Scalar deltaT, boost::shared_ptr<Variant> Temp, unsigned int seed);
		
		//! Sets the Stochastic Bath Temperature
		void setT(boost::shared_ptr<Variant> Temp);

		//! Resets the simulation timestep
		void setDeltaT(Scalar deltaT) 
			{
			assert(m_bdfc);
			m_bdfc->setDeltaT(deltaT); 
			Integrator::setDeltaT(deltaT);
			}

		//! Sets the type-dependant drag coefficient
		/*! \param type Particle type index to set the coefficient on
			\param gamma Drag coefficient to set
		*/
		void setGamma(unsigned int type, Scalar gamma)
			{
			assert(m_bdfc);
			m_bdfc->setParams(type,gamma);
			}
				
		//! Removes all ForceComputes from the list
		virtual void removeForceComputes();
		
		//! Take one timestep forward
		virtual void update(unsigned int timestep);
		
	protected:
		boost::shared_ptr<Variant> m_T;			//!< The Temperature of the Stochastic Bath
		unsigned int m_seed;					//!< The seed for the RNG of the Stochastic Bath 
		bool m_bath;							//!< Whether the bath has been set or not
		
		//! Attaches the Stochastic Bath Temperature
		void addStochasticBath();
		
		//! The StochasticForceCompute to use in applying the temperature bath
		boost::shared_ptr<StochasticForceCompute> m_bdfc;
	};
	
//! Exports the BD_NVTUpdater class to python
void export_BD_NVTUpdater();
	
#endif
