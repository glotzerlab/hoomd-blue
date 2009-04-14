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

// $Id: StochasticForceCompute.h 1119 2008-09-22 21:01:37Z phillicl $
// $URL: http://svn2.assembla.com/svn/hoomd/trunk/src/computes/StochasticForceCompute.h $

/*! \file StochasticForceCompute.h
	\brief Declares the StochasticForceCompute class
*/

#include <boost/shared_ptr.hpp>
#include "ForceCompute.h"
#include "saruprng.h"
#include "Variant.h"

#ifndef __StochasticForceCompute__
#define __StochasticForceCompute__


//! Adds a stochastic force to each particle and a drag force based on its velocity
/*! 
	\ingroup computes
*/
class StochasticForceCompute :	public ForceCompute
	{
	public:
		//! Constructor
		StochasticForceCompute(boost::shared_ptr<ParticleData> pdata, Scalar deltaT, boost::shared_ptr<Variant> Temp, unsigned int seed);

		//! Destructor
		~StochasticForceCompute();
		
		//! Sets force parameters
		virtual void setParams(unsigned int typ, Scalar gamma);
		
		//! Sets Temperature Parameter
		virtual void setT(boost::shared_ptr<Variant> Temp) { m_T = Temp; }
		
		//! Sets timestep Parameter
		virtual void setDeltaT(Scalar deltaT) { m_dt = deltaT; }		
		
		//! Returns a list of log quantities this compute calculates
		virtual std::vector< std::string > getProvidedLogQuantities(); 
		
	protected:
		//! Computes forces
		virtual void computeForces(unsigned int timestep);
		
		boost::shared_ptr<Variant> m_T;			//!< Temperature of the bath
		Scalar m_dt;							//!< friction coefficient of the bath
		unsigned int m_seed; 					//!< initializing seed for RNG
		unsigned int m_ntypes; 					//!< Store the number of particle types
		Scalar * __restrict__ m_gamma;			//!< Parameter for computing forces (m_ntypes by m_ntypes array)
		boost::shared_ptr<Saru> m_saru; 		//!< Store the instantiation of the Saru Random Number Geneator Class
	};


//! Exports the StochasticForceCompute class to python
void export_StochasticForceCompute();


#endif
