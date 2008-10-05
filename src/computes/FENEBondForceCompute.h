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

// $Id: FENEFENEBondForceCompute.h 1125 2008-08-31 15:20:43Z phillicl $
// $URL: https://svn2.assembla.com/svn/hoomd/tags/hoomd-0.7.0/src/computes/FENEBondForceCompute.h $

#include <boost/shared_ptr.hpp>

#include "ForceCompute.h"
#include "BondData.h"

#include <vector>

/*! \file FENEBondForceCompute.h
	\brief Declares a class for computing fene bonds
*/

#ifndef __FENEBONDFORCECOMPUTE_H__
#define __FENEBONDFORCECOMPUTE_H__

//! Computes bond forces on each particle
/*! This is just a simple FENE spring bond force: V = -1/2 K | r_0| ^ 2*ln[1-(|r|/|R_0|)^2]/ .  
	R_0 is the maximum length of the spring.  When r is greater than R_0, V = infinity.
	
	After construction, this class will not compute this potential for any particles by default.
	Bonds where this potential is computed must be defined in the BondData attached to the ParticleData.
	\ingroup computes
*/
class FENEBondForceCompute : public ForceCompute
	{
	public:
		//! Constructs the compute
		FENEBondForceCompute(boost::shared_ptr<ParticleData> pdata);
		
		//! Destructor
		~FENEBondForceCompute();
		
		//! Set the parameters
		virtual void setParams(unsigned int type, Scalar K, Scalar r_0, Scalar sigma, Scalar epsilon);
		
		//! Returns a list of log quantities this compute calculates
		virtual std::vector< std::string > getProvidedLogQuantities(); 
		
		//! Calculates the requested log value and returns it
		virtual Scalar getLogValue(const std::string& quantity);

	protected:
		Scalar *m_K;	//!< K parameter for multiple bond tyes
		Scalar *m_r_0;	//!< r_0 parameter for multiple bond types
		Scalar *m_lj1;	//!< lj1 for multiple bond types
		Scalar *m_lj2;	//!< lj2 for multiple bond types 
		Scalar *m_epsilon;
		
		boost::shared_ptr<BondData> m_bond_data;	//!< Bond data to use in computing bonds
		
		//! Actually compute the forces
		virtual void computeForces(unsigned int timestep);
	};
	
#ifdef USE_PYTHON
//! Exports the BondForceCompute class to python
void export_FENEBondForceCompute();
#endif

#endif
