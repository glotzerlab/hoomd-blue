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

#include "ForceCompute.h"

#include <vector>

/*! \file BondForceCompute.h
	\brief Declares a class for computing harmonic bonds
*/

#ifndef __BONDFORCECOMPUTE_H__
#define __BONDFORCECOMPUTE_H__

//! Computes bond forces on each particle
/*! This is just a simple harmonic bond force: V = K (|vec{Delta r}| - r_0|) ^ 2 . 
	
	After construction, this class will not compute this potential for any particles by default.
	Bonds where this potential is computed must be added by the addBond() method. Bonded pairs are
	identified by TAG, not index so that it doesn't matter if the particles are resorted
	in memory or not. 
	\ingroup computes
*/
class BondForceCompute : public ForceCompute
	{
	public:
		//! Constructs the compute
		BondForceCompute(boost::shared_ptr<ParticleData> pdata, Scalar K, Scalar r_0);
		
		//! Set the parameters
		virtual void setParams(Scalar K, Scalar r_0);
		
		//! Add a bond
		virtual void addBond(unsigned int tag1, unsigned int tag2);
	
	protected:
		Scalar m_K;		//!< K parameter
		Scalar m_r_0;	//!< r_0 parameter
		
		//! A simple struct for storing the two particles of a bonded pair
		struct BondPair
			{
			//! Constructor
			/*! \param tag1 Tag of the first particle in the bond
				\param tag2 Tag of the second particle in the bond
			*/
			BondPair(unsigned int tag1, unsigned int tag2) : m_tag1(tag1), m_tag2(tag2) {}
			
			unsigned int m_tag1;	//!< Tag of the first particle in the bond
			unsigned int m_tag2;	//!< Tag of the second particle in the bond
			};
		
		std::vector< BondPair > m_bonds;	//!< A list of all the bonds to compute
		
		//! Actually compute the forces
		virtual void computeForces(unsigned int timestep);
	};
	
#ifdef USE_PYTHON
//! Exports the BondForceCompute class to python
void export_BondForceCompute();
#endif

#endif
