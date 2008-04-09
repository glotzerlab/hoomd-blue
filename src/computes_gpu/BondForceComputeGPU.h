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

#include "BondForceCompute.h"
#include "gpu_forces.h"

#include <boost/shared_ptr.hpp>
#include <boost/signals.hpp>

/*! \file BondForceComputeGPU.h
	\brief Declares a class for computing harmonic bonds on the GPU
*/

#ifndef __BONDFORCECOMPUTEGPU_H__
#define __BONDFORCECOMPUTEGPU_H__

//! Implements the harmonic bond force calculation on the GPU
/*! Bond forces are calucated much faster on the GPU. This class has the same public
	interface as BondForceCompute so that they can be used interchangably. 
		
	\b Developer information: <br>
	This class operates as a wrapper around CUDA code written in C and compiled by 
	nvcc. See bondforce_kernel.cu for detailed internal documentation.
	\sa BondForceCompute
	\ingroup computes
*/
class BondForceComputeGPU : public BondForceCompute
	{
	public:
		//! Constructs the compute
		BondForceComputeGPU(boost::shared_ptr<ParticleData> pdata, Scalar K, Scalar r_0);
		//! Destructor
		~BondForceComputeGPU();
		
		//! Add a bond
		virtual void addBond(unsigned int tag1, unsigned int tag2);
		
		//! Sets the block size to run on the device
		/*! \param block_size Block size to set
		*/
		void setBlockSize(int block_size) { m_block_size = block_size; }
		
	protected:
		bool m_dirty;	//!< Dirty flag to track if the bond table has changed
		int m_block_size;				//!< Block size to run calculation on
		gpu_bondtable_data m_gpu_bondtable;	//!< The actual bond table data
		boost::signals::connection m_sort_connection;	
		
		//! Helper function to set the dirty flag when particles are resort
		void setDirty() { m_dirty = true; }

		//! Helper function to update the bond table on the device
		void updateBondTable();
		
		//! Helper function to reallocate the bond table on the device
		void reallocateBondTable(int height);
		
		//! Actually compute the forces
		virtual void computeForces(unsigned int timestep);
	};
	
#ifdef USE_PYTHON
//! Export the BondForceComputeGPU class to python
void export_BondForceComputeGPU();
#endif

#endif
