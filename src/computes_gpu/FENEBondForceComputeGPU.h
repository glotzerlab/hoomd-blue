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

// $Id: FENEBondForceComputeGPU.h 1127 2008-08-31 16:54:39Z phillicl $
// $URL: https://svn2.assembla.com/svn/hoomd/tags/hoomd-0.7.0/src/computes_gpu/FENEBondForceComputeGPU.h $

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 )
#endif

#include "FENEBondForceCompute.h"
#include "gpu_forces.h"

#include <boost/shared_ptr.hpp>
#include <boost/signals.hpp>

/*! \file FENEBondForceComputeGPU.h
	\brief Declares the FENEBondForceGPU class
*/

#ifndef __FENEBONDFORCECOMPUTEGPU_H__
#define __FENEBONDFORCECOMPUTEGPU_H__

//! Implements the fene bond force calculation on the GPU
/*! Bond forces are calucated much faster on the GPU. This class has the same public
	interface as BondForceCompute so that they can be used interchangably. 
	
	Per-type parameters are stored in a simple global memory area pointed to by
	\a m_gpu_params. They are stored as float4's with the \a x component being K and the
	\a y component being r_0.
	
	\b Developer information: <br>
	This class operates as a wrapper around CUDA code written in C and compiled by 
	nvcc. See bondforce_kernel.cu for detailed internal documentation.
	\sa BondForceCompute
	\ingroup computes
*/
class FENEBondForceComputeGPU : public FENEBondForceCompute
	{
	public:
		//! Constructs the compute
		FENEBondForceComputeGPU(boost::shared_ptr<ParticleData> pdata);
		//! Destructor
		~FENEBondForceComputeGPU();
		
		//! Sets the block size to run on the device
		/*! \param block_size Block size to set
		*/
		void setBlockSize(int block_size) { m_block_size = block_size; }
		
		//! Set the parameters
		virtual void setParams(unsigned int type, Scalar K, Scalar r_0, Scalar sigma, Scalar epsilon);
				
	protected:
		int m_block_size;		//!< Block size to run calculation on
		float4 *m_gpu_params;	//! Parameters stored on the GPU
		float4 *m_host_params;	//! Host parameters
				
		//! Actually compute the forces
		virtual void computeForces(unsigned int timestep);
	};
	
//! Export the BondForceComputeGPU class to python
void export_FENEBondForceComputeGPU();

#endif

#ifdef WIN32
#pragma warning( pop )
#endif

