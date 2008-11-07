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

// $Id: YukawaForceComputeGPU.h 1325 2008-10-06 13:42:07Z joaander $
// $URL: http://svn2.assembla.com/svn/hoomd/trunk/src/computes_gpu/YukawaForceComputeGPU.h $

#include "YukawaForceCompute.h"
#include "NeighborList.h"
#include "gpu_nlist.h"
#include "YukawaForceGPU.cuh"

#include <boost/shared_ptr.hpp>

/*! \file YukawaForceComputeGPU.h
	\brief Declares a class for computing yukawa forces on the GPU
*/

#ifndef __YUKAWAFORCECOMPUTEGPU_H__
#define __YUKAWAFORCECOMPUTEGPU_H__

//! Computes Yukawa forces on each particle using the GPU
/*! Yukawa pair forces are calucated much faster on the GPU. This class 
	has the same public	interface as YukawaForceCompute so that they can be used interchangably. 
	
	\b Developer information: <br>
	This class operates as a wrapper around CUDA code written in C and compiled by 
	nvcc. See yukawaforcesum_kernel.cu for detailed internal documentation.
	\ingroup computes
*/
class YukawaForceComputeGPU : public YukawaForceCompute
	{
	public:
		//! Constructs the compute
		YukawaForceComputeGPU(boost::shared_ptr<ParticleData> pdata, boost::shared_ptr<NeighborList> nlist, Scalar r_cut, Scalar kappa);
		
		//! Destructor
		virtual ~YukawaForceComputeGPU();
		
		//! Set the parameters for a single type pair
		virtual void setParams(unsigned int typ1, unsigned int typ2, Scalar epsilon);
		
		//! Sets the block size to run at
		void setBlockSize(int block_size);

	protected:
		vector<float *> d_coeffs;		//!< Pointer to the coefficients on the GPU
		float * h_coeffs;				//!< Pointer to the coefficients on the host
		int m_block_size;				//!< The block size to run on the GPU

		//! Actually compute the forces
		virtual void computeForces(unsigned int timestep);
	};
	
//! Exports the YukawaForceComputeGPU class to python
void export_YukawaForceComputeGPU();

#endif
