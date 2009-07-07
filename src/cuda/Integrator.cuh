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

/*! \file Integrator.cuh
	\brief Declares methods and data structures used by the Integrator class on the GPU
*/

#ifndef __INTEGRATOR_CUH__
#define __INTEGRATOR_CUH__

#include "ParticleData.cuh"
#include "RigidData.cuh"

//! Sums up the net acceleration on the GPU for Integrator
cudaError_t gpu_integrator_sum_accel(const gpu_pdata_arrays &pdata, float4** force_list, int num_forces);


// only define device function when compiled with NVCC
#ifdef NVCC

//! A more efficient, GPU force sum to be used in an existing kernel
/*! This __device__ method uses the same data structures as gpu_integrator_sum_accel, 
	but can be used in an existing kernel to avoid the overhead of additional kernel
	launches.
	
	\param idx_local Thread index
	\param local_num number of particles local to this GPU
	\param force_data_ptrs list of force data pointers
	\param num_forces number of force pointes in the list

	\a force_data_ptrs contains up to 32 pointers. Each points to N float4's in memory
	All forces are summed into pdata.accel. 

	\returns Net force on the particle \a idx_local

	\note Uses a small amount of shared memory. Every thread participating in the kernel must 
	call this function. Coalescing is achieved when idx_local = blockDim.x*blockIDx.x + threadIdx.x
*/
__device__ float4 gpu_integrator_sum_forces_inline(unsigned int idx_local, unsigned int local_num, float4 **force_data_ptrs, int num_forces)
	{
	// each block loads in the pointers
	__shared__ float4 *force_ptrs[32];
	if (threadIdx.x < 32)
		force_ptrs[threadIdx.x] = force_data_ptrs[threadIdx.x];
	__syncthreads();

	float4 net_force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	if (idx_local < local_num)
		{
		// sum the net force
		for (int i = 0; i < num_forces; i++)
			{
			float4 *d_force = force_ptrs[i];
			float4 f = d_force[idx_local];
		
			net_force.x += f.x;
			net_force.y += f.y;
			net_force.z += f.z;
			}
		}
	// return the result
	return net_force;
	}
#endif

#endif
