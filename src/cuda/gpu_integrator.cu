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

#include "gpu_integrator.h"

/*! \file gpu_integrator.cu
	\brief Contains kernel and driver code for summing forces on the GPU
*/

//! Kernel for summing forces on the GPU
/*! \param pdata Particle data arrays
	\param force_data_ptrs list of force data pointers
	\param num_forces number of force pointes in the list

	\a force_data_ptrs contains up to 32 pointers. Each points to N float4's in memory
	All forces are summed into pdata.accel. 

	\note mass is assumed to be 1.0 at this stage
*/
__global__ void integrator_sum_forces_kernel(gpu_pdata_arrays pdata, float4 **force_data_ptrs, int num_forces)
	{
	// each block loads in the pointers
	__shared__ float4 *force_ptrs[32];
	if (threadIdx.x < 32)
		force_ptrs[threadIdx.x] = force_data_ptrs[threadIdx.x];
	__syncthreads();

	// calculate the index we will be handling
	int pidx = blockDim.x * blockIdx.x + threadIdx.x;

	if (pidx < pdata.N)
		{
		// sum the acceleration
		float4 accel = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		for (int i = 0; i < num_forces; i++)
			{
			float4 *d_force = force_ptrs[i];
			float4 f = d_force[pidx];
	
			accel.x += f.x;
			accel.y += f.y;
			accel.z += f.z;
			}
	
		// write out the result
		pdata.accel[pidx] = accel;
		}
	}

/*! Every force on every particle is summed up into \a pdata.accel

    \param pdata Particle data to write force sum to
    \param force_list List of pointers to force data to sum
    \param num_forces Number of forces in \a force_list

    \returns Any error code from the kernel call retrieved via cudaGetLastError()
    \note Always returns cudaSuccess in release builds for performance reasons
*/
cudaError_t integrator_sum_forces(gpu_pdata_arrays *pdata, float4** force_list, int num_forces)
	{
	// sanity check
	assert(pdata);
	assert(force_list);
	assert(num_forces < 32);

	const int M = 256;

	integrator_sum_forces_kernel<<< pdata->N/M+1, M >>>(*pdata, force_list, num_forces);

	#ifdef NDEBUG
    return cudaSuccess;
    #else
    cudaThreadSynchronize();
    return cudaGetLastError();
    #endif
    }
