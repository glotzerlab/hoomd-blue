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

#include "gpu_pdata.h"
#include "gpu_updaters.h"
#include "gpu_integrator.h"
#include "gpu_settings.h"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

#include <stdio.h>

//! The texture for reading the pdata pos array
texture<float4, 1, cudaReadModeElementType> pdata_pos_tex;
texture<float4, 1, cudaReadModeElementType> pdata_vel_tex;
texture<float4, 1, cudaReadModeElementType> pdata_accel_tex;

extern __shared__ float nvt_sdata[];

/*! \file gpu_nvt_kernel.cu
	\brief Contains code for the NVT kernel on the GPU
*/

extern "C" __global__ void nvt_pre_step_kernel(gpu_pdata_arrays pdata, gpu_nvt_data d_nvt_data, float denominv, float deltaT, gpu_boxsize box)
	{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int pidx = idx + pdata.local_beg;
	// do Nose-Hoover integrate
	
	float vsq;	
	if (idx < pdata.local_num)
		{
		// update positions to the next timestep and update velocities to the next half step
		float4 pos = tex1Dfetch(pdata_pos_tex, pidx);
		
		float px = pos.x;
		float py = pos.y;
		float pz = pos.z;
		float pw = pos.w;
		
		float4 vel = tex1Dfetch(pdata_vel_tex, pidx);
		float4 accel = tex1Dfetch(pdata_accel_tex, pidx);
		
		vel.x = (vel.x + (1.0f/2.0f) * accel.x * deltaT) * denominv;
		px += vel.x * deltaT;
		
		vel.y = (vel.y + (1.0f/2.0f) * accel.y * deltaT) * denominv;
		py += vel.y * deltaT;
		
		vel.z = (vel.z + (1.0f/2.0f) * accel.z * deltaT) * denominv;
		pz += vel.z * deltaT;
		
		// time to fix the periodic boundary conditions	
		px -= box.Lx * rintf(px * box.Lxinv);
		py -= box.Ly * rintf(py * box.Lyinv);
		pz -= box.Lz * rintf(pz * box.Lzinv);
	
		float4 pos2;
		pos2.x = px;
		pos2.y = py;
		pos2.z = pz;
		pos2.w = pw;
						
		// write out the results
		pdata.pos[pidx] = pos2;
		pdata.vel[pidx] = vel;
	
		// now we need to do the partial K sums
	
		// compute our contribution to the sum
		// NOTE: mass = 1.0
		vsq = vel.x*vel.x + vel.y*vel.y + vel.z*vel.z;
		}
	else
		{
		vsq = 0.0f;
		}
		
	nvt_sdata[threadIdx.x] = vsq;
	__syncthreads();

	// reduce the sum in parallel
	int offs = blockDim.x >> 1;
	while (offs > 0)
		{
		if (threadIdx.x < offs)
			nvt_sdata[threadIdx.x] += nvt_sdata[threadIdx.x + offs];
		offs >>= 1;
		__syncthreads();
		}

	// write out our partial sum
	if (threadIdx.x == 0)
		{
		d_nvt_data.partial_Ksum[blockIdx.x] = nvt_sdata[0];
		}
	}

cudaError_t nvt_pre_step(gpu_pdata_arrays *pdata, gpu_boxsize *box, gpu_nvt_data *d_nvt_data, float Xi, float deltaT)
	{
    assert(pdata);
	assert(d_nvt_data);

    // setup the grid to run the kernel
    int M = d_nvt_data->block_size;
    dim3 grid( d_nvt_data->NBlocks, 1, 1);
    dim3 threads(M, 1, 1);

	// bind the textures
	cudaError_t error = cudaBindTexture(0, pdata_pos_tex, pdata->pos, sizeof(float4) * pdata->N);
	if (error != cudaSuccess)
		return error;

	error = cudaBindTexture(0, pdata_vel_tex, pdata->vel, sizeof(float4) * pdata->N);
	if (error != cudaSuccess)
		return error;

	error = cudaBindTexture(0, pdata_accel_tex, pdata->accel, sizeof(float4) * pdata->N);
	if (error != cudaSuccess)
		return error;
	
	// run the kernel
    nvt_pre_step_kernel<<< grid, threads, M * sizeof(float) >>>(*pdata, *d_nvt_data, 1.0f / (1.0f + deltaT/2.0f * Xi), deltaT, *box);
	if (!g_gpu_error_checking)
		{
		return cudaSuccess;
		}
	else
		{
		cudaThreadSynchronize();
		return cudaGetLastError();
		}
	}


extern "C" __global__ void nvt_step_kernel(gpu_pdata_arrays pdata, gpu_nvt_data d_nvt_data, float4 **force_data_ptrs, int num_forces, float Xi, float deltaT)
	{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int pidx = idx + pdata.local_beg;
	
	float4 accel = integrator_sum_forces_inline(idx, pidx, pdata.local_num, force_data_ptrs, num_forces);
	if (idx < pdata.local_num)
		{
		float4 vel = tex1Dfetch(pdata_vel_tex, pidx);
			
		vel.x += (1.0f/2.0f) * deltaT * (accel.x - Xi * vel.x);
		vel.y += (1.0f/2.0f) * deltaT * (accel.y - Xi * vel.y);
		vel.z += (1.0f/2.0f) * deltaT * (accel.z - Xi * vel.z);
		
		// write out data
		pdata.vel[pidx] = vel;
		// since we calculate the acceleration, we need to write it for the next step
		pdata.accel[pidx] = accel;
		}
	}


cudaError_t nvt_step(gpu_pdata_arrays *pdata, gpu_nvt_data *d_nvt_data, float4 **force_data_ptrs, int num_forces, float Xi, float deltaT)
	{
    assert(pdata);
	assert(d_nvt_data);

    // setup the grid to run the kernel
    int M = d_nvt_data->block_size;
    dim3 grid( d_nvt_data->NBlocks, 1, 1);
    dim3 threads(M, 1, 1);

	// bind the texture
	cudaError_t error = cudaBindTexture(0, pdata_vel_tex, pdata->vel, sizeof(float4) * pdata->N);
	if (error != cudaSuccess)
		return error;

    // run the kernel
    nvt_step_kernel<<< grid, threads, M*sizeof(float) >>>(*pdata, *d_nvt_data, force_data_ptrs, num_forces, Xi, deltaT);
	
	if (!g_gpu_error_checking)
		{
		return cudaSuccess;
		}
	else
		{
		cudaThreadSynchronize();
		return cudaGetLastError();
		}
	}
	

// This kernel is designed to be a 1-block kernel for summing the total Ksum
extern "C" __global__ void nvt_reduce_ksum_kernel(gpu_nvt_data d_nvt_data)
	{
	float Ksum = 0.0f;

	// sum up the values in the partial sum via a sliding window
	for (int start = 0; start < d_nvt_data.NBlocks; start += blockDim.x)
		{
		__syncthreads();
		if (start + threadIdx.x < d_nvt_data.NBlocks)
			nvt_sdata[threadIdx.x] = d_nvt_data.partial_Ksum[start + threadIdx.x];
		else
			nvt_sdata[threadIdx.x] = 0.0f;
		__syncthreads();

		// reduce the sum in parallel
		int offs = blockDim.x >> 1;
		while (offs > 0)
			{
			if (threadIdx.x < offs)
				nvt_sdata[threadIdx.x] += nvt_sdata[threadIdx.x + offs];
			offs >>= 1;
			__syncthreads();
			}

		// everybody sums up Ksum
		Ksum += nvt_sdata[0];
		}
	
	if (threadIdx.x == 0)
		*d_nvt_data.Ksum = Ksum;
	}
	
cudaError_t nvt_reduce_ksum(gpu_nvt_data *d_nvt_data)
	{
	assert(d_nvt_data);

    // setup the grid to run the kernel
    int M = 128;
    dim3 grid( 1, 1, 1);
    dim3 threads(M, 1, 1);

    // run the kernel
    nvt_reduce_ksum_kernel<<< grid, threads, M*sizeof(float) >>>(*d_nvt_data);
	if (!g_gpu_error_checking)
		{
		return cudaSuccess;
		}
	else
		{
		cudaThreadSynchronize();
		return cudaGetLastError();
		}
	}	


// vim:syntax=cpp
