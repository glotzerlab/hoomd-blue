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
#include "gpu_utils.h"
#include "gpu_updaters.h"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

#include <stdio.h>

extern __shared__ float nvt_sdata[];

/*! \file gpu_nvt_kernel.cu
	\brief Contains code for the NVT kernel on the GPU
*/

extern "C" __global__ void nvt_pre_step_kernel(gpu_pdata_arrays pdata, gpu_nvt_data d_nvt_data, float deltaT, gpu_boxsize box)
	{
	int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	// do Nose-Hoover integrate
	
	// start off by calculating the denominator that will multiply velocities
	__shared__ float Xi, denominv;
	
	if (threadIdx.x == 0)
		{	
		Xi = *d_nvt_data.Xi;
		denominv = 1.0f / (1.0f + deltaT/2.0f * Xi);
		}
	__syncthreads();
	
	float vsq;	
	if (pidx < pdata.N)
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

void nvt_pre_step(gpu_pdata_arrays *pdata, gpu_boxsize *box, gpu_nvt_data *d_nvt_data, float deltaT)
	{
    assert(pdata);
	assert(d_nvt_data);

    // setup the grid to run the kernel
    int M = d_nvt_data->block_size;
    dim3 grid( d_nvt_data->NBlocks, 1, 1);
    dim3 threads(M, 1, 1);

    // run the kernel
    nvt_pre_step_kernel<<< grid, threads, M * sizeof(float) >>>(*pdata, *d_nvt_data, deltaT, *box);
    CUT_CHECK_ERROR("Kernel execution failed");
	}


extern "C" __global__ void nvt_step_kernel(gpu_pdata_arrays pdata, gpu_nvt_data d_nvt_data, float4 **force_data_ptrs, int num_forces, float deltaT, float Q, float T, float g)
	{
	int pidx = blockIdx.x * blockDim.x + threadIdx.x;

	// every block updates a local copy of Xi
	__shared__ float Xi;
	if (threadIdx.x == 0)
		{
		Xi = *d_nvt_data.Xi;
		float Ksum = *d_nvt_data.Ksum;
		Xi += deltaT / Q * (Ksum - g * T);
		}
	__syncthreads();

	// only one writes out the new Xi
	if (pidx == 0)
		*d_nvt_data.Xi_dbl = Xi;

	float4 accel = integrator_sum_forces_inline(pidx, pdata.N, force_data_ptrs, num_forces);
	if (pidx < pdata.N)
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


void nvt_step(gpu_pdata_arrays *pdata, gpu_nvt_data *d_nvt_data, float4 **force_data_ptrs, int num_forces, float deltaT, float Q, float T)
	{
    assert(pdata);
	assert(d_nvt_data);

    // setup the grid to run the kernel
    int M = d_nvt_data->block_size;
    dim3 grid( d_nvt_data->NBlocks, 1, 1);
    dim3 threads(M, 1, 1);

    // run the kernel
    nvt_step_kernel<<< grid, threads, M*sizeof(float) >>>(*pdata, *d_nvt_data, force_data_ptrs, num_forces, deltaT, Q, T, 3.0f * pdata->N);
    CUT_CHECK_ERROR("Kernel execution failed");
	
	// swap the double buffered Xi
	float *tmp = d_nvt_data->Xi;
	d_nvt_data->Xi = d_nvt_data->Xi_dbl;
	d_nvt_data->Xi_dbl = tmp;
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
	
void nvt_reduce_ksum(gpu_nvt_data *d_nvt_data)
	{
	assert(d_nvt_data);

    // setup the grid to run the kernel
    int M = 128;
    dim3 grid( 1, 1, 1);
    dim3 threads(M, 1, 1);

    // run the kernel
    nvt_reduce_ksum_kernel<<< grid, threads, M*sizeof(float) >>>(*d_nvt_data);
    CUT_CHECK_ERROR("Kernel execution failed");
	}	


/*! \todo check that block_size is a power of 2
	\todo allow initialization of Xi
*/
void nvt_alloc_data(gpu_nvt_data *d_nvt_data, int N, int block_size)
	{
	assert(d_nvt_data);
	d_nvt_data->block_size = block_size;
	d_nvt_data->NBlocks = N / block_size + 1;

	CUDA_SAFE_CALL( cudaMalloc((void**) &d_nvt_data->partial_Ksum, d_nvt_data->NBlocks * sizeof(float)) );
	CUDA_SAFE_CALL( cudaMalloc((void**) &d_nvt_data->Ksum, sizeof(float)) );
	CUDA_SAFE_CALL( cudaMalloc((void**) &d_nvt_data->Xi, sizeof(float)) );
	CUDA_SAFE_CALL( cudaMalloc((void**) &d_nvt_data->Xi_dbl, sizeof(float)) );

	// initialize Xi to 1.0
	float Xi = 1.0;
	CUDA_SAFE_CALL( cudaMemcpy(d_nvt_data->Xi, &Xi, sizeof(float), cudaMemcpyHostToDevice) );
	}

void nvt_free_data(gpu_nvt_data *d_nvt_data)
	{
	assert(d_nvt_data);

	CUDA_SAFE_CALL( cudaFree(d_nvt_data->partial_Ksum) );
	d_nvt_data->partial_Ksum = NULL;
	CUDA_SAFE_CALL( cudaFree(d_nvt_data->Ksum) );
	d_nvt_data->Ksum = NULL;
	CUDA_SAFE_CALL( cudaFree(d_nvt_data->Xi) );
	d_nvt_data->Xi = NULL;
	CUDA_SAFE_CALL( cudaFree(d_nvt_data->Xi_dbl) );
	d_nvt_data->Xi_dbl = NULL;
	}

// vim:syntax=cpp
