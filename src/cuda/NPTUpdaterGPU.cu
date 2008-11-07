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


#include "NPTUpdaterGPU.cuh"
#include "Integrator.cuh"
#include "gpu_settings.h"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

#include <stdio.h>

/*! \file NPTUpdaterGPU.cu
	\brief Defines GPU kernel code for NPT integration on the GPU. Used by NPTUpdaterGPU.
*/

//! Texture for reading the pdata pos array
texture<float4, 1, cudaReadModeElementType> pdata_pos_tex;
//! Texture for reading the pdata vel array
texture<float4, 1, cudaReadModeElementType> pdata_vel_tex;
//! Texture for reading the pdata accel array
texture<float4, 1, cudaReadModeElementType> pdata_accel_tex;

//! Shared data used by NPT kernels for sum reductions
extern __shared__ float npt_sdata[];

//! Sums virials from many different ForceComputes all in an inline function that can be included in any kernel.
/*! \param idx_local Local index of the running thread
	\param local_num Number of particles local to this GPU
	\param virial_data_ptrs Pointer to a list of pointers which are the arrays of virial data from the various ForceComputes
	\param num_virials Number of virials listed in \a virial_data_ptrs
	
	\note Every thread in the grid must call this function: it needs to __syncthreads()
	\note A maximum of 32 virials can be given in virial_data_ptrs
	
	gpu_integrator_sum_virials_inline() is designed to be run on one thread per particle with the
	normal thread breakdown of idx_local = threadIdx.x + blockDim.x * blockIdx.x. Full memory coalescing
	is achieved when this is the case. Each thread loops through the data pointers (which are cached
	in shared memory) and sums up the virial for particle idx_local.
	
	This inlined call is designed to be used from within other kernels.
*/
__device__ float gpu_integrator_sum_virials_inline(unsigned int idx_local, unsigned int local_num, float **virial_data_ptrs, int num_virials)
	{
	// each block loads in the pointers
	__shared__ float *virial_ptrs[32];
	if (threadIdx.x < 32)
		virial_ptrs[threadIdx.x] = virial_data_ptrs[threadIdx.x];
	__syncthreads();

	float virial = 0.0f;
	if (idx_local < local_num)
		{
		// sum the virials
		for (int i = 0; i < num_virials; i++)
			{
			float *d_virial = virial_ptrs[i];
			float v = d_virial[idx_local];
			virial += v;
			}
		}
	// return the result
	return virial;
	}

//! Sums the varoius virials on the GPU
/*! \param nptdata NPT data storage structure
	\param pdata Particle data arrays
	\param virial_data_ptrs list of virial data pointers
	\param num_virials number of virial points in the list

	\a virial_data_ptrs contains up to 32 pointers. Each points to pdata.local_num float's in memory
	All virials are summed into nptdata.virial
	
	gpu_integrator_sum_virials_kernel() is a simple driver to that uses gpu_integrator_sum_virials_inline()
	to compute the per-particle virial sums into the memory provided in nptdata.virial. One thread
	per particle is run with an arbitrary block size (a multiple of the warp size for coalescing).
*/
extern "C" __global__ void gpu_integrator_sum_virials_kernel(gpu_npt_data nptdata, gpu_pdata_arrays pdata, float **virial_data_ptrs, int num_virials)
	{
	// calculate the index we will be handling
	int idx_local = blockDim.x * blockIdx.x + threadIdx.x;

	float virial = gpu_integrator_sum_virials_inline(idx_local, pdata.local_num, virial_data_ptrs, num_virials);

	if (idx_local < pdata.local_num)
		{
		// write out the result
		nptdata.virial[idx_local] = virial;
		}
	}

/*! Every virial on every particle is summed up into \a nptpdata.virial

	\param nptdata NPT data storage structure
	\param pdata Particle data to write virial sum to
	\param virial_list List of pointers to virial data to sum
	\param num_virials Number of forces in \a virial_list

	\returns Any error code from the kernel call retrieved via cudaGetLastError()
	
	This is just a kernel driver for gpu_integrator_sum_virials_kernel(). See it for more details.
*/
cudaError_t gpu_integrator_sum_virials(const gpu_npt_data &nptdata, const gpu_pdata_arrays &pdata, float** virial_list, int num_virials)
	{
	// sanity check
	assert(num_virials < 32);

	const int block_size = 192;

	gpu_integrator_sum_virials_kernel<<< pdata.local_num/block_size+1, block_size >>>(nptdata, pdata, virial_list, num_virials);

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


//! For inexplicable reasons, the author has decided that is is best not to document his code
/*! \param pdata Particle data arrays to integrate forward 1/2 step
	\param box Box dimensions that the particles are in
	\param d_npt_data NPT data structure for storing data specific to NPT integration
	\param exp_v_fac For inexplicable reasons, the author has decided that is is best not to document his code
	\param exp_r_fac For inexplicable reasons, the author has decided that is is best not to document his code
	\param deltaT Time to advance (for one full step)
	\param box_len_scale For inexplicable reasons, the author has decided that is is best not to document his code
	
	\todo document me
*/
extern "C" __global__ void gpu_npt_pre_step_kernel(gpu_pdata_arrays pdata, gpu_boxsize box, gpu_npt_data d_npt_data, float exp_v_fac, float exp_r_fac, float deltaT, float box_len_scale)
	{
	int idx_local = blockIdx.x * blockDim.x + threadIdx.x;
	int idx_global = idx_local + pdata.local_beg;
	// do Nose-Hoover integrate ??? Copied and pasted comment doesn't apply
	
	if (idx_local < pdata.local_num)
		{
		// update positions to the next timestep and update velocities to the next half step
		float4 pos = tex1Dfetch(pdata_pos_tex, idx_global);
		
		float px = pos.x;
		float py = pos.y;
		float pz = pos.z;
		float pw = pos.w;
		
		float4 vel = tex1Dfetch(pdata_vel_tex, idx_global);
		float4 accel = tex1Dfetch(pdata_accel_tex, idx_global);
		
		vel.x = vel.x*exp_v_fac*exp_v_fac + (1.0f/2.0f) * deltaT*exp_v_fac*accel.x;
		px = px*exp_r_fac*exp_r_fac + vel.x*exp_r_fac*deltaT;

		vel.y = vel.y*exp_v_fac*exp_v_fac + (1.0f/2.0f) * deltaT*exp_v_fac*accel.y;
		py = py*exp_r_fac*exp_r_fac + vel.y*exp_r_fac*deltaT;

		vel.z = vel.z*exp_v_fac*exp_v_fac + (1.0f/2.0f) * deltaT*exp_v_fac*accel.z;
		pz = pz*exp_r_fac*exp_r_fac + vel.z*exp_r_fac*deltaT;

	
		// time to fix the periodic boundary conditions	
		px -= box_len_scale*box.Lx * rintf(px * box.Lxinv/box_len_scale);
		py -= box_len_scale*box.Ly * rintf(py * box.Lyinv/box_len_scale);
		pz -= box_len_scale*box.Lz * rintf(pz * box.Lzinv/box_len_scale);
	
		float4 pos2;
		pos2.x = px;
		pos2.y = py;
		pos2.z = pz;
		pos2.w = pw;
						
		// write out the results
		pdata.pos[idx_global] = pos2;
		pdata.vel[idx_global] = vel;
		}
	
	}

/*! \param pdata Particle Data to operate on
	\param box Current box dimensions the particles are in
	\param d_npt_data NPT specific data structures
	\param Xi For inexplicable reasons, the author has decided that is is best not to document his code
	\param Eta For inexplicable reasons, the author has decided that is is best not to document his code
	\param deltaT Time to move forward in one whole step

	This is just a kernel driver for gpu_integrator_pre_step_kernel(). See it for more details.
*/
cudaError_t gpu_npt_pre_step(const gpu_pdata_arrays &pdata, const gpu_boxsize &box, const gpu_npt_data &d_npt_data, float Xi, float Eta, float deltaT)
	{
	// setup the grid to run the kernel
	int block_size = d_npt_data.block_size;
	dim3 grid( d_npt_data.NBlocks, 1, 1);
	dim3 threads(block_size, 1, 1);

	// bind the textures
	cudaError_t error = cudaBindTexture(0, pdata_pos_tex, pdata.pos, sizeof(float4) * pdata.N);
	if (error != cudaSuccess)
		return error;

	error = cudaBindTexture(0, pdata_vel_tex, pdata.vel, sizeof(float4) * pdata.N);
	if (error != cudaSuccess)
		return error;

	error = cudaBindTexture(0, pdata_accel_tex, pdata.accel, sizeof(float4) * pdata.N);
	if (error != cudaSuccess)
		return error;
	
	// run the kernel
	float exp_v_fac = exp(-1.0f/4.0f*(Eta+Xi)*deltaT);
	float exp_r_fac = exp(1.0f/2.0f*Eta*deltaT);
	float box_len_scale = exp(Eta*deltaT);
	
	gpu_npt_pre_step_kernel<<< grid, threads >>>(pdata, box, d_npt_data, exp_v_fac, exp_r_fac, deltaT, box_len_scale);

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

//! For inexplicable reasons, the author has decided that is is best not to document his code
/*! \param pdata Particle data arrays to integrate forward 1/2 step
	\param d_npt_data NPT data structure for storing data specific to NPT integration
	\param force_data_ptrs Pointers to the forces in device memory
	\param num_forces Number of forces in \a force_data_ptrs
	\param exp_v_fac For inexplicable reasons, the author has decided that is is best not to document his code
	\param deltaT Time to advance (for one full step)
	
	\todo document me
*/
extern "C" __global__ void gpu_npt_step_kernel(gpu_pdata_arrays pdata, gpu_npt_data d_npt_data, float4 **force_data_ptrs, int num_forces, float exp_v_fac, float deltaT)
	{
	int idx_local = blockIdx.x * blockDim.x + threadIdx.x;
	int idx_global = idx_local + pdata.local_beg;
	
	// note assumes mac is 1.0
	float4 accel = gpu_integrator_sum_forces_inline(idx_local, pdata.local_num, force_data_ptrs, num_forces);
	if (idx_local < pdata.local_num)
		{
		float4 vel = tex1Dfetch(pdata_vel_tex, idx_global);
			
		vel.x = vel.x*exp_v_fac*exp_v_fac + (1.0f/2.0f)*deltaT*exp_v_fac*accel.x;
		vel.y = vel.y*exp_v_fac*exp_v_fac + (1.0f/2.0f)*deltaT*exp_v_fac*accel.y;
		vel.z = vel.z*exp_v_fac*exp_v_fac + (1.0f/2.0f)*deltaT*exp_v_fac*accel.z;
		
		// write out data
		pdata.vel[idx_global] = vel;
		// since we calculate the acceleration, we need to write it for the next step
		pdata.accel[idx_global] = accel;
		}
	}

/*! \param pdata Particle Data to operate on
	\param d_npt_data NPT specific data structures
	\param force_data_ptrs Pointers to the forces in device memory
	\param num_forces Number of forces in \a force_data_ptrs
	\param Xi For inexplicable reasons, the author has decided that is is best not to document his code
	\param Eta For inexplicable reasons, the author has decided that is is best not to document his code
	\param deltaT Time to move forward in one whole step

	This is just a kernel driver for gpu_npt_step_kernel(). See it for more details.
*/
cudaError_t gpu_npt_step(const gpu_pdata_arrays &pdata, const gpu_npt_data &d_npt_data, float4 **force_data_ptrs, int num_forces, float Xi, float Eta, float deltaT)
	{
	  // setup the grid to run the kernel
	  int block_size = d_npt_data.block_size;
	  dim3 grid( d_npt_data.NBlocks, 1, 1);
	  dim3 threads(block_size, 1, 1);
	  float exp_v_fac = exp(-1.0f/4.0f*(Eta+Xi)*deltaT);

	  // bind the texture
	  cudaError_t error = cudaBindTexture(0, pdata_vel_tex, pdata.vel, sizeof(float4) * pdata.N);
	  if (error != cudaSuccess)
		return error;

	  // run the kernel
	  gpu_npt_step_kernel<<< grid, threads >>>(pdata, d_npt_data, force_data_ptrs, num_forces, exp_v_fac, deltaT);
	  
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
	
//! Completes the sums of m*v^2 over every particle in the simulation
/*! \param d_npt_data NPT specific data structures
	
	\pre gpu_npt_temperature_kernel() must be called first to fill out the partial sums in \a d_npt_data.
	\a d_npt_data.NBlocks partial sums are written there to be added up here.
	
	gpu_npt_reduce_ksum_kernel() is a very simple 1-block kernel run that completes the partial sums
	and writes the final m*v^2 sum for this GPU out to *d_npt_data.Ksum. It must be run with one
	block and a power of 2 for a block size with blockDim.x*sizeof(float) bytes of dynamic shared 
	memory allocated.
	
	The kernel works by going over the list of partial sums with a sliding window blockDim.x threads
	wide. Each thread participates in a fully coalesced load of the partial sums and then a parallel 
	reduction is employed to complete the sum.
*/
extern "C" __global__ void gpu_npt_reduce_ksum_kernel(gpu_npt_data d_npt_data)
	{
	float Ksum = 0.0f;

	// sum up the values in the partial sum via a sliding window
	for (int start = 0; start < d_npt_data.NBlocks; start += blockDim.x)
		{
		__syncthreads();
		if (start + threadIdx.x < d_npt_data.NBlocks)
			npt_sdata[threadIdx.x] = d_npt_data.partial_Ksum[start + threadIdx.x];
		else
			npt_sdata[threadIdx.x] = 0.0f;
		__syncthreads();

		// reduce the sum in parallel
		int offs = blockDim.x >> 1;
		while (offs > 0)
			{
			if (threadIdx.x < offs)
				npt_sdata[threadIdx.x] += npt_sdata[threadIdx.x + offs];
			offs >>= 1;
			__syncthreads();
			}

		// everybody sums up Ksum
		Ksum += npt_sdata[0];
		}
	
	if (threadIdx.x == 0)
		{
		*d_npt_data.Ksum = Ksum;
		}
	}
	
/*! \param d_npt_data NPT specific data structures
	
	This is just a driver for gpu_npt_reduce_ksum_kernel(). See it for more details.
*/
cudaError_t gpu_npt_reduce_ksum(const gpu_npt_data &d_npt_data)
	{
	// setup the grid to run the kernel
	int block_size = 128;
	dim3 grid( 1, 1, 1);
	dim3 threads(block_size, 1, 1);
	
	// run the kernel
	gpu_npt_reduce_ksum_kernel<<< grid, threads, block_size*sizeof(float) >>>(d_npt_data);
	
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

//! Computes the first-pass m*v^2 sum
/*! \param d_npt_data NPT specific data structures
	\param pdata Particle data to compute temperature of
	
	\a d_npt_data.NBlocks blocks are to be run with \a d_npt_data.block_size width. Each thread
	reads in the velocity of a single particle, calculates m*v^2 and then each block makes a 
	parallel reduction pass to compute the partial m*v^2 sums. \a d_npt_data.NBlocks partial sums
	are written out to \a d_npt_data.partial_Ksum which will be later summed in gpu_npt_reduce_ksum_kernel().
*/
extern "C" __global__ void gpu_npt_temperature_kernel(gpu_npt_data d_npt_data, gpu_pdata_arrays pdata)
	{
	int idx_local = blockIdx.x * blockDim.x + threadIdx.x;
	int idx_global = idx_local + pdata.local_beg;
	
	float vsq;
	if (idx_local < pdata.local_num)
		{
		float4 vel = tex1Dfetch(pdata_vel_tex, idx_global);
		vsq = vel.x*vel.x + vel.y*vel.y + vel.z*vel.z;
		}
	else
		{
		vsq = 0.0f;
		}
		
	npt_sdata[threadIdx.x] = vsq;
	__syncthreads();

	// reduce the sum in parallel
	int offs = blockDim.x >> 1;
	while (offs > 0)
		{
		if (threadIdx.x < offs)
			npt_sdata[threadIdx.x] += npt_sdata[threadIdx.x + offs];
		offs >>= 1;
		__syncthreads();
		}

	// write out our partial sum
	if (threadIdx.x == 0)
		{
		d_npt_data.partial_Ksum[blockIdx.x] = npt_sdata[0];
		}
	}

/*! \param d_npt_data NPT specific data structures
	\param pdata Particle data to compute temperature of
	
	This is just a driver for gpu_npt_temperature_kernel(). See it for more details.
*/
cudaError_t gpu_npt_temperature(const gpu_npt_data &d_npt_data, const gpu_pdata_arrays &pdata)
	{
	// setup the grid to run the kernel
	int block_size = d_npt_data.block_size;
	dim3 grid(d_npt_data.NBlocks, 1, 1);
	dim3 threads(block_size, 1, 1);

	// bind velocity to the texture
	cudaError_t error = cudaBindTexture(0, pdata_vel_tex, pdata.vel, sizeof(float4) * pdata.N);
	if (error != cudaSuccess)
		return error;

	// run the kernel
	gpu_npt_temperature_kernel<<< grid, threads, block_size*sizeof(float) >>>(d_npt_data, pdata);
	
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


//! Completes the virial sum over every particle in the simulation
/*! \param d_npt_data NPT specific data structures
	
	\pre gpu_npt_pressure_kernel() must be called first to fill out the partial sums in \a d_npt_data.
	\a d_npt_data.NBlocks partial sums are written there to be added up here.
	
	gpu_npt_reduce_psum_kernel() is a very simple 1-block kernel run that completes the partial sums
	and writes the final virial sum for this GPU out to *d_npt_data.Psum. It must be run with one
	block and a power of 2 for a block size with blockDim.x*sizeof(float) bytes of dynamic shared 
	memory allocated.
	
	The kernel works by going over the list of partial sums with a sliding window blockDim.x threads
	wide. Each thread participates in a fully coalesced load of the partial sums and then a parallel 
	reduction is employed to complete the sum.
*/
extern "C" __global__ void gpu_npt_reduce_psum_kernel(gpu_npt_data d_npt_data)
	{
	float Psum = 0.0f;

	// sum up the values in the partial sum via a sliding window
	for (int start = 0; start < d_npt_data.NBlocks; start += blockDim.x)
		{
		__syncthreads();
		if (start + threadIdx.x < d_npt_data.NBlocks)
			npt_sdata[threadIdx.x] = d_npt_data.partial_Psum[start + threadIdx.x];
		else
			npt_sdata[threadIdx.x] = 0.0f;
		__syncthreads();

		// reduce the sum in parallel
		int offs = blockDim.x >> 1;
		while (offs > 0)
			{
			if (threadIdx.x < offs)
				npt_sdata[threadIdx.x] += npt_sdata[threadIdx.x + offs];
			offs >>= 1;
			__syncthreads();
			}

		// everybody sums up Psum
		Psum += npt_sdata[0];
		}
	
	if (threadIdx.x == 0)
		{
		*d_npt_data.Psum = Psum;
	  	}
	}

/*! \param d_npt_data NPT specific data structures
		
	This is just a driver for gpu_npt_reduce_psum_kernel(). See it for more details.
*/
cudaError_t gpu_npt_reduce_psum(const gpu_npt_data &d_npt_data)
	{
	// setup the grid to run the kernel
	int block_size = 128;
	dim3 grid( 1, 1, 1);
	dim3 threads(block_size, 1, 1);
	
	// run the kernel
	gpu_npt_reduce_psum_kernel<<< grid, threads, block_size*sizeof(float) >>>(d_npt_data);
	//printf("d_npt_data.Psum = %f\n", (*d_npt_data).Psum);
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

//! Computes the first-pass virial sum
/*! \param d_npt_data NPT specific data structures
	\param pdata Particle data to compute temperature of
	
	\a d_npt_data.NBlocks blocks are to be run with \a d_npt_data.block_size width. Each thread
	reads in the total virial on a single particle and then each block makes a 
	parallel reduction pass to compute the partial virial sums. \a d_npt_data.NBlocks partial sums
	are written out to \a d_npt_data.partial_Psum which will be later summed in gpu_npt_reduce_psum_kernel().
*/
extern "C" __global__ void gpu_npt_pressure_kernel(gpu_npt_data d_npt_data, gpu_pdata_arrays pdata)
	{
	int idx_local = blockIdx.x * blockDim.x + threadIdx.x;
	// do Nose-Hoover integrate ??? copied and pasted comment doesn't apply
	
	//printf("pdata.local_num = %d\n",  pdata.local_num);
	
	float virial = 0.0f;
	if (idx_local < pdata.local_num)
		{
		virial = d_npt_data.virial[idx_local];
		}

	npt_sdata[threadIdx.x] = virial;
	__syncthreads();

	// reduce the sum in parallel
	int offs = blockDim.x >> 1;
	while (offs > 0)
		{
		if (threadIdx.x < offs)
			npt_sdata[threadIdx.x] += npt_sdata[threadIdx.x + offs];
		offs >>= 1;
		__syncthreads();
		}

	// write out our partial sum
	if (threadIdx.x == 0)
		{
		d_npt_data.partial_Psum[blockIdx.x] = npt_sdata[0];
		}
	}

/*! \param d_npt_data NPT specific data structures
	\param pdata Particle data to compute temperature of
	
	This is just a driver function for gpu_npt_pressure_kernel(). See it for more details.
*/
cudaError_t gpu_npt_pressure(const gpu_npt_data &d_npt_data, const gpu_pdata_arrays &pdata)
	{
	// setup the grid to run the kernel
	int block_size = d_npt_data.block_size;
	dim3 grid(d_npt_data.NBlocks, 1, 1);
	dim3 threads(block_size, 1, 1);

	// run the kernel
	gpu_npt_pressure_kernel<<< grid, threads, block_size*sizeof(float) >>>(d_npt_data, pdata);
	
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
