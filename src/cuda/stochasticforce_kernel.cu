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

// $Id: stochasticforce_kernel.cu 1269 2008-09-22 22:37:32Z phillicl $
// $URL: http://svn2.assembla.com/svn/hoomd/trunk/src/cuda/stochasticforce_kernel.cu $

#include "gpu_forces.h"
#include "gpu_pdata.h"
#include "gpu_settings.h"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

#define RNG_MAX             ((unsigned int)4294967295)

/*! \file stochasticforce_kernel.cu
	\brief Contains code for the stochastic force sum kernel on the GPU
	\details Functions in this file are NOT to be called by anyone not knowing 
		exactly what they are doing. They are designed to be used solely by 
		StochasticForceComputeGPU.
*/


// Inlined device function that updates rng_state
// Random number generation based on this paper: http://www.jstatsoft.org/v08/i14/
// With 16 bytes of state, RNG period is 2^128
__device__ inline void xorshift_RNG(uint4 &rng_state) {
    unsigned int tmp;

    tmp = (rng_state.x ^ (rng_state.x << 11)); 
    rng_state.x = rng_state.y;
    rng_state.y = rng_state.z;
    rng_state.z = rng_state.w;
    rng_state.w = ((rng_state.w ^ (rng_state.w >> 19)) ^ (tmp ^ (tmp >> 8)));
}



///////////////////////////////////////// Stochastic params

//! Texture for reading particle velocities
texture<float4, 1, cudaReadModeElementType> pdata_vel_tex;

//! Texture for reading particle positions
texture<float, 1, cudaReadModeElementType> pdata_type_tex;

//! Kernel for calculating stochastic forces
/*! This kernel is called to apply stochastic heat bath forces to all N particles in conjunction with a Brownian Dynamics Simulations

	\param d_forces Device memory array to write calculated forces to
	\param pdata Particle data on the GPU to calculate forces on
	\param dt Timestep of the simulation
	\param T Temperature of the bath
	\param gammas Gamma coefficients that govern the coupling of the particle to the bath.
	\param gamma_length length of the gamma array (number of particle types)
	\param d_state The state vector for the RNG.
	
	\a gammas is a pointer to an array in memory. \c gamma[i].x is \a gamma for the particle type \a i.
	The values in d_gammas are read into shared memory, so \c gamma_length*sizeof(float) bytes of extern 
	shared memory must be allocated for the kernel call.
	
	Developer information:
	Each block will calculate the forces on a block of particles.
	Each thread will calculate the total force on one particle.
	The RNG state vectors should permit a coalesced read, but this fact should be checked.
	
*/
extern "C" __global__ void stochasticForces_kernel(gpu_force_data_arrays force_data, gpu_pdata_arrays pdata, float dt, float T, float *d_gammas, int gamma_length, uint4 * d_state)
	{
	
	// read in the gammas (1 dimensional array)
	extern __shared__ float s_gammas[];
	for (int cur_offset = 0; cur_offset < gamma_length; cur_offset += blockDim.x)
		{
		if (cur_offset + threadIdx.x < gamma_length)
			s_gammas[cur_offset + threadIdx.x] = d_gammas[cur_offset + threadIdx.x];
		}
	__syncthreads();
	
	// start by identifying which particle we are to handle
	int idx_local = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx_local >= pdata.local_num)
		return;
	
	int idx_global = idx_local + pdata.local_beg;

	// read in the velocity of our particle. Texture reads of float4's are faster than global reads on compute 1.0 hardware
	// (MEM TRANSFER: 16 bytes)
	float4 vel = tex1Dfetch(pdata_vel_tex, idx_global);

	// read in the position of our particle. Texture reads of float4's are faster than global reads on compute 1.0 hardware
	// (MEM TRANSFER: 16 bytes)
	float type_f = tex1Dfetch(pdata_type_tex, idx_global*4 + 3);
	int typ = __float_as_int(type_f);
	
	// Calculate Coefficient of Friction
	//type = 0;   //May use this for benchmarking the impact of doing a second texture read just for particle type
	float coeff_fric = sqrtf(6.0f * s_gammas[typ] * T/ dt);
	
	// initialize the force to 0
	float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    // Random Number Generator
    float INV_RNG_MAX = 1.0f / __uint2float_rz(RNG_MAX);
	
    // Each thread of each block has a unique state that is loaded once from global memory
	uint4 rng_state;
    rng_state = d_state[blockIdx.x * blockDim.x + threadIdx.x];

	// Generate random number and generate x, y, and z forces respectively
	xorshift_RNG(rng_state);
	force.x += (2.0f*(__uint2float_rz(rng_state.w) * INV_RNG_MAX)-1.0f) * coeff_fric - s_gammas[typ]*vel.x;
	xorshift_RNG(rng_state);
	force.y += (2.0f*(__uint2float_rz(rng_state.w) * INV_RNG_MAX)-1.0f) * coeff_fric - s_gammas[typ]*vel.y;
	xorshift_RNG(rng_state);
	force.z += (2.0f*(__uint2float_rz(rng_state.w) * INV_RNG_MAX)-1.0f) * coeff_fric - s_gammas[typ]*vel.z;

	// stochastic forces do not contribute to potential energy

	// now that the force calculation is complete, write out the result (MEM TRANSFER: 16 bytes)
	force_data.force[idx_global] = force;
//	force_data.force[idx_global] = make_float4(coeff_fric, dt, T, s_gammas[typ]);  //doing this just to verify it works.
//	force_data.force[idx_global] = make_float4(vel.x, vel.y, vel.z, 0.0f);  //doing this just to verify it works.
	
    // State is written back to global memory
    d_state[blockIdx.x * blockDim.x + threadIdx.x] = rng_state;
	}


/*! \param force_data Force data on GPU to write forces to
	\param pdata Particle data on the GPU to perform the calculation on
	\param dt Timestep
	\param T Temperature values
	\param d_gammas The coefficients of friction for each particle type
	\param gamma_length  The length of d_gamma array
	\param d_state The state vector for the RNG used in thread
	\param M Block size to execute
	
	\returns Any error code resulting from the kernel launch
	\note Always returns cudaSuccess in release builds to avoid the cudaThreadSynchronize()
*/
cudaError_t gpu_stochasticforce(const gpu_force_data_arrays& force_data, gpu_pdata_arrays *pdata, float dt, float T, float *d_gammas, uint4 *d_state, int gamma_length, int M)
	{
	assert(pdata);
	assert(d_gammas);
	assert(d_state);
	assert(gamma_length > 0);

	// setup the grid to run the kernel
	dim3 grid( (int)ceil((double)pdata->local_num/ (double)M), 1, 1);
	dim3 threads(M, 1, 1);

	// bind the velocity texture
	cudaError_t error = cudaBindTexture(0, pdata_vel_tex, pdata->vel, sizeof(float4) * pdata->N);
	if (error != cudaSuccess)
		return error;

	// bind the position texture  (this is done only to retrieve the particle type)
	error = cudaBindTexture(0, pdata_type_tex, pdata->pos, sizeof(float4) * pdata->N);
	if (error != cudaSuccess)
		return error;
		
    // run the kernel
    stochasticForces_kernel<<< grid, threads, sizeof(float)*gamma_length>>>(force_data, *pdata, dt, T, d_gammas, gamma_length, d_state);

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
