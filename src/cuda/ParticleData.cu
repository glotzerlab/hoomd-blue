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

#include "ParticleData.cuh"
#include "gpu_settings.h"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file ParticleData.cu
 	\brief Contains GPU kernel code and data structure functions used by ParticleData
*/

//! Kernel for un-interleaving float4 input into float output
/*! \param d_out Device pointer to write un-interleaved output
	\param d_in Device pointer to read interleaved input
	\param N Number of elements in input
	\param pitch Spacing of arrays through the output

	\pre N/block_size + 1 blocks are run on the device
*/
extern "C" __global__ void uninterleave_float4_kernel(float *d_out, float4 *d_in, int N, int pitch)
    {
	int pidx  = blockDim.x * blockIdx.x + threadIdx.x;
	
	if (pidx < N)
		{
		float4 in = d_in[pidx];
		
		d_out[pidx] = in.x;
		d_out[pidx+pitch] = in.y;
		d_out[pidx+pitch+pitch] = in.z;
		d_out[pidx+pitch+pitch+pitch] = in.w;
		}
	}


/*! The most efficient data storage on the device is to put x,y,z,type into a float4
	data structure. The most efficient storage on the CPU is x,y,z,type each as 
	separate arrays. Translation between the two is best done on the device, and
	memory transfers done with one big cudaMemcpy. This function, and its sister
	gpu_interleave_float4() perform the transformation between a float* with x,y,z,type
	packed non-interleaved to a float4* storing the same values interleaved. 

	Performance is best when pitch is a multiple of 64.

	\param d_out Device pointer to write output to
	\param d_in Device pointer to read input from
	\param N Number of elements to interleave
	\param pitch Spacing between \c x[0] and \c y[0] in \a d_out

	\post A code snipped best describes what is done:
	\verbatim 
	d_out[i] = d_in[i].x
	d_out[i+pitch] = d_in[i].y
	d_out[i+pitch*2] = d_in[i].z
	d_out[i+pitch*3] = d_in[i].w
	\endverbatim

	\returns Any error code from the kernel call retrieved via cudaGetLastError()
	\note Always returns cudaSuccess in release builds for performance reasons
*/
cudaError_t gpu_uninterleave_float4(float *d_out, float4 *d_in, int N, int pitch)
	{
	assert(pitch >= N);
	assert(d_out);
	assert(d_in);
	assert(N > 0);

	const int M = 64;
	uninterleave_float4_kernel<<< N/M+1, M >>>(d_out, d_in, N, pitch);

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

//! Kernel for interleaving float input into float4 output
/*! \param d_out Device pointer to write interleaved output
	\param d_in Device pointer to read non-interleaved input
	\param N Number of elements in output
	\param pitch Spacing of arrays through the input

	\pre N/block_size + 1 blocks are run on the device
*/
extern "C" __global__ void interleave_float4_kernel(float4 *d_out, float *d_in, int N, int pitch)
    {
    int pidx  = blockDim.x * blockIdx.x + threadIdx.x;

    if (pidx < N)
        {
        float x = d_in[pidx];
        float y = d_in[pidx+pitch];
        float z = d_in[pidx+pitch+pitch];
        float w = d_in[pidx+pitch+pitch+pitch];

        float4 out;
        out.x = x;
        out.y = y;
        out.z = z;
        out.w = w;
        d_out[pidx] = out;
        }
    }

/*! See gpu_uninterleave_float4() for details.
	\param d_out Device pointer to write output to
	\param d_in Device pointer to read input from
	\param N Number of elements to interleave
	\param pitch Spacing between \c x[0] and \c y[0] in \a d_in
	
	\returns Any error code from the kernel call retrieved via cudaGetLastError()
	\note Always returns cudaSuccess in release builds for performance reasons
*/
cudaError_t gpu_interleave_float4(float4 *d_out, float *d_in, int N, int pitch)
	{
	assert(pitch >= N);
	assert(d_out);
	assert(d_in);
	assert(N > 0);

	const int M = 64;
	interleave_float4_kernel<<< N/M+1, M >>>(d_out, d_in, N, pitch);

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
 	
////////////////////////////////////////////////////////////////////
// Unit testing functions

//! Kernel for filling \a pdata with nonsense numbers for test purposes
/*! \param pdata Particle data to populate
*/
__global__ void pdata_test_fill(gpu_pdata_arrays pdata)
	{
	// start by identifying the particle index of this particle
	int pidx = blockIdx.x * blockDim.x + threadIdx.x;

	if (pidx < pdata.N)
		{
		// write out a nonsense test pattern
		float4 pos;
		pos.x = float(pidx);
		pos.y = float(pidx)*0.5f;
		pos.z = float(pidx)*0.4f;
		pos.w = float(pidx)*0.2f;
		pdata.pos[pidx] = pos;
		
		float4 vel;
		vel.x = float(pidx)*10.0f;
		vel.y = float(pidx)*5.0f;
		vel.z = float(pidx)*4.0f;
		pdata.vel[pidx] = vel;
		
		float4 accel;
		accel.x = float(pidx)*20.0f;
		accel.y = float(pidx)*15.0f;
		accel.z = float(pidx)*14.0f;
		pdata.accel[pidx] = accel;
		
		pdata.tag[pidx] = pidx*30;
		pdata.rtag[pidx] = pidx*40;
		}
	}

/*! \param pdata Particle data where the arrays will be populated with garbage
	\post Device memory is filled out with a nonsense test pattern
 	Read the pdata_test_fill() code to see what the pattern is

	\returns Error result from the kernel call
*/ 
cudaError_t gpu_generate_pdata_test(gpu_pdata_arrays *pdata)
	{
	assert(pdata);
	
	// setup the grid to run the kernel
	int M = 128;
	dim3 grid(pdata->N/M+1, 1, 1);
	dim3 threads(M, 1, 1);
	
	// run the kernel
	pdata_test_fill<<< grid, threads >>>(*pdata);
	cudaThreadSynchronize();
	return cudaGetLastError();
	}

//! Texture for reading particle positions
texture<float4, 1, cudaReadModeElementType> pdata_pos_tex;

//! Kernel for testing texture read capability
/*! \param pdata Particle data to write to
	\post \c pdata.vel[i] holds the value read from \c pdata_pos_tex at location \c i
*/
__global__ void pdata_texread_test(gpu_pdata_arrays pdata)
	{
	// start by identifying the particle index of this particle
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int pidx = idx + pdata.local_beg;

	if (idx < pdata.local_num)
		{
		float4 pos = tex1Dfetch(pdata_pos_tex, pidx);
		pdata.vel[pidx] = pos;
		}
	}

/*!	\param pdata Particle data arrays to write the velocity 
	\pre The texture that the caller wants read from was bound with gpu_bind_pdata_textures()
	\post The vel device memory is filled out with what is read from the position texture
	\note Designed to be used for unit testing texture reads

	\returns Error result from the kernel call
*/
cudaError_t gpu_pdata_texread_test(const gpu_pdata_arrays &pdata)
	{	
	// setup the grid to run the kernel
	int M = 128;
	dim3 grid(pdata.local_num/M+1, 1, 1);
	dim3 threads(M, 1, 1);

	// bind the textures
	cudaError_t error = cudaBindTexture(0, pdata_pos_tex, pdata.pos, sizeof(float4) * pdata.N);
	if (error != cudaSuccess)
		return error;
	
	// run the kernel
	pdata_texread_test<<< grid, threads >>>(pdata);
	cudaThreadSynchronize();
	return cudaGetLastError();
	}

