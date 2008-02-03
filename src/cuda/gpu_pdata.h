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

#ifndef _CUDA_PDATA_H_
#define _CUDA_PDATA_H_

#include <stdio.h>
#include <cuda_runtime_api.h>

/*! \file gpu_pdata.h
 	\brief Declares extremely low-level functions for working with particle data on the GPU
 	\ingroup cuda_code
*/

#ifdef NVCC
//! The texture for reading the pdata pos array
texture<float4, 1, cudaReadModeElementType> pdata_pos_tex;
texture<float4, 1, cudaReadModeElementType> pdata_vel_tex;
texture<float4, 1, cudaReadModeElementType> pdata_accel_tex;
#endif

extern "C" {

//! Structure of arrays of the particle data as it resides on the GPU
/*! Stores pointers to the particles positions, velocities, acceleartions, and particle tags.
	Particle type information is most likely needed along with the position, so the type
	is encoded in the 4th float in the position float4 as an integer. Device code
	can decode this type data with __float_as_int();
	
	A second part of the particle data that lives on the device, not included in this structure
	is the texture that references the particle positions. The texture is a simple 1D texture
	bound to device memory. However, since this C structure is a portion of a c++ master, 
	it is possible that two c++ classes may exist at the same time. But there is only one
	texture reference, so master classes should be certain to call gpu_bind_pdata_tex()
	before calling any kernel that needs to use it.

	All the pointers in this structure will be allocated on the device.
*/
struct gpu_pdata_arrays
	{
	float4 *pos;	//!< Particle position in \c x,\c y,\c z, particle type as an int in \c w
	float4 *vel;	//!< Particle velocity in \c x, \c y, \c z, nothing in \c w
	float4 *accel;	//!< Particle acceleration in \c x, \c y, \c z, nothing in \c w
	unsigned int *tag;	//!< Particle tag
	unsigned int *rtag;	//!< Particle rtag 
	
	unsigned int N;	//!< Number of particles in the arrays
	};

//! Store the box size on the GPU
/*	\note For performance reasons, the GPU code is allowed to assume that the box goes
	from -L/2 to L/2, and the box dimensions in this structure must reflect that.
*/
struct gpu_boxsize
	{
	float Lx;	//!< Length of the box in the x-direction
	float Ly;	//!< Length of the box in the y-direction
	float Lz;	//!< Length of teh box in the z-direction
	float Lxinv;//!< 1.0f/Lx
	float Lyinv;//!< 1.0f/Ly
	float Lzinv;//!< 1.0f/Lz
	};
	
//! Binds the texture on the device to this data array
cudaError_t gpu_bind_pdata_textures(gpu_pdata_arrays *pdata);

//! Helper kernel for un-interleaving data
cudaError_t gpu_uninterleave_float4(float *d_out, float4 *d_in, int N, int pitch);
//! Helper kernel for interleaving data
cudaError_t gpu_interleave_float4(float4 *d_out, float *d_in, int N, int pitch);

//! Generate a test pattern in the data on the GPU (for unit testing)
cudaError_t gpu_generate_pdata_test(gpu_pdata_arrays *pdata);
//! Read from the pos texture and write into the vel array (for unit testing)
cudaError_t gpu_pdata_texread_test(gpu_pdata_arrays *pdata); 
}

#endif	
