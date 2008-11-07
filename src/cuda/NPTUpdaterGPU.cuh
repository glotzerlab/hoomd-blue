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

#ifndef _CUDA_UPDATERS_H_
#define _CUDA_UPDATERS_H_

#include <stdio.h>
#include <cuda_runtime.h>

#include "ParticleData.cuh"

/*! \file NVTUpdaterGPU.cuh
	\brief Declares GPU kernel code for NPT integration on the GPU. Used by NPTUpdaterGPU.
*/

//! Data structure storing needed intermediate values for NPT integration
struct gpu_npt_data
	{
	float *partial_Ksum; //!< NBlocks elements, each is a partial sum of m*v^2
	float *Ksum;	//!< fully reduced Ksum on one GPU
	float *partial_Psum; //!< NBlocks elements, each is a partial sum of virials
	float *Psum;  //!< fully reduced Psum on one GPU
	float *virial;  //!< Stores virials
	int NBlocks;	//!< Number of blocks in the computation
	int block_size; //!< Block size of the kernel to be run on the device (must be a power of 2)
	};


//! Sums virials on the GPU. Used by NPTUpdaterGPU.
cudaError_t gpu_integrator_sum_virials(const gpu_npt_data &nptdata, const gpu_pdata_arrays &pdata, float** virial_list, int num_virials);

//! Kernel driver for the the first step of the computation called by NPTUpdaterGPU
cudaError_t gpu_npt_pre_step(const gpu_pdata_arrays &pdata, const gpu_boxsize &box, const gpu_npt_data &d_npt_data, float Xi, float Eta, float deltaT);

//! Kernel driver for the the second step of the computation called by NPTUpdaterGPU
cudaError_t gpu_npt_step(const gpu_pdata_arrays &pdata, const gpu_npt_data &d_npt_data, float4 **force_data_ptrs, int num_forces, float Xi, float Eta, float deltaT);

//! Kernel driver for calculating the final pass Ksum on the GPU. Used by NPTUpdaterGPU
cudaError_t gpu_npt_reduce_ksum(const gpu_npt_data &d_npt_data);

//! Kernel driver for calculating the initial pass Ksum on the GPU. Used by NPTUpdaterGPU
cudaError_t gpu_npt_temperature(const gpu_npt_data &d_npt_data, const gpu_pdata_arrays &pdata);

//! Kernel driver for calculating the final pass Psum on the GPU. Used by NPTUpdaterGPU
cudaError_t gpu_npt_reduce_psum(const gpu_npt_data &d_npt_data);

//! Kernel driver for calculating the initial pass Ksum on the GPU. Used by NPTUpdaterGPU
cudaError_t gpu_npt_pressure(const gpu_npt_data &d_npt_data, const gpu_pdata_arrays &pdata);

#endif
