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

#include "gpu_forces.h"
#include "gpu_utils.h"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file gpu_forces.cu
 	\brief Contains CPU code and kernels generic to all gpu force computations 
*/

/////////////////////////////////////////////////////////////////////
// Bond table
void gpu_alloc_bondtable_data(gpu_bondtable_data *blist, unsigned int N, unsigned int height)
	{
	assert(blist);
	// pad N up 512 elements so that when the device reads past the end of the array, it isn't reading junk memory
	unsigned int padded_N = N + 512;
	size_t pitch;
	
	// allocate and zero device memory
	CUDA_SAFE_CALL( cudaMallocPitch( (void**) &blist->d_array.list, &pitch, padded_N*sizeof(unsigned int), height));
	// want pitch in elements, not bytes
	blist->d_array.pitch = pitch / sizeof(int);
	blist->d_array.height = height;
	CUDA_SAFE_CALL( cudaMemset( (void*) blist->d_array.list, 0, pitch * height) );
	
	// allocate and zero host memory
	CUDA_SAFE_CALL( cudaMallocHost( (void**)&blist->h_array.list, pitch * height) );
	memset((void*)blist->h_array.list, 0, pitch*height);
	blist->h_array.pitch = pitch / sizeof(int);
	blist->h_array.height = height;
	
	blist->N = N;
	}
	
//! Free memory
void gpu_free_bondtable_data(gpu_bondtable_data *blist)
	{
	assert(blist);
	CUDA_SAFE_CALL( cudaFree(blist->d_array.list) );
	blist->d_array.list = NULL;
	CUDA_SAFE_CALL( cudaFreeHost(blist->h_array.list) );
	blist->h_array.list = NULL;
	}
	
//! Copy data to the device
void gpu_copy_bontable_data_htod(gpu_bondtable_data *blist)
	{
	assert(blist);

	CUDA_SAFE_CALL( cudaMemcpy(blist->d_array.list, blist->h_array.list, 
			sizeof(unsigned int) * blist->d_array.height * blist->d_array.pitch, 
			cudaMemcpyHostToDevice) );
	}

// vim:syntax=cpp
