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

#include "gpu_nlist.h"
#include "gpu_nlist_nvcc.h"
#include "gpu_utils.h"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file gpu_nlist.h
	\brief Contains code for working with the neighbor list data structure on the GPU
*/

/*! \post The neighborlist memory is allocated both on the host and the device, and initialized to 0
*/
void gpu_alloc_nlist_data(gpu_nlist_data *nlist, unsigned int N, unsigned int height)
	{
	assert(nlist);
	// pad N up 512 elements so that when the device reads past the end of the array, it isn't reading junk memory
	unsigned int padded_N = N + 512;
	size_t pitch;
	
	// allocate and zero device memory
	CUDA_SAFE_CALL( cudaMallocPitch( (void**) &nlist->d_array.list, &pitch, padded_N*sizeof(unsigned int), height));
	// want pitch in elements, not bytes
	nlist->d_array.pitch = pitch / sizeof(int);
	nlist->d_array.height = height;
	CUDA_SAFE_CALL( cudaMemset( (void*) nlist->d_array.list, 0, pitch * height) );
	
	CUDA_SAFE_CALL( cudaMalloc( (void**) &nlist->d_array.last_updated_pos, nlist->d_array.pitch*sizeof(float4)) );
	CUDA_SAFE_CALL( cudaMemset( (void*) nlist->d_array.last_updated_pos, 0, nlist->d_array.pitch * sizeof(float4)) );

	CUDA_SAFE_CALL( cudaMalloc( (void**) &nlist->d_array.needs_update, sizeof(int)) );
	
	CUDA_SAFE_CALL( cudaMalloc( (void**) &nlist->d_array.exclusions, nlist->d_array.pitch*sizeof(uint4)) );
	CUDA_SAFE_CALL( cudaMemset( (void*) nlist->d_array.exclusions, 0xff, nlist->d_array.pitch * sizeof(uint4)) );

	// allocate and zero host memory
	CUDA_SAFE_CALL( cudaMallocHost( (void**)&nlist->h_array.list, pitch * height) );
	memset((void*)nlist->h_array.list, 0, pitch*height);
	
	CUDA_SAFE_CALL( cudaMallocHost( (void**)&nlist->h_array.exclusions, N * sizeof(uint4)) );
	memset((void*)nlist->h_array.exclusions, 0xff, N * sizeof(uint4));
		
	nlist->h_array.pitch = pitch / sizeof(int);
	nlist->h_array.height = height;
	
	nlist->N = N;
	
	///****************** TEMPORARY: needs to be moved to a select function
	nlist_exclude_tex.addressMode[0] = cudaAddressModeClamp;
	nlist_exclude_tex.addressMode[1] = cudaAddressModeClamp;
	nlist_exclude_tex.filterMode = cudaFilterModePoint;
	nlist_exclude_tex.normalized = false;
	// Bind the array to the texture
   	cudaBindTexture(0, nlist_exclude_tex, nlist->d_array.exclusions, sizeof(uint4) * N);
	
	}
	
/*! \post memory is freed and pointers are set to NULL
*/
void gpu_free_nlist_data(gpu_nlist_data *nlist)
	{
	assert(nlist);
	
	CUDA_SAFE_CALL( cudaFreeHost(nlist->h_array.list) );
	nlist->h_array.list = NULL;
	CUDA_SAFE_CALL( cudaFreeHost(nlist->h_array.exclusions) );
	nlist->h_array.exclusions = NULL;

	CUDA_SAFE_CALL( cudaFree(nlist->d_array.list) );
	nlist->d_array.list = NULL;
	CUDA_SAFE_CALL( cudaFree(nlist->d_array.exclusions) );
	nlist->d_array.exclusions = NULL;	
	CUDA_SAFE_CALL( cudaFree(nlist->d_array.last_updated_pos) );
	nlist->d_array.last_updated_pos = NULL;
	CUDA_SAFE_CALL( cudaFree(nlist->d_array.needs_update) );
	nlist->d_array.needs_update = NULL;
	}

/*! \post The neighbor list host array (h_array) is copied to 
 			device memory (d_array)
 */
void gpu_copy_nlist_data_htod(gpu_nlist_data *nlist)
	{
	assert(nlist);

	CUDA_SAFE_CALL( cudaMemcpy(nlist->d_array.list, nlist->h_array.list, 
			sizeof(unsigned int) * nlist->d_array.height * nlist->d_array.pitch, 
			cudaMemcpyHostToDevice) );
	}
	
	
/*! \post The neighbor list device array (d_array) is copied to 
 			host memory (h_array)
 */
void gpu_copy_nlist_data_dtoh(gpu_nlist_data *nlist)
	{
	assert(nlist);
	
	CUDA_SAFE_CALL( cudaMemcpy(nlist->h_array.list, nlist->d_array.list, 
			sizeof(unsigned int) * nlist->d_array.height * nlist->d_array.pitch, 
			cudaMemcpyDeviceToHost) );
	}
	
	
/*! \todo document me
*/
void gpu_copy_exclude_data_htod(gpu_nlist_data *nlist)
	{
	assert(nlist);

	CUDA_SAFE_CALL( cudaMemcpy(nlist->d_array.exclusions, nlist->h_array.exclusions, 
			sizeof(uint4) * nlist->N, 
			cudaMemcpyHostToDevice) );
	}


__global__ void nlist_data_test_fill(gpu_nlist_array nlist)
	{
	// start by identifying the particle index of this particle
	int pidx = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = 0; i < nlist.height; i++)
		nlist.list[i*nlist.pitch + pidx] = 2*pidx + i;
	}

/*! \post The neighborlist device memory is filled out with a test pattern.
 			See the code of nlist_data_test_fill() for details on what that pattern is.
 			This is intended to be used for unit testing only.
 */
void gpu_generate_nlist_data_test(gpu_nlist_data *nlist)
	{
	assert(nlist);
	
	// setup the grid to run the kernel
	int M = 128;
	dim3 grid(nlist->N/M+1, 1, 1);
	dim3 threads(M, 1, 1);
	
	// run the kernel
	nlist_data_test_fill<<< grid, threads >>>(nlist->d_array);
	CUT_CHECK_ERROR("Kernel execution failed");
	}
	

/////////////////////////////////////////////////////////
// bin data

void gpu_alloc_bin_data(gpu_bin_data *bins, unsigned int Mx, unsigned int My, unsigned int Mz, unsigned int Nmax)
	{		
	assert(bins);
	// setup the dimensions
	bins->d_array.Mx = bins->h_array.Mx = Mx;
	bins->d_array.My = bins->h_array.My = My;
	bins->d_array.Mz = bins->h_array.Mz = Mz;

	// use mallocPitch to make sure that memory accesses are coalesced	
	size_t pitch;

	// allocate and zero device memory
	if (Mx*My*Mz*Nmax >= 500000*128)
		printf("Allocating abnormally large cell list: %d %d %d %d\n", Mx, My, Mz, Nmax);

	CUDA_SAFE_CALL( cudaMallocPitch( (void**) &bins->d_array.idxlist, &pitch, Nmax*sizeof(unsigned int), Mx*My*Mz));
	// want pitch in elements, not bytes
	Nmax = pitch / sizeof(unsigned int);
	CUDA_SAFE_CALL( cudaMemset( (void*) bins->d_array.idxlist, 0, pitch * Mx*My*Mz) );
	
	// allocate the bin coord array
	CUDA_SAFE_CALL( cudaMalloc( (void**) &bins->d_array.bin_coord, Mx*My*Mz*sizeof(uint4)) );

	// allocate and zero host memory
	CUDA_SAFE_CALL( cudaMallocHost( (void**)&bins->h_array.idxlist, pitch * Mx*My*Mz) );
	memset((void*)bins->h_array.idxlist, 0, pitch*Mx*My*Mz);
	
	// allocate the bin coord array
	bins->h_array.bin_coord = (uint4*)malloc(sizeof(uint4) * Mx*My*Mz);
	// initialize the coord array
	for (int i = 0; i < Mx; i++)
		{
		for (int j = 0; j < My; j++)
			{
			for (int k = 0; k < Mz; k++)
				{
				int bin = i*Mz*My + j*Mz + k;
				bins->h_array.bin_coord[bin].x = i;
				bins->h_array.bin_coord[bin].y = j;
				bins->h_array.bin_coord[bin].z = k;
				bins->h_array.bin_coord[bin].w = 0;
				}
			}
		}
	// copy it to the device. This only needs to be done once
	CUDA_SAFE_CALL( cudaMemcpy(bins->d_array.bin_coord, bins->h_array.bin_coord, sizeof(uint4)*Mx*My*Mz, cudaMemcpyHostToDevice) );

	// assign allocated pitch
	bins->d_array.Nmax = bins->h_array.Nmax = Nmax;
	//printf("allocated Nmax = %d / allocated nbins = %d\n", Nmax, Mx*My*Mz);
	}

void gpu_free_bin_data(gpu_bin_data *bins)
	{
	assert(bins);
	// free the device memory
	CUDA_SAFE_CALL( cudaFree(bins->d_array.idxlist) );
	CUDA_SAFE_CALL( cudaFree(bins->d_array.bin_coord) );
	// free the hsot memory
	CUDA_SAFE_CALL( cudaFreeHost(bins->h_array.idxlist) );
	free(bins->h_array.bin_coord);

	// set pointers to NULL so no one will think they are valid 
	bins->d_array.idxlist = NULL;
	bins->h_array.idxlist = NULL;
	}

void gpu_copy_bin_data_htod(gpu_bin_data *bins)
	{
	assert(bins);

	unsigned int nbytes = bins->d_array.Mx * bins->d_array.My * bins->d_array.Mz * bins->d_array.Nmax * sizeof(unsigned int);

	CUDA_SAFE_CALL( cudaMemcpy(bins->d_array.idxlist, bins->h_array.idxlist, 
			nbytes, cudaMemcpyHostToDevice) );
	}

void gpu_copy_bin_data_dtoh(gpu_bin_data *bins)
	{
	assert(bins);

	unsigned int nbytes = bins->d_array.Mx * bins->d_array.My * bins->d_array.Mz * bins->d_array.Nmax * sizeof(unsigned int);

	CUDA_SAFE_CALL( cudaMemcpy(bins->h_array.idxlist, bins->d_array.idxlist, 
			nbytes, cudaMemcpyDeviceToHost) );
	}



//////////////////////////////////////////////////////////
__global__ void gpu_nlist_needs_update_check_kernel(gpu_pdata_arrays pdata, gpu_nlist_array nlist, float r_buffsq, gpu_boxsize box)
	{
	// each thread will compare vs it's old position to see if the list needs updating
	// if that is true, write a 1 to nlist_needs_updating
	// it is possible that writes will collide, but at least one will succeed and that is all that matters
	int pidx = blockIdx.x * blockDim.x + threadIdx.x;

	if (pidx < pdata.N)
		{
		float4 cur_pos = pdata.pos[pidx];
		float4 last_pos = nlist.last_updated_pos[pidx];
		float dx = cur_pos.x - last_pos.x;
		float dy = cur_pos.y - last_pos.y;
		float dz = cur_pos.z - last_pos.z;
	
		dx = dx - box.Lx * rintf(dx * box.Lxinv);
		dy = dy - box.Ly * rintf(dy * box.Lyinv);
		dz = dz - box.Lz * rintf(dz * box.Lzinv);
	
		float drsq = dx*dx + dy*dy + dz*dz;

		if (drsq >= r_buffsq && pidx < pdata.N)
			{
			*nlist.needs_update = 1;
			}
		}
	}

//! Check if the neighborlist needs updating
int gpu_nlist_needs_update_check(gpu_pdata_arrays *pdata, gpu_boxsize *box, gpu_nlist_data *nlist, float r_buffsq)
	{
	assert(pdata);
	assert(nlist);
	
	// start by zeroing the value on the device
	int result = 0;
	CUDA_SAFE_CALL( cudaMemcpy(nlist->d_array.needs_update, &result, 
			sizeof(int), cudaMemcpyHostToDevice) );
	
	// run the kernel
    int M = 128;
    dim3 grid( (pdata->N/M) + 1, 1, 1);
    dim3 threads(M, 1, 1);

    // run the kernel
    gpu_nlist_needs_update_check_kernel<<< grid, threads >>>(*pdata, nlist->d_array, r_buffsq, *box);
    CUT_CHECK_ERROR("Kernel execution failed");

	CUDA_SAFE_CALL( cudaMemcpy(&result, nlist->d_array.needs_update,
			sizeof(int), cudaMemcpyDeviceToHost) );
	return result;
	}

// vim:syntax=cpp
