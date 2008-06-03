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
#include "gpu_pdata.h"
#include <stdio.h>

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file nlist_nsq_kernel.cu
	\brief Contains kernel code that implements the O(N^2) kernel on the GPU
*/

const int NLIST_BLOCK_SIZE = 128;
// generate the neighbor list
extern "C" __global__ void generateNlistNSQ(gpu_pdata_arrays pdata, gpu_nlist_array nlist, float r_maxsq, gpu_boxsize box) 
	{
	// each thread is to compute the neighborlist for a single particle i
	// each block will load a bunch of particles into shared mem and then each thread will compare it's particle
	// to each particle in shmem to see if they are a neighbor. Since all threads in the block access the same 
	// shmem element at the same time, the value is broadcast and there are no bank conflicts

	// the way this funciton loads data, all data arrays need to be padded so they have a multiple of 
	// blockDim.x elements. 

	// shared data to store all of the particles we compare against
	__shared__ float sdata[NLIST_BLOCK_SIZE*4];
	
	// load in the particle
	int pidx = blockIdx.x * NLIST_BLOCK_SIZE + threadIdx.x;

	float4 pos = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	if (pidx < pdata.N)
		pos = pdata.pos[pidx];
		
	float px = pos.x;
	float py = pos.y;
	float pz = pos.z;

	// track the number of neighbors added so far
	int n_neigh = 0;
	
	uint4 exclude = make_uint4(0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff);
	if (pidx < pdata.N)
		exclude = nlist.exclusions[pidx];
	
	// each block is going to loop over all N particles (this assumes memory is padded to a multiple of blockDim.x)
	// in blocks of blockDim.x
	for (int start = 0; start < pdata.N; start += NLIST_BLOCK_SIZE)
		{
		// load data
		float4 neigh_pos = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		if (start + threadIdx.x < pdata.N)
			neigh_pos = pdata.pos[start + threadIdx.x];
		
		// make sure everybody is caught up before we stomp on the memory
		__syncthreads();
		sdata[threadIdx.x] = neigh_pos.x;
		sdata[threadIdx.x + NLIST_BLOCK_SIZE] = neigh_pos.y;
		sdata[threadIdx.x + 2*NLIST_BLOCK_SIZE] = neigh_pos.z;
		sdata[threadIdx.x + 3*NLIST_BLOCK_SIZE] = neigh_pos.w; //< unused, but try to get compiler to fully coalesce reads

		// ensure all data is loaded
		__syncthreads();

		// now each thread loops over every particle in shmem, but doesn't loop past the end of the particle list (since
		// the block might extend that far)
		int end_offset= NLIST_BLOCK_SIZE;
		end_offset = min(end_offset, pdata.N - start);

		if (pidx < pdata.N)
			{ 

			for (int cur_offset = 0; cur_offset < end_offset; cur_offset++)
				{
				// calculate dr
				float dx = px - sdata[cur_offset];
				dx = dx - box.Lx * rintf(dx * box.Lxinv);
				
				if (dx*dx < r_maxsq)
					{
					float dy = py - sdata[cur_offset + NLIST_BLOCK_SIZE];
					dy = dy - box.Ly * rintf(dy * box.Lyinv);
				
					if (dy*dy < r_maxsq)
						{
						float dz = pz - sdata[cur_offset + 2*NLIST_BLOCK_SIZE];
						dz = dz - box.Lz * rintf(dz * box.Lzinv);
				
						float drsq = dx*dx + dy*dy + dz*dz;
	
						// we don't add if we are comparing to ourselves, and we don't add if we are above the cut
						if ((drsq < r_maxsq) && ((start + cur_offset) != pidx) && exclude.x != (start + cur_offset) && exclude.y != (start + cur_offset) && exclude.z != (start + cur_offset) && exclude.w != (start + cur_offset))
							{
							nlist.list[pidx + (1 + n_neigh)*nlist.pitch] = start+cur_offset;
							n_neigh++;
							}
					
						}
					}
				}
			}
		}

	// now that we are done: update the first row that lists the number of neighbors
	if (pidx < pdata.N)
		{
		nlist.list[pidx] = n_neigh;
		nlist.last_updated_pos[pidx] = pdata.pos[pidx];
		}
	}

// driver function for the kernel
cudaError_t gpu_nlist_nsq(gpu_pdata_arrays *pdata, gpu_boxsize *box, gpu_nlist_array *nlist, float r_maxsq)
	{
	assert(pdata);
	assert(nlist);
	
	// setup the grid to run the kernel
	int M = NLIST_BLOCK_SIZE;
	dim3 grid( (pdata->N/M) + 1, 1, 1);
	dim3 threads(M, 1, 1);
	
	// run the kernel
	generateNlistNSQ<<< grid, threads >>>(*pdata, *nlist, r_maxsq, *box);
	#ifdef NDEBUG
	return cudaSuccess;
	#else
	cudaThreadSynchronize();
	return cudaGetLastError();
	#endif	
	}

// vim:syntax=cpp
