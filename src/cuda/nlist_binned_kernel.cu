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

// textures used in this kernel: commented out because textures are managed globally currently
texture<float4, 2, cudaReadModeElementType> nlist_idxlist_tex;
texture<unsigned int, 2, cudaReadModeElementType> bin_adj_tex;
texture<unsigned int, 1, cudaReadModeElementType> mem_location_tex;

/*! \file nlist_binned_kernel.cu
	\brief Contains code for the kernel that implements the binned O(N) neighbor list on the GPU
*/

// upating the list from the bins will involve looping over all the particles in the bin and 
// comparing them to all particles in this and neighboring bins
// each block will process one bin. Since each particle is only placed in one bin, each block thus processes
// a block of particles (though which particles it processes is random)
// Empty bin entries will be set to 0xffffffff to allow for efficient handling

#define EMPTY_BIN 0xffffffff

extern "C" __global__ void updateFromBins_new(gpu_pdata_arrays pdata, gpu_bin_array bins, gpu_nlist_array nlist, float r_maxsq, unsigned int actual_Nmax, gpu_boxsize box, float scalex, float scaley, float scalez)
	{
	// each thread is going to compute the neighbor list for a single particle
	int my_pidx = blockDim.x * blockIdx.x + threadIdx.x;
	
	// quit early if we are past the end of the array
	if (my_pidx >= pdata.N)
		return;
	
	// first, determine which bin this particle belongs to
	float4 my_pos = pdata.pos[my_pidx];
	uint4 exclude = nlist.exclusions[my_pidx];
	
	unsigned int ib = (unsigned int)((my_pos.x+box.Lx/2.0f)*scalex);
	unsigned int jb = (unsigned int)((my_pos.y+box.Ly/2.0f)*scaley);
	unsigned int kb = (unsigned int)((my_pos.z+box.Lz/2.0f)*scalez);

	// need to handle the case where the particle is exactly at the box hi
	if (ib == bins.Mx)
		ib = 0;
	if (jb == bins.My)
		jb = 0;
	if (kb == bins.Mz)
		kb = 0;

	int my_bin = tex1Dfetch(mem_location_tex, ib*(bins.Mz*bins.My) + jb * bins.Mz + kb);

	// each thread will determine the neighborlist of a single particle
	int n_neigh = 0;	// count number of neighbors found so far
	
	// loop over all adjacent bins
	for (unsigned int cur_adj = 0; cur_adj < 27; cur_adj++)
		{
		int neigh_bin = tex2D(bin_adj_tex, my_bin, cur_adj);
		
		// printf("%d ", neigh_bin);
		
		// now, we are set to loop through the array
		for (int cur_offset = 0; cur_offset < actual_Nmax; cur_offset++)
			{
			float4 cur_neigh_blob = tex2D(nlist_idxlist_tex, neigh_bin, cur_offset);
			float3 neigh_pos;
			neigh_pos.x = cur_neigh_blob.x;
			neigh_pos.y = cur_neigh_blob.y;
			neigh_pos.z = cur_neigh_blob.z;
			int cur_neigh = __float_as_int(cur_neigh_blob.w);
			
			if (cur_neigh != EMPTY_BIN)
				{
				float dx = my_pos.x - neigh_pos.x;
				dx = dx - box.Lx * rintf(dx * box.Lxinv);

				float dy = my_pos.y - neigh_pos.y;
				dy = dy - box.Ly * rintf(dy * box.Lyinv);

				float dz = my_pos.z - neigh_pos.z;
				dz = dz - box.Lz * rintf(dz * box.Lzinv);

				float dr = dx*dx + dy*dy + dz*dz;
				int not_excluded = (exclude.x != cur_neigh) & (exclude.y != cur_neigh) & (exclude.z != cur_neigh) & (exclude.w != cur_neigh);
				
				if (dr < r_maxsq && (my_pidx != cur_neigh) && not_excluded)
					{
					// check for overflow
					if (n_neigh < nlist.height)
						{
						nlist.list[my_pidx + n_neigh*nlist.pitch] = cur_neigh;
						n_neigh++;
						}
					else
						*nlist.overflow = 1;
					}
				}
			}
		}
		
	// printf("\n");
	
	nlist.n_neigh[my_pidx] = n_neigh;
	nlist.last_updated_pos[my_pidx] = my_pos;
	}
	
cudaError_t gpu_nlist_binned(gpu_pdata_arrays *pdata, gpu_boxsize *box, gpu_bin_array *bins, gpu_nlist_array *nlist, float r_maxsq, int curNmax, int block_size)
	{
	assert(bins);
	assert(pdata);
	assert(nlist);
	assert(block_size > 0);

	// setup the grid to run the kernel
	int nblocks = (int)ceil((double)pdata->N/ (double)block_size);
	
	dim3 grid(nblocks, 1, 1);
	dim3 threads(block_size, 1, 1);

	// bind the textures
	nlist_idxlist_tex.normalized = false;
	nlist_idxlist_tex.filterMode = cudaFilterModePoint;
	cudaError_t error = cudaBindTextureToArray(nlist_idxlist_tex, bins->idxlist_array);
	if (error != cudaSuccess)
		return error;
		
	bin_adj_tex.normalized = false;
	bin_adj_tex.filterMode = cudaFilterModePoint;
	error = cudaBindTextureToArray(bin_adj_tex, bins->bin_adj_array);
	if (error != cudaSuccess)
		return error;
		
	error = cudaBindTexture(0, mem_location_tex, bins->mem_location, sizeof(unsigned int)*bins->Mx*bins->My*bins->Mz);
	if (error != cudaSuccess)
		return error;
			
	// zero the overflow check
	error = cudaMemset(nlist->overflow, 0, sizeof(int));
	if (error != cudaSuccess)
		return error;
	
	// make even bin dimensions
	float binx = (box->Lx) / float(bins->Mx);
	float biny = (box->Ly) / float(bins->My);
	float binz = (box->Lz) / float(bins->Mz);

	// precompute scale factors to eliminate division in inner loop
	float scalex = 1.0f / binx;
	float scaley = 1.0f / biny;
	float scalez = 1.0f / binz;

	// run the kernel
	updateFromBins_new<<< grid, threads>>>(*pdata, *bins, *nlist, r_maxsq, curNmax, *box, scalex, scaley, scalez);
	
	#ifdef NDEBUG
	return cudaSuccess;
	#else
	cudaThreadSynchronize();
	return cudaGetLastError();
	#endif
	}

// vim:syntax=cpp
