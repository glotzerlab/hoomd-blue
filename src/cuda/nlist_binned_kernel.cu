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
texture<unsigned int, 2, cudaReadModeElementType> nlist_idxlist_tex;
texture<uint4, 1, cudaReadModeElementType> nlist_bincoord_tex;

//! Texture for reading particle positions
texture<float4, 1, cudaReadModeElementType> pdata_pos_tex;

/*! \file nlist_binned_kernel.cu
	\brief Contains code for the kernel that implements the binned O(N) neighbor list on the GPU
*/

// upating the list from the bins will involve looping over all the particles in the bin and 
// comparing them to all particles in this and neighboring bins
// each block will process one bin. Since each particle is only placed in one bin, each block thus processes
// a block of particles (though which particles it processes is random)
// Empty bin entries will be set to 0xffffffff to allow for efficient handling

#define EMPTY_BIN 0xffffffff

// the shared data will hold the particle positions of all the particles in the neighboring bins as well as the 
// particle index of the particle (stored as an int to be accessed with __float_as_int)
// px,py,pz will have to be checked vs all of these
//extern __shared__ float bin_sdata[];

/*__device__ int wrap_bin_i(int i, int M)
	{
	if (i < 0)
		i += M;
	if (i >= M)
		i -= M;
	return i;
	}
	
__device__ int handle_bin(int a, int b, int c, int my_pidx, float px, float py, float pz, int n_neigh, gpu_pdata_arrays pdata, gpu_bin_array bins, gpu_nlist_array nlist, float r_maxsq, gpu_boxsize box, uint4 exclude)
	{
	// now: we finally know the current bin to compare to
	int neigh_bin = a*bins.Mz*bins.My + b*bins.Mz + c;

	int neigh_pidx = bins.idxlist[neigh_bin*bins.Nmax + threadIdx.x];
	
	// read in the neighbor position
	float4 neigh_pos = make_float4(0.0f,0.0f,0.0f,0.0f);
	neigh_pos = tex1Dfetch(pdata_pos_tex, neigh_pidx);
	
	// put it into shared memory
	__syncthreads();
	
	bin_sdata[threadIdx.x] = neigh_pos.x;
	bin_sdata[threadIdx.x + blockDim.x] = neigh_pos.y;
	bin_sdata[threadIdx.x + 2*blockDim.x] = neigh_pos.z;
	bin_sdata[threadIdx.x + 3*blockDim.x] = __int_as_float(neigh_pidx);
	
	__syncthreads();
	
	if (my_pidx != EMPTY_BIN)
	{
	// now, we are set to loop through the array
	for (int cur_offset = 0; cur_offset < blockDim.x; cur_offset++)
		{
		int cur_neigh = __float_as_int(bin_sdata[cur_offset + 3*blockDim.x]);
		if (cur_neigh == EMPTY_BIN)
			return n_neigh;

		float dx = px - bin_sdata[cur_offset];
		float dy = py - bin_sdata[cur_offset + blockDim.x];
		float dz = pz - bin_sdata[cur_offset + 2*blockDim.x];
		
		dx = dx - box.Lx * rintf(dx * box.Lxinv);
		dy = dy - box.Ly * rintf(dy * box.Lyinv);
		dz = dz - box.Lz * rintf(dz * box.Lzinv);

		float dr = dx*dx + dy*dy + dz*dz;
	
		int not_excluded = (exclude.x != cur_neigh) & (exclude.y != cur_neigh) & (exclude.z != cur_neigh) & (exclude.w != cur_neigh);
		if (dr < r_maxsq && (my_pidx != cur_neigh) && not_excluded)
			{
			nlist.list[my_pidx + (1 + n_neigh)*nlist.pitch] = cur_neigh;
			n_neigh++;
			}
		}
	}
	
	return n_neigh;
	}

// unrolled version (non full list)
extern "C" __global__ void updateFromBins(gpu_pdata_arrays pdata, gpu_bin_array bins, gpu_nlist_array nlist, float r_maxsq, gpu_boxsize box)
	{
	// each block is going to compute the neighborlist for all of the particles in blockIdx.x
	int my_bin = blockIdx.x;

	// each thread will determine the neighborlist of a single particle: threadIdx.x in the current bin
	int n_neigh = 0;	// count number of neighbors found so far
	
	// we will need to loop over all neighboring bins. In order to do that, we need to know what bin we are actually in!
	// this could be a messy bunch of modulus operations, so we just read it out of an array that has been pre-computed for us :)
	__shared__ volatile int my_i, my_j, my_k, my_w;
	__syncthreads();
	if (threadIdx.x == 0)
		{
		uint4 coords = bins.bin_coord[my_bin];
		my_i = coords.x;
		my_j = coords.y;
		my_k = coords.z;
		my_w = coords.w;
		}
	__syncthreads();

	// pull in the particle of this thread that will be compared to everything else
	int my_pidx = bins.idxlist[my_bin*bins.Nmax + threadIdx.x];

	float4 my_pos = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	uint4 exclude = make_uint4(EMPTY_BIN, EMPTY_BIN, EMPTY_BIN, EMPTY_BIN);
	//if (my_pidx != EMPTY_BIN)
		//{
		my_pos = tex1Dfetch(pdata_pos_tex, my_pidx);
		//my_pos = pdata.pos[my_pidx];
	
		exclude = tex1Dfetch(nlist_exclude_tex, my_pidx);
		//exclude = nlist.exclusions[my_pidx];
		//}

	float px = my_pos.x;
	float py = my_pos.y;
	float pz = my_pos.z;

	// loop through the 27 neighboring bins (unrolled)
	// -1 -1 -1

	__syncthreads();
	if (threadIdx.x == 0)
		{
		my_i--;
		my_j--;
		my_k--;
		my_i = wrap_bin_i(my_i, bins.Mx);
		my_j = wrap_bin_i(my_j, bins.My);
		my_k = wrap_bin_i(my_k, bins.Mz);
		}
	__syncthreads();
	n_neigh = handle_bin(my_i, my_j, my_k, my_pidx, px, py, pz, n_neigh, pdata, bins, nlist, r_maxsq, box, exclude);
	
	// -1 -1 0
	__syncthreads();
	if (threadIdx.x == 0)
		{
		my_k++;
		my_k = wrap_bin_i(my_k, bins.Mz);
		}
	__syncthreads();
	n_neigh = handle_bin(my_i, my_j, my_k, my_pidx, px, py, pz, n_neigh, pdata, bins, nlist, r_maxsq, box, exclude);
	
	// -1 -1 1
	__syncthreads();
	if (threadIdx.x == 0)
		{
		my_k++;
		my_k = wrap_bin_i(my_k, bins.Mz);
		}
	__syncthreads();
	n_neigh = handle_bin(my_i, my_j, my_k, my_pidx, px, py, pz, n_neigh, pdata, bins, nlist, r_maxsq, box, exclude);
	
	// -1 0 1;
	__syncthreads();
	if (threadIdx.x == 0)
		{
		my_j++;
		my_j = wrap_bin_i(my_j, bins.My);
		}
	__syncthreads();
	n_neigh = handle_bin(my_i, my_j, my_k, my_pidx, px, py, pz, n_neigh, pdata, bins, nlist, r_maxsq, box, exclude);
	
	// -1 0 0
	__syncthreads();
	if (threadIdx.x == 0)
		{
		my_k--;
		my_k = wrap_bin_i(my_k, bins.Mz);
		}
	__syncthreads();
	n_neigh = handle_bin(my_i, my_j, my_k, my_pidx, px, py, pz, n_neigh, pdata, bins, nlist, r_maxsq, box, exclude);
	
	// -1 0 -1
	__syncthreads();
	if (threadIdx.x == 0)
		{
		my_k--;
		my_k = wrap_bin_i(my_k, bins.Mz);
		}
	__syncthreads();
	n_neigh = handle_bin(my_i, my_j, my_k, my_pidx, px, py, pz, n_neigh, pdata, bins, nlist, r_maxsq, box, exclude);
	
	// -1 1 -1
	__syncthreads();
	if (threadIdx.x == 0)
		{
		my_j++;
		my_j = wrap_bin_i(my_j, bins.My);
		}
	__syncthreads();
	n_neigh = handle_bin(my_i, my_j, my_k, my_pidx, px, py, pz, n_neigh, pdata, bins, nlist, r_maxsq, box, exclude);
	
	// -1 1 0
	__syncthreads();
	if (threadIdx.x == 0)
		{
		my_k++;
		my_k = wrap_bin_i(my_k, bins.Mz);
		}
	__syncthreads();
	n_neigh = handle_bin(my_i, my_j, my_k, my_pidx, px, py, pz, n_neigh, pdata, bins, nlist, r_maxsq, box, exclude);

	// -1 1 1
	__syncthreads();
	if (threadIdx.x == 0)
		{
		my_k++;
		my_k = wrap_bin_i(my_k, bins.Mz);
		}
	__syncthreads();
	n_neigh = handle_bin(my_i, my_j, my_k, my_pidx, px, py, pz, n_neigh, pdata, bins, nlist, r_maxsq, box, exclude);
		
	// 0 1 1
	__syncthreads();
	if (threadIdx.x == 0)
		{
		my_i++;
		my_i = wrap_bin_i(my_i, bins.Mx);
		}
	__syncthreads();
	n_neigh = handle_bin(my_i, my_j, my_k, my_pidx, px, py, pz, n_neigh, pdata, bins, nlist, r_maxsq, box, exclude);
	
	// 0 1 0
	__syncthreads();
	if (threadIdx.x == 0)
		{
		my_k--;
		my_k = wrap_bin_i(my_k, bins.Mz);
		}
	__syncthreads();
	n_neigh = handle_bin(my_i, my_j, my_k, my_pidx, px, py, pz, n_neigh, pdata, bins, nlist, r_maxsq, box, exclude);
	
	// 0 1 -1
	__syncthreads();
	if (threadIdx.x == 0)
		{
		my_k--;
		my_k = wrap_bin_i(my_k, bins.Mz);
		}
	__syncthreads();
	n_neigh = handle_bin(my_i, my_j, my_k, my_pidx, px, py, pz, n_neigh, pdata, bins, nlist, r_maxsq, box, exclude);
	
	// 0 0 -1
	__syncthreads();
	if (threadIdx.x == 0)
		{
		my_j--;
		my_j = wrap_bin_i(my_j, bins.My);
		}
	__syncthreads();
	n_neigh = handle_bin(my_i, my_j, my_k, my_pidx, px, py, pz, n_neigh, pdata, bins, nlist, r_maxsq, box, exclude);

	// 0 0 0
	__syncthreads();
	if (threadIdx.x == 0)
		{
		my_k++;
		my_k = wrap_bin_i(my_k, bins.Mz);
		}
	__syncthreads();
	n_neigh = handle_bin(my_i, my_j, my_k, my_pidx, px, py, pz, n_neigh, pdata, bins, nlist, r_maxsq, box, exclude);
	
	// 0 0 1
	__syncthreads();
	if (threadIdx.x == 0)
		{
		my_k++;
		my_k = wrap_bin_i(my_k, bins.Mz);
		}
	__syncthreads();
	n_neigh = handle_bin(my_i, my_j, my_k, my_pidx, px, py, pz, n_neigh, pdata, bins, nlist, r_maxsq, box, exclude);

	// 0 -1 1
	__syncthreads();
	if (threadIdx.x == 0)
		{
		my_j--;
		my_j = wrap_bin_i(my_j, bins.My);
		}
	__syncthreads();
	n_neigh = handle_bin(my_i, my_j, my_k, my_pidx, px, py, pz, n_neigh, pdata, bins, nlist, r_maxsq, box, exclude);
	
	// 0 -1 0
	__syncthreads();
	if (threadIdx.x == 0)
		{
		my_k--;
		my_k = wrap_bin_i(my_k, bins.Mz);
		}
	__syncthreads();
	n_neigh = handle_bin(my_i, my_j, my_k, my_pidx, px, py, pz, n_neigh, pdata, bins, nlist, r_maxsq, box, exclude);
	
	// 0 -1 -1
	__syncthreads();
	if (threadIdx.x == 0)
		{
		my_k--;
		my_k = wrap_bin_i(my_k, bins.Mz);
		}
	__syncthreads();
	n_neigh = handle_bin(my_i, my_j, my_k, my_pidx, px, py, pz, n_neigh, pdata, bins, nlist, r_maxsq, box, exclude);
	
	// 1 -1 -1
	__syncthreads();
	if (threadIdx.x == 0)
		{
		my_i++;
		my_i = wrap_bin_i(my_i, bins.Mx);
		}
	__syncthreads();
	n_neigh = handle_bin(my_i, my_j, my_k, my_pidx, px, py, pz, n_neigh, pdata, bins, nlist, r_maxsq, box, exclude);
	
	// 1 1 0
	__syncthreads();
	if (threadIdx.x == 0)
		{
		my_k++;
		my_k = wrap_bin_i(my_k, bins.Mz);
		}
	__syncthreads();
	n_neigh = handle_bin(my_i, my_j, my_k, my_pidx, px, py, pz, n_neigh, pdata, bins, nlist, r_maxsq, box, exclude);
	
	// 1 -1 1
	__syncthreads();
	if (threadIdx.x == 0)
		{
		my_k++;
		my_k = wrap_bin_i(my_k, bins.Mz);
		}
	__syncthreads();
	n_neigh = handle_bin(my_i, my_j, my_k, my_pidx, px, py, pz, n_neigh, pdata, bins, nlist, r_maxsq, box, exclude);
	
	// 1 0 1
	__syncthreads();
	if (threadIdx.x == 0)
		{
		my_j++;
		my_j = wrap_bin_i(my_j, bins.My);
		}
	__syncthreads();
	n_neigh = handle_bin(my_i, my_j, my_k, my_pidx, px, py, pz, n_neigh, pdata, bins, nlist, r_maxsq, box, exclude);
	
	// 1 0 0
	__syncthreads();
	if (threadIdx.x == 0)
		{
		my_k--;
		my_k = wrap_bin_i(my_k, bins.Mz);
		}
	__syncthreads();
	n_neigh = handle_bin(my_i, my_j, my_k, my_pidx, px, py, pz, n_neigh, pdata, bins, nlist, r_maxsq, box, exclude);
	
	// 1 0 -1
	__syncthreads();
	if (threadIdx.x == 0)
		{
		my_k--;
		my_k = wrap_bin_i(my_k, bins.Mz);
		}
	__syncthreads();
	n_neigh = handle_bin(my_i, my_j, my_k, my_pidx, px, py, pz, n_neigh, pdata, bins, nlist, r_maxsq, box, exclude);
	
	// 1 1 -1
	__syncthreads();
	if (threadIdx.x == 0)
		{
		my_j++;
		my_j = wrap_bin_i(my_j, bins.My);
		}
	__syncthreads();
	n_neigh = handle_bin(my_i, my_j, my_k, my_pidx, px, py, pz, n_neigh, pdata, bins, nlist, r_maxsq, box, exclude);
	
	// 1 1 0
	__syncthreads();
	if (threadIdx.x == 0)
		{
		my_k++;
		my_k = wrap_bin_i(my_k, bins.Mz);
		}
	__syncthreads();
	n_neigh = handle_bin(my_i, my_j, my_k, my_pidx, px, py, pz, n_neigh, pdata, bins, nlist, r_maxsq, box, exclude);
	
	// 1 1 1
	__syncthreads();
	if (threadIdx.x == 0)
		{
		my_k++;
		my_k = wrap_bin_i(my_k, bins.Mz);
		}
	__syncthreads();
	n_neigh = handle_bin(my_i, my_j, my_k, my_pidx, px, py, pz, n_neigh, pdata, bins, nlist, r_maxsq, box, exclude);

	// now, at the very end, we need to write out the number of particles in each list
	if (my_pidx != EMPTY_BIN)
		{
		nlist.list[my_pidx] = n_neigh;
		nlist.last_updated_pos[my_pidx] = pdata.pos[my_pidx];
		}
	}*/


// non-unrolled version 
/*extern "C" __global__ void updateFromBins(gpu_pdata_arrays pdata, gpu_bin_array bins, gpu_nlist_array nlist, float r_maxsq, gpu_boxsize box)
	{
	// each block is going to compute the neighborlist for all of the particles in blockIdx.x
	int my_bin = blockIdx.x;

	// each thread will determine the neighborlist of a single particle: threadIdx.x in the current bin
	int n_neigh = 0;	// count number of neighbors found so far
	
	// we will need to loop over all neighboring bins. In order to do that, we need to know what bin we are actually in!
	// this could be a messy bunch of modulus operations, so we just read it out of an array that has been pre-computed for us :)
	uint4 coords = bins.bin_coord[my_bin];
	int my_i = coords.x;
	int my_j = coords.y;
	int my_k = coords.z;

	// pull in the particle of this thread that will be compared to everything else
	int my_pidx = bins.idxlist[my_bin*bins.Nmax + threadIdx.x];
	
	float4 my_pos = tex1Dfetch(pdata_pos_tex, my_pidx);
	float px = my_pos.x;
	float py = my_pos.y;
	float pz = my_pos.z;
	
	//uint4 exclude = tex1Dfetch(nlist_exclude_tex, my_pidx);

	// the shared data will hold the particle positions of all the particles in the neighboring bins as well as the 
	// particle index of the particle (stored as an int to be accessed with __float_as_int)
	// px,py,pz will have to be checked vs all of these
	extern __shared__ float bin_sdata[];

	// loop through the 27 neighboring bins
	for (int cur_i = int(my_i) - 1; cur_i <= int(my_i) + 1; cur_i++)
		{
		for (int cur_j = int(my_j) - 1; cur_j <= int(my_j) + 1; cur_j++)
			{
			for (int cur_k = int(my_k) - 1; cur_k <= int(my_k) + 1; cur_k++)
				{
				// apply boundary conditions to the current bin
				int a = cur_i;
				if (a < 0) 
					a += bins.Mx;
				if (a >= bins.Mx)
					a -= bins.Mx;

				int b = cur_j;
				if (b < 0) 
					b += bins.My;
				if (b >= bins.My)
					b -= bins.My;

				int c = cur_k;
				if (c < 0) 
					c += bins.Mz;
				if (c >= bins.Mz)
					c -= bins.Mz;
					
				//n_neigh = handle_bin(a, b, c, my_pidx, px, py, pz, n_neigh, pdata, bins, nlist, r_maxsq, box);
			
				// now: we finally know the current bin to compare to
				int neigh_bin = a*bins.Mz*bins.My + b*bins.Mz + c;
				int neigh_pidx = bins.idxlist[neigh_bin*bins.Nmax + threadIdx.x];
				
				// read in the neighbor position
				float4 neigh_pos = tex1Dfetch(pdata_pos_tex, neigh_pidx);
				// put it into shared memory
				
				__syncthreads();
				
				bin_sdata[threadIdx.x] = neigh_pos.x;
				bin_sdata[threadIdx.x + blockDim.x] = neigh_pos.y;
				bin_sdata[threadIdx.x + 2*blockDim.x] = neigh_pos.z;
				bin_sdata[threadIdx.x + 3*blockDim.x] = __int_as_float(neigh_pidx);
			
				__syncthreads();

				if (my_pidx != EMPTY_BIN)
					{
				// now, we are set to loop through the array
				for (int cur_offset = 0; cur_offset < blockDim.x; cur_offset++)
					{
					int cur_neigh = __float_as_int(bin_sdata[cur_offset + 3*blockDim.x]);
					if (cur_neigh == EMPTY_BIN)
						break;
					
					float dx = px - bin_sdata[cur_offset];
					dx = dx - box.Lx * rintf(dx * box.Lxinv);
            
					float dy = py - bin_sdata[cur_offset + blockDim.x];
					dy = dy - box.Ly * rintf(dy * box.Lyinv);

					float dz = pz - bin_sdata[cur_offset + 2*blockDim.x];
					dz = dz - box.Lz * rintf(dz * box.Lzinv);

					float dr = dx*dx + dy*dy + dz*dz;
					
					//int not_excluded = (exclude.x != cur_neigh) & (exclude.y != cur_neigh) & (exclude.z != cur_neigh) & (exclude.w != cur_neigh);
					if (dr < r_maxsq && (my_pidx != cur_neigh))
						{
						nlist.list[my_pidx + (1 + n_neigh)*nlist.pitch] = cur_neigh;
						n_neigh++;
						}
					}
					}
				}
			}
		}
		
	if (my_pidx != EMPTY_BIN)
		{
		// now, at the very end, we need to write out the number of particles in each list
		nlist.list[my_pidx] = n_neigh;
		nlist.last_updated_pos[my_pidx] = pdata.pos[my_pidx];
		}
	}*/

// non-unrolled version (with idx full) 
/*__global__ void updateFromBins(gpu_pdata_arrays pdata, gpu_bin_array bins, gpu_nlist_array nlist, float r_maxsq, gpu_boxsize box)
	{
	// each block is going to compute the neighborlist for all of the particles in blockIdx.x
	int my_bin = blockIdx.x;

	// each thread will determine the neighborlist of a single particle: threadIdx.x in the current bin
	int n_neigh = 0;	// count number of neighbors found so far
	
	// pull in the particle of this thread that will be compared to everything else
	int my_pidx = bins.idxlist[my_bin*bins.Nmax + threadIdx.x];
	
	float4 my_pos = tex1Dfetch(pdata_pos_tex, my_pidx);
	float px = my_pos.x;
	float py = my_pos.y;
	float pz = my_pos.z;

	// the shared data will hold the particle positions of all the particles in the neighboring bins as well as the 
	// particle index of the particle (stored as an int to be accessed with __float_as_int)
	// px,py,pz will have to be checked vs all of these
	extern __shared__ float bin_sdata[];

	// loop through the 27 neighboring bins
	for (int i = 0; i < 27; i++)
		{
		int neigh_pidx = bins.idxlist_full[(my_bin*27 + i)*bins.Nmax + threadIdx.x];
		
		// read in the neighbor position
		float4 neigh_pos = tex1Dfetch(pdata_pos_tex, neigh_pidx);
		// put it into shared memory
				
		__syncthreads();
				
		bin_sdata[threadIdx.x] = neigh_pos.x;
		bin_sdata[threadIdx.x + blockDim.x] = neigh_pos.y;
		bin_sdata[threadIdx.x + 2*blockDim.x] = neigh_pos.z;
		bin_sdata[threadIdx.x + 3*blockDim.x] = __int_as_float(neigh_pidx);
			
		__syncthreads();
	
		// because of the way the data is packed, we can break out of this loop if bin_sdata[3*blockDim.x] is EMPTY, as all
		// others after it will be empty too
		if (__float_as_int(bin_sdata[3*blockDim.x]) == EMPTY_BIN)
			break;

		// try a simple, divergent warp method
		if (my_pidx != EMPTY_BIN)
			{

		// now, we are set to loop through the array
		for (int cur_offset = 0; cur_offset < blockDim.x; cur_offset++)
			{
			float dx = px - bin_sdata[cur_offset];
			dx = dx - box.Lx * rintf(dx * box.Lxinv);
            
			float dy = py - bin_sdata[cur_offset + blockDim.x];
			dy = dy - box.Ly * rintf(dy * box.Lyinv);

			float dz = pz - bin_sdata[cur_offset + 2*blockDim.x];
			dz = dz - box.Lz * rintf(dz * box.Lzinv);

			int cur_neigh = __float_as_int(bin_sdata[cur_offset + 3*blockDim.x]);
			
			float dr = dx*dx + dy*dy + dz*dz;
					
			if ((my_pidx != EMPTY_BIN) && dr < r_maxsq && (cur_neigh != EMPTY_BIN) && (my_pidx != cur_neigh))
				{
				nlist.list[my_pidx + (1 + n_neigh)*nlist.pitch] = cur_neigh;
				n_neigh++;
				}
			}

			}
		}
	
	// now, at the very end, we need to write out the number of particles in each list
	nlist.list[my_pidx] = n_neigh;
	float4 my_pos2;
	my_pos2.x = px;
	my_pos2.y = py;
	my_pos2.z = pz;

	if (my_pidx < pdata.N)
		nlist.last_updated_pos[my_pidx] = my_pos2;
	}*/

/*void gpu_nlist_binned(gpu_pdata_arrays *pdata, gpu_boxsize *box, gpu_bin_data *bins, gpu_nlist_data *nlist, float r_maxsq, int curNmax)
	{
	assert(bins);
    assert(pdata);
    assert(nlist);

    // setup the grid to run the kernel
    int M = curNmax;
	int N = bins->d_array.Mx * bins->d_array.My * bins->d_array.Mz;
    dim3 grid(N, 1, 1);
    dim3 threads(M, 1, 1);

    // checks on kernel configuration arguments
	assert(M >= 1);
	assert(N <= 65535);

    // run the kernel
    updateFromBins<<< grid, threads, M*sizeof(float)*4 >>>(*pdata, bins->d_array, nlist->d_array, r_maxsq, *box);
    CUT_CHECK_ERROR("Kernel execution failed");
    }*/

/////////////////////// OLD METHOD ABOVE
/////////////////////// NEW METHOD BELOW


extern "C" __global__ void updateFromBins_new(gpu_pdata_arrays pdata, gpu_bin_array bins, gpu_nlist_array nlist, float r_maxsq, unsigned int actual_Nmax, gpu_boxsize box)
	{
	// each thread is going to compute the neighbor list for a single particle
	int my_pidx = blockDim.x * blockIdx.x + threadIdx.x;
	
	// quit early if we are past the end of the array
	if (my_pidx >= pdata.N)
		return;
	
	// first, determine which bin this particle belongs to
	float4 my_pos = tex1Dfetch(pdata_pos_tex, my_pidx);
	
	// make even bin dimensions
	float binx = (box.Lx) / float(bins.Mx);
	float biny = (box.Ly) / float(bins.My);
	float binz = (box.Lz) / float(bins.Mz);

	// precompute scale factors to eliminate division in inner loop
	float scalex = 1.0f / binx;
	float scaley = 1.0f / biny;
	float scalez = 1.0f / binz;
	
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
			
	int my_bin = ib*(bins.Mz*bins.My) + jb * bins.Mz + kb;	

	// each thread will determine the neighborlist of a single particle
	int n_neigh = 0;	// count number of neighbors found so far
	
	// we will need to loop over all neighboring bins. In order to do that, we need to know what bin we are actually in!
	// this could be a messy bunch of modulus operations, so we just read it out of an array that has been pre-computed for us :)
	uint4 coords = tex1Dfetch(nlist_bincoord_tex, my_bin);
	int my_i = coords.x;
	int my_j = coords.y;
	int my_k = coords.z;
	
	// loop through the 27 neighboring bins
	for (int cur_i = int(my_i) - 1; cur_i <= int(my_i) + 1; cur_i++)
		{
		for (int cur_j = int(my_j) - 1; cur_j <= int(my_j) + 1; cur_j++)
			{
			for (int cur_k = int(my_k) - 1; cur_k <= int(my_k) + 1; cur_k++)
				{
				// apply boundary conditions to the current bin
				int a = cur_i;
				if (a < 0) 
					a += bins.Mx;
				if (a >= bins.Mx)
					a -= bins.Mx;

				int b = cur_j;
				if (b < 0) 
					b += bins.My;
				if (b >= bins.My)
					b -= bins.My;

				int c = cur_k;
				if (c < 0) 
					c += bins.Mz;
				if (c >= bins.Mz)
					c -= bins.Mz;
					
				// now: we finally know the current bin to compare to
				int neigh_bin = a*bins.Mz*bins.My + b*bins.Mz + c;
				
				// now, we are set to loop through the array
				for (int cur_offset = 0; cur_offset < actual_Nmax; cur_offset++)
					{
					unsigned int cur_neigh = tex2D(nlist_idxlist_tex, cur_offset, neigh_bin);
					
					if (cur_neigh != EMPTY_BIN)
						{
						float4 neigh_pos = tex1Dfetch(pdata_pos_tex, cur_neigh);
					
						float dx = my_pos.x - neigh_pos.x;
						dx = dx - box.Lx * rintf(dx * box.Lxinv);
	
						float dy = my_pos.y - neigh_pos.y;
						dy = dy - box.Ly * rintf(dy * box.Lyinv);
	
						float dz = my_pos.z - neigh_pos.z;
						dz = dz - box.Lz * rintf(dz * box.Lzinv);
	
						float dr = dx*dx + dy*dy + dz*dz;
						
						if (dr < r_maxsq && (my_pidx != cur_neigh))
							{
							nlist.list[my_pidx + (1 + n_neigh)*nlist.pitch] = cur_neigh;
							n_neigh++;
							}
						}
					}
				}
			}
		}
		
	nlist.list[my_pidx] = n_neigh;
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
	cudaError_t error = cudaBindTexture(0, pdata_pos_tex, pdata->pos, sizeof(float4) * pdata->N);
	if (error != cudaSuccess)
		return error;

	error = cudaBindTexture(0, nlist_bincoord_tex, bins->bin_coord, sizeof(uint4)*bins->Mx*bins->My*bins->Mz);
	if (error != cudaSuccess)
		return error;

	nlist_idxlist_tex.normalized = false;
	nlist_idxlist_tex.filterMode = cudaFilterModePoint;
	error = cudaBindTextureToArray(nlist_idxlist_tex, bins->idxlist_array);
	if (error != cudaSuccess)
		return error;

	// run the kernel
	updateFromBins_new<<< grid, threads>>>(*pdata, *bins, *nlist, r_maxsq, curNmax, *box);
	
	#ifdef NDEBUG
	return cudaSuccess;
	#else
	cudaThreadSynchronize();
	return cudaGetLastError();
	#endif
	}

// vim:syntax=cpp
