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

#ifndef _CUDA_NLIST_H_
#define _CUDA_NLIST_H_

#include <stdio.h>
#include <cuda_runtime_api.h>

#include "gpu_pdata.h"

/*! \file gpu_nlist.h
	\brief Declares structures and functions for working with neighbor lists on the GPU
	\ingroup cuda_code
*/

extern "C" {

//! Structure of arrays of the neighborlist data as it resides on the GPU
/*! The list is arranged as height rows of \a N columns with the given pitch.
	Each column i is the neighborlist for particle i. 
	\a n_neigh lists the number of neighbors in each column
	Each column has a fixed height \c height.
*/
struct gpu_nlist_array
	{
	unsigned int *n_neigh;
	unsigned int *list;
	unsigned int height;
	unsigned int pitch;
	float4 *last_updated_pos;
	int *needs_update;
	int *overflow;
	
	uint4 *exclusions;
	};

struct gpu_bin_array
        {
        // these are 4D arrays with indices i,j,k,n. i,j,k index the bin and each goes from 0 to Mx-1,My-1,Mz-1 respectively.
        // index into the data with idxdata[i*Nmax*Mz*My + j*Nmax*Mz + k*Nmax  + n]
		// n goes from 0 to Nmax - 1.
        unsigned int Mx,My,Mz,Nmax,Nparticles,coord_idxlist_width;
		
        // idxdata stores the index of the particles in the bins
		unsigned int *idxlist;
		cudaArray *idxlist_array;
		
		// coord_idxlist stores the coordinates and the index of the particles in the bins
		float4 *coord_idxlist;
		cudaArray *coord_idxlist_array;
		
		// mem_location maps a bin index to the actual bin location that should be read from memory
		unsigned int *mem_location;
		
		// bin_adj_array holds the neighboring bins of each bin in memory (x = idx (0:26), y = bin)
		cudaArray *bin_adj_array;
		};

//! Generate the neighborlist (N^2 algorithm)
cudaError_t gpu_nlist_nsq(gpu_pdata_arrays *pdata, gpu_boxsize *box, gpu_nlist_array *nlist, float r_maxsq);

//! Generate the neighborlist from bins (O(N) algorithm)
cudaError_t gpu_nlist_binned(gpu_pdata_arrays *pdata, gpu_boxsize *box, gpu_bin_array *bins, gpu_nlist_array *nlist, float r_maxsq, int curNmax, int block_size);

//! Take the idxlist and generate coord_idxlist
cudaError_t gpu_nlist_idxlist2coord(gpu_pdata_arrays *pdata, gpu_bin_array *bins, int curNmax, int block_size);
	
//! Check if the neighborlist needs updating
cudaError_t gpu_nlist_needs_update_check(gpu_pdata_arrays *pdata, gpu_boxsize *box, gpu_nlist_array *nlist, float r_buffsq, int *result);
}

#endif

