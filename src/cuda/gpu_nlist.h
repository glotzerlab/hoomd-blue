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
#include <cuda_runtime.h>

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
	unsigned int *n_neigh;	//!< n_neigh[i] is the number of neighbors of particle with index i
	unsigned int *list;		//!< list[i*pitch + j] is the index of the j'th neighbor of particle i
	unsigned int height;	//!< Maximum number of neighbors that can be stored for any particle
	unsigned int pitch;		//!< width of the list in elements
	float4 *last_updated_pos;	//!< Holds the positions of the particles as they were when the list was last updated
	int *needs_update;		//!< Flag set to 1 when the neighbor list needs to be updated
	int *overflow;			//!< Flag set to 1 when the neighbor list overflows and needs to be expanded
	
	uint4 *exclusions;		//!< exclusions[i] lists all the particles that are to be excluded from being neighbors with [i] by index
	};

//! Structure of arrays storing the bins particles are placed in on the GPU
/*! This structure is in a current state of flux. Consider it documented as being
	poorly documented :)
	\todo update documentation
*/
struct gpu_bin_array
        {
        // these are 4D arrays with indices i,j,k,n. i,j,k index the bin and each goes from 0 to Mx-1,My-1,Mz-1 respectively.
        // index into the data with idxdata[i*Nmax*Mz*My + j*Nmax*Mz + k*Nmax  + n]
		// n goes from 0 to Nmax - 1.
        unsigned int Mx;	//!< X-dimension of the cell grid
        unsigned int My;	//!< Y-dimension of the cell grid
        unsigned int Mz;	//!< Z-dimension of the cell grid
        unsigned int Nmax;	//!< Maximum number of particles each cell can hold
        unsigned int Nparticles;		//!< Total number of particles binned
        unsigned int coord_idxlist_width;	//!< Width of the coord_idxlist data
		
        unsigned int *idxlist;	//!< \a Mx x \a My x \a Mz x \a Nmax 4D array holding the indices of the particles in each cell
		cudaArray *idxlist_array;	//!< An array memory copy of \a idxlist for 2D texturing
		
		float4 *coord_idxlist;	//!< \a Mx x \a My x \a Mz x \a Nmax 4D array holding the positions and indices of the particles in each cell (x,y,z are position and w holds the index)
		cudaArray *coord_idxlist_array;	//!< An array memory copy of \a coord_idxlist for 2D texturing
		
		unsigned int *mem_location;		//!< Maps a bin index i*Nmax*Mz*My + j*Nmax*Mz + k*Nmax to the actual location in memory where it is stored
		
		cudaArray *bin_adj_array;	//!< bin_adj_array holds the neighboring bins of each bin in memory (x = idx (0:26), y = neighboring bin memory location)
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

