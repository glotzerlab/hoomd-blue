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
/*! The list is arranged as height rows of padded_N columns with the given pitch.
	Each column i is the neighborlist for particle i. 
	The first row lists the number of elements in the column.
	Each column has a fixed height \c height.
	\todo selecting of nlist data for exclusions texture ????
*/
struct gpu_nlist_array
	{
	unsigned int *list;
	unsigned int height;
	unsigned int pitch;
	float4 *last_updated_pos;
	int *needs_update;
	
	uint4 *exclusions;
	};

//! A larger strucutre that stores both the gpu data and the cpu mirror
/*! \todo document me
*/
struct gpu_nlist_data
	{
	gpu_nlist_array d_array;
	gpu_nlist_array h_array;
	unsigned int N;
	};

struct gpu_bin_array
        {
        // these are 4D arrays with indices i,j,k,n. i,j,k index the bin and each goes from 0 to Mx-1,My-1,Mz-1 respectively.
        // index into the data with idxdata[i*Nmax*Mz*My + j*Nmax*Mz + k*Nmax  + n]
		// n goes from 0 to Nmax - 1.
        unsigned int Mx,My,Mz,Nmax,Nparticles;
		
        // idxdata stores the index of the particles in the bins
		unsigned int *idxlist;
		cudaArray *idxlist_array;
		
		uint4 *bin_coord;	// holds the i,j,k coordinates of the bins indexed by i*Mz*My + j*Mz + k.
		};

struct gpu_bin_data
	{
	gpu_bin_array d_array;
	gpu_bin_array h_array;
	};

//! Allocates memory
void gpu_alloc_nlist_data(gpu_nlist_data *nlist, unsigned int N, unsigned int height);
//! Free memory
void gpu_free_nlist_data(gpu_nlist_data *nlist);
//! copy data from the host (h_array) to the device (d_array)
void gpu_copy_nlist_data_htod(gpu_nlist_data *nlist);
//! copy data from the device (d_array) to the host (h_array)
void gpu_copy_nlist_data_dtoh(gpu_nlist_data *nlist);

//! copy exclude data from the host (h_array) to the device (d_array)
void gpu_copy_exclude_data_htod(gpu_nlist_data *nlist);


//! Allocates memory
void gpu_alloc_bin_data(gpu_bin_data *bins, unsigned int Mx, unsigned int My, unsigned int Mz, unsigned int Nmax);
//! Frees memory
void gpu_free_bin_data(gpu_bin_data *bins);
//! copy data from the host (h_array) to the device (d_array)
void gpu_copy_bin_data_htod(gpu_bin_data *bins);
//! copy data from the device (d_array) to the host (h_array)
void gpu_copy_bin_data_dtoh(gpu_bin_data *bins);

//! Generate a test pattern in the data on the GPU (for unit testing)
void gpu_generate_nlist_data_test(gpu_nlist_data *nlist);

//! Generate the neighborlist (N^2 algorithm)
void gpu_nlist_nsq(gpu_pdata_arrays *pdata, gpu_boxsize *box, gpu_nlist_data *nlist, float r_maxsq);
//! Generate the neighborlist from bins (O(N) algorithm)
void gpu_nlist_binned(gpu_pdata_arrays *pdata, gpu_boxsize *box, gpu_bin_data *bins, gpu_nlist_data *nlist, float r_maxsq, int curNmax);
	
//! Check if the neighborlist needs updating
int gpu_nlist_needs_update_check(gpu_pdata_arrays *pdata, gpu_boxsize *box, gpu_nlist_data *nlist, float r_buffsq);
}

#endif

