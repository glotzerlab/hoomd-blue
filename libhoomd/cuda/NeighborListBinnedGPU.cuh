/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
// Maintainer: joaander

#ifndef _NEIGHBORLISTBINNEDGPU_CUH_
#define _NEIGHBORLISTBINNEDGPU_CUH_

#include <cuda_runtime.h>

#include "NeighborList.cuh"

/*! \file NeighborListBinnedGPU.cuh
    \brief Declares data structures and methods used by BinnedNeighborListGPU
*/

//! Structure of arrays storing the bins particles are placed in on the GPU
/*! This structure is in a current state of flux. Consider it documented as being
    poorly documented :)

    These are 4D arrays with indices i,j,k,n. i,j,k index the bin and each goes from 0 to Mx-1,My-1,Mz-1 respectively.
    Index into the data with idxdata[i*Nmax*Mz*My + j*Nmax*Mz + k*Nmax  + n] where n goes from 0 to Nmax - 1.

    \ingroup gpu_data_structs
*/
struct gpu_bin_array
    {
    unsigned int Mx;    //!< X-dimension of the cell grid
    unsigned int My;    //!< Y-dimension of the cell grid
    unsigned int Mz;    //!< Z-dimension of the cell grid
    unsigned int Nmax;  //!< Maximum number of particles each cell can hold
    unsigned int Nparticles;        //!< Total number of particles binned
    unsigned int coord_idxlist_width;   //!< Width of the coord_idxlist data
    
    unsigned int *bin_size; //!< \a nbins length array listing the number of particles in each bin
    unsigned int *idxlist;  //!< \a Mx x \a My x \a Mz x \a Nmax 4D array holding the indices of the particles in each cell
    cudaArray *idxlist_array;   //!< An array memory copy of \a idxlist for 2D texturing
    
    float4 *coord_idxlist;  //!< \a Mx x \a My x \a Mz x \a Nmax 4D array holding the positions and indices of the particles in each cell (x,y,z are position and w holds the index)
    cudaArray *coord_idxlist_array; //!< An array memory copy of \a coord_idxlist for 2D texturing
    
    cudaArray *bin_adj_array;   //!< bin_adj_array holds the neighboring bins of each bin in memory (x = idx (0:26), y = neighboring bin memory location)
    };


//! Take the idxlist and generate coord_idxlist
cudaError_t gpu_nlist_idxlist2coord(gpu_pdata_arrays *pdata, gpu_bin_array *bins, int curNmax, int block_size);

//! Kernel driver for GPU computation in BinnedNeighborListGPU
cudaError_t gpu_compute_nlist_binned(const gpu_nlist_array &nlist,
                                     const gpu_pdata_arrays &pdata,
                                     const gpu_boxsize &box,
                                     const gpu_bin_array &bins,
                                     float r_maxsq,
                                     int curNmax,
                                     int block_size,
                                     bool ulf_workaround,
                                     bool exclude_same_body);

#endif

