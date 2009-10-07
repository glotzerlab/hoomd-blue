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

#include "NeighborListBinnedGPU.cuh"
#include "gpu_settings.h"

#include <stdio.h>

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file NeighborListBinnedGPU.cuh
    \brief Defines GPU code and data structure methods used in BinnedNeighborListGPU
*/

//! sentinel value signifying an empty slot in a bin
#define EMPTY_BIN 0xffffffff

//! Texture for reading pdata pos
texture<float4, 1, cudaReadModeElementType> pdata_pos_tex;
//! Texture for reading the idxlist from the bin arrays
texture<unsigned int, 2, cudaReadModeElementType> bin_idxlist_tex;
//! Texture for reading the bin_size array
texture<unsigned int, 1, cudaReadModeElementType> bin_size_tex;

//! Transposes the bin idx list on the GPU
/*! \param pdata Particle data to read positions from
    \param bins Bin data for the particles

    This kernel reads in bins.idxlist (generated on the host) and writes out a transposed
    array in binds.coord_idxlist. The transposed array also includes the particle position
    along with the index for quick reading and use in the neighbor list build kernel.
*/
extern "C" __global__ void gpu_nlist_idxlist2coord_kernel(gpu_pdata_arrays pdata, gpu_bin_array bins)
    {
    // each thread writes the coord_idxlist of a single bin
    unsigned int binidx = threadIdx.x + blockDim.x*blockIdx.x;
    
    unsigned int nbins = bins.Mx*bins.My*bins.Mz;
    
    // return if we are past the array bouds
    if (binidx >= nbins)
        return;
        
    // read the particle idx
    unsigned int pidx = tex2D(bin_idxlist_tex, blockIdx.y, binidx);
    
    // if the particle idx is valid, read in the position
    float4 coord_idx = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (pidx != EMPTY_BIN)
        coord_idx = tex1Dfetch(pdata_pos_tex, pidx);
        
    // add the index to the coord_idx
    coord_idx.w = __int_as_float(pidx);
    
    // write it out to the coord_idxlist
    bins.coord_idxlist[bins.coord_idxlist_width * blockIdx.y + binidx] = coord_idx;
    }

//! Transposes the bin idx list on the GPU
/*! \param pdata Particle data to read positions from
    \param bins Bin data for the particles
    \param curNmax Maximum number of particles in any of the bins
    \param block_size Block size to run on the device

    see gpu_nlist_idxlist2coord_kernel for details
*/
cudaError_t gpu_nlist_idxlist2coord(gpu_pdata_arrays *pdata, gpu_bin_array *bins, int curNmax, int block_size)
    {
    assert(bins);
    assert(pdata);
    assert(block_size > 0);
    
    // setup the grid to run the kernel
    int nblocks_x = (int)ceil((double)(bins->Mx*bins->My*bins->Mz) / (double)block_size);
    
    dim3 grid(nblocks_x, curNmax, 1);
    dim3 threads(block_size, 1, 1);
    
    // bind the textures
    bin_idxlist_tex.normalized = false;
    bin_idxlist_tex.filterMode = cudaFilterModePoint;
    cudaError_t error = cudaBindTextureToArray(bin_idxlist_tex, bins->idxlist_array);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, pdata_pos_tex, pdata->pos, sizeof(float4)*pdata->N);
    if (error != cudaSuccess)
        return error;
        
    // run the kernel
    gpu_nlist_idxlist2coord_kernel<<< grid, threads>>>(*pdata, *bins);
    
    if (!g_gpu_error_checking)
        {
        return cudaSuccess;
        }
    else
        {
        cudaThreadSynchronize();
        return cudaGetLastError();
        }
    }

//! Texture for reading coord_idxlist from the binned particle data
texture<float4, 2, cudaReadModeElementType> nlist_coord_idxlist_tex;
//! Texture for reading the bins adjacent to a given bin
texture<unsigned int, 2, cudaReadModeElementType> bin_adj_tex;


//! Generates the neighbor list from the binned particles
/*! \param nlist Neighbor list to write out to
    \param pdata Particle data to generate the neighbor list for
    \param box Box dimensions for handling periodic boundary conditions
    \param bins The binned particles
    \param r_maxsq Precalculated value for r_max*r_max
    \param actual_Nmax Number of particles currently in the largest bin
    \param scalex Scale factor to convert particle position to bin i-coordinate
    \param scaley Scale factor to convert particle position to bin j-coordinate
    \param scalez Scale factor to convert particle position to bin k-coordinate

    This kernel runs one thread per particle: Thread \a i handles particle \a i.
    The bin of particle \a i is first determined. Neighboring bins are read from
    the bin_adj_tex. Partilces in each neighboring bin are read from the nlist_coord_idxlist_tex
    and compared to see if they are neighbors.

    This whole processes involves a lot of looping over and reading randomly from
    textures, but the 2D texture cache helps a lot and the process reaches near
    peak bandwidth on the device. It seems more wasteful than a one block per cell
    method, but actually is ~40% faster.
*/
template <bool ulf_workaround> 
__global__ void gpu_compute_nlist_binned_kernel(gpu_nlist_array nlist,
                                                float4 *d_pos,
                                                unsigned int local_beg,
                                                unsigned int local_num,
                                                gpu_boxsize box,
                                                gpu_bin_array bins,
                                                float r_maxsq,
                                                unsigned int actual_Nmax,
                                                float scalex,
                                                float scaley,
                                                float scalez)
    {
    // each thread is going to compute the neighbor list for a single particle
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int my_pidx = idx + local_beg;
    
    // quit early if we are past the end of the array
    if (idx >= local_num)
        return;
        
    // first, determine which bin this particle belongs to
    // MEM TRANSFER: 32 bytes
    float4 my_pos = d_pos[my_pidx];
    uint4 exclude = nlist.exclusions[my_pidx];
#if defined(LARGE_EXCLUSION_LIST)
    uint4 exclude2 = nlist.exclusions2[my_pidx];
    uint4 exclude3 = nlist.exclusions3[my_pidx];
    uint4 exclude4 = nlist.exclusions4[my_pidx];
#endif
    
    // FLOPS: 9
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
        
    // MEM TRANSFER: 4 bytes
    int my_bin = ib*(bins.Mz*bins.My) + jb * bins.Mz + kb;
    
    // each thread will determine the neighborlist of a single particle
    int n_neigh = 0;    // count number of neighbors found so far
    
    // loop over all adjacent bins
    for (unsigned int cur_adj = 0; cur_adj < 27; cur_adj++)
        {
        // MEM TRANSFER: 4 bytes
        int neigh_bin = tex2D(bin_adj_tex, my_bin, cur_adj);
        unsigned int size = tex1Dfetch(bin_size_tex, neigh_bin);
        
        // prefetch
        float4 next_neigh_blob = tex2D(nlist_coord_idxlist_tex, neigh_bin, 0);
        
        unsigned int loop_count = size;
        if (ulf_workaround)
            loop_count = actual_Nmax;
            
        // now, we are set to loop through the array
        for (int cur_offset = 0; cur_offset < loop_count; cur_offset++)
            {
            if (!ulf_workaround || cur_offset < size)
                {
                // MEM TRANSFER: 16 bytes
                float4 cur_neigh_blob = next_neigh_blob;
                // no guard branch needed since the texture will just clamp and we will never use the last read value
                next_neigh_blob = tex2D(nlist_coord_idxlist_tex, neigh_bin, cur_offset+1);
                
                float3 neigh_pos;
                neigh_pos.x = cur_neigh_blob.x;
                neigh_pos.y = cur_neigh_blob.y;
                neigh_pos.z = cur_neigh_blob.z;
                int cur_neigh = __float_as_int(cur_neigh_blob.w);
                
                // FLOPS: 15
                float dx = my_pos.x - neigh_pos.x;
                dx = dx - box.Lx * rintf(dx * box.Lxinv);
                
                float dy = my_pos.y - neigh_pos.y;
                dy = dy - box.Ly * rintf(dy * box.Lyinv);
                
                float dz = my_pos.z - neigh_pos.z;
                dz = dz - box.Lz * rintf(dz * box.Lzinv);
                
                // FLOPS: 5
                float dr = dx*dx + dy*dy + dz*dz;
                int not_excluded = (exclude.x != cur_neigh) & (exclude.y != cur_neigh) & (exclude.z != cur_neigh) & (exclude.w != cur_neigh);
#if defined(LARGE_EXCLUSION_LIST)
                not_excluded &= (exclude2.x != cur_neigh) & (exclude2.y != cur_neigh) & (exclude2.z != cur_neigh) & (exclude2.w != cur_neigh);
                not_excluded &= (exclude3.x != cur_neigh) & (exclude3.y != cur_neigh) & (exclude3.z != cur_neigh) & (exclude3.w != cur_neigh);
                not_excluded &= (exclude4.x != cur_neigh) & (exclude4.y != cur_neigh) & (exclude4.z != cur_neigh) & (exclude4.w != cur_neigh);
#endif
                
                // FLOPS: 1 / MEM TRANSFER total = N * estimated number of neighbors * 4
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
        
    // MEM TRANSFER 8 bytes
    nlist.n_neigh[my_pidx] = n_neigh;
    nlist.last_updated_pos[my_pidx] = my_pos;
    }

//! Generate the neighbor list on the GPU
/*! \param nlist Neighbor list to write out to
    \param pdata Particle data to generate the neighbor list for
    \param box Box dimensions for handling periodic boundary conditions
    \param bins The binned particles
    \param r_maxsq Precalculated value for r_max*r_max
    \param curNmax Number of particles currently in the largest bin
    \param block_size Block size to run the kernel on the device
    \param ulf_workaround Set to true to enable an attempted workaround for ULFs on compute 1.1 devices

    See updateFromBins_new for more information
*/
cudaError_t gpu_compute_nlist_binned(const gpu_nlist_array &nlist,
                                     const gpu_pdata_arrays &pdata,
                                     const gpu_boxsize &box,
                                     const gpu_bin_array &bins,
                                     float r_maxsq,
                                     int curNmax,
                                     int block_size,
                                     bool ulf_workaround)
    {
    assert(block_size > 0);
    
    // setup the grid to run the kernel
    int nblocks = (int)ceil((double)pdata.local_num/ (double)block_size);
    
    dim3 grid(nblocks, 1, 1);
    dim3 threads(block_size, 1, 1);
    
    // bind the textures
    nlist_coord_idxlist_tex.normalized = false;
    nlist_coord_idxlist_tex.filterMode = cudaFilterModePoint;
    cudaError_t error = cudaBindTextureToArray(nlist_coord_idxlist_tex, bins.coord_idxlist_array);
    if (error != cudaSuccess)
        return error;
        
    bin_adj_tex.normalized = false;
    bin_adj_tex.filterMode = cudaFilterModePoint;
    error = cudaBindTextureToArray(bin_adj_tex, bins.bin_adj_array);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, bin_size_tex, bins.bin_size, sizeof(unsigned int)*bins.Mx*bins.My*bins.Mz);
    if (error != cudaSuccess)
        return error;
        
    // zero the overflow check
    error = cudaMemset(nlist.overflow, 0, sizeof(int));
    if (error != cudaSuccess)
        return error;
        
    // make even bin dimensions
    float binx = (box.Lx) / float(bins.Mx);
    float biny = (box.Ly) / float(bins.My);
    float binz = (box.Lz) / float(bins.Mz);
    
    // precompute scale factors to eliminate division in inner loop
    float scalex = 1.0f / binx;
    float scaley = 1.0f / biny;
    float scalez = 1.0f / binz;
    
    // run the kernel
    if (ulf_workaround)
        gpu_compute_nlist_binned_kernel<true><<< grid, threads>>>(nlist, pdata.pos, pdata.local_beg, pdata.local_num, box, bins, r_maxsq, curNmax, scalex, scaley, scalez);
    else
        gpu_compute_nlist_binned_kernel<false><<< grid, threads>>>(nlist, pdata.pos, pdata.local_beg, pdata.local_num, box, bins, r_maxsq, curNmax, scalex, scaley, scalez);
        
    if (!g_gpu_error_checking)
        {
        return cudaSuccess;
        }
    else
        {
        cudaThreadSynchronize();
        return cudaGetLastError();
        }
    }

// vim:syntax=cpp

