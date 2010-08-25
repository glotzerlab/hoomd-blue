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

/*! \file NeighborListGPUBinned.cu
    \brief Defines GPU kernel code for neighbor list processing on the GPU
*/

#include "NeighborListGPUBinned.cuh"

#include "NeighborListGPU.cuh"
#include <stdio.h>

/*! \param d_result Device pointer to a single uint. Will be set to 1 if an update is needed
    \param d_last_pos Particle positions at the time the nlist was last updated
    \param d_pos Current particle positions
    \param N Number of particles
    \param box Box dimensions
    \param maxshiftsq The maximum drsq a particle can have before an update is needed
    
    gpu_nlist_needs_update_check_new_kernel() executes one thread per particle. Every particle's current position is
    compared to its last position. If the particle has moved a distance more than sqrt(\a maxshiftsq), then *d_result
    is set to 1. Consequently, d_result must be set to 0 prior to launching this kernel.
*/
__global__ void gpu_nlist_needs_update_check_new_kernel(unsigned int *d_result,
                                                        const float4 *d_last_pos,
                                                        const float4 *d_pos,
                                                        const unsigned int N,
                                                        const gpu_boxsize box,
                                                        const float maxshiftsq)
    {
    // each thread will compare vs it's old position to see if the list needs updating
    // if that is true, write a 1 to nlist_needs_updating
    // it is possible that writes will collide, but at least one will succeed and that is all that matters
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
        {
        float4 cur_pos = d_pos[idx];
        float4 last_pos = d_last_pos[idx];
        float dx = cur_pos.x - last_pos.x;
        float dy = cur_pos.y - last_pos.y;
        float dz = cur_pos.z - last_pos.z;

        dx = dx - box.Lx * rintf(dx * box.Lxinv);
        dy = dy - box.Ly * rintf(dy * box.Lyinv);
        dz = dz - box.Lz * rintf(dz * box.Lzinv);

        float drsq = dx*dx + dy*dy + dz*dz;
        
        if (drsq >= maxshiftsq)
            {
            *d_result = 1;
            }
        }
    }

cudaError_t gpu_nlist_needs_update_check_new(unsigned int *d_result,
                                             const float4 *d_last_pos,
                                             const float4 *d_pos,
                                             const unsigned int N,
                                             const gpu_boxsize& box,
                                             const float maxshiftsq)
    {
    int zero = 0;
    cudaMemcpy(d_result, &zero, sizeof(unsigned int), cudaMemcpyHostToDevice);
    
    unsigned int block_size = 128;
    int n_blocks = (int)ceil(float(N)/(float)block_size);
    gpu_nlist_needs_update_check_new_kernel<<<n_blocks, block_size>>>(d_result,
                                                                      d_last_pos,
                                                                      d_pos,
                                                                      N,
                                                                      box,
                                                                      maxshiftsq);
    
    return cudaSuccess;
    }

//! Number of elements of the exclusion list to process in each batch
const unsigned int FILTER_BATCH_SIZE = 4;

/*! \param d_n_neigh Number of neighbors for each particle (read/write)
    \param d_nlist Neighbor list for each particle (read/write)
    \param nli Indexer for indexing into d_nlist
    \param d_n_ex Number of exclusions for each particle
    \param d_ex_list List of exclusions for each particle
    \param exli Indexer for indexing into d_ex_list
    \param N Number of particles
    \param ex_start Start filtering the nlist from exclusion number \a ex_start
    
    gpu_nlist_filter_kernel() processes the neighbor list \a d_nlist and removes any entries that are excluded. To allow
    for an arbitrary large number of exclusions, these are processed in batch sizes of FILTER_BATCH_SIZE. The kernel
    must be called multiple times in order to fully remove all exclusions from the nlist. 
    
    \note The driver gpu_nlist_filter properly makes as many calls as are necessary, it only needs to be called once.
    
    \b Implementation
    
    One thread is run for each particle. Exclusions \a ex_start, \a ex_start + 1, ... are loaded in for that particle
    (or the thread returns if there are no exlusions past that point). The thread then loops over the neighbor list,
    comparing each entry to the list of exclusions. If the entry is not excluded, it is written back out. \a d_n_neigh
    is updated to reflect the current number of particles in the list at the end of the kernel call.
*/
__global__ void gpu_nlist_filter_kernel(unsigned int *d_n_neigh,
                                        unsigned int *d_nlist,
                                        const Index2D nli,
                                        const unsigned int *d_n_ex,
                                        const unsigned int *d_ex_list,
                                        const Index2D exli,
                                        const unsigned int N,
                                        const unsigned int ex_start)
    {
    // compute the particle index this thread operates on
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    // quit now if this thread is processing past the end of the particle list
    if (idx >= N)
        return;
    
    const unsigned int n_neigh = d_n_neigh[idx];
    const unsigned int n_ex = d_n_ex[idx];
    unsigned int new_n_neigh = 0;
    
    // quit now if the ex_start flag is past the end of n_ex
    if (ex_start >= n_ex)
        return;
    
    // count the number of exclusions to process in this thread
    const unsigned int n_ex_process = n_ex - ex_start;
    
    // load the exclusion list into "local" memory - fully unrolled loops should dump this into registers
    unsigned int l_ex_list[FILTER_BATCH_SIZE];
    #pragma unroll
    for (unsigned int cur_ex_idx = 0; cur_ex_idx < FILTER_BATCH_SIZE; cur_ex_idx++)
        {
        if (cur_ex_idx < n_ex_process)
            l_ex_list[cur_ex_idx] = d_ex_list[exli(idx, cur_ex_idx + ex_start)];
        else
            l_ex_list[cur_ex_idx] = 0xffffffff;
        }
    
    // loop over the list, regenerating it as we go
    for (unsigned int cur_neigh_idx = 0; cur_neigh_idx < n_neigh; cur_neigh_idx++)
        {
        unsigned int cur_neigh = d_nlist[nli(idx, cur_neigh_idx)];
        
        // test if excluded
        bool excluded = false;
        #pragma unroll
        for (unsigned int cur_ex_idx = 0; cur_ex_idx < FILTER_BATCH_SIZE; cur_ex_idx++)
            {
            if (cur_neigh == l_ex_list[cur_ex_idx])
                excluded = true;
            }
        
        // add it back to the list if it is not excluded
        if (!excluded)
            {
            if (new_n_neigh != cur_neigh_idx)
                d_nlist[nli(idx, new_n_neigh)] = cur_neigh;
            new_n_neigh++;
            }
        }
    
    // update the number of neighbors
    d_n_neigh[idx] = new_n_neigh;
    }

cudaError_t gpu_nlist_filter(unsigned int *d_n_neigh,
                             unsigned int *d_nlist,
                             const Index2D& nli,
                             const unsigned int *d_n_ex,
                             const unsigned int *d_ex_list,
                             const Index2D& exli,
                             const unsigned int N)
    {
    // determine parameters for kernel launch
    unsigned int block_size = 192;
    int n_blocks = (int)ceil(float(N)/(float)block_size);
    
    // split the processing of the full exclusion list up into a number of batches
    unsigned int n_batches = (unsigned int)ceil(float(exli.getH())/(float)FILTER_BATCH_SIZE);
    unsigned int ex_start = 0;
    for (unsigned int batch = 0; batch < n_batches; batch++)
        {
        gpu_nlist_filter_kernel<<<n_blocks, block_size>>>(d_n_neigh,
                                                          d_nlist,
                                                          nli,
                                                          d_n_ex,
                                                          d_ex_list,
                                                          exli,
                                                          N,
                                                          ex_start);
        
        ex_start += FILTER_BATCH_SIZE;
        }
    
    return cudaSuccess;
    }
