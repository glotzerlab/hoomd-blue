/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2014 The Regents of
the University of Michigan All rights reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: joaander

/*! \file NeighborListGPUBinned.cu
    \brief Defines GPU kernel code for neighbor list processing on the GPU
*/

#include "NeighborListGPUBinned.cuh"

#include "NeighborListGPU.cuh"
#include <stdio.h>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

/*! \param d_result Device pointer to a single uint. Will be set to 1 if an update is needed
    \param d_last_pos Particle positions at the time the nlist was last updated
    \param d_pos Current particle positions
    \param N Number of particles
    \param box Box dimensions
    \param d_rcut_max The maximum rcut(i,j) that any particle i participates in
    \param rcut_shift The amount to add to rcut_max to get the buffer width
    \param ntypes The number of particle types
    \param lambda_min Minimum contraction of deformation tensor
    \param lambda Diagonal deformation tensor (for orthorhombic boundaries)
    \param checkn

    gpu_nlist_needs_update_check_new_kernel() executes one thread per particle. Every particle's current position is
    compared to its last position. If the particle has moved a distance more than the buffer width, then *d_result
    is set to \a checkn.
*/
__global__ void gpu_nlist_needs_update_check_new_kernel(unsigned int *d_result,
                                                        const Scalar4 *d_last_pos,
                                                        const Scalar4 *d_pos,
                                                        const unsigned int N,
                                                        const BoxDim box,
                                                        const Scalar *d_rcut_max,
                                                        const Scalar rcut_shift,
                                                        const unsigned int ntypes,
                                                        const Scalar lambda_min,
                                                        const Scalar3 lambda,
                                                        const unsigned int checkn)
    {
    // cache delta max into shared memory
    // shared data for per type pair parameters
    extern __shared__ unsigned char s_data[];
    
    // pointer for the r_listsq data
    Scalar *s_maxshiftsq = (Scalar *)(&s_data[0]);

    // load in the per type pair r_list
    for (unsigned int cur_offset = 0; cur_offset < ntypes; cur_offset += blockDim.x)
        {
        if (cur_offset + threadIdx.x < ntypes)
            {
            const Scalar rcut_max_i = d_rcut_max[cur_offset + threadIdx.x];
            const Scalar rmax = rcut_max_i + rcut_shift;
            const Scalar delta_max = (rmax*lambda_min - rcut_max_i)/Scalar(2.0);
            s_maxshiftsq[cur_offset + threadIdx.x] = (delta_max > 0) ? delta_max*delta_max : 0.0f;
            }
        }
    __syncthreads();
    
    
    // each thread will compare vs it's old position to see if the list needs updating
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
        {
        Scalar4 cur_postype = d_pos[idx];
        Scalar3 cur_pos = make_scalar3(cur_postype.x, cur_postype.y, cur_postype.z);
        const unsigned int cur_type = __scalar_as_int(cur_postype.w);
        Scalar4 last_postype = d_last_pos[idx];
        Scalar3 last_pos = make_scalar3(last_postype.x, last_postype.y, last_postype.z);

        Scalar3 dx = cur_pos - lambda*last_pos;
        dx = box.minImage(dx);

        if (dot(dx, dx) >= s_maxshiftsq[cur_type])
            atomicMax(d_result, checkn);
        }
    }

cudaError_t gpu_nlist_needs_update_check_new(unsigned int *d_result,
                                             const Scalar4 *d_last_pos,
                                             const Scalar4 *d_pos,
                                             const unsigned int N,
                                             const BoxDim& box,
                                             const Scalar *d_rcut_max,
                                             const Scalar rcut_shift,
                                             const unsigned int ntypes,
                                             const Scalar lambda_min,
                                             const Scalar3 lambda,
                                             const unsigned int checkn)
    {
    const unsigned int shared_bytes = sizeof(Scalar) * ntypes;
    
    unsigned int block_size = 128;
    int n_blocks = N/block_size+1;
    gpu_nlist_needs_update_check_new_kernel<<<n_blocks, block_size, shared_bytes>>>(d_result,
                                                                                    d_last_pos,
                                                                                    d_pos,
                                                                                    N,
                                                                                    box,
                                                                                    d_rcut_max,
                                                                                    rcut_shift,
                                                                                    ntypes,
                                                                                    lambda_min,
                                                                                    lambda,
                                                                                    checkn);

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
                                        const unsigned int *d_head_list,
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
    const unsigned int my_head = d_head_list[idx];
    for (unsigned int cur_neigh_idx = 0; cur_neigh_idx < n_neigh; cur_neigh_idx++)
        {
        unsigned int cur_neigh = d_nlist[my_head + cur_neigh_idx];

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
                d_nlist[my_head + new_n_neigh] = cur_neigh;
            new_n_neigh++;
            }
        }

    // update the number of neighbors
    d_n_neigh[idx] = new_n_neigh;
    }

cudaError_t gpu_nlist_filter(unsigned int *d_n_neigh,
                             unsigned int *d_nlist,
                             const unsigned int *d_head_list,
                             const unsigned int *d_n_ex,
                             const unsigned int *d_ex_list,
                             const Index2D& exli,
                             const unsigned int N,
                             const unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void *)gpu_nlist_filter_kernel);
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(block_size, max_block_size);

    // determine parameters for kernel launch
    int n_blocks = N/run_block_size + 1;

    // split the processing of the full exclusion list up into a number of batches
    unsigned int n_batches = (unsigned int)ceil(double(exli.getH())/double(FILTER_BATCH_SIZE));
    unsigned int ex_start = 0;
    for (unsigned int batch = 0; batch < n_batches; batch++)
        {
        gpu_nlist_filter_kernel<<<n_blocks, run_block_size>>>(d_n_neigh,
                                                              d_nlist,
                                                              d_head_list,
                                                              d_n_ex,
                                                              d_ex_list,
                                                              exli,
                                                              N,
                                                              ex_start);

        ex_start += FILTER_BATCH_SIZE;
        }

    return cudaSuccess;
    }

//! GPU kernel to update the exclusions list
__global__ void gpu_update_exclusion_list_kernel(const unsigned int *tags,
                                                  const unsigned int *rtags,
                                                  const unsigned int *n_ex_tag,
                                                  const unsigned int *ex_list_tag,
                                                  const Index2D ex_list_tag_indexer,
                                                  unsigned int *n_ex_idx,
                                                  unsigned int *ex_list_idx,
                                                  const Index2D ex_list_indexer,
                                                  const unsigned int N)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    unsigned int tag = tags[idx];

    unsigned int n = n_ex_tag[tag];

    // copy over number of exclusions
    n_ex_idx[idx] = n;

    for (unsigned int offset = 0; offset < n; offset++)
        {
        unsigned int ex_tag = ex_list_tag[ex_list_tag_indexer(tag, offset)];
        unsigned int ex_idx = rtags[ex_tag];

        ex_list_idx[ex_list_indexer(idx, offset)] = ex_idx;
        }
    }


//! GPU function to update the exclusion list on the device
/*! \param d_tag Array of particle tags
    \param d_rtag Array of reverse-lookup tag->idx
    \param d_n_ex_tag List of number of exclusions per tag
    \param d_ex_list_tag 2D Exclusion list per tag
    \param ex_list_tag_indexer Indexer for per-tag exclusion list
    \param d_n_ex_idx List of number of exclusions per idx
    \param d_ex_list_idx Exclusion list per idx
    \param ex_list_indexer Indexer for per-idx exclusion list
    \param N number of particles
 */
cudaError_t gpu_update_exclusion_list(const unsigned int *d_tag,
                                const unsigned int *d_rtag,
                                const unsigned int *d_n_ex_tag,
                                const unsigned int *d_ex_list_tag,
                                const Index2D& ex_list_tag_indexer,
                                unsigned int *d_n_ex_idx,
                                unsigned int *d_ex_list_idx,
                                const Index2D& ex_list_indexer,
                                const unsigned int N)
    {
    unsigned int block_size = 512;

    gpu_update_exclusion_list_kernel<<<N/block_size + 1, block_size>>>(d_tag,
                                                                       d_rtag,
                                                                       d_n_ex_tag,
                                                                       d_ex_list_tag,
                                                                       ex_list_tag_indexer,
                                                                       d_n_ex_idx,
                                                                       d_ex_list_idx,
                                                                       ex_list_indexer,
                                                                       N);

    return cudaSuccess;
    }

//! GPU kernel to do a preliminary sizing on particles
__global__ void gpu_nlist_init_head_list_kernel(unsigned int *d_head_list,
                                                unsigned int *d_req_size_nlist,
                                                const unsigned int *d_Nmax,
                                                const Scalar4 *d_pos,
                                                const unsigned int N,
                                                const unsigned int ntypes)
    {
    // cache the d_Nmax into shared memory for fast reads
    extern __shared__ unsigned char sh[];
    unsigned int *s_Nmax = (unsigned int *)(&sh[0]);
    for (unsigned int cur_offset = 0; cur_offset < ntypes; cur_offset += blockDim.x)
        {
        if (cur_offset + threadIdx.x < ntypes)
            {
            s_Nmax[cur_offset + threadIdx.x] = d_Nmax[cur_offset + threadIdx.x];
            }
        }
    __syncthreads();
    
    
    // compute the bin index this thread operates on
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    // quit now if this thread is processing past the end of the particle list
    if (idx >= N)
        return;

    const Scalar4 postype_i = d_pos[idx];
    const unsigned int type_i = __scalar_as_int(postype_i.w);
    const unsigned int Nmax_i = d_Nmax[type_i];
    
    d_head_list[idx] = Nmax_i;
    
    // last thread sets the number of particles in the memory req as well
    if (idx == (N-1))
        {
        *d_req_size_nlist = Nmax_i;
        }
    }

__global__ void gpu_nlist_get_nlist_mem_kernel(unsigned int *d_req_size_nlist,
                                               const unsigned int *d_head_list,
                                               const unsigned int N)
    {
    *d_req_size_nlist += d_head_list[N-1];
    }

cudaError_t gpu_nlist_build_head_list(unsigned int *d_head_list,
                                      unsigned int *d_req_size_nlist,
                                      const unsigned int *d_Nmax,
                                      const Scalar4 *d_pos,
                                      const unsigned int N,
                                      const unsigned int ntypes,
                                      const unsigned int block_size)
    {   
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
       cudaFuncAttributes attr;
       cudaFuncGetAttributes(&attr, (const void *)gpu_nlist_init_head_list_kernel);
       max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(block_size, max_block_size);
    unsigned int shared_bytes = ntypes*sizeof(unsigned int);
    
    // initialize each particle with its number of neighbors
    gpu_nlist_init_head_list_kernel<<<N/run_block_size + 1, run_block_size, shared_bytes>>>(d_head_list,
                                                                                            d_req_size_nlist,
                                                                                            d_Nmax,
                                                                                            d_pos,
                                                                                            N,
                                                                                            ntypes);
    
    // thrust exclusive scan does a prefix sum to compute the "head" list of each, starting from 0
    thrust::device_ptr<unsigned int> t_head_list = thrust::device_pointer_cast(d_head_list);
    thrust::exclusive_scan(t_head_list, t_head_list + N, t_head_list);
    
    // compute the total number on the GPU from the last element (it would be nice if the thrust code could do this)
    // it would be better to write our own scan algorithm to avoid the extra kernel overhead,
    // but this is better than a data copy of the whole head list, and faster to code up
    gpu_nlist_get_nlist_mem_kernel<<<1,1>>>(d_req_size_nlist,
                                            d_head_list,
                                            N);
    
    return cudaSuccess;
    }
    
