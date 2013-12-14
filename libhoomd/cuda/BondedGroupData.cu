/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

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

// Maintainer: jglaser

#include "ParticleData.cuh"
#include "BondedGroupData.cuh"

#include "moderngpu/kernels/scan.cuh"
#include "moderngpu/kernels/mergesort.cuh"
#include "moderngpu/kernels/intervalmove.cuh"

/*! \file BondedGroupData.cu
    \brief Implements the helper functions (GPU version) for updating the GPU bonded group tables
*/

template<unsigned int group_size, typename group_t>
__global__ void gpu_count_groups_kernel(
    const unsigned int n_groups,
    const group_t *d_group_table,
    const unsigned int *d_group_type,
    const unsigned int *d_rtag,
    unsigned int *d_scratch_idx,
    group_t *d_scratch_g,
    unsigned int *d_n_groups,
    unsigned int max_n_groups,
    unsigned int *d_condition,
    unsigned int next_flag
    )
    {
    unsigned int group_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (group_idx >= n_groups) return;
   
    group_t g = d_group_table[group_idx];

    group_t p;
    // last element = group type
    p.idx[group_size-1] = d_group_type[group_idx];

    #pragma unroll
    for (unsigned int i = 0; i < group_size; ++i)
        {
        unsigned int tag_i = g.tag[i];
        unsigned int pidx_i = d_rtag[tag_i];

        // detect incomplete groups
        if (pidx_i == NOT_LOCAL)
            atomicMax(d_condition, next_flag+1+group_idx);

        // construct compact group representation, excluding particle index i
        unsigned int j = 0;
        #pragma unroll
        for (unsigned int k = 0; k < group_size; ++k)
            {
            if (i == k) continue;
            unsigned int tag_k = g.tag[k];
            unsigned int pidx_k = d_rtag[tag_k];

            p.idx[j++] = pidx_k;
            }

        // write out to temporary array
        d_scratch_g[i*n_groups+group_idx] = p;
        d_scratch_idx[i*n_groups+group_idx] = pidx_i;

        // atomically increment number of groups
        unsigned int n = 0;
        if (pidx_i != NOT_LOCAL)
           n = atomicInc(&d_n_groups[pidx_i],0xffffffff);

        if (n >= max_n_groups)
            // set flag to indicate we need to grow the output array
            atomicMax(d_condition,next_flag);
        }
    }

template<typename group_t>
__global__ void gpu_group_scatter_kernel(
    unsigned int n_scratch,
    const group_t *d_scratch_g,
    const unsigned int *d_scratch_idx,
    const unsigned int *d_offset,
    group_t *d_pidx_group_table,
    unsigned int pidx_group_table_pitch
    )
    {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_scratch) return;

    unsigned int pidx = d_scratch_idx[i];
    unsigned int offset = d_offset[i]*pidx_group_table_pitch + pidx;
    d_pidx_group_table[offset] = d_scratch_g[i];
    }

template<unsigned int group_size, typename group_t>
void gpu_update_group_table(
    const unsigned int n_groups,
    const unsigned int N,
    const group_t* d_group_table,
    const unsigned int *d_group_type,
    const unsigned int *d_rtag,
    unsigned int *d_n_groups,
    unsigned int max_n_groups,
    unsigned int *d_condition,
    unsigned int next_flag,
    unsigned int &flag,
    group_t *d_pidx_group_table,
    const unsigned int pidx_group_table_pitch,
    cached_allocator& alloc,
    mgpu::ContextPtr mgpu_context
    )
    {
    // allocate temporary buffers
    group_t *d_scratch_g = (group_t *)alloc.allocate(group_size*n_groups*sizeof(group_t));
    unsigned int *d_scratch_idx = (unsigned int *)alloc.allocate(group_size*n_groups*sizeof(unsigned int));

    // construct scratch table by expanding the group table by particle index
    unsigned int block_size = 512;
    unsigned n_blocks = n_groups / block_size + 1;

    // reset number of groups
    cudaMemsetAsync(d_n_groups, 0, sizeof(unsigned int)*N);

    gpu_count_groups_kernel<group_size><<<n_blocks, block_size>>>(
        n_groups,
        d_group_table,
        d_group_type,
        d_rtag,
        d_scratch_idx,
        d_scratch_g,
        d_n_groups,
        max_n_groups,
        d_condition,
        next_flag);

    // read back flag
    cudaMemcpy(&flag, d_condition, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    if (! (flag >= next_flag))
        {
        // we are good, fill group table

        // sort groups by particle index
        mgpu::MergesortPairs(d_scratch_idx, d_scratch_g, group_size*n_groups, *mgpu_context);

        // allocate temporary array
        unsigned int *d_offsets = (unsigned int *)alloc.allocate(group_size*n_groups*sizeof(unsigned int));
        // determine output offsets of segments
        unsigned int *d_seg_offsets = (unsigned int *)alloc.allocate(N*sizeof(unsigned int));
	    mgpu::Scan<mgpu::MgpuScanTypeExc>(d_n_groups, N, (unsigned int) 0, mgpu::plus<unsigned int>(),
            (unsigned int *) NULL, (unsigned int *)NULL, d_seg_offsets,*mgpu_context);

        // use IntervalMove to perform a segmented scan of d_scratch_idx,
        // using sement offsets as an input
        mgpu::constant_iterator<unsigned int> const_it(0);
        mgpu::counting_iterator<unsigned int> count_it(0);
        mgpu::IntervalMove(group_size*n_groups, const_it, d_seg_offsets, d_seg_offsets, N,
            count_it, d_offsets, *mgpu_context);

        // scatter groups to destinations
        block_size = 512;
        n_blocks = group_size*n_groups/block_size + 1;

        gpu_group_scatter_kernel<<<n_blocks, block_size>>>(
            n_groups*group_size,
            d_scratch_g,
            d_scratch_idx,
            d_offsets,
            d_pidx_group_table,
            pidx_group_table_pitch);
 
        alloc.deallocate((char *) d_offsets,0);
        alloc.deallocate((char *) d_seg_offsets, 0);
        } 

    // release temporary arrays
    alloc.deallocate((char *) d_scratch_g,0);
    alloc.deallocate((char *) d_scratch_idx,0);
    }

/*
 * Explicit template instantiations
 */

//! BondData
template void gpu_update_group_table<2>(
    const unsigned int n_groups,
    const unsigned int N,
    const union group_storage<2> *d_group_table,
    const unsigned int *d_group_type,
    const unsigned int *d_rtag,
    unsigned int *d_n_groups,
    unsigned int max_n_groups,
    unsigned int *d_condition,
    unsigned int next_flag,
    unsigned int &flag,
    union group_storage<2> *d_pidx_group_table,
    const unsigned int pidx_group_table_pitch,
    cached_allocator& alloc,
    mgpu::ContextPtr mgpu_context
    );
