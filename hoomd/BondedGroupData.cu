// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

#include "ParticleData.cuh"
#include "BondedGroupData.cuh"

#include "hoomd/extern/kernels/scan.cuh"
#include "hoomd/extern/kernels/mergesort.cuh"
#include "hoomd/extern/kernels/intervalmove.cuh"

/*! \file BondedGroupData.cu
    \brief Implements the helper functions (GPU version) for updating the GPU bonded group tables
*/

template<unsigned int group_size, typename group_t>
__global__ void gpu_count_groups_kernel(
    const unsigned int n_groups,
    const group_t *d_group_table,
    const unsigned int *d_rtag,
    unsigned int *d_scratch_idx,
    unsigned int *d_scratch_g,
    unsigned int *d_n_groups,
    unsigned int max_n_groups,
    unsigned int *d_condition,
    unsigned int next_flag
    )
    {
    unsigned int group_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (group_idx >= n_groups) return;

    group_t g = d_group_table[group_idx];

    #pragma unroll
    for (unsigned int i = 0; i < group_size; ++i)
        {
        unsigned int tag_i = g.tag[i];
        unsigned int pidx_i = d_rtag[tag_i];

        // detect incomplete groups
        if (pidx_i == NOT_LOCAL)
            atomicMax(d_condition, next_flag+1+group_idx);

        // write out group_idx to temporary array
        d_scratch_g[i*n_groups+group_idx] = group_idx;
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

template<unsigned int group_size, typename group_t>
__global__ void gpu_group_scatter_kernel(
    unsigned int n_scratch,
    const unsigned int *d_scratch_g,
    const unsigned int *d_scratch_idx,
    const unsigned int *d_offset,
    const group_t *d_members,
    const typeval_union *d_group_typeval,
    const unsigned int *d_rtag,
    group_t *d_pidx_group_table,
    unsigned int *d_pidx_gpos_table,
    unsigned int pidx_group_table_pitch,
    bool has_type_mapping
    )
    {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_scratch) return;

    unsigned int pidx = d_scratch_idx[i];
    unsigned int offset = d_offset[i]*pidx_group_table_pitch + pidx;

    // load group
    unsigned int group_idx = d_scratch_g[i];
    group_t g = d_members[group_idx];

    // construct compact group representation, excluding particle pidx
    group_t p;

    if (has_type_mapping)
        {
        // last element = group type
        p.idx[group_size-1] = d_group_typeval[group_idx].type;
        }
    else
        {
        // last element = group index
        p.idx[group_size-1] = group_idx;
        }

    unsigned int j = 0;

    // position in group
    unsigned int gpos = 0;

    #pragma unroll
    for (unsigned int k = 0; k < group_size; ++k)
        {
        unsigned int tag_k = g.tag[k];
        unsigned int pidx_k = d_rtag[tag_k];
        if (pidx_k == pidx)
            {
            gpos = k;
            continue;
            }

        p.idx[j++] = pidx_k;
        }

    d_pidx_group_table[offset] = p;
    d_pidx_gpos_table[offset] = gpos;
    }

template<unsigned int group_size, typename group_t>
void gpu_update_group_table(
    const unsigned int n_groups,
    const unsigned int N,
    const group_t* d_group_table,
    const typeval_union *d_group_typeval,
    const unsigned int *d_rtag,
    unsigned int *d_n_groups,
    unsigned int max_n_groups,
    unsigned int *d_condition,
    unsigned int next_flag,
    unsigned int &flag,
    group_t *d_pidx_group_table,
    unsigned int *d_pidx_gpos_table,
    const unsigned int pidx_group_table_pitch,
    unsigned int *d_scratch_g,
    unsigned int *d_scratch_idx,
    unsigned int *d_offsets,
    unsigned int *d_seg_offsets,
    bool has_type_mapping,
    mgpu::ContextPtr mgpu_context
    )
    {
    // construct scratch table by expanding the group table by particle index
    unsigned int block_size = 512;
    unsigned n_blocks = n_groups / block_size + 1;

    // reset number of groups
    cudaMemsetAsync(d_n_groups, 0, sizeof(unsigned int)*N);

    gpu_count_groups_kernel<group_size><<<n_blocks, block_size>>>(
        n_groups,
        d_group_table,
        d_rtag,
        d_scratch_idx,
        d_scratch_g,
        d_n_groups,
        max_n_groups,
        d_condition,
        next_flag);

    // read back flag
    cudaMemcpyAsync(&flag, d_condition, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    if (! (flag >= next_flag) && n_groups)
        {
        // we are good, fill group table

        // sort groups by particle index
        mgpu::MergesortPairs(d_scratch_idx, d_scratch_g, group_size*n_groups, *mgpu_context);

        mgpu::Scan<mgpu::MgpuScanTypeExc>(d_n_groups, N, (unsigned int) 0, mgpu::plus<unsigned int>(),
            (unsigned int *) NULL, (unsigned int *)NULL, d_seg_offsets,*mgpu_context);

        // use IntervalMove to perform a segmented scan of d_scratch_idx,
        // using segment offsets as input
        mgpu::constant_iterator<unsigned int> const_it(0);
        mgpu::counting_iterator<unsigned int> count_it(0);
        mgpu::IntervalMove(group_size*n_groups, const_it, d_seg_offsets, d_seg_offsets, N,
            count_it, d_offsets, *mgpu_context);

        // scatter groups to destinations
        block_size = 512;
        n_blocks = group_size*n_groups/block_size + 1;

        gpu_group_scatter_kernel<group_size><<<n_blocks, block_size>>>(
            n_groups*group_size,
            d_scratch_g,
            d_scratch_idx,
            d_offsets,
            d_group_table,
            d_group_typeval,
            d_rtag,
            d_pidx_group_table,
            d_pidx_gpos_table,
            pidx_group_table_pitch,
            has_type_mapping);
        }
    }

/*
 * Explicit template instantiations
 */

//! BondData
template void gpu_update_group_table<2>(
    const unsigned int n_groups,
    const unsigned int N,
    const union group_storage<2> *d_group_table,
    const typeval_union *d_group_typeval,
    const unsigned int *d_rtag,
    unsigned int *d_n_groups,
    unsigned int max_n_groups,
    unsigned int *d_condition,
    unsigned int next_flag,
    unsigned int &flag,
    group_storage<2> *d_pidx_group_table,
    unsigned int *d_pidx_gpos_table,
    const unsigned int pidx_group_table_pitch,
    unsigned int *d_scratch_g,
    unsigned int *d_scratch_idx,
    unsigned int *d_offsets,
    unsigned int *d_seg_offsets,
    bool has_type_mapping,
    mgpu::ContextPtr mgpu_context
    );

//! AngleData
template void gpu_update_group_table<3>(
    const unsigned int n_groups,
    const unsigned int N,
    const union group_storage<3> *d_group_table,
    const typeval_union *d_group_typeval,
    const unsigned int *d_rtag,
    unsigned int *d_n_groups,
    unsigned int max_n_groups,
    unsigned int *d_condition,
    unsigned int next_flag,
    unsigned int &flag,
    group_storage<3> *d_pidx_group_table,
    unsigned int *d_pidx_gpos_table,
    const unsigned int pidx_group_table_pitch,
    unsigned int *d_scratch_g,
    unsigned int *d_scratch_idx,
    unsigned int *d_offsets,
    unsigned int *d_seg_offsets,
    bool has_type_mapping,
    mgpu::ContextPtr mgpu_context
    );

//! DihedralData and ImproperData
template void gpu_update_group_table<4>(
    const unsigned int n_groups,
    const unsigned int N,
    const union group_storage<4> *d_group_table,
    const typeval_union *d_group_typeval,
    const unsigned int *d_rtag,
    unsigned int *d_n_groups,
    unsigned int max_n_groups,
    unsigned int *d_condition,
    unsigned int next_flag,
    unsigned int &flag,
    group_storage<4> *d_pidx_group_table,
    unsigned int *d_pidx_gpos_table,
    const unsigned int pidx_group_table_pitch,
    unsigned int *d_scratch_g,
    unsigned int *d_scratch_idx,
    unsigned int *d_offsets,
    unsigned int *d_seg_offsets,
    bool has_type_mapping,
    mgpu::ContextPtr mgpu_context
    );
