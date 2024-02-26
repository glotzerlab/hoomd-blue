// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "BondedGroupData.cuh"
#include "ParticleData.cuh"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#include <hip/hip_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#pragma GCC diagnostic pop

/*! \file BondedGroupData.cu
    \brief Implements the helper functions (GPU version) for updating the GPU bonded group tables
*/

namespace hoomd
    {
template<unsigned int group_size, typename group_t>
__global__ void gpu_count_groups_kernel(const unsigned int n_groups,
                                        const group_t* d_group_table,
                                        const unsigned int* d_rtag,
                                        unsigned int* d_scratch_idx,
                                        unsigned int* d_scratch_g,
                                        unsigned int* d_n_groups,
                                        unsigned int max_n_groups,
                                        unsigned int* d_condition,
                                        unsigned int next_flag)
    {
    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_idx >= n_groups)
        return;

    group_t g = d_group_table[group_idx];

    for (unsigned int i = 0; i < group_size; ++i)
        {
        unsigned int tag_i = g.tag[i];
        unsigned int pidx_i = d_rtag[tag_i];

        // detect incomplete groups
        if (pidx_i == NOT_LOCAL)
            atomicMax(d_condition, next_flag + 1 + group_idx);

        // write out group_idx to temporary array
        d_scratch_g[i * n_groups + group_idx] = group_idx;
        d_scratch_idx[i * n_groups + group_idx] = pidx_i;

        // atomically increment number of groups
        unsigned int n = 0;
        if (pidx_i != NOT_LOCAL)
            n = atomicInc(&d_n_groups[pidx_i], 0xffffffff);

        if (n >= max_n_groups)
            // set flag to indicate we need to grow the output array
            atomicMax(d_condition, next_flag);
        }
    }

template<unsigned int group_size, typename group_t>
__global__ void gpu_group_scatter_kernel(unsigned int n_scratch,
                                         const unsigned int* d_scratch_g,
                                         const unsigned int* d_scratch_idx,
                                         const unsigned int* d_offset,
                                         const group_t* d_members,
                                         const typeval_union* d_group_typeval,
                                         const unsigned int* d_rtag,
                                         group_t* d_pidx_group_table,
                                         unsigned int* d_pidx_gpos_table,
                                         unsigned int pidx_group_table_pitch,
                                         bool has_type_mapping)
    {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_scratch)
        return;

    unsigned int pidx = d_scratch_idx[i];
    unsigned int offset = d_offset[i] * pidx_group_table_pitch + pidx;

    // load group
    unsigned int group_idx = d_scratch_g[i];
    group_t g = d_members[group_idx];

    // construct compact group representation, excluding particle pidx
    group_t p;

    if (has_type_mapping)
        {
        // last element = group type
        p.idx[group_size - 1] = d_group_typeval[group_idx].type;
        }
    else
        {
        // last element = group index
        p.idx[group_size - 1] = group_idx;
        }

    unsigned int j = 0;

    // position in group
    unsigned int gpos = 0;

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
void gpu_update_group_table(const unsigned int n_groups,
                            const unsigned int N,
                            const group_t* d_group_table,
                            const typeval_union* d_group_typeval,
                            const unsigned int* d_rtag,
                            unsigned int* d_n_groups,
                            unsigned int max_n_groups,
                            unsigned int* d_condition,
                            unsigned int next_flag,
                            unsigned int& flag,
                            group_t* d_pidx_group_table,
                            unsigned int* d_pidx_gpos_table,
                            const unsigned int pidx_group_table_pitch,
                            unsigned int* d_scratch_g,
                            unsigned int* d_scratch_idx,
                            unsigned int* d_offsets,
                            bool has_type_mapping,
                            CachedAllocator& alloc)
    {
    // construct scratch table by expanding the group table by particle index
    unsigned int block_size = 256;
    unsigned n_blocks = n_groups / block_size + 1;

    // reset number of groups
    hipMemsetAsync(d_n_groups, 0, sizeof(unsigned int) * N);

    hipLaunchKernelGGL(HIP_KERNEL_NAME(gpu_count_groups_kernel<group_size>),
                       dim3(n_blocks),
                       dim3(block_size),
                       0,
                       0,
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
    hipMemcpy(&flag, d_condition, sizeof(unsigned int), hipMemcpyDeviceToHost);

    if (!(flag >= next_flag) && n_groups)
        {
        // we are good, fill group table
        // sort groups by particle idx
        thrust::device_ptr<unsigned int> scratch_idx(d_scratch_idx);
        thrust::device_ptr<unsigned int> scratch_g(d_scratch_g);
#ifdef __HIP_PLATFORM_HCC__
        thrust::sort_by_key(thrust::hip::par(alloc),
#else
        thrust::sort_by_key(thrust::cuda::par(alloc),
#endif
                            scratch_idx,
                            scratch_idx + group_size * n_groups,
                            scratch_g);

        // perform a segmented scan of d_scratch_idx
        thrust::device_ptr<unsigned int> offsets(d_offsets);
        thrust::constant_iterator<unsigned int> const_it(1);
#ifdef __HIP_PLATFORM_HCC__
        thrust::exclusive_scan_by_key(thrust::hip::par(alloc),
#else
        thrust::exclusive_scan_by_key(thrust::cuda::par(alloc),
#endif
                                      scratch_idx,
                                      scratch_idx + group_size * n_groups,
                                      const_it,
                                      offsets);

        // scatter groups to destinations
        block_size = 256;
        n_blocks = (group_size * n_groups) / block_size + 1;

        hipLaunchKernelGGL(gpu_group_scatter_kernel<group_size>,
                           dim3(n_blocks),
                           dim3(block_size),
                           0,
                           0,
                           n_groups * group_size,
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
template void gpu_update_group_table<2>(const unsigned int n_groups,
                                        const unsigned int N,
                                        const union group_storage<2>* d_group_table,
                                        const typeval_union* d_group_typeval,
                                        const unsigned int* d_rtag,
                                        unsigned int* d_n_groups,
                                        unsigned int max_n_groups,
                                        unsigned int* d_condition,
                                        unsigned int next_flag,
                                        unsigned int& flag,
                                        group_storage<2>* d_pidx_group_table,
                                        unsigned int* d_pidx_gpos_table,
                                        const unsigned int pidx_group_table_pitch,
                                        unsigned int* d_scratch_g,
                                        unsigned int* d_scratch_idx,
                                        unsigned int* d_offsets,
                                        bool has_type_mapping,
                                        CachedAllocator& alloc);

//! AngleData
template void gpu_update_group_table<3>(const unsigned int n_groups,
                                        const unsigned int N,
                                        const union group_storage<3>* d_group_table,
                                        const typeval_union* d_group_typeval,
                                        const unsigned int* d_rtag,
                                        unsigned int* d_n_groups,
                                        unsigned int max_n_groups,
                                        unsigned int* d_condition,
                                        unsigned int next_flag,
                                        unsigned int& flag,
                                        group_storage<3>* d_pidx_group_table,
                                        unsigned int* d_pidx_gpos_table,
                                        const unsigned int pidx_group_table_pitch,
                                        unsigned int* d_scratch_g,
                                        unsigned int* d_scratch_idx,
                                        unsigned int* d_offsets,
                                        bool has_type_mapping,
                                        CachedAllocator& alloc);

//! DihedralData and ImproperData
template void gpu_update_group_table<4>(const unsigned int n_groups,
                                        const unsigned int N,
                                        const union group_storage<4>* d_group_table,
                                        const typeval_union* d_group_typeval,
                                        const unsigned int* d_rtag,
                                        unsigned int* d_n_groups,
                                        unsigned int max_n_groups,
                                        unsigned int* d_condition,
                                        unsigned int next_flag,
                                        unsigned int& flag,
                                        group_storage<4>* d_pidx_group_table,
                                        unsigned int* d_pidx_gpos_table,
                                        const unsigned int pidx_group_table_pitch,
                                        unsigned int* d_scratch_g,
                                        unsigned int* d_scratch_idx,
                                        unsigned int* d_offsets,
                                        bool has_type_mapping,
                                        CachedAllocator& alloc);

//! MeshTriangleData
template void gpu_update_group_table<6>(const unsigned int n_groups,
                                        const unsigned int N,
                                        const union group_storage<6>* d_group_table,
                                        const typeval_union* d_group_typeval,
                                        const unsigned int* d_rtag,
                                        unsigned int* d_n_groups,
                                        unsigned int max_n_groups,
                                        unsigned int* d_condition,
                                        unsigned int next_flag,
                                        unsigned int& flag,
                                        group_storage<6>* d_pidx_group_table,
                                        unsigned int* d_pidx_gpos_table,
                                        const unsigned int pidx_group_table_pitch,
                                        unsigned int* d_scratch_g,
                                        unsigned int* d_scratch_idx,
                                        unsigned int* d_offsets,
                                        bool has_type_mapping,
                                        CachedAllocator& alloc);
    } // end namespace hoomd
