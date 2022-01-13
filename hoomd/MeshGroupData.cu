#include "MeshGroupData.cuh"
#include "ParticleData.cuh"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#include <hip/hip_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#pragma GCC diagnostic pop

/*! \file MeshGroupData.cu
    \brief Implements the helper functions (GPU version) for updating the GPU bonded group tables
*/

namespace hoomd
    {
template<unsigned int group_size, typename group_t>
__global__ void gpu_count_mesh_kernel(const unsigned int n_groups,
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

    unsigned int group_size_half = group_size / 2;

    for (unsigned int i = 0; i < group_size_half; ++i)
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
__global__ void gpu_mesh_scatter_kernel(unsigned int n_scratch,
                                        const unsigned int* d_scratch_g,
                                        const unsigned int* d_scratch_idx,
                                        const unsigned int* d_offset,
                                        const group_t* d_members,
                                        const typeval_union* d_group_typeval,
                                        const unsigned int* d_rtag,
                                        group_t* d_pidx_group_table,
                                        unsigned int pidx_group_table_pitch)
    {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_scratch)
        return;

    unsigned int pidx = d_scratch_idx[i];
    unsigned int offset = d_offset[i] * pidx_group_table_pitch + pidx;

    unsigned int group_size_half = group_size / 2;

    // load group
    unsigned int group_idx = d_scratch_g[i];
    group_t g = d_members[group_idx];

    // construct compact group representation, excluding particle pidx
    group_t p;

    p.idx[group_size - 1] = d_group_typeval[group_idx].type;

    for (unsigned int k = group_size_half; k < group_size; ++k)
        {
        p.idx[k - 1] = g.tag[k];
        }

    unsigned int j = 0;

    for (unsigned int k = 0; k < group_size_half; ++k)
        {
        unsigned int tag_k = g.tag[k];
        unsigned int pidx_k = d_rtag[tag_k];
        if (pidx_k == pidx)
            {
            continue;
            }

        p.idx[j++] = pidx_k;
        }

    d_pidx_group_table[offset] = p;
    }

template<unsigned int group_size, typename group_t>
void gpu_update_mesh_table(const unsigned int n_groups,
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
                           const unsigned int pidx_group_table_pitch,
                           unsigned int* d_scratch_g,
                           unsigned int* d_scratch_idx,
                           unsigned int* d_offsets,
                           CachedAllocator& alloc)
    {
    // construct scratch table by expanding the group table by particle index
    unsigned int block_size = 256;
    unsigned n_blocks = n_groups / block_size + 1;

    unsigned int group_size_half = group_size / 2;

    // reset number of groups
    hipMemsetAsync(d_n_groups, 0, sizeof(unsigned int) * N);

    hipLaunchKernelGGL(HIP_KERNEL_NAME(gpu_count_mesh_kernel<group_size>),
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
                            scratch_idx + group_size_half * n_groups,
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
                                      scratch_idx + group_size_half * n_groups,
                                      const_it,
                                      offsets);

        // scatter groups to destinations
        block_size = 256;
        n_blocks = (group_size_half * n_groups) / block_size + 1;

        hipLaunchKernelGGL(gpu_mesh_scatter_kernel<group_size>,
                           dim3(n_blocks),
                           dim3(block_size),
                           0,
                           0,
                           n_groups * group_size_half,
                           d_scratch_g,
                           d_scratch_idx,
                           d_offsets,
                           d_group_table,
                           d_group_typeval,
                           d_rtag,
                           d_pidx_group_table,
                           pidx_group_table_pitch);
        }
    }

/*
 * Explicit template instantiations
 */

//! MeshBondData and MeshTriangleData
template void gpu_update_mesh_table<4>(const unsigned int n_groups,
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
                                       const unsigned int pidx_group_table_pitch,
                                       unsigned int* d_scratch_g,
                                       unsigned int* d_scratch_idx,
                                       unsigned int* d_offsets,
                                       CachedAllocator& alloc);

//! MeshTriangleData
template void gpu_update_mesh_table<6>(const unsigned int n_groups,
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
                                       const unsigned int pidx_group_table_pitch,
                                       unsigned int* d_scratch_g,
                                       unsigned int* d_scratch_idx,
                                       unsigned int* d_offsets,
                                       CachedAllocator& alloc);
    } // end namespace hoomd
