// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/SorterGPU.cu
 * \brief Defines GPU functions and kernels used by mpcd::SorterGPU
 */

#include "CellListGPU.cuh"

#include "hoomd/extern/cub/cub/device/device_select.cuh"

namespace mpcd
{
namespace gpu
{
namespace kernel
{
__global__ void sort_apply(Scalar4 *d_pos_alt,
                           Scalar4 *d_vel_alt,
                           unsigned int *d_tag_alt,
                           const Scalar4 *d_pos,
                           const Scalar4 *d_vel,
                           const unsigned int *d_tag,
                           const unsigned int *d_order,
                           const unsigned int N)
    {
    // one thread per particle
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    const unsigned int old_idx = d_order[idx];
    d_pos_alt[idx] = d_pos[old_idx];
    d_vel_alt[idx] = d_vel[old_idx];
    d_tag_alt[idx] = d_tag[old_idx];
    }

__global__ void sort_set_sentinel(unsigned int *d_cell_list,
                              const unsigned int *d_cell_np,
                              const Index2D cli,
                              const unsigned int sentinel,
                              const unsigned int N_cli)
    {
    // one thread per cell-list entry
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_cli)
        return;

    // convert the entry 1D index into a 2D index
    const unsigned int cell = idx / cli.getW();
    const unsigned int offset = idx - (cell * cli.getW());

    // if the offset lies outside the number of particles in the cell, fill it with sentinel
    const unsigned int np = d_cell_np[cell];
    if (offset >= np)
        d_cell_list[idx] = sentinel;
    }

__global__ void sort_gen_reverse(unsigned int *d_rorder,
                                 const unsigned int *d_order,
                                 const unsigned int N)
    {
    // one thread per particle
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    // inverse map the ordering
    // d_order maps the new index (idx) onto the old index (pid), so we need to flip this around
    const unsigned int pid = d_order[idx];
    d_rorder[pid] = idx;
    }

} // end namespace kernel
cudaError_t sort_apply(Scalar4 *d_pos_alt,
                       Scalar4 *d_vel_alt,
                       unsigned int *d_tag_alt,
                       const Scalar4 *d_pos,
                       const Scalar4 *d_vel,
                       const unsigned int *d_tag,
                       const unsigned int *d_order,
                       const unsigned int N,
                       const unsigned int block_size)
    {
    if (N == 0) return cudaSuccess;

    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::sort_apply);
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(N / run_block_size + 1);
    mpcd::gpu::kernel::sort_apply<<<grid, run_block_size>>>(d_pos_alt,
                                                            d_vel_alt,
                                                            d_tag_alt,
                                                            d_pos,
                                                            d_vel,
                                                            d_tag,
                                                            d_order,
                                                            N);

    return cudaSuccess;
    }

cudaError_t sort_set_sentinel(unsigned int *d_cell_list,
                              const unsigned int *d_cell_np,
                              const Index2D& cli,
                              const unsigned int sentinel,
                              const unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::sort_set_sentinel);
        max_block_size = attr.maxThreadsPerBlock;
        }

    const unsigned int N_cli = cli.getNumElements();

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(N_cli / run_block_size + 1);
    mpcd::gpu::kernel::sort_set_sentinel<<<grid, run_block_size>>>(d_cell_list,
                                                                   d_cell_np,
                                                                   cli,
                                                                   sentinel,
                                                                   N_cli);

    return cudaSuccess;
    }

//! Less-than comparison functor for cell-list compaction
struct LessThan
    {
    __host__ __device__ __forceinline__
    LessThan(unsigned int compare_)
        : compare(compare_) {}

    __host__ __device__ __forceinline__
    bool operator()(const unsigned int& val) const
        {
        return (val < compare);
        }

    unsigned int compare;
    };

cudaError_t sort_cell_compact(unsigned int *d_order,
                              unsigned int *d_num_select,
                              void *d_tmp_storage,
                              size_t& tmp_storage_bytes,
                              const unsigned int *d_cell_list,
                              const unsigned int num_items,
                              const unsigned int N_mpcd)
    {
    LessThan selector(N_mpcd);
    cub::DeviceSelect::If(d_tmp_storage, tmp_storage_bytes, d_cell_list, d_order, d_num_select, num_items, selector);

    return cudaSuccess;
    }

cudaError_t sort_gen_reverse(unsigned int *d_rorder,
                             const unsigned int *d_order,
                             const unsigned int N,
                             const unsigned int block_size)
    {
    if (N == 0) return cudaSuccess;

    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::sort_gen_reverse);
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(N / run_block_size + 1);
    mpcd::gpu::kernel::sort_gen_reverse<<<grid, run_block_size>>>(d_rorder, d_order, N);

    return cudaSuccess;
    }
} // end namespace gpu
} // end namespace mpcd
