// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/SorterGPU.cu
 * \brief Defines GPU functions and kernels used by mpcd::SorterGPU
 */

#include "CellListGPU.cuh"

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
} // end namespace gpu
} // end namespace mpcd