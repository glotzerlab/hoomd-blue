// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/SorterGPU.cu
 * \brief Defines GPU functions and kernels used by mpcd::SorterGPU
 */

#include "CellListGPU.cuh"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#pragma GCC diagnostic pop

namespace hoomd
    {
namespace mpcd
    {
namespace gpu
    {
namespace kernel
    {
//! Kernel to apply sorted particle order
/*!
 * \param d_pos_alt Alternate array of particle positions (output)
 * \param d_vel_alt Alternate array of particle velocities (output)
 * \param d_tag_alt Alternate array of particle tags (output)
 * \param d_pos Particle positions
 * \param d_vel Particle velocities
 * \param d_tag Particle tags
 * \param d_order Mapping of new particle index onto old particle index
 * \param N Number of particles
 *
 * \b Implementation
 * Using one thread per particle, particle data is reordered from the old arrays
 * into the new arrays. This coalesces writes but fragments reads.
 */
__global__ void sort_apply(Scalar4* d_pos_alt,
                           Scalar4* d_vel_alt,
                           unsigned int* d_tag_alt,
                           const Scalar4* d_pos,
                           const Scalar4* d_vel,
                           const unsigned int* d_tag,
                           const unsigned int* d_order,
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

//! Kernel to set the empty-cell-entry sentinel
/*!
 * \param d_order Particle order to fill with indices
 * \param d_cell_id auxiliary array to sort cell id of mpcd particles
 * \param d_vel velocities and cell ids of mpcd particles
 * \param N Number of particles
 *
 * \b Implementation
 * Fills order with indices 0 to N and the auxiliary array d_cell_id with
 * the cell ids of the mpcd particles in preparation to sort the values
 * in d_order by the cell ids.
 */
__global__ void
set_order(unsigned int* d_order, unsigned int* d_cell_id, const Scalar4* d_vel, unsigned int N)
    {
    // one thread per particle
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    d_order[idx] = idx;
    d_cell_id[idx] = __scalar_as_int(d_vel[idx].w);
    }
    } // end namespace kernel

/*!
 * \param d_pos_alt Alternate array of particle positions (output)
 * \param d_vel_alt Alternate array of particle velocities (output)
 * \param d_tag_alt Alternate array of particle tags (output)
 * \param d_pos Particle positions
 * \param d_vel Particle velocities
 * \param d_tag Particle tags
 * \param d_order Mapping of new particle index onto old particle index
 * \param N Number of particles
 * \param block_size Number of threads per block
 *
 * \returns cudaSuccess on completion
 *
 * \sa mpcd::gpu::kernel::sort_apply
 */
cudaError_t sort_apply(Scalar4* d_pos_alt,
                       Scalar4* d_vel_alt,
                       unsigned int* d_tag_alt,
                       const Scalar4* d_pos,
                       const Scalar4* d_vel,
                       const unsigned int* d_tag,
                       const unsigned int* d_order,
                       const unsigned int N,
                       const unsigned int block_size)
    {
    if (N == 0)
        return cudaSuccess;

    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::sort_apply);
    max_block_size = attr.maxThreadsPerBlock;

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

/*!
 * \param d_order Compacted MPCD particle indexes in cell-list order (output)
 * \param d_cell_id auxiliary array to sort cell id of each mpcd particles
 * \param d_vel Cell indexes of the MPCD particles
 * \param N Number of particles
 * \param block_size Number of threads per block

 * \returns cudaSuccess on completion
 *
 * \sa mpcd::gpu::kernel::set_order
 */
cudaError_t set_order(unsigned int* d_order,
                      unsigned int* d_cell_id,
                      const Scalar4* d_vel,
                      const unsigned int N,
                      const unsigned int block_size)
    {
    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::set_order);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(N / run_block_size + 1);
    mpcd::gpu::kernel::set_order<<<grid, run_block_size>>>(d_order, d_cell_id, d_vel, N);
    return cudaSuccess;
    }

/*!
 * \param d_order Compacted MPCD particle indexes in cell-list order (output)
 * \param d_cell_id auxiliary array to sort cell id of each mpcd particles
 * \param N Number of particles

 * \returns cudaSuccess on completion
 */
cudaError_t compute_order(unsigned int* d_order, unsigned int* d_cell_id, const unsigned int N)
    {
    thrust::device_ptr<unsigned int> order(d_order);
    thrust::device_ptr<unsigned int> cell_id(d_cell_id);
    thrust::sort_by_key(thrust::device, cell_id, cell_id + N, order);
    return cudaSuccess;
    }
    } // end namespace gpu
    } // end namespace mpcd

    } // end namespace hoomd
