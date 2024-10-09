// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/RejectionVirtualParticleFillerGPU.cu
 * \brief Definition of CUDA kernels for mpcd::RejectionVirtualParticleFillerGPU
 */

#include <hipcub/hipcub.hpp>

#include "RejectionVirtualParticleFillerGPU.cuh"

namespace hoomd
    {
namespace mpcd
    {
namespace gpu
    {
namespace kernel
    {

/*!
 * \b implementation
 * Using one thread per particle, we assign the particle position, velocity and tags using the
 * compacted indices array as an input.
 */
__global__ void copy_virtual_particles(unsigned int* d_keep_indices,
                                       Scalar4* d_pos,
                                       Scalar4* d_vel,
                                       unsigned int* d_tags,
                                       const Scalar4* d_tmp_pos,
                                       const Scalar4* d_tmp_vel,
                                       const unsigned int first_idx,
                                       const unsigned int first_tag,
                                       const unsigned int n_virtual,
                                       const unsigned int block_size)
    {
    // one thread per virtual particle
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_virtual)
        return;

    // d_keep_indices holds accepted particle indices from the temporary arrays
    const unsigned int tmp_pidx = d_keep_indices[idx];
    const unsigned int pidx = first_idx + idx;
    d_pos[pidx] = d_tmp_pos[tmp_pidx];
    d_vel[pidx] = d_tmp_vel[tmp_pidx];
    d_tags[pidx] = first_tag + idx;
    }

    } // end namespace kernel

cudaError_t __attribute__((visibility("default")))
compact_virtual_particle_indices(void* d_tmp,
                                 size_t& tmp_bytes,
                                 const bool* d_keep_particles,
                                 const unsigned int num_particles,
                                 unsigned int* d_keep_indices,
                                 unsigned int* d_num_keep)
    {
    cub::CountingInputIterator<int> itr(0);
    cub::DeviceSelect::Flagged(d_tmp,
                               tmp_bytes,
                               itr,
                               d_keep_particles,
                               d_keep_indices,
                               d_num_keep,
                               num_particles);
    return cudaSuccess;
    }

cudaError_t __attribute__((visibility("default")))
copy_virtual_particles(unsigned int* d_keep_indices,
                       Scalar4* d_pos,
                       Scalar4* d_vel,
                       unsigned int* d_tags,
                       const Scalar4* d_tmp_pos,
                       const Scalar4* d_tmp_vel,
                       const unsigned int first_idx,
                       const unsigned int first_tag,
                       const unsigned int n_virtual,
                       const unsigned int block_size)
    {
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::copy_virtual_particles);
    const unsigned int max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(n_virtual / run_block_size + 1);
    mpcd::gpu::kernel::copy_virtual_particles<<<grid, run_block_size>>>(d_keep_indices,
                                                                        d_pos,
                                                                        d_vel,
                                                                        d_tags,
                                                                        d_tmp_pos,
                                                                        d_tmp_vel,
                                                                        first_idx,
                                                                        first_tag,
                                                                        n_virtual,
                                                                        block_size);

    return cudaSuccess;
    }
    } // namespace gpu
    } // namespace mpcd
    } // namespace hoomd
