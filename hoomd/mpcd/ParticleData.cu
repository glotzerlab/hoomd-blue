// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/ParticleData.cu
 * \brief Defines GPU functions and kernels used by mpcd::ParticleData
 */

#ifdef ENABLE_MPI

#include "ParticleData.cuh"

#include "hoomd/extern/cub/cub/device/device_scan.cuh"

namespace mpcd
{
namespace gpu
{
namespace kernel
{
//! Kernel to partition particle data
/*!
 * \param d_out
 * \param mask
 * \param d_pos Device array of particle positions
 * \param d_vel Device array of particle velocities
 * \param d_tag Device array of particle tags
 * \param d_pos_alt Device array of particle positions (output)
 * \param d_vel_alt Device array of particle velocities (output)
 * \param d_tag_alt Device array of particle tags (output)
 * \param d_out Output array for packed particle data
 * \param d_comm_flags Communication flags (nonzero if particle should be migrated)
 * \param d_comm_flags_out Packed communication flags
 * \param d_scan Result of exclusive prefix sum
 * \param N Number of local particles
 *
 * Particles are removed by performing a selection using the result of an
 * exclusive prefix sum, stored in \a d_scan. The scan recovers the indexes
 * of the particles. A simple example illustrating the implementation follows:
 *
 * \verbatim
 * Particles:   0 1 2 3 4
 * Flags:       0|1 1|0 0
 * d_scan       0|0 1|2 2
 *              ---------
 * scan_keep:   0|1 1|1 2
 *              ---------
 * keep:        0,3,4 -> 0,1,2
 * remove:      1,2 -> 0,1
 * \endverbatim
 */
__global__ void remove_particles(mpcd::detail::pdata_element *d_out,
                                 const unsigned int mask,
                                 const Scalar4 *d_pos,
                                 const Scalar4 *d_vel,
                                 const unsigned int *d_tag,
                                 const unsigned int *d_comm_flags,
                                 Scalar4 *d_pos_alt,
                                 Scalar4 *d_vel_alt,
                                 unsigned int *d_tag_alt,
                                 unsigned int *d_comm_flags_alt,
                                 const unsigned int *d_scan,
                                 const unsigned int N)
    {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx >= N) return;

    unsigned int scan_remove = d_scan[idx];
    unsigned int scan_keep = idx - scan_remove;

    if (d_comm_flags[idx] & mask)
        {
        mpcd::detail::pdata_element p;
        p.pos = d_pos[idx];
        p.vel = d_vel[idx];
        p.tag = d_tag[idx];
        p.comm_flag = d_comm_flags[idx];

        d_out[scan_remove] = p;
        }
    else
        {
        d_pos_alt[scan_keep] = d_pos[idx];
        d_vel_alt[scan_keep] = d_vel[idx];
        d_tag_alt[scan_keep] = d_tag[idx];
        d_comm_flags_alt[scan_keep] = d_comm_flags[idx];
        }
    }

//! Kernel to transform communication flags for prefix sum
/*!
 * \param d_tmp Temporary storage to hold transformation (output)
 * \param d_comm_flags Communication flags
 * \param mask Bitwise mask for \a d_comm_flags
 * \param N Number of local particles
 *
 * Any communication flags that are bitwise AND with \a mask are transformed to
 * a 1 and stored in \a d_tmp.
 */
__global__ void mark_removed_particles(unsigned int *d_tmp,
                                       const unsigned int *d_comm_flags,
                                       const unsigned int mask,
                                       const unsigned int N)
    {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx >= N) return;
    d_tmp[idx] = (d_comm_flags[idx] & mask) ? 1 : 0;
    }
} // end namespace kernel
} // end namespace gpu
} // end namespace mpcd


cudaError_t mpcd::gpu::mark_removed_particles(unsigned int *d_tmp_flag,
                                              const unsigned int *d_comm_flags,
                                              const unsigned int mask,
                                              const unsigned int N,
                                              const unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::mark_removed_particles);
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(N / run_block_size + 1);
    mpcd::gpu::kernel::mark_removed_particles<<<grid, run_block_size>>>(d_tmp_flag,
                                                                        d_comm_flags,
                                                                        mask,
                                                                        N);
    return cudaSuccess;
    }

cudaError_t mpcd::gpu::scan_removed_particles(void *d_tmp,
                                              size_t& tmp_bytes,
                                              unsigned int *d_tmp_flag,
                                              const unsigned int N)
    {
    // in place scan is supported
    // https://groups.google.com/d/msg/cub-users/pEsYSNc2Rn4/_4ulOuwWDcoJ
    cub::DeviceScan::ExclusiveSum(d_tmp, tmp_bytes, d_tmp_flag, d_tmp_flag, N);

    return cudaSuccess;
    }

/*!
 * \param d_out Output array for packed particle data
 * \param mask Bitwise mask for \a d_comm_flags
 * \param d_pos Device array of particle positions
 * \param d_vel Device array of particle velocities
 * \param d_tag Device array of particle tags
 * \param d_comm_flags Device array of communication flags
 * \param d_pos_alt Device array of particle positions (output)
 * \param d_vel_alt Device array of particle velocities (output)
 * \param d_tag_alt Device array of particle tags (output)
 * \param d_comm_flags_alt Device array of communication flags (output)
 * \param d_scan Output from device scan of temporary flags
 * \param N Current number of particles
 *
 * \returns cudaSuccess on completion.
 */
cudaError_t mpcd::gpu::remove_particles(mpcd::detail::pdata_element *d_out,
                                        const unsigned int mask,
                                        const Scalar4 *d_pos,
                                        const Scalar4 *d_vel,
                                        const unsigned int *d_tag,
                                        const unsigned int *d_comm_flags,
                                        Scalar4 *d_pos_alt,
                                        Scalar4 *d_vel_alt,
                                        unsigned int *d_tag_alt,
                                        unsigned int *d_comm_flags_alt,
                                        unsigned int *d_scan,
                                        const unsigned int N)
    {
    // partition particle data into local and removed particles
    unsigned int block_size = 512;
    unsigned int n_blocks = N/block_size+1;

    mpcd::gpu::kernel::remove_particles<<<n_blocks, block_size>>>(d_out,
                                                                  mask,
                                                                  d_pos,
                                                                  d_vel,
                                                                  d_tag,
                                                                  d_comm_flags,
                                                                  d_pos_alt,
                                                                  d_vel_alt,
                                                                  d_tag_alt,
                                                                  d_comm_flags_alt,
                                                                  d_scan,
                                                                  N);
    return cudaSuccess;
    }


namespace mpcd
{
namespace gpu
{
namespace kernel
{
//! Kernel to partition particle data
/*!
 * \param old_nparticles old local particle count
 * \param num_add_ptls Number of particles in input array
 * \param d_pos Device array of particle positions
 * \param d_vel Device array of particle velocities
 * \param d_tag Device array of particle tags
 * \param d_comm_flags Device array of communication flags
 * \param d_in Device array of packed input particle data
 * \param mask Bitwise mask for received particles to unmask
 *
 * Particle data is appended to the end of the particle data arrays from the
 * packed buffer. Communication flags of new particles are unmasked.
 */
__global__ void add_particles(unsigned int old_nparticles,
                              unsigned int num_add_ptls,
                              Scalar4 *d_pos,
                              Scalar4 *d_vel,
                              unsigned int *d_tag,
                              unsigned int *d_comm_flags,
                              const mpcd::detail::pdata_element *d_in,
                              const unsigned int mask)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_add_ptls) return;

    mpcd::detail::pdata_element p = d_in[idx];

    unsigned int add_idx = old_nparticles + idx;
    d_pos[add_idx] = p.pos;
    d_vel[add_idx] = p.vel;
    d_tag[add_idx] = p.tag;
    d_comm_flags[add_idx] = p.comm_flag & ~mask;
    }
} // end namespace kernel
} // end namespace gpu
} // end namespace mpcd

/*!
 * \param old_nparticles old local particle count
 * \param num_add_ptls Number of particles in input array
 * \param d_pos Device array of particle positions
 * \param d_vel Device array of particle velocities
 * \param d_tag Device array of particle tags
 * \param d_comm_flags Device array of communication flags
 * \param d_in Device array of packed input particle data
 * \param mask Bitwise mask for received particles to unmask
 *
 * Particle data is appended to the end of the particle data arrays from the
 * packed buffer. Communication flags of new particles are unmasked.
 */
void mpcd::gpu::add_particles(unsigned int old_nparticles,
                              unsigned int num_add_ptls,
                              Scalar4 *d_pos,
                              Scalar4 *d_vel,
                              unsigned int *d_tag,
                              unsigned int *d_comm_flags,
                              const mpcd::detail::pdata_element *d_in,
                              const unsigned int mask)
    {
    unsigned int block_size = 512;
    unsigned int n_blocks = num_add_ptls/block_size + 1;

    mpcd::gpu::kernel::add_particles<<<n_blocks, block_size>>>(old_nparticles,
                                                               num_add_ptls,
                                                               d_pos,
                                                               d_vel,
                                                               d_tag,
                                                               d_comm_flags,
                                                               d_in,
                                                               mask);
    }

#endif // ENABLE_MPI
