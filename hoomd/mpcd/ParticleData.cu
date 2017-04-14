// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/ParticleData.cu
 * \brief Defines GPU functions and kernels used by mpcd::ParticleData
 */

#ifdef ENABLE_MPI

#include "ParticleData.cuh"

#include "hoomd/extern/cub/cub/device/device_partition.cuh"
#include "hoomd/extern/cub/cub/thread/thread_load.cuh"

namespace mpcd
{
namespace gpu
{
namespace kernel
{
//! Kernel to partition particle data
/*!
 * \param d_out Packed output buffer
 * \param d_pos Device array of particle positions
 * \param d_vel Device array of particle velocities
 * \param d_tag Device array of particle tags
 * \param d_pos_alt Device array of particle positions (output)
 * \param d_vel_alt Device array of particle velocities (output)
 * \param d_tag_alt Device array of particle tags (output)
 * \param d_out Output array for packed particle data
 * \param d_comm_flags Communication flags (nonzero if particle should be migrated)
 * \param d_comm_flags_out Packed communication flags
 * \param d_keep_ids Partitioned indexes of particles to keep (bottom) or remove (top)
 * \param n_keep Number of particles to keep
 * \param N Number of local particles
 *
 * Particles are removed using the result of cub::DevicePartition, which constructs
 * a list of particles to keep and remove.
 */
__global__ void remove_particles(mpcd::detail::pdata_element *d_out,
                                 const Scalar4 *d_pos,
                                 const Scalar4 *d_vel,
                                 const unsigned int *d_tag,
                                 const unsigned int *d_comm_flags,
                                 Scalar4 *d_pos_alt,
                                 Scalar4 *d_vel_alt,
                                 unsigned int *d_tag_alt,
                                 unsigned int *d_comm_flags_alt,
                                 const unsigned int *d_keep_ids,
                                 const unsigned int n_keep,
                                 const unsigned int N)
    {
    // one thread per particle
    const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // read static data out of textures
    const unsigned int pid = cub::ThreadLoad<cub::LOAD_LDG>(d_keep_ids + idx);
    const Scalar4 pos = cub::ThreadLoad<cub::LOAD_LDG>(d_pos + pid);
    const Scalar4 vel = cub::ThreadLoad<cub::LOAD_LDG>(d_vel + pid);
    const unsigned int tag = cub::ThreadLoad<cub::LOAD_LDG>(d_tag + pid);
    const unsigned int flag = cub::ThreadLoad<cub::LOAD_LDG>(d_comm_flags + pid);

    if (idx >= n_keep)
        {
        mpcd::detail::pdata_element p;
        p.pos = pos;
        p.vel = vel;
        p.tag = tag;
        p.comm_flag = flag;
        d_out[idx - n_keep] = p;
        }
    else
        {
        d_pos_alt[idx] = pos;
        d_vel_alt[idx] = vel;
        d_tag_alt[idx] = tag;
        d_comm_flags_alt[idx] = flag;
        }
    }

//! Kernel to transform communication flags for prefix sum
/*!
 * \param d_keep_flags Flag to keep (1) or remove (0) a particle (output)
 * \param d_tmp_ids Particle indexes that will later be partitioned (0 to \a N-1)
 * \param d_comm_flags Communication flags
 * \param mask Bitwise mask for \a d_comm_flags
 * \param N Number of local particles
 *
 * Any communication flags that are bitwise AND with \a mask are transformed to
 * a 0 and stored in \a d_keep_flags, otherwise a 1 is set. The particle indexes
 * are also filled into \a d_tmp_ids.
 */
__global__ void mark_removed_particles(unsigned char *d_keep_flags,
                                       unsigned int *d_tmp_ids,
                                       const unsigned int *d_comm_flags,
                                       const unsigned int mask,
                                       const unsigned int N)
    {
    // one thread per particle
    const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= N) return;

    d_tmp_ids[idx] = idx;
    d_keep_flags[idx] = (d_comm_flags[idx] & mask) ? 0 : 1;
    }
} // end namespace kernel
} // end namespace gpu
} // end namespace mpcd

/*!
 * \param d_keep_flags Flag to keep (1) or remove (0) a particle (output)
 * \param d_tmp_ids Particle indexes that will later be partitioned (0 to \a N-1)
 * \param d_comm_flags Communication flags
 * \param mask Bitwise mask for \a d_comm_flags
 * \param N Number of local particles
 * \param block_size Number of threads per block
 *
 * \sa mpcd::gpu::kernel::mark_removed_particles
 */
cudaError_t mpcd::gpu::mark_removed_particles(unsigned char *d_keep_flags,
                                              unsigned int *d_tmp_ids,
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
    mpcd::gpu::kernel::mark_removed_particles<<<grid, run_block_size>>>(d_keep_flags,
                                                                        d_tmp_ids,
                                                                        d_comm_flags,
                                                                        mask,
                                                                        N);
    return cudaSuccess;
    }

/*!
 * \param d_tmp Temporary storage
 * \param tmp_bytes Number of bytes in temporary storage
 * \param d_tmp_ids Temporary particle indexes to partition
 * \param d_keep_flags Flags to keep (1) or remove (0) particles
 * \param d_keep_ids Partitioned indexes of particles to keep (bottom) or remove (top)
 * \param d_num_keep Number of particles to keep
 * \param N Number of particles
 *
 * \returns cudaSuccess on completion
 *
 * \b Implementation
 * This is a wrapper to a cub::DevicePartition::Flagged, and as such requires
 * two calls in order for the partitioning to take effect. In the first call,
 * temporary storage is sized and returned in \a tmp_bytes. The caller must then
 * allocate this memory into \a d_tmp, and call the method a second time. The
 * particle indexes in \a d_tmp_ids are then partition into \a d_keep_ids, with
 * the particles to keep first in the array (in their original order), while
 * the removed particles are put into a reverse order at the end of the array.
 * The number of particles to keep is stored into \a d_num_keep.
 */
cudaError_t mpcd::gpu::partition_particles(void *d_tmp,
                                           size_t& tmp_bytes,
                                           const unsigned int *d_tmp_ids,
                                           const unsigned char *d_keep_flags,
                                           unsigned int *d_keep_ids,
                                           unsigned int *d_num_keep,
                                           const unsigned int N)
    {
    cub::DevicePartition::Flagged(d_tmp, tmp_bytes, d_tmp_ids, d_keep_flags, d_keep_ids, d_num_keep, N);
    return cudaSuccess;
    }

/*!
 * \param d_out Output array for packed particle data
 * \param d_pos Device array of particle positions
 * \param d_vel Device array of particle velocities
 * \param d_tag Device array of particle tags
 * \param d_comm_flags Device array of communication flags
 * \param d_pos_alt Device array of particle positions (output)
 * \param d_vel_alt Device array of particle velocities (output)
 * \param d_tag_alt Device array of particle tags (output)
 * \param d_comm_flags_alt Device array of communication flags (output)
 * \param d_keep_ids Partitioned indexes of particles to keep (bottom) or remove (top)
 * \param n_keep Number of particles to keep
 * \param N Current number of particles
 * \param block_size Number of threads per block
 *
 * \returns cudaSuccess on completion.
 *
 * \sa mpcd::gpu::kernel::remove_particles
 */
cudaError_t mpcd::gpu::remove_particles(mpcd::detail::pdata_element *d_out,
                                        const Scalar4 *d_pos,
                                        const Scalar4 *d_vel,
                                        const unsigned int *d_tag,
                                        const unsigned int *d_comm_flags,
                                        Scalar4 *d_pos_alt,
                                        Scalar4 *d_vel_alt,
                                        unsigned int *d_tag_alt,
                                        unsigned int *d_comm_flags_alt,
                                        unsigned int *d_keep_ids,
                                        const unsigned int n_keep,
                                        const unsigned int N,
                                        const unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::remove_particles);
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(N / run_block_size + 1);
    mpcd::gpu::kernel::remove_particles<<<grid, run_block_size>>>(d_out,
                                                                  d_pos,
                                                                  d_vel,
                                                                  d_tag,
                                                                  d_comm_flags,
                                                                  d_pos_alt,
                                                                  d_vel_alt,
                                                                  d_tag_alt,
                                                                  d_comm_flags_alt,
                                                                  d_keep_ids,
                                                                  n_keep,
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
 * \param block_size Number of threads per block
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
                              const unsigned int mask,
                              const unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::add_particles);
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(num_add_ptls / run_block_size + 1);
    mpcd::gpu::kernel::add_particles<<<grid, run_block_size>>>(old_nparticles,
                                                               num_add_ptls,
                                                               d_pos,
                                                               d_vel,
                                                               d_tag,
                                                               d_comm_flags,
                                                               d_in,
                                                               mask);
    }

#endif // ENABLE_MPI
