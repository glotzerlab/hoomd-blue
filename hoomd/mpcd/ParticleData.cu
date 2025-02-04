// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/ParticleData.cu
 * \brief Defines GPU functions and kernels used by mpcd::ParticleData
 */

#ifdef ENABLE_MPI

#include "ParticleData.cuh"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#include <cub/device/device_partition.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#pragma GCC diagnostic pop

namespace hoomd
    {
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
 * \param d_comm_flags Communication flags (nonzero if particle should be migrated)
 * \param d_remove_ids Partitioned indexes of particles to remove (first) followed by keep (last)
 * \param n_remove Number of particles to remove
 * \param N Number of local particles
 *
 * Particles are removed using the result of cub::DevicePartition, which constructs
 * a list of particles to keep and remove.
 */
__global__ void remove_particles(mpcd::detail::pdata_element* d_out,
                                 Scalar4* d_pos,
                                 Scalar4* d_vel,
                                 unsigned int* d_tag,
                                 unsigned int* d_comm_flags,
                                 const unsigned int* d_remove_ids,
                                 const unsigned int n_remove,
                                 const unsigned int N)
    {
    // one thread per particle
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_remove)
        return;
    const unsigned int pid = d_remove_ids[idx];

    // pack a comm element
    mpcd::detail::pdata_element p;
    p.pos = d_pos[pid];
    p.vel = d_vel[pid];
    p.tag = d_tag[pid];
    p.comm_flag = d_comm_flags[pid];
    d_out[idx] = p;

    // now fill myself back in with another particle if that exists
    idx += n_remove;
    if (idx >= N)
        return;
    const unsigned int take_pid = d_remove_ids[idx];

    d_pos[pid] = d_pos[take_pid];
    d_vel[pid] = d_vel[take_pid];
    d_tag[pid] = d_tag[take_pid];
    d_comm_flags[pid] = d_comm_flags[take_pid];
    }

//! Kernel to transform communication flags for prefix sum
/*!
 * \param d_remove_flags Flag to remove (1) or keep (0) a particle (output)
 * \param d_comm_flags Communication flags
 * \param mask Bitwise mask for \a d_comm_flags
 * \param N Number of local particles
 *
 * Any communication flags that are bitwise AND with \a mask are transformed to
 * a 1 and stored in \a d_remove_flags, otherwise a 0 is set.
 */
__global__ void mark_removed_particles(unsigned char* d_remove_flags,
                                       const unsigned int* d_comm_flags,
                                       const unsigned int mask,
                                       const unsigned int N)
    {
    // one thread per particle
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    d_remove_flags[idx] = (d_comm_flags[idx] & mask) ? 1 : 0;
    }
    } // end namespace kernel
    } // end namespace gpu
    } // end namespace mpcd

/*!
 * \param d_remove_flags Flag to remove (1) or keep (0) a particle (output)
 * \param d_comm_flags Communication flags
 * \param mask Bitwise mask for \a d_comm_flags
 * \param N Number of local particles
 * \param block_size Number of threads per block
 *
 * \sa mpcd::gpu::kernel::mark_removed_particles
 */
cudaError_t mpcd::gpu::mark_removed_particles(unsigned char* d_remove_flags,
                                              const unsigned int* d_comm_flags,
                                              const unsigned int mask,
                                              const unsigned int N,
                                              const unsigned int block_size)
    {
    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::mark_removed_particles);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(N / run_block_size + 1);
    mpcd::gpu::kernel::mark_removed_particles<<<grid, run_block_size>>>(d_remove_flags,
                                                                        d_comm_flags,
                                                                        mask,
                                                                        N);
    return cudaSuccess;
    }

/*!
 * \param d_tmp Temporary storage
 * \param tmp_bytes Number of bytes in temporary storage
 * \param d_remove_flags Flags to remove (1) or keep (0) particles
 * \param d_remove_ids Partitioned indexes of particles to remove (first) or keep (last)
 * \param d_num_remove Number of particles to remove
 * \param N Number of particles
 *
 * \returns cudaSuccess on completion
 *
 * \b Implementation
 * This is a wrapper to a cub::DevicePartition::Flagged, and as such requires
 * two calls in order for the partitioning to take effect. In the first call,
 * temporary storage is sized and returned in \a tmp_bytes. The caller must then
 * allocate this memory into \a d_tmp, and call the method a second time. The
 * particle indexes are then partitioned into \a d_remove_ids, with
 * the particles to remove first in the array (in their original order), while
 * the kept particles are put into a reverse order at the end of the array.
 * The number of particles to keep is stored into \a d_num_remove.
 */
cudaError_t mpcd::gpu::partition_particles(void* d_tmp,
                                           size_t& tmp_bytes,
                                           const unsigned char* d_remove_flags,
                                           unsigned int* d_remove_ids,
                                           unsigned int* d_num_remove,
                                           const unsigned int N)
    {
    cub::CountingInputIterator<unsigned int> ids(0);
    cub::DevicePartition::Flagged(d_tmp,
                                  tmp_bytes,
                                  ids,
                                  d_remove_flags,
                                  d_remove_ids,
                                  d_num_remove,
                                  N);
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
 * \param d_remove_ids Partitioned indexes of particles to remove (first) or keep (last)
 * \param n_remove Number of particles to remove
 * \param N Current number of particles
 * \param block_size Number of threads per block
 *
 * \returns cudaSuccess on completion.
 *
 * \sa mpcd::gpu::kernel::remove_particles
 */
cudaError_t mpcd::gpu::remove_particles(mpcd::detail::pdata_element* d_out,
                                        Scalar4* d_pos,
                                        Scalar4* d_vel,
                                        unsigned int* d_tag,
                                        unsigned int* d_comm_flags,
                                        unsigned int* d_remove_ids,
                                        const unsigned int n_remove,
                                        const unsigned int N,
                                        const unsigned int block_size)
    {
    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::remove_particles);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(n_remove / run_block_size + 1);
    mpcd::gpu::kernel::remove_particles<<<grid, run_block_size>>>(d_out,
                                                                  d_pos,
                                                                  d_vel,
                                                                  d_tag,
                                                                  d_comm_flags,
                                                                  d_remove_ids,
                                                                  n_remove,
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
                              Scalar4* d_pos,
                              Scalar4* d_vel,
                              unsigned int* d_tag,
                              unsigned int* d_comm_flags,
                              const mpcd::detail::pdata_element* d_in,
                              const unsigned int mask)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_add_ptls)
        return;

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
                              Scalar4* d_pos,
                              Scalar4* d_vel,
                              unsigned int* d_tag,
                              unsigned int* d_comm_flags,
                              const mpcd::detail::pdata_element* d_in,
                              const unsigned int mask,
                              const unsigned int block_size)
    {
    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::add_particles);
    max_block_size = attr.maxThreadsPerBlock;

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
    } // end namespace hoomd
#endif // ENABLE_MPI
