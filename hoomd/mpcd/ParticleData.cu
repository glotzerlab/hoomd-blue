// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/ParticleData.cu
 * \brief Defines GPU functions and kernels used by mpcd::ParticleData
 */

#ifdef ENABLE_MPI

#include "ParticleData.cuh"

#include "hoomd/extern/kernels/scan.cuh"

namespace mpcd
{
namespace gpu
{
namespace kernel
{
//! Kernel to partition particle data
/*!
 * \param N Number of local particles
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
__global__ void remove_particles(
    const unsigned int N,
    const Scalar4 *d_pos,
    const Scalar4 *d_vel,
    const unsigned int *d_tag,
    Scalar4 *d_pos_alt,
    Scalar4 *d_vel_alt,
    unsigned int *d_tag_alt,
    mpcd::detail::pdata_element *d_out,
    unsigned int *d_comm_flags,
    unsigned int *d_comm_flags_out,
    const unsigned int *d_scan)
    {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx >= N) return;

    bool remove = d_comm_flags[idx];

    unsigned int scan_remove = d_scan[idx];
    unsigned int scan_keep = idx - scan_remove;

    if (remove)
        {
        mpcd::detail::pdata_element p;
        p.pos = d_pos[idx];
        p.vel = d_vel[idx];
        p.tag = d_tag[idx];

        d_out[scan_remove] = p;
        d_comm_flags_out[scan_remove] = d_comm_flags[idx];

        // reset communication flags
        d_comm_flags[idx] = 0;
        }
    else
        {
        d_pos_alt[scan_keep] = d_pos[idx];
        d_vel_alt[scan_keep] = d_vel[idx];
        d_tag_alt[scan_keep] = d_tag[idx];
        }

    }

//! Kernel to transform communication flags for prefix sum
/*!
 * \param N Number of local particles
 * \param d_comm_flags Communication flags
 * \param d_tmp Temporary storage to hold transformation
 *
 * Any communication flags which are nonzero are transformed to a 1 and stored
 * in \a d_tmp.
 */
__global__ void select_sent_particles(unsigned int N,
                                      unsigned int *d_comm_flags,
                                      unsigned int *d_tmp)
    {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx >= N) return;
    d_tmp[idx] = d_comm_flags[idx] ? 1 : 0;
    }
} // end namespace kernel
} // end namespace gpu
} // end namespace mpcd

/*!
 * \param N Number of local particles
 * \param d_pos Device array of particle positions
 * \param d_vel Device array of particle velocities
 * \param d_tag Device array of particle tags
 * \param d_pos_alt Device array of particle positions (output)
 * \param d_vel_alt Device array of particle velocities (output)
 * \param d_tag_alt Device array of particle tags (output)
 * \param d_out Output array for packed particle data
 * \param d_comm_flags Communication flags (nonzero if particle should be migrated)
 * \param d_comm_flags_out Packed communication flags
 * \param max_n_out Maximum number of elements to write to output array
 * \param d_tmp Temporary storage space for device scan
 * \param mgpu_context Modern GPU context for exclusive scan
 *
 * \returns Number of elements marked for removal
 *
 * Particles are removed in three stages:
 * 1. Particles have their communication flags transformed into 0 or 1.
 * 2. An exclusive prefix sum is performed to map the indexes for packing, and
 *    count the number of particles to remove.
 * 3. Particles are removed if sufficient storage has been allocated for packing.
 */
unsigned int mpcd::gpu::remove_particles(unsigned int N,
                                         const Scalar4 *d_pos,
                                         const Scalar4 *d_vel,
                                         const unsigned int *d_tag,
                                         Scalar4 *d_pos_alt,
                                         Scalar4 *d_vel_alt,
                                         unsigned int *d_tag_alt,
                                         mpcd::detail::pdata_element *d_out,
                                         unsigned int *d_comm_flags,
                                         unsigned int *d_comm_flags_out,
                                         unsigned int max_n_out,
                                         unsigned int *d_tmp,
                                         mgpu::ContextPtr mgpu_context)
    {
    unsigned int n_out;

    // partition particle data into local and removed particles
    unsigned int block_size =512;
    unsigned int n_blocks = N/block_size+1;

    // select nonzero communication flags
    mpcd::gpu::kernel::select_sent_particles<<<n_blocks, block_size>>>(N, d_comm_flags, d_tmp);

    // perform a scan over the array of ones and zeroes
    mgpu::Scan<mgpu::MgpuScanTypeExc>(d_tmp,
        N, (unsigned int) 0, mgpu::plus<unsigned int>(),
        (unsigned int *)NULL, &n_out, d_tmp, *mgpu_context);

    // Don't write past end of buffer
    if (n_out <= max_n_out)
        {
        // partition particle data into local and removed particles
        unsigned int block_size =512;
        unsigned int n_blocks = N/block_size+1;

        mpcd::gpu::kernel::remove_particles<<<n_blocks, block_size>>>(N,
                                                                      d_pos,
                                                                      d_vel,
                                                                      d_tag,
                                                                      d_pos_alt,
                                                                      d_vel_alt,
                                                                      d_tag_alt,
                                                                      d_out,
                                                                      d_comm_flags,
                                                                      d_comm_flags_out,
                                                                      d_tmp);
        }

    // return elements written to output stream
    return n_out;
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
 * \param d_in Device array of packed input particle data
 * \param d_comm_flags Device array of communication flags
 *
 * Particle data is appended to the end of the particle data arrays from the
 * packed buffer. Communication flags of new particles are zeroed.
 */
__global__ void add_particles(unsigned int old_nparticles,
                              unsigned int num_add_ptls,
                              Scalar4 *d_pos,
                              Scalar4 *d_vel,
                              unsigned int *d_tag,
                              const mpcd::detail::pdata_element *d_in,
                              unsigned int *d_comm_flags)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_add_ptls) return;

    mpcd::detail::pdata_element p = d_in[idx];

    unsigned int add_idx = old_nparticles + idx;
    d_pos[add_idx] = p.pos;
    d_vel[add_idx] = p.vel;
    d_tag[add_idx] = p.tag;
    d_comm_flags[add_idx] = 0;
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
 * \param d_in Device array of packed input particle data
 * \param d_comm_flags Device array of communication flags
 *
 * Particle data is appended to the end of the particle data arrays from the
 * packed buffer. Communication flags of new particles are zeroed.
 */
void mpcd::gpu::add_particles(unsigned int old_nparticles,
                              unsigned int num_add_ptls,
                              Scalar4 *d_pos,
                              Scalar4 *d_vel,
                              unsigned int *d_tag,
                              const mpcd::detail::pdata_element *d_in,
                              unsigned int *d_comm_flags)
    {
    unsigned int block_size = 512;
    unsigned int n_blocks = num_add_ptls/block_size + 1;

    mpcd::gpu::kernel::add_particles<<<n_blocks, block_size>>>(old_nparticles,
                                                               num_add_ptls,
                                                               d_pos,
                                                               d_vel,
                                                               d_tag,
                                                               d_in,
                                                               d_comm_flags);
    }

#endif // ENABLE_MPI
