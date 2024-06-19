// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/ParallelPlateGeometryFillerGPU.cu
 * \brief Defines GPU functions and kernels used by mpcd::ParallelPlateGeometryFillerGPU
 */

#include "ParticleDataUtilities.h"
#include "PlanarPoreGeometryFillerGPU.cuh"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"

namespace hoomd
    {
namespace mpcd
    {
namespace gpu
    {
namespace kernel
    {
/*!
 * \param d_pos Particle positions
 * \param d_vel Particle velocities
 * \param d_tag Particle tags
 * \param box Local simulation box
 * \param d_boxes List of 2d bounding boxes for filling
 * \param d_ranges Particle ranges for each box
 * \param num_boxes Number of bounding boxes to fill
 * \param N_tot Total number of particles
 * \param type Type of fill particles
 * \param first_tag First tag of filled particles
 * \param first_idx First (local) particle index of filled particles
 * \param vel_factor Scale factor for uniform normal velocities consistent with particle mass /
 * temperature \param timestep Current timestep \param seed User seed to PRNG for drawing velocities
 *
 * \b Implementation:
 *
 * Using one thread per particle, the thread is assigned to a fill range matching a 2d bounding box,
 * which defines a cuboid of volume to fill. The thread index is translated into a particle tag
 * and local particle index. A random position is drawn within the cuboid. A random velocity
 * is drawn consistent with the speed of the moving wall.
 */
__global__ void slit_pore_draw_particles(Scalar4* d_pos,
                                         Scalar4* d_vel,
                                         unsigned int* d_tag,
                                         const BoxDim box,
                                         const Scalar4* d_boxes,
                                         const uint2* d_ranges,
                                         const unsigned int num_boxes,
                                         const unsigned int N_tot,
                                         const unsigned int type,
                                         const unsigned int first_tag,
                                         const unsigned int first_idx,
                                         const Scalar vel_factor,
                                         const uint64_t timestep,
                                         const uint16_t seed)
    {
    // num_boxes should be 6, so this will all fit in shmem
    extern __shared__ char s_data[];
    Scalar4* s_boxes = (Scalar4*)(&s_data[0]);
    uint2* s_ranges = (uint2*)(&s_data[sizeof(Scalar4) * num_boxes]);
    for (unsigned int offset = 0; offset < num_boxes; offset += blockDim.x)
        {
        if (offset + threadIdx.x < num_boxes)
            {
            const unsigned int boxid = offset + threadIdx.x;
            s_boxes[boxid] = d_boxes[boxid];
            s_ranges[boxid] = d_ranges[boxid];
            }
        }
    __syncthreads();

    // one thread per particle
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_tot)
        return;

    // linear search for box matching thread (num_boxes is small)
    Scalar3 lo = box.getLo();
    Scalar3 hi = box.getHi();
    for (unsigned int boxid = 0; boxid < num_boxes; ++boxid)
        {
        const uint2 range = s_ranges[boxid];
        if (idx >= range.x && idx < range.y)
            {
            const Scalar4 fillbox = s_boxes[boxid];
            lo.x = fillbox.x;
            hi.x = fillbox.y;
            lo.y = fillbox.z;
            hi.y = fillbox.w;
            break;
            }
        }

    // particle tag and index
    const unsigned int tag = first_tag + idx;
    const unsigned int pidx = first_idx + idx;
    d_tag[pidx] = tag;

    // initialize random number generator for positions and velocity
    hoomd::RandomGenerator rng(
        hoomd::Seed(hoomd::RNGIdentifier::PlanarPoreGeometryFiller, timestep, seed),
        hoomd::Counter(tag));
    d_pos[pidx] = make_scalar4(hoomd::UniformDistribution<Scalar>(lo.x, hi.x)(rng),
                               hoomd::UniformDistribution<Scalar>(lo.y, hi.y)(rng),
                               hoomd::UniformDistribution<Scalar>(lo.z, hi.z)(rng),
                               __int_as_scalar(type));

    hoomd::NormalDistribution<Scalar> gen(vel_factor, 0.0);
    Scalar3 vel;
    gen(vel.x, vel.y, rng);
    vel.z = gen(rng);
    // TODO: should these be given zero net-momentum contribution (relative to the frame of
    // reference?)
    d_vel[pidx] = make_scalar4(vel.x, vel.y, vel.z, __int_as_scalar(mpcd::detail::NO_CELL));
    }
    } // end namespace kernel

/*!
 * \param d_pos Particle positions
 * \param d_vel Particle velocities
 * \param d_tag Particle tags
 * \param box Local simulation box
 * \param d_boxes List of 2d bounding boxes for filling
 * \param d_ranges Particle ranges for each box
 * \param num_boxes Number of bounding boxes to fill
 * \param N_tot Total number of particles
 * \param mass Mass of fill particles
 * \param type Type of fill particles
 * \param first_tag First tag of filled particles
 * \param first_idx First (local) particle index of filled particles
 * \param kT Temperature for fill particles
 * \param timestep Current timestep
 * \param seed User seed to PRNG for drawing velocities
 * \param block_size Number of threads per block
 *
 * \sa kernel::slit_pore_draw_particles
 */
cudaError_t slit_pore_draw_particles(Scalar4* d_pos,
                                     Scalar4* d_vel,
                                     unsigned int* d_tag,
                                     const BoxDim& box,
                                     const Scalar4* d_boxes,
                                     const uint2* d_ranges,
                                     const unsigned int num_boxes,
                                     const unsigned int N_tot,
                                     const Scalar mass,
                                     const unsigned int type,
                                     const unsigned int first_tag,
                                     const unsigned int first_idx,
                                     const Scalar kT,
                                     const uint64_t timestep,
                                     const uint16_t seed,
                                     const unsigned int block_size)
    {
    if (N_tot == 0)
        return cudaSuccess;

    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)kernel::slit_pore_draw_particles);
    max_block_size = attr.maxThreadsPerBlock;

    // precompute factor for rescaling the velocities since it is the same for all particles
    const Scalar vel_factor = fast::sqrt(kT / mass);

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(N_tot / run_block_size + 1);
    const size_t shared_bytes = num_boxes * (sizeof(Scalar4) + sizeof(uint2));
    kernel::slit_pore_draw_particles<<<grid, run_block_size, shared_bytes>>>(d_pos,
                                                                             d_vel,
                                                                             d_tag,
                                                                             box,
                                                                             d_boxes,
                                                                             d_ranges,
                                                                             num_boxes,
                                                                             N_tot,
                                                                             type,
                                                                             first_tag,
                                                                             first_idx,
                                                                             vel_factor,
                                                                             timestep,
                                                                             seed);

    return cudaSuccess;
    }

    } // end namespace gpu
    } // end namespace mpcd
    } // end namespace hoomd
