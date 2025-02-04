// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/ParallelPlateGeometryFillerGPU.cu
 * \brief Defines GPU functions and kernels used by mpcd::ParallelPlateGeometryFillerGPU
 */

#include "ParallelPlateGeometryFillerGPU.cuh"
#include "ParticleDataUtilities.h"
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
 * \param geom Slit geometry to fill
 * \param y_min Lower bound to lower fill region
 * \param y_max Upper bound to upper fill region
 * \param box Local simulation box
 * \param type Type of fill particles
 * \param N_lo Number of particles to fill in lower region
 * \param N_hi Number of particles to fill in upper region
 * \param first_tag First tag of filled particles
 * \param first_idx First (local) particle index of filled particles
 * \param vel_factor Scale factor for uniform normal velocities consistent with particle mass /
 * temperature
 * \param timestep Current timestep
 * \param seed User seed to PRNG for drawing velocities
 * \param filler_id ID for this filler
 *
 * \b Implementation:
 *
 * Using one thread per particle (in both slabs), the thread is assigned to fill either the lower
 * or upper region. This defines a local cuboid of volume to fill. The thread index is translated
 * into a particle tag and local particle index. A random position is drawn within the cuboid. A
 * random velocity is drawn consistent with the speed of the moving wall.
 */
__global__ void slit_draw_particles(Scalar4* d_pos,
                                    Scalar4* d_vel,
                                    unsigned int* d_tag,
                                    const mpcd::ParallelPlateGeometry geom,
                                    const Scalar y_min,
                                    const Scalar y_max,
                                    const BoxDim box,
                                    const unsigned int type,
                                    const unsigned int N_lo,
                                    const unsigned int N_tot,
                                    const unsigned int first_tag,
                                    const unsigned int first_idx,
                                    const Scalar vel_factor,
                                    const uint64_t timestep,
                                    const uint16_t seed,
                                    const unsigned int filler_id)
    {
    // one thread per particle
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_tot)
        return;

    // determine the fill region based on current index
    signed char sign = (idx >= N_lo) - (idx < N_lo);
    Scalar3 lo = box.getLo();
    Scalar3 hi = box.getHi();
    if (sign == -1) // bottom
        {
        lo.y = y_min;
        hi.y = -geom.getH();
        }
    else // top
        {
        lo.y = geom.getH();
        hi.y = y_max;
        }

    // particle tag and index
    const unsigned int tag = first_tag + idx;
    const unsigned int pidx = first_idx + idx;
    d_tag[pidx] = tag;

    // initialize random number generator for positions and velocity
    hoomd::RandomGenerator rng(
        hoomd::Seed(hoomd::RNGIdentifier::VirtualParticleFiller, timestep, seed),
        hoomd::Counter(tag, filler_id));
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
    d_vel[pidx] = make_scalar4(vel.x + sign * geom.getSpeed(),
                               vel.y,
                               vel.z,
                               __int_as_scalar(mpcd::detail::NO_CELL));
    }
    } // end namespace kernel

/*!
 * \param d_pos Particle positions
 * \param d_vel Particle velocities
 * \param d_tag Particle tags
 * \param geom Slit geometry to fill
 * \param y_min Lower bound to lower fill region
 * \param y_max Upper bound to upper fill region
 * \param box Local simulation box
 * \param mass Mass of fill particles
 * \param type Type of fill particles
 * \param N_lo Number of particles to fill in lower region
 * \param N_hi Number of particles to fill in upper region
 * \param first_tag First tag of filled particles
 * \param first_idx First (local) particle index of filled particles
 * \param kT Temperature for fill particles
 * \param timestep Current timestep
 * \param seed User seed to PRNG for drawing velocities
 * \param filler_id ID for this filler
 * \param block_size Number of threads per block
 *
 * \sa kernel::slit_draw_particles
 */
cudaError_t slit_draw_particles(Scalar4* d_pos,
                                Scalar4* d_vel,
                                unsigned int* d_tag,
                                const mpcd::ParallelPlateGeometry& geom,
                                const Scalar y_min,
                                const Scalar y_max,
                                const BoxDim& box,
                                const Scalar mass,
                                const unsigned int type,
                                const unsigned int N_lo,
                                const unsigned int N_hi,
                                const unsigned int first_tag,
                                const unsigned int first_idx,
                                const Scalar kT,
                                const uint64_t timestep,
                                const uint16_t seed,
                                const unsigned int filler_id,
                                const unsigned int block_size)
    {
    const unsigned int N_tot = N_lo + N_hi;
    if (N_tot == 0)
        return cudaSuccess;

    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)kernel::slit_draw_particles);
    max_block_size = attr.maxThreadsPerBlock;

    // precompute factor for rescaling the velocities since it is the same for all particles
    const Scalar vel_factor = fast::sqrt(kT / mass);

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(N_tot / run_block_size + 1);
    kernel::slit_draw_particles<<<grid, run_block_size>>>(d_pos,
                                                          d_vel,
                                                          d_tag,
                                                          geom,
                                                          y_min,
                                                          y_max,
                                                          box,
                                                          type,
                                                          N_lo,
                                                          N_tot,
                                                          first_tag,
                                                          first_idx,
                                                          vel_factor,
                                                          timestep,
                                                          seed,
                                                          filler_id);

    return cudaSuccess;
    }

    } // end namespace gpu
    } // end namespace mpcd
    } // end namespace hoomd
