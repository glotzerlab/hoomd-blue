// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/RejectionVirtualParticleFillerGPU.h
 * \brief Declaration and definition of RejectionVirtualParticleFillerGPU
 */

#ifndef MPCD_REJECTION_VIRTUAL_PARTICLE_FILLER_GPU_CUH_
#define MPCD_REJECTION_VIRTUAL_PARTICLE_FILLER_GPU_CUH_

#include <cuda_runtime.h>
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
#endif

#include "ParticleDataUtilities.h"
#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"

#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"

namespace hoomd
    {
namespace mpcd
    {
namespace gpu
    {

//! Common arguments passed to all geometry filling kernels
struct draw_virtual_particles_args_t
    {
    //! Constructor
    draw_virtual_particles_args_t(Scalar4* _d_tmp_pos,
                                  Scalar4* _d_tmp_vel,
                                  bool* _d_keep_particles,
                                  const Scalar3 _lo,
                                  const Scalar3 _hi,
                                  const unsigned int _first_tag,
                                  const Scalar _vel_factor,
                                  const unsigned int _type,
                                  const unsigned int _N_virt_max,
                                  const unsigned int _timestep,
                                  const unsigned int _seed,
                                  const unsigned int _filler_id,
                                  const unsigned int _block_size)
        : d_tmp_pos(_d_tmp_pos), d_tmp_vel(_d_tmp_vel), d_keep_particles(_d_keep_particles),
          lo(_lo), hi(_hi), first_tag(_first_tag), vel_factor(_vel_factor), type(_type),
          N_virt_max(_N_virt_max), timestep(_timestep), seed(_seed), filler_id(_filler_id),
          block_size(_block_size)
        {
        }

    Scalar4* d_tmp_pos;
    Scalar4* d_tmp_vel;
    bool* d_keep_particles;
    const Scalar3 lo;
    const Scalar3 hi;
    const unsigned int first_tag;
    const Scalar vel_factor;
    const unsigned int type;
    const unsigned int N_virt_max;
    const unsigned int timestep;
    const unsigned int seed;
    const unsigned int filler_id;
    const unsigned int block_size;
    };

// Function declarations
template<class Geometry>
cudaError_t draw_virtual_particles(const draw_virtual_particles_args_t& args, const Geometry& geom);

cudaError_t compact_virtual_particle_indices(void* d_tmp,
                                             size_t& tmp_bytes,
                                             const bool* d_keep_particles,
                                             const unsigned int num_particles,
                                             unsigned int* d_keep_indices,
                                             unsigned int* d_num_keep);

cudaError_t copy_virtual_particles(unsigned int* d_keep_indices,
                                   Scalar4* d_pos,
                                   Scalar4* d_vel,
                                   unsigned int* d_tags,
                                   const Scalar4* d_tmp_pos,
                                   const Scalar4* d_tmp_vel,
                                   const unsigned int first_idx,
                                   const unsigned int first_tag,
                                   const unsigned int n_virtual,
                                   const unsigned int block_size);

#ifdef __HIPCC__
namespace kernel
    {

//! Kernel to draw virtual particles outside any given geometry
/*!
 * \param d_tmp_pos Temporary positions
 * \param d_tmp_vel Temporary velocities
 * \param d_keep_particles Particle tracking - in/out of given geometry
 * \param lo Left extrema of the sim-box
 * \param hi Right extrema of the sim-box
 * \param first_tag First tag (rng argument)
 * \param vel_factor Scale factor for uniform normal velocities consistent with particle mass /
 * temperature
 * \param type Particle type for filling
 * \param N_virt_max Maximum no. of virtual particles that can exist
 * \param timestep Current timestep
 * \param seed User seed for RNG
 * \param filler_id Identifier for the filler (rng argument)
 *
 * \tparam Geometry type of the confined geometry \a geom
 *
 * \b implementation
 * We assign one thread per particle to draw random particle positions within the box and velocities
 * consistent with system temperature. Along with this a boolean array tracks if the particles are
 * in/out of bounds of the given geometry.
 */
template<class Geometry>
__global__ void draw_virtual_particles(Scalar4* d_tmp_pos,
                                       Scalar4* d_tmp_vel,
                                       bool* d_keep_particles,
                                       const Scalar3 lo,
                                       const Scalar3 hi,
                                       const unsigned int first_tag,
                                       const Scalar vel_factor,
                                       const unsigned int type,
                                       const unsigned int N_virt_max,
                                       const unsigned int timestep,
                                       const unsigned int seed,
                                       const unsigned int filler_id,
                                       const unsigned int block_size,
                                       const Geometry geom)
    {
    // one thread per particle
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_virt_max)
        return;

    // initialize random number generator for positions and velocity
    const unsigned int tag = first_tag + idx;
    hoomd::RandomGenerator rng(
        hoomd::Seed(hoomd::RNGIdentifier::VirtualParticleFiller, timestep, seed),
        hoomd::Counter(tag, filler_id));
    Scalar3 pos = make_scalar3(hoomd::UniformDistribution<Scalar>(lo.x, hi.x)(rng),
                               hoomd::UniformDistribution<Scalar>(lo.y, hi.y)(rng),
                               hoomd::UniformDistribution<Scalar>(lo.z, hi.z)(rng));
    d_tmp_pos[idx] = make_scalar4(pos.x, pos.y, pos.z, __int_as_scalar(type));

    // check if particle is inside/outside the confining geometry
    d_keep_particles[idx] = geom.isOutside(pos);

    hoomd::NormalDistribution<Scalar> gen(vel_factor, 0.0);
    Scalar3 vel;
    gen(vel.x, vel.y, rng);
    vel.z = gen(rng);
    d_tmp_vel[idx] = make_scalar4(vel.x, vel.y, vel.z, __int_as_scalar(mpcd::detail::NO_CELL));
    }

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

/*!
 * \param args Common arguments for all geometries
 * \param geom Confined geometry
 *
 * \tparam Geometry type of the confined geometry \a geom
 *
 * \sa mpcd::gpu::kernel::draw_virtual_particles
 */
template<class Geometry>
cudaError_t draw_virtual_particles(const draw_virtual_particles_args_t& args, const Geometry& geom)
    {
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::draw_virtual_particles<Geometry>);
    const unsigned int max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(args.block_size, max_block_size);
    dim3 grid(args.N_virt_max / run_block_size + 1);
    mpcd::gpu::kernel::draw_virtual_particles<Geometry>
        <<<grid, run_block_size>>>(args.d_tmp_pos,
                                   args.d_tmp_vel,
                                   args.d_keep_particles,
                                   args.lo,
                                   args.hi,
                                   args.first_tag,
                                   args.vel_factor,
                                   args.type,
                                   args.N_virt_max,
                                   args.timestep,
                                   args.seed,
                                   args.filler_id,
                                   args.block_size,
                                   geom);

    return cudaSuccess;
    }

cudaError_t compact_virtual_particle_indices(void* d_tmp,
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

cudaError_t copy_virtual_particles(unsigned int* d_keep_indices,
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

#endif // __HIPCC__

    } // end namespace gpu
    } // end namespace mpcd
    } // namespace hoomd

#endif // MPCD_REJECTION_VIRTUAL_PARTICLE_FILLER_GPU_CUH_
