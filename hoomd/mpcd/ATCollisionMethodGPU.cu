// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/ATCollisionMethodGPU.cu
 * \brief Defines GPU functions and kernels used by mpcd::ATCollisionMethodGPU
 */

#include "ATCollisionMethodGPU.cuh"
#include "ParticleDataUtilities.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/RNGIdentifiers.h"

namespace mpcd
{
namespace gpu
{
namespace kernel
{
__global__ void at_draw_velocity(Scalar4 *d_alt_vel,
                                 Scalar4 *d_alt_vel_embed,
                                 const unsigned int *d_tag,
                                 const Scalar mpcd_mass,
                                 const unsigned int *d_embed_idx,
                                 const Scalar4 *d_vel_embed,
                                 const unsigned int *d_tag_embed,
                                 const unsigned int timestep,
                                 const unsigned int seed,
                                 const Scalar T,
                                 const unsigned int N_mpcd,
                                 const unsigned int N_tot)
    {
    // one thread per particle
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_tot)
        return;

    unsigned int pidx;
    unsigned int tag; Scalar mass;
    if (idx < N_mpcd)
        {
        pidx = idx;
        mass = mpcd_mass;
        tag = d_tag[idx];
        }
    else
        {
        pidx = d_embed_idx[idx-N_mpcd];
        mass = d_vel_embed[pidx].w;
        tag = d_tag_embed[pidx];
        }

    // draw random velocities from normal distribution
    hoomd::RandomGenerator rng(hoomd::RNGIdentifier::ATCollisionMethod, seed, tag, timestep);
    hoomd::NormalDistribution<Scalar> gen(fast::sqrt(T/mass), 0.0);
    Scalar3 vel;
    gen(vel.x, vel.y, rng);
    vel.z = gen(rng);

    // save out velocities
    if (idx < N_mpcd)
        {
        d_alt_vel[pidx] = make_scalar4(vel.x, vel.y, vel.z, __int_as_scalar(mpcd::detail::NO_CELL));
        }
    else
        {
        d_alt_vel_embed[pidx] = make_scalar4(vel.x, vel.y, vel.z, mass);
        }
    }

__global__ void at_apply_velocity(Scalar4 *d_vel,
                                  Scalar4 *d_vel_embed,
                                  const Scalar4 *d_vel_alt,
                                  const unsigned int *d_embed_idx,
                                  const Scalar4 *d_vel_alt_embed,
                                  const unsigned int *d_embed_cell_ids,
                                  const double4 *d_cell_vel,
                                  const double4 *d_rand_vel,
                                  const unsigned int N_mpcd,
                                  const unsigned int N_tot)
    {
    // one thread per particle
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_tot)
        return;

    unsigned int cell, pidx;
    Scalar4 vel_rand;
    if (idx < N_mpcd)
        {
        pidx = idx;
        const Scalar4 vel_cell = d_vel[idx];
        cell = __scalar_as_int(vel_cell.w);
        vel_rand = d_vel_alt[idx];
        }
    else
        {
        pidx = d_embed_idx[idx-N_mpcd];
        cell = d_embed_cell_ids[idx-N_mpcd];
        vel_rand = d_vel_alt_embed[pidx];
        }

    // load cell data
    const double4 v_c = d_cell_vel[cell];
    const double4 vrand_c = d_rand_vel[cell];

    // compute new velocity using the cell + the random draw
    const Scalar3 vnew = make_scalar3(v_c.x - vrand_c.x + vel_rand.x,
                                      v_c.y - vrand_c.y + vel_rand.y,
                                      v_c.z - vrand_c.z + vel_rand.z);

    if (idx < N_mpcd)
        {
        d_vel[pidx] = make_scalar4(vnew.x, vnew.y, vnew.z, __int_as_scalar(cell));
        }
    else
        {
        d_vel_embed[pidx] = make_scalar4(vnew.x, vnew.y, vnew.z, vel_rand.w);
        }
    }

} // end namespace kernel

cudaError_t at_draw_velocity(Scalar4 *d_alt_vel,
                             Scalar4 *d_alt_vel_embed,
                             const unsigned int *d_tag,
                             const Scalar mpcd_mass,
                             const unsigned int *d_embed_idx,
                             const Scalar4 *d_vel_embed,
                             const unsigned int *d_tag_embed,
                             const unsigned int timestep,
                             const unsigned int seed,
                             const Scalar T,
                             const unsigned int N_mpcd,
                             const unsigned int N_tot,
                             const unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::at_draw_velocity);
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(block_size, max_block_size);

    dim3 grid(N_tot / run_block_size + 1);
    mpcd::gpu::kernel::at_draw_velocity<<<grid, run_block_size>>>(d_alt_vel,
                                                                  d_alt_vel_embed,
                                                                  d_tag,
                                                                  mpcd_mass,
                                                                  d_embed_idx,
                                                                  d_vel_embed,
                                                                  d_tag_embed,
                                                                  timestep,
                                                                  seed,
                                                                  T,
                                                                  N_mpcd,
                                                                  N_tot);

    return cudaSuccess;
    }

cudaError_t at_apply_velocity(Scalar4 *d_vel,
                              Scalar4 *d_vel_embed,
                              const Scalar4 *d_vel_alt,
                              const unsigned int *d_embed_idx,
                              const Scalar4 *d_vel_alt_embed,
                              const unsigned int *d_embed_cell_ids,
                              const double4 *d_cell_vel,
                              const double4 *d_rand_vel,
                              const unsigned int N_mpcd,
                              const unsigned int N_tot,
                              const unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::at_apply_velocity);
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(block_size, max_block_size);

    dim3 grid(N_tot / run_block_size + 1);
    mpcd::gpu::kernel::at_apply_velocity<<<grid, run_block_size>>>(d_vel,
                                                                   d_vel_embed,
                                                                   d_vel_alt,
                                                                   d_embed_idx,
                                                                   d_vel_alt_embed,
                                                                   d_embed_cell_ids,
                                                                   d_cell_vel,
                                                                   d_rand_vel,
                                                                   N_mpcd,
                                                                   N_tot);

    return cudaSuccess;
    }

} // end namespace gpu
} // end namespace mpcd
