// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/SRDCollisionMethodGPU.cu
 * \brief Defines GPU functions and kernels used by mpcd::SRDCollisionMethodGPU
 */

#include "SRDCollisionMethodGPU.cuh"
#include "hoomd/extern/saruprngCUDA.h"

#define MPCD_2PI 6.283185307179586

namespace mpcd
{
namespace gpu
{
namespace kernel
{
__global__ void srd_draw_vectors(Scalar3 *d_rotvec,
                                 const Index3D ci,
                                 const int3 origin,
                                 const uint3 global_dim,
                                 const Index3D global_ci,
                                 const unsigned int timestep,
                                 const unsigned int seed,
                                 const unsigned int Ncell)
    {
    // one thread per cell
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Ncell)
        return;

    // get local cell triple from 1d index
    const uint3 cell = ci.getTriple(idx);
    // shift local cell by local origin, and wrap through global boundaries
    int3 global_cell = make_int3(origin.x + (int)cell.x,
                                 origin.y + (int)cell.y,
                                 origin.z + (int)cell.z);
    if (global_cell.x >= (int)global_dim.x) global_cell.x -= global_dim.x;
    else if (global_cell.x < 0) global_cell.x += global_dim.x;

    if (global_cell.y >= (int)global_dim.y) global_cell.y -= global_dim.y;
    else if (global_cell.y < 0) global_cell.y += global_dim.y;

    if (global_cell.z >= (int)global_dim.z) global_cell.z -= global_dim.z;
    else if (global_cell.z < 0) global_cell.z += global_dim.z;

    // convert global triple to 1d global index
    const unsigned int global_idx = global_ci(global_cell.x, global_cell.y, global_cell.z);

    // Initialize the PRNG using the cell index, timestep, and seed for the hash
    SaruGPU saru(global_idx, timestep, seed);

    // calculate the random rotation vector for the cell
    const double theta = saru.d(0, MPCD_2PI);
    const double u = saru.d(-1.0, 1.0);

    /*
     * Sometimes numbers get drawn really close to -1 or +1, and the machine precision difference is a really
     * small (negative) number. This causes sqrt() to fail with nan error, so we need to handle those cases by
     * forcing the sqrt() to 0.0.
     */
    double sqrtu = 0.0;
    const double one_minus_u2 = 1.0-u*u;
    if (one_minus_u2 > 0.0)
        {
        sqrtu = slow::sqrt(one_minus_u2);
        }

    d_rotvec[idx] = make_double3(sqrtu * slow::cos(theta), sqrtu*slow::sin(theta), u);
    }
__global__ void srd_rotate(Scalar4 *d_vel,
                           Scalar4 *d_vel_embed,
                           const unsigned int *d_embed_group,
                           const unsigned int *d_embed_cell_ids,
                           const Scalar4 *d_cell_vel,
                           const Scalar3 *d_rotvec,
                           const double cos_a,
                           const double one_minus_cos_a,
                           const double sin_a,
                           const unsigned int N_mpcd,
                           const unsigned int N_tot)
    {
    // one thread per particle
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N_tot)
        return;

    // load particle data
    double3 vel;
    unsigned int cell;
    // these properties are needed for the embedded particles only
    unsigned int idx(0); double mass(0);
    if (tid < N_mpcd)
        {
        const Scalar4 vel_cell = d_vel[tid];
        vel = make_double3(vel_cell.x, vel_cell.y, vel_cell.z);
        cell = __scalar_as_int(vel_cell.w);
        }
    else
        {
        idx = d_embed_group[tid - N_mpcd];

        const Scalar4 vel_mass = d_vel_embed[idx];
        vel = make_double3(vel_mass.x, vel_mass.y, vel_mass.z);
        mass = vel_mass.w;
        cell = d_embed_cell_ids[tid - N_mpcd];
        }

    // subtract average velocity
    const Scalar4 avg_vel = d_cell_vel[cell];
    vel.x -= avg_vel.x;
    vel.y -= avg_vel.y;
    vel.z -= avg_vel.z;

    // get rotation vector
    double3 rot_vec = d_rotvec[cell];

    // perform the rotation in double precision
    double3 new_vel;
    new_vel.x = (cos_a + rot_vec.x*rot_vec.x*one_minus_cos_a) * vel.x;
    new_vel.x += (rot_vec.x*rot_vec.y*one_minus_cos_a - sin_a*rot_vec.z) * vel.y;
    new_vel.x += (rot_vec.x*rot_vec.z*one_minus_cos_a + sin_a*rot_vec.y) * vel.z;

    new_vel.y = (cos_a + rot_vec.y*rot_vec.y*one_minus_cos_a) * vel.y;
    new_vel.y += (rot_vec.x*rot_vec.y*one_minus_cos_a + sin_a*rot_vec.z) * vel.x;
    new_vel.y += (rot_vec.y*rot_vec.z*one_minus_cos_a - sin_a*rot_vec.x) * vel.z;

    new_vel.z = (cos_a + rot_vec.z*rot_vec.z*one_minus_cos_a) * vel.z;
    new_vel.z += (rot_vec.x*rot_vec.z*one_minus_cos_a - sin_a*rot_vec.y) * vel.x;
    new_vel.z += (rot_vec.y*rot_vec.z*one_minus_cos_a + sin_a*rot_vec.x) * vel.y;

    new_vel.x += avg_vel.x;
    new_vel.y += avg_vel.y;
    new_vel.z += avg_vel.z;

    // set the new velocity
    if (tid < N_mpcd)
        {
        d_vel[tid] = make_scalar4(new_vel.x, new_vel.y, new_vel.z, __int_as_scalar(cell));
        }
    else
        {
        d_vel_embed[idx] = make_scalar4(new_vel.x, new_vel.y, new_vel.z, mass);
        }
    }
} // end namespace kernel

cudaError_t srd_draw_vectors(Scalar3 *d_rotvec,
                             const Index3D& ci,
                             const int3 origin,
                             const uint3 global_dim,
                             const Index3D& global_ci,
                             const unsigned int timestep,
                             const unsigned int seed,
                             const unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::srd_draw_vectors);
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(block_size, max_block_size);

    const unsigned int Ncell = ci.getNumElements();
    dim3 grid(Ncell / run_block_size + 1);
    mpcd::gpu::kernel::srd_draw_vectors<<<grid, run_block_size>>>(d_rotvec,
                                                                  ci,
                                                                  origin,
                                                                  global_dim,
                                                                  global_ci,
                                                                  timestep,
                                                                  seed,
                                                                  Ncell);

    return cudaSuccess;
    }

cudaError_t srd_rotate(Scalar4 *d_vel,
                       Scalar4 *d_vel_embed,
                       const unsigned int *d_embed_group,
                       const unsigned int *d_embed_cell_ids,
                       const Scalar4 *d_cell_vel,
                       const Scalar3 *d_rotvec,
                       const double angle,
                       const unsigned int N_mpcd,
                       const unsigned int N_tot,
                       const unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::srd_rotate);
        max_block_size = attr.maxThreadsPerBlock;
        }

    // precompute angles for rotation
    const double cos_a = slow::cos(angle);
    const double one_minus_cos_a = 1.0 - cos_a;
    const double sin_a = slow::sin(angle);

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(N_tot / run_block_size + 1);
    mpcd::gpu::kernel::srd_rotate<<<grid, run_block_size>>>(d_vel,
                                                            d_vel_embed,
                                                            d_embed_group,
                                                            d_embed_cell_ids,
                                                            d_cell_vel,
                                                            d_rotvec,
                                                            cos_a,
                                                            one_minus_cos_a,
                                                            sin_a,
                                                            N_mpcd,
                                                            N_tot);

    return cudaSuccess;
    }

} // end namespace gpu
} // end namespace mpcd
