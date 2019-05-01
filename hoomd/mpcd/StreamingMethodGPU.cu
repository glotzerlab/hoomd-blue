// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/StreamingMethodGPU.cu
 * \brief Defines GPU functions and kernels used by mpcd::StreamingMethodGPU
 */

#include "StreamingMethodGPU.cuh"

namespace mpcd
{
namespace gpu
{
namespace kernel
{

//! Kernel to stream particles ballistically
/*!
 * \param d_pos Particle positions
 * \param d_vel Particle velocities
 * \param box Simulation box
 * \param dt Timestep to stream
 * \param N Number of particles
 *
 * \b Implementation
 * Using one thread per particle, the particle position and velocity is loaded.
 * The particles are propagated forward ballistically:
 * \f[
 *      r(t + \Delta t) = r(t) + v(t) \Delta t
 * \f]
 * Particles crossing a periodic global boundary are wrapped back into the simulation box.
 * The particle positions are updated.
 */
__global__ void stream(Scalar4 *d_pos,
                       const Scalar4 *d_vel,
                       const BoxDim box,
                       const Scalar dt,
                       const unsigned int N)
    {
    // one thread per particle
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    const Scalar4 postype = d_pos[idx];
    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
    const unsigned int type = __scalar_as_int(postype.w);

    const Scalar4 vel_cell = d_vel[idx];
    const Scalar3 vel = make_scalar3(vel_cell.x, vel_cell.y, vel_cell.z);

    // propagate the particle to its new position ballistically
    pos += dt * vel;

    // wrap and update the position
    int3 image = make_int3(0,0,0);
    box.wrap(pos, image);

    d_pos[idx] = make_scalar4(pos.x, pos.y, pos.z, __int_as_scalar(type));
    }

} // end namespace kernel

/*!
 * \param d_pos Particle positions
 * \param d_vel Particle velocities
 * \param box Simulation box
 * \param dt Timestep to stream
 * \param N Number of particles
 * \param block_size Number of threads per block
 *
 * \sa mpcd::gpu::kernel::stream
 */
cudaError_t stream(Scalar4 *d_pos,
                   const Scalar4 *d_vel,
                   const BoxDim& box,
                   const Scalar dt,
                   const unsigned int N,
                   const unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::stream);
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(N / run_block_size + 1);
    mpcd::gpu::kernel::stream<<<grid, run_block_size>>>(d_pos, d_vel, box, dt, N);

    return cudaSuccess;
    }

} // end namespace gpu
} // end namespace mpcd
