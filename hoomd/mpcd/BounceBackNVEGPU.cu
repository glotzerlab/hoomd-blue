// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file BounceBackNVEGPU.cu
 * \brief Template specialization of CUDA kernels for BounceBackNVEGPU geometries. Each instance of
 * the nve_bounce_step_one must be templated explicitly for each geometry.
 */

#include "BounceBackNVEGPU.cuh"
#include "StreamingGeometry.h"

namespace hoomd
    {
namespace mpcd
    {
namespace gpu
    {
//! Template instantiation of slit geometry streaming
template cudaError_t
nve_bounce_step_one<mpcd::detail::SlitGeometry>(const bounce_args_t& args,
                                                const mpcd::detail::SlitGeometry& geom);

//! Template instantiation of slit pore geometry streaming
template cudaError_t
nve_bounce_step_one<mpcd::detail::SlitPoreGeometry>(const bounce_args_t& args,
                                                    const mpcd::detail::SlitPoreGeometry& geom);

namespace kernel
    {
//! Kernel for applying second step of velocity Verlet algorithm with bounce back
/*!
 * \param d_vel Particle velocities
 * \param d_accel Particle accelerations
 * \param d_net_force Net force on each particle
 * \param d_group Indexes in particle group
 * \param dt Timestep
 * \param N Number of particles in group
 *
 * \b Implementation:
 * Using one thread per particle, the particle velocities are updated according to the second step
 * of the velocity Verlet algorithm. This is the standard update as in MD, and is only reimplemented
 * here in case future modifications are necessary.
 */
__global__ void nve_bounce_step_two(Scalar4* d_vel,
                                    Scalar3* d_accel,
                                    const Scalar4* d_net_force,
                                    const unsigned int* d_group,
                                    const Scalar dt,
                                    const unsigned int N)
    {
    // one thread per particle
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;
    const unsigned int pid = d_group[idx];

    const Scalar4 net_force = d_net_force[pid];
    Scalar3 accel = make_scalar3(net_force.x, net_force.y, net_force.z);
    Scalar4 vel = d_vel[pid];
    accel.x /= vel.w;
    accel.y /= vel.w;
    accel.z /= vel.w;

    // then, update the velocity
    vel.x += Scalar(0.5) * accel.x * dt;
    vel.y += Scalar(0.5) * accel.y * dt;
    vel.z += Scalar(0.5) * accel.z * dt;

    d_vel[pid] = vel;
    d_accel[pid] = accel;
    }
    } // end namespace kernel

/*!
 * \param d_vel Particle velocities
 * \param d_accel Particle accelerations
 * \param d_net_force Net force on each particle
 * \param d_group Indexes in particle group
 * \param dt Timestep
 * \param N Number of particles in group
 * \param block_size Number of threads per block
 *
 * \sa kernel::nve_bounce_step_two
 */
cudaError_t nve_bounce_step_two(Scalar4* d_vel,
                                Scalar3* d_accel,
                                const Scalar4* d_net_force,
                                const unsigned int* d_group,
                                const Scalar dt,
                                const unsigned int N,
                                const unsigned int block_size)
    {
    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)kernel::nve_bounce_step_two);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(N / run_block_size + 1);
    kernel::nve_bounce_step_two<<<grid, run_block_size>>>(d_vel,
                                                          d_accel,
                                                          d_net_force,
                                                          d_group,
                                                          dt,
                                                          N);

    return cudaSuccess;
    }

    } // end namespace gpu
    } // end namespace mpcd
    } // end namespace hoomd
