// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file BounceBackNVEGPU.cuh
 * \brief Declaration of CUDA kernels for BounceBackNVEGPU
 */

#ifndef MPCD_BOUNCE_BACK_NVE_GPU_CUH_
#define MPCD_BOUNCE_BACK_NVE_GPU_CUH_

#include <cuda_runtime.h>

#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"

namespace mpcd
{
namespace gpu
{

//! Common arguments for bounce-back integrator
struct bounce_args_t
    {
    //! Constructor
    bounce_args_t(Scalar4 *_d_pos,
                  int3 *_d_image,
                  Scalar4 *_d_vel,
                  const Scalar3 *_d_accel,
                  const unsigned int *_d_group,
                  const Scalar _dt,
                  const BoxDim& _box,
                  const unsigned int _N,
                  const unsigned int _block_size)
        : d_pos(_d_pos), d_image(_d_image), d_vel(_d_vel), d_accel(_d_accel), d_group(_d_group),
          dt(_dt), box(_box), N(_N), block_size(_block_size)
        { }

    Scalar4 *d_pos;                 //!< Particle positions
    int3 *d_image;                  //!< Particle images
    Scalar4 *d_vel;                 //!< Particle velocities
    const Scalar3 *d_accel;         //!< Particle accelerations
    const unsigned int *d_group;    //!< Indexes in particle group
    const Scalar dt;                //!< Timestep
    const BoxDim& box;              //!< Simulation box
    const unsigned int N;           //!< Number of particles in group
    const unsigned int block_size;  //!< Number of threads per block
    };

//! Kernel driver to apply step one of the velocity Verlet algorithm with bounce-back rules
template<class Geometry>
cudaError_t nve_bounce_step_one(const bounce_args_t& args, const Geometry& geom);

//! Kernel driver to apply step two of the velocity Verlet algorithm with bounce-back rules
cudaError_t nve_bounce_step_two(Scalar4 *d_vel,
                                Scalar3 *d_accel,
                                const Scalar4 *d_net_force,
                                const unsigned int *d_group,
                                const Scalar dt,
                                const unsigned int N,
                                const unsigned int block_size);

#ifdef NVCC
namespace kernel
{
//! Kernel for applying first step of velocity Verlet algorithm with bounce-back
/*!
 * \param d_pos Particle positions
 * \param d_image Particle images
 * \param d_vel Particle velocities
 * \param d_accel Particle accelerations
 * \param d_group Indexes in particle group
 * \param dt Timestep
 * \param box Simulation box
 * \param N Number of particles in group
 * \param geom Bounce-back geometry
 *
 * \tparam Geometry type of bounce-back geometry
 *
 * \b Implementation:
 * Using one thread per particle, the bounce-back equations of motion are applied within the velocity Verlet algorithm.
 * This amounts to first updating the particle velocities according to the current acceleration, then integration the
 * position forward while respecting bounce-back conditions any time the particle crosses a boundary.
 */
template<class Geometry>
__global__ void nve_bounce_step_one(Scalar4 *d_pos,
                                    int3 *d_image,
                                    Scalar4 *d_vel,
                                    const Scalar3 *d_accel,
                                    const unsigned int *d_group,
                                    const Scalar dt,
                                    const BoxDim box,
                                    const unsigned int N,
                                    const Geometry geom)
    {
    // one thread per particle
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;
    const unsigned int pid = d_group[idx];

    // load velocity + mass
    const Scalar4 velmass = d_vel[pid];
    Scalar3 vel = make_scalar3(velmass.x, velmass.y, velmass.z);
    const Scalar mass = velmass.w;

    // update velocity first according to verlet step
    const Scalar3 accel = d_accel[pid];
    vel += Scalar(0.5) * dt * accel;

    // load poosition and type
    const Scalar4 postype = d_pos[pid];
    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
    const Scalar type = postype.w;

    // update position while bouncing-back velocity
    Scalar dt_remain = dt;
    bool collide = false;
    do
        {
        pos += dt_remain * vel;
        collide = geom.detectCollision(pos, vel, dt_remain);
        }
    while (dt_remain > 0 && collide);

    // wrap final position
    int3 img = d_image[pid];
    box.wrap(pos, img);

    // write position and velocity back out
    d_pos[pid] = make_scalar4(pos.x, pos.y, pos.z, type);
    d_vel[pid] = make_scalar4(vel.x, vel.y, vel.z, mass);
    d_image[pid] = img;
    }

} // end namespace kernel

/*!
 * \param args Common bounce-back integration arguments
 * \param geom Bounce-back geometry
 *
 * \tparam Geometry type of bounce-back geometry
 *
 * This function \b must be explicitly templated once for each streaming geometry.
 *
 * \sa kernel::nve_bounce_step_one
 */
template<class Geometry>
cudaError_t nve_bounce_step_one(const bounce_args_t& args, const Geometry& geom)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)kernel::nve_bounce_step_one<Geometry>);
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(args.block_size, max_block_size);
    dim3 grid(args.N / run_block_size + 1);
    kernel::nve_bounce_step_one<Geometry><<<grid, run_block_size>>>(args.d_pos,
                                                                    args.d_image,
                                                                    args.d_vel,
                                                                    args.d_accel,
                                                                    args.d_group,
                                                                    args.dt,
                                                                    args.box,
                                                                    args.N,
                                                                    geom);

    return cudaSuccess;
    }
#endif // NVCC

} // end namespace gpu
} // end namespace mpcd
#endif // MPCD_BOUNCE_BACK_NVE_GPU_CUH_
