// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

#ifndef MPCD_CONFINED_STREAMING_METHOD_GPU_CUH_
#define MPCD_CONFINED_STREAMING_METHOD_GPU_CUH_

/*!
 * \file mpcd/ConfinedStreamingMethodGPU.cuh
 * \brief Declaration of CUDA kernels for mpcd::ConfinedStreamingMethodGPU
 */

#include "ExternalField.h"
#include "ParticleDataUtilities.h"
#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"

namespace mpcd
{
namespace gpu
{

//! Common arguments passed to all streaming kernels
struct stream_args_t
    {
    //! Constructor
    stream_args_t(Scalar4 *_d_pos,
                  Scalar4 *_d_vel,
                  const Scalar _mass,
                  const mpcd::ExternalField* _field,
                  const BoxDim& _box,
                  const Scalar _dt,
                  const unsigned int _N,
                  const unsigned int _block_size)
        : d_pos(_d_pos), d_vel(_d_vel), mass(_mass), field(_field), box(_box), dt(_dt), N(_N), block_size(_block_size)
        { }

    Scalar4 *d_pos;                     //!< Particle positions
    Scalar4 *d_vel;                     //!< Particle velocities
    const Scalar mass;                  //!< Particle mass
    const mpcd::ExternalField* field;   //!< Applied external field on particles
    const BoxDim& box;                  //!< Simulation box
    const Scalar dt;                    //!< Timestep
    const unsigned int N;               //!< Number of particles
    const unsigned int block_size;      //!< Number of threads per block
    };

//! Kernel driver to stream particles ballistically
template<class Geometry>
cudaError_t confined_stream(const stream_args_t& args, const Geometry& geom);

#ifdef NVCC
namespace kernel
{

//! Kernel to stream particles ballistically
/*!
 * \param d_pos Particle positions
 * \param d_vel Particle velocities
 * \param mass Particle mass
 * \param box Simulation box
 * \param dt Timestep to stream
 * \param field Applied external field
 * \param N Number of particles
 * \param geom Confined geometry
 *
 * \tparam Geometry type of the confined geometry \a geom
 *
 * \b Implementation
 * Using one thread per particle, the particle position and velocity is loaded.
 * The particles are propagated forward ballistically subject to an external force \a f:
 * \f[
 *      v(t + \Delta t/2) = v(t) + (f/m) \Delta t / 2
 *      r(t + \Delta t) = r(t) + v(t+\Delta t/2) \Delta t
 *      v(t + \Delta t) = v(t + \Delta t/2) + (f/m) \Delta t / 2
 * \f]
 * Particles crossing a periodic global boundary are wrapped back into the simulation box.
 * Particles are appropriately reflected from the boundaries defined by \a geom during the
 * position update step. The particle positions and velocities are updated accordingly.
 */
template<class Geometry>
__global__ void confined_stream(Scalar4 *d_pos,
                                Scalar4 *d_vel,
                                const Scalar mass,
                                const mpcd::ExternalField* field,
                                const BoxDim box,
                                const Scalar dt,
                                const unsigned int N,
                                const Geometry geom)
    {
    // one thread per particle
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    const Scalar4 postype = d_pos[idx];
    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
    const unsigned int type = __scalar_as_int(postype.w);

    const Scalar4 vel_cell = d_vel[idx];
    Scalar3 vel = make_scalar3(vel_cell.x, vel_cell.y, vel_cell.z);
    // estimate next velocity based on current acceleration
    if (field)
        {
        vel += Scalar(0.5) * dt * field->evaluate(pos) / mass;
        }

    // propagate the particle to its new position ballistically
    Scalar dt_remain = dt;
    bool collide = true;
    do
        {
        pos += dt_remain * vel;
        collide = geom.detectCollision(pos, vel, dt_remain);
        }
    while (dt_remain > 0 && collide);
    // finalize velocity update
    if (field)
        {
        vel += Scalar(0.5) * dt * field->evaluate(pos) / mass;
        }

    // wrap and update the position
    int3 image = make_int3(0,0,0);
    box.wrap(pos, image);

    d_pos[idx] = make_scalar4(pos.x, pos.y, pos.z, __int_as_scalar(type));
    d_vel[idx] = make_scalar4(vel.x, vel.y, vel.z, __int_as_scalar(mpcd::detail::NO_CELL));
    }

} // end namespace kernel

/*!
 * \param args Common arguments for a streaming kernel
 * \param geom Confined geometry
 *
 * \tparam Geometry type of the confined geometry \a geom
 *
 * \sa mpcd::gpu::kernel::confined_stream
 */
template<class Geometry>
cudaError_t confined_stream(const stream_args_t& args, const Geometry& geom)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::confined_stream<Geometry>);
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(args.block_size, max_block_size);
    dim3 grid(args.N / run_block_size + 1);
    mpcd::gpu::kernel::confined_stream<Geometry><<<grid, run_block_size>>>(args.d_pos, args.d_vel, args.mass, args.field, args.box, args.dt, args.N, geom);

    return cudaSuccess;
    }
#endif // NVCC

} // end namespace gpu
} // end namespace mpcd

#endif // MPCD_CONFINED_STREAMING_METHOD_GPU_CUH_
