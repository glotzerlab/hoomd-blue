// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hip/hip_runtime.h"
// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

/*! \file TwoStepRATTLENVEGPU.cuh
    \brief Declares GPU kernel code for RATTLENVE integration on the GPU. Used by
   TwoStepRATTLENVEGPU.
*/

#pragma once

#include "hoomd/CachedAllocator.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/ParticleData.cuh"
#include "hoomd/VectorMath.h"

#include "TwoStepRATTLENVEGPU.cuh"
#include "hoomd/GPUPartition.cuh"

#include <assert.h>
#include <type_traits>

#ifndef __TWO_STEP_RATTLE_NVE_GPU_CUH__
#define __TWO_STEP_RATTLE_NVE_GPU_CUH__

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
hipError_t gpu_rattle_nve_step_one(Scalar4* d_pos,
                                   Scalar4* d_vel,
                                   const Scalar3* d_accel,
                                   int3* d_image,
                                   unsigned int* d_group_members,
                                   const GPUPartition& gpu_partition,
                                   const BoxDim& box,
                                   Scalar deltaT,
                                   bool limit,
                                   Scalar limit_val,
                                   unsigned int block_size);

hipError_t gpu_rattle_nve_angular_step_one(Scalar4* d_orientation,
                                           Scalar4* d_angmom,
                                           const Scalar3* d_inertia,
                                           const Scalar4* d_net_torque,
                                           unsigned int* d_group_members,
                                           const GPUPartition& gpu_partition,
                                           Scalar deltaT,
                                           Scalar scale,
                                           const unsigned int block_size);

template<class Manifold>
hipError_t gpu_rattle_nve_step_two(Scalar4* d_pos,
                                   Scalar4* d_vel,
                                   Scalar3* d_accel,
                                   unsigned int* d_group_members,
                                   const GPUPartition& gpu_partition,
                                   Scalar4* d_net_force,
                                   Manifold manifold,
                                   Scalar tolerance,
                                   Scalar deltaT,
                                   bool limit,
                                   Scalar limit_val,
                                   bool zero_force,
                                   unsigned int block_size);

hipError_t gpu_rattle_nve_angular_step_two(const Scalar4* d_orientation,
                                           Scalar4* d_angmom,
                                           const Scalar3* d_inertia,
                                           const Scalar4* d_net_torque,
                                           unsigned int* d_group_members,
                                           const GPUPartition& gpu_partition,
                                           Scalar deltaT,
                                           Scalar scale,
                                           const unsigned int block_size);

template<class Manifold>
hipError_t gpu_include_rattle_force_nve(const Scalar4* d_pos,
                                        const Scalar4* d_vel,
                                        Scalar3* d_accel,
                                        Scalar4* d_net_force,
                                        Scalar* d_net_virial,
                                        unsigned int* d_group_members,
                                        const GPUPartition& gpu_partition,
                                        size_t net_virial_pitch,
                                        Manifold manifold,
                                        Scalar tolerance,
                                        Scalar deltaT,
                                        bool zero_force,
                                        unsigned int block_size);

#ifdef __HIPCC__

/*! \file TwoStepNVEGPU.cu
    \brief Defines GPU kernel code for NVE integration on the GPU. Used by TwoStepNVEGPU.
*/

//! Takes the second half-step forward in the velocity-verlet NVE integration on a group of
//! particles
/*! \param d_vel array of particle velocities
    \param d_accel array of particle accelerations
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param group_size Number of members in the group
    \param d_net_force Net force on each particle
    \param deltaT Amount of real time to step forward in one time step
    \param limit If \a limit is true, then the dynamics will be limited so that particles do not
   move a distance further than \a limit_val in one step. \param limit_val Length to limit particle
   distance movement to \param zero_force Set to true to always assign an acceleration of 0 to all
   particles in the group This kernel is implemented in a very similar manner to
   gpu_rattle_nve_step_one_kernel(), see it for design details.
*/
template<class Manifold>
__global__ void gpu_rattle_nve_step_two_kernel(Scalar4* d_pos,
                                               Scalar4* d_vel,
                                               Scalar3* d_accel,
                                               unsigned int* d_group_members,
                                               const unsigned int nwork,
                                               const unsigned int offset,
                                               Scalar4* d_net_force,
                                               Manifold manifold,
                                               Scalar tolerance,
                                               Scalar deltaT,
                                               bool limit,
                                               Scalar limit_val,
                                               bool zero_force)
    {
    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int work_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (work_idx < nwork)
        {
        const unsigned int group_idx = work_idx + offset;
        unsigned int idx = d_group_members[group_idx];

        Scalar3 pos = make_scalar3(d_pos[idx].x, d_pos[idx].y, d_pos[idx].z);

        // read in the net forc and calculate the acceleration MEM TRANSFER: 16 bytes
        Scalar3 accel = make_scalar3(Scalar(0.0), Scalar(0.0), Scalar(0.0));

        // read the current particle velocity (MEM TRANSFER: 16 bytes)
        Scalar4 vel = d_vel[idx];

        if (!zero_force)
            {
            Scalar4 net_force = d_net_force[idx];
            accel = make_scalar3(net_force.x, net_force.y, net_force.z);
            // MEM TRANSFER: 4 bytes   FLOPS: 3
            Scalar mass = vel.w;
            accel.x /= mass;
            accel.y /= mass;
            accel.z /= mass;
            }

        // v(t+deltaT) = v(t+deltaT/2) + 1/2 * a(t+deltaT)*deltaT

        // update the velocity (FLOPS: 6)

        Scalar mu = 0;
        Scalar inv_alpha = -Scalar(1.0 / 2.0) * deltaT;
        inv_alpha = Scalar(1.0) / inv_alpha;
        Scalar mass = vel.w;
        Scalar inv_mass = Scalar(1.0) / mass;

        Scalar3 normal = manifold.derivative(pos);

        Scalar3 next_vel;
        next_vel.x = vel.x + Scalar(1.0 / 2.0) * deltaT * accel.x;
        next_vel.y = vel.y + Scalar(1.0 / 2.0) * deltaT * accel.y;
        next_vel.z = vel.z + Scalar(1.0 / 2.0) * deltaT * accel.z;

        Scalar3 residual;
        Scalar resid;
        Scalar3 vel_dot;

        const unsigned int maxiteration = 10;
        unsigned int iteration = 0;
        do
            {
            iteration++;
            vel_dot.x = accel.x - mu * inv_mass * normal.x;
            vel_dot.y = accel.y - mu * inv_mass * normal.y;
            vel_dot.z = accel.z - mu * inv_mass * normal.z;

            residual.x = vel.x - next_vel.x + Scalar(1.0 / 2.0) * deltaT * vel_dot.x;
            residual.y = vel.y - next_vel.y + Scalar(1.0 / 2.0) * deltaT * vel_dot.y;
            residual.z = vel.z - next_vel.z + Scalar(1.0 / 2.0) * deltaT * vel_dot.z;
            resid = dot(normal, next_vel) * inv_mass;

            Scalar ndotr = dot(normal, residual);
            Scalar ndotn = dot(normal, normal);
            Scalar beta = (mass * resid + ndotr) / ndotn;
            next_vel.x = next_vel.x - normal.x * beta + residual.x;
            next_vel.y = next_vel.y - normal.y * beta + residual.y;
            next_vel.z = next_vel.z - normal.z * beta + residual.z;
            mu = mu - mass * beta * inv_alpha;

            resid = fabs(resid);
            Scalar vec_norm = sqrt(dot(residual, residual));
            if (vec_norm > resid)
                resid = vec_norm;

            } while (resid * mass > tolerance && iteration < maxiteration);

        vel.x += (Scalar(1.0) / Scalar(2.0)) * (accel.x - mu * inv_mass * normal.x) * deltaT;
        vel.y += (Scalar(1.0) / Scalar(2.0)) * (accel.y - mu * inv_mass * normal.y) * deltaT;
        vel.z += (Scalar(1.0) / Scalar(2.0)) * (accel.z - mu * inv_mass * normal.z) * deltaT;

        if (limit)
            {
            Scalar vel_len = sqrtf(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z);
            if ((vel_len * deltaT) > limit_val)
                {
                vel.x = vel.x / vel_len * limit_val / deltaT;
                vel.y = vel.y / vel_len * limit_val / deltaT;
                vel.z = vel.z / vel_len * limit_val / deltaT;
                }
            }

        // write out data (MEM TRANSFER: 32 bytes)
        d_vel[idx] = vel;
        // since we calculate the acceleration, we need to write it for the next step
        d_accel[idx] = accel;
        }
    }

/*! \param d_vel array of particle velocities
    \param d_accel array of particle accelerations
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param group_size Number of members in the group
    \param d_net_force Net force on each particle
    \param deltaT Amount of real time to step forward in one time step
    \param limit If \a limit is true, then the dynamics will be limited so that particles do not
   move a distance further than \a limit_val in one step. \param limit_val Length to limit particle
   distance movement to \param zero_force Set to true to always assign an acceleration of 0 to all
   particles in the group This is just a driver for gpu_rattle_nve_step_two_kernel(), see it for
   details.
*/
template<class Manifold>
hipError_t gpu_rattle_nve_step_two(Scalar4* d_pos,
                                   Scalar4* d_vel,
                                   Scalar3* d_accel,
                                   unsigned int* d_group_members,
                                   const GPUPartition& gpu_partition,
                                   Scalar4* d_net_force,
                                   Manifold manifold,
                                   Scalar tolerance,
                                   Scalar deltaT,
                                   bool limit,
                                   Scalar limit_val,
                                   bool zero_force,
                                   unsigned int block_size)
    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_rattle_nve_step_two_kernel<Manifold>);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    // iterate over active GPUs in reverse, to end up on first GPU when returning from this function
    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);

        unsigned int nwork = range.second - range.first;

        // setup the grid to run the kernel
        dim3 grid((nwork / run_block_size) + 1, 1, 1);
        dim3 threads(run_block_size, 1, 1);

        // run the kernel
        hipLaunchKernelGGL((gpu_rattle_nve_step_two_kernel<Manifold>),
                           dim3(grid),
                           dim3(threads),
                           0,
                           0,
                           d_pos,
                           d_vel,
                           d_accel,
                           d_group_members,
                           nwork,
                           range.first,
                           d_net_force,
                           manifold,
                           tolerance,
                           deltaT,
                           limit,
                           limit_val,
                           zero_force);
        }
    return hipSuccess;
    }

template<class Manifold>
__global__ void gpu_include_rattle_force_nve_kernel(const Scalar4* d_pos,
                                                    const Scalar4* d_vel,
                                                    Scalar3* d_accel,
                                                    Scalar4* d_net_force,
                                                    Scalar* d_net_virial,
                                                    unsigned int* d_group_members,
                                                    const unsigned int nwork,
                                                    const unsigned int offset,
                                                    size_t net_virial_pitch,
                                                    Manifold manifold,
                                                    Scalar tolerance,
                                                    Scalar deltaT,
                                                    bool zero_force)
    {
    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int work_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (work_idx < nwork)
        {
        const unsigned int group_idx = work_idx + offset;
        unsigned int idx = d_group_members[group_idx];

        // do velocity verlet update
        // r(t+deltaT) = r(t) + v(t)*deltaT + (1/2)a(t)*deltaT^2
        // v(t+deltaT/2) = v(t) + (1/2)a*deltaT

        // read the particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 postype = d_pos[idx];
        Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);

        Scalar3 normal
            = manifold.derivative(pos); // the normal vector to which the particles are confined.

        // read the particle's velocity and acceleration (MEM TRANSFER: 32 bytes)
        Scalar4 velmass = d_vel[idx];
        Scalar3 vel = make_scalar3(velmass.x, velmass.y, velmass.z);

        Scalar3 accel = make_scalar3(Scalar(0.0), Scalar(0.0), Scalar(0.0));
        if (!zero_force)
            accel = d_accel[idx];

        // read the particle's velocity and acceleration (MEM TRANSFER: 32 bytes)
        Scalar4 forcetype = d_net_force[idx];
        Scalar3 force = make_scalar3(forcetype.x, forcetype.y, forcetype.z);

        Scalar virial0 = d_net_virial[0 * net_virial_pitch + idx];
        Scalar virial1 = d_net_virial[1 * net_virial_pitch + idx];
        Scalar virial2 = d_net_virial[2 * net_virial_pitch + idx];
        Scalar virial3 = d_net_virial[3 * net_virial_pitch + idx];
        Scalar virial4 = d_net_virial[4 * net_virial_pitch + idx];
        Scalar virial5 = d_net_virial[5 * net_virial_pitch + idx];

        Scalar lambda = 0.0;
        Scalar inv_mass = Scalar(1.0) / velmass.w;
        Scalar deltaT_half = Scalar(1.0 / 2.0) * deltaT;
        Scalar inv_alpha = -deltaT_half * deltaT * inv_mass;
        inv_alpha = Scalar(1.0) / inv_alpha;

        Scalar3 next_pos = pos;
        Scalar3 residual;
        Scalar resid;
        Scalar3 half_vel;

        const unsigned int maxiteration = 10;
        unsigned int iteration = 0;
        do
            {
            iteration++;
            half_vel = vel + deltaT_half * accel - deltaT_half * inv_mass * lambda * normal;

            residual = pos - next_pos + deltaT * half_vel;
            resid = manifold.implicitFunction(next_pos);

            Scalar3 next_normal = manifold.derivative(next_pos);
            Scalar nndotr = dot(next_normal, residual);
            Scalar nndotn = dot(next_normal, normal);
            Scalar beta = (resid + nndotr) / nndotn;

            next_pos = next_pos - beta * normal + residual;
            lambda = lambda - beta * inv_alpha;

            resid = fabs(resid);
            Scalar vec_norm = sqrt(dot(residual, residual));
            if (vec_norm > resid)
                resid = vec_norm;

            } while (resid > tolerance && iteration < maxiteration);

        accel = accel - lambda * normal;

        force = force - inv_mass * lambda * normal;

        virial0 -= lambda * normal.x * pos.x;
        virial1 -= 0.5 * lambda * (normal.x * pos.y + normal.y * pos.x);
        virial2 -= 0.5 * lambda * (normal.x * pos.z + normal.z * pos.x);
        virial3 -= lambda * normal.y * pos.y;
        virial4 -= 0.5 * lambda * (normal.y * pos.z + normal.z * pos.y);
        virial5 -= lambda * normal.z * pos.z;

        d_net_force[idx] = make_scalar4(force.x, force.y, force.z, forcetype.w);
        d_accel[idx] = accel;
        d_net_virial[0 * net_virial_pitch + idx] = virial0;
        d_net_virial[1 * net_virial_pitch + idx] = virial1;
        d_net_virial[2 * net_virial_pitch + idx] = virial2;
        d_net_virial[3 * net_virial_pitch + idx] = virial3;
        d_net_virial[4 * net_virial_pitch + idx] = virial4;
        d_net_virial[5 * net_virial_pitch + idx] = virial5;
        }
    }

template<class Manifold>
hipError_t gpu_include_rattle_force_nve(const Scalar4* d_pos,
                                        const Scalar4* d_vel,
                                        Scalar3* d_accel,
                                        Scalar4* d_net_force,
                                        Scalar* d_net_virial,
                                        unsigned int* d_group_members,
                                        const GPUPartition& gpu_partition,
                                        size_t net_virial_pitch,
                                        Manifold manifold,
                                        Scalar tolerance,
                                        Scalar deltaT,
                                        bool zero_force,
                                        unsigned int block_size)
    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_include_rattle_force_nve_kernel<Manifold>);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    // iterate over active GPUs in reverse, to end up on first GPU when returning from this function
    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);

        unsigned int nwork = range.second - range.first;

        // setup the grid to run the kernel
        dim3 grid((nwork / run_block_size) + 1, 1, 1);
        dim3 threads(run_block_size, 1, 1);

        // run the kernel
        hipLaunchKernelGGL((gpu_include_rattle_force_nve_kernel<Manifold>),
                           dim3(grid),
                           dim3(threads),
                           0,
                           0,
                           d_pos,
                           d_vel,
                           d_accel,
                           d_net_force,
                           d_net_virial,
                           d_group_members,
                           nwork,
                           range.first,
                           net_virial_pitch,
                           manifold,
                           tolerance,
                           deltaT,
                           zero_force);
        }

    return hipSuccess;
    }

#endif

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif //__TWO_STEP_RATTLE_NVE_GPU_CUH__
