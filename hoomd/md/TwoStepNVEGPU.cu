// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "TwoStepNVEGPU.cuh"
#include "hoomd/VectorMath.h"

#include <assert.h>

/*! \file TwoStepNVEGPU.cu
    \brief Defines GPU kernel code for NVE integration on the GPU. Used by TwoStepNVEGPU.
*/

//! Takes the first half-step forward in the velocity-verlet NVE integration on a group of particles
/*! \param d_pos array of particle positions
    \param d_vel array of particle velocities
    \param d_accel array of particle accelerations
    \param d_image array of particle images
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param group_size Number of members in the group
    \param box Box dimensions for periodic boundary condition handling
    \param deltaT timestep
    \param limit If \a limit is true, then the dynamics will be limited so that particles do not move
        a distance further than \a limit_val in one step.
    \param limit_val Length to limit particle distance movement to
    \param zero_force Set to true to always assign an acceleration of 0 to all particles in the group

    This kernel must be executed with a 1D grid of any block size such that the number of threads is greater than or
    equal to the number of members in the group. The kernel's implementation simply reads one particle in each thread
    and updates that particle.

    <b>Performance notes:</b>
    Particle properties are read via the texture cache to optimize the bandwidth obtained with sparse groups. The writes
    in sparse groups will not be coalesced. However, because ParticleGroup sorts the index list the writes will be as
    contiguous as possible leading to fewer memory transactions on compute 1.3 hardware and more cache hits on Fermi.
*/
extern "C" __global__
void gpu_nve_step_one_kernel(Scalar4 *d_pos,
                             Scalar4 *d_vel,
                             const Scalar3 *d_accel,
                             int3 *d_image,
                             unsigned int *d_group_members,
                             const unsigned int nwork,
                             const unsigned int offset,
                             BoxDim box,
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

        // do velocity verlet update
        // r(t+deltaT) = r(t) + v(t)*deltaT + (1/2)a(t)*deltaT^2
        // v(t+deltaT/2) = v(t) + (1/2)a*deltaT

        // read the particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 postype = d_pos[idx];
        Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);

        // read the particle's velocity and acceleration (MEM TRANSFER: 32 bytes)
        Scalar4 velmass = d_vel[idx];
        Scalar3 vel = make_scalar3(velmass.x, velmass.y, velmass.z);

        Scalar3 accel = make_scalar3(Scalar(0.0), Scalar(0.0), Scalar(0.0));
        if (!zero_force)
            accel = d_accel[idx];

        // update the position (FLOPS: 15)
        Scalar3 dx = vel * deltaT + (Scalar(1.0)/Scalar(2.0)) * accel * deltaT * deltaT;

        // limit the movement of the particles
        if (limit)
            {
            Scalar len = sqrtf(dot(dx, dx));
            if (len > limit_val)
                dx = dx / len * limit_val;
            }

        // FLOPS: 3
        pos += dx;

        // update the velocity (FLOPS: 9)
        vel += (Scalar(1.0)/Scalar(2.0)) * accel * deltaT;

        // read in the particle's image (MEM TRANSFER: 16 bytes)
        int3 image = d_image[idx];

        // fix the periodic boundary conditions (FLOPS: 15)
        box.wrap(pos, image);

        // write out the results (MEM_TRANSFER: 48 bytes)
        d_pos[idx] = make_scalar4(pos.x, pos.y, pos.z, postype.w);
        d_vel[idx] = make_scalar4(vel.x, vel.y, vel.z, velmass.w);
        d_image[idx] = image;
        }
    }

/*! \param d_pos array of particle positions
    \param d_vel array of particle velocities
    \param d_accel array of particle accelerations
    \param d_image array of particle images
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param group_size Number of members in the group
    \param box Box dimensions for periodic boundary condition handling
    \param deltaT timestep
    \param limit If \a limit is true, then the dynamics will be limited so that particles do not move
        a distance further than \a limit_val in one step.
    \param limit_val Length to limit particle distance movement to
    \param zero_force Set to true to always assign an acceleration of 0 to all particles in the group

    See gpu_nve_step_one_kernel() for full documentation, this function is just a driver.
*/
cudaError_t gpu_nve_step_one(Scalar4 *d_pos,
                             Scalar4 *d_vel,
                             const Scalar3 *d_accel,
                             int3 *d_image,
                             unsigned int *d_group_members,
                             const GPUPartition& gpu_partition,
                             const BoxDim& box,
                             Scalar deltaT,
                             bool limit,
                             Scalar limit_val,
                             bool zero_force,
                             unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)gpu_nve_step_one_kernel);
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(block_size, max_block_size);

    // iterate over active GPUs in reverse, to end up on first GPU when returning from this function
    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);

        unsigned int nwork = range.second - range.first;

        // setup the grid to run the kernel
        dim3 grid( (nwork/run_block_size) + 1, 1, 1);
        dim3 threads(run_block_size, 1, 1);

        // run the kernel
        gpu_nve_step_one_kernel<<< grid, threads >>>(d_pos, d_vel, d_accel, d_image, d_group_members, nwork, range.first, box, deltaT, limit, limit_val, zero_force);
        }

    return cudaSuccess;
    }

//! NO_SQUISH angular part of the first half step
/*! \param d_orientation array of particle orientations
    \param d_angmom array of particle conjugate quaternions
    \param d_inertia array of moments of inertia
    \param d_net_torque array of net torques
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param group_size Number of members in the group
    \param deltaT timestep
*/
__global__ void gpu_nve_angular_step_one_kernel(Scalar4 *d_orientation,
                             Scalar4 *d_angmom,
                             const Scalar3 *d_inertia,
                             const Scalar4 *d_net_torque,
                             const unsigned int *d_group_members,
                             const unsigned int nwork,
                             const unsigned int offset,
                             Scalar deltaT,
                             Scalar scale)
    {
    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int work_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (work_idx < nwork)
        {
        const unsigned int group_idx = work_idx + offset;
        unsigned int idx = d_group_members[group_idx];

        // read the particle's orientation, conjugate quaternion, moment of inertia and net torque
        quat<Scalar> q(d_orientation[idx]);
        quat<Scalar> p(d_angmom[idx]);
        vec3<Scalar> t(d_net_torque[idx]);
        vec3<Scalar> I(d_inertia[idx]);

        // rotate torque into principal frame
        t = rotate(conj(q),t);

        // check for zero moment of inertia
        bool x_zero, y_zero, z_zero;
        x_zero = (I.x < Scalar(EPSILON)); y_zero = (I.y < Scalar(EPSILON)); z_zero = (I.z < Scalar(EPSILON));

        // ignore torque component along an axis for which the moment of inertia zero
        if (x_zero) t.x = Scalar(0.0);
        if (y_zero) t.y = Scalar(0.0);
        if (z_zero) t.z = Scalar(0.0);

        // advance p(t)->p(t+deltaT/2), q(t)->q(t+deltaT)
        p += deltaT*q*t;

        p = p*scale;

        quat<Scalar> p1, p2, p3; // permutated quaternions
        quat<Scalar> q1, q2, q3;
        Scalar phi1, cphi1, sphi1;
        Scalar phi2, cphi2, sphi2;
        Scalar phi3, cphi3, sphi3;

        if (!z_zero)
            {
            p3 = quat<Scalar>(-p.v.z,vec3<Scalar>(p.v.y,-p.v.x,p.s));
            q3 = quat<Scalar>(-q.v.z,vec3<Scalar>(q.v.y,-q.v.x,q.s));
            phi3 = Scalar(1./4.)/I.z*dot(p,q3);
            cphi3 = slow::cos(Scalar(1./2.)*deltaT*phi3);
            sphi3 = slow::sin(Scalar(1./2.)*deltaT*phi3);

            p=cphi3*p+sphi3*p3;
            q=cphi3*q+sphi3*q3;
            }

        if (!y_zero)
            {
            p2 = quat<Scalar>(-p.v.y,vec3<Scalar>(-p.v.z,p.s,p.v.x));
            q2 = quat<Scalar>(-q.v.y,vec3<Scalar>(-q.v.z,q.s,q.v.x));
            phi2 = Scalar(1./4.)/I.y*dot(p,q2);
            cphi2 = slow::cos(Scalar(1./2.)*deltaT*phi2);
            sphi2 = slow::sin(Scalar(1./2.)*deltaT*phi2);

            p=cphi2*p+sphi2*p2;
            q=cphi2*q+sphi2*q2;
            }

        if (!x_zero)
            {
            p1 = quat<Scalar>(-p.v.x,vec3<Scalar>(p.s,p.v.z,-p.v.y));
            q1 = quat<Scalar>(-q.v.x,vec3<Scalar>(q.s,q.v.z,-q.v.y));
            phi1 = Scalar(1./4.)/I.x*dot(p,q1);
            cphi1 = slow::cos(deltaT*phi1);
            sphi1 = slow::sin(deltaT*phi1);

            p=cphi1*p+sphi1*p1;
            q=cphi1*q+sphi1*q1;
            }

        if (! y_zero)
            {
            p2 = quat<Scalar>(-p.v.y,vec3<Scalar>(-p.v.z,p.s,p.v.x));
            q2 = quat<Scalar>(-q.v.y,vec3<Scalar>(-q.v.z,q.s,q.v.x));
            phi2 = Scalar(1./4.)/I.y*dot(p,q2);
            cphi2 = slow::cos(Scalar(1./2.)*deltaT*phi2);
            sphi2 = slow::sin(Scalar(1./2.)*deltaT*phi2);

            p=cphi2*p+sphi2*p2;
            q=cphi2*q+sphi2*q2;
            }

        if (! z_zero)
            {
            p3 = quat<Scalar>(-p.v.z,vec3<Scalar>(p.v.y,-p.v.x,p.s));
            q3 = quat<Scalar>(-q.v.z,vec3<Scalar>(q.v.y,-q.v.x,q.s));
            phi3 = Scalar(1./4.)/I.z*dot(p,q3);
            cphi3 = slow::cos(Scalar(1./2.)*deltaT*phi3);
            sphi3 = slow::sin(Scalar(1./2.)*deltaT*phi3);

            p=cphi3*p+sphi3*p3;
            q=cphi3*q+sphi3*q3;
            }

        // renormalize (improves stability)
        q = q*(Scalar(1.0)/slow::sqrt(norm2(q)));

        d_orientation[idx] = quat_to_scalar4(q);
        d_angmom[idx] = quat_to_scalar4(p);
        }
    }

/*! \param d_orientation array of particle orientations
    \param d_angmom array of particle conjugate quaternions
    \param d_inertia array of moments of inertia
    \param d_net_torque array of net torques
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param group_size Number of members in the group
    \param deltaT timestep
*/
cudaError_t gpu_nve_angular_step_one(Scalar4 *d_orientation,
                             Scalar4 *d_angmom,
                             const Scalar3 *d_inertia,
                             const Scalar4 *d_net_torque,
                             unsigned int *d_group_members,
                             const GPUPartition& gpu_partition,
                             Scalar deltaT,
                             Scalar scale,
                             const unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void *)gpu_nve_angular_step_one_kernel);
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(block_size, max_block_size);

    // iterate over active GPUs in reverse, to end up on first GPU when returning from this function
    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);

        unsigned int nwork = range.second - range.first;

        // setup the grid to run the kernel
        dim3 grid( (nwork/run_block_size) + 1, 1, 1);
        dim3 threads(run_block_size, 1, 1);

        // run the kernel
        gpu_nve_angular_step_one_kernel<<< grid, threads >>>(d_orientation, d_angmom, d_inertia, d_net_torque, d_group_members, nwork, range.first, deltaT, scale);
        }

    return cudaSuccess;
    }


//! Takes the second half-step forward in the velocity-verlet NVE integration on a group of particles
/*! \param d_vel array of particle velocities
    \param d_accel array of particle accelerations
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param group_size Number of members in the group
    \param d_net_force Net force on each particle
    \param deltaT Amount of real time to step forward in one time step
    \param limit If \a limit is true, then the dynamics will be limited so that particles do not move
        a distance further than \a limit_val in one step.
    \param limit_val Length to limit particle distance movement to
    \param zero_force Set to true to always assign an acceleration of 0 to all particles in the group

    This kernel is implemented in a very similar manner to gpu_nve_step_one_kernel(), see it for design details.
*/
extern "C" __global__
void gpu_nve_step_two_kernel(
                            Scalar4 *d_vel,
                            Scalar3 *d_accel,
                            unsigned int *d_group_members,
                            const unsigned int nwork,
                            const unsigned int offset,
                            Scalar4 *d_net_force,
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
        vel.x += (Scalar(1.0)/Scalar(2.0)) * accel.x * deltaT;
        vel.y += (Scalar(1.0)/Scalar(2.0)) * accel.y * deltaT;
        vel.z += (Scalar(1.0)/Scalar(2.0)) * accel.z * deltaT;

        if (limit)
            {
            Scalar vel_len = sqrtf(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z);
            if ( (vel_len*deltaT) > limit_val)
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
    \param limit If \a limit is true, then the dynamics will be limited so that particles do not move
        a distance further than \a limit_val in one step.
    \param limit_val Length to limit particle distance movement to
    \param zero_force Set to true to always assign an acceleration of 0 to all particles in the group

    This is just a driver for gpu_nve_step_two_kernel(), see it for details.
*/
cudaError_t gpu_nve_step_two(Scalar4 *d_vel,
                             Scalar3 *d_accel,
                             unsigned int *d_group_members,
                             const GPUPartition& gpu_partition,
                             Scalar4 *d_net_force,
                             Scalar deltaT,
                             bool limit,
                             Scalar limit_val,
                             bool zero_force,
                             unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void *)gpu_nve_step_two_kernel);
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(block_size, max_block_size);

    // iterate over active GPUs in reverse, to end up on first GPU when returning from this function
    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);

        unsigned int nwork = range.second - range.first;

        // setup the grid to run the kernel
        dim3 grid( (nwork/run_block_size) + 1, 1, 1);
        dim3 threads(run_block_size, 1, 1);

        // run the kernel
        gpu_nve_step_two_kernel<<< grid, threads >>>(d_vel,
                                                     d_accel,
                                                     d_group_members,
                                                     nwork,
                                                     range.first,
                                                     d_net_force,
                                                     deltaT,
                                                     limit,
                                                     limit_val,
                                                     zero_force);
        }
    return cudaSuccess;
    }

//! NO_SQUISH angular part of the second half step
/*! \param d_orientation array of particle orientations
    \param d_angmom array of particle conjugate quaternions
    \param d_inertia array of moments of inertia
    \param d_net_torque array of net torques
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param group_size Number of members in the group
    \param deltaT timestep
*/
__global__ void gpu_nve_angular_step_two_kernel(const Scalar4 *d_orientation,
                             Scalar4 *d_angmom,
                             const Scalar3 *d_inertia,
                             const Scalar4 *d_net_torque,
                             unsigned int *d_group_members,
                             const unsigned int nwork,
                             const unsigned int offset,
                             Scalar deltaT,
                             Scalar scale)
    {
    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int work_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (work_idx < nwork)
        {
        const unsigned int group_idx = work_idx + offset;
        unsigned int idx = d_group_members[group_idx];

        // read the particle's orientation, conjugate quaternion, moment of inertia and net torque
        quat<Scalar> q(d_orientation[idx]);
        quat<Scalar> p(d_angmom[idx]);
        vec3<Scalar> t(d_net_torque[idx]);
        vec3<Scalar> I(d_inertia[idx]);

        // rotate torque into principal frame
        t = rotate(conj(q),t);

        // check for zero moment of inertia
        bool x_zero, y_zero, z_zero;
        x_zero = (I.x < Scalar(EPSILON)); y_zero = (I.y < Scalar(EPSILON)); z_zero = (I.z < Scalar(EPSILON));

        // ignore torque component along an axis for which the moment of inertia zero
        if (x_zero) t.x = Scalar(0.0);
        if (y_zero) t.y = Scalar(0.0);
        if (z_zero) t.z = Scalar(0.0);

        // rescale
        p = p*scale;

        // advance p(t)->p(t+deltaT/2), q(t)->q(t+deltaT)
        p += deltaT*q*t;

        d_angmom[idx] = quat_to_scalar4(p);
        }
    }

/*! \param d_orientation array of particle orientations
    \param d_angmom array of particle conjugate quaternions
    \param d_inertia array of moments of inertia
    \param d_net_torque array of net torques
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param group_size Number of members in the group
    \param deltaT timestep
*/
cudaError_t gpu_nve_angular_step_two(const Scalar4 *d_orientation,
                             Scalar4 *d_angmom,
                             const Scalar3 *d_inertia,
                             const Scalar4 *d_net_torque,
                             unsigned int *d_group_members,
                             const GPUPartition& gpu_partition,
                             Scalar deltaT,
                             Scalar scale,
                             const unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void *)gpu_nve_angular_step_two_kernel);
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(block_size, max_block_size);

    // iterate over active GPUs in reverse, to end up on first GPU when returning from this function
    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);

        unsigned int nwork = range.second - range.first;

        // setup the grid to run the kernel
        dim3 grid( (nwork/run_block_size) + 1, 1, 1);
        dim3 threads(run_block_size, 1, 1);

        // run the kernel
        gpu_nve_angular_step_two_kernel<<< grid, threads >>>(d_orientation, d_angmom, d_inertia, d_net_torque, d_group_members, nwork, range.first, deltaT, scale);
        }

    return cudaSuccess;
    }
