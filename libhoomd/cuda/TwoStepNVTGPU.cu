/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2014 The Regents of
the University of Michigan All rights reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: joaander

#include "TwoStepNVTGPU.cuh"
#include "VectorMath.h"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file TwoStepNVTGPU.cu
    \brief Defines GPU kernel code for NVT integration on the GPU. Used by TwoStepNVTGPU.
*/

//! Takes the first 1/2 step forward in the NVT integration step
/*! \param d_pos array of particle positions
    \param d_vel array of particle velocities
    \param d_accel array of particle accelerations
    \param d_image array of particle images
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param box Box dimensions for periodic boundary condition handling
    \param denominv Intermediate variable computed on the host and used in the NVT integration step
    \param deltaT Amount of real time to step forward in one time step


    Take the first half step forward in the NVT integration.

    See gpu_nve_step_one_kernel() for some performance notes on how to handle the group data reads efficiently.
*/
extern "C" __global__
void gpu_nvt_step_one_kernel(Scalar4 *d_pos,
                             Scalar4 *d_vel,
                             const Scalar3 *d_accel,
                             int3 *d_image,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             BoxDim box,
                             Scalar denominv,
                             Scalar deltaT)
    {
    // determine which particle this thread works on
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];

        // update positions to the next timestep and update velocities to the next half step
        Scalar4 postype = d_pos[idx];
        Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);

        Scalar4 velmass = d_vel[idx];
        Scalar3 vel = make_scalar3(velmass.x, velmass.y, velmass.z);
        Scalar3 accel = d_accel[idx];

        // perform update computation
        vel = (vel + (Scalar(1.0)/Scalar(2.0)) * accel * deltaT) * denominv;
        pos += vel * deltaT;

        // read in the image flags
        int3 image = d_image[idx];

        // time to fix the periodic boundary conditions
        box.wrap(pos, image);

        // write out the results
        d_pos[idx] = make_scalar4(pos.x, pos.y, pos.z, postype.w);
        d_vel[idx] = make_scalar4(vel.x, vel.y, vel.z, velmass.w);
        d_image[idx] = image;
        }
    }

/*! \param d_pos array of particle positions
    \param d_vel array of particle velocities
    \param d_accel array of particle accelerations
    \param d_image array of particle images
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param box Box dimensions for periodic boundary condition handling
    \param block_size Size of the block to run
    \param Xi Current value of the NVT degree of freedom Xi
    \param deltaT Amount of real time to step forward in one time step
*/
cudaError_t gpu_nvt_step_one(Scalar4 *d_pos,
                             Scalar4 *d_vel,
                             const Scalar3 *d_accel,
                             int3 *d_image,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             const BoxDim& box,
                             unsigned int block_size,
                             Scalar Xi,
                             Scalar deltaT)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, gpu_nvt_step_one_kernel);
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(block_size, max_block_size);

    // setup the grid to run the kernel
    dim3 grid( (group_size/run_block_size) + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    // run the kernel
    gpu_nvt_step_one_kernel<<< grid, threads >>>(d_pos,
                         d_vel,
                         d_accel,
                         d_image,
                         d_group_members,
                         group_size,
                         box,
                         Scalar(1.0) / (Scalar(1.0) + deltaT/Scalar(2.0) * Xi),
                         deltaT);
    return cudaSuccess;
    }

//! Takes the second 1/2 step forward in the NVT integration step
/*! \param d_vel array of particle velocities
    \param d_accel array of particle accelerations
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param d_net_force Net force on each particle
    \param Xi current value of the NVT degree of freedom Xi
    \param deltaT Amount of real time to step forward in one time step
*/
extern "C" __global__
void gpu_nvt_step_two_kernel(Scalar4 *d_vel,
                             Scalar3 *d_accel,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             Scalar4 *d_net_force,
                             Scalar Xi,
                             Scalar deltaT)
    {
    // determine which particle this thread works on
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];

        // read in the net force and calculate the acceleration
        Scalar4 net_force = d_net_force[idx];
        Scalar3 accel = make_scalar3(net_force.x,net_force.y,net_force.z);

        Scalar4 vel = d_vel[idx];

        Scalar mass = vel.w;
        accel.x /= mass;
        accel.y /= mass;
        accel.z /= mass;

        vel.x += (Scalar(1.0)/Scalar(2.0)) * deltaT * (accel.x - Xi * vel.x);
        vel.y += (Scalar(1.0)/Scalar(2.0)) * deltaT * (accel.y - Xi * vel.y);
        vel.z += (Scalar(1.0)/Scalar(2.0)) * deltaT * (accel.z - Xi * vel.z);

        // write out data
        d_vel[idx] = vel;
        // since we calculate the acceleration, we need to write it for the next step
        d_accel[idx] = accel;
        }
    }

/*! \param d_vel array of particle velocities
    \param d_accel array of particle accelerations
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param d_net_force Net force on each particle
    \param block_size Size of the block to execute on the device
    \param Xi current value of the NVT degree of freedom Xi
    \param deltaT Amount of real time to step forward in one time step
*/
cudaError_t gpu_nvt_step_two(Scalar4 *d_vel,
                             Scalar3 *d_accel,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             Scalar4 *d_net_force,
                             unsigned int block_size,
                             Scalar Xi,
                             Scalar deltaT)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, gpu_nvt_step_two_kernel);
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(block_size, max_block_size);

    // setup the grid to run the kernel
    dim3 grid( (group_size/run_block_size) + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    // run the kernel
    gpu_nvt_step_two_kernel<<< grid, threads >>>(d_vel, d_accel, d_group_members, group_size, d_net_force, Xi, deltaT);

    return cudaSuccess;
    }

//! NO_SQUISH angular part of the first half step
/*! \param d_orientation array of particle orientations
    \param d_angmom array of particle conjugate quaternions
    \param d_inertia array of moments of inertia
    \param d_net_torque array of net torques
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param deltaT timestep
*/
__global__ void gpu_nvt_angular_step_one_kernel(Scalar4 *d_orientation,
                             Scalar4 *d_angmom,
                             const Scalar3 *d_inertia,
                             const Scalar4 *d_net_torque,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             Scalar deltaT,
                             Scalar exp_fac)
    {
    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (group_idx < group_size)
        {
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

        p = p*exp_fac;

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
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param deltaT timestep
*/
cudaError_t gpu_nvt_angular_step_one(Scalar4 *d_orientation,
                             Scalar4 *d_angmom,
                             const Scalar3 *d_inertia,
                             const Scalar4 *d_net_torque,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             Scalar deltaT,
                             Scalar exp_fac)
    {
    // setup the grid to run the kernel
    int block_size = 256;
    dim3 grid( (group_size/block_size) + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    gpu_nvt_angular_step_one_kernel<<< grid, threads >>>(d_orientation, d_angmom, d_inertia, d_net_torque, d_group_members, group_size, deltaT, exp_fac);

    return cudaSuccess;
    }

//! NO_SQUISH angular part of the second half step
/*! \param d_orientation array of particle orientations
    \param d_angmom array of particle conjugate quaternions
    \param d_inertia array of moments of inertia
    \param d_net_torque array of net torques
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param deltaT timestep
*/
__global__ void gpu_nvt_angular_step_two_kernel(const Scalar4 *d_orientation,
                             Scalar4 *d_angmom,
                             const Scalar3 *d_inertia,
                             const Scalar4 *d_net_torque,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             Scalar deltaT,
                             Scalar exp_fac)
    {
    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (group_idx < group_size)
        {
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

        // apply thermostat
        p = p*exp_fac;

        // advance p(t)->p(t+deltaT/2), q(t)->q(t+deltaT)
        p += deltaT*q*t;

        d_angmom[idx] = quat_to_scalar4(p);
        }
    }

/*! \param d_orientation array of particle orientations
    \param d_angmom array of particle conjugate quaternions
    \param d_inertia array of moments of inertia
    \param d_net_torque array of net torques
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param deltaT timestep
*/
cudaError_t gpu_nvt_angular_step_two(const Scalar4 *d_orientation,
                             Scalar4 *d_angmom,
                             const Scalar3 *d_inertia,
                             const Scalar4 *d_net_torque,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             Scalar deltaT,
                             Scalar exp_fac)
    {
    // setup the grid to run the kernel
    int block_size = 256;
    dim3 grid( (group_size/block_size) + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    gpu_nvt_angular_step_two_kernel<<< grid, threads >>>(d_orientation, d_angmom, d_inertia, d_net_torque, d_group_members, group_size, deltaT, exp_fac);

    return cudaSuccess;
    }
// vim:syntax=cpp
