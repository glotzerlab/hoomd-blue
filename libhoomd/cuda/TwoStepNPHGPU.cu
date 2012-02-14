/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

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

// Maintainer: jglaser

#include "TwoStepNPHGPU.cuh"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file TwoStepNPHGPU.cu
    \brief Defines GPU kernel code for NPT integration on the GPU. Used by TwoStepNPTGPU.
*/

/*! \param d_pos array of particle positions
    \param d_vel array of particle velocities
    \param d_accel array of particle accelerations
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param L_old box lengths at beginning of time step
    \param L_halfstep box lengths at t+deltaT/2
    \param L_final box lengths at t+deltaT
    \param deltaT Time to advance (for one full step)
*/
extern "C" __global__
void gpu_nph_step_one_kernel(Scalar4 *d_pos,
                             Scalar4 *d_vel,
                             const Scalar3 *d_accel,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             float3 L_old,
                             float3 L_halfstep,
                             float3 L_final,
                             float deltaT)
    {
    // determine which particle this thread works on
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // propagate velocity from t to t+1/2*deltaT and position from t to t+deltaT
    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];

        // fetch particle position
        float4 pos = d_pos[idx];

        float px = pos.x;
        float py = pos.y;
        float pz = pos.z;
        float pw = pos.w;

        // fetch particle velocity and acceleration
        float4 vel = d_vel[idx];
        Scalar3 accel = d_accel[idx];

        float4 veltmp;

        // propagate velocity by half a time step and position by the full time step
        // according to the Nose-Hoover barostat
        veltmp.x = vel.x + (1.0f/2.0f) * deltaT*accel.x;
        px = (L_final.x/L_old.x) *(px + veltmp.x*deltaT*L_old.x*L_old.x/L_halfstep.x/L_halfstep.x);
        vel.x = L_old.x/L_final.x*veltmp.x;

        veltmp.y = vel.y + (1.0f/2.0f) * deltaT*accel.y;
        py = (L_final.y/L_old.y) *(py + veltmp.y*deltaT*L_old.y*L_old.y/L_halfstep.y/L_halfstep.y);
        vel.y = L_old.y/L_final.y*veltmp.y;

        veltmp.z = vel.z + (1.0f/2.0f) * deltaT*accel.z;
        pz = (L_final.z/L_old.z) *(pz + veltmp.z*deltaT*L_old.z*L_old.z/L_halfstep.z/L_halfstep.z);
        vel.z = L_old.z/L_final.z*veltmp.z;

        Scalar4 pos2;
        pos2.x = px;
        pos2.y = py;
        pos2.z = pz;
        pos2.w = pw;

        // write out the results
        d_pos[idx] = pos2;
        d_vel[idx] = vel;
        }
    }

/*! \param d_pos array of particle positions
    \param d_vel array of particle velocities
    \param d_accel array of particle accelerations
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param L_old box lengths at beginning of time step
    \param L_halfstep box lengths at t+deltaT/2
    \param L_final box box lengths at t+deltaT
    \param deltaT Time to move forward in one whole step

    This is just a kernel driver for gpu_nph_step_one_kernel(). See it for more details.
*/
cudaError_t gpu_nph_step_one(Scalar4 *d_pos,
                             Scalar4 *d_vel,
                             const Scalar3 *d_accel,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             float3 L_old,
                             float3 L_halfstep,
                             float3 L_final,
                             float deltaT)
    {
    // setup the grid to run the kernel
    unsigned int block_size = 256;
    dim3 grid( (group_size / block_size) + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    gpu_nph_step_one_kernel<<< grid, threads >>>(d_pos,
                                                 d_vel,
                                                 d_accel,
                                                 d_group_members,
                                                 group_size,
                                                 L_old,
                                                 L_halfstep,
                                                 L_final,
                                                 deltaT);

    return cudaSuccess;
    }

/*! \param N number of particles
    \param d_pos array of particle positions
    \param d_image array of particle images
    \param box The new box the particles where the particles now reside

    Wrap particles into new box
*/
extern "C" __global__
void gpu_nph_wrap_particles_kernel(const unsigned int N,
                             Scalar4 *d_pos,
                             int3 *d_image,
                             gpu_boxsize box)
    {
    // determine which particle this thread works on
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // scale ALL particles in the box
    if (idx < N)
        {
        // fetch particle position
        float4 pos = d_pos[idx];

        float px = pos.x;
        float py = pos.y;
        float pz = pos.z;
        float pw = pos.w;

        // read in the image flags
        int3 image = d_image[idx];

        // fix periodic boundary conditions
        float x_shift = rintf(px * box.Lxinv);
        px -= box.Lx * x_shift;
        image.x += (int)x_shift;

        float y_shift = rintf(py * box.Lyinv);
        py -= box.Ly * y_shift;
        image.y += (int)y_shift;

        float z_shift = rintf(pz * box.Lzinv);
        pz -= box.Lz * z_shift;
        image.z += (int)z_shift;

        Scalar4 pos2;
        pos2.x = px;
        pos2.y = py;
        pos2.z = pz;
        pos2.w = pw;

        // write out the results
        d_pos[idx] = pos2;
        d_image[idx] = image;
        }
    }

/*! \param N number of particles
    \param d_pos array of particle positions
    \param d_image array of particle images
    \param box The new box the particles where the particles now reside

    This is just a kernel driver for gpu_nph_wrap_particles_kernel(). See it for more details.
*/
cudaError_t gpu_nph_wrap_particles(const unsigned int N,
                             Scalar4 *d_pos,
                             int3 *d_image,
                             const gpu_boxsize& box)
    {
    // setup the grid to run the kernel
    unsigned int block_size=256;
    dim3 grid( (N / block_size) + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    gpu_nph_wrap_particles_kernel<<< grid, threads >>>(N, d_pos, d_image, box);

    return cudaSuccess;
    }

/*! \param d_vel array of particle velocities
    \param d_accel array of particle accelerations
    \param net_force array of net forces
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param deltaT Time to advance (for one full step)
*/
extern "C" __global__
void gpu_nph_step_two_kernel(Scalar4 *d_vel,
                             Scalar3 *d_accel,
                             float4 *net_force,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             float deltaT)
    {
    // determine which particle this thread works on
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];

        // fetch velocities
        float4 vel = d_vel[idx];

        // read in the net force and compute the acceleration
        float4 accel = net_force[idx];
        float mass = vel.w;
        accel.x /= mass;
        accel.y /= mass;
        accel.z /= mass;

        // propagate velocities from t+1/2*deltaT to t+deltaT
        vel.x +=  (1.0f/2.0f)*deltaT*accel.x;
        vel.y +=  (1.0f/2.0f)*deltaT*accel.y;
        vel.z +=  (1.0f/2.0f)*deltaT*accel.z;

        // write out data
        d_vel[idx] = vel;
        // since we calculate the acceleration, we need to write it for the next step
        d_accel[idx] = make_scalar3(accel.x,accel.y,accel.z);
        }
    }

/*! \param d_vel array of particle velocities
    \param d_accel array of particle accelerations
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param d_net_force Net force on each particle
    \param deltaT Time to move forward in one whole step

    This is just a kernel driver for gpu_nph_step_two_kernel(). See it for more details.
*/
cudaError_t gpu_nph_step_two(Scalar4 *d_vel,
                             Scalar3 *d_accel,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             float4 *d_net_force,
                             float deltaT)
    {
    // setup the grid to run the kernel
    unsigned int block_size=256;
    dim3 grid( (group_size / block_size) + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    gpu_nph_step_two_kernel<<< grid, threads >>>(d_vel, d_accel, d_net_force, d_group_members, group_size, deltaT);

    return cudaSuccess;
    }

// vim:syntax=cpp

