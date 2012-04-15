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

// Maintainer: joaander

#include "TwoStepNVTGPU.cuh"

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
    \param no_wrap_flag Flags to indicate whether periodic boundary conditions should be applied

    
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
                             float denominv,
                             float deltaT,
                             unsigned char no_wrap_flag)
    {
    bool no_wrap_x = no_wrap_flag & 1;
    bool no_wrap_y = no_wrap_flag & 2;
    bool no_wrap_z = no_wrap_flag & 4;

    // determine which particle this thread works on
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];

        // update positions to the next timestep and update velocities to the next half step
        float4 postype = d_pos[idx];
        float3 pos = make_float3(postype.x, postype.y, postype.z);

        Scalar4 velmass = d_vel[idx];
        Scalar3 vel = make_float3(velmass.x, velmass.y, velmass.z);
        float3 accel = d_accel[idx];

        // perform update computation
        vel = (vel + (1.0f/2.0f) * accel * deltaT) * denominv;
        pos += vel * deltaT;

        // read in the image flags
        int3 image = d_image[idx];

        // time to fix the periodic boundary conditions
        box.wrap(pos, image);

        // write out the results
        d_pos[idx] = make_float4(pos.x, pos.y, pos.z, postype.w);
        d_vel[idx] = make_float4(vel.x, vel.y, vel.z, velmass.w);
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
    \param no_wrap_particles Per-direction flag to indicate whether periodic boundary conditions should be applied
*/
cudaError_t gpu_nvt_step_one(Scalar4 *d_pos,
                             Scalar4 *d_vel,
                             const Scalar3 *d_accel,
                             int3 *d_image,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             const BoxDim& box,
                             unsigned int block_size,
                             float Xi,
                             float deltaT,
                             bool no_wrap_particles[])
    {
    // setup the grid to run the kernel
    dim3 grid( (group_size/block_size) + 1, 1, 1);
    dim3 threads(block_size, 1, 1);
   
    unsigned char no_wrap_flag = ((no_wrap_particles[0] ? 1 : 0 ) << 0)
                                |((no_wrap_particles[1] ? 1 : 0 ) << 1)
                                |((no_wrap_particles[2] ? 1 : 0 ) << 2);

    // run the kernel
    gpu_nvt_step_one_kernel<<< grid, threads, block_size * sizeof(float) >>>(d_pos,
                                                                             d_vel,
                                                                             d_accel,
                                                                             d_image,
                                                                             d_group_members,
                                                                             group_size,
                                                                             box,
                                                                             1.0f / (1.0f + deltaT/2.0f * Xi),
                                                                             deltaT,
                                                                             no_wrap_flag);
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
                             float4 *d_net_force,
                             float Xi,
                             float deltaT)
    {
    // determine which particle this thread works on
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];

        // read in the net force and calculate the acceleration
        Scalar4 net_force = d_net_force[idx];
        Scalar3 accel = make_scalar3(net_force.x,net_force.y,net_force.z);

        float4 vel = d_vel[idx];

        float mass = vel.w;
        accel.x /= mass;
        accel.y /= mass;
        accel.z /= mass;
        
        vel.x += (1.0f/2.0f) * deltaT * (accel.x - Xi * vel.x);
        vel.y += (1.0f/2.0f) * deltaT * (accel.y - Xi * vel.y);
        vel.z += (1.0f/2.0f) * deltaT * (accel.z - Xi * vel.z);
        
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
                             float4 *d_net_force,
                             unsigned int block_size,
                             float Xi,
                             float deltaT)
    {
    // setup the grid to run the kernel
    dim3 grid( (group_size/block_size) + 1, 1, 1);
    dim3 threads(block_size, 1, 1);
    
    // run the kernel
    gpu_nvt_step_two_kernel<<< grid, threads >>>(d_vel, d_accel, d_group_members, group_size, d_net_force, Xi, deltaT);
    
    return cudaSuccess;
    }

// vim:syntax=cpp

