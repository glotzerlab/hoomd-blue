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

#include "TwoStepBerendsenGPU.cuh"

#include <assert.h>

/*! \file TwoStepBerendsenGPU.cu
    \brief CUDA kernels for BerendsenGPU
*/

// First, the kernel code for the Berendsen thermostat
//! Kernel that applies the first step of a Berendsen integration to a group of particles
/*! \param d_pos array of particle positions
    \param d_vel array of particle velocties
    \param d_accel array of particle accelerations
    \param d_image array of particle images
    \param d_group_members Device array listing the indicies of the members of the group to integrate
    \param group_size Number of members in the group
    \param box Box dimensions for applying periodic boundary conditions
    \param lambda Intermediate variable computed on the host and used in integrating the velocity
    \param deltaT Length of one timestep

    This kernel executes one thread per particle and applies the theromstat to each each. It can be
    run with any 1D block size as long as block_size * num_blocks is >= the number of particles.
*/
extern "C" __global__
void gpu_berendsen_step_one_kernel(Scalar4 *d_pos,
                                   Scalar4 *d_vel,
                                   const Scalar3 *d_accel,
                                   int3 *d_image,
                                   unsigned int *d_group_members,
                                   const unsigned int group_size,
                                   const BoxDim box,
                                   const float lambda,
                                   const float deltaT)
    {
    // determine the particle index for this thread
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];

        // read the particle position
        float4 postype = d_pos[idx];
        float3 pos = make_float3(postype.x, postype.y, postype.z);

        // read the particle velocity and acceleration
        float4 velmass = d_vel[idx];
        float3 vel = make_float3(velmass.x, velmass.y, velmass.z);
        float3 accel = d_accel[idx];

        // integrate velocity and position forward in time
        vel = lambda * (vel + accel * deltaT / 2.0f);
        pos += vel * deltaT;

        // read in the image flags
        int3 image = d_image[idx];

        // apply the periodic boundary conditions
        box.wrap(pos, image);

        // write the results
        d_pos[idx] = make_float4(pos.x, pos.y, pos.z, postype.w);
        d_vel[idx] = make_float4(vel.x, vel.y, vel.z, velmass.w);
        d_image[idx] = image;
        }
    }

//! Kernel that applies the first step of a Berendsen integration to a group of particles
/*! \param d_vel array of particle velocties
    \param d_accel array of particle accelerations
    \param d_group_members Device array listing the indicies of the members of the group to integrate
    \param group_size Number of members in the group
    \param d_net_force Current net force on the particles
    \param deltaT Length of one timestep

    This kernel executes one thread per particle and applies the theromstat to each each. It can be
    run with any 1D block size as long as block_size * num_blocks is >= the number of particles.
*/
extern "C" __global__
void gpu_berendsen_step_two_kernel(Scalar4 *d_vel,
                                   Scalar3 *d_accel,
                                   unsigned int *d_group_members,
                                   const unsigned int group_size,
                                   const float4 *d_net_force,
                                   const float deltaT)
    {
    // determine the particle index for this thread
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];

        // read in the velocity
        float4 velmass = d_vel[idx];
        float3 vel = make_float3(velmass.x, velmass.y, velmass.z);
        float mass = velmass.w;

        // read in the net force and calculate the acceleration
        Scalar4 net_force_energy = d_net_force[idx];
        Scalar3 net_force = make_scalar3(net_force_energy.x, net_force_energy.y, net_force_energy.z);
        Scalar3 accel = net_force / mass;

        // integrate the velocity
        vel += accel * deltaT / 2.0f;

        // write out the velocity and acceleration
        d_vel[idx] = make_scalar4(vel.x, vel.y, vel.z, velmass.w);
        d_accel[idx] = accel;
        }
    }

cudaError_t gpu_berendsen_step_one(Scalar4 *d_pos,
                                   Scalar4 *d_vel,
                                   const Scalar3 *d_accel,
                                   int3 *d_image,
                                   unsigned int *d_group_members,
                                   unsigned int group_size,
                                   const BoxDim& box,
                                   unsigned int block_size,
                                   float lambda,
                                   float deltaT)
    {
    // setup the grid to run the kernel
    dim3 grid( (group_size / block_size) + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    gpu_berendsen_step_one_kernel<<< grid, threads, block_size * sizeof(float) >>>(d_pos,
                                                                                   d_vel,
                                                                                   d_accel,
                                                                                   d_image,
                                                                                   d_group_members,
                                                                                   group_size,
                                                                                   box,
                                                                                   lambda,
                                                                                   deltaT);

    return cudaSuccess;
    }

cudaError_t gpu_berendsen_step_two(Scalar4 *d_vel,
                                   Scalar3 *d_accel,
                                   unsigned int *d_group_members,
                                   unsigned int group_size,
                                   float4 *d_net_force,
                                   unsigned int block_size,
                                   float deltaT)
    {
    // setup the grid to run the kernel
    dim3 grid( (group_size / block_size) + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    gpu_berendsen_step_two_kernel<<< grid, threads, block_size * sizeof(float) >>>(d_vel,
                                                                                   d_accel,
                                                                                   d_group_members,
                                                                                   group_size,
                                                                                   d_net_force,
                                                                                   deltaT);

    return cudaSuccess;
    }

