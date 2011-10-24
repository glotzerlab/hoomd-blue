/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "TwoStepBerendsenGPU.cuh"

#include <assert.h>

/*! \file TwoStepBerendsenGPU.cu
    \brief CUDA kernels for BerendsenGPU
*/

// First, the kernel code for the Berendsen thermostat
//! Kernel that applies the first step of a Berendsen integration to a group of particles
/*! \param pdata Particle data arrays to apply Berendsen thermostat to
    \param d_group_members Device array listing the indicies of the members of the group to integrate
    \param group_size Number of members in the group
    \param box Box dimensions for applying periodic boundary conditions
    \param lambda Intermediate variable computed on the host and used in integrating the velocity
    \param deltaT Length of one timestep

    This kernel executes one thread per particle and applies the theromstat to each each. It can be
    run with any 1D block size as long as block_size * num_blocks is >= the number of particles.
*/
extern "C" __global__
void gpu_berendsen_step_one_kernel(gpu_pdata_arrays pdata,
                                   unsigned int *d_group_members,
                                   unsigned int group_size,
                                   gpu_boxsize box,
                                   float lambda,
                                   float deltaT)
    {
    // determine the particle index for this thread
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];

        // read the particle position
        float4 pos = pdata.pos[idx];

        // we need to use temporary variables to reduce global memory access
        float px = pos.x;
        float py = pos.y;
        float pz = pos.z;
        float pw = pos.w;

        // read the particle velocity and acceleration
        float4 vel = pdata.vel[idx];
        float4 accel = pdata.accel[idx];

        // integrate velocity and position forward in time
        vel.x = lambda * (vel.x + accel.x * deltaT / 2.0f);
        px += vel.x * deltaT;

        vel.y = lambda * (vel.y + accel.y * deltaT / 2.0f);
        py += vel.y * deltaT;

        vel.z = lambda * (vel.z + accel.z * deltaT / 2.0f);
        pz += vel.z * deltaT;

        // read in the image flags
        int4 image = pdata.image[idx];

        // apply the periodic boundary conditions
        float x_shift = rintf(px * box.Lxinv);
        px -= box.Lx * x_shift;
        image.x += (int)x_shift;

        float y_shift = rintf(py * box.Lyinv);
        py -= box.Ly * y_shift;
        image.y += (int) y_shift;

        float z_shift = rintf(pz * box.Lzinv);
        pz -= box.Lz * z_shift;
        image.z += (int)z_shift;

        // another temporary variable
        float4 pos2;
        pos2.x = px;
        pos2.y = py;
        pos2.z = pz;
        pos2.w = pw;

        // write the results
        pdata.pos[idx] = pos2;
        pdata.vel[idx] = vel;
        pdata.image[idx] = image;
        }
    }

//! Kernel that applies the first step of a Berendsen integration to a group of particles
/*! \param pdata Particle data arrays to apply Berendsen thermostat to
    \param d_group_members Device array listing the indicies of the members of the group to integrate
    \param group_size Number of members in the group
    \param d_net_force Current net force on the particles
    \param deltaT Length of one timestep

    This kernel executes one thread per particle and applies the theromstat to each each. It can be
    run with any 1D block size as long as block_size * num_blocks is >= the number of particles.
*/
extern "C" __global__
void gpu_berendsen_step_two_kernel(gpu_pdata_arrays pdata,
                                   unsigned int *d_group_members,
                                   unsigned int group_size,
                                   float4 *d_net_force,
                                   float deltaT)
    {
    // determine the particle index for this thread
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];

        // read in the net force and calculate the acceleration
        float4 accel = d_net_force[idx];
        float mass = pdata.mass[idx];
        accel.x /= mass;
        accel.y /= mass;
        accel.z /= mass;

        // read in teh velocity
        float4 vel = pdata.vel[idx];

        // integrate the velocity
        vel.x = (vel.x + accel.x * deltaT / 2.0f);
        vel.y = (vel.y + accel.y * deltaT / 2.0f);
        vel.z = (vel.z + accel.z * deltaT / 2.0f);

        // write out the velocity and acceleration
        pdata.vel[idx] = vel;
        pdata.accel[idx] = accel;
        }
    }

cudaError_t gpu_berendsen_step_one(const gpu_pdata_arrays &pdata,
                                   unsigned int *d_group_members,
                                   unsigned int group_size,
                                   const gpu_boxsize &box,
                                   unsigned int block_size,
                                   float lambda,
                                   float deltaT)
    {
    // setup the grid to run the kernel
    dim3 grid( (group_size / block_size) + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    gpu_berendsen_step_one_kernel<<< grid, threads, block_size * sizeof(float) >>>(pdata,
                                                                                   d_group_members,
                                                                                   group_size,
                                                                                   box,
                                                                                   lambda,
                                                                                   deltaT);

    return cudaSuccess;
    }

cudaError_t gpu_berendsen_step_two(const gpu_pdata_arrays &pdata,
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
    gpu_berendsen_step_two_kernel<<< grid, threads, block_size * sizeof(float) >>>(pdata,
                                                                                   d_group_members,
                                                                                   group_size,
                                                                                   d_net_force,
                                                                                   deltaT);

    return cudaSuccess;
    }

