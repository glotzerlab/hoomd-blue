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

* All publications based on HOOMD-blue, including any reports or published
results obtained, in whole or in part, with HOOMD-blue, will acknowledge its use
according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website
at: http://codeblue.umich.edu/hoomd-blue/.

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
/*! \param pdata Particle Data to step forward in time
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param box Box dimensions for periodic boundary condition handling
    \param denominv Intermediate variable computed on the host and used in the NVT integration step
    \param deltaT Amount of real time to step forward in one time step
    
    Take the first half step forward in the NVT integration.
    
    See gpu_nve_step_one_kernel() for some performance notes on how to handle the group data reads efficiently.
*/
extern "C" __global__ 
void gpu_nvt_step_one_kernel(gpu_pdata_arrays pdata,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             gpu_boxsize box,
                             float denominv,
                             float deltaT)
    {
    // determine which particle this thread works on
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];
   
        // update positions to the next timestep and update velocities to the next half step
        float4 pos = pdata.pos[idx];
        
        float px = pos.x;
        float py = pos.y;
        float pz = pos.z;
        float pw = pos.w;
        
        float4 vel = pdata.vel[idx];
        float4 accel = pdata.accel[idx];
        
        vel.x = (vel.x + (1.0f/2.0f) * accel.x * deltaT) * denominv;
        px += vel.x * deltaT;
        
        vel.y = (vel.y + (1.0f/2.0f) * accel.y * deltaT) * denominv;
        py += vel.y * deltaT;
        
        vel.z = (vel.z + (1.0f/2.0f) * accel.z * deltaT) * denominv;
        pz += vel.z * deltaT;
        
        // read in the image flags
        int4 image = pdata.image[idx];
        
        // time to fix the periodic boundary conditions
        float x_shift = rintf(px * box.Lxinv);
        px -= box.Lx * x_shift;
        image.x += (int)x_shift;
        
        float y_shift = rintf(py * box.Lyinv);
        py -= box.Ly * y_shift;
        image.y += (int)y_shift;
        
        float z_shift = rintf(pz * box.Lzinv);
        pz -= box.Lz * z_shift;
        image.z += (int)z_shift;
        
        float4 pos2;
        pos2.x = px;
        pos2.y = py;
        pos2.z = pz;
        pos2.w = pw;
        
        // write out the results
        pdata.pos[idx] = pos2;
        pdata.vel[idx] = vel;
        pdata.image[idx] = image;
        }
    }

/*! \param pdata Particle Data to step forward in time
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param box Box dimensions for periodic boundary condition handling
    \param block_size Size of the block to run
    \param Xi Current value of the NVT degree of freedom Xi
    \param deltaT Amount of real time to step forward in one time step
*/
cudaError_t gpu_nvt_step_one(const gpu_pdata_arrays &pdata,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             const gpu_boxsize &box,
                             unsigned int block_size,
                             float Xi,
                             float deltaT)
    {
    // setup the grid to run the kernel
    dim3 grid( (group_size/block_size) + 1, 1, 1);
    dim3 threads(block_size, 1, 1);
    
    // run the kernel
    gpu_nvt_step_one_kernel<<< grid, threads, block_size * sizeof(float) >>>(pdata,
                                                                             d_group_members,
                                                                             group_size,
                                                                             box,
                                                                             1.0f / (1.0f + deltaT/2.0f * Xi),
                                                                             deltaT);
    return cudaSuccess;
    }

//! The texture for reading the net force
texture<float4, 1, cudaReadModeElementType> net_force_tex;

//! Takes the second 1/2 step forward in the NVT integration step
/*! \param pdata Particle Data to step forward in time
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param d_net_force Net force on each particle
    \param Xi current value of the NVT degree of freedom Xi
    \param deltaT Amount of real time to step forward in one time step
*/
extern "C" __global__ 
void gpu_nvt_step_two_kernel(gpu_pdata_arrays pdata,
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
        float4 accel = d_net_force[idx];
        float mass = pdata.mass[idx];
        accel.x /= mass;
        accel.y /= mass;
        accel.z /= mass;
        
        float4 vel = pdata.vel[idx];
        
        vel.x += (1.0f/2.0f) * deltaT * (accel.x - Xi * vel.x);
        vel.y += (1.0f/2.0f) * deltaT * (accel.y - Xi * vel.y);
        vel.z += (1.0f/2.0f) * deltaT * (accel.z - Xi * vel.z);
        
        // write out data
        pdata.vel[idx] = vel;
        // since we calculate the acceleration, we need to write it for the next step
        pdata.accel[idx] = accel;
        }
    }

/*! \param pdata Particle Data to step forward in time
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param d_net_force Net force on each particle
    \param block_size Size of the block to execute on the device
    \param Xi current value of the NVT degree of freedom Xi
    \param deltaT Amount of real time to step forward in one time step
*/
cudaError_t gpu_nvt_step_two(const gpu_pdata_arrays &pdata,
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
    gpu_nvt_step_two_kernel<<< grid, threads >>>(pdata, d_group_members, group_size, d_net_force, Xi, deltaT);
    
    return cudaSuccess;
    }

// vim:syntax=cpp

