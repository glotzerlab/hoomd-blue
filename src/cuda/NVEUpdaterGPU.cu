/*
Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
Source Software License
Copyright (c) 2008 Ames Laboratory Iowa State University
All rights reserved.

Redistribution and use of HOOMD, in source and binary forms, with or
without modification, are permitted, provided that the following
conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names HOOMD's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
// Maintainer: joaander

#include "Integrator.cuh"
#include "NVEUpdaterGPU.cuh"
#include "gpu_settings.h"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

#include <stdio.h>

/*! \file NVEUpdaterGPU.cu
    \brief Defines GPU kernel code for NVE integration on the GPU. Used by NVEUpdaterGPU.
*/

//! The texture for reading the pdata pos array
texture<float4, 1, cudaReadModeElementType> pdata_pos_tex;
//! The texture for reading the pdata vel array
texture<float4, 1, cudaReadModeElementType> pdata_vel_tex;
//! The texture for reading the pdata accel array
texture<float4, 1, cudaReadModeElementType> pdata_accel_tex;
//! The texture for reading in the pdata image array
texture<int4, 1, cudaReadModeElementType> pdata_image_tex;

//! Takes the first half-step forward in the velocity-verlet NVE integration
/*! \param pdata Particle data to step forward 1/2 step
    \param deltaT timestep
    \param limit If \a limit is true, then the dynamics will be limited so that particles do not move
        a distance further than \a limit_val in one step.
    \param limit_val Length to limit particle distance movement to
    \param box Box dimensions for periodic boundary condition handling
*/
extern "C" __global__ void gpu_nve_pre_step_kernel(gpu_pdata_arrays pdata, gpu_boxsize box, float deltaT, bool limit, float limit_val)
    {
    int idx_local = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_global = idx_local + pdata.local_beg;
    // do velocity verlet update
    // r(t+deltaT) = r(t) + v(t)*deltaT + (1/2)a(t)*deltaT^2
    // v(t+deltaT/2) = v(t) + (1/2)a*deltaT
    
    if (idx_local < pdata.local_num)
        {
        // read the particle's posision (MEM TRANSFER: 16 bytes)
        float4 pos = tex1Dfetch(pdata_pos_tex, idx_global);
        
        float px = pos.x;
        float py = pos.y;
        float pz = pos.z;
        float pw = pos.w;
        
        // read the particle's velocity and acceleration (MEM TRANSFER: 32 bytes)
        float4 vel = tex1Dfetch(pdata_vel_tex, idx_global);
        float4 accel = tex1Dfetch(pdata_accel_tex, idx_global);
        
        // update the position (FLOPS: 15)
        float dx = vel.x * deltaT + (1.0f/2.0f) * accel.x * deltaT * deltaT;
        float dy = vel.y * deltaT + (1.0f/2.0f) * accel.y * deltaT * deltaT;
        float dz = vel.z * deltaT + (1.0f/2.0f) * accel.z * deltaT * deltaT;
        
        // limit the movement of the particles
        if (limit)
            {
            float len = sqrtf(dx*dx + dy*dy + dz*dz);
            if (len > limit_val)
                {
                dx = dx / len * limit_val;
                dy = dy / len * limit_val;
                dz = dz / len * limit_val;
                }
            }
            
        // FLOPS: 3
        px += dx;
        py += dy;
        pz += dz;
        
        // update the velocity (FLOPS: 9)
        vel.x += (1.0f/2.0f) * accel.x * deltaT;
        vel.y += (1.0f/2.0f) * accel.y * deltaT;
        vel.z += (1.0f/2.0f) * accel.z * deltaT;
        
        // read in the particle's image
        // read the particle's velocity and acceleration (MEM TRANSFER: 16 bytes)
        int4 image = tex1Dfetch(pdata_image_tex, idx_global);
        
        // time to fix the periodic boundary conditions (FLOPS: 15)
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
        
        // write out the results (MEM_TRANSFER: 48 bytes)
        pdata.pos[idx_global] = pos2;
        pdata.vel[idx_global] = vel;
        pdata.image[idx_global] = image;
        }
    }

/*! \param pdata Particle data to step forward 1/2 step
    \param box Box dimensions for periodic boundary condition handling
    \param deltaT Amount of real time to step forward in one time step
    \param limit If \a limit is true, then the dynamics will be limited so that particles do not move
        a distance further than \a limit_val in one step.
    \param limit_val Length to limit particle distance movement to
*/
cudaError_t gpu_nve_pre_step(const gpu_pdata_arrays &pdata, const gpu_boxsize &box, float deltaT, bool limit, float limit_val)
    {
    // setup the grid to run the kernel
    int block_size = 256;
    dim3 grid( (pdata.local_num/block_size) + 1, 1, 1);
    dim3 threads(block_size, 1, 1);
    
    // bind the textures
    cudaError_t error = cudaBindTexture(0, pdata_pos_tex, pdata.pos, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, pdata_vel_tex, pdata.vel, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, pdata_accel_tex, pdata.accel, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, pdata_image_tex, pdata.image, sizeof(int4) * pdata.N);
    if (error != cudaSuccess)
        return error;
        
    // run the kernel
    gpu_nve_pre_step_kernel<<< grid, threads >>>(pdata, box, deltaT, limit, limit_val);
    
    if (!g_gpu_error_checking)
        {
        return cudaSuccess;
        }
    else
        {
        cudaThreadSynchronize();
        return cudaGetLastError();
        }
    }

//! Takes the 2nd 1/2 step forward in the velocity-verlet NVE integration scheme
/*! \param pdata Particle data to step forward in time
    \param force_data_ptrs List of pointers to forces on each particle
    \param num_forces Number of forces listed in \a force_data_ptrs
    \param deltaT Amount of real time to step forward in one time step
    \param limit If \a limit is true, then the dynamics will be limited so that particles do not move
        a distance further than \a limit_val in one step.
    \param limit_val Length to limit particle distance movement to
*/
extern "C" __global__ void gpu_nve_step_kernel(gpu_pdata_arrays pdata, float4 **force_data_ptrs, int num_forces, float deltaT, bool limit, float limit_val)
    {
    int idx_local = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_global = idx_local + pdata.local_beg;
    // v(t+deltaT) = v(t+deltaT/2) + 1/2 * a(t+deltaT)*deltaT
    
    // note: assumes mass = 1.0
    // sum the acceleration on this particle: (MEM TRANSFER: 16 bytes * number of forces FLOPS: 3 * number of forces)
    float4 accel = gpu_integrator_sum_forces_inline(idx_local, pdata.local_num, force_data_ptrs, num_forces);
    if (idx_local < pdata.local_num)
        {
        // MEM TRANSFER: 4 bytes   FLOPS: 3
        float mass = pdata.mass[idx_global];
        accel.x /= mass;
        accel.y /= mass;
        accel.z /= mass;
        
        // read the current particle velocity (MEM TRANSFER: 16 bytes)
        float4 vel = tex1Dfetch(pdata_vel_tex, idx_global);
        
        // update the velocity (FLOPS: 6)
        vel.x += (1.0f/2.0f) * accel.x * deltaT;
        vel.y += (1.0f/2.0f) * accel.y * deltaT;
        vel.z += (1.0f/2.0f) * accel.z * deltaT;
        
        if (limit)
            {
            float vel_len = sqrtf(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z);
            if ( (vel_len*deltaT) > limit_val)
                {
                vel.x = vel.x / vel_len * limit_val / deltaT;
                vel.y = vel.y / vel_len * limit_val / deltaT;
                vel.z = vel.z / vel_len * limit_val / deltaT;
                }
            }
            
        // write out data (MEM TRANSFER: 32 bytes)
        pdata.vel[idx_global] = vel;
        // since we calculate the acceleration, we need to write it for the next step
        pdata.accel[idx_global] = accel;
        }
    }

/*! \param pdata Particle data to step forward in time
    \param force_data_ptrs List of pointers to forces on each particle
    \param num_forces Number of forces listed in \a force_data_ptrs
    \param deltaT Amount of real time to step forward in one time step
    \param limit If \a limit is true, then the dynamics will be limited so that particles do not move
        a distance further than \a limit_val in one step.
    \param limit_val Length to limit particle distance movement to
*/
cudaError_t gpu_nve_step(const gpu_pdata_arrays &pdata, float4 **force_data_ptrs, int num_forces, float deltaT, bool limit, float limit_val)
    {
    
    // setup the grid to run the kernel
    int block_size = 192;
    dim3 grid( (pdata.local_num/block_size) + 1, 1, 1);
    dim3 threads(block_size, 1, 1);
    
    // bind the texture
    cudaError_t error = cudaBindTexture(0, pdata_vel_tex, pdata.vel, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;
        
    // run the kernel
    gpu_nve_step_kernel<<< grid, threads >>>(pdata, force_data_ptrs, num_forces, deltaT, limit, limit_val);
    
    if (!g_gpu_error_checking)
        {
        return cudaSuccess;
        }
    else
        {
        cudaThreadSynchronize();
        return cudaGetLastError();
        }
    }

