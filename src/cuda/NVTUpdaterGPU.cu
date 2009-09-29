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

#include "NVTUpdaterGPU.cuh"
#include "Integrator.cuh"
#include "gpu_settings.h"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

#include <stdio.h>

//! The texture for reading the pdata pos array
texture<float4, 1, cudaReadModeElementType> pdata_pos_tex;
//! The texture for reading the pdata vel array
texture<float4, 1, cudaReadModeElementType> pdata_vel_tex;
//! The texture for reading the pdata accel array
texture<float4, 1, cudaReadModeElementType> pdata_accel_tex;
//! The texture for reading in the pdata image array
texture<int4, 1, cudaReadModeElementType> pdata_image_tex;

//! Shared memory used in reducing the mv^2 sum
extern __shared__ float nvt_sdata[];

/*! \file NVTUpdaterGPU.cu
    \brief Defines GPU kernel code for NVT integration on the GPU. Used by NVTUpdaterGPU.
*/

//! Takes the first 1/2 step forward in the NVT integration step
/*! \param pdata Particle Data to step forward in time
    \param d_nvt_data Temporary data storage used in the NVT temperature calculation
    \param denominv Intermediate variable computed on the host and used in the NVT integration step
    \param deltaT Amount of real time to step forward in one time step
    \param box Box dimensions for periodic boundary condition handling
*/
extern "C" __global__ 
void gpu_nvt_pre_step_kernel(gpu_pdata_arrays pdata,
							 gpu_boxsize box,
							 gpu_nvt_data d_nvt_data,
							 float denominv,
							 float deltaT)
    {
    int idx_local = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_global = idx_local + pdata.local_beg;
    // do Nose-Hoover integrate
    
    float psq2; //p^2 * 2
    if (idx_local < pdata.local_num)
        {
        // update positions to the next timestep and update velocities to the next half step
        float4 pos = tex1Dfetch(pdata_pos_tex, idx_global);
        
        float px = pos.x;
        float py = pos.y;
        float pz = pos.z;
        float pw = pos.w;
        
        float4 vel = tex1Dfetch(pdata_vel_tex, idx_global);
        float4 accel = tex1Dfetch(pdata_accel_tex, idx_global);
        
        vel.x = (vel.x + (1.0f/2.0f) * accel.x * deltaT) * denominv;
        px += vel.x * deltaT;
        
        vel.y = (vel.y + (1.0f/2.0f) * accel.y * deltaT) * denominv;
        py += vel.y * deltaT;
        
        vel.z = (vel.z + (1.0f/2.0f) * accel.z * deltaT) * denominv;
        pz += vel.z * deltaT;
        
        // read in the image flags
        int4 image = tex1Dfetch(pdata_image_tex, idx_global);
        
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
        pdata.pos[idx_global] = pos2;
        pdata.vel[idx_global] = vel;
        pdata.image[idx_global] = image;
        
        // now we need to do the partial K sums
        
        // compute our contribution to the sum
        float mass = pdata.mass[idx_global];
        psq2 = mass * (vel.x*vel.x + vel.y*vel.y + vel.z*vel.z);
        }
    else
        {
        psq2 = 0.0f;
        }
        
    nvt_sdata[threadIdx.x] = psq2;
    __syncthreads();
    
    // reduce the sum in parallel
    int offs = blockDim.x >> 1;
    while (offs > 0)
        {
        if (threadIdx.x < offs)
            nvt_sdata[threadIdx.x] += nvt_sdata[threadIdx.x + offs];
        offs >>= 1;
        __syncthreads();
        }
        
    // write out our partial sum
    if (threadIdx.x == 0)
        {
        d_nvt_data.partial_Ksum[blockIdx.x] = nvt_sdata[0];
        }
    }

/*! \param pdata Particle Data to step forward in time
    \param box Box dimensions for periodic boundary condition handling
    \param d_nvt_data Temporary data storage used in the NVT temperature calculation
    \param Xi Current value of the NVT degree of freedom Xi
    \param deltaT Amount of real time to step forward in one time step
*/
cudaError_t gpu_nvt_pre_step(const gpu_pdata_arrays &pdata,
							 const gpu_boxsize &box,
							 const gpu_nvt_data &d_nvt_data,
							 float Xi,
							 float deltaT)
    {
    // setup the grid to run the kernel
    int block_size = d_nvt_data.block_size;
    dim3 grid( d_nvt_data.NBlocks, 1, 1);
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
    gpu_nvt_pre_step_kernel<<< grid, threads, block_size * sizeof(float) >>>(pdata, box, d_nvt_data, 1.0f / (1.0f + deltaT/2.0f * Xi), deltaT);
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

//! Takes the second 1/2 step forward in the NVT integration step
/*! \param pdata Particle Data to step forward in time
    \param d_nvt_data Temporary data storage used in the NVT temperature calculation
    \param force_data_ptrs List of pointers to forces on each particle
    \param num_forces Number of forces listed in \a force_data_ptrs
    \param Xi current value of the NVT degree of freedom Xi
    \param deltaT Amount of real time to step forward in one time step
*/
extern "C" __global__ 
void gpu_nvt_step_kernel(gpu_pdata_arrays pdata,
						 gpu_nvt_data d_nvt_data,
						 float4 **force_data_ptrs,
						 int num_forces,
						 float Xi,
						 float deltaT)
    {
    int idx_local = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_global = idx_local + pdata.local_beg;
    
    // note: assumes mass=1.0
    float4 accel = gpu_integrator_sum_forces_inline(idx_local, pdata.local_num, force_data_ptrs, num_forces);
    if (idx_local < pdata.local_num)
        {
        float mass = pdata.mass[idx_global];
        accel.x /= mass;
        accel.y /= mass;
        accel.z /= mass;
        
        float4 vel = tex1Dfetch(pdata_vel_tex, idx_global);
        
        vel.x += (1.0f/2.0f) * deltaT * (accel.x - Xi * vel.x);
        vel.y += (1.0f/2.0f) * deltaT * (accel.y - Xi * vel.y);
        vel.z += (1.0f/2.0f) * deltaT * (accel.z - Xi * vel.z);
        
        // write out data
        pdata.vel[idx_global] = vel;
        // since we calculate the acceleration, we need to write it for the next step
        pdata.accel[idx_global] = accel;
        }
    }

/*! \param pdata Particle Data to step forward in time
    \param d_nvt_data Temporary data storage used in the NVT temperature calculation
    \param force_data_ptrs List of pointers to forces on each particle
    \param num_forces Number of forces listed in \a force_data_ptrs
    \param Xi current value of the NVT degree of freedom Xi
    \param deltaT Amount of real time to step forward in one time step
*/
cudaError_t gpu_nvt_step(const gpu_pdata_arrays &pdata,
						 const gpu_nvt_data &d_nvt_data,
						 float4 **force_data_ptrs,
						 int num_forces,
						 float Xi,
						 float deltaT)
    {
    // setup the grid to run the kernel
    int block_size = d_nvt_data.block_size;
    dim3 grid( d_nvt_data.NBlocks, 1, 1);
    dim3 threads(block_size, 1, 1);
    
    // bind the texture
    cudaError_t error = cudaBindTexture(0, pdata_vel_tex, pdata.vel, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;
        
    // run the kernel
    gpu_nvt_step_kernel<<< grid, threads >>>(pdata, d_nvt_data, force_data_ptrs, num_forces, Xi, deltaT);
    
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


//! Makes the final mv^2 sum on the GPU
/*! \param d_nvt_data Temporary NVT data holding the partial sums

    nvt_pre_step_kernel reduces the mv^2 sum per block. This kernel completes the task
    and makes the final mv^2 sum on each GPU. It is up to the host to read these
    values and get the final total.

    This kernel is designed to be a 1-block kernel for summing the total Ksum
*/
extern "C" __global__ void gpu_nvt_reduce_ksum_kernel(gpu_nvt_data d_nvt_data)
    {
    float Ksum = 0.0f;
    
    // sum up the values in the partial sum via a sliding window
    for (int start = 0; start < d_nvt_data.NBlocks; start += blockDim.x)
        {
        __syncthreads();
        if (start + threadIdx.x < d_nvt_data.NBlocks)
            nvt_sdata[threadIdx.x] = d_nvt_data.partial_Ksum[start + threadIdx.x];
        else
            nvt_sdata[threadIdx.x] = 0.0f;
        __syncthreads();
        
        // reduce the sum in parallel
        int offs = blockDim.x >> 1;
        while (offs > 0)
            {
            if (threadIdx.x < offs)
                nvt_sdata[threadIdx.x] += nvt_sdata[threadIdx.x + offs];
            offs >>= 1;
            __syncthreads();
            }
            
        // everybody sums up Ksum
        Ksum += nvt_sdata[0];
        }
        
    if (threadIdx.x == 0)
        *d_nvt_data.Ksum = Ksum;
    }

/*! \param d_nvt_data Temporary NVT data holding the partial sums

    this is just a driver for nvt_reduce_ksum kernel: see it for details
*/
cudaError_t gpu_nvt_reduce_ksum(const gpu_nvt_data &d_nvt_data)
    {
    // setup the grid to run the kernel
    int block_size = 256;
    dim3 grid( 1, 1, 1);
    dim3 threads(block_size, 1, 1);
    
    // run the kernel
    gpu_nvt_reduce_ksum_kernel<<< grid, threads, block_size*sizeof(float) >>>(d_nvt_data);
    
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


// vim:syntax=cpp
