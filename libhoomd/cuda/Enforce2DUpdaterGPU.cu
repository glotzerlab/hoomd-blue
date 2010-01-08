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

// $Id$
// $URL$
// Maintainer: joaander

#include "Enforce2DUpdaterGPU.cuh"
#include "gpu_settings.h"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

#include <stdio.h>

/*! \file Enforce2DUpdaterGPU.cu
    \brief Defines GPU kernel code for constraining systems to a 2D plane on 
    the GPU. Used by Enforce2DUpdaterGPU.
*/

//! The texture for reading the pdata vel array
texture<float4, 1, cudaReadModeElementType> pdata_vel_tex;
//! The texture for reading the pdata accel array
texture<float4, 1, cudaReadModeElementType> pdata_accel_tex;

//! Constrains partcles to the xy plane on the GPU
/*! \param pdata Particle data to constrain to xy plane
*/
extern "C" __global__ 
void gpu_enforce2d_kernel(gpu_pdata_arrays pdata)
    {
    int idx_local = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_global = idx_local + pdata.local_beg;
    
    if (idx_local < pdata.local_num)
        {        
        // read the particle's velocity and acceleration (MEM TRANSFER: 32 bytes)
        float4 vel = tex1Dfetch(pdata_vel_tex, idx_global);
        float4 accel = tex1Dfetch(pdata_accel_tex, idx_global);
                
        // zero the z-velocity and z-acceleration(FLOPS: ?)
        vel.z = 0.0f;
        accel.z = 0.0f;
                
        // write out the results (MEM_TRANSFER: 32 bytes)
        pdata.vel[idx_global] = vel;
        pdata.accel[idx_global] = accel;
        }
    }

/*! \param pdata Particle data to constrain to xy plane
*/
cudaError_t gpu_enforce2d(const gpu_pdata_arrays &pdata)
    {
    // setup the grid to run the kernel
    int block_size = 256;
    dim3 grid( (pdata.local_num/block_size) + 1, 1, 1);
    dim3 threads(block_size, 1, 1);
            
    cudaError_t error = cudaBindTexture(0, pdata_vel_tex, pdata.vel, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, pdata_accel_tex, pdata.accel, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;
                
    // run the kernel
    gpu_enforce2d_kernel<<< grid, threads >>>(pdata);
    
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

