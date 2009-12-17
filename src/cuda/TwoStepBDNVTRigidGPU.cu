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
// Maintainer: ndtrung

#include "TwoStepBDNVTGPU.cuh"
#include "TwoStepBDNVTRigidGPU.cuh"
#include "gpu_settings.h"

#include "saruprngCUDA.h"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file TwoStepBDNVTGPU.cu
    \brief Defines GPU kernel code for BDNVT integration on the GPU. Used by TwoStepBDNVTGPU.
*/

//! The texture for reading the pdata vel array
texture<float4, 1, cudaReadModeElementType> pdata_vel_tex;
//! The texture for reading the net force array
texture<float4, 1, cudaReadModeElementType> net_force_tex;
//! The texture for reading the particle mass array
texture<float, 1, cudaReadModeElementType> pdata_mass_tex;
//! The texture for raeding particle types
texture<unsigned int, 1, cudaReadModeElementType> pdata_type_tex;
//! Texture for reading particle diameters
texture<float, 1, cudaReadModeElementType> pdata_diam_tex;
//! Texture for reading particle tags
texture<unsigned int, 1, cudaReadModeElementType> pdata_tag_tex;

//! Shared memory array for gpu_bdnvt_step_two_kernel()
extern __shared__ float s_gammas[];

//! Takes the first half-step forward in the BDNVT integration on a group of particles with
/*! \param pdata Particle data to step forward in time
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param d_net_force Net force on each particle
    \param d_gamma List of per-type gammas
    \param n_types Number of particle types in the simulation
    \param gamma_diam If true, use particle diameters as gamma. If false, read from d_gamma
    \param timestep Current timestep of the simulation
    \param seed User chosen random number seed
    \param T Temperature set point
    \param deltaT Amount of real time to step forward in one time step
    \param limit If \a limit is true, then the dynamics will be limited so that particles do not move
        a distance further than \a limit_val in one step.
    \param limit_val Length to limit particle distance movement to
    
    This kernel is implemented in a very similar manner to gpu_nve_step_one_kernel(), see it for design details.
    
    Random number generation is done per thread with Saru's 3-seed constructor. The seeds are, the time step,
    the particle tag, and the user-defined seed.
    
    This kernel must be launched with enough dynamic shared memory per block to read in d_gamma
*/
extern "C" __global__ 
void gpu_bdnvt_bdforce_kernel(gpu_pdata_arrays pdata,
                              unsigned int *d_group_members,
                              unsigned int group_size,
                              float4 *d_net_force,
                              float *d_gamma,
                              unsigned int n_types,
                              bool gamma_diam,
                              unsigned int timestep,
                              unsigned int seed,
                              float T,
                              float deltaT,
                              bool zero_force)
    {
    if (!gamma_diam)
        {
        // read in the gammas (1 dimensional array)
        for (int cur_offset = 0; cur_offset < n_types; cur_offset += blockDim.x)
            {
            if (cur_offset + threadIdx.x < n_types)
                s_gammas[cur_offset + threadIdx.x] = d_gamma[cur_offset + threadIdx.x];
            }
        __syncthreads();
        }
    
    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];
        
        // calculate the additional BD force
        // read the current particle velocity (MEM TRANSFER: 16 bytes)
        float4 vel = tex1Dfetch(pdata_vel_tex, idx);
        // read in the tag of our particle.
        // (MEM TRANSFER: 4 bytes)
        unsigned int ptag = tex1Dfetch(pdata_tag_tex, idx);
        
        // calculate the magintude of the random force
        float gamma;
        if (gamma_diam)
            {
            // read in the tag of our particle.
            // (MEM TRANSFER: 4 bytes)
            gamma = tex1Dfetch(pdata_diam_tex, idx);
            }
        else
            {
            // read in the type of our particle. A texture read of only the fourth part of the position float4
            // (where type is stored) is used.
            unsigned int typ = tex1Dfetch(pdata_type_tex, idx*4 + 3);
            gamma = s_gammas[typ];
            }
        
        float coeff = sqrtf(6.0f * gamma * T / deltaT);
        float3 bd_force = make_float3(0.0f, 0.0f, 0.0f);
        
        //Initialize the Random Number Generator and generate the 3 random numbers
        SaruGPU s(ptag, timestep, seed); // 3 dimensional seeding
    
        float randomx=s.f(-1.0, 1.0);
        float randomy=s.f(-1.0, 1.0);
        float randomz=s.f(-1.0, 1.0);
        
        bd_force.x = randomx*coeff - gamma*vel.x;
        bd_force.y = randomy*coeff - gamma*vel.y;
        bd_force.z = randomz*coeff - gamma*vel.z;
        
        // read in the net force
        float4 fi = tex1Dfetch(net_force_tex, idx);
        
        // write out data (MEM TRANSFER: 32 bytes)
        d_net_force[idx].x = fi.x + bd_force.x;
        d_net_force[idx].y = fi.y + bd_force.y;
        d_net_force[idx].z = fi.z + bd_force.z;
        d_net_force[idx].w = 0.0f;
        }
    }

/*! \param pdata Particle data to step forward in time
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param d_net_force Net force on each particle
    \param bdnvt_args Collected arguments for gpu_bdnvt_step_two_kernel()
    \param deltaT Amount of real time to step forward in one time step
    \param limit If \a limit is true, then the dynamics will be limited so that particles do not move
        a distance further than \a limit_val in one step.
    \param limit_val Length to limit particle distance movement to
    
*/
cudaError_t gpu_bdnvt_force(const gpu_pdata_arrays &pdata,
                               unsigned int *d_group_members,
                               unsigned int group_size,
                               float4 *d_net_force,
                               const bdnvt_step_two_args& bdnvt_args,
                               float deltaT,
                               bool zero_force)
    {
    
    // setup the grid to run the kernel
    int block_size = 256;
    dim3 grid( (group_size/block_size) + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    // bind the textures
    cudaError_t error = cudaBindTexture(0, pdata_vel_tex, pdata.vel, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;
    
    error = cudaBindTexture(0, pdata_mass_tex, pdata.mass, sizeof(float) * pdata.N);
    if (error != cudaSuccess)
        return error;
    
    error = cudaBindTexture(0, net_force_tex, d_net_force, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;
    
    error = cudaBindTexture(0, pdata_type_tex, pdata.pos, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;
    
    error = cudaBindTexture(0, pdata_diam_tex, pdata.diameter, sizeof(float) * pdata.N);
    if (error != cudaSuccess)
        return error;
    
    error = cudaBindTexture(0, pdata_tag_tex, pdata.tag, sizeof(unsigned int) * pdata.N);
    if (error != cudaSuccess)
        return error;
    
    // run the kernel
    gpu_bdnvt_bdforce_kernel<<< grid, threads, sizeof(float)*bdnvt_args.n_types >>>
                                                  (pdata,
                                                   d_group_members,
                                                   group_size,
                                                   d_net_force,
                                                   bdnvt_args.d_gamma,
                                                   bdnvt_args.n_types,
                                                   bdnvt_args.gamma_diam,
                                                   bdnvt_args.timestep,
                                                   bdnvt_args.seed,
                                                   bdnvt_args.T,
                                                   deltaT,
                                                   zero_force);
    
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

