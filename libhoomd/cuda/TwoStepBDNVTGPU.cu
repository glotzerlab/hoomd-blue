/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
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

// Maintainer: joaander

#include "TwoStepBDNVTGPU.cuh"

#include "saruprngCUDA.h"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file TwoStepBDNVTGPU.cu
    \brief Defines GPU kernel code for BDNVT integration on the GPU. Used by TwoStepBDNVTGPU.
*/

//! Shared memory array for gpu_bdnvt_step_two_kernel()
extern __shared__ float s_gammas[];

//! Shared memory used in reducing sums for bd energy tally
extern __shared__ float bdtally_sdata[];

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
    \param D Dimensionality of the system
    \param limit If \a limit is true, then the dynamics will be limited so that particles do not move
        a distance further than \a limit_val in one step.
    \param limit_val Length to limit particle distance movement to
    \param tally Boolean indicating whether energy tally is performed or not
    \param d_partial_sum_bdenergy Placeholder for the partial sum

    This kernel is implemented in a very similar manner to gpu_nve_step_one_kernel(), see it for design details.
    
    This kernel will tally the energy transfer from the bd thermal reservoir and the particle system
    
    Random number generation is done per thread with Saru's 3-seed constructor. The seeds are, the time step,
    the particle tag, and the user-defined seed.
    
    This kernel must be launched with enough dynamic shared memory per block to read in d_gamma
*/
extern "C" __global__ 
void gpu_bdnvt_step_two_kernel(gpu_pdata_arrays pdata,
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
                              float D,
                              bool limit,
                              float limit_val,
                              bool tally,
                              float *d_partial_sum_bdenergy)
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
 
    float bd_energy_transfer = 0;
       
    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];
        
        // ******** first, calculate the additional BD force
        // read the current particle velocity (MEM TRANSFER: 16 bytes)
        float4 vel = pdata.vel[idx];
        // read in the tag of our particle.
        // (MEM TRANSFER: 4 bytes)
        unsigned int ptag = pdata.tag[idx];
        
        // calculate the magnitude of the random force
        float gamma;
        if (gamma_diam)
            {
            // read in the tag of our particle.
            // (MEM TRANSFER: 4 bytes)
            gamma = pdata.diameter[idx];
            }
        else
            {
            // read in the type of our particle. A texture read of only the fourth part of the position float4
            // (where type is stored) is used.
            unsigned int typ = __float_as_int(pdata.pos[idx].w);
            gamma = s_gammas[typ];
            }
        
        float coeff = sqrtf(6.0f * gamma * T / deltaT);
        float3 bd_force = make_float3(0.0f, 0.0f, 0.0f);
        
        //Initialize the Random Number Generator and generate the 3 random numbers
        SaruGPU s(ptag, timestep + seed); // 2 dimensional seeding
    
        float randomx=s.f(-1.0, 1.0);
        float randomy=s.f(-1.0, 1.0);
        float randomz=s.f(-1.0, 1.0);
        
        bd_force.x = randomx*coeff - gamma*vel.x;
        bd_force.y = randomy*coeff - gamma*vel.y;
        if (D > 2.0f)
            bd_force.z = randomz*coeff - gamma*vel.z;
        
        // read in the net force and calculate the acceleration MEM TRANSFER: 16 bytes
        float4 accel = d_net_force[idx];
        // MEM TRANSFER: 4 bytes   FLOPS: 3
        float mass = pdata.mass[idx];
        float minv = 1.0f / mass;
        accel.x = (accel.x + bd_force.x) * minv;
        accel.y = (accel.y + bd_force.y) * minv;
        accel.z = (accel.z + bd_force.z) * minv;
        
        // v(t+deltaT) = v(t+deltaT/2) + 1/2 * a(t+deltaT)*deltaT
        // update the velocity (FLOPS: 6)
        vel.x += (1.0f/2.0f) * accel.x * deltaT;
        vel.y += (1.0f/2.0f) * accel.y * deltaT;
        vel.z += (1.0f/2.0f) * accel.z * deltaT;
        
        // tally the energy transfer from the bd thermal reservor to the particles (FLOPS: 6)
        bd_energy_transfer =  bd_force.x *vel.x +  bd_force.y * vel.y +  bd_force.z * vel.z;
                        
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
        pdata.vel[idx] = vel;
        // since we calculate the acceleration, we need to write it for the next step
        pdata.accel[idx] = accel;
        }

    if (tally)
        {
        // don't ovewrite values in the s_gammas array with bd_energy transfer
        __syncthreads();
        bdtally_sdata[threadIdx.x] = bd_energy_transfer;
        __syncthreads();
        
        // reduce the sum in parallel
        int offs = blockDim.x >> 1;
        while (offs > 0)
            {
            if (threadIdx.x < offs)
                bdtally_sdata[threadIdx.x] += bdtally_sdata[threadIdx.x + offs];
            offs >>= 1;
            __syncthreads();
            }
        
        // write out our partial sum
        if (threadIdx.x == 0)
            {
            d_partial_sum_bdenergy[blockIdx.x] = bdtally_sdata[0];
            }
        }
    }


    
//! Kernel function for reducing a partial sum to a full sum (one value)
/*! \param d_sum Placeholder for the sum
    \param d_partial_sum Array containing the parial sum
    \param num_blocks Number of blocks to execute
*/
extern "C" __global__ 
    void gpu_bdtally_reduce_partial_sum_kernel(float *d_sum, 
                                            float* d_partial_sum, 
                                            unsigned int num_blocks)
    {
    float sum = 0.0f;
    
    // sum up the values in the partial sum via a sliding window
    for (int start = 0; start < num_blocks; start += blockDim.x)
        {
        __syncthreads();
        if (start + threadIdx.x < num_blocks)
            bdtally_sdata[threadIdx.x] = d_partial_sum[start + threadIdx.x];
        else
            bdtally_sdata[threadIdx.x] = 0.0f;
        __syncthreads();
        
        // reduce the sum in parallel
        int offs = blockDim.x >> 1;
        while (offs > 0)
            {
            if (threadIdx.x < offs)
                bdtally_sdata[threadIdx.x] += bdtally_sdata[threadIdx.x + offs];
            offs >>= 1;
            __syncthreads();
            }
            
        // everybody sums up sum2K
        sum += bdtally_sdata[0];
        }
        
    if (threadIdx.x == 0)
        *d_sum = sum;
    }
    

/*! \param pdata Particle data to step forward in time
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param d_net_force Net force on each particle
    \param bdnvt_args Collected arguments for gpu_bdnvt_step_two_kernel()
    \param deltaT Amount of real time to step forward in one time step
    \param D Dimensionality of the system
    \param limit If \a limit is true, then the dynamics will be limited so that particles do not move
        a distance further than \a limit_val in one step.
    \param limit_val Length to limit particle distance movement to
        
    This is just a driver for gpu_nve_step_two_kernel(), see it for details.
*/
cudaError_t gpu_bdnvt_step_two(const gpu_pdata_arrays &pdata,
                               unsigned int *d_group_members,
                               unsigned int group_size,
                               float4 *d_net_force,
                               const bdnvt_step_two_args& bdnvt_args,
                               float deltaT,
                               float D,
                               bool limit,
                               float limit_val)
    {
    
    // setup the grid to run the kernel
    dim3 grid(bdnvt_args.num_blocks, 1, 1);
    dim3 grid1(1, 1, 1);
    dim3 threads(bdnvt_args.block_size, 1, 1);
    dim3 threads1(256, 1, 1);

    // run the kernel
    gpu_bdnvt_step_two_kernel<<< grid,
                                 threads,
                                 max((unsigned int)(sizeof(float)*bdnvt_args.n_types),
                                     (unsigned int)(bdnvt_args.block_size*sizeof(float))) 
                             >>>(pdata,
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
                                 D,
                                 limit,
                                 limit_val,
                                 bdnvt_args.tally,
                                 bdnvt_args.d_partial_sum_bdenergy);
                                                   
    // run the summation kernel
    if (bdnvt_args.tally) 
        gpu_bdtally_reduce_partial_sum_kernel<<<grid1,
                                                threads1,
                                                bdnvt_args.block_size*sizeof(float)
                                             >>>(&bdnvt_args.d_sum_bdenergy[0], 
                                                 bdnvt_args.d_partial_sum_bdenergy, 
                                                 bdnvt_args.num_blocks);    

                                                   
                                                   
    return cudaSuccess;
    }

