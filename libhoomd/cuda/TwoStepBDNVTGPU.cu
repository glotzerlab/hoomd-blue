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
extern __shared__ Scalar s_gammas[];

//! Shared memory used in reducing sums for bd energy tally
extern __shared__ Scalar bdtally_sdata[];

//! Takes the second half-step forward in the BDNVT integration on a group of particles with
/*! \param d_pos array of particle positions and types
    \param d_vel array of particle positions and masses
    \param d_accel array of particle accelerations
    \param d_diameter array of particle diameters
    \param d_tag array of particle tags
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
void gpu_bdnvt_step_two_kernel(const Scalar4 *d_pos,
                              Scalar4 *d_vel,
                              Scalar3 *d_accel,
                              const Scalar *d_diameter,
                              const unsigned int *d_tag,
                              unsigned int *d_group_members,
                              unsigned int group_size,
                              Scalar4 *d_net_force,
                              Scalar *d_gamma,
                              unsigned int n_types,
                              bool gamma_diam,
                              unsigned int timestep,
                              unsigned int seed,
                              Scalar T,
                              Scalar deltaT,
                              Scalar D,
                              bool limit,
                              Scalar limit_val,
                              bool tally,
                              Scalar *d_partial_sum_bdenergy)
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
 
    Scalar bd_energy_transfer = 0;
       
    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];
        
        // ******** first, calculate the additional BD force
        // read the current particle velocity (MEM TRANSFER: 16 bytes)
        Scalar4 vel = d_vel[idx];
        // read in the tag of our particle.
        // (MEM TRANSFER: 4 bytes)
        unsigned int ptag = d_tag[idx];
        
        // calculate the magnitude of the random force
        Scalar gamma;
        if (gamma_diam)
            {
            // read in the tag of our particle.
            // (MEM TRANSFER: 4 bytes)
            gamma = d_diameter[idx];
            }
        else
            {
            // read in the type of our particle. A texture read of only the fourth part of the position Scalar4
            // (where type is stored) is used.
            unsigned int typ = __scalar_as_int(d_pos[idx].w);
            gamma = s_gammas[typ];
            }
        
        Scalar coeff = sqrtf(Scalar(6.0) * gamma * T / deltaT);
        Scalar3 bd_force = make_scalar3(Scalar(0.0), Scalar(0.0), Scalar(0.0));
        
        //Initialize the Random Number Generator and generate the 3 random numbers
        SaruGPU s(ptag, timestep + seed); // 2 dimensional seeding
    
        Scalar randomx=s.f(-1.0, 1.0);
        Scalar randomy=s.f(-1.0, 1.0);
        Scalar randomz=s.f(-1.0, 1.0);
        
        bd_force.x = randomx*coeff - gamma*vel.x;
        bd_force.y = randomy*coeff - gamma*vel.y;
        if (D > Scalar(2.0))
            bd_force.z = randomz*coeff - gamma*vel.z;
        
        // read in the net force and calculate the acceleration MEM TRANSFER: 16 bytes
        Scalar4 net_force = d_net_force[idx];
        Scalar3 accel = make_scalar3(net_force.x,net_force.y,net_force.z);
        // MEM TRANSFER: 4 bytes   FLOPS: 3
        Scalar mass = vel.w;
        Scalar minv = Scalar(1.0) / mass;
        accel.x = (accel.x + bd_force.x) * minv;
        accel.y = (accel.y + bd_force.y) * minv;
        accel.z = (accel.z + bd_force.z) * minv;
        
        // v(t+deltaT) = v(t+deltaT/2) + 1/2 * a(t+deltaT)*deltaT
        // update the velocity (FLOPS: 6)
        vel.x += (Scalar(1.0)/Scalar(2.0)) * accel.x * deltaT;
        vel.y += (Scalar(1.0)/Scalar(2.0)) * accel.y * deltaT;
        vel.z += (Scalar(1.0)/Scalar(2.0)) * accel.z * deltaT;
        
        // tally the energy transfer from the bd thermal reservor to the particles (FLOPS: 6)
        bd_energy_transfer =  bd_force.x *vel.x +  bd_force.y * vel.y +  bd_force.z * vel.z;
                        
        if (limit)
            {
            Scalar vel_len = sqrtf(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z);
            if ( (vel_len*deltaT) > limit_val)
                {
                vel.x = vel.x / vel_len * limit_val / deltaT;
                vel.y = vel.y / vel_len * limit_val / deltaT;
                vel.z = vel.z / vel_len * limit_val / deltaT;
                }
            }
            
        // write out data (MEM TRANSFER: 32 bytes)
        d_vel[idx] = vel;
        // since we calculate the acceleration, we need to write it for the next step
        d_accel[idx] = accel;
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
    void gpu_bdtally_reduce_partial_sum_kernel(Scalar *d_sum, 
                                            Scalar* d_partial_sum, 
                                            unsigned int num_blocks)
    {
    Scalar sum = Scalar(0.0);
    
    // sum up the values in the partial sum via a sliding window
    for (int start = 0; start < num_blocks; start += blockDim.x)
        {
        __syncthreads();
        if (start + threadIdx.x < num_blocks)
            bdtally_sdata[threadIdx.x] = d_partial_sum[start + threadIdx.x];
        else
            bdtally_sdata[threadIdx.x] = Scalar(0.0);
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
    

/*! \param d_pos array of particle positions and types
    \param d_vel array of particle positions and masses
    \param d_accel array of particle accelerations
    \param d_diameter array of particle diameters
    \param d_tag array of particle tags
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
cudaError_t gpu_bdnvt_step_two(const Scalar4 *d_pos,
                              Scalar4 *d_vel,
                              Scalar3 *d_accel,
                              const Scalar *d_diameter,
                              const unsigned int *d_tag,
                               unsigned int *d_group_members,
                               unsigned int group_size,
                               Scalar4 *d_net_force,
                               const bdnvt_step_two_args& bdnvt_args,
                               Scalar deltaT,
                               Scalar D,
                               bool limit,
                               Scalar limit_val)
    {
    
    // setup the grid to run the kernel
    dim3 grid(bdnvt_args.num_blocks, 1, 1);
    dim3 grid1(1, 1, 1);
    dim3 threads(bdnvt_args.block_size, 1, 1);
    dim3 threads1(256, 1, 1);

    // run the kernel
    gpu_bdnvt_step_two_kernel<<< grid,
                                 threads,
                                 max((unsigned int)(sizeof(Scalar)*bdnvt_args.n_types),
                                     (unsigned int)(bdnvt_args.block_size*sizeof(Scalar))) 
                             >>>(d_pos,
                                 d_vel,
                                 d_accel,
                                 d_diameter,
                                 d_tag,
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
                                                bdnvt_args.block_size*sizeof(Scalar)
                                             >>>(&bdnvt_args.d_sum_bdenergy[0], 
                                                 bdnvt_args.d_partial_sum_bdenergy, 
                                                 bdnvt_args.num_blocks);    

                                                   
                                                   
    return cudaSuccess;
    }

