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

// Maintainer: askeys

#include "FIREEnergyMinimizerGPU.cuh"
#include "TextureTools.h"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

#include <stdio.h>

/*! \file FIREEnergyMinimizerGPU.cu
    \brief Defines GPU kernel code for one performing one FIRE energy 
    minimization iteration on the GPU. Used by FIREEnergyMinimizerGPU.
*/

//! Shared memory used in reducing sums
extern __shared__ Scalar fire_sdata[];
//! Shared memory used in simultaneously reducing three sums (currently unused)
extern __shared__ Scalar fire_sdata1[];
//! Shared memory used in simultaneously reducing three sums (currently unused)
extern __shared__ Scalar fire_sdata2[];
//! Shared memory used in simultaneously reducing three sums (currently unused)
extern __shared__ Scalar fire_sdata3[];

//! The kernel function to zeros velocities, called by gpu_fire_zero_v()
/*! \param d_vel device array of particle velocities
    \param d_group_members Device array listing the indicies of the mebers of the group to zero
    \param group_size Number of members in the group
*/
extern "C" __global__ 
void gpu_fire_zero_v_kernel(Scalar4 *d_vel,
                            unsigned int *d_group_members,
                            unsigned int group_size)
    {
    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];
     
        // read the particle's velocity (MEM TRANSFER: 32 bytes)
        Scalar4 vel = d_vel[idx];
                
        // zero the velocity(FLOPS: ?)
        vel.x = Scalar(0.0);
        vel.y = Scalar(0.0);
        vel.z = Scalar(0.0);
                
        // write out the results (MEM_TRANSFER: 32 bytes)
        d_vel[idx] = vel;
        }
    }


/*! \param d_vel device array of particle velocities
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group

This function is just the driver for gpu_fire_zero_v_kernel(), see that function
for details.
*/
cudaError_t gpu_fire_zero_v(Scalar4 *d_vel,
                            unsigned int *d_group_members,
                            unsigned int group_size)
    {
    // setup the grid to run the kernel
    int block_size = 256;
    dim3 grid( (group_size/block_size) + 1, 1, 1);
    dim3 threads(block_size, 1, 1);
            
    // run the kernel
    gpu_fire_zero_v_kernel<<< grid, threads >>>(d_vel,
                                                d_group_members,
                                                group_size);
    
    return cudaSuccess;
    }

//! Kernel function for reducing the potential energy to a partial sum
/*! \param d_group_members Device array listing the indicies of the mebers of the group to sum
    \param group_size Number of members in the group
    \param d_net_force Pointer to the force array for all particles
    \param d_partial_sum_pe Placeholder for the partial sum
*/
extern "C" __global__ 
    void gpu_fire_reduce_pe_partial_kernel(unsigned int *d_group_members,
                                           unsigned int group_size,
                                           Scalar4* d_net_force, 
                                           Scalar* d_partial_sum_pe)
    {
    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    Scalar pe = 0;
    
    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];
        // read the particle's force and extract the pe from w component (MEM TRANSFER: 32 bytes)
        
        Scalar4 force = d_net_force[idx];
        pe = force.w;
        
        // Uncoalesced Memory Read replace by Texture Read above.  Scalars4* d_net_force still being passed to support this
        // defunct structure.
        //pe = d_net_force[idx].w;  
        } 
               
        fire_sdata[threadIdx.x] = pe;
        __syncthreads();
    
    // reduce the sum in parallel
    int offs = blockDim.x >> 1;
    while (offs > 0)
        {
        if (threadIdx.x < offs)
            fire_sdata[threadIdx.x] += fire_sdata[threadIdx.x + offs];
        offs >>= 1;
        __syncthreads();
        }
    
    // write out our partial sum
    if (threadIdx.x == 0)
        {
        d_partial_sum_pe[blockIdx.x] = fire_sdata[0];
        }
    
    }
    
//! Kernel function for reducing a partial sum to a full sum (one value)
/*! \param d_sum Placeholder for the sum
    \param d_partial_sum Array containing the parial sum
    \param num_blocks Number of blocks to execute
*/
extern "C" __global__ 
    void gpu_fire_reduce_partial_sum_kernel(Scalar *d_sum, 
                                            Scalar* d_partial_sum, 
                                            unsigned int num_blocks)
    {
    Scalar sum = Scalar(0.0);
    
    // sum up the values in the partial sum via a sliding window
    for (int start = 0; start < num_blocks; start += blockDim.x)
        {
        __syncthreads();
        if (start + threadIdx.x < num_blocks)
            fire_sdata[threadIdx.x] = d_partial_sum[start + threadIdx.x];
        else
            fire_sdata[threadIdx.x] = Scalar(0.0);
        __syncthreads();
        
        // reduce the sum in parallel
        int offs = blockDim.x >> 1;
        while (offs > 0)
            {
            if (threadIdx.x < offs)
                fire_sdata[threadIdx.x] += fire_sdata[threadIdx.x + offs];
            offs >>= 1;
            __syncthreads();
            }
            
        // everybody sums up sum2K
        sum += fire_sdata[0];
        }
        
    if (threadIdx.x == 0)
        *d_sum = sum;
    }

/*!  \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param d_net_force Array containing the net forces
    \param d_sum_pe Placeholder for the sum of the PE
    \param d_partial_sum_pe Array containing the parial sum of the PE
    \param block_size The size of one block
    \param num_blocks Number of blocks to execute

    This is a driver for gpu_fire_reduce_pe_partial_kernel() and 
    gpu_fire_reduce_partial_sum_kernel(), see them for details
*/
cudaError_t gpu_fire_compute_sum_pe(unsigned int *d_group_members,
                                    unsigned int group_size,
                                    Scalar4* d_net_force, 
                                    Scalar* d_sum_pe, 
                                    Scalar* d_partial_sum_pe, 
                                    unsigned int block_size, 
                                    unsigned int num_blocks)
    {
    

    // setup the grid to run the kernel
    dim3 grid(num_blocks, 1, 1);
    dim3 threads(block_size, 1, 1);
    
    // run the kernel
    gpu_fire_reduce_pe_partial_kernel<<< grid, threads, block_size*sizeof(Scalar) >>>(d_group_members,
                                                                                     group_size,
                                                                                     d_net_force, 
                                                                                     d_partial_sum_pe);
                                                                                     
    gpu_fire_reduce_partial_sum_kernel<<< grid, threads, block_size*sizeof(Scalar) >>>(d_sum_pe, 
                                                                                      d_partial_sum_pe, 
                                                                                      num_blocks);
    
    return cudaSuccess;
    }

//! Kernel function to compute the partial sum over the P term in the FIRE algorithm
/*! \param d_vel particle velocities and masses on the device
    \param d_accel particle accelerations on the device
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param d_partial_sum_P Array to hold the partial sum
*/
extern "C" __global__ 
    void gpu_fire_reduce_P_partial_kernel(const Scalar4 *d_vel,
                                          const Scalar3 *d_accel,
                                          unsigned int *d_group_members,
                                          unsigned int group_size,    
                                          Scalar* d_partial_sum_P)
    {
    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    Scalar P = 0;
    
    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];
     
        Scalar3 a = d_accel[idx];
        Scalar4 v = d_vel[idx];
        P = a.x*v.x + a.y*v.y + a.z*v.z;
        }
    
    fire_sdata[threadIdx.x] = P;
    __syncthreads();

    // reduce the sum in parallel
    int offs = blockDim.x >> 1;
    while (offs > 0)
        {
        if (threadIdx.x < offs)
            fire_sdata[threadIdx.x] += fire_sdata[threadIdx.x + offs];
        offs >>= 1;
        __syncthreads();
        }
    
    // write out our partial sum
    if (threadIdx.x == 0)
        d_partial_sum_P[blockIdx.x] = fire_sdata[0];
        
    }
    
//! Kernel function to compute the partial sum over the vsq term in the FIRE algorithm
/*! \param d_vel device array of particle velocities
    \param d_group_members Array listing members of the group
    \param group_size Number of members in the group
    \param d_partial_sum_vsq Array to hold the partial sum
*/
extern "C" __global__ 
    void gpu_fire_reduce_vsq_partial_kernel(const Scalar4 *d_vel,
                                            unsigned int *d_group_members,
                                            unsigned int group_size,
                                            Scalar* d_partial_sum_vsq)
    {
    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    Scalar vsq = 0;
    
    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];
    
        Scalar4 v = d_vel[idx];
        vsq = v.x*v.x + v.y*v.y + v.z*v.z;
        }
    
    fire_sdata[threadIdx.x] = vsq;
    __syncthreads();
          
    // reduce the sum in parallel
    int offs = blockDim.x >> 1;
    while (offs > 0)
        {
        if (threadIdx.x < offs)
            fire_sdata[threadIdx.x] += fire_sdata[threadIdx.x + offs];
        offs >>= 1;
        __syncthreads();
        }
        
    // write out our partial sum
    if (threadIdx.x == 0)
        d_partial_sum_vsq[blockIdx.x] = fire_sdata[0];
        
    }
    
//! Kernel function to compute the partial sum over the asq term in the FIRE algorithm
/*! \param d_accel device array of particle accelerations
    \param d_group_members Array listing members of the group
    \param group_size Number of members in the group
    \param d_partial_sum_asq Array to hold the partial sum
*/
extern "C" __global__ 
    void gpu_fire_reduce_asq_partial_kernel(const Scalar3 *d_accel,
                                            unsigned int *d_group_members,
                                            unsigned int group_size,
                                            Scalar* d_partial_sum_asq)
    {
    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    Scalar asq = 0;

    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];
    
        Scalar3 a = d_accel[idx];
        asq = a.x*a.x + a.y*a.y + a.z*a.z;
        }

    fire_sdata[threadIdx.x] = asq;
    __syncthreads();
    
    // reduce the sum in parallel
    int offs = blockDim.x >> 1;
    while (offs > 0)
        {
        if (threadIdx.x < offs)
            fire_sdata[threadIdx.x] += fire_sdata[threadIdx.x + offs];
        offs >>= 1;
        __syncthreads();
        }
    
    // write out our partial sum
    if (threadIdx.x == 0)
        d_partial_sum_asq[blockIdx.x] = fire_sdata[0];
        
    }

//! Kernel function to simultaneously compute the partial sum over P, vsq and asq for the FIRE algorithm
/*! \param d_vel device array of particle velocities
    \param d_accel device array of particle accelerations
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param d_partial_sum_P Array to hold the partial sum over P (v*a)
    \param d_partial_sum_vsq Array to hold the partial sum over vsq (v*v)
    \param d_partial_sum_asq Array to hold the partial sum over asq (a*a)
    \note this function is never used, but could be implemented to improve performance
*/
extern "C" __global__ 
    void gpu_fire_reduce_all_partial_kernel(const Scalar4 *d_vel,
                                            const Scalar3 *d_accel,
                                            unsigned int *d_group_members,
                                            unsigned int group_size,
                                            Scalar* d_partial_sum_P, 
                                            Scalar* d_partial_sum_vsq, 
                                            Scalar* d_partial_sum_asq)
    {
    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    Scalar P, vsq, asq; 
    P=0;
    vsq=0;
    asq=0;
    
    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];
    
        Scalar3 a = d_accel[idx];
        Scalar4 v = d_vel[idx];
        P = a.x*v.x + a.y*v.y + a.z*v.z;
        vsq = v.x*v.x + v.y*v.y + v.z*v.z;
        asq = a.x*a.x + a.y*a.y + a.z*a.z;
        }
        
    fire_sdata1[threadIdx.x] = P;
    fire_sdata2[threadIdx.x] = vsq;
    fire_sdata3[threadIdx.x] = asq;
    __syncthreads();

    // reduce the sum in parallel
    int offs = blockDim.x >> 1;
    while (offs > 0)
        {
        if (threadIdx.x < offs)
            {
            fire_sdata1[threadIdx.x] += fire_sdata1[threadIdx.x + offs];
            fire_sdata2[threadIdx.x] += fire_sdata2[threadIdx.x + offs];
            fire_sdata3[threadIdx.x] += fire_sdata3[threadIdx.x + offs];
            }
        offs >>= 1;
        __syncthreads();
        }
    
    // write out our partial sum
    if (threadIdx.x == 0)
        {
        d_partial_sum_P[blockIdx.x] = fire_sdata1[0];
        d_partial_sum_vsq[blockIdx.x] = fire_sdata2[0];
        d_partial_sum_asq[blockIdx.x] = fire_sdata3[0];
        }
    
    }

//! Kernel function to simultaneously reduce three partial sums at the same time
/*! \param d_sum Array to hold the sums
    \param d_partial_sum1 Array containing a precomputed partial sum
    \param d_partial_sum2 Array containing a precomputed partial sum
    \param d_partial_sum3 Array containing a precomputed partial sum
    \param num_blocks The number of blocks to execute
    \note this function is never used, but could be implemented to improve performance
*/
extern "C" __global__ 
    void gpu_fire_reduce_partial_sum_3_kernel(Scalar *d_sum, 
                                              Scalar* d_partial_sum1, 
                                              Scalar* d_partial_sum2, 
                                              Scalar* d_partial_sum3, 
                                              unsigned int num_blocks)
    {
    Scalar sum1 = Scalar(0.0);
    Scalar sum2 = Scalar(0.0);
    Scalar sum3 = Scalar(0.0);
    
    // sum up the values in the partial sum via a sliding window
    for (int start = 0; start < num_blocks; start += blockDim.x)
        {
        __syncthreads();
        if (start + threadIdx.x < num_blocks)
            {
            fire_sdata1[threadIdx.x] = d_partial_sum1[start + threadIdx.x];
            fire_sdata2[threadIdx.x] = d_partial_sum2[start + threadIdx.x];
            fire_sdata3[threadIdx.x] = d_partial_sum3[start + threadIdx.x];
            }
        else
            {
            fire_sdata1[threadIdx.x] = Scalar(0.0);
            fire_sdata2[threadIdx.x] = Scalar(0.0);
            fire_sdata3[threadIdx.x] = Scalar(0.0);
            }
        __syncthreads();
        
        // reduce the sum in parallel
        int offs = blockDim.x >> 1;
        while (offs > 0)
            {
            if (threadIdx.x < offs)
                {
                fire_sdata1[threadIdx.x] += fire_sdata1[threadIdx.x + offs];
                fire_sdata2[threadIdx.x] += fire_sdata2[threadIdx.x + offs];
                fire_sdata3[threadIdx.x] += fire_sdata3[threadIdx.x + offs];
                }
            offs >>= 1;
            __syncthreads();
            }
            
        sum1 += fire_sdata1[0];
        sum2 += fire_sdata2[0];
        sum3 += fire_sdata3[0];
        }
        
    if (threadIdx.x == 0)
        {
        d_sum[0] = sum1;
        d_sum[1] = sum2;
        d_sum[2] = sum3;
        }
    }

/*! \param N number of particles in system
    \param d_vel array of particle velocities
    \param d_accel array of particle accelerations
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param d_sum_all Array to hold the sum over P, vsq, and asq
    \param d_partial_sum_P Array to hold the partial sum over P (a*v)
    \param d_partial_sum_vsq Array to hold the partial sum over vsq (v*v)
    \param d_partial_sum_asq Array to hold the partial sum over asq (a*a)
    \param block_size is the size of one block
    \param num_blocks is the number of blocks to execute
    \note Currently the sums are performed consecutively. The efficiency of this 
        function could be improved by computing all three sums simultaneously
    This is a driver for gpu_fire_reduce_{X}_partial_kernel() (where X = P, vsq, asq)
    and gpu_fire_reduce_partial_sum_kernel(), see them for details
*/
cudaError_t gpu_fire_compute_sum_all(
                                    const unsigned int N,
                                    const Scalar4 *d_vel,
                                    const Scalar3 *d_accel,
                                    unsigned int *d_group_members,
                                    unsigned int group_size,
                                    Scalar* d_sum_all, 
                                    Scalar* d_partial_sum_P, 
                                    Scalar* d_partial_sum_vsq, 
                                    Scalar* d_partial_sum_asq, 
                                    unsigned int block_size, 
                                    unsigned int num_blocks)
    {
    // setup the grid to run the kernel
    dim3 grid(num_blocks, 1, 1);
    dim3 grid1(1, 1, 1);
    dim3 threads(block_size, 1, 1);
    dim3 threads1(256, 1, 1);

    // run the kernels
    gpu_fire_reduce_P_partial_kernel<<< grid, threads, block_size*sizeof(Scalar) >>>(  d_vel,

      d_accel,
                                                                                      d_group_members,
                                                                                      group_size,
                                                                                      d_partial_sum_P);

    gpu_fire_reduce_partial_sum_kernel<<< grid1, threads1, block_size*sizeof(Scalar) >>>(&d_sum_all[0], 
                                                                                      d_partial_sum_P, 
                                                                                      num_blocks);

    gpu_fire_reduce_vsq_partial_kernel<<< grid, threads, block_size*sizeof(Scalar) >>>(d_vel,
                                                                                      d_group_members,
                                                                                      group_size,
                                                                                      d_partial_sum_vsq);

    gpu_fire_reduce_partial_sum_kernel<<< grid1, threads1, block_size*sizeof(Scalar) >>>(&d_sum_all[1], 
                                                                                      d_partial_sum_vsq, 
                                                                                      num_blocks);

    gpu_fire_reduce_asq_partial_kernel<<< grid, threads, block_size*sizeof(Scalar) >>>(d_accel,
                                                                                      d_group_members,
                                                                                      group_size,
                                                                                      d_partial_sum_asq);

    gpu_fire_reduce_partial_sum_kernel<<< grid1, threads1, block_size*sizeof(Scalar) >>>(&d_sum_all[2], 
                                                                                      d_partial_sum_asq, 
                                                                                      num_blocks);

    /*
    //do all three sums at once:
    gpu_fire_reduce_all_partial_kernel<<< grid, threads, block_size*sizeof(Scalar) >>>(d_vel, d_accel, d_partial_sum_P, d_partial_sum_vsq, d_partial_sum_asq);
    gpu_fire_reduce_partial_sum_3_kernel<<< grid, threads, block_size*sizeof(Scalar) >>>(d_sum_all, d_partial_sum_P, d_partial_sum_vsq, d_partial_sum_asq, num_blocks);
    */
    
    return cudaSuccess;
    }


//! Kernel function to update the velocties used by the FIRE algorithm
/*! \param d_vel Array of velocities to update
    \param d_accel Array of accelerations
    \param d_group_members Device array listing the indicies of the mebers of the group to update
    \param group_size Number of members in the grou
    \param alpha Alpha coupling parameter used by the FIRE algorithm
    \param vnorm Magnitude of the (3*N) dimensional velocity vector
    \param invfnorm 1 over the magnitude of the (3*N) dimensional force vector
*/
extern "C" __global__ 
    void gpu_fire_update_v_kernel(Scalar4 *d_vel,
                                  const Scalar3 *d_accel,
                                  unsigned int *d_group_members,
                                  unsigned int group_size,
                                  Scalar alpha, 
                                  Scalar vnorm, 
                                  Scalar invfnorm)
    {
    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];       
        // read the particle's velocity and acceleration (MEM TRANSFER: 32 bytes)
        Scalar4 v = d_vel[idx];
        Scalar3 a = d_accel[idx];
                        
        v.x = v.x*(Scalar(1.0)-alpha) + alpha*a.x*invfnorm*vnorm;
        v.y = v.y*(Scalar(1.0)-alpha) + alpha*a.y*invfnorm*vnorm;
        v.z = v.z*(Scalar(1.0)-alpha) + alpha*a.z*invfnorm*vnorm;
                        
        // write out the results (MEM_TRANSFER: 32 bytes)
        d_vel[idx] = v;
        }
    }


/*! \param d_vel array of particle velocities to update
    \param d_accel array of particle accelerations
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param alpha Alpha coupling parameter used by the FIRE algorithm
    \param vnorm Magnitude of the (3*N) dimensional velocity vector
    \param invfnorm 1 over the magnitude of the (3*N) dimensional force vector
    
    This function is a driver for gpu_fire_update_v_kernel(), see it for details.
*/
cudaError_t gpu_fire_update_v(Scalar4 *d_vel,
                              const Scalar3 *d_accel,
                              unsigned int *d_group_members,
                              unsigned int group_size,
                              Scalar alpha, 
                              Scalar vnorm, 
                              Scalar invfnorm)
    {
    // setup the grid to run the kernel
    int block_size = 256;
    dim3 grid( (group_size/block_size) + 1, 1, 1);
    dim3 threads(block_size, 1, 1);
            
    // run the kernel
    gpu_fire_update_v_kernel<<< grid, threads >>>(d_vel,
                                                  d_accel,
                                                  d_group_members,
                                                  group_size,
                                                  alpha, 
                                                  vnorm, 
                                                  invfnorm);
    
    return cudaSuccess;
    }

