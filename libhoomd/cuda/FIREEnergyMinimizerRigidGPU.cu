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

// Maintainer: ndtrung

#include "FIREEnergyMinimizerRigidGPU.cuh"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

#include <stdio.h>

/*! \file FIREEnergyMinimizerRigidGPU.cu
    \brief Defines GPU kernel code for one performing one FIRE energy 
    minimization iteration on the GPU. Used by FIREEnergyMinimizerRigidGPU.
*/

//! The texture for reading the rigid data body indices array
texture<unsigned int, 1, cudaReadModeElementType> rigid_data_body_indices_tex;
//! The texture for reading the rigid data vel array
texture<float4, 1, cudaReadModeElementType> rigid_data_vel_tex;
//! The texture for reading the rigid data angvel array
texture<float4, 1, cudaReadModeElementType> rigid_data_angvel_tex;
//! The texture for reading the rigid data angmom array
texture<float4, 1, cudaReadModeElementType> rigid_data_angmom_tex;
//! The texture for reading the rigid data force array
texture<float4, 1, cudaReadModeElementType> rigid_data_force_tex;
//! The texture for reading the rigid data torque array
texture<float4, 1, cudaReadModeElementType> rigid_data_torque_tex;

//! The texture for reading the net force array
texture<float4, 1, cudaReadModeElementType> net_force_tex;

//! Shared memory used in reducing sums
extern __shared__ float fire_sdata[];
    
#pragma mark ZERO_VELOCITY_KERNEL

//! The kernel function to zeros velocities, called by gpu_fire_rigid_zero_v()
/*! \param rdata_vel Body velocities
    \param rdata_angmom Angular momenta
    \param n_group_bodies Number of rigid bodies in my group
    \param n_bodies Number of rigid bodies
*/
extern "C" __global__ void gpu_fire_rigid_zero_v_kernel(float4* rdata_vel, 
                                                float4* rdata_angmom,
                                                unsigned int n_group_bodies,
                                                unsigned int n_bodies)
    {
    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (group_idx < n_group_bodies)
        {
        unsigned int idx_body = tex1Dfetch(rigid_data_body_indices_tex, group_idx);
        float4 vel = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 angmom = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        
        if (idx_body < n_bodies)
            {
            rdata_vel[idx_body] = vel;
            rdata_angmom[idx_body] = angmom;
            }
        }
    }


/*! \param rdata Rigid data to zero velocities for

This function is just the driver for gpu_fire_rigid_zero_v_kernel(), see that function
for details.
*/
cudaError_t gpu_fire_rigid_zero_v(gpu_rigid_data_arrays rdata)
    {
    unsigned int n_group_bodies = rdata.n_group_bodies;
    unsigned int n_bodies = rdata.n_bodies;
    
    // setup the grid to run the kernel
    unsigned int block_size = 256;
    unsigned int num_blocks = n_group_bodies / block_size + 1;
    dim3 grid(num_blocks, 1, 1);
    dim3 threads(block_size, 1, 1);
    
    cudaError_t error = cudaBindTexture(0, rigid_data_body_indices_tex, rdata.body_indices, sizeof(float) * n_group_bodies);
    if (error != cudaSuccess)
        return error;
        
    // run the kernel
    gpu_fire_rigid_zero_v_kernel<<< grid, threads >>>(rdata.vel, rdata.angmom, n_group_bodies, n_bodies);
    
    return cudaSuccess;
    }

#pragma mark SUMMING_POWER_KERNEL

/*! Kernel function to simultaneously compute the partial sum over Pt, vsq and fsq for the FIRE algorithm
    \param d_sum_Pt Array to hold the sum over Pt (f*v), v2 and f2
    \param n_group_bodies Number of rigid bodies in my group
    \param n_bodies Number of rigid bodies
*/
extern "C" __global__ void gpu_fire_rigid_reduce_Pt_kernel(float* d_sum_Pt, 
                                                            unsigned int n_group_bodies, 
                                                            unsigned int n_bodies)
    {
    unsigned int idx_global = blockIdx.x * blockDim.x + threadIdx.x;
    
    float* body_Pt = fire_sdata;
    float* body_vsq = &fire_sdata[blockDim.x];
    float* body_fsq = &fire_sdata[2*blockDim.x];
    
    float4 force, vel;
    float Pt = 0.0f;
    float vsq = 0.0f;
    float fsq = 0.0f;
    
    // sum up the values via a sliding window
    for (int start = 0; start < n_group_bodies; start += blockDim.x)
        {
        if (start + threadIdx.x < n_group_bodies)
            {
            unsigned int idx_body = tex1Dfetch(rigid_data_body_indices_tex, start + threadIdx.x);
            float Ptrans = 0.0f;
            float v2 = 0.0f;
            float f2 = 0.0;
            
            if (idx_body < n_bodies)
                {
                force = tex1Dfetch(rigid_data_force_tex, idx_body);
                vel = tex1Dfetch(rigid_data_vel_tex, idx_body);
                Ptrans = force.x * vel.x + force.y * vel.y + force.z * vel.z;
                v2 = vel.x * vel.x + vel.y * vel.y + vel.z * vel.z;
                f2 = force.x * force.x + force.y * force.y + force.z * force.z;
                }
            
            body_Pt[threadIdx.x] = Ptrans;
            body_vsq[threadIdx.x] = v2;
            body_fsq[threadIdx.x] = f2;
            }
        else
            {
            body_Pt[threadIdx.x] = 0.0f;
            body_vsq[threadIdx.x] = 0.0f;
            body_fsq[threadIdx.x] = 0.0f;
            }
        
        __syncthreads();
        
        // reduce the sum within a block
        int offset = blockDim.x >> 1;
        while (offset > 0)
            {
            if (threadIdx.x < offset)
                {
                body_Pt[threadIdx.x] += body_Pt[threadIdx.x + offset];
                body_vsq[threadIdx.x] += body_vsq[threadIdx.x + offset];
                body_fsq[threadIdx.x] += body_fsq[threadIdx.x + offset];
                }
            offset >>= 1;
            __syncthreads();
            }
            
        // everybody sums up to the local variables
        Pt += body_Pt[0];
        vsq += body_vsq[0];
        fsq += body_fsq[0];
        }
        
    __syncthreads();
    
    // only one thread write to the global memory
    if (idx_global == 0)
        {
        d_sum_Pt[0] = Pt;
        d_sum_Pt[1] = vsq;
        d_sum_Pt[2] = fsq;
        }
    }
    
/*! Kernel function to simultaneously compute the partial sum over Pr, wsq and tsq for the FIRE algorithm
    \param d_sum_Pr Array to hold the sum over Pr (t*w), w2 and t2
    \param n_group_bodies Number of rigid bodies in my group
    \param n_bodies Number of rigid bodies
*/
extern "C" __global__ void gpu_fire_rigid_reduce_Pr_kernel(float* d_sum_Pr, 
                                                            unsigned int n_group_bodies, 
                                                            unsigned int n_bodies)
    {
    unsigned int idx_global = blockDim.x * blockIdx.x + threadIdx.x;
    
    float* body_Pr = fire_sdata;
    float* body_wsq = &fire_sdata[blockDim.x];
    float* body_tsq = &fire_sdata[2*blockDim.x];
    
    float4 torque, angvel;
    float Pr = 0.0f;
    float wsq = 0.0f;
    float tsq = 0.0f;
    
    // sum up the values via a sliding window
    for (unsigned int start = 0; start < n_group_bodies; start += blockDim.x)
        {
        if (start + threadIdx.x < n_group_bodies)
            {
            unsigned int idx_body = tex1Dfetch(rigid_data_body_indices_tex, start + threadIdx.x);
            float Prot = 0.0f;
            float w2 = 0.0f;
            float t2 = 0.0f;
            
            if (idx_body < n_bodies)
                {
                torque = tex1Dfetch(rigid_data_torque_tex, idx_body);
                angvel = tex1Dfetch(rigid_data_angvel_tex, idx_body);
                Prot = torque.x * angvel.x + torque.y * angvel.y + torque.z * angvel.z;
                w2 = angvel.x * angvel.x + angvel.y * angvel.y + angvel.z * angvel.z;
                t2 = torque.x * torque.x + torque.y * torque.y + torque.z * torque.z;
                }
            
            body_Pr[threadIdx.x] = Prot;
            body_wsq[threadIdx.x] = w2;
            body_tsq[threadIdx.x] = t2;
            }
        else
            {
            body_Pr[threadIdx.x] = 0.0f;
            body_wsq[threadIdx.x] = 0.0f;
            body_tsq[threadIdx.x] = 0.0f;
            }
        
        __syncthreads();
        
        // reduce the sum within a block
        int offset = blockDim.x >> 1;
        while (offset > 0)
            {
            if (threadIdx.x < offset)
                {
                body_Pr[threadIdx.x] += body_Pr[threadIdx.x + offset];
                body_wsq[threadIdx.x] += body_wsq[threadIdx.x + offset];
                body_tsq[threadIdx.x] += body_tsq[threadIdx.x + offset];
                }
            offset >>= 1;
            __syncthreads();
            }
            
        // everybody sums up to the local variables
        Pr += body_Pr[0];
        wsq += body_wsq[0];
        tsq += body_tsq[0];
        }
        
    __syncthreads();
    
    // only one thread write to the global memory
    if (idx_global == 0)
        {
        d_sum_Pr[0] = Pr;
        d_sum_Pr[1] = wsq;
        d_sum_Pr[2] = tsq;
        }
    }


/*! Summing the translational and rotational powers across the rigid bodies
    \param rdata Rigid data to compute the sums for
    \param d_sum_Pt Array to hold the sum over Pt
    \param d_sum_Pr Array to hold the sum over Pr
*/
cudaError_t gpu_fire_rigid_compute_sum_all(const gpu_rigid_data_arrays& rdata, 
                                        float* d_sum_Pt, 
                                        float* d_sum_Pr)
    {
    unsigned int n_bodies = rdata.n_bodies;
    unsigned int n_group_bodies = rdata.n_group_bodies;
    
    cudaError_t error = cudaBindTexture(0, rigid_data_body_indices_tex, rdata.body_indices, sizeof(float) * n_group_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_vel_tex, rdata.vel, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;

    error = cudaBindTexture(0, rigid_data_angvel_tex, rdata.angvel, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_force_tex, rdata.force, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
    
    error = cudaBindTexture(0, rigid_data_torque_tex, rdata.torque, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;

    // setup the grid to run the kernel
    unsigned int block_size = 128;
    dim3 grid( 1, 1, 1);
    dim3 threads(block_size, 1, 1);
    
    // run the kernels
    gpu_fire_rigid_reduce_Pt_kernel<<< grid, threads, 3 * block_size * sizeof(float) >>>(d_sum_Pt, n_group_bodies, n_bodies);
   
    gpu_fire_rigid_reduce_Pr_kernel<<< grid, threads, 3 * block_size * sizeof(float) >>>(d_sum_Pr, n_group_bodies, n_bodies);

    
    return cudaSuccess;
    }


#pragma mark UPDATE_VELOCITY_KERNEL

//! Kernel function to update the velocties used by the FIRE algorithm
/*! \param rdata_vel Body velocities to be updated
    \param rdata_angmom Angular momenta to be updated 
    \param alpha Alpha coupling parameter used by the FIRE algorithm
    \param factor_t factor equal to alpha*vnorm/fnorm
    \param factor_r factor equal to alpha*wnorm/tnorm
    \param n_group_bodies Number of rigid bodies in my group
    \param n_bodies Number of rigid bodies
    \param local_beg Starting body index in this card
*/
extern "C" __global__ void gpu_fire_rigid_update_v_kernel(float4* rdata_vel, 
                                                        float4* rdata_angmom,
                                                        float alpha, 
                                                        float factor_t,
                                                        float factor_r,
                                                        unsigned int n_group_bodies,
                                                        unsigned int n_bodies,
                                                        unsigned int local_beg)
    {
    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x + local_beg;
    
    if (group_idx < n_group_bodies)
        {
        unsigned int idx_body = tex1Dfetch(rigid_data_body_indices_tex, group_idx);
        if (idx_body < n_bodies)
            {        
            // read the body data (MEM TRANSFER: 32 bytes)
            float4 vel = tex1Dfetch(rigid_data_vel_tex, idx_body);
            float4 angmom = tex1Dfetch(rigid_data_angmom_tex, idx_body);
            float4 force = tex1Dfetch(rigid_data_force_tex, idx_body);
            float4 torque = tex1Dfetch(rigid_data_torque_tex, idx_body);
            
            float4 vel2;
            vel2.x = vel.x * (1.0f - alpha) + force.x * factor_t;
            vel2.y = vel.y * (1.0f - alpha) + force.y * factor_t;
            vel2.z = vel.z * (1.0f - alpha) + force.z * factor_t;
            
            float4 angmom2;
            angmom2.x = angmom.x * (1.0f - alpha) + torque.x * factor_r;
            angmom2.y = angmom.y * (1.0f - alpha) + torque.y * factor_r;
            angmom2.z = angmom.z * (1.0f - alpha) + torque.z * factor_r;                
            
            // write out the results (MEM_TRANSFER: 32 bytes)
            rdata_vel[idx_body] = vel2;
            rdata_angmom[idx_body] = angmom2;
            }
        }
    }

/*! \param rdata Rigid data to update the velocities for
    \param alpha Alpha coupling parameter used by the FIRE algorithm
    \param factor_t factor equal to alpha*vnorm/fnorm
    \param factor_r factor equal to alpha*wnorm/tnorm
    This function is a driver for gpu_fire_rigid_update_v_kernel(), see it for details.
*/
cudaError_t gpu_fire_rigid_update_v(gpu_rigid_data_arrays rdata, 
                                                    float alpha, 
                                                    float factor_t,
                                                    float factor_r)
    {
    unsigned int n_bodies = rdata.n_bodies;
    unsigned int n_group_bodies = rdata.n_group_bodies;
    unsigned int local_beg = rdata.local_beg;
    
    // setup the grid to run the kernel
    unsigned int block_size = 256;
    unsigned int num_blocks = n_group_bodies / block_size + 1;
    dim3 grid(num_blocks, 1, 1);
    dim3 threads(block_size, 1, 1);
    
    cudaError_t error = cudaBindTexture(0, rigid_data_body_indices_tex, rdata.body_indices, sizeof(float) * n_group_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_vel_tex, rdata.vel, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;

    error = cudaBindTexture(0, rigid_data_angmom_tex, rdata.angmom, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;

    error = cudaBindTexture(0, rigid_data_force_tex, rdata.force, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
    
    error = cudaBindTexture(0, rigid_data_torque_tex, rdata.torque, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    // run the kernel
    gpu_fire_rigid_update_v_kernel<<< grid, threads >>>(rdata.vel,
                                                    rdata.angmom,
                                                    alpha, 
                                                    factor_t,
                                                    factor_r,
                                                    n_group_bodies,
                                                    n_bodies,
                                                    local_beg);
    
    return cudaSuccess;
    }

