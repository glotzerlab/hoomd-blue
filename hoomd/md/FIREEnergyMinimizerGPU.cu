// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: askeys

#include "FIREEnergyMinimizerGPU.cuh"
#include "hoomd/TextureTools.h"
#include "hoomd/VectorMath.h"

#include <assert.h>

#include <stdio.h>

/*! \file FIREEnergyMinimizerGPU.cu
    \brief Defines GPU kernel code for one performing one FIRE energy
    minimization iteration on the GPU. Used by FIREEnergyMinimizerGPU.
*/

//! Shared memory used in reducing sums
extern __shared__ Scalar fire_sdata[];

//! The kernel function to zeros velocities, called by gpu_fire_zero_v()
/*! \param d_vel device array of particle velocities
    \param d_group_members Device array listing the indices of the members of the group to zero
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


//! The kernel function to zero angular momenta, called by gpu_fire_zero_angmom()
extern "C" __global__
void gpu_fire_zero_angmom_kernel(Scalar4 *d_angmom,
                            unsigned int *d_group_members,
                            unsigned int group_size)
    {
    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];

        // write out the results (MEM_TRANSFER: 32 bytes)
        d_angmom[idx] = make_scalar4(0,0,0,0);
        }
    }


/*! \param d_vel device array of particle velocities
    \param d_group_members Device array listing the indices of the members of the group to integrate
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

cudaError_t gpu_fire_zero_angmom(Scalar4 *d_angmom,
                            unsigned int *d_group_members,
                            unsigned int group_size)
    {
    // setup the grid to run the kernel
    int block_size = 256;
    dim3 grid( (group_size/block_size) + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    gpu_fire_zero_angmom_kernel<<< grid, threads >>>(d_angmom,
                                                d_group_members,
                                                group_size);

    return cudaSuccess;
    }


//! Kernel function for reducing the potential energy to a partial sum
/*! \param d_group_members Device array listing the indices of the members of the group to sum
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
    \param d_partial_sum Array containing the partial sum
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

/*!  \param d_group_members Device array listing the indices of the members of the group to integrate
    \param group_size Number of members in the group
    \param d_net_force Array containing the net forces
    \param d_sum_pe Placeholder for the sum of the PE
    \param d_partial_sum_pe Array containing the partial sum of the PE
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
    \param d_group_members Device array listing the indices of the members of the group to integrate
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

// Angular terms
__global__ void gpu_fire_reduce_Pr_partial_kernel(const Scalar4 *d_angmom,
                                          const Scalar4 *d_orientation,
                                          const Scalar3 *d_inertia,
                                          const Scalar4 *d_net_torque,
                                          unsigned int *d_group_members,
                                          unsigned int group_size,
                                          Scalar* d_partial_sum_Pr)
    {
    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    Scalar Pr = 0;

    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];

        vec3<Scalar> t(d_net_torque[idx]);
        quat<Scalar> p(d_angmom[idx]);
        quat<Scalar> q(d_orientation[idx]);
        vec3<Scalar> I(d_inertia[idx]);

        // rotate torque into principal frame
        t = rotate(conj(q),t);

        // check for zero moment of inertia
        bool x_zero, y_zero, z_zero;
        x_zero = (I.x < EPSILON); y_zero = (I.y < EPSILON); z_zero = (I.z < EPSILON);

        // ignore torque component along an axis for which the moment of inertia zero
        if (x_zero) t.x = 0;
        if (y_zero) t.y = 0;
        if (z_zero) t.z = 0;

        // s is the pure imaginary quaternion with im. part equal to true angular velocity
        vec3<Scalar> s = (Scalar(1./2.) * conj(q) * p).v;

        // rotational power = torque * angvel
        Pr = dot(t,s);
        }

    fire_sdata[threadIdx.x] = Pr;
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
        d_partial_sum_Pr[blockIdx.x] = fire_sdata[0];

    }

// Norm of angular velocity vector
__global__ void gpu_fire_reduce_wnorm_partial_kernel(const Scalar4 *d_angmom,
                                          const Scalar4 *d_orientation,
                                          unsigned int *d_group_members,
                                          unsigned int group_size,
                                          Scalar* d_partial_sum_w)
    {
    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    Scalar w = 0;

    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];

        quat<Scalar> p(d_angmom[idx]);
        quat<Scalar> q(d_orientation[idx]);
        vec3<Scalar> s = (Scalar(1./2.) * conj(q) * p).v;

        w = dot(s,s);
        }

    fire_sdata[threadIdx.x] = w;
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
        d_partial_sum_w[blockIdx.x] = fire_sdata[0];

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

__global__ void gpu_fire_reduce_tsq_partial_kernel(const Scalar4 *d_net_torque,
                                            const Scalar4 *d_orientation,
                                            const Scalar3 *d_inertia,
                                            unsigned int *d_group_members,
                                            unsigned int group_size,
                                            Scalar* d_partial_sum_tsq)
    {
    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    Scalar tsq = 0;

    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];

        vec3<Scalar> t(d_net_torque[idx]);
        quat<Scalar> q(d_orientation[idx]);
        vec3<Scalar> I(d_inertia[idx]);

        // rotate torque into principal frame
        t = rotate(conj(q),t);

        // check for zero moment of inertia
        bool x_zero, y_zero, z_zero;
        x_zero = (I.x < EPSILON); y_zero = (I.y < EPSILON); z_zero = (I.z < EPSILON);

        // ignore torque component along an axis for which the moment of inertia zero
        if (x_zero) t.x = 0;
        if (y_zero) t.y = 0;
        if (z_zero) t.z = 0;

        tsq = dot(t,t);
        }

    fire_sdata[threadIdx.x] = tsq;
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
        d_partial_sum_tsq[blockIdx.x] = fire_sdata[0];

    }



/*! \param N number of particles in system
    \param d_vel array of particle velocities
    \param d_accel array of particle accelerations
    \param d_group_members Device array listing the indices of the members of the group to integrate
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

    return cudaSuccess;
    }

cudaError_t gpu_fire_compute_sum_all_angular(const unsigned int N,
                                    const Scalar4 *d_orientation,
                                    const Scalar3 *d_inertia,
                                    const Scalar4 *d_angmom,
                                    const Scalar4 *d_net_torque,
                                    unsigned int *d_group_members,
                                    unsigned int group_size,
                                    Scalar* d_sum_all,
                                    Scalar* d_partial_sum_Pr,
                                    Scalar* d_partial_sum_wnorm,
                                    Scalar* d_partial_sum_tsq,
                                    unsigned int block_size,
                                    unsigned int num_blocks)
    {
    // setup the grid to run the kernel
    dim3 grid(num_blocks, 1, 1);
    dim3 grid1(1, 1, 1);
    dim3 threads(block_size, 1, 1);
    dim3 threads1(256, 1, 1);

    // run the kernels
    gpu_fire_reduce_Pr_partial_kernel<<< grid, threads, block_size*sizeof(Scalar) >>>(  d_angmom,
                                                                                        d_orientation,
                                                                                        d_inertia,
                                                                                        d_net_torque,
                                                                                      d_group_members,
                                                                                      group_size,
                                                                                      d_partial_sum_Pr);

    gpu_fire_reduce_partial_sum_kernel<<< grid1, threads1, block_size*sizeof(Scalar) >>>(&d_sum_all[0],
                                                                                      d_partial_sum_Pr,
                                                                                      num_blocks);

    gpu_fire_reduce_wnorm_partial_kernel<<< grid, threads, block_size*sizeof(Scalar) >>>(d_angmom,
                                                                                       d_orientation,
                                                                                      d_group_members,
                                                                                      group_size,
                                                                                      d_partial_sum_wnorm);

    gpu_fire_reduce_partial_sum_kernel<<< grid1, threads1, block_size*sizeof(Scalar) >>>(&d_sum_all[1],
                                                                                      d_partial_sum_wnorm,
                                                                                      num_blocks);

    gpu_fire_reduce_tsq_partial_kernel<<< grid, threads, block_size*sizeof(Scalar) >>>(d_net_torque,
                                                                                      d_orientation,
                                                                                      d_inertia,
                                                                                      d_group_members,
                                                                                      group_size,
                                                                                      d_partial_sum_tsq);

    gpu_fire_reduce_partial_sum_kernel<<< grid1, threads1, block_size*sizeof(Scalar) >>>(&d_sum_all[2],
                                                                                      d_partial_sum_tsq,
                                                                                      num_blocks);

    return cudaSuccess;
    }



//! Kernel function to update the velocities used by the FIRE algorithm
/*! \param d_vel Array of velocities to update
    \param d_accel Array of accelerations
    \param d_group_members Device array listing the indices of the members of the group to update
    \param group_size Number of members in the grou
    \param alpha Alpha coupling parameter used by the FIRE algorithm
    \param factor_t Combined factor vnorm/fnorm*alpha, or 1 if fnorm==0
*/
extern "C" __global__
    void gpu_fire_update_v_kernel(Scalar4 *d_vel,
                                  const Scalar3 *d_accel,
                                  unsigned int *d_group_members,
                                  unsigned int group_size,
                                  Scalar alpha,
                                  Scalar factor_t)
    {
    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];
        // read the particle's velocity and acceleration (MEM TRANSFER: 32 bytes)
        Scalar4 v = d_vel[idx];
        Scalar3 a = d_accel[idx];

        v.x = v.x*(Scalar(1.0)-alpha) + a.x*factor_t;
        v.y = v.y*(Scalar(1.0)-alpha) + a.y*factor_t;
        v.z = v.z*(Scalar(1.0)-alpha) + a.z*factor_t;

        // write out the results (MEM_TRANSFER: 32 bytes)
        d_vel[idx] = v;
        }
    }


/*! \param d_vel array of particle velocities to update
    \param d_accel array of particle accelerations
    \param d_group_members Device array listing the indices of the members of the group to integrate
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
                              Scalar factor_t)
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
                                                  factor_t);

    return cudaSuccess;
    }

 __global__ void gpu_fire_update_angmom_kernel(const Scalar4 *d_net_torque,
                                  const Scalar4 *d_orientation,
                                  const Scalar3 *d_inertia,
                                  Scalar4 *d_angmom,
                                  unsigned int *d_group_members,
                                  unsigned int group_size,
                                  Scalar alpha,
                                  Scalar factor_r)
    {
    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];
        quat<Scalar> q(d_orientation[idx]);
        vec3<Scalar> t(d_net_torque[idx]);
        quat<Scalar> p(d_angmom[idx]);
        vec3<Scalar> I(d_inertia[idx]);

        // rotate torque into principal frame
        t = rotate(conj(q),t);

        // check for zero moment of inertia
        bool x_zero, y_zero, z_zero;
        x_zero = (I.x < EPSILON); y_zero = (I.y < EPSILON); z_zero = (I.z < EPSILON);

        // ignore torque component along an axis for which the moment of inertia zero
        if (x_zero) t.x = 0;
        if (y_zero) t.y = 0;
        if (z_zero) t.z = 0;

        p = p*Scalar(1.0-alpha) + Scalar(2.0)*q*t*factor_r;

        d_angmom[idx] = quat_to_scalar4(p);
        }
    }

cudaError_t gpu_fire_update_angmom(const Scalar4 *d_net_torque,
                              const Scalar4 *d_orientation,
                              const Scalar3 *d_inertia,
                              Scalar4 *d_angmom,
                              unsigned int *d_group_members,
                              unsigned int group_size,
                              Scalar alpha,
                              Scalar factor_r)
    {
    // setup the grid to run the kernel
    int block_size = 256;
    dim3 grid( (group_size/block_size) + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    gpu_fire_update_angmom_kernel<<< grid, threads >>>(d_net_torque,
                                                  d_orientation,
                                                  d_inertia,
                                                  d_angmom,
                                                  d_group_members,
                                                  group_size,
                                                  alpha,
                                                  factor_r);

    return cudaSuccess;
    }
