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
// Maintainer: ndtrung

#include "QuaternionMath.h"
#include "TwoStepNVERigidGPU.cuh"
#include <stdio.h>

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file TwoStepNVERigidGPU.cu
    \brief Defines GPU kernel code for NVE integration on the GPU. Used by NVEUpdaterGPU.
*/

//! Flag for invalid particle index, identical to the sentinel value NO_INDEX in RigidData.h
#define INVALID_INDEX 0xffffffff 

#pragma mark RIGID_STEP_ONE_KERNEL
/*! Takes the first half-step forward for rigid bodies in the velocity-verlet NVE integration
    \param rdata_com Body center of mass
    \param rdata_vel Body translational velocity
    \param rdata_angmom Angular momentum
    \param rdata_angvel Angular velocity
    \param rdata_orientation Quaternion
    \param rdata_ex_space x-axis unit vector
    \param rdata_ey_space y-axis unit vector
    \param rdata_ez_space z-axis unit vector
    \param rdata_body_image Body image
    \param rdata_conjqm Conjugate quaternion momentum
    \param d_rigid_mass Body mass
    \param d_rigid_mi Body inertia moments
    \param d_rigid_force Body forces
    \param d_rigid_torque Body torques
    \param d_rigid_group Body indices
    \param n_group_bodies Number of rigid bodies in my group
    \param n_bodies Total number of rigid bodies
    \param deltaT Timestep 
    \param box Box dimensions for periodic boundary condition handling
*/
extern "C" __global__ void gpu_nve_rigid_step_one_body_kernel(float4* rdata_com, 
                                                        float4* rdata_vel, 
                                                        float4* rdata_angmom, 
                                                        float4* rdata_angvel,
                                                        float4* rdata_orientation, 
                                                        float4* rdata_ex_space, 
                                                        float4* rdata_ey_space, 
                                                        float4* rdata_ez_space, 
                                                        int3* rdata_body_image, 
                                                        float4* rdata_conjqm,
                                                        float *d_rigid_mass,
                                                        float4 *d_rigid_mi,
                                                        float4 *d_rigid_force,
                                                        float4 *d_rigid_torque,
                                                        unsigned int *d_rigid_group,
                                                        unsigned int n_group_bodies,
                                                        unsigned int n_bodies, 
                                                        gpu_boxsize box, 
                                                        float deltaT)
    {
    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (group_idx >= n_group_bodies)
        return;
    
    // do velocity verlet update
    // v(t+deltaT/2) = v(t) + (1/2)a*deltaT
    // r(t+deltaT) = r(t) + v(t+deltaT/2)*deltaT
    float body_mass;
    float4 moment_inertia, com, vel, angmom, orientation, ex_space, ey_space, ez_space, force, torque;
    int3 body_image;
    float dt_half = 0.5f * deltaT;
        
    unsigned int idx_body = d_rigid_group[group_idx];
    body_mass = d_rigid_mass[idx_body];
    moment_inertia = d_rigid_mi[idx_body];
    com = rdata_com[idx_body];
    vel = rdata_vel[idx_body];
    angmom = rdata_angmom[idx_body];
    orientation = rdata_orientation[idx_body];
    body_image = rdata_body_image[idx_body];
    force = d_rigid_force[idx_body];
    torque = d_rigid_torque[idx_body];
    
    exyzFromQuaternion(orientation, ex_space, ey_space, ez_space);
        
    // update velocity
    float dtfm = dt_half / body_mass;
    
    float4 vel2;
    vel2.x = vel.x + dtfm * force.x;
    vel2.y = vel.y + dtfm * force.y;
    vel2.z = vel.z + dtfm * force.z;
    vel2.w = vel.w;
    
    // update position
    float4 pos2;
    pos2.x = com.x + vel2.x * deltaT;
    pos2.y = com.y + vel2.y * deltaT;
    pos2.z = com.z + vel2.z * deltaT;
    pos2.w = com.w;
    
    // time to fix the periodic boundary conditions
    float x_shift = rintf(pos2.x * box.Lxinv);
    pos2.x -= box.Lx * x_shift;
    body_image.x += (int)x_shift;
    
    float y_shift = rintf(pos2.y * box.Lyinv);
    pos2.y -= box.Ly * y_shift;
    body_image.y += (int)y_shift;
    
    float z_shift = rintf(pos2.z * box.Lzinv);
    pos2.z -= box.Lz * z_shift;
    body_image.z += (int)z_shift;

    // update the angular momentum and angular velocity
    float4 angmom2;
    angmom2.x = angmom.x + dt_half * torque.x;
    angmom2.y = angmom.y + dt_half * torque.y;
    angmom2.z = angmom.z + dt_half * torque.z;
    angmom2.w = 0.0f;
    
    float4 angvel2;
    advanceQuaternion(angmom2, moment_inertia, angvel2, ex_space, ey_space, ez_space, deltaT, orientation);

    // write out the results
    rdata_com[idx_body] = pos2;
    rdata_vel[idx_body] = vel2;
    rdata_angmom[idx_body] = angmom2;
    rdata_angvel[idx_body] = angvel2;
    rdata_orientation[idx_body] = orientation;
    rdata_ex_space[idx_body] = ex_space;
    rdata_ey_space[idx_body] = ey_space;
    rdata_ez_space[idx_body] = ez_space;
    rdata_body_image[idx_body] = body_image;
    }

// Takes the first 1/2 step forward in the NVE integration step
/*! \param pdata Particle data to step forward 1/2 step
    \param rigid_data Rigid body data to step forward 1/2 step
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param d_net_force Particle net forces
    \param box Box dimensions for periodic boundary condition handling
    \param deltaT Amount of real time to step forward in one time step
*/
cudaError_t gpu_nve_rigid_step_one(const gpu_pdata_arrays& pdata, 
                                   const gpu_rigid_data_arrays& rigid_data,
                                   float4 *d_pdata_orientation,
                                   unsigned int *d_group_members,
                                   unsigned int group_size,
                                   float4 *d_net_force,
                                   const gpu_boxsize &box, 
                                   float deltaT)
    {
    assert(d_net_force);
    assert(d_group_members);
    assert(rigid_data.com);
    assert(rigid_data.vel);
    assert(rigid_data.angmom);
    assert(rigid_data.angvel);
    assert(rigid_data.orientation);
    assert(rigid_data.ex_space);
    assert(rigid_data.ey_space);
    assert(rigid_data.ez_space);
    assert(rigid_data.body_image);
    assert(rigid_data.conjqm);
    assert(rigid_data.body_mass);
    assert(rigid_data.moment_inertia);
    assert(rigid_data.force);
    assert(rigid_data.torque);
    assert(rigid_data.body_indices);
//     
    unsigned int n_bodies = rigid_data.n_bodies;
    unsigned int n_group_bodies = rigid_data.n_group_bodies;
    
    // setup the grid to run the kernel for rigid bodies
    int block_size = 64;
    int n_blocks = n_group_bodies / block_size + 1;
    dim3 body_grid(n_blocks, 1, 1);
    dim3 body_threads(block_size, 1, 1);
    
    gpu_nve_rigid_step_one_body_kernel<<< body_grid, body_threads >>>(rigid_data.com, 
                                                           rigid_data.vel, 
                                                           rigid_data.angmom, 
                                                           rigid_data.angvel,
                                                           rigid_data.orientation, 
                                                           rigid_data.ex_space, 
                                                           rigid_data.ey_space, 
                                                           rigid_data.ez_space, 
                                                           rigid_data.body_image, 
                                                           rigid_data.conjqm,
                                                           rigid_data.body_mass,
                                                           rigid_data.moment_inertia,
                                                           rigid_data.force,
                                                           rigid_data.torque,
                                                           rigid_data.body_indices,
                                                           n_group_bodies, 
                                                           n_bodies, 
                                                           box,
                                                           deltaT);
    

    return cudaSuccess;
    }

#pragma mark RIGID_FORCE_KERNEL
//! Shared memory for body force and torque reduction, required allocation when the kernel is called
extern __shared__ float3 sum[];

//! Calculates the body forces and torques by summing the constituent particle forces using a fixed sliding window size
/*! \param rdata_force Body forces
    \param rdata_torque Body torques
    \param d_rigid_group Body indices
    \param d_rigid_orientation Body orientation
    \param d_particle_orientation Particle orientation (quaternion)
    \param d_rigid_particle_idx Particle index of a local particle in the body
    \param d_rigid_particle_dis Position of a particle in the body frame
    \param d_net_force Particle net forces
    \param d_net_torque Particle net torques
    \param n_group_bodies Number of rigid bodies in my group
    \param n_bodies Total number of rigid bodies
    \param nmax Maximum number of particles in a rigid body
    \param window_size Window size for reduction
    \param thread_mask Block size minus 1, used for idenifying the first thread in the block
    \param n_bodies_per_block Number of bodies per block
    \param box Box dimensions for periodic boundary condition handling
    
    Compute the force and torque sum on all bodies in the system from their constituent particles. n_bodies_per_block
    bodies are handled within each block of execution on the GPU. The reason for this is to decrease
    over-parallelism and use the GPU cores more effectively when bodies are smaller than the block size. Otherwise,
    small bodies leave many threads in the block idle with nothing to do.
    
    On start, the properties common to each body are read in, computed, and stored in shared memory for all the threads
    working on that body to access. Then, the threads loop over all particles that are part of the body with
    a sliding window. Each loop of the window computes the force and torque for block_size/n_bodies_per_block particles
    in as many threads in parallel. These quantities are summed over enough windows to cover the whole body.
    
    The block_size/n_bodies_per_block partial sums are stored in shared memory. Then n_bodies_per_block partial
    reductions are performed in parallel using all threads to sum the total force and torque on each body. This looks
    just like a normal reduction, except that it terminates at a certain level in the tree. To make the math
    for the partial reduction work out, block_size must be a power of 2 as must n_bodies_per_block.
    
    Performance testing on GF100 with many different bodies of different sizes ranging from 4-256 particles per body
    has found that the optimum block size for most bodies is 64 threads. Performance increases for all body sizes
    as n_bodies_per_block is increased, but only up to 8. n_bodies_per_block=16 slows performance significantly.
    Based on these performance results, this kernel is hardcoded to handle only 1,2,4,8 n_bodies_per_block
    with a power of 2 block size (hardcoded to 64 in the kernel launch).
    
    However, there is one issue to the n_bodies_per_block parallelism reduction. If the reduction results in too few
    blocks, performance can actually be reduced. For example, if there are only 64 bodies running at the "most optimal"
    n_bodies_per_block=8 results in only 8 blocks on the GPU! That isn't even enough to heat up all 15 SMs on GF100.
    Even though n_bodies_per_block=1 is not fully optimal per block, running 64 slow blocks in parallel is faster than
    running 8 fast blocks in parallel. Testing on GF100 determines that 60 blocks is the dividing line (makes sense - 
    that's 4 blocks active on each SM).
*/
extern "C" __global__ void gpu_rigid_force_sliding_kernel(float4* rdata_force, 
                                                 float4* rdata_torque,
                                                 unsigned int *d_rigid_group,
                                                 float4* d_rigid_orientation,
                                                 float4* d_particle_orientation,
                                                 unsigned int* d_rigid_particle_idx,
                                                 float4* d_rigid_particle_dis,
                                                 float4* d_net_force,
                                                 float4* d_net_torque,
                                                 unsigned int n_group_bodies, 
                                                 unsigned int n_bodies, 
                                                 unsigned int nmax,
                                                 unsigned int window_size,
                                                 unsigned int thread_mask,
                                                 unsigned int n_bodies_per_block,
                                                 gpu_boxsize box)
    {
    // determine which body (0 ... n_bodies_per_block-1) this thread is working on
    // assign threads 0, 1, 2, ... to body 0, n, n+1, n+2, ... to body 1, and so on.
    unsigned int m = threadIdx.x / (blockDim.x / n_bodies_per_block);
    
    // body_force and body_torque are each shared memory arrays with 1 element per threads
    float3 *body_force = sum;
    float3 *body_torque = &sum[blockDim.x];
    
    // store ex_space, ey_space, ez_space, and the body index in shared memory. Up to 8 bodies per block can
    // be handled.
    __shared__ float4 ex_space[8], ey_space[8], ez_space[8];
    __shared__ int idx_body[8];

    // each thread makes partial sums of force and torque of all the particles that this thread loops over
    float3 sum_force = make_float3(0.0f, 0.0f, 0.0f);
    float3 sum_torque = make_float3(0.0f, 0.0f, 0.0f);
    
    // thread_mask is a bitmask that masks out the high bits in threadIdx.x.
    // threadIdx.x & thread_mask is an index from 0 to block_size/n_bodies_per_block-1 and determines what offset
    // this thread is to use when accessing the particles in the body
    if ((threadIdx.x & thread_mask) == 0)
        {
        // thread 0 for this body reads in the body id and orientation and stores them in shared memory
        int group_idx = blockIdx.x*n_bodies_per_block + m;
        if (group_idx < n_group_bodies)
            {
            idx_body[m] = d_rigid_group[group_idx];
            float4 orientation = d_rigid_orientation[idx_body[m]];
            exyzFromQuaternion(orientation, ex_space[m], ey_space[m], ez_space[m]);
            }
        else
            {
            idx_body[m] =-1;
            }
        }
    
    __syncthreads();
    
    // compute the number of windows that we need to loop over
    unsigned int n_windows = nmax / window_size + 1;
        
    // slide the window throughout the block
    for (unsigned int start = 0; start < n_windows; start++)
        {
        float4 fi = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 ti = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        
        // determine the index with this body that this particle should handle
        unsigned int k = start * window_size + (threadIdx.x & thread_mask);
        
        // if that index is in the body we are actually handling a real body
        if (k < nmax && idx_body[m] != -1)
            {
            // determine the particle idx of the particle
            int localidx = idx_body[m] * nmax + k;
            unsigned int pidx = d_rigid_particle_idx[localidx];
            
            // if this particle is actually in the body
            if (pidx != INVALID_INDEX)
                {
                // calculate body force and torques
                float4 particle_pos = d_rigid_particle_dis[localidx];
                fi = d_net_force[pidx];

                //will likely need to rotate these components too
                ti = d_net_torque[pidx];

                // tally the force in the per thread counter
                sum_force.x += fi.x;
                sum_force.y += fi.y;
                sum_force.z += fi.z;
                
                // This might require more calculations but more stable
                // particularly when rigid bodies are bigger than half the box
                float3 ri;
                ri.x = ex_space[m].x * particle_pos.x + ey_space[m].x * particle_pos.y 
                        + ez_space[m].x * particle_pos.z;
                ri.y = ex_space[m].y * particle_pos.x + ey_space[m].y * particle_pos.y 
                        + ez_space[m].y * particle_pos.z;
                ri.z = ex_space[m].z * particle_pos.x + ey_space[m].z * particle_pos.y 
                        + ez_space[m].z * particle_pos.z;

                //need to update here     
                // tally the torque in the per thread counter
                sum_torque.x += ri.y * fi.z - ri.z * fi.y + ti.x;
                sum_torque.y += ri.z * fi.x - ri.x * fi.z + ti.y;
                sum_torque.z += ri.x * fi.y - ri.y * fi.x + ti.z;
                }
            }
        }

    __syncthreads();
    
    // put the partial sums into shared memory
    body_force[threadIdx.x] = sum_force;
    body_torque[threadIdx.x] = sum_torque;
   
    // perform a set of partial reductions. Each block_size/n_bodies_per_block threads performs a sum reduction
    // just within its own group
    unsigned int offset = min(window_size, nmax) >> 1;
    while (offset > 0)
        {
        if ((threadIdx.x & thread_mask) < offset)
            {
            body_force[threadIdx.x].x += body_force[threadIdx.x + offset].x;
            body_force[threadIdx.x].y += body_force[threadIdx.x + offset].y;
            body_force[threadIdx.x].z += body_force[threadIdx.x + offset].z;
            
            body_torque[threadIdx.x].x += body_torque[threadIdx.x + offset].x;
            body_torque[threadIdx.x].y += body_torque[threadIdx.x + offset].y;
            body_torque[threadIdx.x].z += body_torque[threadIdx.x + offset].z;
            }
            
        offset >>= 1;
        
        __syncthreads();
        }
    
    // thread 0 within this body writes out the total force and torque for the body
    if ((threadIdx.x & thread_mask) == 0 && idx_body[m] != -1)
        {
        rdata_force[idx_body[m]] = make_float4(body_force[threadIdx.x].x, body_force[threadIdx.x].y, body_force[threadIdx.x].z, 0.0f);
        rdata_torque[idx_body[m]] = make_float4(body_torque[threadIdx.x].x, body_torque[threadIdx.x].y, body_torque[threadIdx.x].z, 0.0f);
        }
    }


/*! \param pdata Particle data to step forward 1/2 step
    \param rigid_data Rigid body data to step forward 1/2 step
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param d_net_force Particle net forces
    \param d_net_torque Particle net torques
    \param box Box dimensions for periodic boundary condition handling
    \param deltaT Amount of real time to step forward in one time step
*/
cudaError_t gpu_rigid_force(const gpu_pdata_arrays &pdata, 
                                   const gpu_rigid_data_arrays& rigid_data,
                                   unsigned int *d_group_members,
                                   unsigned int group_size, 
                                   float4 *d_net_force,
                                   float4 *d_net_torque,
                                   const gpu_boxsize &box,
                                   float deltaT)
    {
    unsigned int n_bodies = rigid_data.n_bodies;
    unsigned int n_group_bodies = rigid_data.n_group_bodies;
    unsigned int nmax = rigid_data.nmax;
    
    unsigned int n_bodies_per_block;
    unsigned int target_num_blocks = 60;
    if (n_group_bodies / 8 >= target_num_blocks)
        n_bodies_per_block = 8;
    else
    if (n_group_bodies / 4 >= target_num_blocks)
        n_bodies_per_block = 4;
    else
    if (n_group_bodies / 2 >= target_num_blocks)
        n_bodies_per_block = 2;
    else
        n_bodies_per_block = 1;

    unsigned int block_size = 64;
    unsigned int window_size = block_size / n_bodies_per_block;
    unsigned int thread_mask = window_size - 1;
    
    dim3 force_grid(n_group_bodies / n_bodies_per_block + 1, 1, 1);
    dim3 force_threads(block_size, 1, 1);

    gpu_rigid_force_sliding_kernel<<< force_grid, force_threads, 2 * block_size * sizeof(float3) >>>(rigid_data.force, 
                                                                                            rigid_data.torque,
                                                                                            rigid_data.body_indices,
                                                                                            rigid_data.orientation,
                                                                                            rigid_data.particle_orientation,
                                                                                            rigid_data.particle_indices,
                                                                                            rigid_data.particle_pos,
                                                                                            d_net_force,
                                                                                            d_net_torque,
                                                                                            n_group_bodies,
                                                                                            n_bodies,
                                                                                            nmax,
                                                                                            window_size,
                                                                                            thread_mask,
                                                                                            n_bodies_per_block,
                                                                                            box);

                                                
                                                 
                                                 

    return cudaSuccess;
    }

#pragma mark RIGID_STEP_TWO_KERNEL
/*! Takes the second half-step forward for rigid bodies in the velocity-verlet NVE integration
    \param rdata_vel Body translational velocity
    \param rdata_angmom Angular momentum
    \param rdata_angvel Angular velocity
    \param rdata_orientation Quaternion
    \param rdata_conjqm Conjugate quaternion momentum
    \param d_rigid_mass Body mass
    \param d_rigid_mi Body inertia moments
    \param d_rigid_force Body forces
    \param d_rigid_torque Body torques
    \param d_rigid_group Body indices
    \param n_group_bodies Number of rigid bodies in my group
    \param n_bodies Total number of rigid bodies
    \param deltaT Timestep 
    \param box Box dimensions for periodic boundary condition handling
*/
extern "C" __global__ void gpu_nve_rigid_step_two_body_kernel(float4* rdata_vel, 
                                                         float4* rdata_angmom, 
                                                         float4* rdata_angvel,
                                                         float4* rdata_orientation,
                                                         float4* rdata_conjqm,
                                                         float *d_rigid_mass,
                                                         float4 *d_rigid_mi,
                                                         float4 *d_rigid_force,
                                                         float4 *d_rigid_torque,
                                                         unsigned int *d_rigid_group,
                                                         unsigned int n_group_bodies,
                                                         unsigned int n_bodies, 
                                                         gpu_boxsize box, 
                                                         float deltaT)
    {
    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (group_idx >= n_group_bodies)
        return;
    
    float body_mass;
    float4 moment_inertia, vel, angmom, orientation, ex_space, ey_space, ez_space, force, torque;
    float dt_half = 0.5f * deltaT;
    
    unsigned int idx_body = d_rigid_group[group_idx];
    
    // Update body velocity and angmom
    // update the velocity
    body_mass = d_rigid_mass[idx_body];
    vel = rdata_vel[idx_body];
    angmom = rdata_angmom[idx_body];
    force = d_rigid_force[idx_body];
    torque = d_rigid_torque[idx_body];
    moment_inertia = d_rigid_mi[idx_body];
    orientation = rdata_orientation[idx_body];
    
    exyzFromQuaternion(orientation, ex_space, ey_space, ez_space);
        
    float dtfm = dt_half / body_mass;
    float4 vel2;
    vel2.x = vel.x + dtfm * force.x;
    vel2.y = vel.y + dtfm * force.y;
    vel2.z = vel.z + dtfm * force.z;
    vel2.w = 0.0f;

    // update angular momentum
    float4 angmom2;
    angmom2.x = angmom.x + dt_half * torque.x;
    angmom2.y = angmom.y + dt_half * torque.y;
    angmom2.z = angmom.z + dt_half * torque.z;
    angmom2.w = 0.0f;
    
    // update angular velocity        
    float4 angvel2;
    computeAngularVelocity(angmom2, moment_inertia, ex_space, ey_space, ez_space, angvel2);
    
    // write out results
    rdata_vel[idx_body] = vel2;
    rdata_angmom[idx_body] = angmom2;
    rdata_angvel[idx_body] = angvel2;
    }

// Take the second 1/2 step forward in the NVE integration step
/*! \param pdata Particle data to step forward 1/2 step
    \param rigid_data Rigid body data to step forward 1/2 step
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param d_net_force Particle net forces
    \param d_net_virial Particle net virial
    \param box Box dimensions for periodic boundary condition handling
    \param deltaT Amount of real time to step forward in one time step
*/
cudaError_t gpu_nve_rigid_step_two(const gpu_pdata_arrays &pdata, 
                                   const gpu_rigid_data_arrays& rigid_data,
                                   float4 *d_pdata_orientation,
                                   unsigned int *d_group_members,
                                   unsigned int group_size,
                                   float4 *d_net_force,
                                   float *d_net_virial,
                                   const gpu_boxsize &box,
                                   float deltaT)
    {
    unsigned int n_bodies = rigid_data.n_bodies;
    unsigned int n_group_bodies = rigid_data.n_group_bodies;
    
    unsigned int block_size = 64;
    unsigned int n_blocks = n_group_bodies / block_size + 1;
    dim3 body_grid(n_blocks, 1, 1);
    dim3 body_threads(block_size, 1, 1);
    gpu_nve_rigid_step_two_body_kernel<<< body_grid, body_threads >>>(rigid_data.vel, 
                                                                      rigid_data.angmom, 
                                                                      rigid_data.angvel,
                                                                      rigid_data.orientation,
                                                                      rigid_data.conjqm,
                                                                      rigid_data.body_mass,
                                                                      rigid_data.moment_inertia,
                                                                      rigid_data.force,
                                                                      rigid_data.torque,
                                                                      rigid_data.body_indices,
                                                                      n_group_bodies,
                                                                      n_bodies, 
                                                                      box, 
                                                                      deltaT);
    
    return cudaSuccess;
    }
