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

// Maintainer: ndtrung

#include "QuaternionMath.h"
#include "TwoStepNPTRigidGPU.cuh"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file TwoStepNPTRigidGPU.cu
    \brief Defines GPU kernel code for NPT integration on the GPU. Used by TwoStepNPTRigidGPU.
*/

// Flag for invalid particle index, identical to the sentinel value NO_INDEX in RigidData.h
#define INVALID_INDEX 0xffffffff

/*! Maclaurine expansion
    \param x Point to take the expansion

*/
__device__ float nph_maclaurin_series(float x)
    {
    float x2, x4;
    x2 = x * x;
    x4 = x2 * x2;
    return (1.0f + (1.0f/6.0f) * x2 + (1.0f/120.0f) * x4 + (1.0f/5040.0f) * x2 * x4 + (1.0f/362880.0f) * x4 * x4);
    }

/*! Kernel to zero virial contribution from particles from rigid bodies
    \param d_virial_rigid Virial contribution from particles in rigid bodies
    \param local_num Number of particles in this card
*/
extern "C" __global__ void gpu_nph_rigid_zero_virial_rigid_kernel(float *d_virial_rigid, 
                                                                 unsigned int local_num)
    {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // particle's index

    if (idx < local_num)
        {
        d_virial_rigid[idx] = 0.0f;
        }

    }

/*! Takes the first half-step forward for rigid bodies in the velocity-verlet NVT integration 
    \param rdata_com Body center of mass
    \param d_rigid_group Body group
    \param n_group_bodies Number of rigid bodies in my group
    \param n_bodies Total umber of rigid bodies
    \param box Box dimensions for periodic boundary condition handling
    \param npt_rdata Thermostat/barostat data
*/

extern "C" __global__ void gpu_nph_rigid_remap_kernel(float4* rdata_com,
                                                      unsigned int *d_rigid_group,
                                                      unsigned int n_group_bodies,
                                                      unsigned int n_bodies, 
                                                      BoxDim box,
                                                      float npt_rdata_dilation,
                                                      unsigned int npt_rdata_dimension,
                                                      float4* npt_rdata_new_box)
    {
    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_idx >= n_group_bodies)
        return;

    unsigned int idx_body = d_rigid_group[group_idx];

    float oldlo, oldhi, ctr;
    float3 pos, delta;

    Scalar3 lo = box.getLo();
    Scalar3 hi = box.getHi();
    Scalar3 L = box.getL();

    float4 com = rdata_com[idx_body];

    delta.x = com.x - lo.x;
    delta.y = com.y - lo.y;
    delta.z = com.z - lo.z;

    pos.x = 1.0f / L.x * delta.x;
    pos.y = 1.0f / L.y * delta.y;
    pos.z = 1.0f / L.z * delta.z;

    // reset box to new size/shape
    oldlo = lo.x;
    oldhi = hi.x;
    ctr = 0.5f * (oldlo + oldhi);
    lo.x = (oldlo - ctr) * npt_rdata_dilation + ctr;
    hi.x = (oldhi - ctr) * npt_rdata_dilation + ctr;
    L.x = hi.x - lo.x;

    oldlo = lo.y;
    oldhi = hi.y;
    ctr = 0.5f * (oldlo + oldhi);
    lo.y = (oldlo - ctr) * npt_rdata_dilation + ctr;
    hi.y = (oldhi - ctr) * npt_rdata_dilation + ctr;
    L.y = hi.y - lo.y;

    if (npt_rdata_dimension == 3)
        {
        oldlo = lo.z;
        oldhi = hi.z;
        ctr = 0.5f * (oldlo + oldhi);
        lo.z = (oldlo - ctr) * npt_rdata_dilation + ctr;
        hi.z = (oldhi - ctr) * npt_rdata_dilation + ctr;
        L.z = hi.z - lo.z;
        }

    // convert rigid body COMs back to box coords
    float3 newboxlo = -L/2.0f;
    pos = L * pos + newboxlo;

    // write out results
    rdata_com[idx_body] = make_float4(pos.x, pos.y, pos.z, 0.0f);

    if (idx_body == 0)
        {
        *npt_rdata_new_box = make_float4(L.x, L.y, L.z, 0.0f);
        }
    }


#pragma mark RIGID_STEP_ONE_KERNEL
/*! Takes the first half-step forward for rigid bodies in the velocity-verlet NVT integration 
    \param rdata_com Body center of mass
    \param rdata_vel Body velocity
    \param rdata_angmom Angular momentum
    \param rdata_angvel Angular velocity
    \param rdata_orientation Quaternion
    \param rdata_body_image Body image 
    \param rdata_conjqm Conjugate quaternion momentum
    \param d_rigid_mass Body mass
    \param d_rigid_mi Body inertia moments
    \param n_group_bodies Number of rigid bodies in my group
    \param d_rigid_force Body forces
    \param d_rigid_torque Body torques
    \param d_rigid_group Body indices
    \param n_bodies Total umber of rigid bodies
    \param npt_rdata_epsilon_dot Barostat velocity
    \param npt_rdata_partial_Ksum_t Body translational kinetic energy 
    \param npt_rdata_partial_Ksum_r Body rotation kinetic energy
    \param npt_rdata_nf_t Translational degrees of freedom
    \param npt_rdata_nf_r Translational degrees of freedom
    \param npt_rdata_dimension System dimesion
    \param box Box dimensions for periodic boundary condition handling
    \param deltaT Timestep 
    
*/

extern "C" __global__ void gpu_nph_rigid_step_one_body_kernel(float4* rdata_com, 
                                                            float4* rdata_vel, 
                                                            float4* rdata_angmom, 
                                                            float4* rdata_angvel,
                                                            float4* rdata_orientation, 
                                                            int3* rdata_body_image, 
                                                            float4* rdata_conjqm,
                                                            float *d_rigid_mass,
                                                            float4 *d_rigid_mi,
                                                            float4 *d_rigid_force,
                                                            float4 *d_rigid_torque,
                                                            unsigned int *d_rigid_group, 
                                                            unsigned int n_group_bodies,  
                                                            unsigned int n_bodies, 
                                                            float npt_rdata_epsilon_dot, 
                                                            float* npt_rdata_partial_Ksum_t, 
                                                            float* npt_rdata_partial_Ksum_r,
                                                            unsigned int npt_rdata_nf_t,
                                                            unsigned int npt_rdata_nf_r,
                                                            unsigned int npt_rdata_dimension, 
                                                            BoxDim box, 
                                                            float deltaT)
    {
    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (group_idx >= n_group_bodies)
        return;

    // do velocity verlet update
    // v(t+deltaT/2) = v(t) + (1/2)a*deltaT
    // r(t+deltaT) = r(t) + v(t+deltaT/2)*deltaT
    float body_mass;
    float4 moment_inertia, com, vel, orientation, ex_space, ey_space, ez_space, force, torque, conjqm;
    int3 body_image;
    float4 mbody, tbody, fquat;

    float dt_half = 0.5f * deltaT;
    float onednft, onednfr, tmp, scale_t, scale_r, scale_v, akin_t, akin_r;

    onednft = 1.0f + (float) (npt_rdata_dimension) / (float) (npt_rdata_nf_t+npt_rdata_nf_r);
    onednfr = (float) (npt_rdata_dimension) / (float) (npt_rdata_nf_t+npt_rdata_nf_r);

    tmp = -1.0f * dt_half * onednft * npt_rdata_epsilon_dot;
    scale_t = __expf(tmp);
    tmp = -1.0f * dt_half * onednfr * npt_rdata_epsilon_dot;
    scale_r = __expf(tmp);
    tmp = dt_half * npt_rdata_epsilon_dot;
    scale_v = deltaT * __expf(tmp) * nph_maclaurin_series(tmp);

    unsigned int idx_body = d_rigid_group[group_idx];
    body_mass = d_rigid_mass[idx_body];
    moment_inertia = d_rigid_mi[idx_body];
    com = rdata_com[idx_body];
    vel = rdata_vel[idx_body];
    orientation = rdata_orientation[idx_body];
    body_image = rdata_body_image[idx_body];
    force = d_rigid_force[idx_body];
    torque = d_rigid_torque[idx_body];
    conjqm = rdata_conjqm[idx_body];

    exyzFromQuaternion(orientation, ex_space, ey_space, ez_space);

    // update velocity
    float dtfm = dt_half / body_mass;

    float4 vel2;
    vel2.x = vel.x + dtfm * force.x;
    vel2.y = vel.y + dtfm * force.y;
    vel2.z = vel.z + dtfm * force.z;
    vel2.x *= scale_t;
    vel2.y *= scale_t;
    vel2.z *= scale_t;
    vel2.w = vel.w;

    tmp = vel2.x * vel2.x + vel2.y * vel2.y + vel2.z * vel2.z;
    akin_t = body_mass * tmp;

    // update position
    float3 pos2;
    pos2.x = com.x + vel2.x * scale_v;
    pos2.y = com.y + vel2.y * scale_v;
    pos2.z = com.z + vel2.z * scale_v;

    // time to fix the periodic boundary conditions
    box.wrap(pos2, body_image);

    matrix_dot(ex_space, ey_space, ez_space, torque, tbody);
    quatvec(orientation, tbody, fquat);

    float4 conjqm2;
    conjqm2.x = conjqm.x + deltaT * fquat.x;
    conjqm2.y = conjqm.y + deltaT * fquat.y;
    conjqm2.z = conjqm.z + deltaT * fquat.z;
    conjqm2.w = conjqm.w + deltaT * fquat.w;

    conjqm2.x *= scale_r;
    conjqm2.y *= scale_r;
    conjqm2.z *= scale_r;
    conjqm2.w *= scale_r;

    // use no_squish rotate to update p and q
    no_squish_rotate(3, conjqm2, orientation, moment_inertia, dt_half);
    no_squish_rotate(2, conjqm2, orientation, moment_inertia, dt_half);
    no_squish_rotate(1, conjqm2, orientation, moment_inertia, deltaT);
    no_squish_rotate(2, conjqm2, orientation, moment_inertia, dt_half);
    no_squish_rotate(3, conjqm2, orientation, moment_inertia, dt_half);

    // update the exyz_space
    // transform p back to angmom
    // update angular velocity
    float4 angmom2;
    exyzFromQuaternion(orientation, ex_space, ey_space, ez_space);
    invquatvec(orientation, conjqm2, mbody);
    transpose_dot(ex_space, ey_space, ez_space, mbody, angmom2);

    angmom2.x *= 0.5f;
    angmom2.y *= 0.5f;
    angmom2.z *= 0.5f;

    float4 angvel2;
    computeAngularVelocity(angmom2, moment_inertia, ex_space, ey_space, ez_space, angvel2);

    akin_r = angmom2.x * angvel2.x + angmom2.y * angvel2.y + angmom2.z * angvel2.z;

    // write out the results (MEM_TRANSFER: ? bytes)
    rdata_com[idx_body] = make_float4(pos2.x, pos2.y, pos2.z, com.w);
    rdata_vel[idx_body] = vel2;
    rdata_angmom[idx_body] = angmom2;
    rdata_angvel[idx_body] = angvel2;
    rdata_orientation[idx_body] = orientation;
    rdata_body_image[idx_body] = body_image;
    rdata_conjqm[idx_body] = conjqm2;

    npt_rdata_partial_Ksum_t[group_idx] = akin_t;
    npt_rdata_partial_Ksum_r[group_idx] = akin_r;
    }

/*! \param rigid_data Rigid body data to step forward 1/2 step
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param d_net_force Particle net forces
    \param box Box dimensions for periodic boundary condition handling
    \param npt_rdata Thermostat/barostat data
    \param deltaT Amount of real time to step forward in one time step
    
*/
cudaError_t gpu_nph_rigid_step_one(const gpu_rigid_data_arrays& rigid_data,
                                   unsigned int *d_group_members,
                                   unsigned int group_size,
                                   float4 *d_net_force,
                                   const BoxDim& box, 
                                   const gpu_npt_rigid_data& npt_rdata,
                                   float deltaT)
    {
    unsigned int n_bodies = rigid_data.n_bodies;
    unsigned int n_group_bodies = rigid_data.n_group_bodies;
    
    // setup the grid to run the kernel for rigid bodies
    int block_size = 64;
    int n_blocks = n_group_bodies / block_size + 1;
    dim3 body_grid(n_blocks, 1, 1);
    dim3 body_threads(block_size, 1, 1);
    gpu_nph_rigid_step_one_body_kernel<<< body_grid, body_threads  >>>(rigid_data.com, 
                                                                       rigid_data.vel, 
                                                                       rigid_data.angmom, 
                                                                       rigid_data.angvel,
                                                                       rigid_data.orientation, 
                                                                       rigid_data.body_image, 
                                                                       rigid_data.conjqm,
                                                                       rigid_data.body_mass,
                                                                       rigid_data.moment_inertia,
                                                                       rigid_data.force,
                                                                       rigid_data.torque,
                                                                       rigid_data.body_indices,
                                                                       n_group_bodies,
                                                                       n_bodies, 
                                                                       npt_rdata.epsilon_dot, 
                                                                       npt_rdata.partial_Ksum_t,
                                                                       npt_rdata.partial_Ksum_r,
                                                                       npt_rdata.nf_t,
                                                                       npt_rdata.nf_r,
                                                                       npt_rdata.dimension,
                                                                       box, 
                                                                       deltaT);
       
    gpu_nph_rigid_remap_kernel<<< body_grid, body_threads >>>(rigid_data.com,
                                                              rigid_data.body_indices,
                                                              n_group_bodies,
                                                              n_bodies,
                                                              box, 
                                                              npt_rdata.dilation,
                                                              npt_rdata.dimension,
                                                              npt_rdata.new_box);

                                                                    
    return cudaSuccess;
    }

#pragma mark RIGID_STEP_TWO_KERNEL
//! Takes the 2nd 1/2 step forward in the velocity-verlet NPH integration scheme
/*!  
    \param rdata_vel Body velocity
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
    \param npt_rdata_eta_dot_t0 Thermostat translational part 
    \param npt_rdata_eta_dot_r0 Thermostat rotational part
    \param npt_rdata_epsilon_dot Barostat velocity
    \param npt_rdata_partial_Ksum_t Body translational kinetic energy 
    \param npt_rdata_partial_Ksum_r Body rotation kinetic energy
    \param npt_rdata_nf_t Translational degrees of freedom
    \param npt_rdata_nf_r Translational degrees of freedom
    \param npt_rdata_dimension System dimesion
    \param deltaT Timestep 
    \param box Box dimensions for periodic boundary condition handling
*/

extern "C" __global__ void gpu_nph_rigid_step_two_body_kernel(float4* rdata_vel, 
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
                                                              float npt_rdata_epsilon_dot, 
                                                              float* npt_rdata_partial_Ksum_t,
                                                              float* npt_rdata_partial_Ksum_r,
                                                              unsigned int npt_rdata_nf_t,
                                                              unsigned int npt_rdata_nf_r,
                                                              unsigned int npt_rdata_dimension,
                                                              BoxDim box, 
                                                              float deltaT)
    {
    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (group_idx >= n_group_bodies)
        return;
        
    float body_mass;
    float4 moment_inertia, vel, ex_space, ey_space, ez_space, orientation, conjqm;
    float4 force, torque;
    float4 mbody, tbody, fquat;
    
    float dt_half = 0.5f * deltaT;
    float onednft, onednfr, tmp, scale_t, scale_r, akin_t, akin_r;
    
    onednft = 1.0f + (float) (npt_rdata_dimension) / (float) (npt_rdata_nf_t+npt_rdata_nf_r);
    onednfr = (float) (npt_rdata_dimension) / (float) (npt_rdata_nf_t+npt_rdata_nf_r);

    tmp = -1.0f * dt_half * onednft * npt_rdata_epsilon_dot;
    scale_t = exp(tmp);
    tmp = -1.0f * dt_half * onednfr * npt_rdata_epsilon_dot;
    scale_r = exp(tmp);
    
    unsigned int idx_body = d_rigid_group[group_idx];

    // Update body velocity and angmom
    body_mass = d_rigid_mass[idx_body];
    moment_inertia = d_rigid_mi[idx_body];
    vel = rdata_vel[idx_body];
    force = d_rigid_force[idx_body];
    torque = d_rigid_torque[idx_body];
    orientation = rdata_orientation[idx_body];
    conjqm = rdata_conjqm[idx_body];
    
    exyzFromQuaternion(orientation, ex_space, ey_space, ez_space);
    
    float dtfm = dt_half / body_mass;
    
    // update the velocity
    float4 vel2;
    vel2.x = scale_t * vel.x + dtfm * force.x;
    vel2.y = scale_t * vel.y + dtfm * force.y;
    vel2.z = scale_t * vel.z + dtfm * force.z;
    vel2.w = 0.0f;
    
    tmp = vel2.x * vel2.x + vel2.y * vel2.y + vel2.z * vel2.z;
    akin_t = body_mass * tmp;
    
    // update angular momentum
    matrix_dot(ex_space, ey_space, ez_space, torque, tbody);
    quatvec(orientation, tbody, fquat);
    
    float4  conjqm2, angmom2;
    conjqm2.x = scale_r * conjqm.x + deltaT * fquat.x;
    conjqm2.y = scale_r * conjqm.y + deltaT * fquat.y;
    conjqm2.z = scale_r * conjqm.z + deltaT * fquat.z;
    conjqm2.w = scale_r * conjqm.w + deltaT * fquat.w;
    
    invquatvec(orientation, conjqm2, mbody);
    transpose_dot(ex_space, ey_space, ez_space, mbody, angmom2);
    
    angmom2.x *= 0.5f;
    angmom2.y *= 0.5f;
    angmom2.z *= 0.5f;
    angmom2.w = 0.0f;
    
    // update angular velocity
    float4 angvel2;
    computeAngularVelocity(angmom2, moment_inertia, ex_space, ey_space, ez_space, angvel2);
    
    akin_r = angmom2.x * angvel2.x + angmom2.y * angvel2.y + angmom2.z * angvel2.z;
    
    rdata_vel[idx_body] = vel2;
    rdata_angmom[idx_body] = angmom2;
    rdata_angvel[idx_body] = angvel2;
    rdata_conjqm[idx_body] = conjqm2;
    
    npt_rdata_partial_Ksum_t[group_idx] = akin_t;
    npt_rdata_partial_Ksum_r[group_idx] = akin_r;
    }

/*! \param rigid_data Rigid body data to step forward 1/2 step
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param d_net_force Particle net forces
    \param d_net_virial Particle net virial
    \param box Box dimensions for periodic boundary condition handling
    \param npt_rdata Thermostat/barostat data
    \param deltaT Amount of real time to step forward in one time step
    
*/
cudaError_t gpu_nph_rigid_step_two( const gpu_rigid_data_arrays& rigid_data,
                                    unsigned int *d_group_members,
                                    unsigned int group_size,
                                    float4 *d_net_force,
                                    float *d_net_virial,
                                    const BoxDim& box, 
                                    const gpu_npt_rigid_data& npt_rdata,
                                    float deltaT)
    {
    unsigned int n_bodies = rigid_data.n_bodies;
    unsigned int n_group_bodies = rigid_data.n_group_bodies;
                                                                                                                                                            
    unsigned int block_size = 64;
    unsigned int n_blocks = n_group_bodies / block_size + 1;                                
    dim3 body_grid(n_blocks, 1, 1);
    dim3 body_threads(block_size, 1, 1);                                                 
    gpu_nph_rigid_step_two_body_kernel<<< body_grid, body_threads >>>(rigid_data.vel, 
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
                                                                    npt_rdata.epsilon_dot, 
                                                                    npt_rdata.partial_Ksum_t,
                                                                    npt_rdata.partial_Ksum_r,
                                                                    npt_rdata.nf_t,
                                                                    npt_rdata.nf_r,
                                                                    npt_rdata.dimension, 
                                                                    box, 
                                                                    deltaT);

                                                                                                                                             
    return cudaSuccess;
    }

#pragma mark RIGID_KINETIC_ENERGY_REDUCTION

//! Shared memory for kinetic energy reduction
extern __shared__ float nph_rigid_sdata[];

/*! Summing the kinetic energy of rigid bodies
    \param npt_rdata Thermostat data for rigid bodies 
    
*/
extern "C" __global__ void gpu_nph_rigid_reduce_ksum_kernel(float* npt_rdata_partial_Ksum_t,
                                                            float* npt_rdata_partial_Ksum_r,
                                                            float* npt_rdata_Ksum_t,
                                                            float* npt_rdata_Ksum_r,
                                                            unsigned int n_bodies)
    {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float* body_ke_t = nph_rigid_sdata;
    float* body_ke_r = &nph_rigid_sdata[blockDim.x];
    
    float Ksum_t = 0.0f, Ksum_r=0.0f;
    
    // sum up the values in the partial sum via a sliding window
    for (int start = 0; start < n_bodies; start += blockDim.x)
        {
        if (start + threadIdx.x < n_bodies)
            {
            body_ke_t[threadIdx.x] = npt_rdata_partial_Ksum_t[start + threadIdx.x];
            body_ke_r[threadIdx.x] = npt_rdata_partial_Ksum_r[start + threadIdx.x];
            }
        else
            {
            body_ke_t[threadIdx.x] = 0.0f;
            body_ke_r[threadIdx.x] = 0.0f;
            }
        __syncthreads();
        
        // reduce the sum within a block
        int offset = blockDim.x >> 1;
        while (offset > 0)
            {
            if (threadIdx.x < offset)
                {
                body_ke_t[threadIdx.x] += body_ke_t[threadIdx.x + offset];
                body_ke_r[threadIdx.x] += body_ke_r[threadIdx.x + offset];
                }
            offset >>= 1;
            __syncthreads();
            }
            
        // everybody sums up Ksum
        Ksum_t += body_ke_t[0];
        Ksum_r += body_ke_r[0];
        }
        
    __syncthreads();
    
    
    if (global_idx == 0)
        {
        *npt_rdata_Ksum_t = Ksum_t;
        *npt_rdata_Ksum_r = Ksum_r;
        }
        
    }

/*! 
    \param npt_rdata Thermostat/barostat data for rigid bodies 
*/
cudaError_t gpu_nph_rigid_reduce_ksum(const gpu_npt_rigid_data& npt_rdata)
    {
    // setup the grid to run the kernel
    int block_size = 128;
    dim3 grid( 1, 1, 1);
    dim3 threads(block_size, 1, 1);
    
    // run the kernel: double the block size to accomodate Ksum_t and Ksum_r
    gpu_nph_rigid_reduce_ksum_kernel<<< grid, threads, 2 * block_size * sizeof(float) >>>(npt_rdata.partial_Ksum_t,
                                                                                          npt_rdata.partial_Ksum_r,
                                                                                          npt_rdata.Ksum_t,
                                                                                          npt_rdata.Ksum_r,
                                                                                          npt_rdata.n_bodies);
    
    return cudaSuccess;
    }

