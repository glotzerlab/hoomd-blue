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

// $Id: TwoStepNPTRigidGPU.cu 2763 2010-02-15 19:05:06Z ndtrung $
// $URL: http://codeblue.umich.edu/hoomd-blue/svn/branches/rigid-bodies/libhoomd/cuda/TwoStepNPTRigidGPU.cu $
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

//! Flag for invalid particle index, identical to the sentinel value NO_INDEX in RigidData.h
#define INVALID_INDEX 0xffffffff

/*! Maclaurine expansion
    \param x Point to take the expansion

*/
__device__ float maclaurin_series(float x)
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
extern "C" __global__ void gpu_npt_rigid_zero_virial_rigid_kernel(float *d_virial_rigid, 
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

extern "C" __global__ void gpu_npt_rigid_remap_kernel(float4* rdata_com,
                                                        unsigned int *d_rigid_group,
                                                        unsigned int n_group_bodies,
                                                        unsigned int n_bodies, 
                                                        gpu_boxsize box,
                                                        gpu_npt_rigid_data npt_rdata)
    {
    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_idx >= n_group_bodies)
        return;
        
    unsigned int idx_body = d_rigid_group[group_idx];
        
    float oldlo, oldhi, ctr;
    float xlo, xhi, ylo, yhi, zlo, zhi, Lx, Ly, Lz;
    float4 pos, delta;
    float dilation = npt_rdata.dilation;
    
    xlo = -box.Lx/2.0f;
    xhi = box.Lx/2.0f;
    ylo = -box.Ly/2.0f;
    yhi = box.Ly/2.0f;
    zlo = -box.Lz/2.0f;
    zhi = box.Lz/2.0f;
    Lz = zhi - zlo;
    
    float4 com = rdata_com[idx_body];
    
    delta.x = com.x - xlo;
    delta.y = com.y - ylo;
    delta.z = com.z - zlo;

    pos.x = box.Lxinv * delta.x;
    pos.y = box.Lyinv * delta.y;
    pos.z = box.Lzinv * delta.z;
    
    // reset box to new size/shape
    oldlo = xlo;
    oldhi = xhi;
    ctr = 0.5f * (oldlo + oldhi);
    xlo = (oldlo - ctr) * dilation + ctr;
    xhi = (oldhi - ctr) * dilation + ctr;
    Lx = xhi - xlo;
    
    oldlo = ylo;
    oldhi = yhi;
    ctr = 0.5f * (oldlo + oldhi);
    ylo = (oldlo - ctr) * dilation + ctr;
    yhi = (oldhi - ctr) * dilation + ctr;
    Ly = yhi - ylo;
    
    if (npt_rdata.dimension == 3)
        {
        oldlo = zlo;
        oldhi = zhi;
        ctr = 0.5f * (oldlo + oldhi);
        zlo = (oldlo - ctr) * dilation + ctr;
        zhi = (oldhi - ctr) * dilation + ctr;
        Lz = zhi - zlo;
        }
    
    // convert rigid body COMs back to box coords
    float4 newboxlo;
    newboxlo.x = -Lx/2.0f;
    newboxlo.y = -Ly/2.0f;
    newboxlo.z = -Lz/2.0f;
    
    pos.x = Lx * pos.x + newboxlo.x;
    pos.y = Ly * pos.y + newboxlo.y;
    pos.z = Lz * pos.z + newboxlo.z;
    
    // write out results
    rdata_com[idx_body].x = pos.x;
    rdata_com[idx_body].y = pos.y;
    rdata_com[idx_body].z = pos.z;
    
    if (idx_body == 0)
        {
        *(npt_rdata.new_box) = make_float4(Lx, Ly, Lz, 0.0f);
        }
    }

    
#pragma mark RIGID_STEP_ONE_KERNEL
/*! Takes the first half-step forward for rigid bodies in the velocity-verlet NVT integration 
    \param rdata_com Body center of mass
    \param rdata_vel Body velocity
    \param rdata_angmom Angular momentum
    \param rdata_angvel Angular velocity
    \param rdata_orientation Quaternion
    \param rdata_ex_space x-axis unit vector
    \param rdata_ey_space y-axis unit vector
    \param rdata_ez_space z-axis unit vector
    \param rdata_body_imagex Body image in x-direction
    \param rdata_body_imagey Body image in y-direction
    \param rdata_body_imagez Body image in z-direction
    \param rdata_conjqm Conjugate quaternion momentum
    \param d_rigid_mass Body mass
    \param d_rigid_mi Body inertia moments
    \param n_group_bodies Number of rigid bodies in my group
    \param d_rigid_force Body forces
    \param d_rigid_torque Body torques
    \param d_rigid_group Body indices
    \param n_bodies Total umber of rigid bodies
    \param npt_rdata_eta_dot_t0 Thermostat translational velocity 
    \param npt_rdata_eta_dot_r0 Thermostat rotational velocity
    \param npt_rdata_epsilon_dot Barostat velocity
    \param npt_rdata_partial_Ksum_t Body translational kinetic energy 
    \param npt_rdata_partial_Ksum_r Body rotation kinetic energy
    \param npt_rdata_nf_t Translational degrees of freedom
    \param npt_rdata_nf_r Translational degrees of freedom
    \param npt_rdata_dimension System dimesion
    \param box Box dimensions for periodic boundary condition handling
    \param deltaT Timestep 
    
*/

extern "C" __global__ void gpu_npt_rigid_step_one_body_kernel(float4* rdata_com, 
                                                            float4* rdata_vel, 
                                                            float4* rdata_angmom, 
                                                            float4* rdata_angvel,
                                                            float4* rdata_orientation, 
                                                            float4* rdata_ex_space, 
                                                            float4* rdata_ey_space, 
                                                            float4* rdata_ez_space, 
                                                            int* rdata_body_imagex, 
                                                            int* rdata_body_imagey, 
                                                            int* rdata_body_imagez,
                                                            float4* rdata_conjqm,
                                                            float *d_rigid_mass,
                                                            float4 *d_rigid_mi,
                                                            float4 *d_rigid_force,
                                                            float4 *d_rigid_torque,
                                                            unsigned int *d_rigid_group, 
                                                            unsigned int n_group_bodies,  
                                                            unsigned int n_bodies, 
                                                            float npt_rdata_eta_dot_t0, 
                                                            float npt_rdata_eta_dot_r0,
                                                            float npt_rdata_epsilon_dot, 
                                                            float* npt_rdata_partial_Ksum_t, 
                                                            float* npt_rdata_partial_Ksum_r,
                                                            unsigned int npt_rdata_nf_t,
                                                            unsigned int npt_rdata_nf_r,
                                                            unsigned int npt_rdata_dimension, 
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
    float4 moment_inertia, com, vel, orientation, ex_space, ey_space, ez_space, force, torque, conjqm;
    int body_imagex, body_imagey, body_imagez;
    float4 mbody, tbody, fquat;

    float dt_half = 0.5f * deltaT;
    float onednft, onednfr, tmp, scale_t, scale_r, scale_v, akin_t, akin_r;

    onednft = 1.0f + (float) (npt_rdata_dimension) / (float) (npt_rdata_nf_t);
    onednfr = (float) (npt_rdata_dimension) / (float) (npt_rdata_nf_r);

    tmp = -1.0f * dt_half * (npt_rdata_eta_dot_t0 + onednft * npt_rdata_epsilon_dot);
    scale_t = exp(tmp);
    tmp = -1.0f * dt_half * (npt_rdata_eta_dot_r0 + onednfr * npt_rdata_epsilon_dot);
    scale_r = exp(tmp);
    tmp = dt_half * npt_rdata_epsilon_dot;
    scale_v = deltaT * __expf(tmp) * maclaurin_series(tmp);

    unsigned int idx_body = d_rigid_group[group_idx];
    body_mass = d_rigid_mass[idx_body];
    moment_inertia = d_rigid_mi[idx_body];
    com = rdata_com[idx_body];
    vel = rdata_vel[idx_body];
    orientation = rdata_orientation[idx_body];
    body_imagex = rdata_body_imagex[idx_body];
    body_imagey = rdata_body_imagey[idx_body];
    body_imagez = rdata_body_imagez[idx_body];
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
    float4 pos2;
    pos2.x = com.x + vel2.x * scale_v;
    pos2.y = com.y + vel2.y * scale_v;
    pos2.z = com.z + vel2.z * scale_v;
    pos2.w = com.w;
    
    // read in body's image
    // time to fix the periodic boundary conditions
    float x_shift = rintf(pos2.x * box.Lxinv);
    pos2.x -= box.Lx * x_shift;
    body_imagex += (int)x_shift;
    
    float y_shift = rintf(pos2.y * box.Lyinv);
    pos2.y -= box.Ly * y_shift;
    body_imagey += (int)y_shift;
    
    float z_shift = rintf(pos2.z * box.Lzinv);
    pos2.z -= box.Lz * z_shift;
    body_imagez += (int)z_shift;
    
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

    rdata_com[idx_body] = pos2;
    rdata_vel[idx_body] = vel2;
    rdata_angmom[idx_body] = angmom2;
    rdata_angvel[idx_body] = angvel2;
    rdata_orientation[idx_body] = orientation;
    rdata_ex_space[idx_body] = ex_space;
    rdata_ey_space[idx_body] = ey_space;
    rdata_ez_space[idx_body] = ez_space;
    rdata_body_imagex[idx_body] = body_imagex;
    rdata_body_imagey[idx_body] = body_imagey;
    rdata_body_imagez[idx_body] = body_imagez;
    rdata_conjqm[idx_body] = conjqm2;
    
    npt_rdata_partial_Ksum_t[idx_body] = akin_t;
    npt_rdata_partial_Ksum_r[idx_body] = akin_r;
    }

/*! \param pdata Particle data to step forward 1/2 step
    \param rigid_data Rigid body data to step forward 1/2 step
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param d_net_force Particle net forces
    \param box Box dimensions for periodic boundary condition handling
    \param npt_rdata Thermostat/barostat data
    \param deltaT Amount of real time to step forward in one time step
    
*/
cudaError_t gpu_npt_rigid_step_one(const gpu_pdata_arrays& pdata,       
                                   const gpu_rigid_data_arrays& rigid_data,
                                   float4 *d_pdata_orientation,
                                   unsigned int *d_group_members,
                                   unsigned int group_size,
                                   float4 *d_net_force,
                                   const gpu_boxsize &box, 
                                   const gpu_npt_rigid_data& npt_rdata,
                                   float deltaT)
    {
    unsigned int n_bodies = rigid_data.n_bodies;
    unsigned int n_group_bodies = rigid_data.n_group_bodies;
    unsigned int nmax = rigid_data.nmax;
    
    // setup the grid to run the kernel for rigid bodies
    int block_size = 64;
    int n_blocks = n_group_bodies / block_size + 1;
    dim3 body_grid(n_blocks, 1, 1);
    dim3 body_threads(block_size, 1, 1);
    gpu_npt_rigid_step_one_body_kernel<<< body_grid, body_threads  >>>(rigid_data.com, 
                                                                        rigid_data.vel, 
                                                                        rigid_data.angmom, 
                                                                        rigid_data.angvel,
                                                                        rigid_data.orientation, 
                                                                        rigid_data.ex_space, 
                                                                        rigid_data.ey_space, 
                                                                        rigid_data.ez_space, 
                                                                        rigid_data.body_imagex, 
                                                                        rigid_data.body_imagey, 
                                                                        rigid_data.body_imagez,
                                                                        rigid_data.conjqm,
                                                                        rigid_data.body_mass,
                                                                        rigid_data.moment_inertia,
                                                                        rigid_data.force,
                                                                        rigid_data.torque,
                                                                        rigid_data.body_indices,
                                                                        n_group_bodies,
                                                                        n_bodies, 
                                                                        npt_rdata.eta_dot_t0, 
                                                                        npt_rdata.eta_dot_r0,
                                                                        npt_rdata.epsilon_dot, 
                                                                        npt_rdata.partial_Ksum_t,
                                                                        npt_rdata.partial_Ksum_r,
                                                                        npt_rdata.nf_t,
                                                                        npt_rdata.nf_r,
                                                                        npt_rdata.dimension,
                                                                        box, 
                                                                        deltaT);
       
    gpu_npt_rigid_remap_kernel<<< body_grid, body_threads >>>(rigid_data.com,
                                                                rigid_data.body_indices,
                                                                n_group_bodies,
                                                                n_bodies,
                                                                box, 
                                                                npt_rdata);

    block_size = 192;
    dim3 particle_grid(group_size/block_size+1, 1, 1);
    dim3 particle_threads(block_size, 1, 1);
    
    gpu_rigid_setxv_kernel<true><<< particle_grid, particle_threads >>>(pdata.pos, 
                                                                        pdata.vel,
                                                                        d_pdata_orientation,
                                                                        pdata.image,
                                                                        d_group_members,
                                                                        group_size,
                                                                        rigid_data.particle_offset,
                                                                        pdata.body,
                                                                        rigid_data.body_indices,
                                                                        rigid_data.orientation,
                                                                        rigid_data.com,
                                                                        rigid_data.vel,
                                                                        rigid_data.angvel,
                                                                        rigid_data.body_imagex,
                                                                        rigid_data.body_imagey,
                                                                        rigid_data.body_imagez,
                                                                        rigid_data.particle_indices,
                                                                        rigid_data.particle_pos,
                                                                        rigid_data.particle_orientation,
                                                                        n_group_bodies,
                                                                        pdata.N,
                                                                        nmax,
                                                                        box);
                                                                    
    return cudaSuccess;
    }

#pragma mark RIGID_STEP_TWO_KERNEL
//! Takes the 2nd 1/2 step forward in the velocity-verlet NPT integration scheme
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

extern "C" __global__ void gpu_npt_rigid_step_two_body_kernel(float4* rdata_vel, 
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
                                                          float npt_rdata_eta_dot_t0, 
                                                          float npt_rdata_eta_dot_r0,
                                                          float npt_rdata_epsilon_dot, 
                                                          float* npt_rdata_partial_Ksum_t,
                                                          float* npt_rdata_partial_Ksum_r,
                                                          unsigned int npt_rdata_nf_t,
                                                          unsigned int npt_rdata_nf_r,
                                                          unsigned int npt_rdata_dimension,
                                                          gpu_boxsize box, 
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
    float   onednft, onednfr, tmp, scale_t, scale_r, akin_t, akin_r;
    
    onednft = 1.0f + (float) (npt_rdata_dimension) / (float) (npt_rdata_nf_t);
    onednfr = (float) (npt_rdata_dimension) / (float) (npt_rdata_nf_r);

    tmp = -1.0f * dt_half * (npt_rdata_eta_dot_t0 + onednft * npt_rdata_epsilon_dot);
    scale_t = exp(tmp);
    tmp = -1.0f * dt_half * (npt_rdata_eta_dot_r0 + onednfr * npt_rdata_epsilon_dot);
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
    
    npt_rdata_partial_Ksum_t[idx_body] = akin_t;
    npt_rdata_partial_Ksum_r[idx_body] = akin_r;
    }

/*! \param pdata Particle data to step forward 1/2 step
    \param rigid_data Rigid body data to step forward 1/2 step
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param d_net_force Particle net forces
    \param d_net_virial Particle net virial
    \param box Box dimensions for periodic boundary condition handling
    \param npt_rdata Thermostat/barostat data
    \param deltaT Amount of real time to step forward in one time step
    
*/
cudaError_t gpu_npt_rigid_step_two(const gpu_pdata_arrays &pdata, 
                                    const gpu_rigid_data_arrays& rigid_data,
                                    float4 *d_pdata_orientation,
                                    unsigned int *d_group_members,
                                    unsigned int group_size,
                                    float4 *d_net_force,
                                    float *d_net_virial,
                                    const gpu_boxsize &box, 
                                    const gpu_npt_rigid_data& npt_rdata,
                                    float deltaT)
    {
    unsigned int n_bodies = rigid_data.n_bodies;
    unsigned int n_group_bodies = rigid_data.n_group_bodies;
    unsigned int nmax = rigid_data.nmax;
                                                                                                                                                            
    unsigned int block_size = 64;
    unsigned int n_blocks = n_group_bodies / block_size + 1;                                
    dim3 body_grid(n_blocks, 1, 1);
    dim3 body_threads(block_size, 1, 1);                                                 
    gpu_npt_rigid_step_two_body_kernel<<< body_grid, body_threads >>>(rigid_data.vel, 
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
                                                                    npt_rdata.eta_dot_t0, 
                                                                    npt_rdata.eta_dot_r0,
                                                                    npt_rdata.epsilon_dot, 
                                                                    npt_rdata.partial_Ksum_t,
                                                                    npt_rdata.partial_Ksum_r,
                                                                    npt_rdata.nf_t,
                                                                    npt_rdata.nf_r,
                                                                    npt_rdata.dimension, 
                                                                    box, 
                                                                    deltaT);

    block_size = 192;
    dim3 particle_grid(group_size/block_size+1, 1, 1);
    dim3 particle_threads(block_size, 1, 1);
    
    gpu_rigid_setxv_kernel<false><<< particle_grid, particle_threads >>>(pdata.pos, 
                                                                        pdata.vel,
                                                                        d_pdata_orientation,
                                                                        pdata.image,
                                                                        d_group_members,
                                                                        group_size,
                                                                        rigid_data.particle_offset,
                                                                        pdata.body,
                                                                        rigid_data.body_indices,
                                                                        rigid_data.orientation,
                                                                        rigid_data.com,
                                                                        rigid_data.vel,
                                                                        rigid_data.angvel,
                                                                        rigid_data.body_imagex,
                                                                        rigid_data.body_imagey,
                                                                        rigid_data.body_imagez,
                                                                        rigid_data.particle_indices,
                                                                        rigid_data.particle_pos,
                                                                        rigid_data.particle_orientation,
                                                                        n_group_bodies,
                                                                        pdata.N,
                                                                        nmax,
                                                                        box);
                                                                                                                                             
    return cudaSuccess;
    }

#pragma mark RIGID_KINETIC_ENERGY_REDUCTION

//! Shared memory for kinetic energy reduction
extern __shared__ float npt_rigid_sdata[];

/*! Summing the kinetic energy of rigid bodies
    \param npt_rdata Thermostat data for rigid bodies 
    
*/
extern "C" __global__ void gpu_npt_rigid_reduce_ksum_kernel(gpu_npt_rigid_data npt_rdata)
    {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float* body_ke_t = npt_rigid_sdata;
    float* body_ke_r = &npt_rigid_sdata[blockDim.x];
    
    float Ksum_t = 0.0f, Ksum_r=0.0f;
    
    // sum up the values in the partial sum via a sliding window
    for (int start = 0; start < npt_rdata.n_bodies; start += blockDim.x)
        {
        if (start + threadIdx.x < npt_rdata.n_bodies)
            {
            body_ke_t[threadIdx.x] = npt_rdata.partial_Ksum_t[start + threadIdx.x];
            body_ke_r[threadIdx.x] = npt_rdata.partial_Ksum_r[start + threadIdx.x];
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
        *npt_rdata.Ksum_t = Ksum_t;
        *npt_rdata.Ksum_r = Ksum_r;
        }
        
    }

/*! 
    \param npt_rdata Thermostat/barostat data for rigid bodies 
*/
cudaError_t gpu_npt_rigid_reduce_ksum(const gpu_npt_rigid_data& npt_rdata)
    {
    // setup the grid to run the kernel
    int block_size = 128;
    dim3 grid( 1, 1, 1);
    dim3 threads(block_size, 1, 1);
    
    // run the kernel: double the block size to accomodate Ksum_t and Ksum_r
    gpu_npt_rigid_reduce_ksum_kernel<<< grid, threads, 2 * block_size * sizeof(float) >>>(npt_rdata);
    
    return cudaSuccess;
    }

