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

#include "TwoStepNVERigidGPU.cuh"
#include "gpu_settings.h"

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

//! The texture for reading the pdata pos array
texture<float4, 1, cudaReadModeElementType> pdata_pos_tex;
//! The texture for reading the pdata vel array
texture<float4, 1, cudaReadModeElementType> pdata_vel_tex;
//! The texture for reading the pdata accel array
texture<float4, 1, cudaReadModeElementType> pdata_accel_tex;
//! The texture for reading in the pdata image array
texture<int4, 1, cudaReadModeElementType> pdata_image_tex;
//! The texture for reading in the pdata mass array
texture<float, 1, cudaReadModeElementType> pdata_mass_tex;

//! The texture for reading the rigid data body mass array
texture<float, 1, cudaReadModeElementType> rigid_data_body_mass_tex;
//! The texture for reading the rigid data moment of inertia array
texture<float4, 1, cudaReadModeElementType> rigid_data_moment_inertia_tex;
//! The texture for reading the rigid data com array
texture<float4, 1, cudaReadModeElementType> rigid_data_com_tex;
//! The texture for reading the rigid data vel array
texture<float4, 1, cudaReadModeElementType> rigid_data_vel_tex;
//! The texture for reading the rigid data angualr momentum array
texture<float4, 1, cudaReadModeElementType> rigid_data_angmom_tex;
//! The texture for reading the rigid data angular velocity array
texture<float4, 1, cudaReadModeElementType> rigid_data_angvel_tex;
//! The texture for reading the rigid data orientation array
texture<float4, 1, cudaReadModeElementType> rigid_data_orientation_tex;
//! The texture for reading the rigid data ex space array
texture<float4, 1, cudaReadModeElementType> rigid_data_exspace_tex;
//! The texture for reading the rigid data ey space array
texture<float4, 1, cudaReadModeElementType> rigid_data_eyspace_tex;
//! The texture for reading the rigid data ez space array
texture<float4, 1, cudaReadModeElementType> rigid_data_ezspace_tex;
//! The texture for reading in the rigid data body image array
texture<int, 1, cudaReadModeElementType> rigid_data_body_imagex_tex;
//! The texture for reading in the rigid data body image array
texture<int, 1, cudaReadModeElementType> rigid_data_body_imagey_tex;
//! The texture for reading in the rigid data body image array
texture<int, 1, cudaReadModeElementType> rigid_data_body_imagez_tex;
//! The texture for reading the rigid data particle position array
texture<float4, 1, cudaReadModeElementType> rigid_data_particle_pos_tex;
//! The texture for reading the rigid data particle indices array
texture<unsigned int, 1, cudaReadModeElementType> rigid_data_particle_indices_tex;
//! The texture for reading the rigid data force array
texture<float4, 1, cudaReadModeElementType> rigid_data_force_tex;
//! The texture for reading the rigid data torque array
texture<float4, 1, cudaReadModeElementType> rigid_data_torque_tex;

#pragma mark HELPER
//! Helper functions for rigid body quaternion update

/*! Convert the body axes from the quaternion
    \param quat Quaternion
    \param ex_space x-axis unit vector
    \param ey_space y-axis unit vector
    \param ez_space z-axis unit vector

*/
__device__ void exyzFromQuaternion(float4& quat, float4& ex_space, float4& ey_space, float4& ez_space)
    {
    // ex_space
    ex_space.x = quat.x * quat.x + quat.y * quat.y - quat.z * quat.z - quat.w * quat.w;
    ex_space.y = 2.0 * (quat.y * quat.z + quat.x * quat.w);
    ex_space.z = 2.0 * (quat.y * quat.w - quat.x * quat.z);
    
    // ey_space
    ey_space.x = 2.0 * (quat.y * quat.z - quat.x * quat.w);
    ey_space.y = quat.x * quat.x - quat.y * quat.y + quat.z * quat.z - quat.w * quat.w;
    ey_space.z = 2.0 * (quat.z * quat.w + quat.x * quat.y);
    
    // ez_space
    ez_space.x = 2.0 * (quat.y * quat.w + quat.x * quat.z);
    ez_space.y = 2.0 * (quat.z * quat.w - quat.x * quat.y);
    ez_space.z = quat.x * quat.x - quat.y * quat.y - quat.z * quat.z + quat.w * quat.w;
    }

/*! Compute angular velocity from angular momentum 
    \param angmom Angular momentum
    \param moment_inertia Moment of inertia
    \param ex_space x-axis unit vector
    \param ey_space y-axis unit vector
    \param ez_space z-axis unit vector
    \param angvel Returned angular velocity
*/
__device__ void computeAngularVelocity(float4& angmom, float4& moment_inertia, float4& ex_space, float4& ey_space, float4& ez_space, float4& angvel)
    {
    //! Angular velocity in the body frame
    float4 angbody;
    
    //! angbody = angmom_body / moment_inertia = transpose(rotation_matrix) * angmom / moment_inertia
    if (moment_inertia.x == 0.0) angbody.x = 0.0;
    else angbody.x = (ex_space.x * angmom.x + ex_space.y * angmom.y + ex_space.z * angmom.z) / moment_inertia.x;
    
    if (moment_inertia.y == 0.0) angbody.y = 0.0;
    else angbody.y = (ey_space.x * angmom.x + ey_space.y * angmom.y + ey_space.z * angmom.z) / moment_inertia.y;
    
    if (moment_inertia.z == 0.0) angbody.z = 0.0;
    else angbody.z = (ez_space.x * angmom.x + ez_space.y * angmom.y + ez_space.z * angmom.z) / moment_inertia.z;
    
    //! Convert to angbody to the space frame: angvel = rotation_matrix * angbody
    angvel.x = angbody.x * ex_space.x + angbody.y * ey_space.x + angbody.z * ez_space.x;
    angvel.y = angbody.x * ex_space.y + angbody.y * ey_space.y + angbody.z * ez_space.y;
    angvel.z = angbody.x * ex_space.z + angbody.y * ey_space.z + angbody.z * ez_space.z;
    }

/* Quaternion multiply: c = a * b where a = (0, a)
 */

__device__ void multiply(float4& a, float4& b, float4& c)
    {
    c.x = -(a.x * b.y + a.y * b.z + a.z * b.w);
    c.y =   b.x * a.x + a.y * b.w - a.z * b.z;
    c.z =   b.x * a.y + a.z * b.y - a.x * b.w;
    c.w =   b.x * a.z + a.x * b.z - a.y * b.y;
    }

/* Normalize a quaternion
 */

__device__ void normalize(float4 &q)
    {
    float norm = 1.0 / sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    q.x *= norm;
    q.y *= norm;
    q.z *= norm;
    q.w *= norm;
    }

/*! Advance the quaternion using angular momentum and angular velocity
    \param angmom Angular momentum
    \param moment_inertia Moment of inertia
    \param angvel Returned angular velocity
    \param ex_space x-axis unit vector
    \param ey_space y-axis unit vector
    \param ez_space z-axis unit vector
    \param quat Returned quaternion
    \param deltaT Time step
    
 */
__device__ void advanceQuaternion(float4& angmom, float4& moment_inertia, float4& angvel, float4& ex_space, float4& ey_space, float4& ez_space, float4& quat, float deltaT)
    {
    float4 qhalf, qfull, omegaq;
    float dtq = 0.5 * deltaT;
    
    computeAngularVelocity(angmom, moment_inertia, ex_space, ey_space, ez_space, angvel);
    
    // Compute (w q)
    multiply(angvel, quat, omegaq);
    
    // Full update q from dq/dt = 1/2 w q
    qfull.x = quat.x + dtq * omegaq.x;
    qfull.y = quat.y + dtq * omegaq.y;
    qfull.z = quat.z + dtq * omegaq.z;
    qfull.w = quat.w + dtq * omegaq.w;
    normalize(qfull);
    
    // 1st half update from dq/dt = 1/2 w q
    qhalf.x = quat.x + 0.5 * dtq * omegaq.x;
    qhalf.y = quat.y + 0.5 * dtq * omegaq.y;
    qhalf.z = quat.z + 0.5 * dtq * omegaq.z;
    qhalf.w = quat.w + 0.5 * dtq * omegaq.w;
    normalize(qhalf);
    
    // Udpate ex, ey, ez from qhalf = update A
    exyzFromQuaternion(qhalf, ex_space, ey_space, ez_space);
    
    // Compute angular velocity from new ex_space, ey_space and ex_space
    computeAngularVelocity(angmom, moment_inertia, ex_space, ey_space, ez_space, angvel);
    
    // Compute (w qhalf)
    multiply(angvel, qhalf, omegaq);
    
    // 2nd half update from dq/dt = 1/2 w q
    qhalf.x += 0.5 * dtq * omegaq.x;
    qhalf.y += 0.5 * dtq * omegaq.y;
    qhalf.z += 0.5 * dtq * omegaq.z;
    qhalf.w += 0.5 * dtq * omegaq.w;
    normalize(qhalf);
    
    // Corrected Richardson update
    quat.x = 2.0 * qhalf.x - qfull.x;
    quat.y = 2.0 * qhalf.y - qfull.y;
    quat.z = 2.0 * qhalf.z - qfull.z;
    quat.w = 2.0 * qhalf.w - qfull.w;
    normalize(quat);
    
    exyzFromQuaternion(quat, ex_space, ey_space, ez_space);
    }

#pragma mark RIGID_STEP_ONE_KERNEL

/*! Takes the first half-step forward for rigid bodies in the velocity-verlet NVE integration
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
    \param n_bodies Number of rigid bodies
    \param local_beg Starting body index in this card
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
                                                        int* rdata_body_imagex, 
                                                        int* rdata_body_imagey, 
                                                        int* rdata_body_imagez, 
                                                        unsigned int n_bodies, 
                                                        unsigned int local_beg,
                                                        gpu_boxsize box, 
                                                        float deltaT)
    {
    unsigned int idx_body = blockIdx.x * blockDim.x + threadIdx.x + local_beg;
    
    // do velocity verlet update
    // v(t+deltaT/2) = v(t) + (1/2)a*deltaT
    // r(t+deltaT) = r(t) + v(t+deltaT/2)*deltaT
    float body_mass;
    float4 moment_inertia, com, vel, angmom, orientation, ex_space, ey_space, ez_space, force, torque;
    int body_imagex, body_imagey, body_imagez;
    float4 ri = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 pos2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 vel2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    // ri, body_mass and moment_inertia is used throughout the kernel
    if (idx_body < n_bodies)
        {
        body_mass = tex1Dfetch(rigid_data_body_mass_tex, idx_body);
        moment_inertia = tex1Dfetch(rigid_data_moment_inertia_tex, idx_body);
        com = tex1Dfetch(rigid_data_com_tex, idx_body);
        vel = tex1Dfetch(rigid_data_vel_tex, idx_body);
        angmom = tex1Dfetch(rigid_data_angmom_tex, idx_body);
        orientation = tex1Dfetch(rigid_data_orientation_tex, idx_body);
        ex_space = tex1Dfetch(rigid_data_exspace_tex, idx_body);
        ey_space = tex1Dfetch(rigid_data_eyspace_tex, idx_body);
        ez_space = tex1Dfetch(rigid_data_ezspace_tex, idx_body);
        body_imagex = tex1Dfetch(rigid_data_body_imagex_tex, idx_body);
        body_imagey = tex1Dfetch(rigid_data_body_imagey_tex, idx_body);
        body_imagez = tex1Dfetch(rigid_data_body_imagez_tex, idx_body);
        force = tex1Dfetch(rigid_data_force_tex, idx_body);
        torque = tex1Dfetch(rigid_data_torque_tex, idx_body);
        
        // update velocity
        float dtfm = (1.0f/2.0f) * deltaT / body_mass;
        
        vel2.x = vel.x + dtfm * force.x;
        vel2.y = vel.y + dtfm * force.y;
        vel2.z = vel.z + dtfm * force.z;
        vel2.w = vel.w;
        
        // update position
        pos2.x = com.x + vel2.x * deltaT;
        pos2.y = com.y + vel2.y * deltaT;
        pos2.z = com.z + vel2.z * deltaT;
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
        
        // update the angular momentum
        float4 angmom2;
        angmom2.x = angmom.x + (1.0f/2.0f) * deltaT * torque.x;
        angmom2.y = angmom.y + (1.0f/2.0f) * deltaT * torque.y;
        angmom2.z = angmom.z + (1.0f/2.0f) * deltaT * torque.z;
        
        float4 angvel2;
        advanceQuaternion(angmom2, moment_inertia, angvel2, ex_space, ey_space, ez_space, orientation, deltaT);
        
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
            
        }
    }

/*!
    \param pdata_pos Particle position
    \param pdata_vel Particle velocity
    \param pdata_image Particle image
    \param n_bodies Number of rigid bodies
    \param local_beg Starting body index in this card
    \param box Box dimensions for periodic boundary condition handling
*/
extern "C" __global__ void gpu_rigid_step_one_particle_kernel(float4* pdata_pos,
                                                        float4* pdata_vel,
                                                        int4* pdata_image,
                                                        unsigned int n_bodies, 
                                                        unsigned int local_beg,
                                                        gpu_boxsize box)
    {
    unsigned int idx_particle = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idx_body = blockIdx.x + local_beg;
    
    float4 com, vel, angvel, ex_space, ey_space, ez_space, particle_pos;
    float4 ri = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    // ri, body_mass and moment_inertia is used throughout the kernel
    if (idx_body < n_bodies)
        {
        com = tex1Dfetch(rigid_data_com_tex, idx_body);
        vel = tex1Dfetch(rigid_data_vel_tex, idx_body);
        angvel = tex1Dfetch(rigid_data_angvel_tex, idx_body);
        ex_space = tex1Dfetch(rigid_data_exspace_tex, idx_body);
        ey_space = tex1Dfetch(rigid_data_eyspace_tex, idx_body);
        ez_space = tex1Dfetch(rigid_data_ezspace_tex, idx_body);
        particle_pos = tex1Dfetch(rigid_data_particle_pos_tex, idx_particle);
                    
        unsigned int idx_particle_index = tex1Dfetch(rigid_data_particle_indices_tex, idx_particle);        
        // Since we use nmax for all rigid bodies, there might be some empty slot for particles in a rigid body
        // the particle index of these empty slots is set to be INVALID_INDEX.
        if (idx_particle_index != INVALID_INDEX)
            {
            int4 image = tex1Dfetch(pdata_image_tex, idx_particle_index);
            
            // compute ri with new orientation
            ri.x = ex_space.x * particle_pos.x + ey_space.x * particle_pos.y + ez_space.x * particle_pos.z;
            ri.y = ex_space.y * particle_pos.x + ey_space.y * particle_pos.y + ez_space.y * particle_pos.z;
            ri.z = ex_space.z * particle_pos.x + ey_space.z * particle_pos.y + ez_space.z * particle_pos.z;
            
            // x_particle = com + ri
            float4 ppos;
            ppos.x = com.x + ri.x;
            ppos.y = com.y + ri.y;
            ppos.z = com.z + ri.z;
            ppos.w = com.w;
            
            // time to fix the periodic boundary conditions (FLOPS: 15)
            float x_shift = rintf(ppos.x * box.Lxinv);
            ppos.x -= box.Lx * x_shift;
            image.x += (int)x_shift;
            
            float y_shift = rintf(ppos.y * box.Lyinv);
            ppos.y -= box.Ly * y_shift;
            image.y += (int)y_shift;
            
            float z_shift = rintf(ppos.z * box.Lzinv);
            ppos.z -= box.Lz * z_shift;
            image.z += (int)z_shift;
            
            // v_particle = vel + angvel x ri
            float4 pvel;
            pvel.x = vel.x + angvel.y * ri.z - angvel.z * ri.y;
            pvel.y = vel.y + angvel.z * ri.x - angvel.x * ri.z;
            pvel.z = vel.z + angvel.x * ri.y - angvel.y * ri.x;
            pvel.w = 0.0;
            
            // write out the results (MEM_TRANSFER: ? bytes)
            pdata_pos[idx_particle_index] = ppos;
            pdata_vel[idx_particle_index] = pvel;
            pdata_image[idx_particle_index] = image;
            }
        }
    }

//! Takes the first 1/2 step forward in the NVE integration step
/*! \param pdata Particle data to step forward 1/2 step
    \param rigid_data Rigid body data to step forward 1/2 step
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param box Box dimensions for periodic boundary condition handling
    \param deltaT Amount of real time to step forward in one time step
*/
cudaError_t gpu_nve_rigid_step_one(const gpu_pdata_arrays& pdata, 
                                   const gpu_rigid_data_arrays& rigid_data, 
                                   unsigned int *d_group_members,
                                   unsigned int group_size,
                                   const gpu_boxsize &box, 
                                   float deltaT)
    {
    unsigned int n_bodies = rigid_data.n_bodies;
    unsigned int local_beg = rigid_data.local_beg;
    unsigned int nmax = rigid_data.nmax;
    
    // bind the textures for rigid bodies:
    // body mass, com, vel, angmom, angvel, orientation, ex_space, ey_space, ez_space, body images, particle pos, particle indices, force and torque
    
    cudaError_t error = cudaBindTexture(0, rigid_data_body_mass_tex, rigid_data.body_mass, sizeof(float) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_moment_inertia_tex, rigid_data.moment_inertia, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_com_tex, rigid_data.com, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_vel_tex, rigid_data.vel, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_angvel_tex, rigid_data.angvel, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_angmom_tex, rigid_data.angmom, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_orientation_tex, rigid_data.orientation, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_exspace_tex, rigid_data.ex_space, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_eyspace_tex, rigid_data.ey_space, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_ezspace_tex, rigid_data.ez_space, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_body_imagex_tex, rigid_data.body_imagex, sizeof(int) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_body_imagey_tex, rigid_data.body_imagey, sizeof(int) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_body_imagez_tex, rigid_data.body_imagez, sizeof(int) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_particle_pos_tex, rigid_data.particle_pos, sizeof(float4) * n_bodies * nmax);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_particle_indices_tex, rigid_data.particle_indices, sizeof(unsigned int) * n_bodies * nmax);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_force_tex, rigid_data.force, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_torque_tex, rigid_data.torque, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    // setup the grid to run the kernel for rigid bodies
    int block_size = 64;
    int n_blocks = n_bodies / block_size + 1;
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
                                                           rigid_data.body_imagex, 
                                                           rigid_data.body_imagey, 
                                                           rigid_data.body_imagez, 
                                                           n_bodies, 
                                                           local_beg,
                                                           box, 
                                                           deltaT);
    
    // get the body information after the above update
    error = cudaBindTexture(0, rigid_data_com_tex, rigid_data.com, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_vel_tex, rigid_data.vel, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_angvel_tex, rigid_data.angvel, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_exspace_tex, rigid_data.ex_space, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_eyspace_tex, rigid_data.ey_space, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_ezspace_tex, rigid_data.ez_space, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;

    // bind the textures for particles: pos, vel, accel and image of ALL particles
    error = cudaBindTexture(0, pdata_pos_tex, pdata.pos, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, pdata_vel_tex, pdata.vel, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, pdata_image_tex, pdata.image, sizeof(int4) * pdata.N);
    if (error != cudaSuccess)
        return error;

    block_size = nmax; // maximum number of particles in a rigid body: each thread in a block takes care of a particle in a rigid body
    dim3 particle_grid(n_bodies, 1, 1);
    dim3 particle_threads(block_size, 1, 1);
    
    gpu_rigid_step_one_particle_kernel<<< particle_grid, particle_threads >>>(pdata.pos, 
                                                                 pdata.vel, 
                                                                 pdata.image,
                                                                 n_bodies, 
                                                                 local_beg,
                                                                 box);
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

    
#pragma mark RIGID_FORCE_KERNEL

//! The texture for reading the net force array
texture<float4, 1, cudaReadModeElementType> net_force_tex;

//! Shared memory for body force and torque reduction, required allocation when the kernel is called
extern __shared__ float4 sum[];

//! Takes the 2nd 1/2 step forward in the velocity-verlet NVE integration scheme
/*! \param rdata_force Particle data to step forward in time
    \param rdata_torque List of pointers to forces on each particle
    \param n_bodies Number of forces listed in \a force_data_ptrs
    \param local_beg Amount of real time to step forward in one time step
    \param nmax Maximum number of particles in a rigid body
    \param window_size Window size for reduction
    \param box Box dimensions for periodic boundary condition handling
*/
extern "C" __global__ void gpu_rigid_force_kernel(float4* rdata_force, 
                                                 float4* rdata_torque, 
                                                 unsigned int n_bodies, 
                                                 unsigned int local_beg,
                                                 unsigned int nmax,
                                                 unsigned int window_size,
                                                 gpu_boxsize box)
    {
    int idx_particle = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_body = blockIdx.x + local_beg;
    
    float4 *body_force = sum;
    float4 *body_torque = &sum[blockDim.x];
    
    body_force[threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    body_torque[threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    unsigned int idx_particle_index = tex1Dfetch(rigid_data_particle_indices_tex, idx_particle);
    float4 com, pos, ri;
    float4 fi = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 torquei = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    if (idx_body < n_bodies && idx_particle_index != INVALID_INDEX)
        {
        // calculate body force and torques
        com = tex1Dfetch(rigid_data_com_tex, idx_body);

        pos = tex1Dfetch(pdata_pos_tex, idx_particle_index);
        fi = tex1Dfetch(net_force_tex, idx_particle_index);
        
        body_force[threadIdx.x].x = fi.x;
        body_force[threadIdx.x].y = fi.y;
        body_force[threadIdx.x].z = fi.z;
        body_force[threadIdx.x].w = fi.w;
        
        // project the position in the body frame to the space frame: ri = rotation_matrix * particle_pos
        ri.x = pos.x - com.x;
        ri.y = pos.y - com.y;
        ri.z = pos.z - com.z;
        
        float x_shift = rintf(ri.x * box.Lxinv);
        ri.x -= box.Lx * x_shift;
        float y_shift = rintf(ri.y * box.Lyinv);
        ri.y -= box.Ly * y_shift;
        float z_shift = rintf(ri.z * box.Lzinv);
        ri.z -= box.Lz * z_shift;
        
        torquei.x = ri.y * fi.z - ri.z * fi.y;
        torquei.y = ri.z * fi.x - ri.x * fi.z;
        torquei.z = ri.x * fi.y - ri.y * fi.x;
        torquei.w = 0.0;
        
        body_torque[threadIdx.x].x = torquei.x;
        body_torque[threadIdx.x].y = torquei.y;
        body_torque[threadIdx.x].z = torquei.z;
        body_torque[threadIdx.x].w = torquei.w;
        }
        
    __syncthreads();
    
    unsigned int offset = blockDim.x >> 1;
    while (offset > 0)
        {
        if (threadIdx.x < offset)
            {
            body_force[threadIdx.x].x += body_force[threadIdx.x + offset].x;
            body_force[threadIdx.x].y += body_force[threadIdx.x + offset].y;
            body_force[threadIdx.x].z += body_force[threadIdx.x + offset].z;
            body_force[threadIdx.x].w += body_force[threadIdx.x + offset].w;
            
            body_torque[threadIdx.x].x += body_torque[threadIdx.x + offset].x;
            body_torque[threadIdx.x].y += body_torque[threadIdx.x + offset].y;
            body_torque[threadIdx.x].z += body_torque[threadIdx.x + offset].z;
            body_torque[threadIdx.x].w += body_torque[threadIdx.x + offset].w;
            }
            
        offset >>= 1;
        
        __syncthreads();
        }
    if (idx_body < n_bodies)
        {
        // Every thread now has its own copy of body force and torque
        float4 force2 = body_force[0];
        float4 torque2 = body_torque[0];
        
        rdata_force[idx_body] = force2;
        rdata_torque[idx_body] = torque2;
        }

           
    }

/*! \param pdata Particle data to step forward 1/2 step
    \param rigid_data Rigid body data to step forward 1/2 step
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param d_net_force Particle net forces
    \param box Box dimensions for periodic boundary condition handling
    \param deltaT Amount of real time to step forward in one time step
*/
cudaError_t gpu_rigid_force(const gpu_pdata_arrays &pdata, 
                                   const gpu_rigid_data_arrays& rigid_data,
                                   unsigned int *d_group_members,
                                   unsigned int group_size, 
                                   float4 *d_net_force,
                                   const gpu_boxsize &box,
                                   float deltaT)
    {
    unsigned int n_bodies = rigid_data.n_bodies;
    unsigned int local_beg = rigid_data.local_beg;
    unsigned int nmax = rigid_data.nmax;
    
    // bind the textures for ALL rigid bodies
    cudaError_t error = cudaBindTexture(0, rigid_data_com_tex, rigid_data.com, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    // bind the textures for particles
    error = cudaBindTexture(0, pdata_pos_tex, pdata.pos, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;        
   
    error = cudaBindTexture(0, net_force_tex, d_net_force, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;   
        
    // run the kernel: the shared memory size is used for dynamic memory allocation of extern __shared__ sum
    // tested with separate extern __shared__ body_force and body_torque each with size of nmax * sizeof(float4) but it did not work
    unsigned int window_size = nmax;
    unsigned int block_size = nmax; // each thread in a block takes care of a particle in a rigid body
    dim3 force_grid(n_bodies, 1, 1);
    dim3 force_threads(block_size, 1, 1); 
    gpu_rigid_force_kernel<<< force_grid, force_threads, 2 * window_size * sizeof(float4) >>>(rigid_data.force, 
                                                                                             rigid_data.torque, 
                                                                                             n_bodies, 
                                                                                             local_beg,
                                                                                             nmax,
                                                                                             window_size,
                                                                                             box);
           
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


#pragma mark RIGID_STEP_TWO_KERNEL

/*! Takes the second half-step forward for rigid bodies in the velocity-verlet NVE integration
    \param rdata_vel Body velocity
    \param rdata_angmom Angular momentum
    \param rdata_angvel Angular velocity
    \param n_bodies Number of rigid bodies
    \param local_beg Starting body index in this card
    \param nmax Maximum number of particles in a rigid body
    \param deltaT Timestep 
    \param box Box dimensions for periodic boundary condition handling
*/
extern "C" __global__ void gpu_nve_rigid_step_two_body_kernel(float4* rdata_vel, 
                                                         float4* rdata_angmom, 
                                                         float4* rdata_angvel,
                                                         unsigned int n_bodies, 
                                                         unsigned int local_beg,
                                                         unsigned int nmax,
                                                         gpu_boxsize box, 
                                                         float deltaT)
    {
    int idx_body = blockIdx.x * blockDim.x + threadIdx.x + local_beg;
    
    float body_mass;
    float4 moment_inertia, vel, angmom, ex_space, ey_space, ez_space;
    float4 force, torque;
            
    // Update body velocity and angmom
    if (idx_body < n_bodies)
        {        
        // update the velocity
        body_mass = tex1Dfetch(rigid_data_body_mass_tex, idx_body);
        vel = tex1Dfetch(rigid_data_vel_tex, idx_body);
        angmom = tex1Dfetch(rigid_data_angmom_tex, idx_body);
        force = tex1Dfetch(rigid_data_force_tex, idx_body);
        torque = tex1Dfetch(rigid_data_torque_tex, idx_body);
        moment_inertia = tex1Dfetch(rigid_data_moment_inertia_tex, idx_body);
        ex_space = tex1Dfetch(rigid_data_exspace_tex, idx_body);
        ey_space = tex1Dfetch(rigid_data_eyspace_tex, idx_body);
        ez_space = tex1Dfetch(rigid_data_ezspace_tex, idx_body);
        
        float dtfm = (1.0f/2.0f) * deltaT / body_mass;
        float4 vel2;
        vel2.x = vel.x + dtfm * force.x;
        vel2.y = vel.y + dtfm * force.y;
        vel2.z = vel.z + dtfm * force.z;
        vel2.w = 0.0;
        
        // update angular momentum
        float4 angmom2;
        angmom2.x = angmom.x + (1.0f/2.0f) * deltaT * torque.x;
        angmom2.y = angmom.y + (1.0f/2.0f) * deltaT * torque.y;
        angmom2.z = angmom.z + (1.0f/2.0f) * deltaT * torque.z;
        angmom2.w = 0.0;
        
        // update angular velocity        
        float4 angvel2;
        computeAngularVelocity(angmom2, moment_inertia, ex_space, ey_space, ez_space, angvel2);
        
        rdata_vel[idx_body] = vel2;
        rdata_angmom[idx_body] = angmom2;
        rdata_angvel[idx_body] = angvel2;
        }
    }

/*!
    \param pdata_vel Particle velocity
    \param n_bodies Number of rigid bodies
    \param local_beg Starting body index in this card
    \param nmax Maximum number of particles in a rigid body
    \param box Box dimensions for periodic boundary condition handling
*/
extern "C" __global__ void gpu_rigid_step_two_particle_kernel(float4* pdata_vel, 
                                                         unsigned int n_bodies, 
                                                         unsigned int local_beg,
                                                         unsigned int nmax,
                                                         gpu_boxsize box)
    {
    int idx_particle = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_body = blockIdx.x + local_beg;
    
    unsigned int idx_particle_index = tex1Dfetch(rigid_data_particle_indices_tex, idx_particle);
    float4 vel, angvel, ex_space, ey_space, ez_space, particle_pos, ri;
    
    if (idx_body < n_bodies && idx_particle_index != INVALID_INDEX)
        {
        vel = tex1Dfetch(rigid_data_vel_tex, idx_body);
        angvel = tex1Dfetch(rigid_data_angvel_tex, idx_body);
        ex_space = tex1Dfetch(rigid_data_exspace_tex, idx_body);
        ey_space = tex1Dfetch(rigid_data_eyspace_tex, idx_body);
        ez_space = tex1Dfetch(rigid_data_ezspace_tex, idx_body);
        particle_pos = tex1Dfetch(rigid_data_particle_pos_tex, idx_particle);
            
        ri.x = ex_space.x * particle_pos.x + ey_space.x * particle_pos.y + ez_space.x * particle_pos.z;
        ri.y = ex_space.y * particle_pos.x + ey_space.y * particle_pos.y + ez_space.y * particle_pos.z;
        ri.z = ex_space.z * particle_pos.x + ey_space.z * particle_pos.y + ez_space.z * particle_pos.z;
        
        // v_particle = v_com + angvel x xr
        float4 pvel;
        pvel.x = vel.x + angvel.y * ri.z - angvel.z * ri.y;
        pvel.y = vel.y + angvel.z * ri.x - angvel.x * ri.z;
        pvel.z = vel.z + angvel.x * ri.y - angvel.y * ri.x;
        pvel.w = 0.0;
        
        // write out the results
        pdata_vel[idx_particle_index] = pvel;
        }
    }


//! Take 1/2 first 1/2 step forward in the NVE integration step
/*! \param pdata Particle data to step forward 1/2 step
    \param rigid_data Rigid body data to step forward 1/2 step
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param d_net_force Particle net forces
    \param box Box dimensions for periodic boundary condition handling
    \param deltaT Amount of real time to step forward in one time step
*/
cudaError_t gpu_nve_rigid_step_two(const gpu_pdata_arrays &pdata, 
                                   const gpu_rigid_data_arrays& rigid_data,
                                   unsigned int *d_group_members,
                                   unsigned int group_size, 
                                   float4 *d_net_force,
                                   const gpu_boxsize &box,
                                   float deltaT)
    {
    unsigned int n_bodies = rigid_data.n_bodies;
    unsigned int local_beg = rigid_data.local_beg;
    unsigned int nmax = rigid_data.nmax;
    
    // bind the textures for ALL rigid bodies
    cudaError_t error = cudaBindTexture(0, rigid_data_body_mass_tex, rigid_data.body_mass, sizeof(float) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_moment_inertia_tex, rigid_data.moment_inertia, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_com_tex, rigid_data.com, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_vel_tex, rigid_data.vel, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_angvel_tex, rigid_data.angvel, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_angmom_tex, rigid_data.angmom, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_exspace_tex, rigid_data.ex_space, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_eyspace_tex, rigid_data.ey_space, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_ezspace_tex, rigid_data.ez_space, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_particle_pos_tex, rigid_data.particle_pos, sizeof(float4) * n_bodies * nmax);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_particle_indices_tex, rigid_data.particle_indices, sizeof(unsigned int) * n_bodies * nmax);
    if (error != cudaSuccess)
        return error;
    
    // bind the textures for particles
    error = cudaBindTexture(0, pdata_pos_tex, pdata.pos, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;        
   
    error = cudaBindTexture(0, net_force_tex, d_net_force, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;   
            
    error = cudaBindTexture(0, rigid_data_force_tex, rigid_data.force, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_torque_tex, rigid_data.torque, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    unsigned int block_size = 64;
    unsigned int n_blocks = n_bodies / block_size + 1;                                
    dim3 body_grid(n_blocks, 1, 1);
    dim3 body_threads(block_size, 1, 1);                                                 
    gpu_nve_rigid_step_two_body_kernel<<< body_grid, body_threads >>>(rigid_data.vel, 
                                                                      rigid_data.angmom, 
                                                                      rigid_data.angvel,
                                                                      n_bodies, 
                                                                      local_beg,
                                                                      nmax, 
                                                                      box, 
                                                                      deltaT);
    
    // get the body information after the above update
    error = cudaBindTexture(0, rigid_data_vel_tex, rigid_data.vel, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_angvel_tex, rigid_data.angvel, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
 
                                                                                                                                    
    block_size = nmax; // each thread in a block takes care of a particle in a rigid body
    dim3 particle_grid(n_bodies, 1, 1);
    dim3 particle_threads(block_size, 1, 1);                                                
    gpu_rigid_step_two_particle_kernel<<< particle_grid, particle_threads >>>(pdata.vel, 
                                                    n_bodies, 
                                                    local_beg,
                                                    nmax, 
                                                    box);
            
           
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
