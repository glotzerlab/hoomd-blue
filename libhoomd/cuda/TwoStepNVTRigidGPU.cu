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

#include "TwoStepNVTRigidGPU.cuh"
#include "gpu_settings.h"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file TwoStepNVTRigidGPU.cu
    \brief Defines GPU kernel code for NVT integration on the GPU. Used by TwoStepNVTGPU.
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

//! The texture for reading the rigid data conjugate qm array
texture<float4, 1, cudaReadModeElementType> nvt_rdata_conjqm_tex;


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

/*! Apply evolution operators to quat, quat momentum (see ref. Miller)
    \param k Direction
    \param p Thermostat angular momentum conjqm
    \param q Quaternion
    \param inertia Moment of inertia
    \param dt Time step
*/

__device__ void no_squish_rotate(unsigned int k, float4& p, float4& q, float4& inertia, float dt)
    {
    float phi, c_phi, s_phi;
    float4 kp, kq;
    
    // apply permuation operator on p and q, get kp and kq
    if (k == 1)
        {
        kq.x = -q.y;  kp.x = -p.y;
        kq.y =  q.x;  kp.y =  p.x;
        kq.z =  q.w;  kp.z =  p.w;
        kq.w = -q.z;  kp.w = -p.z;
        }
    else if (k == 2)
        {
        kq.x = -q.z;  kp.x = -p.z;
        kq.y = -q.w;  kp.y = -p.w;
        kq.z =  q.x;  kp.z =  p.x;
        kq.w =  q.y;  kp.w =  p.y;
        }
    else if (k == 3)
        {
        kq.x = -q.w;  kp.x = -p.w;
        kq.y =  q.z;  kp.y =  p.z;
        kq.z = -q.y;  kp.z = -p.y;
        kq.w =  q.x;  kp.w =  p.x;
        }
    else
        {
        kq.x = 0.0f;  kp.x = 0.0f;
        kq.y = 0.0f;  kp.y = 0.0f;
        kq.z = 0.0f;  kp.z = 0.0f;
        kq.w = 0.0f;  kp.w = 0.0f;
        }
            
    // obtain phi, cosines and sines
    
    phi = p.x * kq.x + p.y * kq.y + p.z * kq.z + p.w * kq.w;
    
    float inertia_t;
    if (k == 1) inertia_t = inertia.x;
    else if (k == 2) inertia_t = inertia.y;
    else if (k == 3) inertia_t = inertia.z;
    else inertia_t = 0.0f;
    if (inertia_t < 1e-7) phi *= 0.0;
    else phi /= 4.0 * inertia_t;
    
    c_phi = __cosf(dt * phi);
    s_phi = __sinf(dt * phi);
    
    // advance p and q
    
    p.x = c_phi * p.x + s_phi * kp.x;
    p.y = c_phi * p.y + s_phi * kp.y;
    p.z = c_phi * p.z + s_phi * kp.z;
    p.w = c_phi * p.w + s_phi * kp.w;
    
    q.x = c_phi * q.x + s_phi * kq.x;
    q.y = c_phi * q.y + s_phi * kq.y;
    q.z = c_phi * q.z + s_phi * kq.z;
    q.w = c_phi * q.w + s_phi * kq.w;
    }

/*! Quaternion multiply: c = a * b 
    \param a Quaternion
    \param b A three component vector
    \param c Returned quaternion
*/
__device__ void quat_multiply(float4& a, float4& b, float4& c)
    {
    c.x = -a.y * b.x - a.z * b.y - a.w * b.z;
    c.y =  a.x * b.x - a.w * b.y + a.z * b.z;
    c.z =  a.w * b.x + a.x * b.y - a.y * b.z;
    c.w = -a.z * b.x + a.y * b.y + a.x * b.z;
    }

/*! Inverse quaternion multiply: c = inv(a) * b 
    \param a Quaternion
    \param b A four component vector
    \param c A three component vector
*/

__device__ void inv_quat_multiply(float4& a, float4& b, float4& c)
    {
    c.x = -a.y * b.x + a.x * b.y + a.w * b.z - a.z * b.w;
    c.y = -a.z * b.x - a.w * b.y + a.x * b.z + a.y * b.w;
    c.z = -a.w * b.x + a.z * b.y - a.y * b.z + a.x * b.w;
    }

/*! Matrix dot: c = dot(A, b) 
    \param ax The first row of A
    \param ay The second row of A
    \param az The third row of A
    \param b A three component vector
    \param c A three component vector    
*/

__device__ void matrix_dot(float4& ax, float4& ay, float4& az, float4& b, float4& c)
    {
    c.x = ax.x * b.x + ax.y * b.y + ax.z * b.z;
    c.y = ay.x * b.x + ay.y * b.y + ay.z * b.z;
    c.z = az.x * b.x + az.y * b.y + az.z * b.z;
    }

/*! Matrix transpose dot: c = dot(trans(A), b) 
    \param ax The first row of A
    \param ay The second row of A
    \param az The third row of A
    \param b A three component vector
    \param c A three component vector
*/
__device__ void transpose_dot(float4& ax, float4& ay, float4& az, float4& b, float4& c)
    {
    c.x = ax.x * b.x + ay.x * b.y + az.x * b.z;
    c.y = ax.y * b.x + ay.y * b.y + az.y * b.z;
    c.z = ax.z * b.x + ay.z * b.y + az.z * b.z;
    }

/*! Taylor expansion
    \param x Point to take the expansion

*/
__device__ float taylor_exp(float x)
    {
    float x2, x3, x4, x5;
    x2 = x * x;
    x3 = x2 * x;
    x4 = x2 * x2;
    x5 = x4 * x;
    return (1.0 + x + x2 / 2.0 + x3 / 6.0 + x4 / 24.0 + x5 / 120.0);
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
    \param n_bodies Number of rigid bodies
    \param local_beg Starting body index in this card
    \param nvt_rdata_eta_dot_t0 Thermostat translational part 
    \param nvt_rdata_eta_dot_r0 Thermostat rotational part
    \param nvt_rdata_conjqm Thermostat angular momentum
    \param nvt_rdata_partial_Ksum_t Body translational kinetic energy 
    \param nvt_rdata_partial_Ksum_r Body rotation kinetic energy
    \param deltaT Timestep 
    \param box Box dimensions for periodic boundary condition handling
*/

extern "C" __global__ void gpu_nvt_rigid_step_one_body_kernel(float4* rdata_com, 
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
                                                            float nvt_rdata_eta_dot_t0, 
                                                            float nvt_rdata_eta_dot_r0, 
                                                            float4* nvt_rdata_conjqm, 
                                                            float* nvt_rdata_partial_Ksum_t, 
                                                            float* nvt_rdata_partial_Ksum_r, 
                                                            gpu_boxsize box, 
                                                            float deltaT)
    {
    unsigned int idx_body = blockIdx.x * blockDim.x + threadIdx.x + local_beg;
    
    // do velocity verlet update
    // v(t+deltaT/2) = v(t) + (1/2)a*deltaT
    // r(t+deltaT) = r(t) + v(t+deltaT/2)*deltaT
    float body_mass;
    float4 moment_inertia, com, vel, orientation, ex_space, ey_space, ez_space, force, torque, conjqm;
    int body_imagex, body_imagey, body_imagez;
    float4 pos2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 vel2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 mbody, tbody, fquat;
    
    float dt_half = 0.5 * deltaT;
    float   tmp, scale_t, scale_r, akin_t, akin_r;
    tmp = -1.0 * dt_half * nvt_rdata_eta_dot_t0;
    scale_t = __expf(tmp);
    tmp = -1.0 * dt_half * nvt_rdata_eta_dot_r0;
    scale_r = __expf(tmp);
    
    // ri, body_mass and moment_inertia is used throughout the kernel
    if (idx_body < n_bodies)
        {
        body_mass = tex1Dfetch(rigid_data_body_mass_tex, idx_body);
        moment_inertia = tex1Dfetch(rigid_data_moment_inertia_tex, idx_body);
        com = tex1Dfetch(rigid_data_com_tex, idx_body);
        vel = tex1Dfetch(rigid_data_vel_tex, idx_body);
        orientation = tex1Dfetch(rigid_data_orientation_tex, idx_body);
        ex_space = tex1Dfetch(rigid_data_exspace_tex, idx_body);
        ey_space = tex1Dfetch(rigid_data_eyspace_tex, idx_body);
        ez_space = tex1Dfetch(rigid_data_ezspace_tex, idx_body);
        body_imagex = tex1Dfetch(rigid_data_body_imagex_tex, idx_body);
        body_imagey = tex1Dfetch(rigid_data_body_imagey_tex, idx_body);
        body_imagez = tex1Dfetch(rigid_data_body_imagez_tex, idx_body);
        force = tex1Dfetch(rigid_data_force_tex, idx_body);
        torque = tex1Dfetch(rigid_data_torque_tex, idx_body);
        
        conjqm = tex1Dfetch(nvt_rdata_conjqm_tex, idx_body);
        
        // update velocity
        float dtfm = dt_half / body_mass;
        
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
        
        matrix_dot(ex_space, ey_space, ez_space, torque, tbody);
        quat_multiply(orientation, tbody, fquat);
        
        float4 conjqm2;
        conjqm2.x = conjqm.x + deltaT * fquat.x;
        conjqm2.y = conjqm.y + deltaT * fquat.y;
        conjqm2.z = conjqm.z + deltaT * fquat.z;
        conjqm2.w = conjqm.w + deltaT * fquat.w;
        
        conjqm2.x *= scale_r;
        conjqm2.y *= scale_r;
        conjqm2.z *= scale_r;
        conjqm2.w *= scale_r;
        
        // step 1.4 to 1.13 - use no_squish rotate to update p and q
        
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
        inv_quat_multiply(orientation, conjqm2, mbody);
        transpose_dot(ex_space, ey_space, ez_space, mbody, angmom2);
        
        angmom2.x *= 0.5;
        angmom2.y *= 0.5;
        angmom2.z *= 0.5;
        
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
        
        nvt_rdata_conjqm[idx_body] = conjqm2;
        nvt_rdata_partial_Ksum_t[idx_body] = akin_t;
        nvt_rdata_partial_Ksum_r[idx_body] = akin_r;
        
            
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
extern "C" __global__ void gpu_rigid_nvt_step_one_particle_kernel(float4* pdata_pos,
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
    int body_imagex = 0;
    int body_imagey = 0;
    int body_imagez = 0;
        
    // ri, body_mass and moment_inertia is used throughout the kernel
    if (idx_body < n_bodies)
        {
        com = tex1Dfetch(rigid_data_com_tex, idx_body);
        vel = tex1Dfetch(rigid_data_vel_tex, idx_body);
        angvel = tex1Dfetch(rigid_data_angvel_tex, idx_body);
        ex_space = tex1Dfetch(rigid_data_exspace_tex, idx_body);
        ey_space = tex1Dfetch(rigid_data_eyspace_tex, idx_body);
        ez_space = tex1Dfetch(rigid_data_ezspace_tex, idx_body);
        body_imagex = tex1Dfetch(rigid_data_body_imagex_tex, idx_body);
        body_imagey = tex1Dfetch(rigid_data_body_imagey_tex, idx_body);
        body_imagez = tex1Dfetch(rigid_data_body_imagez_tex, idx_body);
        particle_pos = tex1Dfetch(rigid_data_particle_pos_tex, idx_particle);
                    
        unsigned int idx_particle_index = tex1Dfetch(rigid_data_particle_indices_tex, idx_particle);        
        // Since we use nmax for all rigid bodies, there might be some empty slot for particles in a rigid body
        // the particle index of these empty slots is set to be INVALID_INDEX.
        // Setting particle images here is different from normal point particles
        // because the particle position is set from the body center of mass: 
        // two adjacent wraps do not mean the particle has moved twice the box lengths, 
        // so we have to check with the old position
        if (idx_particle_index != INVALID_INDEX)
            {
            float4 pos = tex1Dfetch(pdata_pos_tex, idx_particle_index);
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
            ppos.w = pos.w;
            
            // time to fix the periodic boundary conditions (FLOPS: 15)
            float x_shift = rintf(ppos.x * box.Lxinv);
            ppos.x -= box.Lx * x_shift;
            image.x = body_imagex;
            image.x += (int)x_shift;
            
            float y_shift = rintf(ppos.y * box.Lyinv);
            ppos.y -= box.Ly * y_shift;
            image.y = body_imagey;
            image.y += (int)y_shift;
            
            float z_shift = rintf(ppos.z * box.Lzinv);
            ppos.z -= box.Lz * z_shift;
            image.z = body_imagez;
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

/*!
    \param pdata_pos Particle position
    \param pdata_vel Particle velocity
    \param pdata_image Particle image
    \param n_bodies Number of rigid bodies
    \param local_beg Starting body index in this card
    \param nmax Maximum number of particles in a rigid body
    \param block_size Block size
    \param box Box dimensions for periodic boundary condition handling
*/
extern "C" __global__ void gpu_rigid_nvt_step_one_particle_sliding_kernel(float4* pdata_pos,
                                                        float4* pdata_vel,
                                                        int4* pdata_image,
                                                        unsigned int n_bodies, 
                                                        unsigned int local_beg,
                                                        unsigned int nmax,
                                                        unsigned int block_size,
                                                        gpu_boxsize box)
    {
    unsigned int idx_body = blockIdx.x + local_beg;
    
    float4 particle_pos, ri;
    
    float4 com = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 vel = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 angvel = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 ex_space = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 ey_space = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 ez_space = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    int body_imagex = 0;
    int body_imagey = 0;
    int body_imagez = 0;
    
    if (idx_body < n_bodies)
        {
        com = tex1Dfetch(rigid_data_com_tex, idx_body);
        vel = tex1Dfetch(rigid_data_vel_tex, idx_body);
        angvel = tex1Dfetch(rigid_data_angvel_tex, idx_body);
        ex_space = tex1Dfetch(rigid_data_exspace_tex, idx_body);
        ey_space = tex1Dfetch(rigid_data_eyspace_tex, idx_body);
        ez_space = tex1Dfetch(rigid_data_ezspace_tex, idx_body);
        body_imagex = tex1Dfetch(rigid_data_body_imagex_tex, idx_body);
        body_imagey = tex1Dfetch(rigid_data_body_imagey_tex, idx_body);
        body_imagez = tex1Dfetch(rigid_data_body_imagez_tex, idx_body);
        }

    unsigned int n_windows = nmax / block_size;
    
    for (unsigned int start = 0; start < n_windows; start++)
        {
        int idx_particle = idx_body * nmax + start * block_size + threadIdx.x;
        unsigned int idx_particle_index = tex1Dfetch(rigid_data_particle_indices_tex, idx_particle);  
        
        if (idx_body < n_bodies && idx_particle_index != INVALID_INDEX)
            {
            particle_pos = tex1Dfetch(rigid_data_particle_pos_tex, idx_particle);
            float4 pos = tex1Dfetch(pdata_pos_tex, idx_particle_index);
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
            ppos.w = pos.w;
            
            // time to fix the periodic boundary conditions (FLOPS: 15)
            float x_shift = rintf(ppos.x * box.Lxinv);
            ppos.x -= box.Lx * x_shift;
            image.x = body_imagex;
            image.x += (int)x_shift;
            
            float y_shift = rintf(ppos.y * box.Lyinv);
            ppos.y -= box.Ly * y_shift;
            image.y = body_imagey;
            image.y += (int)y_shift;
            
            float z_shift = rintf(ppos.z * box.Lzinv);
            ppos.z -= box.Lz * z_shift;
            image.z = body_imagez;
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

/*! \param pdata Particle data to step forward 1/2 step
    \param rigid_data Rigid body data to step forward 1/2 step
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param box Box dimensions for periodic boundary condition handling
    \param nvt_rdata Thermostat data
    \param deltaT Amount of real time to step forward in one time step
    
*/
cudaError_t gpu_nvt_rigid_step_one(const gpu_pdata_arrays& pdata,       
                                    const gpu_rigid_data_arrays& rigid_data,
                                    unsigned int *d_group_members,
                                    unsigned int group_size,
                                    const gpu_boxsize &box, 
                                    const gpu_nvt_rigid_data& nvt_rdata,
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
        
    error = cudaBindTexture(0, nvt_rdata_conjqm_tex, nvt_rdata.conjqm, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
    
    // setup the grid to run the kernel for rigid bodies
    int block_size = 64;
    int n_blocks = n_bodies / block_size + 1;
    dim3 body_grid(n_blocks, 1, 1);
    dim3 body_threads(block_size, 1, 1);
/*    gpu_nvt_rigid_step_one_body_kernel<<< body_grid, body_threads  >>>(rigid_data.com, 
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
                                                                        nvt_rdata.eta_dot_t0, 
                                                                        nvt_rdata.eta_dot_r0, 
                                                                        nvt_rdata.conjqm,
                                                                        nvt_rdata.partial_Ksum_t,
                                                                        nvt_rdata.partial_Ksum_r, 
                                                                        box, 
                                                                        deltaT);
 */   
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

    if (nmax <= 32)
        {
        block_size = nmax; // maximum number of particles in a rigid body: each thread in a block takes care of a particle in a rigid body
        dim3 particle_grid(n_bodies, 1, 1);
        dim3 particle_threads(block_size, 1, 1);
        
        gpu_rigid_nvt_step_one_particle_kernel<<< particle_grid, particle_threads >>>(pdata.pos, 
                                                                     pdata.vel, 
                                                                     pdata.image,
                                                                     n_bodies, 
                                                                     local_beg,
                                                                     box);
        }
    else
        {
        block_size = 16; 
        dim3 particle_grid(n_bodies, 1, 1);
        dim3 particle_threads(block_size, 1, 1);
        
        gpu_rigid_nvt_step_one_particle_sliding_kernel<<< particle_grid, particle_threads >>>(pdata.pos, 
                                                                     pdata.vel, 
                                                                     pdata.image,
                                                                     n_bodies, 
                                                                     local_beg,
                                                                     nmax,
                                                                     block_size,
                                                                     box);
        }
                                                                    
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


//! Takes the 2nd 1/2 step forward in the velocity-verlet NVT integration scheme
/*!  
    \param rdata_vel Body velocity
    \param rdata_angmom Angular momentum
    \param rdata_angvel Angular velocity
    \param n_bodies Number of rigid bodies
    \param local_beg Starting body index in this card
    \param nvt_rdata_eta_dot_t0 Thermostat translational part 
    \param nvt_rdata_eta_dot_r0 Thermostat rotational part
    \param nvt_rdata_conjqm Thermostat angular momentum
    \param nvt_rdata_partial_Ksum_t Body translational kinetic energy 
    \param nvt_rdata_partial_Ksum_r Body rotation kinetic energy
    \param deltaT Timestep 
    \param box Box dimensions for periodic boundary condition handling
*/

extern "C" __global__ void gpu_nvt_rigid_step_two_body_kernel(float4* rdata_vel, 
                                                          float4* rdata_angmom, 
                                                          float4* rdata_angvel,
                                                          unsigned int n_bodies, 
                                                          unsigned int local_beg, 
                                                          float nvt_rdata_eta_dot_t0, 
                                                          float nvt_rdata_eta_dot_r0, 
                                                          float4* nvt_rdata_conjqm,
                                                          float* nvt_rdata_partial_Ksum_t,
                                                          float* nvt_rdata_partial_Ksum_r,
                                                          gpu_boxsize box, 
                                                          float deltaT)
    {
    int idx_body = blockIdx.x * blockDim.x + threadIdx.x + local_beg;
    
    float body_mass;
    float4 moment_inertia, vel, ex_space, ey_space, ez_space, orientation, conjqm;
    float4 force, torque;
    float4 mbody, tbody, fquat;
    
    float dt_half = 0.5 * deltaT;
    float   tmp, scale_t, scale_r, akin_t, akin_r;
    tmp = -1.0 * dt_half * nvt_rdata_eta_dot_t0;
    scale_t = __expf(tmp);
    tmp = -1.0 * dt_half * nvt_rdata_eta_dot_r0;
    scale_r = __expf(tmp);
    
    // Update body velocity and angmom
    if (idx_body < n_bodies)
        {        
        body_mass = tex1Dfetch(rigid_data_body_mass_tex, idx_body);
        moment_inertia = tex1Dfetch(rigid_data_moment_inertia_tex, idx_body);
        vel = tex1Dfetch(rigid_data_vel_tex, idx_body);
        force = tex1Dfetch(rigid_data_force_tex, idx_body);
        torque = tex1Dfetch(rigid_data_torque_tex, idx_body);
        ex_space = tex1Dfetch(rigid_data_exspace_tex, idx_body);
        ey_space = tex1Dfetch(rigid_data_eyspace_tex, idx_body);
        ez_space = tex1Dfetch(rigid_data_ezspace_tex, idx_body);
        orientation = tex1Dfetch(rigid_data_orientation_tex, idx_body);
        
        conjqm = tex1Dfetch(nvt_rdata_conjqm_tex, idx_body);
        
        float dtfm = dt_half / body_mass;
        
        // update the velocity
        float4 vel2;
        vel2.x = scale_t * vel.x + dtfm * force.x;
        vel2.y = scale_t * vel.y + dtfm * force.y;
        vel2.z = scale_t * vel.z + dtfm * force.z;
        vel2.w = 0.0;
        
        tmp = vel2.x * vel2.x + vel2.y * vel2.y + vel2.z * vel2.z;
        akin_t = body_mass * tmp;
        
        // update angular momentum
        matrix_dot(ex_space, ey_space, ez_space, torque, tbody);
        quat_multiply(orientation, tbody, fquat);
        
        float4  conjqm2, angmom2;
        conjqm2.x = scale_r * conjqm.x + deltaT * fquat.x;
        conjqm2.y = scale_r * conjqm.y + deltaT * fquat.y;
        conjqm2.z = scale_r * conjqm.z + deltaT * fquat.z;
        conjqm2.w = scale_r * conjqm.w + deltaT * fquat.w;
        
        inv_quat_multiply(orientation, conjqm2, mbody);
        transpose_dot(ex_space, ey_space, ez_space, mbody, angmom2);
        
        angmom2.x *= 0.5;
        angmom2.y *= 0.5;
        angmom2.z *= 0.5;
        angmom2.w = 0.0;
        
        // update angular velocity
        float4 angvel2;
        computeAngularVelocity(angmom2, moment_inertia, ex_space, ey_space, ez_space, angvel2);
        
        akin_r = angmom2.x * angvel2.x + angmom2.y * angvel2.y + angmom2.z * angvel2.z;
        
        rdata_vel[idx_body] = vel2;
        rdata_angmom[idx_body] = angmom2;
        rdata_angvel[idx_body] = angvel2;
        
        nvt_rdata_conjqm[idx_body] = conjqm2;
        nvt_rdata_partial_Ksum_t[idx_body] = akin_t;
        nvt_rdata_partial_Ksum_r[idx_body] = akin_r;
        
        }
    }

/*!
    \param pdata_vel Particle velocity
    \param n_bodies Number of rigid bodies
    \param local_beg Starting body index in this card
    \param nmax Maximum number of particles in a rigid body
    \param box Box dimensions for periodic boundary condition handling
*/
extern "C" __global__ void gpu_rigid_nvt_step_two_particle_kernel(float4* pdata_vel, 
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

/*!
    \param pdata_vel Particle velocity
    \param n_bodies Number of rigid bodies
    \param local_beg Starting body index in this card
    \param nmax Maximum number of particles in a rigid body
    \param block_size Block size
    \param box Box dimensions for periodic boundary condition handling
*/
extern "C" __global__ void gpu_rigid_nvt_step_two_particle_sliding_kernel(float4* pdata_vel, 
                                                         unsigned int n_bodies, 
                                                         unsigned int local_beg,
                                                         unsigned int nmax,
                                                         unsigned int block_size,
                                                         gpu_boxsize box)
    {
    int idx_body = blockIdx.x + local_beg;
    
    float4 particle_pos, ri;
    
    float4 vel = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 angvel = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 ex_space = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 ey_space = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 ez_space = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    if (idx_body < n_bodies)
        {
        vel = tex1Dfetch(rigid_data_vel_tex, idx_body);
        angvel = tex1Dfetch(rigid_data_angvel_tex, idx_body);
        ex_space = tex1Dfetch(rigid_data_exspace_tex, idx_body);
        ey_space = tex1Dfetch(rigid_data_eyspace_tex, idx_body);
        ez_space = tex1Dfetch(rigid_data_ezspace_tex, idx_body);
        }
    
    unsigned int n_windows = nmax / block_size;
    
    for (unsigned int start = 0; start < n_windows; start++)
        {
        int idx_particle = idx_body * nmax + start * block_size + threadIdx.x;
		unsigned int idx_particle_index = tex1Dfetch(rigid_data_particle_indices_tex, idx_particle);
    
        if (idx_body < n_bodies && idx_particle_index != INVALID_INDEX)
            {
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

    }

/*! \param pdata Particle data to step forward 1/2 step
    \param rigid_data Rigid body data to step forward 1/2 step
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param d_net_force Particle net forces
    \param box Box dimensions for periodic boundary condition handling
    \param nvt_rdata Thermostat data
    \param deltaT Amount of real time to step forward in one time step
    
*/
cudaError_t gpu_nvt_rigid_step_two(const gpu_pdata_arrays &pdata, 
                                    const gpu_rigid_data_arrays& rigid_data, 
                                    unsigned int *d_group_members,
                                    unsigned int group_size,
                                    float4 *d_net_force,
                                    const gpu_boxsize &box, 
                                    const gpu_nvt_rigid_data& nvt_rdata,
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
        
    error = cudaBindTexture(0, rigid_data_orientation_tex, rigid_data.orientation, sizeof(float4) * n_bodies);
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

    error = cudaBindTexture(0, nvt_rdata_conjqm_tex, nvt_rdata.conjqm, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    // bind the textures for particles
    
    error = cudaBindTexture(0, pdata_pos_tex, pdata.pos, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;
                                                                                          
    unsigned int block_size = 64;
    unsigned int n_blocks = n_bodies / block_size + 1;                                
    dim3 body_grid(n_blocks, 1, 1);
    dim3 body_threads(block_size, 1, 1);                                                 
    gpu_nvt_rigid_step_two_body_kernel<<< body_grid, body_threads >>>(rigid_data.vel, 
                                                                    rigid_data.angmom, 
                                                                    rigid_data.angvel,
                                                                    n_bodies, 
                                                                    local_beg,
                                                                    nvt_rdata.eta_dot_t0, 
                                                                    nvt_rdata.eta_dot_r0, 
                                                                    nvt_rdata.conjqm,
                                                                    nvt_rdata.partial_Ksum_t,
                                                                    nvt_rdata.partial_Ksum_r, 
                                                                    box, 
                                                                    deltaT);

    // get the body information after the above update
    error = cudaBindTexture(0, rigid_data_vel_tex, rigid_data.vel, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_angvel_tex, rigid_data.angvel, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
 
                                                                                                                                    
    if (nmax <= 32)
        {                                                                                                                                    
        block_size = nmax; // each thread in a block takes care of a particle in a rigid body
        dim3 particle_grid(n_bodies, 1, 1);
        dim3 particle_threads(block_size, 1, 1);                                                
        gpu_rigid_nvt_step_two_particle_kernel<<< particle_grid, particle_threads >>>(pdata.vel, 
                                                        n_bodies, 
                                                        local_beg,
                                                        nmax, 
                                                        box);
        }
    else
        {
        block_size = 16; 
        dim3 particle_grid(n_bodies, 1, 1);
        dim3 particle_threads(block_size, 1, 1);                                                
        gpu_rigid_nvt_step_two_particle_sliding_kernel<<< particle_grid, particle_threads >>>(pdata.vel, 
                                                        n_bodies, 
                                                        local_beg,
                                                        nmax,
                                                        block_size, 
                                                        box);
        } 
                                                                                                                                             
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

#pragma mark RIGID_KINETIC_ENERGY_REDUCTION

//! Shared memory for kinetic energy reduction
extern __shared__ float nvt_rigid_sdata[];

/*! Summing the kinetic energy of rigid bodies
    \param nvt_rdata Thermostat data for rigid bodies 
    
*/
extern "C" __global__ void gpu_nvt_rigid_reduce_ksum_kernel(gpu_nvt_rigid_data nvt_rdata)
    {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float* body_ke_t = nvt_rigid_sdata;
    float* body_ke_r = &nvt_rigid_sdata[blockDim.x];
    
    float Ksum_t = 0.0f, Ksum_r=0.0f;
    
    // sum up the values in the partial sum via a sliding window
    for (int start = 0; start < nvt_rdata.n_bodies; start += blockDim.x)
        {
        if (start + threadIdx.x < nvt_rdata.n_bodies)
            {
            body_ke_t[threadIdx.x] = nvt_rdata.partial_Ksum_t[start + threadIdx.x];
            body_ke_r[threadIdx.x] = nvt_rdata.partial_Ksum_r[start + threadIdx.x];
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
        *nvt_rdata.Ksum_t = Ksum_t;
        *nvt_rdata.Ksum_r = Ksum_r;
        }
        
    }

/*! 
    \param nvt_rdata Thermostat data for rigid bodies 
    
*/
cudaError_t gpu_nvt_rigid_reduce_ksum(const gpu_nvt_rigid_data& nvt_rdata)
    {
    // setup the grid to run the kernel
    int block_size = 128;
    dim3 grid( 1, 1, 1);
    dim3 threads(block_size, 1, 1);
    
    // run the kernel: double the block size to accomodate Ksum_t and Ksum_r
    gpu_nvt_rigid_reduce_ksum_kernel<<< grid, threads, 2 * block_size * sizeof(float) >>>(nvt_rdata);
    
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

