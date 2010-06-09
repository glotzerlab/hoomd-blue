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

#include "TwoStepNPTRigidGPU.cuh"
#include "gpu_settings.h"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file TwoStepNPTRigidGPU.cu
    \brief Defines GPU kernel code for NPT integration on the GPU. Used by TwoStepNPTRigidGPU.
*/

//! Absolute threshold for zero moment of inertia
#define EPSILON 1e-3

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

//! The texture for reading the rigid data body indices array
texture<unsigned int, 1, cudaReadModeElementType> rigid_data_body_indices_tex;
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
texture<float4, 1, cudaReadModeElementType> npt_rdata_conjqm_tex;

//! The texture for reading the net virial array
texture<float, 1, cudaReadModeElementType> net_virial_tex;

//! The texture for reading the net force array
texture<float4, 1, cudaReadModeElementType> net_force_tex;

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
    if (moment_inertia.x < EPSILON) angbody.x = 0.0;
    else angbody.x = (ex_space.x * angmom.x + ex_space.y * angmom.y + ex_space.z * angmom.z) / moment_inertia.x;
    
    if (moment_inertia.y < EPSILON) angbody.y = 0.0;
    else angbody.y = (ey_space.x * angmom.x + ey_space.y * angmom.y + ey_space.z * angmom.z) / moment_inertia.y;
    
    if (moment_inertia.z < EPSILON) angbody.z = 0.0;
    else angbody.z = (ez_space.x * angmom.x + ez_space.y * angmom.y + ez_space.z * angmom.z) / moment_inertia.z;
    
    //! Convert to angbody to the space frame: angvel = rotation_matrix * angbody
    angvel.x = angbody.x * ex_space.x + angbody.y * ey_space.x + angbody.z * ez_space.x;
    angvel.y = angbody.x * ex_space.y + angbody.y * ey_space.y + angbody.z * ez_space.y;
    angvel.z = angbody.x * ex_space.z + angbody.y * ey_space.z + angbody.z * ez_space.z;
    }

/*! Normalize a quaternion
    \param q Quaternion to be normalized
 */

__device__ void normalize(float4 &q)
    {
    float norm = 1.0 / sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    q.x *= norm;
    q.y *= norm;
    q.z *= norm;
    q.w *= norm;
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
    normalize(q);
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

/*! Maclaurine expansion
    \param x Point to take the expansion

*/
__device__ float maclaurin_series(float x)
    {
    float x2, x4;
    x2 = x * x;
    x4 = x2 * x2;
    return (1.0 + (1.0/6.0) * x2 + (1.0/120.0) * x4 + (1.0/5040.0) * x2 * x4 + (1.0/362880.0) * x4 * x4);
    }

#pragma mark RIGID_ZERO_VIRIAL_KERNEL

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
        d_virial_rigid[idx] = 0.0;
        }

    }
    
#pragma mark RIGID_REMAP_KERNEL

/*! Takes the first half-step forward for rigid bodies in the velocity-verlet NVT integration 
    \param rdata_com Body center of mass
    \param n_group_bodies Number of rigid bodies in my group
    \param n_bodies Total umber of rigid bodies
    \param local_beg Starting body index in this card
    \param box Box dimensions for periodic boundary condition handling
    \param npt_rdata Thermostat/barostat data
*/

extern "C" __global__ void gpu_npt_rigid_remap_kernel(float4* rdata_com,
                                                      unsigned int n_group_bodies,
                                                      unsigned int n_bodies, 
                                                      unsigned int local_beg, 
                                                      gpu_boxsize box,
                                                      gpu_npt_rigid_data npt_rdata)
    {
    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x + local_beg;
    if (group_idx < n_group_bodies)
        {
        unsigned int idx_body = tex1Dfetch(rigid_data_body_indices_tex, group_idx);
        
        if (idx_body < n_bodies)
            {
            float oldlo, oldhi, ctr;
            float xlo, xhi, ylo, yhi, zlo, zhi, Lx, Ly, Lz;
            float4 pos, delta;
            float dilation = npt_rdata.dilation;
            
            xlo = -box.Lx/2.0;
            xhi = box.Lx/2.0;
            ylo = -box.Ly/2.0;
            yhi = box.Ly/2.0;
            zlo = -box.Lz/2.0;
            zhi = box.Lz/2.0;
            
            float4 com = tex1Dfetch(rigid_data_com_tex, idx_body);
            
            delta.x = com.x - xlo;
            delta.y = com.y - ylo;
            delta.z = com.z - zlo;

            pos.x = box.Lxinv * delta.x;
            pos.y = box.Lyinv * delta.y;
            pos.z = box.Lzinv * delta.z;
            
            // reset box to new size/shape
            oldlo = xlo;
            oldhi = xhi;
            ctr = 0.5 * (oldlo + oldhi);
            xlo = (oldlo - ctr) * dilation + ctr;
            xhi = (oldhi - ctr) * dilation + ctr;
            Lx = xhi - xlo;
            
            oldlo = ylo;
            oldhi = yhi;
            ctr = 0.5 * (oldlo + oldhi);
            ylo = (oldlo - ctr) * dilation + ctr;
            yhi = (oldhi - ctr) * dilation + ctr;
            Ly = yhi - ylo;
            
            oldlo = zlo;
            oldhi = zhi;
            ctr = 0.5 * (oldlo + oldhi);
            zlo = (oldlo - ctr) * dilation + ctr;
            zhi = (oldhi - ctr) * dilation + ctr;
            Lz = zhi - zlo;
            
            // convert rigid body COMs back to box coords
            float4 newboxlo;
            newboxlo.x = -Lx/2.0;
            newboxlo.y = -Ly/2.0;
            newboxlo.z = -Lz/2.0;
            
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
    \param n_group_bodies Number of rigid bodies in my group
    \param n_bodies Total umber of rigid bodies
    \param local_beg Starting body index in this card
    \param npt_rdata_eta_dot_t0 Thermostat translational velocity 
    \param npt_rdata_eta_dot_r0 Thermostat rotational velocity
    \param npt_rdata_epsilon_dot Barostat velocity
    \param npt_rdata_conjqm Thermostat angular momentum
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
                                                            unsigned int n_group_bodies,  
                                                            unsigned int n_bodies, 
                                                            unsigned int local_beg,
                                                            float npt_rdata_eta_dot_t0, 
                                                            float npt_rdata_eta_dot_r0,
                                                            float npt_rdata_epsilon_dot, 
                                                            float4* npt_rdata_conjqm, 
                                                            float* npt_rdata_partial_Ksum_t, 
                                                            float* npt_rdata_partial_Ksum_r,
                                                            unsigned int npt_rdata_nf_t,
                                                            unsigned int npt_rdata_nf_r,
                                                            unsigned int npt_rdata_dimension, 
                                                            gpu_boxsize box, 
                                                            float deltaT)
    {
    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x + local_beg;
    
    // do velocity verlet update
    // v(t+deltaT/2) = v(t) + (1/2)a*deltaT
    // r(t+deltaT) = r(t) + v(t+deltaT/2)*deltaT
    if (group_idx < n_group_bodies)
        {
        float body_mass;
        float4 moment_inertia, com, vel, orientation, ex_space, ey_space, ez_space, force, torque, conjqm;
        int body_imagex, body_imagey, body_imagez;
        float4 pos2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 vel2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 mbody, tbody, fquat;
        
        float dt_half = 0.5 * deltaT;
        float onednft, onednfr, tmp, scale_t, scale_r, scale_v, akin_t, akin_r;
        
        onednft = 1.0 + (float) (npt_rdata_dimension) / (float) (npt_rdata_nf_t);
        onednfr = (float) (npt_rdata_dimension) / (float) (npt_rdata_nf_r);

        tmp = -1.0 * dt_half * (npt_rdata_eta_dot_t0 + onednft * npt_rdata_epsilon_dot);
        scale_t = exp(tmp);
        tmp = -1.0 * dt_half * (npt_rdata_eta_dot_r0 + onednfr * npt_rdata_epsilon_dot);
        scale_r = exp(tmp);
        tmp = dt_half * npt_rdata_epsilon_dot;
        scale_v = deltaT * __expf(tmp) * maclaurin_series(tmp);
        
        unsigned int idx_body = tex1Dfetch(rigid_data_body_indices_tex, group_idx);
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
            
            conjqm = tex1Dfetch(npt_rdata_conjqm_tex, idx_body);
            
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
            
            npt_rdata_conjqm[idx_body] = conjqm2;
            npt_rdata_partial_Ksum_t[idx_body] = akin_t;
            npt_rdata_partial_Ksum_r[idx_body] = akin_r;
            }
        }
    }

/*!
    \param pdata_pos Particle position
    \param pdata_vel Particle velocity
    \param pdata_image Particle image
    \param d_net_virial Particle virial
    \param n_group_bodies Number of rigid bodies in my group
    \param n_bodies Number of rigid bodies
    \param local_beg Starting body index in this card
    \param new_box Box dimensions for periodic boundary condition handling
    \param deltaT Time step
*/
extern "C" __global__ void gpu_npt_rigid_step_one_particle_kernel(float4* pdata_pos,
                                                        float4* pdata_vel,
                                                        int4* pdata_image,
                                                        float *d_net_virial,
                                                        unsigned int n_group_bodies,
                                                        unsigned int n_bodies, 
                                                        unsigned int local_beg,
                                                        float4* new_box,
                                                        float deltaT)
    {
    unsigned int group_idx = blockIdx.x + local_beg;
    
    __shared__ float4 com, vel, angvel, ex_space, ey_space, ez_space;
    __shared__ int body_imagex, body_imagey, body_imagez;
    
    int idx_body = -1;
    
    float dt_half = 0.5 * deltaT; 
    float4 boxsize = *new_box;
    float4 invboxsize;
    invboxsize.x = 1.0 / boxsize.x;
    invboxsize.y = 1.0 / boxsize.y;
    invboxsize.z = 1.0 / boxsize.z;
     
    if (group_idx < n_group_bodies)
        {
        idx_body = tex1Dfetch(rigid_data_body_indices_tex, group_idx); 
        
        if (idx_body < n_bodies && threadIdx.x == 0)
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
        }
        
    __syncthreads();
    
    if (idx_body >= 0 && idx_body < n_bodies)
        {
        unsigned int idx_particle = idx_body * blockDim.x + threadIdx.x;
        unsigned int idx_particle_index = tex1Dfetch(rigid_data_particle_indices_tex, idx_particle);        
        // Since we use nmax for all rigid bodies, there might be some empty slot for particles in a rigid body
        // the particle index of these empty slots is set to be INVALID_INDEX.
        if (idx_particle_index != INVALID_INDEX)
            {
            float4 particle_pos = tex1Dfetch(rigid_data_particle_pos_tex, idx_particle);
            float4 old_pos = tex1Dfetch(pdata_pos_tex, idx_particle_index);
            float4 old_vel = tex1Dfetch(pdata_vel_tex, idx_particle_index);
            float massone = tex1Dfetch(pdata_mass_tex, idx_particle_index);
            int4 image = tex1Dfetch(pdata_image_tex, idx_particle_index);
            float4 pforce = tex1Dfetch(net_force_tex, idx_particle_index);
            float virial = tex1Dfetch(net_virial_tex, idx_particle_index);
            
            // compute ri with new orientation
            float4 ri;
            ri.x = ex_space.x * particle_pos.x + ey_space.x * particle_pos.y + ez_space.x * particle_pos.z;
            ri.y = ex_space.y * particle_pos.x + ey_space.y * particle_pos.y + ez_space.y * particle_pos.z;
            ri.z = ex_space.z * particle_pos.x + ey_space.z * particle_pos.y + ez_space.z * particle_pos.z;
            
            // x_particle = com + ri
            float4 ppos;
            ppos.x = com.x + ri.x;
            ppos.y = com.y + ri.y;
            ppos.z = com.z + ri.z;
            ppos.w = old_pos.w;
            
            // time to fix the periodic boundary conditions (FLOPS: 15)
            float x_shift = rintf(ppos.x * invboxsize.x);
            ppos.x -= boxsize.x * x_shift;
            image.x = body_imagex;
            image.x += (int)x_shift;
            
            float y_shift = rintf(ppos.y * invboxsize.y);
            ppos.y -= boxsize.y * y_shift;
            image.y = body_imagey;
            image.y += (int)y_shift;
            
            float z_shift = rintf(ppos.z * invboxsize.z);
            ppos.z -= boxsize.z * z_shift;
            image.z = body_imagez;
            image.z += (int)z_shift;
            
            // v_particle = vel + angvel x ri
            float4 pvel;
            pvel.x = vel.x + angvel.y * ri.z - angvel.z * ri.y;
            pvel.y = vel.y + angvel.z * ri.x - angvel.x * ri.z;
            pvel.z = vel.z + angvel.x * ri.y - angvel.y * ri.x;
            pvel.w = 0.0;
            
            float4 fc;
            fc.x = massone * (pvel.x - old_vel.x) / dt_half - pforce.x;
            fc.y = massone * (pvel.y - old_vel.y) / dt_half - pforce.y;
            fc.z = massone * (pvel.z - old_vel.z) / dt_half - pforce.z; 
            
            float pvirial = virial + (0.5 * (old_pos.x * fc.x + old_pos.y * fc.y + old_pos.z * fc.z) / 3.0);
                            
            // write out the results (MEM_TRANSFER: ? bytes)
            pdata_pos[idx_particle_index] = ppos;
            pdata_vel[idx_particle_index] = pvel;
            pdata_image[idx_particle_index] = image;
            d_net_virial[idx_particle_index] = pvirial;
            }
        }
    }

/*!
    \param pdata_pos Particle position
    \param pdata_vel Particle velocity
    \param pdata_image Particle image
    \param d_virial_rigid Virial contribution from rigid bodies
    \param n_group_bodies Number of rigid bodies in my group
    \param n_bodies Total number of rigid bodies
    \param local_beg Starting body index in this card
    \param nmax Maximum number of particles in a rigid body
    \param block_size Block size
    \param new_box Box dimensions for periodic boundary condition handling
    \param deltaT Time step
*/
extern "C" __global__ void gpu_npt_rigid_step_one_particle_sliding_kernel(float4* pdata_pos,
                                                        float4* pdata_vel,
                                                        int4* pdata_image,
                                                        float *d_net_virial,
                                                        unsigned int n_group_bodies,
                                                        unsigned int n_bodies, 
                                                        unsigned int local_beg,
                                                        unsigned int nmax,
                                                        unsigned int block_size,
                                                        float4* new_box,
                                                        float deltaT)
    {
    unsigned int group_idx = blockIdx.x + local_beg;
    
    __shared__ float4 com, vel, angvel, ex_space, ey_space, ez_space;
    __shared__ int body_imagex, body_imagey, body_imagez;
    
    float dt_half = 0.5 * deltaT;
    
    float4 boxsize = *new_box;
    float4 invboxsize;
    invboxsize.x = 1.0 / boxsize.x;
    invboxsize.y = 1.0 / boxsize.y;
    invboxsize.z = 1.0 / boxsize.z;
    
    int idx_body = -1;
    
    if (group_idx < n_group_bodies)
        {
        idx_body = tex1Dfetch(rigid_data_body_indices_tex, group_idx);
        
        if (idx_body < n_bodies && threadIdx.x == 0)
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
        }
        
    __syncthreads();
    
    unsigned int n_windows = nmax / block_size + 1;
    for (unsigned int start = 0; start < n_windows; start++)
        {
        if (idx_body >= 0 && idx_body < n_bodies)
            {
            int idx_particle = idx_body * nmax + start * block_size + threadIdx.x;
            if (idx_particle < nmax * n_bodies && start * block_size + threadIdx.x < nmax)
                {
                unsigned int idx_particle_index = tex1Dfetch(rigid_data_particle_indices_tex, idx_particle);  
                
                if (idx_particle_index != INVALID_INDEX)
                    {
                    float4 particle_pos = tex1Dfetch(rigid_data_particle_pos_tex, idx_particle);
                    float4 old_pos = tex1Dfetch(pdata_pos_tex, idx_particle_index);
                    float4 old_vel = tex1Dfetch(pdata_vel_tex, idx_particle_index);
                    int4 image = tex1Dfetch(pdata_image_tex, idx_particle_index);
                    float massone = tex1Dfetch(pdata_mass_tex, idx_particle_index);
                    float4 pforce = tex1Dfetch(net_force_tex, idx_particle_index);
                    float virial = tex1Dfetch(net_virial_tex, idx_particle_index);
                    
                    // compute ri with new orientation
                    float4 ri;
                    ri.x = ex_space.x * particle_pos.x + ey_space.x * particle_pos.y + ez_space.x * particle_pos.z;
                    ri.y = ex_space.y * particle_pos.x + ey_space.y * particle_pos.y + ez_space.y * particle_pos.z;
                    ri.z = ex_space.z * particle_pos.x + ey_space.z * particle_pos.y + ez_space.z * particle_pos.z;
                    
                    // x_particle = com + ri
                    float4 ppos;
                    ppos.x = com.x + ri.x;
                    ppos.y = com.y + ri.y;
                    ppos.z = com.z + ri.z;
                    ppos.w = old_pos.w;
                    
                    // time to fix the periodic boundary conditions (FLOPS: 15)
                    float x_shift = rintf(ppos.x * invboxsize.x);
                    ppos.x -= boxsize.x * x_shift;
                    image.x = body_imagex;
                    image.x += (int)x_shift;
                    
                    float y_shift = rintf(ppos.y * invboxsize.y);
                    ppos.y -= boxsize.y * y_shift;
                    image.y = body_imagey;
                    image.y += (int)y_shift;
                    
                    float z_shift = rintf(ppos.z * invboxsize.z);
                    ppos.z -= boxsize.z * z_shift;
                    image.z = body_imagez;
                    image.z += (int)z_shift;
                    
                    // v_particle = vel + angvel x ri
                    float4 pvel;
                    pvel.x = vel.x + angvel.y * ri.z - angvel.z * ri.y;
                    pvel.y = vel.y + angvel.z * ri.x - angvel.x * ri.z;
                    pvel.z = vel.z + angvel.x * ri.y - angvel.y * ri.x;
                    pvel.w = 0.0;
                    
                    float4 fc;
                    fc.x = massone * (pvel.x - old_vel.x) / dt_half - pforce.x;
                    fc.y = massone * (pvel.y - old_vel.y) / dt_half - pforce.y;
                    fc.z = massone * (pvel.z - old_vel.z) / dt_half - pforce.z; 
                    
                    float pvirial = virial + (0.5 * (old_pos.x * fc.x + old_pos.y * fc.y + old_pos.z * fc.z) / 3.0);
                    
                    // write out the results (MEM_TRANSFER: ? bytes)
                    pdata_pos[idx_particle_index] = ppos;
                    pdata_vel[idx_particle_index] = pvel;
                    pdata_image[idx_particle_index] = image;
                    d_net_virial[idx_particle_index] = pvirial;
                    }
                }
            }
        }
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
    unsigned int local_beg = rigid_data.local_beg;
    unsigned int nmax = rigid_data.nmax;
    
    // bind the textures for rigid bodies:
    // body mass, com, vel, angmom, angvel, orientation, ex_space, ey_space, ez_space, body images, particle pos, 
    // particle indices, force and torque
    cudaError_t error = cudaBindTexture(0, rigid_data_body_indices_tex, rigid_data.body_indices, sizeof(float) * n_group_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_body_mass_tex, rigid_data.body_mass, sizeof(float) * n_bodies);
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
        
    error = cudaBindTexture(0, npt_rdata_conjqm_tex, npt_rdata.conjqm, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
    
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
                                                                        n_group_bodies,
                                                                        n_bodies, 
                                                                        local_beg, 
                                                                        npt_rdata.eta_dot_t0, 
                                                                        npt_rdata.eta_dot_r0,
                                                                        npt_rdata.epsilon_dot, 
                                                                        npt_rdata.conjqm,
                                                                        npt_rdata.partial_Ksum_t,
                                                                        npt_rdata.partial_Ksum_r,
                                                                        npt_rdata.nf_t,
                                                                        npt_rdata.nf_r,
                                                                        npt_rdata.dimension,
                                                                        box, 
                                                                        deltaT);
    
    // update COMs
    error = cudaBindTexture(0, rigid_data_com_tex, rigid_data.com, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
    
    gpu_npt_rigid_remap_kernel<<< body_grid, body_threads >>>(rigid_data.com,
                                                                n_group_bodies,
                                                                n_bodies,
                                                                local_beg,
                                                                box, 
                                                                npt_rdata);

    
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

    error = cudaBindTexture(0, rigid_data_body_imagex_tex, rigid_data.body_imagex, sizeof(int) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_body_imagey_tex, rigid_data.body_imagey, sizeof(int) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_body_imagez_tex, rigid_data.body_imagez, sizeof(int) * n_bodies);
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
    
    error = cudaBindTexture(0, pdata_mass_tex, pdata.mass, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;
    
    error = cudaBindTexture(0, net_force_tex, d_net_force, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error; 
        
    error = cudaBindTexture(0, net_virial_tex, d_net_virial, sizeof(float) * pdata.N);
    if (error != cudaSuccess)
        return error;
    
    if (nmax <= 32)
        {
        block_size = nmax; // maximum number of particles in a rigid body: each thread in a block takes care of a particle in a rigid body
        dim3 particle_grid(n_group_bodies, 1, 1);
        dim3 particle_threads(block_size, 1, 1);
        
        gpu_npt_rigid_step_one_particle_kernel<<< particle_grid, particle_threads >>>(pdata.pos, 
                                                                     pdata.vel, 
                                                                     pdata.image,
                                                                     d_net_virial,
                                                                     n_group_bodies,
                                                                     n_bodies, 
                                                                     local_beg,
                                                                     npt_rdata.new_box,
                                                                     deltaT);
        }
    else
        {
        block_size = 32; 
        dim3 particle_grid(n_group_bodies, 1, 1);
        dim3 particle_threads(block_size, 1, 1);
        
        gpu_npt_rigid_step_one_particle_sliding_kernel<<< particle_grid, particle_threads >>>(pdata.pos, 
                                                                     pdata.vel, 
                                                                     pdata.image,
                                                                     d_net_virial,
                                                                     n_group_bodies,
                                                                     n_bodies, 
                                                                     local_beg,
                                                                     nmax,
                                                                     block_size,
                                                                     npt_rdata.new_box,
                                                                     deltaT);
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


//! Takes the 2nd 1/2 step forward in the velocity-verlet NPT integration scheme
/*!  
    \param rdata_vel Body velocity
    \param rdata_angmom Angular momentum
    \param rdata_angvel Angular velocity
    \param n_group_bodies Number of rigid bodies in my group
    \param n_bodies Total number of rigid bodies
    \param local_beg Starting body index in this card
    \param npt_rdata_eta_dot_t0 Thermostat translational part 
    \param npt_rdata_eta_dot_r0 Thermostat rotational part
    \param npt_rdata_epsilon_dot Barostat velocity
    \param npt_rdata_conjqm Thermostat angular momentum
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
                                                          unsigned int n_group_bodies,
                                                          unsigned int n_bodies, 
                                                          unsigned int local_beg, 
                                                          float npt_rdata_eta_dot_t0, 
                                                          float npt_rdata_eta_dot_r0,
                                                          float npt_rdata_epsilon_dot, 
                                                          float4* npt_rdata_conjqm,
                                                          float* npt_rdata_partial_Ksum_t,
                                                          float* npt_rdata_partial_Ksum_r,
                                                          unsigned int npt_rdata_nf_t,
                                                          unsigned int npt_rdata_nf_r,
                                                          unsigned int npt_rdata_dimension,
                                                          gpu_boxsize box, 
                                                          float deltaT)
    {
    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x + local_beg;
    
    if (group_idx < n_group_bodies)
        {
        float body_mass;
        float4 moment_inertia, vel, ex_space, ey_space, ez_space, orientation, conjqm;
        float4 force, torque;
        float4 mbody, tbody, fquat;
        
        float dt_half = 0.5 * deltaT;
        float   onednft, onednfr, tmp, scale_t, scale_r, akin_t, akin_r;
        
        onednft = 1.0 + (float) (npt_rdata_dimension) / (float) (npt_rdata_nf_t);
        onednfr = (float) (npt_rdata_dimension) / (float) (npt_rdata_nf_r);

        tmp = -1.0 * dt_half * (npt_rdata_eta_dot_t0 + onednft * npt_rdata_epsilon_dot);
        scale_t = exp(tmp);
        tmp = -1.0 * dt_half * (npt_rdata_eta_dot_r0 + onednfr * npt_rdata_epsilon_dot);
        scale_r = exp(tmp);
        
        unsigned int idx_body = tex1Dfetch(rigid_data_body_indices_tex, group_idx);
        
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
            
            conjqm = tex1Dfetch(npt_rdata_conjqm_tex, idx_body);
            
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
            
            npt_rdata_conjqm[idx_body] = conjqm2;
            npt_rdata_partial_Ksum_t[idx_body] = akin_t;
            npt_rdata_partial_Ksum_r[idx_body] = akin_r;
            }
        }
    }

/*!
    \param pdata_vel Particle velocity
    \param d_virial_rigid Virial contribution from rigid bodies
    \param n_group_bodies Number of rigid bodies in my group
    \param n_bodies Total number of rigid bodies
    \param local_beg Starting body index in this card
    \param nmax Maximum number of particles in a rigid body
    \param box Box dimensions for periodic boundary condition handling
    \param deltaT Time step
*/
extern "C" __global__ void gpu_npt_rigid_step_two_particle_kernel(float4* pdata_vel,
                                                         float* d_net_virial,
                                                         unsigned int n_group_bodies,  
                                                         unsigned int n_bodies, 
                                                         unsigned int local_beg,
                                                         unsigned int nmax,
                                                         gpu_boxsize box,
                                                         float deltaT)
    {
    unsigned int group_idx = blockIdx.x + local_beg;
    
    __shared__ float4 vel, angvel, ex_space, ey_space, ez_space;
    
    float dt_half = 0.5 * deltaT;
    
    int idx_body = -1; 
    
    if (group_idx < n_group_bodies)
        {
        idx_body = tex1Dfetch(rigid_data_body_indices_tex, group_idx);
        if (idx_body < n_bodies && threadIdx.x == 0)
            {
            vel = tex1Dfetch(rigid_data_vel_tex, idx_body);
            angvel = tex1Dfetch(rigid_data_angvel_tex, idx_body);
            ex_space = tex1Dfetch(rigid_data_exspace_tex, idx_body);
            ey_space = tex1Dfetch(rigid_data_eyspace_tex, idx_body);
            ez_space = tex1Dfetch(rigid_data_ezspace_tex, idx_body);
            }
        }
        
    __syncthreads();
    
    if (idx_body >= 0 && idx_body < n_bodies)
        {
        unsigned int idx_particle = idx_body * blockDim.x + threadIdx.x;
        unsigned int idx_particle_index = tex1Dfetch(rigid_data_particle_indices_tex, idx_particle);
        if (idx_particle_index != INVALID_INDEX)
            {
            float4 particle_pos = tex1Dfetch(rigid_data_particle_pos_tex, idx_particle);
            float4 old_pos = tex1Dfetch(pdata_pos_tex, idx_particle_index);
            float4 old_vel = tex1Dfetch(pdata_vel_tex, idx_particle_index);
            float massone = tex1Dfetch(pdata_mass_tex, idx_particle_index);
            float4 pforce = tex1Dfetch(net_force_tex, idx_particle_index);
            float virial = tex1Dfetch(net_virial_tex, idx_particle_index);
            
            float4 ri;
            ri.x = ex_space.x * particle_pos.x + ey_space.x * particle_pos.y + ez_space.x * particle_pos.z;
            ri.y = ex_space.y * particle_pos.x + ey_space.y * particle_pos.y + ez_space.y * particle_pos.z;
            ri.z = ex_space.z * particle_pos.x + ey_space.z * particle_pos.y + ez_space.z * particle_pos.z;
            
            // v_particle = v_com + angvel x xr
            float4 pvel;
            pvel.x = vel.x + angvel.y * ri.z - angvel.z * ri.y;
            pvel.y = vel.y + angvel.z * ri.x - angvel.x * ri.z;
            pvel.z = vel.z + angvel.x * ri.y - angvel.y * ri.x;
            pvel.w = 0.0;
            
            float4 fc;
            fc.x = massone * (pvel.x - old_vel.x) / dt_half - pforce.x;
            fc.y = massone * (pvel.y - old_vel.y) / dt_half - pforce.y;
            fc.z = massone * (pvel.z - old_vel.z) / dt_half - pforce.z; 
            
            float pvirial = virial + (0.5 * (old_pos.x * fc.x + old_pos.y * fc.y + old_pos.z * fc.z) / 3.0);
                                
            // write out the results
            pdata_vel[idx_particle_index] = pvel;
            d_net_virial[idx_particle_index] = pvirial;
            }
        }
    }

/*!
    \param pdata_vel Particle velocity
    \param d_net_virial Particel virial
    \param n_group_bodies Number of rigid bodies in my group
    \param n_bodies Total number of rigid bodies
    \param local_beg Starting body index in this card
    \param nmax Maximum number of particles in a rigid body
    \param block_size Block size
    \param box Box dimensions for periodic boundary condition handling
    \param deltaT Time step
*/
extern "C" __global__ void gpu_npt_rigid_step_two_particle_sliding_kernel(float4* pdata_vel,
                                                         float* d_net_virial,
                                                         unsigned int n_group_bodies, 
                                                         unsigned int n_bodies, 
                                                         unsigned int local_beg,
                                                         unsigned int nmax,
                                                         unsigned int block_size,
                                                         gpu_boxsize box,
                                                         float deltaT)
    {
    unsigned int group_idx = blockIdx.x + local_beg;
    
    __shared__ float4 vel, angvel, ex_space, ey_space, ez_space;
    
    float dt_half = 0.5 * deltaT;
    
    int idx_body = -1;
    
    if (group_idx < n_group_bodies)
        {
        idx_body = tex1Dfetch(rigid_data_body_indices_tex, group_idx);
        if (idx_body < n_bodies && threadIdx.x == 0)
            {
            vel = tex1Dfetch(rigid_data_vel_tex, idx_body);
            angvel = tex1Dfetch(rigid_data_angvel_tex, idx_body);
            ex_space = tex1Dfetch(rigid_data_exspace_tex, idx_body);
            ey_space = tex1Dfetch(rigid_data_eyspace_tex, idx_body);
            ez_space = tex1Dfetch(rigid_data_ezspace_tex, idx_body);
            }
        }
        
    __syncthreads();
    
    unsigned int n_windows = nmax / block_size + 1;
    
    for (unsigned int start = 0; start < n_windows; start++)
        {
        if (idx_body >= 0 && idx_body < n_bodies)
            {
            int idx_particle = idx_body * nmax + start * block_size + threadIdx.x;
            if (idx_particle < nmax * n_bodies && start * block_size + threadIdx.x < nmax)
                {
                unsigned int idx_particle_index = tex1Dfetch(rigid_data_particle_indices_tex, idx_particle);
            
                if (idx_body < n_bodies && idx_particle_index != INVALID_INDEX)
                    {
                    float4 particle_pos = tex1Dfetch(rigid_data_particle_pos_tex, idx_particle);
                    float4 old_pos = tex1Dfetch(pdata_pos_tex, idx_particle_index);
                    float4 old_vel = tex1Dfetch(pdata_vel_tex, idx_particle_index);
                    float massone = tex1Dfetch(pdata_mass_tex, idx_particle_index);
                    float4 pforce = tex1Dfetch(net_force_tex, idx_particle_index);
                    float virial = tex1Dfetch(net_virial_tex, idx_particle_index);
                    
                    float4 ri;
                    ri.x = ex_space.x * particle_pos.x + ey_space.x * particle_pos.y + ez_space.x * particle_pos.z;
                    ri.y = ex_space.y * particle_pos.x + ey_space.y * particle_pos.y + ez_space.y * particle_pos.z;
                    ri.z = ex_space.z * particle_pos.x + ey_space.z * particle_pos.y + ez_space.z * particle_pos.z;
                    
                    // v_particle = v_com + angvel x xr
                    float4 pvel;
                    pvel.x = vel.x + angvel.y * ri.z - angvel.z * ri.y;
                    pvel.y = vel.y + angvel.z * ri.x - angvel.x * ri.z;
                    pvel.z = vel.z + angvel.x * ri.y - angvel.y * ri.x;
                    pvel.w = 0.0;
                    
                    float4 fc;
                    fc.x = massone * (pvel.x - old_vel.x) / dt_half - pforce.x;
                    fc.y = massone * (pvel.y - old_vel.y) / dt_half - pforce.y;
                    fc.z = massone * (pvel.z - old_vel.z) / dt_half - pforce.z; 
                    
                    float pvirial = virial + (0.5 * (old_pos.x * fc.x + old_pos.y * fc.y + old_pos.z * fc.z) / 3.0);
                                    
                    // write out the results
                    pdata_vel[idx_particle_index] = pvel;
                    d_net_virial[idx_particle_index] = pvirial;
                    }
                }
            }
        }

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
cudaError_t gpu_npt_rigid_step_two(const gpu_pdata_arrays &pdata, 
                                    const gpu_rigid_data_arrays& rigid_data, 
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
    unsigned int local_beg = rigid_data.local_beg;
    unsigned int nmax = rigid_data.nmax;
    
    // bind the textures for ALL rigid bodies
    cudaError_t error = cudaBindTexture(0, rigid_data_body_indices_tex, rigid_data.body_indices, sizeof(float) * n_group_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_body_mass_tex, rigid_data.body_mass, sizeof(float) * n_bodies);
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

    error = cudaBindTexture(0, npt_rdata_conjqm_tex, npt_rdata.conjqm, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
                                                                                                                                                                            
    unsigned int block_size = 64;
    unsigned int n_blocks = n_group_bodies / block_size + 1;                                
    dim3 body_grid(n_blocks, 1, 1);
    dim3 body_threads(block_size, 1, 1);                                                 
    gpu_npt_rigid_step_two_body_kernel<<< body_grid, body_threads >>>(rigid_data.vel, 
                                                                    rigid_data.angmom, 
                                                                    rigid_data.angvel,
                                                                    n_group_bodies,
                                                                    n_bodies, 
                                                                    local_beg,
                                                                    npt_rdata.eta_dot_t0, 
                                                                    npt_rdata.eta_dot_r0,
                                                                    npt_rdata.epsilon_dot, 
                                                                    npt_rdata.conjqm,
                                                                    npt_rdata.partial_Ksum_t,
                                                                    npt_rdata.partial_Ksum_r,
                                                                    npt_rdata.nf_t,
                                                                    npt_rdata.nf_r,
                                                                    npt_rdata.dimension, 
                                                                    box, 
                                                                    deltaT);

    // get the body information after the above update
    error = cudaBindTexture(0, rigid_data_vel_tex, rigid_data.vel, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_angvel_tex, rigid_data.angvel, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
    
    // bind the textures for particles
    
    error = cudaBindTexture(0, pdata_pos_tex, pdata.pos, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, pdata_vel_tex, pdata.vel, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;
    
    error = cudaBindTexture(0, pdata_mass_tex, pdata.mass, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;
    
    error = cudaBindTexture(0, net_force_tex, d_net_force, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error; 
        
    error = cudaBindTexture(0, net_virial_tex, d_net_virial, sizeof(float) * pdata.N);
    if (error != cudaSuccess)
        return error;
                                                                                                                                    
    if (nmax <= 32)
        {                                                                                                                                    
        block_size = nmax; // each thread in a block takes care of a particle in a rigid body
        dim3 particle_grid(n_group_bodies, 1, 1);
        dim3 particle_threads(block_size, 1, 1);                                                
        gpu_npt_rigid_step_two_particle_kernel<<< particle_grid, particle_threads >>>(pdata.vel,
                                                        d_net_virial,
                                                        n_group_bodies,
                                                        n_bodies, 
                                                        local_beg,
                                                        nmax, 
                                                        box,
                                                        deltaT);
        }
    else
        {
        block_size = 32; 
        dim3 particle_grid(n_group_bodies, 1, 1);
        dim3 particle_threads(block_size, 1, 1);                                                
        gpu_npt_rigid_step_two_particle_sliding_kernel<<< particle_grid, particle_threads >>>(pdata.vel,
                                                        d_net_virial,
                                                        n_group_bodies,
                                                        n_bodies, 
                                                        local_beg,
                                                        nmax,
                                                        block_size, 
                                                        box,
                                                        deltaT);
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

#pragma mark RIGID_PRESSUREL_KERNEL

//! Computes the first-pass virial and 2K sum for all particles
/*! \param d_partial_sum2K Stores one partial 2K sum per block run
    \param d_partial_sumW Stores one partial W sum per block run
    \param pdata Particle data to use when computing the sum
    \param d_net_virial The per particle net virial

    \a One thread is run per particle. That thread reads in the particles velocity and mass, then computes the
    value 2K and W. These values are then reduced in a sum within each block. Thus, the block size must be a power of
    two. The final reduction to a single value is performed in another kernel.
*/
extern "C" __global__ void gpu_npt_rigid_pressure_kernel2(float *d_partial_sum2K,
                              float *d_partial_sumW,
                              gpu_pdata_arrays pdata,
                              float *d_net_virial)
    {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // particle's index
    
    // *** First sum the virial
    float virial = 0.0f;
    if (idx < pdata.local_num)
        {
        virial = d_net_virial[idx];
        }
    
    npt_rigid_sdata[threadIdx.x] = virial;
    __syncthreads();
    
    // reduce the sum in parallel
    int offs = blockDim.x >> 1;
    while (offs > 0)
        {
        if (threadIdx.x < offs)
            npt_rigid_sdata[threadIdx.x] += npt_rigid_sdata[threadIdx.x + offs];
        offs >>= 1;
        __syncthreads();
        }
        
    // write out our partial sum
    if (threadIdx.x == 0)
        {
        d_partial_sumW[blockIdx.x] = npt_rigid_sdata[0];
        }

    // *** Then sum 2K
    float psq2 = 0.0f;
    if (idx < pdata.local_num)
        {
        float4 vel = pdata.vel[idx];
        float mass = pdata.mass[idx];
        psq2 = mass*(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z);
        }
    
    npt_rigid_sdata[threadIdx.x] = psq2;
    __syncthreads();
    
    // reduce the sum in parallel
    offs = blockDim.x >> 1;
    while (offs > 0)
        {
        if (threadIdx.x < offs)
            npt_rigid_sdata[threadIdx.x] += npt_rigid_sdata[threadIdx.x + offs];
        offs >>= 1;
        __syncthreads();
        }
        
    // write out our partial sum
    if (threadIdx.x == 0)
        {
        d_partial_sum2K[blockIdx.x] = npt_rigid_sdata[0];
        }
    }

/*! \param d_partial_sum2K Stores one partial 2K sum per block run
    \param d_partial_sumW Stores one partial W sum per block run
    \param pdata Particle data to use when computing the sum
    \param d_net_virial The per particle net virial
    \param block_size Size of the block to execute on the GPU
    \param num_blocks Number of blocks to execute on the GPU

    This is just a driver function for gpu_npt_pressure_kernel(). See it for more details.
*/
cudaError_t gpu_npt_rigid_pressure2(float *d_partial_sum2K,
                              float *d_partial_sumW,
                              gpu_pdata_arrays pdata,
                              float *d_net_virial,
                              unsigned int block_size,
                              unsigned int num_blocks)
    {
    // setup the grid to run the kernel
    dim3 grid(num_blocks, 1, 1);
    dim3 threads(block_size, 1, 1);
    
    // run the kernel
    gpu_npt_rigid_pressure_kernel2<<< grid, threads, block_size*sizeof(float) >>>(d_partial_sum2K,
                                                                           d_partial_sumW,
                                                                           pdata,
                                                                           d_net_virial);
    
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

#pragma mark RIGID_REDUCE_KINETIC_KERNEL
    
//! Makes the final 2K sum on the GPU
/*! \param d_sum2K Pointer to write the final sum to
    \param d_partial_sum2K Already computed partial sums
    \param num_blocks Number of partial sums in \a d_partial_sum2k

    nvt_step_one_kernel reduces the 2K sum per block. This kernel completes the task
    and makes the final 2K sum on the GPU. It is up to the host to read this value to do something with the final total

    This kernel is designed to be a really simple 1-block kernel for summing the total Ksum.
    
    \note \a num_blocks is the number of partial sums! Not the number of blocks executed in this kernel.
*/
extern "C" __global__ void gpu_npt_rigid_reduce_sum2K_kernel(float *d_sum2K, float *d_partial_sum2K, unsigned int num_blocks)
    {
    float sum2K = 0.0f;
    
    // sum up the values in the partial sum via a sliding window
    for (int start = 0; start < num_blocks; start += blockDim.x)
        {
        __syncthreads();
        if (start + threadIdx.x < num_blocks)
            npt_rigid_sdata[threadIdx.x] = d_partial_sum2K[start + threadIdx.x];
        else
            npt_rigid_sdata[threadIdx.x] = 0.0f;
        __syncthreads();
        
        // reduce the sum in parallel
        int offs = blockDim.x >> 1;
        while (offs > 0)
            {
            if (threadIdx.x < offs)
                npt_rigid_sdata[threadIdx.x] += npt_rigid_sdata[threadIdx.x + offs];
            offs >>= 1;
            __syncthreads();
            }
            
        // everybody sums up sum2K
        sum2K += npt_rigid_sdata[0];
        }
        
    if (threadIdx.x == 0)
        *d_sum2K = sum2K;
    }

/*! \param d_sum2K Pointer to write the final sum to
    \param d_partial_sum2K Already computed partial sums
    \param num_blocks Number of partial sums in \a d_partial_sum2k

    this is just a driver for gpu_nvt_reduce_sum2K_kernel() see it for details
*/
cudaError_t gpu_npt_rigid_reduce_sum2K(float *d_sum2K, float *d_partial_sum2K, unsigned int num_blocks)
    {
    // setup the grid to run the kernel
    int block_size = 256;
    dim3 grid( 1, 1, 1);
    dim3 threads(block_size, 1, 1);
    
    // run the kernel
    gpu_npt_rigid_reduce_sum2K_kernel<<< grid, threads, block_size*sizeof(float) >>>(d_sum2K, d_partial_sum2K, num_blocks);
    
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
