/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
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

// $Id$
// $URL$
// Maintainer: joaander

#ifndef __QUATERNION_MATH_H__
#define __QUATERNION_MATH_H__

#include "HOOMDMath.h"

// need to declare these class methods with __device__ qualifiers when building in nvcc
//! DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

// call different optimized cos functions on the host / device
//! __COS is __cosf when included in nvcc and cos when included into the host compiler
#ifdef NVCC
#define __COS __cosf
#else
#define __COS cos
#endif

// call different optimized sin functions on the host / device
//! __SIN is __sinf when included in nvcc and sin when included into the host compiler
#ifdef NVCC
#define __SIN __sinf
#else
#define __SIN sin
#endif

// call different optimized sqrt functions on the host / device
//! RSQRT is rsqrtf when included in nvcc and 1.0 / sqrt(x) when included into the host compiler
#ifdef NVCC
#define RSQRT(x) rsqrtf( (x) )
#else
#define RSQRT(x) Scalar(1.0) / sqrt( (x) )
#endif

//! Normalize a quaternion
/*! 
    \param q Quaternion to be normalized
*/
DEVICE inline void normalize(Scalar4 &q)
    {
    Scalar norm = RSQRT(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    q.x *= norm;
    q.y *= norm;
    q.z *= norm;
    q.w *= norm;
    }

//! Apply evolution operators to quat, quat momentum (see ref. Miller)
/*!
    \param k Direction
    \param p Thermostat angular momentum conjqm
    \param q Quaternion
    \param inertia Moment of inertia
    \param dt Time step
*/
DEVICE inline void no_squish_rotate(unsigned int k, Scalar4& p, Scalar4& q, Scalar4& inertia, Scalar dt)
    {
    Scalar phi, c_phi, s_phi;
    Scalar4 kp, kq;
    
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
        kq.x = 0.0;  kp.x = 0.0;
        kq.y = 0.0;  kp.y = 0.0;
        kq.z = 0.0;  kp.z = 0.0;
        kq.w = 0.0;  kp.w = 0.0;
        }
            
    // obtain phi, cosines and sines
    
    phi = p.x * kq.x + p.y * kq.y + p.z * kq.z + p.w * kq.w;
    
    Scalar inertia_t;
    if (k == 1) inertia_t = inertia.x;
    else if (k == 2) inertia_t = inertia.y;
    else if (k == 3) inertia_t = inertia.z;
    else inertia_t = Scalar(0.0);
    if (fabs(inertia_t) < EPSILON) phi *= 0.0;
    else phi /= (4.0 * inertia_t);
    
    c_phi = __COS(dt * phi);
    s_phi = __SIN(dt * phi);
    
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
    
//! Compute orientation (ex_space, ey_space, ez_space) from quaternion- re-implement from RigidData for self-containing purposes
/*! \param quat Quaternion
    \param ex_space x-axis unit vector
    \param ey_space y-axis unit vector
    \param ez_space z-axis unit vector
*/
DEVICE inline void exyzFromQuaternion(Scalar4 &quat, Scalar4 &ex_space, Scalar4 &ey_space, Scalar4 &ez_space)
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

//! Compute angular velocity from angular momentum
/*!  Convert the angular momentum from world frame to body frame.
    Compute angular velocity in the body frame (angbody).
    Convert the angular velocity from body frame back to world frame.

    Rotation matrix is formed by arranging ex_space, ey_space and ez_space vectors into columns.
    In this code, rotation matrix is used to map a vector in a body frame into the space frame:
        x_space = rotation_matrix * x_body
    The reverse operation is to convert a vector in the space frame to a body frame:
        x_body = transpose(rotation matrix) * x_space
    
    \param angmom Angular momentum
    \param moment_inertia Moment of inertia
    \param ex_space x-axis unit vector
    \param ey_space y-axis unit vector
    \param ez_space z-axis unit vector
    \param angvel Returned angular velocity
*/
DEVICE inline void computeAngularVelocity(Scalar4& angmom,
                                          Scalar4& moment_inertia,
                                          Scalar4& ex_space,
                                          Scalar4& ey_space,
                                          Scalar4& ez_space,
                                          Scalar4& angvel)
    {
    // Angular velocity in the body frame
    Scalar angbody[3];
    
    // angbody = angmom_body / moment_inertia = transpose(rotation_matrix) * angmom / moment_inertia
    if (moment_inertia.x < EPSILON) angbody[0] = 0.0;
    else angbody[0] = (ex_space.x * angmom.x + ex_space.y * angmom.y
                           + ex_space.z * angmom.z) / moment_inertia.x;
                           
    if (moment_inertia.y < EPSILON) angbody[1] = 0.0;
    else angbody[1] = (ey_space.x * angmom.x + ey_space.y * angmom.y
                           + ey_space.z * angmom.z) / moment_inertia.y;
                           
    if (moment_inertia.z < EPSILON) angbody[2] = 0.0;
    else angbody[2] = (ez_space.x * angmom.x + ez_space.y * angmom.y
                           + ez_space.z * angmom.z) / moment_inertia.z;
                           
    // Convert to angbody to the space frame: angvel = rotation_matrix * angbody
    angvel.x = angbody[0] * ex_space.x + angbody[1] * ey_space.x + angbody[2] * ez_space.x;
    angvel.y = angbody[0] * ex_space.y + angbody[1] * ey_space.y + angbody[2] * ez_space.y;
    angvel.z = angbody[0] * ex_space.z + angbody[1] * ey_space.z + angbody[2] * ez_space.z;
    }

//! Quaternion multiply: c = a * b where a = (0, a)
/*  \param a Quaternion
    \param b Quaternion
    \param c Returned quaternion
*/
DEVICE inline void vec_multiply(Scalar4 &a, Scalar4 &b, Scalar4 &c)
    {
    c.x = -(a.x * b.y + a.y * b.z + a.z * b.w);
    c.y =   b.x * a.x + a.y * b.w - a.z * b.z;
    c.z =   b.x * a.y + a.z * b.y - a.x * b.w;
    c.w =   b.x * a.z + a.x * b.z - a.y * b.y;
    }

//! Advance the quaternion using angular momentum and angular velocity
/*  \param angmom Angular momentum
    \param moment_inerta Moment of inertia
    \param angvel Returned angular velocity
    \param ex_space x-axis unit vector
    \param ey_space y-axis unit vector
    \param ez_space z-axis unit vector
    \param deltaT delta T step size
    \param quat Returned quaternion
*/
DEVICE inline void advanceQuaternion(Scalar4& angmom,
                                     Scalar4 &moment_inertia,
                                     Scalar4 &angvel,
                                     Scalar4& ex_space,
                                     Scalar4& ey_space,
                                     Scalar4& ez_space,
                                     Scalar deltaT,
                                     Scalar4 &quat)
    {
    Scalar4 qhalf, qfull, omegaq;
    Scalar dtq = 0.5 * deltaT;
    
    computeAngularVelocity(angmom, moment_inertia, ex_space, ey_space, ez_space, angvel);
    
    // Compute (w q)
    vec_multiply(angvel, quat, omegaq);
    
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
    vec_multiply(angvel, qhalf, omegaq);
    
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
    
//! Quaternion multiply: c = a * b 
/*! \param a Quaternion
    \param b A three component vector
    \param c Returned quaternion
*/
DEVICE inline void quat_multiply(Scalar4& a, Scalar4& b, Scalar4& c)
    {
    c.x = -a.y * b.x - a.z * b.y - a.w * b.z;
    c.y =  a.x * b.x - a.w * b.y + a.z * b.z;
    c.z =  a.w * b.x + a.x * b.y - a.y * b.z;
    c.w = -a.z * b.x + a.y * b.y + a.x * b.z;
    }

//! Inverse quaternion multiply: c = inv(a) * b 
/*! \param a Quaternion
    \param b A four component vector
    \param c A three component vector
*/
DEVICE inline void inv_quat_multiply(Scalar4& a, Scalar4& b, Scalar4& c)
    {
    c.x = -a.y * b.x + a.x * b.y + a.w * b.z - a.z * b.w;
    c.y = -a.z * b.x - a.w * b.y + a.x * b.z + a.y * b.w;
    c.z = -a.w * b.x + a.z * b.y - a.y * b.z + a.x * b.w;
    }

//! Matrix dot: c = dot(A, b) 
/*! \param ax The first row of A
    \param ay The second row of A
    \param az The third row of A
    \param b A three component vector
    \param c A three component vector    
*/
DEVICE inline void matrix_dot(Scalar4& ax, Scalar4& ay, Scalar4& az, Scalar4& b, Scalar4& c)
    {
    c.x = ax.x * b.x + ax.y * b.y + ax.z * b.z;
    c.y = ay.x * b.x + ay.y * b.y + ay.z * b.z;
    c.z = az.x * b.x + az.y * b.y + az.z * b.z;
    }

//! Matrix transpose dot: c = dot(trans(A), b) 
/*! \param ax The first row of A
    \param ay The second row of A
    \param az The third row of A
    \param b A three component vector
    \param c A three component vector
*/
DEVICE inline void transpose_dot(Scalar4& ax, Scalar4& ay, Scalar4& az, Scalar4& b, Scalar4& c)
    {
    c.x = ax.x * b.x + ay.x * b.y + az.x * b.z;
    c.y = ax.y * b.x + ay.y * b.y + az.y * b.z;
    c.z = ax.z * b.x + ay.z * b.y + az.z * b.z;
    }

#endif
