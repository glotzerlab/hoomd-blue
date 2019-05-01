// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file QuaternionMath.h
    \brief Quaternion math utility functions
*/

#ifndef __QUATERNION_MATH_H__
#define __QUATERNION_MATH_H__

#include "hoomd/HOOMDMath.h"

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

//! Normalize a quaternion
/*!
    \param q Quaternion to be normalized
*/
DEVICE inline void normalize(Scalar4 &q)
    {
    Scalar norm = fast::rsqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
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

    // apply permutation operator on p and q, get kp and kq
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
        kq.x = Scalar(0.0);  kp.x = Scalar(0.0);
        kq.y = Scalar(0.0);  kp.y = Scalar(0.0);
        kq.z = Scalar(0.0);  kp.z = Scalar(0.0);
        kq.w = Scalar(0.0);  kp.w = Scalar(0.0);
        }

    // obtain phi, cosines and sines

    phi = p.x * kq.x + p.y * kq.y + p.z * kq.z + p.w * kq.w;

    Scalar inertia_t;
    if (k == 1) inertia_t = inertia.x;
    else if (k == 2) inertia_t = inertia.y;
    else if (k == 3) inertia_t = inertia.z;
    else inertia_t = Scalar(0.0);
    if (fabs(inertia_t) < EPSILON) phi *= Scalar(0.0);
    else phi /= (Scalar(4.0) * inertia_t);

    c_phi = fast::cos(dt * phi);
    s_phi = fast::sin(dt * phi);

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
    ex_space.y = Scalar(2.0) * (quat.y * quat.z + quat.x * quat.w);
    ex_space.z = Scalar(2.0) * (quat.y * quat.w - quat.x * quat.z);

    // ey_space
    ey_space.x = Scalar(2.0) * (quat.y * quat.z - quat.x * quat.w);
    ey_space.y = quat.x * quat.x - quat.y * quat.y + quat.z * quat.z - quat.w * quat.w;
    ey_space.z = Scalar(2.0) * (quat.z * quat.w + quat.x * quat.y);

    // ez_space
    ez_space.x = Scalar(2.0) * (quat.y * quat.w + quat.x * quat.z);
    ez_space.y = Scalar(2.0) * (quat.z * quat.w - quat.x * quat.y);
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
    if (moment_inertia.x < EPSILON) angbody[0] = Scalar(0.0);
    else angbody[0] = (ex_space.x * angmom.x + ex_space.y * angmom.y
                           + ex_space.z * angmom.z) / moment_inertia.x;

    if (moment_inertia.y < EPSILON) angbody[1] = Scalar(0.0);
    else angbody[1] = (ey_space.x * angmom.x + ey_space.y * angmom.y
                           + ey_space.z * angmom.z) / moment_inertia.y;

    if (moment_inertia.z < EPSILON) angbody[2] = Scalar(0.0);
    else angbody[2] = (ez_space.x * angmom.x + ez_space.y * angmom.y
                           + ez_space.z * angmom.z) / moment_inertia.z;

    // Convert to angbody to the space frame: angvel = rotation_matrix * angbody
    angvel.x = angbody[0] * ex_space.x + angbody[1] * ey_space.x + angbody[2] * ez_space.x;
    angvel.y = angbody[0] * ex_space.y + angbody[1] * ey_space.y + angbody[2] * ez_space.y;
    angvel.z = angbody[0] * ex_space.z + angbody[1] * ey_space.z + angbody[2] * ez_space.z;
    }

//! Quaternion multiply: c = a * b where a = (0, a)
/*  \param a Vector
    \param b Quaternion
    \param c Returned quaternion
*/
DEVICE inline void vecquat(Scalar4 &a, Scalar4 &b, Scalar4 &c)
    {
    c.x = -(a.x * b.y + a.y * b.z + a.z * b.w);
    c.y =   b.x * a.x + a.y * b.w - a.z * b.z;
    c.z =   b.x * a.y + a.z * b.y - a.x * b.w;
    c.w =   b.x * a.z + a.x * b.z - a.y * b.y;
    }

//! Advance the quaternion using angular momentum and angular velocity
/*  \param angmom Angular momentum
    \param moment_inertia Moment of inertia
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
    Scalar dtq = Scalar(0.5) * deltaT;

    computeAngularVelocity(angmom, moment_inertia, ex_space, ey_space, ez_space, angvel);

    // Compute (w q)
    vecquat(angvel, quat, omegaq);

    // Full update q from dq/dt = 1/2 w q
    qfull.x = quat.x + dtq * omegaq.x;
    qfull.y = quat.y + dtq * omegaq.y;
    qfull.z = quat.z + dtq * omegaq.z;
    qfull.w = quat.w + dtq * omegaq.w;
    normalize(qfull);

    // 1st half update from dq/dt = 1/2 w q
    qhalf.x = quat.x + Scalar(0.5) * dtq * omegaq.x;
    qhalf.y = quat.y + Scalar(0.5) * dtq * omegaq.y;
    qhalf.z = quat.z + Scalar(0.5) * dtq * omegaq.z;
    qhalf.w = quat.w + Scalar(0.5) * dtq * omegaq.w;
    normalize(qhalf);

    // Update ex, ey, ez from qhalf = update A
    exyzFromQuaternion(qhalf, ex_space, ey_space, ez_space);

    // Compute angular velocity from new ex_space, ey_space and ex_space
    computeAngularVelocity(angmom, moment_inertia, ex_space, ey_space, ez_space, angvel);

    // Compute (w qhalf)
    vecquat(angvel, qhalf, omegaq);

    // 2nd half update from dq/dt = 1/2 w q
    qhalf.x += Scalar(0.5) * dtq * omegaq.x;
    qhalf.y += Scalar(0.5) * dtq * omegaq.y;
    qhalf.z += Scalar(0.5) * dtq * omegaq.z;
    qhalf.w += Scalar(0.5) * dtq * omegaq.w;
    normalize(qhalf);

    // Corrected Richardson update
    quat.x = Scalar(2.0) * qhalf.x - qfull.x;
    quat.y = Scalar(2.0) * qhalf.y - qfull.y;
    quat.z = Scalar(2.0) * qhalf.z - qfull.z;
    quat.w = Scalar(2.0) * qhalf.w - qfull.w;
    normalize(quat);

    exyzFromQuaternion(quat, ex_space, ey_space, ez_space);
    }

//! Quaternion multiply: c = a * b
/*! \param a Quaternion
    \param b A three component vector
    \param c Returned quaternion
*/
DEVICE inline void quatvec(Scalar4& a, Scalar4& b, Scalar4& c)
    {
    c.x = -a.y * b.x - a.z * b.y - a.w * b.z;
    c.y =  a.x * b.x - a.w * b.y + a.z * b.z;
    c.z =  a.w * b.x + a.x * b.y - a.y * b.z;
    c.w = -a.z * b.x + a.y * b.y + a.x * b.z;
    }

//! Inverse quaternion multiply: c = inv(a) * b
/*! \param a Quaternion
    \param b A three component vector
    \param c A three component vector
*/
DEVICE inline void invquatvec(Scalar4& a, Scalar4& b, Scalar4& c)
    {
    c.x = -a.y * b.x + a.x * b.y + a.w * b.z - a.z * b.w;
    c.y = -a.z * b.x - a.w * b.y + a.x * b.z + a.y * b.w;
    c.z = -a.w * b.x + a.z * b.y - a.y * b.z + a.x * b.w;
    }


//! Quaternion quaternion multiply: c = a * b
/*! \param a Quaternion
    \param b Quaternion
    \param c Quaternion
*/
DEVICE inline void quatquat(const Scalar4& a, const Scalar4& b, Scalar4& c)
{
  c.x = a.x*b.x - a.y*b.y - a.z*b.z - a.w*b.w;
  c.y = a.x*b.y + a.y*b.x + a.z*b.w - a.w*b.z;
  c.z = a.x*b.z - a.y*b.w + a.z*b.x + a.w*b.y;
  c.w = a.x*b.w + a.y*b.z - a.z*b.y + a.w*b.x;
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

//! Hermitian conjugate of quaternion
/*! \param a The quaternion to conjugate
    \param b The Hermitian conjugate of a
*/
DEVICE inline void quatconj(const Scalar4& a, Scalar4& b)
    {
    b.x = a.x;
    b.y = -a.y;
    b.z = -a.z;
    b.w = -a.w;
    }

//! Convert between quaternion and ZYX Euler angles. q is equivalent to R(z,psi)R(y,theta)R(x,phi)
/*! \param phi Rotation angle about intrinsic x axis
    \param theta Rotation angle about intrinsic y axis
    \param psi Rotation angle about intrinsic z axis
    \param q Output quaternion
*/
DEVICE inline void eulerToQuat(const Scalar phi,const Scalar theta, const Scalar psi, Scalar4& q)
    {
    Scalar cosphi_2 = fast::cos(0.5*phi);
    Scalar sinphi_2 = fast::sin(0.5*phi);
    Scalar costheta_2 = fast::cos(0.5*theta);
    Scalar sintheta_2 = fast::sin(0.5*theta);
    Scalar cospsi_2 = fast::cos(0.5*psi);
    Scalar sinpsi_2 = fast::sin(0.5*psi);
    q.x =  cosphi_2*costheta_2*cospsi_2 + sinphi_2*sintheta_2*sinpsi_2;
    q.y =  sinphi_2*costheta_2*cospsi_2 - cosphi_2*sintheta_2*sinpsi_2;
    q.z =  cosphi_2*sintheta_2*cospsi_2 + sinphi_2*costheta_2*sinpsi_2;
    q.w =  cosphi_2*costheta_2*sinpsi_2 - sinphi_2*sintheta_2*cospsi_2;
    normalize(q);
    }

//! Convert between quaternion and ZYX Euler angles. q is equivalent to R(z,psi)R(y,theta)R(x,phi)
/*!  \param q Input quaternion
     \param phi output rotation angle about intrinsic x axis
    \param theta Output rotation angle about intrinsic y axis
    \param psi Output rotation angle about intrinsic z axis
*/
DEVICE inline void quatToEuler(const Scalar4 q, Scalar& phi, Scalar& theta, Scalar& psi)
    {
    phi = atan2(q.x*q.y+q.z*q.w,Scalar(0.5)-q.x*q.x-q.y*q.y);
    theta = asin(Scalar(2.0)*(q.x*q.z-q.w*q.y));
    psi = atan2(q.x*q.w+q.y*q.z,Scalar(0.5)-q.y*q.y-q.z*q.z);
    }

//! Rotate a vector with a quaternion
/*! \param a Three-component vector to be rotated
    \param q Quaternion used to rotate vector a
    \param b Resulted three-component vector
*/
DEVICE inline void quatrot(const Scalar3& a, const Scalar4& q, Scalar3& b)
    {
    Scalar4 a4 = {0.0, a.x, a.y, a.z};
    Scalar4 qc;
    quatconj(q, qc);

    // b4 = q a4 qc
    Scalar4 tmp, b4;
    quatquat(q, a4, tmp);
    quatquat(tmp, qc, b4);

    // get the last three components of b4
    b.x = b4.y;
    b.y = b4.z;
    b.z = b4.w;
    }

//! Rotate a vector with three Euler angles: http://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
/*! \param a Three-component vector to be rotated
    \param phi (or gamma) in radian
    \param theta (or beta) in radian
    \param psi (or alpha) in radian
    \param b Resulted three-component vector
*/
DEVICE inline void eulerrot(const Scalar3& a,
                            const Scalar& phi,
                            const Scalar& theta,
                            const Scalar& psi,
                            Scalar3& b)
    {
    // rotation matrix R with the columns are ex, ey and ez
    Scalar3 ex, ey, ez;
    ex.x = fast::cos(theta) * fast::cos(psi);
    ex.y = fast::cos(theta) * fast::sin(psi);
    ex.z = Scalar(-1.0) * fast::sin(theta);

    ey.x = Scalar(-1.0) * fast::cos(phi) * fast::sin(psi) + fast::sin(phi) * fast::sin(theta) * fast::cos(psi);
    ey.y = fast::cos(theta) * fast::cos(psi) + fast::sin(phi) * fast::sin(theta) * fast::sin(psi);
    ey.z = fast::sin(phi) * fast::cos(theta);

    ez.x = fast::sin(phi) * fast::sin(psi) + fast::cos(phi) * fast::sin(theta) * fast::cos(psi);
    ez.y = Scalar(-1.0) * fast::sin(phi) * fast::cos(psi) + fast::cos(phi) * fast::sin(theta) * fast::sin(psi);
    ez.z = fast::cos(phi) * fast::cos(theta);

    // rotate b using the rotation matrix: b = R a
    b.x = ex.x * a.x + ey.x * a.y + ez.x * a.z;
    b.y = ex.y * a.x + ey.y * a.y + ez.y * a.z;
    b.z = ex.z * a.x + ey.z * a.y + ez.z * a.z;
    }

/*! \param q Quaternion describing the particles current orientation
    \param R Output rotation matrix from the lab to the particle frame
*/
DEVICE inline void quatToR(const Scalar4& q, Scalar* R)
    {
    Scalar q0_2 = q.x * q.x;
    Scalar q1_2 = q.y * q.y;
    Scalar q2_2 = q.z * q.z;
    Scalar q3_2 = q.w * q.w;
    Scalar two_q0q1 = Scalar(2.0) * q.x * q.y;
    Scalar two_q0q2 = Scalar(2.0) * q.x * q.z;
    Scalar two_q0q3 = Scalar(2.0) * q.x * q.w;
    Scalar two_q1q2 = Scalar(2.0) * q.y * q.z;
    Scalar two_q1q3 = Scalar(2.0) * q.y * q.w;
    Scalar two_q2q3 = Scalar(2.0) * q.z * q.w;

    R[0] = q0_2 + q1_2 - q2_2 -q3_2;
    R[1] = two_q1q2 - two_q0q3;
    R[2] = two_q0q2 + two_q1q3;

    R[3] = two_q1q2 + two_q0q3;
    R[4] = q0_2 - q1_2 + q2_2 - q3_2;
    R[5] = two_q2q3 - two_q0q1;

    R[6] = two_q1q3 - two_q0q2;
    R[7] = two_q0q1 + two_q2q3;
    R[8] = q0_2 - q1_2 - q2_2 + q3_2;
    }

#endif
