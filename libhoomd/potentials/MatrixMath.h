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
// Maintainer: baschult     

#ifndef __MATRIX_MATH_H__
#define __MATRIX_MATH_H__

#include "HOOMDMath.h"

// need to declare these class methods with __device__ qualifiers when building in nvcc
//! DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

// call different optimized sqrt functions on the host / device
//! RSQRT is rsqrtf when included in nvcc and 1.0 / sqrt(x) when included into the host compiler
#ifdef NVCC
#define _SQRT sqrtf
#else
#define _SQRT sqrt
#endif

#define INDEX3(i,j) i*3+j
//! Copy the elements of a matrix
DEVICE inline void matCopy3(Scalar*M , Scalar* N)
    {
    M[0]=N[0];
    M[1]=N[1];
    M[2]=N[2];
    M[3]=N[3];
    M[4]=N[4];
    M[5]=N[5];
    M[6]=N[6];
    M[7]=N[7];
    M[8]=N[8];
    }

//! Multiply 3x3 matrices N_ij = diag_ik * m_kj
DEVICE inline void matDiagMult3(const Scalar3& diag,const Scalar* M,Scalar* N)
    {
    N[INDEX3(0,0)]=diag.x*M[INDEX3(0,0)];
    N[INDEX3(0,1)]=diag.x*M[INDEX3(0,1)];
    N[INDEX3(0,2)]=diag.x*M[INDEX3(0,2)];
    N[INDEX3(1,0)]=diag.y*M[INDEX3(1,0)];
    N[INDEX3(1,1)]=diag.y*M[INDEX3(1,1)];
    N[INDEX3(1,2)]=diag.y*M[INDEX3(1,2)];
    N[INDEX3(2,0)]=diag.z*M[INDEX3(2,0)];
    N[INDEX3(2,1)]=diag.z*M[INDEX3(2,1)];
    N[INDEX3(2,2)]=diag.z*M[INDEX3(2,2)];
    }

//! Multiply 3x3 matrices A^t B = C... C_ij=A_ki B_kj
DEVICE inline void matTransMatMult3(const Scalar* A,const  Scalar* B, Scalar* C)
    {
    C[INDEX3(0,0)]=A[INDEX3(0,0)]*B[INDEX3(0,0)]+A[INDEX3(0,1)]*B[INDEX3(0,1)]+A[INDEX3(0,2)]*B[INDEX3(0,2)];
    C[INDEX3(0,1)]=A[INDEX3(0,0)]*B[INDEX3(1,0)]+A[INDEX3(0,1)]*B[INDEX3(1,1)]+A[INDEX3(0,2)]*B[INDEX3(1,2)];
    C[INDEX3(0,2)]=A[INDEX3(0,0)]*B[INDEX3(2,0)]+A[INDEX3(0,1)]*B[INDEX3(2,1)]+A[INDEX3(0,2)]*B[INDEX3(2,2)];
    C[INDEX3(1,0)]=A[INDEX3(1,0)]*B[INDEX3(0,0)]+A[INDEX3(1,1)]*B[INDEX3(0,1)]+A[INDEX3(1,2)]*B[INDEX3(0,2)];
    C[INDEX3(1,1)]=A[INDEX3(1,0)]*B[INDEX3(1,0)]+A[INDEX3(1,1)]*B[INDEX3(0,1)]+A[INDEX3(1,2)]*B[INDEX3(1,2)];
    C[INDEX3(1,2)]=A[INDEX3(1,0)]*B[INDEX3(2,0)]+A[INDEX3(1,1)]*B[INDEX3(0,1)]+A[INDEX3(1,2)]*B[INDEX3(2,2)];
    C[INDEX3(2,0)]=A[INDEX3(2,0)]*B[INDEX3(0,0)]+A[INDEX3(2,1)]*B[INDEX3(0,1)]+A[INDEX3(2,2)]*B[INDEX3(0,2)];
    C[INDEX3(2,1)]=A[INDEX3(2,0)]*B[INDEX3(1,0)]+A[INDEX3(2,1)]*B[INDEX3(1,1)]+A[INDEX3(2,2)]*B[INDEX3(1,2)];
    C[INDEX3(2,2)]=A[INDEX3(2,0)]*B[INDEX3(2,0)]+A[INDEX3(2,1)]*B[INDEX3(2,1)]+A[INDEX3(2,2)]*B[INDEX3(2,2)];
    }

DEVICE inline void matMatAdd3(const Scalar* A, const Scalar* B, Scalar* C)
    {
    C[INDEX3(0,0)]=A[INDEX3(0,0)]+B[INDEX3(0,0)];
    C[INDEX3(0,1)]=A[INDEX3(0,1)]+B[INDEX3(0,1)];
    C[INDEX3(0,2)]=A[INDEX3(0,2)]+B[INDEX3(0,2)];
    C[INDEX3(1,0)]=A[INDEX3(1,0)]+B[INDEX3(1,0)];
    C[INDEX3(1,1)]=A[INDEX3(1,1)]+B[INDEX3(1,1)];
    C[INDEX3(1,2)]=A[INDEX3(1,2)]+B[INDEX3(1,2)];
    C[INDEX3(2,0)]=A[INDEX3(2,0)]+B[INDEX3(2,0)];
    C[INDEX3(2,1)]=A[INDEX3(2,1)]+B[INDEX3(2,1)];
    C[INDEX3(2,2)]=A[INDEX3(2,2)]+B[INDEX3(2,2)];
    }

//! Compute x= M v
DEVICE inline void matVecMult3(const Scalar* M, const Scalar* v, Scalar* x)
    {
    x[0]=v[0]*M[INDEX3(0,0)]+v[1]*M[INDEX3(0,1)]+v[2]*M[INDEX3(0,2)];
    x[1]=v[0]*M[INDEX3(1,0)]+v[1]*M[INDEX3(1,1)]+v[2]*M[INDEX3(1,2)];
    x[2]=v[0]*M[INDEX3(2,0)]+v[1]*M[INDEX3(2,1)]+v[2]*M[INDEX3(2,2)];
    
    }

//! Compute x = (v^t M)
DEVICE inline void rowVecMatMult3(const Scalar* v, const Scalar* M, Scalar* x)
    {
    x[0]=v[0]*M[INDEX3(0,0)]+v[1]*M[INDEX3(1,0)]+v[2]*M[INDEX3(2,0)];
    x[1]=v[0]*M[INDEX3(0,1)]+v[1]*M[INDEX3(1,1)]+v[2]*M[INDEX3(2,1)];
    x[2]=v[0]*M[INDEX3(0,2)]+v[1]*M[INDEX3(1,2)]+v[2]*M[INDEX3(2,2)];
    }

//! Compute x= M v
DEVICE inline void matVecMult3(const Scalar* M, const Scalar3& v, Scalar* x)
    {
    x[0]=v.x*M[INDEX3(0,0)]+v.y*M[INDEX3(0,1)]+v.z*M[INDEX3(0,2)];
    x[1]=v.x*M[INDEX3(1,0)]+v.y*M[INDEX3(1,1)]+v.z*M[INDEX3(1,2)];
    x[2]=v.x*M[INDEX3(2,0)]+v.y*M[INDEX3(2,1)]+v.z*M[INDEX3(2,2)];
    
    }

//! Compute x = (v^t M)
DEVICE inline void rowVecMatMult3(const Scalar3& v, const Scalar* M, Scalar* x)
    {
    x[0]=v.x*M[INDEX3(0,0)]+v.y*M[INDEX3(1,0)]+v.z*M[INDEX3(2,0)];
    x[1]=v.x*M[INDEX3(0,1)]+v.y*M[INDEX3(1,1)]+v.z*M[INDEX3(2,1)];
    x[2]=v.x*M[INDEX3(0,2)]+v.y*M[INDEX3(1,2)]+v.z*M[INDEX3(2,2)];
    }


//! Compute determinant of 3x3 mat, ||M||
DEVICE inline void det3(const Scalar* M, Scalar& det)
    {
    det=M[INDEX3(0,0)]*(M[INDEX3(1,1)]*M[INDEX3(2,2)]-M[INDEX3(1,2)]*M[INDEX3(2,1)]) +
        M[INDEX3(0,1)]*(M[INDEX3(1,2)]*M[INDEX3(2,0)]-M[INDEX3(1,0)]*M[INDEX3(2,2)]) +
        M[INDEX3(0,2)]*(M[INDEX3(1,0)]*M[INDEX3(2,1)]-M[INDEX3(1,2)]*M[INDEX3(0,2)]);
    }
DEVICE inline void matInverse3(const Scalar* M, Scalar* M_inv)
    { 
    Scalar det;
    det3(M,det);

    M_inv[INDEX3(0,0)] = (M[INDEX3(1,1)]*M[INDEX3(2,2)]-M[INDEX3(1,2)]*M[INDEX3(2,1)]) / det;
    M_inv[INDEX3(0,1)] = -(M[INDEX3(0,1)]*M[INDEX3(2,2)]-M[INDEX3(0,2)]*M[INDEX3(2,1)]) / det;
    M_inv[INDEX3(0,2)] = (M[INDEX3(0,1)]*M[INDEX3(1,2)]-M[INDEX3(0,2)]*M[INDEX3(1,1)]) / det;
    M_inv[INDEX3(1,0)] = -(M[INDEX3(1,0)]*M[INDEX3(2,2)]-M[INDEX3(1,2)]*M[INDEX3(2,0)]) / det;
    M_inv[INDEX3(1,1)] = (M[INDEX3(0,0)]*M[INDEX3(2,2)]-M[INDEX3(0,2)]*M[INDEX3(2,0)]) / det;
    M_inv[INDEX3(1,2)] = -(M[INDEX3(0,0)]*M[INDEX3(1,2)]-M[INDEX3(0,2)]*M[INDEX3(1,0)]) / det;
    M_inv[INDEX3(2,0)] = (M[INDEX3(1,0)]*M[INDEX3(2,1)]-M[INDEX3(1,1)]*M[INDEX3(2,0)]) / det;
    M_inv[INDEX3(2,1)] = -(M[INDEX3(0,0)]*M[INDEX3(2,1)]-M[INDEX3(0,1)]*M[INDEX3(2,0)]) / det;
    M_inv[INDEX3(2,2)] = (M[INDEX3(0,0)]*M[INDEX3(1,1)]-M[INDEX3(0,1)]*M[INDEX3(1,0)]) / det;
    }

//!Compute x=a.b
DEVICE inline void dot3(const Scalar* a, const Scalar* b, Scalar& x)
    {
    x=a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
    }

//! Compute a X b = c;
DEVICE inline void cross3(const Scalar* a, const Scalar* b, Scalar* c)
    {
    c[0]=a[1]*b[2]-b[1]*a[2];
    c[1]=b[0]*a[2]-b[2]*a[0];
    c[2]=a[0]*b[1]-b[0]*a[1];
    }

//! Compute the norm of a 3-vector
DEVICE inline void norm3(const Scalar3& r,Scalar& norm)
    {
    norm=_SQRT(r.x*r.x+r.y*r.y+r.z*r.z);
    }

#endif
