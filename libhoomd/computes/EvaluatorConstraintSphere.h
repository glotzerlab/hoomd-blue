/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
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

// Maintainer: joaander

#ifndef __EVALUATOR_CONSTRAINT_SPHERE_H__
#define __EVALUATOR_CONSTRAINT_SPHERE_H__

#include "HOOMDMath.h"

/*! \file EvaluatorConstraintSphere.h
    \brief Defines the constraint evaluator class for spheres
*/

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
#define RSQRT(x) rsqrtf( (x) )
#else
#define RSQRT(x) Scalar(1.0) / sqrt( (x) )
#endif

//! Class for evaluating sphere constraints
/*! <b>General Overview</b>
    EvaluatorConstraintSphere is a low level computation helper class to aid in evaluating particle constraints on a
    sphere. Given a sphere at a given position and radius, it will find the nearest point on the sphere to a given
    point.
*/
class EvaluatorConstraintSphere
    {
    public:
        //! Constructs the constraint evaluator
        /*! \param _P Position of the sphere
            \param _r   Radius of the sphere
        */
        DEVICE EvaluatorConstraintSphere(Scalar3 _P, Scalar _r)
            : P(_P), r(_r)
            {
            }
        
        //! Evaluate the closest point on the sphere
        /*! \param U unconstrained point
            
            \return Nearest point on the sphere
        */
        DEVICE Scalar3 evalClosest(const Scalar3& U)
            {
            // compute the vector pointing from P to V
            Scalar3 V;
            V.x = U.x - P.x;
            V.y = U.y - P.y;
            V.z = U.z - P.z;
            
            // compute 1/magnitude of V
            Scalar magVinv = RSQRT(V.x*V.x + V.y*V.y + V.z*V.z);
            
            // compute Vhat, the unit vector pointing in the direction of V
            Scalar3 Vhat;
            Vhat.x = magVinv * V.x;
            Vhat.y = magVinv * V.y;
            Vhat.z = magVinv * V.z;
            
            // compute resulting constrained point
            Scalar3 C;
#if(NVCC && __CUDA_ARCH__ < 200)
            C.x = P.x + __fmul_rn(Vhat.x,r);
            C.y = P.y + __fmul_rn(Vhat.y,r);
            C.z = P.z + __fmul_rn(Vhat.z,r);
#else
            C.x = P.x + Vhat.x * r;
            C.y = P.y + Vhat.y * r;
            C.z = P.z + Vhat.z * r;
#endif

            return C;
            }
        
    protected:
        Scalar3 P;      //!< Position of the sphere
        Scalar r;       //!< radius of the sphere
    };


#endif // __PAIR_EVALUATOR_LJ_H__

