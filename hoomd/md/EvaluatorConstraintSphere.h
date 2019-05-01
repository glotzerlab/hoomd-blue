// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#ifndef __EVALUATOR_CONSTRAINT_SPHERE_H__
#define __EVALUATOR_CONSTRAINT_SPHERE_H__

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorConstraintSphere.h
    \brief Defines the constraint evaluator class for spheres
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
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
            Scalar magVinv = fast::rsqrt(V.x*V.x + V.y*V.y + V.z*V.z);

            // compute Vhat, the unit vector pointing in the direction of V
            Scalar3 Vhat;
            Vhat.x = magVinv * V.x;
            Vhat.y = magVinv * V.y;
            Vhat.z = magVinv * V.z;

            // compute resulting constrained point
            Scalar3 C;
            C.x = P.x + Vhat.x * r;
            C.y = P.y + Vhat.y * r;
            C.z = P.z + Vhat.z * r;

            return C;
            }

    protected:
        Scalar3 P;      //!< Position of the sphere
        Scalar r;       //!< radius of the sphere
    };


#endif // __PAIR_EVALUATOR_LJ_H__
