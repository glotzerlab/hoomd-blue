// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#ifndef __EVALUATOR_CONSTRAINT_H__
#define __EVALUATOR_CONSTRAINT_H__

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorConstraint.h
    \brief Defines basic evaluation methods common to all constraint force implementations
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

//! Class for evaluating constraint forces
/*! <b>General Overview</b>
    EvaluatorConstraint is a low level computation class for use on the CPU and GPU. It provides basic functionality
    needed by all constraint forces.
*/
class EvaluatorConstraint
    {
    public:
        //! Constructs the constraint evaluator
        /*! \param _X Current position at the time of the constraint force calculation
            \param V Current velocity at the time of the constraint force calculation
            \param F Current net force at the time of the constraint force calculation
            \param _m Mass of the particle
            \param _deltaT Step size delta t
        */
        DEVICE EvaluatorConstraint(Scalar3 _X, Scalar3 V, Scalar3 F, Scalar _m, Scalar _deltaT)
            : X(_X), m(_m), deltaT(_deltaT)
            {
            // perform step 2 of this velocity verlet update and step 1 of the next to get
            // U = X(t+2deltaT) given X = X(t+deltaT)
            Scalar minv = Scalar(1.0)/m;
            Scalar dtsqdivm = deltaT*deltaT * minv;
            U.x = X.x + V.x * deltaT + F.x * dtsqdivm;
            U.y = X.y + V.y * deltaT + F.y * dtsqdivm;
            U.z = X.z + V.z * deltaT + F.z * dtsqdivm;
            }

        //! Evaluate the unconstrained position update U
        /*! \returns The unconstrained position update U
        */
        DEVICE Scalar3 evalU()
            {
            return U;
            }

        //! Evaluate the additional constraint force
        /*! \param FC output parameter where the computed force is written
            \param virial array of six scalars the computed virial tensor is written
            \param C constrained position particle will be moved to at the next step
            \return Additional force \a F needed to satisfy the constraint
        */
        DEVICE void evalConstraintForce(Scalar3& FC, Scalar *virial, const Scalar3& C)
            {
            // subtract a constrained update from U and get that F = (C-U)*m/dt^2
            Scalar moverdtsq = m / (deltaT * deltaT);
            FC.x = (C.x - U.x) * moverdtsq;
            FC.y = (C.y - U.y) * moverdtsq;
            FC.z = (C.z - U.z) * moverdtsq;

            // compute virial
            virial[0] = FC.x * X.x;
            virial[1] = Scalar(1./2.)*(FC.y * X.x + FC.x * X.y);
            virial[2] = Scalar(1./2.)*(FC.z * X.x + FC.x * X.z);
            virial[3] = FC.y * X.y;
            virial[4] = Scalar(1./2.)*(FC.z * X.y + FC.y * X.z);
            virial[5] = FC.z * X.z;
            }

    protected:
        Scalar3 U;      //!< Unconstrained position update
        Scalar3 X;      //!< Current particle position
        Scalar m;       //!< Saved mass value
        Scalar deltaT;  //!< Saved delta T value

    };


#endif // __PAIR_EVALUATOR_LJ_H__
