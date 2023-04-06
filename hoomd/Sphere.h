// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: pschoenh

/*! \file Sphere.h
    \brief Defines the Sphere class
*/

#pragma once

#include "HOOMDMath.h"
#include "VectorMath.h"

// Don't include MPI when compiling with __HIPCC__ or an LLVM JIT build
#if defined(ENABLE_MPI) && !defined(__HIPCC__) && !defined(HOOMD_LLVMJIT_BUILD)
#include "HOOMDMPI.h"
#endif

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#endif

namespace hoomd
    {
//! Stores the radius of the sphere, on which particle coordinates are defined
/*! This class stores the sphere radius of the spherical coordinate system on which the simulation
    is carried out, in three embedding dimensions. It also provides some helper methods
    to project particles form spherical back into Euclidean coordinates.
    Coordinates are stored as a unit quaternions (q_pos and orientation), which transform a four vector, such as a particle
    director or a particle center position v, like this:
        q'(v) = q_pos*v*(q_pos^-1),
    where * is the quaternion multiplication.
    The standard position is a purely imaginary quaternion, (0,R,0,0).
 */

struct
#ifndef __HIPCC__
    __attribute__((visibility("default")))
#endif
    Sphere
    {
    public:
        //! Default constructor
        HOSTDEVICE Sphere()
            : R(1.0)
            { }

        /*! Define spherical boundary conditions
            \param R Radius of the (hyper-) sphere
         */
        HOSTDEVICE Sphere(Scalar _R)
            : R(_R) {}

        //! Get the hypersphere radius
        HOSTDEVICE Scalar getR() const
            {
            return R;
            }

        //! Set the hypersphere radius
        void setR(Scalar _R)
            {
            R = _R;
            }

        //! Return the simulation volume
        Scalar getVolume() const
            {
               return Scalar(4.0*M_PI*R*R);
            }

        /*! Convert a spherical coordinate into a cartesian one
            \param q_l The first unit quaternion specifying particle position and orientation
            \param q_r The second unit quaternion specifying particle position and orientation
            \returns the projection as a 3-vector
         */
        template<class Real>
        HOSTDEVICE vec3<Real> sphericalToCartesian(const quat<Real>& q_pos) const
            {
            return rotate(q_pos,vec3<Real>(R,0,0));

            }

    private:
        Scalar R;        //!< Sphere radius
    };

    } // end namespace hoomd
#undef HOSTDEVICE
