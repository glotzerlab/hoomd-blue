// Copyright (c) 2009-2020 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: pschoenh

/*! \file Hypersphere.h
    \brief Defines the Hypersphere class
*/

#pragma once

#include "HOOMDMath.h"
#include "VectorMath.h"

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#endif

//! Stores the radius of the hypersphere, on which particle coordinates are defined
/*! This class stores the hypersphere radius of the spherical coordinate system on which the simulation
    is carried out, in three or four embedding dimensions. It also provides some helper methods
    to project particles back into 3d.

    Coordinates are stored as a set of two unit quaternions (q_l and q_r), which transform a four vector, such as a particle
    director or a particle center position  v, like this:

        q'(v) = q_l*v*q_r,

    where * is the quaternion multiplication.

    The standard position is a purely imaginary quaternion, (R,0,0,0).

    On the hypersphere, improper transformations would require storing an extra parity bit and are currently not implemented.

    For more details, see Sinkovits, Barr and Luijten JCP 136, 144111 (2012).
 */

struct __attribute__((visibility("default"))) Hypersphere
    {
    public:
        //! Default constructor
        HOSTDEVICE Hypersphere()
            : R(1.0)
            { }

        /*! Define spherical boundary conditions
            \param R Radius of the (hyper-) sphere
         */
        HOSTDEVICE Hypersphere(Scalar _R)
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
               return Scalar(2.0*M_PI*M_PI*R*R*R);
            }

        /*! Convert a hyperspherical coordinate into a cartesian one

            \param q_l The first unit quaternion specifying particle position and orientation
            \param q_r The second unit quaternion specifying particle position and orientation
            \returns the projection as a 3-vector
         */
        template<class Real>
        HOSTDEVICE quat<Real> hypersphericalToCartesian(const quat<Real>& q_l, const quat<Real>& q_r) const
            {
            return q_l*quat<Real>(R,vec3<Real>(0,0,0))*q_r;
            }

    private:
        Scalar R;        //!< Hypersphere radius
    };
#undef HOSTDEVICE
