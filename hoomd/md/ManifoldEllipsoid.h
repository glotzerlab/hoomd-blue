// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __MANIFOLD_CLASS_ELLIPSOID_H__
#define __MANIFOLD_CLASS_ELLIPSOID_H__

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"

#ifndef __HIPCC__
#include <pybind11/pybind11.h>
#endif

/*! \file ManifoldEllipsoid.h
    \brief Defines the manifold class for the Ellipsoid surface
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host
// compiler
#ifdef __HIPCC__
#define DEVICE __device__
#else
#define DEVICE
#endif

namespace hoomd
    {
namespace md
    {
//! Class for constructing the Ellipsoid surface
/*! <b>General Overview</b>

    ManifoldEllipsoid is a low level computation class that computes the distance and normal vector
   to the Ellipsoid surface.

    <b>Ellipsoid specifics</b>

    ManifoldEllipsoid constructs the surface:
    1 = ((x-P_x)/a)^2 + ((y-P_y)/b)^2 + ((z-P_z)/c)^2

    These are the parameters:
    - \a P_x = center position of the ellipsoid in x-direction;
    - \a P_y = center position of the ellipsoid in y-direction;
    - \a P_z = center position of the ellipsoid in z-direction;
    - \a a = axis of the ellipsoid in x-direction;
    - \a b = axis of the ellipsoid in y-direction;
    - \a c = axis of the ellipsoid in z-direction;

*/

class ManifoldEllipsoid
    {
    public:
    //! Constructs the manifold class
    /*! \param _Px center position in x-direction
        \param _Py center position in y-direction
        \param _Pz center position in z-direction
        \param _a x-axis
        \param _b y-axis
        \param _c z-axis
    */
    DEVICE ManifoldEllipsoid(Scalar _a, Scalar _b, Scalar _c, Scalar3 _P)
        : Px(_P.x), Py(_P.y), Pz(_P.z), a(_a), b(_b), c(_c), inv_a2(Scalar(1.0) / (_a * _a)),
          inv_b2(Scalar(1.0) / (_b * _b)), inv_c2(Scalar(1.0) / (_c * _c))
        {
        }

    //! Evaluate implicit function
    /*! \param point Point at which surface is calculated

        \return result of the nodal function at input point
    */

    DEVICE Scalar implicitFunction(const Scalar3& point)
        {
        return inv_a2 * (point.x - Px) * (point.x - Px) + inv_b2 * (point.y - Py) * (point.y - Py)
               + inv_c2 * (point.z - Pz) * (point.z - Pz) - 1;
        }

    //! Evaluate derivative of implicit function
    /*! \param point Point at surface is calculated

        \return normal of the Ellipsoid surface at input point
    */

    DEVICE Scalar3 derivative(const Scalar3& point)
        {
        return make_scalar3(2 * inv_a2 * (point.x - Px),
                            2 * inv_b2 * (point.y - Py),
                            2 * inv_c2 * (point.z - Pz));
        }

    DEVICE bool fitsInsideBox(const BoxDim& box)
        {
        Scalar3 lo = box.getLo();
        Scalar3 hi = box.getHi();

        if (Px + a > hi.x || Px - a < lo.x || Py + b > hi.y || Py - b < lo.y || Pz + c > hi.z
            || Pz - c < lo.z)
            {
            return false; // Ellipsoid does not fit inside box
            }
        else
            {
            return true;
            }
        }

    Scalar getA()
        {
        return a;
        };

    Scalar getB()
        {
        return b;
        };

    Scalar getC()
        {
        return c;
        };

#ifndef __HIPCC__
    pybind11::tuple getP()
        {
        return pybind11::make_tuple(Px, Py, Pz);
        }
#endif

    static unsigned int dimension()
        {
        return 2;
        }

    protected:
    Scalar Px;
    Scalar Py;
    Scalar Pz;
    Scalar a;
    Scalar b;
    Scalar c;
    Scalar inv_a2;
    Scalar inv_b2;
    Scalar inv_c2;
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __MANIFOLD_CLASS_ELLIPSOID_H__
