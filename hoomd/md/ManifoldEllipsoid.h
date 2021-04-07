// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: pschoenhoefer

#ifndef __MANIFOLD_CLASS_ELLIPSOID_H__
#define __MANIFOLD_CLASS_ELLIPSOID_H__

#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"
#include <pybind11/pybind11.h>

/*! \file ManifoldEllipsoid.h
    \brief Defines the manifold class for the Ellipsoid minimal surface
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef __HIPCC__
#define DEVICE __device__
#else
#define DEVICE
#endif

//! Class for constructing the Ellipsoid minimal surface
/*! <b>General Overview</b>

    ManifoldEllipsoid is a low level computation class that computes the distance and normal vector to the Ellipsoid surface.

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
            : Px(_P.x), Py(_P.y), Pz(_P.z), inv_a2(Scalar(1.0)/(_a*_a)), inv_b2(Scalar(1.0)/(_b*_b)), inv_c2(Scalar(1.0)/(_c*_c))
            {
            }

        //! Evaluate implicit function
        /*! \param point Point at which surface is calculated

            \return result of the nodal function at input point
        */

        DEVICE Scalar implicit_function(const Scalar3& point)
        {
            return inv_a2*(point.x - Px)*(point.x - Px) + inv_b2*(point.y - Py)*(point.y - Py) + inv_c2*(point.z - Pz)*(point.z - Pz) - 1;
        }

        //! Evaluate deriviative of implicit function
        /*! \param point Point at surface is calculated

            \return normal of the Ellipsoid surface at input point
        */

        DEVICE Scalar3 derivative(const Scalar3& point)
        {
            return make_scalar3(2*inv_a2*(point.x - Px), 2*inv_b2*(point.y - Py), 2*inv_c2*(point.z - Pz));
        }

        DEVICE bool adjust_to_box(const BoxDim& box)
        {
            Scalar3 lo = box.getLo();
            Scalar3 hi = box.getHi();
            Scalar ia = Scalar(1.0)/fast::sqrt(inv_a2);
            Scalar ib = Scalar(1.0)/fast::sqrt(inv_b2);
            Scalar ic = Scalar(1.0)/fast::sqrt(inv_c2);
            
            if (Px + ia > hi.x || Px - ia < lo.x ||
                Py + ib > hi.y || Py - ib < lo.y ||
                Pz + ic > hi.z || Pz - ic < lo.z)
                {
                return false; // Ellipsoid does not fit inside box
                }
                else 
                {
                return true;
                }
        }

        pybind11::dict getDict()
        {
            pybind11::dict v;
            v["a"] = sqrt(1.0/inv_a2);
            v["b"] = sqrt(1.0/inv_b2);
            v["c"] = sqrt(1.0/inv_c2);
            v["P"] = pybind11::make_tuple(Px, Py, Pz);
            return v;
        }

        static unsigned int dimension()
            {
            return 2;
            }

    protected:
        Scalar Px;
        Scalar Py;
        Scalar Pz;
        Scalar inv_a2;
        Scalar inv_b2;
        Scalar inv_c2;
    };

//! Exports the Ellipsoid manifold class to python
void export_ManifoldEllipsoid(pybind11::module& m);

#endif // __MANIFOLD_CLASS_ELLIPSOID_H__
