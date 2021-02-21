// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: pschoenhoefer

#ifndef __MANIFOLD_CLASS_CYLINDER_H__
#define __MANIFOLD_CLASS_CYLINDER_H__

#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"
#include <pybind11/pybind11.h>

/*! \file ManifoldCylinder.h
    \brief Defines the manifold class for the Cylinder minimal surface
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef __HIPCC__
#define DEVICE __device__
#else
#define DEVICE
#endif

//! Class for constructing the Cylinder minimal surface
/*! <b>General Overview</b>

    ManifoldCylinder is a low level computation class that computes the distance and normal vector to the Cylinder surface.

    <b>Cylinder specifics</b>

    ManifoldCylinder constructs the surface:
    R^2 = (x-P_x)^2 + (y-P_y)^2

    These are the parameters:
    - \a P_x = center position of the cylinder in x-direction;
    - \a P_y = center position of the cylinder in y-direction;
    - \a P_z = center position of the cylinder in z-direction;
    - \a R = radius of the cylinder;

*/

class ManifoldCylinder
    {
    public:

        //! Constructs the manifold class
        /*! \param _Px center position in x-direction
            \param _Py center position in y-direction
            \param _Pz center position in z-direction
            \param _R radius
        */
        DEVICE ManifoldCylinder(const Scalar _R, const Scalar3 _P)
            : Px(_P.x), Py(_P.y), Pz(_P.z), R(_R*_R)
            {
            }

        //! Evaluate implicit function
        /*! \param point Point at which surface is calculated

            \return result of the nodal function at input point
        */

        DEVICE Scalar implicit_function(const Scalar3& point)
        {
            return  (point.x - Px)*(point.x - Px) + (point.y - Py)*(point.y - Py) - R;
        }

        //! Evaluate deriviative of implicit function
        /*! \param point Point at surface is calculated

            \return normal of the Cylinder surface at input point
        */

        DEVICE Scalar3 derivative(const Scalar3& point)
        {
            return make_scalar3(2*(point.x - Px), 2*(point.y - Py), 0);
        }

        DEVICE bool validate(const BoxDim& box)
        {
         Scalar3 lo = box.getLo();
         Scalar3 hi = box.getHi();
         Scalar sqR = fast::sqrt(R);
         if (Px + sqR > hi.x || Px - sqR < lo.x ||
             Py + sqR > hi.y || Py - sqR < lo.y ||
             Pz > hi.z || Pz < lo.z)
            {
            return true;
            }
            else
            { 
            return false;
            }
        }

        static unsigned int dimension()
            {
            return 2;
            }

    protected:
        Scalar Px;
        Scalar Py;
        Scalar Pz;
        Scalar R;
    };

//! Exports the Cylinder manifold class to python
void export_ManifoldCylinder(pybind11::module& m);

#endif // __MANIFOLD_CLASS_CYLINDER_H__
