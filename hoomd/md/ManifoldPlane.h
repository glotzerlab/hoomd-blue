// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: pschoenhoefer

#ifndef __MANIFOLD_CLASS_PLANE_H__
#define __MANIFOLD_CLASS_PLANE_H__

#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"
#include <pybind11/pybind11.h>

/*! \file ManifoldPlane.h
    \brief Defines the manifold class for the Plane surface
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef __HIPCC__
#define DEVICE __device__
#else
#define DEVICE
#endif

//! Class for constructing the Plane surface
/*! <b>General Overview</b>

    ManifoldPlane is a low level computation class that computes the distance and normal vector to the xy surface.

    <b>Plane specifics</b>

    ManifoldPlane constructs the surface:
    shift = z

    These are the parameters:
    - \a shift = shift of the xy-plane in z-direction;

*/

class ManifoldPlane
    {
    public:
        //! Constructs the manifold class
        /*! \param _shift in z direction
        */
        DEVICE ManifoldPlane(const Scalar _shift)
            : shift(_shift)
            {
            }

        //! Evaluate implicit function
        /*! \param point Point at which surface is calculated

            \return result of the nodal function at input point
        */

        DEVICE Scalar implicit_function(const Scalar3& point)
        {
            return point.z - shift;
        }

        //! Evaluate deriviative of implicit function
        /*! \param point Point at surface is calculated

            \return normal of the Plane surface at input point
        */

        DEVICE Scalar3 derivative(const Scalar3& point)
        {
            return make_scalar3(0,0,1);
        }

        DEVICE bool check_fit_to_box(const BoxDim& box)
        {
        Scalar3 lo = box.getLo();
        Scalar3 hi = box.getHi();
        if (shift > hi.z || shift < lo.z)
            {
            return false; // Plane does not fit inside box
            }
            else
            {
            return true;
            }
        }

        pybind11::dict getDict()
        {
            pybind11::dict v;
            v["shift"] = shift;
            return v;
        }

        Scalar getShift(){ return shift;};

        static unsigned int dimension()
            {
            return 2;
            }

    protected:
        Scalar shift;
    };

//! Exports the Plane manifold class to python
void export_ManifoldPlane(pybind11::module& m);

#endif // __MANIFOLD_CLASS_PLANE_H__
