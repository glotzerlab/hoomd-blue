// Copyright (c) 2009-2020 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: pschoenhoefer

#ifndef __MANIFOLD_CLASS_PLANE_H__
#define __MANIFOLD_CLASS_PLANE_H__

#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

using namespace std;

/*! \file ManifoldClassPlane.h
    \brief Defines the manifold class for the Plane minimal surface
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef __HIPCC__
#define DEVICE __device__
#else
#define DEVICE
#endif

//! Class for constructing the Plane minimal surface
/*! <b>General Overview</b>

    ManifoldClassPlane is a low level computation class that computes the distance and normal vector to the xy surface.

    <b>Plane specifics</b>

    ManifoldClassPlane constructs the surface:
    shift = z

    These are the parameters:
    - \a shift = shift of the xy-plane in z-direction;

*/

class ManifoldClassPlane
    {
    public:
        //! Constructs the manifold class
        /*! \param _shift in z direction
        */
        DEVICE ManifoldClassPlane(const Scalar _shift)
            : shift(_shift)
            {
            }

        //! Evaluate implicit function
        /*! \param point Point at which surface is calculated

            \return result of the nodal function at input point
        */

        DEVICE Scalar implicit_function(const Scalar3 point)
        {
            return point.z - shift;	
        }

        //! Evaluate deriviative of implicit function
        /*! \param point Point at surface is calculated

            \return normal of the Plane surface at input point
        */

        DEVICE Scalar3 derivative(const Scalar3 point)
        {
            Scalar3 delta;
            delta.x = 0;
            delta.y = 0;	
            delta.z = 1;	
            return delta;
        }

        DEVICE bool validate(const BoxDim box);

        //! Get the name of this manifold
        /*! \returns The manifold name. Must be short and all lowercase, as this is the name manifolds will be logged as
            via analyze.log.
        */
        static std::string getName()
            {
            return std::string("Plane");
            }

    protected:
        Scalar shift;   
    };

//! Exports the Plane manifold class to python
void export_ManifoldClassPlane(pybind11::module& m);

#endif // __MANIFOLD_CLASS_PLANE_H__
