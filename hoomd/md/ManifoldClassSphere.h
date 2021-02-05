// Copyright (c) 2009-2020 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: pschoenhoefer

#ifndef __MANIFOLD_CLASS_SPHERE_H__
#define __MANIFOLD_CLASS_SPHERE_H__

#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

using namespace std;

/*! \file ManifoldClassSphere.h
    \brief Defines the manifold class for the Sphere minimal surface
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef __HIPCC__
#define DEVICE __device__
#else
#define DEVICE
#endif

//! Class for constructing the Sphere minimal surface
/*! <b>General Overview</b>

    ManifoldClassSphere is a low level computation class that computes the distance and normal vector to the Sphere surface.

    <b>Sphere specifics</b>

    ManifoldClassSphere constructs the surface:
    R^2 = (x-P_x)^2 + (y-P_y)^2 + (z-P_z)^2 

    These are the parameters:
    - \a P_x = center position of the sphere in x-direction;
    - \a P_y = center position of the sphere in y-direction;
    - \a P_z = center position of the sphere in z-direction;
    - \a R = radius of the sphere;

*/

class ManifoldClassSphere
    {
    public:
        //! Constructs the manifold class
        /*! \param _Px center position in x-direction
            \param _Py center position in y-direction
            \param _Pz center position in z-direction
            \param _R radius 
        */
        DEVICE ManifoldClassSphere(const Scalar _R, const Scalar3 _P)
            : Px(_P.x), Py(_P.y), Pz(_P.z), R(_R*_R)
            {
            }

        //! Evaluate implicit function
        /*! \param point Point at which surface is calculated

            \return result of the nodal function at input point
        */

        DEVICE Scalar implicit_function(const Scalar3 point)
        {
            return  (point.x - Px)*(point.x - Px) + (point.y - Py)*(point.y - Py) + (point.z - Pz)*(point.z - Pz) - R;	
        }

        //! Evaluate deriviative of implicit function
        /*! \param point Point at surface is calculated

            \return normal of the Sphere surface at input point
        */

        DEVICE Scalar3 derivative(const Scalar3 point)
        {
            Scalar3 delta;
            delta.x = 2*(point.x - Px);
            delta.y = 2*(point.y - Py);	
            delta.z = 2*(point.z - Pz);	
            return delta;
        }


        DEVICE bool validate(const BoxDim box);


        //! Get the name of this manifold
        /*! \returns The manifold name. Must be short and all lowercase, as this is the name manifolds will be logged as
            via analyze.log.
        */
        static std::string getName()
            {
            return std::string("Sphere");
            }

    protected:
        Scalar Px;       
        Scalar Py;       
        Scalar Pz;       
        Scalar R;        
    };

//! Exports the Sphere manifold class to python
void export_ManifoldClassSphere(pybind11::module& m);

#endif // __MANIFOLD_CLASS_SPHERE_H__
