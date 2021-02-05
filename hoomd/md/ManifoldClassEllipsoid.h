// Copyright (c) 2009-2020 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: pschoenhoefer

#ifndef __MANIFOLD_CLASS_ELLIPSOID_H__
#define __MANIFOLD_CLASS_ELLIPSOID_H__

#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

using namespace std;

/*! \file ManifoldClassEllipsoid.h
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

    ManifoldClassEllipsoid is a low level computation class that computes the distance and normal vector to the Ellipsoid surface.

    <b>Ellipsoid specifics</b>

    ManifoldClassEllipsoid constructs the surface:
    1 = ((x-P_x)/a)^2 + ((y-P_y)/b)^2 + ((z-P_z)/c)^2 

    These are the parameters:
    - \a P_x = center position of the ellipsoid in x-direction;
    - \a P_y = center position of the ellipsoid in y-direction;
    - \a P_z = center position of the ellipsoid in z-direction;
    - \a a = axis of the ellipsoid in x-direction;
    - \a b = axis of the ellipsoid in y-direction;
    - \a c = axis of the ellipsoid in z-direction;

*/

class ManifoldClassEllipsoid
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
        DEVICE ManifoldClassEllipsoid(Scalar _a, Scalar _b, Scalar _c, Scalar3 _P)
            : Px(_P.x), Py(_P.y), Pz(_P.z), a(Scalar(1.0)/(_a*_a)), b(Scalar(1.0)/(_b*_b)), c(Scalar(1.0)/(_c*_c))
            {
            }

        //! Evaluate implicit function
        /*! \param point Point at which surface is calculated

            \return result of the nodal function at input point
        */

        DEVICE Scalar implicit_function(Scalar3 point)
        {
            return  a*(point.x - Px)*(point.x - Px) + b*(point.y - Py)*(point.y - Py) + c*(point.z - Pz)*(point.z - Pz) - 1;	
        }

        //! Evaluate deriviative of implicit function
        /*! \param point Point at surface is calculated

            \return normal of the Ellipsoid surface at input point
        */

        DEVICE Scalar3 derivative(Scalar3 point)
        {
            Scalar3 delta;
            delta.x = 2*a*(point.x - Px);
            delta.y = 2*b*(point.y - Py);	
            delta.z = 2*c*(point.z - Pz);	
            return delta;
        }

        DEVICE bool validate(const BoxDim box);
	//
        //! Get the name of this manifold
        /*! \returns The manifold name. Must be short and all lowercase, as this is the name manifolds will be logged as
            via analyze.log.
        */
        static std::string getName()
            {
            return std::string("Ellipsoid");
            }

    protected:
        Scalar Px;       
        Scalar Py;       
        Scalar Pz;       
        Scalar a;        
        Scalar b;        
        Scalar c;        
    };

//! Exports the Ellipsoid manifold class to python
void export_ManifoldClassEllipsoid(pybind11::module& m);

#endif // __MANIFOLD_CLASS_ELLIPSOID_H__
