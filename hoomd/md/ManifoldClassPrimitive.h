// Copyright (c) 2009-2020 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: pschoenhoefer

#ifndef __MANIFOLD_CLASS_PRIMITIVE_H__
#define __MANIFOLD_CLASS_PRIMITIVE_H__

#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

using namespace std;

/*! \file ManifoldClassPrimitive.h
    \brief Defines the manifold class for the Primitive minimal surface
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef __HIPCC__
#define DEVICE __device__
#else
#define DEVICE
#endif

//! Class for constructing the Primitive minimal surface
/*! <b>General Overview</b>

    ManifoldClassPrimitive is a low level computation class that computes the distance and normal vector to the Primitive surface.

    <b>Primitive specifics</b>

    ManifoldClassPrimitive constructs the surface:
    R^2 = (x-P_x)^2 + (y-P_y)^2 + (z-P_z)^2 

    These are the parameters:
    \a Nx The number of unitcells in x-direction
    \a Ny The number of unitcells in y-direction
    \a Nz The number of unitcells in z-direction
    \a epsilon Defines the specific constant mean curvture companion 

*/

class ManifoldClassPrimitive
    {
    public:
        //! Constructs the manifold class
         /* \param _Nx The number of unitcells in x-direction
            \param _Ny The number of unitcells in y-direction
            \param _Nz The number of unitcells in z-direction
            \param _epsilon Defines the specific constant mean curvture companion 
        */
        DEVICE ManifoldClassPrimitive(const int _Nx, const int _Ny, const int _Nz, const Scalar _epsilon)
            : Nx(_Nx), Ny(_Ny), Nz(_Nz), epsilon(_epsilon)
            {
            }

        //! Evaluate implicit function
        /*! \param point Point at which surface is calculated

            \return result of the nodal function at input point
        */

        DEVICE Scalar implicit_function(const Scalar3 point)
        {
            return  slow::cos(Lx*point.x) + slow::cos(Ly*point.y) + slow::cos(Lz*point.z) - epsilon;
        }

        //! Evaluate deriviative of implicit function
        /*! \param point Point at surface is calculated

            \return normal of the Primitive surface at input point
        */

        DEVICE Scalar3 derivative(const Scalar3 point)
        {
            Scalar3 delta;
            delta.x = -Lx*slow::sin(Lx*point.x);
            delta.y = -Ly*slow::sin(Ly*point.y);
            delta.z = -Lz*slow::sin(Lz*point.z);
            return delta;
        }

        DEVICE bool validate(const BoxDim box)
        {
         Scalar3 box_length = box.getHi() - box.getLo();

         Lx = 2*M_PI*Nx/box_length.x;
         Ly = 2*M_PI*Ny/box_length.y;
         Lz = 2*M_PI*Nz/box_length.z;
         
         return false;
        }

        //! Get the name of this manifold
        /*! \returns The manifold name. Must be short and all lowercase, as this is the name manifolds will be logged as
            via analyze.log.
        */
        static std::string getName()
            {
            return std::string("Primitive");
            }

    protected:
        int Nx;
        int Ny;
        int Nz;
        Scalar Lx;       
        Scalar Ly;       
        Scalar Lz;       
        Scalar epsilon;        
    };

//! Exports the Primitive manifold class to python
void export_ManifoldClassPrimitive(pybind11::module& m)
    {
    py::class_< ManifoldClassPrimitive, std::shared_ptr<ManifoldClassPrimitive> >(m, "ManifoldClassPrimitive")
    .def(py::init<int, int, int, Scalar >())
    .def("implicit_function", &ManifoldClassPrimitive::implicit_function)
    .def("derivative", &ManifoldClassPrimitive::derivative)
    ;
    }

#endif // __MANIFOLD_CLASS_PRIMITIVE_H__
