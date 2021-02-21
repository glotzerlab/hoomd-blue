// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: pschoenhoefer

#ifndef __MANIFOLD_CLASS_DIAMOND_H__
#define __MANIFOLD_CLASS_DIAMOND_H__

#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"
#include <pybind11/pybind11.h>

//namespace py = pybind11;

/*! \file ManifoldDiamond.h
    \brief Defines the manifold class for the Diamond minimal surface
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef __HIPCC__
#define DEVICE __device__
#else
#define DEVICE
#endif

//! Class for constructing the Diamond minimal surface
/*! <b>General Overview</b>

    ManifoldDiamond is a low level computation class that computes the distance and normal vector to the Diamond surface.

    <b>Diamond specifics</b>

    These are the parameters:
    \a Nx The number of unitcells in x-direction
    \a Ny The number of unitcells in y-direction
    \a Nz The number of unitcells in z-direction
    \a epsilon Defines the specific constant mean curvture companion

*/

class ManifoldDiamond
    {
    public:
        //! Constructs the manifold class
         /* \param _Nx The number of unitcells in x-direction
            \param _Ny The number of unitcells in y-direction
            \param _Nz The number of unitcells in z-direction
            \param _epsilon Defines the specific constant mean curvture companion
        */
        DEVICE ManifoldDiamond(const int _Nx, const int _Ny, const int _Nz, const Scalar _epsilon)
            : Nx(_Nx), Ny(_Ny), Nz(_Nz), Lx(0), Ly(0), Lz(0), epsilon(_epsilon)
            {
            }

        //! Evaluate implicit function
        /*! \param point Point at which surface is calculated

            \return result of the nodal function at input point
        */

        DEVICE Scalar implicit_function(const Scalar3& point)
        {
            return fast::cos(Lx*point.x)*fast::cos(Ly*point.y)*fast::cos(Lz*point.z) - fast::sin(Lx*point.x)*fast::sin(Ly*point.y)*fast::sin(Lz*point.z) - epsilon;
        }

        //! Evaluate deriviative of implicit function
        /*! \param point Point at surface is calculated

            \return normal of the Diamond surface at input point
        */

        DEVICE Scalar3 derivative(const Scalar3& point)
        {

            Scalar cx,sx;
            fast::sincos(Lx*point.x,sx,cx);
            Scalar cy,sy;
            fast::sincos(Ly*point.y,sy,cy);
            Scalar cz,sz;
            fast::sincos(Lz*point.z,sz,cz);

            return make_scalar3(-Lx*(sx*cy*cz + cx*sy*sz),-Ly*(cx*sy*cz + sx*cy*sz),-Lz*(cx*cy*sz + sx*sy*cz));
        }

        DEVICE bool validate(const BoxDim& box)
        {
            Scalar3 box_length = box.getHi() - box.getLo();
        
            Lx = M_PI*Nx/box_length.x;
            Ly = M_PI*Ny/box_length.y;
            Lz = M_PI*Nz/box_length.z;
        
            return false;
        }

        //! Get the name of this manifold
        /*! \returns The manifold name. Must be short and all lowercase, as this is the name manifolds will be logged as
            via analyze.log.
        */
        static std::string getName()
            {
            return std::string("Diamond");
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

//! Exports the Diamond manifold class to python
void export_ManifoldDiamond(pybind11::module& m);

#endif // __MANIFOLD_CLASS_DIAMOND_H__
