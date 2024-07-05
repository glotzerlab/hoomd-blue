// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __MANIFOLD_CLASS_GYROID_H__
#define __MANIFOLD_CLASS_GYROID_H__

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"

#ifndef __HIPCC__
#include <pybind11/pybind11.h>
#endif

/*! \file ManifoldGyroid.h
    \brief Defines the manifold class for the Gyroid minimal surface
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
//! Class for constructing the Gyroid minimal surface
/*! <b>General Overview</b>

    ManifoldGyroid is a low level computation class that computes the distance and normal vector to
   the Gyroid surface.

    <b>Gyroid specifics</b>

    ManifoldGyroid constructs the surface:
    R^2 = (x-P_x)^2 + (y-P_y)^2 + (z-P_z)^2

    These are the parameters:
    \a Nx The number of unitcells in x-direction
    \a Ny The number of unitcells in y-direction
    \a Nz The number of unitcells in z-direction
    \a epsilon Defines the specific constant mean curvture companion

*/

class ManifoldGyroid
    {
    public:
    //! Constructs the manifold class
    /* \param _N vector determining the number of unitcells in x-, y-, and z-direction
       \param _epsilon Defines the specific constant mean curvture companion
   */
    DEVICE ManifoldGyroid(const int3 _N, const Scalar _epsilon)
        : Nx(_N.x), Ny(_N.y), Nz(_N.z), Lx(0), Ly(0), Lz(0), epsilon(_epsilon)
        {
        }

    //! Evaluate implicit function
    /*! \param point Point at which surface is calculated

        \return result of the nodal function at input point
    */

    DEVICE Scalar implicitFunction(const Scalar3& point)
        {
        Scalar cx, sx;
        fast::sincos(Lx * point.x, sx, cx);
        Scalar cy, sy;
        fast::sincos(Ly * point.y, sy, cy);
        Scalar cz, sz;
        fast::sincos(Lz * point.z, sz, cz);
        return sx * cy + sy * cz + sz * cx - epsilon;
        }

    //! Evaluate derivative of implicit function
    /*! \param point Point at surface is calculated

        \return normal of the Gyroid surface at input point
    */

    DEVICE Scalar3 derivative(const Scalar3& point)
        {
        Scalar cx, sx;
        fast::sincos(Lx * point.x, sx, cx);
        Scalar cy, sy;
        fast::sincos(Ly * point.y, sy, cy);
        Scalar cz, sz;
        fast::sincos(Lz * point.z, sz, cz);
        return make_scalar3(Lx * (cx * cy - sz * sx),
                            Ly * (cy * cz - sx * sy),
                            Lz * (cz * cx - sy * sz));
        }

    DEVICE bool fitsInsideBox(const BoxDim& box)
        {
        Scalar3 box_length = box.getHi() - box.getLo();

        Lx = 2 * M_PI * Nx / box_length.x;
        Ly = 2 * M_PI * Ny / box_length.y;
        Lz = 2 * M_PI * Nz / box_length.z;

        return true; // Gyroid surface is adjusted to box automatically and, therefore, is always
                     // accepted
        }

    static unsigned int dimension()
        {
        return 2;
        }

#ifndef __HIPCC__
    pybind11::tuple getN()
        {
        return pybind11::make_tuple(Nx, Ny, Nz);
        }
#endif

    Scalar getEpsilon()
        {
        return epsilon;
        };

    protected:
    int Nx;
    int Ny;
    int Nz;
    Scalar Lx;
    Scalar Ly;
    Scalar Lz;
    Scalar epsilon;
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __MANIFOLD_CLASS_GYROID_H__
