// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __MANIFOLD_CLASS_XY_PLANE_H__
#define __MANIFOLD_CLASS_XY_PLANE_H__

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"

/*! \file ManifoldXYPlane.h
    \brief Defines the manifold class for the XYPlane surface
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
//! Class for constructing the XYPlane surface
/*! <b>General Overview</b>

    ManifoldXYPlane is a low level computation class that computes the distance and normal vector to
   the xy surface.

    <b>XYPlane specifics</b>

    ManifoldXYPlane constructs the surface:
    shift = z

    These are the parameters:
    - \a shift = shift of the xy-plane in z-direction;

*/

class ManifoldXYPlane
    {
    public:
    //! Constructs the manifold class
    /*! \param _shift in z direction
     */
    DEVICE ManifoldXYPlane(const Scalar _shift) : shift(_shift) { }

    //! Evaluate implicit function
    /*! \param point Point at which surface is calculated

        \return result of the nodal function at input point
    */

    DEVICE Scalar implicitFunction(const Scalar3& point)
        {
        return point.z - shift;
        }

    //! Evaluate derivative of implicit function
    /*! \param point Point at surface is calculated

        \return normal of the XYPlane surface at input point
    */

    DEVICE Scalar3 derivative(const Scalar3& point)
        {
        return make_scalar3(0, 0, 1);
        }

    DEVICE bool fitsInsideBox(const BoxDim& box)
        {
        Scalar3 lo = box.getLo();
        Scalar3 hi = box.getHi();
        if (shift > hi.z || shift < lo.z)
            {
            return false; // XYPlane does not fit inside box
            }
        else
            {
            return true;
            }
        }

    Scalar getShift()
        {
        return shift;
        };

    static unsigned int dimension()
        {
        return 2;
        }

    protected:
    Scalar shift;
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __MANIFOLD_CLASS_XY_PLANE_H__
