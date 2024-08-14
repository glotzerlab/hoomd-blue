// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __MANIFOLD_CLASS_XY_PLANE_H__
#define __MANIFOLD_CLASS_XY_PLANE_H__

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"

/*! \file ManifoldCuboid.h
    \brief Defines the manifold class for the Cuboid surface
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
//! Class for constructing the Cuboid surface
/*! <b>General Overview</b>

    ManifoldCuboid is a low level computation class that computes the distance and normal vector to
   a cuboid surface.

    <b>Cuboid specifics</b>

    ManifoldCuboid constructs the surface:
    shift = z

    These are the parameters:
    - \a shift = shift of the xy-plane in z-direction;

*/

class ManifoldCuboid
    {
    public:
    //! Constructs the manifold class
    /*! \param _Px center position in x-direction
        \param _Py center position in y-direction
        \param _Pz center position in z-direction
        \param _ax side length in x-direction
        \param _ay side length in y-direction
        \param _az side length in z-direction
    */
    DEVICE ManifoldCuboid(const Scalar3 _a, const Scalar3 _P) 
	: Px(_P.x), Py(_P.y), Pz(_P.z), ax(_a.x/2), ay(_a.y/2), az(_a.z/2) 
    	{
       	}

    //! Evaluate implicit function
    /*! \param point Point at which surface is calculated

        \return result of the nodal function at input point
    */

    DEVICE Scalar implicitFunction(const Scalar3& point)
        {
	Scalar distance;
	Scalar distance_test;

	if (Px < point.x)
	   {
           distance = point.x - Px - ax;
	   }
	else
	   {
           distance = Px - ax - point.x;
	   }

	if (Py < point.y)
	   {
           distance_test = point.y - Py - ay;
	   }
	else
	   {
           distance_test = Py - ay - point.y;
	   }

	if(distance > 0)
	   {
	   if(distance > distance_test && distance_test > 0)
	      {
	      distance = distance_test;
	      }
	   }
	else
	   {
	   if(distance < distance_test)
	      {
	      distance = distance_test;
	      }
	   }

	if (Pz < point.z)
	   {
           distance_test = point.z - Pz - az;
	   }
	else
	   {
           distance_test = Pz - az - point.z;
	   }

	if(distance > 0)
	   {
	   if(distance > distance_test && distance_test > 0)
	      {
	      distance = distance_test;
	      }
	   }
	else
	   {
	   if(distance < distance_test)
	      {
	      distance = distance_test;
	      }
	   }
	
        return distance;
        }

    //! Evaluate derivative of implicit function
    /*! \param point Point at surface is calculated

        \return normal of the Cuboid surface at input point
    */

    DEVICE Scalar3 derivative(const Scalar3& point)
        {
	Scalar distance;
	Scalar distance_test;
	Scalar3 normal;

	if (Px < point.x)
	   {
           normal = make_scalar3(1,0,0);
           distance = point.x - Px - ax;
	   }
	else
	   {
           distance = Px - ax - point.x;
           normal = make_scalar3(-1,0,0);
	   }

	Scalar dir = 1;
	if (Py < point.y)
	   {
           distance_test = point.y - Py - ay;
	   }
	else
	   {
           distance_test = Py - ay - point.y;
	   dir = -1;
	   }

	if(distance > 0)
	   {
	   if(distance > distance_test && distance_test > 0)
	      {
	      distance = distance_test;
              normal = make_scalar3(0,dir,0);
	      }
	   }
	else
	   {
	   if(distance < distance_test)
	      {
	      distance = distance_test;
              normal = make_scalar3(0,dir,0);
	      }
	   }

	dir = 1;
	if (Pz < point.z)
	   {
           distance_test = point.z - Pz - az;
	   }
	else
	   {
           distance_test = Pz - az - point.z;
	   dir = -1;
	   }

	if(distance > 0)
	   {
	   if(distance > distance_test && distance_test > 0)
              normal = make_scalar3(0,0,dir);
	   }
	else
	   {
	   if(distance < distance_test)
              normal = make_scalar3(0,0,dir);
	   }
	
        return normal;
        }

    DEVICE bool fitsInsideBox(const BoxDim& box)
        {
        /*Scalar3 lo = box.getLo();
        Scalar3 hi = box.getHi();
        if (Px+ax > hi.x || Px-ax < lo.x || Py+ay > hi.y || Py-ay < lo.y || Pz+az > hi.z || Pz-az < lo.z )
            {
            return false; // Cuboid does not fit inside box
            }
        else
            {*/
            return true;
            //}
        }

#ifndef __HIPCC__
    pybind11::tuple getP()
        {
        return pybind11::make_tuple(Px, Py, Pz);
        }

    pybind11::tuple getA()
        {
        return pybind11::make_tuple(ax*2, ay*2, az*2);
        }
#endif

    static unsigned int dimension()
        {
        return 2;
        }

    protected:
    Scalar Px;
    Scalar Py;
    Scalar Pz;
    Scalar ax;
    Scalar ay;
    Scalar az;
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __MANIFOLD_CLASS_XY_PLANE_H__
