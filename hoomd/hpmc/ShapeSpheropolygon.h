// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"
#include "hoomd/VectorMath.h"
#include "ShapeSphere.h"    //< For the base template of test_overlap
#include "ShapeConvexPolygon.h"

#ifndef __SHAPE_SPHEROPOLYGON_H__
#define __SHAPE_SPHEROPOLYGON_H__

/*! \file ShapeSpheropolygon.h
    \brief Defines the spheropolygon shape
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#include <iostream>
#endif

namespace hpmc
{

namespace detail
{

//! Support function for ShapeSpheropolygon
/*! SupportFuncSpheropolygon is a functor that computes the support function for ShapeSpheropolygon. For a given
    input vector in local coordinates, it finds the vertex most in that direction and then extends it out further
    by the sweep radius.

    \ingroup minkowski
*/
class SupportFuncSpheropolygon
    {
    public:
        //! Construct a support function for a spheropolygon
        /*! \param _verts Polygon vertices
        */
        DEVICE SupportFuncSpheropolygon(const poly2d_verts& _verts)
            : verts(_verts)
            {
            }

        //! Compute the support function
        /*! \param n Normal vector input (in the local frame)
            \returns Local coords of the point furthest in the direction of n
        */
        DEVICE vec2<OverlapReal> operator() (const vec2<OverlapReal>& n) const
            {
            // get the support function of the underlying convex polyhedron
            vec2<OverlapReal> max_poly = SupportFuncConvexPolygon(verts)(n);
            // add to that the support mapping of the sphere
            vec2<OverlapReal> max_sphere = (verts.sweep_radius * fast::rsqrt(dot(n,n))) * n;

            return max_poly + max_sphere;
            }

    private:
        const poly2d_verts& verts;      //!< Vertices of the polygon
    };

}; // end namespace detail

//! Spheropolygon shape template
/*! ShapeSpheropolygon represents a convex polygon swept out by a sphere. For simplicity, it uses the same poly2d_verts
    struct as ShapeConvexPolygon. ShapeSpheropolygon interprets two fields in that struct that ShapeConvexPolygon
    ignores. The first is sweep_radius which defines the radius of the sphere to sweep around the polygon. The 2nd
    is ignore. When two shapes are checked for overlap, if both of them have ignore set to true (non-zero) then
    there is assumed to be no collision. This is intended for use with the penetrable hard-sphere model for depletants,
    but could be useful in other cases.

    The parameter defining a polygon is a structure containing a list of N vertices. They are assumed to be listed
    in counter-clockwise order and centered on 0,0. In fact, it is **required** that the origin is inside the shape,
    and it is best if the origin is the center of mass.

    \ingroup shape
*/
struct ShapeSpheropolygon
    {
    //! Define the parameter type
    typedef detail::poly2d_verts param_type;

    //! Initialize a polygon
    DEVICE ShapeSpheropolygon(const quat<Scalar>& _orientation, const param_type& _params)
        : orientation(_orientation), verts(_params)
        {
        }

    //! Does this shape have an orientation
    DEVICE bool hasOrientation() const {
        if (verts.N > 1)
            {
            return true;
            }
        else
            {
            return false;
            }
        }

    //!Ignore flag for acceptance statistics
    DEVICE bool ignoreStatistics() const { return verts.ignore; }

    //! Get the circumsphere diameter
    DEVICE OverlapReal getCircumsphereDiameter() const
        {
        // return the precomputed diameter
        return verts.diameter;
        }

    //! Get the in-circle radius
    DEVICE OverlapReal getInsphereRadius() const
        {
        // not implemented
        return OverlapReal(0.0);
        }

    #ifndef NVCC
    std::string getShapeSpec() const
        {
        std::ostringstream shapedef;
        unsigned int nverts = verts.N;
        if (nverts == 1)
            {
            shapedef << "{\"type\": \"Sphere\", " << "\"diameter\": " << verts.diameter << "}";
            }
        else if (nverts == 2)
            {
            throw std::runtime_error("Shape definition not supported for 2-vertex spheropolygons");
            }
        else
            {
            shapedef << "{\"type\": \"Polygon\", \"rounding_radius\": " << verts.sweep_radius << ", \"vertices\": [";
            for (unsigned int i = 0; i < nverts-1; i++)
                {
                shapedef << "[" << verts.x[i] << ", " << verts.y[i] << "], ";
                }
            shapedef << "[" << verts.x[nverts-1] << ", " << verts.y[nverts-1] << "]]}";
            }
        return shapedef.str();
        }
    #endif

    //! Return the bounding box of the shape in world coordinates
    DEVICE detail::AABB getAABB(const vec3<Scalar>& pos) const
        {
        return detail::AABB(pos, verts.diameter/Scalar(2));
        }

    //! Returns true if this shape splits the overlap check over several threads of a warp using threadIdx.x
    HOSTDEVICE static bool isParallel() { return false; }

    quat<Scalar> orientation;    //!< Orientation of the polygon

    const detail::poly2d_verts& verts;     //!< Vertices
    };

//! Check if circumspheres overlap
/*! \param r_ab Vector defining the position of shape b relative to shape a (r_b - r_a)
    \param a first shape
    \param b second shape
    \returns true if the circumspheres of both shapes overlap

    \ingroup shape
*/
DEVICE inline bool check_circumsphere_overlap(const vec3<Scalar>& r_ab, const ShapeSpheropolygon& a,
    const ShapeSpheropolygon &b)
    {
    vec2<OverlapReal> dr(r_ab.x, r_ab.y);

    OverlapReal rsq = dot(dr,dr);
    OverlapReal DaDb = a.getCircumsphereDiameter() + b.getCircumsphereDiameter();
    return (rsq*OverlapReal(4.0) <= DaDb * DaDb);
    }

//! Convex polygon overlap test
/*! \param r_ab Vector defining the position of shape b relative to shape a (r_b - r_a)
    \param a first shape
    \param b second shape
    \param err in/out variable incremented when error conditions occur in the overlap test
    \returns true when *a* and *b* overlap, and false when they are disjoint

    \ingroup shape
*/
template <>
DEVICE inline bool test_overlap<ShapeSpheropolygon,ShapeSpheropolygon>(const vec3<Scalar>& r_ab,
                                                                       const ShapeSpheropolygon& a,
                                                                       const ShapeSpheropolygon& b,
                                                                       unsigned int& err)
    {
    vec2<OverlapReal> dr(r_ab.x, r_ab.y);

    /*return detail::gjke_2d(detail::SupportFuncSpheropolygon(a.verts),
                           detail::SupportFuncSpheropolygon(b.verts),
                           dr,
                           quat<OverlapReal>(a.orientation),
                           quat<OverlapReal>(b.orientation));*/

    return detail::xenocollide_2d(detail::SupportFuncSpheropolygon(a.verts),
                                  detail::SupportFuncSpheropolygon(b.verts),
                                  dr,
                                  quat<OverlapReal>(a.orientation),
                                  quat<OverlapReal>(b.orientation),
                                  err);
    }

}; // end namespace hpmc

#undef HOSTDEVICE
#undef DEVICE
#endif // __SHAPE_SPHEROPOLYGON_H__
