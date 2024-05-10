// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ShapeConvexPolygon.h"
#include "ShapeSphere.h" //< For the base template of test_overlap
#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"

#ifndef __SHAPE_SPHEROPOLYGON_H__
#define __SHAPE_SPHEROPOLYGON_H__

/*! \file ShapeSpheropolygon.h
    \brief Defines the spheropolygon shape
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __device__ when included in nvcc and blank when included into the host compiler
#ifdef __HIPCC__
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#include <iostream>
#endif

namespace hoomd
    {
namespace hpmc
    {
namespace detail
    {
//! Support function for ShapeSpheropolygon
/*! SupportFuncSpheropolygon is a functor that computes the support function for ShapeSpheropolygon.
   For a given input vector in local coordinates, it finds the vertex most in that direction and
   then extends it out further by the sweep radius.

    \ingroup minkowski
*/
class SupportFuncSpheropolygon
    {
    public:
    //! Construct a support function for a spheropolygon
    /*! \param _verts Polygon vertices
     */
    DEVICE SupportFuncSpheropolygon(const PolygonVertices& _verts) : verts(_verts) { }

    //! Compute the support function
    /*! \param n Normal vector input (in the local frame)
        \returns Local coords of the point furthest in the direction of n
    */
    DEVICE vec2<ShortReal> operator()(const vec2<ShortReal>& n) const
        {
        // get the support function of the underlying convex polyhedron
        vec2<ShortReal> max_poly = SupportFuncConvexPolygon(verts)(n);
        // add to that the support mapping of the sphere
        vec2<ShortReal> max_sphere = (verts.sweep_radius * fast::rsqrt(dot(n, n))) * n;

        return max_poly + max_sphere;
        }

    private:
    const PolygonVertices& verts; //!< Vertices of the polygon
    };

    }; // end namespace detail

//! Spheropolygon shape template
/*! ShapeSpheropolygon represents a convex polygon swept out by a sphere. For simplicity, it uses
   the same PolygonVertices struct as ShapeConvexPolygon. ShapeSpheropolygon interprets two fields
   in that struct that ShapeConvexPolygon ignores. The first is sweep_radius which defines the
   radius of the sphere to sweep around the polygon. The 2nd is ignore. When two shapes are checked
   for overlap, if both of them have ignore set to true (non-zero) then there is assumed to be no
   collision. This is intended for use with the penetrable hard-sphere model for depletants, but
   could be useful in other cases.

    The parameter defining a polygon is a structure containing a list of N vertices. They are
   assumed to be listed in counter-clockwise order and centered on 0,0. In fact, it is **required**
   that the origin is inside the shape, and it is best if the origin is the center of mass.

    \ingroup shape
*/
struct ShapeSpheropolygon
    {
    //! Define the parameter type
    typedef detail::PolygonVertices param_type;

    //! Temporary storage for depletant insertion
    typedef struct
        {
        } depletion_storage_type;

    //! Initialize a polygon
    DEVICE ShapeSpheropolygon(const quat<Scalar>& _orientation, const param_type& _params)
        : orientation(_orientation), verts(_params)
        {
        }

    //! Does this shape have an orientation
    DEVICE bool hasOrientation() const
        {
        if (verts.N > 1)
            {
            return true;
            }
        else
            {
            return false;
            }
        }

    //! Ignore flag for acceptance statistics
    DEVICE bool ignoreStatistics() const
        {
        return verts.ignore;
        }

    //! Get the circumsphere diameter
    DEVICE ShortReal getCircumsphereDiameter() const
        {
        // return the precomputed diameter
        return verts.diameter;
        }

    //! Get the in-circle radius
    DEVICE ShortReal getInsphereRadius() const
        {
        // not implemented
        return ShortReal(0.0);
        }

    //! Return the bounding box of the shape in world coordinates
    DEVICE hoomd::detail::AABB getAABB(const vec3<Scalar>& pos) const
        {
        return hoomd::detail::AABB(pos, verts.diameter / Scalar(2));
        }

    //! Return a tight fitting OBB
    DEVICE detail::OBB getOBB(const vec3<Scalar>& pos) const
        {
        // just use the AABB for now
        return detail::OBB(getAABB(pos));
        }

    //! Returns true if this shape splits the overlap check over several threads of a warp using
    //! threadIdx.x
    HOSTDEVICE static bool isParallel()
        {
        return false;
        }

    //! Retrns true if the overlap check supports sweeping both shapes by a sphere of given radius
    HOSTDEVICE static bool supportsSweepRadius()
        {
        return false;
        }

    quat<Scalar> orientation; //!< Orientation of the polygon

    const detail::PolygonVertices& verts; //!< Vertices
    };

//! Convex polygon overlap test
/*! \param r_ab Vector defining the position of shape b relative to shape a (r_b - r_a)
    \param a first shape
    \param b second shape
    \param err in/out variable incremented when error conditions occur in the overlap test
    \returns true when *a* and *b* overlap, and false when they are disjoint

    \ingroup shape
*/
template<>
DEVICE inline bool test_overlap<ShapeSpheropolygon, ShapeSpheropolygon>(const vec3<Scalar>& r_ab,
                                                                        const ShapeSpheropolygon& a,
                                                                        const ShapeSpheropolygon& b,
                                                                        unsigned int& err)
    {
    vec2<ShortReal> dr(ShortReal(r_ab.x), ShortReal(r_ab.y));

    /*return detail::gjke_2d(detail::SupportFuncSpheropolygon(a.verts),
                           detail::SupportFuncSpheropolygon(b.verts),
                           dr,
                           quat<ShortReal>(a.orientation),
                           quat<ShortReal>(b.orientation));*/

    return detail::xenocollide_2d(detail::SupportFuncSpheropolygon(a.verts),
                                  detail::SupportFuncSpheropolygon(b.verts),
                                  dr,
                                  quat<ShortReal>(a.orientation),
                                  quat<ShortReal>(b.orientation),
                                  err);
    }

#ifndef __HIPCC__
template<> inline std::string getShapeSpec(const ShapeSpheropolygon& spoly)
    {
    std::ostringstream shapedef;
    auto& verts = spoly.verts;
    unsigned int nverts = verts.N;
    if (nverts == 0)
        {
        throw std::runtime_error("Shape definition not supported for 0-vertex spheropolygons");
        }
    if (nverts == 1)
        {
        shapedef << "{\"type\": \"Sphere\", " << "\"diameter\": " << verts.diameter << "}";
        }
    else if (nverts == 2)
        {
        throw std::runtime_error("Shape definition not supported for 2-vertex spheropolygons.");
        }
    else
        {
        shapedef << "{\"type\": \"Polygon\", \"rounding_radius\": " << verts.sweep_radius
                 << ", \"vertices\": [";
        for (unsigned int i = 0; i < nverts - 1; i++)
            {
            shapedef << "[" << verts.x[i] << ", " << verts.y[i] << "], ";
            }
        shapedef << "[" << verts.x[nverts - 1] << ", " << verts.y[nverts - 1] << "]]}";
        }
    return shapedef.str();
    }
#endif

    } // end namespace hpmc
    } // end namespace hoomd

#undef HOSTDEVICE
#undef DEVICE
#endif // __SHAPE_SPHEROPOLYGON_H__
