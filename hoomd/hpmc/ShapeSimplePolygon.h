// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"
#include "hoomd/VectorMath.h"
#include "ShapeSphere.h"    //< For the base template of test_overlap
#include "ShapeConvexPolygon.h"

#ifndef __SHAPE_SIMPLE_POLYGON_H__
#define __SHAPE_SIMPLE_POLYGON_H__

/*! \file ShapeSimplePolygon.h
    \brief Defines the simple polygon shape
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __device__ when included in nvcc and blank when included into the host compiler
#ifdef __HIPCC__
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#endif

namespace hpmc
{

//! Simple Polygon shape template
/*! ShapeSimplePolygon implements IntegratorHPMC's shape protocol. It uses the same data structures as
    ShapeConvexPolygon, but the overlap check is generalized to support simple polygons (i.e. concave).

    The parameter defining a polygon is a structure containing a list of N vertices. They are assumed to be listed
    in counter-clockwise order and centered on 0,0.

    \ingroup shape
*/
struct ShapeSimplePolygon
    {
    //! Define the parameter type
    typedef detail::poly2d_verts param_type;

    //! Initialize a polygon
    DEVICE ShapeSimplePolygon(const quat<Scalar>& _orientation, const param_type& _params)
        : orientation(_orientation), verts(_params)
        {
        }

    //! Does this shape have an orientation
    DEVICE bool hasOrientation() const { return true; }

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
        return Scalar(0.0);
        }

    #ifndef __HIPCC__
    std::string getShapeSpec() const
        {
        std::ostringstream shapedef;
        shapedef << "{\"type\": \"Polygon\", \"rounding_radius\": " << verts.sweep_radius << ", \"vertices\": [";
        for (unsigned int i = 0; i < verts.N-1; i++)
            {
            shapedef << "[" << verts.x[i] << ", " << verts.y[i] << "], ";
            }
        shapedef << "[" << verts.x[verts.N-1] << ", " << verts.y[verts.N-1] << "]]}";
        return shapedef.str();
        }
    #endif

    //! Return the bounding box of the shape in world coordinates
    DEVICE detail::AABB getAABB(const vec3<Scalar>& pos) const
        {
        return detail::AABB(pos, verts.diameter/Scalar(2));
        }

    //! Return a tight fitting OBB
    DEVICE detail::OBB getOBB(const vec3<Scalar>& pos) const
        {
        // just use the AABB for now
        return detail::OBB(getAABB(pos));
        }

    //! Returns true if this shape splits the overlap check over several threads of a warp using threadIdx.x
    HOSTDEVICE static bool isParallel() { return false; }

    //! Retrns true if the overlap check supports sweeping both shapes by a sphere of given radius
    HOSTDEVICE static bool supportsSweepRadius()
        {
        return false;
        }

    quat<Scalar> orientation;    //!< Orientation of the polygon

    const detail::poly2d_verts& verts;     //!< Vertices
    };

namespace detail
{

//! Test if a point is inside a polygon
/*! \param verts Polygon vertices
    \param p Point
    \returns true if the point is inside the polygon

    \note \a p is *in the polygon's reference frame!*

    \ingroup overlap
*/
DEVICE inline bool is_inside(const vec2<OverlapReal>& p, const poly2d_verts& verts)
    {
    // code for concave test from: http://alienryderflex.com/polygon/
    unsigned int nvert = verts.N;

    unsigned int  i, j=nvert-1;
    bool oddNodes=false;

    for (i = 0; i < nvert; i++)
        {
        // if (polyY[i]<y && polyY[j]>=y ||  polyY[j]<y && polyY[i]>=y)
        if ((verts.y[i] < p.y && verts.y[j] >= p.y) ||  (verts.y[j] < p.y && verts.y[i] >= p.y))
            {
            // if (polyX[i]+(y-polyY[i])/(polyY[j]-polyY[i])*(polyX[j]-polyX[i])<x)
            if (verts.x[i]+(p.y-verts.y[i])/(verts.y[j]-verts.y[i])*(verts.x[j]-verts.x[i]) < p.x)
                {
                oddNodes=!oddNodes;
                }
            }
        j=i;
        }

    return oddNodes;
    }

//! Test if 3 points are in ccw order
DEVICE inline unsigned int tri_orientation(const vec2<OverlapReal>& a, const vec2<OverlapReal>& b, const vec2<OverlapReal>& c)
    {
    const OverlapReal precision_tol = 1e-6;
    OverlapReal v = ((c.y - a.y)*(b.x - a.x) - (b.y - a.y)*(c.x - a.x));

    if (fabs(v) < precision_tol)
        return 0;
    else if (v > 0)
        return 1;
    else
        return 2;
    }


//! Test if two line segments intersect
/*! \param a vertex of segment 1
    \param b vertex of segment 1
    \param c vertex of segment 2
    \param d vertex of segment 2
    \returns true when the two line segments intersect

    \ingroup overlap
*/
DEVICE inline bool segment_intersect(const vec2<OverlapReal>& a, const vec2<OverlapReal>& b, const vec2<OverlapReal>& c, const vec2<OverlapReal>& d)
    {
    // implemented following the algorithm in: http://www.dcs.gla.ac.uk/~pat/52233/slides/Geometry1x1.pdf
    unsigned int o1 = tri_orientation(a, c, d);
    unsigned int o2 = tri_orientation(b, c, d);
    unsigned int o3 = tri_orientation(a, b, c);
    unsigned int o4 = tri_orientation(a, b, d);

    // general case
    if (o1 != o2 && o3 != o4)
        return true;

    // special case
    if (o1 == 0 && o2 == 0 && o3 == 0 && o4 == 0)
        {
        // all points a,b,c,d are in a line. Project onto that line and see if the intervals overlap or not.
        vec2<OverlapReal> v = b-a;

        OverlapReal p1 = dot(a,v);
        OverlapReal p2 = dot(b,v);
        OverlapReal min_1 = min(p1, p2);
        OverlapReal max_1 = max(p1, p2);

        OverlapReal p3 = dot(c,v);
        OverlapReal p4 = dot(d,v);

        if ((p3 > min_1 && p3 < max_1) || (p4 > min_1 && p4 < max_1))
            return true;
        }

    return false;
    }

//! Test if two simple polygons overlap
/*! \param a First polygon
    \param b Second polygon
    \param dr Vector pointing from a's center to b's center, in the space frame
    \param qa Orientation of first polygon
    \param qb Orientation of second polygon
    \returns true when the two polygons overlap

    Shape *a* is at the origin. (in other words, we are solving this in the frame of *a*). Normal vectors can be rotated
    from the frame of a to b simply by rotating by ab_r, which is equal to conj(b.orientation) * a.orientation.
    This comes from the following

        - first, go from the frame of *a* into the space frame (qa))
        - then, go into back into the *b* frame (conj(qb))

    Transforming points from one frame into another takes a bit more care. The easiest way to think about it is this:

        - Rotate from the *a* frame into the space frame (rotate by *qa*).
        - Then translate into a frame with *b*'s origin at the center (subtract dr).
        - Then rotate into the *b* frame (rotate by conj(*qb*))

    Putting that all together, we get: \f$ q_b^* \cdot (q_a \vec{v} q_a^* - \vec{a}) a_b \f$. That's a lot of quats to
    store and a lot of rotations to do. Distributing gives \f$ q_b^* q_a \vec{v} q_a^* q_b - q_b^* \vec{a} q_b \f$.
    The first rotation is by the already computed ab_r! The 2nd only needs to be computed once

    \pre Polygon vertices are in **counter-clockwise** order
    \pre The shape is simple (no self crossings)

    \ingroup overlap
*/
DEVICE inline bool test_simple_polygon_overlap(const poly2d_verts& a,
                                               const poly2d_verts& b,
                                               const vec2<OverlapReal>& dr,
                                               const quat<OverlapReal>& qa,
                                               const quat<OverlapReal>& qb)
    {
    // construct a quaternion that rotates from a's coordinate system into b's
    quat<OverlapReal> ab_r = conj(qb) * qa;
    vec2<OverlapReal> ab_t = rotate(conj(qb), dr);

    // loop through all edges in a. As we loop through them, transform into b's coordinate system and do the checks
    // there
    unsigned int j = a.N-1;
    vec2<OverlapReal> prev_a = rotate(ab_r, vec2<OverlapReal>(a.x[j], a.y[j])) - ab_t;
    for (unsigned int i = 0; i < a.N; i++)
        {
        vec2<OverlapReal> cur_a = rotate(ab_r, vec2<OverlapReal>(a.x[i], a.y[i])) - ab_t;

        // check if this edge in a intersects any edge in b
        unsigned int k = b.N-1;
        vec2<OverlapReal> prev_b = vec2<OverlapReal>(b.x[k], b.y[k]);
        for (unsigned int l = 0; l < b.N; l++)
            {
            vec2<OverlapReal> cur_b = vec2<OverlapReal>(b.x[l], b.y[l]);

            if (segment_intersect(prev_a, cur_a, prev_b, cur_b))
                return true;

            k = l;
            prev_b = cur_b;
            }

        // check if any vertex from a is in inside b
        if (is_inside(cur_a, b))
            return true;

        // save previous vertex for next iteration
        j = i;
        prev_a = cur_a;
        }

    // switch coordinate systems
    ab_r = conj(ab_r);
    ab_t = rotate(conj(qa), -dr);

    for (unsigned int i = 0; i < a.N; i++)
        {
        vec2<OverlapReal> cur = rotate(ab_r, vec2<OverlapReal>(b.x[i], b.y[i])) - ab_t;

        // check if any vertex from b is in inside a
        if (is_inside(cur, a))
            return true;
        }

    // if we get here, there is no overlap
    return false;
    }

}; // end namespace detail

//! Simple polygon overlap test
/*!
    \param r_ab Vector defining the position of shape b relative to shape a (r_b - r_a)
    \param a Shape a
    \param b Shape b
    \param err in/out variable incremented when error conditions occur in the overlap test
    \param sweep_radius Additional sphere radius to sweep the shapes with
    \returns true if the two shapes overlap
    \ingroup shape
*/
template <>
DEVICE inline bool test_overlap<ShapeSimplePolygon,ShapeSimplePolygon>(const vec3<Scalar>& r_ab,
                                                                       const ShapeSimplePolygon& a,
                                                                       const ShapeSimplePolygon& b,
                                                                       unsigned int& err,
                                                                       Scalar sweep_radius_a,
                                                                       Scalar sweep_radius_b)
    {
    // trivial rejection: first check if the circumscribing spheres overlap
    vec2<OverlapReal> dr(r_ab.x, r_ab.y);

    return detail::test_simple_polygon_overlap(a.verts,
                                               b.verts,
                                               dr,
                                               quat<OverlapReal>(a.orientation),
                                               quat<OverlapReal>(b.orientation));
    }

#ifndef __HIPCC__
template<>
inline std::string getShapeSpec(const ShapeSimplePolygon& poly)
    {
    std::ostringstream shapedef;
    auto& verts = poly.verts;
    shapedef << "{\"type\": \"Polygon\", \"rounding_radius\": " << poly.verts.sweep_radius << ", \"vertices\": [";
    for (unsigned int i = 0; i < verts.N-1; i++)
        {
        shapedef << "[" << verts.x[i] << ", " << verts.y[i] << "], ";
        }
    shapedef << "[" << verts.x[verts.N-1] << ", " << verts.y[verts.N-1] << "]]}";
    return shapedef.str();
    }
#endif

}; // end namespace hpmc

#undef DEVICE
#undef HOSTDEVCE

#endif //__SHAPE_CONVEX_POLYGON_H__
