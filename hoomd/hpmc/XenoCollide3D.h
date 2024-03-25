// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "MinkowskiMath.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include <cstdio>

#ifndef __XENOCOLLIDE_3D_H__
#define __XENOCOLLIDE_3D_H__

/*! \file XenoCollide3D.h
    \brief Implements XenoCollide in 3D
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __device__ when included in nvcc and blank when included into the host compiler
#ifdef __HIPCC__
#define DEVICE __device__
#else
#define DEVICE
#endif

namespace hoomd
    {
namespace hpmc
    {
namespace detail
    {
const unsigned int XENOCOLLIDE_3D_MAX_ITERATIONS = 1024;

//! XenoCollide overlap check in 3D
/*! \tparam SupportFuncA Support function class type for shape A
    \tparam SupportFuncB Support function class type for shape B
    \param sa Support function for shape A
    \param sb Support function for shape B
    \param ab_t Vector pointing from a's center to b's center, in frame A
    \param q Orientation of shape B in frame A
    \param R Approximate radius of Minkowski difference for scaling tolerance value
    \param err_count Error counter to increment whenever an infinite loop is encountered
    \returns true when the two shapes overlap and false when they are disjoint.

    XenoCollide is a generic algorithm for detecting overlaps between two shapes. It operates with
   the support function of each of the two shapes. To enable generic use of this algorithm on a
   variety of shapes, those support functions are passed in as templated functors. Each functor
   might store a reference to data (i.e. polyhedron verts), but the only public interface that
   XenoCollide will use is to call the operator() on the functor and give it the normal vector n *in
   the **local** coordinates* of that shape. Local coordinates are used to avoid massive memory
   usage needed to store a translated copy of each shape.

    The initial implementation is designed primarily for polygons. Shapes with curved surfaces could
   be used, but they require an additional termination condition that comes with a tolerance. When
   and if such shapes are needed, we can update this function to optionally implement that tolerance
   (via another template parameter).

    The parameters of this class closely follow those of test_overlap_separating_planes, since they
   were found to be a good breakdown of the problem into coordinate systems. Specifically, overlaps
   are checked in a coordinate system where particle *A* is at the origin, and particle *B* is at
   position *ab_t*. Particle A has orientation (1,0,0,0) and particle B has orientation *q*.

    The recommended way of using this code is to specify the support functor in the same file as the
   shape data (e.g. ShapeConvexPolyhedron.h). Then include XenoCollide3D.h and call xenocollide_3d
   where needed.

    **Normalization**
    In _Games Programming Gems_, the book normalizes all vectors passed into S. This is unnecessary
   in some circumstances and we avoid it for performance reasons. Support functions that require the
   use of normal n vectors should normalize it when needed.

    \ingroup minkowski
*/
template<class SupportFuncA, class SupportFuncB>
DEVICE inline bool xenocollide_3d(const SupportFuncA& sa,
                                  const SupportFuncB& sb,
                                  const vec3<ShortReal>& ab_t,
                                  const quat<ShortReal>& q,
                                  const ShortReal R,
                                  unsigned int& err_count)
    {
    // This implementation of XenoCollide is hand-written from the description of the algorithm on
    // page 171 of _Games Programming Gems 7_

    vec3<ShortReal> v0, v1, v2, v3, v4, n;
    CompositeSupportFunc3D<SupportFuncA, SupportFuncB> S(sa, sb, ab_t, q);
    ShortReal d;
    const ShortReal precision_tol
        = ShortReal(1e-7); // precision tolerance for single-precision floats near 1.0

    // square root of precision tolerance, in distance units
    const ShortReal root_tol = ShortReal(3e-4) * R;

    if (fabs(ab_t.x) < root_tol && fabs(ab_t.y) < root_tol && fabs(ab_t.z) < root_tol)
        {
        // Interior point is at origin => particles overlap
        return true;
        }

    // Phase 1: Portal Discovery
    // ------
    // Find the origin ray v0 from the origin to an interior point of the Minkowski difference.
    // The easiest origin ray is the position of b minus the position of a, or more simply:
    v0 = ab_t;

    // ------
    // find a candidate portal of three support points
    //
    // find support v1 in the direction of the origin
    v1 = S(-v0); // should be guaranteed ||v1|| > 0

    /* if (dot(v1, v1 - v0) <= 0) // by convexity */
    if (dot(v1, v0) > ShortReal(0.0))
        return false; // origin is outside v1 support plane

    // find support v2 perpendicular to v0, v1 plane
    n = cross(v1, v0);
    // cross product is zero if v0,v1 colinear with origin, but we have already determined origin is
    // within v1 support plane. If origin is on a line between v1 and v0, particles overlap.
    // if (dot(n, n) < tol)
    // cross product has units of length**2, multiply tolerance by R**2 to put in the same units
    if (fabs(n.x) < precision_tol * R * R && fabs(n.y) < precision_tol * R * R
        && fabs(n.z) < precision_tol * R * R)
        return true;

    v2 = S(n); // Convexity should guarantee ||v2|| > 0, but v2 == v1 may be possible in edge cases
               // of {B}-{A}
    // particles do not overlap if origin outside v2 support plane
    if (dot(v2, n) < ShortReal(0.0))
        return false;

    // Find next support direction perpendicular to plane (v1,v0,v2)
    n = cross(v1 - v0, v2 - v0);
    // Maintain known handedness of the portal: make sure plane normal points towards origin
    if (dot(n, v0) > ShortReal(0.0))
        {
        v1.swap(v2);
        n = -n;
        }

    // ------
    // while (origin ray does not intersect candidate) choose new candidate
    bool intersects = false;
    unsigned int count = 0;
    while (!intersects)
        {
        count++;

        if (count >= XENOCOLLIDE_3D_MAX_ITERATIONS)
            {
            err_count++;
            return true;
            }

        // Get the next support point
        v3 = S(n);
        if (dot(v3, n) <= 0)
            return false; // check if origin outside v3 support plane

        // If origin lies on opposite side of a plane from the third support point, use outer-facing
        // plane normal to find a new support point. Check (v3,v0,v1) if (dot(cross(v3 - v0, v1 -
        // v0), -v0) < 0)
        // -> if (dot(cross(v1 - v0, v3 - v0), v0) < 0)
        // A little bit of algebra shows that dot(cross(a - c, b - c), c) == dot(cross(a, b), c)
        if (dot(cross(v1, v3), v0) < ShortReal(0.0))
            {
            // replace v2 and find new support direction
            v2 = v3; // preserve handedness
            n = cross(v1 - v0, v2 - v0);
            continue; // continue iterating to find valid portal
            }
        // Check (v2, v0, v3)
        if (dot(cross(v3, v2), v0) < ShortReal(0.0))
            {
            // replace v1 and find new support direction
            v1 = v3;
            n = cross(v1 - v0, v2 - v0);
            continue;
            }

        // If we've made it this far, we have a valid portal and can proceed to refine the portal
        intersects = true;
        }

    // Phase 2: Portal Refinement
    count = 0;
    while (true)
        {
        count++;

        // ----
        // if (origin inside portal) return true
        n = cross(v2 - v1, v3 - v1); // by construction, this is the outer-facing normal

        // check if origin is inside (or overlapping)
        // the = is important, because in an MC simulation you are guaranteed to find cases where
        // edges and or vertices touch exactly
        if (dot(v1, n) >= ShortReal(0.0))
            {
            return true;
            }

        // ----
        // find support in direction of portal's outer facing normal
        v4 = S(n);

        // ----
        // if (origin outside support plane) return false
        if (dot(v4, n) < ShortReal(0.0))
            {
            return false;
            }

        // Perform tolerance checks
        // are we within an epsilon of the surface of the shape? If yes, done, one way or another
        const ShortReal tol_multiplier = 10000;
        n = cross(v2 - v1, v3 - v1);
        d = dot((v4 - v1) * tol_multiplier, n);
        // d is in units of length**3, multiply by R**2 * |n| to put in the same units
        ShortReal tol = precision_tol * tol_multiplier * R * R * fast::sqrt(dot(n, n));

        // First, check if v4 is on plane (v2,v1,v3)
        if (fabs(d) < tol)
            return false; // no more refinement possible, but not intersection detected

        // Second, check if origin is on plane (v2,v1,v3) and has been missed by other checks
        d = dot(v1 * tol_multiplier, n);
        if (fabs(d) < tol)
            return true;

        if (count >= XENOCOLLIDE_3D_MAX_ITERATIONS)
            {
            err_count++;
            /*
            // Output useful info if we are in an infinite loop
            printf(
                "Could not resolve overlap check. b.pos - a.pos = (%f %f %f)\n",
                ab_t.x, ab_t.y, ab_t.z
                );
            printf(
                "a.orientation = (%f %f %f %f)\n",
                qa.s, qa.v.x, qa.v.y, qa.v.z
                );
            printf(
                "b.orientation = (%f %f %f %f)\n",
                qb.s, qb.v.x, qb.v.y, qb.v.z
                );
            printf(
                "v1=(%f %f %f) v2=(%f %f %f) v3=(%f %f %f) v4=(%f %f %f)) d=%f Target<%f\n",
                v1.x, v1.y, v1.z,
                v2.x, v2.y, v2.z,
                v3.x, v3.y, v3.z,
                v4.x, v4.y, v4.z,
                d, tol
                );
            */
            return true;
            }

        // ----
        // Choose new portal. Two of its edges will be from the planes (v4,v0,v1), (v4,v0,v2),
        // (v4,v0,v3). Find which two have the origin on the same side. MEI: As I understand this
        // statement, I don't believe it is correct. An _inside_ needs to be defined and used. The
        // only way I can think to do this is to consider all three pairs of planes to find which
        // pair has the origin between them. Need to better understand and document this. The
        // following code was directly adapted from example code.

        // Test origin against the three planes that separate the new portal candidates: (v1,v4,v0)
        // (v2,v4,v0) (v3,v4,v0) Note:  We're taking advantage of the triple product identities here
        // as an optimization
        //        (v1 % v4) * v0 == v1 * (v4 % v0)    > 0 if origin inside (v1, v4, v0)
        //        (v2 % v4) * v0 == v2 * (v4 % v0)    > 0 if origin inside (v2, v4, v0)
        //        (v3 % v4) * v0 == v3 * (v4 % v0)    > 0 if origin inside (v3, v4, v0)
        vec3<ShortReal> x = cross(v4, v0);
        if (dot(v1, x) > ShortReal(0.0))
            {
            if (dot(v2, x) > ShortReal(0.0))
                v1 = v4; // Inside v1 & inside v2 ==> eliminate v1
            else
                v3 = v4; // Inside v1 & outside v2 ==> eliminate v3
            }
        else
            {
            if (dot(v3, x) > ShortReal(0.0))
                v2 = v4; // Outside v1 & inside v3 ==> eliminate v2
            else
                v1 = v4; // Outside v1 & outside v3 ==> eliminate v1
            }
        }
    }
    } // namespace detail

    } // end namespace hpmc
    } // end namespace hoomd

#undef DEVICE
#endif // __XENOCOLLIDE_3D_H__
