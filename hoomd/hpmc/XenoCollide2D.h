// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include <stdio.h>

#ifndef __XENOCOLLIDE_2D_H__
#define __XENOCOLLIDE_2D_H__

/*! \file XenoCollide2D.h
    \brief Implements XenoCollide in 2D
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __device__ when included in nvcc and blank when included into the host compiler
#ifdef __HIPCC__
#define DEVICE __device__
#else
#define DEVICE
#include <iostream>
#endif

namespace hoomd
    {
namespace hpmc
    {
namespace detail
    {
const unsigned int XENOCOLLIDE_2D_MAX_ITERATIONS = 1024;

//! Composite support functor
/*! \tparam SupportFuncA Support function class type for shape A
    \tparam SupportFuncB Support function class type for shape B

    Helper functor that computes the support function of the Minkowski difference B-A from the given
   two support functions. The given support functions are kept in local coords and
   translations/rotations are performed going in and out so that the input *n* and final result are
   in the space frame (where a is at the origin).

    This operation is performed many times in XenoCollide, so this convenience class simplifies the
   calling code from having too many rotations/translations.

    \ingroup minkowski
*/
template<class SupportFuncA, class SupportFuncB> class CompositeSupportFunc2D
    {
    public:
    //! Construct a composite support function
    /*! \param _sa Support function for shape A
        \param _sb Support function for shape B
        \param _ab_t Vector pointing from a's center to b's center, in the space frame
        \param _qa Orientation of shape A
        \param _qb Orientation of shape B
    */
    DEVICE CompositeSupportFunc2D(const SupportFuncA& _sa,
                                  const SupportFuncB& _sb,
                                  const vec2<ShortReal>& _ab_t,
                                  const quat<ShortReal>& _qa,
                                  const quat<ShortReal>& _qb)
        : sa(_sa), sb(_sb), ab_t(_ab_t), Ra(_qa), Rb(_qb)
        {
        }

    //! Compute the support function
    /*! \param n Normal vector input (in the space frame)
        \returns S_B(n) - S_A(n) in world space coords (transformations put n into local coords for
       S_A and S_b)
    */
    DEVICE vec2<ShortReal> operator()(const vec2<ShortReal>& n) const
        {
        // translation/rotation formula comes from pg 168 of "Games Programming Gems 7"
        vec2<ShortReal> SB_n = Rb * sb(transpose(Rb) * n) + ab_t;
        vec2<ShortReal> SA_n = Ra * sa(transpose(Ra) * (-n));
        return SB_n - SA_n;
        }

    private:
    const SupportFuncA& sa; //!< Support function for shape A
    const SupportFuncB& sb; //!< Support function for shape B
    const vec2<ShortReal>&
        ab_t; //!< Vector pointing from a's center to b's center, in the space frame
    const rotmat2<ShortReal> Ra; //!< Orientation of shape A
    const rotmat2<ShortReal> Rb; //!< Orientation of shape B
    };

//! XenoCollide overlap check in 2D
/*! \tparam SupportFuncA Support function class type for shape A
    \tparam SupportFuncB Support function class type for shape B
    \param sa Support function for shape A
    \param sb Support function for shape B
    \param ab_t Vector pointing from a's center to b's center, in the space frame
    \param qa Orientation of shape A
    \param qb Orientation of shape B
    \param err_count Error counter to increment whenever an infinite loop is encountered
    \returns true when the two shapes overlap and false when they are disjoint.

    XenoCollide is a generic algorithm for detecting overlaps between two shapes. It operates with
   the support function of each of the two shapes. To enable generic use of this algorithm on a
   variety of shapes, those support functions are passed in as templated functors. Each functor
   might store a reference to data (i.e. polygon verts), but the only public interface that
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
   position *ab_t*. Each particle has it's own orientation *qa* and *qb*.

    The recommended way of using this code is to specify the support functor in the same file as the
   shape data (e.g. ShapeConvexPolygon.h). Then include XenoCollide2D.h and call xenocollide_2d
   where needed.

    **Normalization**
    In _Games Programming Gems_, the book normalizes all vectors passed into S. This is unnecessary
   in some circumstances and we avoid it for performance reasons. Support functions that require the
   use of normal n vectors should normalize it when needed.

    This implementation works, but is minimally tested and only on shapes of diameter 1. gjkm_2d()
   is the production code for overlap detection of convex shapes.

    \ingroup minkowski
*/
template<class SupportFuncA, class SupportFuncB>
DEVICE inline bool xenocollide_2d(const SupportFuncA& sa,
                                  const SupportFuncB& sb,
                                  const vec2<ShortReal>& ab_t,
                                  const quat<ShortReal>& qa,
                                  const quat<ShortReal>& qb,
                                  unsigned int& err_count)
    {
    // This implementation of XenoCollide is hand-written from the description of the algorithm on
    // page 171 of _Games Programming Gems 7_

    vec2<ShortReal> v0, v1, v2, v3, v10_perp, v21_perp, v30_perp;
    const ShortReal tol_multiplier = 10000;
    const ShortReal tol = ShortReal(1e-7) * tol_multiplier;
    CompositeSupportFunc2D<SupportFuncA, SupportFuncB> S(sa, sb, ab_t, qa, qb);

    // Phase 1: Portal Discovery
    // ------
    // find the origin ray v0
    // The easiest origin ray is the position of b minus the position of a, or more simply:
    v0 = ab_t;

    // ------
    // find a candidate portal
    v1 = S(-v0);

    // need a vector perpendicular to v1-v0 to find v2.
    v10_perp = perp(v1 - v0);

    // There are two possibilities, choose the one that points toward the origin
    if (dot(v1, v10_perp) > 0)
        v10_perp = -v10_perp;

    v2 = S(v10_perp);

    // ------
    // while (origin ray does not intersect candidate) choose new candidate
    // In 2D, this step is not necessary (https://xenocollide.snethen.com/mpr2d.html). By choosing
    // the correct normal above, we have a proper portal by construction. The 3D version of this
    // code will need to implement the loop, however. See the book.

    // Phase 2: Portal Refinement
    unsigned int count = 0;
    while (1)
        {
        // ----
        // if (origin inside portal) return true
        v21_perp = perp(v2 - v1);

        // Make v21_perp point away from the interior point
        if (dot(v1 - v0, v21_perp) < 0)
            v21_perp = -v21_perp;

        // check if origin is inside (or overlapping)
        // the = is important, because in an MC simulation you are guaranteed to find cases where
        // edges and or vertices touch exactly
        if (dot(v1, v21_perp) >= 0)
            {
            return true;
            }

        // ----
        // find support in direction of portal
        v3 = S(v21_perp);

        // ----
        // if (origin outside support plane) return false
        if (dot(v3, v21_perp) < 0)
            {
            return false;
            }

        // ----
        // if (support plane close to portal) return false
        // JAA - I originally thought that this was only a tolerance for curved surfaces. However,
        // it is also necessary for polygons. Here is an example edge case where it causes problems:
        // v0 = (-0.0552826,1.3569); v1 = (0.0180616,-0.00930899); v2 = (-0.870821,0.448825); v3
        // =(-0.870821,0.448825); the portal v1,v2 has narrowed down to the surface of B-A (as
        // evidenced by v3==v2). Thus, the two dots above SHOULD be comparing identical planes and
        // one if should succeed. However, due to truncation error and the fact that v1 is used in
        // the first if and v2 (because v2==v3) in the 2nd, both ifs fail and the code loops
        // infinitely. We can return either a hit or a miss in this case because we can't resolve
        // the difference within floating point precision

        // are we within an epsilon of the surface of the shape? If yes, done (overlap)
        vec2<ShortReal> d = ((v3 - v1) - project(v3 - v1, v2 - v1)) * tol_multiplier;

        if (dot(d, d) < tol * tol * dot(v3, v3))
            return true;

        if (count >= XENOCOLLIDE_2D_MAX_ITERATIONS)
            {
            err_count++;
            return true;
            }

        // ----
        // choose new portal, it is either v3 and v2 or v3 and v1.
        // To determine which, construct a line segment from v0 to v3 and test if the origin is on
        // the same side of this line as v1. If so, then the next portal is v3 and v1 - if not, then
        // the next portal is v3 and v2
        v30_perp = perp(v3 - v0);
        // make v30_perp point toward v1
        if (dot(v1 - v3, v30_perp) < 0)
            v30_perp = -v30_perp;

        // now check which side the origin is on
        if (dot(v3, v30_perp) < 0)
            {
            // on v1 side, make the new portal be v3 and v1
            v2 = v3;
            }
        else
            {
            // on v2 side, make the new portal be v3 and v2
            v1 = v3;
            }

        count++;
        }
    }

    } // end namespace detail

    } // end namespace hpmc
    } // end namespace hoomd

#endif // __XENOCOLLIDE_2D_H__
