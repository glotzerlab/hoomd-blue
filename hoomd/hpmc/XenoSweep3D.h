// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "MinkowskiMath.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include <cstdio>

#include "XenoCollide3D.h"

#ifndef __XENOSWEEP_3D_H__
#define __XENOSWEEP_3D_H__

/*! \file XenoSweep3D.h
    \brief Implements XenoCollide in 3D
*/

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
//! XenoSweep in 3D
/*! \tparam SupportFuncA Support function class type for shape A
    \tparam SupportFuncB Support function class type for shape B
    \param sa Support function for shape A
    \param sb Support function for shape B
    \param ab_t Vector pointing from a's center to b's center, in frame A
    \param q Orientation of shape B in frame A
    \param direction sweeping direction in frame A
    \param R Approximate radius of Minkowski difference for scaling tolerance value
    \param err_count Error counter to increment whenever an infinite loop is encountered
    \param newCollisionPlaneVector Gives a normal on the final portal
    \returns positive for two particles that collide in a given distance, negative for error values.
   -1: resultNoCollision, -2: resultNoForwardCollisionOrOverlapping, -3: resultOverlapping

    XenoSweep is an extension of the XenoCollide algorithm. Described in
    arxiv:2104.06829 https://arxiv.org/abs/2104.06829

    We try to find the surface element of the minkowski difference, that will be hit by the origin.
    It works well for polyhedra. As spherical/round shapes do not have a defined surface element
   that we can find, convergence is slow and way less efficient.

    \ingroup minkowski
*/
template<class SupportFuncA, class SupportFuncB>
DEVICE inline ShortReal xenosweep_3d(const SupportFuncA& sa,
                                     const SupportFuncB& sb,
                                     const vec3<ShortReal>& ab_t,
                                     const quat<ShortReal>& q,
                                     const vec3<ShortReal>& direction,
                                     const ShortReal R,
                                     unsigned int& err_count,
                                     vec3<ShortReal>& collisionPlaneVector)
    {
    vec3<ShortReal> v0, v1, v2, v3, v4, n, x;
    CompositeSupportFunc3D<SupportFuncA, SupportFuncB> S(sa, sb, ab_t, q);
    ShortReal d;

#if HOOMD_SHORTREAL_SIZE == 32
    // precision tolerance for single-precision floats near 1.0
    const ShortReal precision_tol = ShortReal(5e-7);

    // square root of precision tolerance
    const ShortReal root_tol = ShortReal(3e-4);
#else
    // precision tolerance for double-precision floats near 1.0
    const ShortReal precision_tol = ShortReal(1e-14); // 4e-14 for overlap check

    // square root of precision tolerance
    const ShortReal root_tol = ShortReal(2e-7); // 1e-7 for overlap check
#endif

    const ShortReal resultNoCollision = -1.0;
    const ShortReal resultNoForwardCollisionOrOverlapping = -2.0;
    // If there is a known overlap we will return a value <= -3 (which is related to how early the
    // overlap was detected). const ShortReal resultOverlapping = -3.0;

    if (fabs(ab_t.x) < root_tol && fabs(ab_t.y) < root_tol && fabs(ab_t.z) < root_tol)
        {
        // Interior point is at origin => particles overlap
        return resultNoCollision;
        }

    // Phase 1: Portal Discovery
    // ------
    // The portal-discovery for sweeps is a 2D-Collision detection with all
    // points projected into the plane perpendicular to the sweeping direction.
    //
    // vector 0 : the sweeping direction.
    v0 = direction;

    // ------
    // find a candidate portal of three support points
    //
    // find support v1 in the direction reverse to the sweeping direction
    v1 = ab_t; // S(-v0);

    // find support v2 in reversed v0 direction minus the part covered by v1
    n = dot(v0, v0) * v1 - dot(v1, v0) * v0;
    v2 = S(-n);

    n = cross(v0, v2 - v1);
    if (dot(n, v1) < ShortReal(0.0))
        {
        v1.swap(v2);
        n = -n;
        }

    unsigned int count = 0;
    while (count < XENOCOLLIDE_3D_MAX_ITERATIONS)
        {
        count++;

        // Get the next support point
        v3 = S(-n);

        if (dot(n, v3) >= ShortReal(0.0))
            {
            // The origin is not within the projected minkowski sum on the plane
            // perpendicular to the sweeping direction. No forward collision possible.
            return resultNoCollision;
            }

        // Updating v1 or v2.
        // consider the portals v1-v3 and v2-v3. The new portal should contain the
        // origin on the far side of v0. If neither portal does, the origin is inside;
        // then we have a portal for phase 2.
        x = cross(v3, v0);
        if (dot(x, v1) < ShortReal(0.0))
            {
            v2 = v3;
            }
        else if (dot(x, v2) > ShortReal(0.0))
            {
            v1 = v3;
            }
        else
            break;

        // Finally update the normal vector for the next iteration.
        n = cross(v0, v2 - v1);
        }

    // Error-Handling: We exceeded XENOCOLLIDE_3D_MAX_ITERATIONS, thus return an error.
    if (count == XENOCOLLIDE_3D_MAX_ITERATIONS)
        {
        err_count++;
        return resultNoForwardCollisionOrOverlapping;
        }

    // Phase 2: Portal Refinement (in 3D)
    count = 0;
    while (count < XENOCOLLIDE_3D_MAX_ITERATIONS)
        {
        count++;

        n = cross(v1 - v2, v3 - v1);

        // check if origin is inside (or overlapping, or behind, or touching)
        // the = is important, because in an MC simulation you are guaranteed
        // to find cases where edges and or vertices touch exactly.
        //
        // Only return in the first iteration (we may assume, that we search
        // in the wrong direction) or a few steps later, such that the portal
        // describes the intersection well.
        if (dot(v1, n) <= ShortReal(0.0) and (count == 1 or count > 3))
            {
            collisionPlaneVector = n;
            return resultNoForwardCollisionOrOverlapping - ShortReal(count);
            }

        // ----
        // find support in direction of portal's outer facing normal
        v4 = S(-n);

        // Perform tolerance checks
        // are we within an epsilon of the surface of the shape? If yes, done, one way
        // or another
        const ShortReal tol_multiplier = 10000;
        d = dot((v1 - v4) * tol_multiplier, n);
        ShortReal tol = precision_tol * tol_multiplier * R * fast::sqrt(dot(n, n));

        // First, check if v4 is on plane (v2,v1,v3)
        if (-tol < d and d < tol)
            {
            collisionPlaneVector = n;
            return dot(n, v4) / dot(n, v0);
            }

        // As v4 is not in plane yet, update the portal for the next iteration.
        x = cross(v4, v0);
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

    // If we end up here, we exceeded XENOCOLLIDE_3D_MAX_ITERATIONS
    // As best guess, take the current portal as approximant
    err_count++;
    collisionPlaneVector = n;
    return dot(n, v1) / dot(n, v0);
    }
    } // namespace detail
    } // end namespace hpmc
    } // end namespace hoomd

#undef DEVICE
#endif // __XENOCOLLIDE_3D_H__
