// Copyright (c) 2009-2017 The Regents of the University of Michigan
// Copyright (c) 2017-2019 Marco Klement, Michael Engel
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "hoomd/HOOMDMath.h"
#include "HPMCPrecisionSetup.h"
#include "hoomd/VectorMath.h"
#include "MinkowskiMath.h"
#include <cstdio>

#include "XenoCollide3D.h"

#ifndef __XENOSWEEP_3D_H__
#define __XENOSWEEP_3D_H__

/*! \file XenoSweep3D.h
    \brief Implements XenoCollide in 3D
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

// #ifdef SINGLE_PRECISION
// #error maybe we should use doubles... [is: SINGLE_PRECISION]
// #else
// #ifdef ENABLE_HPMC_MIXED_PRECISION
// #error maybe we should use doubles... [is: MIXED_PRECISION]
// #endif
// #endif

namespace hpmc
{

namespace detail
{

//const unsigned int XENOCOLLIDE_3D_MAX_ITERATIONS = 1024;

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
    \returns positive for two particles that collide in a given distance, negative for error values. -1: resultNoCollision, -2: resultNoForwardCollisionOrOverlapping, -3: resultOverlapping

    XenoSweep is an extension of the XenoCollide algorithm. Described in arXiv:xxxx.xxxxx | [insert-paper-here]
    
    We try to find the surface element of the minkowski difference, that will be hit by the origin.
    It works well for polyhedra. As spherical/round shapes do not have a defined surface element that we can find, convergence is slow and way less efficient.

    \ingroup minkowski
*/
template<class SupportFuncA, class SupportFuncB>
DEVICE inline OverlapReal xenosweep_3d(const SupportFuncA& sa,
                                       const SupportFuncB& sb,
                                       const vec3<OverlapReal>& ab_t,
                                       const quat<OverlapReal>& q,
                                       const vec3<OverlapReal>& direction,
                                       const OverlapReal R,
                                       unsigned int& err_count,
                                       vec3<OverlapReal>& collisionPlaneVector
)
    {

    vec3<OverlapReal> v0, v1, v2, v3, v4, n, x;
    CompositeSupportFunc3D<SupportFuncA, SupportFuncB> S(sa, sb, ab_t, q);
    OverlapReal d;
    
    #if defined(SINGLE_PRECISION) || defined(ENABLE_HPMC_MIXED_PRECISION)
        // precision tolerance for single-precision floats near 1.0
        const OverlapReal precision_tol = 5e-7;

        // square root of precision tolerance
        const OverlapReal root_tol = 3e-4;
    #else
        // precision tolerance for double-precision floats near 1.0
        const OverlapReal precision_tol = 1e-14; // 4e-14 for overlap check

        // square root of precision tolerance
        const OverlapReal root_tol = 2e-7; // 1e-7 for overlap check
    #endif
    

    const OverlapReal resultNoCollision = -1.0;
    const OverlapReal resultNoForwardCollisionOrOverlapping = -2.0;
    // If there is a known overlap we will return a value <= -3 (which is related to how early the overlap was detected).
    // const OverlapReal resultOverlapping = -3.0;
    
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
    v1 = ab_t;//S(-v0);

    // find support v2 in reversed v0 direction minus the part covered by v1 
    n = dot(v0,v0)*v1 - dot(v1,v0)*v0;
    v2 = S(-n);

    n = cross(v0, v2 - v1);
    if( dot(n,v1) < OverlapReal(0.0))
        {
        v1.swap(v2);
        n = -n;
        }
        
    unsigned int count = 0;
    while (true) // 2D-Refinement-Loop
        {
        count++;

        if (count >= XENOCOLLIDE_3D_MAX_ITERATIONS)
            {
            err_count++;
            
            #ifdef XENOERRORPUTS
            printf("[EE] XenoSweep3D above max iterations in 2D-Iteration\n"
                   "     ab_t      = vec3<OverlapReal>( %10f , %10f , %10f) ; \n"
                   "     direction = vec3<OverlapReal>( %10f , %10f , %10f) ; \n"
                   "     q         = quat<OverlapReal>( %10f , vec3<OverlapReal>( %10f , %10f , %10f) ; \n",
                   ab_t.x, ab_t.y, ab_t.z,
                   direction.x, direction.y, direction.z,
                   q.s, q.v.x, q.v.y, q.v.z
                   );
            #endif
            
            return resultNoForwardCollisionOrOverlapping;
            }

        // Get the next support point
        v3 = S(-n);
        
        //TODO Fix exceeding limit.
        //  Ideas:
        //  Numerical tolerant check for "dot on line"?
        
        if( dot(n,v3) >= OverlapReal(0.0) )
            {
            // The origin is not within the projected minkowski sum on the plane
            // perpendicular to the sweeping direction. No forward collision possible.
            return resultNoCollision;
            }
        
        // Updating v1 or v2.
        // consider the portals v1-v3 and v2-v3. The new portal should contain the
        // origin on the far side of v0. If neither portal does, the origin is inside;
        // then we have a portal for phase 2.
        x  = cross( v3, v0 );
        if( dot(x,v1) < OverlapReal(0.0) )
            {
            v2 = v3;
            }
        else if( dot(x,v2) > OverlapReal(0.0) )
            {
            v1 = v3;
            }
        else break;
        
        // Finally update the normal vector for the next iteration.
        n = cross(v0, v2-v1);
        }
        

    // Phase 2: Portal Refinement (in 3D)
    count = 0;
    while (true) // 3D-Refinement-Loop
        {
        count++;

        n = cross(v1 - v2, v3 - v1);

        // check if origin is inside (or overlapping, or behind, or touching)
        // the = is important, because in an MC simulation you are guaranteed
        // to find cases where edges and or vertices touch exactly.
        if (dot(v1, n) <= OverlapReal(0.0))
            {
//             return (count==1?resultNoForwardCollisionOrOverlapping:resultOverlapping);
            collisionPlaneVector = n;
            return resultNoForwardCollisionOrOverlapping - count + 1;
            }

        // ----
        // find support in direction of portal's outer facing normal
        v4 = S(-n);

        // Perform tolerance checks
        // are we within an epsilon of the surface of the shape? If yes, done, one way
        // or another
        const OverlapReal tol_multiplier = 10000;
        d = dot((v1 - v4) * tol_multiplier , n);
        OverlapReal tol = precision_tol * tol_multiplier * R * fast::sqrt(dot(n,n));
        //OverlapReal tol = precision_tol * tol_multiplier * R * sqrt(dot(n,n));

        // First, check if v4 is on plane (v2,v1,v3) or "behind"
        if (d < tol)
            {
            collisionPlaneVector = n;
            return dot(n,v1) / dot(n,v0);
            }

        #ifdef XENOERRORPUTS
        printf("[II] New v4 with %e > %e\n"
                "     v1      = vec3<OverlapReal>( %10f , %10f , %10f) ; \n"
                "     v2      = vec3<OverlapReal>( %10f , %10f , %10f) ; \n"
                "     v3      = vec3<OverlapReal>( %10f , %10f , %10f) ; \n"
                "     v4      = vec3<OverlapReal>( %10f , %10f , %10f) ; \n"
                "     n       = vec3<OverlapReal>( %10f , %10f , %10f) ; \n",
                d, tol,
                v1.x, v1.y, v1.z,
                v2.x, v2.y, v2.z,
                v3.x, v3.y, v3.z,
                v4.x, v4.y, v4.z,
                 n.x,  n.y,  n.z
                );
        #endif

        
        if (count >= XENOCOLLIDE_3D_MAX_ITERATIONS)
            {
            err_count++;

            #ifdef XENOERRORPUTS
            printf("[EE] XenoSweep3D above max iterations!\n"
                   "     ab_t      = vec3<OverlapReal>( %10f , %10f , %10f) ; \n"
                   "     direction = vec3<OverlapReal>( %10f , %10f , %10f) ; \n"
                   "     q         = quat<OverlapReal>( %10f , vec3<OverlapReal>( %10f , %10f , %10f) ; \n",
                   ab_t.x, ab_t.y, ab_t.z,
                   direction.x, direction.y, direction.z,
                   q.s, q.v.x, q.v.y, q.v.z
            );
            printf("     Ended with %e > %e  Final Vectors:\n"
                    "     v1      = vec3<OverlapReal>( %10f , %10f , %10f) ; \n"
                    "     v2      = vec3<OverlapReal>( %10f , %10f , %10f) ; \n"
                    "     v3      = vec3<OverlapReal>( %10f , %10f , %10f) ; \n"
                    "     v4      = vec3<OverlapReal>( %10f , %10f , %10f) ; \n"
                    "     n       = vec3<OverlapReal>( %10f , %10f , %10f) ; \n",
                    d, tol,
                    v1.x, v1.y, v1.z,
                    v2.x, v2.y, v2.z,
                    v3.x, v3.y, v3.z,
                    v4.x, v4.y, v4.z,
                    n.x,  n.y,  n.z

            );
            #endif

            collisionPlaneVector = n;
            return dot(n,v1) / dot(n,v0);
            // Return current distance, as it should be a better estimate than ignoring each other.
            //             return noForwardCollisionOrOverlapping;
            }

        x = cross(v4, v0);
        if (dot(v1, x) > OverlapReal(0.0))
            {
            if (dot(v2, x) > OverlapReal(0.0))
                v1 = v4;    // Inside v1 & inside v2 ==> eliminate v1
            else
                v3 = v4;                   // Inside v1 & outside v2 ==> eliminate v3
            }
        else
            {
            if (dot(v3, x) > OverlapReal(0.0))
                v2 = v4;    // Outside v1 & inside v3 ==> eliminate v2
            else
                v1 = v4;                   // Outside v1 & outside v3 ==> eliminate v1
            }

        }
    }
} // end namespace hpmc::detail

}; // end namespace hpmc

#endif // __XENOCOLLIDE_3D_H__
