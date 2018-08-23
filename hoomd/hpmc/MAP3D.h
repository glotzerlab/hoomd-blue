// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "hoomd/HOOMDMath.h"
#include "HPMCPrecisionSetup.h"
#include "hoomd/VectorMath.h"

#include "hoomd/hpmc/MinkowskiMath.h"

#ifndef __MAP_3D_H__
#define __MAP_3D_H__

/*! \file MAP3D.h
    \brief Implements the method of alternating projections for convex sets in 3D
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

namespace hpmc
{

namespace detail
{

const unsigned int MAP_3D_MAX_ITERATIONS = 1024;

//! Test for pairwise overlap using the method of alternating projections
/*! \param a First shape to test for intersection
    \param b Second shape to test for intersection
    \param sa first support function
    \param sb second support function
    \param pa first projection function
    \param pb second projection function
    \param dr Position of second shape relative to first
    \param err Output variable that is incremented upon non-convergence
 */
template <class Shape,
          class SupportFuncA, class SupportFuncB,
          class ProjectionFuncA, class ProjectionFuncB>
bool map_two(const Shape& a, const Shape& b,
    const SupportFuncA& sa, const SupportFuncB& sb,
    const ProjectionFuncA& pa, const ProjectionFuncB& pb,
    vec3<OverlapReal>& dr, unsigned int &err)
    {
    quat<OverlapReal> qa(a.orientation);
    quat<OverlapReal> qb(b.orientation);
    quat<OverlapReal> q(conj(qa)* qb);
    vec3<OverlapReal> dr_rot(rotate(conj(qa), dr));

    CompositeSupportFunc3D<SupportFuncA, SupportFuncB> S(sa, sb, dr_rot, q);
    vec3<OverlapReal> p(0,0,0);

    unsigned int it = 0;
    err = 0;

    const OverlapReal root_tol = 3e-4;   // square root of precision tolerance

    vec3<OverlapReal> v;

    while (it++ <= MAP_3D_MAX_ITERATIONS)
        {
        vec3<OverlapReal> proj = rotate(qa,pa(rotate(conj(qa),p)));

        p = dr+rotate(qb,pb(rotate(conj(qb),proj-dr)));

        if (fabs(p.x - proj.x) <= root_tol && fabs(p.y - proj.y) <= root_tol && fabs(p.z - proj.z) <= root_tol)
            {
            // the point p is in the intersection
            return true;
            }

        // conversely, check if we found a separating hyperplane
        v = rotate(conj(qa),proj - p);
        if (dot(S(v), v) < OverlapReal(0.0))
            {
            return false;   // origin is outside v1 support plane
            }
        }

    // maximum number of iterations reached, return overlap and indicate error
    err++;
    return true;
    }

} // end namespace hpmc::detail

}; // end namespace hpmc

#endif // __MAP_3D_H__
