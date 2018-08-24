// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "hoomd/HOOMDMath.h"
#include "HPMCPrecisionSetup.h"
#include "hoomd/VectorMath.h"

#include "hoomd/hpmc/MinkowskiMath.h"

#ifndef __MAP_3D_H__
#define __MAP_3D_H__

/*! \file MAP3D.h
    \brief Implements von Neumann's method of alternating projections for convex sets in 3D
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
DEVICE inline bool map_two(const Shape& a, const Shape& b,
    const SupportFuncA& sa, const SupportFuncB& sb,
    const ProjectionFuncA& pa, const ProjectionFuncB& pb,
    const vec3<OverlapReal>& dr, unsigned int &err)
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

//! Test for a common point in the intersection of three shapes using the method of barycenters
/*! \param a First shape to test for intersection
    \param b Second shape to test for intersection
    \param sa first support function
    \param sb second support function
    \param pa first projection function
    \param pb second projection function
    \param ab_t Position of second shape relative to first
    \param ac_t Position of third shape relative to first
    \param err Output variable that is incremented upon non-convergence

    see Pierra, G. Mathematical Programming (1984) 28: 96. http://doi.org/10.1007/BF02612715

    as cited in
    Bauschke, H.H. & Borwein, J.M. Set-Valued Anal (1993) 1: 185. http://doi.org/10.1007/BF01027691
 */
template <class ShapeA, class ShapeB, class ShapeC,
          class SupportFuncA, class SupportFuncB, class SupportFuncC,
          class ProjectionFuncA, class ProjectionFuncB, class ProjectionFuncC>
DEVICE inline bool map_three(const ShapeA& a, const ShapeB& b, const ShapeC& c,
    const SupportFuncA& sa, const SupportFuncB& sb, const SupportFuncC& sc,
    const ProjectionFuncA& pa, const ProjectionFuncB& pb, const ProjectionFuncC& pc,
    const vec3<OverlapReal>& ab_t, const vec3<OverlapReal>& ac_t, unsigned int &err)
    {
    quat<OverlapReal> qa(a.orientation);
    quat<OverlapReal> qb(b.orientation);
    quat<OverlapReal> qc(c.orientation);

    // element of the cartesian product C = A x B x C
    vec3<OverlapReal> q_a;
    vec3<OverlapReal> q_b;
    vec3<OverlapReal> q_c;

    // element of the diagonal space D
    vec3<OverlapReal> b_diag;

    unsigned int it = 0;
    err = 0;

    const OverlapReal tol = 3e-4;   // square root of precision tolerance

    vec3<OverlapReal> v_a, v_b, v_c;

    while (it++ <= MAP_3D_MAX_ITERATIONS)
        {
        // first step: project b on to product space
        q_a = rotate(qa,pa(rotate(conj(qa),b_diag)));
        q_b = ab_t+rotate(qa,pa(rotate(conj(qb),b_diag-ab_t)));
        q_c = ac_t+rotate(qa,pa(rotate(conj(qc),b_diag-ac_t)));

        // second step: project back into diagonal space (barycenter)
        b_diag = (q_a+q_b+q_c)/OverlapReal(3.0);

        if (fabs(q_a.x - b_diag.x) <= tol && fabs(q_a.y - b_diag.y) <= tol && fabs(q_a.z - b_diag.z) <= tol &&
            fabs(q_b.x - b_diag.x) <= tol && fabs(q_b.y - b_diag.y) <= tol && fabs(q_b.z - b_diag.z) <= tol &&
            fabs(q_c.x - b_diag.x) <= tol && fabs(q_c.y - b_diag.y) <= tol && fabs(q_c.z - b_diag.z) <= tol)
            {
            // the point b is a common point
            return true;
            }

        // conversely, check if we found a separating hyperplane between C and D

        // get the support vertex in the direction b - q
        v_a = rotate(qa,sa(rotate(conj(qa),b_diag - q_a)));
        v_b = ab_t + rotate(qb,sb(rotate(conj(qb),b_diag - q_b)));
        v_c = ac_t + rotate(qc,sc(rotate(conj(qc),b_diag - q_c)));

        if ( (dot(v_a-b_diag, b_diag-q_a) + dot(v_b-b_diag, b_diag - q_b) + dot(v_c-b_diag, b_diag-q_c)) < OverlapReal(0.0))
            return false;   // found a separating plane
        }

    // maximum number of iterations reached, return overlap and indicate error
    err++;
    return true;
    }

} // end namespace hpmc::detail

}; // end namespace hpmc

#endif // __MAP_3D_H__
