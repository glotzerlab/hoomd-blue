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
    \param sweep_radius Radius of sphere to sweep the shapes by
 */
template <class Shape,
          class SupportFuncA, class SupportFuncB,
          class ProjectionFuncA, class ProjectionFuncB>
DEVICE inline bool map_two(const Shape& a, const Shape& b,
    const SupportFuncA& sa, const SupportFuncB& sb,
    const ProjectionFuncA& pa, const ProjectionFuncB& pb,
    const vec3<OverlapReal>& dr, unsigned int &err,
    Scalar sweep_radius)
    {
    quat<OverlapReal> qa(a.orientation);
    quat<OverlapReal> qb(b.orientation);
    quat<OverlapReal> q(conj(qa)* qb);
    vec3<OverlapReal> dr_rot(rotate(conj(qa), dr));

    OverlapReal r = sweep_radius;

    CompositeSupportFunc3D<SupportFuncA, SupportFuncB> S(sa, sb, dr_rot, q);
    vec3<OverlapReal> p(0,0,0);

    unsigned int it = 0;
    err = 0;

    const OverlapReal tol(1e-7);

    vec3<OverlapReal> v;

    while (it++ <= MAP_3D_MAX_ITERATIONS)
        {
        vec3<OverlapReal> proj = rotate(qa,pa(rotate(conj(qa),p)));

        p = dr+rotate(qb,pb(rotate(conj(qb),proj-dr)));

        if (dot(p-proj,p-proj) <= OverlapReal(4.0)*r*r+tol)
            {
            // the point p is in the intersection
            return true;
            }

        // conversely, check if we found a separating hyperplane
        v = rotate(conj(qa),proj - p);
        if (dot(S(v), v) < -r)
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
    \param sweep_radius Radius of sphere to sweep all shapes by

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
    const vec3<OverlapReal>& ab_t, const vec3<OverlapReal>& ac_t, unsigned int &err,
    Scalar sweep_radius)
    {
    quat<OverlapReal> qa(a.orientation);
    quat<OverlapReal> qb(b.orientation);
    quat<OverlapReal> qc(c.orientation);

    OverlapReal r = sweep_radius;

    // element of the cartesian product C = A x B x C
    vec3<OverlapReal> q_a;
    vec3<OverlapReal> q_b;
    vec3<OverlapReal> q_c;

    // element of the diagonal space D
    vec3<OverlapReal> b_diag(0.0,0.0,0.0);

    unsigned int it = 0;
    err = 0;

    vec3<OverlapReal> v_a, v_b, v_c;

    const OverlapReal tol(1e-7);

    while (it++ <= MAP_3D_MAX_ITERATIONS)
        {
        // first step: project b onto product space
        q_a = rotate(qa,pa(rotate(conj(qa),b_diag)));
        q_b = ab_t+rotate(qb,pb(rotate(conj(qb),b_diag-ab_t)));
        q_c = ac_t+rotate(qc,pc(rotate(conj(qc),b_diag-ac_t)));

        // second step: project back into diagonal space (barycenter)
        b_diag = (q_a+q_b+q_c)/OverlapReal(3.0);

        // test if all closest points lie in the sphere
        if ((dot(q_a-b_diag,q_a-b_diag) <= r*r+tol) &&
            (dot(q_b-b_diag,q_b-b_diag) <= r*r+tol) &&
            (dot(q_c-b_diag,q_c-b_diag) <= r*r+tol))
            return true;

        // if not, check if we found a separating hyperplane between C and the linear subspace D

        // get the support vertex in the direction b - q
        v_a = rotate(qa,sa(rotate(conj(qa),b_diag - q_a)));
        v_b = ab_t + rotate(qb,sb(rotate(conj(qb),b_diag - q_b)));
        v_c = ac_t + rotate(qc,sc(rotate(conj(qc),b_diag - q_c)));

        const OverlapReal sqrt3(1.7320508075688772935);
        OverlapReal norm = fast::sqrt(dot(b_diag-q_a,b_diag-q_a)+dot(b_diag-q_b,b_diag-q_b)+dot(b_diag-q_c,b_diag-q_c));
        if ( (dot(b_diag-v_a, b_diag - q_a) + dot(b_diag-v_b, b_diag - q_b) + dot(b_diag-v_c, b_diag - q_c)) > sqrt3*r*norm)
            return false;   // found a separating plane
        }

    // maximum number of iterations reached, return overlap and indicate error
    err++;
    return true;
    }

} // end namespace hpmc::detail

}; // end namespace hpmc

#endif // __MAP_3D_H__
