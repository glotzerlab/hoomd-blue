// Copyright (c) 2009-2019 The Regents of the University of Michigan
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
#ifdef __HIPCC__
#define DEVICE __device__
#else
#define DEVICE
#endif

namespace hpmc
{

namespace detail
{

const unsigned int MAP_3D_MAX_ITERATIONS = 1024;

//! Test for a common point in the intersection of three shapes using the extrapolated parallel projections method
/*! \param a First shape to test
    \param b Second shape to test
    \param c Third shape to test
    \param sa first support function
    \param sb second support function
    \param sc third support function
    \param pa first projection function
    \param pb second projection function
    \param pc third projection function
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
    vec3<OverlapReal> b_prime_diag;
    vec3<OverlapReal> u_diag(0.0,0.0,0.0);

    unsigned int it = 0;

    vec3<OverlapReal> v_a, v_b, v_c;

    const OverlapReal tol(4e-7); // for squares of distances, in *double* precision

    /*
     * Tuning parameters
     */

    // amount of extrapolation to apply (0 <= lambda <=1)
    OverlapReal lambda(1.0);
    const unsigned int k = 10;
    const OverlapReal f(0.9);

    OverlapReal sum;
    OverlapReal normsq;

    while (it++ <= MAP_3D_MAX_ITERATIONS)
        {
        // first step: project b onto product space
        q_a = rotate(qa,pa(rotate(conj(qa),u_diag)));
        q_b = ab_t+rotate(qb,pb(rotate(conj(qb),u_diag-ab_t)));
        q_c = ac_t+rotate(qc,pc(rotate(conj(qc),u_diag-ac_t)));

        // test for common point
        if (dot(q_a-u_diag,q_a-u_diag) + dot(q_b-u_diag,q_b-u_diag) + dot(q_c-u_diag,q_c-u_diag) <= tol)
            return true;

        // project back into diagonal space (barycenter)
        b_prime_diag = (q_a+q_b+q_c)/OverlapReal(3.0);

        //  check if we found a separating hyperplane between C and the linear subspace D

        // get the support vertex in the direction b' - q
        sum = OverlapReal(0.0);

        vec3<OverlapReal> n;
        n = b_prime_diag - q_a;
        if (dot(n,n) > OverlapReal(0.0))
            {
            v_a = rotate(qa,sa(rotate(conj(qa),b_prime_diag - q_a)));
            sum += dot(b_prime_diag-v_a, b_prime_diag - q_a);
            }

        n = b_prime_diag - q_b;
        if (dot(n,n) > OverlapReal(0.0))
            {
            v_b = ab_t + rotate(qb,sb(rotate(conj(qb),b_prime_diag - q_b)));
            sum += dot(b_prime_diag-v_b, b_prime_diag - q_b);
            }

        n = b_prime_diag - q_c;
        if (dot(n,n) > OverlapReal(0.0))
            {
            v_c = ac_t + rotate(qc,sc(rotate(conj(qc),b_prime_diag - q_c)));
            sum += dot(b_prime_diag-v_c, b_prime_diag - q_c);
            }

        normsq = dot(b_prime_diag-q_a,b_prime_diag-q_a)+
            dot(b_prime_diag-q_b,b_prime_diag-q_b)+
            dot(b_prime_diag-q_c,b_prime_diag-q_c);
        if (sum > -fast::sqrt(normsq*tol))
            return false;   // found a separating plane

        // second step, extrapolation
        OverlapReal coeff = dot(q_a-u_diag,q_a-u_diag) + dot(q_b-u_diag,q_b-u_diag) + dot(q_c-u_diag,q_c-u_diag);
        coeff /= OverlapReal(3.0)*dot(b_prime_diag-u_diag,b_prime_diag-u_diag);

        u_diag += (OverlapReal(1.0) + lambda*(coeff-OverlapReal(1.0)))*(b_prime_diag-u_diag);

        //  reduce lambda every k iterations
        if (it % k == 0)
            lambda *= f;
        }

    // maximum number of iterations reached, return overlap and indicate error
    err++;
    return true;
    }

//! Test for pairwise overlap using the extrapolated parallel projections method
/*! \param a First shape to test for intersection
    \param b Second shape to test for intersection
    \param sa first support function
    \param sb second support function
    \param pa first projection function
    \param pb second projection function
    \param ab_t Position of second shape relative to first
    \param err Output variable that is incremented upon non-convergence
 */
template <class ShapeA, class ShapeB,
          class SupportFuncA, class SupportFuncB,
          class ProjectionFuncA, class ProjectionFuncB>
DEVICE inline bool map_two(const ShapeA& a, const ShapeB& b,
    const SupportFuncA& sa, const SupportFuncB& sb,
    const ProjectionFuncA& pa, const ProjectionFuncB& pb,
    const vec3<OverlapReal>& ab_t, unsigned int &err)
    {
    quat<OverlapReal> qa(a.orientation);
    quat<OverlapReal> qb(b.orientation);

    // element of the cartesian product C = A x B
    vec3<OverlapReal> q_a;
    vec3<OverlapReal> q_b;

    // element of the diagonal space D
    vec3<OverlapReal> b_prime_diag;
    vec3<OverlapReal> u_diag(0.0,0.0,0.0);

    unsigned int it = 0;

    vec3<OverlapReal> v_a, v_b;

    // this seems to be a reasonable lower bound on the single-precision error, when
    // reducing further one should see an increased number of forced loop terminations
    const OverlapReal tol(4e-7); // for squares of distances

    /*
     * Tuning parameters
     */

    // amount of extrapolation to apply (0 <= lambda <=1)
    OverlapReal lambda(1.0);
    const unsigned int k = 20;
    const OverlapReal f(0.999);

    OverlapReal sum;
    OverlapReal normsq;

    while (it++ <= MAP_3D_MAX_ITERATIONS)
        {
        // first step: project b onto product space
        q_a = rotate(qa,pa(rotate(conj(qa),u_diag)));
        q_b = ab_t+rotate(qb,pb(rotate(conj(qb),u_diag-ab_t)));

        // test for common point
        if (dot(q_a-u_diag,q_a-u_diag) + dot(q_b-u_diag,q_b-u_diag) <= tol)
            return true;

        // project back into diagonal space (barycenter)
        b_prime_diag = (q_a+q_b)/OverlapReal(2.0);

        //  check if we found a separating hyperplane between C and the linear subspace D

        // get the support vertex in the direction b' - q
        sum = OverlapReal(0.0);

        vec3<OverlapReal> n;
        n = b_prime_diag - q_a;
        if (dot(n,n) > OverlapReal(0.0))
            {
            v_a = rotate(qa,sa(rotate(conj(qa),b_prime_diag - q_a)));
            sum += dot(b_prime_diag-v_a, b_prime_diag - q_a);
            }

        n = b_prime_diag - q_b;
        if (dot(n,n) > OverlapReal(0.0))
            {
            v_b = ab_t + rotate(qb,sb(rotate(conj(qb),b_prime_diag - q_b)));
            sum += dot(b_prime_diag-v_b, b_prime_diag - q_b);
            }

        normsq = dot(b_prime_diag-q_a,b_prime_diag-q_a)+dot(b_prime_diag-q_b,b_prime_diag-q_b);
        if (sum > -fast::sqrt(normsq*tol))
            return false;   // found a separating plane

        // second step, extrapolation
        OverlapReal coeff = dot(q_a-u_diag,q_a-u_diag) + dot(q_b-u_diag,q_b-u_diag);
        coeff /= OverlapReal(2.0)*dot(b_prime_diag-u_diag,b_prime_diag-u_diag);

        u_diag += (OverlapReal(1.0) + lambda*(coeff-OverlapReal(1.0)))*(b_prime_diag-u_diag);

        //  reduce lambda every k iterations
        if (it % k == 0)
            lambda *= f;
        }

    // maximum number of iterations reached, return overlap and indicate error
    err++;
    return true;
    }

} // end namespace hpmc::detail

}; // end namespace hpmc

#endif // __MAP_3D_H__
