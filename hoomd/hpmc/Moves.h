// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/HOOMDMath.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/VectorMath.h"

#include "hoomd/AABB.h"
#include "hoomd/BoxDim.h"
#include "hoomd/hpmc/OBB.h"

/*! \file Moves.h
    \brief Trial move generators
*/

#ifndef __MOVES_H__
#define __MOVES_H__

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host
// compiler
#ifdef __HIPCC__
#define DEVICE __device__
#else
#define DEVICE
#endif

namespace hoomd
    {
//! hpmc namespace
namespace hpmc
    {
//! Translation move
/*! \param v Vector to translate (in/out)
    \param rng RNG to utilize in the move
    \param d Maximum move distance
    \param dim Dimension

    When \a dim == 2, only x and y components are moved.
*/
DEVICE inline void move_translate(vec3<Scalar>& v, RandomGenerator& rng, Scalar d, unsigned int dim)
    {
    hoomd::UniformDistribution<Scalar> uniform(-d, d);

    // Generate a random vector inside a sphere of radius d
    vec3<Scalar> dr(Scalar(0.0), Scalar(0.0), Scalar(0.0));
    do
        {
        dr.x = uniform(rng);
        dr.y = uniform(rng);
        if (dim != 2)
            dr.z = uniform(rng);
        } while (dot(dr, dr) > d * d);

    // apply the move vector
    v += dr;
    }

//! Rotation move
/*! \param orientation Quaternion to rotate (in/out)
    \param rng RNG to utilize in the move
    \param a Rotation magnitude
    \param dim Dimension

    When \a dim == 2, a random rotation about (0,0,1) is generated. When \a dim == 3 a random 3D
   rotation is generated.
*/
template<unsigned int dim>
DEVICE void move_rotate(quat<Scalar>& orientation, RandomGenerator& rng, Scalar a)
    {
    if (dim == 2)
        {
        a /= Scalar(2.0);
        Scalar alpha = hoomd::UniformDistribution<Scalar>(-a, a)(rng);
        ;
        quat<Scalar> q(fast::cos(alpha),
                       fast::sin(alpha)
                           * vec3<Scalar>(Scalar(0), Scalar(0), Scalar(1))); // rotation quaternion
        orientation = orientation * q;
        orientation = orientation * (fast::rsqrt(norm2(orientation)));
        }
    else
        {
        // Based on Karney 2007: doi.org/10.1016/j.jmgm.2006.04.002
        hoomd::NormalDistribution<Scalar> normal(a);
        quat<Scalar> small_rotation = quat<Scalar>();

        while (true)
            {
            vec3<Scalar> s = vec3<Scalar>(normal(rng), normal(rng), normal(rng));
            Scalar theta = fast::sqrt(dot(s, s));

            // Reject moves with |s| > pi to ensure detailed balance. Otherwise, there
            // is a shorter path between the proposed move and the start (theta - pi),
            // which has a different rejection probability.
            if (theta > M_PI)
                {
                continue;
                }

            // Lift the normally distributed values to SO(3) with the exponential map
            Scalar half_theta = 0.5 * theta;
            Scalar w = fast::cos(half_theta);

            Scalar v_factor = fast::sin(half_theta) / theta;
            if (!isfinite(v_factor))
                {
                v_factor = 0.5;
                }

            vec3<Scalar> v = s * v_factor;
            small_rotation = quat<Scalar>(w, v);
            break;
            }

        orientation = small_rotation * orientation;

        // renormalize
        orientation = orientation * (fast::rsqrt(norm2(orientation)));
        }
    }

//! Helper function to test if a particle is in an active region
/*! \param pos Position of the particle
    \param box simulation box
    \param ghost_fraction Fraction of the box in the inactive zone
*/
DEVICE inline bool isActive(Scalar3 pos, const BoxDim& box, Scalar3 ghost_fraction)
    {
    // Determine if particle is in the active region
    Scalar3 f = box.makeFraction(pos);
    uchar3 periodic = box.getPeriodic();
    if ((!periodic.x && (f.x < Scalar(0.0) || f.x >= (Scalar(1.0) - ghost_fraction.x)))
        || (!periodic.y && (f.y < Scalar(0.0) || f.y >= (Scalar(1.0) - ghost_fraction.y)))
        || (!periodic.z && (f.z < Scalar(0.0) || f.z >= (Scalar(1.0) - ghost_fraction.z))))
        {
        return false;
        }
    return true;
    }

//! Helper function to generate a random element of SO(3)
// see Shoemake, Uniform random rotations, Graphics Gems III, p.142-132
// and
// https://math.stackexchange.com/questions/131336/uniform-random-quaternion-in-a-restricted-angle-range
template<class RNG>
DEVICE inline quat<Scalar> generateRandomOrientation(RNG& rng, unsigned int ndim)
    {
    // 2D just needs a random rotation in the plane
    if (ndim == 2)
        {
        Scalar angle = hoomd::UniformDistribution<Scalar>(-M_PI, M_PI)(rng);
        vec3<Scalar> axis(Scalar(0), Scalar(0), Scalar(1));
        return quat<Scalar>::fromAxisAngle(axis, angle);
        }
    else
        {
        Scalar u1 = hoomd::detail::generate_canonical<Scalar>(rng);
        Scalar u2 = hoomd::detail::generate_canonical<Scalar>(rng);
        Scalar u3 = hoomd::detail::generate_canonical<Scalar>(rng);
        return quat<Scalar>(
            fast::sqrt(u1) * fast::cos(Scalar(2.0 * M_PI) * u3),
            vec3<Scalar>(fast::sqrt(Scalar(1.0) - u1) * fast::sin(Scalar(2.0 * M_PI) * u2),
                         fast::sqrt(Scalar(1.0 - u1)) * fast::cos(Scalar(2.0 * M_PI) * u2),
                         fast::sqrt(u1) * fast::sin(Scalar(2.0 * M_PI) * u3)));
        }
    }

/*! Reflect a point in R3 around a line (pi rotation), given by a point p through which it passes
    and a rotation quaternion
 */
DEVICE inline vec3<Scalar> lineReflection(vec3<Scalar> pos, vec3<Scalar> p, quat<Scalar> q)
    {
    // find closest point on line
    vec3<Scalar> n = q.v;
    Scalar t = dot(pos - p, n);
    vec3<Scalar> r = p + t * n;

    // pivot around that point
    return r - (pos - r);
    }

    }; // end namespace hpmc
    } // namespace hoomd

#undef DEVICE

#endif //__MOVES_H__
