// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"

/*! \file Moves.h
    \brief Trial move generators
*/

#ifndef __MOVES_H__
#define __MOVES_H__

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

//! hpmc namespace
namespace hpmc
{

//! Translation move
/*! \param v Vector to translate (in/out)
    \param rng Saru RNG to utilize in the move
    \param d Maximum move distance
    \param dim Dimension

    When \a dim == 2, only x and y components are moved.
*/
template <class RNG>
DEVICE inline void move_translate(vec3<Scalar>& v, RNG& rng, Scalar d, unsigned int dim)
    {
    // Generate a random vector inside a sphere of radius d
    vec3<Scalar> dr(Scalar(0.0), Scalar(0.0), Scalar(0.0));
    do
        {
        dr.x = rng.s(-d, d);
        dr.y = rng.s(-d, d);
        if (dim != 2)
            dr.z = rng.s(-d, d);
        } while(dot(dr,dr) > d*d);

    // apply the move vector
    v += dr;
    }

//! Rotation move
/*! \param orientation Quaternion to rotate (in/out)
    \param rng Saru RNG to utilize in the move
    \param a Rotation magnitude
    \param dim Dimension

    When \a dim == 2, a random rotation about (0,0,1) is generated. When \a dim == 3 a random 3D rotation is generated.
*/
template <class RNG>
DEVICE void move_rotate(quat<Scalar>& orientation, RNG& rng, Scalar a, unsigned int dim)
    {
    if (dim==2)
        {
        a /= Scalar(2.0);
        Scalar alpha = rng.s(-a, a);
        quat<Scalar> q(cosf(alpha), (Scalar)sinf(alpha) * vec3<Scalar>(Scalar(0),Scalar(0),Scalar(1))); // rotation quaternion
        orientation = orientation * q;
        orientation = orientation * (fast::rsqrt(norm2(orientation)));
        }
    else
        {
        // Frenkel and Smit reference Allen and Tildesley, referencing Vesley(1982), referencing Marsaglia(1972).
        // Generate a random unit quaternion. Scale it to a small rotation and apply.
        quat<Scalar> q;
        Scalar s1, s2, s3;

        do
            {
            q.s = rng.s(Scalar(-1.0),Scalar(1.0));
            q.v.x = rng.s(Scalar(-1.0),Scalar(1.0));
            }
        while ((s1 = q.s * q.s + q.v.x * q.v.x) >= Scalar(1.0));

        do
            {
            q.v.y = rng.s(Scalar(-1.0),Scalar(1.0));
            q.v.z = rng.s(Scalar(-1.0),Scalar(1.0));
            }
        while ((s2 = q.v.y * q.v.y + q.v.z * q.v.z) >= Scalar(1.0) || s2 == Scalar(0.0));

        s3 = fast::sqrt((Scalar(1.0) - s1) / s2);
        q.v.y *= s3;
        q.v.z *= s3;

        // generate new trial orientation
        orientation += a * q;

        // renormalize
        orientation = orientation * (fast::rsqrt(norm2(orientation)));
        }
    }

//! Select a random index
/*! \param rng Saru RNG to utilize in the move
    \param max Maximum index to select
    \returns a random number 0 <= i <= max with uniform probability.

    **Method**

    First, round max+1 up to the next nearest power of two -> max2. Then draw random numbers in the range [0 ... max2)
    using 32-but random values and a bitwise and with max2-1. Return the first random number found in the range.
*/
template <class RNG>
DEVICE inline unsigned int rand_select(RNG& rng, unsigned int max)
    {
    // handle degenerate case where max==0
    if (max == 0)
        return 0;

    // algorithm to round up to the nearest power of two from https://en.wikipedia.org/wiki/Power_of_two
    unsigned int n = max+1;
    n = n - 1;
    n = n | (n >> 1);
    n = n | (n >> 2);
    n = n | (n >> 4);
    n = n | (n >> 8);
    n = n | (n >> 16);
    // Note: leaving off the n = n + 1 because we are going to & with next highest power of 2 -1

    unsigned int result;
    do
        {
        result = rng.u32() & n;
        } while(result > max);

    return result;
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
    if ( (!periodic.x && (f.x < Scalar(0.0) || f.x >= (Scalar(1.0) - ghost_fraction.x))) ||
         (!periodic.y && (f.y < Scalar(0.0) || f.y >= (Scalar(1.0) - ghost_fraction.y))) ||
         (!periodic.z && (f.z < Scalar(0.0) || f.z >= (Scalar(1.0) - ghost_fraction.z))) )
        {
        return false;
        }
    return true;
    }

//! Helper function to generate a random element of SO(3)
// see Shoemake, Uniform random rotations, Graphics Gems III, p.142-132
// and http://math.stackexchange.com/questions/131336/uniform-random-quaternion-in-a-restricted-angle-range
template<class RNG>
DEVICE inline quat<Scalar> generateRandomOrientation(RNG& rng)
    {
    Scalar u1 = rng.template s<Scalar>();
    Scalar u2 = rng.template s<Scalar>();
    Scalar u3 = rng.template s<Scalar>();
    return quat<Scalar>(fast::sqrt(u1)*fast::cos(Scalar(2.0*M_PI)*u3),
        vec3<Scalar>(fast::sqrt(Scalar(1.0)-u1)*fast::sin(Scalar(2.0*M_PI)*u2),
            fast::sqrt(Scalar(1.0-u1))*fast::cos(Scalar(2.0*M_PI)*u2),
            fast::sqrt(u1)*fast::sin(Scalar(2.0*M_PI)*u3)));

    }

}; // end namespace hpmc

#endif //__MOVES_H__
