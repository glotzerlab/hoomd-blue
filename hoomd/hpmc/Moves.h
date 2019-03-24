// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"

#include "hoomd/AABB.h"
#include "hoomd/hpmc/HPMCPrecisionSetup.h"
#include "hoomd/hpmc/OBB.h"

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

/* Generate a uniformly distributed random position in a sphere
 * \param rng Saru RNG
 * \param pos_sphere Center of insertion sphere
 * \param R radius of insertion sphere
 */
template<class RNG>
DEVICE inline vec3<Scalar> generatePositionInSphere(RNG& rng, vec3<Scalar> pos_sphere, Scalar R)
    {
    // draw a random vector in the excluded volume sphere of the colloid
    Scalar theta = rng.template s<Scalar>(Scalar(0.0),Scalar(2.0*M_PI));
    Scalar z = rng.template s<Scalar>(Scalar(-1.0),Scalar(1.0));

    // random normalized vector
    vec3<Scalar> n(fast::sqrt(Scalar(1.0)-z*z)*fast::cos(theta),fast::sqrt(Scalar(1.0)-z*z)*fast::sin(theta),z);

    // draw random radial coordinate in test sphere
    Scalar r3 = rng.template s<Scalar>(0,Scalar(1.0));
    Scalar r = R*fast::pow(r3,Scalar(1.0/3.0));

    // test depletant position
    vec3<Scalar> pos_in_sphere = pos_sphere+r*n;

    return pos_in_sphere;
    }

/* Generate a uniformly distributed random position in a spherical cap
 *
 * \param rng The random number generator
 * \param pos_sphere Center of sphere
 * \param R radius of sphere
 * \param h height of spherical cap (0<h<2*R)
 * \param d Vector normal to the cap
 */
template<class RNG>
inline vec3<Scalar> generatePositionInSphericalCap(RNG& rng, const vec3<Scalar>& pos_sphere,
     Scalar R, Scalar h, const vec3<Scalar>& d)
    {
    // pick a z coordinate in the spherical cap s.t. V(z) ~ uniform
    Scalar theta = Scalar(2.0*M_PI)*rng.template s<Scalar>();
    Scalar R3=R*R*R;
    Scalar V_cap = Scalar(M_PI/3.0)*h*h*(Scalar(3.0)*R-h);
    Scalar V = V_cap*rng.template s<Scalar>();
    const Scalar sqrt3(1.7320508075688772935);

    // convert the cap volume into a z coordinate in the sphere, using the correct root of the cubic polynomial
    Scalar arg = Scalar(1./3.)*atan2(fast::sqrt((Scalar(4.0*M_PI)*R3-Scalar(3.0)*V)*3*V),Scalar(2.0*M_PI)*R3-Scalar(3.0)*V);
    Scalar z = R*(fast::cos(arg)-sqrt3*fast::sin(arg));

    // pick a point in disk of radius sqrt(R^2-z^2)
    Scalar r = fast::sqrt(rng.template s<Scalar>()*(R*R-z*z));

    // unit vector in cap direction
    vec3<Scalar> n = d/sqrt(dot(d,d));

    // find two unit vectors normal to n
    vec3<Scalar> ez(0,0,1);
    vec3<Scalar> n1, n2;
    vec3<Scalar> c = cross(n,ez);
    if (dot(c,c)==0.0)
        {
        n1 = vec3<Scalar>(1,0,0);
        n2 = vec3<Scalar>(0,1,0);
        }
    else
        {
        n1 = c/sqrt(dot(c,c));
        c = cross(n,n1);
        n2 = c/sqrt(dot(c,c));
        }

    vec3<Scalar> r_cone = n1*r*cos(theta)+n2*r*sin(theta)+n*z;

    // test depletant position
    return pos_sphere+r_cone;
    }

/* Generate a uniformly distributed random position in an AABB
 *
 * \param rng The random number generator
 * \param aabb The AABB to sample in
 */
template<class RNG>
DEVICE inline vec3<Scalar> generatePositionInAABB(RNG& rng, const detail::AABB& aabb)
    {
    vec3<Scalar> p;
    vec3<Scalar> lower = aabb.getLower();
    vec3<Scalar> upper = aabb.getUpper();

    p.x = rng.template s<Scalar>(lower.x, upper.x);
    p.y = rng.template s<Scalar>(lower.y, upper.y);
    p.z = rng.template s<Scalar>(lower.z, upper.z);

    return p;
    }

/* Generate a uniformly distributed random position in an OBB
 *
 * \param rng The random number generator
 * \param aabb The OBB to sample in
 * \param dim Dimensionality of system
 */
template<class RNG>
DEVICE inline vec3<OverlapReal> generatePositionInOBB(RNG& rng, const detail::OBB& obb, unsigned int dim)
    {
    vec3<OverlapReal> p;
    vec3<OverlapReal> lower = -obb.lengths;
    vec3<OverlapReal> upper = obb.lengths;

    p.x = rng.template s<OverlapReal>(lower.x, upper.x);
    p.y = rng.template s<OverlapReal>(lower.y, upper.y);
    if (dim == 3)
        p.z = rng.template s<OverlapReal>(lower.z, upper.z);
    else
        p.z = OverlapReal(0.0);

    return rotate(obb.rotation,p)+obb.center;
    }

/*! Reflect a point in R3 around a line (pi rotation), given by a point p through which it passes
    and a rotation quaternion
 */
inline vec3<Scalar> lineReflection(vec3<Scalar> pos, vec3<Scalar> p, quat<Scalar> q)
    {
    // find closest point on line
    vec3<Scalar> n = q.v;
    Scalar t = dot(pos-p,n);
    vec3<Scalar> r = p + t*n;

    // pivot around that point
    return r - (pos - r);
    }

}; // end namespace hpmc

#undef DEVICE

#endif //__MOVES_H__
