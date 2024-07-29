// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file WallData.h
    \brief Contains declarations for all types (currently Sphere, Cylinder, and
    Plane) of WallData and associated utilities.
 */

#pragma once

#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"

#ifdef __HIPCC__
#define DEVICE __device__
#else
#define DEVICE
#endif

//! SphereWall Constructor
/*! \param r Radius of the sphere
    \param origin The x,y,z coordinates of the center of the sphere
    \param inside Determines which half space is evaluated.
*/

#ifdef SINGLE_PRECISION
#define ALIGN_SCALAR 4
#else
#define ALIGN_SCALAR 8
#endif

namespace hoomd
    {
namespace md
    {
struct __attribute__((visibility("default"))) SphereWall
    {
    SphereWall(Scalar rad = 0.0,
               Scalar3 orig = make_scalar3(0.0, 0.0, 0.0),
               bool ins = true,
               bool open_ = true)
        : origin(vec3<Scalar>(orig)), r(rad), inside(ins), open(open_)
        {
        }
    vec3<Scalar> origin; // need to order datatype in descending order of type size for Fermi
    Scalar r;
    bool inside;
    bool open;
    } __attribute__((aligned(ALIGN_SCALAR))); // align according to first member of vec3<Scalar>

//! CylinderWall Constructor
/*! \param r Radius of the sphere
    \param origin The x,y,z coordinates of a point on the cylinder axis
    \param axis A x,y,z vector along the cylinder axis used to define the axis
    \param quatAxisToZRot (Calculated not input) The quaternion which rotates the simulation space
   such that the axis of the cylinder is parallel to the z' axis \param inside Determines which half
   space is evaluated.
*/
struct __attribute__((visibility("default"))) CylinderWall
    {
    CylinderWall(Scalar rad = 0.0,
                 Scalar3 orig = make_scalar3(0.0, 0.0, 0.0),
                 Scalar3 zorient = make_scalar3(0.0, 0.0, 1.0),
                 bool ins = true,
                 bool open_ = true)
        : origin(vec3<Scalar>(orig)), axis(vec3<Scalar>(zorient)), r(rad), inside(ins), open(open_)
        {
        vec3<Scalar> zVec = axis;
        vec3<Scalar> zNorm(0.0, 0.0, 1.0);

        // method source: http://lolengine.net/blog/2014/02/24/quaternion-from-two-vectors-final
        // easily simplified due to zNorm being a normalized vector
        Scalar normVec = sqrt(dot(zVec, zVec));
        Scalar realPart = normVec + dot(zNorm, zVec);
        vec3<Scalar> w;

        if (realPart < Scalar(1.0e-6) * normVec)
            {
            realPart = Scalar(0.0);
            w = vec3<Scalar>(0.0, -1.0, 0.0);
            }
        else
            {
            w = cross(zNorm, zVec);
            realPart = Scalar(realPart);
            }
        quatAxisToZRot = quat<Scalar>(realPart, w);
        Scalar norm = fast::rsqrt(norm2(quatAxisToZRot));
        quatAxisToZRot = conj(norm * quatAxisToZRot);
        }
    quat<Scalar>
        quatAxisToZRot; // need to order datatype in descending order of type size for Fermi
    vec3<Scalar> origin;
    vec3<Scalar> axis;
    Scalar r;
    bool inside;
    bool open;
    } __attribute__((aligned(ALIGN_SCALAR))); // align according to first member of quat<Scalar>

//! ConeWall Constructor
/*! \param r1 Radius of the circle 1
    \param r2 Radius of the circle 2
    \param d distance between two circles
    \param origin The x,y,z coordinates of a point on the cylinder axis
    \param axis A x,y,z vector along the cylinder axis used to define the axis
    \param quatAxisToZRot (Calculated not input) The quaternion which rotates the simulation space
   such that the axis of the cylinder is parallel to the z' axis \param inside Determines which half
   space is evaluated.
*/
struct __attribute__((visibility("default"))) ConeWall
    {
    ConeWall(Scalar rad1 = 0.0,
             Scalar rad2 = 0.0,
             Scalar dist = 0.0,
             Scalar3 orig = make_scalar3(0.0, 0.0, 0.0),
             Scalar3 zorient = make_scalar3(0.0, 0.0, 1.0),
             bool ins = true,
             bool open_ = true)
        : origin(vec3<Scalar>(orig)), axis(vec3<Scalar>(zorient)), r1(rad1), r2(rad2), d(dist), inside(ins), open(open_)
        {
        vec3<Scalar> zVec = axis;
        vec3<Scalar> zNorm(0.0, 0.0, 1.0);

        // method source: http://lolengine.net/blog/2014/02/24/quaternion-from-two-vectors-final
        // easily simplified due to zNorm being a normalized vector
        Scalar normVec = sqrt(dot(zVec, zVec));
        Scalar realPart = normVec + dot(zNorm, zVec);
        vec3<Scalar> w;

        if (realPart < Scalar(1.0e-6) * normVec)
            {
            realPart = Scalar(0.0);
            w = vec3<Scalar>(0.0, -1.0, 0.0);
            }
        else
            {
            w = cross(zNorm, zVec);
            realPart = Scalar(realPart);
            }
        quatAxisToZRot = quat<Scalar>(realPart, w);
        Scalar norm = fast::rsqrt(norm2(quatAxisToZRot));
        quatAxisToZRot = conj(norm * quatAxisToZRot);
        
        angle = fast::atan((r2 - r1)/d)
        }
    quat<Scalar>
        quatAxisToZRot; // need to order datatype in descending order of type size for Fermi
    Scalar angle;
    vec3<Scalar> origin;
    vec3<Scalar> axis;
    Scalar r1;
    Scalar r2;
    Scalar d;
    bool inside;
    bool open;
    } __attribute__((aligned(ALIGN_SCALAR))); // align according to first member of quat<Scalar>

//! PlaneWall Constructor
/*! \param origin The x,y,z coordinates of a point on the cylinder axis
    \param normal The x,y,z normal vector of the plane (normalized upon input)
*/
struct __attribute__((visibility("default"))) PlaneWall
    {
    PlaneWall(Scalar3 orig = make_scalar3(0.0, 0.0, 0.0),
              Scalar3 norm = make_scalar3(0.0, 0.0, 1.0),
              bool open_ = true)
        : normal(vec3<Scalar>(norm)), origin(vec3<Scalar>(orig)), open(open_)
        {
        vec3<Scalar> nVec;
        nVec = normal;
        Scalar invNormLength;
        invNormLength = fast::rsqrt(nVec.x * nVec.x + nVec.y * nVec.y + nVec.z * nVec.z);
        normal = nVec * invNormLength;
        }
    vec3<Scalar> normal;
    vec3<Scalar> origin;
    bool open;
    } __attribute__((aligned(ALIGN_SCALAR))); // align according to first member of vec3<Scalar>

//! Wall vector to point for a sphere wall geometry
/* Returns 0 vector when all normal directions are equal
 */
DEVICE inline Scalar3
distVectorWallToPoint(const SphereWall& wall, const vec3<Scalar>& position, bool& in_active_space)
    {
    const vec3<Scalar> dist_from_origin = position - wall.origin;
    const Scalar euclidean_dist = sqrt(dot(dist_from_origin, dist_from_origin));
    if (euclidean_dist == 0.0)
        {
        in_active_space = wall.open;
        return make_scalar3(0.0, 0.0, 0.0);
        }
    in_active_space
        = ((euclidean_dist < wall.r) && wall.inside) || ((euclidean_dist > wall.r) && !wall.inside);
    return vec_to_scalar3((1 - (wall.r / euclidean_dist)) * dist_from_origin);
    };

//! Wall vector to point for a cylinder wall geometry
/* Returns 0 vector when all normal directions are equal
 */
DEVICE inline Scalar3
distVectorWallToPoint(const CylinderWall& wall, const vec3<Scalar>& position, bool& in_active_space)
    {
    const vec3<Scalar> dist_from_origin = position - wall.origin;
    vec3<Scalar> rotated_distance = rotate(wall.quatAxisToZRot, dist_from_origin);
    rotated_distance.z = 0.0;
    const Scalar euclidean_dist = sqrt(dot(rotated_distance, rotated_distance));
    if (euclidean_dist == 0.0)
        {
        in_active_space = wall.open;
        return make_scalar3(0.0, 0.0, 0.0);
        }
    in_active_space = ((euclidean_dist < wall.r) && wall.inside)
                      || ((euclidean_dist > wall.r) && !(wall.inside));
    const vec3<Scalar> dx = (1 - (wall.r / euclidean_dist)) * rotated_distance;
    return vec_to_scalar3(rotate(conj(wall.quatAxisToZRot), dx));
    };

//! Wall vector to point for a cone wall geometry
/* Returns 0 vector when all normal directions are equal
 */
DEVICE inline Scalar3
distVectorWallToPoint(const ConeWall& wall, const vec3<Scalar>& position, bool& in_active_space)
    {
    const vec3<Scalar> dist_from_origin = position - wall.origin;
    vec3<Scalar> rotated_distance = rotate(wall.quatAxisToZRot, dist_from_origin);
    // rotated_distance.z = 0.0;
    const Scalar euclidean_dist = sqrt(dot(rotated_distance, rotated_distance));
    if (euclidean_dist == 0.0)
        {
        in_active_space = wall.open;
        return make_scalar3(0.0, 0.0, 0.0);
        }
    in_active_space = ((euclidean_dist < wall.r) && wall.inside)
                      || ((euclidean_dist > wall.r) && !(wall.inside));
    const vec3<Scalar> dx = (1 - (wall.r / euclidean_dist)) * rotated_distance;
    return vec_to_scalar3(rotate(conj(wall.quatAxisToZRot), dx));
    };

//! Wall vector to point for a plane wall geometry
DEVICE inline Scalar3
distVectorWallToPoint(const PlaneWall& wall, const vec3<Scalar>& position, bool& in_active_space)
    {
    Scalar distance = dot(wall.normal, position) - dot(wall.normal, wall.origin);
    if (distance == 0)
        {
        in_active_space = wall.open;
        return make_scalar3(0.0, 0.0, 0.0);
        }
    in_active_space = distance > 0.0;
    return vec_to_scalar3(distance * wall.normal);
    };

//! Distance of point to inside sphere wall geometry, not really distance, +- based on if it's
//! inside or not
DEVICE inline Scalar distWall(const SphereWall& wall, const vec3<Scalar>& position)
    {
    vec3<Scalar> t = position;
    t -= wall.origin;
    vec3<Scalar> shiftedPos(t);
    Scalar rxyz2
        = shiftedPos.x * shiftedPos.x + shiftedPos.y * shiftedPos.y + shiftedPos.z * shiftedPos.z;
    Scalar d = wall.r - sqrt(rxyz2);
    d = wall.inside ? d : -d;
    return d;
    };

//! Distance of point to inside cylinder wall geometry, not really distance, +- based on if it's
//! inside or not
DEVICE inline Scalar distWall(const CylinderWall& wall, const vec3<Scalar>& position)
    {
    vec3<Scalar> t = position;
    t -= wall.origin;
    vec3<Scalar> shiftedPos = rotate(wall.quatAxisToZRot, t);
    Scalar rxy2 = shiftedPos.x * shiftedPos.x + shiftedPos.y * shiftedPos.y;
    Scalar d = wall.r - sqrt(rxy2);
    d = wall.inside ? d : -d;
    return d;
    };

//! Distance of point to inside cone wall geometry, not really distance, +- based on if it's
//! inside or not
DEVICE inline Scalar distWall(const ConeWall& wall, const vec3<Scalar>& position)
    {
    vec3<Scalar> t = position;
    t -= wall.origin;
    vec3<Scalar> shiftedPos = rotate(wall.quatAxisToZRot, t);
    Scalar rxy2 = shiftedPos.x * shiftedPos.x + shiftedPos.y * shiftedPos.y;
    Scalar d = wall.r - sqrt(rxy2);
    d = wall.inside ? d : -d;
    return d;
    };

//! Distance of point to inside plane wall geometry, not really distance, +- based on if it's inside
//! or not
DEVICE inline Scalar distWall(const PlaneWall& wall, const vec3<Scalar>& position)
    {
    return dot(wall.normal, position) - dot(wall.normal, wall.origin);
    };

    } // end namespace md
    } // end namespace hoomd
