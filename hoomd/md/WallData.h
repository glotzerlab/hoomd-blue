// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: jproc

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

#ifndef __HIPCC__
#include <pybind11/pybind11.h>
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

struct __attribute__((visibility("default"))) SphereWall
    {
    SphereWall(Scalar rad = 0.0, Scalar3 orig = make_scalar3(0.0, 0.0, 0.0), bool ins = true)
        : origin(vec3<Scalar>(orig)), r(rad), inside(ins)
        {
        }
    vec3<Scalar> origin; // need to order datatype in descending order of type size for Fermi
    Scalar r;
    bool inside;
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
                 bool ins = true)
        : origin(vec3<Scalar>(orig)), axis(vec3<Scalar>(zorient)), r(rad), inside(ins)
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
        quatAxisToZRot = norm * quatAxisToZRot;
        }
    quat<Scalar>
        quatAxisToZRot; // need to order datatype in descending order of type size for Fermi
    vec3<Scalar> origin;
    vec3<Scalar> axis;
    Scalar r;
    bool inside;
    } __attribute__((aligned(ALIGN_SCALAR))); // align according to first member of quat<Scalar>

//! PlaneWall Constructor
/*! \param origin The x,y,z coordinates of a point on the cylinder axis
    \param normal The x,y,z normal vector of the plane (normalized upon input)
    \param inside Determines which half space is evaluated.
*/
struct __attribute__((visibility("default"))) PlaneWall
    {
    PlaneWall(Scalar3 orig = make_scalar3(0.0, 0.0, 0.0),
              Scalar3 norm = make_scalar3(0.0, 0.0, 1.0),
              bool ins = true)
        : normal(vec3<Scalar>(norm)), origin(vec3<Scalar>(orig)), inside(ins)
        {
        vec3<Scalar> nVec;
        nVec = normal;
        Scalar invNormLength;
        invNormLength = fast::rsqrt(nVec.x * nVec.x + nVec.y * nVec.y + nVec.z * nVec.z);
        normal = nVec * invNormLength;
        }
    vec3<Scalar> normal;
    vec3<Scalar> origin;
    bool inside;
    } __attribute__((aligned(ALIGN_SCALAR))); // align according to first member of vec3<Scalar>

//! Point to wall vector for a sphere wall geometry
/* Returns 0 vector when all normal directions are equal
 */
DEVICE inline vec3<Scalar>
vecPtToWall(const SphereWall& wall, const vec3<Scalar>& position, bool& inside)
    {
    vec3<Scalar> t = position;
    t -= wall.origin;
    vec3<Scalar> shiftedPos(t);
    Scalar rxyz = sqrt(dot(shiftedPos, shiftedPos));
    if (rxyz > 0.0)
        {
        inside = (((rxyz <= wall.r) && wall.inside) || ((rxyz > wall.r) && !(wall.inside))) ? true
                                                                                            : false;
        t *= wall.r / rxyz;
        vec3<Scalar> dx = t - shiftedPos;
        return dx;
        }
    else
        {
        inside = (wall.inside) ? true : false;
        return vec3<Scalar>(0.0, 0.0, 0.0);
        }
    };

//! Point to wall vector for a cylinder wall geometry
/* Returns 0 vector when all normal directions are equal
 */
DEVICE inline vec3<Scalar>
vecPtToWall(const CylinderWall& wall, const vec3<Scalar>& position, bool& inside)
    {
    vec3<Scalar> t = position;
    t -= wall.origin;
    vec3<Scalar> shiftedPos = rotate(wall.quatAxisToZRot, t);
    shiftedPos.z = 0.0;
    Scalar rxy = sqrt(dot(shiftedPos, shiftedPos));
    if (rxy > 0.0)
        {
        inside = (((rxy <= wall.r) && wall.inside) || ((rxy > wall.r) && !(wall.inside))) ? true
                                                                                          : false;
        t = (wall.r / rxy) * shiftedPos;
        vec3<Scalar> dx = t - shiftedPos;
        dx = rotate(conj(wall.quatAxisToZRot), dx);
        return dx;
        }
    else
        {
        inside = (wall.inside) ? true : false;
        return vec3<Scalar>(0.0, 0.0, 0.0);
        }
    };

//! Point to wall vector for a plane wall geometry
DEVICE inline vec3<Scalar>
vecPtToWall(const PlaneWall& wall, const vec3<Scalar>& position, bool& inside)
    {
    vec3<Scalar> t = position;
    Scalar d = dot(wall.normal, t) - dot(wall.normal, wall.origin);
    inside = (((d >= 0.0) && wall.inside) || ((d < 0.0) && !(wall.inside))) ? true : false;
    vec3<Scalar> dx = -d * wall.normal;
    return dx;
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
    d = (wall.inside) ? d : -d;
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
    d = (wall.inside) ? d : -d;
    return d;
    };

//! Distance of point to inside plane wall geometry, not really distance, +- based on if it's inside
//! or not
DEVICE inline Scalar distWall(const PlaneWall& wall, const vec3<Scalar>& position)
    {
    vec3<Scalar> t = position;
    Scalar d = dot(wall.normal, t) - dot(wall.normal, wall.origin);
    d = (wall.inside) ? d : -d;
    return d;
    };

#ifndef __HIPCC__
// Export all wall data types into Python. This is needed to allow for syncing Python and C++
// list/array data structures containing walls for WallPotential objects.
void export_wall_data(pybind11::module& m)
    {
    pybind11::class_<SphereWall>(m, "SphereWall")
        .def(pybind11::init(
            [](Scalar radius, pybind11::tuple origin, bool inside)
                {
                return SphereWall(
                    radius,
                    make_scalar3(origin[0].cast<Scalar>(),
                                 origin[1].cast<Scalar>(),
                                 origin[2].cast<Scalar>()),
                    inside);
                }),
            pybind11::arg("radius"), pybind11::arg("origin"), pybind11::arg("inside")
            )
        .def_property_readonly("radius", [](const SphereWall& wall) {return wall.r;} )
        .def_property_readonly("origin", [](const SphereWall& wall)
            {
            return pybind11::make_tuple(wall.origin.x, wall.origin.y, wall.origin.z);
            })
        .def_property_readonly("inside", [](const SphereWall& wall) {return wall.inside;} );

    pybind11::class_<CylinderWall>(m, "CylinderWall")
        .def(pybind11::init(
            [](Scalar radius, pybind11::tuple origin, pybind11::tuple z_orientation, bool inside)
                {
                return CylinderWall(radius,
                                  make_scalar3(origin[0].cast<Scalar>(), origin[1].cast<Scalar>(), origin[2].cast<Scalar>()),
                                  make_scalar3(z_orientation[0].cast<Scalar>(), z_orientation[1].cast<Scalar>(), z_orientation[2].cast<Scalar>()),
                                  inside);
                }),
            pybind11::arg("radius"), pybind11::arg("origin"), pybind11::arg("axis"), pybind11::arg("inside")
            )
        .def_property_readonly("radius", [](const CylinderWall& wall) {return wall.r;} )
        .def_property_readonly("origin", [](const CylinderWall& wall)
            {
            return pybind11::make_tuple(wall.origin.x, wall.origin.y, wall.origin.z);
            })
        .def_property_readonly("axis", [](const CylinderWall& wall)
            {
            return pybind11::make_tuple(wall.axis.x, wall.axis.y, wall.axis.z);
            })
        .def_property_readonly("inside", [](const CylinderWall& wall) {return wall.inside;} );

    pybind11::class_<PlaneWall>(m, "PlaneWall")
        .def(pybind11::init(
            [](pybind11::tuple origin, pybind11::tuple normal, bool inside)
                {
                return PlaneWall(make_scalar3(origin[0].cast<Scalar>(), origin[1].cast<Scalar>(), origin[2].cast<Scalar>()),
                                 make_scalar3(normal[0].cast<Scalar>(), normal[1].cast<Scalar>(), normal[2].cast<Scalar>()),
                                 inside);
                }),
            pybind11::arg("origin"), pybind11::arg("normal"), pybind11::arg("inside")
            )
        .def_property_readonly("origin", [](const PlaneWall& wall)
            {
            return pybind11::make_tuple(wall.origin.x, wall.origin.y, wall.origin.z);
            })
        .def_property_readonly("normal", [](const PlaneWall& wall)
            {
            return pybind11::make_tuple(wall.normal.x, wall.normal.y, wall.normal.z);
            })
        .def_property_readonly("inside", [](const PlaneWall& wall) { return wall.inside; } );
    }
#endif
