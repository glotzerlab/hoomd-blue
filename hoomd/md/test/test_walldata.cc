// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// this include is necessary to get MPI included before anything else to support intel MPI
#include "hoomd/ExecutionConfiguration.h"

#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN();

#include "hoomd/md/WallData.h"

#include <cstdlib>
#include <memory>
#include <vector>

using namespace hoomd;
using namespace hoomd::md;

UP_TEST(construction)
    {
    // test each geometry definition structure constructors
    SphereWall defSphere = SphereWall();
    MY_CHECK_SMALL(defSphere.r, tol_small);
    MY_CHECK_SMALL(defSphere.origin.x, tol_small);
    MY_CHECK_SMALL(defSphere.origin.y, tol_small);
    MY_CHECK_SMALL(defSphere.origin.z, tol_small);
    UP_ASSERT(defSphere.inside == true);

    SphereWall Sphere = SphereWall(1.25, make_scalar3(1.0, 2.8, 39.27), false);
    MY_CHECK_CLOSE(Sphere.r, 1.25, tol);
    MY_CHECK_CLOSE(Sphere.origin.x, 1.0, tol);
    MY_CHECK_CLOSE(Sphere.origin.y, 2.8, tol);
    MY_CHECK_CLOSE(Sphere.origin.z, 39.27, tol);
    UP_ASSERT(Sphere.inside == false);

    CylinderWall defCylinder = CylinderWall();
    MY_CHECK_SMALL(defCylinder.r, tol_small);
    MY_CHECK_SMALL(defCylinder.origin.x, tol_small);
    MY_CHECK_SMALL(defCylinder.origin.y, tol_small);
    MY_CHECK_SMALL(defCylinder.origin.z, tol_small);
    MY_CHECK_SMALL(defCylinder.axis.x, tol_small);
    MY_CHECK_SMALL(defCylinder.axis.y, tol_small);
    MY_CHECK_CLOSE(defCylinder.axis.z, 1.0, tol);
    UP_ASSERT(defCylinder.inside == true);

    CylinderWall Cylinder = CylinderWall(1.25,
                                         make_scalar3(584.1, 27.6, 14.1),
                                         make_scalar3(4.2, 57.21, 0.21),
                                         false);
    MY_CHECK_CLOSE(Cylinder.r, 1.25, tol);
    MY_CHECK_CLOSE(Cylinder.origin.x, 584.1, tol);
    MY_CHECK_CLOSE(Cylinder.origin.y, 27.6, tol);
    MY_CHECK_CLOSE(Cylinder.origin.z, 14.1, tol);
    MY_CHECK_CLOSE(Cylinder.axis.x, 4.2, tol);
    MY_CHECK_CLOSE(Cylinder.axis.y, 57.21, tol);
    MY_CHECK_CLOSE(Cylinder.axis.z, 0.21, tol);
    UP_ASSERT(Cylinder.inside == false);

    PlaneWall defPlane = PlaneWall();
    MY_CHECK_SMALL(defPlane.origin.x, tol_small);
    MY_CHECK_SMALL(defPlane.origin.y, tol_small);
    MY_CHECK_SMALL(defPlane.origin.z, tol_small);
    MY_CHECK_SMALL(defPlane.normal.x, tol_small);
    MY_CHECK_SMALL(defPlane.normal.y, tol_small);
    MY_CHECK_CLOSE(defPlane.normal.z, 1.0, tol);

    PlaneWall Plane = PlaneWall(make_scalar3(5.214, 1.32, 8.143), make_scalar3(3.62, 2.151, 8.1));
    MY_CHECK_CLOSE(Plane.origin.x, 5.214, tol);
    MY_CHECK_CLOSE(Plane.origin.y, 1.32, tol);
    MY_CHECK_CLOSE(Plane.origin.z, 8.143, tol);
    MY_CHECK_CLOSE(Plane.normal.x, 0.39653241, tol);
    MY_CHECK_CLOSE(Plane.normal.y, 0.23561912, tol);
    MY_CHECK_CLOSE(Plane.normal.z, 0.88726866, tol);
    }

UP_TEST(sphere_wall_math)
    {
    SphereWall Sphere = SphereWall(5.0, make_scalar3(2.0, 3.0, 1.0));
    bool inside = true;

    // test inside
    auto x = vec3<Scalar>(1.2, 1.6, 2.0);
    auto vx = distVectorWallToPoint(Sphere, x, inside);
    Scalar dx = distWall(Sphere, x);
    MY_CHECK_CLOSE(vx.x, 1.30818511, tol);
    MY_CHECK_CLOSE(vx.y, 2.28932394, tol);
    MY_CHECK_CLOSE(vx.z, -1.63523138, tol);
    MY_CHECK_CLOSE(dx, 3.1026334, tol);
    UP_ASSERT(inside == true);

    // test outside
    x = vec3<Scalar>(13.2, 3.0, 1.0);
    vx = distVectorWallToPoint(Sphere, x, inside);
    dx = distWall(Sphere, x);
    MY_CHECK_CLOSE(vx.x, 6.2, tol);
    MY_CHECK_SMALL(vx.y, tol_small);
    MY_CHECK_SMALL(vx.z, tol_small);
    MY_CHECK_CLOSE(dx, -6.2, tol);
    UP_ASSERT(inside == false);

    // test center
    x = vec3<Scalar>(2.0, 3.0, 1.0);
    vx = distVectorWallToPoint(Sphere, x, inside);
    dx = distWall(Sphere, x);
    MY_CHECK_SMALL(vx.x, tol_small);
    MY_CHECK_SMALL(vx.y, tol_small);
    MY_CHECK_SMALL(vx.z, tol_small);
    MY_CHECK_CLOSE(dx, 5.0, tol);
    UP_ASSERT(inside == true);

    // test on surface
    x = vec3<Scalar>(2.0, 3.0, 6.0);
    vx = distVectorWallToPoint(Sphere, x, inside);
    dx = distWall(Sphere, x);
    MY_CHECK_SMALL(vx.x, tol_small);
    MY_CHECK_SMALL(vx.y, tol_small);
    MY_CHECK_SMALL(vx.z, tol_small);
    MY_CHECK_SMALL(dx, tol_small);

    SphereWall invSphere = SphereWall(0.5, make_scalar3(1.0, 2.0, 3.0), false);

    // test "inside"
    x = vec3<Scalar>(1.2, 3.6, 2.0);
    vx = distVectorWallToPoint(invSphere, x, inside);
    dx = distWall(invSphere, x);
    MY_CHECK_CLOSE(vx.x, 0.14729537, tol);
    MY_CHECK_CLOSE(vx.y, 1.17836298, tol);
    MY_CHECK_CLOSE(vx.z, -0.73647686, tol);
    MY_CHECK_CLOSE(dx, 1.3973666, tol);
    UP_ASSERT(inside == true);

    // test "outside"
    x = vec3<Scalar>(1.2, 2.1, 3.0);
    vx = distVectorWallToPoint(invSphere, x, inside);
    dx = distWall(invSphere, x);
    MY_CHECK_CLOSE(vx.x, -0.2472136, tol);
    MY_CHECK_CLOSE(vx.y, -0.1236068, tol);
    MY_CHECK_SMALL(vx.z, tol_small);
    MY_CHECK_CLOSE(dx, -0.2763932073, tol);
    UP_ASSERT(inside == false);

    // test on surface
    x = vec3<Scalar>(1.0, 2.5, 3.0);
    vx = distVectorWallToPoint(invSphere, x, inside);
    dx = distWall(invSphere, x);
    MY_CHECK_SMALL(vx.x, tol_small);
    MY_CHECK_SMALL(vx.y, tol_small);
    MY_CHECK_SMALL(vx.z, tol_small);
    MY_CHECK_SMALL(dx, tol_small);
    }

UP_TEST(cylinder_wall_math)
    {
    CylinderWall Cylinder = CylinderWall(3.5, make_scalar3(2.1, 1.1, 0.7), make_scalar3(2.0, 0, 0));
    bool inside = true;

    // test inside
    auto x = vec3<Scalar>(30.2, 1.6, 2.0);
    auto vx = distVectorWallToPoint(Cylinder, x, inside);
    Scalar dx = distWall(Cylinder, x);
    MY_CHECK_SMALL(vx.x, tol_small);
    MY_CHECK_CLOSE(vx.y, -0.75642678, tol);
    MY_CHECK_CLOSE(vx.z, -1.96670962, tol);
    MY_CHECK_CLOSE(dx, 2.10716117, tol);
    UP_ASSERT(inside == true);

    // test outside
    x = vec3<Scalar>(2.2, 5.6, 1.0);
    vx = distVectorWallToPoint(Cylinder, x, inside);
    dx = distWall(Cylinder, x);
    MY_CHECK_SMALL(vx.x, tol_small);
    MY_CHECK_CLOSE(vx.y, 1.00775195, tol);
    MY_CHECK_CLOSE(vx.z, 0.06718346, tol);
    MY_CHECK_CLOSE(dx, -1.00998891, tol);
    UP_ASSERT(inside == false);

    // test center
    x = vec3<Scalar>(2.1, 1.1, 0.7);
    vx = distVectorWallToPoint(Cylinder, x, inside);
    dx = distWall(Cylinder, x);
    MY_CHECK_SMALL(vx.x, tol_small);
    MY_CHECK_SMALL(vx.y, tol_small);
    MY_CHECK_SMALL(vx.z, tol_small);
    MY_CHECK_CLOSE(dx, 3.5, tol);
    UP_ASSERT(inside == true);

    // test on surface
    x = vec3<Scalar>(2.1, 4.6, 0.7);
    vx = distVectorWallToPoint(Cylinder, x, inside);
    dx = distWall(Cylinder, x);
    MY_CHECK_SMALL(vx.x, tol_small);
    MY_CHECK_SMALL(vx.y, tol_small);
    MY_CHECK_SMALL(vx.z, tol_small);
    MY_CHECK_SMALL(dx, tol_small);
    UP_ASSERT(inside == true);

    CylinderWall invCylinder
        = CylinderWall(3.5, make_scalar3(2.1, 1.1, 0.7), make_scalar3(1.5, 0, 0), false);

    // test "inside"
    x = vec3<Scalar>(2.2, 5.6, 1.0);
    vx = distVectorWallToPoint(invCylinder, x, inside);
    dx = distWall(invCylinder, x);
    MY_CHECK_SMALL(vx.x, tol_small);
    MY_CHECK_CLOSE(vx.y, 1.00775195, tol);
    MY_CHECK_CLOSE(vx.z, 0.06718346, tol);
    MY_CHECK_CLOSE(dx, 1.00998891, tol);
    UP_ASSERT(inside == true);

    // test "outside"
    x = vec3<Scalar>(30.2, 1.6, 2.0);
    vx = distVectorWallToPoint(invCylinder, x, inside);
    dx = distWall(invCylinder, x);
    MY_CHECK_SMALL(vx.x, tol_small);
    MY_CHECK_CLOSE(vx.y, -0.75642678, tol);
    MY_CHECK_CLOSE(vx.z, -1.96670962, tol);
    MY_CHECK_CLOSE(dx, -2.10716117, tol);
    UP_ASSERT(inside == false);

    // test on surface
    x = vec3<Scalar>(2.1, 1.1, 4.2);
    vx = distVectorWallToPoint(invCylinder, x, inside);
    dx = distWall(invCylinder, x);
    MY_CHECK_SMALL(vx.x, tol_small);
    MY_CHECK_SMALL(vx.y, tol_small);
    MY_CHECK_SMALL(vx.z, tol_small);
    MY_CHECK_SMALL(dx, tol_small);
    UP_ASSERT(inside == true);
    }

UP_TEST(plane_wall_math)
    {
    PlaneWall Plane = PlaneWall(make_scalar3(4.1, 2.1, 8.7), make_scalar3(-1.1, -0.3, -5.0));
    bool inside = true;

    // test inside
    auto x = vec3<Scalar>(30.2, 1.6, 2.0);
    auto vx = distVectorWallToPoint(Plane, x, inside);
    Scalar dx = distWall(Plane, x);
    MY_CHECK_CLOSE(vx.x, -0.20661597, tol);
    MY_CHECK_CLOSE(vx.y, -0.05634981, tol);
    MY_CHECK_CLOSE(vx.z, -0.9391635, tol);
    MY_CHECK_CLOSE(dx, 0.963272306319, tol);
    UP_ASSERT(inside == true);

    // test outside
    x = vec3<Scalar>(5.1, 2.2, 8.9);
    vx = distVectorWallToPoint(Plane, x, inside);
    dx = distWall(Plane, x);
    MY_CHECK_CLOSE(vx.x, 0.08908745, tol);
    MY_CHECK_CLOSE(vx.y, 0.02429658, tol);
    MY_CHECK_CLOSE(vx.z, 0.40494297, tol);
    MY_CHECK_CLOSE(dx, -0.4153380592, tol);
    UP_ASSERT(inside == false);

    // test on surface
    x = vec3<Scalar>(4.1, 2.1, 8.7);
    vx = distVectorWallToPoint(Plane, x, inside);
    dx = distWall(Plane, x);
    MY_CHECK_SMALL(vx.x, tol_small);
    MY_CHECK_SMALL(vx.y, tol_small);
    MY_CHECK_SMALL(vx.z, tol_small);
    MY_CHECK_SMALL(dx, tol_small);
    UP_ASSERT(inside == true);

    PlaneWall invPlane = PlaneWall(make_scalar3(4.1, 2.1, 8.7), make_scalar3(1.1, 0.3, 5.0));

    // test on surface
    x = vec3<Scalar>(4.1, 2.1, 8.7);
    vx = distVectorWallToPoint(invPlane, x, inside);
    dx = distWall(invPlane, x);
    MY_CHECK_SMALL(vx.x, tol_small);
    MY_CHECK_SMALL(vx.y, tol_small);
    MY_CHECK_SMALL(vx.z, tol_small);
    MY_CHECK_SMALL(dx, tol_small);
    }
