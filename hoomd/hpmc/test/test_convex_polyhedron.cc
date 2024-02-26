// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/hpmc/IntegratorHPMC.h"
#include "hoomd/hpmc/Moves.h"
#include "hoomd/hpmc/ShapeConvexPolyhedron.h"

#include "hoomd/extern/quickhull/QuickHull.hpp"

#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN();

#include <iostream>
#include <string>

#include <pybind11/pybind11.h>

using namespace hoomd;
using namespace hoomd::hpmc;
using namespace std;
using namespace hoomd::hpmc::detail;

unsigned int err_count;

UP_TEST(construction)
    {
    quat<Scalar> o(1.0, vec3<Scalar>(-3.0, 9.0, 6.0));

    vector<vec3<ShortReal>> vlist;
    vlist.push_back(vec3<Scalar>(0, 0, 0));
    vlist.push_back(vec3<Scalar>(1, 0, 0));
    vlist.push_back(vec3<Scalar>(0, 1.25, 0));
    vlist.push_back(vec3<Scalar>(0, 0, 1.1));
    PolyhedronVertices verts(vlist, 0, 0);

    ShapeConvexPolyhedron a(o, verts);

    MY_CHECK_CLOSE(a.orientation.s, o.s, tol);
    MY_CHECK_CLOSE(a.orientation.v.x, o.v.x, tol);
    MY_CHECK_CLOSE(a.orientation.v.y, o.v.y, tol);
    MY_CHECK_CLOSE(a.orientation.v.z, o.v.z, tol);

    UP_ASSERT_EQUAL(a.verts.N, verts.N);
    for (unsigned int i = 0; i < verts.N; i++)
        {
        MY_CHECK_CLOSE(a.verts.x[i], verts.x[i], tol);
        MY_CHECK_CLOSE(a.verts.y[i], verts.y[i], tol);
        MY_CHECK_CLOSE(a.verts.z[i], verts.z[i], tol);
        }

    UP_ASSERT(a.hasOrientation());

    MY_CHECK_CLOSE(a.getCircumsphereDiameter(), 2.5, tol);
    }

UP_TEST(support)
    {
    // Find the support of a tetrahedron
    quat<Scalar> o;

    vector<vec3<ShortReal>> vlist;
    vlist.push_back(vec3<ShortReal>(-0.5, -0.5, -0.5));
    vlist.push_back(vec3<ShortReal>(-0.5, 0.5, 0.5));
    vlist.push_back(vec3<ShortReal>(0.5, -0.5, 0.5));
    vlist.push_back(vec3<ShortReal>(0.5, 0.5, -0.5));
    PolyhedronVertices verts(vlist, 0, 0);

    ShapeConvexPolyhedron a(o, verts);
    SupportFuncConvexPolyhedron sa = SupportFuncConvexPolyhedron(verts);
    vec3<ShortReal> v1, v2;

    v1 = sa(vec3<ShortReal>(-0.5, -0.5, -0.5));
    v2 = vec3<ShortReal>(-0.5, -0.5, -0.5);
    UP_ASSERT(v1 == v2);
    v1 = sa(vec3<ShortReal>(-0.125, 0.125, 0.125));
    v2 = vec3<ShortReal>(-0.5, 0.5, 0.5);
    UP_ASSERT(v1 == v2);
    v1 = sa(vec3<ShortReal>(1, -1, 1));
    v2 = vec3<ShortReal>(0.5, -0.5, 0.5);
    UP_ASSERT(v1 == v2);
    v1 = sa(vec3<ShortReal>(ShortReal(0.51), ShortReal(0.49), ShortReal(-0.1)));
    v2 = vec3<ShortReal>(0.5, 0.5, -0.5);
    UP_ASSERT(v1 == v2);
    }

UP_TEST(overlap_octahedron_no_rot)
    {
    // first set of simple overlap checks is two octahedra at unit orientation
    vec3<Scalar> r_ij;
    quat<Scalar> o;

    // build a square
    vector<vec3<ShortReal>> vlist;
    vlist.push_back(vec3<ShortReal>(-0.5, -0.5, 0));
    vlist.push_back(vec3<ShortReal>(0.5, -0.5, 0));
    vlist.push_back(vec3<ShortReal>(0.5, 0.5, 0));
    vlist.push_back(vec3<ShortReal>(-0.5, 0.5, 0));
    vlist.push_back(vec3<ShortReal>(0, 0, ShortReal(0.707106781186548)));
    vlist.push_back(vec3<ShortReal>(0, 0, ShortReal(-0.707106781186548)));
    PolyhedronVertices verts(vlist, 0, 0);

    ShapeConvexPolyhedron a(o, verts);
    ShapeConvexPolyhedron b(o, verts);

    // zeroth test: exactly overlapping shapes
    r_ij = vec3<Scalar>(0.0, 0.0, 0.0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    // first test, separate squares by a large distance
    r_ij = vec3<Scalar>(10, 0, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    // next test, set them close, but not overlapping - from all four sides of base
    r_ij = vec3<Scalar>(1.1, 0, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(-1.1, 0, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(0, 1.1, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(0, -1.1, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    // now test them close, but slightly offset and not overlapping - from all four sides
    r_ij = vec3<Scalar>(1.1, 0.2, 0.1);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(-1.1, 0.2, 0.1);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(-0.2, 1.1, 0.1);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(-0.2, -1.1, 0.1);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    // and finally, make them overlap slightly in each direction
    r_ij = vec3<Scalar>(0.9, 0.2, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(-0.9, 0.2, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(-0.2, 0.9, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(-0.2, -0.9, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    // torture test, overlap along most of a line
    r_ij = vec3<Scalar>(1.0, 0.2, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));
    }

UP_TEST(overlap_cube_no_rot)
    {
    // first set of simple overlap checks is two squares at unit orientation
    vec3<Scalar> r_ij;
    quat<Scalar> o;

    // build a square
    vector<vec3<ShortReal>> vlist;
    vlist.push_back(vec3<ShortReal>(-0.5, -0.5, -0.5));
    vlist.push_back(vec3<ShortReal>(0.5, -0.5, -0.5));
    vlist.push_back(vec3<ShortReal>(0.5, 0.5, -0.5));
    vlist.push_back(vec3<ShortReal>(-0.5, 0.5, -0.5));
    vlist.push_back(vec3<ShortReal>(-0.5, -0.5, 0.5));
    vlist.push_back(vec3<ShortReal>(0.5, -0.5, 0.5));
    vlist.push_back(vec3<ShortReal>(0.5, 0.5, 0.5));
    vlist.push_back(vec3<ShortReal>(-0.5, 0.5, 0.5));
    PolyhedronVertices verts(vlist, 0, 0);

    ShapeConvexPolyhedron a(o, verts);

    // first test, separate squares by a large distance
    ShapeConvexPolyhedron b(o, verts);
    r_ij = vec3<Scalar>(10, 0, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    // next test, set them close, but not overlapping - from all four sides
    r_ij = vec3<Scalar>(1.1, 0, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(-1.1, 0, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(0, 1.1, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(0, -1.1, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    // now test them close, but slightly offset and not overlapping - from all four sides
    r_ij = vec3<Scalar>(1.1, 0.2, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(-1.1, 0.2, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(-0.2, 1.1, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(-0.2, -1.1, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    // make them overlap slightly in each direction
    r_ij = vec3<Scalar>(0.9, 0.2, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(-0.9, 0.2, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(-0.2, 0.9, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(-0.2, -0.9, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    // Make them overlap a lot
    r_ij = vec3<Scalar>(0.2, 0, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(0, 0.2, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(0.2, tol, tol);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(0.1, 0.2, 0.1);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    // torture test, overlap along most of a line
    r_ij = vec3<Scalar>(1.0, 0.2, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));
    }

UP_TEST(overlap_cube_rot1)
    {
    // second set of simple overlap checks is two cubes, with one rotated by 45 degrees
    vec3<Scalar> r_ij;
    quat<Scalar> o_a;
    Scalar alpha = M_PI / 4.0;
    quat<Scalar> o_b(cos(alpha / 2.0),
                     (Scalar)sin(alpha / 2.0) * vec3<Scalar>(0, 0, 1)); // rotation quaternion

    // build a square
    vector<vec3<ShortReal>> vlist;
    vlist.push_back(vec3<ShortReal>(-0.5, -0.5, -0.5));
    vlist.push_back(vec3<ShortReal>(0.5, -0.5, -0.5));
    vlist.push_back(vec3<ShortReal>(0.5, 0.5, -0.5));
    vlist.push_back(vec3<ShortReal>(-0.5, 0.5, -0.5));
    vlist.push_back(vec3<ShortReal>(-0.5, -0.5, 0.5));
    vlist.push_back(vec3<ShortReal>(0.5, -0.5, 0.5));
    vlist.push_back(vec3<ShortReal>(0.5, 0.5, 0.5));
    vlist.push_back(vec3<ShortReal>(-0.5, 0.5, 0.5));
    PolyhedronVertices verts(vlist, 0, 0);

    ShapeConvexPolyhedron a(o_a, verts);

    // first test, separate squares by a large distance
    ShapeConvexPolyhedron b(o_b, verts);
    r_ij = vec3<Scalar>(10, 0, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    // next test, set them close, but not overlapping - from all four sides
    r_ij = vec3<Scalar>(1.3, 0, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(-1.3, 0, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(0, 1.3, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(0, -1.3, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    // now test them close, but slightly offset and not overlapping - from all four sides
    r_ij = vec3<Scalar>(1.3, 0.2, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(-1.3, 0.2, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(-0.2, 1.3, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(-0.2, -1.3, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    // and finally, make them overlap slightly in each direction
    r_ij = vec3<Scalar>(1.2, 0.2, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(-1.2, 0.2, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(-0.2, 1.2, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(-0.2, -1.2, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));
    }

UP_TEST(overlap_cube_rot2)
    {
    // third set of simple overlap checks is two cubes, with the other one rotated by 45 degrees
    vec3<Scalar> r_ij;
    quat<Scalar> o_a;
    Scalar alpha = M_PI / 4.0;
    quat<Scalar> o_b(cos(alpha / 2.0),
                     (Scalar)sin(alpha / 2.0) * vec3<Scalar>(0, 0, 1)); // rotation quaternion

    // build a cube
    vector<vec3<ShortReal>> vlist;
    vlist.push_back(vec3<ShortReal>(-0.5, -0.5, -0.5));
    vlist.push_back(vec3<ShortReal>(0.5, -0.5, -0.5));
    vlist.push_back(vec3<ShortReal>(0.5, 0.5, -0.5));
    vlist.push_back(vec3<ShortReal>(-0.5, 0.5, -0.5));
    vlist.push_back(vec3<ShortReal>(-0.5, -0.5, 0.5));
    vlist.push_back(vec3<ShortReal>(0.5, -0.5, 0.5));
    vlist.push_back(vec3<ShortReal>(0.5, 0.5, 0.5));
    vlist.push_back(vec3<ShortReal>(-0.5, 0.5, 0.5));
    PolyhedronVertices verts(vlist, 0, 0);

    ShapeConvexPolyhedron a(o_b, verts);

    // first test, separate cubes by a large distance
    ShapeConvexPolyhedron b(o_a, verts);
    r_ij = vec3<Scalar>(10, 0, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    // next test, set them close, but not overlapping - from all four sides
    r_ij = vec3<Scalar>(1.3, 0, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(-1.3, 0, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(0, 1.3, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(0, -1.3, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    // now test them close, but slightly offset and not overlapping - from all four sides
    r_ij = vec3<Scalar>(1.3, 0.2, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(-1.3, 0.2, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(-0.2, 1.3, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(-0.2, -1.3, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    // and finally, make them overlap slightly in each direction
    r_ij = vec3<Scalar>(1.2, 0.2, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(-1.2, 0.2, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(-0.2, 1.2, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(-0.2, -1.2, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));
    }

UP_TEST(overlap_cube_rot3)
    {
    // two cubes, with one rotated by 45 degrees around two axes. This lets us look at edge-edge and
    // point-face collisions
    quat<Scalar> o_a;
    vec3<Scalar> r_ij;
    Scalar alpha = M_PI / 4.0;
    // rotation around x and then z
    const quat<Scalar> q1(cos(alpha / 2.0), (Scalar)sin(alpha / 2.0) * vec3<Scalar>(1, 0, 0));
    const quat<Scalar> q2(cos(alpha / 2.0), (Scalar)sin(alpha / 2.0) * vec3<Scalar>(0, 0, 1));
    quat<Scalar> o_b(q2 * q1);

    // build a cube
    vector<vec3<ShortReal>> vlist;
    vlist.push_back(vec3<ShortReal>(-0.5, -0.5, -0.5));
    vlist.push_back(vec3<ShortReal>(0.5, -0.5, -0.5));
    vlist.push_back(vec3<ShortReal>(0.5, 0.5, -0.5));
    vlist.push_back(vec3<ShortReal>(-0.5, 0.5, -0.5));
    vlist.push_back(vec3<ShortReal>(-0.5, -0.5, 0.5));
    vlist.push_back(vec3<ShortReal>(0.5, -0.5, 0.5));
    vlist.push_back(vec3<ShortReal>(0.5, 0.5, 0.5));
    vlist.push_back(vec3<ShortReal>(-0.5, 0.5, 0.5));
    PolyhedronVertices verts(vlist, 0, 0);

    ShapeConvexPolyhedron a(o_a, verts);

    // first test, separate squares by a large distance
    ShapeConvexPolyhedron b(o_b, verts);
    r_ij = vec3<Scalar>(10, 0, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    // next test, set them close, but not overlapping - from four sides
    r_ij = vec3<Scalar>(1.4, 0, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(-1.4, 0, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(0, 1.4, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(0, -1.4, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    // now test them close, but slightly offset and not overlapping - from four sides
    r_ij = vec3<Scalar>(1.4, 0.2, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(-1.4, 0.2, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(-0.2, 1.4, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(-0.2, -1.4, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    // Test point-face overlaps
    r_ij = vec3<Scalar>(0, 1.2, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(0, 1.2, 0.1);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(0.1, 1.2, 0.1);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(1.2, 0, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(1.2, 0.1, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(1.2, 0.1, 0.1);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    // Test edge-edge overlaps
    r_ij = vec3<Scalar>(-0.9, 0.9, 0.0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(-0.9, 0.899, 0.001);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(0.9, -0.9, 0.0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count)); // this and only this test failed :-(
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(-0.9, 0.899, 0.001);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(-0.9, 0.9, 0.1);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));
    }

UP_TEST(closest_pt_cube_no_rot)
    {
    //! Test that projection of points is working for a cube
    vec3<Scalar> r_ij;
    quat<Scalar> o;

    // build a cube
    vector<vec3<ShortReal>> vlist;
    vlist.push_back(vec3<ShortReal>(-0.5, -0.5, -0.5));
    vlist.push_back(vec3<ShortReal>(0.5, -0.5, -0.5));
    vlist.push_back(vec3<ShortReal>(0.5, 0.5, -0.5));
    vlist.push_back(vec3<ShortReal>(-0.5, 0.5, -0.5));
    vlist.push_back(vec3<ShortReal>(-0.5, -0.5, 0.5));
    vlist.push_back(vec3<ShortReal>(0.5, -0.5, 0.5));
    vlist.push_back(vec3<ShortReal>(0.5, 0.5, 0.5));
    vlist.push_back(vec3<ShortReal>(-0.5, 0.5, 0.5));
    PolyhedronVertices verts(vlist, 0, 0);

    ShapeConvexPolyhedron a(o, verts);

    // a point inside
    ProjectionFuncConvexPolyhedron P(a.verts);

    vec3<ShortReal> p = P(vec3<ShortReal>(0, 0, 0));
    MY_CHECK_CLOSE(p.x, 0, tol);
    MY_CHECK_CLOSE(p.y, 0, tol);
    MY_CHECK_CLOSE(p.z, 0, tol);

    // a point out on the x axis
    p = P(vec3<ShortReal>(1, 0, 0));
    MY_CHECK_CLOSE(p.x, 0.5, tol);
    MY_CHECK_CLOSE(p.y, 0, tol);
    MY_CHECK_CLOSE(p.z, 0, tol);

    // a point out on the y axis
    p = P(vec3<ShortReal>(0, 2, 0));
    MY_CHECK_CLOSE(p.x, 0, tol);
    MY_CHECK_CLOSE(p.y, 0.5, tol);
    MY_CHECK_CLOSE(p.z, 0, tol);

    // a point out on the z axis
    p = P(vec3<ShortReal>(0, 0, 3));
    MY_CHECK_CLOSE(p.x, 0, tol);
    MY_CHECK_CLOSE(p.y, 0, tol);
    MY_CHECK_CLOSE(p.z, 0.5, tol);

    // a point nearest to the +yz face
    p = P(vec3<ShortReal>(1, .25, .25));
    MY_CHECK_CLOSE(p.x, 0.5, tol);
    MY_CHECK_CLOSE(p.y, 0.25, tol);
    MY_CHECK_CLOSE(p.z, 0.25, tol);

    // a point nearest to a corner
    p = P(vec3<ShortReal>(1, 1, 1));
    MY_CHECK_CLOSE(p.x, 0.5, tol);
    MY_CHECK_CLOSE(p.y, 0.5, tol);
    MY_CHECK_CLOSE(p.z, 0.5, tol);
    }
