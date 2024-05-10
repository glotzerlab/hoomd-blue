// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/ExecutionConfiguration.h"

#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN();

#include "hoomd/hpmc/IntegratorHPMC.h"
#include "hoomd/hpmc/Moves.h"
#include "hoomd/hpmc/ShapeSpheropolyhedron.h"

#include "hoomd/extern/quickhull/QuickHull.hpp"

#include <iostream>
#include <string>

#include <memory>
#include <pybind11/pybind11.h>

using namespace hoomd;
using namespace hoomd::hpmc;
using namespace std;
using namespace hoomd::hpmc::detail;

unsigned int err_count = 0;

// helper function to compute poly radius
PolyhedronVertices setup_verts(const vector<vec3<ShortReal>> vlist, ShortReal sweep_radius)
    {
    PolyhedronVertices result((unsigned int)vlist.size(), false);
    result.sweep_radius = sweep_radius;
    result.ignore = 0;

    // extract the verts from the python list and compute the radius on the way
    ShortReal radius_sq = ShortReal(0.0);
    for (unsigned int i = 0; i < vlist.size(); i++)
        {
        vec3<ShortReal> vert = vlist[i];
        result.x[i] = vert.x;
        result.y[i] = vert.y;
        result.z[i] = vert.z;
        radius_sq = std::max(radius_sq, dot(vert, vert));
        }
    for (unsigned int i = (unsigned int)vlist.size(); i < result.N; i++)
        {
        result.x[i] = 0;
        result.y[i] = 0;
        result.z[i] = 0;
        }

    // set the diameter
    result.diameter = 2 * (sqrt(radius_sq) + sweep_radius);

    if (vlist.size() >= 3)
        {
        // compute convex hull of vertices
        typedef quickhull::Vector3<ShortReal> vec;

        std::vector<vec> qh_pts;
        for (unsigned int i = 0; i < vlist.size(); i++)
            {
            vec vert;
            vert.x = vlist[i].x;
            vert.y = vlist[i].y;
            vert.z = vlist[i].z;
            qh_pts.push_back(vert);
            }

        quickhull::QuickHull<ShortReal> qh;
        auto hull = qh.getConvexHull(qh_pts, false, true);
        auto indexBuffer = hull.getIndexBuffer();

        result.hull_verts = ManagedArray<unsigned int>((unsigned int)indexBuffer.size(), false);
        result.n_hull_verts = (unsigned int)indexBuffer.size();

        for (unsigned int i = 0; i < indexBuffer.size(); i++)
            result.hull_verts[i] = (unsigned int)indexBuffer[i];
        }
    return result;
    }

UP_TEST(construction)
    {
    quat<Scalar> o(1.0, vec3<Scalar>(-3.0, 9.0, 6.0));

    vector<vec3<ShortReal>> vlist;
    vlist.push_back(vec3<Scalar>(0, 0, 0));
    vlist.push_back(vec3<Scalar>(1, 0, 0));
    vlist.push_back(vec3<Scalar>(0, 1.25, 0));
    vlist.push_back(vec3<Scalar>(0, 0, 1.1));
    PolyhedronVertices verts = setup_verts(vlist, 0.25);

    ShapeSpheropolyhedron a(o, verts);

    MY_CHECK_CLOSE(a.orientation.s, o.s, tol);
    MY_CHECK_CLOSE(a.orientation.v.x, o.v.x, tol);
    MY_CHECK_CLOSE(a.orientation.v.y, o.v.y, tol);
    MY_CHECK_CLOSE(a.orientation.v.z, o.v.z, tol);

    UP_ASSERT_EQUAL(a.verts.N, verts.N);
    for (unsigned int i = 0; i < verts.N; i++)
        {
        MY_CHECK_CLOSE(a.verts.x[i], verts.x[i], tol);
        MY_CHECK_CLOSE(a.verts.y[i], verts.y[i], tol);
        }
    UP_ASSERT_EQUAL(a.verts.sweep_radius, verts.sweep_radius);
    UP_ASSERT_EQUAL(a.verts.ignore, verts.ignore);

    UP_ASSERT(a.hasOrientation());

    MY_CHECK_CLOSE(a.getCircumsphereDiameter(), 3.0, tol);
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
    PolyhedronVertices verts = setup_verts(vlist, 0.0);

    ShapeSpheropolyhedron a(o, verts);
    SupportFuncConvexPolyhedron sa = SupportFuncConvexPolyhedron(verts, verts.sweep_radius);
    vec3<ShortReal> v1, v2;

    v1 = sa(vec3<ShortReal>(-0.5, -0.5, -0.5));
    v2 = vec3<ShortReal>(-0.5, -0.5, -0.5);
    UP_ASSERT(v1 == v2);
    v1 = sa(vec3<ShortReal>(-ShortReal(0.1), ShortReal(0.1), ShortReal(0.1)));
    v2 = vec3<ShortReal>(-0.5, 0.5, 0.5);
    UP_ASSERT(v1 == v2);
    v1 = sa(vec3<ShortReal>(1, -1, 1));
    v2 = vec3<ShortReal>(0.5, -0.5, 0.5);
    UP_ASSERT(v1 == v2);
    v1 = sa(vec3<ShortReal>(ShortReal(0.51), ShortReal(0.49), -ShortReal(0.1)));
    v2 = vec3<ShortReal>(0.5, 0.5, -0.5);
    UP_ASSERT(v1 == v2);
    }

UP_TEST(sphere)
    {
    // test sphere: zero-vertex special case
    vec3<Scalar> r_ij;
    quat<Scalar> o;

    // build a sphere
    vector<vec3<ShortReal>> vlist;
    PolyhedronVertices verts = setup_verts(vlist, 0.5);

    // test overla
    ShapeSpheropolyhedron a(o, verts);
    ShapeSpheropolyhedron b(o, verts);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));
    r_ij = vec3<Scalar>(.2, .2, .1);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));
    r_ij = vec3<Scalar>(-.2, -.2, -.1);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    // test non-overlap using calculated circumsphere
    r_ij = vec3<Scalar>(3, 0, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));
    r_ij = vec3<Scalar>(2, 2, 1);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));
    r_ij = vec3<Scalar>(-2, -2, -1);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    // test non-overlap using Minkowski difference
    verts.diameter = 10.0;
    ShapeSpheropolyhedron c(o, verts);
    ShapeSpheropolyhedron d(o, verts);
    r_ij = vec3<Scalar>(3, 0, 0);
    UP_ASSERT(!test_overlap(r_ij, c, d, err_count));
    UP_ASSERT(!test_overlap(-r_ij, d, c, err_count));
    r_ij = vec3<Scalar>(2, 2, 1);
    UP_ASSERT(!test_overlap(r_ij, c, d, err_count));
    UP_ASSERT(!test_overlap(-r_ij, d, c, err_count));
    r_ij = vec3<Scalar>(-2, -2, -1);
    UP_ASSERT(!test_overlap(r_ij, c, d, err_count));
    UP_ASSERT(!test_overlap(-r_ij, d, c, err_count));
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
    vlist.push_back(vec3<ShortReal>(0, 0, -ShortReal(0.707106781186548)));
    PolyhedronVertices verts = setup_verts(vlist, 0.0);

    ShapeSpheropolyhedron a(o, verts);

    // first test, separate squares by a large distance
    ShapeSpheropolyhedron b(o, verts);
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
    // this torture test works because 1.0 and 0.5 (the polygon verts) are exactly representable in
    // floating point checking this is important because in a large MC simulation, you are certainly
    // going to find cases where edges or vertices touch exactly
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
    PolyhedronVertices verts = setup_verts(vlist, 0.0);

    ShapeSpheropolyhedron a(o, verts);

    // first test, separate squares by a large distance
    ShapeSpheropolyhedron b(o, verts);
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
    // this torture test works because 1.0 and 0.5 (the polygon verts) are exactly representable in
    // floating point checking this is important because in a large MC simulation, you are certainly
    // going to find cases where edges or vertices touch exactly
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
    PolyhedronVertices verts = setup_verts(vlist, 0.0);

    ShapeSpheropolyhedron a(o_a, verts);

    // first test, separate squares by a large distance
    ShapeSpheropolyhedron b(o_b, verts);
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
    PolyhedronVertices verts = setup_verts(vlist, 0.0);

    ShapeSpheropolyhedron a(o_b, verts);

    // first test, separate cubes by a large distance
    ShapeSpheropolyhedron b(o_a, verts);
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
    vec3<Scalar> r_ij;
    quat<Scalar> o_a;
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
    PolyhedronVertices verts = setup_verts(vlist, 0.0);

    ShapeSpheropolyhedron a(o_a, verts);

    // first test, separate squares by a large distance
    ShapeSpheropolyhedron b(o_b, verts);
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

UP_TEST(overlap_cube_precise)
    {
    // Rounding radius
    Scalar R = 0.1;

    // set the precision
    Scalar offset = 0.00001 / 10.0;

    // test two squares just touching and barely separated to test precision
    vec3<Scalar> r_ij;
    quat<Scalar> o;

    // build a square
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
    PolyhedronVertices verts = setup_verts(vlist, ShortReal(R));

    ShapeSpheropolyhedron a(o, verts);
    ShapeSpheropolyhedron b(o, verts);

    // test face-face non-overlaps
    r_ij = vec3<Scalar>(1 + 2 * R + offset, 0, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(-(1 + 2 * R + offset), 0, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(0, 1 + 2 * R + offset, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(0, -(1 + 2 * R + offset), 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(0, 0, 1 + 2 * R + offset);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(0, 0, -(1 + 2 * R + offset));
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    // test face-face overlaps
    r_ij = vec3<Scalar>(1 + 2 * R - offset, 0, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(-(1 + 2 * R - offset), 0, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(0, 1 + 2 * R - offset, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(0, -(1 + 2 * R - offset), 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(0, 0, 1 + 2 * R - offset);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(0, 0, -(1 + 2 * R - offset));
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    // edge-edge test
    // rotate cubes by 45 degrees about z
    Scalar alpha = M_PI / 4.0;
    quat<Scalar> o_45(cos(alpha / 2.0),
                      (Scalar)sin(alpha / 2.0) * vec3<Scalar>(0, 0, 1)); // rotation quaternion
    a.orientation = o_45;
    b.orientation = o_45;

    // check non-overlapping
    Scalar d = sqrt(2) + 2 * R + offset;
    r_ij = vec3<Scalar>(d, 0, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(-d, 0, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(0, d, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(0, -d, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    // check overlaps
    d = sqrt(2) + 2 * R - offset;
    r_ij = vec3<Scalar>(d, 0, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(-d, 0, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(0, d, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(0, -d, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    // edge-face test
    // rotate only one cube by 45 degrees
    a.orientation = o;
    b.orientation = o_45;

    // check non-overlapping
    d = 0.5 * (1 + sqrt(2)) + 2 * R + offset;
    r_ij = vec3<Scalar>(d, 0, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(-d, 0, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(0, d, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(0, -d, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, b, a, err_count));

    // check overlaps
    d = 0.5 * (1 + sqrt(2)) + 2 * R - offset;
    r_ij = vec3<Scalar>(d, 0, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(-d, 0, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(0, d, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(0, -d, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));
    }

UP_TEST(closest_pt_rounded_cube_no_rot)
    {
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

    ShortReal R = 0.5;
    PolyhedronVertices verts = setup_verts(vlist, R);

    ShapeSpheropolyhedron a(o, verts);

    ProjectionFuncConvexPolyhedron P(verts, verts.sweep_radius);

    // a point inside
    vec3<ShortReal> p = P(vec3<ShortReal>(0, 0, 0));
    MY_CHECK_CLOSE(p.x, 0, tol);
    MY_CHECK_CLOSE(p.y, 0, tol);
    MY_CHECK_CLOSE(p.z, 0, tol);

    // a point out on the x axis
    p = P(vec3<ShortReal>(1.25, 0, 0));
    MY_CHECK_CLOSE(p.x, 1, tol);
    MY_CHECK_CLOSE(p.y, 0, tol);
    MY_CHECK_CLOSE(p.z, 0, tol);

    // a point out on the y axis
    p = P(vec3<ShortReal>(0, 2, 0));
    MY_CHECK_CLOSE(p.x, 0, tol);
    MY_CHECK_CLOSE(p.y, 1, tol);
    MY_CHECK_CLOSE(p.z, 0, tol);

    // a point out on the z axis
    p = P(vec3<ShortReal>(0, 0, 3));
    MY_CHECK_CLOSE(p.x, 0, tol);
    MY_CHECK_CLOSE(p.y, 0, tol);
    MY_CHECK_CLOSE(p.z, 1, tol);

    // a point nearest to the +yz face
    p = P(vec3<ShortReal>(1.5, .25, .25));
    MY_CHECK_CLOSE(p.x, 1, tol);
    MY_CHECK_CLOSE(p.y, 0.25, tol);
    MY_CHECK_CLOSE(p.z, 0.25, tol);

    // a point nearest to a corner
    p = P(vec3<ShortReal>(2, 2, 2));
    MY_CHECK_CLOSE(p.x, 0.5 + 0.5 / sqrt(3), tol);
    MY_CHECK_CLOSE(p.y, 0.5 + 0.5 / sqrt(3), tol);
    MY_CHECK_CLOSE(p.z, 0.5 + 0.5 / sqrt(3), tol);

    // a point close to, but inside, the surface
    p = P(vec3<ShortReal>(0.75, 0, 0));
    std::cout << p.x << " " << p.y << " " << p.z << std::endl;
    MY_CHECK_CLOSE(p.x, 0.75, tol);
    MY_CHECK_CLOSE(p.y, 0, tol);
    MY_CHECK_CLOSE(p.z, 0, tol);
    }
