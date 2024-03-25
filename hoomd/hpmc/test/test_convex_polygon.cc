// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/hpmc/IntegratorHPMC.h"
#include "hoomd/hpmc/Moves.h"
#include "hoomd/hpmc/ShapeConvexPolygon.h"

#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN();

#include <iostream>
#include <string>

#include <pybind11/pybind11.h>

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/RandomNumbers.h"

using namespace hoomd;
using namespace hoomd::hpmc;
using namespace hoomd::hpmc::detail;
using namespace std;

unsigned int err_count = 0;

// helper function to compute poly radius
PolygonVertices setup_verts(const vector<vec2<ShortReal>> vlist)
    {
    if (vlist.size() > MAX_POLY2D_VERTS)
        throw runtime_error("Too many polygon vertices");

    PolygonVertices result;
    result.N = (unsigned int)vlist.size();
    result.ignore = 0;

    // extract the verts from the python list and compute the radius on the way
    ShortReal radius_sq = ShortReal(0.0);
    for (unsigned int i = 0; i < vlist.size(); i++)
        {
        vec2<ShortReal> vert = vlist[i];
        result.x[i] = vert.x;
        result.y[i] = vert.y;
        radius_sq = std::max(radius_sq, dot(vert, vert));
        }
    for (unsigned int i = (unsigned int)vlist.size(); i < MAX_POLY2D_VERTS; i++)
        {
        result.x[i] = 0;
        result.y[i] = 0;
        }

    // set the diameter
    result.diameter = 2 * sqrt(radius_sq);

    return result;
    }

UP_TEST(construction)
    {
    quat<Scalar> o(1.0, vec3<Scalar>(-3.0, 9.0, 6.0));

    std::vector<vec2<ShortReal>> vlist;
    vlist.push_back(vec2<ShortReal>(0, 0));
    vlist.push_back(vec2<ShortReal>(1, 0));
    vlist.push_back(vec2<ShortReal>(0, 1.25));
    PolygonVertices verts = setup_verts(vlist);

    ShapeConvexPolygon a(o, verts);
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

    UP_ASSERT(a.hasOrientation());

    MY_CHECK_CLOSE(a.getCircumsphereDiameter(), 2.5, tol);
    }

UP_TEST(overlap_square_no_rot)
    {
    // first set of simple overlap checks is two squares at unit orientation
    quat<Scalar> o;
    vec3<Scalar> r_ij;
    BoxDim box(100);

    // build a square
    std::vector<vec2<ShortReal>> vlist;
    vlist.push_back(vec2<ShortReal>(-0.5, -0.5));
    vlist.push_back(vec2<ShortReal>(0.5, -0.5));
    vlist.push_back(vec2<ShortReal>(0.5, 0.5));
    vlist.push_back(vec2<ShortReal>(-0.5, 0.5));
    PolygonVertices verts = setup_verts(vlist);

    ShapeConvexPolygon a(o, verts);

    // first test, separate squares by a large distance
    ShapeConvexPolygon b(o, verts);
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

UP_TEST(overlap_square_rot1)
    {
    // second set of simple overlap checks is two squares, with one rotated by 45 degrees
    vec3<Scalar> r_ij;
    quat<Scalar> o_a;
    Scalar alpha = M_PI / 4.0;
    quat<Scalar> o_b(cos(alpha / 2.0),
                     (Scalar)sin(alpha / 2.0) * vec3<Scalar>(0, 0, 1)); // rotation quaternion

    BoxDim box(100);

    // build a square
    std::vector<vec2<ShortReal>> vlist;
    vlist.push_back(vec2<ShortReal>(-0.5, -0.5));
    vlist.push_back(vec2<ShortReal>(0.5, -0.5));
    vlist.push_back(vec2<ShortReal>(0.5, 0.5));
    vlist.push_back(vec2<ShortReal>(-0.5, 0.5));
    PolygonVertices verts = setup_verts(vlist);

    ShapeConvexPolygon a(o_a, verts);

    // first test, separate squares by a large distance
    ShapeConvexPolygon b(o_b, verts);
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

UP_TEST(overlap_square_rot2)
    {
    // third set of simple overlap checks is two squares, with the other one rotated by 45 degrees
    vec3<Scalar> r_ij;
    quat<Scalar> o_a;
    Scalar alpha = M_PI / 4.0;
    quat<Scalar> o_b(cos(alpha / 2.0),
                     (Scalar)sin(alpha / 2.0) * vec3<Scalar>(0, 0, 1)); // rotation quaternion

    BoxDim box(100);

    // build a square
    std::vector<vec2<ShortReal>> vlist;
    vlist.push_back(vec2<ShortReal>(-0.5, -0.5));
    vlist.push_back(vec2<ShortReal>(0.5, -0.5));
    vlist.push_back(vec2<ShortReal>(0.5, 0.5));
    vlist.push_back(vec2<ShortReal>(-0.5, 0.5));
    PolygonVertices verts = setup_verts(vlist);

    ShapeConvexPolygon a(o_b, verts);

    // first test, separate squares by a large distance
    ShapeConvexPolygon b(o_a, verts);
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

UP_TEST(overlap_square_tri)
    {
    // and now a more complicated test, a square and a triangle - both rotated
    vec3<Scalar> r_ij;

    Scalar alpha = -M_PI / 4.0;
    quat<Scalar> o_a(cos(alpha / 2.0),
                     (Scalar)sin(alpha / 2.0) * vec3<Scalar>(0, 0, 1)); // rotation quaternion
    alpha = M_PI;
    quat<Scalar> o_b(cos(alpha / 2.0),
                     (Scalar)sin(alpha / 2.0) * vec3<Scalar>(0, 0, 1)); // rotation quaternion

    BoxDim box(100);

    // build a square
    std::vector<vec2<ShortReal>> vlist_a;
    vlist_a.push_back(vec2<ShortReal>(-0.5, -0.5));
    vlist_a.push_back(vec2<ShortReal>(0.5, -0.5));
    vlist_a.push_back(vec2<ShortReal>(0.5, 0.5));
    vlist_a.push_back(vec2<ShortReal>(-0.5, 0.5));
    PolygonVertices verts_a = setup_verts(vlist_a);

    std::vector<vec2<ShortReal>> vlist_b;
    vlist_b.push_back(vec2<ShortReal>(-0.5, -0.5));
    vlist_b.push_back(vec2<ShortReal>(0.5, -0.5));
    vlist_b.push_back(vec2<ShortReal>(0.5, 0.5));
    PolygonVertices verts_b = setup_verts(vlist_b);

    ShapeConvexPolygon a(o_a, verts_a);

    // first test, separate squares by a large distance
    ShapeConvexPolygon b(o_b, verts_b);
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

    // and finally, make them overlap slightly in each direction
    r_ij = vec3<Scalar>(1.2, 0.2, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(-0.7, -0.2, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(0.4, 1.1, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));

    r_ij = vec3<Scalar>(-0.2, -1.2, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, b, a, err_count));
    }
