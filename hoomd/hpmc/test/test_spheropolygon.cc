// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/ExecutionConfiguration.h"

#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN();

#include "hoomd/hpmc/IntegratorHPMC.h"
#include "hoomd/hpmc/Moves.h"
#include "hoomd/hpmc/ShapeSpheropolygon.h"

#include <iostream>
#include <string>

#include <memory>
#include <pybind11/pybind11.h>

using namespace hoomd;
using namespace hoomd::hpmc;
using namespace std;
using namespace hoomd::hpmc::detail;

unsigned int err_count;

// helper function to compute poly radius
PolygonVertices setup_verts(const vector<vec2<ShortReal>> vlist, ShortReal sweep_radius = 0.0)
    {
    if (vlist.size() > MAX_POLY2D_VERTS)
        throw runtime_error("Too many polygon vertices");

    PolygonVertices result;
    result.N = (unsigned int)vlist.size();
    result.sweep_radius = sweep_radius;
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
    result.diameter = 2 * (sqrt(radius_sq) + sweep_radius);

    return result;
    }

UP_TEST(construction)
    {
    quat<Scalar> o(1.0, vec3<Scalar>(-3.0, 9.0, 6.0));

    std::vector<vec2<ShortReal>> vlist;
    vlist.push_back(vec2<ShortReal>(0, 0));
    vlist.push_back(vec2<ShortReal>(1, 0));
    vlist.push_back(vec2<ShortReal>(0, 1.25));
    PolygonVertices verts = setup_verts(vlist, 0.25);

    ShapeSpheropolygon a(o, verts);

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

    MY_CHECK_CLOSE(a.getCircumsphereDiameter(), 3.0, tol);
    MY_CHECK_CLOSE(a.verts.sweep_radius, 0.25, tol);
    }

UP_TEST(overlap_disk)
    {
    // check the overlap of two disks using spheropolygons
    vec3<Scalar> r_ij;
    quat<Scalar> o;
    BoxDim box(100);

    // build a square
    std::vector<vec2<ShortReal>> vlist;
    vlist.push_back(vec2<ShortReal>(0, 0));
    PolygonVertices verts = setup_verts(vlist, 0.5);

    ShapeSpheropolygon a(o, verts);

    // first test, separate squares by a large distance
    ShapeSpheropolygon b(o, verts);
    r_ij = vec3<Scalar>(10, 0, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, a, b, err_count));

    // next test, set them close, but not overlapping - from all four sides
    r_ij = vec3<Scalar>(1.01, 0, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(-1.01, 0, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(0, 1.01, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(0, -1.01, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, a, b, err_count));

    // now test them close, but slightly offset and not overlapping - from all four sides
    r_ij = vec3<Scalar>(1.01, 0.2, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(-1.01, 0.2, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(-0.2, 1.01, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(-0.2, -1.01, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, a, b, err_count));

    // and finally, make them overlap slightly in each direction
    r_ij = vec3<Scalar>(0.99, 0, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(-0.99, 0.0, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(0, 0.99, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(0, -0.99, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, a, b, err_count));
    }

UP_TEST(overlap_square_no_rot)
    {
    // first set of simple overlap checks is two squares at unit orientation
    vec3<Scalar> r_ij;
    quat<Scalar> o;
    BoxDim box(100);

    // build a square
    std::vector<vec2<ShortReal>> vlist;
    vlist.push_back(vec2<ShortReal>(-0.5, -0.5));
    vlist.push_back(vec2<ShortReal>(0.5, -0.5));
    vlist.push_back(vec2<ShortReal>(0.5, 0.5));
    vlist.push_back(vec2<ShortReal>(-0.5, 0.5));
    PolygonVertices verts = setup_verts(vlist, ShortReal(0.1));

    ShapeSpheropolygon a(o, verts);

    // first test, separate squares by a large distance
    ShapeSpheropolygon b(o, verts);
    r_ij = vec3<Scalar>(10, 0, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, a, b, err_count));

    // next test, set them close, but not overlapping - from all four sides
    r_ij = vec3<Scalar>(1.25, 0, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(-1.25, 0, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(0, 1.25, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(0, -1.25, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, a, b, err_count));

    // now test them close, but slightly offset and not overlapping - from all four sides
    r_ij = vec3<Scalar>(1.25, 0.2, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(-1.25, 0.2, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(-0.2, 1.25, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(-0.2, -1.25, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, a, b, err_count));

    // and finally, make them overlap slightly in each direction
    r_ij = vec3<Scalar>(1.1, 0.2, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(-1.1, 0.2, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(-0.2, 1.1, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(-0.2, -1.1, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, a, b, err_count));

    // one additional test: line up the squares corner to corner
    r_ij = vec3<Scalar>(1.142, 1.142, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(1.142, -1.142, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(-1.142, 1.142, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(-1.142, -1.142, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(1.140, 1.140, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(1.140, -1.140, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(-1.140, 1.140, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(-1.140, -1.140, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, a, b, err_count));
    }

UP_TEST(overlap_square_disk)
    {
    // first set of simple overlap checks is two squares at unit orientation
    vec3<Scalar> r_ij;
    quat<Scalar> o;
    BoxDim box(100);

    // build a square
    std::vector<vec2<ShortReal>> vlist_sq;
    vlist_sq.push_back(vec2<ShortReal>(-0.5, -0.5));
    vlist_sq.push_back(vec2<ShortReal>(0.5, -0.5));
    vlist_sq.push_back(vec2<ShortReal>(0.5, 0.5));
    vlist_sq.push_back(vec2<ShortReal>(-0.5, 0.5));
    PolygonVertices verts_sq = setup_verts(vlist_sq, 0.0);

    std::vector<vec2<ShortReal>> vlist_dsk;
    vlist_dsk.push_back(vec2<ShortReal>(0, 0));
    PolygonVertices verts_dsk = setup_verts(vlist_dsk, 0.5);

    ShapeSpheropolygon a(o, verts_sq);

    // first test, separate squares by a large distance
    ShapeSpheropolygon b(o, verts_dsk);
    r_ij = vec3<Scalar>(10, 0, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, a, b, err_count));

    // next test, set them close, but not overlapping - from all four sides
    r_ij = vec3<Scalar>(1.01, 0, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(-1.01, 0, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(0, 1.01, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(0, -1.01, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, a, b, err_count));

    // now test them close, but slightly offset and not overlapping - from all four sides
    r_ij = vec3<Scalar>(1.01, 0.2, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(-1.01, 0.2, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(-0.2, 1.01, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(-0.2, -1.01, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, a, b, err_count));

    // and finally, make them overlap slightly in each direction
    r_ij = vec3<Scalar>(0.99, 0.2, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(-0.99, 0.2, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(-0.2, 0.99, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(-0.2, -0.99, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, a, b, err_count));
    }

UP_TEST(overlap_square_precise)
    {
    // Rounding radius
    Scalar R = 0.1;

    // set the precision
    Scalar offset = 0.00001 / 10.0;

    // test two squares just touching and barely separated to test precision
    vec3<Scalar> r_ij;
    quat<Scalar> o;
    BoxDim box(100);

    // build a square
    std::vector<vec2<ShortReal>> vlist;
    vlist.push_back(vec2<ShortReal>(-0.5, -0.5));
    vlist.push_back(vec2<ShortReal>(0.5, -0.5));
    vlist.push_back(vec2<ShortReal>(0.5, 0.5));
    vlist.push_back(vec2<ShortReal>(-0.5, 0.5));
    PolygonVertices verts = setup_verts(vlist, ShortReal(R));

    ShapeSpheropolygon a(o, verts);
    ShapeSpheropolygon b(o, verts);

    // test edge-edge non-overlaps
    r_ij = vec3<Scalar>(1 + 2 * R + offset, 0, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(-(1 + 2 * R + offset), 0, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(0, 1 + 2 * R + offset, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(0, -(1 + 2 * R + offset), 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, a, b, err_count));

    // test edge-edge overlaps
    r_ij = vec3<Scalar>(1 + 2 * R - offset, 0, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(-(1 + 2 * R - offset), 0, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(0, 1 + 2 * R - offset, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(0, -(1 + 2 * R - offset), 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, a, b, err_count));

    // vertex-vertex test
    // rotate squares by 45 degrees
    Scalar alpha = M_PI / 4.0;
    quat<Scalar> o_45(cos(alpha / 2.0),
                      (Scalar)sin(alpha / 2.0) * vec3<Scalar>(0, 0, 1)); // rotation quaternion
    a.orientation = o_45;
    b.orientation = o_45;

    // check non-overlapping
    Scalar d = sqrt(2) + 2 * R + offset;
    r_ij = vec3<Scalar>(d, 0, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(-d, 0, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(0, d, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(0, -d, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, a, b, err_count));

    // check overlaps
    d = sqrt(2) + 2 * R - offset;
    r_ij = vec3<Scalar>(d, 0, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(-d, 0, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(0, d, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(0, -d, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, a, b, err_count));

    // vertex-edge test
    // rotate only one square by 45 degrees
    a.orientation = o;
    b.orientation = o_45;

    // check non-overlapping
    d = 0.5 * (1 + sqrt(2)) + 2 * R + offset;
    r_ij = vec3<Scalar>(d, 0, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(-d, 0, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(0, d, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(0, -d, 0);
    UP_ASSERT(!test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(!test_overlap(-r_ij, a, b, err_count));

    // check overlaps
    d = 0.5 * (1 + sqrt(2)) + 2 * R - offset;
    r_ij = vec3<Scalar>(d, 0, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(-d, 0, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(0, d, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, a, b, err_count));

    r_ij = vec3<Scalar>(0, -d, 0);
    UP_ASSERT(test_overlap(r_ij, a, b, err_count));
    UP_ASSERT(test_overlap(-r_ij, a, b, err_count));
    }
