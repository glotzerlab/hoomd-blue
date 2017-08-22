

#include "hoomd/hpmc/IntegratorHPMC.h"
#include "hoomd/hpmc/Moves.h"
#include "hoomd/hpmc/ShapeSimplePolygon.h"

#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN();




#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <string>

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include <memory>


#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"

using namespace hpmc;
using namespace std;
using namespace hpmc::detail;

unsigned int err_count;

// helper function to compute poly radius
poly2d_verts setup_verts(const vector< vec2<OverlapReal> > vlist)
    {
    if (vlist.size() > MAX_POLY2D_VERTS)
        throw runtime_error("Too many polygon vertices");

    poly2d_verts result;
    result.N = vlist.size();
    result.ignore = 0;

    // extract the verts from the python list and compute the radius on the way
    OverlapReal radius_sq = OverlapReal(0.0);
    for (unsigned int i = 0; i < vlist.size(); i++)
        {
        vec2<OverlapReal> vert = vlist[i];
        result.x[i] = vert.x;
        result.y[i] = vert.y;
        radius_sq = std::max(radius_sq, dot(vert, vert));
        }
    for (unsigned int i = vlist.size(); i < MAX_POLY2D_VERTS; i++)
        {
        result.x[i] = 0;
        result.y[i] = 0;
        }

    // set the diameter
    result.diameter = 2*sqrt(radius_sq);

    return result;
    }

// void write_gle_header(ostream& o, OverlapReal s)
//     {
//     o << "size " << s << " " << s << endl;
//     o << "set hei 0.15" << endl;
//     o << "set just tc" << endl;
//     }

// void write_poly_gle(ostream& o, const ShapeSimplePolygon& poly, vec3<OverlapReal> pos)
//     {
//     // first transform all verts into screen space (rotation only, translation handled by an rmove)
//     poly2d_verts ss_verts = poly.verts;

//     for (unsigned int i = 0; i < ss_verts.N; i++)
//         ss_verts.v[i] = rotate(quat<OverlapReal>(poly.orientation), ss_verts.v[i]);

//     o << "rmove " << pos.x << " " << pos.y << endl;
//     o << "rmove " << ss_verts.v[0].x << " " << ss_verts.v[0].y << endl;

//     for (unsigned int i = 0; i < ss_verts.N-1; i++)
//         {
//         vec2<OverlapReal> line = ss_verts.v[i+1] - ss_verts.v[i];
//         o << "text " << i << endl;
//         o << "rline " << line.x << " " << line.y << endl;
//         }

//     o << "text " << ss_verts.N-1 << endl;
//     vec2<OverlapReal> line = ss_verts.v[0] - ss_verts.v[ss_verts.N-1];
//     o << "rline " << line.x << " " << line.y << endl;
//     }

//////////// Convex tests
// The tests below are copied and pasted from the convex tests, the simple polygon test should pass all of them

UP_TEST( construction )
    {
    quat<Scalar> o(1.0, vec3<Scalar>(-3.0, 9.0, 6.0));

    std::vector< vec2<OverlapReal> > vlist;
    vlist.push_back(vec2<OverlapReal>(0,0));
    vlist.push_back(vec2<OverlapReal>(1,0));
    vlist.push_back(vec2<OverlapReal>(0,1.25));
    poly2d_verts verts = setup_verts(vlist);

    ShapeSimplePolygon a(o, verts);

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

UP_TEST( overlap_square_no_rot )
    {
    // first set of simple overlap checks is two squares at unit orientation
    vec3<Scalar> r_ij;
    quat<Scalar> o;
    BoxDim box(100);

    // build a square
    std::vector< vec2<OverlapReal> > vlist;
    vlist.push_back(vec2<OverlapReal>(-0.5,-0.5));
    vlist.push_back(vec2<OverlapReal>(0.5,-0.5));
    vlist.push_back(vec2<OverlapReal>(0.5,0.5));
    vlist.push_back(vec2<OverlapReal>(-0.5,0.5));
    poly2d_verts verts = setup_verts(vlist);

    ShapeSimplePolygon a(o, verts);

    // first test, separate squares by a large distance
    ShapeSimplePolygon b(o, verts);
    r_ij = vec3<Scalar>(10,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // next test, set them close, but not overlapping - from all four sides
    r_ij = vec3<Scalar>(1.1,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(-1.1,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0,1.1,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0,-1.1,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // now test them close, but slightly offset and not overlapping - from all four sides
    r_ij = vec3<Scalar>(1.1,0.2,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(-1.1,0.2,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(-0.2,1.1,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(-0.2,-1.1,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // and finally, make them overlap slightly in each direction
    r_ij = vec3<Scalar>(0.9,0.2,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(-0.9,0.2,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(-0.2,0.9,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(-0.2,-0.9,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));
    }


UP_TEST( overlap_square_rot1 )
    {
    // second set of simple overlap checks is two squares, with one rotated by 45 degrees
    vec3<Scalar> r_ij;
    quat<Scalar> o_a;
    Scalar alpha = M_PI/4.0;
    quat<Scalar> o_b(cos(alpha/2.0), (Scalar)sin(alpha/2.0) * vec3<Scalar>(0,0,1)); // rotation quaternion

    BoxDim box(100);

    // build a square
    std::vector< vec2<OverlapReal> > vlist;
    vlist.push_back(vec2<OverlapReal>(-0.5,-0.5));
    vlist.push_back(vec2<OverlapReal>(0.5,-0.5));
    vlist.push_back(vec2<OverlapReal>(0.5,0.5));
    vlist.push_back(vec2<OverlapReal>(-0.5,0.5));
    poly2d_verts verts = setup_verts(vlist);

    ShapeSimplePolygon a(o_a, verts);

    // first test, separate squares by a large distance
    ShapeSimplePolygon b(o_b, verts);
    r_ij = vec3<Scalar>(10,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // next test, set them close, but not overlapping - from all four sides
    r_ij = vec3<Scalar>(1.3,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(-1.3,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0,1.3,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0,-1.3,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // now test them close, but slightly offset and not overlapping - from all four sides
    r_ij = vec3<Scalar>(1.3,0.2,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(-1.3,0.2,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(-0.2,1.3,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(-0.2,-1.3,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // and finally, make them overlap slightly in each direction
    r_ij = vec3<Scalar>(1.2,0.2,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(-1.2,0.2,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(-0.2,1.2,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(-0.2,-1.2,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));
    }

UP_TEST( overlap_square_rot2 )
    {
    // third set of simple overlap checks is two squares, with the other one rotated by 45 degrees
    vec3<Scalar> r_ij;
    quat<Scalar> o_a;
    Scalar alpha = M_PI/4.0;
    quat<Scalar> o_b(cos(alpha/2.0), (Scalar)sin(alpha/2.0) * vec3<Scalar>(0,0,1)); // rotation quaternion

    BoxDim box(100);

    // build a square
    std::vector< vec2<OverlapReal> > vlist;
    vlist.push_back(vec2<OverlapReal>(-0.5,-0.5));
    vlist.push_back(vec2<OverlapReal>(0.5,-0.5));
    vlist.push_back(vec2<OverlapReal>(0.5,0.5));
    vlist.push_back(vec2<OverlapReal>(-0.5,0.5));
    poly2d_verts verts = setup_verts(vlist);

    ShapeSimplePolygon a(o_b, verts);

    // first test, separate squares by a large distance
    ShapeSimplePolygon b(o_a, verts);
    r_ij = vec3<Scalar>(10,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // next test, set them close, but not overlapping - from all four sides
    r_ij = vec3<Scalar>(1.3,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(-1.3,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0,1.3,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0,-1.3,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // now test them close, but slightly offset and not overlapping - from all four sides
    r_ij = vec3<Scalar>(1.3,0.2,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(-1.3,0.2,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(-0.2,1.3,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(-0.2,-1.3,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // and finally, make them overlap slightly in each direction
    r_ij = vec3<Scalar>(1.2,0.2,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(-1.2,0.2,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(-0.2,1.2,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(-0.2,-1.2,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));
    }

UP_TEST( overlap_square_tri )
    {
    // and now a more complicated test, a square and a triangle - both rotated
    vec3<Scalar> r_ij;
    Scalar alpha = -M_PI/4.0;
    quat<Scalar> o_a(cos(alpha/2.0), (Scalar)sin(alpha/2.0) * vec3<Scalar>(0,0,1)); // rotation quaternion
    alpha = M_PI;
    quat<Scalar> o_b(cos(alpha/2.0), (Scalar)sin(alpha/2.0) * vec3<Scalar>(0,0,1)); // rotation quaternion

    BoxDim box(100);

    // build a square
    std::vector< vec2<OverlapReal> > vlist_a;
    vlist_a.push_back(vec2<OverlapReal>(-0.5,-0.5));
    vlist_a.push_back(vec2<OverlapReal>(0.5,-0.5));
    vlist_a.push_back(vec2<OverlapReal>(0.5,0.5));
    vlist_a.push_back(vec2<OverlapReal>(-0.5,0.5));
    poly2d_verts verts_a = setup_verts(vlist_a);

    std::vector< vec2<OverlapReal> > vlist_b;
    vlist_b.push_back(vec2<OverlapReal>(-0.5,-0.5));
    vlist_b.push_back(vec2<OverlapReal>(0.5,-0.5));
    vlist_b.push_back(vec2<OverlapReal>(0.5,0.5));
    poly2d_verts verts_b = setup_verts(vlist_b);

    ShapeSimplePolygon a(o_a, verts_a);

    // first test, separate squares by a large distance
    ShapeSimplePolygon b(o_b, verts_b);
    r_ij = vec3<Scalar>(10,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // next test, set them close, but not overlapping - from all four sides
    r_ij = vec3<Scalar>(1.3,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(-1.3,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0,1.3,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0,-1.3,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // and finally, make them overlap slightly in each direction
    r_ij = vec3<Scalar>(1.2,0.2,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(-0.7,-0.2,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0.4,1.1,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(-0.2,-1.2,0);
    //write_two_polys("test.gle", a, b);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));
    }

UP_TEST( overlap_concave_norot )
    {
    // test two concave darts
    vec3<Scalar> r_ij;
    Scalar alpha = 0;
    quat<Scalar> o_a(cos(alpha/2.0), (Scalar)sin(alpha/2.0) * vec3<Scalar>(0,0,1)); // rotation quaternion
    alpha = 0;
    quat<Scalar> o_b(cos(alpha/2.0), (Scalar)sin(alpha/2.0) * vec3<Scalar>(0,0,1)); // rotation quaternion

    BoxDim box(20);

    // build a dart
    std::vector< vec2<OverlapReal> > vlist_a;
    vlist_a.push_back(vec2<OverlapReal>(-0.5,0));
    vlist_a.push_back(vec2<OverlapReal>(0.5,-0.5));
    vlist_a.push_back(vec2<OverlapReal>(0,0));
    vlist_a.push_back(vec2<OverlapReal>(0.5,0.5));
    poly2d_verts verts_a = setup_verts(vlist_a);

    ShapeSimplePolygon a(o_a, verts_a);

    // first test, one shape inside the concavity in the other, but not overlapping
    ShapeSimplePolygon b(o_b, verts_a);
    r_ij = vec3<Scalar>(0.6,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // now set them to just overlap
    // this tests the vertex in object test
    r_ij = vec3<Scalar>(0.49,0,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    // move so one is under the other
    r_ij = vec3<Scalar>(0,1.1,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // adjust so that the wings overlap without any vertices inside either the other shape
    // this tests the edge edge checks
    r_ij = vec3<Scalar>(0,0.9,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));
    }

UP_TEST( no_overlap_bug )
    {
    // exhibit bug
    vec3<Scalar> r_ij;
    r_ij.x = std::strtof("0x1.27978ff361599p+0",NULL);
    r_ij.y = std::strtof("0x1.b744249169932p-2",NULL);
    r_ij.z = std::strtof("0x0.0000000000000p+0",NULL);

    quat<Scalar> o_i, o_j;
    o_i.s = std::strtof("0x1.fd4b3acfae4a6p-1",NULL);
    o_i.v.x = std::strtof("-0x1.7e15bb920cc00p-62",NULL);
    o_i.v.y = std::strtof("0x1.cfead6fedd6a4p-58",NULL);
    o_i.v.z = std::strtof("-0x1.a4925729b867bp-4",NULL);
    o_j.s = std::strtof("0x1.fd4b3acfae4a6p-1",NULL);
    o_j.v.x = std::strtof("-0x1.7e15bb920cc00p-62",NULL);
    o_j.v.y = std::strtof("0x1.cfead6fedd6a4p-58",NULL);
    o_j.v.z = std::strtof("-0x1.a4925729b867bp-4",NULL);

    std::vector< vec2<OverlapReal> > vlist;
    vlist.push_back(vec2<OverlapReal>());
    vlist.push_back(vec2<OverlapReal>());
    vlist.push_back(vec2<OverlapReal>());
    vlist.push_back(vec2<OverlapReal>());

    vlist[0].x = std::strtof("-0x1.7954933782004p+0",NULL);
    vlist[1].x = std::strtof("0x1.0d56d990fbff8p-1",NULL);
    vlist[2].x = std::strtof("0x1.7954933782004p+0",NULL);
    vlist[3].x = std::strtof("-0x1.0d56d990fbff8p-1",NULL);

    vlist[0].y = std::strtof("-0x1.313d7d92ca92fp-2",NULL);
    vlist[1].y = std::strtof("-0x1.313d7d92ca92fp-2",NULL);
    vlist[2].y = std::strtof("0x1.313d7d92ca92fp-2",NULL);
    vlist[3].y = std::strtof("0x1.313d7d92ca92fp-2",NULL);

    poly2d_verts verts = setup_verts(vlist);
    ShapeSimplePolygon i(o_i, verts);
    ShapeSimplePolygon j(o_j, verts);


    cout << "boxMatrix 10 0 0 0 10 0 0 0 10\n";
    cout << "poly3d 4 " << i.verts.x[0] << ' ' << i.verts.y[0] << " 0 ";
    cout << i.verts.x[1] << ' ' << i.verts.y[1] << " 0 ";
    cout << i.verts.x[2] << ' ' << i.verts.y[2] << " 0 ";
    cout << i.verts.x[3] << ' ' << i.verts.y[3] << " 0 ";
    cout << "88888844 0 0 0 " << o_i.s << ' ' << o_i.v.x << ' ' << o_i.v.y << ' ' << o_i.v.z << '\n';
    cout << "poly3d 4 " << j.verts.x[0] << ' ' << j.verts.y[0] << " 0 ";
    cout << j.verts.x[1] << ' ' << j.verts.y[1] << " 0 ";
    cout << j.verts.x[2] << ' ' << j.verts.y[2] << " 0 ";
    cout << j.verts.x[3] << ' ' << j.verts.y[3] << " 0 ";
    cout << "888888cc " << r_ij.x << ' ' << r_ij.y << ' ' << r_ij.z << ' ';
    cout << o_j.s << ' ' << o_j.v.x << ' ' << o_j.v.y << ' ' << o_j.v.z << '\n';
    cout << "eof\n";

    UP_ASSERT(!test_overlap(r_ij, i, j, err_count));
    UP_ASSERT(!test_overlap(-r_ij, j, i, err_count));
    }

UP_TEST( no_overlap_bug2 )
    {
    // exhibit bug
    vec3<Scalar> r_ij;
    r_ij.x = 0.99173007432252847;
    r_ij.y = 0;
    r_ij.z = 0;

    quat<Scalar> o_i, o_j;
    o_i.s = 0.38942358823656648;
    o_i.v.x = -3.7387022415048261e-17;
    o_i.v.y = -5.6398584095352949e-17;
    o_i.v.z = 0.92105877604252651;
    o_j = o_i;

    std::vector< vec2<OverlapReal> > vlist;
    vlist.push_back(vec2<OverlapReal>());
    vlist.push_back(vec2<OverlapReal>());
    vlist.push_back(vec2<OverlapReal>());
    vlist.push_back(vec2<OverlapReal>());

    vlist[0].x = -1.31406224;
    vlist[1].x = 0.685937762;
    vlist[2].x = 1.31406224;
    vlist[3].x = -0.685937762;

    vlist[0].y = -0.323377937;
    vlist[1].y = -0.323377937;
    vlist[2].y = 0.323377937;
    vlist[3].y = 0.323377937;

    poly2d_verts verts = setup_verts(vlist);
    ShapeSimplePolygon i(o_i, verts);
    ShapeSimplePolygon j(o_j, verts);


    cout << "boxMatrix 10 0 0 0 10 0 0 0 10\n";
    cout << "poly3d 4 " << i.verts.x[0] << ' ' << i.verts.y[0] << " 0 ";
    cout << i.verts.x[1] << ' ' << i.verts.y[1] << " 0 ";
    cout << i.verts.x[2] << ' ' << i.verts.y[2] << " 0 ";
    cout << i.verts.x[3] << ' ' << i.verts.y[3] << " 0 ";
    cout << "88888844 0 0 0 " << o_i.s << ' ' << o_i.v.x << ' ' << o_i.v.y << ' ' << o_i.v.z << '\n';
    cout << "poly3d 4 " << j.verts.x[0] << ' ' << j.verts.y[0] << " 0 ";
    cout << j.verts.x[1] << ' ' << j.verts.y[1] << " 0 ";
    cout << j.verts.x[2] << ' ' << j.verts.y[2] << " 0 ";
    cout << j.verts.x[3] << ' ' << j.verts.y[3] << " 0 ";
    cout << "888888cc " << r_ij.x << ' ' << r_ij.y << ' ' << r_ij.z << ' ';
    cout << o_j.s << ' ' << o_j.v.x << ' ' << o_j.v.y << ' ' << o_j.v.z << '\n';
    cout << "eof\n";

    UP_ASSERT(!test_overlap(r_ij, i, j, err_count));
    UP_ASSERT(!test_overlap(-r_ij, j, i, err_count));
    }


UP_TEST( no_overlap_bug3 )
    {
    // exhibit bug
    vec3<Scalar> r_ij;
    r_ij.x = 1.8710188222821107;
    r_ij.y = 0;
    r_ij.z = 0;

    quat<Scalar> o_i, o_j;
    o_i.s = 0.48003594826374163;
    o_i.v.x = -3.1838615581524877e-17;
    o_i.v.y = 5.3715997840230335e-17;
    o_i.v.z = -0.87724881782452713;
    o_j = o_i;

    std::vector< vec2<OverlapReal> > vlist;
    vlist.push_back(vec2<OverlapReal>());
    vlist.push_back(vec2<OverlapReal>());
    vlist.push_back(vec2<OverlapReal>());
    vlist.push_back(vec2<OverlapReal>());

    vlist[0].x = -0.541489005;
    vlist[1].x = 1.45851099;
    vlist[2].x = 0.541489005;
    vlist[3].x = -1.45851099;

    vlist[0].y = -0.716278672;
    vlist[1].y = -0.716278672;
    vlist[2].y = 0.716278672;
    vlist[3].y = 0.716278672;

    poly2d_verts verts = setup_verts(vlist);
    ShapeSimplePolygon i(o_i, verts);
    ShapeSimplePolygon j(o_j, verts);


    cout << "boxMatrix 10 0 0 0 10 0 0 0 10\n";
    cout << "poly3d 4 " << i.verts.x[0] << ' ' << i.verts.y[0] << " 0 ";
    cout << i.verts.x[1] << ' ' << i.verts.y[1] << " 0 ";
    cout << i.verts.x[2] << ' ' << i.verts.y[2] << " 0 ";
    cout << i.verts.x[3] << ' ' << i.verts.y[3] << " 0 ";
    cout << "88888844 0 0 0 " << o_i.s << ' ' << o_i.v.x << ' ' << o_i.v.y << ' ' << o_i.v.z << '\n';
    cout << "poly3d 4 " << j.verts.x[0] << ' ' << j.verts.y[0] << " 0 ";
    cout << j.verts.x[1] << ' ' << j.verts.y[1] << " 0 ";
    cout << j.verts.x[2] << ' ' << j.verts.y[2] << " 0 ";
    cout << j.verts.x[3] << ' ' << j.verts.y[3] << " 0 ";
    cout << "888888cc " << r_ij.x << ' ' << r_ij.y << ' ' << r_ij.z << ' ';
    cout << o_j.s << ' ' << o_j.v.x << ' ' << o_j.v.y << ' ' << o_j.v.z << '\n';
    cout << "eof\n";

    UP_ASSERT(!test_overlap(r_ij, i, j, err_count));
    UP_ASSERT(!test_overlap(-r_ij, j, i, err_count));
    }


/*UP_TEST( visual )
    {
    // place these randomly and draw them with GLE colored red if they overlap
    BoxDim box(100);

    // build a dart
    poly2d_verts verts_a;
    verts_a.N = 4;
    verts_a.v[0] = vec2<Scalar>(-0.5,0);
    verts_a.v[1] = vec2<Scalar>(0.5,-0.5);
    verts_a.v[2] = vec2<Scalar>(0,0);
    verts_a.v[3] = vec2<Scalar>(0.5,0.5);
    set_radius(verts_a);

    ShapeSimplePolygon a(quat<Scalar>(), verts_a);

    // build an indented square
    poly2d_verts verts_b;
    verts_b.N = 5;
    verts_b.v[0] = vec2<Scalar>(-0.5,-0.5);
    verts_b.v[1] = vec2<Scalar>(0.5,-0.5);
    verts_b.v[2] = vec2<Scalar>(0.5,0.5);
    verts_b.v[3] = vec2<Scalar>(0,0);
    verts_b.v[3] = vec2<Scalar>(-0.5,0.5);
    set_radius(verts_b);

    ShapeSimplePolygon b(vec3<Scalar>(0,0,0), quat<Scalar>(), verts_b);

    // start up the random number generator
    Saru rng(123, 456, 789);

    // do 16x16 tests, each drawn in a 4x4cm space
    unsigned int m = 16;
    Scalar w = 4;
    Scalar s = m*w;

    // place the shapes from max_d to min_d spacing apart
    Scalar max_d = (a.getCircumsphereDiameter() + b.getCircumsphereDiameter()) / 2.0;
    Scalar min_d = -max_d;

    // write to the GLE file as we go
    ofstream f("visual.gle");
    write_gle_header(f, s);
    f << "set lwidth 0.01" << endl;

    for (unsigned int i = 0; i < m; i++)
        for (unsigned int j = 0; j < m; j++)
            {
            // place the shapes randomly
            Scalar alpha = rng.s(-M_PI, M_PI);
            a.orientation = quat<Scalar>(cos(alpha/2.0), (Scalar)sin(alpha/2.0) * vec3<Scalar>(0,0,1));
            a.pos = vec3<Scalar>(0,0,0);

            alpha = rng.s(-M_PI, M_PI);
            b.orientation = quat<Scalar>(cos(alpha/2.0), (Scalar)sin(alpha/2.0) * vec3<Scalar>(0,0,1));
            r_ij = vec3<Scalar>(rng.s(min_d, max_d),rng.s(min_d, max_d), 0);

            // translate to the current space on the plot
            f << "begin translate " << i*w + w/2.0 << " " << j*w + w/2.0 << endl;

            // color the lines red if they overlap, black otherwise
            if (test_overlap(r_ij,a,b,err_count))
                f << "set color red" << endl;
            else
                f << "set color black" << endl;

            // draw the polygons
            f << "amove " << 0 << " " << 0 << endl;
            f << "text " << i << " " << j << endl;
            f << "amove " << 0 << " " << 0 << endl;
            write_poly_gle(f, a);

            f << "amove " << 0 << " " << 0 << endl;
            write_poly_gle(f, b);

            f << "end translate" << endl << endl;
            }

    }*/
