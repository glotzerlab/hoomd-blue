
#include "hoomd/hpmc/IntegratorHPMC.h"
#include "hoomd/hpmc/Moves.h"
#include "hoomd/hpmc/ShapeConvexPolygon.h"

#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN();

#include <iostream>
#include <string>

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/RandomNumbers.h"

using namespace hpmc;
using namespace hpmc::detail;
using namespace std;

unsigned int err_count = 0;

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

// void write_gle_header(ostream& o, Scalar s)
//     {
//     o << "size " << s << " " << s << endl;
//     o << "set hei 0.15" << endl;
//     o << "set just tc" << endl;
//     }

// void write_poly_gle(ostream& o, const ShapeConvexPolygon& poly, const vec3<Scalar>& pos)
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

UP_TEST( construction )
    {
    quat<Scalar> o(1.0, vec3<Scalar>(-3.0, 9.0, 6.0));

    std::vector< vec2<OverlapReal> > vlist;
    vlist.push_back(vec2<OverlapReal>(0,0));
    vlist.push_back(vec2<OverlapReal>(1,0));
    vlist.push_back(vec2<OverlapReal>(0,1.25));
    poly2d_verts verts = setup_verts(vlist);

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

UP_TEST( overlap_square_no_rot )
    {
    // first set of simple overlap checks is two squares at unit orientation
    quat<Scalar> o;
    vec3<Scalar> r_ij;
    BoxDim box(100);

    // build a square
    std::vector< vec2<OverlapReal> > vlist;
    vlist.push_back(vec2<OverlapReal>(-0.5,-0.5));
    vlist.push_back(vec2<OverlapReal>(0.5,-0.5));
    vlist.push_back(vec2<OverlapReal>(0.5,0.5));
    vlist.push_back(vec2<OverlapReal>(-0.5,0.5));
    poly2d_verts verts = setup_verts(vlist);

    ShapeConvexPolygon a(o, verts);

    // first test, separate squares by a large distance
    ShapeConvexPolygon b(o, verts);
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

    // torture test, overlap along most of a line
    // this torture test works because 1.0 and 0.5 (the polygon verts) are exactly representable in floating point
    // checking this is important because in a large MC simulation, you are certainly going to find cases where edges or
    // vertices touch exactly
    r_ij = vec3<Scalar>(1.0,0.2,0);
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

    ShapeConvexPolygon a(o_a, verts);

    // first test, separate squares by a large distance
    ShapeConvexPolygon b(o_b, verts);
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

    ShapeConvexPolygon a(o_b, verts);

    // first test, separate squares by a large distance
    ShapeConvexPolygon b(o_a, verts);
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

    ShapeConvexPolygon a(o_a, verts_a);

    // first test, separate squares by a large distance
    ShapeConvexPolygon b(o_b, verts_b);
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
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));
    }

/*UP_TEST( visual )
    {
    // place these randomly and draw them with GLE colored red if they overlap
    BoxDim box(100);

    // build a hexagon
    poly2d_verts verts_a;
    verts_a.N = 6;
    verts_a.v[0] = vec2<Scalar>(-0.5,-0.5);
    verts_a.v[1] = vec2<Scalar>(0.1,-0.4);
    verts_a.v[2] = vec2<Scalar>(0.6,0.0);
    verts_a.v[3] = vec2<Scalar>(0,0.5);
    verts_a.v[4] = vec2<Scalar>(-0.3,0.5);
    verts_a.v[5] = vec2<Scalar>(-0.6,0);
    set_radius(verts_a);

    ShapeConvexPolygon a(vec3<Scalar>(0,0,0), quat<Scalar>(), verts_a);

    // build a triangle
    poly2d_verts verts_b;
    verts_b.N = 3;
    verts_b.v[0] = vec2<Scalar>(-0.5,-0.5);
    verts_b.v[1] = vec2<Scalar>(0.5,-0.5);
    verts_b.v[2] = vec2<Scalar>(0.5,0.5);
    set_radius(verts_b);

    ShapeConvexPolygon b(vec3<Scalar>(0,0,0), quat<Scalar>(), verts_b);

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
