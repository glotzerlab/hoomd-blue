

#include "hoomd/hpmc/IntegratorHPMC.h"
#include "hoomd/hpmc/Moves.h"
#include "hoomd/hpmc/ShapePolyhedron.h"
#include "hoomd/AABBTree.h"

//! Name the unit test module
#define BOOST_TEST_MODULE ShapePolyhedron
#include "boost_utf_configure.h"

#include <iostream>
#include <string>

#include <boost/bind.hpp>
#include <boost/python.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

using namespace hpmc;
using namespace std;
using namespace hpmc::detail;

unsigned int err_count;

/*!
 * Currently, we test convex shapes only
 */

// helper function to compute poly radius
void set_radius(poly3d_data& data)
    {
    OverlapReal radius_sq = OverlapReal(0.0);
    for (unsigned int i = 0; i < data.verts.N; i++)
        {
        vec3<OverlapReal> vert(data.verts.x[i], data.verts.y[i], data.verts.z[i]);
        radius_sq = std::max(radius_sq, dot(vert, vert));
        }

    // set the diameter
    data.verts.diameter = 2*sqrt(radius_sq);
    }

hpmc::detail::GPUTree build_tree(poly3d_data &data)
    {
    hpmc::detail::AABBTree tree;
    hpmc::detail::AABB *aabbs;
    int retval = posix_memalign((void**)&aabbs, 32, sizeof(hpmc::detail::AABB)*data.n_faces);
    if (retval != 0)
        {
        throw std::runtime_error("Error allocating aligned AABB memory.");
        }

    // construct bounding box tree
    for (unsigned int i = 0; i < data.n_faces; ++i)
        {
        vec3<OverlapReal> cm(0,0,0);
        unsigned int l = data.face_offs[i+1] - data.face_offs[i];
        for (unsigned int j = data.face_offs[i]; j < data.face_offs[i+1]; ++j)
            {
            vec3<OverlapReal> v(data.verts.x[data.face_verts[j]], data.verts.y[data.face_verts[j]], data.verts.z[data.face_verts[j]]);
            cm += v;
            }
        cm = cm*OverlapReal(1.0/l);
        vec3<OverlapReal> lo = cm;
        vec3<OverlapReal> hi = cm;
        for (unsigned int j = data.face_offs[i]; j < data.face_offs[i+1]; ++j)
            {
            vec3<OverlapReal> v(data.verts.x[data.face_verts[j]], data.verts.y[data.face_verts[j]], data.verts.z[data.face_verts[j]]);
            if (v.x < lo.x) lo.x = v.x;
            if (v.y < lo.y) lo.y = v.y;
            if (v.z < lo.z) lo.z = v.z;

            if (v.x > hi.x) hi.x = v.x;
            if (v.y > hi.y) hi.y = v.y;
            if (v.z > hi.z) hi.z = v.z;
            }
        aabbs[i] = hpmc::detail::AABB(vec3<Scalar>(lo.x,lo.y,lo.z),vec3<Scalar>(hi.x,hi.y,hi.z));
        }
    tree.buildTree(aabbs, data.n_faces);
    GPUTree gpu_tree(tree,aabbs);
    free(aabbs);
    return gpu_tree;
    }

BOOST_AUTO_TEST_CASE( construction )
    {
    quat<Scalar> o(1.0, vec3<Scalar>(-3.0, 9.0, 6.0));

    poly3d_data data;
    data.verts.N = 4;
    data.n_edges = 4;
    data.n_faces = 1;
    data.verts.x[0] = 0; data.verts.y[0] = 0; data.verts.z[0] = 0;
    data.verts.x[1] = 1; data.verts.y[1] = 0; data.verts.z[1] = 0;
    data.verts.x[2] = 0; data.verts.y[2] = 1.25; data.verts.z[2] = 0;
    data.verts.x[3] = 0; data.verts.y[3] = 0; data.verts.z[3] = 1.1;
    data.edges[0] = 0; data.edges[1] = 1;
    data.edges[2] = 1; data.edges[3] = 2;
    data.edges[4] = 2; data.edges[5] = 3;
    data.edges[5] = 3; data.edges[6] = 0;
    data.face_verts[0] = 0;
    data.face_verts[1] = 1;
    data.face_verts[2] = 2;
    data.face_verts[3] = 3;
    data.face_offs[0] = 0;
    data.face_offs[1] = 4;
    data.ignore = 0;
    set_radius(data);

    ShapePolyhedron::param_type p;
    p.data = data;
    p.tree = build_tree(data);
    ShapePolyhedron a(o, p);

    MY_BOOST_CHECK_CLOSE(a.orientation.s, o.s, tol);
    MY_BOOST_CHECK_CLOSE(a.orientation.v.x, o.v.x, tol);
    MY_BOOST_CHECK_CLOSE(a.orientation.v.y, o.v.y, tol);
    MY_BOOST_CHECK_CLOSE(a.orientation.v.z, o.v.z, tol);

    BOOST_REQUIRE_EQUAL(a.data.verts.N, data.verts.N);
    for (unsigned int i = 0; i < data.verts.N; i++)
        {
        MY_BOOST_CHECK_CLOSE(a.data.verts.x[i], data.verts.x[i], tol);
        MY_BOOST_CHECK_CLOSE(a.data.verts.y[i], data.verts.y[i], tol);
        MY_BOOST_CHECK_CLOSE(a.data.verts.z[i], data.verts.z[i], tol);
        }

    BOOST_REQUIRE_EQUAL(a.data.n_edges, data.n_edges);
    for (unsigned int i = 0; i < data.n_edges; i++)
        {
        BOOST_CHECK_EQUAL(a.data.edges[2*i], data.edges[2*i]);
        BOOST_CHECK_EQUAL(a.data.edges[2*i+1], data.edges[2*i+1]);
        }

    BOOST_REQUIRE_EQUAL(a.data.n_faces, data.n_faces);
    for (unsigned int i = 0; i < data.n_faces; i++)
        {
        BOOST_CHECK_EQUAL(a.data.face_offs[i], data.face_offs[i]);
        unsigned int offs = a.data.face_offs[i];
        for (unsigned int j = offs; j < a.data.face_offs[i+1]; ++j)
            BOOST_CHECK_EQUAL(a.data.face_verts[j], data.face_verts[j]);
        }

    BOOST_CHECK_EQUAL(a.data.face_offs[data.n_faces], data.face_offs[data.n_faces]);
    BOOST_CHECK(a.hasOrientation());

    MY_BOOST_CHECK_CLOSE(a.getCircumsphereDiameter(), 2.5, tol);
    }

BOOST_AUTO_TEST_CASE( overlap_octahedron_no_rot )
    {
    // first set of simple overlap checks is two octahedra at unit orientation
    vec3<Scalar> r_ij;
    quat<Scalar> o;
    BoxDim box(100);

    // build an octahedron
    poly3d_data data;
    BOOST_REQUIRE(MAX_POLY3D_VERTS >= 6);
    data.verts.N = 6;
    BOOST_REQUIRE(MAX_POLY3D_FACES >= 8);
    data.n_faces = 8;
    BOOST_REQUIRE(MAX_POLY3D_EDGES >= 12);
    data.n_edges = 12;

    data.verts.x[0] = -0.5; data.verts.y[0] = -0.5; data.verts.z[0] = 0; // vec3<OverlapReal>(-0.5,-0.5,0);
    data.verts.x[1] = 0.5; data.verts.y[1] = -0.5; data.verts.z[1] = 0; //vec3<OverlapReal>(0.5,-0.5,0);
    data.verts.x[2] = 0.5; data.verts.y[2] = 0.5; data.verts.z[2] = 0; //vec3<OverlapReal>(0.5,0.5,0);
    data.verts.x[3] = -0.5; data.verts.y[3] = 0.5; data.verts.z[3] = 0; // vec3<OverlapReal>(-0.5,0.5,0);
    data.verts.x[4] = 0; data.verts.y[4] = 0; data.verts.z[4] = 0.707106781186548; // vec3<OverlapReal>(0,0,0.707106781186548);
    data.verts.x[5] = 0; data.verts.y[5] = 0; data.verts.z[5] = -0.707106781186548; // vec3<OverlapReal>(0,0,-0.707106781186548);
    data.edges[2*0] = 0; data.edges[2*0+1] = 1;
    data.edges[2*1] = 1; data.edges[2*1+1] = 2;
    data.edges[2*2] = 2; data.edges[2*2+1] = 3;
    data.edges[2*3] = 3; data.edges[2*3+1] = 0;
    data.edges[2*4] = 0; data.edges[2*4+1] = 4;
    data.edges[2*5] = 1; data.edges[2*5+1] = 4;
    data.edges[2*6] = 2; data.edges[2*6+1] = 4;
    data.edges[2*7] = 3; data.edges[2*7+1] = 4;
    data.edges[2*8] = 0; data.edges[2*8+1] = 5;
    data.edges[2*9] = 1; data.edges[2*9+1] = 5;
    data.edges[2*10] = 2; data.edges[2*10+1] = 5;
    data.edges[2*11] = 3; data.edges[2*11+1] = 5;
    BOOST_REQUIRE(MAX_POLY3D_FACE_VERTS >= 3);
    data.face_offs[0] = 0;
    data.face_verts[0] = 0; data.face_verts[1] = 4; data.face_verts[2] = 1;
    data.face_offs[1] = 3;
    data.face_verts[3] = 1; data.face_verts[4] = 4; data.face_verts[5] = 2;
    data.face_offs[2] = 6;
    data.face_verts[6] = 2; data.face_verts[7] = 4; data.face_verts[8] = 3;
    data.face_offs[3] = 9;
    data.face_verts[9] = 3; data.face_verts[10] = 4; data.face_verts[11] = 0;
    data.face_offs[4] = 12;
    data.face_verts[12] = 0; data.face_verts[13] = 5; data.face_verts[14] = 1;
    data.face_offs[5] = 15;
    data.face_verts[15] = 1; data.face_verts[16] = 5; data.face_verts[17] = 2;
    data.face_offs[6] = 18;
    data.face_verts[18] = 2; data.face_verts[19] = 5; data.face_verts[20] = 3;
    data.face_offs[7] = 21;
    data.face_verts[21] = 3; data.face_verts[22] = 5; data.face_verts[23] = 0;
    data.face_offs[8] = 24;
    data.ignore = 0;
    set_radius(data);

    ShapePolyhedron::param_type p;
    p.data = data;
    p.tree = build_tree(data);

    ShapePolyhedron a(o, p);
    ShapePolyhedron b(o, p);

    // zeroth test: exactly overlapping shapes
    r_ij =  vec3<Scalar>(0.0, 0.0, 0.0);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    // first test, separate octahedrons by a large distance
    r_ij =  vec3<Scalar>(10,0,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    // next test, set them close, but not overlapping - from all four sides of base
    r_ij =  vec3<Scalar>(1.1,0,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-1.1,0,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,1.1,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,-1.1,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    // now test them close, but slightly offset and not overlapping - from all four sides
    r_ij =  vec3<Scalar>(1.1,0.2,0.1);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-1.1,0.2,0.1);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,1.1,0.1);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,-1.1,0.1);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    // and finally, make them overlap slightly in each direction
    r_ij =  vec3<Scalar>(0.9,0.2,0);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.9,0.2,0);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,0.9,0);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,-0.9,0);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    // torture test, overlap along most of a line
    r_ij =  vec3<Scalar>(1.0,0.2,0);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));
    }

BOOST_AUTO_TEST_CASE( overlap_cube_no_rot )
    {
    // first set of simple overlap checks is two cubes at unit orientation
    vec3<Scalar> r_ij;
    quat<Scalar> o;
    BoxDim box(100);

    // build a cube
    poly3d_data data;
    BOOST_REQUIRE(MAX_POLY3D_VERTS >= 8);
    data.verts.N = 8;
    data.verts.x[0] = -0.5; data.verts.y[0] = -0.5; data.verts.z[0] = -0.5;  // vec3<OverlapReal>(-0.5,-0.5,-0.5);
    data.verts.x[1] = 0.5; data.verts.y[1] = -0.5; data.verts.z[1] = -0.5;  //vec3<OverlapReal>(0.5,-0.5,-0.5);
    data.verts.x[2] = 0.5; data.verts.y[2] = 0.5; data.verts.z[2] = -0.5;  //vec3<OverlapReal>(0.5,0.5,-0.5);
    data.verts.x[3] = -0.5; data.verts.y[3] = 0.5; data.verts.z[3] = -0.5;  //vec3<OverlapReal>(-0.5,0.5,-0.5);
    data.verts.x[4] = -0.5; data.verts.y[4] = -0.5; data.verts.z[4] = 0.5;  //vec3<OverlapReal>(-0.5,-0.5,0.5);
    data.verts.x[5] = 0.5; data.verts.y[5] = -0.5; data.verts.z[5] = 0.5;  //vec3<OverlapReal>(0.5,-0.5,0.5);
    data.verts.x[6] = 0.5; data.verts.y[6] = 0.5; data.verts.z[6] = 0.5;  //vec3<OverlapReal>(0.5,0.5,0.5);
    data.verts.x[7] = -0.5; data.verts.y[7] = 0.5; data.verts.z[7] = 0.5;  //vec3<OverlapReal>(-0.5,0.5,0.5);

    BOOST_REQUIRE(MAX_POLY3D_EDGES >= 12);
    data.n_edges = 12;
    data.edges[2*0] = 0; data.edges[2*0+1] = 1;
    data.edges[2*1] = 1; data.edges[2*1+1] = 2;
    data.edges[2*2] = 2; data.edges[2*2+1] = 3;
    data.edges[2*3] = 3; data.edges[2*3+1] = 0;
    data.edges[2*4] = 0; data.edges[2*4+1] = 4;
    data.edges[2*5] = 1; data.edges[2*5+1] = 5;
    data.edges[2*6] = 2; data.edges[2*6+1] = 6;
    data.edges[2*7] = 3; data.edges[2*7+1] = 7;
    data.edges[2*8] = 4; data.edges[2*8+1] = 5;
    data.edges[2*9] = 5; data.edges[2*9+1] = 6;
    data.edges[2*10] = 6; data.edges[2*10+1] = 7;
    data.edges[2*11] = 7; data.edges[2*11+1] = 4;

    BOOST_REQUIRE(MAX_POLY3D_FACES >= 6);
    data.n_faces = 6;
    BOOST_REQUIRE(MAX_POLY3D_FACE_VERTS >= 4);
    data.face_offs[0] = 0;
    data.face_verts[0] = 0; data.face_verts[1] = 1; data.face_verts[2] = 2; data.face_verts[3] = 3;
    data.face_offs[1] = 4;
    data.face_verts[4] = 0; data.face_verts[5] = 3; data.face_verts[6] = 7; data.face_verts[7] = 4;
    data.face_offs[2] = 8;
    data.face_verts[8] = 0; data.face_verts[9] = 4; data.face_verts[10] = 5; data.face_verts[11] = 1;
    data.face_offs[3] = 12;
    data.face_verts[12] = 4; data.face_verts[13] = 5; data.face_verts[14] = 6; data.face_verts[15] = 7;
    data.face_offs[4] = 16;
    data.face_verts[16] = 6; data.face_verts[17] = 7; data.face_verts[18] = 3; data.face_verts[19] = 2;
    data.face_offs[5] = 20;
    data.face_verts[20] = 1; data.face_verts[21] = 2; data.face_verts[22] = 6; data.face_verts[23] = 5;
    data.face_offs[6] = 24;

    data.ignore=0;

    set_radius(data);

    ShapePolyhedron::param_type p;
    p.data = data;
    p.tree = build_tree(data);
    ShapePolyhedron a(o, p);

    // first test, separate cubes by a large distance
    ShapePolyhedron b(o, p);
    r_ij = vec3<Scalar>(10,0,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    // next test, set them close, but not overlapping - from all four sides
    r_ij =  vec3<Scalar>(1.1,0,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-1.1,0,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,1.1,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,-1.1,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    // now test them close, but slightly offset and not overlapping - from all four sides
    r_ij =  vec3<Scalar>(1.1,0.2,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-1.1,0.2,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,1.1,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,-1.1,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    // make them overlap slightly in each direction
    r_ij =  vec3<Scalar>(0.9,0.2,0);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.9,0.2,0);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,0.9,0);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,-0.9,0);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    // Make them overlap a lot
    r_ij =  vec3<Scalar>(0.2,0,0);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,0.2,0);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0.2,tol,tol);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0.1,0.2,0.1);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    // torture test, overlap along most of a line
    r_ij =  vec3<Scalar>(1.0,0.2,0);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));
    }


BOOST_AUTO_TEST_CASE( overlap_cube_rot1 )
    {
    // second set of simple overlap checks is two cubes, with one rotated by 45 degrees
    vec3<Scalar> r_ij;
    quat<Scalar> o_a;
    Scalar alpha = M_PI/4.0;
    quat<Scalar> o_b(cos(alpha/2.0), (Scalar)sin(alpha/2.0) * vec3<Scalar>(0,0,1)); // rotation quaternion

    BoxDim box(100);

    // build a cube
    poly3d_data data;
    BOOST_REQUIRE(MAX_POLY3D_VERTS >= 8);
    data.verts.N = 8;
    data.verts.x[0] = -0.5; data.verts.y[0] = -0.5; data.verts.z[0] = -0.5;  // vec3<OverlapReal>(-0.5,-0.5,-0.5);
    data.verts.x[1] = 0.5; data.verts.y[1] = -0.5; data.verts.z[1] = -0.5;  //vec3<OverlapReal>(0.5,-0.5,-0.5);
    data.verts.x[2] = 0.5; data.verts.y[2] = 0.5; data.verts.z[2] = -0.5;  //vec3<OverlapReal>(0.5,0.5,-0.5);
    data.verts.x[3] = -0.5; data.verts.y[3] = 0.5; data.verts.z[3] = -0.5;  //vec3<OverlapReal>(-0.5,0.5,-0.5);
    data.verts.x[4] = -0.5; data.verts.y[4] = -0.5; data.verts.z[4] = 0.5;  //vec3<OverlapReal>(-0.5,-0.5,0.5);
    data.verts.x[5] = 0.5; data.verts.y[5] = -0.5; data.verts.z[5] = 0.5;  //vec3<OverlapReal>(0.5,-0.5,0.5);
    data.verts.x[6] = 0.5; data.verts.y[6] = 0.5; data.verts.z[6] = 0.5;  //vec3<OverlapReal>(0.5,0.5,0.5);
    data.verts.x[7] = -0.5; data.verts.y[7] = 0.5; data.verts.z[7] = 0.5;  //vec3<OverlapReal>(-0.5,0.5,0.5);

    BOOST_REQUIRE(MAX_POLY3D_EDGES >= 12);
    data.n_edges = 12;
    data.edges[2*0] = 0; data.edges[2*0+1] = 1;
    data.edges[2*1] = 1; data.edges[2*1+1] = 2;
    data.edges[2*2] = 2; data.edges[2*2+1] = 3;
    data.edges[2*3] = 3; data.edges[2*3+1] = 0;
    data.edges[2*4] = 0; data.edges[2*4+1] = 4;
    data.edges[2*5] = 1; data.edges[2*5+1] = 5;
    data.edges[2*6] = 2; data.edges[2*6+1] = 6;
    data.edges[2*7] = 3; data.edges[2*7+1] = 7;
    data.edges[2*8] = 4; data.edges[2*8+1] = 5;
    data.edges[2*9] = 5; data.edges[2*9+1] = 6;
    data.edges[2*10] = 6; data.edges[2*10+1] = 7;
    data.edges[2*11] = 7; data.edges[2*11+1] = 4;

    BOOST_REQUIRE(MAX_POLY3D_FACES >= 6);
    data.n_faces = 6;
    BOOST_REQUIRE(MAX_POLY3D_FACE_VERTS >= 4);
    data.face_offs[0] = 0;
    data.face_verts[0] = 0; data.face_verts[1] = 1; data.face_verts[2] = 2; data.face_verts[3] = 3;
    data.face_offs[1] = 4;
    data.face_verts[4] = 0; data.face_verts[5] = 3; data.face_verts[6] = 7; data.face_verts[7] = 4;
    data.face_offs[2] = 8;
    data.face_verts[8] = 0; data.face_verts[9] = 4; data.face_verts[10] = 5; data.face_verts[11] = 1;
    data.face_offs[3] = 12;
    data.face_verts[12] = 4; data.face_verts[13] = 5; data.face_verts[14] = 6; data.face_verts[15] = 7;
    data.face_offs[4] = 16;
    data.face_verts[16] = 6; data.face_verts[17] = 7; data.face_verts[18] = 3; data.face_verts[19] = 2;
    data.face_offs[5] = 20;
    data.face_verts[20] = 1; data.face_verts[21] = 2; data.face_verts[22] = 6; data.face_verts[23] = 5;
    data.face_offs[6] = 24;

    data.ignore=0;

    set_radius(data);

    ShapePolyhedron::param_type p;
    p.data = data;
    p.tree = build_tree(data);
    ShapePolyhedron a(o_a, p);

    // first test, separate cubes by a large distance
    ShapePolyhedron b(o_b, p);
    r_ij = vec3<Scalar>(10,0,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    // next test, set them close, but not overlapping - from all four sides
    r_ij =  vec3<Scalar>(1.3,0,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-1.3,0,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,1.3,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,-1.3,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    // now test them close, but slightly offset and not overlapping - from all four sides
    r_ij =  vec3<Scalar>(1.3,0.2,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-1.3,0.2,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,1.3,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,-1.3,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    // and finally, make them overlap slightly in each direction
    r_ij =  vec3<Scalar>(1.2,0.2,0);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-1.2,0.2,0);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,1.2,0);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,-1.2,0);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));
    }

BOOST_AUTO_TEST_CASE( overlap_cube_rot2 )
    {
    // third set of simple overlap checks is two cubes, with the other one rotated by 45 degrees
    vec3<Scalar> r_ij;
    quat<Scalar> o_a;
    Scalar alpha = M_PI/4.0;
    quat<Scalar> o_b(cos(alpha/2.0), (Scalar)sin(alpha/2.0) * vec3<Scalar>(0,0,1)); // rotation quaternion

    BoxDim box(100);

    // build a cube
    poly3d_data data;
    BOOST_REQUIRE(MAX_POLY3D_VERTS >= 8);
    data.verts.N = 8;
    data.verts.x[0] = -0.5; data.verts.y[0] = -0.5; data.verts.z[0] = -0.5;  // vec3<OverlapReal>(-0.5,-0.5,-0.5);
    data.verts.x[1] = 0.5; data.verts.y[1] = -0.5; data.verts.z[1] = -0.5;  //vec3<OverlapReal>(0.5,-0.5,-0.5);
    data.verts.x[2] = 0.5; data.verts.y[2] = 0.5; data.verts.z[2] = -0.5;  //vec3<OverlapReal>(0.5,0.5,-0.5);
    data.verts.x[3] = -0.5; data.verts.y[3] = 0.5; data.verts.z[3] = -0.5;  //vec3<OverlapReal>(-0.5,0.5,-0.5);
    data.verts.x[4] = -0.5; data.verts.y[4] = -0.5; data.verts.z[4] = 0.5;  //vec3<OverlapReal>(-0.5,-0.5,0.5);
    data.verts.x[5] = 0.5; data.verts.y[5] = -0.5; data.verts.z[5] = 0.5;  //vec3<OverlapReal>(0.5,-0.5,0.5);
    data.verts.x[6] = 0.5; data.verts.y[6] = 0.5; data.verts.z[6] = 0.5;  //vec3<OverlapReal>(0.5,0.5,0.5);
    data.verts.x[7] = -0.5; data.verts.y[7] = 0.5; data.verts.z[7] = 0.5;  //vec3<OverlapReal>(-0.5,0.5,0.5);

    BOOST_REQUIRE(MAX_POLY3D_EDGES >= 12);
    data.n_edges = 12;
    data.edges[2*0] = 0; data.edges[2*0+1] = 1;
    data.edges[2*1] = 1; data.edges[2*1+1] = 2;
    data.edges[2*2] = 2; data.edges[2*2+1] = 3;
    data.edges[2*3] = 3; data.edges[2*3+1] = 0;
    data.edges[2*4] = 0; data.edges[2*4+1] = 4;
    data.edges[2*5] = 1; data.edges[2*5+1] = 5;
    data.edges[2*6] = 2; data.edges[2*6+1] = 6;
    data.edges[2*7] = 3; data.edges[2*7+1] = 7;
    data.edges[2*8] = 4; data.edges[2*8+1] = 5;
    data.edges[2*9] = 5; data.edges[2*9+1] = 6;
    data.edges[2*10] = 6; data.edges[2*10+1] = 7;
    data.edges[2*11] = 7; data.edges[2*11+1] = 4;

    BOOST_REQUIRE(MAX_POLY3D_FACES >= 6);
    data.n_faces = 6;
    BOOST_REQUIRE(MAX_POLY3D_FACE_VERTS >= 4);
    data.face_offs[0] = 0;
    data.face_verts[0] = 0; data.face_verts[1] = 1; data.face_verts[2] = 2; data.face_verts[3] = 3;
    data.face_offs[1] = 4;
    data.face_verts[4] = 0; data.face_verts[5] = 3; data.face_verts[6] = 7; data.face_verts[7] = 4;
    data.face_offs[2] = 8;
    data.face_verts[8] = 0; data.face_verts[9] = 4; data.face_verts[10] = 5; data.face_verts[11] = 1;
    data.face_offs[3] = 12;
    data.face_verts[12] = 4; data.face_verts[13] = 5; data.face_verts[14] = 6; data.face_verts[15] = 7;
    data.face_offs[4] = 16;
    data.face_verts[16] = 6; data.face_verts[17] = 7; data.face_verts[18] = 3; data.face_verts[19] = 2;
    data.face_offs[5] = 20;
    data.face_verts[20] = 1; data.face_verts[21] = 2; data.face_verts[22] = 6; data.face_verts[23] = 5;
    data.face_offs[6] = 24;

    data.ignore = 0;

    set_radius(data);

    ShapePolyhedron::param_type p;
    p.data = data;
    p.tree = build_tree(data);
    ShapePolyhedron a(o_b, p);

    // first test, separate cubes by a large distance
    ShapePolyhedron b(o_a, p);
    r_ij = vec3<Scalar>(10,0,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    // next test, set them close, but not overlapping - from all four sides
    r_ij =  vec3<Scalar>(1.3,0,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-1.3,0,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,1.3,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,-1.3,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    // now test them close, but slightly offset and not overlapping - from all four sides
    r_ij =  vec3<Scalar>(1.3,0.2,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-1.3,0.2,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,1.3,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,-1.3,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    // and finally, make them overlap slightly in each direction
    r_ij =  vec3<Scalar>(1.2,0.2,0);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-1.2,0.2,0);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,1.2,0);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,-1.2,0);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));
    }

BOOST_AUTO_TEST_CASE( overlap_cube_rot3 )
    {
    // two cubes, with one rotated by 45 degrees around two axes. This lets us look at edge-edge and point-face collisions
    quat<Scalar> o_a;
    vec3<Scalar> r_ij;
    Scalar alpha = M_PI/4.0;
    // rotation around x and then z
    const quat<Scalar> q1(cos(alpha/2.0), (Scalar)sin(alpha/2.0) * vec3<Scalar>(1,0,0));
    const quat<Scalar> q2(cos(alpha/2.0), (Scalar)sin(alpha/2.0) * vec3<Scalar>(0,0,1));
    quat<Scalar> o_b(q2 * q1);

    BoxDim box(100);

    // build a cube
    poly3d_data data;
    BOOST_REQUIRE(MAX_POLY3D_VERTS >= 8);
    data.verts.N = 8;
    data.verts.x[0] = -0.5; data.verts.y[0] = -0.5; data.verts.z[0] = -0.5;  // vec3<OverlapReal>(-0.5,-0.5,-0.5);
    data.verts.x[1] = 0.5; data.verts.y[1] = -0.5; data.verts.z[1] = -0.5;  //vec3<OverlapReal>(0.5,-0.5,-0.5);
    data.verts.x[2] = 0.5; data.verts.y[2] = 0.5; data.verts.z[2] = -0.5;  //vec3<OverlapReal>(0.5,0.5,-0.5);
    data.verts.x[3] = -0.5; data.verts.y[3] = 0.5; data.verts.z[3] = -0.5;  //vec3<OverlapReal>(-0.5,0.5,-0.5);
    data.verts.x[4] = -0.5; data.verts.y[4] = -0.5; data.verts.z[4] = 0.5;  //vec3<OverlapReal>(-0.5,-0.5,0.5);
    data.verts.x[5] = 0.5; data.verts.y[5] = -0.5; data.verts.z[5] = 0.5;  //vec3<OverlapReal>(0.5,-0.5,0.5);
    data.verts.x[6] = 0.5; data.verts.y[6] = 0.5; data.verts.z[6] = 0.5;  //vec3<OverlapReal>(0.5,0.5,0.5);
    data.verts.x[7] = -0.5; data.verts.y[7] = 0.5; data.verts.z[7] = 0.5;  //vec3<OverlapReal>(-0.5,0.5,0.5);

    BOOST_REQUIRE(MAX_POLY3D_EDGES >= 12);
    data.n_edges = 12;
    data.edges[2*0] = 0; data.edges[2*0+1] = 1;
    data.edges[2*1] = 1; data.edges[2*1+1] = 2;
    data.edges[2*2] = 2; data.edges[2*2+1] = 3;
    data.edges[2*3] = 3; data.edges[2*3+1] = 0;
    data.edges[2*4] = 0; data.edges[2*4+1] = 4;
    data.edges[2*5] = 1; data.edges[2*5+1] = 5;
    data.edges[2*6] = 2; data.edges[2*6+1] = 6;
    data.edges[2*7] = 3; data.edges[2*7+1] = 7;
    data.edges[2*8] = 4; data.edges[2*8+1] = 5;
    data.edges[2*9] = 5; data.edges[2*9+1] = 6;
    data.edges[2*10] = 6; data.edges[2*10+1] = 7;
    data.edges[2*11] = 7; data.edges[2*11+1] = 4;

    BOOST_REQUIRE(MAX_POLY3D_FACES >= 6);
    data.n_faces = 6;
    BOOST_REQUIRE(MAX_POLY3D_FACE_VERTS >= 4);
    data.face_offs[0] = 0;
    data.face_verts[0] = 0; data.face_verts[1] = 1; data.face_verts[2] = 2; data.face_verts[3] = 3;
    data.face_offs[1] = 4;
    data.face_verts[4] = 0; data.face_verts[5] = 3; data.face_verts[6] = 7; data.face_verts[7] = 4;
    data.face_offs[2] = 8;
    data.face_verts[8] = 0; data.face_verts[9] = 4; data.face_verts[10] = 5; data.face_verts[11] = 1;
    data.face_offs[3] = 12;
    data.face_verts[12] = 4; data.face_verts[13] = 5; data.face_verts[14] = 6; data.face_verts[15] = 7;
    data.face_offs[4] = 16;
    data.face_verts[16] = 6; data.face_verts[17] = 7; data.face_verts[18] = 3; data.face_verts[19] = 2;
    data.face_offs[5] = 20;
    data.face_verts[20] = 1; data.face_verts[21] = 2; data.face_verts[22] = 6; data.face_verts[23] = 5;
    data.face_offs[6] = 24;

    data.ignore=0;

    set_radius(data);

    ShapePolyhedron::param_type p;
    p.data = data;
    p.tree = build_tree(data);
    ShapePolyhedron a(o_a, p);

    // first test, separate cubes by a large distance
    ShapePolyhedron b(o_b, p);
    r_ij = vec3<Scalar>(10,0,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    // next test, set them close, but not overlapping - from four sides
    r_ij =  vec3<Scalar>(1.4,0,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-1.4,0,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,1.4,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,-1.4,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    // now test them close, but slightly offset and not overlapping - from four sides
    r_ij =  vec3<Scalar>(1.4,0.2,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-1.4,0.2,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,1.4,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,-1.4,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    // Test point-face overlaps
    r_ij =  vec3<Scalar>(0,1.2,0);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,1.2,0.1);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0.1,1.2,0.1);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(1.2,0,0);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(1.2,0.1,0);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(1.2,0.1,0.1);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    // Test edge-edge overlaps
    r_ij =  vec3<Scalar>(-0.9,0.9,0.0);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.9,0.899,0.001);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0.9,-0.9,0.0);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count)); // this and only this test failed :-(
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.9,0.899,0.001);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.9,0.9,0.1);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    }

BOOST_AUTO_TEST_CASE( cubes_contained )
    {
    // one cube contained in the other
    vec3<Scalar> r_ij;
    quat<Scalar> o;
    BoxDim box(100);

    // build a cube
    poly3d_data data_a;
    BOOST_REQUIRE(MAX_POLY3D_VERTS >= 8);
    data_a.verts.N = 8;
    data_a.verts.x[0] = -0.5; data_a.verts.y[0] = -0.5; data_a.verts.z[0] = -0.5;  // vec3<OverlapReal>(-0.5,-0.5,-0.5);
    data_a.verts.x[1] = 0.5; data_a.verts.y[1] = -0.5; data_a.verts.z[1] = -0.5;  //vec3<OverlapReal>(0.5,-0.5,-0.5);
    data_a.verts.x[2] = 0.5; data_a.verts.y[2] = 0.5; data_a.verts.z[2] = -0.5;  //vec3<OverlapReal>(0.5,0.5,-0.5);
    data_a.verts.x[3] = -0.5; data_a.verts.y[3] = 0.5; data_a.verts.z[3] = -0.5;  //vec3<OverlapReal>(-0.5,0.5,-0.5);
    data_a.verts.x[4] = -0.5; data_a.verts.y[4] = -0.5; data_a.verts.z[4] = 0.5;  //vec3<OverlapReal>(-0.5,-0.5,0.5);
    data_a.verts.x[5] = 0.5; data_a.verts.y[5] = -0.5; data_a.verts.z[5] = 0.5;  //vec3<OverlapReal>(0.5,-0.5,0.5);
    data_a.verts.x[6] = 0.5; data_a.verts.y[6] = 0.5; data_a.verts.z[6] = 0.5;  //vec3<OverlapReal>(0.5,0.5,0.5);
    data_a.verts.x[7] = -0.5; data_a.verts.y[7] = 0.5; data_a.verts.z[7] = 0.5;  //vec3<OverlapReal>(-0.5,0.5,0.5);

    BOOST_REQUIRE(MAX_POLY3D_EDGES >= 12);
    data_a.n_edges = 12;
    data_a.edges[2*0] = 0; data_a.edges[2*0+1] = 1;
    data_a.edges[2*1] = 1; data_a.edges[2*1+1] = 2;
    data_a.edges[2*2] = 2; data_a.edges[2*2+1] = 3;
    data_a.edges[2*3] = 3; data_a.edges[2*3+1] = 0;
    data_a.edges[2*4] = 0; data_a.edges[2*4+1] = 4;
    data_a.edges[2*5] = 1; data_a.edges[2*5+1] = 5;
    data_a.edges[2*6] = 2; data_a.edges[2*6+1] = 6;
    data_a.edges[2*7] = 3; data_a.edges[2*7+1] = 7;
    data_a.edges[2*8] = 4; data_a.edges[2*8+1] = 5;
    data_a.edges[2*9] = 5; data_a.edges[2*9+1] = 6;
    data_a.edges[2*10] = 6; data_a.edges[2*10+1] = 7;
    data_a.edges[2*11] = 7; data_a.edges[2*11+1] = 4;

    BOOST_REQUIRE(MAX_POLY3D_FACES >= 6);
    data_a.n_faces = 6;

    BOOST_REQUIRE(MAX_POLY3D_FACE_VERTS >= 4);
    data_a.face_offs[0] = 0;
    data_a.face_verts[0] = 0; data_a.face_verts[1] = 1; data_a.face_verts[2] = 2; data_a.face_verts[3] = 3;
    data_a.face_offs[1] = 4;
    data_a.face_verts[4] = 0; data_a.face_verts[5] = 3; data_a.face_verts[6] = 7; data_a.face_verts[7] = 4;
    data_a.face_offs[2] = 8;
    data_a.face_verts[8] = 0; data_a.face_verts[9] = 4; data_a.face_verts[10] = 5; data_a.face_verts[11] = 1;
    data_a.face_offs[3] = 12;
    data_a.face_verts[12] = 4; data_a.face_verts[13] = 5; data_a.face_verts[14] = 6; data_a.face_verts[15] = 7;
    data_a.face_offs[4] = 16;
    data_a.face_verts[16] = 6; data_a.face_verts[17] = 7; data_a.face_verts[18] = 3; data_a.face_verts[19] = 2;
    data_a.face_offs[5] = 20;
    data_a.face_verts[20] = 1; data_a.face_verts[21] = 2; data_a.face_verts[22] = 6; data_a.face_verts[23] = 5;
    data_a.face_offs[6] = 24;

    data_a.ignore=0;

    // the second cube is 0.1 of the size of the first one
    poly3d_data data_b = data_a;
    OverlapReal scale = 0.1;
    for (unsigned int j = 0; j < 8; ++j)
        {
        data_b.verts.x[j] *= scale;
        data_b.verts.y[j] *= scale;
        data_b.verts.z[j] *= scale;
        }


    set_radius(data_a);
    set_radius(data_b);

    ShapePolyhedron::param_type p_a;
    p_a.data = data_a;
    p_a.tree = build_tree(data_a);
    ShapePolyhedron a(o, p_a);

    ShapePolyhedron::param_type p_b;
    p_b.data = data_b;
    p_b.tree = build_tree(data_b);
    ShapePolyhedron b(o, p_b);

    // first test, separate cubes by a large distance
    r_ij = vec3<Scalar>(10,0,0);
    BOOST_CHECK(!test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(!test_overlap(-r_ij,b,a,err_count));

    // place the small cube in the center of the large one
    r_ij = vec3<Scalar>(0,0,0);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    // place the small cube in the large one, but slightly offset
    r_ij = vec3<Scalar>(0.1,0,0);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(-0.1,0,0);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0.0,0.1,0);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0.0,-0.1,0);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0.0,0.0,0.1);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0.0,0.0,-0.1);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    // rotate cube a around two axes
    Scalar alpha = M_PI/4.0;
    const quat<Scalar> q1(cos(alpha/2.0), (Scalar)sin(alpha/2.0) * vec3<Scalar>(1,0,0));
    const quat<Scalar> q2(cos(alpha/2.0), (Scalar)sin(alpha/2.0) * vec3<Scalar>(0,0,1));
    quat<Scalar> o_new(q2 * q1);

    a.orientation = o_new;

    r_ij = vec3<Scalar>(0.1,0,0);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(-0.1,0,0);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0.0,0.1,0);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0.0,-0.1,0);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0.0,0.0,0.1);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0.0,0.0,-0.1);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    // now also rotate cube b
    b.orientation = o_new;

    r_ij = vec3<Scalar>(0.1,0,0);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(-0.1,0,0);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0.0,0.1,0);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0.0,-0.1,0);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0.0,0.0,0.1);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0.0,0.0,-0.1);
    BOOST_CHECK(test_overlap(r_ij,a,b,err_count));
    BOOST_CHECK(test_overlap(-r_ij,b,a,err_count));
    }
