#include "hoomd/hpmc/IntegratorHPMC.h"
#include "hoomd/hpmc/Moves.h"
#include "hoomd/hpmc/ShapePolyhedron.h"
#include "hoomd/AABBTree.h"

#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN();




#include <iostream>
#include <string>

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

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
    data.verts.diameter = 2*(sqrt(radius_sq)+data.verts.sweep_radius);
    }

ShapePolyhedron::gpu_tree_type build_tree(poly3d_data &data)
    {
    ShapePolyhedron::gpu_tree_type::obb_tree_type tree;
    hpmc::detail::OBB *obbs;
    int retval = posix_memalign((void**)&obbs, 32, sizeof(hpmc::detail::OBB)*data.n_faces);
    if (retval != 0)
        {
        throw std::runtime_error("Error allocating aligned OBB memory.");
        }

    std::vector<std::vector<vec3<OverlapReal> > > internal_coordinates;
    // construct bounding box tree
    for (unsigned int i = 0; i < data.n_faces; ++i)
        {
        std::vector< vec3<OverlapReal> > face_vec;

        for (unsigned int j = data.face_offs[i]; j < data.face_offs[i+1]; ++j)
            {
            vec3<OverlapReal> v(data.verts.x[data.face_verts[j]], data.verts.y[data.face_verts[j]], data.verts.z[data.face_verts[j]]);
            face_vec.push_back(v);
            }
        obbs[i] = hpmc::detail::compute_obb(face_vec, data.verts.sweep_radius);
        internal_coordinates.push_back(face_vec);
        }
    tree.buildTree(obbs, internal_coordinates, data.verts.sweep_radius, data.n_faces);
    ShapePolyhedron::gpu_tree_type gpu_tree(tree);
    free(obbs);
    return gpu_tree;
    }

UP_TEST( construction )
    {
    quat<Scalar> o(1.0, vec3<Scalar>(-3.0, 9.0, 6.0));

    poly3d_data data(4,1,4,false);
    data.verts.sweep_radius=0.0f;
    data.verts.x[0] = 0; data.verts.y[0] = 0; data.verts.z[0] = 0;
    data.verts.x[1] = 1; data.verts.y[1] = 0; data.verts.z[1] = 0;
    data.verts.x[2] = 0; data.verts.y[2] = 1.25; data.verts.z[2] = 0;
    data.verts.x[3] = 0; data.verts.y[3] = 0; data.verts.z[3] = 1.1;
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

    MY_CHECK_CLOSE(a.orientation.s, o.s, tol);
    MY_CHECK_CLOSE(a.orientation.v.x, o.v.x, tol);
    MY_CHECK_CLOSE(a.orientation.v.y, o.v.y, tol);
    MY_CHECK_CLOSE(a.orientation.v.z, o.v.z, tol);

    UP_ASSERT_EQUAL(a.data.verts.N, data.verts.N);
    for (unsigned int i = 0; i < data.verts.N; i++)
        {
        MY_CHECK_CLOSE(a.data.verts.x[i], data.verts.x[i], tol);
        MY_CHECK_CLOSE(a.data.verts.y[i], data.verts.y[i], tol);
        MY_CHECK_CLOSE(a.data.verts.z[i], data.verts.z[i], tol);
        }

    UP_ASSERT_EQUAL(a.data.n_faces, data.n_faces);
    for (unsigned int i = 0; i < data.n_faces; i++)
        {
        UP_ASSERT_EQUAL(a.data.face_offs[i], data.face_offs[i]);
        unsigned int offs = a.data.face_offs[i];
        for (unsigned int j = offs; j < a.data.face_offs[i+1]; ++j)
            UP_ASSERT_EQUAL(a.data.face_verts[j], data.face_verts[j]);
        }

    UP_ASSERT_EQUAL(a.data.face_offs[data.n_faces], data.face_offs[data.n_faces]);
    UP_ASSERT(a.hasOrientation());

    MY_CHECK_CLOSE(a.getCircumsphereDiameter(), 2.5, tol);
    }

UP_TEST( overlap_octahedron_no_rot )
    {
    // first set of simple overlap checks is two octahedra at unit orientation
    vec3<Scalar> r_ij;
    quat<Scalar> o;
    BoxDim box(100);

    // build an octahedron
    poly3d_data data(6,8,24,false);
    data.verts.sweep_radius = 0.0f;

    data.verts.x[0] = -0.5; data.verts.y[0] = -0.5; data.verts.z[0] = 0; // vec3<OverlapReal>(-0.5,-0.5,0);
    data.verts.x[1] = 0.5; data.verts.y[1] = -0.5; data.verts.z[1] = 0; //vec3<OverlapReal>(0.5,-0.5,0);
    data.verts.x[2] = 0.5; data.verts.y[2] = 0.5; data.verts.z[2] = 0; //vec3<OverlapReal>(0.5,0.5,0);
    data.verts.x[3] = -0.5; data.verts.y[3] = 0.5; data.verts.z[3] = 0; // vec3<OverlapReal>(-0.5,0.5,0);
    data.verts.x[4] = 0; data.verts.y[4] = 0; data.verts.z[4] = 0.707106781186548; // vec3<OverlapReal>(0,0,0.707106781186548);
    data.verts.x[5] = 0; data.verts.y[5] = 0; data.verts.z[5] = -0.707106781186548; // vec3<OverlapReal>(0,0,-0.707106781186548);
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
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    // first test, separate octahedrons by a large distance
    r_ij =  vec3<Scalar>(10,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // next test, set them close, but not overlapping - from all four sides of base
    r_ij =  vec3<Scalar>(1.1,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-1.1,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,1.1,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,-1.1,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // now test them close, but slightly offset and not overlapping - from all four sides
    r_ij =  vec3<Scalar>(1.1,0.2,0.1);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-1.1,0.2,0.1);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,1.1,0.1);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,-1.1,0.1);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // and finally, make them overlap slightly in each direction
    r_ij =  vec3<Scalar>(0.9,0.2,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.9,0.2,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,0.9,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,-0.9,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    // torture test, overlap along most of a line
    r_ij =  vec3<Scalar>(1.0,0.2,0);

    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));
    }

UP_TEST( overlap_cube_no_rot )
    {
    // first set of simple overlap checks is two cubes at unit orientation
    vec3<Scalar> r_ij;
    quat<Scalar> o;
    BoxDim box(100);

    // build a cube
    poly3d_data data(8,6,24,false);
    data.verts.sweep_radius = 0.0f;
    data.verts.x[0] = -0.5; data.verts.y[0] = -0.5; data.verts.z[0] = -0.5;  // vec3<OverlapReal>(-0.5,-0.5,-0.5);
    data.verts.x[1] = 0.5; data.verts.y[1] = -0.5; data.verts.z[1] = -0.5;  //vec3<OverlapReal>(0.5,-0.5,-0.5);
    data.verts.x[2] = 0.5; data.verts.y[2] = 0.5; data.verts.z[2] = -0.5;  //vec3<OverlapReal>(0.5,0.5,-0.5);
    data.verts.x[3] = -0.5; data.verts.y[3] = 0.5; data.verts.z[3] = -0.5;  //vec3<OverlapReal>(-0.5,0.5,-0.5);
    data.verts.x[4] = -0.5; data.verts.y[4] = -0.5; data.verts.z[4] = 0.5;  //vec3<OverlapReal>(-0.5,-0.5,0.5);
    data.verts.x[5] = 0.5; data.verts.y[5] = -0.5; data.verts.z[5] = 0.5;  //vec3<OverlapReal>(0.5,-0.5,0.5);
    data.verts.x[6] = 0.5; data.verts.y[6] = 0.5; data.verts.z[6] = 0.5;  //vec3<OverlapReal>(0.5,0.5,0.5);
    data.verts.x[7] = -0.5; data.verts.y[7] = 0.5; data.verts.z[7] = 0.5;  //vec3<OverlapReal>(-0.5,0.5,0.5);

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
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // next test, set them close, but not overlapping - from all four sides
    r_ij =  vec3<Scalar>(1.1,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-1.1,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,1.1,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,-1.1,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // now test them close, but slightly offset and not overlapping - from all four sides
    r_ij =  vec3<Scalar>(1.1,0.2,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-1.1,0.2,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,1.1,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,-1.1,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // make them overlap slightly in each direction
    r_ij =  vec3<Scalar>(0.9,0.2,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.9,0.2,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,0.9,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,-0.9,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    // Make them overlap a lot
    r_ij =  vec3<Scalar>(0.2,0,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,0.2,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0.2,tol,tol);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0.1,0.2,0.1);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    // torture test, overlap along most of a line
    r_ij =  vec3<Scalar>(1.0,0.2,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));
    }


UP_TEST( overlap_cube_rot1 )
    {
    // second set of simple overlap checks is two cubes, with one rotated by 45 degrees
    vec3<Scalar> r_ij;
    quat<Scalar> o_a;
    Scalar alpha = M_PI/4.0;
    quat<Scalar> o_b(cos(alpha/2.0), (Scalar)sin(alpha/2.0) * vec3<Scalar>(0,0,1)); // rotation quaternion

    BoxDim box(100);

    // build a cube
    poly3d_data data(8,6,24,false);
    data.verts.sweep_radius = 0.0f;
    data.verts.x[0] = -0.5; data.verts.y[0] = -0.5; data.verts.z[0] = -0.5;  // vec3<OverlapReal>(-0.5,-0.5,-0.5);
    data.verts.x[1] = 0.5; data.verts.y[1] = -0.5; data.verts.z[1] = -0.5;  //vec3<OverlapReal>(0.5,-0.5,-0.5);
    data.verts.x[2] = 0.5; data.verts.y[2] = 0.5; data.verts.z[2] = -0.5;  //vec3<OverlapReal>(0.5,0.5,-0.5);
    data.verts.x[3] = -0.5; data.verts.y[3] = 0.5; data.verts.z[3] = -0.5;  //vec3<OverlapReal>(-0.5,0.5,-0.5);
    data.verts.x[4] = -0.5; data.verts.y[4] = -0.5; data.verts.z[4] = 0.5;  //vec3<OverlapReal>(-0.5,-0.5,0.5);
    data.verts.x[5] = 0.5; data.verts.y[5] = -0.5; data.verts.z[5] = 0.5;  //vec3<OverlapReal>(0.5,-0.5,0.5);
    data.verts.x[6] = 0.5; data.verts.y[6] = 0.5; data.verts.z[6] = 0.5;  //vec3<OverlapReal>(0.5,0.5,0.5);
    data.verts.x[7] = -0.5; data.verts.y[7] = 0.5; data.verts.z[7] = 0.5;  //vec3<OverlapReal>(-0.5,0.5,0.5);

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
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // next test, set them close, but not overlapping - from all four sides
    r_ij =  vec3<Scalar>(1.3,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-1.3,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,1.3,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,-1.3,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // now test them close, but slightly offset and not overlapping - from all four sides
    r_ij =  vec3<Scalar>(1.3,0.2,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-1.3,0.2,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,1.3,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,-1.3,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // and finally, make them overlap slightly in each direction
    r_ij =  vec3<Scalar>(1.2,0.2,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-1.2,0.2,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,1.2,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,-1.2,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));
    }

UP_TEST( overlap_cube_rot2 )
    {
    // third set of simple overlap checks is two cubes, with the other one rotated by 45 degrees
    vec3<Scalar> r_ij;
    quat<Scalar> o_a;
    Scalar alpha = M_PI/4.0;
    quat<Scalar> o_b(cos(alpha/2.0), (Scalar)sin(alpha/2.0) * vec3<Scalar>(0,0,1)); // rotation quaternion

    BoxDim box(100);

    // build a cube
    poly3d_data data(8,6,24,false);
    data.verts.sweep_radius = 0.0f;
    data.verts.x[0] = -0.5; data.verts.y[0] = -0.5; data.verts.z[0] = -0.5;  // vec3<OverlapReal>(-0.5,-0.5,-0.5);
    data.verts.x[1] = 0.5; data.verts.y[1] = -0.5; data.verts.z[1] = -0.5;  //vec3<OverlapReal>(0.5,-0.5,-0.5);
    data.verts.x[2] = 0.5; data.verts.y[2] = 0.5; data.verts.z[2] = -0.5;  //vec3<OverlapReal>(0.5,0.5,-0.5);
    data.verts.x[3] = -0.5; data.verts.y[3] = 0.5; data.verts.z[3] = -0.5;  //vec3<OverlapReal>(-0.5,0.5,-0.5);
    data.verts.x[4] = -0.5; data.verts.y[4] = -0.5; data.verts.z[4] = 0.5;  //vec3<OverlapReal>(-0.5,-0.5,0.5);
    data.verts.x[5] = 0.5; data.verts.y[5] = -0.5; data.verts.z[5] = 0.5;  //vec3<OverlapReal>(0.5,-0.5,0.5);
    data.verts.x[6] = 0.5; data.verts.y[6] = 0.5; data.verts.z[6] = 0.5;  //vec3<OverlapReal>(0.5,0.5,0.5);
    data.verts.x[7] = -0.5; data.verts.y[7] = 0.5; data.verts.z[7] = 0.5;  //vec3<OverlapReal>(-0.5,0.5,0.5);

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
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // next test, set them close, but not overlapping - from all four sides
    r_ij =  vec3<Scalar>(1.3,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-1.3,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,1.3,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,-1.3,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // now test them close, but slightly offset and not overlapping - from all four sides
    r_ij =  vec3<Scalar>(1.3,0.2,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-1.3,0.2,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,1.3,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,-1.3,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // and finally, make them overlap slightly in each direction
    r_ij =  vec3<Scalar>(1.2,0.2,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-1.2,0.2,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,1.2,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,-1.2,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));
    }

UP_TEST( overlap_cube_rot3 )
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
    poly3d_data data(8,6,24,false);
    data.verts.sweep_radius = 0.0f;
    data.verts.x[0] = -0.5; data.verts.y[0] = -0.5; data.verts.z[0] = -0.5;  // vec3<OverlapReal>(-0.5,-0.5,-0.5);
    data.verts.x[1] = 0.5; data.verts.y[1] = -0.5; data.verts.z[1] = -0.5;  //vec3<OverlapReal>(0.5,-0.5,-0.5);
    data.verts.x[2] = 0.5; data.verts.y[2] = 0.5; data.verts.z[2] = -0.5;  //vec3<OverlapReal>(0.5,0.5,-0.5);
    data.verts.x[3] = -0.5; data.verts.y[3] = 0.5; data.verts.z[3] = -0.5;  //vec3<OverlapReal>(-0.5,0.5,-0.5);
    data.verts.x[4] = -0.5; data.verts.y[4] = -0.5; data.verts.z[4] = 0.5;  //vec3<OverlapReal>(-0.5,-0.5,0.5);
    data.verts.x[5] = 0.5; data.verts.y[5] = -0.5; data.verts.z[5] = 0.5;  //vec3<OverlapReal>(0.5,-0.5,0.5);
    data.verts.x[6] = 0.5; data.verts.y[6] = 0.5; data.verts.z[6] = 0.5;  //vec3<OverlapReal>(0.5,0.5,0.5);
    data.verts.x[7] = -0.5; data.verts.y[7] = 0.5; data.verts.z[7] = 0.5;  //vec3<OverlapReal>(-0.5,0.5,0.5);

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
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // next test, set them close, but not overlapping - from four sides
    r_ij =  vec3<Scalar>(1.4,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-1.4,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,1.4,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,-1.4,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // now test them close, but slightly offset and not overlapping - from four sides
    r_ij =  vec3<Scalar>(1.4,0.2,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-1.4,0.2,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,1.4,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,-1.4,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // Test point-face overlaps
    r_ij =  vec3<Scalar>(0,1.2,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,1.2,0.1);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0.1,1.2,0.1);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(1.2,0,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(1.2,0.1,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(1.2,0.1,0.1);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    // Test edge-edge overlaps
    r_ij =  vec3<Scalar>(-0.9,0.9,0.0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.9,0.899,0.001);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0.9,-0.9,0.0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count)); // this and only this test failed :-(
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.9,0.899,0.001);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.9,0.9,0.1);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    }

UP_TEST( cubes_contained )
    {
    // one cube contained in the other
    vec3<Scalar> r_ij;
    quat<Scalar> o;
    BoxDim box(100);

    // build a cube
    poly3d_data data_a(8,6,24,false);
    data_a.verts.sweep_radius = 0.0f;
    data_a.verts.x[0] = -0.5; data_a.verts.y[0] = -0.5; data_a.verts.z[0] = -0.5;  // vec3<OverlapReal>(-0.5,-0.5,-0.5);
    data_a.verts.x[1] = 0.5; data_a.verts.y[1] = -0.5; data_a.verts.z[1] = -0.5;  //vec3<OverlapReal>(0.5,-0.5,-0.5);
    data_a.verts.x[2] = 0.5; data_a.verts.y[2] = 0.5; data_a.verts.z[2] = -0.5;  //vec3<OverlapReal>(0.5,0.5,-0.5);
    data_a.verts.x[3] = -0.5; data_a.verts.y[3] = 0.5; data_a.verts.z[3] = -0.5;  //vec3<OverlapReal>(-0.5,0.5,-0.5);
    data_a.verts.x[4] = -0.5; data_a.verts.y[4] = -0.5; data_a.verts.z[4] = 0.5;  //vec3<OverlapReal>(-0.5,-0.5,0.5);
    data_a.verts.x[5] = 0.5; data_a.verts.y[5] = -0.5; data_a.verts.z[5] = 0.5;  //vec3<OverlapReal>(0.5,-0.5,0.5);
    data_a.verts.x[6] = 0.5; data_a.verts.y[6] = 0.5; data_a.verts.z[6] = 0.5;  //vec3<OverlapReal>(0.5,0.5,0.5);
    data_a.verts.x[7] = -0.5; data_a.verts.y[7] = 0.5; data_a.verts.z[7] = 0.5;  //vec3<OverlapReal>(-0.5,0.5,0.5);

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
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // place the small cube in the center of the large one
    r_ij = vec3<Scalar>(0,0,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    // place the small cube in the large one, but slightly offset
    r_ij = vec3<Scalar>(0.1,0,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(-0.1,0,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0.0,0.1,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0.0,-0.1,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0.0,0.0,0.1);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0.0,0.0,-0.1);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    // rotate cube a around two axes
    Scalar alpha = M_PI/4.0;
    const quat<Scalar> q1(cos(alpha/2.0), (Scalar)sin(alpha/2.0) * vec3<Scalar>(1,0,0));
    const quat<Scalar> q2(cos(alpha/2.0), (Scalar)sin(alpha/2.0) * vec3<Scalar>(0,0,1));
    quat<Scalar> o_new(q2 * q1);

    a.orientation = o_new;

    r_ij = vec3<Scalar>(0.1,0,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(-0.1,0,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0.0,0.1,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0.0,-0.1,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0.0,0.0,0.1);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0.0,0.0,-0.1);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    // now also rotate cube b
    b.orientation = o_new;

    r_ij = vec3<Scalar>(0.1,0,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(-0.1,0,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0.0,0.1,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0.0,-0.1,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0.0,0.0,0.1);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij = vec3<Scalar>(0.0,0.0,-0.1);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));
    }

UP_TEST( overlap_sphero_octahedron_no_rot )
    {
    // first set of simple overlap checks is two octahedra at unit orientation
    vec3<Scalar> r_ij;
    quat<Scalar> o;
    BoxDim box(100);

    // build an octahedron
    poly3d_data data(6,8,24,false);
    data.verts.sweep_radius = 0.1f;

    data.verts.x[0] = -0.5; data.verts.y[0] = -0.5; data.verts.z[0] = 0; // vec3<OverlapReal>(-0.5,-0.5,0);
    data.verts.x[1] = 0.5; data.verts.y[1] = -0.5; data.verts.z[1] = 0; //vec3<OverlapReal>(0.5,-0.5,0);
    data.verts.x[2] = 0.5; data.verts.y[2] = 0.5; data.verts.z[2] = 0; //vec3<OverlapReal>(0.5,0.5,0);
    data.verts.x[3] = -0.5; data.verts.y[3] = 0.5; data.verts.z[3] = 0; // vec3<OverlapReal>(-0.5,0.5,0);
    data.verts.x[4] = 0; data.verts.y[4] = 0; data.verts.z[4] = 0.707106781186548; // vec3<OverlapReal>(0,0,0.707106781186548);
    data.verts.x[5] = 0; data.verts.y[5] = 0; data.verts.z[5] = -0.707106781186548; // vec3<OverlapReal>(0,0,-0.707106781186548);
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
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    // first test, separate octahedrons by a large distance
    r_ij =  vec3<Scalar>(10,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // next test, set them close, but not overlapping - from all four sides of base
    r_ij =  vec3<Scalar>(1.3,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-1.3,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,1.3,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0,-1.3,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // now test them close, but slightly offset and not overlapping - from all four sides
    r_ij =  vec3<Scalar>(1.3,0.2,0.1);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-1.3,0.2,0.1);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,1.3,0.1);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-0.2,-1.3,0.1);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // and finally, make them overlap slightly in each direction
    r_ij =  vec3<Scalar>(1.1,0.2,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-1.1,0.2,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-1.1,0.9,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(-1.1,-0.9,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    // torture test, overlap along most of a line
    r_ij =  vec3<Scalar>(1.1999,0.2,0);

    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(1.20001,0.2,0);

    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));
    }


UP_TEST( overlap_octahedron_sphere )
    {
    // a cube and a sphere
    vec3<Scalar> r_ij;
    quat<Scalar> o;
    BoxDim box(100);

    // build an octahedron
    poly3d_data data_a(6,8,24,false);
    data_a.verts.sweep_radius = 0.0f;

    data_a.verts.x[0] = -0.5; data_a.verts.y[0] = -0.5; data_a.verts.z[0] = 0; // vec3<OverlapReal>(-0.5,-0.5,0);
    data_a.verts.x[1] = 0.5; data_a.verts.y[1] = -0.5; data_a.verts.z[1] = 0; //vec3<OverlapReal>(0.5,-0.5,0);
    data_a.verts.x[2] = 0.5; data_a.verts.y[2] = 0.5; data_a.verts.z[2] = 0; //vec3<OverlapReal>(0.5,0.5,0);
    data_a.verts.x[3] = -0.5; data_a.verts.y[3] = 0.5; data_a.verts.z[3] = 0; // vec3<OverlapReal>(-0.5,0.5,0);
    data_a.verts.x[4] = 0; data_a.verts.y[4] = 0; data_a.verts.z[4] = 0.707106781186548; // vec3<OverlapReal>(0,0,0.707106781186548);
    data_a.verts.x[5] = 0; data_a.verts.y[5] = 0; data_a.verts.z[5] = -0.707106781186548; // vec3<OverlapReal>(0,0,-0.707106781186548);
    data_a.face_offs[0] = 0;
    data_a.face_verts[0] = 0; data_a.face_verts[1] = 4; data_a.face_verts[2] = 1;
    data_a.face_offs[1] = 3;
    data_a.face_verts[3] = 1; data_a.face_verts[4] = 4; data_a.face_verts[5] = 2;
    data_a.face_offs[2] = 6;
    data_a.face_verts[6] = 2; data_a.face_verts[7] = 4; data_a.face_verts[8] = 3;
    data_a.face_offs[3] = 9;
    data_a.face_verts[9] = 3; data_a.face_verts[10] = 4; data_a.face_verts[11] = 0;
    data_a.face_offs[4] = 12;
    data_a.face_verts[12] = 0; data_a.face_verts[13] = 5; data_a.face_verts[14] = 1;
    data_a.face_offs[5] = 15;
    data_a.face_verts[15] = 1; data_a.face_verts[16] = 5; data_a.face_verts[17] = 2;
    data_a.face_offs[6] = 18;
    data_a.face_verts[18] = 2; data_a.face_verts[19] = 5; data_a.face_verts[20] = 3;
    data_a.face_offs[7] = 21;
    data_a.face_verts[21] = 3; data_a.face_verts[22] = 5; data_a.face_verts[23] = 0;
    data_a.face_offs[8] = 24;
    data_a.ignore = 0;
    set_radius(data_a);

    poly3d_data data_b(1,1,1,false);
    data_b.verts.sweep_radius = 0.5;
    data_b.verts.x[0] = data_b.verts.y[0] = data_b.verts.z[0] = 0;
    data_b.face_offs[0] = 0;
    data_b.face_verts[0] = 0;
    data_b.face_offs[1] = 1;
    data_b.ignore = 0;
    set_radius(data_b);

    ShapePolyhedron::param_type cube_p;
    cube_p.data = data_a;
    cube_p.tree = build_tree(data_a);
    ShapePolyhedron a(o, cube_p);

    ShapePolyhedron::param_type sphere_p;
    sphere_p.data = data_b;
    sphere_p.tree = build_tree(data_b);
    ShapePolyhedron b(o, sphere_p);

    // not overlapping
    r_ij =  vec3<Scalar>(2.0,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // not touching
    r_ij =  vec3<Scalar>(1.0001,0,0);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    // touching
    r_ij =  vec3<Scalar>(0.9999,0,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    // partially overlapping
    r_ij =  vec3<Scalar>(0.5,0,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    // contained
    r_ij =  vec3<Scalar>(0,0,0);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));

    // sphere below octahedron
    r_ij =  vec3<Scalar>(0.0,0.0,-1.0/sqrt(2)-0.5-0.0001);
    UP_ASSERT(!test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(!test_overlap(-r_ij,b,a,err_count));

    r_ij =  vec3<Scalar>(0.0,0.0,-1.0/sqrt(2)-0.5+0.0001);
    UP_ASSERT(test_overlap(r_ij,a,b,err_count));
    UP_ASSERT(test_overlap(-r_ij,b,a,err_count));
    }
