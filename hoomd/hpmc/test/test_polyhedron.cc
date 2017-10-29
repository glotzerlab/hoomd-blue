#include "hoomd/hpmc/IntegratorHPMC.h"
#include "hoomd/hpmc/Moves.h"
#include "hoomd/hpmc/ShapePolyhedron.h"
#include "hoomd/AABBTree.h"
#include "hoomd/extern/quickhull/QuickHull.hpp"

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
    for (unsigned int i = 0; i < data.n_verts; i++)
        {
        radius_sq = std::max(radius_sq, dot(data.verts[i],data.verts[i]));
        }

    // set the diameter
    data.convex_hull_verts.diameter = 2*(sqrt(radius_sq)+data.sweep_radius);
    }

GPUTree build_tree(poly3d_data &data)
    {
    OBBTree tree;
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

        unsigned int n_vert = 0;
        for (unsigned int j = data.face_offs[i]; j < data.face_offs[i+1]; ++j)
            {
            vec3<OverlapReal> v = data.verts[data.face_verts[j]];
            face_vec.push_back(v);
            n_vert++;
            }

        std::vector<OverlapReal> vertex_radii(n_vert,data.sweep_radius);
        obbs[i] = hpmc::detail::compute_obb(face_vec, vertex_radii, false);
        internal_coordinates.push_back(face_vec);
        }
    unsigned int capacity = 4;
    tree.buildTree(obbs, internal_coordinates, data.sweep_radius, data.n_faces, capacity);
    GPUTree gpu_tree(tree);
    free(obbs);
    return gpu_tree;
    }

void initialize_convex_hull(poly3d_data &data)
    {
    // for simplicity, use all vertices instead of convex hull
    for (unsigned int i = 0; i < data.n_verts; ++i)
        {
        data.convex_hull_verts.x[i] = data.verts[i].x;
        data.convex_hull_verts.y[i] = data.verts[i].y;
        data.convex_hull_verts.z[i] = data.verts[i].z;
        }
    }

UP_TEST( construction )
    {
    quat<Scalar> o(1.0, vec3<Scalar>(-3.0, 9.0, 6.0));

    poly3d_data data(4,1,4,4,false);
    data.sweep_radius=data.convex_hull_verts.sweep_radius=0.0f;
    data.verts[0] = vec3<OverlapReal>(0,0,0);
    data.verts[1] = vec3<OverlapReal>(1,0,0);
    data.verts[2] = vec3<OverlapReal>(0,1.25,0);
    data.verts[3] = vec3<OverlapReal>(0,0,1.1);
    data.face_verts[0] = 0;
    data.face_verts[1] = 1;
    data.face_verts[2] = 2;
    data.face_verts[3] = 3;
    data.face_offs[0] = 0;
    data.face_offs[1] = 4;
    data.ignore = 0;
    set_radius(data);
    initialize_convex_hull(data);

    ShapePolyhedron::param_type p = data;
    p.tree = build_tree(data);
    ShapePolyhedron a(o, p);

    MY_CHECK_CLOSE(a.orientation.s, o.s, tol);
    MY_CHECK_CLOSE(a.orientation.v.x, o.v.x, tol);
    MY_CHECK_CLOSE(a.orientation.v.y, o.v.y, tol);
    MY_CHECK_CLOSE(a.orientation.v.z, o.v.z, tol);

    UP_ASSERT_EQUAL(a.data.n_verts, data.n_verts);
    for (unsigned int i = 0; i < data.n_verts; i++)
        {
        MY_CHECK_CLOSE(a.data.verts[i].x, data.verts[i].x, tol);
        MY_CHECK_CLOSE(a.data.verts[i].y, data.verts[i].y, tol);
        MY_CHECK_CLOSE(a.data.verts[i].z, data.verts[i].z, tol);
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
    poly3d_data data(6,8,24,6,false);
    data.sweep_radius=data.convex_hull_verts.sweep_radius=0.0f;

    data.verts[0] = vec3<OverlapReal>(-0.5,-0.5,0);
    data.verts[1] = vec3<OverlapReal>(0.5,-0.5,0);
    data.verts[2] = vec3<OverlapReal>(0.5,0.5,0);
    data.verts[3] = vec3<OverlapReal>(-0.5,0.5,0);
    data.verts[4] = vec3<OverlapReal>(0,0,0.707106781186548);
    data.verts[5] = vec3<OverlapReal>(0,0,-0.707106781186548);
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
    initialize_convex_hull(data);

    ShapePolyhedron::param_type p = data;
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


UP_TEST( overlap_sphero_octahedron_no_rot )
    {
    // first set of simple overlap checks is two octahedra at unit orientation
    vec3<Scalar> r_ij;
    quat<Scalar> o;
    BoxDim box(100);

    // build an octahedron
    poly3d_data data(6,8,24,6,false);
    data.sweep_radius=data.convex_hull_verts.sweep_radius=0.1f;

    data.verts[0] = vec3<OverlapReal>(-0.5,-0.5,0);
    data.verts[1] = vec3<OverlapReal>(0.5,-0.5,0);
    data.verts[2] = vec3<OverlapReal>(0.5,0.5,0);
    data.verts[3] = vec3<OverlapReal>(-0.5,0.5,0);
    data.verts[4] = vec3<OverlapReal>(0,0,0.707106781186548);
    data.verts[5] = vec3<OverlapReal>(0,0,-0.707106781186548);
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
    initialize_convex_hull(data);

    ShapePolyhedron::param_type p = data;
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
    // an octahedron and a sphere
    vec3<Scalar> r_ij;
    quat<Scalar> o;
    BoxDim box(100);

    // build an octahedron
    poly3d_data data_a(6,8,24,6,false);

    memset((void *)&data_a.verts[0], 0, sizeof(vec3<OverlapReal>)*6);
    memset((void*)&data_a.face_offs[0], 0, sizeof(unsigned int)*9);
    memset((void*)&data_a.face_verts[0], 0, sizeof(unsigned int)*24);

    data_a.sweep_radius=data_a.convex_hull_verts.sweep_radius=0.0f;

    data_a.verts[0] = vec3<OverlapReal>(-0.5,-0.5,0);
    data_a.verts[1] = vec3<OverlapReal>(0.5,-0.5,0);
    data_a.verts[2] = vec3<OverlapReal>(0.5,0.5,0);
    data_a.verts[3] = vec3<OverlapReal>(-0.5,0.5,0);
    data_a.verts[4] = vec3<OverlapReal>(0,0,0.707106781186548);
    data_a.verts[5] = vec3<OverlapReal>(0,0,-0.707106781186548);
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
    initialize_convex_hull(data_a);

    poly3d_data data_b(1,1,1,1,false);

    memset((void *)&data_a.verts[0], 0, sizeof(vec3<OverlapReal>)*1);
    memset((void*)&data_a.face_offs[0], 0, sizeof(unsigned int)*1);
    memset((void*)&data_a.face_verts[0], 0, sizeof(unsigned int)*1);

    data_b.sweep_radius=data_b.convex_hull_verts.sweep_radius=0.5f;
    data_b.verts[0] = vec3<OverlapReal>(0,0,0);
    data_b.face_offs[0] = 0;
    data_b.face_verts[0] = 0;
    data_b.face_offs[1] = 1;
    data_b.ignore = 0;
    set_radius(data_b);

    ShapePolyhedron::param_type octahedron_p = data_a;
    octahedron_p.tree = build_tree(data_a);
    ShapePolyhedron a(o, octahedron_p);

    ShapePolyhedron::param_type sphere_p = data_b;
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
