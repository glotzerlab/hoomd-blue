
#include "hoomd/ExecutionConfiguration.h"

//! Name the unit test module
#define BOOST_TEST_MODULE ShapeSphereUnion
#include "boost_utf_configure.h"

#include "hoomd/hpmc/IntegratorHPMC.h"
#include "hoomd/hpmc/Moves.h"
#include "hoomd/hpmc/ShapeUnion.h"

#include <iostream>
#include <string>

#include <boost/bind.hpp>
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include <boost/function.hpp>
#include <memory>

using namespace hpmc;
using namespace std;
using namespace hpmc::detail;

unsigned int err_count;

template<class Shape>
void build_tree(union_params<Shape>& data)
    {
    union_gpu_tree_type::obb_tree_type tree;
    hpmc::detail::OBB *obbs;
    int retval = posix_memalign((void**)&obbs, 32, sizeof(hpmc::detail::OBB)*data.N);
    if (retval != 0)
        {
        throw std::runtime_error("Error allocating aligned AABB memory.");
        }

    // construct bounding box tree
    for (unsigned int i = 0; i < data.N; ++i)
        {
        Shape dummy(quat<Scalar>(data.morientation[i]), data.mparams[i]);
        obbs[i] = OBB(dummy.getAABB(data.mpos[i]));
        }

    tree.buildTree(obbs, data.N);
    free(obbs);
    data.tree = union_gpu_tree_type(tree);
    }

BOOST_AUTO_TEST_CASE( construction )
    {
    // parameters
    quat<Scalar> o(1.0, vec3<Scalar>(-3.0, 9.0, 6.0));
    o = o * (Scalar)(Scalar(1.0)/sqrt(norm2(o)));

    // lopsided dumbbell: spheres of radius 0.25 and 0.5, located at x= +/- 0.25
    Scalar x_i(-0.25);
    Scalar x_j(0.25);
    Scalar R_i(0.25);
    Scalar R_j(0.5);
    Scalar R(x_j + R_j);
    ShapeSphere::param_type par_i;
    par_i.radius = R_i;
    par_i.ignore = 0;
    ShapeSphere::param_type par_j;
    par_j.radius = R_j;
    par_i.ignore = 0;

    union_params<ShapeSphere> params;
    params.N = 2;
    params.diameter = 2*R;
    params.mpos[0] = vec3<Scalar>(x_i, 0, 0);
    params.mpos[1] = vec3<Scalar>(x_j, 0, 0);
    params.morientation[0] = o;
    params.morientation[1] = o;
    params.mparams[0] = par_i;
    params.mparams[1] = par_j;
    build_tree(params);

    // construct and check
    ShapeUnion<ShapeSphere> a(o, params);
    MY_BOOST_CHECK_CLOSE(a.orientation.s, o.s, tol);
    MY_BOOST_CHECK_CLOSE(a.orientation.v.x, o.v.x, tol);
    MY_BOOST_CHECK_CLOSE(a.orientation.v.y, o.v.y, tol);
    MY_BOOST_CHECK_CLOSE(a.orientation.v.z, o.v.z, tol);
    MY_BOOST_CHECK_CLOSE(a.members.diameter, R*2, tol);

    MY_BOOST_CHECK_CLOSE(a.members.morientation[0].s, o.s, tol);
    MY_BOOST_CHECK_CLOSE(a.members.morientation[0].v.x, o.v.x, tol);
    MY_BOOST_CHECK_CLOSE(a.members.morientation[1].v.y, o.v.y, tol);
    MY_BOOST_CHECK_CLOSE(a.members.morientation[1].v.z, o.v.z, tol);
    MY_BOOST_CHECK_CLOSE(a.members.mpos[0].x, x_i, tol);
    MY_BOOST_CHECK_CLOSE(a.members.mpos[1].x, x_j, tol);

    BOOST_CHECK(a.hasOrientation());

    MY_BOOST_CHECK_CLOSE(a.getCircumsphereDiameter(), R*2, tol);
    }

BOOST_AUTO_TEST_CASE( non_overlap )
    {
    // parameters
    quat<Scalar> o;
    quat<Scalar> o_a;
    quat<Scalar> o_b;
    vec3<Scalar> r_a;
    vec3<Scalar> r_b;

    // dumbbell: spheres of radius 0.25, located at x= +/- 0.25
    Scalar x_i(-0.25);
    Scalar x_j(0.25);
    Scalar R_i(0.25);
    Scalar R_j(0.25);
    Scalar R(x_j + R_j);
    ShapeSphere::param_type par_i;
    par_i.radius = R_i;
    par_i.ignore = 0;
    ShapeSphere::param_type par_j;
    par_j.radius = R_j;
    par_i.ignore = 0;

    union_params<ShapeSphere> params;
    params.N = 2;
    params.diameter = 2*R;
    params.mpos[0] = vec3<Scalar>(x_i, 0, 0);
    params.mpos[1] = vec3<Scalar>(x_j, 0, 0);
    params.morientation[0] = o;
    params.morientation[1] = o;
    params.mparams[0] = par_i;
    params.mparams[1] = par_j;
    params.ignore = 0;
    build_tree(params);

    // create two identical dumbbells
    ShapeUnion<ShapeSphere> a(o_a, params);
    ShapeUnion<ShapeSphere> b(o_b, params);

    // trivial orientation
    r_a = vec3<Scalar>(0,0,0);
    r_b = vec3<Scalar>(1.01, 0, 0);
    BOOST_CHECK(!test_overlap(r_b - r_a, a, b, err_count));
    BOOST_CHECK(!test_overlap(r_a - r_b, b, a, err_count));
    r_b = vec3<Scalar>(0, 0.51, 0);
    BOOST_CHECK(!test_overlap(r_b - r_a, a, b, err_count));
    BOOST_CHECK(!test_overlap(r_a - r_b, b, a, err_count));

    // rotate vertical: pi/2 about y axis
    Scalar alpha = M_PI/2.0;
    o_a = quat<Scalar>(cos(alpha/2.0), (Scalar)sin(alpha/2.0) * vec3<Scalar>(0,1,0));
    o_b = o_a;
    a.orientation = o_a;
    b.orientation = o_b;
    r_b = vec3<Scalar>(0.51, 0, 0);
    BOOST_CHECK(!test_overlap(r_b - r_a, a, b, err_count));
    BOOST_CHECK(!test_overlap(r_a - r_b, b, a, err_count));
    r_b = vec3<Scalar>(0, 1.01, 0);
    BOOST_CHECK(!test_overlap(r_b - r_a, a, b, err_count));
    BOOST_CHECK(!test_overlap(r_a - r_b, b, a, err_count));

    // 'a' x-axis aligned, 'b' z-axis aligned
    a.orientation = quat<Scalar>();
    r_b = vec3<Scalar>(0.75, 0, 0);
    BOOST_CHECK(!test_overlap(r_b - r_a, a, b, err_count));
    BOOST_CHECK(!test_overlap(r_a - r_b, b, a, err_count));
    r_b = vec3<Scalar>(0, 0, 0.75);
    BOOST_CHECK(!test_overlap(r_b - r_a, a, b, err_count));
    BOOST_CHECK(!test_overlap(r_a - r_b, b, a, err_count));
    r_b = vec3<Scalar>(0.76, 0, 0.25);
    BOOST_CHECK(!test_overlap(r_b - r_a, a, b, err_count));
    BOOST_CHECK(!test_overlap(r_a - r_b, b, a, err_count));
    }

BOOST_AUTO_TEST_CASE( overlapping_dumbbells )
    {
    // parameters
    quat<Scalar> o;
    quat<Scalar> o_a;
    quat<Scalar> o_b;
    vec3<Scalar> r_a;
    vec3<Scalar> r_b;

    // dumbbell: spheres of radius 0.25, located at x= +/- 0.25
    Scalar x_i(-0.25);
    Scalar x_j(0.25);
    Scalar R_i(0.25);
    Scalar R_j(0.25);
    Scalar R(x_j + R_j);
    ShapeSphere::param_type par_i;
    par_i.radius = R_i;
    par_i.ignore = 0;
    ShapeSphere::param_type par_j;
    par_j.radius = R_j;
    par_i.ignore = 0;

    union_params<ShapeSphere> params;
    params.N = 2;
    params.diameter = 2*R;
    params.mpos[0] = vec3<Scalar>(x_i, 0, 0);
    params.mpos[1] = vec3<Scalar>(x_j, 0, 0);
    params.morientation[0] = o;
    params.morientation[1] = o;
    params.mparams[0] = par_i;
    params.mparams[1] = par_j;
    params.ignore = 0;
    build_tree(params);

    // create two identical dumbbells
    ShapeUnion<ShapeSphere> a(o_a, params);
    ShapeUnion<ShapeSphere> b(o_b, params);

    // trivial orientation
    r_a = vec3<Scalar>(0,0,0);
    r_b = vec3<Scalar>(0.99, 0, 0);
    BOOST_CHECK(test_overlap(r_b - r_a, a, b, err_count));
    BOOST_CHECK(test_overlap(r_a - r_b, b, a, err_count));
    r_b = vec3<Scalar>(0, 0.49, 0);
    BOOST_CHECK(test_overlap(r_b - r_a, a, b, err_count));
    BOOST_CHECK(test_overlap(r_a - r_b, b, a, err_count));

    // rotate vertical: pi/2 about y axis
    Scalar alpha = M_PI/2.0;
    o_a = quat<Scalar>(cos(alpha/2.0), (Scalar)sin(alpha/2.0) * vec3<Scalar>(0,1,0));
    o_b = o_a;
    a.orientation = o_a;
    b.orientation = o_b;
    r_b = vec3<Scalar>(0.49, 0, 0);
    BOOST_CHECK(test_overlap(r_b - r_a, a, b, err_count));
    BOOST_CHECK(test_overlap(r_a - r_b, b, a, err_count));
    r_b = vec3<Scalar>(0, 0, 0.99);
    BOOST_CHECK(test_overlap(r_b - r_a, a, b, err_count));
    BOOST_CHECK(test_overlap(r_a - r_b, b, a, err_count));

    // 'a' x-axis aligned, 'b' z-axis aligned
    a.orientation = quat<Scalar>();
    r_b = vec3<Scalar>(0.68, 0, 0);
    BOOST_CHECK(test_overlap(r_b - r_a, a, b, err_count));
    BOOST_CHECK(test_overlap(r_a - r_b, b, a, err_count));
    r_b = vec3<Scalar>(0, 0, 0.68);
    BOOST_CHECK(test_overlap(r_b - r_a, a, b, err_count));
    BOOST_CHECK(test_overlap(r_a - r_b, b, a, err_count));
    r_b = vec3<Scalar>(0.74, 0, 0.25);
    BOOST_CHECK(test_overlap(r_b - r_a, a, b, err_count));
    BOOST_CHECK(test_overlap(r_a - r_b, b, a, err_count));
    }
