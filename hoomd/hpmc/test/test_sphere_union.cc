// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/ExecutionConfiguration.h"

#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN();

#include "hoomd/hpmc/IntegratorHPMC.h"
#include "hoomd/hpmc/Moves.h"
#include "hoomd/hpmc/ShapeUnion.h"

#include <iostream>
#include <string>

#include <memory>
#include <pybind11/pybind11.h>

using namespace hoomd;
using namespace hoomd::hpmc;
using namespace std;
using namespace hoomd::hpmc::detail;

unsigned int err_count;

template<class Shape> void build_tree(typename ShapeUnion<Shape>::param_type& data)
    {
    OBBTree tree;
    hpmc::detail::OBB* obbs;
    obbs = new hpmc::detail::OBB[data.N];

    // construct bounding box tree
    for (unsigned int i = 0; i < data.N; ++i)
        {
        Shape dummy(quat<Scalar>(data.morientation[i]), data.mparams[i]);
        obbs[i] = OBB(dummy.getAABB(data.mpos[i]));
        }

    tree.buildTree(obbs, data.N, 4, true);
    delete[] obbs;
    data.tree = GPUTree(tree);
    }

UP_TEST(construction)
    {
    // parameters
    quat<Scalar> o(1.0, vec3<Scalar>(-3.0, 9.0, 6.0));
    o = o * (Scalar)(Scalar(1.0) / sqrt(norm2(o)));

    // lopsided dumbbell: spheres of radius 0.25 and 0.5, located at x= +/- 0.25
    Scalar x_i(-0.25);
    Scalar x_j(0.25);
    Scalar R_i(0.25);
    Scalar R_j(0.5);
    Scalar R(x_j + R_j);
    ShapeSphere::param_type par_i;
    par_i.radius = ShortReal(R_i);
    par_i.ignore = 0;
    ShapeSphere::param_type par_j;
    par_j.radius = ShortReal(R_j);
    par_i.ignore = 0;

    ShapeUnion<ShapeSphere>::param_type params(2);
    params.diameter = ShortReal(2 * R);
    params.mpos[0] = vec3<Scalar>(x_i, 0, 0);
    params.mpos[1] = vec3<Scalar>(x_j, 0, 0);
    params.morientation[0] = o;
    params.morientation[1] = o;
    params.mparams[0] = par_i;
    params.mparams[1] = par_j;
    params.moverlap[0] = 1;
    params.moverlap[1] = 1;
    build_tree<ShapeSphere>(params);

    // construct and check
    ShapeUnion<ShapeSphere> a(o, params);
    MY_CHECK_CLOSE(a.orientation.s, o.s, tol);
    MY_CHECK_CLOSE(a.orientation.v.x, o.v.x, tol);
    MY_CHECK_CLOSE(a.orientation.v.y, o.v.y, tol);
    MY_CHECK_CLOSE(a.orientation.v.z, o.v.z, tol);
    MY_CHECK_CLOSE(a.members.diameter, R * 2, tol);

    MY_CHECK_CLOSE(a.members.morientation[0].s, o.s, tol);
    MY_CHECK_CLOSE(a.members.morientation[0].v.x, o.v.x, tol);
    MY_CHECK_CLOSE(a.members.morientation[1].v.y, o.v.y, tol);
    MY_CHECK_CLOSE(a.members.morientation[1].v.z, o.v.z, tol);
    MY_CHECK_CLOSE(a.members.mpos[0].x, x_i, tol);
    MY_CHECK_CLOSE(a.members.mpos[1].x, x_j, tol);

    UP_ASSERT(a.hasOrientation());

    MY_CHECK_CLOSE(a.getCircumsphereDiameter(), R * 2, tol);
    }

UP_TEST(non_overlap)
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
    par_i.radius = ShortReal(R_i);
    par_i.ignore = 0;
    ShapeSphere::param_type par_j;
    par_j.radius = ShortReal(R_j);
    par_i.ignore = 0;

    ShapeUnion<ShapeSphere>::param_type params(2);
    params.diameter = ShortReal(2 * R);
    params.mpos[0] = vec3<Scalar>(x_i, 0, 0);
    params.mpos[1] = vec3<Scalar>(x_j, 0, 0);
    params.morientation[0] = o;
    params.morientation[1] = o;
    params.mparams[0] = par_i;
    params.mparams[1] = par_j;
    params.ignore = 0;
    params.moverlap[0] = 1;
    params.moverlap[1] = 1;
    build_tree<ShapeSphere>(params);

    // create two identical dumbbells
    ShapeUnion<ShapeSphere> a(o_a, params);
    ShapeUnion<ShapeSphere> b(o_b, params);

    // trivial orientation
    r_a = vec3<Scalar>(0, 0, 0);
    r_b = vec3<Scalar>(1.01, 0, 0);
    UP_ASSERT(!test_overlap(r_b - r_a, a, b, err_count));
    UP_ASSERT(!test_overlap(r_a - r_b, b, a, err_count));
    r_b = vec3<Scalar>(0, 0.51, 0);
    UP_ASSERT(!test_overlap(r_b - r_a, a, b, err_count));
    UP_ASSERT(!test_overlap(r_a - r_b, b, a, err_count));

    // rotate vertical: pi/2 about y axis
    Scalar alpha = M_PI / 2.0;
    o_a = quat<Scalar>(cos(alpha / 2.0), (Scalar)sin(alpha / 2.0) * vec3<Scalar>(0, 1, 0));
    o_b = o_a;
    a.orientation = o_a;
    b.orientation = o_b;
    r_b = vec3<Scalar>(0.51, 0, 0);
    UP_ASSERT(!test_overlap(r_b - r_a, a, b, err_count));
    UP_ASSERT(!test_overlap(r_a - r_b, b, a, err_count));
    r_b = vec3<Scalar>(0, 1.01, 0);
    UP_ASSERT(!test_overlap(r_b - r_a, a, b, err_count));
    UP_ASSERT(!test_overlap(r_a - r_b, b, a, err_count));

    // 'a' x-axis aligned, 'b' z-axis aligned
    a.orientation = quat<Scalar>();
    r_b = vec3<Scalar>(0.75, 0, 0);
    UP_ASSERT(!test_overlap(r_b - r_a, a, b, err_count));
    UP_ASSERT(!test_overlap(r_a - r_b, b, a, err_count));
    r_b = vec3<Scalar>(0, 0, 0.75);
    UP_ASSERT(!test_overlap(r_b - r_a, a, b, err_count));
    UP_ASSERT(!test_overlap(r_a - r_b, b, a, err_count));
    r_b = vec3<Scalar>(0.76, 0, 0.25);
    UP_ASSERT(!test_overlap(r_b - r_a, a, b, err_count));
    UP_ASSERT(!test_overlap(r_a - r_b, b, a, err_count));
    }

UP_TEST(overlapping_dumbbells)
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
    par_i.radius = ShortReal(R_i);
    par_i.ignore = 0;
    ShapeSphere::param_type par_j;
    par_j.radius = ShortReal(R_j);
    par_i.ignore = 0;

    ShapeUnion<ShapeSphere>::param_type params(2);
    params.diameter = ShortReal(2 * R);
    params.mpos[0] = vec3<Scalar>(x_i, 0, 0);
    params.mpos[1] = vec3<Scalar>(x_j, 0, 0);
    params.morientation[0] = o;
    params.morientation[1] = o;
    params.mparams[0] = par_i;
    params.mparams[1] = par_j;
    params.ignore = 0;
    params.moverlap[0] = 1;
    params.moverlap[1] = 1;
    build_tree<ShapeSphere>(params);

    // create two identical dumbbells
    ShapeUnion<ShapeSphere> a(o_a, params);
    ShapeUnion<ShapeSphere> b(o_b, params);

    // trivial orientation
    r_a = vec3<Scalar>(0, 0, 0);
    r_b = vec3<Scalar>(0.99, 0, 0);
    UP_ASSERT(test_overlap(r_b - r_a, a, b, err_count));
    UP_ASSERT(test_overlap(r_a - r_b, b, a, err_count));
    r_b = vec3<Scalar>(0, 0.49, 0);
    UP_ASSERT(test_overlap(r_b - r_a, a, b, err_count));
    UP_ASSERT(test_overlap(r_a - r_b, b, a, err_count));

    // rotate vertical: pi/2 about y axis
    Scalar alpha = M_PI / 2.0;
    o_a = quat<Scalar>(cos(alpha / 2.0), (Scalar)sin(alpha / 2.0) * vec3<Scalar>(0, 1, 0));
    o_b = o_a;
    a.orientation = o_a;
    b.orientation = o_b;
    r_b = vec3<Scalar>(0.49, 0, 0);
    UP_ASSERT(test_overlap(r_b - r_a, a, b, err_count));
    UP_ASSERT(test_overlap(r_a - r_b, b, a, err_count));
    r_b = vec3<Scalar>(0, 0, 0.99);
    UP_ASSERT(test_overlap(r_b - r_a, a, b, err_count));
    UP_ASSERT(test_overlap(r_a - r_b, b, a, err_count));

    // 'a' x-axis aligned, 'b' z-axis aligned
    a.orientation = quat<Scalar>();
    r_b = vec3<Scalar>(0.68, 0, 0);
    UP_ASSERT(test_overlap(r_b - r_a, a, b, err_count));
    UP_ASSERT(test_overlap(r_a - r_b, b, a, err_count));
    r_b = vec3<Scalar>(0, 0, 0.68);
    UP_ASSERT(test_overlap(r_b - r_a, a, b, err_count));
    UP_ASSERT(test_overlap(r_a - r_b, b, a, err_count));
    r_b = vec3<Scalar>(0.74, 0, 0.25);
    UP_ASSERT(test_overlap(r_b - r_a, a, b, err_count));
    UP_ASSERT(test_overlap(r_a - r_b, b, a, err_count));
    }
