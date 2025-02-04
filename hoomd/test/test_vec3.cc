// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// this include is necessary to get MPI included before anything else to support intel MPI
#include "hoomd/ExecutionConfiguration.h"

#include "upp11_config.h"

#include <iostream>

#include <math.h>

#include <memory>
#include <pybind11/pybind11.h>

#include "hoomd/VectorMath.h"

using namespace hoomd;

HOOMD_UP_MAIN();

UP_TEST(construction)
    {
    // test each constructor separately
    vec3<Scalar> a;
    MY_CHECK_SMALL(a.x, tol_small);
    MY_CHECK_SMALL(a.y, tol_small);
    MY_CHECK_SMALL(a.z, tol_small);

    vec3<Scalar> b(123, 86, -103);
    MY_CHECK_CLOSE(b.x, 123, tol);
    MY_CHECK_CLOSE(b.y, 86, tol);
    MY_CHECK_CLOSE(b.z, -103, tol);

    Scalar3 s3 = make_scalar3(-10, 25, 92);
    vec3<Scalar> c(s3);
    MY_CHECK_CLOSE(c.x, s3.x, tol);
    MY_CHECK_CLOSE(c.y, s3.y, tol);
    MY_CHECK_CLOSE(c.z, s3.z, tol);

    Scalar4 s4 = make_scalar4(18, -22, 78, 12);
    vec3<Scalar> d(s4);
    MY_CHECK_CLOSE(d.x, s4.x, tol);
    MY_CHECK_CLOSE(d.y, s4.y, tol);
    MY_CHECK_CLOSE(d.z, s4.z, tol);

    vec3<float> e(123, 86, -103);
    MY_CHECK_CLOSE(vec3<float>(e).x, 123, tol);
    MY_CHECK_CLOSE(vec3<float>(e).y, 86, tol);
    MY_CHECK_CLOSE(vec3<float>(e).z, -103, tol);
    MY_CHECK_CLOSE(vec3<double>(e).x, 123, tol);
    MY_CHECK_CLOSE(vec3<double>(e).y, 86, tol);
    MY_CHECK_CLOSE(vec3<double>(e).z, -103, tol);

    vec3<double> f(123, 86, -103);
    MY_CHECK_CLOSE(vec3<float>(f).x, 123, tol);
    MY_CHECK_CLOSE(vec3<float>(f).y, 86, tol);
    MY_CHECK_CLOSE(vec3<float>(f).z, -103, tol);
    MY_CHECK_CLOSE(vec3<double>(f).x, 123, tol);
    MY_CHECK_CLOSE(vec3<double>(f).y, 86, tol);
    MY_CHECK_CLOSE(vec3<double>(f).z, -103, tol);

    // Test assignment
    vec3<float> g;
    vec3<double> h;
    g = vec3<float>(121, 12, -10);
    MY_CHECK_CLOSE(g.x, 121, tol);
    MY_CHECK_CLOSE(g.y, 12, tol);
    MY_CHECK_CLOSE(g.z, -10, tol);
    g = vec3<double>(-122, 15, 3);
    MY_CHECK_CLOSE(g.x, -122, tol);
    MY_CHECK_CLOSE(g.y, 15, tol);
    MY_CHECK_CLOSE(g.z, 3, tol);
    h = vec3<float>(18, 12, -1000);
    MY_CHECK_CLOSE(h.x, 18, tol);
    MY_CHECK_CLOSE(h.y, 12, tol);
    MY_CHECK_CLOSE(h.z, -1000, tol);
    h = vec3<double>(55, -64, 1);
    MY_CHECK_CLOSE(h.x, 55, tol);
    MY_CHECK_CLOSE(h.y, -64, tol);
    MY_CHECK_CLOSE(h.z, 1, tol);
    }

UP_TEST(component_wise)
    {
    vec3<Scalar> a(1, 2, 3);
    vec3<Scalar> b(4, 6, 8);
    vec3<Scalar> c;

    // test each component-wise operator separately
    c = a + b;
    MY_CHECK_CLOSE(c.x, 5, tol);
    MY_CHECK_CLOSE(c.y, 8, tol);
    MY_CHECK_CLOSE(c.z, 11, tol);

    c = a - b;
    MY_CHECK_CLOSE(c.x, -3, tol);
    MY_CHECK_CLOSE(c.y, -4, tol);
    MY_CHECK_CLOSE(c.z, -5, tol);

    c = a * b;
    MY_CHECK_CLOSE(c.x, 4, tol);
    MY_CHECK_CLOSE(c.y, 12, tol);
    MY_CHECK_CLOSE(c.z, 24, tol);

    c = a / b;
    MY_CHECK_CLOSE(c.x, 1.0 / 4.0, tol);
    MY_CHECK_CLOSE(c.y, 2.0 / 6.0, tol);
    MY_CHECK_CLOSE(c.z, 3.0 / 8.0, tol);

    c = -a;
    MY_CHECK_CLOSE(c.x, -1, tol);
    MY_CHECK_CLOSE(c.y, -2, tol);
    MY_CHECK_CLOSE(c.z, -3, tol);
    }

UP_TEST(assignment_component_wise)
    {
    vec3<Scalar> a(1, 2, 3);
    vec3<Scalar> b(4, 6, 8);
    vec3<Scalar> c;

    // test each component-wise operator separately
    c = a += b;
    MY_CHECK_CLOSE(c.x, 5, tol);
    MY_CHECK_CLOSE(c.y, 8, tol);
    MY_CHECK_CLOSE(c.z, 11, tol);
    MY_CHECK_CLOSE(a.x, 5, tol);
    MY_CHECK_CLOSE(a.y, 8, tol);
    MY_CHECK_CLOSE(a.z, 11, tol);

    a = vec3<Scalar>(1, 2, 3);
    c = a -= b;
    MY_CHECK_CLOSE(c.x, -3, tol);
    MY_CHECK_CLOSE(c.y, -4, tol);
    MY_CHECK_CLOSE(c.z, -5, tol);
    MY_CHECK_CLOSE(a.x, -3, tol);
    MY_CHECK_CLOSE(a.y, -4, tol);
    MY_CHECK_CLOSE(a.z, -5, tol);

    a = vec3<Scalar>(1, 2, 3);
    c = a *= b;
    MY_CHECK_CLOSE(c.x, 4, tol);
    MY_CHECK_CLOSE(c.y, 12, tol);
    MY_CHECK_CLOSE(c.z, 24, tol);
    MY_CHECK_CLOSE(a.x, 4, tol);
    MY_CHECK_CLOSE(a.y, 12, tol);
    MY_CHECK_CLOSE(a.z, 24, tol);

    a = vec3<Scalar>(1, 2, 3);
    c = a /= b;
    MY_CHECK_CLOSE(c.x, 1.0 / 4.0, tol);
    MY_CHECK_CLOSE(c.y, 2.0 / 6.0, tol);
    MY_CHECK_CLOSE(c.z, 3.0 / 8.0, tol);
    MY_CHECK_CLOSE(a.x, 1.0 / 4.0, tol);
    MY_CHECK_CLOSE(a.y, 2.0 / 6.0, tol);
    MY_CHECK_CLOSE(a.z, 3.0 / 8.0, tol);
    }

UP_TEST(scalar)
    {
    vec3<Scalar> a(1, 2, 3);
    Scalar b(4);
    vec3<Scalar> c;

    // test each component-wise operator separately
    c = a * b;
    MY_CHECK_CLOSE(c.x, 4, tol);
    MY_CHECK_CLOSE(c.y, 8, tol);
    MY_CHECK_CLOSE(c.z, 12, tol);

    c = b * a;
    MY_CHECK_CLOSE(c.x, 4, tol);
    MY_CHECK_CLOSE(c.y, 8, tol);
    MY_CHECK_CLOSE(c.z, 12, tol);

    c = a / b;
    MY_CHECK_CLOSE(c.x, 1.0 / 4.0, tol);
    MY_CHECK_CLOSE(c.y, 2.0 / 4.0, tol);
    MY_CHECK_CLOSE(c.z, 3.0 / 4.0, tol);
    }

UP_TEST(assignment_scalar)
    {
    vec3<Scalar> a(1, 2, 3);
    Scalar b(4);

    // test each component-wise operator separately
    a = vec3<Scalar>(1, 2, 3);
    a *= b;
    MY_CHECK_CLOSE(a.x, 4, tol);
    MY_CHECK_CLOSE(a.y, 8, tol);
    MY_CHECK_CLOSE(a.z, 12, tol);

    a = vec3<Scalar>(1, 2, 3);
    a /= b;
    MY_CHECK_CLOSE(a.x, 1.0 / 4.0, tol);
    MY_CHECK_CLOSE(a.y, 2.0 / 4.0, tol);
    MY_CHECK_CLOSE(a.z, 3.0 / 4.0, tol);
    }

UP_TEST(vector_ops)
    {
    vec3<Scalar> a(1, 2, 3);
    vec3<Scalar> b(6, 5, 4);
    vec3<Scalar> c;
    Scalar d;

    // test each vector operation
    d = dot(a, b);
    MY_CHECK_CLOSE(d, 28, tol);

    c = cross(a, b);
    MY_CHECK_CLOSE(c.x, -7, tol);
    MY_CHECK_CLOSE(c.y, 14, tol);
    MY_CHECK_CLOSE(c.z, -7, tol);
    }

UP_TEST(vec_to_scalar)
    {
    vec3<Scalar> a(1, 2, 3);
    Scalar w(4);
    Scalar3 m;
    Scalar4 n;

    // test convenience functions for converting between types
    m = vec_to_scalar3(a);
    MY_CHECK_CLOSE(m.x, 1, tol);
    MY_CHECK_CLOSE(m.y, 2, tol);
    MY_CHECK_CLOSE(m.z, 3, tol);

    n = vec_to_scalar4(a, w);
    MY_CHECK_CLOSE(n.x, 1, tol);
    MY_CHECK_CLOSE(n.y, 2, tol);
    MY_CHECK_CLOSE(n.z, 3, tol);
    MY_CHECK_CLOSE(n.w, 4, tol);

    // test mapping of Scalar{3,4} to vec3
    a = vec3<Scalar>(0.0, 0.0, 0.0);
    a = vec3<Scalar>(m);
    MY_CHECK_CLOSE(a.x, 1.0, tol);
    MY_CHECK_CLOSE(a.y, 2.0, tol);
    MY_CHECK_CLOSE(a.z, 3.0, tol);

    a = vec3<Scalar>(0.0, 0.0, 0.0);
    a = vec3<Scalar>(n);
    MY_CHECK_CLOSE(a.x, 1.0, tol);
    MY_CHECK_CLOSE(a.y, 2.0, tol);
    MY_CHECK_CLOSE(a.z, 3.0, tol);
    }

UP_TEST(comparison)
    {
    vec3<Scalar> a(1.1, 2.1, .1);
    vec3<Scalar> b = a;
    vec3<Scalar> c(.1, 1.1, 2.1);

    // test equality
    UP_ASSERT(a == b);
    UP_ASSERT(!(a == c));

    // test inequality
    UP_ASSERT(!(a != b));
    UP_ASSERT(a != c);
    }

UP_TEST(test_swap)
    {
    vec3<Scalar> a(1.1, 2.2, 0.0);
    vec3<Scalar> b(3.3, 4.4, 0.0);
    vec3<Scalar> c(1.1, 2.2, 0.0);
    vec3<Scalar> d(3.3, 4.4, 0.0);

    // test swap
    a.swap(b);
    UP_ASSERT(a == d);
    UP_ASSERT(b == c);
    }
