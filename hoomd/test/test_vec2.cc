// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// this include is necessary to get MPI included before anything else to support intel MPI
#include "hoomd/ExecutionConfiguration.h"

#include "upp11_config.h"

HOOMD_UP_MAIN();


#include <iostream>

#include <math.h>

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include <memory>

#include "hoomd/VectorMath.h"

UP_TEST( construction )
    {
    // test each constructor separately
    vec2<Scalar> a;
    MY_CHECK_SMALL(a.x, tol_small);
    MY_CHECK_SMALL(a.y, tol_small);

    vec2<Scalar> b(123, 86);
    MY_CHECK_CLOSE(b.x, 123, tol);
    MY_CHECK_CLOSE(b.y, 86, tol);
    }

UP_TEST( component_wise )
    {
    vec2<Scalar> a(1,2);
    vec2<Scalar> b(4,6);
    vec2<Scalar> c;

    // test each component-wise operator separately
    c = a + b;
    MY_CHECK_CLOSE(c.x, 5, tol);
    MY_CHECK_CLOSE(c.y, 8, tol);

    c = a - b;
    MY_CHECK_CLOSE(c.x, -3, tol);
    MY_CHECK_CLOSE(c.y, -4, tol);

    c = a * b;
    MY_CHECK_CLOSE(c.x, 4, tol);
    MY_CHECK_CLOSE(c.y, 12, tol);

    c = a / b;
    MY_CHECK_CLOSE(c.x, 1.0/4.0, tol);
    MY_CHECK_CLOSE(c.y, 2.0/6.0, tol);

    c = -a;
    MY_CHECK_CLOSE(c.x, -1, tol);
    MY_CHECK_CLOSE(c.y, -2, tol);
    }

UP_TEST( assignment_component_wise )
    {
    vec2<Scalar> a(1,2);
    vec2<Scalar> b(4,6);
    vec2<Scalar> c;

    // test each component-wise operator separately
    c = a += b;
    MY_CHECK_CLOSE(c.x, 5, tol);
    MY_CHECK_CLOSE(c.y, 8, tol);
    MY_CHECK_CLOSE(a.x, 5, tol);
    MY_CHECK_CLOSE(a.y, 8, tol);

    a = vec2<Scalar>(1,2);
    c = a -= b;
    MY_CHECK_CLOSE(c.x, -3, tol);
    MY_CHECK_CLOSE(c.y, -4, tol);
    MY_CHECK_CLOSE(a.x, -3, tol);
    MY_CHECK_CLOSE(a.y, -4, tol);

    a = vec2<Scalar>(1,2);
    c = a *= b;
    MY_CHECK_CLOSE(c.x, 4, tol);
    MY_CHECK_CLOSE(c.y, 12, tol);
    MY_CHECK_CLOSE(a.x, 4, tol);
    MY_CHECK_CLOSE(a.y, 12, tol);

    a = vec2<Scalar>(1,2);
    c = a /= b;
    MY_CHECK_CLOSE(c.x, 1.0/4.0, tol);
    MY_CHECK_CLOSE(c.y, 2.0/6.0, tol);
    MY_CHECK_CLOSE(a.x, 1.0/4.0, tol);
    MY_CHECK_CLOSE(a.y, 2.0/6.0, tol);
    }

UP_TEST( scalar )
    {
    vec2<Scalar> a(1,2);
    Scalar b(4);
    vec2<Scalar> c;

    // test each component-wise operator separately
    c = a * b;
    MY_CHECK_CLOSE(c.x, 4, tol);
    MY_CHECK_CLOSE(c.y, 8, tol);

    c = b * a;
    MY_CHECK_CLOSE(c.x, 4, tol);
    MY_CHECK_CLOSE(c.y, 8, tol);

    c = a / b;
    MY_CHECK_CLOSE(c.x, 1.0/4.0, tol);
    MY_CHECK_CLOSE(c.y, 2.0/4.0, tol);
    }

UP_TEST( assignment_scalar )
    {
    vec2<Scalar> a(1,2);
    Scalar b(4);

    // test each component-wise operator separately
    a = vec2<Scalar>(1,2);
    a *= b;
    MY_CHECK_CLOSE(a.x, 4, tol);
    MY_CHECK_CLOSE(a.y, 8, tol);

    a = vec2<Scalar>(1,2);
    a /= b;
    MY_CHECK_CLOSE(a.x, 1.0/4.0, tol);
    MY_CHECK_CLOSE(a.y, 2.0/4.0, tol);
    }

UP_TEST( vector_ops )
    {
    vec2<Scalar> a(1,2);
    vec2<Scalar> b(6,5);
    vec2<Scalar> c;
    Scalar d;

    // test each vector operation
    d = dot(a,b);
    MY_CHECK_CLOSE(d, 16, tol);

    c = perp(a);
    MY_CHECK_SMALL(dot(a,c), tol_small);
    }

UP_TEST( comparison )
    {
    vec2<Scalar> a(1.1,2.1);
    vec2<Scalar> b = a;
    vec2<Scalar> c(2.1,1.1);

    // test equality
    UP_ASSERT(a==b);
    UP_ASSERT(!(a==c));

    // test inequality
    UP_ASSERT(!(a!=b));
    UP_ASSERT(a!=c);
    }

UP_TEST( projection )
    {
    vec2<Scalar> a(3.4,5.5);
    vec2<Scalar> b(0.1,0);
    vec2<Scalar> c;

    // test projection
    c = project(a,b);
    MY_CHECK_CLOSE(c.x, 3.4, tol);
    MY_CHECK_SMALL(c.y, tol_small);
    }

UP_TEST( perpdot_product )
    {
    vec2<Scalar> a(3.4,5.5);
    vec2<Scalar> b(0.1,-4.2);

    // test projection
    Scalar c = perpdot(a,b);
    MY_CHECK_CLOSE(c, -14.83, tol);
    }

UP_TEST( test_swap )
    {
    vec2<Scalar> a(1.1, 2.2);
    vec2<Scalar> b(3.3, 4.4);
    vec2<Scalar> c(1.1, 2.2);
    vec2<Scalar> d(3.3, 4.4);

    // test swap
    a.swap(b);
    UP_ASSERT(a==d);
    UP_ASSERT(b==c);
    }

UP_TEST(test_assignment )
    {
    // Test assignment
    vec2<float> g;
    vec2<double> h;
    g = vec2<float>(121, 12);
    MY_CHECK_CLOSE(g.x, 121, tol);
    MY_CHECK_CLOSE(g.y, 12, tol);
    g = vec2<double>(-122, 15);
    MY_CHECK_CLOSE(g.x, -122, tol);
    MY_CHECK_CLOSE(g.y, 15, tol);
    h = vec2<float>(18, 12);
    MY_CHECK_CLOSE(h.x, 18, tol);
    MY_CHECK_CLOSE(h.y, 12, tol);
    h = vec2<double>(55, -64);
    MY_CHECK_CLOSE(h.x, 55, tol);
    MY_CHECK_CLOSE(h.y, -64, tol);
    }
