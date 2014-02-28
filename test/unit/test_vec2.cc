//! Name the unit test module
#define BOOST_TEST_MODULE vec2
#include "boost_utf_configure.h"

#include <iostream>

#include <math.h>

#include <boost/bind.hpp>
#include <boost/python.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include "VectorMath.h"

BOOST_AUTO_TEST_CASE( construction )
    {
    // test each constructor separately
    vec2<Scalar> a;
    MY_BOOST_CHECK_SMALL(a.x, tol_small);
    MY_BOOST_CHECK_SMALL(a.y, tol_small);

    vec2<Scalar> b(123, 86);
    MY_BOOST_CHECK_CLOSE(b.x, 123, tol);
    MY_BOOST_CHECK_CLOSE(b.y, 86, tol);
    }

BOOST_AUTO_TEST_CASE( component_wise )
    {
    vec2<Scalar> a(1,2);
    vec2<Scalar> b(4,6);
    vec2<Scalar> c;

    // test each component-wise operator separately
    c = a + b;
    MY_BOOST_CHECK_CLOSE(c.x, 5, tol);
    MY_BOOST_CHECK_CLOSE(c.y, 8, tol);

    c = a - b;
    MY_BOOST_CHECK_CLOSE(c.x, -3, tol);
    MY_BOOST_CHECK_CLOSE(c.y, -4, tol);

    c = a * b;
    MY_BOOST_CHECK_CLOSE(c.x, 4, tol);
    MY_BOOST_CHECK_CLOSE(c.y, 12, tol);

    c = a / b;
    MY_BOOST_CHECK_CLOSE(c.x, 1.0/4.0, tol);
    MY_BOOST_CHECK_CLOSE(c.y, 2.0/6.0, tol);

    c = -a;
    MY_BOOST_CHECK_CLOSE(c.x, -1, tol);
    MY_BOOST_CHECK_CLOSE(c.y, -2, tol);
    }

BOOST_AUTO_TEST_CASE( assignment_component_wise )
    {
    vec2<Scalar> a(1,2);
    vec2<Scalar> b(4,6);
    vec2<Scalar> c;

    // test each component-wise operator separately
    c = a += b;
    MY_BOOST_CHECK_CLOSE(c.x, 5, tol);
    MY_BOOST_CHECK_CLOSE(c.y, 8, tol);
    MY_BOOST_CHECK_CLOSE(a.x, 5, tol);
    MY_BOOST_CHECK_CLOSE(a.y, 8, tol);

    a = vec2<Scalar>(1,2);
    c = a -= b;
    MY_BOOST_CHECK_CLOSE(c.x, -3, tol);
    MY_BOOST_CHECK_CLOSE(c.y, -4, tol);
    MY_BOOST_CHECK_CLOSE(a.x, -3, tol);
    MY_BOOST_CHECK_CLOSE(a.y, -4, tol);

    a = vec2<Scalar>(1,2);
    c = a *= b;
    MY_BOOST_CHECK_CLOSE(c.x, 4, tol);
    MY_BOOST_CHECK_CLOSE(c.y, 12, tol);
    MY_BOOST_CHECK_CLOSE(a.x, 4, tol);
    MY_BOOST_CHECK_CLOSE(a.y, 12, tol);

    a = vec2<Scalar>(1,2);
    c = a /= b;
    MY_BOOST_CHECK_CLOSE(c.x, 1.0/4.0, tol);
    MY_BOOST_CHECK_CLOSE(c.y, 2.0/6.0, tol);
    MY_BOOST_CHECK_CLOSE(a.x, 1.0/4.0, tol);
    MY_BOOST_CHECK_CLOSE(a.y, 2.0/6.0, tol);
    }

BOOST_AUTO_TEST_CASE( scalar )
    {
    vec2<Scalar> a(1,2);
    Scalar b(4);
    vec2<Scalar> c;

    // test each component-wise operator separately
    c = a * b;
    MY_BOOST_CHECK_CLOSE(c.x, 4, tol);
    MY_BOOST_CHECK_CLOSE(c.y, 8, tol);

    c = b * a;
    MY_BOOST_CHECK_CLOSE(c.x, 4, tol);
    MY_BOOST_CHECK_CLOSE(c.y, 8, tol);

    c = a / b;
    MY_BOOST_CHECK_CLOSE(c.x, 1.0/4.0, tol);
    MY_BOOST_CHECK_CLOSE(c.y, 2.0/4.0, tol);
    }

BOOST_AUTO_TEST_CASE( assignment_scalar )
    {
    vec2<Scalar> a(1,2);
    Scalar b(4);

    // test each component-wise operator separately
    a = vec2<Scalar>(1,2);
    a *= b;
    MY_BOOST_CHECK_CLOSE(a.x, 4, tol);
    MY_BOOST_CHECK_CLOSE(a.y, 8, tol);

    a = vec2<Scalar>(1,2);
    a /= b;
    MY_BOOST_CHECK_CLOSE(a.x, 1.0/4.0, tol);
    MY_BOOST_CHECK_CLOSE(a.y, 2.0/4.0, tol);
    }

BOOST_AUTO_TEST_CASE( vector_ops )
    {
    vec2<Scalar> a(1,2);
    vec2<Scalar> b(6,5);
    vec2<Scalar> c;
    Scalar d;

    // test each vector operation
    d = dot(a,b);
    MY_BOOST_CHECK_CLOSE(d, 16, tol);

    c = perp(a);
    MY_BOOST_CHECK_SMALL(dot(a,c), tol_small);
    }

BOOST_AUTO_TEST_CASE( comparison )
    {
    vec2<Scalar> a(1.1,2.1);
    vec2<Scalar> b = a;
    vec2<Scalar> c(2.1,1.1);

    // test equality
    BOOST_CHECK(a==b);
    BOOST_CHECK(!(a==c));

    // test inequality
    BOOST_CHECK(!(a!=b));
    BOOST_CHECK(a!=c);
    }

BOOST_AUTO_TEST_CASE( projection )
    {
    vec2<Scalar> a(3.4,5.5);
    vec2<Scalar> b(0.1,0);
    vec2<Scalar> c;

    // test projection
    c = project(a,b);
    MY_BOOST_CHECK_CLOSE(c.x, 3.4, tol);
    MY_BOOST_CHECK_SMALL(c.y, tol_small);
    }

BOOST_AUTO_TEST_CASE( perpdot_product )
    {
    vec2<Scalar> a(3.4,5.5);
    vec2<Scalar> b(0.1,-4.2);

    // test projection
    Scalar c = perpdot(a,b);
    MY_BOOST_CHECK_CLOSE(c, -14.83, tol);
    }

BOOST_AUTO_TEST_CASE( test_swap )
    {
    vec2<Scalar> a(1.1, 2.2);
    vec2<Scalar> b(3.3, 4.4);
    vec2<Scalar> c(1.1, 2.2);
    vec2<Scalar> d(3.3, 4.4);

    // test swap
    swap(a, b);
    BOOST_CHECK(a==d);
    BOOST_CHECK(b==c);
    }
