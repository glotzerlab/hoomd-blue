//! Name the unit test module
#define BOOST_TEST_MODULE vec3
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
    quat<Scalar> a;
    MY_BOOST_CHECK_CLOSE(a.s, 1.0, tol);
    MY_BOOST_CHECK_SMALL(a.v.x, tol_small);
    MY_BOOST_CHECK_SMALL(a.v.y, tol_small);
    MY_BOOST_CHECK_SMALL(a.v.z, tol_small);

    quat<Scalar> b(123, vec3<Scalar>(86, -103, 12));
    MY_BOOST_CHECK_CLOSE(b.s, 123, tol);
    MY_BOOST_CHECK_CLOSE(b.v.x, 86, tol);
    MY_BOOST_CHECK_CLOSE(b.v.y, -103, tol);
    MY_BOOST_CHECK_CLOSE(b.v.z, 12, tol);

    // note this mapping is for hoomd compatibility
    Scalar4 s4 = make_scalar4(-10, 25, 92, 12);
    quat<Scalar> c(s4);
    MY_BOOST_CHECK_CLOSE(c.s, s4.x, tol);
    MY_BOOST_CHECK_CLOSE(c.v.x, s4.y, tol);
    MY_BOOST_CHECK_CLOSE(c.v.y, s4.z, tol);
    MY_BOOST_CHECK_CLOSE(c.v.z, s4.w, tol);
    }

BOOST_AUTO_TEST_CASE( arithmetic )
    {
    quat<Scalar> a(1, vec3<Scalar>(2,3,4));
    quat<Scalar> b(4, vec3<Scalar>(6,8,10));
    quat<Scalar> c;
    vec3<Scalar> v;
    Scalar s=3;

    // test each operator separately
    c = a * s;
    MY_BOOST_CHECK_CLOSE(c.s, 3, tol);
    MY_BOOST_CHECK_CLOSE(c.v.x, 6, tol);
    MY_BOOST_CHECK_CLOSE(c.v.y, 9, tol);
    MY_BOOST_CHECK_CLOSE(c.v.z, 12, tol);

    c = s * a;
    MY_BOOST_CHECK_CLOSE(c.s, 3, tol);
    MY_BOOST_CHECK_CLOSE(c.v.x, 6, tol);
    MY_BOOST_CHECK_CLOSE(c.v.y, 9, tol);
    MY_BOOST_CHECK_CLOSE(c.v.z, 12, tol);

    c = a + b;
    MY_BOOST_CHECK_CLOSE(c.s, 5, tol);
    MY_BOOST_CHECK_CLOSE(c.v.x, 8, tol);
    MY_BOOST_CHECK_CLOSE(c.v.y, 11, tol);
    MY_BOOST_CHECK_CLOSE(c.v.z, 14, tol);

    c = a;
    c += b;
    MY_BOOST_CHECK_CLOSE(c.s, 5, tol);
    MY_BOOST_CHECK_CLOSE(c.v.x, 8, tol);
    MY_BOOST_CHECK_CLOSE(c.v.y, 11, tol);
    MY_BOOST_CHECK_CLOSE(c.v.z, 14, tol);

    c = a * b;
    MY_BOOST_CHECK_CLOSE(c.s, -72, tol);
    MY_BOOST_CHECK_CLOSE(c.v.x, 12, tol);
    MY_BOOST_CHECK_CLOSE(c.v.y, 24, tol);
    MY_BOOST_CHECK_CLOSE(c.v.z, 24, tol);

    c = a * vec3<Scalar>(6, 8, 10);
    MY_BOOST_CHECK_CLOSE(c.s, -76, tol);
    MY_BOOST_CHECK_CLOSE(c.v.x, 4, tol);
    MY_BOOST_CHECK_CLOSE(c.v.y, 12, tol);
    MY_BOOST_CHECK_CLOSE(c.v.z, 8, tol);

    c = vec3<Scalar>(6, 8, 10) * a;
    MY_BOOST_CHECK_CLOSE(c.s, -76, tol);
    MY_BOOST_CHECK_CLOSE(c.v.x, 8, tol);
    MY_BOOST_CHECK_CLOSE(c.v.y, 4, tol);
    MY_BOOST_CHECK_CLOSE(c.v.z, 12, tol);

    s = norm2(a);
    MY_BOOST_CHECK_CLOSE(s, 30, tol);

    c = conj(a);
    MY_BOOST_CHECK_CLOSE(c.s, a.s, tol);
    MY_BOOST_CHECK_CLOSE(c.v.x, -a.v.x, tol);
    MY_BOOST_CHECK_CLOSE(c.v.y, -a.v.y, tol);
    MY_BOOST_CHECK_CLOSE(c.v.z, -a.v.z, tol);

    v = rotate(a, vec3<Scalar>(1,1,1));
    MY_BOOST_CHECK_CLOSE(v.x, 6, tol);
    MY_BOOST_CHECK_CLOSE(v.y, 30, tol);
    MY_BOOST_CHECK_CLOSE(v.z, 42, tol);
    }

BOOST_AUTO_TEST_CASE( hoomd_compat )
    {
    // test convenience function for conversion to hoomd quaternions
    quat<Scalar> q(1, vec3<Scalar>(2,3,4));
    Scalar4 hq;
    hq = quat_to_scalar4(q);
    MY_BOOST_CHECK_CLOSE(hq.x, 1, tol);
    MY_BOOST_CHECK_CLOSE(hq.y, 2, tol);
    MY_BOOST_CHECK_CLOSE(hq.z, 3, tol);
    MY_BOOST_CHECK_CLOSE(hq.w, 4, tol);
    }

// test some quaternion identities for more sanity checking

BOOST_AUTO_TEST_CASE( conjugation_norm )
    {
    quat<Scalar> p(1, vec3<Scalar>(2,3,4));
    quat<Scalar> a;
    Scalar s1, s2;

    // conjugation and the norm
    s1 = norm2(p);
    a = p * conj(p);
    s2 = sqrt(norm2(a));
    MY_BOOST_CHECK_CLOSE(s1, s2, tol);
    }

BOOST_AUTO_TEST_CASE( multiplicative_norm )
    {
    quat<Scalar> p(1, vec3<Scalar>(2,3,4));
    quat<Scalar> q(0.4, vec3<Scalar>(0.3,0.2,0.1));
    quat<Scalar> a;
    Scalar s1, s2;

    // multiplicative norm
    s1 = norm2(p*q);
    s2 = norm2(p) * norm2(q);
    MY_BOOST_CHECK_CLOSE(s1, s2, tol);
    }

BOOST_AUTO_TEST_CASE( rotation )
    {
    Scalar pi = M_PI;
    Scalar alpha = pi/2.0; // angle of rotation
    vec3<Scalar> v(1,1,1); // test vector
    quat<Scalar> q1(cos(alpha/2.0), (Scalar)sin(alpha/2.0) * vec3<Scalar>(0,0,1)); // rotation quaternions
    quat<Scalar> q2(cos(alpha/2.0), (Scalar)sin(alpha/2.0) * vec3<Scalar>(1,0,0));
    quat<Scalar> q3;
    vec3<Scalar> a;

    a = (q1 * v * conj(q1)).v;
    MY_BOOST_CHECK_CLOSE(a.x, -1, tol);
    MY_BOOST_CHECK_CLOSE(a.y, 1, tol);
    MY_BOOST_CHECK_CLOSE(a.z, 1, tol);

    a = rotate(q1,v);
    MY_BOOST_CHECK_CLOSE(a.x, -1, tol);
    MY_BOOST_CHECK_CLOSE(a.y, 1, tol);
    MY_BOOST_CHECK_CLOSE(a.z, 1, tol);

    // test rotation composition
    a = (q2 * q1 * v * conj(q1) * conj(q2)).v;
    MY_BOOST_CHECK_CLOSE(a.x, -1, tol);
    MY_BOOST_CHECK_CLOSE(a.y, -1, tol);
    MY_BOOST_CHECK_CLOSE(a.z, 1, tol);

    q3 = q2 * q1;
    a = (q3 * v * conj(q3)).v;
    MY_BOOST_CHECK_CLOSE(a.x, -1, tol);
    MY_BOOST_CHECK_CLOSE(a.y, -1, tol);
    MY_BOOST_CHECK_CLOSE(a.z, 1, tol);

    a = rotate(q3,v);
    MY_BOOST_CHECK_CLOSE(a.x, -1, tol);
    MY_BOOST_CHECK_CLOSE(a.y, -1, tol);
    MY_BOOST_CHECK_CLOSE(a.z, 1, tol);

    // inverse rotation
    a = rotate(conj(q3), a);
    MY_BOOST_CHECK_CLOSE(a.x, v.x, tol);
    MY_BOOST_CHECK_CLOSE(a.y, v.y, tol);
    MY_BOOST_CHECK_CLOSE(a.z, v.z, tol);
    }

BOOST_AUTO_TEST_CASE( rotation_2 )
    {
    // test rotating a vec2
    Scalar pi = M_PI;
    Scalar alpha = pi/2.0; // angle of rotation
    vec2<Scalar> v(1,1); // test vector
    quat<Scalar> q1(cos(alpha/2.0), (Scalar)sin(alpha/2.0) * vec3<Scalar>(0,0,1)); // rotation quaternions
    vec2<Scalar> a;

    a = rotate(q1,v);
    MY_BOOST_CHECK_CLOSE(a.x, -1, tol);
    MY_BOOST_CHECK_CLOSE(a.y, 1, tol);
    }
