//! Name the unit test module
#define BOOST_TEST_MODULE rotmat2
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
    rotmat2<Scalar> A;
    MY_BOOST_CHECK_CLOSE(A.row0.x, 1.0, tol);
    MY_BOOST_CHECK_SMALL(A.row0.y, tol_small);
    MY_BOOST_CHECK_SMALL(A.row1.x, tol_small);
    MY_BOOST_CHECK_CLOSE(A.row1.y, 1.0, tol);

    rotmat2<Scalar> B(vec2<Scalar>(1,2), vec2<Scalar>(3,4));
    MY_BOOST_CHECK_CLOSE(B.row0.x, 1.0, tol);
    MY_BOOST_CHECK_CLOSE(B.row0.y, 2.0, tol);
    MY_BOOST_CHECK_CLOSE(B.row1.x, 3.0, tol);
    MY_BOOST_CHECK_CLOSE(B.row1.y, 4.0, tol);

    Scalar pi = M_PI;
    Scalar alpha = pi/2.0; // angle of rotation
    quat<Scalar> q1(cos(alpha/2.0), (Scalar)sin(alpha/2.0) * vec3<Scalar>(0,0,1)); // rotation quaternions
    rotmat2<Scalar> C(q1);
    MY_BOOST_CHECK_SMALL(C.row0.x, tol_small);
    MY_BOOST_CHECK_CLOSE(C.row0.y, -1.0, tol);
    MY_BOOST_CHECK_CLOSE(C.row1.x, 1.0, tol);
    MY_BOOST_CHECK_SMALL(C.row1.y, tol_small);
    }

BOOST_AUTO_TEST_CASE( rotation_2 )
    {
    // test rotating a vec2
    Scalar pi = M_PI;
    Scalar alpha = pi/2.0; // angle of rotation
    vec2<Scalar> v(1,1); // test vector
    quat<Scalar> q1(cos(alpha/2.0), (Scalar)sin(alpha/2.0) * vec3<Scalar>(0,0,1)); // rotation quaternions
    vec2<Scalar> a;

    rotmat2<Scalar> A(q1);

    a = A*v;
    MY_BOOST_CHECK_CLOSE(a.x, -1, tol);
    MY_BOOST_CHECK_CLOSE(a.y, 1, tol);
    }


BOOST_AUTO_TEST_CASE( transpose_2 )
    {
    // test rotating a vec2
    Scalar pi = M_PI;
    Scalar alpha = pi/2.0; // angle of rotation
    vec2<Scalar> v(0.1234,1.76123); // test vector
    quat<Scalar> q1(cos(alpha/2.0), (Scalar)sin(alpha/2.0) * vec3<Scalar>(0,0,1)); // rotation quaternions
    vec2<Scalar> a, b;

    rotmat2<Scalar> A(q1), B;

    a = A*v;
    b = transpose(A)*a;
    MY_BOOST_CHECK_CLOSE(b.x, v.x, tol);
    MY_BOOST_CHECK_CLOSE(b.y, v.y, tol);
    }
