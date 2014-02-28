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
    rotmat3<Scalar> A;
    MY_BOOST_CHECK_CLOSE(A.row0.x, 1.0, tol);
    MY_BOOST_CHECK_SMALL(A.row0.y, tol_small);
    MY_BOOST_CHECK_SMALL(A.row0.z, tol_small);
    MY_BOOST_CHECK_SMALL(A.row1.x, tol_small);
    MY_BOOST_CHECK_CLOSE(A.row1.y, 1.0, tol);
    MY_BOOST_CHECK_SMALL(A.row1.z, tol_small);
    MY_BOOST_CHECK_SMALL(A.row2.x, tol_small);
    MY_BOOST_CHECK_SMALL(A.row2.y, tol_small);
    MY_BOOST_CHECK_CLOSE(A.row2.z, 1.0, tol);

    rotmat3<Scalar> B(vec3<Scalar>(1,2,3), vec3<Scalar>(4,5,6), vec3<Scalar>(7,8,9));
    MY_BOOST_CHECK_CLOSE(B.row0.x, 1.0, tol);
    MY_BOOST_CHECK_CLOSE(B.row0.y, 2.0, tol);
    MY_BOOST_CHECK_CLOSE(B.row0.z, 3.0, tol);
    MY_BOOST_CHECK_CLOSE(B.row1.x, 4.0, tol);
    MY_BOOST_CHECK_CLOSE(B.row1.y, 5.0, tol);
    MY_BOOST_CHECK_CLOSE(B.row1.z, 6.0, tol);
    MY_BOOST_CHECK_CLOSE(B.row2.x, 7.0, tol);
    MY_BOOST_CHECK_CLOSE(B.row2.y, 8.0, tol);
    MY_BOOST_CHECK_CLOSE(B.row2.z, 9.0, tol);

    Scalar pi = M_PI;
    Scalar alpha = pi/2.0; // angle of rotation
    quat<Scalar> q1(cos(alpha/2.0), (Scalar)sin(alpha/2.0) * vec3<Scalar>(0,0,1)); // rotation quaternions
    rotmat3<Scalar> C(q1);
    MY_BOOST_CHECK_SMALL(C.row0.x, tol_small);
    MY_BOOST_CHECK_CLOSE(C.row0.y, -1.0, tol);
    MY_BOOST_CHECK_SMALL(C.row0.z, tol_small);
    MY_BOOST_CHECK_CLOSE(C.row1.x, 1.0, tol);
    MY_BOOST_CHECK_SMALL(C.row1.y, tol_small);
    MY_BOOST_CHECK_SMALL(C.row1.z, tol_small);
    MY_BOOST_CHECK_SMALL(C.row2.x, tol_small);
    MY_BOOST_CHECK_SMALL(C.row2.y, tol_small);
    MY_BOOST_CHECK_CLOSE(C.row2.z, 1.0, tol);
    }

BOOST_AUTO_TEST_CASE( rotation_1 )
    {
    // test rotating a vec3
    Scalar pi = M_PI;
    Scalar alpha = pi/2.0; // angle of rotation
    vec3<Scalar> v(1,1,1); // test vector
    quat<Scalar> q1(cos(alpha/2.0), (Scalar)sin(alpha/2.0) * vec3<Scalar>(0,0,1)); // rotation quaternions
    vec3<Scalar> a;

    rotmat3<Scalar> A(q1);

    a = A*v;
    MY_BOOST_CHECK_CLOSE(a.x, -1, tol);
    MY_BOOST_CHECK_CLOSE(a.y, 1, tol);
    MY_BOOST_CHECK_CLOSE(a.z, 1, tol);
    }

BOOST_AUTO_TEST_CASE( rotation_2 )
    {
    // test rotating a vec3
    Scalar pi = M_PI;
    Scalar alpha = pi/2.0; // angle of rotation
    vec3<Scalar> v(1,1,1); // test vector
    quat<Scalar> q1(cos(alpha/2.0), (Scalar)sin(alpha/2.0) * vec3<Scalar>(0,1,0)); // rotation quaternions
    vec3<Scalar> a;

    rotmat3<Scalar> A(q1);

    a = A*v;
    MY_BOOST_CHECK_CLOSE(a.x, 1, tol);
    MY_BOOST_CHECK_CLOSE(a.y, 1, tol);
    MY_BOOST_CHECK_CLOSE(a.z, -1, tol);
    }

BOOST_AUTO_TEST_CASE( rotation_3 )
    {
    // test rotating a vec3
    Scalar pi = M_PI;
    Scalar alpha = pi/2.0; // angle of rotation
    vec3<Scalar> v(1,1,1); // test vector
    quat<Scalar> q1(cos(alpha/2.0), (Scalar)sin(alpha/2.0) * vec3<Scalar>(1,0,0)); // rotation quaternions
    vec3<Scalar> a;

    rotmat3<Scalar> A(q1);

    a = A*v;
    MY_BOOST_CHECK_CLOSE(a.x, 1, tol);
    MY_BOOST_CHECK_CLOSE(a.y, -1, tol);
    MY_BOOST_CHECK_CLOSE(a.z, 1, tol);
    }
