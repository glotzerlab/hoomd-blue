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
    rotmat2<Scalar> A;
    MY_CHECK_CLOSE(A.row0.x, 1.0, tol);
    MY_CHECK_SMALL(A.row0.y, tol_small);
    MY_CHECK_SMALL(A.row1.x, tol_small);
    MY_CHECK_CLOSE(A.row1.y, 1.0, tol);

    rotmat2<Scalar> B(vec2<Scalar>(1,2), vec2<Scalar>(3,4));
    MY_CHECK_CLOSE(B.row0.x, 1.0, tol);
    MY_CHECK_CLOSE(B.row0.y, 2.0, tol);
    MY_CHECK_CLOSE(B.row1.x, 3.0, tol);
    MY_CHECK_CLOSE(B.row1.y, 4.0, tol);

    Scalar pi = M_PI;
    Scalar alpha = pi/2.0; // angle of rotation
    quat<Scalar> q1(cos(alpha/2.0), (Scalar)sin(alpha/2.0) * vec3<Scalar>(0,0,1)); // rotation quaternions
    rotmat2<Scalar> C(q1);
    MY_CHECK_SMALL(C.row0.x, tol_small);
    MY_CHECK_CLOSE(C.row0.y, -1.0, tol);
    MY_CHECK_CLOSE(C.row1.x, 1.0, tol);
    MY_CHECK_SMALL(C.row1.y, tol_small);

    rotmat2<Scalar> D = rotmat2<Scalar>::fromAngle(alpha);
    MY_CHECK_SMALL(D.row0.x, tol_small);
    MY_CHECK_CLOSE(D.row0.y, -1.0, tol);
    MY_CHECK_CLOSE(D.row1.x, 1.0, tol);
    MY_CHECK_SMALL(D.row1.y, tol_small);
    }

UP_TEST( rotation_2 )
    {
    // test rotating a vec2
    Scalar pi = M_PI;
    Scalar alpha = pi/2.0; // angle of rotation
    vec2<Scalar> v(1,1); // test vector
    quat<Scalar> q1(cos(alpha/2.0), (Scalar)sin(alpha/2.0) * vec3<Scalar>(0,0,1)); // rotation quaternions
    vec2<Scalar> a;

    rotmat2<Scalar> A(q1);

    a = A*v;
    MY_CHECK_CLOSE(a.x, -1, tol);
    MY_CHECK_CLOSE(a.y, 1, tol);
    }


UP_TEST( transpose_2 )
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
    MY_CHECK_CLOSE(b.x, v.x, tol);
    MY_CHECK_CLOSE(b.y, v.y, tol);
    }
