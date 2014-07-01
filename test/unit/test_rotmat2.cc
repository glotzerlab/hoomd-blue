/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2014 The Regents of
the University of Michigan All rights reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

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

    rotmat2<Scalar> D = rotmat2<Scalar>::fromAngle(alpha);
    MY_BOOST_CHECK_SMALL(D.row0.x, tol_small);
    MY_BOOST_CHECK_CLOSE(D.row0.y, -1.0, tol);
    MY_BOOST_CHECK_CLOSE(D.row1.x, 1.0, tol);
    MY_BOOST_CHECK_SMALL(D.row1.y, tol_small);
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
