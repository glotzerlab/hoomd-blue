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
