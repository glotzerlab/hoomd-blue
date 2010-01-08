/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
// Maintainer: joaander

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <iostream>

//! Name the unit test module
#define BOOST_TEST_MODULE Index1D tests
#include "boost_utf_configure.h"

#include "Index1D.h"

using namespace std;
using namespace boost;

/*! \file lj_force_test.cc
    \brief Implements unit tests for LJForceCompute and descendants
    \ingroup unit_tests
*/

//! boost test case for 1x1 Index2D
BOOST_AUTO_TEST_CASE( Index2D_1 )
    {
    Index2D a(1);
    BOOST_CHECK_EQUAL(a.getNumElements(), (unsigned int)1);
    BOOST_CHECK_EQUAL(a(0,0), (unsigned int)0);
    }

//! boost test case for 2x2 Index2D
BOOST_AUTO_TEST_CASE( Index2D_2 )
    {
    Index2D a(2);
    BOOST_CHECK_EQUAL(a.getNumElements(), (unsigned int)4);
    BOOST_CHECK_EQUAL(a(0,0), (unsigned int)0);
    BOOST_CHECK_EQUAL(a(1,0), (unsigned int)1);
    BOOST_CHECK_EQUAL(a(0,1), (unsigned int)2);
    BOOST_CHECK_EQUAL(a(1,1), (unsigned int)3);
    }

//! boost test case for 3x3 Index2D
BOOST_AUTO_TEST_CASE( Index2D_3 )
    {
    Index2D a(3);
    BOOST_CHECK_EQUAL(a.getNumElements(), (unsigned int)9);
    BOOST_CHECK_EQUAL(a(0,0), (unsigned int)0);
    BOOST_CHECK_EQUAL(a(1,0), (unsigned int)1);
    BOOST_CHECK_EQUAL(a(2,0), (unsigned int)2);
    BOOST_CHECK_EQUAL(a(0,1), (unsigned int)3);
    BOOST_CHECK_EQUAL(a(1,1), (unsigned int)4);
    BOOST_CHECK_EQUAL(a(2,1), (unsigned int)5);
    BOOST_CHECK_EQUAL(a(0,2), (unsigned int)6);
    BOOST_CHECK_EQUAL(a(1,2), (unsigned int)7);
    BOOST_CHECK_EQUAL(a(2,2), (unsigned int)8);
    }

//! boost test case for 20x20 Index2D
BOOST_AUTO_TEST_CASE( Index2D_20 )
    {
    Index2D a(20);
    BOOST_CHECK_EQUAL(a.getNumElements(), (unsigned int)20*20);
    
    for (unsigned int i=0; i < 20; i++)
        for (unsigned int j=0; j < 20; j++)
            {
            BOOST_CHECK_EQUAL(a(i,j), j*20+i);
            }
    }

//! boost test case for 1x1 Index2DUpperTriangler
BOOST_AUTO_TEST_CASE( Index2DUpperTriangular_1 )
    {
    Index2DUpperTriangular a(1);
    BOOST_CHECK_EQUAL(a.getNumElements(), (unsigned int)1);
    BOOST_CHECK_EQUAL(a(0,0), (unsigned int)0);
    }

//! boost test case for 2x2 Index2DUpperTriangler
BOOST_AUTO_TEST_CASE( Index2DUpperTriangular_2 )
    {
    Index2DUpperTriangular a(2);
    BOOST_CHECK_EQUAL(a.getNumElements(), (unsigned int)3);
    BOOST_CHECK_EQUAL(a(0,0), (unsigned int)0);
    BOOST_CHECK_EQUAL(a(1,0), (unsigned int)1);
    BOOST_CHECK_EQUAL(a(1,1), (unsigned int)2);
    }

//! boost test case for 3x3 Index2DUpperTriangler
BOOST_AUTO_TEST_CASE( Index2DUpperTriangular_3 )
    {
    Index2DUpperTriangular a(3);
    BOOST_CHECK_EQUAL(a.getNumElements(), (unsigned int)6);
    BOOST_CHECK_EQUAL(a(0,0), (unsigned int)0);
    BOOST_CHECK_EQUAL(a(1,0), (unsigned int)1);
    BOOST_CHECK_EQUAL(a(2,0), (unsigned int)2);
    BOOST_CHECK_EQUAL(a(1,1), (unsigned int)3);
    BOOST_CHECK_EQUAL(a(2,1), (unsigned int)4);
    BOOST_CHECK_EQUAL(a(2,2), (unsigned int)5);
    }

//! boost test case for 20x20 Index2DUpperTriangular
BOOST_AUTO_TEST_CASE( Index2DUpperTriangular_20 )
    {
    Index2DUpperTriangular a(20);
    BOOST_CHECK_EQUAL(a.getNumElements(), (unsigned int)20*21/2);
    
    unsigned int cur_idx = 0;
    for (unsigned int i=0; i < 20; i++)
        for (unsigned int j=i; j < 20; j++)
            {
            BOOST_CHECK_EQUAL(a(i,j), cur_idx);
            BOOST_CHECK_EQUAL(a(j,i), cur_idx);
            cur_idx++;
            }
    }

#ifdef WIN32
#pragma warning( pop )
#endif

