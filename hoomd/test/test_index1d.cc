// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// this include is necessary to get MPI included before anything else to support intel MPI
#include "hoomd/ExecutionConfiguration.h"

#include <iostream>

//! Name the unit test module
UP_TEST(Index1D tests)
#include "boost_utf_configure.h"

#include "hoomd/Index1D.h"

using namespace std;
using namespace boost;

/*! \file lj_force_test.cc
    \brief Implements unit tests for LJForceCompute and descendants
    \ingroup unit_tests
*/

//! boost test case for 1x1 Index2D
UP_TEST( Index2D_1 )
    {
    Index2D a(1);
    UPP_ASSERT_EQUAL(a.getNumElements(), (unsigned int)1);
    UPP_ASSERT_EQUAL(a(0,0), (unsigned int)0);
    }

//! boost test case for 2x2 Index2D
UP_TEST( Index2D_2 )
    {
    Index2D a(2);
    UPP_ASSERT_EQUAL(a.getNumElements(), (unsigned int)4);
    UPP_ASSERT_EQUAL(a(0,0), (unsigned int)0);
    UPP_ASSERT_EQUAL(a(1,0), (unsigned int)1);
    UPP_ASSERT_EQUAL(a(0,1), (unsigned int)2);
    UPP_ASSERT_EQUAL(a(1,1), (unsigned int)3);
    }

//! boost test case for 3x3 Index2D
UP_TEST( Index2D_3 )
    {
    Index2D a(3);
    UPP_ASSERT_EQUAL(a.getNumElements(), (unsigned int)9);
    UPP_ASSERT_EQUAL(a(0,0), (unsigned int)0);
    UPP_ASSERT_EQUAL(a(1,0), (unsigned int)1);
    UPP_ASSERT_EQUAL(a(2,0), (unsigned int)2);
    UPP_ASSERT_EQUAL(a(0,1), (unsigned int)3);
    UPP_ASSERT_EQUAL(a(1,1), (unsigned int)4);
    UPP_ASSERT_EQUAL(a(2,1), (unsigned int)5);
    UPP_ASSERT_EQUAL(a(0,2), (unsigned int)6);
    UPP_ASSERT_EQUAL(a(1,2), (unsigned int)7);
    UPP_ASSERT_EQUAL(a(2,2), (unsigned int)8);
    }

//! boost test case for 2x2x2 Index3D
UP_TEST( Index3D_2 )
    {
    Index3D a(2);
    UPP_ASSERT_EQUAL(a.getNumElements(), (unsigned int)8);
    UPP_ASSERT_EQUAL(a(0,0,0), (unsigned int)0);
    UPP_ASSERT_EQUAL(a(1,0,0), (unsigned int)1);
    UPP_ASSERT_EQUAL(a(0,1,0), (unsigned int)2);
    UPP_ASSERT_EQUAL(a(1,1,0), (unsigned int)3);
    UPP_ASSERT_EQUAL(a(0,0,1), (unsigned int)4);
    UPP_ASSERT_EQUAL(a(1,0,1), (unsigned int)5);
    UPP_ASSERT_EQUAL(a(0,1,1), (unsigned int)6);
    UPP_ASSERT_EQUAL(a(1,1,1), (unsigned int)7);
    }

//! boost test case for 4x3x2 Index3D
UP_TEST( Index3D_432 )
    {
    Index3D a(4,3,2);
    UPP_ASSERT_EQUAL(a.getNumElements(), (unsigned int)24);
    UPP_ASSERT_EQUAL(a(0,0,0), (unsigned int)0);
    UPP_ASSERT_EQUAL(a(1,0,0), (unsigned int)1);
    UPP_ASSERT_EQUAL(a(2,0,0), (unsigned int)2);
    UPP_ASSERT_EQUAL(a(3,0,0), (unsigned int)3);
    UPP_ASSERT_EQUAL(a(0,1,0), (unsigned int)4);
    UPP_ASSERT_EQUAL(a(1,1,0), (unsigned int)5);
    UPP_ASSERT_EQUAL(a(2,1,0), (unsigned int)6);
    UPP_ASSERT_EQUAL(a(3,1,0), (unsigned int)7);
    UPP_ASSERT_EQUAL(a(0,2,0), (unsigned int)8);
    UPP_ASSERT_EQUAL(a(1,2,0), (unsigned int)9);
    UPP_ASSERT_EQUAL(a(2,2,0), (unsigned int)10);
    UPP_ASSERT_EQUAL(a(3,2,0), (unsigned int)11);

    UPP_ASSERT_EQUAL(a(0,0,1), (unsigned int)12);
    UPP_ASSERT_EQUAL(a(1,0,1), (unsigned int)13);
    UPP_ASSERT_EQUAL(a(2,0,1), (unsigned int)14);
    UPP_ASSERT_EQUAL(a(3,0,1), (unsigned int)15);
    UPP_ASSERT_EQUAL(a(0,1,1), (unsigned int)16);
    UPP_ASSERT_EQUAL(a(1,1,1), (unsigned int)17);
    UPP_ASSERT_EQUAL(a(2,1,1), (unsigned int)18);
    UPP_ASSERT_EQUAL(a(3,1,1), (unsigned int)19);
    UPP_ASSERT_EQUAL(a(0,2,1), (unsigned int)20);
    UPP_ASSERT_EQUAL(a(1,2,1), (unsigned int)21);
    UPP_ASSERT_EQUAL(a(2,2,1), (unsigned int)22);
    UPP_ASSERT_EQUAL(a(3,2,1), (unsigned int)23);
    }

//! boost test case for 20x20 Index2D
UP_TEST( Index2D_20 )
    {
    Index2D a(20);
    UPP_ASSERT_EQUAL(a.getNumElements(), (unsigned int)20*20);

    for (unsigned int i=0; i < 20; i++)
        for (unsigned int j=0; j < 20; j++)
            {
            UPP_ASSERT_EQUAL(a(i,j), j*20+i);
            }
    }

//! boost test case for 1x1 Index2DUpperTriangler
UP_TEST( Index2DUpperTriangular_1 )
    {
    Index2DUpperTriangular a(1);
    UPP_ASSERT_EQUAL(a.getNumElements(), (unsigned int)1);
    UPP_ASSERT_EQUAL(a(0,0), (unsigned int)0);
    }

//! boost test case for 2x2 Index2DUpperTriangler
UP_TEST( Index2DUpperTriangular_2 )
    {
    Index2DUpperTriangular a(2);
    UPP_ASSERT_EQUAL(a.getNumElements(), (unsigned int)3);
    UPP_ASSERT_EQUAL(a(0,0), (unsigned int)0);
    UPP_ASSERT_EQUAL(a(1,0), (unsigned int)1);
    UPP_ASSERT_EQUAL(a(1,1), (unsigned int)2);
    }

//! boost test case for 3x3 Index2DUpperTriangler
UP_TEST( Index2DUpperTriangular_3 )
    {
    Index2DUpperTriangular a(3);
    UPP_ASSERT_EQUAL(a.getNumElements(), (unsigned int)6);
    UPP_ASSERT_EQUAL(a(0,0), (unsigned int)0);
    UPP_ASSERT_EQUAL(a(1,0), (unsigned int)1);
    UPP_ASSERT_EQUAL(a(2,0), (unsigned int)2);
    UPP_ASSERT_EQUAL(a(1,1), (unsigned int)3);
    UPP_ASSERT_EQUAL(a(2,1), (unsigned int)4);
    UPP_ASSERT_EQUAL(a(2,2), (unsigned int)5);
    }

//! boost test case for 20x20 Index2DUpperTriangular
UP_TEST( Index2DUpperTriangular_20 )
    {
    Index2DUpperTriangular a(20);
    UPP_ASSERT_EQUAL(a.getNumElements(), (unsigned int)20*21/2);

    unsigned int cur_idx = 0;
    for (unsigned int i=0; i < 20; i++)
        for (unsigned int j=i; j < 20; j++)
            {
            UPP_ASSERT_EQUAL(a(i,j), cur_idx);
            UPP_ASSERT_EQUAL(a(j,i), cur_idx);
            cur_idx++;
            }
    }
