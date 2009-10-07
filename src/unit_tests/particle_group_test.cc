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

/*! \file particle_group_test.cc
    \brief Unit tests for ParticleGroup
    \ingroup unit_tests
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <iostream>

//! Name the boost unit test module
#define BOOST_TEST_MODULE ParticleGroupTests
#include "boost_utf_configure.h"

#include "ParticleData.h"
#include "Initializers.h"
#include "ParticleGroup.h"

using namespace std;
using namespace boost;

//! Need a simple define for requireing two values which are unsigned
#define MY_BOOST_REQUIRE_EQUAL(a,b) BOOST_REQUIRE_EQUAL(a,(unsigned int)(b))
//! Need a simple define for checking two values which are unsigned
#define MY_BOOST_CHECK_EQUAL(a,b) BOOST_CHECK_EQUAL(a,(unsigned int)(b))

//! initializes the particle data used by the tests
shared_ptr<ParticleData> create_pdata()
    {
    // initialize a box with 10 particles of 4 groups
    BoxDim box(10.0);
    shared_ptr<ParticleData> pdata(new ParticleData(10, box, 4));
    ParticleDataArrays arrays = pdata->acquireReadWrite();
    
    // set the types
    arrays.type[0] = 0;
    arrays.type[1] = 2;
    arrays.type[2] = 0;
    arrays.type[3] = 1;
    arrays.type[4] = 3;
    arrays.type[5] = 0;
    arrays.type[6] = 1;
    arrays.type[7] = 2;
    arrays.type[8] = 0;
    arrays.type[9] = 3;
    
    pdata->release();
    return pdata;
    }

//! Checks that ParticleGroup can initialize by particle type
BOOST_AUTO_TEST_CASE( ParticleGroup_type_test )
    {
    shared_ptr<ParticleData> pdata = create_pdata();
    
    // create a group of type 0 and check it
    ParticleGroup type0(pdata, ParticleGroup::type, 0, 0);
    MY_BOOST_REQUIRE_EQUAL(type0.getNumMembers(), 4);
    MY_BOOST_CHECK_EQUAL(type0.getMemberTag(0), 0);
    MY_BOOST_CHECK_EQUAL(type0.getMemberTag(1), 2);
    MY_BOOST_CHECK_EQUAL(type0.getMemberTag(2), 5);
    MY_BOOST_CHECK_EQUAL(type0.getMemberTag(3), 8);
    
    // create a group of type 1 and check it
    ParticleGroup type1(pdata, ParticleGroup::type, 1, 1);
    MY_BOOST_REQUIRE_EQUAL(type1.getNumMembers(), 2);
    MY_BOOST_CHECK_EQUAL(type1.getMemberTag(0), 3);
    MY_BOOST_CHECK_EQUAL(type1.getMemberTag(1), 6);
    
    // create a group of type 2 and check it
    ParticleGroup type2(pdata, ParticleGroup::type, 2, 2);
    MY_BOOST_REQUIRE_EQUAL(type2.getNumMembers(), 2);
    MY_BOOST_CHECK_EQUAL(type2.getMemberTag(0), 1);
    MY_BOOST_CHECK_EQUAL(type2.getMemberTag(1), 7);
    
    // create a group of type 3 and check it
    ParticleGroup type3(pdata, ParticleGroup::type, 3, 3);
    MY_BOOST_REQUIRE_EQUAL(type3.getNumMembers(), 2);
    MY_BOOST_CHECK_EQUAL(type3.getMemberTag(0), 4);
    MY_BOOST_CHECK_EQUAL(type3.getMemberTag(1), 9);
    
    // create a group of all types and check it
    ParticleGroup alltypes(pdata, ParticleGroup::type, 0, 3);
    MY_BOOST_REQUIRE_EQUAL(alltypes.getNumMembers(), 10);
    for (unsigned int i = 0; i < 10; i++)
        MY_BOOST_CHECK_EQUAL(alltypes.getMemberTag(i), i);
    }

//! Checks that ParticleGroup can initialize by particle tag
BOOST_AUTO_TEST_CASE( ParticleGroup_tag_test )
    {
    shared_ptr<ParticleData> pdata = create_pdata();
    
    // create a group of tags 0-4 and check it
    ParticleGroup tags05(pdata, ParticleGroup::tag, 0, 4);
    MY_BOOST_REQUIRE_EQUAL(tags05.getNumMembers(), 5);
    MY_BOOST_CHECK_EQUAL(tags05.getMemberTag(0), 0);
    MY_BOOST_CHECK_EQUAL(tags05.getMemberTag(1), 1);
    MY_BOOST_CHECK_EQUAL(tags05.getMemberTag(2), 2);
    MY_BOOST_CHECK_EQUAL(tags05.getMemberTag(3), 3);
    MY_BOOST_CHECK_EQUAL(tags05.getMemberTag(4), 4);
    
    // create a group of tags 5-9 and check it
    ParticleGroup tags59(pdata, ParticleGroup::tag, 5, 9);
    MY_BOOST_REQUIRE_EQUAL(tags59.getNumMembers(), 5);
    MY_BOOST_CHECK_EQUAL(tags59.getMemberTag(0), 5);
    MY_BOOST_CHECK_EQUAL(tags59.getMemberTag(1), 6);
    MY_BOOST_CHECK_EQUAL(tags59.getMemberTag(2), 7);
    MY_BOOST_CHECK_EQUAL(tags59.getMemberTag(3), 8);
    MY_BOOST_CHECK_EQUAL(tags59.getMemberTag(4), 9);
    }

//! Checks that the ParticleGroup boolean operation work correctly
BOOST_AUTO_TEST_CASE( ParticleGroup_boolean_tests)
    {
    shared_ptr<ParticleData> pdata = create_pdata();
    
    // create a group of tags 0-4
    boost::shared_ptr<ParticleGroup> tags04(new ParticleGroup(pdata, ParticleGroup::tag, 0, 4));
    
    // create a group of type 0
    boost::shared_ptr<ParticleGroup> type0(new ParticleGroup(pdata, ParticleGroup::type, 0, 0));
    
    // make a union of the two groups and check it
    boost::shared_ptr<ParticleGroup> union_group = ParticleGroup::groupUnion(type0, tags04);
    MY_BOOST_REQUIRE_EQUAL(union_group->getNumMembers(), 7);
    MY_BOOST_CHECK_EQUAL(union_group->getMemberTag(0), 0);
    MY_BOOST_CHECK_EQUAL(union_group->getMemberTag(1), 1);
    MY_BOOST_CHECK_EQUAL(union_group->getMemberTag(2), 2);
    MY_BOOST_CHECK_EQUAL(union_group->getMemberTag(3), 3);
    MY_BOOST_CHECK_EQUAL(union_group->getMemberTag(4), 4);
    MY_BOOST_CHECK_EQUAL(union_group->getMemberTag(5), 5);
    MY_BOOST_CHECK_EQUAL(union_group->getMemberTag(6), 8);
    
    // make a intersection group and test it
    boost::shared_ptr<ParticleGroup> intersection_group = ParticleGroup::groupIntersection(type0, tags04);
    MY_BOOST_REQUIRE_EQUAL(intersection_group->getNumMembers(), 2);
    MY_BOOST_CHECK_EQUAL(intersection_group->getMemberTag(0), 0);
    MY_BOOST_CHECK_EQUAL(intersection_group->getMemberTag(1), 2);
    }

