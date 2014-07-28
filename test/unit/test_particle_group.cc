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


/*! \file particle_group_test.cc
    \brief Unit tests for ParticleGroup
    \ingroup unit_tests
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <iostream>

#include "ParticleData.h"
#include "Initializers.h"
#include "ParticleGroup.h"
#include "RigidBodyGroup.h"

using namespace std;
using namespace boost;

//! Name the boost unit test module
#define BOOST_TEST_MODULE ParticleGroupTests
#include "boost_utf_configure.h"

//! initializes the particle data used by the tests
boost::shared_ptr<SystemDefinition> create_sysdef()
    {
    // initialize a box with 10 particles of 4 groups
    BoxDim box(10.0);
    boost::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(10, box, 4));
    boost::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    // set the types
    // currently, the position is only set on the first 3 particles, intended for use in the total and center of mass
    // tests. Later, other particles will be added to test the new particle data selectors
    {
    ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<int3> h_image(pdata->getImages(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_body(pdata->getBodies(), access_location::host, access_mode::readwrite);

    h_pos.data[0].w = __int_as_scalar(0);
    h_pos.data[0].x = Scalar(0.0); h_pos.data[0].y = Scalar(0.0); h_pos.data[0].z = Scalar(0.0);
    h_image.data[0].x = 0; h_image.data[0].y = 0; h_image.data[0].z = 0;
    h_vel.data[0].w = Scalar(1.0); //mass
    h_body.data[0] = 0;

    h_pos.data[1].w = __int_as_scalar(2);
    h_pos.data[1].x = Scalar(1.0); h_pos.data[1].y = Scalar(2.0); h_pos.data[1].z = Scalar(3.0);
    h_image.data[1].x = 1; h_image.data[1].y = -1; h_image.data[1].z = 2;
    h_vel.data[1].w = Scalar(2.0);
    h_body.data[1] = 0;

    h_pos.data[2].w = __int_as_scalar(0);
    h_pos.data[2].x = Scalar(-1.0); h_pos.data[2].y = Scalar(-2.0); h_pos.data[2].z = Scalar(-3.0);
    h_image.data[2].x = 0; h_image.data[2].y = 0; h_image.data[2].z = 0;
    h_vel.data[2].w = Scalar(5.0);
    h_body.data[2] = 1;

    h_pos.data[3].w = __int_as_scalar(1);
    h_pos.data[3].x = Scalar(-4.0); h_pos.data[3].y = Scalar(-4.0); h_pos.data[3].z = Scalar(-4.0);
    h_body.data[3] = 1;

    h_pos.data[4].w = __int_as_scalar(3);
    h_pos.data[4].x = Scalar(-3.5); h_pos.data[4].y = Scalar(-4.5); h_pos.data[4].z = Scalar(-5.0);

    h_pos.data[5].w = __int_as_scalar(0);
    h_pos.data[5].x = Scalar(-5.0); h_pos.data[5].y = Scalar(-4.5); h_pos.data[5].z = Scalar(-3.5);

    h_pos.data[6].w = __int_as_scalar(1);
    h_pos.data[6].x = Scalar(4.0); h_pos.data[6].y = Scalar(4.0); h_pos.data[6].z = Scalar(4.0);

    h_pos.data[7].w = __int_as_scalar(2);
    h_pos.data[7].x = Scalar(3.5); h_pos.data[7].y = Scalar(4.5); h_pos.data[7].z = Scalar(-5.0);

    h_pos.data[8].w = __int_as_scalar(0);
    h_pos.data[8].x = Scalar(5.0); h_pos.data[8].y = Scalar(4.5); h_pos.data[8].z = Scalar(3.5);

    h_pos.data[9].w = __int_as_scalar(3);
    h_pos.data[9].x = Scalar(5.0); h_pos.data[9].y = Scalar(5.0); h_pos.data[9].z = Scalar(5.0);
    }

    sysdef->getRigidData()->initializeData();
    return sysdef;
    }

//! Checks that ParticleGroup can sucessfully initialize
BOOST_AUTO_TEST_CASE( ParticleGroup_basic_test )
    {
    boost::shared_ptr<SystemDefinition> sysdef = create_sysdef();
    boost::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    // create an empty group
    ParticleGroup a;
    // copy construct it
    ParticleGroup b(a);
    // copy it
    ParticleGroup c;
    c = a;
    }

//! Test copy and equals operators
BOOST_AUTO_TEST_CASE( ParticleGroup_copy_test )
    {
    boost::shared_ptr<SystemDefinition> sysdef = create_sysdef();
    boost::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    // create another particle group of all particles
    boost::shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef, 0, pdata->getN()-1));
    ParticleGroup tags_all(sysdef, selector_all);
    // verify it
    BOOST_CHECK_EQUAL_UINT(tags_all.getNumMembers(), pdata->getN());
    BOOST_CHECK_EQUAL_UINT(tags_all.getIndexArray().getNumElements(), pdata->getN());
    for (unsigned int i = 0; i < pdata->getN(); i++)
        {
        BOOST_CHECK_EQUAL_UINT(tags_all.getMemberTag(i), i);
        BOOST_CHECK_EQUAL_UINT(tags_all.getMemberIndex(i), i);
        BOOST_CHECK(tags_all.isMember(i));
        }

    // copy construct it
    ParticleGroup copy1(tags_all);
    // verify it
    BOOST_CHECK_EQUAL_UINT(copy1.getNumMembers(), pdata->getN());
    BOOST_CHECK_EQUAL_UINT(copy1.getIndexArray().getNumElements(), pdata->getN());
    for (unsigned int i = 0; i < pdata->getN(); i++)
        {
        BOOST_CHECK_EQUAL_UINT(copy1.getMemberTag(i), i);
        BOOST_CHECK_EQUAL_UINT(copy1.getMemberIndex(i), i);
        BOOST_CHECK(copy1.isMember(i));
        }

    // copy it
    ParticleGroup copy2;
    copy2 = copy1;
    // verify it
    BOOST_CHECK_EQUAL_UINT(copy2.getNumMembers(), pdata->getN());
    BOOST_CHECK_EQUAL_UINT(copy2.getIndexArray().getNumElements(), pdata->getN());
    for (unsigned int i = 0; i < pdata->getN(); i++)
        {
        BOOST_CHECK_EQUAL_UINT(copy2.getMemberTag(i), i);
        BOOST_CHECK_EQUAL_UINT(copy2.getMemberIndex(i), i);
        BOOST_CHECK(copy2.isMember(i));
        }
    }

//! Checks that ParticleGroup can sucessfully handle particle resorts
BOOST_AUTO_TEST_CASE( ParticleGroup_sort_test )
    {
    boost::shared_ptr<SystemDefinition> sysdef = create_sysdef();
    boost::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    boost::shared_ptr<ParticleSelector> selector04(new ParticleSelectorTag(sysdef, 0, 4));
    ParticleGroup tags04(sysdef, selector04);
    // verify the initial set
    BOOST_CHECK_EQUAL_UINT(tags04.getNumMembers(), 5);
    BOOST_CHECK_EQUAL_UINT(tags04.getIndexArray().getNumElements(), 5);
    for (unsigned int i = 0; i < 5; i++)
        {
        BOOST_CHECK_EQUAL_UINT(tags04.getMemberTag(i), i);
        BOOST_CHECK_EQUAL_UINT(tags04.getMemberIndex(i), i);
        }

    for (unsigned int i = 0; i < pdata->getN(); i++)
        {
        if (i <= 4)
            BOOST_CHECK(tags04.isMember(i));
        else
            BOOST_CHECK(!tags04.isMember(i));
        }

    // resort the particles
    {
    ArrayHandle<unsigned int> h_tag(pdata->getTags(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_rtag(pdata->getRTags(), access_location::host, access_mode::readwrite);

    // set the types
    h_tag.data[0] = 9;
    h_tag.data[1] = 8;
    h_tag.data[2] = 7;
    h_tag.data[3] = 6;
    h_tag.data[4] = 5;
    h_tag.data[5] = 4;
    h_tag.data[6] = 3;
    h_tag.data[7] = 2;
    h_tag.data[8] = 1;
    h_tag.data[9] = 0;

    h_rtag.data[0] = 9;
    h_rtag.data[1] = 8;
    h_rtag.data[2] = 7;
    h_rtag.data[3] = 6;
    h_rtag.data[4] = 5;
    h_rtag.data[5] = 4;
    h_rtag.data[6] = 3;
    h_rtag.data[7] = 2;
    h_rtag.data[8] = 1;
    h_rtag.data[9] = 0;
    }

    pdata->notifyParticleSort();

    // verify that the group has updated
    BOOST_CHECK_EQUAL_UINT(tags04.getNumMembers(), 5);
    BOOST_CHECK_EQUAL_UINT(tags04.getIndexArray().getNumElements(), 5);
    for (unsigned int i = 0; i < 5; i++)
        {
        BOOST_CHECK_EQUAL_UINT(tags04.getMemberTag(i), i);
        // indices are in sorted order (tags 0-4 are particles 9-5)
        BOOST_CHECK_EQUAL_UINT(tags04.getMemberIndex(i), i + 5);
        }
    {
    ArrayHandle<unsigned int> h_tag(pdata->getTags(), access_location::host, access_mode::readwrite);
    for (unsigned int i = 0; i < pdata->getN(); i++)
        {
        if (h_tag.data[i] <= 4)
            BOOST_CHECK(tags04.isMember(i));
        else
            BOOST_CHECK(!tags04.isMember(i));
        }
    }
    }

//! Checks that ParticleGroup can initialize by particle type
BOOST_AUTO_TEST_CASE( ParticleGroup_type_test )
    {
    boost::shared_ptr<SystemDefinition> sysdef = create_sysdef();
    boost::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    // create a group of type 0 and check it
    boost::shared_ptr<ParticleSelector> selector0(new ParticleSelectorType(sysdef, 0, 0));
    ParticleGroup type0(sysdef, selector0);
    BOOST_REQUIRE_EQUAL_UINT(type0.getNumMembers(), 4);
    BOOST_CHECK_EQUAL_UINT(type0.getIndexArray().getNumElements(), 4);

    BOOST_CHECK_EQUAL_UINT(type0.getMemberTag(0), 0);
    BOOST_CHECK_EQUAL_UINT(type0.getMemberTag(1), 2);
    BOOST_CHECK_EQUAL_UINT(type0.getMemberTag(2), 5);
    BOOST_CHECK_EQUAL_UINT(type0.getMemberTag(3), 8);

    // create a group of type 1 and check it
    boost::shared_ptr<ParticleSelector> selector1(new ParticleSelectorType(sysdef, 1, 1));
    ParticleGroup type1(sysdef, selector1);
    BOOST_REQUIRE_EQUAL_UINT(type1.getNumMembers(), 2);
    BOOST_CHECK_EQUAL_UINT(type1.getIndexArray().getNumElements(), 2);
    BOOST_CHECK_EQUAL_UINT(type1.getMemberTag(0), 3);
    BOOST_CHECK_EQUAL_UINT(type1.getMemberTag(1), 6);

    // create a group of type 2 and check it
    boost::shared_ptr<ParticleSelector> selector2(new ParticleSelectorType(sysdef, 2, 2));
    ParticleGroup type2(sysdef, selector2);
    BOOST_REQUIRE_EQUAL_UINT(type2.getNumMembers(), 2);
    BOOST_CHECK_EQUAL_UINT(type2.getIndexArray().getNumElements(), 2);
    BOOST_CHECK_EQUAL_UINT(type2.getMemberTag(0), 1);
    BOOST_CHECK_EQUAL_UINT(type2.getMemberTag(1), 7);

    // create a group of type 3 and check it
    boost::shared_ptr<ParticleSelector> selector3(new ParticleSelectorType(sysdef, 3, 3));
    ParticleGroup type3(sysdef, selector3);
    BOOST_REQUIRE_EQUAL_UINT(type3.getNumMembers(), 2);
    BOOST_CHECK_EQUAL_UINT(type3.getIndexArray().getNumElements(), 2);
    BOOST_CHECK_EQUAL_UINT(type3.getMemberTag(0), 4);
    BOOST_CHECK_EQUAL_UINT(type3.getMemberTag(1), 9);

    // create a group of all types and check it
    boost::shared_ptr<ParticleSelector> selector_all(new ParticleSelectorType(sysdef, 0, 3));
    ParticleGroup alltypes(sysdef, selector_all);
    BOOST_REQUIRE_EQUAL_UINT(alltypes.getNumMembers(), 10);
    BOOST_CHECK_EQUAL_UINT(alltypes.getIndexArray().getNumElements(), 10);
    for (unsigned int i = 0; i < 10; i++)
        BOOST_CHECK_EQUAL_UINT(alltypes.getMemberTag(i), i);
    }

//! Checks that ParticleGroup can initialize to the empty set
BOOST_AUTO_TEST_CASE( ParticleGroup_empty_test )
    {
    boost::shared_ptr<SystemDefinition> sysdef = create_sysdef();
    boost::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    // create a group of type 100 and check it
    boost::shared_ptr<ParticleSelector> selector100(new ParticleSelectorType(sysdef, 100, 100));
    ParticleGroup empty(sysdef, selector100);
    BOOST_REQUIRE_EQUAL_UINT(empty.getNumMembers(), 0);
    BOOST_CHECK_EQUAL_UINT(empty.getIndexArray().getNumElements(), 0);
    }

//! Checks that ParticleGroup can initialize by particle body
BOOST_AUTO_TEST_CASE( ParticleGroup_body_test )
    {
    boost::shared_ptr<SystemDefinition> sysdef = create_sysdef();
    boost::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    // create a group of rigid bodies and check it
    boost::shared_ptr<ParticleSelector> selector_body_true(new ParticleSelectorRigid(sysdef, true));
    ParticleGroup type_true(sysdef, selector_body_true);
    BOOST_REQUIRE_EQUAL_UINT(type_true.getNumMembers(), 4);
    BOOST_CHECK_EQUAL_UINT(type_true.getMemberTag(0), 0);
    BOOST_CHECK_EQUAL_UINT(type_true.getMemberTag(1), 1);
    BOOST_CHECK_EQUAL_UINT(type_true.getMemberTag(2), 2);
    BOOST_CHECK_EQUAL_UINT(type_true.getMemberTag(3), 3);

    // create a group of non rigid particles and check it
    boost::shared_ptr<ParticleSelector> selector_body_false(new ParticleSelectorRigid(sysdef, false));
    ParticleGroup type_false(sysdef, selector_body_false);
    BOOST_REQUIRE_EQUAL_UINT(type_false.getNumMembers(), 6);
    BOOST_CHECK_EQUAL_UINT(type_false.getMemberTag(0), 4);
    BOOST_CHECK_EQUAL_UINT(type_false.getMemberTag(1), 5);
    BOOST_CHECK_EQUAL_UINT(type_false.getMemberTag(2), 6);
    BOOST_CHECK_EQUAL_UINT(type_false.getMemberTag(3), 7);
    BOOST_CHECK_EQUAL_UINT(type_false.getMemberTag(4), 8);
    BOOST_CHECK_EQUAL_UINT(type_false.getMemberTag(5), 9);
    }

//! Checks that RigidBodyGroup can successfully initialize when given all bodies
BOOST_AUTO_TEST_CASE( RigidBodyGroup_all_test )
    {
    boost::shared_ptr<SystemDefinition> sysdef = create_sysdef();
    boost::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    // create a group of rigid bodies and check it
    boost::shared_ptr<ParticleSelector> selector_body_true(new ParticleSelectorRigid(sysdef, true));
    boost::shared_ptr<ParticleGroup> body_true(new ParticleGroup(sysdef, selector_body_true));

    // create a rigid body group
    RigidBodyGroup body_group(sysdef, body_true);
    BOOST_CHECK_EQUAL_UINT(body_group.getNumMembers(), 2);
    BOOST_CHECK(body_group.isMember(0));
    BOOST_CHECK(body_group.isMember(1));

    ArrayHandle<unsigned int> h_member_idx(body_group.getIndexArray(), access_location::host, access_mode::read);
    BOOST_CHECK_EQUAL_UINT(body_group.getIndexArray().getNumElements(), 2);
    BOOST_CHECK_EQUAL_UINT(h_member_idx.data[0], 0);
    BOOST_CHECK_EQUAL_UINT(h_member_idx.data[1], 1);
    }

//! Checks that RigidBodyGroup can successfully initialize when given all bodies
BOOST_AUTO_TEST_CASE( RigidBodyGroup_one_test )
    {
    boost::shared_ptr<SystemDefinition> sysdef = create_sysdef();
    boost::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    // create a group of rigid bodies and check it
    boost::shared_ptr<ParticleSelector> selector_body(new ParticleSelectorTag(sysdef, 2, 3));
    boost::shared_ptr<ParticleGroup> body_particles(new ParticleGroup(sysdef, selector_body));

    // create a rigid body group
    RigidBodyGroup body_group(sysdef, body_particles);
    BOOST_CHECK_EQUAL_UINT(body_group.getNumMembers(), 1);
    BOOST_CHECK(!body_group.isMember(0));
    BOOST_CHECK(body_group.isMember(1));

    ArrayHandle<unsigned int> h_member_idx(body_group.getIndexArray(), access_location::host, access_mode::read);
    BOOST_CHECK_EQUAL_UINT(body_group.getIndexArray().getNumElements(), 1);
    BOOST_CHECK_EQUAL_UINT(h_member_idx.data[0], 1);
    }


//! Checks that ParticleGroup can initialize by particle tag
BOOST_AUTO_TEST_CASE( ParticleGroup_tag_test )
    {
    boost::shared_ptr<SystemDefinition> sysdef = create_sysdef();
    boost::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    // create a group of tags 0-4 and check it
    boost::shared_ptr<ParticleSelector> selector04(new ParticleSelectorTag(sysdef, 0, 4));
    ParticleGroup tags05(sysdef, selector04);
    BOOST_REQUIRE_EQUAL_UINT(tags05.getNumMembers(), 5);
    BOOST_CHECK_EQUAL_UINT(tags05.getIndexArray().getNumElements(), 5);
    BOOST_CHECK_EQUAL_UINT(tags05.getMemberTag(0), 0);
    BOOST_CHECK_EQUAL_UINT(tags05.getMemberTag(1), 1);
    BOOST_CHECK_EQUAL_UINT(tags05.getMemberTag(2), 2);
    BOOST_CHECK_EQUAL_UINT(tags05.getMemberTag(3), 3);
    BOOST_CHECK_EQUAL_UINT(tags05.getMemberTag(4), 4);

    // create a group of tags 5-9 and check it
    boost::shared_ptr<ParticleSelector> selector59(new ParticleSelectorTag(sysdef, 5, 9));
    ParticleGroup tags59(sysdef, selector59);
    BOOST_REQUIRE_EQUAL_UINT(tags59.getNumMembers(), 5);
    BOOST_CHECK_EQUAL_UINT(tags59.getIndexArray().getNumElements(), 5);
    BOOST_CHECK_EQUAL_UINT(tags59.getMemberTag(0), 5);
    BOOST_CHECK_EQUAL_UINT(tags59.getMemberTag(1), 6);
    BOOST_CHECK_EQUAL_UINT(tags59.getMemberTag(2), 7);
    BOOST_CHECK_EQUAL_UINT(tags59.getMemberTag(3), 8);
    BOOST_CHECK_EQUAL_UINT(tags59.getMemberTag(4), 9);
    }

//! Checks that ParticleGroup can initialize by cuboid
BOOST_AUTO_TEST_CASE( ParticleGroup_cuboid_test )
    {
    boost::shared_ptr<SystemDefinition> sysdef = create_sysdef();
    boost::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    // create a group containing only particle 0
    boost::shared_ptr<ParticleSelector> selector0(new ParticleSelectorCuboid(sysdef,
                                                                      make_scalar3(-0.5, -0.5, -0.5),
                                                                      make_scalar3( 0.5,  0.5,  0.5)));
    ParticleGroup tags0(sysdef, selector0);
    BOOST_REQUIRE_EQUAL_UINT(tags0.getNumMembers(), 1);
    BOOST_CHECK_EQUAL_UINT(tags0.getIndexArray().getNumElements(), 1);
    BOOST_CHECK_EQUAL_UINT(tags0.getMemberTag(0), 0);

    // create a group containing particles 0 and 1
    boost::shared_ptr<ParticleSelector> selector1(new ParticleSelectorCuboid(sysdef,
                                                                      make_scalar3(-0.5, -0.5, -0.5),
                                                                      make_scalar3( 1.5,  2.5,  3.5)));
    ParticleGroup tags1(sysdef, selector1);
    BOOST_REQUIRE_EQUAL_UINT(tags1.getNumMembers(), 2);
    BOOST_CHECK_EQUAL_UINT(tags1.getIndexArray().getNumElements(), 2);
    BOOST_CHECK_EQUAL_UINT(tags1.getMemberTag(0), 0);
    BOOST_CHECK_EQUAL_UINT(tags1.getMemberTag(1), 1);

    // create a group containing particles 0, 1 and 2
    boost::shared_ptr<ParticleSelector> selector2(new ParticleSelectorCuboid(sysdef,
                                                                      make_scalar3(-1.5, -2.5, -3.5),
                                                                      make_scalar3( 1.5,  2.5,  3.5)));
    ParticleGroup tags2(sysdef, selector2);
    BOOST_REQUIRE_EQUAL_UINT(tags2.getNumMembers(), 3);
    BOOST_CHECK_EQUAL_UINT(tags2.getIndexArray().getNumElements(), 3);
    BOOST_CHECK_EQUAL_UINT(tags2.getMemberTag(0), 0);
    BOOST_CHECK_EQUAL_UINT(tags2.getMemberTag(1), 1);
    BOOST_CHECK_EQUAL_UINT(tags2.getMemberTag(2), 2);
    }

//! Checks that the ParticleGroup boolean operation work correctly
BOOST_AUTO_TEST_CASE( ParticleGroup_boolean_tests)
    {
    boost::shared_ptr<SystemDefinition> sysdef = create_sysdef();
    boost::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    // create a group of tags 0-4
    boost::shared_ptr<ParticleSelector> selector04(new ParticleSelectorTag(sysdef, 0, 4));
    boost::shared_ptr<ParticleGroup> tags04(new ParticleGroup(sysdef, selector04));

    // create a group of type 0
    boost::shared_ptr<ParticleSelector> selector0(new ParticleSelectorType(sysdef, 0, 0));
    boost::shared_ptr<ParticleGroup> type0(new ParticleGroup(sysdef, selector0));

    // make a union of the two groups and check it
    boost::shared_ptr<ParticleGroup> union_group = ParticleGroup::groupUnion(type0, tags04);
    BOOST_REQUIRE_EQUAL_UINT(union_group->getNumMembers(), 7);
    BOOST_CHECK_EQUAL_UINT(union_group->getIndexArray().getNumElements(), 7);
    BOOST_CHECK_EQUAL_UINT(union_group->getMemberTag(0), 0);
    BOOST_CHECK_EQUAL_UINT(union_group->getMemberTag(1), 1);
    BOOST_CHECK_EQUAL_UINT(union_group->getMemberTag(2), 2);
    BOOST_CHECK_EQUAL_UINT(union_group->getMemberTag(3), 3);
    BOOST_CHECK_EQUAL_UINT(union_group->getMemberTag(4), 4);
    BOOST_CHECK_EQUAL_UINT(union_group->getMemberTag(5), 5);
    BOOST_CHECK_EQUAL_UINT(union_group->getMemberTag(6), 8);

    // make a intersection group and test it
    boost::shared_ptr<ParticleGroup> intersection_group = ParticleGroup::groupIntersection(type0, tags04);
    BOOST_REQUIRE_EQUAL_UINT(intersection_group->getNumMembers(), 2);
    BOOST_CHECK_EQUAL_UINT(intersection_group->getIndexArray().getNumElements(), 2);
    BOOST_CHECK_EQUAL_UINT(intersection_group->getMemberTag(0), 0);
    BOOST_CHECK_EQUAL_UINT(intersection_group->getMemberTag(1), 2);
    }

//! Checks that the ParticleGroup::getTotalMass works correctly
BOOST_AUTO_TEST_CASE( ParticleGroup_total_mass_tests)
    {
    boost::shared_ptr<SystemDefinition> sysdef = create_sysdef();
    boost::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    ParticleGroup group1(sysdef, boost::shared_ptr<ParticleSelector>(new ParticleSelectorTag(sysdef, 0, 0)));
    MY_BOOST_CHECK_CLOSE(group1.getTotalMass(), 1.0, tol);

    ParticleGroup group2(sysdef, boost::shared_ptr<ParticleSelector>(new ParticleSelectorTag(sysdef, 0, 1)));
    MY_BOOST_CHECK_CLOSE(group2.getTotalMass(), 3.0, tol);

    ParticleGroup group3(sysdef, boost::shared_ptr<ParticleSelector>(new ParticleSelectorTag(sysdef, 0, 2)));
    MY_BOOST_CHECK_CLOSE(group3.getTotalMass(), 8.0, tol);
    }

//! Checks that the ParticleGroup::getCenterOfMass works correctly
BOOST_AUTO_TEST_CASE( ParticleGroup_center_of_mass_tests)
    {
    boost::shared_ptr<SystemDefinition> sysdef = create_sysdef();
    boost::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    Scalar3 com;
    ParticleGroup group1(sysdef, boost::shared_ptr<ParticleSelector>(new ParticleSelectorTag(sysdef, 0, 0)));
    com = group1.getCenterOfMass();
    MY_BOOST_CHECK_SMALL(com.x, tol_small);
    MY_BOOST_CHECK_SMALL(com.y, tol_small);
    MY_BOOST_CHECK_SMALL(com.z, tol_small);

    ParticleGroup group2(sysdef, boost::shared_ptr<ParticleSelector>(new ParticleSelectorTag(sysdef, 0, 1)));
    com = group2.getCenterOfMass();
    MY_BOOST_CHECK_CLOSE(com.x, 7.3333333333, tol);
    MY_BOOST_CHECK_CLOSE(com.y, -5.3333333333, tol);
    MY_BOOST_CHECK_CLOSE(com.z, 15.333333333, tol);

    ParticleGroup group3(sysdef, boost::shared_ptr<ParticleSelector>(new ParticleSelectorTag(sysdef, 0, 2)));
    com = group3.getCenterOfMass();
    MY_BOOST_CHECK_CLOSE(com.x, 2.125, tol);
    MY_BOOST_CHECK_CLOSE(com.y, -3.25, tol);
    MY_BOOST_CHECK_CLOSE(com.z, 3.875, tol);
    }
