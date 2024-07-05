// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// this include is necessary to get MPI included before anything else to support intel MPI
#include "hoomd/ExecutionConfiguration.h"

/*! \file particle_group_test.cc
    \brief Unit tests for ParticleGroup
    \ingroup unit_tests
*/

#include <iostream>

#include "hoomd/Initializers.h"
#include "hoomd/ParticleData.h"
#include "hoomd/ParticleGroup.h"

using namespace std;

#include "upp11_config.h"

HOOMD_UP_MAIN();

//! initializes the particle data used by the tests
std::shared_ptr<SystemDefinition> create_sysdef()
    {
    // initialize a box with 10 particles of 4 groups
    BoxDim box(10.0);
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(10, box, 4));
    std::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

        // set the types
        // currently, the position is only set on the first 3 particles, intended for use in the
        // total and center of mass tests. Later, other particles will be added to test the new
        // particle data selectors
        {
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(),
                                   access_location::host,
                                   access_mode::readwrite);
        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::readwrite);
        ArrayHandle<int3> h_image(pdata->getImages(),
                                  access_location::host,
                                  access_mode::readwrite);
        ArrayHandle<unsigned int> h_body(pdata->getBodies(),
                                         access_location::host,
                                         access_mode::readwrite);

        h_pos.data[0].w = __int_as_scalar(0);
        h_pos.data[0].x = Scalar(0.0);
        h_pos.data[0].y = Scalar(0.0);
        h_pos.data[0].z = Scalar(0.0);
        h_image.data[0].x = 0;
        h_image.data[0].y = 0;
        h_image.data[0].z = 0;
        h_vel.data[0].w = Scalar(1.0); // mass
        h_body.data[0] = 0;

        h_pos.data[1].w = __int_as_scalar(2);
        h_pos.data[1].x = Scalar(1.0);
        h_pos.data[1].y = Scalar(2.0);
        h_pos.data[1].z = Scalar(3.0);
        h_image.data[1].x = 1;
        h_image.data[1].y = -1;
        h_image.data[1].z = 2;
        h_vel.data[1].w = Scalar(2.0);
        h_body.data[1] = 0;

        h_pos.data[2].w = __int_as_scalar(0);
        h_pos.data[2].x = Scalar(-1.0);
        h_pos.data[2].y = Scalar(-2.0);
        h_pos.data[2].z = Scalar(-3.0);
        h_image.data[2].x = 0;
        h_image.data[2].y = 0;
        h_image.data[2].z = 0;
        h_vel.data[2].w = Scalar(5.0);
        h_body.data[2] = 1;

        h_pos.data[3].w = __int_as_scalar(1);
        h_pos.data[3].x = Scalar(-4.0);
        h_pos.data[3].y = Scalar(-4.0);
        h_pos.data[3].z = Scalar(-4.0);
        h_body.data[3] = 1;

        h_pos.data[4].w = __int_as_scalar(3);
        h_pos.data[4].x = Scalar(-3.5);
        h_pos.data[4].y = Scalar(-4.5);
        h_pos.data[4].z = Scalar(-5.0);

        h_pos.data[5].w = __int_as_scalar(0);
        h_pos.data[5].x = Scalar(-5.0);
        h_pos.data[5].y = Scalar(-4.5);
        h_pos.data[5].z = Scalar(-3.5);

        h_pos.data[6].w = __int_as_scalar(1);
        h_pos.data[6].x = Scalar(4.0);
        h_pos.data[6].y = Scalar(4.0);
        h_pos.data[6].z = Scalar(4.0);

        h_pos.data[7].w = __int_as_scalar(2);
        h_pos.data[7].x = Scalar(3.5);
        h_pos.data[7].y = Scalar(4.5);
        h_pos.data[7].z = Scalar(-5.0);

        h_pos.data[8].w = __int_as_scalar(0);
        h_pos.data[8].x = Scalar(5.0);
        h_pos.data[8].y = Scalar(4.5);
        h_pos.data[8].z = Scalar(3.5);

        h_pos.data[9].w = __int_as_scalar(3);
        h_pos.data[9].x = Scalar(5.0);
        h_pos.data[9].y = Scalar(5.0);
        h_pos.data[9].z = Scalar(5.0);
        }

    return sysdef;
    }

//! Checks that ParticleGroup can successfully initialize
UP_TEST(ParticleGroup_basic_test)
    {
    std::shared_ptr<SystemDefinition> sysdef = create_sysdef();
    std::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    // create an empty group
    ParticleGroup a;
    // copy construct it
    ParticleGroup b(a);
    // copy it
    ParticleGroup c;
    c = a;
    }

//! Test copy and equals operators
UP_TEST(ParticleGroup_copy_test)
    {
    std::shared_ptr<SystemDefinition> sysdef = create_sysdef();
    std::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    // create another particle group of all particles
    std::shared_ptr<ParticleFilter> selector_all(new ParticleFilterAll());
    ParticleGroup tags_all(sysdef, selector_all);
    // verify it
    CHECK_EQUAL_UINT(tags_all.getNumMembers(), pdata->getN());
    CHECK_EQUAL_UINT(tags_all.getIndexArray().getNumElements(), pdata->getN());
    for (unsigned int i = 0; i < pdata->getN(); i++)
        {
        CHECK_EQUAL_UINT(tags_all.getMemberTag(i), i);
        CHECK_EQUAL_UINT(tags_all.getMemberIndex(i), i);
        UP_ASSERT(tags_all.isMember(i));
        }

    // copy construct it
    ParticleGroup copy1(tags_all);
    // verify it
    CHECK_EQUAL_UINT(copy1.getNumMembers(), pdata->getN());
    CHECK_EQUAL_UINT(copy1.getIndexArray().getNumElements(), pdata->getN());
    for (unsigned int i = 0; i < pdata->getN(); i++)
        {
        CHECK_EQUAL_UINT(copy1.getMemberTag(i), i);
        CHECK_EQUAL_UINT(copy1.getMemberIndex(i), i);
        UP_ASSERT(copy1.isMember(i));
        }

    // copy it
    ParticleGroup copy2;
    copy2 = copy1;
    // verify it
    CHECK_EQUAL_UINT(copy2.getNumMembers(), pdata->getN());
    CHECK_EQUAL_UINT(copy2.getIndexArray().getNumElements(), pdata->getN());
    for (unsigned int i = 0; i < pdata->getN(); i++)
        {
        CHECK_EQUAL_UINT(copy2.getMemberTag(i), i);
        CHECK_EQUAL_UINT(copy2.getMemberIndex(i), i);
        UP_ASSERT(copy2.isMember(i));
        }
    }

//! Checks that ParticleGroup can successfully handle particle resorts
UP_TEST(ParticleGroup_sort_test)
    {
    std::shared_ptr<SystemDefinition> sysdef = create_sysdef();
    std::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    std::shared_ptr<ParticleFilter> selector04(
        new ParticleFilterTags(std::vector<unsigned int>({0, 1, 2, 3, 4})));
    ParticleGroup tags04(sysdef, selector04);
    // verify the initial set
    CHECK_EQUAL_UINT(tags04.getNumMembers(), 5);
    CHECK_EQUAL_UINT(tags04.getIndexArray().getNumElements(), 5);
    for (unsigned int i = 0; i < 5; i++)
        {
        CHECK_EQUAL_UINT(tags04.getMemberTag(i), i);
        CHECK_EQUAL_UINT(tags04.getMemberIndex(i), i);
        }

    for (unsigned int i = 0; i < pdata->getN(); i++)
        {
        if (i <= 4)
            UP_ASSERT(tags04.isMember(i));
        else
            UP_ASSERT(!tags04.isMember(i));
        }

        // resort the particles
        {
        ArrayHandle<unsigned int> h_tag(pdata->getTags(),
                                        access_location::host,
                                        access_mode::readwrite);
        ArrayHandle<unsigned int> h_rtag(pdata->getRTags(),
                                         access_location::host,
                                         access_mode::readwrite);

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
    CHECK_EQUAL_UINT(tags04.getNumMembers(), 5);
    CHECK_EQUAL_UINT(tags04.getIndexArray().getNumElements(), 5);
    for (unsigned int i = 0; i < 5; i++)
        {
        CHECK_EQUAL_UINT(tags04.getMemberTag(i), i);
        // indices are in sorted order (tags 0-4 are particles 9-5)
        CHECK_EQUAL_UINT(tags04.getMemberIndex(i), i + 5);
        }
        {
        ArrayHandle<unsigned int> h_tag(pdata->getTags(),
                                        access_location::host,
                                        access_mode::readwrite);
        for (unsigned int i = 0; i < pdata->getN(); i++)
            {
            if (h_tag.data[i] <= 4)
                UP_ASSERT(tags04.isMember(i));
            else
                UP_ASSERT(!tags04.isMember(i));
            }
        }
    }

//! Checks that ParticleGroup can initialize by particle type
UP_TEST(ParticleGroup_type_test)
    {
    std::shared_ptr<SystemDefinition> sysdef = create_sysdef();
    std::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    // create a group of type 0 and check it
    std::shared_ptr<ParticleFilter> selector0(new ParticleFilterType(0, 0));
    ParticleGroup type0(sysdef, selector0);
    CHECK_EQUAL_UINT(type0.getNumMembers(), 4);
    CHECK_EQUAL_UINT(type0.getIndexArray().getNumElements(), 4);

    CHECK_EQUAL_UINT(type0.getMemberTag(0), 0);
    CHECK_EQUAL_UINT(type0.getMemberTag(1), 2);
    CHECK_EQUAL_UINT(type0.getMemberTag(2), 5);
    CHECK_EQUAL_UINT(type0.getMemberTag(3), 8);

    // create a group of type 1 and check it
    std::shared_ptr<ParticleFilter> selector1(new ParticleFilterType(1, 1));
    ParticleGroup type1(sysdef, selector1);
    CHECK_EQUAL_UINT(type1.getNumMembers(), 2);
    CHECK_EQUAL_UINT(type1.getIndexArray().getNumElements(), 2);
    CHECK_EQUAL_UINT(type1.getMemberTag(0), 3);
    CHECK_EQUAL_UINT(type1.getMemberTag(1), 6);

    // create a group of type 2 and check it
    std::shared_ptr<ParticleFilter> selector2(new ParticleFilterType(2, 2));
    ParticleGroup type2(sysdef, selector2);
    CHECK_EQUAL_UINT(type2.getNumMembers(), 2);
    CHECK_EQUAL_UINT(type2.getIndexArray().getNumElements(), 2);
    CHECK_EQUAL_UINT(type2.getMemberTag(0), 1);
    CHECK_EQUAL_UINT(type2.getMemberTag(1), 7);

    // create a group of type 3 and check it
    std::shared_ptr<ParticleFilter> selector3(new ParticleFilterType(3, 3));
    ParticleGroup type3(sysdef, selector3);
    CHECK_EQUAL_UINT(type3.getNumMembers(), 2);
    CHECK_EQUAL_UINT(type3.getIndexArray().getNumElements(), 2);
    CHECK_EQUAL_UINT(type3.getMemberTag(0), 4);
    CHECK_EQUAL_UINT(type3.getMemberTag(1), 9);

    // create a group of all types and check it
    std::shared_ptr<ParticleFilter> selector_all(new ParticleFilterType(0, 3));
    ParticleGroup alltypes(sysdef, selector_all);
    CHECK_EQUAL_UINT(alltypes.getNumMembers(), 10);
    CHECK_EQUAL_UINT(alltypes.getIndexArray().getNumElements(), 10);
    for (unsigned int i = 0; i < 10; i++)
        CHECK_EQUAL_UINT(alltypes.getMemberTag(i), i);
    }

//! Checks that ParticleGroup can initialize to the empty set
UP_TEST(ParticleGroup_empty_test)
    {
    std::shared_ptr<SystemDefinition> sysdef = create_sysdef();
    std::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    // create a group of type 100 and check it
    std::shared_ptr<ParticleFilter> selector100(new ParticleFilterType(100, 100));
    ParticleGroup empty(sysdef, selector100);
    CHECK_EQUAL_UINT(empty.getNumMembers(), 0);
    CHECK_EQUAL_UINT(empty.getIndexArray().getNumElements(), 0);
    }

//! Checks that ParticleGroup can initialize by particle body
UP_TEST(ParticleGroup_body_test)
    {
    std::shared_ptr<SystemDefinition> sysdef = create_sysdef();
    std::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    // create a group of rigid bodies and check it
    std::shared_ptr<ParticleFilter> selector_body_true(new ParticleFilterRigid(true));
    ParticleGroup type_true(sysdef, selector_body_true);
    CHECK_EQUAL_UINT(type_true.getNumMembers(), 4);
    CHECK_EQUAL_UINT(type_true.getMemberTag(0), 0);
    CHECK_EQUAL_UINT(type_true.getMemberTag(1), 1);
    CHECK_EQUAL_UINT(type_true.getMemberTag(2), 2);
    CHECK_EQUAL_UINT(type_true.getMemberTag(3), 3);

    // create a group of non rigid particles and check it
    std::shared_ptr<ParticleFilter> selector_body_false(new ParticleFilterRigid(false));
    ParticleGroup type_false(sysdef, selector_body_false);
    CHECK_EQUAL_UINT(type_false.getNumMembers(), 6);
    CHECK_EQUAL_UINT(type_false.getMemberTag(0), 4);
    CHECK_EQUAL_UINT(type_false.getMemberTag(1), 5);
    CHECK_EQUAL_UINT(type_false.getMemberTag(2), 6);
    CHECK_EQUAL_UINT(type_false.getMemberTag(3), 7);
    CHECK_EQUAL_UINT(type_false.getMemberTag(4), 8);
    CHECK_EQUAL_UINT(type_false.getMemberTag(5), 9);
    }

//! Checks that ParticleGroup can initialize by particle tag
UP_TEST(ParticleGroup_tag_test)
    {
    std::shared_ptr<SystemDefinition> sysdef = create_sysdef();
    std::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    // create a group of tags 0-4 and check it
    std::shared_ptr<ParticleFilter> selector04(
        new ParticleFilterTags(std::vector<unsigned int>({0, 1, 2, 3, 4})));
    ParticleGroup tags05(sysdef, selector04);
    CHECK_EQUAL_UINT(tags05.getNumMembers(), 5);
    CHECK_EQUAL_UINT(tags05.getIndexArray().getNumElements(), 5);
    CHECK_EQUAL_UINT(tags05.getMemberTag(0), 0);
    CHECK_EQUAL_UINT(tags05.getMemberTag(1), 1);
    CHECK_EQUAL_UINT(tags05.getMemberTag(2), 2);
    CHECK_EQUAL_UINT(tags05.getMemberTag(3), 3);
    CHECK_EQUAL_UINT(tags05.getMemberTag(4), 4);

    // create a group of tags 5-9 and check it
    std::shared_ptr<ParticleFilter> selector59(
        new ParticleFilterTags(std::vector<unsigned int>({5, 6, 7, 8, 9})));
    ParticleGroup tags59(sysdef, selector59);
    CHECK_EQUAL_UINT(tags59.getNumMembers(), 5);
    CHECK_EQUAL_UINT(tags59.getIndexArray().getNumElements(), 5);
    CHECK_EQUAL_UINT(tags59.getMemberTag(0), 5);
    CHECK_EQUAL_UINT(tags59.getMemberTag(1), 6);
    CHECK_EQUAL_UINT(tags59.getMemberTag(2), 7);
    CHECK_EQUAL_UINT(tags59.getMemberTag(3), 8);
    CHECK_EQUAL_UINT(tags59.getMemberTag(4), 9);
    }

//! Checks that the ParticleGroup boolean operation work correctly
UP_TEST(ParticleGroup_boolean_tests)
    {
    std::shared_ptr<SystemDefinition> sysdef = create_sysdef();
    std::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    // create a group of tags 0-4
    std::shared_ptr<ParticleFilter> selector04(
        new ParticleFilterTags(std::vector<unsigned int>({0, 1, 2, 3, 4})));
    std::shared_ptr<ParticleGroup> tags04(new ParticleGroup(sysdef, selector04));

    // create a group of type 0
    std::shared_ptr<ParticleFilter> selector0(new ParticleFilterType(0, 0));
    std::shared_ptr<ParticleGroup> type0(new ParticleGroup(sysdef, selector0));

    // make a union of the two groups and check it
    std::shared_ptr<ParticleGroup> union_group = ParticleGroup::groupUnion(type0, tags04);
    CHECK_EQUAL_UINT(union_group->getNumMembers(), 7);
    CHECK_EQUAL_UINT(union_group->getIndexArray().getNumElements(), 7);
    CHECK_EQUAL_UINT(union_group->getMemberTag(0), 0);
    CHECK_EQUAL_UINT(union_group->getMemberTag(1), 1);
    CHECK_EQUAL_UINT(union_group->getMemberTag(2), 2);
    CHECK_EQUAL_UINT(union_group->getMemberTag(3), 3);
    CHECK_EQUAL_UINT(union_group->getMemberTag(4), 4);
    CHECK_EQUAL_UINT(union_group->getMemberTag(5), 5);
    CHECK_EQUAL_UINT(union_group->getMemberTag(6), 8);

    // make a intersection group and test it
    std::shared_ptr<ParticleGroup> intersection_group
        = ParticleGroup::groupIntersection(type0, tags04);
    CHECK_EQUAL_UINT(intersection_group->getNumMembers(), 2);
    CHECK_EQUAL_UINT(intersection_group->getIndexArray().getNumElements(), 2);
    CHECK_EQUAL_UINT(intersection_group->getMemberTag(0), 0);
    CHECK_EQUAL_UINT(intersection_group->getMemberTag(1), 2);
    }

//! Checks that the ParticleGroup::getTotalMass works correctly
UP_TEST(ParticleGroup_total_mass_tests)
    {
    std::shared_ptr<SystemDefinition> sysdef = create_sysdef();
    std::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    ParticleGroup group1(
        sysdef,
        std::shared_ptr<ParticleFilter>(new ParticleFilterTags(std::vector<unsigned int>({0}))));
    MY_CHECK_CLOSE(group1.getTotalMass(), 1.0, tol);

    ParticleGroup group2(
        sysdef,
        std::shared_ptr<ParticleFilter>(new ParticleFilterTags(std::vector<unsigned int>({0, 1}))));
    MY_CHECK_CLOSE(group2.getTotalMass(), 3.0, tol);

    ParticleGroup group3(sysdef,
                         std::shared_ptr<ParticleFilter>(
                             new ParticleFilterTags(std::vector<unsigned int>({0, 1, 2}))));
    MY_CHECK_CLOSE(group3.getTotalMass(), 8.0, tol);
    }

//! Checks that the ParticleGroup::getCenterOfMass works correctly
UP_TEST(ParticleGroup_center_of_mass_tests)
    {
    std::shared_ptr<SystemDefinition> sysdef = create_sysdef();
    std::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    Scalar3 com;
    ParticleGroup group1(
        sysdef,
        std::shared_ptr<ParticleFilter>(new ParticleFilterTags(std::vector<unsigned int>({0}))));
    com = group1.getCenterOfMass();
    MY_CHECK_SMALL(com.x, tol_small);
    MY_CHECK_SMALL(com.y, tol_small);
    MY_CHECK_SMALL(com.z, tol_small);

    ParticleGroup group2(
        sysdef,
        std::shared_ptr<ParticleFilter>(new ParticleFilterTags(std::vector<unsigned int>({0, 1}))));
    com = group2.getCenterOfMass();
    MY_CHECK_CLOSE(com.x, 7.3333333333, tol);
    MY_CHECK_CLOSE(com.y, -5.3333333333, tol);
    MY_CHECK_CLOSE(com.z, 15.333333333, tol);

    ParticleGroup group3(sysdef,
                         std::shared_ptr<ParticleFilter>(
                             new ParticleFilterTags(std::vector<unsigned int>({0, 1, 2}))));
    com = group3.getCenterOfMass();
    MY_CHECK_CLOSE(com.x, 2.125, tol);
    MY_CHECK_CLOSE(com.y, -3.25, tol);
    MY_CHECK_CLOSE(com.z, 3.875, tol);
    }
