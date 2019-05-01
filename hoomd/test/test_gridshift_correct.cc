// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// this include is necessary to get MPI included before anything else to support intel MPI
#include "hoomd/ExecutionConfiguration.h"

#include "upp11_config.h"

#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include "hoomd/System.h"
#include "hoomd/ParticleData.h"

#include <math.h>
#include <memory>

HOOMD_UP_MAIN();
/*! \file test_gridshift_correct.cc
    \brief Unit tests for the ParticleData class in response to origin shifts
    \ingroup unit_tests
*/

//! test case to verify proper operation of ZeroMomentumUpdater
UP_TEST( ParticleDataGridShiftGetMethods )
    {
    // create a simple particle data to test with
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(3, BoxDim(10.0), 4));
    std::shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    BoxDim box = pdata->getBox();
    {
    ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(pdata->getVelocities(), access_location::host, access_mode::readwrite);

    h_pos.data[0].x = h_pos.data[0].y = h_pos.data[0].z = 0.0;
    h_pos.data[1].x = h_pos.data[1].y = h_pos.data[1].z = 1.0;
    }

    // compute a shift and apply it to all particles, and origin
    Scalar3 shift = make_scalar3(0.5,0.125,0.75);
    pdata->translateOrigin(shift);
    {
    ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<int3> h_img(pdata->getImages(), access_location::host, access_mode::readwrite);

    for (unsigned int i = 0; i < pdata->getN(); i++)
        {
        // read in the current position and orientation
        Scalar4 pos_i = h_pos.data[i];
        vec3<Scalar> r_i = vec3<Scalar>(pos_i); // translation from local to global coordinates
        r_i += vec3<Scalar>(shift);
        h_pos.data[i] = vec_to_scalar4(r_i, pos_i.w);
        box.wrap(h_pos.data[i], h_img.data[i]);
        }
    }

    // check that the particle positions are still the original ones
    Scalar3 pos = pdata->getPosition(0);
    MY_CHECK_SMALL(Scalar(pos.x-0.0), tol_small);
    MY_CHECK_SMALL(Scalar(pos.y-0.0), tol_small);
    MY_CHECK_SMALL(Scalar(pos.z-0.0), tol_small);
    pos = pdata->getPosition(1);
    MY_CHECK_SMALL(Scalar(pos.x-1.0), tol_small);
    MY_CHECK_SMALL(Scalar(pos.y-1.0), tol_small);
    MY_CHECK_SMALL(Scalar(pos.z-1.0), tol_small);

    int3 pimg = pdata->getImage(0);
    UP_ASSERT_EQUAL(pimg.x, 0);
    UP_ASSERT_EQUAL(pimg.y, 0);
    UP_ASSERT_EQUAL(pimg.z, 0);
    pimg = pdata->getImage(0);
    UP_ASSERT_EQUAL(pimg.x, 0);
    UP_ASSERT_EQUAL(pimg.y, 0);
    UP_ASSERT_EQUAL(pimg.z, 0);

    // compute a shift that will shift the image of the box
    Scalar3 shift_img = make_scalar3(10.5,10.125,10.75);
    pdata->translateOrigin(shift_img);
    {
    ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<int3> h_img(pdata->getImages(), access_location::host, access_mode::readwrite);

    for (unsigned int i = 0; i < pdata->getN(); i++)
        {
        // read in the current position and orientation
        Scalar4 pos_i = h_pos.data[i];
        vec3<Scalar> r_i = vec3<Scalar>(pos_i); // translation from local to global coordinates
        r_i += vec3<Scalar>(shift_img);
        h_pos.data[i] = vec_to_scalar4(r_i, pos_i.w);
        box.wrap(h_pos.data[i], h_img.data[i]);
        }
    }

    // check that the particle positions are still the original ones
    pos = pdata->getPosition(0);
    MY_CHECK_SMALL(Scalar(pos.x-0.0), tol_small);
    MY_CHECK_SMALL(Scalar(pos.y-0.0), tol_small);
    MY_CHECK_SMALL(Scalar(pos.z-0.0), tol_small);
    pos = pdata->getPosition(1);
    MY_CHECK_SMALL(Scalar(pos.x-1.0), tol_small);
    MY_CHECK_SMALL(Scalar(pos.y-1.0), tol_small);
    MY_CHECK_SMALL(Scalar(pos.z-1.0), tol_small);

    pimg = pdata->getImage(0);
    UP_ASSERT_EQUAL(pimg.x, 0);
    UP_ASSERT_EQUAL(pimg.y, 0);
    UP_ASSERT_EQUAL(pimg.z, 0);
    pimg = pdata->getImage(0);
    UP_ASSERT_EQUAL(pimg.x, 0);
    UP_ASSERT_EQUAL(pimg.y, 0);
    UP_ASSERT_EQUAL(pimg.z, 0);
    }

UP_TEST( ParticleDataGridShiftSetMethods )
    {
    // create a simple particle data to test with
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(3, BoxDim(10.0), 4));
    std::shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    BoxDim box = pdata->getBox();
    {
    ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(pdata->getVelocities(), access_location::host, access_mode::readwrite);

    h_pos.data[0].x = h_pos.data[0].y = h_pos.data[0].z = 0.0;
    h_pos.data[1].x = h_pos.data[1].y = h_pos.data[1].z = 1.0;
    }

    // compute a shift that will shift the image of the box
    Scalar3 shift_img =  make_scalar3(10.5,10.125,10.75);
    pdata->translateOrigin(shift_img);
    {
    ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<int3> h_img(pdata->getImages(), access_location::host, access_mode::readwrite);

    for (unsigned int i = 0; i < pdata->getN(); i++)
        {
        // read in the current position and orientation
        Scalar4 pos_i = h_pos.data[i];
        vec3<Scalar> r_i = vec3<Scalar>(pos_i); // translation from local to global coordinates
        r_i += vec3<Scalar>(shift_img);
        h_pos.data[i] = vec_to_scalar4(r_i, pos_i.w);
        box.wrap(h_pos.data[i], h_img.data[i]);
        }
    }

    // check that the particle positions are still the original ones
    Scalar3 pos = pdata->getPosition(0);
    MY_CHECK_SMALL(Scalar(pos.x-0.0), tol_small);
    MY_CHECK_SMALL(Scalar(pos.y-0.0), tol_small);
    MY_CHECK_SMALL(Scalar(pos.z-0.0), tol_small);
    pos = pdata->getPosition(1);
    MY_CHECK_SMALL(Scalar(pos.x-1.0), tol_small);
    MY_CHECK_SMALL(Scalar(pos.y-1.0), tol_small);
    MY_CHECK_SMALL(Scalar(pos.z-1.0), tol_small);

    int3 pimg = pdata->getImage(0);
    UP_ASSERT_EQUAL(pimg.x, 0);
    UP_ASSERT_EQUAL(pimg.y, 0);
    UP_ASSERT_EQUAL(pimg.z, 0);
    pimg = pdata->getImage(1);
    UP_ASSERT_EQUAL(pimg.x, 0);
    UP_ASSERT_EQUAL(pimg.y, 0);
    UP_ASSERT_EQUAL(pimg.z, 0);


    //OK, now we set the positions using the particle data proxy
    Scalar3 new_pos0 = make_scalar3(0.1,0.5,0.7);
    pdata->setPosition(0,new_pos0);
    Scalar3 new_pos1 = make_scalar3(0.4,0.1,2.75);
    pdata->setPosition(1,new_pos1);

    Scalar3 ret_pos0 = pdata->getPosition(0);
    MY_CHECK_SMALL(ret_pos0.x-new_pos0.x, tol_small);
    MY_CHECK_SMALL(ret_pos0.y-new_pos0.y, tol_small);
    MY_CHECK_SMALL(ret_pos0.z-new_pos0.z, tol_small);

    Scalar3 ret_pos1 = pdata->getPosition(1);
    MY_CHECK_SMALL(ret_pos1.x-new_pos1.x, tol_small);
    MY_CHECK_SMALL(ret_pos1.y-new_pos1.y, tol_small);
    MY_CHECK_SMALL(ret_pos1.z-new_pos1.z, tol_small);

    //OK, now do the same with the images
    int3 new_img0 = make_int3(1,-5,7);
    pdata->setImage(0,new_img0);
    int3 new_img1 = make_int3(4,1,10);
    pdata->setImage(1,new_img1);

    int3 ret_img0 = pdata->getImage(0);
    UP_ASSERT_EQUAL(ret_img0.x-new_img0.x, 0);
    UP_ASSERT_EQUAL(ret_img0.y-new_img0.y, 0);
    UP_ASSERT_EQUAL(ret_img0.z-new_img0.z, 0);

    int3 ret_img1 = pdata->getImage(1);
    UP_ASSERT_EQUAL(ret_img1.x-new_img1.x, 0);
    UP_ASSERT_EQUAL(ret_img1.y-new_img1.y, 0);
    UP_ASSERT_EQUAL(ret_img1.z-new_img1.z, 0);
    }
