// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file pdata_test.cc
    \brief Unit tests for BoxDim, ParticleData, SimpleCubicInitializer, and RandomInitializer
   classes. \ingroup unit_tests
*/

// this include is necessary to get MPI included before anything else to support intel MPI
#include "hoomd/ExecutionConfiguration.h"

#include <iostream>

#include "hoomd/Initializers.h"
#include "hoomd/ParticleData.h"
#include "hoomd/SnapshotSystemData.h"

#include "upp11_config.h"

using namespace std;
using namespace hoomd;

HOOMD_UP_MAIN();

//! Perform some basic tests on the boxdim structure
UP_TEST(BoxDim_basic_test)
    {
    Scalar tol = Scalar(1e-6);

    // test default constructor
    BoxDim a;
    MY_CHECK_CLOSE(a.getLo().x, 0.0, tol);
    MY_CHECK_CLOSE(a.getLo().y, 0.0, tol);
    MY_CHECK_CLOSE(a.getLo().z, 0.0, tol);
    MY_CHECK_CLOSE(a.getHi().x, 0.0, tol);
    MY_CHECK_CLOSE(a.getHi().y, 0.0, tol);
    MY_CHECK_CLOSE(a.getHi().z, 0.0, tol);

    BoxDim b(10.0);
    MY_CHECK_CLOSE(b.getLo().x, -5.0, tol);
    MY_CHECK_CLOSE(b.getLo().y, -5.0, tol);
    MY_CHECK_CLOSE(b.getLo().z, -5.0, tol);
    MY_CHECK_CLOSE(b.getHi().x, 5.0, tol);
    MY_CHECK_CLOSE(b.getHi().y, 5.0, tol);
    MY_CHECK_CLOSE(b.getHi().z, 5.0, tol);
    MY_CHECK_CLOSE(b.getL().x, 10.0, tol);
    MY_CHECK_CLOSE(b.getL().y, 10.0, tol);
    MY_CHECK_CLOSE(b.getL().z, 10.0, tol);
    UP_ASSERT_EQUAL(b.getPeriodic().x, 1);
    UP_ASSERT_EQUAL(b.getPeriodic().y, 1);
    UP_ASSERT_EQUAL(b.getPeriodic().z, 1);

    BoxDim c(10.0, 30.0, 50.0);
    MY_CHECK_CLOSE(c.getLo().x, -5.0, tol);
    MY_CHECK_CLOSE(c.getLo().y, -15.0, tol);
    MY_CHECK_CLOSE(c.getLo().z, -25.0, tol);
    MY_CHECK_CLOSE(c.getHi().x, 5.0, tol);
    MY_CHECK_CLOSE(c.getHi().y, 15.0, tol);
    MY_CHECK_CLOSE(c.getHi().z, 25.0, tol);
    MY_CHECK_CLOSE(c.getL().x, 10.0, tol);
    MY_CHECK_CLOSE(c.getL().y, 30.0, tol);
    MY_CHECK_CLOSE(c.getL().z, 50.0, tol);
    UP_ASSERT_EQUAL(c.getPeriodic().x, 1);
    UP_ASSERT_EQUAL(c.getPeriodic().y, 1);
    UP_ASSERT_EQUAL(c.getPeriodic().z, 1);

    // test for assignment and copy constructor
    c.setPeriodic(make_uchar3(1, 0, 1));
    BoxDim d(c);
    MY_CHECK_CLOSE(d.getLo().x, -5.0, tol);
    MY_CHECK_CLOSE(d.getLo().y, -15.0, tol);
    MY_CHECK_CLOSE(d.getLo().z, -25.0, tol);
    MY_CHECK_CLOSE(d.getHi().x, 5.0, tol);
    MY_CHECK_CLOSE(d.getHi().y, 15.0, tol);
    MY_CHECK_CLOSE(d.getHi().z, 25.0, tol);
    MY_CHECK_CLOSE(d.getL().x, 10.0, tol);
    MY_CHECK_CLOSE(d.getL().y, 30.0, tol);
    MY_CHECK_CLOSE(d.getL().z, 50.0, tol);
    UP_ASSERT_EQUAL(d.getPeriodic().x, 1);
    UP_ASSERT_EQUAL(d.getPeriodic().y, 0);
    UP_ASSERT_EQUAL(d.getPeriodic().z, 1);

    BoxDim e;
    e = c;
    MY_CHECK_CLOSE(e.getLo().x, -5.0, tol);
    MY_CHECK_CLOSE(e.getLo().y, -15.0, tol);
    MY_CHECK_CLOSE(e.getLo().z, -25.0, tol);
    MY_CHECK_CLOSE(e.getHi().x, 5.0, tol);
    MY_CHECK_CLOSE(e.getHi().y, 15.0, tol);
    MY_CHECK_CLOSE(e.getHi().z, 25.0, tol);
    MY_CHECK_CLOSE(e.getL().x, 10.0, tol);
    MY_CHECK_CLOSE(e.getL().y, 30.0, tol);
    MY_CHECK_CLOSE(e.getL().z, 50.0, tol);
    UP_ASSERT_EQUAL(d.getPeriodic().x, 1);
    UP_ASSERT_EQUAL(d.getPeriodic().y, 0);
    UP_ASSERT_EQUAL(d.getPeriodic().z, 1);

    b = b;
    MY_CHECK_CLOSE(b.getLo().x, -5.0, tol);
    MY_CHECK_CLOSE(b.getLo().y, -5.0, tol);
    MY_CHECK_CLOSE(b.getLo().z, -5.0, tol);
    MY_CHECK_CLOSE(b.getHi().x, 5.0, tol);
    MY_CHECK_CLOSE(b.getHi().y, 5.0, tol);
    MY_CHECK_CLOSE(b.getHi().z, 5.0, tol);
    MY_CHECK_CLOSE(b.getL().x, 10.0, tol);
    MY_CHECK_CLOSE(b.getL().y, 10.0, tol);
    MY_CHECK_CLOSE(b.getL().z, 10.0, tol);
    UP_ASSERT_EQUAL(b.getPeriodic().x, 1);
    UP_ASSERT_EQUAL(b.getPeriodic().y, 1);
    UP_ASSERT_EQUAL(b.getPeriodic().z, 1);
    }

UP_TEST(BoxDim_functionality_test1)
    {
    BoxDim b(5.0);
    MY_CHECK_CLOSE(b.getL().x, 5.0, tol);
    MY_CHECK_CLOSE(b.getL().y, 5.0, tol);
    MY_CHECK_CLOSE(b.getL().z, 5.0, tol);

    b.setL(make_scalar3(10.0, 10.0, 10.0));
    MY_CHECK_CLOSE(b.getL().x, 10.0, tol);
    MY_CHECK_CLOSE(b.getL().y, 10.0, tol);
    MY_CHECK_CLOSE(b.getL().z, 10.0, tol);

    Scalar3 v = b.makeFraction(make_scalar3(5.0, 5.0, 5.0));
    MY_CHECK_CLOSE(v.x, 1.0, tol);
    MY_CHECK_CLOSE(v.y, 1.0, tol);
    MY_CHECK_CLOSE(v.z, 1.0, tol);

    v = b.makeFraction(make_scalar3(1.0, -2.0, 3.0));
    MY_CHECK_CLOSE(v.x, 0.6, tol);
    MY_CHECK_CLOSE(v.y, 0.3, tol);
    MY_CHECK_CLOSE(v.z, 0.8, tol);

    v = b.minImage(make_scalar3(1.0, -2.0, 3.0));
    MY_CHECK_CLOSE(v.x, 1.0, tol);
    MY_CHECK_CLOSE(v.y, -2.0, tol);
    MY_CHECK_CLOSE(v.z, 3.0, tol);

    // test minimum image
    v = b.minImage(make_scalar3(6.0, -7.0, 8.0));
    MY_CHECK_CLOSE(v.x, -4.0, tol);
    MY_CHECK_CLOSE(v.y, 3.0, tol);
    MY_CHECK_CLOSE(v.z, -2.0, tol);

    b.setPeriodic(make_uchar3(0, 0, 0));
    v = b.minImage(make_scalar3(6.0, -7.0, 8.0));
    MY_CHECK_CLOSE(v.x, 6.0, tol);
    MY_CHECK_CLOSE(v.y, -7.0, tol);
    MY_CHECK_CLOSE(v.z, 8.0, tol);

    b.setPeriodic(make_uchar3(1, 0, 0));
    v = b.minImage(make_scalar3(6.0, -7.0, 8.0));
    MY_CHECK_CLOSE(v.x, -4.0, tol);
    MY_CHECK_CLOSE(v.y, -7.0, tol);
    MY_CHECK_CLOSE(v.z, 8.0, tol);

    b.setPeriodic(make_uchar3(0, 1, 0));
    v = b.minImage(make_scalar3(6.0, -7.0, 8.0));
    MY_CHECK_CLOSE(v.x, 6.0, tol);
    MY_CHECK_CLOSE(v.y, 3.0, tol);
    MY_CHECK_CLOSE(v.z, 8.0, tol);

    b.setPeriodic(make_uchar3(0, 0, 1));
    v = b.minImage(make_scalar3(6.0, -7.0, 8.0));
    MY_CHECK_CLOSE(v.x, 6.0, tol);
    MY_CHECK_CLOSE(v.y, -7.0, tol);
    MY_CHECK_CLOSE(v.z, -2.0, tol);

    // test wrap
    b.setPeriodic(make_uchar3(1, 1, 1));
    int3 image = make_int3(10, 20, 30);
    v = make_scalar3(1.0, -2.0, 3.0);
    b.wrap(v, image);
    MY_CHECK_CLOSE(v.x, 1.0, tol);
    MY_CHECK_CLOSE(v.y, -2.0, tol);
    MY_CHECK_CLOSE(v.z, 3.0, tol);
    UP_ASSERT_EQUAL(image.x, 10);
    UP_ASSERT_EQUAL(image.y, 20);
    UP_ASSERT_EQUAL(image.z, 30);

    image = make_int3(10, 20, 30);
    v = make_scalar3(6.0, -7.0, 8.0);
    b.wrap(v, image);
    MY_CHECK_CLOSE(v.x, -4.0, tol);
    MY_CHECK_CLOSE(v.y, 3.0, tol);
    MY_CHECK_CLOSE(v.z, -2.0, tol);
    UP_ASSERT_EQUAL(image.x, 11);
    UP_ASSERT_EQUAL(image.y, 19);
    UP_ASSERT_EQUAL(image.z, 31);

    b.setPeriodic(make_uchar3(1, 0, 0));
    image = make_int3(10, 20, 30);
    v = make_scalar3(6.0, -7.0, 8.0);
    b.wrap(v, image);
    MY_CHECK_CLOSE(v.x, -4.0, tol);
    MY_CHECK_CLOSE(v.y, -7.0, tol);
    MY_CHECK_CLOSE(v.z, 8.0, tol);
    UP_ASSERT_EQUAL(image.x, 11);
    UP_ASSERT_EQUAL(image.y, 20);
    UP_ASSERT_EQUAL(image.z, 30);

    b.setPeriodic(make_uchar3(0, 1, 0));
    image = make_int3(10, 20, 30);
    v = make_scalar3(6.0, -7.0, 8.0);
    b.wrap(v, image);
    MY_CHECK_CLOSE(v.x, 6.0, tol);
    MY_CHECK_CLOSE(v.y, 3.0, tol);
    MY_CHECK_CLOSE(v.z, 8.0, tol);
    UP_ASSERT_EQUAL(image.x, 10);
    UP_ASSERT_EQUAL(image.y, 19);
    UP_ASSERT_EQUAL(image.z, 30);

    b.setPeriodic(make_uchar3(0, 0, 1));
    image = make_int3(10, 20, 30);
    v = make_scalar3(6.0, -7.0, 8.0);
    b.wrap(v, image);
    MY_CHECK_CLOSE(v.x, 6.0, tol);
    MY_CHECK_CLOSE(v.y, -7.0, tol);
    MY_CHECK_CLOSE(v.z, -2.0, tol);
    UP_ASSERT_EQUAL(image.x, 10);
    UP_ASSERT_EQUAL(image.y, 20);
    UP_ASSERT_EQUAL(image.z, 31);
    }

UP_TEST(BoxDim_triclinic_test)
    {
    BoxDim b(5.0);

    Scalar tol = Scalar(1e-4);

    Scalar xy = 1.0;
    Scalar xz = .4;
    Scalar yz = .9;

    b.setTiltFactors(xy, xz, yz);
    MY_CHECK_CLOSE(b.getTiltFactorXY(), xy, tol);
    MY_CHECK_CLOSE(b.getTiltFactorXZ(), xz, tol);
    MY_CHECK_CLOSE(b.getTiltFactorYZ(), yz, tol);

    Scalar3 f = make_scalar3(.5, .6, .7);
    Scalar3 L = b.getL();

    Scalar3 pos;
    pos.x
        = b.getLo().x + f.x * L.x + xy * L.y * (f.y - Scalar(0.5)) + xz * L.z * (f.z - Scalar(0.5));
    pos.y = b.getLo().y + f.y * L.y + yz * L.z * (f.z - Scalar(0.5));
    pos.z = b.getLo().z + f.z * L.z;

    // convert pos to fraction
    Scalar3 f2 = b.makeFraction(pos);

    MY_CHECK_CLOSE(f2.x, f.x, tol);
    MY_CHECK_CLOSE(f2.y, f.y, tol);
    MY_CHECK_CLOSE(f2.z, f.z, tol);

    // convert fraction to pos
    Scalar3 pos2 = b.makeCoordinates(f);

    MY_CHECK_CLOSE(pos2.x, pos.x, tol);
    MY_CHECK_CLOSE(pos2.y, pos.y, tol);
    MY_CHECK_CLOSE(pos2.z, pos.z, tol);

    /*
     * test ghost layer padding
     */
    // fractional ghost layer
    Scalar3 npd = b.getNearestPlaneDistance();
    Scalar3 ghost_frac = make_scalar3(0.5, 0.3, 0.0);
    Scalar3 ghost_width = ghost_frac * npd;

    // make fraction with a ghost layer
    Scalar3 f3 = b.makeFraction(pos, ghost_width);

    // compare to new fraction, calculated from old fraction accounting for ghost layer
    Scalar3 f3_cmp = (f2 + ghost_frac) / (make_scalar3(1.0, 1.0, 1.0) + 2.0 * ghost_frac);
    MY_CHECK_CLOSE(f3.x, f3_cmp.x, tol);
    MY_CHECK_CLOSE(f3.y, f3_cmp.y, tol);
    MY_CHECK_CLOSE(f3.z, f3_cmp.z, tol);

    // test minimum image

    // along x coordinate
    Scalar3 dx = make_scalar3(3.0, 1.0, 2.0);
    Scalar3 dx2 = b.minImage(dx);

    MY_CHECK_CLOSE(dx2.x, -2.0, tol);
    MY_CHECK_CLOSE(dx2.y, 1.0, tol);
    MY_CHECK_CLOSE(dx2.z, 2.0, tol);

    dx = make_scalar3(-3.0, 1.0, 2.0);
    dx2 = b.minImage(dx);

    MY_CHECK_CLOSE(dx2.x, 2.0, tol);
    MY_CHECK_CLOSE(dx2.y, 1.0, tol);
    MY_CHECK_CLOSE(dx2.z, 2.0, tol);

    // along y coordinate
    dx = make_scalar3(2.0, 2.6, 1.5);
    dx2 = b.minImage(dx);

    MY_CHECK_CLOSE(dx2.x, 2.0, tol);
    MY_CHECK_CLOSE(dx2.y, -2.4, tol);
    MY_CHECK_CLOSE(dx2.z, 1.5, tol);

    dx = make_scalar3(2.0, -2.6, 1.5);
    dx2 = b.minImage(dx);

    MY_CHECK_CLOSE(dx2.x, 2.0, tol);
    MY_CHECK_CLOSE(dx2.y, 2.4, tol);
    MY_CHECK_CLOSE(dx2.z, 1.5, tol);

    dx = make_scalar3(3.0, 2.6, 1.5);
    dx2 = b.minImage(dx);

    MY_CHECK_CLOSE(dx2.x, -2.0, tol);
    MY_CHECK_CLOSE(dx2.y, -2.4, tol);
    MY_CHECK_CLOSE(dx2.z, 1.5, tol);

    dx = make_scalar3(3.0, -2.6, 1.5);
    dx2 = b.minImage(dx);

    MY_CHECK_CLOSE(dx2.x, -2.0, tol);
    MY_CHECK_CLOSE(dx2.y, 2.4, tol);
    MY_CHECK_CLOSE(dx.z, 1.5, tol);

    // along z coordinate
    dx = make_scalar3(2.1, 1.5, 3.0);
    dx2 = b.minImage(dx);

    MY_CHECK_CLOSE(dx2.x, 0.1, tol);
    MY_CHECK_CLOSE(dx2.y, 2.0, tol);
    MY_CHECK_CLOSE(dx2.z, -2.0, tol);

    dx = make_scalar3(2.1, 1.5, -3.0);
    dx2 = b.minImage(dx);

    MY_CHECK_CLOSE(dx2.x, -0.9, tol);
    MY_CHECK_CLOSE(dx2.y, 1.0, tol);
    MY_CHECK_CLOSE(dx2.z, 2.0, tol);

    // test particle wrap

    // along z direction
    pos = make_scalar3(1.0, 2.0, 2.6);
    int3 img = make_int3(0, 0, 0);

    b.wrap(pos, img);
    MY_CHECK_CLOSE(pos.x, -1.0, tol);
    MY_CHECK_CLOSE(pos.y, -2.5, tol);
    MY_CHECK_CLOSE(pos.z, -2.4, tol);
    UP_ASSERT_EQUAL(img.x, 0);
    UP_ASSERT_EQUAL(img.y, 0);
    UP_ASSERT_EQUAL(img.z, 1);

    pos = make_scalar3(-1.0, -2.0, -2.6);
    img = make_int3(0, 0, 0);
    b.wrap(pos, img);

    MY_CHECK_CLOSE(pos.x, 1.0, tol);
    MY_CHECK_CLOSE(pos.y, 2.5, tol);
    MY_CHECK_CLOSE(pos.z, 2.4, tol);
    UP_ASSERT_EQUAL(img.x, 0);
    UP_ASSERT_EQUAL(img.y, 0);
    UP_ASSERT_EQUAL(img.z, -1);

    // along y direction
    pos = make_scalar3(1.0, 4.0, 1.5);
    img = make_int3(0, 0, 0);

    b.wrap(pos, img);
    MY_CHECK_CLOSE(pos.x, -4.0, tol);
    MY_CHECK_CLOSE(pos.y, -1.0, tol);
    MY_CHECK_CLOSE(pos.z, 1.5, tol);

    UP_ASSERT_EQUAL(img.x, 0);
    UP_ASSERT_EQUAL(img.y, 1);
    UP_ASSERT_EQUAL(img.z, 0);

    pos = make_scalar3(-1.0, -4.0, -1.5);
    img = make_int3(0, 0, 0);

    b.wrap(pos, img);
    MY_CHECK_CLOSE(pos.x, 4.0, tol);
    MY_CHECK_CLOSE(pos.y, 1.0, tol);
    MY_CHECK_CLOSE(pos.z, -1.5, tol);

    UP_ASSERT_EQUAL(img.x, 0);
    UP_ASSERT_EQUAL(img.y, -1);
    UP_ASSERT_EQUAL(img.z, 0);

    // along x direction
    pos = make_scalar3(4.2, 1.5, 1.0);
    img = make_int3(0, 0, 0);

    b.wrap(pos, img);
    MY_CHECK_CLOSE(pos.x, -0.8, tol);
    MY_CHECK_CLOSE(pos.y, 1.5, tol);
    MY_CHECK_CLOSE(pos.z, 1.0, tol);

    UP_ASSERT_EQUAL(img.x, 1);
    UP_ASSERT_EQUAL(img.y, 0);
    UP_ASSERT_EQUAL(img.z, 0);

    pos = make_scalar3(-5.0, -1.5, 1.0);
    img = make_int3(0, 0, 0);

    b.wrap(pos, img);
    MY_CHECK_CLOSE(pos.x, 0.0, tol);
    MY_CHECK_CLOSE(pos.y, -1.5, tol);
    MY_CHECK_CLOSE(pos.z, 1.0, tol);

    UP_ASSERT_EQUAL(img.x, -1);
    UP_ASSERT_EQUAL(img.y, 0);
    UP_ASSERT_EQUAL(img.z, 0);
    }

//! Test operation of the particle data class
UP_TEST(ParticleData_test)
    {
    auto box = std::make_shared<BoxDim>(10.0, 30.0, 50.0);
    std::shared_ptr<ExecutionConfiguration> exec_conf(
        new ExecutionConfiguration(ExecutionConfiguration::CPU));
    ParticleData a(1, box, 1, exec_conf);

    Scalar tol = Scalar(1e-6);

    // make sure the box is working
    const BoxDim& c = a.getBox();
    MY_CHECK_CLOSE(c.getLo().x, -5.0, tol);
    MY_CHECK_CLOSE(c.getLo().y, -15.0, tol);
    MY_CHECK_CLOSE(c.getLo().z, -25.0, tol);
    MY_CHECK_CLOSE(c.getHi().x, 5.0, tol);
    MY_CHECK_CLOSE(c.getHi().y, 15.0, tol);
    MY_CHECK_CLOSE(c.getHi().z, 25.0, tol);

    a.setGlobalBoxL(make_scalar3(5.0, 5.0, 5.0));
    const BoxDim& d = a.getBox();
    MY_CHECK_CLOSE(d.getLo().x, -2.5, tol);
    MY_CHECK_CLOSE(d.getLo().y, -2.5, tol);
    MY_CHECK_CLOSE(d.getLo().z, -2.5, tol);
    MY_CHECK_CLOSE(d.getHi().x, 2.5, tol);
    MY_CHECK_CLOSE(d.getHi().y, 2.5, tol);
    MY_CHECK_CLOSE(d.getHi().z, 2.5, tol);

    // make sure that getN is working
    UP_ASSERT(a.getN() == 1);

        // Test the ability to acquire data
        {
        ArrayHandle<Scalar4> h_pos(a.getPositions(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> h_vel(a.getVelocities(),
                                   access_location::host,
                                   access_mode::readwrite);
        ArrayHandle<Scalar3> h_accel(a.getAccelerations(),
                                     access_location::host,
                                     access_mode::readwrite);
        ArrayHandle<int3> h_image(a.getImages(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> h_charge(a.getCharges(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> h_diameter(a.getDiameters(),
                                       access_location::host,
                                       access_mode::readwrite);
        ArrayHandle<unsigned int> h_tag(a.getTags(), access_location::host, access_mode::readwrite);
        ArrayHandle<unsigned int> h_rtag(a.getRTags(),
                                         access_location::host,
                                         access_mode::readwrite);
        ArrayHandle<unsigned int> h_body(a.getBodies(),
                                         access_location::host,
                                         access_mode::readwrite);

        // begin by verifying that the defaults the class advertises are set
        UP_ASSERT(a.getPositions().getNumElements() == 1);
        UP_ASSERT(a.getVelocities().getNumElements() == 1);
        UP_ASSERT(a.getAccelerations().getNumElements() == 1);
        UP_ASSERT(a.getImages().getNumElements() == 1);
        UP_ASSERT(a.getCharges().getNumElements() == 1);
        UP_ASSERT(a.getDiameters().getNumElements() == 1);
        UP_ASSERT(a.getTags().getNumElements() == 1);
        UP_ASSERT(a.getRTags().size() == 1);
        UP_ASSERT(a.getBodies().getNumElements() == 1);

        MY_CHECK_CLOSE(h_pos.data[0].x, 0.0, tol);
        MY_CHECK_CLOSE(h_pos.data[0].y, 0.0, tol);
        MY_CHECK_CLOSE(h_pos.data[0].z, 0.0, tol);
        MY_CHECK_CLOSE(h_vel.data[0].x, 0.0, tol);
        MY_CHECK_CLOSE(h_vel.data[0].y, 0.0, tol);
        MY_CHECK_CLOSE(h_vel.data[0].z, 0.0, tol);
        MY_CHECK_CLOSE(h_accel.data[0].x, 0.0, tol);
        MY_CHECK_CLOSE(h_accel.data[0].y, 0.0, tol);
        MY_CHECK_CLOSE(h_accel.data[0].z, 0.0, tol);
        MY_CHECK_CLOSE(h_charge.data[0], 0.0, tol);
        MY_CHECK_CLOSE(h_vel.data[0].w, 1.0, tol); // mass
        MY_CHECK_CLOSE(h_diameter.data[0], 1.0, tol);
        UP_ASSERT_EQUAL(h_image.data[0].x, 0);
        UP_ASSERT_EQUAL(h_image.data[0].y, 0);
        UP_ASSERT_EQUAL(h_image.data[0].z, 0);
        UP_ASSERT(__scalar_as_int(h_pos.data[0].w) == 0); // type
        UP_ASSERT(h_rtag.data[0] == 0);
        UP_ASSERT(h_tag.data[0] == 0);
        UP_ASSERT(h_body.data[0] == NO_BODY);

        // set some new values for testing
        h_pos.data[0].x = 1.0;
        h_pos.data[0].y = 2.0;
        h_pos.data[0].z = -2.0;
        h_vel.data[0].x = 11.0;
        h_vel.data[0].y = 12.0;
        h_vel.data[0].z = 13.0;
        h_accel.data[0].x = 21.0;
        h_accel.data[0].y = 22.0;
        h_accel.data[0].z = 23.0;
        h_charge.data[0] = 24.0;
        h_vel.data[0].w = 25.0; // mass
        h_diameter.data[0] = 26.0;
        h_image.data[0].x = 27;
        h_image.data[0].y = 28;
        h_image.data[0].z = 29;
        h_pos.data[0].w = __int_as_scalar(1); // type
        h_body.data[0] = 0;
        }

        // make sure when the data is re-acquired, the values read properly
        {
        ArrayHandle<Scalar4> h_pos(a.getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_vel(a.getVelocities(), access_location::host, access_mode::read);
        ArrayHandle<Scalar3> h_accel(a.getAccelerations(),
                                     access_location::host,
                                     access_mode::read);
        ArrayHandle<int3> h_image(a.getImages(), access_location::host, access_mode::read);
        ArrayHandle<Scalar> h_charge(a.getCharges(), access_location::host, access_mode::read);
        ArrayHandle<Scalar> h_diameter(a.getDiameters(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_tag(a.getTags(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_rtag(a.getRTags(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_body(a.getBodies(), access_location::host, access_mode::read);

        UP_ASSERT(a.getPositions().getNumElements() == 1);
        UP_ASSERT(a.getVelocities().getNumElements() == 1);
        UP_ASSERT(a.getAccelerations().getNumElements() == 1);
        UP_ASSERT(a.getImages().getNumElements() == 1);
        UP_ASSERT(a.getCharges().getNumElements() == 1);
        UP_ASSERT(a.getDiameters().getNumElements() == 1);
        UP_ASSERT(a.getTags().getNumElements() == 1);
        UP_ASSERT(a.getRTags().getNumElements() == 1);
        UP_ASSERT(a.getBodies().getNumElements() == 1);

        MY_CHECK_CLOSE(h_pos.data[0].x, 1.0, tol);
        MY_CHECK_CLOSE(h_pos.data[0].y, 2.0, tol);
        MY_CHECK_CLOSE(h_pos.data[0].z, -2.0, tol);
        MY_CHECK_CLOSE(h_vel.data[0].x, 11.0, tol);
        MY_CHECK_CLOSE(h_vel.data[0].y, 12.0, tol);
        MY_CHECK_CLOSE(h_vel.data[0].z, 13.0, tol);
        MY_CHECK_CLOSE(h_accel.data[0].x, 21.0, tol);
        MY_CHECK_CLOSE(h_accel.data[0].y, 22.0, tol);
        MY_CHECK_CLOSE(h_accel.data[0].z, 23.0, tol);
        MY_CHECK_CLOSE(h_charge.data[0], 24.0, tol);
        MY_CHECK_CLOSE(h_vel.data[0].w, 25.0, tol); // mass
        MY_CHECK_CLOSE(h_diameter.data[0], 26.0, tol);
        UP_ASSERT_EQUAL(h_image.data[0].x, 27);
        UP_ASSERT_EQUAL(h_image.data[0].y, 28);
        UP_ASSERT_EQUAL(h_image.data[0].z, 29);
        UP_ASSERT(__scalar_as_int(h_pos.data[0].w) == 1); // type
        UP_ASSERT(h_rtag.data[0] == 0);
        UP_ASSERT(h_tag.data[0] == 0);
        UP_ASSERT(h_body.data[0] == 0);
        }
    // finally, lets check a larger ParticleData for correctness of the initialization
    const unsigned int N = 1000;
    ParticleData b(N, box, 1, exec_conf);
        {
        ArrayHandle<Scalar4> h_pos(b.getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_vel(b.getVelocities(), access_location::host, access_mode::read);
        ArrayHandle<Scalar3> h_accel(b.getAccelerations(),
                                     access_location::host,
                                     access_mode::read);
        ArrayHandle<int3> h_image(b.getImages(), access_location::host, access_mode::read);
        ArrayHandle<Scalar> h_charge(b.getCharges(), access_location::host, access_mode::read);
        ArrayHandle<Scalar> h_diameter(b.getDiameters(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_tag(b.getTags(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_rtag(b.getRTags(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_body(b.getBodies(), access_location::host, access_mode::read);

        // begin by verifying that the defaults the class advertises are set
        UP_ASSERT(b.getPositions().getNumElements() == N);
        UP_ASSERT(b.getVelocities().getNumElements() == N);
        UP_ASSERT(b.getAccelerations().getNumElements() == N);
        UP_ASSERT(b.getImages().getNumElements() == N);
        UP_ASSERT(b.getCharges().getNumElements() == N);
        UP_ASSERT(b.getDiameters().getNumElements() == N);
        UP_ASSERT(b.getTags().getNumElements() == N);
        UP_ASSERT(b.getRTags().size() == N);
        UP_ASSERT(b.getBodies().getNumElements() == N);

        for (unsigned int i = 0; i < N; i++)
            {
            MY_CHECK_CLOSE(h_pos.data[i].x, 0.0, tol);
            MY_CHECK_CLOSE(h_pos.data[i].y, 0.0, tol);
            MY_CHECK_CLOSE(h_pos.data[i].z, 0.0, tol);
            MY_CHECK_CLOSE(h_vel.data[i].x, 0.0, tol);
            MY_CHECK_CLOSE(h_vel.data[i].y, 0.0, tol);
            MY_CHECK_CLOSE(h_vel.data[i].z, 0.0, tol);
            MY_CHECK_CLOSE(h_accel.data[i].x, 0.0, tol);
            MY_CHECK_CLOSE(h_accel.data[i].y, 0.0, tol);
            MY_CHECK_CLOSE(h_accel.data[i].z, 0.0, tol);
            MY_CHECK_CLOSE(h_charge.data[i], 0.0, tol);
            MY_CHECK_CLOSE(h_vel.data[i].w, 1.0, tol); // mass
            MY_CHECK_CLOSE(h_diameter.data[i], 1.0, tol);
            UP_ASSERT_EQUAL(h_image.data[i].x, 0);
            UP_ASSERT_EQUAL(h_image.data[i].y, 0);
            UP_ASSERT_EQUAL(h_image.data[i].z, 0);
            UP_ASSERT(__scalar_as_int(h_pos.data[i].w) == 0); // type
            UP_ASSERT(h_rtag.data[i] == i);
            UP_ASSERT(h_tag.data[i] == i);
            UP_ASSERT(h_body.data[i] == NO_BODY);
            }
        }
    }

//! Tests the RandomParticleInitializer class
UP_TEST(Random_test)
    {
    // create a fairly dense system with a minimum distance of 0.8
    std::shared_ptr<ExecutionConfiguration> exec_conf(
        new ExecutionConfiguration(ExecutionConfiguration::CPU));
    Scalar min_dist = Scalar(0.8);
    RandomInitializer rand_init(500, Scalar(0.4), min_dist, "ABC");
    std::shared_ptr<SnapshotSystemData<Scalar>> snap = rand_init.getSnapshot();
    ParticleData pdata(snap->particle_data, snap->global_box, exec_conf);

    UP_ASSERT_EQUAL(pdata.getNameByType(0), "ABC");
    UP_ASSERT_EQUAL(pdata.getTypeByName("ABC"), (unsigned int)0);

        {
        ArrayHandle<Scalar4> h_pos(pdata.getPositions(), access_location::host, access_mode::read);

        // check that the distances between particles are OK
        BoxDim box = pdata.getBox();
        Scalar L = box.getL().x;
        for (unsigned int i = 0; i < pdata.getN(); i++)
            {
            UP_ASSERT(h_pos.data[i].x <= box.getHi().x && h_pos.data[i].x >= box.getLo().x);
            UP_ASSERT(h_pos.data[i].y <= box.getHi().y && h_pos.data[i].y >= box.getLo().y);
            UP_ASSERT(h_pos.data[i].z <= box.getHi().z && h_pos.data[i].z >= box.getLo().z);

            for (unsigned int j = 0; j < pdata.getN(); j++)
                {
                if (i == j)
                    continue;

                Scalar dx = h_pos.data[j].x - h_pos.data[i].x;
                Scalar dy = h_pos.data[j].y - h_pos.data[i].y;
                Scalar dz = h_pos.data[j].z - h_pos.data[i].z;

                if (dx < -L / Scalar(2.0))
                    dx += L;
                if (dx > L / Scalar(2.0))
                    dx -= L;

                if (dy < -L / Scalar(2.0))
                    dy += L;
                if (dy > L / Scalar(2.0))
                    dy -= L;

                if (dz < -L / Scalar(2.0))
                    dz += L;
                if (dz > L / Scalar(2.0))
                    dz -= L;

                Scalar dr2 = dx * dx + dy * dy + dz * dz;
                UP_ASSERT(dr2 >= min_dist * min_dist);
                }
            }
        }
    }

/*#include "RandomGenerator.h"
#include "MOL2DumpWriter.h"
UP_TEST( Generator_test )
    {
    vector<string> types;
    for (int i = 0; i < 6; i++)
        types.push_back("A");
    for (int i = 0; i < 7; i++)
        types.push_back("B");
    for (int i = 0; i < 6; i++)
        types.push_back("A");

    vector<string> types2;
    for (int i = 0; i < 7; i++)
        types2.push_back("B");

    std::shared_ptr<PolymerParticleGenerator> poly(new PolymerParticleGenerator(1.2, types, 100));
    std::shared_ptr<PolymerParticleGenerator> poly2(new PolymerParticleGenerator(1.2, types2, 100));
    BoxDim box(40);
    RandomGenerator generator(box, 1);
    generator.setSeparationRadius("A", 0.5);
    generator.setSeparationRadius("B", 0.5);
    generator.addGenerator(20, poly);
    generator.addGenerator(20, poly2);

    generator.generate();

    std::shared_ptr<ParticleData> pdata(new ParticleData(generator));
    MOL2DumpWriter dump(pdata, string("test.mol2"));
    dump.analyze(0);
    }*/
