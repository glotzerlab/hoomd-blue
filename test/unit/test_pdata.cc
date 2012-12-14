/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

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


/*! \file pdata_test.cc
    \brief Unit tests for BoxDim, ParticleData, SimpleCubicInitializer, and RandomInitializer classes.
    \ingroup unit_tests
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <iostream>

#include <boost/bind.hpp>

#include "ParticleData.h"
#include "Initializers.h"

using namespace std;
using namespace boost;

//! Name the boost unit test module
#define BOOST_TEST_MODULE ParticleDataTests
#include "boost_utf_configure.h"

//! Perform some basic tests on the boxdim structure
BOOST_AUTO_TEST_CASE( BoxDim_basic_test )
    {
    Scalar tol = Scalar(1e-6);
    
    // test default constructor
    BoxDim a;
    MY_BOOST_CHECK_CLOSE(a.getLo().x,0.0, tol);
    MY_BOOST_CHECK_CLOSE(a.getLo().y,0.0, tol);
    MY_BOOST_CHECK_CLOSE(a.getLo().z,0.0, tol);
    MY_BOOST_CHECK_CLOSE(a.getHi().x,0.0, tol);
    MY_BOOST_CHECK_CLOSE(a.getHi().y,0.0, tol);
    MY_BOOST_CHECK_CLOSE(a.getHi().z,0.0, tol);
    
    BoxDim b(10.0);
    MY_BOOST_CHECK_CLOSE(b.getLo().x,-5.0, tol);
    MY_BOOST_CHECK_CLOSE(b.getLo().y,-5.0, tol);
    MY_BOOST_CHECK_CLOSE(b.getLo().z,-5.0, tol);
    MY_BOOST_CHECK_CLOSE(b.getHi().x,5.0, tol);
    MY_BOOST_CHECK_CLOSE(b.getHi().y,5.0, tol);
    MY_BOOST_CHECK_CLOSE(b.getHi().z,5.0, tol);
    MY_BOOST_CHECK_CLOSE(b.getL().x,10.0, tol);
    MY_BOOST_CHECK_CLOSE(b.getL().y,10.0, tol);
    MY_BOOST_CHECK_CLOSE(b.getL().z,10.0, tol);
    BOOST_CHECK_EQUAL(b.getPeriodic().x, 1);
    BOOST_CHECK_EQUAL(b.getPeriodic().y, 1);
    BOOST_CHECK_EQUAL(b.getPeriodic().z, 1);

    BoxDim c(10.0, 30.0, 50.0);
    MY_BOOST_CHECK_CLOSE(c.getLo().x,-5.0, tol);
    MY_BOOST_CHECK_CLOSE(c.getLo().y,-15.0, tol);
    MY_BOOST_CHECK_CLOSE(c.getLo().z,-25.0, tol);
    MY_BOOST_CHECK_CLOSE(c.getHi().x,5.0, tol);
    MY_BOOST_CHECK_CLOSE(c.getHi().y,15.0, tol);
    MY_BOOST_CHECK_CLOSE(c.getHi().z,25.0, tol);
    MY_BOOST_CHECK_CLOSE(c.getL().x,10.0, tol);
    MY_BOOST_CHECK_CLOSE(c.getL().y,30.0, tol);
    MY_BOOST_CHECK_CLOSE(c.getL().z,50.0, tol);
    BOOST_CHECK_EQUAL(c.getPeriodic().x, 1);
    BOOST_CHECK_EQUAL(c.getPeriodic().y, 1);
    BOOST_CHECK_EQUAL(c.getPeriodic().z, 1);

    // test for assignment and copy constructor
    c.setPeriodic(make_uchar3(1,0,1));
    BoxDim d(c);
    MY_BOOST_CHECK_CLOSE(d.getLo().x,-5.0, tol);
    MY_BOOST_CHECK_CLOSE(d.getLo().y,-15.0, tol);
    MY_BOOST_CHECK_CLOSE(d.getLo().z,-25.0, tol);
    MY_BOOST_CHECK_CLOSE(d.getHi().x,5.0, tol);
    MY_BOOST_CHECK_CLOSE(d.getHi().y,15.0, tol);
    MY_BOOST_CHECK_CLOSE(d.getHi().z,25.0, tol);
    MY_BOOST_CHECK_CLOSE(d.getL().x,10.0, tol);
    MY_BOOST_CHECK_CLOSE(d.getL().y,30.0, tol);
    MY_BOOST_CHECK_CLOSE(d.getL().z,50.0, tol);
    BOOST_CHECK_EQUAL(d.getPeriodic().x, 1);
    BOOST_CHECK_EQUAL(d.getPeriodic().y, 0);
    BOOST_CHECK_EQUAL(d.getPeriodic().z, 1);

    BoxDim e;
    e = c;
    MY_BOOST_CHECK_CLOSE(e.getLo().x,-5.0, tol);
    MY_BOOST_CHECK_CLOSE(e.getLo().y,-15.0, tol);
    MY_BOOST_CHECK_CLOSE(e.getLo().z,-25.0, tol);
    MY_BOOST_CHECK_CLOSE(e.getHi().x,5.0, tol);
    MY_BOOST_CHECK_CLOSE(e.getHi().y,15.0, tol);
    MY_BOOST_CHECK_CLOSE(e.getHi().z,25.0, tol);
    MY_BOOST_CHECK_CLOSE(e.getL().x,10.0, tol);
    MY_BOOST_CHECK_CLOSE(e.getL().y,30.0, tol);
    MY_BOOST_CHECK_CLOSE(e.getL().z,50.0, tol);
    BOOST_CHECK_EQUAL(d.getPeriodic().x, 1);
    BOOST_CHECK_EQUAL(d.getPeriodic().y, 0);
    BOOST_CHECK_EQUAL(d.getPeriodic().z, 1);

    b = b;
    MY_BOOST_CHECK_CLOSE(b.getLo().x,-5.0, tol);
    MY_BOOST_CHECK_CLOSE(b.getLo().y,-5.0, tol);
    MY_BOOST_CHECK_CLOSE(b.getLo().z,-5.0, tol);
    MY_BOOST_CHECK_CLOSE(b.getHi().x,5.0, tol);
    MY_BOOST_CHECK_CLOSE(b.getHi().y,5.0, tol);
    MY_BOOST_CHECK_CLOSE(b.getHi().z,5.0, tol);
    MY_BOOST_CHECK_CLOSE(b.getL().x,10.0, tol);
    MY_BOOST_CHECK_CLOSE(b.getL().y,10.0, tol);
    MY_BOOST_CHECK_CLOSE(b.getL().z,10.0, tol);
    BOOST_CHECK_EQUAL(b.getPeriodic().x, 1);
    BOOST_CHECK_EQUAL(b.getPeriodic().y, 1);
    BOOST_CHECK_EQUAL(b.getPeriodic().z, 1);
    }

BOOST_AUTO_TEST_CASE( BoxDim_functionality_test1 )
    {
    BoxDim b(5.0);
    MY_BOOST_CHECK_CLOSE(b.getL().x,5.0, tol);
    MY_BOOST_CHECK_CLOSE(b.getL().y,5.0, tol);
    MY_BOOST_CHECK_CLOSE(b.getL().z,5.0, tol);

    b.setL(make_scalar3(10.0, 10.0, 10.0));
    MY_BOOST_CHECK_CLOSE(b.getL().x,10.0, tol);
    MY_BOOST_CHECK_CLOSE(b.getL().y,10.0, tol);
    MY_BOOST_CHECK_CLOSE(b.getL().z,10.0, tol);

    Scalar3 v = b.makeFraction(make_scalar3(5.0, 5.0, 5.0));
    MY_BOOST_CHECK_CLOSE(v.x,1.0, tol);
    MY_BOOST_CHECK_CLOSE(v.y,1.0, tol);
    MY_BOOST_CHECK_CLOSE(v.z,1.0, tol);

    v = b.makeFraction(make_scalar3(1.0, -2.0, 3.0));
    MY_BOOST_CHECK_CLOSE(v.x,0.6, tol);
    MY_BOOST_CHECK_CLOSE(v.y,0.3, tol);
    MY_BOOST_CHECK_CLOSE(v.z,0.8, tol);

    v = b.minImage(make_scalar3(1.0, -2.0, 3.0));
    MY_BOOST_CHECK_CLOSE(v.x,1.0, tol);
    MY_BOOST_CHECK_CLOSE(v.y,-2.0, tol);
    MY_BOOST_CHECK_CLOSE(v.z,3.0, tol);

    // test minimum image
    v = b.minImage(make_scalar3(6.0, -7.0, 8.0));
    MY_BOOST_CHECK_CLOSE(v.x,-4.0, tol);
    MY_BOOST_CHECK_CLOSE(v.y,3.0, tol);
    MY_BOOST_CHECK_CLOSE(v.z,-2.0, tol);

    b.setPeriodic(make_uchar3(0,0,0));
    v = b.minImage(make_scalar3(6.0, -7.0, 8.0));
    MY_BOOST_CHECK_CLOSE(v.x,6.0, tol);
    MY_BOOST_CHECK_CLOSE(v.y,-7.0, tol);
    MY_BOOST_CHECK_CLOSE(v.z,8.0, tol);

    b.setPeriodic(make_uchar3(1,0,0));
    v = b.minImage(make_scalar3(6.0, -7.0, 8.0));
    MY_BOOST_CHECK_CLOSE(v.x,-4.0, tol);
    MY_BOOST_CHECK_CLOSE(v.y,-7.0, tol);
    MY_BOOST_CHECK_CLOSE(v.z,8.0, tol);

    b.setPeriodic(make_uchar3(0,1,0));
    v = b.minImage(make_scalar3(6.0, -7.0, 8.0));
    MY_BOOST_CHECK_CLOSE(v.x,6.0, tol);
    MY_BOOST_CHECK_CLOSE(v.y,3.0, tol);
    MY_BOOST_CHECK_CLOSE(v.z,8.0, tol);

    b.setPeriodic(make_uchar3(0,0,1));
    v = b.minImage(make_scalar3(6.0, -7.0, 8.0));
    MY_BOOST_CHECK_CLOSE(v.x,6.0, tol);
    MY_BOOST_CHECK_CLOSE(v.y,-7.0, tol);
    MY_BOOST_CHECK_CLOSE(v.z,-2.0, tol);

    // test wrap
    b.setPeriodic(make_uchar3(1,1,1));
    int3 image = make_int3(10,20,30);
    v = make_scalar3(1.0, -2.0, 3.0);
    b.wrap(v, image);
    MY_BOOST_CHECK_CLOSE(v.x,1.0, tol);
    MY_BOOST_CHECK_CLOSE(v.y,-2.0, tol);
    MY_BOOST_CHECK_CLOSE(v.z,3.0, tol);
    BOOST_CHECK_EQUAL(image.x, 10);
    BOOST_CHECK_EQUAL(image.y, 20);
    BOOST_CHECK_EQUAL(image.z, 30);

    image = make_int3(10,20,30);
    v = make_scalar3(6.0, -7.0, 8.0);
    b.wrap(v, image);
    MY_BOOST_CHECK_CLOSE(v.x,-4.0, tol);
    MY_BOOST_CHECK_CLOSE(v.y,3.0, tol);
    MY_BOOST_CHECK_CLOSE(v.z,-2.0, tol);
    BOOST_CHECK_EQUAL(image.x, 11);
    BOOST_CHECK_EQUAL(image.y, 19);
    BOOST_CHECK_EQUAL(image.z, 31);

    b.setPeriodic(make_uchar3(1,0,0));
    image = make_int3(10,20,30);
    v = make_scalar3(6.0, -7.0, 8.0);
    b.wrap(v, image);
    MY_BOOST_CHECK_CLOSE(v.x,-4.0, tol);
    MY_BOOST_CHECK_CLOSE(v.y,-7.0, tol);
    MY_BOOST_CHECK_CLOSE(v.z,8.0, tol);
    BOOST_CHECK_EQUAL(image.x, 11);
    BOOST_CHECK_EQUAL(image.y, 20);
    BOOST_CHECK_EQUAL(image.z, 30);

    b.setPeriodic(make_uchar3(0,1,0));
    image = make_int3(10,20,30);
    v = make_scalar3(6.0, -7.0, 8.0);
    b.wrap(v, image);
    MY_BOOST_CHECK_CLOSE(v.x,6.0, tol);
    MY_BOOST_CHECK_CLOSE(v.y,3.0, tol);
    MY_BOOST_CHECK_CLOSE(v.z,8.0, tol);
    BOOST_CHECK_EQUAL(image.x, 10);
    BOOST_CHECK_EQUAL(image.y, 19);
    BOOST_CHECK_EQUAL(image.z, 30);

    b.setPeriodic(make_uchar3(0,0,1));
    image = make_int3(10,20,30);
    v = make_scalar3(6.0, -7.0, 8.0);
    b.wrap(v, image);
    MY_BOOST_CHECK_CLOSE(v.x,6.0, tol);
    MY_BOOST_CHECK_CLOSE(v.y,-7.0, tol);
    MY_BOOST_CHECK_CLOSE(v.z,-2.0, tol);
    BOOST_CHECK_EQUAL(image.x, 10);
    BOOST_CHECK_EQUAL(image.y, 20);
    BOOST_CHECK_EQUAL(image.z, 31);
    }

BOOST_AUTO_TEST_CASE( BoxDim_triclinic_test )
    {
    BoxDim b(5.0);
    
    Scalar tol = Scalar(1e-4);

    Scalar xy = 1.0;
    Scalar xz = .4;
    Scalar yz = .9;

    b.setTiltFactors(xy,xz,yz);
    MY_BOOST_CHECK_CLOSE(b.getTiltFactorXY(), xy,tol);
    MY_BOOST_CHECK_CLOSE(b.getTiltFactorXZ(), xz,tol);
    MY_BOOST_CHECK_CLOSE(b.getTiltFactorYZ(), yz,tol);

    Scalar3 f = make_scalar3(.5,.6,.7);
    Scalar3 L = b.getL();
    
    Scalar3 pos;
    pos.x = b.getLo().x + f.x * L.x + xy * L.y * (f.y-Scalar(0.5)) + xz * L.z * (f.z-Scalar(0.5));
    pos.y = b.getLo().y + f.y * L.y + yz * L.z * (f.z-Scalar(0.5));
    pos.z = b.getLo().z + f.z * L.z;

    // convert pos to fraction
    Scalar3 f2 = b.makeFraction(pos);

    MY_BOOST_CHECK_CLOSE(f2.x,f.x,tol);
    MY_BOOST_CHECK_CLOSE(f2.y,f.y,tol);
    MY_BOOST_CHECK_CLOSE(f2.z,f.z,tol);

    // convert fraction to pos
    Scalar3 pos2 = b.makeCoordinates(f);

    MY_BOOST_CHECK_CLOSE(pos2.x, pos.x,tol);
    MY_BOOST_CHECK_CLOSE(pos2.y, pos.y,tol);
    MY_BOOST_CHECK_CLOSE(pos2.z, pos.z,tol);

    // test minimum image

    // along x coordinate
    Scalar3 dx = make_scalar3(3.0,1.0,2.0);
    Scalar3 dx2 = b.minImage(dx);   

    MY_BOOST_CHECK_CLOSE(dx2.x, -2.0, tol);
    MY_BOOST_CHECK_CLOSE(dx2.y, 1.0, tol);
    MY_BOOST_CHECK_CLOSE(dx2.z, 2.0, tol);

    dx = make_scalar3(-3.0,1.0,2.0);
    dx2 = b.minImage(dx);

    MY_BOOST_CHECK_CLOSE(dx2.x, 2.0, tol);
    MY_BOOST_CHECK_CLOSE(dx2.y, 1.0, tol);
    MY_BOOST_CHECK_CLOSE(dx2.z, 2.0, tol);

    // along y coordinate
    dx = make_scalar3(2.0,2.6,1.5);
    dx2 = b.minImage(dx);

    MY_BOOST_CHECK_CLOSE(dx2.x, 2.0, tol);
    MY_BOOST_CHECK_CLOSE(dx2.y, -2.4, tol);
    MY_BOOST_CHECK_CLOSE(dx2.z, 1.5, tol);

    dx = make_scalar3(2.0,-2.6,1.5);
    dx2 = b.minImage(dx);

    MY_BOOST_CHECK_CLOSE(dx2.x, 2.0, tol);
    MY_BOOST_CHECK_CLOSE(dx2.y, 2.4, tol);
    MY_BOOST_CHECK_CLOSE(dx2.z, 1.5, tol);

    dx = make_scalar3(3.0,2.6,1.5);
    dx2 = b.minImage(dx);

    MY_BOOST_CHECK_CLOSE(dx2.x, -2.0, tol);
    MY_BOOST_CHECK_CLOSE(dx2.y, -2.4, tol);
    MY_BOOST_CHECK_CLOSE(dx2.z, 1.5, tol);

    dx = make_scalar3(3.0,-2.6,1.5);
    dx2 = b.minImage(dx);

    MY_BOOST_CHECK_CLOSE(dx2.x,-2.0,tol);
    MY_BOOST_CHECK_CLOSE(dx2.y,2.4,tol);
    MY_BOOST_CHECK_CLOSE(dx.z, 1.5,tol);

    // along z coordinate
    dx = make_scalar3(2.1,1.5,3.0);
    dx2 = b.minImage(dx);

    MY_BOOST_CHECK_CLOSE(dx2.x, 0.1, tol);
    MY_BOOST_CHECK_CLOSE(dx2.y, 2.0 ,tol);
    MY_BOOST_CHECK_CLOSE(dx2.z, -2.0,tol);

    dx = make_scalar3(2.1,1.5,-3.0);
    dx2 = b.minImage(dx);

    MY_BOOST_CHECK_CLOSE(dx2.x, -0.9, tol);
    MY_BOOST_CHECK_CLOSE(dx2.y, 1.0, tol);
    MY_BOOST_CHECK_CLOSE(dx2.z, 2.0, tol);

    // test particle wrap

    // along z direction
    pos = make_scalar3(1.0,2.0,2.6);
    int3 img = make_int3(0,0,0);

    b.wrap(pos, img);
    MY_BOOST_CHECK_CLOSE(pos.x, -1.0 ,tol);
    MY_BOOST_CHECK_CLOSE(pos.y, -2.5,tol);
    MY_BOOST_CHECK_CLOSE(pos.z, -2.4,tol);
    BOOST_CHECK_EQUAL(img.x, 0);
    BOOST_CHECK_EQUAL(img.y, 0);
    BOOST_CHECK_EQUAL(img.z, 1);

    pos = make_scalar3(-1.0,-2.0,-2.6);
    img = make_int3(0,0,0);
    b.wrap(pos,img);

    MY_BOOST_CHECK_CLOSE(pos.x, 1.0 ,tol);
    MY_BOOST_CHECK_CLOSE(pos.y, 2.5,tol);
    MY_BOOST_CHECK_CLOSE(pos.z, 2.4,tol);
    BOOST_CHECK_EQUAL(img.x, 0);
    BOOST_CHECK_EQUAL(img.y, 0);
    BOOST_CHECK_EQUAL(img.z, -1);

    // along y direction
    pos = make_scalar3(1.0,4.0,1.5);
    img = make_int3(0,0,0);
    
    b.wrap(pos, img);
    MY_BOOST_CHECK_CLOSE(pos.x, -4.0,tol);
    MY_BOOST_CHECK_CLOSE(pos.y, -1.0,tol);
    MY_BOOST_CHECK_CLOSE(pos.z, 1.5,tol);

    BOOST_CHECK_EQUAL(img.x, 0);
    BOOST_CHECK_EQUAL(img.y, 1);
    BOOST_CHECK_EQUAL(img.z, 0);

    pos = make_scalar3(-1.0,-4.0,-1.5);
    img = make_int3(0,0,0);

    b.wrap(pos, img);
    MY_BOOST_CHECK_CLOSE(pos.x, 4.0, tol);
    MY_BOOST_CHECK_CLOSE(pos.y, 1.0, tol);
    MY_BOOST_CHECK_CLOSE(pos.z, -1.5, tol);

    BOOST_CHECK_EQUAL(img.x, 0);
    BOOST_CHECK_EQUAL(img.y, -1);
    BOOST_CHECK_EQUAL(img.z, 0);

    // along x direction
    pos = make_scalar3(4.2,1.5, 1.0);
    img = make_int3(0,0,0);
    
    b.wrap(pos, img);
    MY_BOOST_CHECK_CLOSE(pos.x, -0.8, tol);
    MY_BOOST_CHECK_CLOSE(pos.y, 1.5, tol);
    MY_BOOST_CHECK_CLOSE(pos.z, 1.0, tol);

    BOOST_CHECK_EQUAL(img.x, 1);
    BOOST_CHECK_EQUAL(img.y, 0);
    BOOST_CHECK_EQUAL(img.z, 0);

    pos = make_scalar3(-5.0,-1.5, 1.0);
    img = make_int3(0,0,0);

    b.wrap(pos,img);
    MY_BOOST_CHECK_CLOSE(pos.x, 0.0, tol);
    MY_BOOST_CHECK_CLOSE(pos.y, -1.5, tol);
    MY_BOOST_CHECK_CLOSE(pos.z, 1.0, tol);

    BOOST_CHECK_EQUAL(img.x, -1);
    BOOST_CHECK_EQUAL(img.y, 0);
    BOOST_CHECK_EQUAL(img.z, 0);
    }

//! Test operation of the particle data class
BOOST_AUTO_TEST_CASE( ParticleData_test )
    {
    BoxDim box(10.0, 30.0, 50.0);
    boost::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::CPU));
    ParticleData a(1, box, 1, exec_conf);
    
    Scalar tol = Scalar(1e-6);
    
    // make sure the box is working
    const BoxDim& c = a.getBox();
    MY_BOOST_CHECK_CLOSE(c.getLo().x,-5.0, tol);
    MY_BOOST_CHECK_CLOSE(c.getLo().y,-15.0, tol);
    MY_BOOST_CHECK_CLOSE(c.getLo().z,-25.0, tol);
    MY_BOOST_CHECK_CLOSE(c.getHi().x,5.0, tol);
    MY_BOOST_CHECK_CLOSE(c.getHi().y,15.0, tol);
    MY_BOOST_CHECK_CLOSE(c.getHi().z,25.0, tol);
    
    a.setGlobalBoxL(make_scalar3(5.0, 5.0, 5.0));
    const BoxDim& d = a.getBox();
    MY_BOOST_CHECK_CLOSE(d.getLo().x,-2.5, tol);
    MY_BOOST_CHECK_CLOSE(d.getLo().y,-2.5, tol);
    MY_BOOST_CHECK_CLOSE(d.getLo().z,-2.5, tol);
    MY_BOOST_CHECK_CLOSE(d.getHi().x,2.5, tol);
    MY_BOOST_CHECK_CLOSE(d.getHi().y,2.5, tol);
    MY_BOOST_CHECK_CLOSE(d.getHi().z,2.5, tol);
    
    // make sure that getN is working
    BOOST_CHECK(a.getN() == 1);
    
    // Test the ability to acquire data
    {
    ArrayHandle<Scalar4> h_pos(a.getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(a.getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(a.getAccelerations(), access_location::host, access_mode::readwrite);
    ArrayHandle<int3> h_image(a.getImages(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_charge(a.getCharges(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_diameter(a.getDiameters(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_tag(a.getTags(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_rtag(a.getRTags(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_body(a.getBodies(), access_location::host, access_mode::readwrite);

    // begin by verifying that the defaults the class adversizes are set
    BOOST_CHECK(a.getPositions().getNumElements() == 1);
    BOOST_CHECK(a.getVelocities().getNumElements() == 1);
    BOOST_CHECK(a.getAccelerations().getNumElements() == 1);
    BOOST_CHECK(a.getImages().getNumElements() == 1);
    BOOST_CHECK(a.getCharges().getNumElements() == 1);
    BOOST_CHECK(a.getDiameters().getNumElements() == 1);
    BOOST_CHECK(a.getTags().getNumElements() == 1);
    BOOST_CHECK(a.getRTags().getNumElements() == 1);
    BOOST_CHECK(a.getBodies().getNumElements() == 1);


    MY_BOOST_CHECK_CLOSE(h_pos.data[0].x, 0.0, tol);
    MY_BOOST_CHECK_CLOSE(h_pos.data[0].y, 0.0, tol);
    MY_BOOST_CHECK_CLOSE(h_pos.data[0].z, 0.0, tol);
    MY_BOOST_CHECK_CLOSE(h_vel.data[0].x, 0.0, tol);
    MY_BOOST_CHECK_CLOSE(h_vel.data[0].y, 0.0, tol);
    MY_BOOST_CHECK_CLOSE(h_vel.data[0].z, 0.0, tol);
    MY_BOOST_CHECK_CLOSE(h_accel.data[0].x, 0.0, tol);
    MY_BOOST_CHECK_CLOSE(h_accel.data[0].y, 0.0, tol);
    MY_BOOST_CHECK_CLOSE(h_accel.data[0].z, 0.0, tol);
    MY_BOOST_CHECK_CLOSE(h_charge.data[0], 0.0, tol);
    MY_BOOST_CHECK_CLOSE(h_vel.data[0].w, 1.0, tol); // mass
    MY_BOOST_CHECK_CLOSE(h_diameter.data[0], 1.0, tol);
    BOOST_CHECK_EQUAL(h_image.data[0].x, 0);
    BOOST_CHECK_EQUAL(h_image.data[0].y, 0);
    BOOST_CHECK_EQUAL(h_image.data[0].z, 0);
    BOOST_CHECK(__scalar_as_int(h_pos.data[0].w) == 0); //type
    BOOST_CHECK(h_rtag.data[0] == 0);
    BOOST_CHECK(h_tag.data[0] == 0);
    BOOST_CHECK(h_body.data[0] == NO_BODY);
    
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
    h_image.data[0].x =  27;
    h_image.data[0].y = 28;
    h_image.data[0].z = 29;
    h_pos.data[0].w = __int_as_scalar(1); //type
    h_body.data[0] = 0;
    
    }
    
    // make sure when the data is re-acquired, the values read properly
    {
    ArrayHandle<Scalar4> h_pos(a.getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_vel(a.getVelocities(), access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_accel(a.getAccelerations(), access_location::host, access_mode::read);
    ArrayHandle<int3> h_image(a.getImages(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_charge(a.getCharges(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_diameter(a.getDiameters(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(a.getTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(a.getRTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_body(a.getBodies(), access_location::host, access_mode::read);

    BOOST_CHECK(a.getPositions().getNumElements() == 1);
    BOOST_CHECK(a.getVelocities().getNumElements() == 1);
    BOOST_CHECK(a.getAccelerations().getNumElements() == 1);
    BOOST_CHECK(a.getImages().getNumElements() == 1);
    BOOST_CHECK(a.getCharges().getNumElements() == 1);
    BOOST_CHECK(a.getDiameters().getNumElements() == 1);
    BOOST_CHECK(a.getTags().getNumElements() == 1);
    BOOST_CHECK(a.getRTags().getNumElements() == 1);
    BOOST_CHECK(a.getBodies().getNumElements() == 1);

    MY_BOOST_CHECK_CLOSE(h_pos.data[0].x, 1.0, tol);
    MY_BOOST_CHECK_CLOSE(h_pos.data[0].y, 2.0, tol);
    MY_BOOST_CHECK_CLOSE(h_pos.data[0].z,-2.0, tol);
    MY_BOOST_CHECK_CLOSE(h_vel.data[0].x,11.0, tol);
    MY_BOOST_CHECK_CLOSE(h_vel.data[0].y,12.0, tol);
    MY_BOOST_CHECK_CLOSE(h_vel.data[0].z,13.0, tol);
    MY_BOOST_CHECK_CLOSE(h_accel.data[0].x,21.0, tol);
    MY_BOOST_CHECK_CLOSE(h_accel.data[0].y,22.0, tol);
    MY_BOOST_CHECK_CLOSE(h_accel.data[0].z,23.0, tol);
    MY_BOOST_CHECK_CLOSE(h_charge.data[0],24.0, tol);
    MY_BOOST_CHECK_CLOSE(h_vel.data[0].w,25.0, tol); // mass
    MY_BOOST_CHECK_CLOSE(h_diameter.data[0],26.0, tol);
    BOOST_CHECK_EQUAL(h_image.data[0].x,27);
    BOOST_CHECK_EQUAL(h_image.data[0].y,28);
    BOOST_CHECK_EQUAL(h_image.data[0].z,29);
    BOOST_CHECK(__scalar_as_int(h_pos.data[0].w) == 1); //type
    BOOST_CHECK(h_rtag.data[0] == 0);
    BOOST_CHECK(h_tag.data[0] == 0);
    BOOST_CHECK(h_body.data[0] == 0);

    }
    // finally, lets check a larger ParticleData for correctness of the initialization
    const unsigned int N = 1000;
    ParticleData b(N, box, 1, exec_conf);
    {
    ArrayHandle<Scalar4> h_pos(b.getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_vel(b.getVelocities(), access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_accel(b.getAccelerations(), access_location::host, access_mode::read);
    ArrayHandle<int3> h_image(b.getImages(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_charge(b.getCharges(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_diameter(b.getDiameters(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(b.getTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(b.getRTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_body(b.getBodies(), access_location::host, access_mode::read);

    // begin by verifying that the defaults the class adversizes are set
    BOOST_CHECK(b.getPositions().getNumElements() == N);
    BOOST_CHECK(b.getVelocities().getNumElements() == N);
    BOOST_CHECK(b.getAccelerations().getNumElements() == N);
    BOOST_CHECK(b.getImages().getNumElements() == N);
    BOOST_CHECK(b.getCharges().getNumElements() == N);
    BOOST_CHECK(b.getDiameters().getNumElements() == N);
    BOOST_CHECK(b.getTags().getNumElements() == N);
    BOOST_CHECK(b.getRTags().getNumElements() == N);
    BOOST_CHECK(b.getBodies().getNumElements() == N);


    for (unsigned int i = 0; i < N; i++)
        {
        MY_BOOST_CHECK_CLOSE(h_pos.data[i].x, 0.0, tol);
        MY_BOOST_CHECK_CLOSE(h_pos.data[i].y, 0.0, tol);
        MY_BOOST_CHECK_CLOSE(h_pos.data[i].z, 0.0, tol);
        MY_BOOST_CHECK_CLOSE(h_vel.data[i].x, 0.0, tol);
        MY_BOOST_CHECK_CLOSE(h_vel.data[i].y, 0.0, tol);
        MY_BOOST_CHECK_CLOSE(h_vel.data[i].z, 0.0, tol);
        MY_BOOST_CHECK_CLOSE(h_accel.data[i].x, 0.0, tol);
        MY_BOOST_CHECK_CLOSE(h_accel.data[i].y, 0.0, tol);
        MY_BOOST_CHECK_CLOSE(h_accel.data[i].z, 0.0, tol);
        MY_BOOST_CHECK_CLOSE(h_charge.data[i], 0.0, tol);
        MY_BOOST_CHECK_CLOSE(h_vel.data[i].w, 1.0, tol); // mass
        MY_BOOST_CHECK_CLOSE(h_diameter.data[i], 1.0, tol);
        BOOST_CHECK_EQUAL(h_image.data[i].x, 0);
        BOOST_CHECK_EQUAL(h_image.data[i].y, 0);
        BOOST_CHECK_EQUAL(h_image.data[i].z, 0);
        BOOST_CHECK(__scalar_as_int(h_pos.data[i].w) == 0); //type
        BOOST_CHECK(h_rtag.data[i] == i);
        BOOST_CHECK(h_tag.data[i] == i);
        BOOST_CHECK(h_body.data[i] == NO_BODY);
        }
        
    }
    }

//! Test operation of the simple cubic initializer class
BOOST_AUTO_TEST_CASE( SimpleCubic_test )
    {
    Scalar tol = Scalar(1e-6);
    
    // make a simple one-particle box
    boost::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::CPU));
    SimpleCubicInitializer one(1, 2.0, "ABC");
    ParticleData one_data(one, exec_conf);

    BOOST_CHECK(one_data.getN() == 1);
    {
    ArrayHandle<Scalar4> h_pos(one_data.getPositions(), access_location::host, access_mode::read);

    MY_BOOST_CHECK_CLOSE(h_pos.data[0].x, -1.0, tol);
    MY_BOOST_CHECK_CLOSE(h_pos.data[0].y, -1.0, tol);
    MY_BOOST_CHECK_CLOSE(h_pos.data[0].z, -1.0, tol);
    }
    
    BOOST_CHECK_EQUAL(one_data.getNameByType(0), "ABC");
    BOOST_CHECK_EQUAL(one_data.getTypeByName("ABC"), (unsigned int)0);
    
    // now try an 8-particle one
    SimpleCubicInitializer eight(2, 2.0, "A");
    ParticleData eight_data(eight, exec_conf);
    
    BOOST_CHECK(eight_data.getN() == 8);
    {
    ArrayHandle<Scalar4> h_pos(eight_data.getPositions(), access_location::host, access_mode::read);
    MY_BOOST_CHECK_CLOSE(h_pos.data[0].x, -2.0, tol);
    MY_BOOST_CHECK_CLOSE(h_pos.data[0].y, -2.0, tol);
    MY_BOOST_CHECK_CLOSE(h_pos.data[0].z, -2.0, tol);
    MY_BOOST_CHECK_CLOSE(h_pos.data[1].x, 0.0, tol);
    MY_BOOST_CHECK_CLOSE(h_pos.data[1].y, -2.0, tol);
    MY_BOOST_CHECK_CLOSE(h_pos.data[1].z, -2.0, tol);
    MY_BOOST_CHECK_CLOSE(h_pos.data[2].x, -2.0, tol);
    MY_BOOST_CHECK_CLOSE(h_pos.data[2].y, 0.0, tol);
    MY_BOOST_CHECK_CLOSE(h_pos.data[2].z, -2.0, tol);
    MY_BOOST_CHECK_CLOSE(h_pos.data[3].x, 0.0, tol);
    MY_BOOST_CHECK_CLOSE(h_pos.data[3].y, 0.0, tol);
    MY_BOOST_CHECK_CLOSE(h_pos.data[3].z, -2.0, tol);
    
    MY_BOOST_CHECK_CLOSE(h_pos.data[4].x, -2.0, tol);
    MY_BOOST_CHECK_CLOSE(h_pos.data[4].y, -2.0, tol);
    MY_BOOST_CHECK_CLOSE(h_pos.data[4].z, 0.0, tol);
    MY_BOOST_CHECK_CLOSE(h_pos.data[5].x, 0.0, tol);
    MY_BOOST_CHECK_CLOSE(h_pos.data[5].y, -2.0, tol);
    MY_BOOST_CHECK_CLOSE(h_pos.data[5].z, 0.0, tol);
    MY_BOOST_CHECK_CLOSE(h_pos.data[6].x, -2.0, tol);
    MY_BOOST_CHECK_CLOSE(h_pos.data[6].y, 0.0, tol);
    MY_BOOST_CHECK_CLOSE(h_pos.data[6].z, 0.0, tol);
    MY_BOOST_CHECK_CLOSE(h_pos.data[7].x, 0.0, tol);
    MY_BOOST_CHECK_CLOSE(h_pos.data[7].y, 0.0, tol);
    MY_BOOST_CHECK_CLOSE(h_pos.data[7].z, 0.0, tol);
    }
    }

//! Tests the RandomParticleInitializer class
BOOST_AUTO_TEST_CASE( Random_test )
    {
    // create a fairly dense system with a minimum distance of 0.8
    boost::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::CPU));
    Scalar min_dist = Scalar(0.8);
    RandomInitializer rand_init(500, Scalar(0.4), min_dist, "ABC");
    ParticleData pdata(rand_init, exec_conf);
    
    BOOST_CHECK_EQUAL(pdata.getNameByType(0), "ABC");
    BOOST_CHECK_EQUAL(pdata.getTypeByName("ABC"), (unsigned int)0);
    
    {
    ArrayHandle<Scalar4> h_pos(pdata.getPositions(), access_location::host, access_mode::read);
    
    // check that the distances between particles are OK
    BoxDim box = pdata.getBox();
    Scalar L = box.getL().x;
    for (unsigned int i = 0; i < pdata.getN(); i++)
        {
        BOOST_CHECK(h_pos.data[i].x <= box.getHi().x && h_pos.data[i].x >= box.getLo().x);
        BOOST_CHECK(h_pos.data[i].y <= box.getHi().y && h_pos.data[i].y >= box.getLo().y);
        BOOST_CHECK(h_pos.data[i].z <= box.getHi().z && h_pos.data[i].z >= box.getLo().z);
        
        for (unsigned int j = 0; j < pdata.getN(); j++)
            {
            if (i == j)
                continue;
                
            Scalar dx = h_pos.data[j].x - h_pos.data[i].x;
            Scalar dy = h_pos.data[j].y - h_pos.data[i].y;
            Scalar dz = h_pos.data[j].z - h_pos.data[i].z;
            
            if (dx < -L/Scalar(2.0))
                dx += L;
            if (dx > L/Scalar(2.0))
                dx -= L;
                
            if (dy < -L/Scalar(2.0))
                dy += L;
            if (dy > L/Scalar(2.0))
                dy -= L;
                
            if (dz < -L/Scalar(2.0))
                dz += L;
            if (dz > L/Scalar(2.0))
                dz -= L;
                
            Scalar dr2 = dx*dx + dy*dy + dz*dz;
            BOOST_CHECK(dr2 >= min_dist*min_dist);
            }
        }
        
    }
    }

/*#include "RandomGenerator.h"
#include "MOL2DumpWriter.h"
BOOST_AUTO_TEST_CASE( Generator_test )
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

    boost::shared_ptr<PolymerParticleGenerator> poly(new PolymerParticleGenerator(1.2, types, 100));
    boost::shared_ptr<PolymerParticleGenerator> poly2(new PolymerParticleGenerator(1.2, types2, 100));
    BoxDim box(40);
    RandomGenerator generator(box, 1);
    generator.setSeparationRadius("A", 0.5);
    generator.setSeparationRadius("B", 0.5);
    generator.addGenerator(20, poly);
    generator.addGenerator(20, poly2);

    generator.generate();

    boost::shared_ptr<ParticleData> pdata(new ParticleData(generator));
    MOL2DumpWriter dump(pdata, string("test.mol2"));
    dump.analyze(0);
    }*/

#ifdef WIN32
#pragma warning( pop )
#endif

