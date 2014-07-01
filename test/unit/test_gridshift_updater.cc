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


#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <iostream>

#include <boost/shared_ptr.hpp>

#include "ZeroMomentumUpdater.h"

#include <math.h>

using namespace std;
using namespace boost;

//! label the boost test module
#define BOOST_TEST_MODULE ZeroMomentumUpdaterTests
#include "boost_utf_configure.h"

/*! \file zero_momentum_updater_test.cc
    \brief Unit tests for the ZeroMomentumUpdater class
    \ingroup unit_tests
*/

//! boost test case to verify proper operation of ZeroMomentumUpdater
BOOST_AUTO_TEST_CASE( ParticleDataGridShiftGetMethods )
    {
    // create a simple particle data to test with
    boost::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(3, BoxDim(10.0), 4));
    boost::shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    boost::shared_ptr<Box> box = pdata->getBox();
    {
    ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(pdata->getVelocities(), access_location::host, access_mode::readwrite);

    h_pos.data[0].x = h_pos.data[0].y = h_pos.data[0].z = 0.0;
    h_pos.data[1].x = h_pos.data[1].y = h_pos.data[1].z = 1.0;
    }

    // compute a shift and apply it to all particles, and origin
    Scalar3 shift(0.5,0.125,0.75);
    pdata->translateOrigin(shift);
    {
    ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<int3> h_img(pdata->getImages(), access_location::host, access_mode::readwrite);

    for (unsigned int i = 0; i < pdata->getN(); i++)
        {
        // read in the current position and orientation
        Scalar4 postype_i = h_postype.data[i];
        vec3<Scalar> r_i = vec3<Scalar>(postype_i); // translation from local to global coordinates
        r_i += vec3<Scalar>(shift);
        h_postype.data[i] = vec_to_scalar4(r_i, postype_i.w);
        box.wrap(h_postype.data[i], h_image.data[i]);
        }
    }

    // check that the particle positions are still the original ones
    Scalar3 pdata->getPosition(0);
    MY_BOOST_CHECK_SMALL(pos.x-0.0, tol_small);
    MY_BOOST_CHECK_SMALL(pos.y-0.0, tol_small);
    MY_BOOST_CHECK_SMALL(pos.z-0.0, tol_small);
    Scalar3 pdata->getPosition(1);
    MY_BOOST_CHECK_SMALL(pos.x-1.0, tol_small);
    MY_BOOST_CHECK_SMALL(pos.y-1.0, tol_small);
    MY_BOOST_CHECK_SMALL(pos.z-1.0, tol_small);

    int3 pimg = pdata->getImage(0);
    MY_BOOST_CHECK_EQUAL(pimg.x, 0);
    MY_BOOST_CHECK_EQUAL(pimg.y, 0);
    MY_BOOST_CHECK_EQUAL(pimg.z, 0);
    int3 pimg = pdata->getImage(0);
    MY_BOOST_CHECK_EQUAL(pimg.x, 0);
    MY_BOOST_CHECK_EQUAL(pimg.y, 0);
    MY_BOOST_CHECK_EQUAL(pimg.z, 0);

    // compute a shift that will shift the image of the box
    Scalar3 shift_img(10.5,10.125,10.75);
    pdata->translateOrigin(shift_img);
    {
    ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<int3> h_img(pdata->getImages(), access_location::host, access_mode::readwrite);

    for (unsigned int i = 0; i < pdata->getN(); i++)
        {
        // read in the current position and orientation
        Scalar4 postype_i = h_postype.data[i];
        vec3<Scalar> r_i = vec3<Scalar>(postype_i); // translation from local to global coordinates
        r_i += vec3<Scalar>(shift_img);
        h_postype.data[i] = vec_to_scalar4(r_i, postype_i.w);
        box.wrap(h_postype.data[i], h_image.data[i]);
        }
    }

    // check that the particle positions are still the original ones
    Scalar3 pdata->getPosition(0);
    MY_BOOST_CHECK_SMALL(pos.x-0.0, tol_small);
    MY_BOOST_CHECK_SMALL(pos.y-0.0, tol_small);
    MY_BOOST_CHECK_SMALL(pos.z-0.0, tol_small);
    Scalar3 pdata->getPosition(1);
    MY_BOOST_CHECK_SMALL(pos.x-1.0, tol_small);
    MY_BOOST_CHECK_SMALL(pos.y-1.0, tol_small);
    MY_BOOST_CHECK_SMALL(pos.z-1.0, tol_small);

    int3 pimg = pdata->getImage(0);
    MY_BOOST_CHECK_EQUAL(pimg.x, 0);
    MY_BOOST_CHECK_EQUAL(pimg.y, 0);
    MY_BOOST_CHECK_EQUAL(pimg.z, 0);
    int3 pimg = pdata->getImage(0);
    MY_BOOST_CHECK_EQUAL(pimg.x, 0);
    MY_BOOST_CHECK_EQUAL(pimg.y, 0);
    MY_BOOST_CHECK_EQUAL(pimg.z, 0);
    }

BOOST_AUTO_TEST_CASE( ParticleDataGridShiftSetMethods )
    {
    // create a simple particle data to test with
    boost::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(3, BoxDim(10.0), 4));
    boost::shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    boost::shared_ptr<Box> box = pdata->getBox();
    {
    ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(pdata->getVelocities(), access_location::host, access_mode::readwrite);

    h_pos.data[0].x = h_pos.data[0].y = h_pos.data[0].z = 0.0;
    h_pos.data[1].x = h_pos.data[1].y = h_pos.data[1].z = 1.0;
    }

    // compute a shift that will shift the image of the box
    Scalar3 shift_img(10.5,10.125,10.75);
    pdata->translateOrigin(shift_img);
    {
    ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<int3> h_img(pdata->getImages(), access_location::host, access_mode::readwrite);

    for (unsigned int i = 0; i < pdata->getN(); i++)
        {
        // read in the current position and orientation
        Scalar4 postype_i = h_postype.data[i];
        vec3<Scalar> r_i = vec3<Scalar>(postype_i); // translation from local to global coordinates
        r_i += vec3<Scalar>(shift_img);
        h_postype.data[i] = vec_to_scalar4(r_i, postype_i.w);
        box.wrap(h_postype.data[i], h_image.data[i]);
        }
    }

    // check that the particle positions are still the original ones
    Scalar3 pdata->getPosition(0);
    MY_BOOST_CHECK_SMALL(pos.x-0.0, tol_small);
    MY_BOOST_CHECK_SMALL(pos.y-0.0, tol_small);
    MY_BOOST_CHECK_SMALL(pos.z-0.0, tol_small);
    Scalar3 pdata->getPosition(1);
    MY_BOOST_CHECK_SMALL(pos.x-1.0, tol_small);
    MY_BOOST_CHECK_SMALL(pos.y-1.0, tol_small);
    MY_BOOST_CHECK_SMALL(pos.z-1.0, tol_small);

    int3 pimg = pdata->getImage(0);
    MY_BOOST_CHECK_EQUAL(pimg.x, 0);
    MY_BOOST_CHECK_EQUAL(pimg.y, 0);
    MY_BOOST_CHECK_EQUAL(pimg.z, 0);
    int3 pimg = pdata->getImage(1);
    MY_BOOST_CHECK_EQUAL(pimg.x, 0);
    MY_BOOST_CHECK_EQUAL(pimg.y, 0);
    MY_BOOST_CHECK_EQUAL(pimg.z, 0);


    //OK, now we set the positions using the particle data proxy
    Scalar3 new_pos0(0.1,0.5,0.7);
    pdata->setPosition(0,new_pos0);
    Scalar3 new_pos1(0.4,0.1,10);
    pdata->setPosition(1,new_pos0);

    ret_pos0 = pdata->getPosition(0);
    MY_BOOST_CHECK_SMALL(ret_pos0.x-new_pos0.x, tol_small);
    MY_BOOST_CHECK_SMALL(ret_pos0.y-new_pos0.y, tol_small);
    MY_BOOST_CHECK_SMALL(ret_pos0.z-new_pos0.z, tol_small);

    ret_pos1 = pdata->getPosition(1);
    MY_BOOST_CHECK_SMALL(ret_pos1.x-new_pos1.x, tol_small);
    MY_BOOST_CHECK_SMALL(ret_pos1.y-new_pos1.y, tol_small);
    MY_BOOST_CHECK_SMALL(ret_pos1.z-new_pos1.z, tol_small);

    //OK, now do the same with the images
    int3 new_img0(1,-5,7);
    pdata->setImage(0,new_img0);
    int3 new_img1(4,1,10);
    pdata->setImage(1,new_img1);

    ret_img0 = pdata->getImage(0);
    MY_BOOST_CHECK_EQUAL(ret_img0.x-new_img0.x, 0);
    MY_BOOST_CHECK_EQUAL(ret_img0.y-new_img0.y, 0);
    MY_BOOST_CHECK_EQUAL(ret_img0.z-new_img0.z, 0);

    ret_img1 = pdata->getImage(1);
    MY_BOOST_CHECK_EQUAL(ret_img1.x-new_img1.x, 0);
    MY_BOOST_CHECK_EQUAL(ret_img1.y-new_img1.y, 0);
    MY_BOOST_CHECK_EQUAL(ret_img1.z-new_img1.z, 0);
    }

#ifdef WIN32
#pragma warning( pop )
#endif
