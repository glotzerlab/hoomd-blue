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
BOOST_AUTO_TEST_CASE( ZeroMomentumUpdater_basic )
    {
    // create a simple particle data to test with
    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(2, BoxDim(1000.0), 4));
    shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    
    ParticleDataArrays arrays = pdata->acquireReadWrite();
    arrays.x[0] = arrays.y[0] = arrays.z[0] = 0.0;
    arrays.vx[0] = 1.0; arrays.vy[0] = 2.0; arrays.vz[0] = 3.0;
    arrays.x[1] = arrays.y[1] = arrays.z[1] = 1.0;
    arrays.vx[1] = 4.0; arrays.vy[1] = 5.0; arrays.vz[1] = 6.0;
    pdata->release();
    
    // construct the updater and make sure everything is set properly
    shared_ptr<ZeroMomentumUpdater> zerop(new ZeroMomentumUpdater(sysdef));
    
    // run the updater and check the new temperature
    zerop->update(0);
    
    // check that the momentum is now zero
    arrays = pdata->acquireReadWrite();
    
    // temp variables for holding the sums
    Scalar sum_px = 0.0;
    Scalar sum_py = 0.0;
    Scalar sum_pz = 0.0;
    
    // note: assuming mass == 1 for now
    for (unsigned int i = 0; i < arrays.nparticles; i++)
        {
        sum_px += arrays.vx[i];
        sum_py += arrays.vy[i];
        sum_pz += arrays.vz[i];
        }
    pdata->release();
    
    // calculate the average
    Scalar avg_px = sum_px / Scalar(arrays.nparticles);
    Scalar avg_py = sum_py / Scalar(arrays.nparticles);
    Scalar avg_pz = sum_pz / Scalar(arrays.nparticles);
    
    MY_BOOST_CHECK_SMALL(avg_px, tol_small);
    MY_BOOST_CHECK_SMALL(avg_py, tol_small);
    MY_BOOST_CHECK_SMALL(avg_pz, tol_small);
    }

//! boost test case to verify proper operation of ZeroMomentumUpdater with rigid bodies
BOOST_AUTO_TEST_CASE( ZeroMomentumUpdater_rigid )
    {
    // create a simple particle data to test with
    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(8, BoxDim(1000.0), 4));
    shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    
    // set the first 4 particles to be free particles (position doesn't matter, but set it anyways)
    pdata->setPosition(0, make_scalar3(0,0,0));
    pdata->setPosition(1, make_scalar3(1,1,1));
    pdata->setPosition(2, make_scalar3(2,2,2));
    pdata->setPosition(3, make_scalar3(3,3,3));
    
    pdata->setBody(0, -1);
    pdata->setBody(1, -1);
    pdata->setBody(2, -1);
    pdata->setBody(3, -1);
    
    // pick some random velocities and masses to make life interesting
    pdata->setMass(0, 1.5);
    pdata->setMass(1, 8.5);
    pdata->setMass(2, 2.5);
    pdata->setMass(3, 0.5);
    
    pdata->setVelocity(0, make_scalar3(-1.0, 4.0, 8.0));
    pdata->setVelocity(1, make_scalar3(-3.2, -1.5, -9.2));
    pdata->setVelocity(2, make_scalar3(2.1, 1.1, -4.5));
    pdata->setVelocity(3, make_scalar3(-3.2, 8.7, 4.9));
    
    // now, set up the other 4 particles as 2 dimers
    pdata->setPosition(4, make_scalar3(-1,0,0));
    pdata->setPosition(5, make_scalar3(-2,0,0));
    pdata->setPosition(6, make_scalar3(1,1,0));
    pdata->setPosition(7, make_scalar3(1,2,0));
    
    pdata->setBody(4, 0);
    pdata->setBody(5, 0);
    pdata->setBody(6, 1);
    pdata->setBody(7, 1);
    
    shared_ptr<RigidData> rdata = sysdef->getRigidData();
    // Initialize rigid bodies
    rdata->initializeData();
    
    // give them an initial COM velocity and angular momentum
    rdata->setBodyVel(0, make_scalar3(1.0, 8.0, -3.0));
    rdata->setBodyVel(1, make_scalar3(2.0, 12.0, -2.0));
    
    rdata->setAngMom(0, make_scalar4(0.5, 2.3, 1.4, 0.0));
    rdata->setAngMom(1, make_scalar4(8.2, 3.1, 6.4, 0.0));
    
    rdata->setRV(false);
    
    // construct the updater and make sure everything is set properly
    shared_ptr<ZeroMomentumUpdater> zerop(new ZeroMomentumUpdater(sysdef));
    
    // run the updater and check the new temperature
    zerop->update(0);
    
    // temp variables for holding the sums
    Scalar sum_px = 0.0;
    Scalar sum_py = 0.0;
    Scalar sum_pz = 0.0;
    
    for (unsigned int i = 0; i < pdata->getN(); i++)
        {
        Scalar mass = pdata->getMass(i);
        Scalar3 vel = pdata->getVelocity(i);
        sum_px += mass * vel.x;
        sum_py += mass * vel.y;
        sum_pz += mass * vel.z;
        }
    
    // calculate the average
    Scalar avg_px = sum_px / Scalar(pdata->getN());
    Scalar avg_py = sum_py / Scalar(pdata->getN());
    Scalar avg_pz = sum_pz / Scalar(pdata->getN());
    
    MY_BOOST_CHECK_SMALL(avg_px, tol_small);
    MY_BOOST_CHECK_SMALL(avg_py, tol_small);
    MY_BOOST_CHECK_SMALL(avg_pz, tol_small);
    }

#ifdef WIN32
#pragma warning( pop )
#endif

