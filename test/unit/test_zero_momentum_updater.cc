/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2016 The Regents of
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

// this include is necessary to get MPI included before anything else to support intel MPI
#include "ExecutionConfiguration.h"

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
    boost::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(2, BoxDim(1000.0), 4));
    boost::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    {
    ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(pdata->getVelocities(), access_location::host, access_mode::readwrite);

    h_pos.data[0].x = h_pos.data[0].y = h_pos.data[0].z = 0.0;
    h_vel.data[0].x = 1.0; h_vel.data[0].y = 2.0; h_vel.data[0].z = 3.0;
    h_pos.data[1].x = h_pos.data[1].y = h_pos.data[1].z = 1.0;
    h_vel.data[1].x = 4.0; h_vel.data[1].y = 5.0; h_vel.data[1].z = 6.0;
    }

    // construct the updater and make sure everything is set properly
    boost::shared_ptr<ZeroMomentumUpdater> zerop(new ZeroMomentumUpdater(sysdef));

    // run the updater and check the new temperature
    zerop->update(0);

    // check that the momentum is now zero

    // temp variables for holding the sums
    Scalar sum_px = 0.0;
    Scalar sum_py = 0.0;
    Scalar sum_pz = 0.0;

    // note: assuming mass == 1 for now
    {
    ArrayHandle<Scalar4> h_vel(pdata->getVelocities(), access_location::host, access_mode::read);

    for (unsigned int i = 0; i < pdata->getN(); i++)
        {
        sum_px += h_vel.data[i].x;
        sum_py += h_vel.data[i].y;
        sum_pz += h_vel.data[i].z;
        }
    }

    // calculate the average
    Scalar avg_px = sum_px / Scalar(pdata->getN());
    Scalar avg_py = sum_py / Scalar(pdata->getN());
    Scalar avg_pz = sum_pz / Scalar(pdata->getN());

    MY_BOOST_CHECK_SMALL(avg_px, tol_small);
    MY_BOOST_CHECK_SMALL(avg_py, tol_small);
    MY_BOOST_CHECK_SMALL(avg_pz, tol_small);
    }

