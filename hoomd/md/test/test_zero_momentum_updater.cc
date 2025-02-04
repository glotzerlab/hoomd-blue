// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// this include is necessary to get MPI included before anything else to support intel MPI
#include "hoomd/ExecutionConfiguration.h"

#include <iostream>

#include <memory>

#include "hoomd/md/ZeroMomentumUpdater.h"

#include <math.h>

using namespace std;
using namespace hoomd;
using namespace hoomd::md;

#include "hoomd/test/upp11_config.h"
HOOMD_UP_MAIN();

/*! \file zero_momentum_updater_test.cc
    \brief Unit tests for the ZeroMomentumUpdater class
    \ingroup unit_tests
*/

//! test case to verify proper operation of ZeroMomentumUpdater
UP_TEST(ZeroMomentumUpdater_basic)
    {
    // create a simple particle data to test with
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(2, BoxDim(1000.0), 4));
    std::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

        {
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(),
                                   access_location::host,
                                   access_mode::readwrite);
        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::readwrite);

        h_pos.data[0].x = h_pos.data[0].y = h_pos.data[0].z = 0.0;
        h_vel.data[0].x = 1.0;
        h_vel.data[0].y = 2.0;
        h_vel.data[0].z = 3.0;
        h_pos.data[1].x = h_pos.data[1].y = h_pos.data[1].z = 1.0;
        h_vel.data[1].x = 4.0;
        h_vel.data[1].y = 5.0;
        h_vel.data[1].z = 6.0;
        }

    // construct the updater and make sure everything is set properly
    auto zerop
        = std::make_shared<ZeroMomentumUpdater>(sysdef, std::make_shared<PeriodicTrigger>(1));

    // run the updater and check the new temperature
    zerop->update(0);

    // check that the momentum is now zero

    // temp variables for holding the sums
    Scalar sum_px = 0.0;
    Scalar sum_py = 0.0;
    Scalar sum_pz = 0.0;

        // note: assuming mass == 1 for now
        {
        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::read);

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

    MY_CHECK_SMALL(avg_px, tol_small);
    MY_CHECK_SMALL(avg_py, tol_small);
    MY_CHECK_SMALL(avg_pz, tol_small);
    }
