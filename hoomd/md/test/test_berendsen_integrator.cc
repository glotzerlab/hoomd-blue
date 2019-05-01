// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// this include is necessary to get MPI included before anything else to support intel MPI
#include "hoomd/ExecutionConfiguration.h"

#include <iostream>
#include <memory>

#include "hoomd/md/IntegratorTwoStep.h"
#include "hoomd/md/TwoStepLangevin.h"
#include "hoomd/md/TwoStepBerendsen.h"
#ifdef ENABLE_CUDA
#include "hoomd/md/TwoStepBerendsenGPU.h"
#endif


#include "hoomd/Initializers.h"

#include <math.h>
#include "hoomd/test/upp11_config.h"

using namespace std;

/*! \file test_berendsen_updater.cc
    \brief Implements unit tests for TwoStepBerendsen and descendants
    \ingroup unit_tests
*/

HOOMD_UP_MAIN();

//! Apply the thermostat to 1000 particles in an ideal gas
template <class Berendsen>
void berend_updater_lj_tests(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // check that the berendsen thermostat applied to a system of 1000 LJ particles produces the correct average temperature
    // Build a 1000 particle system with particles scattered on the x, y, and z axes.
    RandomInitializer rand_init(1000, Scalar(0.05), Scalar(1.3), "A");
    std::shared_ptr< SnapshotSystemData<Scalar> > snap;
    snap = rand_init.getSnapshot();
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));

    std::shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    std::shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef, 0, pdata->getN()-1));
    std::shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_all));

    Scalar deltaT = Scalar(0.002);
    Scalar Temp = Scalar(2.0);

    std::shared_ptr<ComputeThermo> thermo(new ComputeThermo(sysdef, group_all));
    thermo->setNDOF(3*1000-3);
    std::shared_ptr<VariantConst> T_variant(new VariantConst(Temp));
    std::shared_ptr<VariantConst> T_variant2(new VariantConst(1.0));

    std::shared_ptr<TwoStepBerendsen> two_step_berendsen(new Berendsen(sysdef, group_all, thermo, 1.0, T_variant));
    std::shared_ptr<IntegratorTwoStep> berendsen_up(new IntegratorTwoStep(sysdef, deltaT));
    berendsen_up->addIntegrationMethod(two_step_berendsen);

    std::shared_ptr<TwoStepLangevin> two_step_bdnvt(new TwoStepLangevin(sysdef, group_all, T_variant2, 268, 1, 1.0, false, false));
    std::shared_ptr<IntegratorTwoStep> bdnvt_up(new IntegratorTwoStep(sysdef, deltaT));
    bdnvt_up->addIntegrationMethod(two_step_bdnvt);
    bdnvt_up->prepRun(0);

    int i;

    // ramp up to temp
    for (i = 0; i < 10000; i++)
        bdnvt_up->update(i);

    Scalar AvgT = Scalar(0);

    // equilibrate with berend
    berendsen_up->prepRun(0);

    for (i = 0; i < 5000; i++)
        berendsen_up->update(i);

    for (; i < 10000; i++)
        {
        berendsen_up->update(i);

        if (i % 10 == 0)
            {
            thermo->compute(i);
            AvgT += thermo->getTemperature();
            }
        }
    AvgT /= Scalar(5000.0/10.0);

    MY_CHECK_CLOSE(AvgT, 2.0, 1);

    // Resetting the Temperature to 1.0
    two_step_berendsen->setT(T_variant2);

    AvgT = Scalar(0);
    for (i = 0; i < 5000; i++)
        berendsen_up->update(i);

    for (; i < 10000; i++)
        {
        if (i % 10 == 0)
            AvgT += thermo->getTemperature();

        berendsen_up->update(i);
        }
    AvgT /= Scalar(5000.0/10.0);

    MY_CHECK_CLOSE(AvgT, 1.0, 1);
    }

//! extended LJ-liquid test for the base class
UP_TEST( TwoStepBerendsen_LJ_tests )
    {
    berend_updater_lj_tests<TwoStepBerendsen>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

#ifdef ENABLE_CUDA
//! extended LJ-liquid test for the GPU class
UP_TEST( TwoStepBerendsenGPU_LJ_tests )
    {
    berend_updater_lj_tests<TwoStepBerendsenGPU>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
#endif
