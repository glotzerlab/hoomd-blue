// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// this include is necessary to get MPI included before anything else to support intel MPI
#include "hoomd/ExecutionConfiguration.h"

#include <iostream>

#include <functional>
#include <memory>

#include "hoomd/md/IntegratorTwoStep.h"
#include "hoomd/md/TwoStepLangevin.h"
#include "hoomd/md/ConstraintSphere.h"
#ifdef ENABLE_CUDA
#include "hoomd/md/ConstraintSphereGPU.h"
#endif

#include <math.h>

using namespace std;
using namespace std::placeholders;

/*! \file constraint_sphere_test.cc
    \brief Implements unit tests for ConstraintSphere and descendants
    \ingroup unit_tests
*/

#include "hoomd/test/upp11_config.h"
HOOMD_UP_MAIN();

//! Typedef'd class factory
typedef std::function<std::shared_ptr<ConstraintSphere> (std::shared_ptr<SystemDefinition> sysdef,
                                                      std::shared_ptr<ParticleGroup> group,
                                                      Scalar3 P,
                                                      Scalar r)> cs_creator_t;

//! Run a BD simulation on 6 particles and validate the constraints
void constraint_sphere_tests(cs_creator_t cs_creator, std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    Scalar3 P = make_scalar3(1.0f, 2.0f, 3.0f);
    Scalar r = 10.0f;

    // Build a 6 particle system with all particles starting at the 6 "corners" of a sphere centered
    // at P with radius r. Use a huge box so boundary conditions don't come into play
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(6, BoxDim(1000000.0), 1, 0, 0, 0, 0, exec_conf));
    std::shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    std::shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef, 0, pdata->getN()-1));
    std::shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_all));

    {
    ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);

    // set the particle initial positions
    h_pos.data[0].x = P.x - r;
    h_pos.data[1].x = P.x + r;
    h_pos.data[2].y = P.y - r;
    h_pos.data[3].y = P.y + r;
    h_pos.data[4].z = P.z - r;
    h_pos.data[5].z = P.z + r;
    }

    Scalar deltaT = Scalar(0.01);
    Scalar Temp = Scalar(2.0);

    // run the particles in a BD simulation with a constraint force applied and verify that the constraint is always
    // satisfied
    std::shared_ptr<VariantConst> T_variant(new VariantConst(Temp));
    std::shared_ptr<TwoStepLangevin> two_step_bdnvt(new TwoStepLangevin(sysdef, group_all, T_variant, 123, 0, 0.0, false, false));
    std::shared_ptr<IntegratorTwoStep> bdnvt_up(new IntegratorTwoStep(sysdef, deltaT));
    bdnvt_up->addIntegrationMethod(two_step_bdnvt);

    std::shared_ptr<ConstraintSphere> cs = cs_creator(sysdef, group_all, P, r);
    bdnvt_up->addForceConstraint(cs);
    bdnvt_up->prepRun(0);

    for (int i = 0; i < 1000; i++)
        {
        bdnvt_up->update(i);


        for (unsigned int j = 0; j < 6; j++)
            {
            Scalar3 V;
            Scalar3 pos = pdata->getPosition(j);
            V.x = pos.x - P.x;
            V.y = pos.y - P.y;
            V.z = pos.z - P.z;

            Scalar current_r = sqrt(V.x*V.x + V.y*V.y + V.z*V.z);
            MY_CHECK_CLOSE(current_r, r, loose_tol);
            }

        }
    }


//! ConstraintSphere factory for the unit tests
std::shared_ptr<ConstraintSphere> base_class_cs_creator(std::shared_ptr<SystemDefinition> sysdef,
                                                   std::shared_ptr<ParticleGroup> group,
                                                   Scalar3 P,
                                                   Scalar r)
    {
    return std::shared_ptr<ConstraintSphere>(new ConstraintSphere(sysdef, group, P, r));
    }

#ifdef ENABLE_CUDA
//! ConstraintSphereGPU factory for the unit tests
std::shared_ptr<ConstraintSphere> gpu_cs_creator(std::shared_ptr<SystemDefinition> sysdef,
                                                   std::shared_ptr<ParticleGroup> group,
                                                   Scalar3 P,
                                                   Scalar r)
    {
    return std::shared_ptr<ConstraintSphere>(new ConstraintSphereGPU(sysdef, group, P, r));
    }
#endif

//! Basic test for the base class
UP_TEST( BDUpdater_tests )
    {
    cs_creator_t cs_creator = bind(base_class_cs_creator, _1, _2, _3, _4);
    constraint_sphere_tests(cs_creator, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

#ifdef ENABLE_CUDA
//! Basic test for the GPU class
UP_TEST( BDUpdaterGPU_tests )
    {
    cs_creator_t cs_creator = bind(gpu_cs_creator, _1, _2, _3, _4);
    constraint_sphere_tests(cs_creator, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
#endif
