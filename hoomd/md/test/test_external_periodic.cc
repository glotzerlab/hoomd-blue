// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// this include is necessary to get MPI included before anything else to support intel MPI
#include "hoomd/ExecutionConfiguration.h"

#include <fstream>
#include <iostream>

#include <functional>
#include <memory>

#include "hoomd/md/PotentialExternal.h"
#ifdef ENABLE_HIP
#include "hoomd/md/PotentialExternalGPU.h"
#endif
#include "hoomd/md/EvaluatorExternalPeriodic.h"

#include "hoomd/Initializers.h"

#include <math.h>

using namespace std;
using namespace std::placeholders;
using namespace hoomd;
using namespace hoomd::md;

typedef PotentialExternal<EvaluatorExternalPeriodic> PotentialExternalPeriodic;
#ifdef ENABLE_HIP
typedef PotentialExternalGPU<EvaluatorExternalPeriodic> PotentialExternalPeriodicGPU;
#endif

/*! \file lj_force_test.cc
    \brief Implements unit tests for PotentialPairLJ and PotentialPairLJGPU and descendants
    \ingroup unit_tests
*/

#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN();

//! Typedef'd LJForceCompute factory
typedef std::function<std::shared_ptr<PotentialExternalPeriodic>(
    std::shared_ptr<SystemDefinition> sysdef)>
    periodicforce_creator;

//! Test the ability of the lj force compute to actually calculate forces
void periodic_force_particle_test(periodicforce_creator periodic_creator,
                                  std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // this 3-particle test subtly checks several conditions
    // the particles are arranged on the x axis,  1   2   3
    // types of the particles : 0, 1, 0

    // periodic boundary conditions will be handled in another test
    std::shared_ptr<SystemDefinition> sysdef_3(
        new SystemDefinition(3, BoxDim(5.0), 2, 0, 0, 0, 0, exec_conf));
    std::shared_ptr<ParticleData> pdata_3 = sysdef_3->getParticleData();

    pdata_3->setPosition(0, make_scalar3(1.7, 0.0, 0.0));
    pdata_3->setPosition(1, make_scalar3(2.0, 0.0, 0.0));
    pdata_3->setPosition(2, make_scalar3(3.5, 0.0, 0.0));
    pdata_3->setType(1, 1);
    std::shared_ptr<PotentialExternalPeriodic> fc_3 = periodic_creator(sysdef_3);

    // first test: setup a sigma of 1.0 so that all forces will be 0
    unsigned int index = 0;
    Scalar orderParameter = 0.5;
    Scalar interfaceWidth = 0.5;
    unsigned int periodicity = 2;
    fc_3->setParams(
        0,
        PotentialExternalPeriodic::param_type(index, orderParameter, interfaceWidth, periodicity));
    fc_3->setParams(
        1,
        PotentialExternalPeriodic::param_type(index, -orderParameter, interfaceWidth, periodicity));

    // compute the forces
    fc_3->compute(0);

        {
        const GlobalArray<Scalar4>& force_array_1 = fc_3->getForceArray();
        const GlobalArray<Scalar>& virial_array_1 = fc_3->getVirialArray();
        size_t pitch = virial_array_1.getPitch();
        ArrayHandle<Scalar4> h_force_1(force_array_1, access_location::host, access_mode::read);
        ArrayHandle<Scalar> h_virial_1(virial_array_1, access_location::host, access_mode::read);
        MY_CHECK_CLOSE(h_force_1.data[0].x, -0.180137, tol);
        MY_CHECK_SMALL(h_force_1.data[0].y, tol_small);
        MY_CHECK_SMALL(h_force_1.data[0].z, tol_small);
        MY_CHECK_CLOSE(h_force_1.data[0].w, -0.0338307, tol);
        MY_CHECK_SMALL(h_virial_1.data[0 * pitch + 0] + h_virial_1.data[3 * pitch + 0]
                           + h_virial_1.data[5 * pitch + 0],
                       tol_small);

        MY_CHECK_CLOSE(h_force_1.data[1].x, 0.189752, tol);
        MY_CHECK_SMALL(h_force_1.data[1].y, tol_small);
        MY_CHECK_SMALL(h_force_1.data[1].z, tol_small);
        MY_CHECK_CLOSE(h_force_1.data[1].w, -0.024571, tol);
        MY_CHECK_SMALL(h_virial_1.data[0 * pitch + 1] + h_virial_1.data[3 * pitch + 1]
                           + h_virial_1.data[5 * pitch + 1],
                       tol_small);

        MY_CHECK_CLOSE(h_force_1.data[2].x, 0.115629, tol);
        MY_CHECK_SMALL(h_force_1.data[2].y, tol_small);
        MY_CHECK_SMALL(h_force_1.data[2].z, tol_small);
        MY_CHECK_CLOSE(h_force_1.data[2].w, -0.0640261, tol);
        MY_CHECK_SMALL(h_virial_1.data[0 * pitch + 2] + h_virial_1.data[3 * pitch + 2]
                           + h_virial_1.data[5 * pitch + 2],
                       tol_small);
        }
    }

#if 0
//! Unit test a comparison between 2 PeriodicForceComputes on a "real" system
void periodic_force_comparison_test(periodicforce_creator periodic_creator1, periodicforce_creator periodic_creator2, std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    const unsigned int N = 5000;

    // create a random particle system to sum forces on
    RandomInitializer rand_init(N, Scalar(0.2), Scalar(0.9), "A");
    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(rand_init, exec_conf));
    shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    shared_ptr<PotentialExternalPeriodic> fc1 = periodic_creator1(sysdef);
    shared_ptr<PotentialExternalPeriodic> fc2 = periodic_creator2(sysdef);

    unsigned int index = 0;
    Scalar orderParameter = 0.5;
    Scalar interfaceWidth = 0.5;
    unsigned int periodicity = 2;
    fc1->setParams(make_scalar4(__int_as_scalar(index),orderParameter,interfaceWidth,__int_as_scalar(periodicity)));
    fc2->setParams(make_scalar4(__int_as_scalar(index),orderParameter,interfaceWidth,__int_as_scalar(periodicity)));

    // compute the forces
    fc1->compute(0);
    fc2->compute(0);

    {
    // verify that the forces are identical (within roundoff errors)
    GlobalArray<Scalar4>& force_array_5 =  fc1->getForceArray();
    GlobalArray<Scalar>& virial_array_5 =  fc1->getVirialArray();
    size_t pitch = virial_array_5.getPitch();
    ArrayHandle<Scalar4> h_force_5(force_array_5,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_5(virial_array_5,access_location::host,access_mode::read);
    GlobalArray<Scalar4>& force_array_6 =  fc2->getForceArray();
    GlobalArray<Scalar>& virial_array_6 =  fc2->getVirialArray();
    ArrayHandle<Scalar4> h_force_6(force_array_6,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_6(virial_array_6,access_location::host,access_mode::read);

    // compare average deviation between the two computes
    double deltaf2 = 0.0;
    double deltape2 = 0.0;
    double deltav2[6];
    for (unsigned int i = 0; i < 6; i++)
        deltav2[i] = 0.0;

    for (unsigned int i = 0; i < N; i++)
        {
        deltaf2 += double(h_force_6.data[i].x - h_force_5.data[i].x) * double(h_force_6.data[i].x - h_force_5.data[i].x);
        deltaf2 += double(h_force_6.data[i].y - h_force_5.data[i].y) * double(h_force_6.data[i].y - h_force_5.data[i].y);
        deltaf2 += double(h_force_6.data[i].z - h_force_5.data[i].z) * double(h_force_6.data[i].z - h_force_5.data[i].z);
        deltape2 += double(h_force_6.data[i].w - h_force_5.data[i].w) * double(h_force_6.data[i].w - h_force_5.data[i].w);
        for (unsigned int j = 0; j < 6; j++)
            deltav2[j] += double(h_virial_6.data[j*pitch+i] - h_virial_5.data[j*pitch+i]) * double(h_virial_6.data[j*pitch+i] - h_virial_5.data[j*pitch+i]);

        // also check that each individual calculation is somewhat close
        }
    deltaf2 /= double(pdata->getN());
    deltape2 /= double(pdata->getN());
    for (unsigned int j = 0; j < 6; j++)
        deltav2[j] /= double(pdata->getN());
    CHECK_SMALL(deltaf2, double(tol_small));
    CHECK_SMALL(deltape2, double(tol_small));
    CHECK_SMALL(deltav2[0], double(tol_small));
    CHECK_SMALL(deltav2[1], double(tol_small));
    CHECK_SMALL(deltav2[2], double(tol_small));
    CHECK_SMALL(deltav2[3], double(tol_small));
    CHECK_SMALL(deltav2[4], double(tol_small));
    CHECK_SMALL(deltav2[5], double(tol_small));
    }
    }
#endif

//! LJForceCompute creator for unit tests
std::shared_ptr<PotentialExternalPeriodic>
base_class_periodic_creator(std::shared_ptr<SystemDefinition> sysdef)
    {
    return std::shared_ptr<PotentialExternalPeriodic>(new PotentialExternalPeriodic(sysdef));
    }

#ifdef ENABLE_HIP
//! LJForceComputeGPU creator for unit tests
std::shared_ptr<PotentialExternalPeriodic>
gpu_periodic_creator(std::shared_ptr<SystemDefinition> sysdef)
    {
    std::shared_ptr<PotentialExternalPeriodicGPU> periodic(
        new PotentialExternalPeriodicGPU(sysdef));
    // the default block size kills valgrind :) reduce it
    //    lj->setBlockSize(64);
    return periodic;
    }
#endif

//! test case for particle test on CPU
UP_TEST(PotentialExternalPeriodic_particle)
    {
    periodicforce_creator periodic_creator_base = bind(base_class_periodic_creator, _1);
    periodic_force_particle_test(periodic_creator_base,
                                 std::shared_ptr<ExecutionConfiguration>(
                                     new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

#ifdef ENABLE_HIP
//! test case for particle test on GPU
UP_TEST(PotentialExternalLamellaGPU_particle)
    {
    periodicforce_creator periodic_creator_gpu = bind(gpu_periodic_creator, _1);
    periodic_force_particle_test(periodic_creator_gpu,
                                 std::shared_ptr<ExecutionConfiguration>(
                                     new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

/*
//! test case for comparing GPU output to base class output
UP_TEST( LJForceGPU_compare )
    {
    ljforce_creator lj_creator_gpu = bind(gpu_lj_creator, _1, _2);
    ljforce_creator lj_creator_base = bind(base_class_lj_creator, _1, _2);
    lj_force_comparison_test(lj_creator_base, lj_creator_gpu,
std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
*/
#endif
