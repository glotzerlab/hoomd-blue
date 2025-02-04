// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// this include is necessary to get MPI included before anything else to support intel MPI
#include "hoomd/ExecutionConfiguration.h"

#include <iostream>

#include <functional>

#include "hoomd/md/CosineSqAngleForceCompute.h"
#ifdef ENABLE_HIP
#include "hoomd/md/CosineSqAngleForceComputeGPU.h"
#endif

#include <stdio.h>

#include "hoomd/Initializers.h"
#include "hoomd/SnapshotSystemData.h"

using namespace std;
using namespace std::placeholders;
using namespace hoomd;
using namespace hoomd::md;

#include "hoomd/test/upp11_config.h"
HOOMD_UP_MAIN();

//! Typedef to make using the std::function factory easier
typedef std::function<std::shared_ptr<CosineSqAngleForceCompute>(
    std::shared_ptr<SystemDefinition> sysdef)>
    angleforce_creator;

//! Perform some simple functionality tests of any BondForceCompute
void angle_force_basic_tests(angleforce_creator af_creator,
                             std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    /////////////////////////////////////////////////////////
    // start with the simplest possible test: 3 particles in a huge box with only one bond type !!!!
    // NO ANGLES
    std::shared_ptr<SystemDefinition> sysdef_3(
        new SystemDefinition(3, BoxDim(1000.0), 1, 1, 1, 0, 0, exec_conf));
    std::shared_ptr<ParticleData> pdata_3 = sysdef_3->getParticleData();

    pdata_3->setPosition(0, make_scalar3(0.0, 0.0, 0.0));
    pdata_3->setPosition(1, make_scalar3(1.0, 0.0, 0.0));
    pdata_3->setPosition(2, make_scalar3(2.0, 0.0, 0.0));

    // create the angle force compute to check
    std::shared_ptr<CosineSqAngleForceCompute> fc_3 = af_creator(sysdef_3);
    fc_3->setParams(0, Scalar(1.0), Scalar(3.14159265359)); // type=0, K=1.0,theta_0=pi

    // compute the force and check the results
    fc_3->compute(0);
        {
        const GlobalArray<Scalar4>& force_array_1 = fc_3->getForceArray();
        const GlobalArray<Scalar>& virial_array_1 = fc_3->getVirialArray();
        size_t pitch = virial_array_1.getPitch();
        ArrayHandle<Scalar4> h_force_1(force_array_1, access_location::host, access_mode::read);
        ArrayHandle<Scalar> h_virial_1(virial_array_1, access_location::host, access_mode::read);

        // check that the force is correct, it should be 0 since we haven't created any angles yet
        MY_CHECK_SMALL(h_force_1.data[0].x, tol);
        MY_CHECK_SMALL(h_force_1.data[0].y, tol);
        MY_CHECK_SMALL(h_force_1.data[0].z, tol);
        MY_CHECK_SMALL(h_force_1.data[0].w, tol);
        MY_CHECK_SMALL(h_virial_1.data[0 * pitch + 0], tol);
        MY_CHECK_SMALL(h_virial_1.data[1 * pitch + 0], tol);
        MY_CHECK_SMALL(h_virial_1.data[2 * pitch + 0], tol);
        MY_CHECK_SMALL(h_virial_1.data[3 * pitch + 0], tol);
        MY_CHECK_SMALL(h_virial_1.data[4 * pitch + 0], tol);
        MY_CHECK_SMALL(h_virial_1.data[5 * pitch + 0], tol);
        }

    // add an angle and check again
    // add type 0 between angle formed by atom 0-1-2
    sysdef_3->getAngleData()->addBondedGroup(Angle(0, 0, 1, 2));
    fc_3->compute(1);

        // this time there should be a force, but it should be 0 because the angle
        // is equal to the equilibrium angle
        {
        const GlobalArray<Scalar4>& force_array_2 = fc_3->getForceArray();
        const GlobalArray<Scalar>& virial_array_2 = fc_3->getVirialArray();
        size_t pitch = virial_array_2.getPitch();
        ArrayHandle<Scalar4> h_force_2(force_array_2, access_location::host, access_mode::read);
        ArrayHandle<Scalar> h_virial_2(virial_array_2, access_location::host, access_mode::read);
        MY_CHECK_SMALL(h_force_2.data[0].x, tol);
        MY_CHECK_SMALL(h_force_2.data[0].y, tol);
        MY_CHECK_SMALL(h_force_2.data[0].z, tol);
        MY_CHECK_SMALL(h_force_2.data[0].w, tol);
        MY_CHECK_SMALL(h_virial_2.data[0 * pitch + 0] + h_virial_2.data[3 * pitch + 0]
                           + h_virial_2.data[5 * pitch + 0],
                       tol);
        }

    // now move the 3rd particle into the next box image,
    // we should still get 0 force
    pdata_3->setPosition(2, make_scalar3(1002.0, 0.0, 0.0));
    fc_3->compute(1);
        {
        const GlobalArray<Scalar4>& force_array_2 = fc_3->getForceArray();
        const GlobalArray<Scalar>& virial_array_2 = fc_3->getVirialArray();
        size_t pitch = virial_array_2.getPitch();
        ArrayHandle<Scalar4> h_force_2(force_array_2, access_location::host, access_mode::read);
        ArrayHandle<Scalar> h_virial_2(virial_array_2, access_location::host, access_mode::read);
        MY_CHECK_SMALL(h_force_2.data[0].x, tol);
        MY_CHECK_SMALL(h_force_2.data[0].y, tol);
        MY_CHECK_SMALL(h_force_2.data[0].z, tol);
        MY_CHECK_SMALL(h_force_2.data[0].w, tol);
        MY_CHECK_SMALL(h_virial_2.data[0 * pitch + 0] + h_virial_2.data[3 * pitch + 0]
                           + h_virial_2.data[5 * pitch + 0],
                       tol);
        }

    // make sure the angle force is 0 if the angle is at equilibrium != pi
    fc_3->setParams(0, Scalar(1.0), Scalar(1.57079632679)); // type=0, K=1.0,theta_0=pi/2
    pdata_3->setPosition(0, make_scalar3(0.0, 0.0, 0.0));
    pdata_3->setPosition(1, make_scalar3(1.0, 0.0, 0.0));
    pdata_3->setPosition(2, make_scalar3(1.0, 1.0, 0.0));
    fc_3->compute(1);
        {
        const GlobalArray<Scalar4>& force_array_2 = fc_3->getForceArray();
        const GlobalArray<Scalar>& virial_array_2 = fc_3->getVirialArray();
        size_t pitch = virial_array_2.getPitch();
        ArrayHandle<Scalar4> h_force_2(force_array_2, access_location::host, access_mode::read);
        ArrayHandle<Scalar> h_virial_2(virial_array_2, access_location::host, access_mode::read);
        MY_CHECK_SMALL(h_force_2.data[0].x, tol);
        MY_CHECK_SMALL(h_force_2.data[0].y, tol);
        MY_CHECK_SMALL(h_force_2.data[0].z, tol);
        MY_CHECK_SMALL(h_force_2.data[0].w, tol);
        MY_CHECK_SMALL(h_virial_2.data[0 * pitch + 0] + h_virial_2.data[3 * pitch + 0]
                           + h_virial_2.data[5 * pitch + 0],
                       tol);
        }

        // rearrange the two particles in memory and see if they are properly updated
        {
        // first move particles back to their original positions, and reset angle params
        pdata_3->setPosition(0, make_scalar3(0.0, 0.0, 0.0));
        pdata_3->setPosition(1, make_scalar3(1.0, 0.0, 0.0));
        pdata_3->setPosition(2, make_scalar3(2.0, 0.0, 0.0));
        fc_3->setParams(0, Scalar(1.0), Scalar(3.14159265359)); // type=0, K=1.0,theta_0=pi

        ArrayHandle<Scalar4> h_pos(pdata_3->getPositions(),
                                   access_location::host,
                                   access_mode::readwrite);
        ArrayHandle<unsigned int> h_tag(pdata_3->getTags(),
                                        access_location::host,
                                        access_mode::readwrite);
        ArrayHandle<unsigned int> h_rtag(pdata_3->getRTags(),
                                         access_location::host,
                                         access_mode::readwrite);

        h_pos.data[1].x = Scalar(0.0); // put atom a at (-1,0,0.1)
        h_pos.data[1].y = Scalar(0.0);
        h_pos.data[1].z = Scalar(0.0);

        h_pos.data[0].x = h_pos.data[0].y = h_pos.data[0].z = Scalar(0.0); // put atom b at (0,0,0)
        h_pos.data[0].x = Scalar(1.0);

        h_tag.data[0] = 1;
        h_tag.data[1] = 0;
        h_rtag.data[0] = 1;
        h_rtag.data[1] = 0;
        }

    // notify that we made the sort
    pdata_3->notifyParticleSort();
    // recompute at the same timestep, the forces should still be updated
    fc_3->compute(1);

        {
        const GlobalArray<Scalar4>& force_array_3 = fc_3->getForceArray();
        const GlobalArray<Scalar>& virial_array_3 = fc_3->getVirialArray();
        size_t pitch = virial_array_3.getPitch();
        ArrayHandle<Scalar4> h_force_3(force_array_3, access_location::host, access_mode::read);
        ArrayHandle<Scalar> h_virial_3(virial_array_3, access_location::host, access_mode::read);

        MY_CHECK_CLOSE(h_force_3.data[1].x, 0.0, tol);
        MY_CHECK_CLOSE(h_force_3.data[1].y, 0.0, tol);
        MY_CHECK_CLOSE(h_force_3.data[1].z, 0.0, tol);
        MY_CHECK_CLOSE(h_force_3.data[1].w, 0.0, tol);
        MY_CHECK_SMALL(h_virial_3.data[0 * pitch + 1] + h_virial_3.data[3 * pitch + 1]
                           + h_virial_3.data[5 * pitch + 1],
                       tol);
        }
    }

//! Compares the output of two CosineSqAngleForceComputes
void angle_force_comparison_tests(angleforce_creator af_creator1,
                                  angleforce_creator af_creator2,
                                  std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    const unsigned int N = 1000;

    // create a particle system to sum forces on
    // just randomly place particles. We don't really care how huge the angle
    // forces get: this is just a unit test
    RandomInitializer rand_init(N, Scalar(0.2), Scalar(0.9), "A");
    std::shared_ptr<SnapshotSystemData<Scalar>> snap = rand_init.getSnapshot();
    snap->angle_data.type_mapping.push_back("A");
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));

    std::shared_ptr<CosineSqAngleForceCompute> fc1 = af_creator1(sysdef);
    std::shared_ptr<CosineSqAngleForceCompute> fc2 = af_creator2(sysdef);
    fc1->setParams(0, Scalar(1.0), Scalar(1.348));
    fc2->setParams(0, Scalar(1.0), Scalar(1.348));

    // add angles
    for (unsigned int i = 0; i < N - 2; i++)
        {
        sysdef->getAngleData()->addBondedGroup(Angle(0, i, i + 1, i + 2));
        }

    // compute the forces
    fc1->compute(0);
    fc2->compute(0);

        {
        const GlobalArray<Scalar4>& force_array_7 = fc1->getForceArray();
        const GlobalArray<Scalar>& virial_array_7 = fc1->getVirialArray();
        size_t pitch = virial_array_7.getPitch();
        ArrayHandle<Scalar4> h_force_7(force_array_7, access_location::host, access_mode::read);
        ArrayHandle<Scalar> h_virial_7(virial_array_7, access_location::host, access_mode::read);
        const GlobalArray<Scalar4>& force_array_8 = fc2->getForceArray();
        const GlobalArray<Scalar>& virial_array_8 = fc2->getVirialArray();
        ArrayHandle<Scalar4> h_force_8(force_array_8, access_location::host, access_mode::read);
        ArrayHandle<Scalar> h_virial_8(virial_array_8, access_location::host, access_mode::read);

        // compare average deviation between the two computes
        double deltaf2 = 0.0;
        double deltape2 = 0.0;
        double deltav2[6];
        for (unsigned int i = 0; i < 6; i++)
            deltav2[i] = 0;

        for (unsigned int i = 0; i < N; i++)
            {
            deltaf2 += double(h_force_8.data[i].x - h_force_7.data[i].x)
                       * double(h_force_8.data[i].x - h_force_7.data[i].x);
            deltaf2 += double(h_force_8.data[i].y - h_force_7.data[i].y)
                       * double(h_force_8.data[i].y - h_force_7.data[i].y);
            deltaf2 += double(h_force_8.data[i].z - h_force_7.data[i].z)
                       * double(h_force_8.data[i].z - h_force_7.data[i].z);
            deltape2 += double(h_force_8.data[i].w - h_force_7.data[i].w)
                        * double(h_force_8.data[i].w - h_force_7.data[i].w);
            for (unsigned int j = 0; j < 6; j++)
                deltav2[j]
                    += double(h_virial_8.data[j * pitch + i] - h_virial_7.data[j * pitch + i])
                       * double(h_virial_8.data[j * pitch + i] - h_virial_7.data[j * pitch + i]);

            // also check that each individual calculation is somewhat close
            }
        deltaf2 /= double(N);
        deltape2 /= double(N);
        for (unsigned int i = 0; i < 6; i++)
            deltav2[i] /= double(N);
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

//! CosineSqAngleForceCompute creator for angle_force_basic_tests()
std::shared_ptr<CosineSqAngleForceCompute>
base_class_af_creator(std::shared_ptr<SystemDefinition> sysdef)
    {
    return std::shared_ptr<CosineSqAngleForceCompute>(new CosineSqAngleForceCompute(sysdef));
    }

#ifdef ENABLE_HIP
//! AngleForceCompute creator for bond_force_basic_tests()
std::shared_ptr<CosineSqAngleForceCompute> gpu_af_creator(std::shared_ptr<SystemDefinition> sysdef)
    {
    return std::shared_ptr<CosineSqAngleForceCompute>(new CosineSqAngleForceComputeGPU(sysdef));
    }
#endif

//! test case for angle forces on the CPU
UP_TEST(CosineSqAngleForceCompute_basic)
    {
    printf(" IN UP_TEST: CPU \n");
    // cout << " IN UP_TEST: CPU \n";
    angleforce_creator af_creator = bind(base_class_af_creator, _1);
    std::shared_ptr<ExecutionConfiguration> exec_conf(
        new ExecutionConfiguration(ExecutionConfiguration::CPU));
    angle_force_basic_tests(af_creator, exec_conf);
    }

#ifdef ENABLE_HIP
//! test case for angle forces on the GPU
UP_TEST(CosineSqAngleForceComputeGPU_basic)
    {
    printf(" IN UP_TEST: GPU \n");
    cout << " IN UP_TEST: GPU \n";
    angleforce_creator af_creator = bind(gpu_af_creator, _1);
    std::shared_ptr<ExecutionConfiguration> exec_conf(
        new ExecutionConfiguration(ExecutionConfiguration::GPU));
    exec_conf->setCUDAErrorChecking(true);
    angle_force_basic_tests(af_creator, exec_conf);
    }

//! test case for comparing bond GPU and CPU BondForceComputes
UP_TEST(CosineSqAngleForceComputeGPU_compare)
    {
    angleforce_creator af_creator_gpu = bind(gpu_af_creator, _1);
    angleforce_creator af_creator = bind(base_class_af_creator, _1);
    angle_force_comparison_tests(af_creator,
                                 af_creator_gpu,
                                 std::shared_ptr<ExecutionConfiguration>(
                                     new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

#endif
