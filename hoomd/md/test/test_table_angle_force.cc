// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// this include is necessary to get MPI included before anything else to support intel MPI
#include "hoomd/ExecutionConfiguration.h"

#include <iostream>

#include <functional>

#include "hoomd/md/TableAngleForceCompute.h"
#include "hoomd/ConstForceCompute.h"
#ifdef ENABLE_CUDA
#include "hoomd/md/TableAngleForceComputeGPU.h"
#endif

#include <stdio.h>

#include "hoomd/Initializers.h"
#include "hoomd/SnapshotSystemData.h"

using namespace std;
using namespace std::placeholders;

#include "hoomd/test/upp11_config.h"
HOOMD_UP_MAIN();

//! Typedef to make using the std::function factory easier
typedef std::function<std::shared_ptr<TableAngleForceCompute>  (std::shared_ptr<SystemDefinition> sysdef,unsigned int width)> angleforce_creator;

//! Perform some simple functionality tests of any BondForceCompute
void angle_force_basic_tests(angleforce_creator tf_creator, std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    /////////////////////////////////////////////////////////
    // start with the simplest possible test: 3 particles in a huge box with only one angle type !!!! NO DIHEDRALS
    std::shared_ptr<SystemDefinition> sysdef_3(new SystemDefinition(3, BoxDim(4.5), 1, 0, 1, 0, 0, exec_conf));
    std::shared_ptr<ParticleData> pdata_3 = sysdef_3->getParticleData();

    pdata_3->setPosition(0,make_scalar3(-1.23,2.0,0.1));
    pdata_3->setPosition(1,make_scalar3(1.0,1.0,1.0));
    pdata_3->setPosition(2,make_scalar3(1.0,0.0,0.5));

    /*
        printf(" Particle 1: x = %f  y = %f  z = %f \n", h_pos.data[0].x, h_pos.data[0].y, h_pos.data[0].z);
        printf(" Particle 2: x = %f  y = %f  z = %f \n", h_pos.data[1].x, h_pos.data[1].y, h_pos.data[1].z);
        printf(" Particle 3: x = %f  y = %f  z = %f \n", h_pos.data[2].x, h_pos.data[2].y, h_pos.data[2].z);
        printf(" Particle 3: x = %f  y = %f  z = %f \n", h_pos.data[3].x, h_pos.data[3].y, h_pos.data[3].z);
        printf("\n");
    */

    // create the angle force compute to check
    unsigned int width = 100;
    std::shared_ptr<TableAngleForceCompute> fc_3 = tf_creator(sysdef_3,width);

    // set up a harmonic potential
    std::vector<Scalar> V, T;

    Scalar kappa = 1.0;
    Scalar phi0 = 0.785398; // pi/4
    for (unsigned int i = 0; i < width; ++i)
        {
        Scalar phi = (Scalar)i/(Scalar)(width-1)*Scalar(M_PI);
        V.push_back(0.5*kappa*(phi-phi0)*(phi-phi0));
        T.push_back(-kappa*(phi-phi0));
        }

    fc_3->setTable(0, V, T);

    // compute the force and check the results
    fc_3->compute(0);

    {
    GlobalArray<Scalar4>& force_array_1 =  fc_3->getForceArray();
    GlobalArray<Scalar>& virial_array_1 =  fc_3->getVirialArray();
    unsigned int pitch = 0;
    ArrayHandle<Scalar4> h_force_1(force_array_1,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_1(virial_array_1,access_location::host,access_mode::read);

    // check that the force is correct, it should be 0 since we haven't created any angles yet
    MY_CHECK_SMALL(h_force_1.data[0].x, tol);
    MY_CHECK_SMALL(h_force_1.data[0].y, tol);
    MY_CHECK_SMALL(h_force_1.data[0].z, tol);
    MY_CHECK_SMALL(h_force_1.data[0].w, tol);
    MY_CHECK_SMALL(h_virial_1.data[0*pitch], tol);
    MY_CHECK_SMALL(h_virial_1.data[1*pitch], tol);
    MY_CHECK_SMALL(h_virial_1.data[2*pitch], tol);
    MY_CHECK_SMALL(h_virial_1.data[3*pitch], tol);
    MY_CHECK_SMALL(h_virial_1.data[4*pitch], tol);
    MY_CHECK_SMALL(h_virial_1.data[5*pitch], tol);
    }

    // add an angle and check again
    sysdef_3->getAngleData()->addBondedGroup(Angle(0,0,1,2)); // add type 0 between angle formed by atom 0-1-2
    fc_3->compute(1);

    Scalar rough_tol = 0.1;
    {
    // this time there should be a force
    GlobalArray<Scalar4>& force_array_2 =  fc_3->getForceArray();
    GlobalArray<Scalar>& virial_array_2 =  fc_3->getVirialArray();
    unsigned int pitch = virial_array_2.getPitch();
    ArrayHandle<Scalar4> h_force_2(force_array_2,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_2(virial_array_2,access_location::host,access_mode::read);

    MY_CHECK_CLOSE(h_force_2.data[0].x, -0.061684, rough_tol);
    MY_CHECK_CLOSE(h_force_2.data[0].y, -0.313469, rough_tol);
    MY_CHECK_CLOSE(h_force_2.data[0].z, -0.195460, rough_tol);
    MY_CHECK_CLOSE(h_force_2.data[0].w, 0.158576, rough_tol);
    MY_CHECK_SMALL(h_virial_2.data[0*pitch+0]
                        +h_virial_2.data[3*pitch+0]
                        +h_virial_2.data[5*pitch+0], rough_tol);

    }

    // rearrange the two particles in memory and see if they are properly updated
    {
    ArrayHandle<Scalar4> h_pos(pdata_3->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_tag(pdata_3->getTags(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_rtag(pdata_3->getRTags(), access_location::host, access_mode::readwrite);

    std::swap(h_pos.data[0],h_pos.data[1]);
    std::swap(h_tag.data[0],h_tag.data[1]);
    std::swap(h_rtag.data[0], h_rtag.data[1]);
    }

    // notify that we made the sort
    pdata_3->notifyParticleSort();
    // recompute at the same timestep, the forces should still be updated
    fc_3->compute(1);

    {
    GlobalArray<Scalar4>& force_array_3 =  fc_3->getForceArray();
    GlobalArray<Scalar>& virial_array_3 =  fc_3->getVirialArray();
    unsigned int pitch = virial_array_3.getPitch();
    ArrayHandle<Scalar4> h_force_3(force_array_3,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_3(virial_array_3,access_location::host,access_mode::read);

    MY_CHECK_CLOSE(h_force_3.data[1].x, -0.061684, rough_tol);
    MY_CHECK_CLOSE(h_force_3.data[1].y, -0.3134695, rough_tol);
    MY_CHECK_CLOSE(h_force_3.data[1].z, -0.195460, rough_tol);
    MY_CHECK_CLOSE(h_force_3.data[1].w, 0.158576, rough_tol);
    MY_CHECK_SMALL(h_virial_3.data[0*pitch+1]
                        +h_virial_3.data[3*pitch+1]
                        +h_virial_3.data[5*pitch+1], rough_tol);
    }

    {
    ArrayHandle<Scalar4> h_pos(pdata_3->getPositions(), access_location::host, access_mode::readwrite);

    // translate all particles and wrap them back into the box
    Scalar3 shift = make_scalar3(-2,0,1);
    int3 img = make_int3(0,0,0);
    const BoxDim& box = pdata_3->getBox();
    h_pos.data[0] = make_scalar4(h_pos.data[0].x+shift.x, h_pos.data[0].y+shift.y,h_pos.data[0].z + shift.z,h_pos.data[0].w);
    box.wrap(h_pos.data[0], img);
    h_pos.data[1] = make_scalar4(h_pos.data[1].x+shift.x, h_pos.data[1].y+shift.y,h_pos.data[1].z + shift.z,h_pos.data[1].w);
    box.wrap(h_pos.data[1], img);
    h_pos.data[2] = make_scalar4(h_pos.data[2].x+shift.x, h_pos.data[2].y+shift.y,h_pos.data[2].z + shift.z,h_pos.data[2].w);
    box.wrap(h_pos.data[2], img);
    }

    fc_3->compute(2);
    {
    GlobalArray<Scalar4>& force_array_3 =  fc_3->getForceArray();
    GlobalArray<Scalar>& virial_array_3 =  fc_3->getVirialArray();
    unsigned int pitch = virial_array_3.getPitch();
    ArrayHandle<Scalar4> h_force_3(force_array_3,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_3(virial_array_3,access_location::host,access_mode::read);

    MY_CHECK_CLOSE(h_force_3.data[1].x, -0.061684, rough_tol);
    MY_CHECK_CLOSE(h_force_3.data[1].y, -0.3134695, rough_tol);
    MY_CHECK_CLOSE(h_force_3.data[1].z, -0.195460, rough_tol);
    MY_CHECK_CLOSE(h_force_3.data[1].w, 0.158576, rough_tol);
    MY_CHECK_SMALL(h_virial_3.data[0*pitch+1]
                        +h_virial_3.data[3*pitch+1]
                        +h_virial_3.data[5*pitch+1], rough_tol);
    }

    }

#if 0
//! Compares the output of two TableAngleForceComputes
void angle_force_comparison_tests(angleforce_creator tf_creator1,
                                     angleforce_creator tf_creator2,
                                     std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    const unsigned int N = 1000;

    // create a particle system to sum forces on
    // just randomly place particles. We don't really care how huge the bond forces get: this is just a unit test
    RandomInitializer rand_init(N, Scalar(0.2), Scalar(0.9), "A");
    std::shared_ptr< SnapshotSystemData<Scalar> > snap = rand_init.getSnapshot();
    snap->angle_data.type_mapping.push_back("A");
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));

    std::shared_ptr<TableAngleForceCompute> fc1 = tf_creator1(sysdef);
    std::shared_ptr<TableAngleForceCompute> fc2 = tf_creator2(sysdef);
    fc1->setParams(0, Scalar(3.0), -1, 3);
    fc2->setParams(0, Scalar(3.0), -1, 3);

    // add angles
    for (unsigned int i = 0; i < N-3; i++)
        {
        sysdef->getAngleData()->addBondedGroup(Angle(0, i, i+1,i+2, i+3));
        }

    // compute the forces
    fc1->compute(0);
    fc2->compute(0);

    // verify that the forces are identical (within roundoff errors)
    {
    GlobalArray<Scalar4>& force_array_7 =  fc1->getForceArray();
    GlobalArray<Scalar>& virial_array_7 =  fc1->getVirialArray();
    unsigned int pitch = virial_array_7.getPitch();
    ArrayHandle<Scalar4> h_force_7(force_array_7,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_7(virial_array_7,access_location::host,access_mode::read);
    GlobalArray<Scalar4>& force_array_8 =  fc2->getForceArray();
    GlobalArray<Scalar>& virial_array_8 =  fc2->getVirialArray();
    ArrayHandle<Scalar4> h_force_8(force_array_8,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_8(virial_array_8,access_location::host,access_mode::read);

    // compare average deviation between the two computes
    double deltaf2 = 0.0;
    double deltape2 = 0.0;
    double deltav2[6];
    for (unsigned int i = 0; i < 6; i++)
        deltav2[i] = 0.0;

    for (unsigned int i = 0; i < N; i++)
        {
        deltaf2 += double(h_force_8.data[i].x - h_force_7.data[i].x) * double(h_force_8.data[i].x - h_force_7.data[i].x);
        deltaf2 += double(h_force_8.data[i].y - h_force_7.data[i].y) * double(h_force_8.data[i].y - h_force_7.data[i].y);
        deltaf2 += double(h_force_8.data[i].z - h_force_7.data[i].z) * double(h_force_8.data[i].z - h_force_7.data[i].z);
        deltape2 += double(h_force_8.data[i].w - h_force_7.data[i].w) * double(h_force_8.data[i].w - h_force_7.data[i].w);
        for (unsigned int j = 0; j < 6; j++)
            deltav2[j] += double(h_virial_8.data[j*pitch+i] - h_virial_7.data[j*pitch+i]) * double(h_virial_8.data[j*pitch+i] - h_virial_7.data[j*pitch+i]);

        // also check that each individual calculation is somewhat close
        }
    deltaf2 /= double(N);
    deltape2 /= double(N);
    for (unsigned int j = 0; j < 6; j++)
        deltav2[j] /= double(N);

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
//! TableAngleForceCompute creator for angle_force_basic_tests()
std::shared_ptr<TableAngleForceCompute> base_class_tf_creator(std::shared_ptr<SystemDefinition> sysdef,unsigned int width)
    {
    return std::shared_ptr<TableAngleForceCompute>(new TableAngleForceCompute(sysdef,width));
    }

#ifdef ENABLE_CUDA
//! AngleForceCompute creator for bond_force_basic_tests()
std::shared_ptr<TableAngleForceCompute> gpu_tf_creator(std::shared_ptr<SystemDefinition> sysdef,unsigned int width)
    {
    return std::shared_ptr<TableAngleForceCompute>(new TableAngleForceComputeGPU(sysdef,width));
    }
#endif

//! test case for angle forces on the CPU
UP_TEST( TableAngleForceCompute_basic )
    {
    printf(" IN UP_TEST: CPU \n");
    angleforce_creator tf_creator = bind(base_class_tf_creator, _1,_2);
    angle_force_basic_tests(tf_creator, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

#ifdef ENABLE_CUDA
//! test case for angle forces on the GPU
UP_TEST( TableAngleForceComputeGPU_basic )
    {
    printf(" IN UP_TEST: GPU \n");
    angleforce_creator tf_creator = bind(gpu_tf_creator, _1,_2);
    angle_force_basic_tests(tf_creator, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
#if 0
//! test case for comparing bond GPU and CPU BondForceComputes
UP_TEST( TableAngleForceComputeGPU_compare )
    {
    angleforce_creator tf_creator_gpu = bind(gpu_tf_creator, _1);
    angleforce_creator tf_creator = bind(base_class_tf_creator, _1);
    angle_force_comparison_tests(tf_creator, tf_creator_gpu, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
#endif
#endif
