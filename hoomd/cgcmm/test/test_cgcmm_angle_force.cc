// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// this include is necessary to get MPI included before anything else to support intel MPI
#include "hoomd/ExecutionConfiguration.h"


#include <iostream>

#include <functional>

#include "hoomd/cgcmm/CGCMMAngleForceCompute.h"
#include "hoomd/ConstForceCompute.h"
#ifdef ENABLE_CUDA
#include "hoomd/cgcmm/CGCMMAngleForceComputeGPU.h"
#endif

#include <stdio.h>

#include "hoomd/Initializers.h"
#include "hoomd/SnapshotSystemData.h"

using namespace std;
using namespace std::placeholders;

#include "hoomd/test/upp11_config.h"
HOOMD_UP_MAIN();

//! Typedef to make using the std::function factory easier
typedef std::function<std::shared_ptr<CGCMMAngleForceCompute>  (std::shared_ptr<SystemDefinition> sysdef)> cgcmm_angleforce_creator;

//! Perform some simple functionality tests of any AngleForceCompute
void angle_force_basic_tests(cgcmm_angleforce_creator af_creator, std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    /////////////////////////////////////////////////////////
    // start with the simplest possible test: 3 particles in a huge box with only one angle type !!!! NO ANGLES
    std::shared_ptr<SystemDefinition> sysdef_3(new SystemDefinition(3, BoxDim(1000.0), 1, 1, 1, 0, 0,  exec_conf));
    std::shared_ptr<ParticleData> pdata_3 = sysdef_3->getParticleData();

    pdata_3->setPosition(0,make_scalar3(-1.23,2.0,0.1)); // put atom a at (-1,0,0.1)
    pdata_3->setPosition(1,make_scalar3(1.0,1.0,1.0)); // put atom b at (0,0,0)
    pdata_3->setPosition(2,make_scalar3(1.0,0.0,0.5)); // put atom c at (1,0,0.5)

    // create the angle force compute to check
    std::shared_ptr<CGCMMAngleForceCompute> fc_3 = af_creator(sysdef_3);
    fc_3->setParams(0, 1.0, 0.785398, 1, 1.0, 2.0); // type=0, K=1.0,theta_0=pi/4=0.785398, cg_type=1, eps=2.0, sigma=1.0

    // compute the force and check the results
    fc_3->compute(0);

    {
    //New Force data access pattern
    GlobalArray<Scalar4>& force_array_1 =  fc_3->getForceArray();
    GlobalArray<Scalar>& virial_array_1 =  fc_3->getVirialArray();
    ArrayHandle<Scalar4> h_force_1(force_array_1,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_1(virial_array_1,access_location::host,access_mode::read);

    // check that the force is correct, it should be 0 since we haven't created any angles yet
    MY_CHECK_SMALL(h_force_1.data[0].x, tol);
    MY_CHECK_SMALL(h_force_1.data[0].y, tol);
    MY_CHECK_SMALL(h_force_1.data[0].z, tol);
    MY_CHECK_SMALL(h_force_1.data[0].w, tol);
    MY_CHECK_SMALL(h_virial_1.data[0*virial_array_1.getPitch()], tol);
    MY_CHECK_SMALL(h_virial_1.data[1*virial_array_1.getPitch()], tol);
    MY_CHECK_SMALL(h_virial_1.data[2*virial_array_1.getPitch()], tol);
    MY_CHECK_SMALL(h_virial_1.data[3*virial_array_1.getPitch()], tol);
    MY_CHECK_SMALL(h_virial_1.data[4*virial_array_1.getPitch()], tol);
    MY_CHECK_SMALL(h_virial_1.data[5*virial_array_1.getPitch()], tol);
    }

    // add an angle and check again
    sysdef_3->getAngleData()->addBondedGroup(Angle(0,0,1,2)); // add type 0 between angle formed by atom 0-1-2
    fc_3->compute(1);

    {
    // this time there should be a force
    GlobalArray<Scalar4>& force_array_2 =  fc_3->getForceArray();
    GlobalArray<Scalar>& virial_array_2 =  fc_3->getVirialArray();
    ArrayHandle<Scalar4> h_force_2(force_array_2,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_2(virial_array_2,access_location::host,access_mode::read);
    MY_CHECK_CLOSE(h_force_2.data[0].x, -0.061684, tol);
    MY_CHECK_CLOSE(h_force_2.data[0].y, -0.313469, tol);
    MY_CHECK_CLOSE(h_force_2.data[0].z, -0.195460, tol);
    MY_CHECK_CLOSE(h_force_2.data[0].w, 0.158576, tol);
    MY_CHECK_SMALL(h_virial_2.data[0*virial_array_2.getPitch()]
                         +h_virial_2.data[3*virial_array_2.getPitch()]
                         +h_virial_2.data[5*virial_array_2.getPitch()], tol);
    }

    // rearrange the two particles in memory and see if they are properly updated
    {
    ArrayHandle<Scalar4> h_pos(pdata_3->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_tag(pdata_3->getTags(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_rtag(pdata_3->getRTags(), access_location::host, access_mode::readwrite);

    h_pos.data[1].x = -1.23;
    h_pos.data[1].y = 2.0;
    h_pos.data[1].z = 0.1;

    h_pos.data[0].x = 1.0;
    h_pos.data[0].y = 1.0;
    h_pos.data[0].z = 1.0;

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
    GlobalArray<Scalar4>& force_array_3 =  fc_3->getForceArray();
    GlobalArray<Scalar>& virial_array_3 =  fc_3->getVirialArray();
    ArrayHandle<Scalar4> h_force_3(force_array_3,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_3(virial_array_3,access_location::host,access_mode::read);

    MY_CHECK_CLOSE(h_force_3.data[1].x, -0.0616840, tol);
    MY_CHECK_CLOSE(h_force_3.data[1].y, -0.3134695, tol);
    MY_CHECK_CLOSE(h_force_3.data[1].z, -0.195460, tol);
    MY_CHECK_CLOSE(h_force_3.data[1].w, 0.158576, tol);
    MY_CHECK_SMALL(h_virial_3.data[0*virial_array_3.getPitch()+1]
                         +h_virial_3.data[3*virial_array_3.getPitch()+1]
                         +h_virial_3.data[5*virial_array_3.getPitch()+1], tol);
    }
    //pdata_3->release();

    ////////////////////////////////////////////////////////////////////
    // now, lets do a more thorough test and include boundary conditions
    // there are way too many permutations to test here, so I will simply
    // test +x, -x, +y, -y, +z, and -z independently
    // build a 6 particle system with particles across each boundary
    // also test more than one type of angle
    unsigned int num_angles_to_test = 2;
    std::shared_ptr<SystemDefinition> sysdef_6(new SystemDefinition(6, BoxDim(20.0, 40.0, 60.0), 1, 1, num_angles_to_test, 0, 0, exec_conf));
    std::shared_ptr<ParticleData> pdata_6 = sysdef_6->getParticleData();

    pdata_6->setPosition(0,make_scalar3(-9.6, 0.0, 0.0));
    pdata_6->setPosition(1,make_scalar3( 9.6, 0.0, 0.0));
    pdata_6->setPosition(2,make_scalar3( 0.0, -19.6, 0.0));
    pdata_6->setPosition(3,make_scalar3( 0.0, 19.6, 0.0));
    pdata_6->setPosition(4,make_scalar3( 0.0, 0.0, -29.6));
    pdata_6->setPosition(5,make_scalar3( 0.0, 0.0,  29.6));

    std::shared_ptr<CGCMMAngleForceCompute> fc_6 = af_creator(sysdef_6);
    fc_6->setParams(0, 1.0, 0.785398, 1, 1.0, 2.0);
    fc_6->setParams(1, 2.0, 1.46, 2, 1.0, 2.0);

    sysdef_6->getAngleData()->addBondedGroup(Angle(0, 0,1,2));
    sysdef_6->getAngleData()->addBondedGroup(Angle(1, 3,4,5));

    fc_6->compute(0);

    {
    // check that the forces are correctly computed
    GlobalArray<Scalar4>& force_array_4 =  fc_6->getForceArray();
    GlobalArray<Scalar>& virial_array_4 =  fc_6->getVirialArray();
    ArrayHandle<Scalar4> h_force_4(force_array_4,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_4(virial_array_4,access_location::host,access_mode::read);


    MY_CHECK_SMALL(h_force_4.data[0].x, tol);
    MY_CHECK_CLOSE(h_force_4.data[0].y, -1.5510634,tol);
    MY_CHECK_SMALL(h_force_4.data[0].z, tol);
    MY_CHECK_CLOSE(h_force_4.data[0].w, 0.256618, tol);
    MY_CHECK_SMALL(h_virial_4.data[0*virial_array_4.getPitch()+0]
                         +h_virial_4.data[3*virial_array_4.getPitch()+0]
                         +h_virial_4.data[5*virial_array_4.getPitch()+0], tol);

    MY_CHECK_CLOSE(h_force_4.data[1].x, -0.0510595, tol);
    MY_CHECK_CLOSE(h_force_4.data[1].y, 1.5760721,tol);
    MY_CHECK_SMALL(h_force_4.data[1].z, tol);
    MY_CHECK_CLOSE(h_force_4.data[1].w, 0.256618, tol);
    MY_CHECK_SMALL(h_virial_4.data[0*virial_array_4.getPitch()+1]
                         +h_virial_4.data[3*virial_array_4.getPitch()+1]
                         +h_virial_4.data[5*virial_array_4.getPitch()+1], tol);

    MY_CHECK_CLOSE(h_force_4.data[2].x, 0.0510595,tol);
    MY_CHECK_CLOSE(h_force_4.data[2].y, -0.0250087, tol);
    MY_CHECK_SMALL(h_force_4.data[2].z, tol);
    MY_CHECK_CLOSE(h_force_4.data[2].w, 0.256618, tol);
    MY_CHECK_SMALL(h_virial_4.data[0*virial_array_4.getPitch()+2]
                         +h_virial_4.data[3*virial_array_4.getPitch()+2]
                         +h_virial_4.data[5*virial_array_4.getPitch()+2], tol);

    MY_CHECK_SMALL(h_force_4.data[3].x, tol);
    MY_CHECK_CLOSE(h_force_4.data[3].y, 0.0515151, tol);
    MY_CHECK_CLOSE(h_force_4.data[3].z, -0.03411135,tol);
    MY_CHECK_CLOSE(h_force_4.data[3].w, 0.400928, tol);
    MY_CHECK_SMALL(h_virial_4.data[0*virial_array_4.getPitch()+3]
                         +h_virial_4.data[3*virial_array_4.getPitch()+3]
                         +h_virial_4.data[5*virial_array_4.getPitch()+3], tol);

    MY_CHECK_SMALL(h_force_4.data[4].x, tol);
    MY_CHECK_CLOSE(h_force_4.data[4].y, -2.793305,tol);
    MY_CHECK_CLOSE(h_force_4.data[4].z, 0.0341109, tol);
    MY_CHECK_CLOSE(h_force_4.data[4].w, 0.400928, tol);
    MY_CHECK_SMALL(h_virial_4.data[0*virial_array_4.getPitch()+4]
                         +h_virial_4.data[3*virial_array_4.getPitch()+4]
                         +h_virial_4.data[5*virial_array_4.getPitch()+4], tol);

    MY_CHECK_SMALL(h_force_4.data[5].x, tol);
    MY_CHECK_CLOSE(h_force_4.data[5].y, 2.74178982,tol);
    MY_CHECK_SMALL(h_force_4.data[5].z, tol);
    MY_CHECK_CLOSE(h_force_4.data[5].w, 0.400928, tol);
    MY_CHECK_SMALL(h_virial_4.data[0*virial_array_4.getPitch()+5]
                         +h_virial_4.data[3*virial_array_4.getPitch()+5]
                         +h_virial_4.data[5*virial_array_4.getPitch()+5], tol);
    }

    //////////////////////////////////////////////////////////////////////
    // THE DREADED 4 PARTICLE TEST -- see CGCMMAngleForceGPU.cu //
    //////////////////////////////////////////////////////////////////////
    // one more test: this one will test two things:
    // 1) That the forces are computed correctly even if the particles are rearranged in memory
    // and 2) That two forces can add to the same particle
    std::shared_ptr<SystemDefinition> sysdef_4(new SystemDefinition(4, BoxDim(100.0, 100.0, 100.0), 1, 1, 3, 0, 0, exec_conf));
    std::shared_ptr<ParticleData> pdata_4 = sysdef_4->getParticleData();

    // make a square of particles
    {
    ArrayHandle<Scalar4> h_pos(pdata_4->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_tag(pdata_4->getTags(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_rtag(pdata_4->getRTags(), access_location::host, access_mode::readwrite);
    h_pos.data[0].x = 0.0; h_pos.data[0].y = 0.0; h_pos.data[0].z = 0.0;
    h_pos.data[1].x = 1.0; h_pos.data[1].y = 0; h_pos.data[1].z = 0.0;
    h_pos.data[2].x = 0.0; h_pos.data[2].y = 1.0; h_pos.data[2].z = 0.0;
    h_pos.data[3].x = 1.0; h_pos.data[3].y = 1.0; h_pos.data[3].z = 0.0;

    h_tag.data[0] = 2;
    h_tag.data[1] = 3;
    h_tag.data[2] = 0;
    h_tag.data[3] = 1;
    h_rtag.data[h_tag.data[0]] = 0;
    h_rtag.data[h_tag.data[1]] = 1;
    h_rtag.data[h_tag.data[2]] = 2;
    h_rtag.data[h_tag.data[3]] = 3;
    }

    // build the angle force compute and try it out
    std::shared_ptr<CGCMMAngleForceCompute> fc_4 = af_creator(sysdef_4);
//  fc_4->setParams(0, 1.5, 1.75, 2, 1.0, 2.0);
    fc_4->setParams(0, 1.0, 0.785398, 1, 1.0, 0.45);
    fc_4->setParams(1, 12.3, 0.21112, 2, 1.0, 0.45);
    fc_4->setParams(2, 22.0, 0.3772, 3, 1.0, 0.65);
    // only add angles on the left, top, and bottom of the square
    sysdef_4->getAngleData()->addBondedGroup(Angle(0, 0,1,2));
    sysdef_4->getAngleData()->addBondedGroup(Angle(1, 1,2,3));
    sysdef_4->getAngleData()->addBondedGroup(Angle(0, 0,1,3));

    fc_4->compute(0);

    {
    GlobalArray<Scalar4>& force_array_5 =  fc_4->getForceArray();
    GlobalArray<Scalar>& virial_array_5 =  fc_4->getVirialArray();
    ArrayHandle<Scalar4> h_force_5(force_array_5,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_5(virial_array_5,access_location::host,access_mode::read);

    MY_CHECK_CLOSE(h_force_5.data[0].x, -3.531810,tol);
    MY_CHECK_CLOSE(h_force_5.data[0].y, -3.531810, tol);
    MY_CHECK_SMALL(h_force_5.data[0].z, tol);
    MY_CHECK_CLOSE(h_force_5.data[0].w, 0.676081, tol);
    MY_CHECK_SMALL(h_virial_5.data[0*virial_array_5.getPitch()+0]
                         +h_virial_5.data[3*virial_array_5.getPitch()+0]
                         +h_virial_5.data[5*virial_array_5.getPitch()+0], tol);

    MY_CHECK_CLOSE(h_force_5.data[1].x, -0.785398,tol);
    MY_CHECK_CLOSE(h_force_5.data[1].y, 7.063621,tol);
    MY_CHECK_SMALL(h_force_5.data[1].z, tol);
    MY_CHECK_CLOSE(h_force_5.data[1].w, 0.778889, tol);
    MY_CHECK_SMALL(h_virial_5.data[0*virial_array_5.getPitch()+1]
                         +h_virial_5.data[3*virial_array_5.getPitch()+1]
                         +h_virial_5.data[5*virial_array_5.getPitch()+1], tol);


    MY_CHECK_SMALL(h_force_5.data[2].x, tol);
    MY_CHECK_CLOSE(h_force_5.data[2].y, -0.785399,tol);
    MY_CHECK_SMALL(h_force_5.data[2].z, tol);
    MY_CHECK_CLOSE(h_force_5.data[2].w, 0.102808, tol);
    MY_CHECK_SMALL(h_virial_5.data[0*virial_array_5.getPitch()+2]
                         +h_virial_5.data[3*virial_array_5.getPitch()+2]
                         +h_virial_5.data[5*virial_array_5.getPitch()+2], tol);


    MY_CHECK_CLOSE(h_force_5.data[3].x, 4.317209,tol);
    MY_CHECK_CLOSE(h_force_5.data[3].y, -2.746412,tol);
    MY_CHECK_SMALL(h_force_5.data[3].z, tol);
    MY_CHECK_CLOSE(h_force_5.data[3].w, 0.778889, tol);
    MY_CHECK_SMALL(h_virial_5.data[0*virial_array_5.getPitch()+3]
                         +h_virial_5.data[3*virial_array_5.getPitch()+3]
                         +h_virial_5.data[5*virial_array_5.getPitch()+3], tol);
    }
    }





//! Compares the output of two CGCMMAngleForceComputes
void angle_force_comparison_tests(cgcmm_angleforce_creator af_creator1, cgcmm_angleforce_creator af_creator2, std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    const unsigned int N = 1000;

    // create a particle system to sum forces on
    // just randomly place particles. We don't really care how huge the angle forces get: this is just a unit test
    RandomInitializer rand_init(N, Scalar(0.2), Scalar(0.9), "A");
    std::shared_ptr< SnapshotSystemData<Scalar> > snap = rand_init.getSnapshot();
    snap->angle_data.type_mapping.push_back("A");
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));

    std::shared_ptr<CGCMMAngleForceCompute> fc1 = af_creator1(sysdef);
    std::shared_ptr<CGCMMAngleForceCompute> fc2 = af_creator2(sysdef);
    fc1->setParams(0, Scalar(1.0), Scalar(1.348), 1, Scalar(1.0), Scalar(0.05));
    fc2->setParams(0, Scalar(1.0), Scalar(1.348), 1, Scalar(1.0), Scalar(0.05));

    // add angles
    for (unsigned int i = 0; i < N-2; i++)
        {
        sysdef->getAngleData()->addBondedGroup(Angle(0, i, i+1,i+2));
        }

    // compute the forces
    fc1->compute(0);
    fc2->compute(0);

    {
    // verify that the forces are identical (within roundoff errors)
    GlobalArray<Scalar4>& force_array_6 =  fc1->getForceArray();
    GlobalArray<Scalar>& virial_array_6 =  fc1->getVirialArray();
    ArrayHandle<Scalar4> h_force_6(force_array_6,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_6(virial_array_6,access_location::host,access_mode::read);
    GlobalArray<Scalar4>& force_array_7 =  fc2->getForceArray();
    GlobalArray<Scalar>& virial_array_7 =  fc2->getVirialArray();
    ArrayHandle<Scalar4> h_force_7(force_array_7,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_7(virial_array_7,access_location::host,access_mode::read);

    // compare average deviation between the two computes
    double deltaf2 = 0.0;
    double deltape2 = 0.0;

    for (unsigned int i = 0; i < N; i++)
        {
        deltaf2 += double(h_force_7.data[i].x - h_force_6.data[i].x) * double(h_force_7.data[i].x - h_force_6.data[i].x);
        deltaf2 += double(h_force_7.data[i].y - h_force_6.data[i].y) * double(h_force_7.data[i].y - h_force_6.data[i].y);
        deltaf2 += double(h_force_7.data[i].z - h_force_6.data[i].z) * double(h_force_7.data[i].z - h_force_6.data[i].z);
        deltape2 += double(h_force_7.data[i].w - h_force_6.data[i].w) * double(h_force_7.data[i].w - h_force_6.data[i].w);

        // also check that each individual calculation is somewhat close
        }
    deltaf2 /= double(sysdef->getParticleData()->getN());
    deltape2 /= double(sysdef->getParticleData()->getN());
    CHECK_SMALL(deltaf2, double(tol_small));
    CHECK_SMALL(deltape2, double(tol_small));
    }
    }


//! CGCMMAngleForceCompute creator for angle_force_basic_tests()
std::shared_ptr<CGCMMAngleForceCompute> base_class_af_creator(std::shared_ptr<SystemDefinition> sysdef)
    {
    return std::shared_ptr<CGCMMAngleForceCompute>(new CGCMMAngleForceCompute(sysdef));
    }

#ifdef ENABLE_CUDA
//! AngleForceCompute creator for angle_force_basic_tests()
std::shared_ptr<CGCMMAngleForceCompute> gpu_af_creator(std::shared_ptr<SystemDefinition> sysdef)
    {
    return std::shared_ptr<CGCMMAngleForceCompute>(new CGCMMAngleForceComputeGPU(sysdef));
    }
#endif

//! test case for angle forces on the CPU
UP_TEST( CGCMMAngleForceCompute_basic )
    {
    printf(" IN UP_TEST: CPU \n");
    cgcmm_angleforce_creator af_creator = bind(base_class_af_creator, _1);
    angle_force_basic_tests(af_creator, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

#ifdef ENABLE_CUDA
//! test case for angle forces on the GPU
UP_TEST( CGCMMAngleForceComputeGPU_basic )
    {
    printf(" IN UP_TEST: GPU \n");
    cgcmm_angleforce_creator af_creator = bind(gpu_af_creator, _1);
    angle_force_basic_tests(af_creator, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }


//! test case for comparing angle GPU and CPU AngleForceComputes
UP_TEST( CGCMMAngleForceComputeGPU_compare )
    {
    cgcmm_angleforce_creator af_creator_gpu = bind(gpu_af_creator, _1);
    cgcmm_angleforce_creator af_creator = bind(base_class_af_creator, _1);
    angle_force_comparison_tests(af_creator, af_creator_gpu, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

#endif
