// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// this include is necessary to get MPI included before anything else to support intel MPI
#include "hoomd/ExecutionConfiguration.h"

#include <iostream>

#include <functional>

#include "hoomd/md/AllBondPotentials.h"
#include "hoomd/ConstForceCompute.h"

#include "hoomd/Initializers.h"
#include "hoomd/SnapshotSystemData.h"

using namespace std;
using namespace std::placeholders;

/*! \file fenebond_force_test.cc
    \brief Implements unit tests for BondForceCompute and child classes
    \ingroup unit_tests
*/

#include "hoomd/test/upp11_config.h"
HOOMD_UP_MAIN();

//! Typedef to make using the std::function factory easier
typedef std::function<std::shared_ptr<PotentialBondFENE>  (std::shared_ptr<SystemDefinition> sysdef)> bondforce_creator;

//! Perform some simple functionality tests of any BondForceCompute
void bond_force_basic_tests(bondforce_creator bf_creator, std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    /////////////////////////////////////////////////////////
    // start with the simplest possible test: 2 particles in a huge box with only one bond type
    std::shared_ptr<SystemDefinition> sysdef_2(new SystemDefinition(2, BoxDim(1000.0), 1, 1, 0, 0, 0,  exec_conf));
    std::shared_ptr<ParticleData> pdata_2 = sysdef_2->getParticleData();
    pdata_2->setFlags(~PDataFlags(0));

    pdata_2->setPosition(0,make_scalar3(0.0,0.0,0.0));
    pdata_2->setPosition(1,make_scalar3(0.9,0.0,0.0));

    // create the bond force compute to check
    std::shared_ptr<PotentialBondFENE> fc_2 = bf_creator(sysdef_2);
    fc_2->setParams(0, make_scalar4(Scalar(1.5), Scalar(1.1), Scalar(1.0), Scalar(1.0)));

    // compute the force and check the results
    fc_2->compute(0);

    {
    GlobalArray<Scalar4>& force_array_1 =  fc_2->getForceArray();
    GlobalArray<Scalar>& virial_array_1 =  fc_2->getVirialArray();
    unsigned int pitch = virial_array_1.getPitch();
    ArrayHandle<Scalar4> h_force_1(force_array_1,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_1(virial_array_1,access_location::host,access_mode::read);
    // check that the force is correct, it should be 0 since we haven't created any bonds yet
    MY_CHECK_SMALL(h_force_1.data[0].x, tol_small);
    MY_CHECK_SMALL(h_force_1.data[0].y, tol_small);
    MY_CHECK_SMALL(h_force_1.data[0].z, tol_small);
    MY_CHECK_SMALL(h_force_1.data[0].w, tol_small);
    MY_CHECK_SMALL(h_virial_1.data[0*pitch+0], tol_small);
    MY_CHECK_SMALL(h_virial_1.data[1*pitch+0], tol_small);
    MY_CHECK_SMALL(h_virial_1.data[2*pitch+0], tol_small);
    MY_CHECK_SMALL(h_virial_1.data[3*pitch+0], tol_small);
    MY_CHECK_SMALL(h_virial_1.data[4*pitch+0], tol_small);
    MY_CHECK_SMALL(h_virial_1.data[5*pitch+0], tol_small);
    }

    // add a bond and check again
    sysdef_2->getBondData()->addBondedGroup(Bond(0, 0, 1));
    fc_2->compute(1);

    {
    // this time there should be a force
    GlobalArray<Scalar4>& force_array_2 =  fc_2->getForceArray();
    GlobalArray<Scalar>& virial_array_2 =  fc_2->getVirialArray();
    unsigned int pitch = virial_array_2.getPitch();
    ArrayHandle<Scalar4> h_force_2(force_array_2,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_2(virial_array_2,access_location::host,access_mode::read);
    MY_CHECK_CLOSE(h_force_2.data[0].x, -30.581156, tol);
    MY_CHECK_SMALL(h_force_2.data[0].y, tol_small);
    MY_CHECK_SMALL(h_force_2.data[0].z, tol_small);
    MY_CHECK_CLOSE(h_force_2.data[0].w, 1.33177578 + 0.25/2, tol);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_2.data[0*pitch+0]
                                       +h_virial_2.data[3*pitch+0]
                                       +h_virial_2.data[5*pitch+0]), 4.58717, tol);

    // check that the two forces are negatives of each other
    MY_CHECK_CLOSE(h_force_2.data[0].x, -h_force_2.data[1].x, tol);
    MY_CHECK_CLOSE(h_force_2.data[0].y, -h_force_2.data[1].y, tol);
    MY_CHECK_CLOSE(h_force_2.data[0].z, -h_force_2.data[1].z, tol);
    MY_CHECK_CLOSE(h_force_2.data[0].w, h_force_2.data[1].w, tol);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_2.data[0*pitch+1]
                                       +h_virial_2.data[3*pitch+1]
                                       +h_virial_2.data[5*pitch+1]), 4.58717, tol);
    }

    // rearrange the two particles in memory and see if they are properly updated
    {
    ArrayHandle<Scalar4> h_pos(pdata_2->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_tag(pdata_2->getTags(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_rtag(pdata_2->getRTags(), access_location::host, access_mode::readwrite);

    h_pos.data[0].x = Scalar(0.9);
    h_pos.data[1].x = Scalar(0.0);
    h_tag.data[0] = 1;
    h_tag.data[1] = 0;
    h_rtag.data[0] = 1;
    h_rtag.data[1] = 0;
    }

    // notify that we made the sort
    pdata_2->notifyParticleSort();
    // recompute at the same timestep, the forces should still be updated
    fc_2->compute(1);

    {
    // this time there should be a force
    GlobalArray<Scalar4>& force_array_3 =  fc_2->getForceArray();
    GlobalArray<Scalar>& virial_array_3 =  fc_2->getVirialArray();
    ArrayHandle<Scalar4> h_force_3(force_array_3,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_3(virial_array_3,access_location::host,access_mode::read);
    MY_CHECK_CLOSE(h_force_3.data[0].x, 30.581156, tol);
    MY_CHECK_CLOSE(h_force_3.data[1].x, -30.581156, tol);
    }

    ////////////////////////////////////////////////////////////////////
    // now, lets do a more thorough test and include boundary conditions
    // there are way too many permutations to test here, so I will simply
    // test +x, -x, +y, -y, +z, and -z independently
    // build a 6 particle system with particles across each boundary
    // also test more than one type of bond
    std::shared_ptr<SystemDefinition> sysdef_6(new SystemDefinition(6, BoxDim(20.0, 40.0, 60.0), 1, 3, 0, 0, 0, exec_conf));
    std::shared_ptr<ParticleData> pdata_6 = sysdef_6->getParticleData();
    pdata_6->setFlags(~PDataFlags(0));

    pdata_6->setPosition(0, make_scalar3(-9.6,0.0,0.0));
    pdata_6->setPosition(1, make_scalar3(9.6, 0.0,0.0));
    pdata_6->setPosition(2, make_scalar3(0.0,-19.6,0.0));
    pdata_6->setPosition(3, make_scalar3(0.0,19.6,0.0));
    pdata_6->setPosition(4, make_scalar3(0.0,0.0,-29.6));
    pdata_6->setPosition(5, make_scalar3(0.0,0.0,29.6));

    std::shared_ptr<PotentialBondFENE> fc_6 = bf_creator(sysdef_6);
    fc_6->setParams(0, make_scalar4(Scalar(1.5), Scalar(1.1), Scalar(1.0), Scalar(1.0)));
    fc_6->setParams(1, make_scalar4(Scalar(2.0*1.5), Scalar(1.1), Scalar(1.0), Scalar(1.0)));
    fc_6->setParams(2, make_scalar4(Scalar(1.5), Scalar(1.0), Scalar(1.0), Scalar(1.0)));

    sysdef_6->getBondData()->addBondedGroup(Bond(0, 0,1));
    sysdef_6->getBondData()->addBondedGroup(Bond(1, 2,3));
    sysdef_6->getBondData()->addBondedGroup(Bond(2, 4,5));

    fc_6->compute(0);

    {
    // check that the forces are correctly computed
    GlobalArray<Scalar4>& force_array_4 =  fc_6->getForceArray();
    GlobalArray<Scalar>& virial_array_4 =  fc_6->getVirialArray();
    unsigned int pitch = virial_array_4.getPitch();
    ArrayHandle<Scalar4> h_force_4(force_array_4,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_4(virial_array_4,access_location::host,access_mode::read);
    MY_CHECK_CLOSE(h_force_4.data[0].x, 187.121131, tol);
    MY_CHECK_SMALL(h_force_4.data[0].y, tol_small);
    MY_CHECK_SMALL(h_force_4.data[0].z, tol_small);
    MY_CHECK_CLOSE(h_force_4.data[0].w, 5.71016443 + 0.25/2, tol);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_4.data[0*pitch+0]
                                       +h_virial_4.data[3*pitch+0]
                                       +h_virial_4.data[5*pitch+0]), 24.9495, tol);

    MY_CHECK_CLOSE(h_force_4.data[1].x, -187.121131, tol);
    MY_CHECK_SMALL(h_force_4.data[1].y, tol_small);
    MY_CHECK_SMALL(h_force_4.data[1].z, tol_small);
    MY_CHECK_CLOSE(h_force_4.data[1].w, 5.71016443 + 0.25/2, tol);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_4.data[0*pitch+1]
                                       +h_virial_4.data[3*pitch+1]
                                       +h_virial_4.data[5*pitch+1]), 24.9495, tol);

    MY_CHECK_SMALL(h_force_4.data[2].x, tol_small);
    MY_CHECK_CLOSE(h_force_4.data[2].y, 184.573762, tol);
    MY_CHECK_SMALL(h_force_4.data[2].z, tol_small);
    MY_CHECK_CLOSE(h_force_4.data[2].w,  6.05171988 + 0.25/2, tol);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_4.data[0*pitch+2]
                                       +h_virial_4.data[3*pitch+2]
                                       +h_virial_4.data[5*pitch+2]), 24.6098, tol);

    MY_CHECK_SMALL(h_force_4.data[3].x, tol_small);
    MY_CHECK_CLOSE(h_force_4.data[3].y, -184.573762, tol);
    MY_CHECK_SMALL(h_force_4.data[3].z, tol_small);
    MY_CHECK_CLOSE(h_force_4.data[3].w, 6.05171988 + 0.25/2, tol);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_4.data[0*pitch+3]
                                       +h_virial_4.data[3*pitch+3]
                                       +h_virial_4.data[5*pitch+3]), 24.6098, tol);

    MY_CHECK_SMALL(h_force_4.data[4].x, tol_small);
    MY_CHECK_SMALL(h_force_4.data[4].y, tol_small);
    MY_CHECK_CLOSE(h_force_4.data[4].z, 186.335166, tol);
    MY_CHECK_CLOSE(h_force_4.data[4].w, 5.7517282 + 0.25/2, tol);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_4.data[0*pitch+4]
                                       +h_virial_4.data[3*pitch+4]
                                       +h_virial_4.data[5*pitch+4]), 24.8447, tol);

    MY_CHECK_SMALL(h_force_4.data[5].x, tol_small);
    MY_CHECK_SMALL(h_force_4.data[5].y, tol_small);
    MY_CHECK_CLOSE(h_force_4.data[5].z, -186.335166, tol);
    MY_CHECK_CLOSE(h_force_4.data[5].w,  5.7517282 + 0.25/2, tol);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_4.data[0*pitch+5]
                                       +h_virial_4.data[3*pitch+5]
                                       +h_virial_4.data[5*pitch+5]), 24.8447, tol);
    }

    // one more test: this one will test two things:
    // 1) That the forces are computed correctly even if the particles are rearranged in memory
    // and 2) That two forces can add to the same particle
    std::shared_ptr<SystemDefinition> sysdef_4(new SystemDefinition(4, BoxDim(100.0, 100.0, 100.0), 1, 1, 0, 0, 0, exec_conf));
    std::shared_ptr<ParticleData> pdata_4 = sysdef_4->getParticleData();
    pdata_4->setFlags(~PDataFlags(0));

    {
    ArrayHandle<Scalar4> h_pos(pdata_4->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_tag(pdata_4->getTags(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_rtag(pdata_4->getRTags(), access_location::host, access_mode::readwrite);

    // make a square of particles
    h_pos.data[0].x = 0.0; h_pos.data[0].y = 0.0; h_pos.data[0].z = 0.0;
    h_pos.data[1].x = 1.0; h_pos.data[1].y = 0; h_pos.data[1].z = 0.0;
    h_pos.data[2].x = 0; h_pos.data[2].y = 1.0; h_pos.data[2].z = 0.0;
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

    // build the bond force compute and try it out
    std::shared_ptr<PotentialBondFENE> fc_4 = bf_creator(sysdef_4);
    fc_4->setParams(0, make_scalar4(Scalar(1.5), Scalar(1.75), Scalar(pow(1.2,12.0)), Scalar(pow(1.2,6.0))));
    // only add bonds on the left, top, and bottom of the square
    sysdef_4->getBondData()->addBondedGroup(Bond(0, 2,3));
    sysdef_4->getBondData()->addBondedGroup(Bond(0, 2,0));
    sysdef_4->getBondData()->addBondedGroup(Bond(0, 0,1));

    fc_4->compute(0);

    {
    GlobalArray<Scalar4>& force_array_5 =  fc_4->getForceArray();
    GlobalArray<Scalar>& virial_array_5 =  fc_4->getVirialArray();
    unsigned int pitch = virial_array_5.getPitch();
    ArrayHandle<Scalar4> h_force_5(force_array_5,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_5(virial_array_5,access_location::host,access_mode::read);
    // the right two particles should only have a force pulling them left
    MY_CHECK_CLOSE(h_force_5.data[1].x, 86.85002865, tol);
    MY_CHECK_CLOSE(h_force_5.data[1].y, 0, tol);
    MY_CHECK_CLOSE(h_force_5.data[1].z, 0, tol);
    MY_CHECK_CLOSE(h_force_5.data[1].w, 7.08810039/2.0, tol);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_5.data[0*pitch+1]
                                       +h_virial_5.data[3*pitch+1]
                                       +h_virial_5.data[5*pitch+1]), 14.475, tol);

    MY_CHECK_CLOSE(h_force_5.data[3].x, 86.85002865, tol);
    MY_CHECK_CLOSE(h_force_5.data[3].y, 0, tol);
    MY_CHECK_CLOSE(h_force_5.data[3].z, 0, tol);
    MY_CHECK_CLOSE(h_force_5.data[3].w, 7.08810039/2.0, tol);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_5.data[0*pitch+3]
                                       +h_virial_5.data[3*pitch+3]
                                       +h_virial_5.data[5*pitch+3]), 14.475, tol);

    // the bottom left particle should have a force pulling up and to the right
    MY_CHECK_CLOSE(h_force_5.data[0].x, -86.850028653, tol);
    MY_CHECK_CLOSE(h_force_5.data[0].y, -86.85002865, tol);
    MY_CHECK_CLOSE(h_force_5.data[0].z, 0, tol);
    MY_CHECK_CLOSE(h_force_5.data[0].w, 7.08810039, tol);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_5.data[0*pitch+0]
                                       +h_virial_5.data[3*pitch+0]
                                       +h_virial_5.data[5*pitch+0]), 2.0*14.475, tol);

    // and the top left particle should have a force pulling down and to the right
    MY_CHECK_CLOSE(h_force_5.data[2].x, -86.85002865, tol);
    MY_CHECK_CLOSE(h_force_5.data[2].y, 86.85002865, tol);
    MY_CHECK_CLOSE(h_force_5.data[2].z, 0, tol);
    MY_CHECK_CLOSE(h_force_5.data[2].w, 7.08810039, tol);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_5.data[0*pitch+2]
                                       +h_virial_5.data[3*pitch+2]
                                       +h_virial_5.data[5*pitch+2]), 2.0*14.475, tol);
    }
    }

//! Compares the output of two PotentialBondFENEs
void bond_force_comparison_tests(bondforce_creator bf_creator1,
                                 bondforce_creator bf_creator2,
                                 std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    const unsigned int M = 10;
    const unsigned int N = M*M*M;

    // create a particle system to sum forces on
    // use a simple cubic array of particles so that random bonds
    // don't result in huge forces on a random particle arrangement
    SimpleCubicInitializer sc_init(M, 1.5, "A");
    std::shared_ptr< SnapshotSystemData<Scalar> > snap = sc_init.getSnapshot();
    snap->bond_data.type_mapping.push_back("A");
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));
    std::shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    pdata->setFlags(~PDataFlags(0));

    std::shared_ptr<PotentialBondFENE> fc1 = bf_creator1(sysdef);
    std::shared_ptr<PotentialBondFENE> fc2 = bf_creator2(sysdef);
    fc1->setParams(0, make_scalar4(Scalar(300.0), Scalar(1.6), Scalar(1.0), Scalar(1.0)));
    fc2->setParams(0, make_scalar4(Scalar(300.0), Scalar(1.6), Scalar(1.0), Scalar(1.0)));

    // displace particles a little so all forces aren't alike
    {
    ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);
    BoxDim box = pdata->getBox();
    for (unsigned int i = 0; i < N; i++)
        {
        h_pos.data[i].x += Scalar((rand())/Scalar(RAND_MAX) - 0.5) * Scalar(0.01);
        h_pos.data[i].y += Scalar((rand())/Scalar(RAND_MAX) - 0.5) * Scalar(0.05);
        h_pos.data[i].z += Scalar((rand())/Scalar(RAND_MAX) - 0.5) * Scalar(0.001);

        int3 img;
        box.wrap(h_pos.data[i], img);
        }
    }

    // add bonds
    for (unsigned int i = 0; i < M; i++)
        for (unsigned int j = 0; j < M; j++)
            for (unsigned int k = 0; k < M-1; k++)
                {
                sysdef->getBondData()->addBondedGroup(Bond(0, i*M*M + j*M + k, i*M*M + j*M + k + 1));
                }


    // compute the forces
    fc1->compute(0);
    fc2->compute(0);

    {
    // verify that the forces are identical (within roundoff errors)
    GlobalArray<Scalar4>& force_array_6 =  fc1->getForceArray();
    GlobalArray<Scalar>& virial_array_6 =  fc1->getVirialArray();
    unsigned int pitch = virial_array_6.getPitch();
    ArrayHandle<Scalar4> h_force_6(force_array_6,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_6(virial_array_6,access_location::host,access_mode::read);
    GlobalArray<Scalar4>& force_array_7 =  fc2->getForceArray();
    GlobalArray<Scalar>& virial_array_7 =  fc2->getVirialArray();
    ArrayHandle<Scalar4> h_force_7(force_array_7,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_7(virial_array_7,access_location::host,access_mode::read);

    // compare average deviation between the two computes
    double deltaf2 = 0.0;
    double deltape2 = 0.0;
    double deltav2[6];
    for (unsigned int i = 0; i < 6; i++)
        deltav2[i] = 0.0;

    for (unsigned int i = 0; i < N; i++)
        {
        deltaf2 += double(h_force_7.data[i].x - h_force_6.data[i].x) * double(h_force_7.data[i].x - h_force_6.data[i].x);
        deltaf2 += double(h_force_7.data[i].y - h_force_6.data[i].y) * double(h_force_7.data[i].y - h_force_6.data[i].y);
        deltaf2 += double(h_force_7.data[i].z - h_force_6.data[i].z) * double(h_force_7.data[i].z - h_force_6.data[i].z);
        deltape2 += double(h_force_7.data[i].w - h_force_6.data[i].w) * double(h_force_7.data[i].w - h_force_6.data[i].w);
        for (unsigned int j = 0; j < 6; j++)
            deltav2[j] += double(h_virial_7.data[j*pitch+i] - h_virial_6.data[j*pitch+i]) * double(h_virial_7.data[j*pitch+i] - h_virial_6.data[j*pitch+i]);

        // also check that each individual calculation is somewhat close
        }
    deltaf2 /= double(pdata->getN());
    deltape2 /= double(pdata->getN());
    for (unsigned int i = 0; i < 6; i++)
        deltav2[i] /= double(pdata->getN());
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

//! PotentialBondFENE creator for bond_force_basic_tests()
std::shared_ptr<PotentialBondFENE> base_class_bf_creator(std::shared_ptr<SystemDefinition> sysdef)
    {
    return std::shared_ptr<PotentialBondFENE>(new PotentialBondFENE(sysdef));
    }

#ifdef ENABLE_CUDA
//! PotentialBondFENE creator for bond_force_basic_tests()
std::shared_ptr<PotentialBondFENE> gpu_bf_creator(std::shared_ptr<SystemDefinition> sysdef)
    {
    return std::shared_ptr<PotentialBondFENE>(new PotentialBondFENEGPU(sysdef));
    }
#endif

//! test case for bond forces on the CPU
UP_TEST( PotentialBondFENE_basic )
    {
    bondforce_creator bf_creator = bind(base_class_bf_creator, _1);
    bond_force_basic_tests(bf_creator, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

#ifdef ENABLE_CUDA
//! test case for bond forces on the GPU
UP_TEST( PotentialBondFENEGPU_basic )
    {
    bondforce_creator bf_creator = bind(gpu_bf_creator, _1);
    bond_force_basic_tests(bf_creator, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! test case for comparing bond GPU and CPU BondForceComputes
UP_TEST( PotentialBondFENEGPU_compare )
    {
    bondforce_creator bf_creator_gpu = bind(gpu_bf_creator, _1);
    bondforce_creator bf_creator = bind(base_class_bf_creator, _1);
    bond_force_comparison_tests(bf_creator, bf_creator_gpu, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

#endif
