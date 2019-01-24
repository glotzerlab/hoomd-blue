// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// this include is necessary to get MPI included before anything else to support intel MPI
#include "hoomd/ExecutionConfiguration.h"

#include <iostream>
#include <fstream>

#include <functional>
#include <memory>

#include "hoomd/md/AllPairPotentials.h"

#include "hoomd/md/NeighborListTree.h"
#include "hoomd/Initializers.h"
#include "hoomd/SnapshotSystemData.h"

#include <math.h>

using namespace std;
using namespace std::placeholders;

/*! \file gaussian_force_test.cc
    \brief Implements unit tests for PotentialPairGauss and descendants
    \ingroup unit_tests
*/

#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN();




//! Typedef'd PotentialPairGauss factory
typedef std::function<std::shared_ptr<PotentialPairGauss> (std::shared_ptr<SystemDefinition> sysdef,
                                                        std::shared_ptr<NeighborList> nlist)> gaussforce_creator;

//! Test the ability of the gauss force compute to actually calculate forces
void gauss_force_particle_test(gaussforce_creator gauss_creator, std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // this 3-particle test subtly checks several conditions
    // the particles are arranged on the x axis,  1   2   3
    // such that 2 is inside the cutoff radius of 1 and 3, but 1 and 3 are outside the cutoff
    // of course, the buffer will be set on the neighborlist so that 3 is included in it
    // thus, this case tests the ability of the force summer to sum more than one force on
    // a particle and ignore a particle outside the radius

    // periodic boundary conditions will be handled in another test
    std::shared_ptr<SystemDefinition> sysdef_3(new SystemDefinition(3, BoxDim(1000.0), 1, 0, 0, 0, 0, exec_conf));
    std::shared_ptr<ParticleData> pdata_3 = sysdef_3->getParticleData();
    pdata_3->setFlags(~PDataFlags(0));

    pdata_3->setPosition(0,make_scalar3(0.0,0.0,0.0));
    pdata_3->setPosition(1,make_scalar3(1.0,0.0,0.0));
    pdata_3->setPosition(2,make_scalar3(2.0,0.0,0.0));

    std::shared_ptr<NeighborListTree> nlist_3(new NeighborListTree(sysdef_3, Scalar(1.3), Scalar(3.0)));
    std::shared_ptr<PotentialPairGauss> fc_3 = gauss_creator(sysdef_3, nlist_3);
    fc_3->setRcut(0, 0, Scalar(1.3));

    // first test: choose a basic sigma
    Scalar epsilon = Scalar(1.15);
    Scalar sigma = Scalar(0.5);
    fc_3->setParams(0,0,make_scalar2(epsilon,sigma));

    // compute the forces
    fc_3->compute(0);

    {
    GlobalArray<Scalar4>& force_array_1 =  fc_3->getForceArray();
    GlobalArray<Scalar>& virial_array_1 =  fc_3->getVirialArray();
    unsigned int pitch = virial_array_1.getPitch();
    ArrayHandle<Scalar4> h_force_1(force_array_1,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_1(virial_array_1,access_location::host,access_mode::read);
    MY_CHECK_CLOSE(h_force_1.data[0].x, -0.622542302888418, tol);
    MY_CHECK_SMALL(h_force_1.data[0].y, tol_small);
    MY_CHECK_SMALL(h_force_1.data[0].z, tol_small);
    MY_CHECK_CLOSE(h_force_1.data[0].w, 0.155635575722105/2.0, tol);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_1.data[0*pitch+0]
                                       +h_virial_1.data[3*pitch+0]
                                       +h_virial_1.data[5*pitch+0]), 0.103757050481403, tol);

    MY_CHECK_SMALL(h_force_1.data[1].x, tol_small);
    MY_CHECK_SMALL(h_force_1.data[1].y, tol_small);
    MY_CHECK_SMALL(h_force_1.data[1].z, tol_small);
    MY_CHECK_CLOSE(h_force_1.data[1].w, 0.155635575722105, tol);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_1.data[0*pitch+1]
                                       +h_virial_1.data[3*pitch+1]
                                       +h_virial_1.data[5*pitch+1]), 0.103757050481403*2, tol);

    MY_CHECK_CLOSE(h_force_1.data[2].x, 0.622542302888418, tol);
    MY_CHECK_SMALL(h_force_1.data[2].y, tol_small);
    MY_CHECK_SMALL(h_force_1.data[2].z, tol_small);
    MY_CHECK_CLOSE(h_force_1.data[2].w, 0.155635575722105/2.0, tol);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_1.data[0*pitch+2]
                                       +h_virial_1.data[3*pitch+2]
                                       +h_virial_1.data[5*pitch+2]), 0.103757050481403, tol);
    }

    // swap the order of particles 0 ans 2 in memory to check that the force compute handles this properly
    {
    ArrayHandle<Scalar4> h_pos(pdata_3->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_tag(pdata_3->getTags(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_rtag(pdata_3->getRTags(), access_location::host, access_mode::readwrite);

    h_pos.data[2].x = h_pos.data[2].y = h_pos.data[2].z = 0.0;
    h_pos.data[0].x = Scalar(2.0); h_pos.data[0].y = h_pos.data[0].z = 0.0;

    h_tag.data[0] = 2;
    h_tag.data[2] = 0;
    h_rtag.data[0] = 2;
    h_rtag.data[2] = 0;
    }

    // notify the particle data that we changed the order
    pdata_3->notifyParticleSort();

    // recompute the forces at the same timestep, they should be updated
    fc_3->compute(1);

    {
    GlobalArray<Scalar4>& force_array_2 =  fc_3->getForceArray();
    GlobalArray<Scalar>& virial_array_2 =  fc_3->getVirialArray();
    ArrayHandle<Scalar4> h_force_2(force_array_2,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_2(virial_array_2,access_location::host,access_mode::read);
    MY_CHECK_CLOSE(h_force_2.data[0].x, 0.622542302888418, tol);
    MY_CHECK_CLOSE(h_force_2.data[2].x, -0.622542302888418, tol);
    }
    }

//! Tests the ability of a PotentialPairGauss to handle periodic boundary conditions
void gauss_force_periodic_test(gaussforce_creator gauss_creator, std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    ////////////////////////////////////////////////////////////////////
    // now, lets do a more thorough test and include boundary conditions
    // there are way too many permutations to test here, so I will simply
    // test +x, -x, +y, -y, +z, and -z independently
    // build a 6 particle system with particles across each boundary
    // also test the ability of the force compute to use different particle types
    std::shared_ptr<SystemDefinition> sysdef_6(new SystemDefinition(6, BoxDim(20.0, 40.0, 60.0), 3, 0, 0, 0, 0, exec_conf));
    std::shared_ptr<ParticleData> pdata_6 = sysdef_6->getParticleData();
    pdata_6->setFlags(~PDataFlags(0));

    pdata_6->setPosition(0, make_scalar3(-9.6,0.0,0.0));
    pdata_6->setPosition(1, make_scalar3(9.6, 0.0,0.0));
    pdata_6->setPosition(2, make_scalar3(0.0,-19.6,0.0));
    pdata_6->setPosition(3, make_scalar3(0.0,19.6,0.0));
    pdata_6->setPosition(4, make_scalar3(0.0,0.0,-29.6));
    pdata_6->setPosition(5, make_scalar3(0.0,0.0,29.6));

    pdata_6->setType(0,0);
    pdata_6->setType(1,1);
    pdata_6->setType(2,2);
    pdata_6->setType(3,0);
    pdata_6->setType(4,2);
    pdata_6->setType(5,1);

    std::shared_ptr<NeighborListTree> nlist_6(new NeighborListTree(sysdef_6, Scalar(1.3), Scalar(3.0)));
    std::shared_ptr<PotentialPairGauss> fc_6 = gauss_creator(sysdef_6, nlist_6);
    fc_6->setRcut(0, 0, Scalar(1.3));
    fc_6->setRcut(0, 1, Scalar(1.3));
    fc_6->setRcut(0, 2, Scalar(1.3));
    fc_6->setRcut(1, 1, Scalar(1.3));
    fc_6->setRcut(1, 2, Scalar(1.3));
    fc_6->setRcut(2, 2, Scalar(1.3));

    // choose a small sigma so that all interactions are attractive
    Scalar epsilon = Scalar(1.0);
    Scalar sigma = Scalar(0.5);

    // make life easy: just change epsilon for the different pairs
    fc_6->setParams(0,0,make_scalar2(epsilon,sigma));
    fc_6->setParams(0,1,make_scalar2(Scalar(2.0)*epsilon,sigma));
    fc_6->setParams(0,2,make_scalar2(Scalar(3.0)*epsilon,sigma));
    fc_6->setParams(1,1,make_scalar2(Scalar(4.0)*epsilon,sigma));
    fc_6->setParams(1,2,make_scalar2(Scalar(5.0)*epsilon,sigma));
    fc_6->setParams(2,2,make_scalar2(Scalar(6.0)*epsilon,sigma));

    fc_6->compute(0);

    {
    GlobalArray<Scalar4>& force_array_3 =  fc_6->getForceArray();
    GlobalArray<Scalar>& virial_array_3 =  fc_6->getVirialArray();
    unsigned int pitch = virial_array_3.getPitch();
    ArrayHandle<Scalar4> h_force_3(force_array_3,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_3(virial_array_3,access_location::host,access_mode::read);
    // particle 0 should be pushed right
    MY_CHECK_CLOSE(h_force_3.data[0].x, 2.224298403625553*0.8, tol);
    MY_CHECK_SMALL(h_force_3.data[0].y, tol_small);
    MY_CHECK_SMALL(h_force_3.data[0].z, tol_small);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_3.data[0*pitch+0]
                                       +h_virial_3.data[3*pitch+0]
                                       +h_virial_3.data[5*pitch+0]), 0.296573120483407*0.8, tol);

    // particle 1 should be pushed left
    MY_CHECK_CLOSE(h_force_3.data[1].x, -2.224298403625553*0.8, tol);
    MY_CHECK_SMALL(h_force_3.data[1].y, tol_small);
    MY_CHECK_SMALL(h_force_3.data[1].z, tol_small);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_3.data[0*pitch+1]
                                       +h_virial_3.data[3*pitch+1]
                                       +h_virial_3.data[5*pitch+1]), 0.296573120483407*0.8, tol);

    // particle 2 should be pushed up
    MY_CHECK_CLOSE(h_force_3.data[2].y, 3.336447605438329*0.8, tol);
    MY_CHECK_SMALL(h_force_3.data[2].x, tol_small);
    MY_CHECK_SMALL(h_force_3.data[2].z, tol_small);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_3.data[0*pitch+2]
                                       +h_virial_3.data[3*pitch+2]
                                       +h_virial_3.data[5*pitch+2]), 0.444859680725111*0.8, tol);

    // particle 3 should be pushed down
    MY_CHECK_CLOSE(h_force_3.data[3].y, -3.336447605438329*0.8, tol);
    MY_CHECK_SMALL(h_force_3.data[3].x, tol_small);
    MY_CHECK_SMALL(h_force_3.data[3].z, tol_small);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_3.data[0*pitch+3]
                                       +h_virial_3.data[3*pitch+3]
                                       +h_virial_3.data[5*pitch+3]), 0.444859680725111*0.8, tol);

    // particle 4 should be pushed forward
    MY_CHECK_CLOSE(h_force_3.data[4].z, 5.560746009063882*0.8, tol);
    MY_CHECK_SMALL(h_force_3.data[4].x, tol_small);
    MY_CHECK_SMALL(h_force_3.data[4].y, tol_small);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_3.data[0*pitch+4]
                                       +h_virial_3.data[3*pitch+4]
                                       +h_virial_3.data[5*pitch+4]),  0.741432801208518*0.8, tol);

    // particle 3 should be pushed back
    MY_CHECK_CLOSE(h_force_3.data[5].z, -5.560746009063882*0.8, tol);
    MY_CHECK_SMALL(h_force_3.data[5].x, tol_small);
    MY_CHECK_SMALL(h_force_3.data[5].y, tol_small);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_3.data[0*pitch+5]
                                       +h_virial_3.data[3*pitch+5]
                                       +h_virial_3.data[5*pitch+5]),  0.741432801208518*0.8, tol);
    }
    }

//! Unit test a comparison between 2 LJForceComputes on a "real" system
void gauss_force_comparison_test(gaussforce_creator gauss_creator1,
                                 gaussforce_creator gauss_creator2,
                                 std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    const unsigned int N = 5000;

    // create a random particle system to sum forces on
    RandomInitializer rand_init(N, Scalar(0.2), Scalar(0.9), "A");
    std::shared_ptr< SnapshotSystemData<Scalar> > snap;
    snap = rand_init.getSnapshot();
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));
    std::shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    pdata->setFlags(~PDataFlags(0));
    std::shared_ptr<NeighborListTree> nlist(new NeighborListTree(sysdef, Scalar(3.0), Scalar(0.8)));

    std::shared_ptr<PotentialPairGauss> fc1 = gauss_creator1(sysdef, nlist);
    std::shared_ptr<PotentialPairGauss> fc2 = gauss_creator2(sysdef, nlist);
    fc1->setRcut(0, 0, Scalar(3.0));
    fc2->setRcut(0, 0, Scalar(3.0));

    // setup some values for epsilon and sigma
    Scalar epsilon = Scalar(1.0);
    Scalar sigma = Scalar(1.2);

    // specify the force parameters
    fc1->setParams(0,0,make_scalar2(epsilon,sigma));
    fc2->setParams(0,0,make_scalar2(epsilon,sigma));

    // compute the forces
    fc1->compute(0);
    fc2->compute(0);

    {
    // verify that the forces are identical (within roundoff errors)
    GlobalArray<Scalar4>& force_array_4 =  fc1->getForceArray();
    GlobalArray<Scalar>& virial_array_4 =  fc1->getVirialArray();
    unsigned int pitch = virial_array_4.getPitch();
    ArrayHandle<Scalar4> h_force_4(force_array_4,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_4(virial_array_4,access_location::host,access_mode::read);
    GlobalArray<Scalar4>& force_array_5 =  fc2->getForceArray();
    GlobalArray<Scalar>& virial_array_5 =  fc2->getVirialArray();
    ArrayHandle<Scalar4> h_force_5(force_array_5,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_5(virial_array_5,access_location::host,access_mode::read);

    // compare average deviation between the two computes
    double deltaf2 = 0.0;
    double deltape2 = 0.0;
    double deltav2[6];
    for (unsigned int i = 0; i < 6;  i++)
        deltav2[i] = 0.0;

    for (unsigned int i = 0; i < N; i++)
        {
        deltaf2 += double(h_force_5.data[i].x - h_force_4.data[i].x) * double(h_force_5.data[i].x - h_force_4.data[i].x);
        deltaf2 += double(h_force_5.data[i].y - h_force_4.data[i].y) * double(h_force_5.data[i].y - h_force_4.data[i].y);
        deltaf2 += double(h_force_5.data[i].z - h_force_4.data[i].z) * double(h_force_5.data[i].z - h_force_4.data[i].z);
        deltape2 += double(h_force_5.data[i].w - h_force_4.data[i].w) * double(h_force_5.data[i].w - h_force_4.data[i].w);
        for (unsigned int j = 0; j < 6; j++)
            deltav2[j] += double(h_virial_5.data[j*pitch+i] - h_virial_4.data[j*pitch+i]) * double(h_virial_5.data[j*pitch+i] - h_virial_4.data[j*pitch+i]);

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

//! Test the ability of the gauss force compute to compute forces with different shift modes
void gauss_force_shift_test(gaussforce_creator gauss_creator, std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // this 2-particle test is just to get a plot of the potential and force vs r cut
    std::shared_ptr<SystemDefinition> sysdef_2(new SystemDefinition(2, BoxDim(1000.0), 1, 0, 0, 0, 0, exec_conf));
    std::shared_ptr<ParticleData> pdata_2 = sysdef_2->getParticleData();
    pdata_2->setFlags(~PDataFlags(0));

    pdata_2->setPosition(0,make_scalar3(0.0,0.0,0.0));
    pdata_2->setPosition(1,make_scalar3(2.8,0.0,0.0));
    std::shared_ptr<NeighborListTree> nlist_2(new NeighborListTree(sysdef_2, Scalar(3.0), Scalar(0.8)));
    std::shared_ptr<PotentialPairGauss> fc_no_shift = gauss_creator(sysdef_2, nlist_2);
    fc_no_shift->setShiftMode(PotentialPairGauss::no_shift);
    fc_no_shift->setRcut(0, 0, Scalar(3.0));
    std::shared_ptr<PotentialPairGauss> fc_shift = gauss_creator(sysdef_2, nlist_2);
    fc_shift->setShiftMode(PotentialPairGauss::shift);
    fc_shift->setRcut(0, 0, Scalar(3.0));

    nlist_2->setStorageMode(NeighborList::full);

    // setup a standard epsilon and sigma
    Scalar epsilon = Scalar(1.0);
    Scalar sigma = Scalar(1.0);
    fc_no_shift->setParams(0,0,make_scalar2(epsilon,sigma));
    fc_shift->setParams(0,0,make_scalar2(epsilon,sigma));

    fc_no_shift->compute(0);
    fc_shift->compute(0);

    {
    GlobalArray<Scalar4>& force_array_6 =  fc_no_shift->getForceArray();
    GlobalArray<Scalar>& virial_array_6 =  fc_no_shift->getVirialArray();
    ArrayHandle<Scalar4> h_force_6(force_array_6,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_6(virial_array_6,access_location::host,access_mode::read);

    MY_CHECK_CLOSE(h_force_6.data[0].x, -0.055555065284237, tol);
    MY_CHECK_CLOSE(h_force_6.data[0].w, 0.019841094744370/2.0, tol);
    MY_CHECK_CLOSE(h_force_6.data[1].x, 0.055555065284237, tol);
    MY_CHECK_CLOSE(h_force_6.data[1].w, 0.019841094744370/2.0, tol);

    GlobalArray<Scalar4>& force_array_7 =  fc_shift->getForceArray();
    GlobalArray<Scalar>& virial_array_7 =  fc_shift->getVirialArray();
    ArrayHandle<Scalar4> h_force_7(force_array_7,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_7(virial_array_7,access_location::host,access_mode::read);

    // shifted just has pe shifted by a given amount
    MY_CHECK_CLOSE(h_force_7.data[0].x, -0.055555065284237, tol);
    MY_CHECK_CLOSE(h_force_7.data[0].w, 0.008732098206128/2.0, tol);
    MY_CHECK_CLOSE(h_force_7.data[1].x, 0.055555065284237, tol);
    MY_CHECK_CLOSE(h_force_7.data[1].w, 0.008732098206128/2.0, tol);
    }

    // check once again to verify that nothing fish happens past r_cut
    pdata_2->setPosition(0,make_scalar3(0.0,0.0,0.0));
    pdata_2->setPosition(1,make_scalar3(3.1,0.0,0.0));

    fc_no_shift->compute(2);
    fc_shift->compute(2);

    {
    GlobalArray<Scalar4>& force_array_8 =  fc_no_shift->getForceArray();
    GlobalArray<Scalar>& virial_array_8 =  fc_no_shift->getVirialArray();
    ArrayHandle<Scalar4> h_force_8(force_array_8,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_8(virial_array_8,access_location::host,access_mode::read);
    GlobalArray<Scalar4>& force_array_9 =  fc_shift->getForceArray();
    GlobalArray<Scalar>& virial_array_9 =  fc_shift->getVirialArray();
    ArrayHandle<Scalar4> h_force_9(force_array_9,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_9(virial_array_9,access_location::host,access_mode::read);

    MY_CHECK_SMALL(h_force_9.data[0].x, tol_small);
    MY_CHECK_SMALL(h_force_9.data[0].w, tol_small);
    MY_CHECK_SMALL(h_force_9.data[1].x, tol_small);
    MY_CHECK_SMALL(h_force_9.data[1].w, tol_small);

    // shifted just has pe shifted by a given amount
    MY_CHECK_SMALL(h_force_9.data[0].x, tol_small);
    MY_CHECK_SMALL(h_force_9.data[0].w, tol_small);
    MY_CHECK_SMALL(h_force_9.data[1].x, tol_small);
    MY_CHECK_SMALL(h_force_9.data[1].w, tol_small);
    }
    }

//! LJForceCompute creator for unit tests
std::shared_ptr<PotentialPairGauss> base_class_gauss_creator(std::shared_ptr<SystemDefinition> sysdef,
                                                        std::shared_ptr<NeighborList> nlist)
    {
    return std::shared_ptr<PotentialPairGauss>(new PotentialPairGauss(sysdef, nlist));
    }

#ifdef ENABLE_CUDA
//! PotentialPairGaussGPU creator for unit tests
std::shared_ptr<PotentialPairGaussGPU> gpu_gauss_creator(std::shared_ptr<SystemDefinition> sysdef,
                                                    std::shared_ptr<NeighborList> nlist)
    {
    nlist->setStorageMode(NeighborList::full);
    std::shared_ptr<PotentialPairGaussGPU> gauss(new PotentialPairGaussGPU(sysdef, nlist));
    return gauss;
    }
#endif

//! test case for particle test on CPU
UP_TEST( GaussForce_particle )
    {
    gaussforce_creator gauss_creator_base = bind(base_class_gauss_creator, _1, _2);
    gauss_force_particle_test(gauss_creator_base, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

//! test case for periodic test on CPU
UP_TEST( GaussForce_periodic )
    {
    gaussforce_creator gauss_creator_base = bind(base_class_gauss_creator, _1, _2);
    gauss_force_periodic_test(gauss_creator_base, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

//! test case for particle test on CPU
UP_TEST( GaussForce_shift )
    {
    gaussforce_creator gauss_creator_base = bind(base_class_gauss_creator, _1, _2);
    gauss_force_shift_test(gauss_creator_base, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

# ifdef ENABLE_CUDA
//! test case for particle test on GPU
UP_TEST( GaussForceGPU_particle )
    {
    gaussforce_creator gauss_creator_gpu = bind(gpu_gauss_creator, _1, _2);
    gauss_force_particle_test(gauss_creator_gpu, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! test case for periodic test on the GPU
UP_TEST( GaussForceGPU_periodic )
    {
    gaussforce_creator gauss_creator_gpu = bind(gpu_gauss_creator, _1, _2);
    gauss_force_periodic_test(gauss_creator_gpu, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! test case for shift test on GPU
UP_TEST( GaussForceGPU_shift )
    {
    gaussforce_creator gauss_creator_gpu = bind(gpu_gauss_creator, _1, _2);
    gauss_force_shift_test(gauss_creator_gpu, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! test case for comparing GPU output to base class output
UP_TEST( GaussForceGPU_compare )
    {
    gaussforce_creator gauss_creator_gpu = bind(gpu_gauss_creator, _1, _2);
    gaussforce_creator gauss_creator_base = bind(base_class_gauss_creator, _1, _2);
    gauss_force_comparison_test(gauss_creator_base, gauss_creator_gpu, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

#endif
