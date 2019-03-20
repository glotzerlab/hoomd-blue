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

#include <math.h>

using namespace std;
using namespace std::placeholders;

/*! \file lj_force_test.cc
    \brief Implements unit tests for PotentialPairLJ and PotentialPairLJGPU and descendants
    \ingroup unit_tests
*/

#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN();




//! Typedef'd LJForceCompute factory
typedef std::function<std::shared_ptr<PotentialPairLJ> (std::shared_ptr<SystemDefinition> sysdef,
                                                     std::shared_ptr<NeighborList> nlist)> ljforce_creator;

//! Test the ability of the lj force compute to actually calculate forces
void lj_force_particle_test(ljforce_creator lj_creator, std::shared_ptr<ExecutionConfiguration> exec_conf)
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

    {
    ArrayHandle<Scalar4> h_pos(pdata_3->getPositions(), access_location::host, access_mode::readwrite);
    h_pos.data[0].x = h_pos.data[0].y = h_pos.data[0].z = 0.0;
    h_pos.data[1].x = Scalar(pow(2.0,1.0/6.0)); h_pos.data[1].y = h_pos.data[1].z = 0.0;
    h_pos.data[2].x = Scalar(2.0*pow(2.0,1.0/6.0)); h_pos.data[2].y = h_pos.data[2].z = 0.0;
    }
    std::shared_ptr<NeighborListTree> nlist_3(new NeighborListTree(sysdef_3, Scalar(1.3), Scalar(3.0)));
    std::shared_ptr<PotentialPairLJ> fc_3 = lj_creator(sysdef_3, nlist_3);
    fc_3->setRcut(0, 0, Scalar(1.3));

    // first test: setup a sigma of 1.0 so that all forces will be 0
    Scalar epsilon = Scalar(1.15);
    Scalar sigma = Scalar(1.0);
    Scalar alpha = Scalar(1.0);
    Scalar lj1 = Scalar(4.0) * epsilon * pow(sigma,Scalar(12.0));
    Scalar lj2 = alpha * Scalar(4.0) * epsilon * pow(sigma,Scalar(6.0));
    fc_3->setParams(0,0,make_scalar2(lj1,lj2));

    // compute the forces
    fc_3->compute(0);

    {
    GlobalArray<Scalar4>& force_array_1 =  fc_3->getForceArray();
    GlobalArray<Scalar>& virial_array_1 =  fc_3->getVirialArray();
    unsigned int pitch = virial_array_1.getPitch();
    ArrayHandle<Scalar4> h_force_1(force_array_1,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_1(virial_array_1,access_location::host,access_mode::read);
    MY_CHECK_SMALL(h_force_1.data[0].x, tol_small);
    MY_CHECK_SMALL(h_force_1.data[0].y, tol_small);
    MY_CHECK_SMALL(h_force_1.data[0].z, tol_small);
    MY_CHECK_CLOSE(h_force_1.data[0].w, -0.575, tol);
    MY_CHECK_SMALL(h_virial_1.data[0*pitch+0]
                        +h_virial_1.data[3*pitch+0]
                        +h_virial_1.data[5*pitch+0], tol_small);

    MY_CHECK_SMALL(h_force_1.data[1].x, tol_small);
    MY_CHECK_SMALL(h_force_1.data[1].y, tol_small);
    MY_CHECK_SMALL(h_force_1.data[1].z, tol_small);
    MY_CHECK_CLOSE(h_force_1.data[1].w, -1.15, tol);
    MY_CHECK_SMALL(h_virial_1.data[0*pitch+1]
                        +h_virial_1.data[3*pitch+1]
                        +h_virial_1.data[5*pitch+1], tol_small);

    MY_CHECK_SMALL(h_force_1.data[2].x, tol_small);
    MY_CHECK_SMALL(h_force_1.data[2].y, tol_small);
    MY_CHECK_SMALL(h_force_1.data[2].z, tol_small);
    MY_CHECK_CLOSE(h_force_1.data[2].w, -0.575, tol);
    MY_CHECK_SMALL(h_virial_1.data[0*pitch+2]
                        +h_virial_1.data[3*pitch+2]
                        +h_virial_1.data[5*pitch+2], tol_small);
    }

    // now change sigma and alpha so we can check that it is computing the right force
    sigma = Scalar(1.2); // < bigger sigma should push particle 0 left and particle 2 right
    alpha = Scalar(0.45);
    lj1 = Scalar(4.0) * epsilon * pow(sigma,Scalar(12.0));
    lj2 = alpha * Scalar(4.0) * epsilon * pow(sigma,Scalar(6.0));
    fc_3->setParams(0,0,make_scalar2(lj1,lj2));
    fc_3->compute(1);

    {
    GlobalArray<Scalar4>& force_array_2 =  fc_3->getForceArray();
    GlobalArray<Scalar>& virial_array_2 =  fc_3->getVirialArray();
    unsigned int pitch = virial_array_2.getPitch();
    ArrayHandle<Scalar4> h_force_2(force_array_2,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_2(virial_array_2,access_location::host,access_mode::read);
    MY_CHECK_CLOSE(h_force_2.data[0].x, -93.09822608552962, tol);
    MY_CHECK_SMALL(h_force_2.data[0].y, tol_small);
    MY_CHECK_SMALL(h_force_2.data[0].z, tol_small);
    MY_CHECK_CLOSE(h_force_2.data[0].w, 3.5815110377468, tol);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_2.data[0*pitch+0]
                                       +h_virial_2.data[3*pitch+0]
                                       +h_virial_2.data[5*pitch+0]), 17.416537590989, tol);

    // center particle should still be a 0 force by symmetry
    MY_CHECK_SMALL(h_force_2.data[1].x, tol_small);
    MY_CHECK_SMALL(h_force_2.data[1].y, tol_small);
    MY_CHECK_SMALL(h_force_2.data[1].z, tol_small);
    // there is still an energy and virial, though
    MY_CHECK_CLOSE(h_force_2.data[1].w, 7.1630220754935, tol);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_2.data[0*pitch+1]
                                       +h_virial_2.data[3*pitch+1]
                                       +h_virial_2.data[5*pitch+1]), 34.833075181975, tol);

    MY_CHECK_CLOSE(h_force_2.data[2].x, 93.09822608552962, tol);
    MY_CHECK_SMALL(h_force_2.data[2].y, tol_small);
    MY_CHECK_SMALL(h_force_2.data[2].z, tol_small);
    MY_CHECK_CLOSE(h_force_2.data[2].w, 3.581511037746, tol);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_2.data[0*pitch+2]
                                       +h_virial_2.data[3*pitch+2]
                                       +h_virial_2.data[5*pitch+2]), 17.416537590989, tol);
    }

    // swap the order of particles 0 ans 2 in memory to check that the force compute handles this properly
    {
    ArrayHandle<Scalar4> h_pos(pdata_3->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_tag(pdata_3->getTags(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_rtag(pdata_3->getRTags(), access_location::host, access_mode::readwrite);

    h_pos.data[2].x = h_pos.data[2].y = h_pos.data[2].z = 0.0;
    h_pos.data[0].x = Scalar(2.0*pow(2.0,1.0/6.0)); h_pos.data[0].y = h_pos.data[0].z = 0.0;

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
    GlobalArray<Scalar4>& force_array_3 =  fc_3->getForceArray();
    GlobalArray<Scalar>& virial_array_3 =  fc_3->getVirialArray();
    ArrayHandle<Scalar4> h_force_3(force_array_3,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_3(virial_array_3,access_location::host,access_mode::read);
    MY_CHECK_CLOSE(h_force_3.data[0].x, 93.09822608552962, tol);
    MY_CHECK_CLOSE(h_force_3.data[2].x, -93.09822608552962, tol);
    }
    }

//! Tests the ability of a LJForceCompute to handle periodic boundary conditions
void lj_force_periodic_test(ljforce_creator lj_creator, std::shared_ptr<ExecutionConfiguration> exec_conf)
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
    std::shared_ptr<PotentialPairLJ> fc_6 = lj_creator(sysdef_6, nlist_6);
    fc_6->setRcut(0, 0, Scalar(1.3));
    fc_6->setRcut(0, 1, Scalar(1.3));
    fc_6->setRcut(0, 2, Scalar(1.3));
    fc_6->setRcut(1, 1, Scalar(1.3));
    fc_6->setRcut(1, 2, Scalar(1.3));
    fc_6->setRcut(2, 2, Scalar(1.3));

    // choose a small sigma so that all interactions are attractive
    Scalar epsilon = Scalar(1.0);
    Scalar sigma = Scalar(0.5);
    Scalar alpha = Scalar(0.45);
    Scalar lj1 = Scalar(4.0) * epsilon * pow(sigma,Scalar(12.0));
    Scalar lj2 = alpha * Scalar(4.0) * epsilon * pow(sigma,Scalar(6.0));

    // make life easy: just change epsilon for the different pairs
    fc_6->setParams(0,0,make_scalar2(lj1,lj2));
    fc_6->setParams(0,1,make_scalar2(Scalar(2.0)*lj1,Scalar(2.0)*lj2));
    fc_6->setParams(0,2,make_scalar2(Scalar(3.0)*lj1,Scalar(3.0)*lj2));
    fc_6->setParams(1,1,make_scalar2(Scalar(4.0)*lj1,Scalar(4.0)*lj2));
    fc_6->setParams(1,2,make_scalar2(Scalar(5.0)*lj1,Scalar(5.0)*lj2));
    fc_6->setParams(2,2,make_scalar2(Scalar(6.0)*lj1,Scalar(6.0)*lj2));

    fc_6->compute(0);

    {
    GlobalArray<Scalar4>& force_array_4 =  fc_6->getForceArray();
    GlobalArray<Scalar>& virial_array_4 =  fc_6->getVirialArray();
    unsigned int pitch = virial_array_4.getPitch();
    ArrayHandle<Scalar4> h_force_4(force_array_4,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_4(virial_array_4,access_location::host,access_mode::read);
    // particle 0 should be pulled left
    MY_CHECK_CLOSE(h_force_4.data[0].x, -1.18299976747949, tol);
    MY_CHECK_SMALL(h_force_4.data[0].y, tol_small);
    MY_CHECK_SMALL(h_force_4.data[0].z, tol_small);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_4.data[0*pitch+0]
                                       +h_virial_4.data[3*pitch+0]
                                       +h_virial_4.data[5*pitch+0]),-0.15773330233059, tol);

    // particle 1 should be pulled right
    MY_CHECK_CLOSE(h_force_4.data[1].x, 1.18299976747949, tol);
    MY_CHECK_SMALL(h_force_4.data[1].y, tol_small);
    MY_CHECK_SMALL(h_force_4.data[1].z, tol_small);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_4.data[0*pitch+1]
                                       +h_virial_4.data[3*pitch+1]
                                       +h_virial_4.data[5*pitch+1]),-0.15773330233059, tol);

    // particle 2 should be pulled down
    MY_CHECK_CLOSE(h_force_4.data[2].y, -1.77449965121923, tol);
    MY_CHECK_SMALL(h_force_4.data[2].x, tol_small);
    MY_CHECK_SMALL(h_force_4.data[2].z, tol_small);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_4.data[0*pitch+2]
                                       +h_virial_4.data[3*pitch+2]
                                       +h_virial_4.data[5*pitch+2]), -0.23659995349591, tol);

    // particle 3 should be pulled up
    MY_CHECK_CLOSE(h_force_4.data[3].y, 1.77449965121923, tol);
    MY_CHECK_SMALL(h_force_4.data[3].x, tol_small);
    MY_CHECK_SMALL(h_force_4.data[3].z, tol_small);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_4.data[0*pitch+3]
                                       +h_virial_4.data[3*pitch+3]
                                       +h_virial_4.data[5*pitch+3]), -0.23659995349591, tol);

    // particle 4 should be pulled back
    MY_CHECK_CLOSE(h_force_4.data[4].z, -2.95749941869871, tol);
    MY_CHECK_SMALL(h_force_4.data[4].x, tol_small);
    MY_CHECK_SMALL(h_force_4.data[4].y, tol_small);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_4.data[0*pitch+4]
                                       +h_virial_4.data[3*pitch+4]
                                       +h_virial_4.data[5*pitch+4]), -0.39433325582651, tol);

    // particle 3 should be pulled forward
    MY_CHECK_CLOSE(h_force_4.data[5].z, 2.95749941869871, tol);
    MY_CHECK_SMALL(h_force_4.data[5].x, tol_small);
    MY_CHECK_SMALL(h_force_4.data[5].y, tol_small);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_4.data[0*pitch+5]
                                       +h_virial_4.data[3*pitch+5]
                                       +h_virial_4.data[5*pitch+5]), -0.39433325582651, tol);
    }
    }

//! Unit test a comparison between 2 LJForceComputes on a "real" system
void lj_force_comparison_test(ljforce_creator lj_creator1, ljforce_creator lj_creator2, std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    const unsigned int N = 5000;

    // create a random particle system to sum forces on
    RandomInitializer rand_init(N, Scalar(0.2), Scalar(0.9), "A");
    std::shared_ptr< SnapshotSystemData<Scalar> > snap = rand_init.getSnapshot();
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));
    std::shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    pdata->setFlags(~PDataFlags(0));

    std::shared_ptr<NeighborListTree> nlist(new NeighborListTree(sysdef, Scalar(3.0), Scalar(0.8)));

    std::shared_ptr<PotentialPairLJ> fc1 = lj_creator1(sysdef, nlist);
    std::shared_ptr<PotentialPairLJ> fc2 = lj_creator2(sysdef, nlist);
    fc1->setRcut(0, 0, Scalar(3.0));
    fc2->setRcut(0, 0, Scalar(3.0));

    // setup some values for alpha and sigma
    Scalar epsilon = Scalar(1.0);
    Scalar sigma = Scalar(1.2);
    Scalar alpha = Scalar(0.45);
    Scalar lj1 = Scalar(4.0) * epsilon * pow(sigma,Scalar(12.0));
    Scalar lj2 = alpha * Scalar(4.0) * epsilon * pow(sigma,Scalar(6.0));

    // specify the force parameters
    fc1->setParams(0,0,make_scalar2(lj1,lj2));
    fc2->setParams(0,0,make_scalar2(lj1,lj2));

    // compute the forces
    fc1->compute(0);
    fc2->compute(0);

    {
    // verify that the forces are identical (within roundoff errors)
    GlobalArray<Scalar4>& force_array_5 =  fc1->getForceArray();
    GlobalArray<Scalar>& virial_array_5 =  fc1->getVirialArray();
    unsigned int pitch = virial_array_5.getPitch();
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

//! Test the ability of the lj force compute to compute forces with different shift modes
void lj_force_shift_test(ljforce_creator lj_creator, std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // this 2-particle test is just to get a plot of the potential and force vs r cut
    std::shared_ptr<SystemDefinition> sysdef_2(new SystemDefinition(2, BoxDim(1000.0), 1, 0, 0, 0, 0, exec_conf));
    std::shared_ptr<ParticleData> pdata_2 = sysdef_2->getParticleData();
    pdata_2->setFlags(~PDataFlags(0));

    {
    ArrayHandle<Scalar4> h_pos(pdata_2->getPositions(), access_location::host, access_mode::readwrite);

    h_pos.data[0].x = h_pos.data[0].y = h_pos.data[0].z = 0.0;
    h_pos.data[1].x = Scalar(2.8); h_pos.data[1].y = h_pos.data[1].z = 0.0;
    }

    std::shared_ptr<NeighborListTree> nlist_2(new NeighborListTree(sysdef_2, Scalar(3.0), Scalar(0.8)));
    std::shared_ptr<PotentialPairLJ> fc_no_shift = lj_creator(sysdef_2, nlist_2);
    fc_no_shift->setRcut(0, 0, Scalar(3.0));
    fc_no_shift->setShiftMode(PotentialPairLJ::no_shift);

    std::shared_ptr<PotentialPairLJ> fc_shift = lj_creator(sysdef_2, nlist_2);
    fc_shift->setRcut(0, 0, Scalar(3.0));
    fc_shift->setShiftMode(PotentialPairLJ::shift);

    std::shared_ptr<PotentialPairLJ> fc_xplor = lj_creator(sysdef_2, nlist_2);
    fc_xplor->setRcut(0, 0, Scalar(3.0));
    fc_xplor->setShiftMode(PotentialPairLJ::xplor);
    fc_xplor->setRon(0, 0, Scalar(2.0));

    nlist_2->setStorageMode(NeighborList::full);

    // setup a standard epsilon and sigma
    Scalar epsilon = Scalar(1.0);
    Scalar sigma = Scalar(1.0);
    Scalar alpha = Scalar(1.0);
    Scalar lj1 = Scalar(4.0) * epsilon * pow(sigma,Scalar(12.0));
    Scalar lj2 = alpha * Scalar(4.0) * epsilon * pow(sigma,Scalar(6.0));
    fc_no_shift->setParams(0,0,make_scalar2(lj1,lj2));
    fc_shift->setParams(0,0,make_scalar2(lj1,lj2));
    fc_xplor->setParams(0,0,make_scalar2(lj1,lj2));

    fc_no_shift->compute(0);
    fc_shift->compute(0);
    fc_xplor->compute(0);

    {
    GlobalArray<Scalar4>& force_array_7 =  fc_no_shift->getForceArray();
    GlobalArray<Scalar>& virial_array_7 =  fc_no_shift->getVirialArray();
    ArrayHandle<Scalar4> h_force_7(force_array_7,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_7(virial_array_7,access_location::host,access_mode::read);

    MY_CHECK_CLOSE(h_force_7.data[0].x, 0.017713272731914, tol);
    MY_CHECK_CLOSE(h_force_7.data[0].w, -0.0041417095577326, tol);
    MY_CHECK_CLOSE(h_force_7.data[1].x, -0.017713272731914, tol);
    MY_CHECK_CLOSE(h_force_7.data[1].w, -0.0041417095577326, tol);

    // shifted just has pe shifted by a given amount
    GlobalArray<Scalar4>& force_array_8 =  fc_shift->getForceArray();
    GlobalArray<Scalar>& virial_array_8 =  fc_shift->getVirialArray();
    ArrayHandle<Scalar4> h_force_8(force_array_8,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_8(virial_array_8,access_location::host,access_mode::read);
    MY_CHECK_CLOSE(h_force_8.data[0].x, 0.017713272731914, tol);
    MY_CHECK_CLOSE(h_force_8.data[0].w, -0.0014019886856134, tol);
    MY_CHECK_CLOSE(h_force_8.data[1].x, -0.017713272731914, tol);
    MY_CHECK_CLOSE(h_force_8.data[1].w, -0.0014019886856134, tol);

    // xplor has slight tweaks
    GlobalArray<Scalar4>& force_array_9 =  fc_xplor->getForceArray();
    GlobalArray<Scalar>& virial_array_9 =  fc_xplor->getVirialArray();
    ArrayHandle<Scalar4> h_force_9(force_array_9,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_9(virial_array_9,access_location::host,access_mode::read);
    MY_CHECK_CLOSE(h_force_9.data[0].x, 0.012335911924312, tol);
    MY_CHECK_CLOSE(h_force_9.data[0].w, -0.001130667359194/2.0, tol);
    MY_CHECK_CLOSE(h_force_9.data[1].x, -0.012335911924312, tol);
    MY_CHECK_CLOSE(h_force_9.data[1].w, -0.001130667359194/2.0, tol);
    }

    // check again, prior to r_on to make sure xplor isn't doing something weird
    {
    ArrayHandle<Scalar4> h_pos(pdata_2->getPositions(), access_location::host, access_mode::readwrite);

    h_pos.data[0].x = h_pos.data[0].y = h_pos.data[0].z = 0.0;
    h_pos.data[1].x = Scalar(1.5); h_pos.data[1].y = h_pos.data[1].z = 0.0;
    }

    fc_no_shift->compute(1);
    fc_shift->compute(1);
    fc_xplor->compute(1);

    {
    GlobalArray<Scalar4>& force_array_10 =  fc_no_shift->getForceArray();
    GlobalArray<Scalar>& virial_array_10 =  fc_no_shift->getVirialArray();
    ArrayHandle<Scalar4> h_force_10(force_array_10,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_10(virial_array_10,access_location::host,access_mode::read);

    MY_CHECK_CLOSE(h_force_10.data[0].x, 1.1580288310461, tol);
    MY_CHECK_CLOSE(h_force_10.data[0].w, -0.16016829713928, tol);
    MY_CHECK_CLOSE(h_force_10.data[1].x, -1.1580288310461, tol);
    MY_CHECK_CLOSE(h_force_10.data[1].w, -0.16016829713928, tol);

    // shifted just has pe shifted by a given amount
    GlobalArray<Scalar4>& force_array_11 =  fc_shift->getForceArray();
    GlobalArray<Scalar>& virial_array_11 =  fc_shift->getVirialArray();
    ArrayHandle<Scalar4> h_force_11(force_array_11,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_11(virial_array_11,access_location::host,access_mode::read);
    MY_CHECK_CLOSE(h_force_11.data[0].x, 1.1580288310461, tol);
    MY_CHECK_CLOSE(h_force_11.data[0].w, -0.15742857626716, tol);
    MY_CHECK_CLOSE(h_force_11.data[1].x, -1.1580288310461, tol);
    MY_CHECK_CLOSE(h_force_11.data[1].w, -0.15742857626716, tol);

    // xplor has slight tweaks
    GlobalArray<Scalar4>& force_array_12 =  fc_xplor->getForceArray();
    GlobalArray<Scalar>& virial_array_12 =  fc_xplor->getVirialArray();
    ArrayHandle<Scalar4> h_force_12(force_array_12,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_12(virial_array_12,access_location::host,access_mode::read);
    MY_CHECK_CLOSE(h_force_12.data[0].x, 1.1580288310461, tol);
    MY_CHECK_CLOSE(h_force_12.data[0].w, -0.16016829713928, tol);
    MY_CHECK_CLOSE(h_force_12.data[1].x, -1.1580288310461, tol);
    MY_CHECK_CLOSE(h_force_12.data[1].w, -0.16016829713928, tol);
    }

    // check once again to verify that nothing fish happens past r_cut
    {
    ArrayHandle<Scalar4> h_pos(pdata_2->getPositions(), access_location::host, access_mode::readwrite);

    h_pos.data[0].x = h_pos.data[0].y = h_pos.data[0].z = 0.0;
    h_pos.data[1].x = Scalar(3.1); h_pos.data[1].y = h_pos.data[1].z = 0.0;
    }

    fc_no_shift->compute(2);
    fc_shift->compute(2);
    fc_xplor->compute(2);

    {
    GlobalArray<Scalar4>& force_array_13 =  fc_no_shift->getForceArray();
    GlobalArray<Scalar>& virial_array_13 =  fc_no_shift->getVirialArray();
    ArrayHandle<Scalar4> h_force_13(force_array_13,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_13(virial_array_13,access_location::host,access_mode::read);

    MY_CHECK_SMALL(h_force_13.data[0].x, tol_small);
    MY_CHECK_SMALL(h_force_13.data[0].w, tol_small);
    MY_CHECK_SMALL(h_force_13.data[1].x, tol_small);
    MY_CHECK_SMALL(h_force_13.data[1].w, tol_small);

    // shifted just has pe shifted by a given amount
    GlobalArray<Scalar4>& force_array_14 =  fc_shift->getForceArray();
    GlobalArray<Scalar>& virial_array_14 =  fc_shift->getVirialArray();
    ArrayHandle<Scalar4> h_force_14(force_array_14,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_14(virial_array_14,access_location::host,access_mode::read);
    MY_CHECK_SMALL(h_force_14.data[0].x, tol_small);
    MY_CHECK_SMALL(h_force_14.data[0].w, tol_small);
    MY_CHECK_SMALL(h_force_14.data[1].x, tol_small);
    MY_CHECK_SMALL(h_force_14.data[1].w, tol_small);

    // xplor has slight tweaks
    GlobalArray<Scalar4>& force_array_15 =  fc_xplor->getForceArray();
    GlobalArray<Scalar>& virial_array_15 =  fc_xplor->getVirialArray();
    ArrayHandle<Scalar4> h_force_15(force_array_15,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_15(virial_array_15,access_location::host,access_mode::read);
    MY_CHECK_SMALL(h_force_15.data[0].x, tol_small);
    MY_CHECK_SMALL(h_force_15.data[0].w, tol_small);
    MY_CHECK_SMALL(h_force_15.data[1].x, tol_small);
    MY_CHECK_SMALL(h_force_15.data[1].w, tol_small);
    }
    }

//! LJForceCompute creator for unit tests
std::shared_ptr<PotentialPairLJ> base_class_lj_creator(std::shared_ptr<SystemDefinition> sysdef,
                                                  std::shared_ptr<NeighborList> nlist)
    {
    return std::shared_ptr<PotentialPairLJ>(new PotentialPairLJ(sysdef, nlist));
    }

#ifdef ENABLE_CUDA
//! LJForceComputeGPU creator for unit tests
std::shared_ptr<PotentialPairLJGPU> gpu_lj_creator(std::shared_ptr<SystemDefinition> sysdef,
                                          std::shared_ptr<NeighborList> nlist)
    {
    nlist->setStorageMode(NeighborList::full);
    std::shared_ptr<PotentialPairLJGPU> lj(new PotentialPairLJGPU(sysdef, nlist));
    return lj;
    }
#endif

//! test case for particle test on CPU
UP_TEST( PotentialPairLJ_particle )
    {
    ljforce_creator lj_creator_base = bind(base_class_lj_creator, _1, _2);
    lj_force_particle_test(lj_creator_base, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

//! test case for periodic test on CPU
UP_TEST( PotentialPairLJ_periodic )
    {
    ljforce_creator lj_creator_base = bind(base_class_lj_creator, _1, _2);
    lj_force_periodic_test(lj_creator_base, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

//! test case for particle test on CPU
UP_TEST( PotentialPairLJ_shift )
    {
    ljforce_creator lj_creator_base = bind(base_class_lj_creator, _1, _2);
    lj_force_shift_test(lj_creator_base, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

# ifdef ENABLE_CUDA
//! test case for particle test on GPU
UP_TEST( LJForceGPU_particle )
    {
    ljforce_creator lj_creator_gpu = bind(gpu_lj_creator, _1, _2);
    lj_force_particle_test(lj_creator_gpu, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! test case for periodic test on the GPU
UP_TEST( LJForceGPU_periodic )
    {
    ljforce_creator lj_creator_gpu = bind(gpu_lj_creator, _1, _2);
    lj_force_periodic_test(lj_creator_gpu, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! test case for shift test on GPU
UP_TEST( LJForceGPU_shift )
    {
    ljforce_creator lj_creator_gpu = bind(gpu_lj_creator, _1, _2);
    lj_force_shift_test(lj_creator_gpu, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! test case for comparing GPU output to base class output
/*UP_TEST( LJForceGPU_compare )
    {
    ljforce_creator lj_creator_gpu = bind(gpu_lj_creator, _1, _2);
    ljforce_creator lj_creator_base = bind(base_class_lj_creator, _1, _2);
    lj_force_comparison_test(lj_creator_base, lj_creator_gpu, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }*/

#endif
