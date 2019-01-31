// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// this include is necessary to get MPI included before anything else to support intel MPI
#include "hoomd/ExecutionConfiguration.h"

#include <iostream>

#include <functional>
#include <memory>

#include "hoomd/md/AllPairPotentials.h"

#include "hoomd/md/NeighborListTree.h"
#include "hoomd/Initializers.h"

#include <math.h>

using namespace std;
using namespace std::placeholders;

/*! \file shiftedlj_force_test.cc
    \brief Implements unit tests for PotentialPairSLJ and descendants
    \ingroup unit_tests
*/

#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN();




//! Typedef'd PotentialPairSLJ factory
typedef std::function<std::shared_ptr<PotentialPairSLJ> (std::shared_ptr<SystemDefinition> sysdef,
                                                      std::shared_ptr<NeighborList> nlist)> shiftedljforce_creator;

//! Test the ability of the shiftedlj force compute to actually calculate forces
void shiftedlj_force_particle_test(shiftedljforce_creator shiftedlj_creator, std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // this 3-particle test subtly checks several conditions
    // the particles are arranged on the x axis,  1   2   3
    // such that 2 is inside the cutoff radius of 1 and 3, but 1 and 3 are outside the cutoff
    // of course, the buffer will be set on the neighborlist so that 3 is included in it
    // thus, this case tests the ability of the force summer to sum more than one force on
    // a particle and ignore a particle outside the radius
    // Also particle 2 would not be within the cutoff of particle 1 if it were not the case that particle 1 has a shifted potential.

    // periodic boundary conditions will be handeled in another test
    std::shared_ptr<SystemDefinition> sysdef_3(new SystemDefinition(3, BoxDim(1000.0), 1, 0, 0, 0, 0, exec_conf));
    std::shared_ptr<ParticleData> pdata_3 = sysdef_3->getParticleData();
    pdata_3->setFlags(~PDataFlags(0));

    {
    ArrayHandle<Scalar4> h_pos(pdata_3->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_diameter(pdata_3->getDiameters(), access_location::host, access_mode::readwrite);

    h_pos.data[0].x = Scalar(-0.2);
    //h_pos.data[0].x = 0;
    h_pos.data[0].y = h_pos.data[0].z = Scalar(0.0);
    h_pos.data[1].x = Scalar(pow(2.0,1.0/6.0)); h_pos.data[1].y = h_pos.data[1].z = 0.0;
    h_pos.data[2].x = Scalar(2.0*pow(2.0,1.0/6.0)); h_pos.data[2].y = h_pos.data[2].z = 0.0;
    h_diameter.data[0]= Scalar(1.2);
    }

    Scalar maxdiam = pdata_3->getMaxDiameter();
    Scalar r_cut = Scalar(1.3);
    Scalar r_alpha = maxdiam/2 - 0.5;
    Scalar r_cut_wc = r_cut + 2 * r_alpha;

    std::shared_ptr<NeighborListTree> nlist_3(new NeighborListTree(sysdef_3, r_cut_wc, Scalar(3.0)));
    std::shared_ptr<PotentialPairSLJ> fc_3 = shiftedlj_creator(sysdef_3, nlist_3);
    fc_3->setRcut(0, 0, r_cut);

    // first test: setup a sigma of 1.0 so that all forces will be 0
    Scalar epsilon = Scalar(1.15);
    Scalar sigma = Scalar(1.0);
    Scalar alpha = Scalar(1.0);
    Scalar shiftedlj1 = Scalar(4.0) * epsilon * pow(sigma,Scalar(12.0));
    Scalar shiftedlj2 = alpha * Scalar(4.0) * epsilon * pow(sigma,Scalar(6.0));
    fc_3->setParams(0,0,make_scalar2(shiftedlj1,shiftedlj2));

    // compute the forces
    fc_3->compute(0);

    {
    GlobalArray<Scalar4>& force_array_1 =  fc_3->getForceArray();
    GlobalArray<Scalar>& virial_array_1 =  fc_3->getVirialArray();
    unsigned int pitch = virial_array_1.getPitch();
    ArrayHandle<Scalar4> h_force_1(force_array_1,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_1(virial_array_1,access_location::host,access_mode::read);
    MY_CHECK_CLOSE(h_force_1.data[0].x, 2.710943702, tol);
    MY_CHECK_SMALL(h_force_1.data[0].y, tol);
    MY_CHECK_SMALL(h_force_1.data[0].z, tol);
    MY_CHECK_CLOSE(h_force_1.data[0].w, -0.482660808, tol);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_1.data[0*pitch+0]
                                       +h_virial_1.data[3*pitch+0]
                                       +h_virial_1.data[5*pitch+0]), -0.597520027, tol);

    MY_CHECK_CLOSE(h_force_1.data[1].x, -2.710943702, tol);
    MY_CHECK_SMALL(h_force_1.data[1].y, tol);
    MY_CHECK_SMALL(h_force_1.data[1].z, tol);
    MY_CHECK_CLOSE(h_force_1.data[1].w, -1.057660808, tol);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_1.data[0*pitch+1]
                                       +h_virial_1.data[3*pitch+1]
                                       +h_virial_1.data[5*pitch+1]), -0.597520027, tol);

    MY_CHECK_SMALL(h_force_1.data[2].x, tol);
    MY_CHECK_SMALL(h_force_1.data[2].y, tol);
    MY_CHECK_SMALL(h_force_1.data[2].z, tol);
    MY_CHECK_CLOSE(h_force_1.data[2].w, -0.575, tol);
    MY_CHECK_SMALL(Scalar(1./3.)*(h_virial_1.data[0*pitch+2]
                                       +h_virial_1.data[3*pitch+2]
                                       +h_virial_1.data[5*pitch+2]), tol);
    }

    // now change sigma and alpha so we can check that it is computing the right force
    sigma = Scalar(1.2); // < bigger sigma should push particle 0 left and particle 2 right
    alpha = Scalar(0.45);
    shiftedlj1 = Scalar(4.0) * epsilon * pow(sigma,Scalar(12.0));
    shiftedlj2 = alpha * Scalar(4.0) * epsilon * pow(sigma,Scalar(6.0));
    fc_3->setParams(0,0,make_scalar2(shiftedlj1,shiftedlj2));
    fc_3->compute(1);

    {
    GlobalArray<Scalar4>& force_array_2 =  fc_3->getForceArray();
    GlobalArray<Scalar>& virial_array_2 =  fc_3->getVirialArray();
    unsigned int pitch = virial_array_2.getPitch();
    ArrayHandle<Scalar4> h_force_2(force_array_2,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_2(virial_array_2,access_location::host,access_mode::read);
    MY_CHECK_CLOSE(h_force_2.data[0].x, -27.05553467, tol);
    MY_CHECK_SMALL(h_force_2.data[0].y, tol);
    MY_CHECK_SMALL(h_force_2.data[0].z, tol);
    MY_CHECK_CLOSE(h_force_2.data[0].w, 0.915093686, tol);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_2.data[0*pitch+0]
                                       +h_virial_2.data[3*pitch+0]
                                       +h_virial_2.data[5*pitch+0]), 5.9633196325, tol);

    // center particle should still be a 0 force by symmetry
    MY_CHECK_CLOSE(h_force_2.data[1].x,-66.0427, tol);
    MY_CHECK_SMALL(h_force_2.data[1].y, 1e-5);
    MY_CHECK_SMALL(h_force_2.data[1].z, 1e-5);
    // there is still an energy and virial, though
    MY_CHECK_CLOSE(h_force_2.data[1].w, 4.496604724, tol);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_2.data[0*pitch+1]
                                       +h_virial_2.data[3*pitch+1]
                                       +h_virial_2.data[5*pitch+1]), 23.37985722, tol);

    MY_CHECK_CLOSE(h_force_2.data[2].x, 93.09822608552962, tol);
    MY_CHECK_SMALL(h_force_2.data[2].y, tol);
    MY_CHECK_SMALL(h_force_2.data[2].z, tol);
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
    MY_CHECK_CLOSE(h_force_3.data[0].x, 336.9779601, tol);
    MY_CHECK_CLOSE(h_force_3.data[2].x, -93.09822608552962, tol);
    }
    }

//! Tests the ability of a ShiftedLJForceCompute to handle periodic boundary conditions.  Also intentionally place a particle outside the cutoff of normally size particle but in the cutoff of a large particle
void shiftedlj_force_periodic_test(shiftedljforce_creator shiftedlj_creator, std::shared_ptr<ExecutionConfiguration> exec_conf)
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
    pdata_6->setPosition(2, make_scalar3(0.0,-19.35,0.0));
    pdata_6->setPosition(3, make_scalar3(0.0,19.6,0.0));
    pdata_6->setPosition(4, make_scalar3(0.0,0.0,-29.1));
    pdata_6->setPosition(5, make_scalar3(0.0,0.0,29.6));

    pdata_6->setType(0,0);
    pdata_6->setType(1,1);
    pdata_6->setType(2,2);
    pdata_6->setType(3,0);
    pdata_6->setType(4,2);
    pdata_6->setType(5,1);

    pdata_6->setDiameter(0,1.2);
    pdata_6->setDiameter(2,1.5);
    pdata_6->setDiameter(4,2.0);

    Scalar maxdiam = pdata_6->getMaxDiameter();
    Scalar r_cut = Scalar(1.3);
    Scalar r_alpha = Scalar(maxdiam/2.0 - 0.5);
    Scalar r_cut_wc = Scalar(r_cut + 2.0 * r_alpha);


    std::shared_ptr<NeighborListTree> nlist_6(new NeighborListTree(sysdef_6, r_cut_wc, Scalar(3.0)));
    std::shared_ptr<PotentialPairSLJ> fc_6 = shiftedlj_creator(sysdef_6, nlist_6);
    fc_6->setRcut(0, 0, r_cut);
    fc_6->setRcut(0, 1, r_cut);
    fc_6->setRcut(0, 2, r_cut);
    fc_6->setRcut(1, 1, r_cut);
    fc_6->setRcut(1, 2, r_cut);
    fc_6->setRcut(2, 2, r_cut);

    // choose a small sigma so that all interactions are attractive
    Scalar epsilon = Scalar(1.0);
    Scalar sigma = Scalar(0.5);
    Scalar alpha = Scalar(0.45);
    Scalar shiftedlj1 = Scalar(4.0) * epsilon * pow(sigma,Scalar(12.0));
    Scalar shiftedlj2 = alpha * Scalar(4.0) * epsilon * pow(sigma,Scalar(6.0));

    // make life easy: just change epsilon for the different pairs
    fc_6->setParams(0,0,make_scalar2(shiftedlj1,shiftedlj2));
    fc_6->setParams(0,1,make_scalar2(Scalar(2.0)*shiftedlj1,Scalar(2.0)*shiftedlj2));
    fc_6->setParams(0,2,make_scalar2(Scalar(3.0)*shiftedlj1,Scalar(3.0)*shiftedlj2));
    fc_6->setParams(1,1,make_scalar2(Scalar(4.0)*shiftedlj1,Scalar(4.0)*shiftedlj2));
    fc_6->setParams(1,2,make_scalar2(Scalar(5.0)*shiftedlj1,Scalar(5.0)*shiftedlj2));
    fc_6->setParams(2,2,make_scalar2(Scalar(6.0)*shiftedlj1,Scalar(6.0)*shiftedlj2));

    fc_6->compute(0);

    {
    GlobalArray<Scalar4>& force_array_4 =  fc_6->getForceArray();
    GlobalArray<Scalar>& virial_array_4 =  fc_6->getVirialArray();
    unsigned int pitch = virial_array_4.getPitch();
    ArrayHandle<Scalar4> h_force_4(force_array_4,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_4(virial_array_4,access_location::host,access_mode::read);
    // particle 0 should be pulled left
    MY_CHECK_CLOSE(h_force_4.data[0].x, -1.679141673, tol);
    MY_CHECK_SMALL(h_force_4.data[0].y, tol);
    MY_CHECK_SMALL(h_force_4.data[0].z, tol);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_4.data[0*pitch+0]
                                       +h_virial_4.data[3*pitch+0]
                                       +h_virial_4.data[5*pitch+0]),-0.223885556, tol);

    // particle 1 should be pulled right
    MY_CHECK_CLOSE(h_force_4.data[1].x, 1.679141673, tol);
    MY_CHECK_SMALL(h_force_4.data[1].y, 1e-5);
    MY_CHECK_SMALL(h_force_4.data[1].z, 1e-5);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_4.data[0*pitch+1]
                                       +h_virial_4.data[3*pitch+1]
                                       +h_virial_4.data[5*pitch+1]), -0.223885556, tol);

    // particle 2 should be pulled down
    MY_CHECK_CLOSE(h_force_4.data[2].y, -1.77449965121923, tol);
    MY_CHECK_SMALL(h_force_4.data[2].x, tol);
    MY_CHECK_SMALL(h_force_4.data[2].z, tol);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_4.data[0*pitch+2]
                                       +h_virial_4.data[3*pitch+2]
                                       +h_virial_4.data[5*pitch+2]), -0.310537439, tol);


    // particle 3 should be pulled up
    MY_CHECK_CLOSE(h_force_4.data[3].y, 1.77449965121923, tol);
    MY_CHECK_SMALL(h_force_4.data[3].x, 1e-5);
    MY_CHECK_SMALL(h_force_4.data[3].z, 1e-5);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_4.data[0*pitch+3]
                                       +h_virial_4.data[3*pitch+3]
                                       +h_virial_4.data[5*pitch+3]), -0.310537439, tol);

    // particle 4 should be pulled back
    MY_CHECK_CLOSE(h_force_4.data[4].z, -2.95749941869871, tol);
    MY_CHECK_SMALL(h_force_4.data[4].x, tol);
    MY_CHECK_SMALL(h_force_4.data[4].y, tol);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_4.data[0*pitch+4]
                                       +h_virial_4.data[3*pitch+4]
                                       +h_virial_4.data[5*pitch+4]), -0.640791541, tol);

    // particle 3 should be pulled forward
    MY_CHECK_CLOSE(h_force_4.data[5].z, 2.95749941869871, tol);
    MY_CHECK_SMALL(h_force_4.data[5].x, 1e-5);
    MY_CHECK_SMALL(h_force_4.data[5].y, 1e-5);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_4.data[0*pitch+5]
                                       +h_virial_4.data[3*pitch+5]
                                       +h_virial_4.data[5*pitch+5]), -0.640791541, tol);
    }
    }

//! Unit test a comparison between 2 ShiftedLJForceComputes on a "real" system
void shiftedlj_force_comparison_test(shiftedljforce_creator shiftedlj_creator1,
                                     shiftedljforce_creator shiftedlj_creator2,
                                     std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    const unsigned int N = 5000;

    // create a random particle system to sum forces on
    RandomInitializer rand_init(N, Scalar(0.05), Scalar(1.3), "A");
    std::shared_ptr< SnapshotSystemData<Scalar> > snap = rand_init.getSnapshot();
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));
    std::shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    pdata->setFlags(~PDataFlags(0));
    std::shared_ptr<NeighborListTree> nlist(new NeighborListTree(sysdef, Scalar(3.0), Scalar(0.8)));

    std::shared_ptr<PotentialPairSLJ> fc1 = shiftedlj_creator1(sysdef, nlist);
    std::shared_ptr<PotentialPairSLJ> fc2 = shiftedlj_creator2(sysdef, nlist);
    fc1->setRcut(0, 0, Scalar(3.0));
    fc2->setRcut(0, 0, Scalar(3.0));

    // setup some values for alpha and sigma
    Scalar epsilon = Scalar(1.0);
    Scalar sigma = Scalar(1.2);
    Scalar alpha = Scalar(0.45);
    Scalar shiftedlj1 = Scalar(4.0) * epsilon * pow(sigma,Scalar(12.0));
    Scalar shiftedlj2 = alpha * Scalar(4.0) * epsilon * pow(sigma,Scalar(6.0));

    // specify the force parameters
    fc1->setParams(0,0,make_scalar2(shiftedlj1,shiftedlj2));
    fc2->setParams(0,0,make_scalar2(shiftedlj1,shiftedlj2));

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

    for (unsigned int j = 0; j < 6; j++)
        deltav2[j] =0;

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

//! PotentialPairSLJ creator for unit tests
std::shared_ptr<PotentialPairSLJ> base_class_shiftedlj_creator(std::shared_ptr<SystemDefinition> sysdef,
                                                          std::shared_ptr<NeighborList> nlist)
    {
    return std::shared_ptr<PotentialPairSLJ>(new PotentialPairSLJ(sysdef, nlist));
    }

#ifdef ENABLE_CUDA
//! PotentialPairSLJGPU creator for unit tests
std::shared_ptr<PotentialPairSLJ> gpu_shiftedlj_creator(std::shared_ptr<SystemDefinition> sysdef,
                                                   std::shared_ptr<NeighborList> nlist)
    {
    nlist->setStorageMode(NeighborList::full);
    std::shared_ptr<PotentialPairSLJGPU> lj(new PotentialPairSLJGPU(sysdef, nlist));
    return lj;
    }
#endif

//! test case for particle test on CPU
UP_TEST( SLJForce_particle )
    {
    shiftedljforce_creator shiftedlj_creator_base = bind(base_class_shiftedlj_creator, _1, _2);
    shiftedlj_force_particle_test(shiftedlj_creator_base, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

//! test case for periodic test on CPU
UP_TEST( SLJForce_periodic )
    {
    shiftedljforce_creator shiftedlj_creator_base = bind(base_class_shiftedlj_creator, _1, _2);
    shiftedlj_force_periodic_test(shiftedlj_creator_base, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }


# ifdef ENABLE_CUDA
//! test case for particle test on CPU - threaded
UP_TEST( SLJForceGPU_particle )
    {
    shiftedljforce_creator shiftedlj_creator_gpu = bind(gpu_shiftedlj_creator, _1, _2);
    shiftedlj_force_particle_test(shiftedlj_creator_gpu, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }


//! test case for periodic test on the GPU
UP_TEST( SLJForceGPU_periodic )
    {
    shiftedljforce_creator shiftedlj_creator_gpu = bind(gpu_shiftedlj_creator, _1, _2);
    shiftedlj_force_periodic_test(shiftedlj_creator_gpu, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! test case for comparing GPU output to base class output
UP_TEST( SLJForceGPU_compare )
    {
    shiftedljforce_creator shiftedlj_creator_gpu = bind(gpu_shiftedlj_creator, _1, _2);
    shiftedljforce_creator shiftedlj_creator_base = bind(base_class_shiftedlj_creator, _1, _2);
    shiftedlj_force_comparison_test(shiftedlj_creator_base,
                                    shiftedlj_creator_gpu,
                                    std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

#endif
