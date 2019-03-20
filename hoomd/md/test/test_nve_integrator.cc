// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.




#include <iostream>

#include <functional>
#include <memory>

#include "hoomd/ConstForceCompute.h"
#include "hoomd/ComputeThermo.h"
#include "hoomd/md/TwoStepNVE.h"
#ifdef ENABLE_CUDA
#include "hoomd/md/TwoStepNVEGPU.h"
#endif

#include "hoomd/md/IntegratorTwoStep.h"

#include "hoomd/md/AllPairPotentials.h"
#include "hoomd/md/AllAnisoPairPotentials.h"
#include "hoomd/md/NeighborListBinned.h"
#include "hoomd/md/NeighborListTree.h"
#include "hoomd/Initializers.h"
#include "hoomd/SnapshotSystemData.h"

#include <math.h>

using namespace std;
using namespace std::placeholders;


/*! \file nve_updater_test.cc
    \brief Implements unit tests for TwoStepNVE and descendants
    \ingroup unit_tests
*/

#include "hoomd/test/upp11_config.h"
HOOMD_UP_MAIN();

//! Typedef'd NVEUpdater class factory
typedef std::function<std::shared_ptr<TwoStepNVE> (std::shared_ptr<SystemDefinition> sysdef,
                                                       std::shared_ptr<ParticleGroup> group)> twostepnve_creator;

//! Integrate 1 particle through time and compare to an analytical solution
void nve_updater_integrate_tests(twostepnve_creator nve_creator, std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // check that the nve updater can actually integrate particle positions and velocities correctly
    // start with a 2 particle system to keep things simple: also put everything in a huge box so boundary conditions
    // don't come into play
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(2, BoxDim(1000.0), 4, 0, 0, 0, 0, exec_conf));
    std::shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    std::shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef, 0, pdata->getN()-1));
    std::shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_all));

    {
    ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(pdata->getVelocities(), access_location::host, access_mode::readwrite);
    // setup a simple initial state
    h_pos.data[0].x = 0.0;
    h_pos.data[0].y = 1.0;
    h_pos.data[0].z = 2.0;
    h_vel.data[0].x = 3.0;
    h_vel.data[0].y = 2.0;
    h_vel.data[0].z = 1.0;

    h_pos.data[1].x = 10.0;
    h_pos.data[1].y = 11.0;
    h_pos.data[1].z = 12.0;
    h_vel.data[1].x = 13.0;
    h_vel.data[1].y = 12.0;
    h_vel.data[1].z = 11.0;
    }

    Scalar deltaT = Scalar(0.0001);
    std::shared_ptr<TwoStepNVE> two_step_nve = nve_creator(sysdef, group_all);
    std::shared_ptr<IntegratorTwoStep> nve_up(new IntegratorTwoStep(sysdef, deltaT));
    nve_up->addIntegrationMethod(two_step_nve);

    // also test the ability of the updater to add two force computes together properly
    std::shared_ptr<ConstForceCompute> fc1(new ConstForceCompute(sysdef, 1.5, 0.0, 0.0));
    nve_up->addForceCompute(fc1);
    std::shared_ptr<ConstForceCompute> fc2(new ConstForceCompute(sysdef, 0.0, 2.5, 0.0));
    nve_up->addForceCompute(fc2);

    nve_up->prepRun(0);

    // verify proper integration compared to x = x0 + v0 t + 1/2 a t^2, v = v0 + a t
    // roundoff errors prevent this from keeping within 0.1% error for long
    for (int i = 0; i < 500; i++)
        {
        {
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(), access_location::host, access_mode::read);

        Scalar t = Scalar(i) * deltaT;
        MY_CHECK_CLOSE(h_pos.data[0].x, 0.0 + 3.0 * t + 1.0/2.0 * 1.5 * t*t, loose_tol);
        MY_CHECK_CLOSE(h_vel.data[0].x, 3.0 + 1.5 * t, loose_tol);

        MY_CHECK_CLOSE(h_pos.data[0].y, 1.0 + 2.0 * t + 1.0/2.0 * 2.5 * t*t, loose_tol);
        MY_CHECK_CLOSE(h_vel.data[0].y, 2.0 + 2.5 * t, loose_tol);

        MY_CHECK_CLOSE(h_pos.data[0].z, 2.0 + 1.0 * t + 1.0/2.0 * 0 * t*t, loose_tol);
        MY_CHECK_CLOSE(h_vel.data[0].z, 1.0 + 0 * t, loose_tol);

        MY_CHECK_CLOSE(h_pos.data[1].x, 10.0 + 13.0 * t + 1.0/2.0 * 1.5 * t*t, loose_tol);
        MY_CHECK_CLOSE(h_vel.data[1].x, 13.0 + 1.5 * t, loose_tol);

        MY_CHECK_CLOSE(h_pos.data[1].y, 11.0 + 12.0 * t + 1.0/2.0 * 2.5 * t*t, loose_tol);
        MY_CHECK_CLOSE(h_vel.data[1].y, 12.0 + 2.5 * t, loose_tol);

        MY_CHECK_CLOSE(h_pos.data[1].z, 12.0 + 11.0 * t + 1.0/2.0 * 0 * t*t, loose_tol);
        MY_CHECK_CLOSE(h_vel.data[1].z, 11.0 + 0 * t, loose_tol);
        }

        nve_up->update(i);
        }
    }

//! Check that the particle movement limit works
void nve_updater_limit_tests(twostepnve_creator nve_creator, std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // create a simple 1 particle system
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(1, BoxDim(1000.0), 1, 0, 0, 0, 0, exec_conf));
    std::shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    std::shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef, 0, pdata->getN()-1));
    std::shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_all));

    {
    ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(pdata->getVelocities(), access_location::host, access_mode::readwrite);

    // setup a simple initial state
    h_pos.data[0].x = 0.0;
    h_pos.data[0].y = 1.0;
    h_pos.data[0].z = 2.0;
    h_vel.data[0].x = 0.0;
    h_vel.data[0].y = 0.0;
    h_vel.data[0].z = 0.0;
    }

    Scalar deltaT = Scalar(0.0001);
    std::shared_ptr<TwoStepNVE> two_step_nve = nve_creator(sysdef, group_all);
    std::shared_ptr<IntegratorTwoStep> nve_up(new IntegratorTwoStep(sysdef, deltaT));
    nve_up->addIntegrationMethod(two_step_nve);

    // set the limit
    Scalar limit = Scalar(0.1);
    two_step_nve->setLimit(limit);

    // create an insanely large force to test the limiting method
    std::shared_ptr<ConstForceCompute> fc1(new ConstForceCompute(sysdef, 1e9, 2e9, 3e9));
    nve_up->addForceCompute(fc1);

    // expected movement vectors
    Scalar dx = limit / sqrt(14.0);
    Scalar dy = 2.0 * limit / sqrt(14.0);
    Scalar dz = 3.0 * limit / sqrt(14.0);

    Scalar vx = limit / sqrt(14.0) / deltaT;
    Scalar vy = 2.0 * limit / sqrt(14.0) / deltaT;
    Scalar vz = 3.0 * limit / sqrt(14.0) / deltaT;

    nve_up->prepRun(0);

    // verify proper integration compared to x = x0 + dx * i
    nve_up->update(0);
    for (int i = 1; i < 500; i++)
        {
        {
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(), access_location::host, access_mode::read);

        MY_CHECK_CLOSE(h_pos.data[0].x, 0.0 + dx * Scalar(i), tol);
        MY_CHECK_CLOSE(h_vel.data[0].x, vx, tol);

        MY_CHECK_CLOSE(h_pos.data[0].y, 1.0 + dy * Scalar(i), tol);
        MY_CHECK_CLOSE(h_vel.data[0].y, vy, tol);

        MY_CHECK_CLOSE(h_pos.data[0].z, 2.0 + dz * Scalar(i), tol);
        MY_CHECK_CLOSE(h_vel.data[0].z, vz, tol);
        }

        nve_up->update(i);
        }
    }


//! Make a few particles jump across the boundary and verify that the updater works
void nve_updater_boundary_tests(twostepnve_creator nve_creator, std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    ////////////////////////////////////////////////////////////////////
    // now, lets do a more thorough test and include boundary conditions
    // there are way too many permutations to test here, so I will simply
    // test +x, -x, +y, -y, +z, and -z independently
    // build a 6 particle system with particles set to move across each boundary
    std::shared_ptr<SystemDefinition> sysdef_6(new SystemDefinition(6, BoxDim(20.0, 40.0, 60.0), 1, 0, 0, 0, 0, exec_conf));
    std::shared_ptr<ParticleData> pdata_6 = sysdef_6->getParticleData();
    std::shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef_6, 0, pdata_6->getN()-1));
    std::shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef_6, selector_all));

    {
    ArrayHandle<Scalar4> h_pos(pdata_6->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(pdata_6->getVelocities(), access_location::host, access_mode::readwrite);
    h_pos.data[0].x = Scalar(-9.6); h_pos.data[0].y = 0; h_pos.data[0].z = 0.0;
    h_vel.data[0].x = Scalar(-0.5);
    h_pos.data[1].x =  Scalar(9.6); h_pos.data[1].y = 0; h_pos.data[1].z = 0.0;
    h_vel.data[1].x = Scalar(0.6);
    h_pos.data[2].x = 0; h_pos.data[2].y = Scalar(-19.6); h_pos.data[2].z = 0.0;
    h_vel.data[2].y = Scalar(-0.5);
    h_pos.data[3].x = 0; h_pos.data[3].y = Scalar(19.6); h_pos.data[3].z = 0.0;
    h_vel.data[3].y = Scalar(0.6);
    h_pos.data[4].x = 0; h_pos.data[4].y = 0; h_pos.data[4].z = Scalar(-29.6);
    h_vel.data[4].z = Scalar(-0.5);
    h_pos.data[5].x = 0; h_pos.data[5].y = 0; h_pos.data[5].z =  Scalar(29.6);
    h_vel.data[5].z = Scalar(0.6);
    }

    Scalar deltaT = 1.0;
    std::shared_ptr<TwoStepNVE> two_step_nve = nve_creator(sysdef_6, group_all);
    std::shared_ptr<IntegratorTwoStep> nve_up(new IntegratorTwoStep(sysdef_6, deltaT));
    nve_up->addIntegrationMethod(two_step_nve);

    // no forces on these particles
    std::shared_ptr<ConstForceCompute> fc1(new ConstForceCompute(sysdef_6, 0, 0.0, 0.0));
    nve_up->addForceCompute(fc1);

    nve_up->prepRun(0);

    // move the particles across the boundary
    nve_up->update(0);

    // check that they go to the proper final position
    {
    ArrayHandle<Scalar4> h_pos(pdata_6->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<int3> h_image(pdata_6->getImages(), access_location::host, access_mode::read);
    MY_CHECK_CLOSE(h_pos.data[0].x, 9.9, tol);
    UP_ASSERT_EQUAL(h_image.data[0].x, -1);
    MY_CHECK_CLOSE(h_pos.data[1].x, -9.8, tol);
    UP_ASSERT_EQUAL(h_image.data[1].x, 1);
    MY_CHECK_CLOSE(h_pos.data[2].y, 19.9, tol);
    UP_ASSERT_EQUAL(h_image.data[2].y, -1);
    MY_CHECK_CLOSE(h_pos.data[3].y, -19.8, tol);
    UP_ASSERT_EQUAL(h_image.data[3].y, 1);
    MY_CHECK_CLOSE(h_pos.data[4].z, 29.9, tol);
    UP_ASSERT_EQUAL(h_image.data[4].z, -1);
    MY_CHECK_CLOSE(h_pos.data[5].z, -29.8, tol);
    UP_ASSERT_EQUAL(h_image.data[5].z, 1);
    }
    }

//! Compares the output from one TwoStepNVE to another
void nve_updater_compare_test(twostepnve_creator nve_creator1,
                              twostepnve_creator nve_creator2,
                              std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    const unsigned int N = 1000;

    // create two identical random particle systems to simulate
    SimpleCubicInitializer cubic_init(10, Scalar(1.2), "A");
    std::shared_ptr< SnapshotSystemData<Scalar> > snap = cubic_init.getSnapshot();

    std::shared_ptr<SystemDefinition> sysdef1(new SystemDefinition(snap, exec_conf));
    std::shared_ptr<ParticleData> pdata1 = sysdef1->getParticleData();
    std::shared_ptr<ParticleSelector> selector_all1(new ParticleSelectorTag(sysdef1, 0, pdata1->getN()-1));
    std::shared_ptr<ParticleGroup> group_all1(new ParticleGroup(sysdef1, selector_all1));

    std::shared_ptr<SystemDefinition> sysdef2(new SystemDefinition(snap, exec_conf));
    std::shared_ptr<ParticleData> pdata2 = sysdef2->getParticleData();
    std::shared_ptr<ParticleSelector> selector_all2(new ParticleSelectorTag(sysdef2, 0, pdata2->getN()-1));
    std::shared_ptr<ParticleGroup> group_all2(new ParticleGroup(sysdef2, selector_all2));

    std::shared_ptr<NeighborListTree> nlist1(new NeighborListTree(sysdef1, Scalar(3.0), Scalar(0.8)));
    std::shared_ptr<NeighborListTree> nlist2(new NeighborListTree(sysdef2, Scalar(3.0), Scalar(0.8)));

    std::shared_ptr<PotentialPairLJ> fc1(new PotentialPairLJ(sysdef1, nlist1));
    fc1->setRcut(0, 0, Scalar(3.0));
    std::shared_ptr<PotentialPairLJ> fc2(new PotentialPairLJ(sysdef2, nlist2));
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

    std::shared_ptr<TwoStepNVE> two_step_nve1 = nve_creator1(sysdef1, group_all1);
    std::shared_ptr<IntegratorTwoStep> nve1(new IntegratorTwoStep(sysdef1, Scalar(0.005)));
    nve1->addIntegrationMethod(two_step_nve1);

    std::shared_ptr<TwoStepNVE> two_step_nve2 = nve_creator2(sysdef2, group_all2);
    std::shared_ptr<IntegratorTwoStep> nve2(new IntegratorTwoStep(sysdef2, Scalar(0.005)));
    nve2->addIntegrationMethod(two_step_nve2);

    nve1->addForceCompute(fc1);
    nve2->addForceCompute(fc2);

    nve1->prepRun(0);
    nve2->prepRun(0);

    // step for only a few time steps and verify that they are the same
    // we can't do much more because these things are chaotic and diverge quickly
    for (int i = 0; i < 5; i++)
        {
        std::cout << "i = " << i << std::endl;
        {
        ArrayHandle<Scalar4> h_pos1(pdata1->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_vel1(pdata1->getVelocities(), access_location::host, access_mode::read);
        ArrayHandle<Scalar3> h_accel1(pdata1->getAccelerations(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_pos2(pdata2->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_vel2(pdata2->getVelocities(), access_location::host, access_mode::read);
        ArrayHandle<Scalar3> h_accel2(pdata2->getAccelerations(), access_location::host, access_mode::read);

        Scalar rough_tol = 2.0;
        //cout << arrays1.x[100] << " " << arrays2.x[100] << endl;

        // check position, velocity and acceleration
        for (unsigned int j = 0; j < N; j++)
            {
            MY_CHECK_CLOSE(h_pos1.data[j].x, h_pos2.data[j].x, rough_tol);
            MY_CHECK_CLOSE(h_pos1.data[j].y, h_pos2.data[j].y, rough_tol);
            MY_CHECK_CLOSE(h_pos1.data[j].z, h_pos2.data[j].z, rough_tol);

            MY_CHECK_CLOSE(h_vel1.data[j].x, h_vel2.data[j].x, rough_tol);
            MY_CHECK_CLOSE(h_vel1.data[j].y, h_vel2.data[j].y, rough_tol);
            MY_CHECK_CLOSE(h_vel1.data[j].z, h_vel2.data[j].z, rough_tol);

            MY_CHECK_CLOSE(h_accel1.data[j].x, h_accel2.data[j].x, rough_tol);
            MY_CHECK_CLOSE(h_accel1.data[j].y, h_accel2.data[j].y, rough_tol);
            MY_CHECK_CLOSE(h_accel1.data[j].z, h_accel2.data[j].z, rough_tol);
            }

        }
        nve1->update(i);
        nve2->update(i);
        }
    }

void nve_updater_aniso_test(std::shared_ptr<ExecutionConfiguration> exec_conf, twostepnve_creator nve_creator)
{
    // initialize random particle system
    SimpleCubicInitializer cubic_init(12, Scalar(1.2), "A");
    std::shared_ptr< SnapshotSystemData<Scalar> > snap = cubic_init.getSnapshot();

    // have to set moment of inertia to actually test aniso integration
    for(unsigned int i(0); i < snap->particle_data.size; ++i)
        snap->particle_data.inertia[i] = vec3<Scalar>(1.0, 1.0, 1.0);

    std::shared_ptr<SystemDefinition> sysdef_1(new SystemDefinition(snap, exec_conf));
    std::shared_ptr<ParticleData> pdata_1 = sysdef_1->getParticleData();
    std::shared_ptr<ParticleSelector> selector_all_1(new ParticleSelectorTag(sysdef_1, 0, pdata_1->getNGlobal()-1));
    std::shared_ptr<ParticleGroup> group_all_1(new ParticleGroup(sysdef_1, selector_all_1));

    Scalar r_cut = Scalar(3.0);
    Scalar r_buff = Scalar(0.4);
    std::shared_ptr<NeighborList> nlist_1(new NeighborListBinned(sysdef_1, r_cut, r_buff));

    nlist_1->setStorageMode(NeighborList::full);
    std::shared_ptr<AnisoPotentialPairGB> fc_1 = std::shared_ptr<AnisoPotentialPairGB>(new AnisoPotentialPairGB(sysdef_1, nlist_1));

    fc_1->setRcut(0, 0, r_cut);

    // setup some values for alpha and sigma
    Scalar epsilon = Scalar(1.0);
    Scalar lperp = Scalar(0.45);
    Scalar lpar = Scalar(0.5);
    fc_1->setParams(0,0,make_scalar3(epsilon,lperp,lpar));
    // If we want accurate calculation of potential energy, we need to apply the
    // energy shift
    fc_1->setShiftMode(AnisoPotentialPairGB::shift);

    Scalar deltaT = Scalar(0.0025);
    std::shared_ptr<IntegratorTwoStep> nve_1(new IntegratorTwoStep(sysdef_1, deltaT));
    std::shared_ptr<ComputeThermo> thermo_1 = std::shared_ptr<ComputeThermo>(new ComputeThermo(sysdef_1,group_all_1));

    std::shared_ptr<TwoStepNVE> two_step_nve_1 = nve_creator(sysdef_1, group_all_1);
;
    nve_1->addIntegrationMethod(two_step_nve_1);
    nve_1->addForceCompute(fc_1);

    unsigned int ndof = nve_1->getNDOF(group_all_1);
    thermo_1->setNDOF(ndof);
    unsigned int ndof_rot = nve_1->getRotationalNDOF(group_all_1);
    thermo_1->setRotationalNDOF(ndof_rot);

    nve_1->prepRun(0);

    PDataFlags flags;
    flags[pdata_flag::potential_energy] = 1;
    flags[pdata_flag::rotational_kinetic_energy] = 1;
    pdata_1->setFlags(flags);

    // equilibrate
    std::cout << "Testing anisotropic mode" << std::endl;
    unsigned int n_equil_steps = 150000;
    std::cout << "Equilibrating for " << n_equil_steps << " time steps..." << std::endl;
    unsigned int i =0;

    for (i=0; i< n_equil_steps; i++)
        {
        nve_1->update(i);
        if (i % 1000 == 0)
            std::cout << i << std::endl;
        }

    // 0.2  % tolerance for conserved quantity
    Scalar H_tol = 0.2;

    // conserved quantity
    thermo_1->compute(i+1);
    Scalar H_ini = thermo_1->getKineticEnergy() + thermo_1->getPotentialEnergy();
    std::cout << "Initial energy: " << H_ini << std::endl;

    int n_measure_steps = 25000;
    std::cout << "Measuring conserved quantity for another " << n_measure_steps << " time steps..." << std::endl;
    for (i=n_equil_steps; i< n_equil_steps+n_measure_steps; i++)
        {
        // get conserved quantity
        nve_1->update(i);

        thermo_1->compute(i+1);

        Scalar H = thermo_1->getKineticEnergy() + thermo_1->getPotentialEnergy();

        if (i % 1000 == 0)
            std::cout << i << ' ' << H << std::endl;

        MY_CHECK_CLOSE(H_ini,H, H_tol);
        }
    }

//! TwoStepNVE factory for the unit tests
std::shared_ptr<TwoStepNVE> base_class_nve_creator(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<ParticleGroup> group)
    {
    return std::shared_ptr<TwoStepNVE>(new TwoStepNVE(sysdef, group));
    }

#ifdef ENABLE_CUDA
//! TwoStepNVEGPU factory for the unit tests
std::shared_ptr<TwoStepNVE> gpu_nve_creator(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<ParticleGroup> group)
    {
    return std::shared_ptr<TwoStepNVE>(new TwoStepNVEGPU(sysdef, group));
    }
#endif


//! test case for base class integration tests
UP_TEST( TwoStepNVE_integrate_tests )
    {
    twostepnve_creator nve_creator = bind(base_class_nve_creator, _1, _2);
    nve_updater_integrate_tests(nve_creator, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

//! test case for base class limit tests
UP_TEST( TwoStepNVE_limit_tests )
    {
    twostepnve_creator nve_creator = bind(base_class_nve_creator, _1, _2);
    nve_updater_limit_tests(nve_creator, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

//! test case for base class boundary tests
UP_TEST( TwoStepNVE_boundary_tests )
    {
    twostepnve_creator nve_creator = bind(base_class_nve_creator, _1, _2);
    nve_updater_boundary_tests(nve_creator, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

//! Performs a basic equilibration test of TwoStepNVE
UP_TEST( TwoStepNVE_aniso_test )
    {
    nve_updater_aniso_test(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)),bind(base_class_nve_creator, _1, _2));
    }

//! Need work on NVEUpdaterGPU with rigid bodies to test these cases
#ifdef ENABLE_CUDA
//! test case for base class integration tests
UP_TEST( TwoStepNVEGPU_integrate_tests )
    {
    twostepnve_creator nve_creator_gpu = bind(gpu_nve_creator, _1, _2);
    nve_updater_integrate_tests(nve_creator_gpu, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! test case for base class limit tests
UP_TEST( TwoStepNVEGPU_limit_tests )
    {
    twostepnve_creator nve_creator = bind(gpu_nve_creator, _1, _2);
    nve_updater_limit_tests(nve_creator, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! test case for base class boundary tests
UP_TEST( TwoStepNVEGPU_boundary_tests )
    {
    twostepnve_creator nve_creator_gpu = bind(gpu_nve_creator, _1, _2);
    nve_updater_boundary_tests(nve_creator_gpu, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! test case for comparing the GPU and CPU NVEUpdaters
UP_TEST( TwoStepNVEGPU_comparison_tests)
    {
    twostepnve_creator nve_creator_gpu = bind(gpu_nve_creator, _1, _2);
    twostepnve_creator nve_creator = bind(base_class_nve_creator, _1, _2);
    nve_updater_compare_test(nve_creator, nve_creator_gpu, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! test case for testing aniso integration
UP_TEST( TwoStepNVEGPU_aniso_tests)
    {
    nve_updater_aniso_test(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)),bind(gpu_nve_creator, _1, _2));
    }

#endif
