// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.



#include <iostream>

#include <functional>

#include "hoomd/ConstForceCompute.h"
#include "hoomd/md/TwoStepNVTMTK.h"
#include "hoomd/ComputeThermo.h"
#ifdef ENABLE_CUDA
#include "hoomd/md/TwoStepNVTMTKGPU.h"
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

/*! \file nvt_updater_test.cc
    \brief Implements unit tests for NVTUpdater and descendants
    \ingroup unit_tests
*/

#include "hoomd/test/upp11_config.h"
HOOMD_UP_MAIN();


//! Typedef'd NVTUpdater class factory
typedef std::function<std::shared_ptr<TwoStepNVTMTK> (std::shared_ptr<SystemDefinition> sysdef,
                                                 std::shared_ptr<ParticleGroup> group,
                                                 std::shared_ptr<ComputeThermo> thermo,
                                                 Scalar Q,
                                                 Scalar T)> twostepnvt_creator;

//! NVTUpdater creator
std::shared_ptr<TwoStepNVTMTK> base_class_nvt_creator(std::shared_ptr<SystemDefinition> sysdef,
                                              std::shared_ptr<ParticleGroup> group,
                                              std::shared_ptr<ComputeThermo> thermo,
                                              Scalar Q,
                                              Scalar T)
    {
    std::shared_ptr<VariantConst> T_variant(new VariantConst(T));
    return std::shared_ptr<TwoStepNVTMTK>(new TwoStepNVTMTK(sysdef, group, thermo, Q, T_variant));
    }

#ifdef ENABLE_CUDA
//! NVTUpdaterGPU factory for the unit tests
std::shared_ptr<TwoStepNVTMTK> gpu_nvt_creator(std::shared_ptr<SystemDefinition> sysdef,
                                       std::shared_ptr<ParticleGroup> group,
                                       std::shared_ptr<ComputeThermo> thermo,
                                       Scalar Q,
                                       Scalar T)
    {
    std::shared_ptr<VariantConst> T_variant(new VariantConst(T));
    return std::shared_ptr<TwoStepNVTMTK>(new TwoStepNVTMTKGPU(sysdef, group, thermo, Q, T_variant));
    }
#endif

void test_nvt_mtk_integrator(std::shared_ptr<ExecutionConfiguration> exec_conf, twostepnvt_creator nvt_creator)
{
    // initialize a particle system
    SimpleCubicInitializer cubic_init(12, Scalar(1.2), "A");
    std::shared_ptr< SnapshotSystemData<Scalar> > snap = cubic_init.getSnapshot();

    std::shared_ptr<SystemDefinition> sysdef_1(new SystemDefinition(snap, exec_conf));
    std::shared_ptr<ParticleData> pdata_1 = sysdef_1->getParticleData();
    std::shared_ptr<ParticleSelector> selector_all_1(new ParticleSelectorTag(sysdef_1, 0, pdata_1->getNGlobal()-1));
    std::shared_ptr<ParticleGroup> group_all_1(new ParticleGroup(sysdef_1, selector_all_1));

    Scalar r_cut = Scalar(3.0);
    Scalar r_buff = Scalar(0.8);
    std::shared_ptr<NeighborListTree> nlist_1(new NeighborListTree(sysdef_1, r_cut, r_buff));
    nlist_1->setRCutPair(0,0,r_cut);
    nlist_1->setStorageMode(NeighborList::full);
    std::shared_ptr<PotentialPairLJ> fc_1 = std::shared_ptr<PotentialPairLJ>(new PotentialPairLJ(sysdef_1, nlist_1));

    fc_1->setRcut(0, 0, r_cut);

    // setup some values for alpha and sigma
    Scalar epsilon = Scalar(1.0);
    Scalar sigma = Scalar(1.0);
    Scalar alpha = Scalar(1.0);
    Scalar lj1 = Scalar(4.0) * epsilon * pow(sigma,Scalar(12.0));
    Scalar lj2 = alpha * Scalar(4.0) * epsilon * pow(sigma,Scalar(6.0));
    fc_1->setParams(0,0,make_scalar2(lj1,lj2));
    // If we want accurate calculation of potential energy, we need to apply the
    // energy shift
    fc_1->setShiftMode(PotentialPairLJ::shift);

    Scalar deltaT = Scalar(0.004);
    Scalar T_ref = Scalar(1.0);
    Scalar tau = Scalar(0.5);
    std::shared_ptr<IntegratorTwoStep> nvt_1(new IntegratorTwoStep(sysdef_1, deltaT));
    std::shared_ptr<ComputeThermo> thermo_1 = std::shared_ptr<ComputeThermo>(new ComputeThermo(sysdef_1,group_all_1));

    // ComputeThermo for integrator
    std::shared_ptr<ComputeThermo> thermo_nvt = std::shared_ptr<ComputeThermo>(new ComputeThermo(sysdef_1,group_all_1));

    std::shared_ptr<TwoStepNVTMTK> two_step_nvt_1 = nvt_creator(sysdef_1, group_all_1, thermo_nvt, tau, T_ref);
;
    nvt_1->addIntegrationMethod(two_step_nvt_1);
    nvt_1->addForceCompute(fc_1);

    unsigned int ndof = nvt_1->getNDOF(group_all_1);
    thermo_nvt->setNDOF(ndof);
    thermo_1->setNDOF(ndof);

    nvt_1->prepRun(0);

    PDataFlags flags;
    flags[pdata_flag::potential_energy] = 1;
    pdata_1->setFlags(flags);

    // equilibrate
    std::cout << "Equilibrating for 10,000 time steps..." << std::endl;
    int i =0;

    for (i=0; i< 10000; i++)
        {
        // get conserved quantity
        nvt_1->update(i);
        if (i % 1000 == 0)
            std::cout << i << std::endl;
        }

    // 0.1 % tolerance for temperature
    Scalar T_tol = .1;
    // 0.02 % tolerance for conserved quantity
    Scalar H_tol = 0.02;

    // conserved quantity
    thermo_1->compute(i+1);
    Scalar H_ini = thermo_1->getKineticEnergy() + thermo_1->getPotentialEnergy();
    bool flag = false;
    H_ini += nvt_1->getLogValue("nvt_mtk_reservoir_energy", flag);

    std::cout << "Measuring temperature and conserved quantity for another 25,000 time steps..." << std::endl;
    Scalar avg_T(0.0);
    int n_measure_steps = 25000;
    for (i=10000; i< 10000+n_measure_steps; i++)
        {
        // get conserved quantity
        nvt_1->update(i);

        if (i % 1000 == 0)
            std::cout << i << std::endl;

        thermo_1->compute(i+1);

        avg_T += thermo_1->getTemperature();

        Scalar H = thermo_1->getKineticEnergy() + thermo_1->getPotentialEnergy();
        H += nvt_1->getLogValue("nvt_mtk_reservoir_energy", flag);
        MY_CHECK_CLOSE(H_ini,H, H_tol);
        }

    avg_T /= n_measure_steps;
    MY_CHECK_CLOSE(T_ref, avg_T, T_tol);
    }

void test_nvt_mtk_integrator_aniso(std::shared_ptr<ExecutionConfiguration> exec_conf, twostepnvt_creator nvt_creator)
{
    SimpleCubicInitializer cubic_init(5, Scalar(1.2), "A");
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
    Scalar T_ref = Scalar(1.0);
    Scalar tau = Scalar(0.5);
    std::shared_ptr<IntegratorTwoStep> nvt_1(new IntegratorTwoStep(sysdef_1, deltaT));
    std::shared_ptr<ComputeThermo> thermo_1 = std::shared_ptr<ComputeThermo>(new ComputeThermo(sysdef_1,group_all_1));

    // ComputeThermo for integrator
    std::shared_ptr<ComputeThermo> thermo_nvt = std::shared_ptr<ComputeThermo>(new ComputeThermo(sysdef_1,group_all_1));

    std::shared_ptr<TwoStepNVTMTK> two_step_nvt_1 = nvt_creator(sysdef_1, group_all_1, thermo_nvt, tau, T_ref);
;
    nvt_1->addIntegrationMethod(two_step_nvt_1);
    nvt_1->addForceCompute(fc_1);

    unsigned int ndof = nvt_1->getNDOF(group_all_1);
    thermo_nvt->setNDOF(ndof);
    thermo_1->setNDOF(ndof);
    unsigned int ndof_rot = nvt_1->getRotationalNDOF(group_all_1);
    thermo_nvt->setRotationalNDOF(ndof_rot);
    thermo_1->setRotationalNDOF(ndof_rot);

    nvt_1->prepRun(0);

    PDataFlags flags;
    flags[pdata_flag::potential_energy] = 1;
    flags[pdata_flag::rotational_kinetic_energy] = 1;
    pdata_1->setFlags(flags);

    bool flag = false;
    // equilibrate
    std::cout << "Testing anisotropic mode" << std::endl;
    unsigned int n_equil_steps = 150000;
    std::cout << "Equilibrating for " << n_equil_steps << " time steps..." << std::endl;
    unsigned int i =0;

    for (i=0; i< n_equil_steps; i++)
        {
        nvt_1->update(i);
        if (i % 1000 == 0)
            std::cout << i << std::endl;
        }

    // 0.1 % tolerance for temperature
    Scalar T_tol = .1;
    // 0.2  % tolerance for conserved quantity
    Scalar H_tol = 0.2;

    // conserved quantity
    thermo_1->compute(i+1);
    Scalar H_ini = thermo_1->getKineticEnergy() + thermo_1->getPotentialEnergy();
    H_ini += nvt_1->getLogValue("nvt_mtk_reservoir_energy", flag);

    int n_measure_steps = 25000;
    std::cout << "Measuring temperature and conserved quantity for another " << n_measure_steps << " time steps..." << std::endl;
    Scalar avg_T(0.0);
    Scalar avg_T_trans(0.0);
    Scalar avg_T_rot(0.0);
    for (i=n_equil_steps; i< n_equil_steps+n_measure_steps; i++)
        {
        // get conserved quantity
        nvt_1->update(i);

        if (i % 1000 == 0)
            std::cout << i << std::endl;

        thermo_1->compute(i+1);

        avg_T += thermo_1->getTemperature();
        avg_T_trans += thermo_1->getTranslationalTemperature();
        avg_T_rot += thermo_1->getRotationalTemperature();

        Scalar H = thermo_1->getKineticEnergy() + thermo_1->getPotentialEnergy();
        H += nvt_1->getLogValue("nvt_mtk_reservoir_energy", flag);
        MY_CHECK_CLOSE(H_ini,H, H_tol);
        }

    avg_T /= n_measure_steps;
    avg_T_trans /= n_measure_steps;
    avg_T_rot /= n_measure_steps;
    MY_CHECK_CLOSE(T_ref, avg_T, T_tol);
    MY_CHECK_CLOSE(T_ref, avg_T_trans, T_tol);
    MY_CHECK_CLOSE(T_ref, avg_T_rot, T_tol);
    }


//! Compares the output from one NVTUpdater to another
void nvt_updater_compare_test(twostepnvt_creator nvt_creator1, twostepnvt_creator nvt_creator2, std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    const unsigned int N = 1000;

    // create two identical random particle systems to simulate
    RandomInitializer rand_init(N, Scalar(0.2), Scalar(0.9), "A");
    std::shared_ptr< SnapshotSystemData<Scalar> > snap;
    rand_init.setSeed(12345);
    snap = rand_init.getSnapshot();

    std::shared_ptr<SystemDefinition> sysdef1(new SystemDefinition(snap, exec_conf));
    std::shared_ptr<ParticleData> pdata1 = sysdef1->getParticleData();
    std::shared_ptr<ParticleSelector> selector_all1(new ParticleSelectorTag(sysdef1, 0, pdata1->getN()-1));
    std::shared_ptr<ParticleGroup> group_all1(new ParticleGroup(sysdef1, selector_all1));

    std::shared_ptr<SystemDefinition> sysdef2(new SystemDefinition(snap, exec_conf));
    std::shared_ptr<ParticleData> pdata2 = sysdef2->getParticleData();
    std::shared_ptr<ParticleSelector> selector_all2(new ParticleSelectorTag(sysdef2, 0, pdata2->getN()-1));
    std::shared_ptr<ParticleGroup> group_all2(new ParticleGroup(sysdef2, selector_all2));

    std::shared_ptr<NeighborListTree> nlist1(new NeighborListTree(sysdef1, Scalar(3.0), Scalar(0.8)));
    nlist1->setRCutPair(0,0,3.0);
    std::shared_ptr<NeighborListTree> nlist2(new NeighborListTree(sysdef2, Scalar(3.0), Scalar(0.8)));
    nlist2->setRCutPair(0,0,3.0);

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

    std::shared_ptr<IntegratorTwoStep> nvt1(new IntegratorTwoStep(sysdef1, Scalar(0.002)));
    std::shared_ptr<ComputeThermo> thermo1(new ComputeThermo(sysdef1, group_all1));
    thermo1->setNDOF(3*N-3);
    std::shared_ptr<TwoStepNVTMTK> two_step_nvt1 = nvt_creator1(sysdef1, group_all1, thermo1, Scalar(0.5), Scalar(1.2));
    nvt1->addIntegrationMethod(two_step_nvt1);

    std::shared_ptr<IntegratorTwoStep> nvt2(new IntegratorTwoStep(sysdef2, Scalar(0.002)));
    std::shared_ptr<ComputeThermo> thermo2(new ComputeThermo(sysdef2, group_all2));
    thermo2->setNDOF(3*N-3);
    std::shared_ptr<TwoStepNVTMTK> two_step_nvt2 = nvt_creator2(sysdef2, group_all2, thermo2, Scalar(0.5), Scalar(1.2));
    nvt2->addIntegrationMethod(two_step_nvt2);

    nvt1->addForceCompute(fc1);
    nvt2->addForceCompute(fc2);

    nvt1->prepRun(0);
    nvt2->prepRun(0);

    // step for 3 time steps and verify that they are the same
    // we can't do much more because these things are chaotic and diverge quickly
    for (int i = 0; i < 5; i++)
        {
        {
        ArrayHandle<Scalar4> h_pos1(pdata1->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_vel1(pdata1->getVelocities(), access_location::host, access_mode::read);
        ArrayHandle<Scalar3> h_accel1(pdata1->getAccelerations(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_pos2(pdata2->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_vel2(pdata2->getVelocities(), access_location::host, access_mode::read);
        ArrayHandle<Scalar3> h_accel2(pdata2->getAccelerations(), access_location::host, access_mode::read);

        //cout << arrays1.x[100] << " " << arrays2.x[100] << endl;
        Scalar rough_tol = 2.0;

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
        nvt1->update(i);
        nvt2->update(i);
        }
    }

//! Performs a basic equilibration test of TwoStepNVTMTK
UP_TEST( TwoStepNVTMTK_basic_test )
    {
    test_nvt_mtk_integrator(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)),bind(base_class_nvt_creator, _1, _2, _3, _4, _5));
    }

//! Performs a basic equilibration test of TwoStepNVTMTK
UP_TEST( TwoStepNVTMTK_basic_aniso_test )
    {
    test_nvt_mtk_integrator_aniso(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)),bind(base_class_nvt_creator, _1, _2, _3, _4, _5));
    }

#ifdef ENABLE_CUDA
//! Performs a basic equilibration test of TwoStepNVTMTKGPU
UP_TEST( TwoStepNVTMTKGPU_basic_test )
    {
    test_nvt_mtk_integrator(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)),bind(gpu_nvt_creator, _1, _2, _3, _4, _5));
    }

//! Performs a basic equilibration test of TwoStepNVTMTKGPU
UP_TEST( TwoStepNVTMTKGPU_basic_aniso_test )
    {
    test_nvt_mtk_integrator_aniso(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)),bind(gpu_nvt_creator, _1, _2, _3, _4, _5));
    }

//! test case for comparing the GPU and CPU NVTUpdaters
UP_TEST( TwoStepNVTMTKGPU_comparison_tests)
    {
    twostepnvt_creator nvt_creator_gpu = bind(gpu_nvt_creator, _1, _2, _3, _4, _5);
    twostepnvt_creator nvt_creator = bind(base_class_nvt_creator, _1, _2, _3, _4, _5);
    nvt_updater_compare_test(nvt_creator, nvt_creator_gpu, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

#endif
