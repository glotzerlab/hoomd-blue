// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// this include is necessary to get MPI included before anything else to support intel MPI
#include "hoomd/ExecutionConfiguration.h"

#include <functional>
#include <memory>

#include <iostream>
#include <fstream>

#include "hoomd/md/PPPMForceCompute.h"
#ifdef ENABLE_CUDA
#include "hoomd/md/PPPMForceComputeGPU.h"
#endif

#include "hoomd/md/NeighborListTree.h"
#include "hoomd/Initializers.h"

#include <math.h>

using namespace std;
using namespace std::placeholders;

/*! \file pppm_force_test.cc
    \brief Implements unit tests for PPPMForceCompute and PPPMForceComputeGPU and descendants
    \ingroup unit_tests
*/

#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN();




//! Typedef'd PPPMForceCompute factory

typedef std::function<std::shared_ptr<PPPMForceCompute> (std::shared_ptr<SystemDefinition> sysdef,
                                                      std::shared_ptr<NeighborList> nlist,
                                                      std::shared_ptr<ParticleGroup> group)> pppmforce_creator;

//! Test the ability of the lj force compute to actually calculate forces
void pppm_force_particle_test(pppmforce_creator pppm_creator, std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // this is a 2-particle of charge 1 and -1
    // due to the complexity of FFTs, the correct results are not analytically computed
    // but instead taken from a known working implementation of the PPPM method
    // The box lengths and grid points are different in each direction

    std::shared_ptr<SystemDefinition> sysdef_2(new SystemDefinition(2, BoxDim(6.0, 10.0, 14.0), 1, 0, 0, 0, 0, exec_conf));
    std::shared_ptr<ParticleData> pdata_2 = sysdef_2->getParticleData();
    pdata_2->setFlags(~PDataFlags(0));

    std::shared_ptr<NeighborListTree> nlist_2(new NeighborListTree(sysdef_2, Scalar(1.0), Scalar(1.0)));
    std::shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef_2, 0, 1));
    std::shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef_2, selector_all));

    {
    ArrayHandle<Scalar4> h_pos(pdata_2->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_charge(pdata_2->getCharges(), access_location::host, access_mode::readwrite);

    h_pos.data[0].x = h_pos.data[0].y = h_pos.data[0].z = 1.0;
    h_charge.data[0] = 1.0;
    h_pos.data[1].x = h_pos.data[1].y = h_pos.data[1].z = 2.0;
    h_charge.data[1] = -1.0;

    }

    std::shared_ptr<PPPMForceCompute> fc_2 = pppm_creator(sysdef_2, nlist_2, group_all);


    // first test: setup a sigma of 1.0 so that all forces will be 0
    int Nx = 10;
    int Ny = 15;
    int Nz = 24;

    int order = 5;
    Scalar kappa = 1.0;
    Scalar rcut = 1.0;
    Scalar volume = 6.0*10.0*14.0;
    fc_2->setParams(Nx, Ny, Nz, order, kappa, rcut);

    // compute the forces
    fc_2->compute(0);

    ArrayHandle<Scalar4> h_force(fc_2->getForceArray(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_virial(fc_2->getVirialArray(), access_location::host, access_mode::read);
    unsigned int pitch = fc_2->getVirialArray().getPitch();

    MY_CHECK_CLOSE(h_force.data[0].x, 0.151335f, tol_small);
    MY_CHECK_CLOSE(h_force.data[0].y, 0.172246f, tol_small);
    MY_CHECK_CLOSE(h_force.data[0].z, 0.179186f, tol_small);
    MY_CHECK_SMALL(h_force.data[0].w, tol_small);
    MY_CHECK_CLOSE(fc_2->getExternalEnergy(), -0.576491f, tol_small);
    MY_CHECK_SMALL(h_virial.data[0*pitch+0]
                        +h_virial.data[3*pitch+0]
                        +h_virial.data[5*pitch+0], tol_small);

    MY_CHECK_CLOSE(fc_2->getExternalVirial(0)/volume, -0.000180413f, tol_small);
    MY_CHECK_CLOSE(fc_2->getExternalVirial(1)/volume, -0.000180153f, tol_small);
    MY_CHECK_CLOSE(fc_2->getExternalVirial(2)/volume, -0.000180394f, tol_small);
    MY_CHECK_CLOSE(fc_2->getExternalVirial(3)/volume, -0.000211184f, tol_small);
    MY_CHECK_CLOSE(fc_2->getExternalVirial(4)/volume, -0.000204873f, tol_small);
    MY_CHECK_CLOSE(fc_2->getExternalVirial(5)/volume, -0.000219209f, tol_small);

    MY_CHECK_CLOSE(h_force.data[1].x, -0.151335f, tol_small);
    MY_CHECK_CLOSE(h_force.data[1].y, -0.172246f, tol_small);
    MY_CHECK_CLOSE(h_force.data[1].z, -0.179186f, tol_small);
    MY_CHECK_SMALL(h_force.data[1].w, tol_small);
    MY_CHECK_SMALL(h_virial.data[0*pitch+1]
                        +h_virial.data[3*pitch+1]
                        +h_virial.data[5*pitch+1], tol_small);

    }

//! Test the ability of the lj force compute to actually calculate forces
void pppm_force_particle_test_triclinic(pppmforce_creator pppm_creator, std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // this is a 2-particle of charge 1 and -1
    // due to the complexity of FFTs, the correct results are not analytically computed
    // but instead taken from a known working implementation of the PPPM method (LAMMPS ewald/disp
    // with lj/long/coul/long at RMS error = 6.14724e-06)
    // The box lengths and grid points are different in each direction

    // set up triclinic box
    Scalar tilt(0.5);
    std::shared_ptr<SystemDefinition> sysdef_2(new SystemDefinition(2, BoxDim(10.0,tilt,tilt,tilt), 1, 0, 0, 0, 0, exec_conf));
    std::shared_ptr<ParticleData> pdata_2 = sysdef_2->getParticleData();
    pdata_2->setFlags(~PDataFlags(0));

    std::shared_ptr<NeighborListTree> nlist_2(new NeighborListTree(sysdef_2, Scalar(1.0), Scalar(1.0)));
    std::shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef_2, 0, 1));
    std::shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef_2, selector_all));

    {
    ArrayHandle<Scalar4> h_pos(pdata_2->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_charge(pdata_2->getCharges(), access_location::host, access_mode::readwrite);

    h_pos.data[0].x = h_pos.data[0].y = h_pos.data[0].z = 0.0;
    h_charge.data[0] = 1.0;
    h_pos.data[1].x = 3.0; h_pos.data[1].y = 3.0; h_pos.data[1].z = 3.0;
    h_charge.data[1] = -1.0;

    }

    std::shared_ptr<PPPMForceCompute> fc_2 = pppm_creator(sysdef_2, nlist_2, group_all);


    int Nx = 128;
    int Ny = 128;
    int Nz = 128;
    int order = 3;
    Scalar kappa = 1.519768; // this value is calculated by charge.pppm
    Scalar rcut = 2.0;
    Scalar volume = 10.0*10.0*10.0;
    fc_2->setParams(Nx, Ny, Nz, order, kappa, rcut);

    // compute the forces
    fc_2->compute(0);

    ArrayHandle<Scalar4> h_force(fc_2->getForceArray(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_virial(fc_2->getVirialArray(), access_location::host, access_mode::read);
    unsigned int pitch = fc_2->getVirialArray().getPitch();

    Scalar rough_tol = 0.02;
    Scalar rough_tol_2 = 10.0;
    MY_CHECK_CLOSE(h_force.data[0].x, 0.00904953, rough_tol);
    MY_CHECK_CLOSE(h_force.data[0].y, 0.0101797, rough_tol);
    MY_CHECK_CLOSE(h_force.data[0].z, 0.0124804, rough_tol);
    MY_CHECK_SMALL(h_force.data[0].w, rough_tol);
    MY_CHECK_SMALL(h_virial.data[0*pitch+0], rough_tol);
    MY_CHECK_SMALL(h_virial.data[1*pitch+0], rough_tol);
    MY_CHECK_SMALL(h_virial.data[2*pitch+0], rough_tol);
    MY_CHECK_SMALL(h_virial.data[3*pitch+0], rough_tol);
    MY_CHECK_SMALL(h_virial.data[4*pitch+0], rough_tol);
    MY_CHECK_SMALL(h_virial.data[5*pitch+0], rough_tol);

    MY_CHECK_CLOSE(fc_2->getExternalEnergy(), -0.2441, rough_tol);
    MY_CHECK_CLOSE(fc_2->getExternalVirial(0)/volume, -5.7313404e-05, rough_tol_2);
    MY_CHECK_CLOSE(fc_2->getExternalVirial(1)/volume, -4.5494677e-05, rough_tol_2);
    MY_CHECK_CLOSE(fc_2->getExternalVirial(2)/volume, -3.9889249e-05, rough_tol_2);
    MY_CHECK_CLOSE(fc_2->getExternalVirial(3)/volume, -7.8745142e-05, rough_tol_2);
    MY_CHECK_CLOSE(fc_2->getExternalVirial(4)/volume, -4.8501155e-05, rough_tol_2);
    MY_CHECK_CLOSE(fc_2->getExternalVirial(5)/volume, -0.00010732774, rough_tol_2);

    MY_CHECK_CLOSE(h_force.data[1].x, -0.00904953, rough_tol);
    MY_CHECK_CLOSE(h_force.data[1].y, -0.0101797, rough_tol);
    MY_CHECK_CLOSE(h_force.data[1].z, -0.0124804, rough_tol);
    MY_CHECK_SMALL(h_force.data[1].w, rough_tol);
    MY_CHECK_SMALL(h_virial.data[0*pitch+1], rough_tol);
    MY_CHECK_SMALL(h_virial.data[1*pitch+1], rough_tol);
    MY_CHECK_SMALL(h_virial.data[2*pitch+1], rough_tol);
    MY_CHECK_SMALL(h_virial.data[3*pitch+1], rough_tol);
    MY_CHECK_SMALL(h_virial.data[4*pitch+1], rough_tol);
    MY_CHECK_SMALL(h_virial.data[5*pitch+1], rough_tol);
    }


//! PPPMForceCompute creator for unit tests
std::shared_ptr<PPPMForceCompute> base_class_pppm_creator(std::shared_ptr<SystemDefinition> sysdef,
                                                     std::shared_ptr<NeighborList> nlist,
                                                     std::shared_ptr<ParticleGroup> group)
    {
    return std::shared_ptr<PPPMForceCompute>(new PPPMForceCompute(sysdef, nlist, group));
    }

#ifdef ENABLE_CUDA
//! PPPMForceComputeGPU creator for unit tests
std::shared_ptr<PPPMForceCompute> gpu_pppm_creator(std::shared_ptr<SystemDefinition> sysdef,
                                              std::shared_ptr<NeighborList> nlist,
                                              std::shared_ptr<ParticleGroup> group)
    {
    nlist->setStorageMode(NeighborList::full);
    return std::shared_ptr<PPPMForceComputeGPU> (new PPPMForceComputeGPU(sysdef, nlist, group));
    }
#endif


//! test case for particle test on CPU
UP_TEST( PPPMForceCompute_basic )
    {
    pppmforce_creator pppm_creator = bind(base_class_pppm_creator, _1, _2, _3);
    pppm_force_particle_test(pppm_creator, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

//! test case for particle test on CPU
UP_TEST( PPPMForceCompute_triclinic )
    {
    pppmforce_creator pppm_creator = bind(base_class_pppm_creator, _1, _2, _3);
    pppm_force_particle_test_triclinic(pppm_creator, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }


#ifdef ENABLE_CUDA
//! test case for bond forces on the GPU
UP_TEST( PPPMForceComputeGPU_basic )
    {
    pppmforce_creator pppm_creator = bind(gpu_pppm_creator, _1, _2, _3);
    pppm_force_particle_test(pppm_creator, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

UP_TEST( PPPMForceComputeGPU_triclinic )
    {
    pppmforce_creator pppm_creator = bind(gpu_pppm_creator, _1, _2, _3);
    pppm_force_particle_test_triclinic(pppm_creator, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

#endif
