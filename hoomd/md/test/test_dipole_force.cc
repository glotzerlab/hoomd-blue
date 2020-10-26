// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// this include is necessary to get MPI included before anything else to support intel MPI
#include "hoomd/ExecutionConfiguration.h"

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <iostream>
#include <fstream>

#include <functional>
#include <memory>

#include "hoomd/md/AllAnisoPairPotentials.h"

#include "hoomd/md/NeighborListTree.h"
#include "hoomd/Initializers.h"

#include <math.h>

using namespace std;
using namespace std::placeholders;

/*! \file test_dipole_force.cc
    \brief Implements unit tests for AnisoPotentialPairDipole and AnisoPotentialPairDipoleGPU
    \ingroup unit_tests
*/

#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN();




typedef std::function<std::shared_ptr<AnisoPotentialPairDipole> (std::shared_ptr<SystemDefinition> sysdef,
                                                     std::shared_ptr<NeighborList> nlist)> dipoleforce_creator;

//! Test the ability of the Gay Berne force compute to actually calculate forces
void dipole_force_particle_test(dipoleforce_creator dipole_creator, std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    std::shared_ptr<SystemDefinition> sysdef_2(new SystemDefinition(2, BoxDim(1000.0), 1, 0, 0, 0, 0, exec_conf));
    std::shared_ptr<ParticleData> pdata_2 = sysdef_2->getParticleData();
    pdata_2->setFlags(~PDataFlags(0));

    {
    ArrayHandle<Scalar4> h_pos(pdata_2->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_charge(pdata_2->getCharges(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_orientation(pdata_2->getOrientationArray(), access_location::host, access_mode::readwrite);

    h_pos.data[0].x = h_pos.data[0].y = h_pos.data[0].z = 0.0;
    h_charge.data[0] = 2.0;
    h_pos.data[1].x = .8; h_pos.data[1].y = .45; h_pos.data[1].z = .9;
    h_charge.data[1] = 1.0;
    h_pos.data[0].w = h_pos.data[1].w = __int_as_scalar(0);

    // default orientation has dipole in (1, 0, 0) direction
    h_orientation.data[0] = make_scalar4(1,0,0,0);
    // rotate particle 1 by 2*pi/3 about the (1, 1, 0) axis
    h_orientation.data[1] = make_scalar4(cos(2*M_PI/6), sin(2*M_PI/6)/sqrt(2), sin(2*M_PI/6)/sqrt(2), 0);
    }
    std::shared_ptr<NeighborList> nlist_2(new NeighborListTree(sysdef_2, Scalar(6.0), Scalar(6.5)));
    std::shared_ptr<AnisoPotentialPairDipole> fc_2 = dipole_creator(sysdef_2, nlist_2);
    fc_2->setRcut(0, 0, Scalar(6.0));

    // Compare with lammps dipole potential, which fixes A=1 and kappa=0
    pair_dipole_params params;
    params.mu = 0.6;
    params.A = 1;
    params.kappa = 0;
    fc_2->setParams(0, 0, params);

    // compute the forces
    fc_2->compute(0);

    {
    GlobalArray<Scalar4>& force_array_1 =  fc_2->getForceArray();
    GlobalArray<Scalar>& virial_array_1 =  fc_2->getVirialArray();
    GlobalArray<Scalar4>& torque_array_1 =  fc_2->getTorqueArray();
    ArrayHandle<Scalar4> h_force_1(force_array_1,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_1(virial_array_1,access_location::host,access_mode::read);
    ArrayHandle<Scalar4> h_torque_1(torque_array_1,access_location::host,access_mode::read);
    MY_CHECK_CLOSE(h_force_1.data[0].x, -1.07832, tol);
    MY_CHECK_CLOSE(h_force_1.data[0].y, -1.26201, tol);
    MY_CHECK_CLOSE(h_force_1.data[0].z, -0.810835, tol);
    MY_CHECK_CLOSE(h_force_1.data[0].w, 0.917602, tol);
    MY_CHECK_CLOSE(h_torque_1.data[0].x, 0, tol);
    MY_CHECK_CLOSE(h_torque_1.data[0].y, 0.154201, tol);
    MY_CHECK_CLOSE(h_torque_1.data[0].z, -0.256091, tol);

    MY_CHECK_CLOSE(h_force_1.data[1].x, 1.07832, tol);
    MY_CHECK_CLOSE(h_force_1.data[1].y, 1.26201, tol);
    MY_CHECK_CLOSE(h_force_1.data[1].z, 0.810835, tol);
    MY_CHECK_CLOSE(h_force_1.data[1].w, 0.917602, tol);
    MY_CHECK_CLOSE(h_torque_1.data[1].x, 0.770933, tol);
    MY_CHECK_CLOSE(h_torque_1.data[1].y, -0.476021, tol);
    MY_CHECK_CLOSE(h_torque_1.data[1].z, -0.268273, tol);
    }
    }

//! LJForceCompute creator for unit tests
std::shared_ptr<AnisoPotentialPairDipole> base_class_dipole_creator(std::shared_ptr<SystemDefinition> sysdef,
                                                  std::shared_ptr<NeighborList> nlist)
    {
    return std::shared_ptr<AnisoPotentialPairDipole>(new AnisoPotentialPairDipole(sysdef, nlist));
    }

#ifdef ENABLE_CUDA
//! LJForceComputeGPU creator for unit tests
std::shared_ptr<AnisoPotentialPairDipoleGPU> gpu_dipole_creator(std::shared_ptr<SystemDefinition> sysdef,
                                          std::shared_ptr<NeighborList> nlist)
    {
    nlist->setStorageMode(NeighborList::full);
    return std::shared_ptr<AnisoPotentialPairDipoleGPU>(new AnisoPotentialPairDipoleGPU(sysdef, nlist));
    }
#endif

//! test case for particle test on CPU
UP_TEST( AnisoPotentialPairDipole_particle )
    {
    dipoleforce_creator dipole_creator_base = bind(base_class_dipole_creator, _1, _2);
    dipole_force_particle_test(dipole_creator_base, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

#ifdef ENABLE_CUDA
//! test case for particle test on GPU
UP_TEST( AnisoPotentialPairDipoleGPU_particle )
    {
    dipoleforce_creator dipole_creator_gpu = bind(gpu_dipole_creator, _1, _2);
    dipole_force_particle_test(dipole_creator_gpu, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
#endif
