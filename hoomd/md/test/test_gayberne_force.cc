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

/*! \file test_gayberne_force.cc
    \brief Implements unit tests for AnisoPotentialPairGB and AnisoPotentialPairGBGPU
    \ingroup unit_tests
*/

#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN();




typedef std::function<std::shared_ptr<AnisoPotentialPairGB> (std::shared_ptr<SystemDefinition> sysdef,
                                                     std::shared_ptr<NeighborList> nlist)> gbforce_creator;

//! Test the ability of the Gay Berne force compute to actually calculate forces
void gb_force_particle_test(gbforce_creator gb_creator, std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    std::shared_ptr<SystemDefinition> sysdef_2(new SystemDefinition(2, BoxDim(1000.0), 1, 0, 0, 0, 0, exec_conf));
    std::shared_ptr<ParticleData> pdata_2 = sysdef_2->getParticleData();
    pdata_2->setFlags(~PDataFlags(0));

    {
    ArrayHandle<Scalar4> h_pos(pdata_2->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_orientation(pdata_2->getOrientationArray(), access_location::host, access_mode::readwrite);
    h_pos.data[0].x = h_pos.data[0].y = h_pos.data[0].z = 0.0;
    h_pos.data[1].x = .8; h_pos.data[1].y = .45; h_pos.data[1].z = .9;
    // particle 0 pointing along z axis
    h_orientation.data[0] = make_scalar4(1,0,0,0);
    // particle 1 pointing along x axis
    quat<Scalar> q = quat<Scalar>::fromAxisAngle(vec3<Scalar>(0,1,0), M_PI/2.0);
    h_orientation.data[1] = quat_to_scalar4(q);
    }
    std::shared_ptr<NeighborList> nlist_2(new NeighborListTree(sysdef_2, Scalar(1.3), Scalar(3.0)));
    std::shared_ptr<AnisoPotentialPairGB> fc_2 = gb_creator(sysdef_2, nlist_2);
    fc_2->setRcut(0, 0, Scalar(3.0));

    pair_gb_params params;
    params.epsilon = Scalar(1.5);
    params.lperp = Scalar(0.3);
    params.lpar = Scalar(0.5);
    fc_2->setParams(0,0,params);

    // compute the forces
    fc_2->compute(0);

    {
    GlobalArray<Scalar4>& force_array_1 =  fc_2->getForceArray();
    GlobalArray<Scalar>& virial_array_1 =  fc_2->getVirialArray();
    GlobalArray<Scalar4>& torque_array_1 =  fc_2->getTorqueArray();
    unsigned int pitch = virial_array_1.getPitch();
    ArrayHandle<Scalar4> h_force_1(force_array_1,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_1(virial_array_1,access_location::host,access_mode::read);
    ArrayHandle<Scalar4> h_torque_1(torque_array_1,access_location::host,access_mode::read);
    MY_CHECK_CLOSE(h_force_1.data[0].x, 0.470778, tol);
    MY_CHECK_CLOSE(h_force_1.data[0].y, 0.402348, tol);
    MY_CHECK_CLOSE(h_force_1.data[0].z, 0.529626, tol);
    MY_CHECK_CLOSE(h_force_1.data[0].w, -0.151892/2.0, tol);
    MY_CHECK_CLOSE(h_virial_1.data[0*pitch+1], -0.188311, tol);
    MY_CHECK_CLOSE(h_virial_1.data[1*pitch+1], -0.105925, tol);
    MY_CHECK_CLOSE(h_virial_1.data[2*pitch+1], -0.21185, tol);
    MY_CHECK_CLOSE(h_virial_1.data[3*pitch+1], -0.0905282, tol);
    MY_CHECK_CLOSE(h_virial_1.data[4*pitch+1], -0.181056, tol);
    MY_CHECK_CLOSE(h_virial_1.data[5*pitch+1], -0.238332, tol);

    MY_CHECK_CLOSE(h_force_1.data[1].x, -0.470778, tol);
    MY_CHECK_CLOSE(h_force_1.data[1].y, -0.402348, tol);
    MY_CHECK_CLOSE(h_force_1.data[1].z, -0.529626, tol);
    MY_CHECK_CLOSE(h_force_1.data[1].w, -0.151892/2.0, tol);
    MY_CHECK_CLOSE(h_virial_1.data[0*pitch+1], -0.188311, tol);
    MY_CHECK_CLOSE(h_virial_1.data[1*pitch+1], -0.105925, tol);
    MY_CHECK_CLOSE(h_virial_1.data[2*pitch+1], -0.21185, tol);
    MY_CHECK_CLOSE(h_virial_1.data[3*pitch+1], -0.0905282, tol);
    MY_CHECK_CLOSE(h_virial_1.data[4*pitch+1], -0.181056, tol);
    MY_CHECK_CLOSE(h_virial_1.data[5*pitch+1], -0.238332, tol);

    MY_CHECK_CLOSE(h_torque_1.data[0].x, -0.123781, tol);
    MY_CHECK_CLOSE(h_torque_1.data[0].y, 0.1165, tol);
    MY_CHECK_SMALL(h_torque_1.data[0].z, tol_small);

    MY_CHECK_SMALL(h_torque_1.data[1].x, tol_small);
    MY_CHECK_CLOSE(h_torque_1.data[1].y, -0.1165, tol);
    MY_CHECK_CLOSE(h_torque_1.data[1].z, 0.110028, tol);
    }
    }

//! LJForceCompute creator for unit tests
std::shared_ptr<AnisoPotentialPairGB> base_class_gb_creator(std::shared_ptr<SystemDefinition> sysdef,
                                                  std::shared_ptr<NeighborList> nlist)
    {
    return std::shared_ptr<AnisoPotentialPairGB>(new AnisoPotentialPairGB(sysdef, nlist));
    }

#ifdef ENABLE_CUDA
//! LJForceComputeGPU creator for unit tests
std::shared_ptr<AnisoPotentialPairGBGPU> gpu_gb_creator(std::shared_ptr<SystemDefinition> sysdef,
                                          std::shared_ptr<NeighborList> nlist)
    {
    nlist->setStorageMode(NeighborList::full);
    return std::shared_ptr<AnisoPotentialPairGBGPU>(new AnisoPotentialPairGBGPU(sysdef, nlist));
    }
#endif

//! test case for particle test on CPU
UP_TEST( AnisoPotentialPairGB_particle )
    {
    gbforce_creator gb_creator_base = bind(base_class_gb_creator, _1, _2);
    gb_force_particle_test(gb_creator_base, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

#ifdef ENABLE_CUDA
//! test case for particle test on GPU
UP_TEST( LJForceGPU_particle )
    {
    gbforce_creator gb_creator_gpu = bind(gpu_gb_creator, _1, _2);
    gb_force_particle_test(gb_creator_gpu, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
#endif
