// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// this include is necessary to get MPI included before anything else to support intel MPI
#include "hoomd/ExecutionConfiguration.h"

#include <fstream>

#include <functional>

#include "hoomd/md/BondTablePotential.h"

#ifdef ENABLE_CUDA
#include "hoomd/md/BondTablePotentialGPU.h"
#endif

#include "hoomd/Initializers.h"

#include "hoomd/test/upp11_config.h"

using namespace std;
using namespace std::placeholders;

/*! \file harmonic_bond_force_test.cc
    \brief Implements unit tests for BondTablePotential and
           BondTablePotentialGPU
    \ingroup unit_tests
*/

HOOMD_UP_MAIN();

//! Typedef to make using the std::function factory easier
typedef std::function<std::shared_ptr<BondTablePotential>  (std::shared_ptr<SystemDefinition> sysdef, unsigned int width)> bondforce_creator;

//! Perform some simple functionality tests of any BondTableForceCompute
void bond_force_basic_tests(bondforce_creator bf_creator, std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    /////////////////////////////////////////////////////////
    // start with the simplest possible test: 2 particles in a huge box with only one bond type
    std::shared_ptr<SystemDefinition> sysdef_2(new SystemDefinition(2, BoxDim(1000.0), 1, 1, 0, 0, 0, exec_conf));
    std::shared_ptr<ParticleData> pdata_2 = sysdef_2->getParticleData();

    pdata_2->setPosition(0,make_scalar3(0.0,0.0,0.0));
    pdata_2->setPosition(1,make_scalar3(1.0,0.0,0.0));

    // create the bond force compute to check
    std::shared_ptr<BondTablePotential> fc_2 = bf_creator(sysdef_2,3);



    // compute the force and check the results
    fc_2->compute(0);
    GlobalArray<Scalar4>& force_array_1 =  fc_2->getForceArray();
    GlobalArray<Scalar>& virial_array_1 =  fc_2->getVirialArray();

    {
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
    sysdef_2->getBondData()->addBondedGroup(Bond(0, 0,1));

    // specify a table to interpolate
    vector<Scalar> V, F;
    V.push_back(10.0);  F.push_back(1.0);
    V.push_back(21.0);  F.push_back(6.0);
    V.push_back(5.0);   F.push_back(2.0);
    fc_2->setTable(0, V, F, 2.0, 4.0);

    // now go to rmin and check for the correct force value
    {
    ArrayHandle<Scalar4> h_pos(pdata_2->getPositions(), access_location::host, access_mode::readwrite);
    h_pos.data[1].x = Scalar(2.0);
    }

    fc_2->compute(1);

    {
    GlobalArray<Scalar4>& force_array_3 =  fc_2->getForceArray();
    GlobalArray<Scalar>& virial_array_3 =  fc_2->getVirialArray();
    unsigned int pitch = virial_array_3.getPitch();
    ArrayHandle<Scalar4> h_force_3(force_array_3,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_3(virial_array_3,access_location::host,access_mode::read);

    MY_CHECK_CLOSE(h_force_3.data[0].x, -1.0, tol);
    MY_CHECK_SMALL(h_force_3.data[0].y, tol_small);
    MY_CHECK_SMALL(h_force_3.data[0].z, tol_small);
    MY_CHECK_CLOSE(h_force_3.data[0].w, 5.0, tol);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_3.data[0*pitch+0]
                                       +h_virial_3.data[3*pitch+0]
                                       +h_virial_3.data[5*pitch+0]), (1.0 / 6.0) * 2.0, tol);

    MY_CHECK_CLOSE(h_force_3.data[1].x, 1.0, tol);
    MY_CHECK_SMALL(h_force_3.data[1].y, tol_small);
    MY_CHECK_SMALL(h_force_3.data[1].z, tol_small);
    MY_CHECK_CLOSE(h_force_3.data[1].w, 5.0, tol);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_3.data[0*pitch+1]
                                       +h_virial_3.data[3*pitch+1]
                                       +h_virial_3.data[5*pitch+1]), (1.0 / 6.0) * 2.0, tol);
    }

    // go halfway in-between two points
    {
    ArrayHandle<Scalar4> h_pos(pdata_2->getPositions(), access_location::host, access_mode::readwrite);
    h_pos.data[1].y = Scalar(3.5);
    h_pos.data[1].x = Scalar(0.0);
    }

    // check the forces
    fc_2->compute(2);

    {
    GlobalArray<Scalar4>& force_array_4 =  fc_2->getForceArray();
    GlobalArray<Scalar>& virial_array_4 =  fc_2->getVirialArray();
    unsigned int pitch = virial_array_4.getPitch();
    ArrayHandle<Scalar4> h_force_4(force_array_4,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_4(virial_array_4,access_location::host,access_mode::read);
    MY_CHECK_CLOSE(h_force_4.data[0].y, -4.0, tol);
    MY_CHECK_SMALL(h_force_4.data[0].x, tol_small);
    MY_CHECK_SMALL(h_force_4.data[0].z, tol_small);
    MY_CHECK_CLOSE(h_force_4.data[0].w, 13.0/2.0, tol);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_4.data[0*pitch+0]
                                       +h_virial_4.data[3*pitch+0]
                                       +h_virial_4.data[5*pitch+0]), (1.0 / 6.0) * 4.0 * 3.5, tol);

    MY_CHECK_CLOSE(h_force_4.data[1].y, 4.0, tol);
    MY_CHECK_SMALL(h_force_4.data[1].x, tol_small);
    MY_CHECK_SMALL(h_force_4.data[1].z, tol_small);
    MY_CHECK_CLOSE(h_force_4.data[1].w, 13.0 / 2.0, tol);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_4.data[0*pitch+1]
                                       +h_virial_4.data[3*pitch+1]
                                       +h_virial_4.data[5*pitch+1]), (1.0 / 6.0) * 4.0 * 3.5, tol);
    }
    }


//! checks to see if BondTablePotential correctly handles multiple types
void bond_force_type_test(bondforce_creator bf_creator, std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // start with the simplest possible test: 3 particles in a huge box with two bond types
    std::shared_ptr<SystemDefinition> sysdef_2(new SystemDefinition(3, BoxDim(1000.0), 1, 2, 0, 0, 0, exec_conf));
    std::shared_ptr<ParticleData> pdata_2 = sysdef_2->getParticleData();

    {
    pdata_2->setPosition(0,make_scalar3(0.0,0.0,0.0));
    pdata_2->setPosition(1,make_scalar3(1.0,0.0,0.0));
    pdata_2->setPosition(2,make_scalar3(1.0,1.0,0.0));
    }

    // create the bond force compute to check
    std::shared_ptr<BondTablePotential> fc_2 = bf_creator(sysdef_2,3);

    // specify a table to interpolate
    vector<Scalar> V, F;
    V.push_back(10.0);  F.push_back(1.0);
    V.push_back(20.0);  F.push_back(6.0);
    V.push_back(5.0);   F.push_back(2.0);
    fc_2->setTable(0, V, F, 1.0, 2.0);

    // specify a second table to interpolate
    V.clear(); F.clear();
    V.push_back(20.0);  F.push_back(2.0);
    V.push_back(40.0);  F.push_back(12.0);
    V.push_back(10.0);   F.push_back(4.0);
    fc_2->setTable(1, V, F, 0.0, 2.0);

    // add a bond
    sysdef_2->getBondData()->addBondedGroup(Bond(0, 0,1));

    // add a second bond
    sysdef_2->getBondData()->addBondedGroup(Bond(1, 1,2));

    // compute and check
    fc_2->compute(0);

    {
    GlobalArray<Scalar4>& force_array_6 =  fc_2->getForceArray();
    GlobalArray<Scalar>& virial_array_6 =  fc_2->getVirialArray();
    unsigned int pitch = virial_array_6.getPitch();
    ArrayHandle<Scalar4> h_force_6(force_array_6,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_6(virial_array_6,access_location::host,access_mode::read);

    MY_CHECK_CLOSE(h_force_6.data[0].x, -1.0, tol);
    MY_CHECK_SMALL(h_force_6.data[0].y, tol_small);
    MY_CHECK_SMALL(h_force_6.data[0].z, tol_small);
    MY_CHECK_CLOSE(h_force_6.data[0].w, 5.0, tol);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_6.data[0*pitch+0]
                                       +h_virial_6.data[3*pitch+0]
                                       +h_virial_6.data[5*pitch+0]), (1.0)*1.0/6.0, tol);

    MY_CHECK_CLOSE(h_force_6.data[1].x, 1.0, tol);
    MY_CHECK_CLOSE(h_force_6.data[1].y, -12.0, tol);
    MY_CHECK_SMALL(h_force_6.data[1].z, tol_small);
    MY_CHECK_CLOSE(h_force_6.data[1].w, 20.0 + 5.0, tol);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_6.data[0*pitch+1]
                                       +h_virial_6.data[3*pitch+1]
                                       +h_virial_6.data[5*pitch+1]), (1*1.0 + 12.0 * 1.0)*1.0/6.0, tol);

    MY_CHECK_SMALL(h_force_6.data[2].x, tol_small);
    MY_CHECK_CLOSE(h_force_6.data[2].y, 12.0, tol);
    MY_CHECK_SMALL(h_force_6.data[2].z, tol_small);
    MY_CHECK_SMALL(h_force_6.data[2].w, 120.0);
    MY_CHECK_CLOSE(Scalar(1./3.)*(h_virial_6.data[0*pitch+2]
                                       +h_virial_6.data[3*pitch+2]
                                       +h_virial_6.data[5*pitch+2]), (12*1.0)*1.0/6.0, tol);

    }
     }



//! BondTablePotential creator for bond_force_basic_tests()
std::shared_ptr<BondTablePotential> base_class_bf_creator(std::shared_ptr<SystemDefinition> sysdef, unsigned int width)
    {
    return std::shared_ptr<BondTablePotential>(new BondTablePotential(sysdef, width));
    }

#ifdef ENABLE_CUDA
//! BondTablePotential creator for bond_force_basic_tests()
std::shared_ptr<BondTablePotential> gpu_bf_creator(std::shared_ptr<SystemDefinition> sysdef, unsigned int width)
    {
    return std::shared_ptr<BondTablePotential>(new BondTablePotentialGPU(sysdef, width));
    }
#endif

//! test case for bond forces on the CPU
UP_TEST( BondTablePotential_basic )
    {
    bondforce_creator bf_creator = bind(base_class_bf_creator, _1, _2);
    bond_force_basic_tests(bf_creator, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

//! test case for bond force type on the CPU
UP_TEST( BondTablePotential_type )
    {
    bondforce_creator bf_creator = bind(base_class_bf_creator, _1, _2);
    bond_force_type_test(bf_creator, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }


#ifdef ENABLE_CUDA
//! test case for bond forces on the GPU
UP_TEST( BondTablePotentialGPU_basic )
    {
    bondforce_creator bf_creator = bind(gpu_bf_creator, _1, _2);
    bond_force_basic_tests(bf_creator, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! test case for bond force type on the GPU
UP_TEST( BondTablePotentialGPU_type )
    {
    bondforce_creator bf_creator = bind(gpu_bf_creator, _1, _2);
    bond_force_type_test(bf_creator, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

#endif
