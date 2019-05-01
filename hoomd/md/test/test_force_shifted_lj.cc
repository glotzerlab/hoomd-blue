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

/*! \file fslj_force_test.cc
    \brief Implements unit tests for PotentialPairForceShiftedLJ and PotentialPairForceShiftedLJGPU and descendants
    \ingroup unit_tests
*/

#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN();




//! Typedef'd LJForceCompute factory
typedef std::function<std::shared_ptr<PotentialPairForceShiftedLJ> (std::shared_ptr<SystemDefinition> sysdef,
                                                     std::shared_ptr<NeighborList> nlist)> ljforce_creator;

//! Test the ability of the lj force compute to actually calculate forces
void fslj_force_particle_test(ljforce_creator lj_creator, std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    std::shared_ptr<SystemDefinition> sysdef_2(new SystemDefinition(2, BoxDim(1000.0), 1, 0, 0, 0, 0, exec_conf));
    std::shared_ptr<ParticleData> pdata_2 = sysdef_2->getParticleData();
    pdata_2->setFlags(~PDataFlags(0));

    pdata_2->setPosition(0, make_scalar3(0.0,0.0,0.0));
    pdata_2->setPosition(1, make_scalar3(1.3,0.0,0.0));

    std::shared_ptr<NeighborListTree> nlist_2(new NeighborListTree(sysdef_2, Scalar(1.5), Scalar(0.4)));
    std::shared_ptr<PotentialPairForceShiftedLJ> fc_no_shift = lj_creator(sysdef_2, nlist_2);
    fc_no_shift->setRcut(0, 0, Scalar(1.5));
    fc_no_shift->setShiftMode(PotentialPairForceShiftedLJ::no_shift);

    std::shared_ptr<PotentialPairForceShiftedLJ> fc_shift = lj_creator(sysdef_2, nlist_2);
    fc_shift->setRcut(0, 0, Scalar(1.5));
    fc_shift->setShiftMode(PotentialPairForceShiftedLJ::shift);

    // setup a standard epsilon and sigma
    Scalar epsilon = Scalar(1.0);
    Scalar sigma = Scalar(1.0);
    Scalar alpha = Scalar(1.0);
    Scalar lj1 = Scalar(4.0) * epsilon * pow(sigma,Scalar(12.0));
    Scalar lj2 = alpha * Scalar(4.0) * epsilon * pow(sigma,Scalar(6.0));
    fc_no_shift->setParams(0,0,make_scalar2(lj1,lj2));
    fc_shift->setParams(0,0,make_scalar2(lj1,lj2));

    fc_no_shift->compute(0);
    fc_shift->compute(0);


    {
    GlobalArray<Scalar4>& force_array_1 =  fc_no_shift->getForceArray();
    GlobalArray<Scalar>& virial_array_1 =  fc_no_shift->getVirialArray();
    ArrayHandle<Scalar4> h_force_1(force_array_1,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_1(virial_array_1,access_location::host,access_mode::read);

    MY_CHECK_CLOSE(h_force_1.data[0].x,  1.0819510987449876 , tol);
    MY_CHECK_CLOSE(h_force_1.data[0].w, -0.21270557412540803, tol);
    MY_CHECK_CLOSE(h_force_1.data[1].x, -1.0819510987449876, tol);
    MY_CHECK_CLOSE(h_force_1.data[1].w, -0.21270557412540803, tol);
    }

    {
    GlobalArray<Scalar4>& force_array_2 =  fc_shift->getForceArray();
    GlobalArray<Scalar>& virial_array_2 =  fc_shift->getVirialArray();
    ArrayHandle<Scalar4> h_force_2(force_array_2,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_2(virial_array_2,access_location::host,access_mode::read);
    MY_CHECK_CLOSE(h_force_2.data[0].x,  1.0819510987449876 , tol);
    MY_CHECK_CLOSE(h_force_2.data[0].w, -0.05253727698612069, tol);
    MY_CHECK_CLOSE(h_force_2.data[1].x, -1.0819510987449876, tol);
    MY_CHECK_CLOSE(h_force_2.data[1].w, -0.05253727698612069, tol);
    }

    }

//! LJForceCompute creator for unit tests
std::shared_ptr<PotentialPairForceShiftedLJ> base_class_lj_creator(std::shared_ptr<SystemDefinition> sysdef,
                                                  std::shared_ptr<NeighborList> nlist)
    {
    return std::shared_ptr<PotentialPairForceShiftedLJ>(new PotentialPairForceShiftedLJ(sysdef, nlist));
    }

#ifdef ENABLE_CUDA
//! LJForceComputeGPU creator for unit tests
std::shared_ptr<PotentialPairForceShiftedLJGPU> gpu_lj_creator(std::shared_ptr<SystemDefinition> sysdef,
                                          std::shared_ptr<NeighborList> nlist)
    {
    nlist->setStorageMode(NeighborList::full);
    std::shared_ptr<PotentialPairForceShiftedLJGPU> lj(new PotentialPairForceShiftedLJGPU(sysdef, nlist));
    return lj;
    }
#endif

//! test case for particle test on CPU
UP_TEST( PotentialPairForceShiftedLJ_particle )
    {
    ljforce_creator lj_creator_base = bind(base_class_lj_creator, _1, _2);
    fslj_force_particle_test(lj_creator_base, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

# ifdef ENABLE_CUDA
//! test case for particle test on GPU
UP_TEST( LJForceGPU_particle )
    {
    ljforce_creator lj_creator_gpu = bind(gpu_lj_creator, _1, _2);
    fslj_force_particle_test(lj_creator_gpu, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
#endif
