// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "hoomd/ExecutionConfiguration.h"
#include "hoomd/md/DynamicBond.h"

#include "hoomd/Initializers.h"
#include "hoomd/SnapshotSystemData.h"
#include "hoomd/md/NeighborListTree.h"

#include <math.h>
#include <iostream>

#include <memory>
#include <functional>

using namespace std;
using namespace std::placeholders;

/*! \file test_dynamic_bond.cc
    \brief Unit tests for the DynamicBond class
    \ingroup unit_tests
 */

#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN();

//! Typedef to make using the std::function factory easier
typedef std::function<std::shared_ptr<DynamicBond>  (std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<ParticleGroup> group, std::shared_ptr<NeighborList> nlist, int seed, Scalar delta_t, int period, unsigned int table_width)> dybond_creator;


//! Test bond creation
void dynamic_bond_create_destroy_test(dybond_creator db_creator, std::shared_ptr<ExecutionConfiguration> exec_conf)
{
    // start with the simplest possible test: 2 particles in a box with only one bond type
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(2, BoxDim(100.0), 1, 1, 0, 0, 0, exec_conf));
    std::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    pdata->setFlags(~PDataFlags(0));
    pdata->setPosition(0, make_scalar3(0.0, 0.0, 0.0));
    pdata->setPosition(1, make_scalar3(1.2, 0.0, 0.0));

    // look at least 2.0 between adjacent particles
    // radius=0.5 each, interparticle distance=1.2, so this should be plenty large)
    Scalar r_cut = Scalar(2.0);
    Scalar r_true = Scalar(0.5);
    Scalar max_diam = Scalar(1.0);

    ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_diameter(pdata->getDiameters(), access_location::host, access_mode::readwrite);

    std::shared_ptr<NeighborListTree> nlist(new NeighborListTree(sysdef, max_diam, r_cut));
    std::shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef, 0, pdata->getN() - 1));
    std::shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_all));

    std::shared_ptr<DynamicBond> dybond = db_creator(sysdef, group_all, nlist, 0, 1, 1, 3);
    dybond->setParams(r_cut, r_true, "harmonic", 1, 1, 1);

    // specify a table to interpolate
    vector<Scalar> XB, M, L;
    XB.push_back(0.0);  M.push_back(0.0); L.push_back(0.0);
    XB.push_back(0.0);  M.push_back(0.0); L.push_back(0.0);
    XB.push_back(0.0);   M.push_back(0.0); L.push_back(0.0);
    dybond->setTable(XB, M, L, 0.0, 4.0);

    std::shared_ptr<BondData> bdata(sysdef->getBondData());
    // Access the GPU bond table for reading
    ArrayHandle<BondData::members_t> h_gpu_bondlist(bdata->getGPUTable(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_gpu_n_bonds(bdata->getNGroupsArray(), access_location::host, access_mode::read);

    dybond->update(0);

    UP_ASSERT_EQUAL(h_gpu_n_bonds.data[0], 0);
    UP_ASSERT_EQUAL(h_gpu_n_bonds.data[1], 0);

    // dybond->setParams(r_cut, "harmonic", 0, 1);
    // dybond->update(1);

    // UP_ASSERT_EQUAL(h_gpu_n_bonds.data[0], 0);
    // UP_ASSERT_EQUAL(h_gpu_n_bonds.data[1], 0);
}

void dynamic_bond_rcut_test(dybond_creator db_creator, std::shared_ptr<ExecutionConfiguration> exec_conf)
{
    // start with the simplest possible test: 2 particles in a box with only one bond type
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(2, BoxDim(100.0), 1, 1, 0, 0, 0, exec_conf));
    std::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    pdata->setFlags(~PDataFlags(0));
    pdata->setPosition(0, make_scalar3(0.0, 0.0, 0.0));
    pdata->setPosition(1, make_scalar3(1.2, 0.0, 0.0));

    Scalar r_cut = Scalar(3.0);
    Scalar r_true = Scalar(0.5);
    Scalar max_diam = Scalar(1.0);

    ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_diameter(pdata->getDiameters(), access_location::host, access_mode::readwrite);

    std::shared_ptr<NeighborListTree> nlist(new NeighborListTree(sysdef, max_diam, r_cut));
    std::shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef, 0, pdata->getN() - 1));
    std::shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_all));

    std::shared_ptr<DynamicBond> dybond = db_creator(sysdef, group_all, nlist, 0, 1, 1, 3);
    dybond->setParams(r_cut, r_true, "harmonic", 1, 1, 1);

    std::shared_ptr<BondData> bdata(sysdef->getBondData());
    // Access the GPU bond table for reading
    ArrayHandle<BondData::members_t> h_gpu_bondlist(bdata->getGPUTable(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_gpu_n_bonds(bdata->getNGroupsArray(), access_location::host, access_mode::read);

    dybond->update(0);

    UP_ASSERT_EQUAL(h_gpu_n_bonds.data[0], 1);
    UP_ASSERT_EQUAL(h_gpu_n_bonds.data[1], 1);

    dybond->setParams(r_cut, r_true, "harmonic", 1, 1, 1);
    dybond->update(1);

    UP_ASSERT_EQUAL(h_gpu_n_bonds.data[0], 0);
    UP_ASSERT_EQUAL(h_gpu_n_bonds.data[1], 0);
}

//! DynamicBond creator for unit tests
std::shared_ptr<DynamicBond> base_class_db_creator(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<ParticleGroup> group, std::shared_ptr<NeighborList> nlist, int seed, Scalar delta_t, int period, unsigned int table_width)
{
    return std::shared_ptr<DynamicBond>(new DynamicBond(sysdef, group, nlist, seed, delta_t, period, table_width));
}

UP_TEST(dybond_create_destroy)
{
    dybond_creator db_creator_base = bind(base_class_db_creator, _1, _2, _3, _4, _5, _6, _7);
    dynamic_bond_create_destroy_test(db_creator_base, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
}

// UP_TEST(dybond_rcut)
// {
//     dybond_creator db_creator_base = bind(base_class_db_creator, _1, _2, _3, _4, _5, _6, _7);
//     dynamic_bond_rcut_test(db_creator_base, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
// }