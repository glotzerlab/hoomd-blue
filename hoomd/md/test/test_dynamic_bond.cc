// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include <iostream>
#include <memory>

#include "hoomd/ExecutionConfiguration.h"
#include "hoomd/md/DynamicBond.h"

#include "hoomd/Initializers.h"
#include "hoomd/SnapshotSystemData.h"
#include "hoomd/md/NeighborListTree.h"
#include <math.h>
#include <functional>
using namespace std;
using namespace std::placeholders;

#include "hoomd/test/upp11_config.h"
HOOMD_UP_MAIN();


/*! \file test_dynamic_bond.cc
    \brief Unit tests for the DynamicBond class
    \ingroup unit_tests
 */


//! Typedef to make using the std::function factory easier
typedef std::function<std::shared_ptr<DynamicBond>  (std::shared_ptr<SystemDefinition> sysdef,
                                                     std::shared_ptr<ParticleGroup> group,
                                                     std::shared_ptr<NeighborList> nlist,
                                                     int seed,
                                                     Scalar delta_t,
                                                     int period
                                                     )> dybond_creator;

void dynamic_bond_initialization_test(dybond_creator db_creator, std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // start with the simplest possible test: 2 particles in a box with only one bond type

    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(2, BoxDim(100.0), 1, 1, 0, 0, 0,  exec_conf));
    std::shared_ptr<ParticleData> pdata = sysdef->getParticleData();


    pdata->setFlags(~PDataFlags(0));
    pdata->setPosition(0,make_scalar3(0.0,0.0,0.0));
    pdata->setPosition(1,make_scalar3(0.9,0.0,0.0));

    Scalar maxdiam = pdata->getMaxDiameter();
    Scalar r_cut = Scalar(1.3);
    Scalar r_alpha = maxdiam/2 - 0.5;
    Scalar r_cut_wc = r_cut + 2 * r_alpha;

    ArrayHandle<Scalar4> h_pos(pdata_3->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_diameter(pdata_3->getDiameters(), access_location::host, access_mode::readwrite);

    std::shared_ptr<NeighborListTree> nlist(new NeighborListTree(sysdef, r_cut_wc, Scalar(3.0)));
    std::shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef, 0, pdata->getN()-1));
    std::shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_all));

    std::shared_ptr<DynamicBond> dybond = db_creator(sysdef, group_all, nlist, 0, 0.05, 1);


    // dybond->setParams(30.0, 10.0, "fene", 6.0, 10, 60);
    }
