// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// this include is necessary to get MPI included before anything else to support intel MPI
#include "hoomd/ExecutionConfiguration.h"

#include <iostream>
#include <algorithm>

#include <memory>

#include "hoomd/md/NeighborList.h"
#include "hoomd/md/NeighborListBinned.h"
#include "hoomd/md/NeighborListStencil.h"
#include "hoomd/md/NeighborListTree.h"
#include "hoomd/Initializers.h"

#ifdef ENABLE_CUDA
#include "hoomd/md/NeighborListGPU.h"
#include "hoomd/md/NeighborListGPUBinned.h"
#include "hoomd/md/NeighborListGPUStencil.h"
#include "hoomd/md/NeighborListGPUTree.h"
#endif

using namespace std;

#include "hoomd/test/upp11_config.h"
HOOMD_UP_MAIN();

//! Performs basic functionality tests on a neighbor list
template <class NL>
void neighborlist_basic_tests(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    /////////////////////////////////////////////////////////
    // start with the simplest possible test: 2 particles in a huge box
    std::shared_ptr<SystemDefinition> sysdef_2(new SystemDefinition(2, BoxDim(25.0), 1, 0, 0, 0, 0, exec_conf));
    std::shared_ptr<ParticleData> pdata_2 = sysdef_2->getParticleData();

    {
    ArrayHandle<Scalar4> h_pos(pdata_2->getPositions(), access_location::host, access_mode::readwrite);

    h_pos.data[0].x = h_pos.data[0].y = h_pos.data[0].z = 0.0;
    h_pos.data[1].x = h_pos.data[1].y = h_pos.data[1].z = 3.25;

    h_pos.data[0].w = 0.0; h_pos.data[1].w = 0.0;
    pdata_2->notifyParticleSort();
    }

    // test construction of the neighborlist
    std::shared_ptr<NeighborList> nlist_2(new NL(sysdef_2, 3.0, 0.25));
    nlist_2->setRCutPair(0,0,3.0);
    nlist_2->compute(1);

    // with the given radius, there should be no neighbors: check that
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist_2->getNNeighArray(), access_location::host, access_mode::read);

        CHECK_EQUAL_UINT(h_n_neigh.data[0], 0);
        CHECK_EQUAL_UINT(h_n_neigh.data[1], 0);
        }

    // adjust the radius to include the particles and see if we get some now
    nlist_2->setRCutPair(0,0,5.5);
    nlist_2->compute(2);
    // some neighbor lists default to full because they don't support half: ignore them
    if (nlist_2->getStorageMode() == NeighborList::half)
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist_2->getNNeighArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_nlist(nlist_2->getNListArray(), access_location::host, access_mode::read);

        CHECK_EQUAL_UINT(h_n_neigh.data[0], 1);
        CHECK_EQUAL_UINT(h_nlist.data[0], 1);
        // since this is a half list, only 0 stores 1 as a neighbor
        CHECK_EQUAL_UINT(h_n_neigh.data[1], 0);
        }

    // change to full mode to check that
    nlist_2->setStorageMode(NeighborList::full);
    nlist_2->setRCutPair(0,0,5.5);
    nlist_2->setRBuff(0.5);
    nlist_2->compute(3);
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist_2->getNNeighArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_nlist(nlist_2->getNListArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_head_list(nlist_2->getHeadList(), access_location::host, access_mode::read);

        CHECK_EQUAL_UINT(h_n_neigh.data[0], 1);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[0] + 0], 1);

        CHECK_EQUAL_UINT(h_n_neigh.data[1], 1);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[1] + 0], 0);
        }


    ////////////////////////////////////////////////////////////////////
    // now, lets do a more thorough test and include boundary conditions
    // there are way too many permutations to test here, so I will simply
    // test +x, -x, +y, -y, +z, and -z independently
    // build a 6 particle system with particles across each boundary

    std::shared_ptr<SystemDefinition> sysdef_6(new SystemDefinition(6, BoxDim(20.0, 40.0, 60.0), 1, 0, 0, 0, 0, exec_conf));
    std::shared_ptr<ParticleData> pdata_6 = sysdef_6->getParticleData();

    {
    ArrayHandle<Scalar4> h_pos(pdata_6->getPositions(), access_location::host, access_mode::readwrite);

    h_pos.data[0].x = Scalar(-9.6); h_pos.data[0].y = 0; h_pos.data[0].z = 0.0; h_pos.data[0].w = 0.0;
    h_pos.data[1].x =  Scalar(9.6); h_pos.data[1].y = 0; h_pos.data[1].z = 0.0; h_pos.data[1].w = 0.0;
    h_pos.data[2].x = 0; h_pos.data[2].y = Scalar(-19.6); h_pos.data[2].z = 0.0; h_pos.data[2].w = 0.0;
    h_pos.data[3].x = 0; h_pos.data[3].y = Scalar(19.6); h_pos.data[3].z = 0.0; h_pos.data[3].w = 0.0;
    h_pos.data[4].x = 0; h_pos.data[4].y = 0; h_pos.data[4].z = Scalar(-29.6); h_pos.data[4].w = 0.0;
    h_pos.data[5].x = 0; h_pos.data[5].y = 0; h_pos.data[5].z =  Scalar(29.6); h_pos.data[5].w = 0.0;

    pdata_6->notifyParticleSort();
    }

    std::shared_ptr<NeighborList> nlist_6(new NL(sysdef_6, 3.0, 0.25));
    nlist_6->setRCutPair(0,0,3.0);
    nlist_6->setStorageMode(NeighborList::full);
    nlist_6->compute(0);
    // verify the neighbor list
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist_6->getNNeighArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_nlist(nlist_6->getNListArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_head_list(nlist_6->getHeadList(), access_location::host, access_mode::read);

        // check for right number of neighbors
        CHECK_EQUAL_UINT(h_n_neigh.data[0], 1);
        CHECK_EQUAL_UINT(h_n_neigh.data[1], 1);
        CHECK_EQUAL_UINT(h_n_neigh.data[2], 1);
        CHECK_EQUAL_UINT(h_n_neigh.data[3], 1);
        CHECK_EQUAL_UINT(h_n_neigh.data[4], 1);
        CHECK_EQUAL_UINT(h_n_neigh.data[5], 1);

        // the answer we expect
        unsigned int check_nbrs[] = {1, 0, 3, 2, 5, 4};

        // validate that the neighbors are correct
        for (unsigned int i=0; i < 6; ++i)
            {
            UP_ASSERT_EQUAL(h_nlist.data[h_head_list.data[i]],check_nbrs[i]);
            }
        }

    // swap the order of the particles around to look for subtle directional bugs
    {
    ArrayHandle<Scalar4> h_pos(pdata_6->getPositions(), access_location::host, access_mode::readwrite);

    h_pos.data[1].x = Scalar(-9.6); h_pos.data[1].y = 0; h_pos.data[1].z = 0.0;
    h_pos.data[0].x =  Scalar(9.6); h_pos.data[0].y = 0; h_pos.data[0].z = 0.0;
    h_pos.data[3].x = 0; h_pos.data[3].y = Scalar(-19.6); h_pos.data[3].z = 0.0;
    h_pos.data[2].x = 0; h_pos.data[2].y = Scalar(19.6); h_pos.data[2].z = 0.0;
    h_pos.data[5].x = 0; h_pos.data[5].y = 0; h_pos.data[5].z = Scalar(-29.6);
    h_pos.data[4].x = 0; h_pos.data[4].y = 0; h_pos.data[4].z =  Scalar(29.6);

    pdata_6->notifyParticleSort();
    }

    // verify the neighbor list
    nlist_6->compute(1);
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist_6->getNNeighArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_nlist(nlist_6->getNListArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_head_list(nlist_6->getHeadList(), access_location::host, access_mode::read);

        // check for right number of nbrs
        CHECK_EQUAL_UINT(h_n_neigh.data[0], 1);
        CHECK_EQUAL_UINT(h_n_neigh.data[1], 1);
        CHECK_EQUAL_UINT(h_n_neigh.data[2], 1);
        CHECK_EQUAL_UINT(h_n_neigh.data[3], 1);
        CHECK_EQUAL_UINT(h_n_neigh.data[4], 1);
        CHECK_EQUAL_UINT(h_n_neigh.data[5], 1);

        // the answer we expect
        unsigned int check_nbrs[] = {1, 0, 3, 2, 5, 4};

        for (unsigned int i=0; i < 6; ++i)
            {
            UP_ASSERT_EQUAL(h_nlist.data[h_head_list.data[i]],check_nbrs[i]);
            }

        }

    // one last test, we should check that more than one neighbor can be generated
    {
    ArrayHandle<Scalar4> h_pos(pdata_6->getPositions(), access_location::host, access_mode::readwrite);

    h_pos.data[0].x = 0; h_pos.data[0].y = 0; h_pos.data[0].z = 0.0;
    h_pos.data[1].x = 0; h_pos.data[1].y = 0; h_pos.data[1].z = 0.0;
    h_pos.data[2].x = 0; h_pos.data[2].y = Scalar(-19.6); h_pos.data[2].z = 0.0;
    h_pos.data[3].x = 0; h_pos.data[3].y = Scalar(19.6); h_pos.data[3].z = 0.0;
    h_pos.data[4].x = 0; h_pos.data[4].y = 0; h_pos.data[4].z = 0;
    h_pos.data[5].x = 0; h_pos.data[5].y = 0; h_pos.data[5].z =  0;

    pdata_6->notifyParticleSort();
    }

    nlist_6->compute(20);
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist_6->getNNeighArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_nlist(nlist_6->getNListArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_head_list(nlist_6->getHeadList(), access_location::host, access_mode::read);

        CHECK_EQUAL_UINT(h_n_neigh.data[0], 3);

        vector<unsigned int> nbrs(3,0);
        for (unsigned int i=0; i < 3; ++i)
            {
            nbrs[i] = h_nlist.data[h_head_list.data[0] + i];
            }

        // sort the neighbors because it doesn't matter what order they are stored in, just that they all are there
        sort(nbrs.begin(), nbrs.end());

        // the answer we expect
        unsigned int check_nbrs[] = {1, 4, 5};

        for (unsigned int i=0; i < 3; ++i)
            {
            UP_ASSERT_EQUAL(nbrs[i],check_nbrs[i]);
            }
        }
    }

//! Test neighborlist functionality with particles with different numbers of neighbors
template <class NL>
void neighborlist_particle_asymm_tests(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    std::shared_ptr<SystemDefinition> sysdef_3(new SystemDefinition(3, BoxDim(40.0, 40.0, 60.0), 2, 0, 0, 0, 0, exec_conf));
    std::shared_ptr<ParticleData> pdata_3 = sysdef_3->getParticleData();
    // check that pair cutoffs are set independently
        {
        ArrayHandle<Scalar4> h_pos(pdata_3->getPositions(), access_location::host, access_mode::readwrite);

        h_pos.data[0].x = 0.0; h_pos.data[0].y = 0.0; h_pos.data[0].z = 0.0; h_pos.data[0].w = __int_as_scalar(1);
        h_pos.data[1].x = Scalar(1.2); h_pos.data[1].y = 0.0; h_pos.data[1].z = 0.0; h_pos.data[1].w = __int_as_scalar(0);
        h_pos.data[2].x = Scalar(3.5); h_pos.data[2].y = 0.0; h_pos.data[2].z = 0.0; h_pos.data[2].w = __int_as_scalar(1);

        pdata_3->notifyParticleSort();
        }

    std::shared_ptr<NeighborList> nlist_3(new NL(sysdef_3, 3.0, 0.25));
    nlist_3->setStorageMode(NeighborList::full);
    nlist_3->setRCutPair(0,0,1.0);
    nlist_3->setRCutPair(1,1,3.0);
    nlist_3->setRCutPair(0,1,2.0);
    nlist_3->compute(0);
    // 1 is neighbor of 0 but not of 2
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist_3->getNNeighArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_nlist(nlist_3->getNListArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_head_list(nlist_3->getHeadList(), access_location::host, access_mode::read);

        CHECK_EQUAL_UINT(h_n_neigh.data[0], 1);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[0] + 0], 1);

        CHECK_EQUAL_UINT(h_n_neigh.data[1], 1);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[1] + 0], 0);

        CHECK_EQUAL_UINT(h_n_neigh.data[2], 0);
        }

    // now change the cutoff so that 2 is neighbors with 0 but not 1
    nlist_3->setRCutPair(1,1,3.5);
    nlist_3->compute(1);
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist_3->getNNeighArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_nlist(nlist_3->getNListArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_head_list(nlist_3->getHeadList(), access_location::host, access_mode::read);

        CHECK_EQUAL_UINT(h_n_neigh.data[0], 2);
        vector<unsigned int> nbrs(2, 0);
        nbrs[0] = h_nlist.data[h_head_list.data[0] + 0];
        nbrs[1] = h_nlist.data[h_head_list.data[0] + 1];
        sort(nbrs.begin(), nbrs.end());
        unsigned int check_nbrs[] = {1,2};

        for (unsigned int i=0; i < 2; ++i)
            {
            UP_ASSERT_EQUAL(nbrs[i],check_nbrs[i]);
            }

        CHECK_EQUAL_UINT(h_n_neigh.data[1], 1);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[1] + 0], 0);

        CHECK_EQUAL_UINT(h_n_neigh.data[2], 1);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[2] + 0], 0);
        }

    // now change the cutoff so that all are neighbors
    nlist_3->setRCutPair(0,1,2.5);
    nlist_3->compute(20);
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist_3->getNNeighArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_nlist(nlist_3->getNListArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_head_list(nlist_3->getHeadList(), access_location::host, access_mode::read);

        CHECK_EQUAL_UINT(h_n_neigh.data[0], 2);

        CHECK_EQUAL_UINT(h_n_neigh.data[1], 2);

        CHECK_EQUAL_UINT(h_n_neigh.data[2], 2);
        }

    // check what happens with particle resize by first keeping number below the 8 default, and then bumping over this
    // do this with size 18 so that NeighborListGPU is forced to use kernel call with multiple levels at m_bin_size = 4
    std::shared_ptr<SystemDefinition> sysdef_18(new SystemDefinition(18, BoxDim(40.0, 40.0, 40.0), 2, 0, 0, 0, 0, exec_conf));
    std::shared_ptr<ParticleData> pdata_18 = sysdef_18->getParticleData();
        {
        ArrayHandle<Scalar4> h_pos(pdata_18->getPositions(), access_location::host, access_mode::readwrite);

        h_pos.data[0].x = 0.0; h_pos.data[0].y = 0.0; h_pos.data[0].z = 0.0; h_pos.data[0].w = __int_as_scalar(1);
        h_pos.data[1].x = 0.0; h_pos.data[1].y = 0.0; h_pos.data[1].z = 0.0; h_pos.data[1].w = __int_as_scalar(1);
        h_pos.data[2].x = 0.0; h_pos.data[2].y = 0.0; h_pos.data[2].z = 0.0; h_pos.data[2].w = __int_as_scalar(1);
        h_pos.data[3].x = Scalar(10.0); h_pos.data[3].y = 0.0; h_pos.data[3].z = 0.0; h_pos.data[3].w = __int_as_scalar(1);
        h_pos.data[4].x = Scalar(0.9); h_pos.data[4].y = 0.0; h_pos.data[4].z = 0.0; h_pos.data[4].w = __int_as_scalar(0);
        h_pos.data[5].x = Scalar(-0.9); h_pos.data[5].y = 0.0; h_pos.data[5].z = 0.0; h_pos.data[5].w = __int_as_scalar(0);
        h_pos.data[6].x = 0.0; h_pos.data[6].y = Scalar(0.9); h_pos.data[6].z = 0.0; h_pos.data[6].w = __int_as_scalar(0);
        h_pos.data[7].x = 0.0; h_pos.data[7].y = Scalar(-0.9); h_pos.data[7].z = 0.0; h_pos.data[7].w = __int_as_scalar(0);
        h_pos.data[8].x = 0.0; h_pos.data[8].y = 0.0; h_pos.data[8].z = Scalar(0.9); h_pos.data[8].w = __int_as_scalar(0);
        h_pos.data[9].x = 0.0; h_pos.data[9].y = 0.0; h_pos.data[9].z = Scalar(-0.9); h_pos.data[9].w = __int_as_scalar(0);
        h_pos.data[10].x = Scalar(0.9); h_pos.data[10].y = 0.0; h_pos.data[10].z = 0.0; h_pos.data[10].w = __int_as_scalar(0);
        h_pos.data[11].x = Scalar(-0.9); h_pos.data[11].y = 0.0; h_pos.data[11].z = 0.0; h_pos.data[11].w = __int_as_scalar(0);
        h_pos.data[12].x = 0.0; h_pos.data[12].y = Scalar(0.9); h_pos.data[12].z = 0.0; h_pos.data[12].w = __int_as_scalar(0);
        h_pos.data[13].x = 0.0; h_pos.data[13].y = Scalar(-0.9); h_pos.data[13].z = 0.0; h_pos.data[13].w = __int_as_scalar(0);
        h_pos.data[14].x = 0.0; h_pos.data[14].y = 0.0; h_pos.data[14].z = Scalar(0.9); h_pos.data[14].w = __int_as_scalar(0);
        h_pos.data[15].x = 0.0; h_pos.data[15].y = 0.0; h_pos.data[15].z = Scalar(-0.9); h_pos.data[15].w = __int_as_scalar(0);
        h_pos.data[16].x = Scalar(-10.0); h_pos.data[16].y = 0.0; h_pos.data[16].z = 0.0; h_pos.data[16].w = __int_as_scalar(1);
        h_pos.data[17].x = 0.0; h_pos.data[17].y = Scalar(10.0); h_pos.data[17].z = 0.0; h_pos.data[17].w = __int_as_scalar(1);

        pdata_18->notifyParticleSort();
        }

    std::shared_ptr<NeighborList> nlist_18(new NL(sysdef_18, 3.0, 0.05));
    nlist_18->setRCutPair(0,0,1.0);
    nlist_18->setRCutPair(1,1,1.0);
    nlist_18->setRCutPair(0,1,1.0);
    nlist_18->setStorageMode(NeighborList::full);
    nlist_18->compute(0);
    // 0-2 have 15 neighbors, 3 and 16 have no neighbors, and all others have 4 neighbors
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist_18->getNNeighArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_nlist(nlist_18->getNListArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_head_list(nlist_18->getHeadList(), access_location::host, access_mode::read);

        // 6x16 + 12x8 = 192
        UP_ASSERT(nlist_18->getNListArray().getPitch() >= 192);
        CHECK_EQUAL_UINT(h_head_list.data[17],176);

        for (unsigned int i=0; i < 18; ++i)
            {
            if (i < 3)
                {
                CHECK_EQUAL_UINT(h_n_neigh.data[i], 14);
                for (unsigned int j=0; j < 14; ++j)
                    {
                    // not the ones far away
                    UP_ASSERT(h_nlist.data[j] != 3 && h_nlist.data[j] != 16 && h_nlist.data[j] != 17);
                    }
                }
            else if (i == 3 || i >= 16)
                {
                CHECK_EQUAL_UINT(h_n_neigh.data[i], 0);
                }
            else
                {
                CHECK_EQUAL_UINT(h_n_neigh.data[i], 4);
                }
            }
        }

    // bring in particle 3, 16, and 17, which should force a resize on particle type 1
        {
        ArrayHandle<Scalar4> h_pos(pdata_18->getPositions(), access_location::host, access_mode::readwrite);
        h_pos.data[3].x = 0.0;
        h_pos.data[16].x = 0.0;
        h_pos.data[17].y = 0.0;

        pdata_18->notifyParticleSort();
        }

    nlist_18->compute(20);
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist_18->getNNeighArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_nlist(nlist_18->getNListArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_head_list(nlist_18->getHeadList(), access_location::host, access_mode::read);

        // 6x24 + 12x8 = 240
        UP_ASSERT(nlist_18->getNListArray().getPitch() >= 240);
        CHECK_EQUAL_UINT(h_head_list.data[17],216);

        for (unsigned int i=0; i < 18; ++i)
            {
            if (i <= 3 || i >= 16)
                {
                CHECK_EQUAL_UINT(h_n_neigh.data[i], 17);
                }
            else
                {
                CHECK_EQUAL_UINT(h_n_neigh.data[i], 7);
                }
            }
        }

    // collapse all particles onto self and force a resize
        {
        ArrayHandle<Scalar4> h_pos(pdata_18->getPositions(), access_location::host, access_mode::readwrite);
        for (unsigned int i=4; i < 16; ++i)
            {
            h_pos.data[i].x = 0.0; h_pos.data[i].y = 0.0; h_pos.data[i].z = 0.0;
            }
        pdata_18->notifyParticleSort();
        }

    nlist_18->compute(40);
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist_18->getNNeighArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_nlist(nlist_18->getNListArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_head_list(nlist_18->getHeadList(), access_location::host, access_mode::read);

        // 18x24 = 432
        UP_ASSERT(nlist_18->getNListArray().getPitch() >= 432);
        CHECK_EQUAL_UINT(h_head_list.data[17],408);

        for (unsigned int i=0; i < 18; ++i)
            {
            CHECK_EQUAL_UINT(h_n_neigh.data[i], 17);
            }
        }
    }

//! Test neighborlist functionality with changing types
template <class NL>
void neighborlist_type_tests(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    std::shared_ptr<SystemDefinition> sysdef_6(new SystemDefinition(6, BoxDim(40.0, 40.0, 40.0), 4, 0, 0, 0, 0, exec_conf));
    std::shared_ptr<ParticleData> pdata_6 = sysdef_6->getParticleData();
    // test 1: 4 types, but missing two in the middle
        {
        ArrayHandle<Scalar4> h_pos(pdata_6->getPositions(), access_location::host, access_mode::readwrite);

        for (unsigned int cur_p=0; cur_p < 6; ++cur_p)
            {
            if (cur_p < 5)
                {
                h_pos.data[cur_p] = make_scalar4(-1.0, 0.0, 0.0, __int_as_scalar(3));
                }
            else
                {
                h_pos.data[cur_p] = make_scalar4(1.0, 0.0, 0.0, __int_as_scalar(0));
                }
            }
        pdata_6->notifyParticleSort();
        }

    std::shared_ptr<NeighborList> nlist_6(new NL(sysdef_6, 3.0, 0.1));
    nlist_6->setStorageMode(NeighborList::full);
    nlist_6->compute(0);

    // everybody should neighbor everybody else
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist_6->getNNeighArray(), access_location::host, access_mode::read);
        CHECK_EQUAL_UINT(h_n_neigh.data[0], 5);
        CHECK_EQUAL_UINT(h_n_neigh.data[1], 5);
        CHECK_EQUAL_UINT(h_n_neigh.data[2], 5);
        CHECK_EQUAL_UINT(h_n_neigh.data[3], 5);
        CHECK_EQUAL_UINT(h_n_neigh.data[4], 5);
        CHECK_EQUAL_UINT(h_n_neigh.data[5], 5);

        ArrayHandle<unsigned int> h_nlist(nlist_6->getNListArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_head_list(nlist_6->getHeadList(), access_location::host, access_mode::read);
        for (unsigned int cur_p = 0; cur_p < 6; ++cur_p)
            {
            vector<unsigned int> nbrs(5,0), check_nbrs;

            // create the sorted list of computed neighbors
            for (unsigned int cur_neigh = 0; cur_neigh < 5; ++cur_neigh)
                {
                nbrs[cur_neigh] = h_nlist.data[h_head_list.data[cur_p] + cur_neigh];
                }
            sort(nbrs.begin(), nbrs.end());

            // create the list of expected neighbors (everybody except for myself)
            check_nbrs.reserve(5);
            for (unsigned int i = 0; i < 6; ++i)
                {
                if (i != cur_p)
                    {
                    check_nbrs.push_back(i);
                    }
                }
            sort(check_nbrs.begin(), check_nbrs.end());

            for (unsigned int i=0; i < 5; ++i)
                {
                UP_ASSERT_EQUAL(nbrs[i],check_nbrs[i]);
                }
            }
        }

    // add a new type
    pdata_6->addType("E");
    nlist_6->setRCut(3.0, 0.1); // update the rcut for the new type
    nlist_6->compute(10);
    // result is unchanged
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist_6->getNNeighArray(), access_location::host, access_mode::read);
        CHECK_EQUAL_UINT(h_n_neigh.data[0], 5);
        CHECK_EQUAL_UINT(h_n_neigh.data[1], 5);
        CHECK_EQUAL_UINT(h_n_neigh.data[2], 5);
        CHECK_EQUAL_UINT(h_n_neigh.data[3], 5);
        CHECK_EQUAL_UINT(h_n_neigh.data[4], 5);
        CHECK_EQUAL_UINT(h_n_neigh.data[5], 5);

        ArrayHandle<unsigned int> h_nlist(nlist_6->getNListArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_head_list(nlist_6->getHeadList(), access_location::host, access_mode::read);
        for (unsigned int cur_p = 0; cur_p < 6; ++cur_p)
            {
            vector<unsigned int> nbrs(5,0), check_nbrs;

            // create the sorted list of computed neighbors
            for (unsigned int cur_neigh = 0; cur_neigh < 5; ++cur_neigh)
                {
                nbrs[cur_neigh] = h_nlist.data[h_head_list.data[cur_p] + cur_neigh];
                }
            sort(nbrs.begin(), nbrs.end());

            // create the list of expected neighbors (everybody except for myself)
            check_nbrs.reserve(5);
            for (unsigned int i = 0; i < 6; ++i)
                {
                if (i != cur_p)
                    {
                    check_nbrs.push_back(i);
                    }
                }
            sort(check_nbrs.begin(), check_nbrs.end());

            for (unsigned int i=0; i < 5; ++i)
                {
                UP_ASSERT_EQUAL(nbrs[i],check_nbrs[i]);
                }
            }
        }

    // add two more empty types
    pdata_6->addType("F");
    pdata_6->addType("G");
    nlist_6->setRCut(3.0, 0.1); // update the rcut for the new type
    nlist_6->compute(20);
    // result is unchanged
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist_6->getNNeighArray(), access_location::host, access_mode::read);
        CHECK_EQUAL_UINT(h_n_neigh.data[0], 5);
        CHECK_EQUAL_UINT(h_n_neigh.data[1], 5);
        CHECK_EQUAL_UINT(h_n_neigh.data[2], 5);
        CHECK_EQUAL_UINT(h_n_neigh.data[3], 5);
        CHECK_EQUAL_UINT(h_n_neigh.data[4], 5);
        CHECK_EQUAL_UINT(h_n_neigh.data[5], 5);

        ArrayHandle<unsigned int> h_nlist(nlist_6->getNListArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_head_list(nlist_6->getHeadList(), access_location::host, access_mode::read);
        for (unsigned int cur_p = 0; cur_p < 6; ++cur_p)
            {
            vector<unsigned int> nbrs(5,0), check_nbrs;

            // create the sorted list of computed neighbors
            for (unsigned int cur_neigh = 0; cur_neigh < 5; ++cur_neigh)
                {
                nbrs[cur_neigh] = h_nlist.data[h_head_list.data[cur_p] + cur_neigh];
                }
            sort(nbrs.begin(), nbrs.end());

            // create the list of expected neighbors (everybody except for myself)
            check_nbrs.reserve(5);
            for (unsigned int i = 0; i < 6; ++i)
                {
                if (i != cur_p)
                    {
                    check_nbrs.push_back(i);
                    }
                }
            sort(check_nbrs.begin(), check_nbrs.end());

            for (unsigned int i=0; i < 5; ++i)
                {
                UP_ASSERT_EQUAL(nbrs[i],check_nbrs[i]);
                }
            }
        }

    pdata_6->addType("H");
    nlist_6->setRCut(3.0,0.1);
    // disable the interaction between type 6 and all other particles
    for (unsigned int cur_type = 0; cur_type < pdata_6->getNTypes(); ++cur_type)
        {
        nlist_6->setRCutPair(6, cur_type, -1.0);
        }
    // shuffle all of the particle types and retest
        {
        ArrayHandle<Scalar4> h_pos(pdata_6->getPositions(), access_location::host, access_mode::readwrite);
        h_pos.data[0].w = __int_as_scalar(2);
        h_pos.data[1].w = __int_as_scalar(4);
        h_pos.data[2].w = __int_as_scalar(0);
        h_pos.data[3].w = __int_as_scalar(1);
        h_pos.data[4].w = __int_as_scalar(7);
        h_pos.data[5].w = __int_as_scalar(6);
        pdata_6->notifyParticleSort();
        }
    nlist_6->compute(30);
    // particle 5 (type 6) should have no neighbors, all others have 4
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist_6->getNNeighArray(), access_location::host, access_mode::read);
        CHECK_EQUAL_UINT(h_n_neigh.data[0], 4);
        CHECK_EQUAL_UINT(h_n_neigh.data[1], 4);
        CHECK_EQUAL_UINT(h_n_neigh.data[2], 4);
        CHECK_EQUAL_UINT(h_n_neigh.data[3], 4);
        CHECK_EQUAL_UINT(h_n_neigh.data[4], 4);
        CHECK_EQUAL_UINT(h_n_neigh.data[5], 0);

        ArrayHandle<unsigned int> h_nlist(nlist_6->getNListArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_head_list(nlist_6->getHeadList(), access_location::host, access_mode::read);
        // just check the first 5 particles, since the last has no neighbors
        for (unsigned int cur_p = 0; cur_p < 5; ++cur_p)
            {
            vector<unsigned int> nbrs(4,0), check_nbrs;

            // create the sorted list of computed neighbors
            for (unsigned int cur_neigh = 0; cur_neigh < 4; ++cur_neigh)
                {
                nbrs[cur_neigh] = h_nlist.data[h_head_list.data[cur_p] + cur_neigh];
                }
            sort(nbrs.begin(), nbrs.end());

            // create the list of expected neighbors (everybody except for myself)
            check_nbrs.reserve(5);
            for (unsigned int i = 0; i < 5; ++i)
                {
                if (i != cur_p)
                    {
                    check_nbrs.push_back(i);
                    }
                }
            sort(check_nbrs.begin(), check_nbrs.end());

            for (unsigned int i=0; i < 4; ++i)
                {
                UP_ASSERT_EQUAL(nbrs[i],check_nbrs[i]);
                }
            }
        }
    }

//! Tests the ability of the neighbor list to exclude particle pairs
template <class NL>
void neighborlist_exclusion_tests(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    std::shared_ptr<SystemDefinition> sysdef_6(new SystemDefinition(6, BoxDim(20.0, 40.0, 60.0), 1, 0, 0, 0, 0, exec_conf));
    std::shared_ptr<ParticleData> pdata_6 = sysdef_6->getParticleData();

    // lets make this test simple: put all 6 particles on top of each other and
    // see if the exclusion code can ignore 4 of the particles
    {
    ArrayHandle<Scalar4> h_pos(pdata_6->getPositions(), access_location::host, access_mode::readwrite);

    h_pos.data[0].x = 0; h_pos.data[0].y = 0; h_pos.data[0].z = 0.0; h_pos.data[0].w = 0.0;
    h_pos.data[1].x = 0; h_pos.data[1].y = 0; h_pos.data[1].z = 0.0; h_pos.data[1].w = 0.0;
    h_pos.data[2].x = 0; h_pos.data[2].y = 0; h_pos.data[2].z = 0.0; h_pos.data[2].w = 0.0;
    h_pos.data[3].x = 0; h_pos.data[3].y = 0; h_pos.data[3].z = 0.0; h_pos.data[3].w = 0.0;
    h_pos.data[4].x = 0; h_pos.data[4].y = 0; h_pos.data[4].z = 0; h_pos.data[4].w = 0.0;
    h_pos.data[5].x = 0; h_pos.data[5].y = 0; h_pos.data[5].z =  0; h_pos.data[5].w = 0.0;

    pdata_6->notifyParticleSort();
    }

    std::shared_ptr<NeighborList> nlist_6(new NL(sysdef_6, 3.0, 0.25));
    nlist_6->setRCutPair(0,0,3.0);
    nlist_6->setStorageMode(NeighborList::full);
    nlist_6->addExclusion(0,1);
    nlist_6->addExclusion(0,2);
    nlist_6->addExclusion(0,3);
    nlist_6->addExclusion(0,4);

    nlist_6->compute(0);
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist_6->getNNeighArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_nlist(nlist_6->getNListArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_head_list(nlist_6->getHeadList(), access_location::host, access_mode::read);

//         UP_ASSERT(nli.getW() >= 6);
        CHECK_EQUAL_UINT(h_n_neigh.data[0], 1);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[0] + 0], 5);

        CHECK_EQUAL_UINT(h_n_neigh.data[1], 4);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[1] + 0], 2);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[1] + 1], 3);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[1] + 2], 4);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[1] + 3], 5);

        CHECK_EQUAL_UINT(h_n_neigh.data[2], 4);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[2] + 0], 1);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[2] + 1], 3);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[2] + 2], 4);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[2] + 3], 5);

        CHECK_EQUAL_UINT(h_n_neigh.data[3], 4);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[3] + 0], 1);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[3] + 1], 2);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[3] + 2], 4);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[3] + 3], 5);

        CHECK_EQUAL_UINT(h_n_neigh.data[4], 4);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[4] + 0], 1);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[4] + 1], 2);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[4] + 2], 3);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[4] + 3], 5);

        CHECK_EQUAL_UINT(h_n_neigh.data[5], 5);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[5] + 0], 0);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[5] + 1], 1);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[5] + 2], 2);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[5] + 3], 3);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[5] + 4], 4);
        }
    }

//! Tests the ability of the neighbor list to exclude particles from the same body
template <class NL>
void neighborlist_body_filter_tests(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    std::shared_ptr<SystemDefinition> sysdef_6(new SystemDefinition(6, BoxDim(20.0, 40.0, 60.0), 1, 0, 0, 0, 0, exec_conf));
    std::shared_ptr<ParticleData> pdata_6 = sysdef_6->getParticleData();

    // lets make this test simple: put all 6 particles on top of each other and
    // see if the exclusion code can ignore 4 of the particles
    {
    ArrayHandle<Scalar4> h_pos(pdata_6->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_body(pdata_6->getBodies(), access_location::host, access_mode::readwrite);

    h_pos.data[0].x = 0; h_pos.data[0].y = 0; h_pos.data[0].z = 0; h_pos.data[0].w = 0.0; h_body.data[0] = NO_BODY;
    h_pos.data[1].x = 0; h_pos.data[1].y = 0; h_pos.data[1].z = 0; h_pos.data[1].w = 0.0; h_body.data[1] = 0;
    h_pos.data[2].x = 0; h_pos.data[2].y = 0; h_pos.data[2].z = 0; h_pos.data[2].w = 0.0; h_body.data[2] = 1;
    h_pos.data[3].x = 0; h_pos.data[3].y = 0; h_pos.data[3].z = 0; h_pos.data[3].w = 0.0; h_body.data[3] = 0;
    h_pos.data[4].x = 0; h_pos.data[4].y = 0; h_pos.data[4].z = 0; h_pos.data[4].w = 0.0; h_body.data[4] = 1;
    h_pos.data[5].x = 0; h_pos.data[5].y = 0; h_pos.data[5].z = 0; h_pos.data[5].w = 0.0; h_body.data[5] = NO_BODY;

    pdata_6->notifyParticleSort();
    }

    std::shared_ptr<NeighborList> nlist_6(new NL(sysdef_6, 3.0, 0.25));
    nlist_6->setRCutPair(0,0,3.0);
    nlist_6->setFilterBody(true);
    nlist_6->setStorageMode(NeighborList::full);

    nlist_6->compute(0);
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist_6->getNNeighArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_nlist(nlist_6->getNListArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_head_list(nlist_6->getHeadList(), access_location::host, access_mode::read);

//         UP_ASSERT(nli.getW() >= 6);
        CHECK_EQUAL_UINT(h_n_neigh.data[0], 5);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[0] + 0], 1);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[0] + 1], 2);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[0] + 2], 3);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[0] + 3], 4);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[0] + 4], 5);

        CHECK_EQUAL_UINT(h_n_neigh.data[1], 4);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[1] + 0], 0);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[1] + 1], 2);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[1] + 2], 4);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[1] + 3], 5);

        CHECK_EQUAL_UINT(h_n_neigh.data[2], 4);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[2] + 0], 0);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[2] + 1], 1);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[2] + 2], 3);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[2] + 3], 5);

        CHECK_EQUAL_UINT(h_n_neigh.data[3], 4);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[3] + 0], 0);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[3] + 1], 2);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[3] + 2], 4);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[3] + 3], 5);

        CHECK_EQUAL_UINT(h_n_neigh.data[4], 4);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[4] + 0], 0);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[4] + 1], 1);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[4] + 2], 3);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[4] + 3], 5);

        CHECK_EQUAL_UINT(h_n_neigh.data[5], 5);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[5] + 0], 0);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[5] + 1], 1);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[5] + 2], 2);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[5] + 3], 3);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[5] + 4], 4);
        }
    }

//! Tests the ability of the neighbor list to filter by diameter
template <class NL>
void neighborlist_diameter_shift_tests(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    /////////////////////////////////////////////////////////
    // start with the simplest possible test: 3 particles in a huge box
    std::shared_ptr<SystemDefinition> sysdef_3(new SystemDefinition(4, BoxDim(25.0), 1, 0, 0, 0, 0, exec_conf));
    std::shared_ptr<ParticleData> pdata_3 = sysdef_3->getParticleData();

    {
    ArrayHandle<Scalar4> h_pos(pdata_3->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_diameter(pdata_3->getDiameters(), access_location::host, access_mode::readwrite);

    h_pos.data[0].x = 0; h_pos.data[0].y = 0; h_pos.data[0].z = 0.0; h_pos.data[0].w = 0.0; h_diameter.data[0] = 3.0;
    h_pos.data[2].x = 0; h_pos.data[2].y = 0; h_pos.data[2].z = 2.5; h_pos.data[2].w = 0.0; h_diameter.data[2] = 2.0;
    h_pos.data[1].x = 0; h_pos.data[1].y = 0; h_pos.data[1].z = -3.0; h_pos.data[1].w = 0.0; h_diameter.data[1] = 1.0;
    h_pos.data[3].x = 0; h_pos.data[3].y = 2.51; h_pos.data[3].z = 0; h_pos.data[3].w = 0.0; h_diameter.data[3] = 0;

    pdata_3->notifyParticleSort();
    }

    // test construction of the neighborlist
    std::shared_ptr<NeighborList> nlist_2(new NL(sysdef_3, 1.5, 0.5));
    nlist_2->setRCutPair(0,0,1.5);
    nlist_2->compute(1);
    nlist_2->setStorageMode(NeighborList::full);

    // with the given settings, there should be no neighbors: check that
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist_2->getNNeighArray(), access_location::host, access_mode::read);

        CHECK_EQUAL_UINT(h_n_neigh.data[0], 0);
        CHECK_EQUAL_UINT(h_n_neigh.data[1], 0);
        CHECK_EQUAL_UINT(h_n_neigh.data[2], 0);
        }

    // enable diameter shifting
    nlist_2->setDiameterShift(true);
    nlist_2->setMaximumDiameter(3.0);
    nlist_2->compute(2);

    // the particle 0 should now be neighbors with 1 and 2
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist_2->getNNeighArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_nlist(nlist_2->getNListArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_head_list(nlist_2->getHeadList(), access_location::host, access_mode::read);

        CHECK_EQUAL_UINT(h_n_neigh.data[0], 2);
            {
            vector<unsigned int> nbrs(2, 0);
            nbrs[0] = h_nlist.data[h_head_list.data[0] + 0];
            nbrs[1] = h_nlist.data[h_head_list.data[0] + 1];
            sort(nbrs.begin(), nbrs.end());
            unsigned int check_nbrs[] = {1,2};
            for (unsigned int i=0; i < 2; ++i)
                {
                UP_ASSERT_EQUAL(nbrs[i],check_nbrs[i]);
                }
            }

        CHECK_EQUAL_UINT(h_n_neigh.data[1], 1);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[1]], 0);

        CHECK_EQUAL_UINT(h_n_neigh.data[2], 1);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[2]], 0);
        }
    }


//! Test two implementations of NeighborList and verify that the output is identical
template <class NLA, class NLB>
void neighborlist_comparison_test(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // construct the particle system
    RandomInitializer init(1000, Scalar(0.016778), Scalar(0.9), "A");
    std::shared_ptr< SnapshotSystemData<Scalar> > snap = init.getSnapshot();
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));
    std::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    std::shared_ptr<NeighborList> nlist1(new NLA(sysdef, Scalar(3.0), Scalar(0.4)));
    nlist1->setRCutPair(0,0,3.0);
    nlist1->setStorageMode(NeighborList::full);

    std::shared_ptr<NeighborList> nlist2(new NLB(sysdef, Scalar(3.0), Scalar(0.4)));
    nlist2->setRCutPair(0,0,3.0);
    nlist2->setStorageMode(NeighborList::full);

    // setup some exclusions: try to fill out all four exclusions for each particle
    for (unsigned int i=0; i < pdata->getN()-2; i++)
        {
        nlist1->addExclusion(i,i+1);
        nlist1->addExclusion(i,i+2);

        nlist2->addExclusion(i,i+1);
        nlist2->addExclusion(i,i+2);
        }

    // compute each of the lists
    nlist1->compute(0);
    nlist2->compute(0);

    // verify that both new ones match the basic
    ArrayHandle<unsigned int> h_n_neigh1(nlist1->getNNeighArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_nlist1(nlist1->getNListArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_head_list1(nlist1->getHeadList(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_n_neigh2(nlist2->getNNeighArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_nlist2(nlist2->getNListArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_head_list2(nlist2->getHeadList(), access_location::host, access_mode::read);

    // temporary vectors for holding the lists: they will be sorted for compariso
    std::vector<unsigned int> tmp_list2;

    // check to make sure that every neighbor matches
    for (unsigned int i = 0; i < pdata->getN(); i++)
        {
        UP_ASSERT(h_n_neigh2.data[i] >= h_n_neigh1.data[i]);

        // test list
        std::vector<unsigned int> test_list(h_n_neigh2.data[i]);
        for (unsigned int j=0; j < h_n_neigh2.data[i]; ++j)
            {
            test_list[j] = h_nlist2.data[h_head_list2.data[i] + j];
            }

        // check all elements from ref list are in the test list
        for (unsigned int j = 0; j < h_n_neigh1.data[i]; ++j)
            {
            const unsigned int ref_idx = h_nlist1.data[h_head_list1.data[i] + j];
            bool found = std::find(test_list.begin(), test_list.end(), ref_idx) != test_list.end();
            if (!found)
                {
                std::cout << "Neighbor " << ref_idx << " from reference list not found in test list for particle " << i << "." << std::endl;
                UP_ASSERT(false);
                }
            }
        }
    }

//! Test that a NeighborList can successfully exclude a ridiculously large number of particles
template <class NL>
void neighborlist_large_ex_tests(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // construct the particle system
    RandomInitializer init(1000, Scalar(0.016778), Scalar(0.9), "A");
    std::shared_ptr< SnapshotSystemData<Scalar> > snap = init.getSnapshot();
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));
    std::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    std::shared_ptr<NeighborList> nlist(new NL(sysdef, Scalar(8.0), Scalar(0.4)));
    nlist->setRCutPair(0,0,8.0);
    nlist->setStorageMode(NeighborList::full);

    // add every single neighbor as an exclusion
    nlist->compute(0);
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist->getNNeighArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_nlist(nlist->getNListArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_head_list(nlist->getHeadList(), access_location::host, access_mode::read);

        for (unsigned int i = 0; i < pdata->getN(); i++)
            {
            for (unsigned int neigh = 0; neigh < h_n_neigh.data[i]; neigh++)
                {
                unsigned int j = h_nlist.data[h_head_list.data[i] + neigh];
                nlist->addExclusion(i,j);
                }
            }
        }

    // compute the nlist again
    nlist->compute(0);

    // verify that there are now 0 neighbors for each particle
    ArrayHandle<unsigned int> h_n_neigh(nlist->getNNeighArray(), access_location::host, access_mode::read);

    // check to make sure that every neighbor matches
    for (unsigned int i = 0; i < pdata->getN(); i++)
        {
        CHECK_EQUAL_UINT(h_n_neigh.data[i], 0);
        }
    }

//! Test that NeighborList can exclude particles correctly when cutoff radius is negative
template <class NL>
void neighborlist_cutoff_exclude_tests(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // Initialize a system of 3 particles each having a distinct type
    std::shared_ptr<SystemDefinition> sysdef_3(new SystemDefinition(3, BoxDim(25.0), 3, 0, 0, 0, 0, exec_conf));
    std::shared_ptr<ParticleData> pdata_3 = sysdef_3->getParticleData();

    // put the particles on top of each other, the worst case scenario for inclusion / exclusion since the distance
    // between them is zero
        {
        ArrayHandle<Scalar4> h_pos(pdata_3->getPositions(), access_location::host, access_mode::overwrite);
        for (unsigned int i=0; i < pdata_3->getN(); ++i)
            {
            h_pos.data[i] = make_scalar4(0.0, 0.0, 0.0, __int_as_scalar(i));
            }
        }

    std::shared_ptr<NeighborList> nlist(new NL(sysdef_3, Scalar(-1.0), Scalar(0.4)));
    // explicitly set the cutoff radius of each pair type to ignore
    for (unsigned int i = 0; i < pdata_3->getNTypes(); ++i)
        {
        for (unsigned int j = i; j < pdata_3->getNTypes(); ++j)
            {
            nlist->setRCutPair(i,j,-1.0);
            }
        }
    nlist->setStorageMode(NeighborList::full);

    // compute the neighbor list, each particle should have no neighbors
    nlist->compute(0);
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist->getNNeighArray(), access_location::host, access_mode::read);
        CHECK_EQUAL_UINT(h_n_neigh.data[0], 0);
        CHECK_EQUAL_UINT(h_n_neigh.data[1], 0);
        CHECK_EQUAL_UINT(h_n_neigh.data[2], 0);
        }

    // turn on cross interaction with B particle
    for (unsigned int i=0; i < pdata_3->getNTypes(); ++i)
        {
        nlist->setRCutPair(1, i, 1.0);
        }
    nlist->compute(1);
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist->getNNeighArray(), access_location::host, access_mode::read);
        CHECK_EQUAL_UINT(h_n_neigh.data[0], 1);
        CHECK_EQUAL_UINT(h_n_neigh.data[1], 2); // B ignores itself, but gets everyone else as a neighbor
        CHECK_EQUAL_UINT(h_n_neigh.data[2], 1);

        ArrayHandle<unsigned int> h_nlist(nlist->getNListArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_head_list(nlist->getHeadList(), access_location::host, access_mode::read);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[0]], 1);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[2]], 1);

        vector<unsigned int> nbrs(2, 0);
        nbrs[0] = h_nlist.data[h_head_list.data[1] + 0];
        nbrs[1] = h_nlist.data[h_head_list.data[1] + 1];
        sort(nbrs.begin(), nbrs.end());
        unsigned int check_nbrs[] = {0,2};

        for (unsigned int i=0; i < 2; ++i)
            {
            UP_ASSERT_EQUAL(nbrs[i],check_nbrs[i]);
            }
        }

    // turn A-C on and B-C off with things very close to the < 0.0 criterion as a pathological case
    nlist->setRCutPair(0, 2, 0.00001);
    nlist->setRCutPair(1, 2, -0.00001);
    nlist->compute(3);
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist->getNNeighArray(), access_location::host, access_mode::read);
        CHECK_EQUAL_UINT(h_n_neigh.data[0], 2);
        CHECK_EQUAL_UINT(h_n_neigh.data[1], 1);
        CHECK_EQUAL_UINT(h_n_neigh.data[2], 1);

        ArrayHandle<unsigned int> h_nlist(nlist->getNListArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_head_list(nlist->getHeadList(), access_location::host, access_mode::read);

        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[1]], 0);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[2]], 0);

        vector<unsigned int> nbrs(2, 0);
        nbrs[0] = h_nlist.data[h_head_list.data[0] + 0];
        nbrs[1] = h_nlist.data[h_head_list.data[0] + 1];
        sort(nbrs.begin(), nbrs.end());
        unsigned int check_nbrs[] = {1,2};

        for (unsigned int i=0; i < 2; ++i)
            {
            UP_ASSERT_EQUAL(nbrs[i],check_nbrs[i]);
            }
        }
    }

//! Tests for correctness of neighbor search in 2d systems
template<class NL>
void neighborlist_2d_tests(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    auto sysdef = std::make_shared<SystemDefinition>(2, BoxDim(10.0, 10.0, 0.01), 1, 0, 0, 0, 0, exec_conf);
    sysdef->setNDimensions(2);
    auto pdata = sysdef->getParticleData();

    auto nlist = std::make_shared<NL>(sysdef, 3.0, 0.25);
    nlist->setRCutPair(0,0,3.0);
    nlist->setStorageMode(NeighborList::full);

    // non-interacting inside the box
        {
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);

        h_pos.data[0].x = h_pos.data[0].y = 0.0;
        h_pos.data[1].x = h_pos.data[1].y = 3.0;
        h_pos.data[0].z = h_pos.data[1].z = 0.0;

        h_pos.data[0].w = __int_as_scalar(0.0); h_pos.data[1].w = __int_as_scalar(0.0);
        }
    nlist->compute(0);
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist->getNNeighArray(), access_location::host, access_mode::read);

        CHECK_EQUAL_UINT(h_n_neigh.data[0], 0);
        CHECK_EQUAL_UINT(h_n_neigh.data[1], 0);
        }

    // interacting inside the box
        {
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);

        h_pos.data[0].x = h_pos.data[0].y = 0.0;
        h_pos.data[1].x = h_pos.data[1].y = 1.0;
        }
    nlist->compute(1);
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist->getNNeighArray(), access_location::host, access_mode::read);

        CHECK_EQUAL_UINT(h_n_neigh.data[0], 1);
        CHECK_EQUAL_UINT(h_n_neigh.data[1], 1);

        ArrayHandle<unsigned int> h_nlist(nlist->getNListArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_head_list(nlist->getHeadList(), access_location::host, access_mode::read);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[0]], 1);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[1]], 0);
        }

    // non-interacting through boundary
        {
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);

        h_pos.data[0].x = h_pos.data[0].y = 4.9;
        h_pos.data[1].x = h_pos.data[1].y = -2.1;
        }
    nlist->compute(2);
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist->getNNeighArray(), access_location::host, access_mode::read);

        CHECK_EQUAL_UINT(h_n_neigh.data[0], 0);
        CHECK_EQUAL_UINT(h_n_neigh.data[1], 0);
        }

    // interacting through boundary
        {
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);

        h_pos.data[0].x = h_pos.data[0].y = 4.9;
        h_pos.data[1].x = h_pos.data[1].y = -4.9;
        }
    nlist->compute(3);
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist->getNNeighArray(), access_location::host, access_mode::read);

        CHECK_EQUAL_UINT(h_n_neigh.data[0], 1);
        CHECK_EQUAL_UINT(h_n_neigh.data[1], 1);

        ArrayHandle<unsigned int> h_nlist(nlist->getNListArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_head_list(nlist->getHeadList(), access_location::host, access_mode::read);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[0]], 1);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[1]], 0);
        }

    // non-interacting through other boundary
        {
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);

        h_pos.data[0].x = -4.9; h_pos.data[0].y = 4.9;
        h_pos.data[1].x = 2.1; h_pos.data[1].y = -2.1;
        }
    nlist->compute(4);
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist->getNNeighArray(), access_location::host, access_mode::read);

        CHECK_EQUAL_UINT(h_n_neigh.data[0], 0);
        CHECK_EQUAL_UINT(h_n_neigh.data[1], 0);
        }

    // interacting through other boundary
        {
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);

        h_pos.data[0].x = -4.9; h_pos.data[0].y = 4.9;
        h_pos.data[1].x = 4.9; h_pos.data[1].y = -4.9;
        }
    nlist->compute(5);
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist->getNNeighArray(), access_location::host, access_mode::read);

        CHECK_EQUAL_UINT(h_n_neigh.data[0], 1);
        CHECK_EQUAL_UINT(h_n_neigh.data[1], 1);

        ArrayHandle<unsigned int> h_nlist(nlist->getNListArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_head_list(nlist->getHeadList(), access_location::host, access_mode::read);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[0]], 1);
        CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[1]], 0);
        }
    }

///////////////
// BINNED CPU
///////////////
//! basic test case for binned class
UP_TEST( NeighborListBinned_basic )
    {
    neighborlist_basic_tests<NeighborListBinned>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
//! exclusion test case for binned class
UP_TEST( NeighborListBinned_exclusion )
    {
    neighborlist_exclusion_tests<NeighborListBinned>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
//! large exclusion test case for binned class
UP_TEST( NeighborListBinned_large_ex )
    {
    neighborlist_large_ex_tests<NeighborListBinned>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
//! body filter test case for binned class
UP_TEST( NeighborListBinned_body_filter)
    {
    neighborlist_body_filter_tests<NeighborListBinned>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
//! diameter filter test case for binned class
UP_TEST( NeighborListBinned_diameter_shift )
    {
    neighborlist_diameter_shift_tests<NeighborListBinned>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
//! particle asymmetry test case for binned class
UP_TEST( NeighborListBinned_particle_asymm )
    {
    neighborlist_particle_asymm_tests<NeighborListBinned>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
//! cutoff exclusion test case for binned class
UP_TEST( NeighborListBinned_cutoff_exclude )
    {
    neighborlist_cutoff_exclude_tests<NeighborListBinned>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
//! type test case for binned class
UP_TEST( NeighborListBinned_type )
    {
    neighborlist_type_tests<NeighborListBinned>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
//! 2d tests for binned class
UP_TEST( NeighborListBinned_2d )
    {
    neighborlist_2d_tests<NeighborListBinned>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

////////////////////
// STENCIL CPU
////////////////////
//! basic test case for stencil class
UP_TEST( NeighborListStencil_basic )
    {
    neighborlist_basic_tests<NeighborListStencil>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

//! exclusion test case for stencil class
UP_TEST( NeighborListStencil_exclusion )
    {
    neighborlist_exclusion_tests<NeighborListStencil>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
//! large exclusion test case for stencil class
UP_TEST( NeighborListStencil_large_ex )
    {
    neighborlist_large_ex_tests<NeighborListStencil>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
//! body filter test case for stencil class
UP_TEST( NeighborListStencil_body_filter)
    {
    neighborlist_body_filter_tests<NeighborListStencil>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
//! diameter filter test case for stencil class
UP_TEST( NeighborListStencil_diameter_shift )
    {
    neighborlist_diameter_shift_tests<NeighborListStencil>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
//! particle asymmetry test case for stencil class
UP_TEST( NeighborListStencil_particle_asymm )
    {
    neighborlist_particle_asymm_tests<NeighborListStencil>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
//! cutoff exclusion test case for stencil class
UP_TEST( NeighborListStencil_cutoff_exclude )
    {
    neighborlist_cutoff_exclude_tests<NeighborListStencil>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
//! type test case for stencil class
UP_TEST( NeighborListStencil_type )
    {
    neighborlist_type_tests<NeighborListStencil>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
//! 2d tests for stencil class
UP_TEST( NeighborListStencil_2d )
    {
    neighborlist_2d_tests<NeighborListStencil>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
//! comparison test case for stencil class
UP_TEST( NeighborListStencil_comparison )
    {
    neighborlist_comparison_test<NeighborListBinned, NeighborListStencil>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

///////////////
// TREE CPU
///////////////
//! basic test case for tree class
UP_TEST( NeighborListTree_basic )
    {
    neighborlist_basic_tests<NeighborListTree>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
//! exclusion test case for tree class
UP_TEST( NeighborListTree_exclusion )
    {
    neighborlist_exclusion_tests<NeighborListTree>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
//! large exclusion test case for tree class
UP_TEST( NeighborListTree_large_ex )
    {
    neighborlist_large_ex_tests<NeighborListTree>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
//! body filter test case for tree class
UP_TEST( NeighborListTree_body_filter)
    {
    neighborlist_body_filter_tests<NeighborListTree>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
//! diameter filter test case for binned class
UP_TEST( NeighborListTree_diameter_shift )
    {
    neighborlist_diameter_shift_tests<NeighborListTree>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
//! particle asymmetry test case for tree class
UP_TEST( NeighborListTree_particle_asymm )
    {
    neighborlist_particle_asymm_tests<NeighborListTree>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
//! cutoff exclusion test case for tree class
UP_TEST( NeighborListTree_cutoff_exclude )
    {
    neighborlist_cutoff_exclude_tests<NeighborListTree>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
//! type test case for tree class
UP_TEST( NeighborListTree_type )
    {
    neighborlist_type_tests<NeighborListTree>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
//! 2d tests for tree class
UP_TEST( NeighborListTree_2d )
    {
    neighborlist_2d_tests<NeighborListTree>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
//! comparison test case for tree class
UP_TEST( NeighborListTree_comparison )
    {
    neighborlist_comparison_test<NeighborListBinned, NeighborListTree>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

#ifdef ENABLE_CUDA
///////////////
// BINNED GPU
///////////////
//! basic test case for GPUBinned class
UP_TEST( NeighborListGPUBinned_basic )
    {
    neighborlist_basic_tests<NeighborListGPUBinned>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
//! exclusion test case for GPUBinned class
UP_TEST( NeighborListGPUBinned_exclusion )
    {
    neighborlist_exclusion_tests<NeighborListGPUBinned>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
//! large exclusion test case for GPUBinned class
UP_TEST( NeighborListGPUBinned_large_ex )
    {
    neighborlist_large_ex_tests<NeighborListGPUBinned>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
//! body filter test case for GPUBinned class
UP_TEST( NeighborListGPUBinned_body_filter)
    {
    neighborlist_body_filter_tests<NeighborListGPUBinned>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
//! diameter filter test case for GPUBinned class
UP_TEST( NeighborListGPUBinned_diameter_shift )
    {
    neighborlist_diameter_shift_tests<NeighborListGPUBinned>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
//! particle asymmetry test case for GPUBinned class
UP_TEST( NeighborListGPUBinned_particle_asymm )
    {
    neighborlist_particle_asymm_tests<NeighborListGPUBinned>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
//! cutoff exclusion test case for GPUBinned class
UP_TEST( NeighborListGPUBinned_cutoff_exclude )
    {
    neighborlist_cutoff_exclude_tests<NeighborListGPUBinned>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
//! type test case for GPUBinned class
UP_TEST( NeighborListGPUBinned_type )
    {
    neighborlist_type_tests<NeighborListGPUBinned>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
//! 2d tests for GPUBinned class
UP_TEST( NeighborListGPUBinned_2d )
    {
    neighborlist_2d_tests<NeighborListGPUBinned>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
//! comparison test case for GPUBinned class
UP_TEST( NeighborListGPUBinned_comparison )
    {
    neighborlist_comparison_test<NeighborListBinned, NeighborListGPUBinned>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

///////////////
// STENCIL GPU
///////////////
//! basic test case for GPUStencil class
UP_TEST( NeighborListGPUStencil_basic )
    {
    neighborlist_basic_tests<NeighborListGPUStencil>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
//! exclusion test case for GPUStencil class
UP_TEST( NeighborListGPUStencil_exclusion )
    {
    neighborlist_exclusion_tests<NeighborListGPUStencil>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
//! large exclusion test case for GPUStencil class
UP_TEST( NeighborListGPUStencil_large_ex )
    {
    neighborlist_large_ex_tests<NeighborListGPUStencil>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
//! body filter test case for GPUStencil class
UP_TEST( NeighborListGPUStencil_body_filter)
    {
    neighborlist_body_filter_tests<NeighborListGPUStencil>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
//! diameter filter test case for GPUStencil class
UP_TEST( NeighborListGPUStencil_diameter_shift )
    {
    neighborlist_diameter_shift_tests<NeighborListGPUStencil>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
//! particle asymmetry test case for GPUStencil class
UP_TEST( NeighborListGPUStencil_particle_asymm )
    {
    neighborlist_particle_asymm_tests<NeighborListGPUStencil>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
//! cutoff exclusion test case for GPUStencil class
UP_TEST( NeighborListGPUStencil_cutoff_exclude )
    {
    neighborlist_cutoff_exclude_tests<NeighborListGPUStencil>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
//! type test case for GPUStencil class
UP_TEST( NeighborListGPUStencil_type )
    {
    neighborlist_type_tests<NeighborListGPUStencil>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
//! 2d tests for GPUStencil class
UP_TEST( NeighborListGPUStencil_2d )
    {
    neighborlist_2d_tests<NeighborListGPUStencil>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
//! comparison test case for GPUStencil class against Stencil on cpu
UP_TEST( NeighborListGPUStencil_cpu_comparison )
    {
    neighborlist_comparison_test<NeighborListStencil, NeighborListGPUStencil>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
//! comparison test case for GPUStencil class against GPUBinned
UP_TEST( NeighborListGPUStencil_binned_comparison )
    {
    neighborlist_comparison_test<NeighborListGPUBinned, NeighborListGPUStencil>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

///////////////
// TREE GPU
///////////////
//! basic test case for GPUTree class
UP_TEST( NeighborListGPUTree_basic )
    {
    std::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::GPU));
    neighborlist_basic_tests<NeighborListGPUTree>(exec_conf);
    }
//! exclusion test case for GPUTree class
UP_TEST( NeighborListGPUTree_exclusion )
    {
    std::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::GPU));
    neighborlist_exclusion_tests<NeighborListGPUTree>(exec_conf);
    }
//! large exclusion test case for GPUTree class
UP_TEST( NeighborListGPUTree_large_ex )
    {
    std::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::GPU));
    neighborlist_large_ex_tests<NeighborListGPUTree>(exec_conf);
    }
//! body filter test case for GPUTree class
UP_TEST( NeighborListGPUTree_body_filter)
    {
    std::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::GPU));
    neighborlist_body_filter_tests<NeighborListGPUTree>(exec_conf);
    }
//! diameter filter test case for GPUTree class
UP_TEST( NeighborListGPUTree_diameter_shift )
    {
    std::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::GPU));
    neighborlist_diameter_shift_tests<NeighborListGPUTree>(exec_conf);
    }
//! particle asymmetry test case for GPUTree class
UP_TEST( NeighborListGPUTree_particle_asymm )
    {
    std::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::GPU));
    neighborlist_particle_asymm_tests<NeighborListGPUTree>(exec_conf);
    }
//! cutoff exclusion test case for GPUTree class
UP_TEST( NeighborListGPUTree_cutoff_exclude )
    {
    std::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::GPU));
    neighborlist_cutoff_exclude_tests<NeighborListGPUTree>(exec_conf);
    }
//! type test case for tree class
UP_TEST( NeighborListGPUTree_type )
    {
    std::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::GPU));
    neighborlist_type_tests<NeighborListGPUTree>(exec_conf);
    }
//! 2d tests for tree class
UP_TEST( NeighborListGPUTree_2d )
    {
    std::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::GPU));
    neighborlist_2d_tests<NeighborListGPUTree>(exec_conf);
    }
//! comparison test case for GPUTree class with itself
UP_TEST( NeighborListGPUTree_cpu_comparison )
    {
    std::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::GPU));
    neighborlist_comparison_test<NeighborListTree, NeighborListGPUTree>(exec_conf);
    }
//! comparison test case for GPUTree class with GPUBinned
UP_TEST( NeighborListGPUTree_binned_comparison )
    {
    std::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::GPU));
    neighborlist_comparison_test<NeighborListGPUBinned, NeighborListGPUTree>(exec_conf);
    }
#endif
