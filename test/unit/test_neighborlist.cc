/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2015 The Regents of
the University of Michigan All rights reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/



#include <iostream>
#include <algorithm>

#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include "NeighborList.h"
#include "NeighborListBinned.h"
#include "NeighborListTree.h"
#include "Initializers.h"

#ifdef ENABLE_CUDA
#include "NeighborListGPU.h"
#include "NeighborListGPUBinned.h"
#include "NeighborListGPUTree.h"
#endif

using namespace std;
using namespace boost;

//! Define the name of the boost test module
#define BOOST_TEST_MODULE NeighborListTest
#include "boost_utf_configure.h"

//! Performs basic functionality tests on a neighbor list
template <class NL>
void neighborlist_basic_tests(boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    /////////////////////////////////////////////////////////
    // start with the simplest possible test: 2 particles in a huge box
    boost::shared_ptr<SystemDefinition> sysdef_2(new SystemDefinition(2, BoxDim(25.0), 1, 0, 0, 0, 0, exec_conf));
    boost::shared_ptr<ParticleData> pdata_2 = sysdef_2->getParticleData();

    {
    ArrayHandle<Scalar4> h_pos(pdata_2->getPositions(), access_location::host, access_mode::readwrite);

    h_pos.data[0].x = h_pos.data[0].y = h_pos.data[0].z = 0.0;
    h_pos.data[1].x = h_pos.data[1].y = h_pos.data[1].z = 3.25;
    
    h_pos.data[0].w = 0.0; h_pos.data[1].w = 0.0;
    pdata_2->notifyParticleSort();
    }

    // test construction of the neighborlist
    boost::shared_ptr<NeighborList> nlist_2(new NL(sysdef_2, 3.0, 0.25));
    nlist_2->setRCutPair(0,0,3.0);
    nlist_2->compute(1);

    // with the given radius, there should be no neighbors: check that
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist_2->getNNeighArray(), access_location::host, access_mode::read);

        BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[0], 0);
        BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[1], 0);
        }
        
    // adjust the radius to include the particles and see if we get some now
    nlist_2->setRCutPair(0,0,5.5);
    nlist_2->compute(2);
    // some neighbor lists default to full because they don't support half: ignore them
    if (nlist_2->getStorageMode() == NeighborList::half)
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist_2->getNNeighArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_nlist(nlist_2->getNListArray(), access_location::host, access_mode::read);

        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[0], 1);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[0], 1);
        // since this is a half list, only 0 stores 1 as a neighbor
        BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[1], 0);
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

        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[0], 1);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[0] + 0], 1);

        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[1], 1);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[1] + 0], 0);
        }


    ////////////////////////////////////////////////////////////////////
    // now, lets do a more thorough test and include boundary conditions
    // there are way too many permutations to test here, so I will simply
    // test +x, -x, +y, -y, +z, and -z independantly
    // build a 6 particle system with particles across each boundary

    boost::shared_ptr<SystemDefinition> sysdef_6(new SystemDefinition(6, BoxDim(20.0, 40.0, 60.0), 1, 0, 0, 0, 0, exec_conf));
    boost::shared_ptr<ParticleData> pdata_6 = sysdef_6->getParticleData();

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

    boost::shared_ptr<NeighborList> nlist_6(new NL(sysdef_6, 3.0, 0.25));
    nlist_6->setRCutPair(0,0,3.0);
    nlist_6->setStorageMode(NeighborList::full);
    nlist_6->compute(0);
    // verify the neighbor list
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist_6->getNNeighArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_nlist(nlist_6->getNListArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_head_list(nlist_6->getHeadList(), access_location::host, access_mode::read);

        // check for right number of nbrs
        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[0], 1);
        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[1], 1);
        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[2], 1);
        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[3], 1);
        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[4], 1);
        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[5], 1);
        
        // populate the neighbor list as a collection for fast compare
        vector<unsigned int> nbrs(6, 0);
        for (unsigned int i=0; i < 6; ++i)
            {
            nbrs[i] = h_nlist.data[h_head_list.data[i]];
            }
        
        // the answer we expect
        unsigned int check_nbrs[] = {1, 0, 3, 2, 5, 4};
        
        BOOST_CHECK_EQUAL_COLLECTIONS(nbrs.begin(), nbrs.end(), check_nbrs, check_nbrs + 6);
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
        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[0], 1);
        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[1], 1);
        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[2], 1);
        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[3], 1);
        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[4], 1);
        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[5], 1);
        
        // populate the neighbor list as a collection for fast compare
        vector<unsigned int> nbrs(6, 0);
        for (unsigned int i=0; i < 6; ++i)
            {
            nbrs[i] = h_nlist.data[h_head_list.data[i]];
            }
        
        // the answer we expect
        unsigned int check_nbrs[] = {1, 0, 3, 2, 5, 4};
        
        BOOST_CHECK_EQUAL_COLLECTIONS(nbrs.begin(), nbrs.end(), check_nbrs, check_nbrs + 6);
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

        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[0], 3);
        
        vector<unsigned int> nbrs(3,0);
        for (unsigned int i=0; i < 3; ++i)
            {
            nbrs[i] = h_nlist.data[h_head_list.data[0] + i];
            }
            
        // sort the neighbors because it doesn't matter what order they are stored in, just that they all are there
        sort(nbrs.begin(), nbrs.end());
        
        // the answer we expect
        unsigned int check_nbrs[] = {1, 4, 5};
        BOOST_CHECK_EQUAL_COLLECTIONS(nbrs.begin(), nbrs.end(), check_nbrs, check_nbrs + 3);
        }
    }
    
//! Test neighborlist functionality with particles with different numbers of neighbors
template <class NL>
void neighborlist_particle_asymm_tests(boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    boost::shared_ptr<SystemDefinition> sysdef_3(new SystemDefinition(3, BoxDim(40.0, 40.0, 60.0), 2, 0, 0, 0, 0, exec_conf));
    boost::shared_ptr<ParticleData> pdata_3 = sysdef_3->getParticleData();
    // check that pair cutoffs are set independently
        {
        ArrayHandle<Scalar4> h_pos(pdata_3->getPositions(), access_location::host, access_mode::readwrite);

        h_pos.data[0].x = 0.0; h_pos.data[0].y = 0.0; h_pos.data[0].z = 0.0; h_pos.data[0].w = __int_as_scalar(1);
        h_pos.data[1].x = Scalar(1.2); h_pos.data[1].y = 0.0; h_pos.data[1].z = 0.0; h_pos.data[1].w = __int_as_scalar(0);
        h_pos.data[2].x = Scalar(3.5); h_pos.data[2].y = 0.0; h_pos.data[2].z = 0.0; h_pos.data[2].w = __int_as_scalar(1);
        
        pdata_3->notifyParticleSort();
        }
        
    boost::shared_ptr<NeighborList> nlist_3(new NL(sysdef_3, 3.0, 0.25));
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

        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[0], 1);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[0] + 0], 1);

        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[1], 1);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[1] + 0], 0);

        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[2], 0);
        }
    
    // now change the cutoff so that 2 is neighbors with 0 but not 1
    nlist_3->setRCutPair(1,1,3.5);
    nlist_3->compute(1);
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist_3->getNNeighArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_nlist(nlist_3->getNListArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_head_list(nlist_3->getHeadList(), access_location::host, access_mode::read);
            
        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[0], 2);
        vector<unsigned int> nbrs(2, 0);
        nbrs[0] = h_nlist.data[h_head_list.data[0] + 0];
        nbrs[1] = h_nlist.data[h_head_list.data[0] + 1];
        sort(nbrs.begin(), nbrs.end());
        unsigned int check_nbrs[] = {1,2};
        BOOST_CHECK_EQUAL_COLLECTIONS(nbrs.begin(), nbrs.end(), check_nbrs, check_nbrs + 2);

        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[1], 1);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[1] + 0], 0);

        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[2], 1);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[2] + 0], 0);
        }
        
    // now change the cutoff so that all are neighbors
    nlist_3->setRCutPair(0,1,2.5);
    nlist_3->compute(20);
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist_3->getNNeighArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_nlist(nlist_3->getNListArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_head_list(nlist_3->getHeadList(), access_location::host, access_mode::read);
            
        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[0], 2);

        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[1], 2);

        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[2], 2);
        }
        
    // check what happens with particle resize by first keeping number below the 8 default, and then bumping over this
    // do this with size 18 so that NeighborListGPU is forced to use kernel call with multiple levels at m_bin_size = 4
    boost::shared_ptr<SystemDefinition> sysdef_18(new SystemDefinition(18, BoxDim(40.0, 40.0, 40.0), 2, 0, 0, 0, 0, exec_conf));
    boost::shared_ptr<ParticleData> pdata_18 = sysdef_18->getParticleData();
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
        
    boost::shared_ptr<NeighborList> nlist_18(new NL(sysdef_18, 3.0, 0.05));
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
        BOOST_REQUIRE(nlist_18->getNListArray().getPitch() >= 192);
        BOOST_CHECK_EQUAL_UINT(h_head_list.data[17],176);
        
        for (unsigned int i=0; i < 18; ++i)
            {
            if (i < 3)
                {
                BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[i], 14);
                for (unsigned int j=0; j < 14; ++j)
                    {
                    // not the ones far away
                    BOOST_CHECK(h_nlist.data[j] != 3 && h_nlist.data[j] != 16 && h_nlist.data[j] != 17);
                    }
                }
            else if (i == 3 || i >= 16)
                {
                BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[i], 0);
                }
            else
                {
                BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[i], 4);
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
        BOOST_REQUIRE(nlist_18->getNListArray().getPitch() >= 240);
        BOOST_CHECK_EQUAL_UINT(h_head_list.data[17],216);
        
        for (unsigned int i=0; i < 18; ++i)
            {
            if (i <= 3 || i >= 16)
                {
                BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[i], 17);
                }
            else
                {
                BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[i], 7);
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
        BOOST_REQUIRE(nlist_18->getNListArray().getPitch() >= 432);
        BOOST_CHECK_EQUAL_UINT(h_head_list.data[17],408);
        
        for (unsigned int i=0; i < 18; ++i)
            {
            BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[i], 17);
            }
        }
    }

//! Test neighborlist functionality with changing types
template <class NL>
void neighborlist_type_tests(boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    boost::shared_ptr<SystemDefinition> sysdef_6(new SystemDefinition(6, BoxDim(40.0, 40.0, 40.0), 4, 0, 0, 0, 0, exec_conf));
    boost::shared_ptr<ParticleData> pdata_6 = sysdef_6->getParticleData();
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

    boost::shared_ptr<NeighborList> nlist_6(new NL(sysdef_6, 3.0, 0.1));
    nlist_6->setStorageMode(NeighborList::full);
    nlist_6->compute(0);

    // everybody should neighbor everybody else
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist_6->getNNeighArray(), access_location::host, access_mode::read);
        BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[0], 5);
        BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[1], 5);
        BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[2], 5);
        BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[3], 5);
        BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[4], 5);
        BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[5], 5);
        
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
            BOOST_CHECK_EQUAL_COLLECTIONS(nbrs.begin(), nbrs.end(), check_nbrs.begin(), check_nbrs.end());
            }
        }

    // add a new type
    pdata_6->addType("E");
    nlist_6->setRCut(3.0, 0.1); // update the rcut for the new type
    nlist_6->compute(10);
    // result is unchanged
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist_6->getNNeighArray(), access_location::host, access_mode::read);
        BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[0], 5);
        BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[1], 5);
        BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[2], 5);
        BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[3], 5);
        BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[4], 5);
        BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[5], 5);
        
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
            BOOST_CHECK_EQUAL_COLLECTIONS(nbrs.begin(), nbrs.end(), check_nbrs.begin(), check_nbrs.end());
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
        BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[0], 5);
        BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[1], 5);
        BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[2], 5);
        BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[3], 5);
        BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[4], 5);
        BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[5], 5);
        
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
            BOOST_CHECK_EQUAL_COLLECTIONS(nbrs.begin(), nbrs.end(), check_nbrs.begin(), check_nbrs.end());
            }
        }

    pdata_6->addType("H");
    nlist_6->setRCut(3.0,0.1);
    // disable the interaction between type 6 and all other particles
    for (unsigned int cur_type = 0; cur_type < pdata_6->getNTypes(); ++cur_type)
        {
        nlist_6->setRCutPair(6, cur_type, 0.001);
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
        BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[0], 4);
        BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[1], 4);
        BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[2], 4);
        BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[3], 4);
        BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[4], 4);
        BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[5], 0);
        
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
            BOOST_CHECK_EQUAL_COLLECTIONS(nbrs.begin(), nbrs.end(), check_nbrs.begin(), check_nbrs.end());
            }
        }
    }
    
//! Tests the ability of the neighbor list to exclude particle pairs
template <class NL>
void neighborlist_exclusion_tests(boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    boost::shared_ptr<SystemDefinition> sysdef_6(new SystemDefinition(6, BoxDim(20.0, 40.0, 60.0), 1, 0, 0, 0, 0, exec_conf));
    boost::shared_ptr<ParticleData> pdata_6 = sysdef_6->getParticleData();

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

    boost::shared_ptr<NeighborList> nlist_6(new NL(sysdef_6, 3.0, 0.25));
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

//         BOOST_REQUIRE(nli.getW() >= 6);
        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[0], 1);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[0] + 0], 5);

        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[1], 4);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[1] + 0], 2);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[1] + 1], 3);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[1] + 2], 4);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[1] + 3], 5);

        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[2], 4);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[2] + 0], 1);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[2] + 1], 3);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[2] + 2], 4);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[2] + 3], 5);

        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[3], 4);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[3] + 0], 1);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[3] + 1], 2);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[3] + 2], 4);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[3] + 3], 5);

        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[4], 4);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[4] + 0], 1);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[4] + 1], 2);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[4] + 2], 3);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[4] + 3], 5);

        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[5], 5);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[5] + 0], 0);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[5] + 1], 1);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[5] + 2], 2);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[5] + 3], 3);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[5] + 4], 4);
        }
    }

//! Tests the ability of the neighbor list to exclude particles from the same body
template <class NL>
void neighborlist_body_filter_tests(boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    boost::shared_ptr<SystemDefinition> sysdef_6(new SystemDefinition(6, BoxDim(20.0, 40.0, 60.0), 1, 0, 0, 0, 0, exec_conf));
    boost::shared_ptr<ParticleData> pdata_6 = sysdef_6->getParticleData();

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

    // this test uses rigid bodies, initialize them
    sysdef_6->getRigidData()->initializeData();

    boost::shared_ptr<NeighborList> nlist_6(new NL(sysdef_6, 3.0, 0.25));
    nlist_6->setRCutPair(0,0,3.0);
    nlist_6->setFilterBody(true);
    nlist_6->setStorageMode(NeighborList::full);

    nlist_6->compute(0);
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist_6->getNNeighArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_nlist(nlist_6->getNListArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_head_list(nlist_6->getHeadList(), access_location::host, access_mode::read);

//         BOOST_REQUIRE(nli.getW() >= 6);
        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[0], 5);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[0] + 0], 1);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[0] + 1], 2);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[0] + 2], 3);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[0] + 3], 4);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[0] + 4], 5);

        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[1], 4);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[1] + 0], 0);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[1] + 1], 2);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[1] + 2], 4);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[1] + 3], 5);

        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[2], 4);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[2] + 0], 0);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[2] + 1], 1);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[2] + 2], 3);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[2] + 3], 5);

        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[3], 4);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[3] + 0], 0);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[3] + 1], 2);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[3] + 2], 4);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[3] + 3], 5);

        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[4], 4);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[4] + 0], 0);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[4] + 1], 1);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[4] + 2], 3);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[4] + 3], 5);

        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[5], 5);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[5] + 0], 0);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[5] + 1], 1);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[5] + 2], 2);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[5] + 3], 3);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[5] + 4], 4);
        }
    }

//! Tests the ability of the neighbor list to filter by diameter
template <class NL>
void neighborlist_diameter_shift_tests(boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    /////////////////////////////////////////////////////////
    // start with the simplest possible test: 3 particles in a huge box
    boost::shared_ptr<SystemDefinition> sysdef_3(new SystemDefinition(4, BoxDim(25.0), 1, 0, 0, 0, 0, exec_conf));
    boost::shared_ptr<ParticleData> pdata_3 = sysdef_3->getParticleData();

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
    boost::shared_ptr<NeighborList> nlist_2(new NL(sysdef_3, 1.5, 0.5));
    nlist_2->setRCutPair(0,0,1.5);
    nlist_2->compute(1);
    nlist_2->setStorageMode(NeighborList::full);

    // with the given settings, there should be no neighbors: check that
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist_2->getNNeighArray(), access_location::host, access_mode::read);

        BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[0], 0);
        BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[1], 0);
        BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[2], 0);
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

        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[0], 2);
            {
            vector<unsigned int> nbrs(2, 0);
            nbrs[0] = h_nlist.data[h_head_list.data[0] + 0];
            nbrs[1] = h_nlist.data[h_head_list.data[0] + 1];
            sort(nbrs.begin(), nbrs.end());
            unsigned int check_nbrs[] = {1,2};
            BOOST_CHECK_EQUAL_COLLECTIONS(nbrs.begin(), nbrs.end(), check_nbrs, check_nbrs + 2);
            }

        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[1], 1);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[1]], 0);

        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[2], 1);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[2]], 0);
        }
    }


//! Test two implementations of NeighborList and verify that the output is identical
template <class NLA, class NLB>
void neighborlist_comparison_test(boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // construct the particle system
    RandomInitializer init(1000, Scalar(0.016778), Scalar(0.9), "A");
    boost::shared_ptr< SnapshotSystemData<Scalar> > snap = init.getSnapshot();
    boost::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));
    boost::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    boost::shared_ptr<NeighborList> nlist1(new NLA(sysdef, Scalar(3.0), Scalar(0.4)));
    nlist1->setRCutPair(0,0,3.0);
    nlist1->setStorageMode(NeighborList::full);

    boost::shared_ptr<NeighborList> nlist2(new NLB(sysdef, Scalar(3.0), Scalar(0.4)));
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

    // temporary vectors for holding the lists: they will be sorted for comparison
    std::vector<unsigned int> tmp_list1;
    std::vector<unsigned int> tmp_list2;

    // check to make sure that every neighbor matches
    for (unsigned int i = 0; i < pdata->getN(); i++)
        {
        BOOST_REQUIRE_EQUAL(h_head_list1.data[i], h_head_list2.data[i]);
        BOOST_REQUIRE_EQUAL(h_n_neigh1.data[i], h_n_neigh2.data[i]);

        tmp_list1.resize(h_n_neigh1.data[i]);
        tmp_list2.resize(h_n_neigh1.data[i]);

        for (unsigned int j = 0; j < h_n_neigh1.data[i]; j++)
            {
            tmp_list1[j] = h_nlist1.data[h_head_list1.data[i] + j];
            tmp_list2[j] = h_nlist2.data[h_head_list2.data[i] + j];
            }

        sort(tmp_list1.begin(), tmp_list1.end());
        sort(tmp_list2.begin(), tmp_list2.end());
        
        BOOST_CHECK_EQUAL_COLLECTIONS(tmp_list1.begin(), tmp_list1.end(), tmp_list2.begin(), tmp_list2.end());
        }
    }

//! Test that a NeighborList can successfully exclude a ridiculously large number of particles
template <class NL>
void neighborlist_large_ex_tests(boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // construct the particle system
    RandomInitializer init(1000, Scalar(0.016778), Scalar(0.9), "A");
    boost::shared_ptr< SnapshotSystemData<Scalar> > snap = init.getSnapshot();
    boost::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));
    boost::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    boost::shared_ptr<NeighborList> nlist(new NL(sysdef, Scalar(8.0), Scalar(0.4)));
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
        BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[i], 0);
        }
    }

//! Test that NeighborList can exclude particles correctly when cutoff radius is negative
template <class NL>
void neighborlist_cutoff_exclude_tests(boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // Initialize a system of 3 particles each having a distinct type
    boost::shared_ptr<SystemDefinition> sysdef_3(new SystemDefinition(3, BoxDim(25.0), 3, 0, 0, 0, 0, exec_conf));
    boost::shared_ptr<ParticleData> pdata_3 = sysdef_3->getParticleData();
    
    // put the particles on top of each other, the worst case scenario for inclusion / exclusion since the distance
    // between them is zero
        {
        ArrayHandle<Scalar4> h_pos(pdata_3->getPositions(), access_location::host, access_mode::overwrite);
        for (unsigned int i=0; i < pdata_3->getN(); ++i)
            {
            h_pos.data[i] = make_scalar4(0.0, 0.0, 0.0, __int_as_scalar(i));
            }
        }

    boost::shared_ptr<NeighborList> nlist(new NL(sysdef_3, Scalar(-1.0), Scalar(0.4)));
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
        BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[0], 0);
        BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[1], 0);
        BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[2], 0);
        }
    
    // turn on cross interaction with B particle
    for (unsigned int i=0; i < pdata_3->getNTypes(); ++i)
        {
        nlist->setRCutPair(1, i, 1.0);
        }
    nlist->compute(1);
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist->getNNeighArray(), access_location::host, access_mode::read);
        BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[0], 1);
        BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[1], 2); // B ignores itself, but gets everyone else as a neighbor
        BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[2], 1);
        
        ArrayHandle<unsigned int> h_nlist(nlist->getNListArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_head_list(nlist->getHeadList(), access_location::host, access_mode::read);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[0]], 1);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[2]], 1);
        
        vector<unsigned int> nbrs(2, 0);
        nbrs[0] = h_nlist.data[h_head_list.data[1] + 0];
        nbrs[1] = h_nlist.data[h_head_list.data[1] + 1];
        sort(nbrs.begin(), nbrs.end());
        unsigned int check_nbrs[] = {0,2};
        BOOST_CHECK_EQUAL_COLLECTIONS(nbrs.begin(), nbrs.end(), check_nbrs, check_nbrs + 2);
        }

    // turn A-C on and B-C off with things very close to the < 0.0 criterion as a pathological case
    nlist->setRCutPair(0, 2, 0.00001);
    nlist->setRCutPair(1, 2, -0.00001);
    nlist->compute(3);
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist->getNNeighArray(), access_location::host, access_mode::read);
        BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[0], 2);
        BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[1], 1);
        BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[2], 1);
        
        ArrayHandle<unsigned int> h_nlist(nlist->getNListArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_head_list(nlist->getHeadList(), access_location::host, access_mode::read);

        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[1]], 0);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[h_head_list.data[2]], 0);

        vector<unsigned int> nbrs(2, 0);
        nbrs[0] = h_nlist.data[h_head_list.data[0] + 0];
        nbrs[1] = h_nlist.data[h_head_list.data[0] + 1];
        sort(nbrs.begin(), nbrs.end());
        unsigned int check_nbrs[] = {1,2};
        BOOST_CHECK_EQUAL_COLLECTIONS(nbrs.begin(), nbrs.end(), check_nbrs, check_nbrs + 2);        
        }
    }

//! basic test case for binned class
BOOST_AUTO_TEST_CASE( NeighborListBinned_basic )
    {
    neighborlist_basic_tests<NeighborListBinned>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
//! exclusion test case for binned class
BOOST_AUTO_TEST_CASE( NeighborListBinned_exclusion )
    {
    neighborlist_exclusion_tests<NeighborListBinned>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
//! large exclusion test case for binned class
BOOST_AUTO_TEST_CASE( NeighborListBinned_large_ex )
    {
    neighborlist_large_ex_tests<NeighborListBinned>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
//! body filter test case for binned class
BOOST_AUTO_TEST_CASE( NeighborListBinned_body_filter)
    {
    neighborlist_body_filter_tests<NeighborListBinned>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
//! diameter filter test case for binned class
BOOST_AUTO_TEST_CASE( NeighborListBinned_diameter_shift )
    {
    neighborlist_diameter_shift_tests<NeighborListBinned>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
//! particle asymmetry test case for binned class
BOOST_AUTO_TEST_CASE( NeighborListBinned_particle_asymm)
    {
    neighborlist_particle_asymm_tests<NeighborListBinned>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
//! cutoff exclusion test case for binned class
BOOST_AUTO_TEST_CASE( NeighborListBinned_cutoff_exclude)
    {
    neighborlist_cutoff_exclude_tests<NeighborListBinned>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
//! type test case for tree class
BOOST_AUTO_TEST_CASE( NeighborListBinned_type)
    {
    neighborlist_type_tests<NeighborListBinned>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

//! basic test case for tree class
BOOST_AUTO_TEST_CASE( NeighborListTree_basic )
    {
    neighborlist_basic_tests<NeighborListTree>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
//! exclusion test case for tree class
BOOST_AUTO_TEST_CASE( NeighborListTree_exclusion )
    {
    neighborlist_exclusion_tests<NeighborListTree>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
//! large exclusion test case for tree class
BOOST_AUTO_TEST_CASE( NeighborListTree_large_ex )
    {
    neighborlist_large_ex_tests<NeighborListTree>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
//! body filter test case for tree class
BOOST_AUTO_TEST_CASE( NeighborListTree_body_filter)
    {
    neighborlist_body_filter_tests<NeighborListTree>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
//! diameter filter test case for binned class
BOOST_AUTO_TEST_CASE( NeighborListTree_diameter_shift )
    {
    neighborlist_diameter_shift_tests<NeighborListTree>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
//! particle asymmetry test case for tree class
BOOST_AUTO_TEST_CASE( NeighborListTree_particle_asymm)
    {
    neighborlist_particle_asymm_tests<NeighborListTree>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
//! cutoff exclusion test case for tree class
BOOST_AUTO_TEST_CASE( NeighborListTree_cutoff_exclude)
    {
    neighborlist_cutoff_exclude_tests<NeighborListTree>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
//! type test case for tree class
BOOST_AUTO_TEST_CASE( NeighborListTree_type)
    {
    neighborlist_type_tests<NeighborListTree>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
//! comparison test case for binned class
BOOST_AUTO_TEST_CASE( NeighborListTree_comparison )
    {
    neighborlist_comparison_test<NeighborListBinned, NeighborListTree>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

#ifdef ENABLE_CUDA
//! basic test case for GPUBinned class
BOOST_AUTO_TEST_CASE( NeighborListGPUBinned_basic )
    {
    neighborlist_basic_tests<NeighborListGPUBinned>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
//! exclusion test case for GPUBinned class
BOOST_AUTO_TEST_CASE( NeighborListGPUBinned_exclusion )
    {
    neighborlist_exclusion_tests<NeighborListGPUBinned>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
//! large exclusion test case for GPUBinned class
BOOST_AUTO_TEST_CASE( NeighborListGPUBinned_large_ex )
    {
    neighborlist_large_ex_tests<NeighborListGPUBinned>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
//! body filter test case for GPUBinned class
BOOST_AUTO_TEST_CASE( NeighborListGPUBinned_body_filter)
    {
    neighborlist_body_filter_tests<NeighborListGPUBinned>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
//! diameter filter test case for binned class
BOOST_AUTO_TEST_CASE( NeighborListGPUBinned_diameter_shift )
    {
    neighborlist_diameter_shift_tests<NeighborListGPUBinned>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
//! comparison test case for GPUBinned class
BOOST_AUTO_TEST_CASE( NeighborListGPUBinned_comparison )
    {
    neighborlist_comparison_test<NeighborListBinned, NeighborListGPUBinned>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
//! particle asymmetry test case for GPUBinned class
BOOST_AUTO_TEST_CASE( NeighborListGPUBinned_particle_asymm)
    {
    neighborlist_particle_asymm_tests<NeighborListGPUBinned>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
//! cutoff exclusion test case for GPUBinned class
BOOST_AUTO_TEST_CASE( NeighborListGPUBinned_cutoff_exclude)
    {
    neighborlist_cutoff_exclude_tests<NeighborListGPUBinned>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
//! type test case for tree class
BOOST_AUTO_TEST_CASE( NeighborListGPUBinned_type)
    {
    neighborlist_type_tests<NeighborListGPUBinned>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! basic test case for GPUTree class
BOOST_AUTO_TEST_CASE( NeighborListGPUTree_basic )
    {
    boost::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::GPU));
    if (exec_conf->getComputeCapability() > 200)
        {
        neighborlist_basic_tests<NeighborListGPUTree>(exec_conf);
        }
    else
        {
        exec_conf->msg->notice(1) << "Skipping GPU tree basic test, unsupported" << endl;
        }
    }
//! exclusion test case for GPUTree class
BOOST_AUTO_TEST_CASE( NeighborListGPUTree_exclusion )
    {
    boost::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::GPU));
    if (exec_conf->getComputeCapability() > 200)
        {
        neighborlist_exclusion_tests<NeighborListGPUTree>(exec_conf);
        }
    else
        {
        exec_conf->msg->notice(1) << "Skipping GPU tree exclusion test, unsupported" << endl;
        }
    }
//! large exclusion test case for GPUTree class
BOOST_AUTO_TEST_CASE( NeighborListGPUTree_large_ex )
    {
    boost::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::GPU));
    if (exec_conf->getComputeCapability() > 200)
        {
        neighborlist_large_ex_tests<NeighborListGPUTree>(exec_conf);
        }
    else
        {
        exec_conf->msg->notice(1) << "Skipping GPU tree large exclusion test, unsupported" << endl;
        }
    }
//! body filter test case for GPUTree class
BOOST_AUTO_TEST_CASE( NeighborListGPUTree_body_filter)
    {
    boost::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::GPU));
    if (exec_conf->getComputeCapability() > 200)
        {
        neighborlist_body_filter_tests<NeighborListGPUTree>(exec_conf);
        }
    else
        {
        exec_conf->msg->notice(1) << "Skipping GPU tree body filter test, unsupported" << endl;
        }
    }
//! diameter filter test case for GPUTree class
BOOST_AUTO_TEST_CASE( NeighborListGPUTree_diameter_shift )
    {
    boost::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::GPU));
    if (exec_conf->getComputeCapability() > 200)
        {
        neighborlist_diameter_shift_tests<NeighborListGPUTree>(exec_conf);
        }
    else
        {
        exec_conf->msg->notice(1) << "Skipping GPU tree diameter shift test, unsupported" << endl;
        }
    }

//! particle asymmetry test case for GPUTree class
BOOST_AUTO_TEST_CASE( NeighborListGPUTree_particle_asymm)
    {
    boost::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::GPU));
    if (exec_conf->getComputeCapability() > 200)
        {
        neighborlist_particle_asymm_tests<NeighborListGPUTree>(exec_conf);
        }
    else
        {
        exec_conf->msg->notice(1) << "Skipping GPU tree particle asymm test, unsupported" << endl;
        }
    }
//! cutoff exclusion test case for GPUTree class
BOOST_AUTO_TEST_CASE( NeighborListGPUTree_cutoff_exclude)
    {
    boost::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::GPU));
    if (exec_conf->getComputeCapability() > 200)
        {
        neighborlist_cutoff_exclude_tests<NeighborListGPUTree>(exec_conf);
        }
    else
        {
        exec_conf->msg->notice(1) << "Skipping GPU tree cutoff exclusion test, unsupported" << endl;
        }
    }
//! type test case for tree class
BOOST_AUTO_TEST_CASE( NeighborListGPUTree_type)
    {
    boost::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::GPU));
    if (exec_conf->getComputeCapability() > 200)
        {
        neighborlist_type_tests<NeighborListGPUTree>(exec_conf);
        }
    else
        {
        exec_conf->msg->notice(1) << "Skipping GPU tree type test, unsupported" << endl;
        }
    }

//! comparison test case for GPUTree class with itself
BOOST_AUTO_TEST_CASE( NeighborListGPUTree_cpu_comparison )
    {
    boost::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::GPU));
    if (exec_conf->getComputeCapability() > 200)
        {
        neighborlist_comparison_test<NeighborListTree, NeighborListGPUTree>(exec_conf);
        }
    else
        {
        exec_conf->msg->notice(1) << "Skipping GPU tree CPU comparison test, unsupported" << endl;
        }
    }
    
//! comparison test case for GPUTree class with GPUBinned
BOOST_AUTO_TEST_CASE( NeighborListGPUTree_binned_comparison )
    {
    boost::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::GPU));
    if (exec_conf->getComputeCapability() > 200)
        {
        neighborlist_comparison_test<NeighborListGPUBinned, NeighborListGPUTree>(exec_conf);
        }
    else
        {
        exec_conf->msg->notice(1) << "Skipping GPU tree GPU comparison test, unsupported" << endl;
        }
    }
#endif
