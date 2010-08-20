/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
// Maintainer: joaander

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <iostream>
#include <algorithm>

#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include "NeighborList.h"
#include "BinnedNeighborList.h"
#include "Initializers.h"

#ifdef ENABLE_CUDA
#include "NeighborListNsqGPU.h"
#include "BinnedNeighborListGPU.h"
#include "NeighborListGPU.h"
#include "NeighborListGPUBinned.h"
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
    shared_ptr<SystemDefinition> sysdef_2(new SystemDefinition(2, BoxDim(25.0), 1, 0, 0, 0, 0, exec_conf));
    shared_ptr<ParticleData> pdata_2 = sysdef_2->getParticleData();
    
    ParticleDataArrays arrays = pdata_2->acquireReadWrite();
    arrays.x[0] = arrays.y[0] = arrays.z[0] = 0.0;
    arrays.x[1] = arrays.y[1] = arrays.z[1] = 3.25;
    pdata_2->release();
    
    // test construction of the neighborlist
    shared_ptr<NeighborList> nlist_2(new NL(sysdef_2, 3.0, 0.25));
    nlist_2->compute(1);
    
    // with the given radius, there should be no neighbors: check that
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist_2->getNNeighArray(), access_location::host, access_mode::read);
        
        BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[0], 0);
        BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[1], 0);
        }
    
    // adjust the radius to include the particles and see if we get some now
    nlist_2->setRCut(5.5, 0.5);
    nlist_2->compute(2);
    // some neighbor lists default to full because they don't support half: ignore them
    if (nlist_2->getStorageMode() == NeighborList::half)
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist_2->getNNeighArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_nlist(nlist_2->getNListArray(), access_location::host, access_mode::read);
        Index2D nli = nlist_2->getNListIndexer();
        
        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[0], 1);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[nli(0,0)], 1);
        // since this is a half list, only 0 stores 1 as a neighbor
        BOOST_CHECK_EQUAL_UINT(h_n_neigh.data[1], 0);
        }
        
    // change to full mode to check that
    nlist_2->setStorageMode(NeighborList::full);
    nlist_2->compute(3);
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist_2->getNNeighArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_nlist(nlist_2->getNListArray(), access_location::host, access_mode::read);
        Index2D nli = nlist_2->getNListIndexer();
        
        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[0], 1);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[nli(0,0)], 1);

        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[1], 1);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[nli(1,0)], 0);
        }
    
    
    ////////////////////////////////////////////////////////////////////
    // now, lets do a more thorough test and include boundary conditions
    // there are way too many permutations to test here, so I will simply
    // test +x, -x, +y, -y, +z, and -z independantly
    // build a 6 particle system with particles across each boundary
    
    shared_ptr<SystemDefinition> sysdef_6(new SystemDefinition(6, BoxDim(20.0, 40.0, 60.0), 1, 0, 0, 0, 0, exec_conf));
    shared_ptr<ParticleData> pdata_6 = sysdef_6->getParticleData();
    
    arrays = pdata_6->acquireReadWrite();
    arrays.x[0] = Scalar(-9.6); arrays.y[0] = 0; arrays.z[0] = 0.0;
    arrays.x[1] =  Scalar(9.6); arrays.y[1] = 0; arrays.z[1] = 0.0;
    arrays.x[2] = 0; arrays.y[2] = Scalar(-19.6); arrays.z[2] = 0.0;
    arrays.x[3] = 0; arrays.y[3] = Scalar(19.6); arrays.z[3] = 0.0;
    arrays.x[4] = 0; arrays.y[4] = 0; arrays.z[4] = Scalar(-29.6);
    arrays.x[5] = 0; arrays.y[5] = 0; arrays.z[5] =  Scalar(29.6);
    pdata_6->release();
    
    shared_ptr<NeighborList> nlist_6(new NL(sysdef_6, 3.0, 0.25));
    nlist_6->setStorageMode(NeighborList::full);
    nlist_6->compute(0);
    // verify the neighbor list
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist_6->getNNeighArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_nlist(nlist_6->getNListArray(), access_location::host, access_mode::read);
        Index2D nli = nlist_6->getNListIndexer();
        
        BOOST_REQUIRE(nli.getW() >= 6);
        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[0], 1);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[nli(0,0)], 1);
        
        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[1], 1);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[nli(1,0)], 0);
        
        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[2], 1);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[nli(2,0)], 3);
        
        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[3], 1);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[nli(3,0)], 2);
        
        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[4], 1);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[nli(4,0)], 5);
        
        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[5], 1);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[nli(5,0)], 4);
        }
    
    // swap the order of the particles around to look for subtle directional bugs
    arrays = pdata_6->acquireReadWrite();
    arrays.x[1] = Scalar(-9.6); arrays.y[1] = 0; arrays.z[1] = 0.0;
    arrays.x[0] =  Scalar(9.6); arrays.y[0] = 0; arrays.z[0] = 0.0;
    arrays.x[3] = 0; arrays.y[3] = Scalar(-19.6); arrays.z[3] = 0.0;
    arrays.x[2] = 0; arrays.y[2] = Scalar(19.6); arrays.z[2] = 0.0;
    arrays.x[5] = 0; arrays.y[5] = 0; arrays.z[5] = Scalar(-29.6);
    arrays.x[4] = 0; arrays.y[4] = 0; arrays.z[4] =  Scalar(29.6);
    pdata_6->release();
    
    // verify the neighbor list
    nlist_6->compute(1);
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist_6->getNNeighArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_nlist(nlist_6->getNListArray(), access_location::host, access_mode::read);
        Index2D nli = nlist_6->getNListIndexer();
        
        BOOST_REQUIRE(nli.getW() >= 6);
        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[0], 1);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[nli(0,0)], 1);
        
        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[1], 1);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[nli(1,0)], 0);
        
        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[2], 1);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[nli(2,0)], 3);
        
        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[3], 1);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[nli(3,0)], 2);
        
        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[4], 1);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[nli(4,0)], 5);
        
        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[5], 1);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[nli(5,0)], 4);
        }
    
    // one last test, we should check that more than one neighbor can be generated
    arrays = pdata_6->acquireReadWrite();
    arrays.x[0] = 0; arrays.y[0] = 0; arrays.z[0] = 0.0;
    arrays.x[1] = 0; arrays.y[1] = 0; arrays.z[1] = 0.0;
    arrays.x[2] = 0; arrays.y[2] = Scalar(-19.6); arrays.z[2] = 0.0;
    arrays.x[3] = 0; arrays.y[3] = Scalar(19.6); arrays.z[3] = 0.0;
    arrays.x[4] = 0; arrays.y[4] = 0; arrays.z[4] = 0;
    arrays.x[5] = 0; arrays.y[5] = 0; arrays.z[5] =  0;
    pdata_6->release();
    
    nlist_6->compute(20);
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist_6->getNNeighArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_nlist(nlist_6->getNListArray(), access_location::host, access_mode::read);
        Index2D nli = nlist_6->getNListIndexer();
        
        BOOST_REQUIRE(nli.getW() >= 6);
        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[0], 3);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[nli(0,0)], 1);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[nli(0,1)], 4);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[nli(0,2)], 5);
        }
    }

//! Tests the ability of the neighbor list to exclude particle pairs
template <class NL>
void neighborlist_exclusion_tests(boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    shared_ptr<SystemDefinition> sysdef_6(new SystemDefinition(6, BoxDim(20.0, 40.0, 60.0), 1, 0, 0, 0, 0, exec_conf));
    shared_ptr<ParticleData> pdata_6 = sysdef_6->getParticleData();
    
    // lets make this test simple: put all 6 particles on top of each other and
    // see if the exclusion code can ignore 4 of the particles
    ParticleDataArrays arrays = pdata_6->acquireReadWrite();
    arrays.x[0] = 0; arrays.y[0] = 0; arrays.z[0] = 0.0;
    arrays.x[1] = 0; arrays.y[1] = 0; arrays.z[1] = 0.0;
    arrays.x[2] = 0; arrays.y[2] = 0; arrays.z[2] = 0.0;
    arrays.x[3] = 0; arrays.y[3] = 0; arrays.z[3] = 0.0;
    arrays.x[4] = 0; arrays.y[4] = 0; arrays.z[4] = 0;
    arrays.x[5] = 0; arrays.y[5] = 0; arrays.z[5] =  0;
    pdata_6->release();
    
    shared_ptr<NeighborList> nlist_6(new NL(sysdef_6, 3.0, 0.25));
    nlist_6->setStorageMode(NeighborList::full);
    nlist_6->addExclusion(0,1);
    nlist_6->addExclusion(0,2);
    nlist_6->addExclusion(0,3);
    nlist_6->addExclusion(0,4);
    
    nlist_6->compute(0);
        {
        ArrayHandle<unsigned int> h_n_neigh(nlist_6->getNNeighArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_nlist(nlist_6->getNListArray(), access_location::host, access_mode::read);
        Index2D nli = nlist_6->getNListIndexer();
        
        BOOST_REQUIRE(nli.getW() >= 6);
        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[0], 1);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[nli(0,0)], 5);
        
        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[1], 4);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[nli(1,0)], 2);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[nli(1,1)], 3);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[nli(1,2)], 4);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[nli(1,3)], 5);
        
        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[2], 4);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[nli(2,0)], 1);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[nli(2,1)], 3);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[nli(2,2)], 4);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[nli(2,3)], 5);

        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[3], 4);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[nli(3,0)], 1);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[nli(3,1)], 2);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[nli(3,2)], 4);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[nli(3,3)], 5);
    
        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[4], 4);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[nli(4,0)], 1);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[nli(4,1)], 2);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[nli(4,2)], 3);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[nli(4,3)], 5);

        BOOST_REQUIRE_EQUAL_UINT(h_n_neigh.data[5], 5);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[nli(5,0)], 0);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[nli(5,1)], 1);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[nli(5,2)], 2);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[nli(5,3)], 3);
        BOOST_CHECK_EQUAL_UINT(h_nlist.data[nli(5,4)], 4);
        }
    }

//! Test two implementations of NeighborList and verify that the output is identical
template <class NLA, class NLB>
void neighborlist_comparison_test(boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // construct the particle system
    RandomInitializer init(1000, Scalar(0.016778), Scalar(0.9), "A");
    
    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(init, exec_conf));
    shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    
    shared_ptr<NeighborList> nlist1(new NLA(sysdef, Scalar(3.0), Scalar(0.4)));
    nlist1->setStorageMode(NeighborList::full);
    
    shared_ptr<NeighborList> nlist2(new NLB(sysdef, Scalar(3.0), Scalar(0.4)));
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
    ArrayHandle<unsigned int> h_n_neigh2(nlist2->getNNeighArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_nlist2(nlist2->getNListArray(), access_location::host, access_mode::read);
    Index2D nli = nlist1->getNListIndexer();
    
    // temporary vectors for holding the lists: they will be sorted for comparison
    std::vector<unsigned int> tmp_list1;
    std::vector<unsigned int> tmp_list2;
    
    // check to make sure that every neighbor matches
    for (unsigned int i = 0; i < pdata->getN(); i++)
        {
        BOOST_REQUIRE_EQUAL(h_n_neigh1.data[i], h_n_neigh2.data[i]);
        
        tmp_list1.resize(h_n_neigh1.data[i]);
        tmp_list2.resize(h_n_neigh1.data[i]);
        
        for (unsigned int j = 0; j < h_n_neigh1.data[i]; j++)
            {
            tmp_list1[j] = h_nlist1.data[nli(i,j)];
            tmp_list2[j] = h_nlist2.data[nli(i,j)];
            }
        
        sort(tmp_list1.begin(), tmp_list1.end());
        sort(tmp_list2.begin(), tmp_list2.end());
        
        for (unsigned int j = 0; j < tmp_list1.size(); j++)
            {
            BOOST_CHECK_EQUAL(tmp_list1[j], tmp_list2[j]);
            }
        }
    }

//! basic test case for base class
BOOST_AUTO_TEST_CASE( NeighborList_basic )
    {
    neighborlist_basic_tests<NeighborList>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
//! exclusion test case for base class
BOOST_AUTO_TEST_CASE( NeighborList_exclusion )
    {
    neighborlist_exclusion_tests<NeighborList>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

#ifdef ENABLE_CUDA

//! basic test case for GPU class
BOOST_AUTO_TEST_CASE( NeighborListGPU_basic )
    {
    neighborlist_basic_tests<NeighborListGPU>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
//! exclusion test case for GPU class
BOOST_AUTO_TEST_CASE( NeighborListGPU_exclusion )
    {
    neighborlist_exclusion_tests<NeighborListGPU>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
//! comparison test case for GPU class
BOOST_AUTO_TEST_CASE( NeighborListGPU_comparison )
    {
    neighborlist_comparison_test<NeighborList, NeighborListGPU>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

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
//! comparison test case for GPUBinned class
BOOST_AUTO_TEST_CASE( NeighborListGPUBinned_comparison )
    {
    neighborlist_comparison_test<NeighborList, NeighborListGPUBinned>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

#endif

/*//! boost test case for BinnedNeighborList
BOOST_AUTO_TEST_CASE( BinnedNeighborList_tests )
    {
    nlist_creator_typ base_creator = bind(base_class_nlist_creator, _1, _2, _3);
    nlist_creator_typ binned_creator = bind(binned_nlist_creator, _1, _2, _3);
    
    neighborlist_basic_tests(binned_creator, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    neighborlist_exclusion_tests(binned_creator, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    neighborlist_comparison_test(base_creator, binned_creator, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

#ifdef ENABLE_CUDA
//! boost test case for NeighborListNsqGPU
BOOST_AUTO_TEST_CASE( NeighborListNsqGPU_tests )
    {
    nlist_creator_typ base_creator = bind(base_class_nlist_creator, _1, _2, _3);
    nlist_creator_typ gpu_creator = bind(gpu_nsq_nlist_creator, _1, _2, _3);
    
    neighborlist_basic_tests(gpu_creator, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    neighborlist_exclusion_tests(gpu_creator, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    neighborlist_comparison_test(base_creator, gpu_creator, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! boost test case for BinnedNeighborListGPU
BOOST_AUTO_TEST_CASE( BinnedNeighborListGPU_tests )
    {
    nlist_creator_typ base_creator = bind(base_class_nlist_creator, _1, _2, _3);
    nlist_creator_typ gpu_creator = bind(gpu_binned_nlist_creator, _1, _2, _3);
    
    neighborlist_basic_tests(gpu_creator, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    neighborlist_exclusion_tests(gpu_creator, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    neighborlist_comparison_test(base_creator, gpu_creator, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

#endif*/

#ifdef WIN32
#pragma warning( pop )
#endif




