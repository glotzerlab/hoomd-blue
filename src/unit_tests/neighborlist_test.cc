/*
Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
Source Software License
Copyright (c) 2008 Ames Laboratory Iowa State University
All rights reserved.

Redistribution and use of HOOMD, in source and binary forms, with or
without modification, are permitted, provided that the following
conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names HOOMD's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <iostream>
#include <algorithm>

//! Define the name of the boost test module
#define BOOST_TEST_MODULE NeighborListTest
#include "boost_utf_configure.h"

#include <boost/test/floating_point_comparison.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include "NeighborList.h"
#include "BinnedNeighborList.h"
#include "Initializers.h"

#ifdef USE_CUDA
#include "NeighborListNsqGPU.h"
#include "BinnedNeighborListGPU.h"
#endif

using namespace std;
using namespace boost;

//! helper macro for checking if two numbers are close
#define MY_BOOST_CHECK_CLOSE(a,b,c) BOOST_CHECK_CLOSE(a,Scalar(b),Scalar(c))

//! Shortcut for defining a factory
typedef boost::function<shared_ptr<NeighborList> (shared_ptr<ParticleData> pdata, Scalar r_cut, Scalar r_buff)> nlist_creator_typ;

//! Performs basic functionality tests on a neighbor list
void neighborlist_basic_tests(nlist_creator_typ nlist_creator, ExecutionConfiguration exec_conf)
	{
	/////////////////////////////////////////////////////////
	// start with the simplest possible test: 2 particles in a huge box
	shared_ptr<ParticleData> pdata_2(new ParticleData(2, BoxDim(25.0), 1, 0, exec_conf));
	ParticleDataArrays arrays = pdata_2->acquireReadWrite();
	arrays.x[0] = arrays.y[0] = arrays.z[0] = 0.0;
	arrays.x[1] = arrays.y[1] = arrays.z[1] = 3.25;
	pdata_2->release();

	// test construction of the neighborlist
	shared_ptr<NeighborList> nlist_2 = nlist_creator(pdata_2, 3.0, 0.25);
	nlist_2->compute(1);

	// with the given radius, there should be no neighbors: check that
	std::vector< std::vector<unsigned int> > list = nlist_2->getList();
	BOOST_CHECK(list[0].size() == 0);
	BOOST_CHECK(list[1].size() == 0);

	// adjust the radius to include the particles and see if we get some now
	nlist_2->setRCut(5.5, 0.5);
	nlist_2->compute(2);
	// some neighbor lists default to full because they don't support half: ignore them
	if (nlist_2->getStorageMode() == NeighborList::half)
		{
		list = nlist_2->getList();
		BOOST_REQUIRE(list[0].size() == 1);
		BOOST_CHECK(list[0][0] == 1);
		BOOST_CHECK(list[1].size() == 0);	//< since this is a half list, only 0 stores 1 as a neighbor
		}
		
	// change to full mode to check that
	nlist_2->setStorageMode(NeighborList::full);
	nlist_2->compute(3);
	list = nlist_2->getList();
	BOOST_REQUIRE(list[0].size() == 1);
	BOOST_CHECK(list[0][0] == 1);
	BOOST_REQUIRE(list[1].size() == 1);
	BOOST_CHECK(list[1][0] == 0);


	////////////////////////////////////////////////////////////////////
	// now, lets do a more thorough test and include boundary conditions
	// there are way too many permutations to test here, so I will simply
	// test +x, -x, +y, -y, +z, and -z independantly
	// build a 6 particle system with particles across each boundary
	shared_ptr<ParticleData> pdata_6(new ParticleData(6, BoxDim(20.0, 40.0, 60.0), 1, 0, exec_conf));
	arrays = pdata_6->acquireReadWrite();
	arrays.x[0] = Scalar(-9.6); arrays.y[0] = 0; arrays.z[0] = 0.0;
	arrays.x[1] =  Scalar(9.6); arrays.y[1] = 0; arrays.z[1] = 0.0;
	arrays.x[2] = 0; arrays.y[2] = Scalar(-19.6); arrays.z[2] = 0.0;
	arrays.x[3] = 0; arrays.y[3] = Scalar(19.6); arrays.z[3] = 0.0;
	arrays.x[4] = 0; arrays.y[4] = 0; arrays.z[4] = Scalar(-29.6);
	arrays.x[5] = 0; arrays.y[5] = 0; arrays.z[5] =  Scalar(29.6);
	pdata_6->release();

	shared_ptr<NeighborList> nlist_6 = nlist_creator(pdata_6, 3.0, 0.25);
	nlist_6->setStorageMode(NeighborList::full);
	nlist_6->compute(0);
	// verify the neighbor list
	list = nlist_6->getList();
	BOOST_REQUIRE(list.size() == 6);
	BOOST_REQUIRE(list[0].size() == 1);  BOOST_CHECK(list[0][0] == 1);
	BOOST_REQUIRE(list[1].size() == 1);  BOOST_CHECK(list[1][0] == 0);
	BOOST_REQUIRE(list[2].size() == 1);  BOOST_CHECK(list[2][0] == 3);
	BOOST_REQUIRE(list[3].size() == 1);  BOOST_CHECK(list[3][0] == 2);
	BOOST_REQUIRE(list[4].size() == 1);  BOOST_CHECK(list[4][0] == 5);
	BOOST_REQUIRE(list[5].size() == 1);  BOOST_CHECK(list[5][0] == 4);

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
	list = nlist_6->getList();
	BOOST_REQUIRE(list.size() == 6);
	BOOST_REQUIRE(list[0].size() == 1);  BOOST_CHECK(list[0][0] == 1);
	BOOST_REQUIRE(list[1].size() == 1);  BOOST_CHECK(list[1][0] == 0);
	BOOST_REQUIRE(list[2].size() == 1);  BOOST_CHECK(list[2][0] == 3);
	BOOST_REQUIRE(list[3].size() == 1);  BOOST_CHECK(list[3][0] == 2);
	BOOST_REQUIRE(list[4].size() == 1);  BOOST_CHECK(list[4][0] == 5);
	BOOST_REQUIRE(list[5].size() == 1);  BOOST_CHECK(list[5][0] == 4);

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
	list = nlist_6->getList();
	BOOST_REQUIRE(list.size() == 6);
	BOOST_REQUIRE(list[0].size() == 3);
	BOOST_CHECK(list[0][0] == 1);
	BOOST_CHECK(list[0][1] == 4);
	BOOST_CHECK(list[0][2] == 5);
	}

//! Tests the ability of the neighbor list to exclude particle pairs
void neighborlist_exclusion_tests(nlist_creator_typ nlist_creator, ExecutionConfiguration exec_conf)
	{
	shared_ptr<ParticleData> pdata_6(new ParticleData(6, BoxDim(20.0, 40.0, 60.0), 1, 0, exec_conf));
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

	shared_ptr<NeighborList> nlist_6 = nlist_creator(pdata_6, 3.0, 0.25);
	nlist_6->setStorageMode(NeighborList::full);
	nlist_6->addExclusion(0,1);
	nlist_6->addExclusion(0,2);
	nlist_6->addExclusion(0,3);
	nlist_6->addExclusion(0,4);

	nlist_6->compute(0);
	std::vector< std::vector<unsigned int> > list = nlist_6->getList();
	BOOST_REQUIRE(list.size() == 6);
	BOOST_REQUIRE(list[0].size() == 1);
	BOOST_CHECK(list[0][0] == 5);
	BOOST_REQUIRE(list[1].size() == 4);
	BOOST_CHECK(list[1][0] == 2);
	BOOST_CHECK(list[1][1] == 3);
	BOOST_CHECK(list[1][2] == 4);
	BOOST_CHECK(list[1][3] == 5);

	BOOST_REQUIRE(list[2].size() == 4);
	BOOST_CHECK(list[2][0] == 1);
	BOOST_CHECK(list[2][1] == 3);
	BOOST_CHECK(list[2][2] == 4);
	BOOST_CHECK(list[2][3] == 5);

	BOOST_REQUIRE(list[3].size() == 4);
	BOOST_CHECK(list[3][0] == 1);
	BOOST_CHECK(list[3][1] == 2);
	BOOST_CHECK(list[3][2] == 4);
	BOOST_CHECK(list[3][3] == 5);

	BOOST_REQUIRE(list[4].size() == 4);
	BOOST_CHECK(list[4][0] == 1);
	BOOST_CHECK(list[4][1] == 2);
	BOOST_CHECK(list[4][2] == 3);
	BOOST_CHECK(list[4][3] == 5);

	BOOST_REQUIRE(list[5].size() == 5);
	BOOST_CHECK(list[5][0] == 0);
	BOOST_CHECK(list[5][1] == 1);
	BOOST_CHECK(list[5][2] == 2);
	BOOST_CHECK(list[5][3] == 3);
	BOOST_CHECK(list[5][4] == 4);
	}

//! Test two implementations of NeighborList and verify that the output is identical
void neighborlist_comparison_test(nlist_creator_typ nlist_creator1, nlist_creator_typ nlist_creator2, ExecutionConfiguration exec_conf)
	{
	// construct the particle system
	RandomInitializer init(1000, Scalar(0.016778), Scalar(0.9), "A");
	
	shared_ptr<ParticleData> pdata(new ParticleData(init, exec_conf));
		
	shared_ptr<NeighborList> nlist1 = nlist_creator1(pdata, Scalar(3.0), Scalar(0.4));
	nlist1->setStorageMode(NeighborList::full);
	
	shared_ptr<NeighborList> nlist2 = nlist_creator2(pdata, Scalar(3.0), Scalar(0.4));
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
	std::vector< std::vector<unsigned int> > list1 = nlist1->getList();
	std::vector< std::vector<unsigned int> > list2 = nlist2->getList();
	
	// temporary vectors for holding the lists: they will be sorted for comparison
	std::vector<unsigned int> tmp_list1;
	std::vector<unsigned int> tmp_list2;
	
	// check to make sure that every neighbor matches
	for (unsigned int i = 0; i < pdata->getN(); i++)
		{
		BOOST_REQUIRE(list1[i].size() == list2[i].size());
		tmp_list1 = list1[i];
		tmp_list2 = list2[i];
		sort(tmp_list1.begin(), tmp_list1.end());
		sort(tmp_list2.begin(), tmp_list2.end());
		
		for (unsigned int j = 0; j < tmp_list1.size(); j++)
			{
			BOOST_CHECK(tmp_list1[j] == tmp_list2[j]);
			}
		}
	}

// define the creators
shared_ptr<NeighborList> base_class_nlist_creator(shared_ptr<ParticleData> pdata, Scalar r_cut, Scalar r_buff)
	{
	return shared_ptr<NeighborList>(new NeighborList(pdata, r_cut, r_buff));
	}
	
shared_ptr<NeighborList> binned_nlist_creator(shared_ptr<ParticleData> pdata, Scalar r_cut, Scalar r_buff)
	{
	return shared_ptr<NeighborList>(new NeighborList(pdata, r_cut, r_buff));
	}
	
#ifdef USE_CUDA
shared_ptr<NeighborList> gpu_nsq_nlist_creator(shared_ptr<ParticleData> pdata, Scalar r_cut, Scalar r_buff)
	{
	return shared_ptr<NeighborList>(new NeighborListNsqGPU(pdata, r_cut, r_buff));
	}
shared_ptr<NeighborList> gpu_binned_nlist_creator(shared_ptr<ParticleData> pdata, Scalar r_cut, Scalar r_buff)
	{
	shared_ptr<BinnedNeighborListGPU> nlist(new BinnedNeighborListGPU(pdata, r_cut, r_buff));
	// the default block size kills valgrind :) reduce it
	nlist->setBlockSize(64);
	return nlist;
	}
#endif

// now it is time for all the tests

//! boost test case for base class
BOOST_AUTO_TEST_CASE( NeighborList_tests )
	{
	nlist_creator_typ base_creator = bind(base_class_nlist_creator, _1, _2, _3);
	neighborlist_basic_tests(base_creator, ExecutionConfiguration(ExecutionConfiguration::CPU, 0));
	neighborlist_exclusion_tests(base_creator, ExecutionConfiguration(ExecutionConfiguration::CPU, 0));
	}

//! boost test case for BinnedNeighborList
BOOST_AUTO_TEST_CASE( BinnedNeighborList_tests )
	{
	nlist_creator_typ base_creator = bind(base_class_nlist_creator, _1, _2, _3);
	nlist_creator_typ binned_creator = bind(binned_nlist_creator, _1, _2, _3);
	
	neighborlist_basic_tests(binned_creator, ExecutionConfiguration(ExecutionConfiguration::CPU, 0));
	neighborlist_exclusion_tests(binned_creator, ExecutionConfiguration(ExecutionConfiguration::CPU, 0));
	neighborlist_comparison_test(base_creator, binned_creator, ExecutionConfiguration(ExecutionConfiguration::CPU, 0));
	}
	
#ifdef USE_CUDA
//! boost test case for NeighborListNsqGPU
BOOST_AUTO_TEST_CASE( NeighborListNsqGPU_tests )
	{
	nlist_creator_typ base_creator = bind(base_class_nlist_creator, _1, _2, _3);
	nlist_creator_typ gpu_creator = bind(gpu_nsq_nlist_creator, _1, _2, _3);
	
	neighborlist_basic_tests(gpu_creator, ExecutionConfiguration(ExecutionConfiguration::GPU, 0));
	neighborlist_exclusion_tests(gpu_creator, ExecutionConfiguration(ExecutionConfiguration::GPU, 0));
	neighborlist_comparison_test(base_creator, gpu_creator, ExecutionConfiguration(ExecutionConfiguration::GPU, 0));
	}
	
//! boost test case for BinnedNeighborListGPU
BOOST_AUTO_TEST_CASE( BinnedNeighborListGPU_tests )
	{
	nlist_creator_typ base_creator = bind(base_class_nlist_creator, _1, _2, _3);
	nlist_creator_typ gpu_creator = bind(gpu_binned_nlist_creator, _1, _2, _3);
	
	neighborlist_basic_tests(gpu_creator, ExecutionConfiguration(ExecutionConfiguration::GPU, 0));
	neighborlist_exclusion_tests(gpu_creator, ExecutionConfiguration(ExecutionConfiguration::GPU, 0));
	neighborlist_comparison_test(base_creator, gpu_creator, ExecutionConfiguration(ExecutionConfiguration::GPU, 0));
	}

//! boost test case for BinnedNeighborListGPU multi-gpu
BOOST_AUTO_TEST_CASE( BinnedNeighborListMultiGPU_tests )
	{
	nlist_creator_typ base_creator = bind(base_class_nlist_creator, _1, _2, _3);
	nlist_creator_typ gpu_creator = bind(gpu_binned_nlist_creator, _1, _2, _3);
	
	vector<unsigned int> gpu_list;
	gpu_list.push_back(0);
	gpu_list.push_back(0);
	gpu_list.push_back(0);
	gpu_list.push_back(0);
	ExecutionConfiguration exec_conf(ExecutionConfiguration::GPU, gpu_list);
	
	neighborlist_comparison_test(base_creator, gpu_creator, exec_conf);
	}

#endif

#ifdef WIN32
#pragma warning( pop )
#endif




