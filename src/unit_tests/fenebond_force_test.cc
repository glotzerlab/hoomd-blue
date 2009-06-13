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

// $Id: fenebond_force_test.cc 1240 2008-09-19 17:19:48Z phillicl $
// $URL: http://svn2.assembla.com/svn/hoomd/trunk/src/unit_tests/fenebond_force_test.cc $
// Maintainer: phillicl

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <iostream>

//! Name the boost unit test module
#define BOOST_TEST_MODULE BondForceTests
#include "boost_utf_configure.h"

#include <boost/test/floating_point_comparison.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>

#include "FENEBondForceCompute.h"
#include "ConstForceCompute.h"
#ifdef ENABLE_CUDA
#include "FENEBondForceComputeGPU.h"
#endif

#include "Initializers.h"

using namespace std;
using namespace boost;

/*! \file fenebond_force_test.cc
	\brief Implements unit tests for BondForceCompute and child classes
	\ingroup unit_tests
*/

//! Helper macro for testing if two numbers are close
#define MY_BOOST_CHECK_CLOSE(a,b,c) BOOST_CHECK_CLOSE(a,Scalar(b),Scalar(c))
//! Helper macro for testing if a number is small
#define MY_BOOST_CHECK_SMALL(a,c) BOOST_CHECK_SMALL(a,Scalar(c))

//! Global tolerance for floating point comparisons
#ifdef SINGLE_PRECISION
const Scalar tol = Scalar(1e-1);
#else
const Scalar tol = 1e-3;
#endif
//! Global tolerance for check_small comparisons
const Scalar tol_small = Scalar(1e-4);

//! Typedef to make using the boost::function factory easier
typedef boost::function<shared_ptr<FENEBondForceCompute>  (shared_ptr<SystemDefinition> sysdef)> bondforce_creator;

//! Perform some simple functionality tests of any BondForceCompute
void bond_force_basic_tests(bondforce_creator bf_creator, ExecutionConfiguration exec_conf)
	{
	#ifdef CUDA
	g_gpu_error_checking = true;
	#endif
	
	/////////////////////////////////////////////////////////
	// start with the simplest possible test: 2 particles in a huge box with only one bond type
	shared_ptr<SystemDefinition> sysdef_2(new SystemDefinition(2, BoxDim(1000.0), 1, 1, 0, 0, 0,  exec_conf));
	shared_ptr<ParticleData> pdata_2 = sysdef_2->getParticleData();

	ParticleDataArrays arrays = pdata_2->acquireReadWrite();
	arrays.x[0] = arrays.y[0] = arrays.z[0] = 0.0;
	arrays.x[1] = Scalar(0.9);
	arrays.y[1] = arrays.z[1] = 0.0;
	pdata_2->release();

	// create the bond force compute to check
	shared_ptr<FENEBondForceCompute> fc_2 = bf_creator(sysdef_2);
	fc_2->setParams(0, Scalar(1.5), Scalar(1.1), Scalar(1.0), Scalar(1.0/4.0));

	// compute the force and check the results
	fc_2->compute(0);
	ForceDataArrays force_arrays = fc_2->acquire();
	// check that the force is correct, it should be 0 since we haven't created any bonds yet
	MY_BOOST_CHECK_SMALL(force_arrays.fx[0], tol_small);
	MY_BOOST_CHECK_SMALL(force_arrays.fy[0], tol_small);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[0], tol_small);
	MY_BOOST_CHECK_SMALL(force_arrays.pe[0], tol_small);
	MY_BOOST_CHECK_SMALL(force_arrays.virial[0], tol_small);
	
	// add a bond and check again
	sysdef_2->getBondData()->addBond(Bond(0, 0, 1));
	fc_2->compute(1);
	
	// this time there should be a force
	force_arrays = fc_2->acquire();
	MY_BOOST_CHECK_CLOSE(force_arrays.fx[0], -30.581156, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fy[0], tol_small);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[0], tol_small);	
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[0], 1.33177578 + 0.25/2, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.virial[0], 4.58717, tol);
		
	// check that the two forces are negatives of each other
	MY_BOOST_CHECK_CLOSE(force_arrays.fx[0], -force_arrays.fx[1], tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[0], -force_arrays.fy[1], tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[0], -force_arrays.fz[1], tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[0], force_arrays.pe[1], tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.virial[1], 4.58717, tol);

	// rearrange the two particles in memory and see if they are properly updated
	arrays = pdata_2->acquireReadWrite();
	arrays.x[0] = Scalar(0.9);
	arrays.x[1] = Scalar(0.0);
	arrays.tag[0] = 1;
	arrays.tag[1] = 0;
	arrays.rtag[0] = 1;
	arrays.rtag[1] = 0;
	pdata_2->release();

	// notify that we made the sort
	pdata_2->notifyParticleSort();
	// recompute at the same timestep, the forces should still be updated
	fc_2->compute(1);
	
	// this time there should be a force
	force_arrays = fc_2->acquire();
	MY_BOOST_CHECK_CLOSE(force_arrays.fx[0], 30.581156, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fx[1], -30.581156, tol);

	////////////////////////////////////////////////////////////////////
	// now, lets do a more thorough test and include boundary conditions
	// there are way too many permutations to test here, so I will simply
	// test +x, -x, +y, -y, +z, and -z independantly
	// build a 6 particle system with particles across each boundary
	// also test more than one type of bond
	shared_ptr<SystemDefinition> sysdef_6(new SystemDefinition(6, BoxDim(20.0, 40.0, 60.0), 1, 3, 0, 0, 0, exec_conf));
	shared_ptr<ParticleData> pdata_6 = sysdef_6->getParticleData();
	
	arrays = pdata_6->acquireReadWrite();
	arrays.x[0] = Scalar(-9.6); arrays.y[0] = 0; arrays.z[0] = 0.0;
	arrays.x[1] =  Scalar(9.6); arrays.y[1] = 0; arrays.z[1] = 0.0;
	arrays.x[2] = 0; arrays.y[2] = Scalar(-19.6); arrays.z[2] = 0.0;
	arrays.x[3] = 0; arrays.y[3] = Scalar(19.6); arrays.z[3] = 0.0;
	arrays.x[4] = 0; arrays.y[4] = 0; arrays.z[4] = Scalar(-29.6);
	arrays.x[5] = 0; arrays.y[5] = 0; arrays.z[5] =  Scalar(29.6);
	pdata_6->release();
	
	shared_ptr<FENEBondForceCompute> fc_6 = bf_creator(sysdef_6);
	fc_6->setParams(0, Scalar(1.5), Scalar(1.1), Scalar(1.0), Scalar(1.0/4.0));
	fc_6->setParams(1, Scalar(2.0*1.5), Scalar(1.1), Scalar(1.0), Scalar(1.0/4.0));
	fc_6->setParams(2, Scalar(1.5), Scalar(1.0), Scalar(1.0), Scalar(1.0/4.0));
	
	sysdef_6->getBondData()->addBond(Bond(0, 0,1));
	sysdef_6->getBondData()->addBond(Bond(1, 2,3));
	sysdef_6->getBondData()->addBond(Bond(2, 4,5));
	
	fc_6->compute(0);
	// check that the forces are correctly computed
	force_arrays = fc_6->acquire();
	MY_BOOST_CHECK_CLOSE(force_arrays.fx[0], 187.121131, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[0], 0, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[0], 0, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[0], 5.71016443 + 0.25/2, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.virial[0], 24.9495, tol);

	MY_BOOST_CHECK_CLOSE(force_arrays.fx[1], -187.121131, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[1], 0, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[1], 0, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[1], 5.71016443 + 0.25/2, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.virial[1], 24.9495, tol);

	MY_BOOST_CHECK_CLOSE(force_arrays.fx[2], 0, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[2], 184.573762, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[2], 0, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[2],  6.05171988 + 0.25/2, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.virial[2], 24.6098, tol);

	MY_BOOST_CHECK_CLOSE(force_arrays.fx[3], 0, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[3], -184.573762, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[3], 0, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[3], 6.05171988 + 0.25/2, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.virial[3], 24.6098, tol);

	MY_BOOST_CHECK_CLOSE(force_arrays.fx[4], 0, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[4], 0, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[4], 186.335166, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[4], 5.7517282 + 0.25/2, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.virial[4], 24.8447, tol);

	MY_BOOST_CHECK_CLOSE(force_arrays.fx[5], 0, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[5], 0, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[5], -186.335166, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[5],  5.7517282 + 0.25/2, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.virial[5], 24.8447, tol);

	// one more test: this one will test two things:
	// 1) That the forces are computed correctly even if the particles are rearranged in memory
	// and 2) That two forces can add to the same particle
	shared_ptr<SystemDefinition> sysdef_4(new SystemDefinition(4, BoxDim(100.0, 100.0, 100.0), 1, 1, 0, 0, 0, exec_conf));
	shared_ptr<ParticleData> pdata_4 = sysdef_4->getParticleData();	

	arrays = pdata_4->acquireReadWrite();
	// make a square of particles
	arrays.x[0] = 0.0; arrays.y[0] = 0.0; arrays.z[0] = 0.0;
	arrays.x[1] = 1.0; arrays.y[1] = 0; arrays.z[1] = 0.0;
	arrays.x[2] = 0; arrays.y[2] = 1.0; arrays.z[2] = 0.0;
	arrays.x[3] = 1.0; arrays.y[3] = 1.0; arrays.z[3] = 0.0;

	arrays.tag[0] = 2;
	arrays.tag[1] = 3;
	arrays.tag[2] = 0;
	arrays.tag[3] = 1;
	arrays.rtag[arrays.tag[0]] = 0;
	arrays.rtag[arrays.tag[1]] = 1;
	arrays.rtag[arrays.tag[2]] = 2;
	arrays.rtag[arrays.tag[3]] = 3;
	pdata_4->release();

	// build the bond force compute and try it out
	shared_ptr<FENEBondForceCompute> fc_4 = bf_creator(sysdef_4);
	fc_4->setParams(0, Scalar(1.5), Scalar(1.75), Scalar(1.2), Scalar(1.0/4.0));
	// only add bonds on the left, top, and bottom of the square
	sysdef_4->getBondData()->addBond(Bond(0, 2,3));
	sysdef_4->getBondData()->addBond(Bond(0, 2,0));
	sysdef_4->getBondData()->addBond(Bond(0, 0,1));

	fc_4->compute(0);
	force_arrays = fc_4->acquire();
	// the right two particles should only have a force pulling them left
	MY_BOOST_CHECK_CLOSE(force_arrays.fx[1], 86.85002865, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[1], 0, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[1], 0, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[1], 7.08810039/2.0, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.virial[1], 14.475, tol);

	MY_BOOST_CHECK_CLOSE(force_arrays.fx[3], 86.85002865, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[3], 0, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[3], 0, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[3], 7.08810039/2.0, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.virial[3], 14.475, tol);

	// the bottom left particle should have a force pulling up and to the right
	MY_BOOST_CHECK_CLOSE(force_arrays.fx[0], -86.850028653, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[0], -86.85002865, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[0], 0, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[0], 7.08810039, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.virial[0], 14.475*2.0, tol);

	// and the top left particle should have a force pulling down and to the right
	MY_BOOST_CHECK_CLOSE(force_arrays.fx[2], -86.85002865, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[2], 86.85002865, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[2], 0, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[2], 7.08810039, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.virial[2], 14.475*2.0, tol);
	}
	
//! Compares the output of two FENEBondForceComputes
void bond_force_comparison_tests(bondforce_creator bf_creator1, bondforce_creator bf_creator2, ExecutionConfiguration exec_conf)
	{
	#ifdef CUDA
	g_gpu_error_checking = true;
	#endif
	
	const unsigned int M = 10;
	const unsigned int N = M*M*M;
	
	// create a particle system to sum forces on
	// use a simple cubic array of particles so that random bonds
	// don't result in huge forces on a random particle arrangement
	SimpleCubicInitializer sc_init(M, 1.5, "A");
	shared_ptr<SystemDefinition> sysdef(new SystemDefinition(sc_init, exec_conf));
	shared_ptr<ParticleData> pdata = sysdef->getParticleData();
	
	shared_ptr<FENEBondForceCompute> fc1 = bf_creator1(sysdef);
	shared_ptr<FENEBondForceCompute> fc2 = bf_creator2(sysdef);
	fc1->setParams(0, Scalar(300.0), Scalar(1.6), Scalar(1.0), Scalar(1.0/4.0));
	fc2->setParams(0, Scalar(300.0), Scalar(1.6), Scalar(1.0), Scalar(1.0/4.0));

	// displace particles a little so all forces aren't alike
	ParticleDataArrays arrays = pdata->acquireReadWrite();
	BoxDim box = pdata->getBox();
	for (unsigned int i = 0; i < N; i++)
		{
		arrays.x[i] += Scalar((rand())/Scalar(RAND_MAX) - 0.5) * Scalar(0.01);
		if (arrays.x[i] < box.xlo)
			arrays.x[i] = box.xlo;
		if (arrays.x[i] > box.xhi)
			arrays.x[i] = box.xhi;
		
		arrays.y[i] += Scalar((rand())/Scalar(RAND_MAX) - 0.5) * Scalar(0.05);
		if (arrays.y[i] < box.ylo)
			arrays.y[i] = box.ylo;
		if (arrays.y[i] > box.yhi)
			arrays.y[i] = box.yhi;
		
		arrays.z[i] += Scalar((rand())/Scalar(RAND_MAX) - 0.5) * Scalar(0.001);
		if (arrays.z[i] < box.zlo)
			arrays.z[i] = box.zlo;
		if (arrays.z[i] > box.zhi)
			arrays.z[i] = box.zhi;
		}
	pdata->release();

	// add bonds
	for (unsigned int i = 0; i < M; i++)
		for (unsigned int j = 0; j < M; j++)
			for (unsigned int k = 0; k < M-1; k++)
				{
				sysdef->getBondData()->addBond(Bond(0, i*M*M + j*M + k, i*M*M + j*M + k + 1));
				}

		
	// compute the forces
	fc1->compute(0);
	fc2->compute(0);
	
	// verify that the forces are identical (within roundoff errors)
	ForceDataArrays arrays1 = fc1->acquire();
	ForceDataArrays arrays2 = fc2->acquire();

	Scalar rough_tol = Scalar(5.0);

	for (unsigned int i = 0; i < N; i++)
		{
		BOOST_CHECK_CLOSE(arrays1.fx[i], arrays2.fx[i], rough_tol);
		BOOST_CHECK_CLOSE(arrays1.fy[i], arrays2.fy[i], rough_tol);
		BOOST_CHECK_CLOSE(arrays1.fz[i], arrays2.fz[i], rough_tol);
		BOOST_CHECK_CLOSE(arrays1.pe[i], arrays2.pe[i], rough_tol);
		}
	}
	
//! FEBEBondForceCompute creator for bond_force_basic_tests()
shared_ptr<FENEBondForceCompute> base_class_bf_creator(shared_ptr<SystemDefinition> sysdef)
	{
	return shared_ptr<FENEBondForceCompute>(new FENEBondForceCompute(sysdef));
	}
	
#ifdef ENABLE_CUDA
//! FENEBondForceCompute creator for bond_force_basic_tests()
shared_ptr<FENEBondForceCompute> gpu_bf_creator(shared_ptr<SystemDefinition> sysdef)
	{
	return shared_ptr<FENEBondForceCompute>(new FENEBondForceComputeGPU(sysdef));
	}
#endif

//! boost test case for bond forces on the CPU
BOOST_AUTO_TEST_CASE( FENEBondForceCompute_basic )
	{
	bondforce_creator bf_creator = bind(base_class_bf_creator, _1);
	bond_force_basic_tests(bf_creator, ExecutionConfiguration(ExecutionConfiguration::CPU, 0));
	}

#ifdef ENABLE_CUDA
//! boost test case for bond forces on the GPU
BOOST_AUTO_TEST_CASE( FENEBondForceComputeGPU_basic )
	{
	bondforce_creator bf_creator = bind(gpu_bf_creator, _1);
	bond_force_basic_tests(bf_creator, ExecutionConfiguration(ExecutionConfiguration::GPU, ExecutionConfiguration::getDefaultGPU()));
	}
	
//! boost test case for comparing bond GPU and CPU BondForceComputes
BOOST_AUTO_TEST_CASE( FENEBondForceComputeGPU_compare )
	{
	bondforce_creator bf_creator_gpu = bind(gpu_bf_creator, _1);
	bondforce_creator bf_creator = bind(base_class_bf_creator, _1);
	bond_force_comparison_tests(bf_creator, bf_creator_gpu, ExecutionConfiguration(ExecutionConfiguration::GPU, ExecutionConfiguration::getDefaultGPU()));
	}
	
//! boost test case for comparing calculation on the CPU to multi-gpu ones
BOOST_AUTO_TEST_CASE( FENEBondForce_MultiGPU_compare)
	{
	vector<int> gpu_list;
	gpu_list.push_back(ExecutionConfiguration::getDefaultGPU());
	gpu_list.push_back(ExecutionConfiguration::getDefaultGPU());
	gpu_list.push_back(ExecutionConfiguration::getDefaultGPU());
	gpu_list.push_back(ExecutionConfiguration::getDefaultGPU());
	ExecutionConfiguration exec_conf(ExecutionConfiguration::GPU, gpu_list);
	
	bondforce_creator bf_creator_gpu = bind(gpu_bf_creator, _1);
	bondforce_creator bf_creator = bind(base_class_bf_creator, _1);
	bond_force_comparison_tests(bf_creator, bf_creator_gpu, exec_conf);
	}
#endif

#ifdef WIN32
#pragma warning( pop )
#endif
