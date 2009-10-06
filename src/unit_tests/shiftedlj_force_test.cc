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
// Maintainer: phillicl

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <iostream>

//! Name the unit test module
#define BOOST_TEST_MODULE ShiftedLJForceTests
#include "boost_utf_configure.h"

#include <boost/test/floating_point_comparison.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include "LJForceCompute.h"

#ifdef ENABLE_CUDA
#include "LJForceComputeGPU.h"
#endif

#include "BinnedNeighborList.h"
#include "Initializers.h"

#include <math.h>

using namespace std;
using namespace boost;

/*! \file shiftedlj_force_test.cc
	\brief Implements unit tests for ShiftedLJForceCompute and descendants
	\ingroup unit_tests
*/

//! Helper macro for testing if two numbers are close
#define MY_BOOST_CHECK_CLOSE(a,b,c) BOOST_CHECK_CLOSE(a,Scalar(b),Scalar(c))
//! Helper macro for testing if a number is small
#define MY_BOOST_CHECK_SMALL(a,c) BOOST_CHECK_SMALL(a,Scalar(c))

//! Tolerance in percent to use for comparing various ShiftedLJForceComputes to each other
#ifdef SINGLE_PRECISION
const Scalar tol = Scalar(2);
#else
const Scalar tol = 1e-6;
#endif

//! Typedef'd ShiftedLJForceCompute factory
typedef boost::function<shared_ptr<LJForceCompute> (shared_ptr<SystemDefinition> sysdef, shared_ptr<NeighborList> nlist, Scalar r_cut)> shiftedljforce_creator;
	
//! Test the ability of the shiftedlj force compute to actually calucate forces
void shiftedlj_force_particle_test(shiftedljforce_creator shiftedlj_creator, ExecutionConfiguration exec_conf)
	{
	#ifdef CUDA
	g_gpu_error_checking = true;
	#endif
	
	// this 3-particle test subtly checks several conditions
	// the particles are arranged on the x axis,  1   2   3
	// such that 2 is inside the cuttoff radius of 1 and 3, but 1 and 3 are outside the cuttoff
	// of course, the buffer will be set on the neighborlist so that 3 is included in it
	// thus, this case tests the ability of the force summer to sum more than one force on
	// a particle and ignore a particle outside the radius
	// Also particle 2 would not be within the cutoff of particle 1 if it were not the case that particle 1 has a shifted potential.
	
	// periodic boundary conditions will be handeled in another test
	shared_ptr<SystemDefinition> sysdef_3(new SystemDefinition(3, BoxDim(1000.0), 1, 0, 0, 0, 0, exec_conf));
	shared_ptr<ParticleData> pdata_3 = sysdef_3->getParticleData();

	ParticleDataArrays arrays = pdata_3->acquireReadWrite();
	arrays.x[0] = -0.2; 
	//arrays.x[0] = 0;
	arrays.y[0] = arrays.z[0] = 0.0;
	arrays.x[1] = Scalar(pow(2.0,1.0/6.0)); arrays.y[1] = arrays.z[1] = 0.0;
	arrays.x[2] = Scalar(2.0*pow(2.0,1.0/6.0)); arrays.y[2] = arrays.z[2] = 0.0;
	arrays.diameter[0]=1.2;
	
	Scalar maxdiam = pdata_3->getMaxDiameter();
	Scalar r_cut = 1.3;
	Scalar r_alpha = maxdiam/2 - 0.5;
	Scalar r_cut_wc = r_cut + 2 * r_alpha;
	pdata_3->release();
	shared_ptr<NeighborList> nlist_3(new NeighborList(sysdef_3, r_cut_wc, Scalar(3.0)));
	shared_ptr<LJForceCompute> fc_3 = shiftedlj_creator(sysdef_3, nlist_3, r_cut);
	fc_3->setSLJ(true);

	
	// first test: setup a sigma of 1.0 so that all forces will be 0
	Scalar epsilon = Scalar(1.15);
	Scalar sigma = Scalar(1.0);
	Scalar alpha = Scalar(1.0);
	Scalar shiftedlj1 = Scalar(4.0) * epsilon * pow(sigma,Scalar(12.0));
	Scalar shiftedlj2 = alpha * Scalar(4.0) * epsilon * pow(sigma,Scalar(6.0));
	fc_3->setParams(0,0,shiftedlj1,shiftedlj2);
	
	// compute the forces
	fc_3->compute(0);
	
	ForceDataArrays force_arrays = fc_3->acquire();
	MY_BOOST_CHECK_CLOSE(force_arrays.fx[0], 2.710943702, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fy[0], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[0], tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[0], -0.482660808, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.virial[0], -0.597520027, tol);

	MY_BOOST_CHECK_CLOSE(force_arrays.fx[1], -2.710943702, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fy[1], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[1], tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[1], -1.057660808, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.virial[1],-0.597520027, tol);

	MY_BOOST_CHECK_SMALL(force_arrays.fx[2], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fy[2], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[2], tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[2], -0.575, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.virial[2], tol);
	
	// now change sigma and alpha so we can check that it is computing the right force
	sigma = Scalar(1.2); // < bigger sigma should push particle 0 left and particle 2 right
	alpha = Scalar(0.45);
	shiftedlj1 = Scalar(4.0) * epsilon * pow(sigma,Scalar(12.0));
	shiftedlj2 = alpha * Scalar(4.0) * epsilon * pow(sigma,Scalar(6.0));	
	fc_3->setParams(0,0,shiftedlj1,shiftedlj2);
	fc_3->compute(1);
	
	force_arrays = fc_3->acquire();
	MY_BOOST_CHECK_CLOSE(force_arrays.fx[0], -27.05553467, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fy[0], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[0], tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[0], 0.915093686, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.virial[0], 5.9633196325, tol);

	// center particle should still be a 0 force by symmetry
	MY_BOOST_CHECK_CLOSE(force_arrays.fx[1],-66.0427, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fy[1], 1e-5);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[1], 1e-5);
	// there is still an energy and virial, though
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[1], 4.496604724, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.virial[1], 23.37985722, tol);

	MY_BOOST_CHECK_CLOSE(force_arrays.fx[2], 93.09822608552962, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fy[2], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[2], tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[2], 3.581511037746, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.virial[2], 17.416537590989, tol);
	
	// swap the order of particles 0 ans 2 in memory to check that the force compute handles this properly
	arrays = pdata_3->acquireReadWrite();
	arrays.x[2] = arrays.y[2] = arrays.z[2] = 0.0;
	arrays.x[0] = Scalar(2.0*pow(2.0,1.0/6.0)); arrays.y[0] = arrays.z[0] = 0.0;

	arrays.tag[0] = 2;
	arrays.tag[2] = 0;
	arrays.rtag[0] = 2;
	arrays.rtag[2] = 0;
	pdata_3->release();
	
	// notify the particle data that we changed the order
	pdata_3->notifyParticleSort();

	// recompute the forces at the same timestep, they should be updated
	fc_3->compute(1);
	force_arrays = fc_3->acquire();
	MY_BOOST_CHECK_CLOSE(force_arrays.fx[0], 336.9779601, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fx[2], -93.09822608552962, tol);
	}

//! Tests the ability of a ShiftedLJForceCompute to handle periodic boundary conditions.  Also intentionally place a particle outside the cutoff of normally size particle but in the cutoff of a large particle
void shiftedlj_force_periodic_test(shiftedljforce_creator shiftedlj_creator, ExecutionConfiguration exec_conf)
	{
	#ifdef CUDA
	g_gpu_error_checking = true;
	#endif
	
	////////////////////////////////////////////////////////////////////
	// now, lets do a more thorough test and include boundary conditions
	// there are way too many permutations to test here, so I will simply
	// test +x, -x, +y, -y, +z, and -z independantly
	// build a 6 particle system with particles across each boundary
	// also test the ability of the force compute to use different particle types
	
	shared_ptr<SystemDefinition> sysdef_6(new SystemDefinition(6, BoxDim(20.0, 40.0, 60.0), 3, 0, 0, 0, 0, exec_conf));
	shared_ptr<ParticleData> pdata_6 = sysdef_6->getParticleData();

	ParticleDataArrays arrays = pdata_6->acquireReadWrite();
	arrays.x[0] = Scalar(-9.6); arrays.y[0] = 0; arrays.z[0] = 0.0;
	arrays.x[1] =  Scalar(9.6); arrays.y[1] = 0; arrays.z[1] = 0.0;
	arrays.x[2] = 0; arrays.y[2] = Scalar(-19.35); arrays.z[2] = 0.0;
	arrays.x[3] = 0; arrays.y[3] = Scalar(19.6); arrays.z[3] = 0.0;
	arrays.x[4] = 0; arrays.y[4] = 0; arrays.z[4] = Scalar(-29.1);
	arrays.x[5] = 0; arrays.y[5] = 0; arrays.z[5] =  Scalar(29.6);
	
	arrays.type[0] = 0;
	arrays.type[1] = 1;
	arrays.type[2] = 2;
	arrays.type[3] = 0;
	arrays.type[4] = 2;
	arrays.type[5] = 1;
	
	arrays.diameter[0]=1.2;
	arrays.diameter[2]=1.5;
	arrays.diameter[4]=2.0; 
	Scalar maxdiam = pdata_6->getMaxDiameter();
	Scalar r_cut = 1.3;
	Scalar r_alpha = maxdiam/2 - 0.5;
	Scalar r_cut_wc = r_cut + 2 * r_alpha;
	
	pdata_6->release();
	
	shared_ptr<NeighborList> nlist_6(new NeighborList(sysdef_6, r_cut_wc, Scalar(3.0)));
	shared_ptr<LJForceCompute> fc_6 = shiftedlj_creator(sysdef_6, nlist_6, r_cut);
	fc_6->setSLJ(true);

		
	// choose a small sigma so that all interactions are attractive
	Scalar epsilon = Scalar(1.0);
	Scalar sigma = Scalar(0.5);
	Scalar alpha = Scalar(0.45);
	Scalar shiftedlj1 = Scalar(4.0) * epsilon * pow(sigma,Scalar(12.0));
	Scalar shiftedlj2 = alpha * Scalar(4.0) * epsilon * pow(sigma,Scalar(6.0));
	
	// make life easy: just change epsilon for the different pairs
	fc_6->setParams(0,0,shiftedlj1,shiftedlj2);
	fc_6->setParams(0,1,Scalar(2.0)*shiftedlj1,Scalar(2.0)*shiftedlj2);
	fc_6->setParams(0,2,Scalar(3.0)*shiftedlj1,Scalar(3.0)*shiftedlj2);
	fc_6->setParams(1,1,Scalar(4.0)*shiftedlj1,Scalar(4.0)*shiftedlj2);
	fc_6->setParams(1,2,Scalar(5.0)*shiftedlj1,Scalar(5.0)*shiftedlj2);
	fc_6->setParams(2,2,Scalar(6.0)*shiftedlj1,Scalar(6.0)*shiftedlj2);
	
	fc_6->compute(0);	
	
	ForceDataArrays force_arrays = fc_6->acquire();
	// particle 0 should be pulled left
	MY_BOOST_CHECK_CLOSE(force_arrays.fx[0], -1.679141673, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fy[0], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[0], tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.virial[0],-0.223885556, tol);

	// particle 1 should be pulled right
	MY_BOOST_CHECK_CLOSE(force_arrays.fx[1], 1.679141673, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fy[1], 1e-5);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[1], 1e-5);
	MY_BOOST_CHECK_CLOSE(force_arrays.virial[1], -0.223885556, tol);
	
	// particle 2 should be pulled down
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[2], -1.77449965121923, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fx[2], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[2], tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.virial[2], -0.310537439, tol);

	// particle 3 should be pulled up
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[3], 1.77449965121923, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fx[3], 1e-5);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[3], 1e-5);
	MY_BOOST_CHECK_CLOSE(force_arrays.virial[3], -0.310537439, tol);
	
	// particle 4 should be pulled back
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[4], -2.95749941869871, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fx[4], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fy[4], tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.virial[4], -0.640791541, tol);

	// particle 3 should be pulled forward
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[5], 2.95749941869871, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fx[5], 1e-5);
	MY_BOOST_CHECK_SMALL(force_arrays.fy[5], 1e-5);
	MY_BOOST_CHECK_CLOSE(force_arrays.virial[5], -0.640791541, tol);
	}
	
//! Unit test a comparison between 2 ShiftedLJForceComputes on a "real" system
void shiftedlj_force_comparison_test(shiftedljforce_creator shiftedlj_creator1, shiftedljforce_creator shiftedlj_creator2, ExecutionConfiguration exec_conf)
	{
	#ifdef CUDA
	g_gpu_error_checking = true;
	#endif
	
	const unsigned int N = 5000;
	
	// create a random particle system to sum forces on
	RandomInitializer rand_init(N, Scalar(0.2), Scalar(0.9), "A");
	shared_ptr<SystemDefinition> sysdef(new SystemDefinition(rand_init, exec_conf));
	shared_ptr<ParticleData> pdata = sysdef->getParticleData();
	
	shared_ptr<BinnedNeighborList> nlist(new BinnedNeighborList(sysdef, Scalar(3.0), Scalar(0.8)));
	
	shared_ptr<LJForceCompute> fc1 = shiftedlj_creator1(sysdef, nlist, Scalar(3.0));
	shared_ptr<LJForceCompute> fc2 = shiftedlj_creator2(sysdef, nlist, Scalar(3.0));
		
	// setup some values for alpha and sigma
	Scalar epsilon = Scalar(1.0);
	Scalar sigma = Scalar(1.2);
	Scalar alpha = Scalar(0.45);
	Scalar shiftedlj1 = Scalar(4.0) * epsilon * pow(sigma,Scalar(12.0));
	Scalar shiftedlj2 = alpha * Scalar(4.0) * epsilon * pow(sigma,Scalar(6.0));
	fc1->setSLJ(true);
	fc2->setSLJ(true);
	
	// specify the force parameters
	fc1->setParams(0,0,shiftedlj1,shiftedlj2);
	fc2->setParams(0,0,shiftedlj1,shiftedlj2);
	
	// compute the forces
	fc1->compute(0);
	fc2->compute(0);
	
	// verify that the forces are identical (within roundoff errors)
	ForceDataArrays arrays1 = fc1->acquire();
	ForceDataArrays arrays2 = fc2->acquire();
	
	for (unsigned int i = 0; i < N; i++)
		{
		BOOST_CHECK_CLOSE(arrays1.fx[i], arrays2.fx[i], tol);
		BOOST_CHECK_CLOSE(arrays1.fy[i], arrays2.fy[i], tol);
		BOOST_CHECK_CLOSE(arrays1.fz[i], arrays2.fz[i], tol);
		BOOST_CHECK_CLOSE(arrays1.pe[i], arrays2.pe[i], tol);
		BOOST_CHECK_CLOSE(arrays1.virial[i], arrays2.virial[i], tol);
		}
	}
	
//! ShiftedLJForceCompute creator for unit tests
shared_ptr<LJForceCompute> base_class_shiftedlj_creator(shared_ptr<SystemDefinition> sysdef, shared_ptr<NeighborList> nlist, Scalar r_cut)
	{
	return shared_ptr<LJForceCompute>(new LJForceCompute(sysdef, nlist, r_cut));
	}
	
	
#ifdef ENABLE_CUDA
//! LJForceComputeGPU creator for unit tests
shared_ptr<LJForceCompute> gpu_shiftedlj_creator(shared_ptr<SystemDefinition> sysdef, shared_ptr<NeighborList> nlist, Scalar r_cut)
	{
	nlist->setStorageMode(NeighborList::full);
	shared_ptr<LJForceComputeGPU> lj(new LJForceComputeGPU(sysdef, nlist, r_cut));
	// the default block size kills valgrind :) reduce it
	lj->setBlockSize(64);
	return lj;
	}
#endif
	
//! boost test case for particle test on CPU
BOOST_AUTO_TEST_CASE( ShiftedLJForce_particle )
	{
	shiftedljforce_creator shiftedlj_creator_base = bind(base_class_shiftedlj_creator, _1, _2, _3);
	shiftedlj_force_particle_test(shiftedlj_creator_base, ExecutionConfiguration(ExecutionConfiguration::CPU));
	}
	
//! boost test case for periodic test on CPU
BOOST_AUTO_TEST_CASE( ShiftedLJForce_periodic )
	{
	shiftedljforce_creator shiftedlj_creator_base = bind(base_class_shiftedlj_creator, _1, _2, _3);
	shiftedlj_force_periodic_test(shiftedlj_creator_base, ExecutionConfiguration(ExecutionConfiguration::CPU));
	}

	
# ifdef ENABLE_CUDA
//! boost test case for particle test on CPU - threaded
BOOST_AUTO_TEST_CASE( ShiftedLJForceGPU_particle )
	{
	shiftedljforce_creator shiftedlj_creator_gpu = bind(gpu_shiftedlj_creator, _1, _2, _3);
	shiftedlj_force_particle_test(shiftedlj_creator_gpu, ExecutionConfiguration(ExecutionConfiguration::GPU));
	}


//! boost test case for periodic test on the GPU
BOOST_AUTO_TEST_CASE( ShiftedLJForceGPU_periodic )
	{
	shiftedljforce_creator shiftedlj_creator_gpu = bind(gpu_shiftedlj_creator, _1, _2, _3);
	shiftedlj_force_periodic_test(shiftedlj_creator_gpu, ExecutionConfiguration(ExecutionConfiguration::GPU));
	}

//! boost test case for comparing GPU output to base class output
BOOST_AUTO_TEST_CASE( ShiftedLJForceGPU_compare )
	{
	shiftedljforce_creator shiftedlj_creator_gpu = bind(gpu_shiftedlj_creator, _1, _2, _3);
	shiftedljforce_creator shiftedlj_creator_base = bind(base_class_shiftedlj_creator, _1, _2, _3);
	shiftedlj_force_comparison_test(shiftedlj_creator_base, shiftedlj_creator_gpu, ExecutionConfiguration(ExecutionConfiguration::GPU));
	}
	
//! boost test case for comparing multi-GPU output to base class output
BOOST_AUTO_TEST_CASE( ShiftedLJForceMultiGPU_compare )
	{
	vector<int> gpu_list;
	gpu_list.push_back(ExecutionConfiguration::getDefaultGPU());
	gpu_list.push_back(ExecutionConfiguration::getDefaultGPU());
	gpu_list.push_back(ExecutionConfiguration::getDefaultGPU());
	gpu_list.push_back(ExecutionConfiguration::getDefaultGPU());
	ExecutionConfiguration exec_conf(gpu_list);

	shiftedljforce_creator shiftedlj_creator_gpu = bind(gpu_shiftedlj_creator, _1, _2, _3);
	shiftedljforce_creator shiftedlj_creator_base = bind(base_class_shiftedlj_creator, _1, _2, _3);
	shiftedlj_force_comparison_test(shiftedlj_creator_base, shiftedlj_creator_gpu, exec_conf);
	}

#endif

#ifdef WIN32
#pragma warning( pop )
#endif
