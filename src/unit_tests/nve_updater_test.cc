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

//! name the boost unit test module
#define BOOST_TEST_MODULE NVEUpdaterTests
#include "boost_utf_configure.h"

#include <boost/test/floating_point_comparison.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include "ConstForceCompute.h"
#include "NVEUpdater.h"
#ifdef ENABLE_CUDA
#include "NVEUpdaterGPU.h"
#endif

#include "BinnedNeighborList.h"
#include "Initializers.h"
#include "LJForceCompute.h"

#include <math.h>

using namespace std;
using namespace boost;

/*! \file nve_updater_test.cc
	\brief Implements unit tests for NVEUpdater and descendants
	\ingroup unit_tests
*/

//! Helper macro for checking if two floating point numbers are close
#define MY_BOOST_CHECK_CLOSE(a,b,c) BOOST_CHECK_CLOSE(a,Scalar(b),Scalar(c))
//! Helper macro for checking if two floating point numbers are small
#define MY_BOOST_CHECK_SMALL(a,c) BOOST_CHECK_SMALL(a,Scalar(c))

//! Tolerance for floating point comparisons
#ifdef SINGLE_PRECISION
const Scalar tol = Scalar(1e-2);
#else
const Scalar tol = 1e-3;
#endif

//! Typedef'd NVEUpdator class factory
typedef boost::function<shared_ptr<NVEUpdater> (shared_ptr<SystemDefinition> sysdef, Scalar deltaT)> nveup_creator;

//! Integrate 1 particle through time and compare to an analytical solution
void nve_updater_integrate_tests(nveup_creator nve_creator, ExecutionConfiguration exec_conf)
	{
	#ifdef CUDA
	g_gpu_error_checking = true;
	#endif
	
	// check that the nve updater can actually integrate particle positions and velocities correctly
	// start with a 2 particle system to keep things simple: also put everything in a huge box so boundary conditions
	// don't come into play
	shared_ptr<SystemDefinition> sysdef(new SystemDefinition(2, BoxDim(1000.0), 4, 0, exec_conf));
	shared_ptr<ParticleData> pdata = sysdef->getParticleData();
	
	ParticleDataArrays arrays = pdata->acquireReadWrite();
	
	// setup a simple initial state
	arrays.x[0] = 0.0;
	arrays.y[0] = 1.0;
	arrays.z[0] = 2.0;
	arrays.vx[0] = 3.0;
	arrays.vy[0] = 2.0;
	arrays.vz[0] = 1.0;

	arrays.x[1] = 10.0;
	arrays.y[1] = 11.0;
	arrays.z[1] = 12.0;
	arrays.vx[1] = 13.0;
	arrays.vy[1] = 12.0;
	arrays.vz[1] = 11.0;
	
	pdata->release();
	
	Scalar deltaT = Scalar(0.0001);
	shared_ptr<NVEUpdater> nve_up = nve_creator(sysdef, deltaT);
	// also test the ability of the updater to add two force computes together properly
	shared_ptr<ConstForceCompute> fc1(new ConstForceCompute(sysdef, 1.5, 0.0, 0.0));
	nve_up->addForceCompute(fc1);
	shared_ptr<ConstForceCompute> fc2(new ConstForceCompute(sysdef, 0.0, 2.5, 0.0));
	nve_up->addForceCompute(fc2);
	
	// verify proper integration compared to x = x0 + v0 t + 1/2 a t^2, v = v0 + a t
	// roundoff errors prevent this from keeping within 0.1% error for long
	for (int i = 0; i < 500; i++)
		{
		arrays = pdata->acquireReadWrite();
		
		Scalar t = Scalar(i) * deltaT;
		MY_BOOST_CHECK_CLOSE(arrays.x[0], 0.0 + 3.0 * t + 1.0/2.0 * 1.5 * t*t, tol);
		MY_BOOST_CHECK_CLOSE(arrays.vx[0], 3.0 + 1.5 * t, tol);
		
		MY_BOOST_CHECK_CLOSE(arrays.y[0], 1.0 + 2.0 * t + 1.0/2.0 * 2.5 * t*t, tol);
		MY_BOOST_CHECK_CLOSE(arrays.vy[0], 2.0 + 2.5 * t, tol);
		
		MY_BOOST_CHECK_CLOSE(arrays.z[0], 2.0 + 1.0 * t + 1.0/2.0 * 0 * t*t, tol);
		MY_BOOST_CHECK_CLOSE(arrays.vz[0], 1.0 + 0 * t, tol);
		
		MY_BOOST_CHECK_CLOSE(arrays.x[1], 10.0 + 13.0 * t + 1.0/2.0 * 1.5 * t*t, tol);
		MY_BOOST_CHECK_CLOSE(arrays.vx[1], 13.0 + 1.5 * t, tol);
		
		MY_BOOST_CHECK_CLOSE(arrays.y[1], 11.0 + 12.0 * t + 1.0/2.0 * 2.5 * t*t, tol);
		MY_BOOST_CHECK_CLOSE(arrays.vy[1], 12.0 + 2.5 * t, tol);
		
		MY_BOOST_CHECK_CLOSE(arrays.z[1], 12.0 + 11.0 * t + 1.0/2.0 * 0 * t*t, tol);
		MY_BOOST_CHECK_CLOSE(arrays.vz[1], 11.0 + 0 * t, tol);		
		
		pdata->release();
		
		nve_up->update(i);
		}
	}
	
//! Check that the particle movement limit works
void nve_updater_limit_tests(nveup_creator nve_creator, ExecutionConfiguration exec_conf)
	{
	#ifdef CUDA
	g_gpu_error_checking = true;
	#endif
	
	// create a simple 1 particle system
	shared_ptr<SystemDefinition> sysdef(new SystemDefinition(1, BoxDim(1000.0), 1, 0, exec_conf));
	shared_ptr<ParticleData> pdata = sysdef->getParticleData();
	
	ParticleDataArrays arrays = pdata->acquireReadWrite();
	
	// setup a simple initial state
	arrays.x[0] = 0.0;
	arrays.y[0] = 1.0;
	arrays.z[0] = 2.0;
	arrays.vx[0] = 0.0;
	arrays.vy[0] = 0.0;
	arrays.vz[0] = 0.0;

	pdata->release();
	
	Scalar deltaT = Scalar(0.0001);
	shared_ptr<NVEUpdater> nve_up = nve_creator(sysdef, deltaT);
	// set the limit
	Scalar limit = Scalar(0.1);
	nve_up->setLimit(limit);

	// create an insanely large force to test the limiting method
	shared_ptr<ConstForceCompute> fc1(new ConstForceCompute(sysdef, 1e9, 2e9, 3e9));
	nve_up->addForceCompute(fc1);
	
	// expected movement vectors
	Scalar dx = limit / sqrt(14.0);
	Scalar dy = 2.0 * limit / sqrt(14.0);
	Scalar dz = 3.0 * limit / sqrt(14.0);
	
	Scalar vx = limit / sqrt(14.0) / deltaT;
	Scalar vy = 2.0 * limit / sqrt(14.0) / deltaT;
	Scalar vz = 3.0 * limit / sqrt(14.0) / deltaT;
	
	// verify proper integration compared to x = x0 + dx * i
	nve_up->update(0);
	for (int i = 1; i < 500; i++)
		{
		arrays = pdata->acquireReadWrite();
		
		MY_BOOST_CHECK_CLOSE(arrays.x[0], 0.0 + dx * Scalar(i), tol);
		MY_BOOST_CHECK_CLOSE(arrays.vx[0], vx, tol);
		
		MY_BOOST_CHECK_CLOSE(arrays.y[0], 1.0 + dy * Scalar(i), tol);
		MY_BOOST_CHECK_CLOSE(arrays.vy[0], vy, tol);
		
		MY_BOOST_CHECK_CLOSE(arrays.z[0], 2.0 + dz * Scalar(i), tol);
		MY_BOOST_CHECK_CLOSE(arrays.vz[0], vz, tol);
		
		pdata->release();
		
		nve_up->update(i);
		}
	}
	
	
//! Make a few particles jump across the boundary and verify that the updater works
void nve_updater_boundary_tests(nveup_creator nve_creator, ExecutionConfiguration exec_conf)
	{
	#ifdef CUDA
	g_gpu_error_checking = true;
	#endif
	
	////////////////////////////////////////////////////////////////////
	// now, lets do a more thorough test and include boundary conditions
	// there are way too many permutations to test here, so I will simply
	// test +x, -x, +y, -y, +z, and -z independantly
	// build a 6 particle system with particles set to move across each boundary
	shared_ptr<SystemDefinition> sysdef_6(new SystemDefinition(6, BoxDim(20.0, 40.0, 60.0), 1, 0, exec_conf));
	shared_ptr<ParticleData> pdata_6 = sysdef_6->getParticleData();
	
	ParticleDataArrays arrays = pdata_6->acquireReadWrite();
	arrays.x[0] = Scalar(-9.6); arrays.y[0] = 0; arrays.z[0] = 0.0;
	arrays.vx[0] = Scalar(-0.5);
	arrays.x[1] =  Scalar(9.6); arrays.y[1] = 0; arrays.z[1] = 0.0;
	arrays.vx[1] = Scalar(0.6);
	arrays.x[2] = 0; arrays.y[2] = Scalar(-19.6); arrays.z[2] = 0.0;
	arrays.vy[2] = Scalar(-0.5);
	arrays.x[3] = 0; arrays.y[3] = Scalar(19.6); arrays.z[3] = 0.0;
	arrays.vy[3] = Scalar(0.6);
	arrays.x[4] = 0; arrays.y[4] = 0; arrays.z[4] = Scalar(-29.6);
	arrays.vz[4] = Scalar(-0.5);
	arrays.x[5] = 0; arrays.y[5] = 0; arrays.z[5] =  Scalar(29.6);
	arrays.vz[5] = Scalar(0.6);
	pdata_6->release();
	
	Scalar deltaT = 1.0;
	shared_ptr<NVEUpdater> nve_up = nve_creator(sysdef_6, deltaT);
	// no forces on these particles
	shared_ptr<ConstForceCompute> fc1(new ConstForceCompute(sysdef_6, 0, 0.0, 0.0));
	nve_up->addForceCompute(fc1);
	
	// move the particles across the boundary
	nve_up->update(0);
	
	// check that they go to the proper final position
	arrays = pdata_6->acquireReadWrite();
	MY_BOOST_CHECK_CLOSE(arrays.x[0], 9.9, tol);
	BOOST_CHECK_EQUAL(arrays.ix[0], -1);
	MY_BOOST_CHECK_CLOSE(arrays.x[1], -9.8, tol);
	BOOST_CHECK_EQUAL(arrays.ix[1], 1);
	MY_BOOST_CHECK_CLOSE(arrays.y[2], 19.9, tol);
	BOOST_CHECK_EQUAL(arrays.iy[2], -1);
	MY_BOOST_CHECK_CLOSE(arrays.y[3], -19.8, tol);
	BOOST_CHECK_EQUAL(arrays.iy[3], 1);
	MY_BOOST_CHECK_CLOSE(arrays.z[4], 29.9, tol);
	BOOST_CHECK_EQUAL(arrays.iz[4], -1);
	MY_BOOST_CHECK_CLOSE(arrays.z[5], -29.8, tol);
	BOOST_CHECK_EQUAL(arrays.iz[5], 1);
	
	pdata_6->release();
	}

//! Compares the output from one NVEUpdater to another
void nve_updater_compare_test(nveup_creator nve_creator1, nveup_creator nve_creator2, ExecutionConfiguration exec_conf)
	{
	#ifdef CUDA
	g_gpu_error_checking = true;
	#endif
	
	const unsigned int N = 1000;
	
	// create two identical random particle systems to simulate
	RandomInitializer rand_init1(N, Scalar(0.2), Scalar(0.9), "A");
	RandomInitializer rand_init2(N, Scalar(0.2), Scalar(0.9), "A");
	rand_init1.setSeed(12345);
	shared_ptr<SystemDefinition> sysdef1(new SystemDefinition(rand_init1, exec_conf));
	shared_ptr<ParticleData> pdata1 = sysdef1->getParticleData();
	rand_init2.setSeed(12345);
	shared_ptr<SystemDefinition> sysdef2(new SystemDefinition(rand_init2, exec_conf));
	shared_ptr<ParticleData> pdata2 = sysdef2->getParticleData();

	shared_ptr<NeighborList> nlist1(new NeighborList(sysdef1, Scalar(3.0), Scalar(0.8)));
	shared_ptr<NeighborList> nlist2(new NeighborList(sysdef2, Scalar(3.0), Scalar(0.8)));
	
	shared_ptr<LJForceCompute> fc1(new LJForceCompute(sysdef1, nlist1, Scalar(3.0)));
	shared_ptr<LJForceCompute> fc2(new LJForceCompute(sysdef2, nlist2, Scalar(3.0)));
		
	// setup some values for alpha and sigma
	Scalar epsilon = Scalar(1.0);
	Scalar sigma = Scalar(1.2);
	Scalar alpha = Scalar(0.45);
	Scalar lj1 = Scalar(4.0) * epsilon * pow(sigma,Scalar(12.0));
	Scalar lj2 = alpha * Scalar(4.0) * epsilon * pow(sigma,Scalar(6.0));
	
	// specify the force parameters
	fc1->setParams(0,0,lj1,lj2);
	fc2->setParams(0,0,lj1,lj2);

	shared_ptr<NVEUpdater> nve1 = nve_creator1(sysdef1, Scalar(0.005));
	shared_ptr<NVEUpdater> nve2 = nve_creator2(sysdef2, Scalar(0.005));

	nve1->addForceCompute(fc1);
	nve2->addForceCompute(fc2);

	// step for only a few time steps and verify that they are the same
	// we can't do much more because these things are chaotic and diverge quickly
	for (int i = 0; i < 5; i++)
		{
		const ParticleDataArraysConst& arrays1 = pdata1->acquireReadOnly();
		const ParticleDataArraysConst& arrays2 = pdata2->acquireReadOnly();

		Scalar rough_tol = 2.0;
		//cout << arrays1.x[100] << " " << arrays2.x[100] << endl;

		// check position, velocity and acceleration
		for (unsigned int j = 0; j < N; j++)
			{
			MY_BOOST_CHECK_CLOSE(arrays1.x[j], arrays2.x[j], rough_tol);
			MY_BOOST_CHECK_CLOSE(arrays1.y[j], arrays2.y[j], rough_tol);
			MY_BOOST_CHECK_CLOSE(arrays1.z[j], arrays2.z[j], rough_tol);
			
			MY_BOOST_CHECK_CLOSE(arrays1.vx[j], arrays2.vx[j], rough_tol);
			MY_BOOST_CHECK_CLOSE(arrays1.vy[j], arrays2.vy[j], rough_tol);
			MY_BOOST_CHECK_CLOSE(arrays1.vz[j], arrays2.vz[j], rough_tol);
			
			MY_BOOST_CHECK_CLOSE(arrays1.ax[j], arrays2.ax[j], rough_tol);
			MY_BOOST_CHECK_CLOSE(arrays1.ay[j], arrays2.ay[j], rough_tol);
			MY_BOOST_CHECK_CLOSE(arrays1.az[j], arrays2.az[j], rough_tol);
			}
		
		pdata1->release();
		pdata2->release();

		nve1->update(i);
		nve2->update(i);
		}
	}
	
//! NVEUpdater factory for the unit tests
shared_ptr<NVEUpdater> base_class_nve_creator(shared_ptr<SystemDefinition> sysdef, Scalar deltaT)
	{
	return shared_ptr<NVEUpdater>(new NVEUpdater(sysdef, deltaT));
	}
	
#ifdef ENABLE_CUDA
//! NVEUpdaterGPU factory for the unit tests
shared_ptr<NVEUpdater> gpu_nve_creator(shared_ptr<SystemDefinition> sysdef, Scalar deltaT)
	{
	return shared_ptr<NVEUpdater>(new NVEUpdaterGPU(sysdef, deltaT));
	}
#endif
	
	
//! boost test case for base class integration tests
BOOST_AUTO_TEST_CASE( NVEUpdater_integrate_tests )
	{
	nveup_creator nve_creator = bind(base_class_nve_creator, _1, _2);
	nve_updater_integrate_tests(nve_creator, ExecutionConfiguration(ExecutionConfiguration::CPU, 0));
	}
	
//! boost test case for base class limit tests
BOOST_AUTO_TEST_CASE( NVEUpdater_limit_tests )
	{
	nveup_creator nve_creator = bind(base_class_nve_creator, _1, _2);
	nve_updater_limit_tests(nve_creator, ExecutionConfiguration(ExecutionConfiguration::CPU, 0));
	}	
	
//! boost test case for base class boundary tests
BOOST_AUTO_TEST_CASE( NVEUpdater_boundary_tests )
	{
	nveup_creator nve_creator = bind(base_class_nve_creator, _1, _2);
	nve_updater_boundary_tests(nve_creator, ExecutionConfiguration(ExecutionConfiguration::CPU, 0));
	}
	
#ifdef ENABLE_CUDA
//! boost test case for base class integration tests
BOOST_AUTO_TEST_CASE( NVEUpdaterGPU_integrate_tests )
	{
	nveup_creator nve_creator_gpu = bind(gpu_nve_creator, _1, _2);
	nve_updater_integrate_tests(nve_creator_gpu, ExecutionConfiguration(ExecutionConfiguration::GPU, ExecutionConfiguration::getDefaultGPU()));
	}
	
//! boost test case for base class limit tests
BOOST_AUTO_TEST_CASE( NVEUpdaterGPU_limit_tests )
	{
	nveup_creator nve_creator = bind(gpu_nve_creator, _1, _2);
	nve_updater_limit_tests(nve_creator, ExecutionConfiguration(ExecutionConfiguration::GPU, ExecutionConfiguration::getDefaultGPU()));
	}		
	
//! boost test case for base class boundary tests
BOOST_AUTO_TEST_CASE( NVEUpdaterGPU_boundary_tests )
	{
	nveup_creator nve_creator_gpu = bind(gpu_nve_creator, _1, _2);
	nve_updater_boundary_tests(nve_creator_gpu, ExecutionConfiguration(ExecutionConfiguration::GPU, ExecutionConfiguration::getDefaultGPU()));
	}

//! boost test case for comparing the GPU and CPU NVEUpdaters
BOOST_AUTO_TEST_CASE( NVEUPdaterGPU_comparison_tests)
	{
	nveup_creator nve_creator_gpu = bind(gpu_nve_creator, _1, _2);
	nveup_creator nve_creator = bind(base_class_nve_creator, _1, _2);
	nve_updater_compare_test(nve_creator, nve_creator_gpu, ExecutionConfiguration(ExecutionConfiguration::GPU, ExecutionConfiguration::getDefaultGPU()));
	}
	
//! boost test case for comkparing CPU to multi-GPU updaters
BOOST_AUTO_TEST_CASE( NVEUPdaterMultiGPU_comparison_tests)
	{
	vector<unsigned int> gpu_list;
	gpu_list.push_back(ExecutionConfiguration::getDefaultGPU());
	gpu_list.push_back(ExecutionConfiguration::getDefaultGPU());
	gpu_list.push_back(ExecutionConfiguration::getDefaultGPU());
	gpu_list.push_back(ExecutionConfiguration::getDefaultGPU());
	ExecutionConfiguration exec_conf(ExecutionConfiguration::GPU, gpu_list);
	
	nveup_creator nve_creator_gpu = bind(gpu_nve_creator, _1, _2);
	nveup_creator nve_creator = bind(base_class_nve_creator, _1, _2);
	nve_updater_compare_test(nve_creator, nve_creator_gpu, exec_conf);
	}
#endif

#ifdef WIN32
#pragma warning( pop )
#endif
