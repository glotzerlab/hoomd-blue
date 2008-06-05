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

#include <iostream>

//! Name the unit test module
#define BOOST_TEST_MODULE LJForceTests
#include "boost_utf_configure.h"

#include <boost/test/floating_point_comparison.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include "LJForceCompute.h"
#include "LJForceComputeThreaded.h"
#ifdef USE_CUDA
#include "LJForceComputeGPU.h"
#endif

#include "BinnedNeighborList.h"
#include "Initializers.h"

#include <math.h>

using namespace std;
using namespace boost;

/*! \file lj_force_test.cc
	\brief Implements unit tests for LJForceCompute and descendants
	\ingroup unit_tests
*/

//! Helper macro for testing if two numbers are close
#define MY_BOOST_CHECK_CLOSE(a,b,c) BOOST_CHECK_CLOSE(a,Scalar(b),Scalar(c))
//! Helper macro for testing if a number is small
#define MY_BOOST_CHECK_SMALL(a,c) BOOST_CHECK_SMALL(a,Scalar(c))

//! Tolerance in percent to use for comparing various LJForceComputes to each other
#ifdef SINGLE_PRECISION
const Scalar tol = Scalar(1);
#else
const Scalar tol = 1e-6;
#endif

//! Typedef'd LJForceCompute factory
typedef boost::function<shared_ptr<LJForceCompute> (shared_ptr<ParticleData> pdata, shared_ptr<NeighborList> nlist, Scalar r_cut)> ljforce_creator;
	
//! Test the ability of the lj force compute to actually calucate forces
/*! \param lj_creator Function that creates a BondForceCompute
	\note With the creator as a parameter, the same code can be used to test any derived child
		of LJForceCompute
*/
void lj_force_particle_test(ljforce_creator lj_creator)
	{
	// this 3-particle test subtly checks several conditions
	// the particles are arranged on the x axis,  1   2   3
	// such that 2 is inside the cuttoff radius of 1 and 3, but 1 and 3 are outside the cuttoff
	// of course, the buffer will be set on the neighborlist so that 3 is included in it
	// thus, this case tests the ability of the force summer to sum more than one force on
	// a particle and ignore a particle outside the radius
	
	// periodic boundary conditions will be handeled in another test
	shared_ptr<ParticleData> pdata_3(new ParticleData(3, BoxDim(1000.0), 1));
	ParticleDataArrays arrays = pdata_3->acquireReadWrite();
	arrays.x[0] = arrays.y[0] = arrays.z[0] = 0.0;
	arrays.x[1] = Scalar(pow(2.0,1.0/6.0)); arrays.y[1] = arrays.z[1] = 0.0;
	arrays.x[2] = Scalar(2.0*pow(2.0,1.0/6.0)); arrays.y[2] = arrays.z[2] = 0.0;
	pdata_3->release();
	shared_ptr<NeighborList> nlist_3(new NeighborList(pdata_3, Scalar(1.3), Scalar(3.0)));
	shared_ptr<LJForceCompute> fc_3 = lj_creator(pdata_3, nlist_3, Scalar(1.3));
	
	// first test: setup a sigma of 1.0 so that all forces will be 0
	Scalar epsilon = Scalar(1.15);
	Scalar sigma = Scalar(1.0);
	Scalar alpha = Scalar(1.0);
	Scalar lj1 = Scalar(48.0) * epsilon * pow(sigma,Scalar(12.0));
	Scalar lj2 = alpha * Scalar(24.0) * epsilon * pow(sigma,Scalar(6.0));
	fc_3->setParams(0,0,lj1,lj2);
	
	// compute the forces
	fc_3->compute(0);
	
	ForceDataArrays force_arrays = fc_3->acquire();
	MY_BOOST_CHECK_SMALL(force_arrays.fx[0], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fy[0], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[0], tol);

	MY_BOOST_CHECK_SMALL(force_arrays.fx[1], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fy[1], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[1], tol);

	MY_BOOST_CHECK_SMALL(force_arrays.fx[2], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fy[2], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[2], tol);
	
	// now change sigma and alpha so we can check that it is computing the right force
	sigma = Scalar(1.2); // < bigger sigma should push particle 0 left and particle 2 right
	alpha = Scalar(0.45);
	lj1 = Scalar(48.0) * epsilon * pow(sigma,Scalar(12.0));
	lj2 = alpha * Scalar(24.0) * epsilon * pow(sigma,Scalar(6.0));	
	fc_3->setParams(0,0,lj1,lj2);
	fc_3->compute(1);
	
	force_arrays = fc_3->acquire();
	MY_BOOST_CHECK_CLOSE(force_arrays.fx[0], -93.09822608552962, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fy[0], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[0], tol);

	// center particle should still be a 0 force by symmetry
	MY_BOOST_CHECK_SMALL(force_arrays.fx[1], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fy[1], 1e-5);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[1], 1e-5);

	MY_BOOST_CHECK_CLOSE(force_arrays.fx[2], 93.09822608552962, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fy[2], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[2], tol);
	
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
	MY_BOOST_CHECK_CLOSE(force_arrays.fx[0], 93.09822608552962, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fx[2], -93.09822608552962, tol);
	}

//! Tests the ability of a LJForceCompute to handle periodic boundary conditions
/*! \param lj_creator Function that creates a BondForceCompute
	\note With the creator as a parameter, the same code can be used to test any derived child
		of LJForceCompute
	\note This test also indirectly verifies the ability of the force compute to handle multiple
		particle types.
*/
void lj_force_periodic_test(ljforce_creator lj_creator)
	{
	////////////////////////////////////////////////////////////////////
	// now, lets do a more thorough test and include boundary conditions
	// there are way too many permutations to test here, so I will simply
	// test +x, -x, +y, -y, +z, and -z independantly
	// build a 6 particle system with particles across each boundary
	// also test the ability of the force compute to use different particle types
	
	shared_ptr<ParticleData> pdata_6(new ParticleData(6, BoxDim(20.0, 40.0, 60.0), 3));
	ParticleDataArrays arrays = pdata_6->acquireReadWrite();
	arrays.x[0] = Scalar(-9.6); arrays.y[0] = 0; arrays.z[0] = 0.0;
	arrays.x[1] =  Scalar(9.6); arrays.y[1] = 0; arrays.z[1] = 0.0;
	arrays.x[2] = 0; arrays.y[2] = Scalar(-19.6); arrays.z[2] = 0.0;
	arrays.x[3] = 0; arrays.y[3] = Scalar(19.6); arrays.z[3] = 0.0;
	arrays.x[4] = 0; arrays.y[4] = 0; arrays.z[4] = Scalar(-29.6);
	arrays.x[5] = 0; arrays.y[5] = 0; arrays.z[5] =  Scalar(29.6);
	
	arrays.type[0] = 0;
	arrays.type[1] = 1;
	arrays.type[2] = 2;
	arrays.type[3] = 0;
	arrays.type[4] = 2;
	arrays.type[5] = 1;
	pdata_6->release();
	
	shared_ptr<NeighborList> nlist_6(new NeighborList(pdata_6, Scalar(1.3), Scalar(3.0)));
	shared_ptr<LJForceCompute> fc_6 = lj_creator(pdata_6, nlist_6, Scalar(1.3));
		
	// first test: setup a sigma of 1.0 so that all forces will be 0
	Scalar epsilon = Scalar(1.0);
	Scalar sigma = Scalar(0.5);
	Scalar alpha = Scalar(0.45);
	Scalar lj1 = Scalar(48.0) * epsilon * pow(sigma,Scalar(12.0));
	Scalar lj2 = alpha * Scalar(24.0) * epsilon * pow(sigma,Scalar(6.0));
	
	// make life easy: just change epsilon for the different pairs
	fc_6->setParams(0,0,lj1,lj2);
	fc_6->setParams(0,1,Scalar(2.0)*lj1,Scalar(2.0)*lj2);
	fc_6->setParams(0,2,Scalar(3.0)*lj1,Scalar(3.0)*lj2);
	fc_6->setParams(1,1,Scalar(4.0)*lj1,Scalar(4.0)*lj2);
	fc_6->setParams(1,2,Scalar(5.0)*lj1,Scalar(5.0)*lj2);
	fc_6->setParams(2,2,Scalar(6.0)*lj1,Scalar(6.0)*lj2);
	
	fc_6->compute(0);	
	
	ForceDataArrays force_arrays = fc_6->acquire();
	// particle 0 should be pulled left
	MY_BOOST_CHECK_CLOSE(force_arrays.fx[0], -1.18299976747949, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fy[0], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[0], tol);

	// particle 1 should be pulled right
	MY_BOOST_CHECK_CLOSE(force_arrays.fx[1], 1.18299976747949, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fy[1], 1e-5);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[1], 1e-5);
	
	// particle 2 should be pulled down
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[2], -1.77449965121923, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fx[2], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[2], tol);

	// particle 3 should be pulled up
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[3], 1.77449965121923, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fx[3], 1e-5);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[3], 1e-5);	
	
	// particle 4 should be pulled back
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[4], -2.95749941869871, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fx[4], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fy[4], tol);

	// particle 3 should be pulled forward
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[5], 2.95749941869871, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fx[5], 1e-5);
	MY_BOOST_CHECK_SMALL(force_arrays.fy[5], 1e-5);
	}
	
//! Unit test a comparison between 2 LJForceComputes on a "real" system
void lj_force_comparison_test(ljforce_creator lj_creator1, ljforce_creator lj_creator2)
	{
	const unsigned int N = 5000;
	
	// create a random particle system to sum forces on
	RandomInitializer rand_init(N, 0.2, 0.9, "A");
	shared_ptr<ParticleData> pdata(new ParticleData(rand_init));
	shared_ptr<BinnedNeighborList> nlist(new BinnedNeighborList(pdata, Scalar(3.0), Scalar(0.8)));
	
	shared_ptr<LJForceCompute> fc1 = lj_creator1(pdata, nlist, Scalar(3.0));
	shared_ptr<LJForceCompute> fc2 = lj_creator2(pdata, nlist, Scalar(3.0));
		
	// setup some values for alpha and sigma
	Scalar epsilon = Scalar(1.0);
	Scalar sigma = Scalar(1.2);
	Scalar alpha = Scalar(0.45);
	Scalar lj1 = Scalar(48.0) * epsilon * pow(sigma,Scalar(12.0));
	Scalar lj2 = alpha * Scalar(24.0) * epsilon * pow(sigma,Scalar(6.0));
	
	// specify the force parameters
	fc1->setParams(0,0,lj1,lj2);
	fc2->setParams(0,0,lj1,lj2);
	
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
		}
	}
	
//! LJForceCompute creator for unit tests
shared_ptr<LJForceCompute> base_class_lj_creator(shared_ptr<ParticleData> pdata, shared_ptr<NeighborList> nlist, Scalar r_cut)
	{
	return shared_ptr<LJForceCompute>(new LJForceCompute(pdata, nlist, r_cut));
	}
	
//! LJForceComputeThreaded for unit tests
shared_ptr<LJForceCompute> threaded_lj_creator(shared_ptr<ParticleData> pdata, shared_ptr<NeighborList> nlist, Scalar r_cut, int nthreads)
	{
	return shared_ptr<LJForceCompute>(new LJForceComputeThreaded(pdata, nlist, r_cut, nthreads));
	}
	
#ifdef USE_CUDA
//! LJForceComputeGPU creator for unit tests
shared_ptr<LJForceCompute> gpu_lj_creator(shared_ptr<ParticleData> pdata, shared_ptr<NeighborList> nlist, Scalar r_cut)
	{
	nlist->setStorageMode(NeighborList::full);
	shared_ptr<LJForceComputeGPU> lj(new LJForceComputeGPU(pdata, nlist, r_cut));
	// the default block size kills valgrind :) reduce it
	lj->setBlockSize(64);
	return lj;
	}
#endif
	
//! boost test case for particle test on CPU
BOOST_AUTO_TEST_CASE( LJForce_particle )
	{
	ljforce_creator lj_creator_base = bind(base_class_lj_creator, _1, _2, _3);
	lj_force_particle_test(lj_creator_base);
	}
	
//! boost test case for periodic test on CPU
BOOST_AUTO_TEST_CASE( LJForce_periodic )
	{
	ljforce_creator lj_creator_base = bind(base_class_lj_creator, _1, _2, _3);
	lj_force_periodic_test(lj_creator_base);
	}
	
//! boost test case for particle test on CPU - threaded
BOOST_AUTO_TEST_CASE( LJForceThreaded_particle )
	{
	ljforce_creator lj_creator_threaded = bind(threaded_lj_creator, _1, _2, _3, 2);
	lj_force_particle_test(lj_creator_threaded);
	}
	
//! boost test case for periodic test on CPU - threaded
BOOST_AUTO_TEST_CASE( LJForceThreaded_periodic )
	{
	ljforce_creator lj_creator_threaded = bind(threaded_lj_creator, _1, _2, _3, 2);
	lj_force_periodic_test(lj_creator_threaded);
	}

//! boost test case for comparing threaded output to base class output
BOOST_AUTO_TEST_CASE( LJForceThreaded_compare )
	{
	ljforce_creator lj_creator_threaded = bind(threaded_lj_creator, _1, _2, _3, 2);
	ljforce_creator lj_creator_base = bind(base_class_lj_creator, _1, _2, _3);
	lj_force_comparison_test(lj_creator_base, lj_creator_threaded);
	}
	
# ifdef USE_CUDA
//! boost test case for particle test on CPU - threaded
BOOST_AUTO_TEST_CASE( LJForceGPU_particle )
	{
	ljforce_creator lj_creator_gpu = bind(gpu_lj_creator, _1, _2, _3);
	lj_force_particle_test(lj_creator_gpu);
	}

//! boost test case for periodic test on the GPU
BOOST_AUTO_TEST_CASE( LJForceGPU_periodic )
	{
	ljforce_creator lj_creator_gpu = bind(gpu_lj_creator, _1, _2, _3);
	lj_force_periodic_test(lj_creator_gpu);
	}

//! boost test case for comparing GPU output to base class output
BOOST_AUTO_TEST_CASE( LJForceGPU_compare )
	{
	ljforce_creator lj_creator_gpu = bind(gpu_lj_creator, _1, _2, _3);
	ljforce_creator lj_creator_base = bind(base_class_lj_creator, _1, _2, _3);
	lj_force_comparison_test(lj_creator_base, lj_creator_gpu);
	}
#endif

