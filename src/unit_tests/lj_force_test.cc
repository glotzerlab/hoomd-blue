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
// Maintainer: joaander

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <iostream>
#include <fstream>

//! Name the unit test module
#define BOOST_TEST_MODULE LJForceTests
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
const Scalar tol = Scalar(4);
#else
const Scalar tol = 1e-6;
#endif
//! Global tolerance for check_small comparisons
const Scalar tol_small = 1e-4;

//! Typedef'd LJForceCompute factory
typedef boost::function<shared_ptr<LJForceCompute> (shared_ptr<ParticleData> pdata, shared_ptr<NeighborList> nlist, Scalar r_cut)> ljforce_creator;
	
//! Test the ability of the lj force compute to actually calucate forces
void lj_force_particle_test(ljforce_creator lj_creator, ExecutionConfiguration exec_conf)
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
	
	// periodic boundary conditions will be handeled in another test
	shared_ptr<ParticleData> pdata_3(new ParticleData(3, BoxDim(1000.0), 1, 0, 0, 0, 0, exec_conf));
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
	Scalar lj1 = Scalar(4.0) * epsilon * pow(sigma,Scalar(12.0));
	Scalar lj2 = alpha * Scalar(4.0) * epsilon * pow(sigma,Scalar(6.0));
	fc_3->setParams(0,0,lj1,lj2);
	
	// compute the forces
	fc_3->compute(0);
	
	ForceDataArrays force_arrays = fc_3->acquire();
	MY_BOOST_CHECK_SMALL(force_arrays.fx[0], tol_small);
	MY_BOOST_CHECK_SMALL(force_arrays.fy[0], tol_small);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[0], tol_small);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[0], -0.575, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.virial[0], tol_small);

	MY_BOOST_CHECK_SMALL(force_arrays.fx[1], tol_small);
	MY_BOOST_CHECK_SMALL(force_arrays.fy[1], tol_small);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[1], tol_small);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[1], -1.15, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.virial[1], tol_small);

	MY_BOOST_CHECK_SMALL(force_arrays.fx[2], tol_small);
	MY_BOOST_CHECK_SMALL(force_arrays.fy[2], tol_small);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[2], tol_small);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[2], -0.575, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.virial[2], tol_small);
	
	// now change sigma and alpha so we can check that it is computing the right force
	sigma = Scalar(1.2); // < bigger sigma should push particle 0 left and particle 2 right
	alpha = Scalar(0.45);
	lj1 = Scalar(4.0) * epsilon * pow(sigma,Scalar(12.0));
	lj2 = alpha * Scalar(4.0) * epsilon * pow(sigma,Scalar(6.0));	
	fc_3->setParams(0,0,lj1,lj2);
	fc_3->compute(1);
	
	force_arrays = fc_3->acquire();
	MY_BOOST_CHECK_CLOSE(force_arrays.fx[0], -93.09822608552962, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fy[0], tol_small);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[0], tol_small);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[0], 3.5815110377468, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.virial[0], 17.416537590989, tol);

	// center particle should still be a 0 force by symmetry
	MY_BOOST_CHECK_SMALL(force_arrays.fx[1], tol_small);
	MY_BOOST_CHECK_SMALL(force_arrays.fy[1], tol_small);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[1], tol_small);
	// there is still an energy and virial, though
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[1], 7.1630220754935, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.virial[1], 34.833075181975, tol);

	MY_BOOST_CHECK_CLOSE(force_arrays.fx[2], 93.09822608552962, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fy[2], tol_small);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[2], tol_small);
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
	MY_BOOST_CHECK_CLOSE(force_arrays.fx[0], 93.09822608552962, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fx[2], -93.09822608552962, tol);
	}

//! Tests the ability of a LJForceCompute to handle periodic boundary conditions
void lj_force_periodic_test(ljforce_creator lj_creator, ExecutionConfiguration exec_conf)
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
	
	shared_ptr<ParticleData> pdata_6(new ParticleData(6, BoxDim(20.0, 40.0, 60.0), 3, 0, 0, 0, 0, exec_conf));
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
		
	// choose a small sigma so that all interactions are attractive
	Scalar epsilon = Scalar(1.0);
	Scalar sigma = Scalar(0.5);
	Scalar alpha = Scalar(0.45);
	Scalar lj1 = Scalar(4.0) * epsilon * pow(sigma,Scalar(12.0));
	Scalar lj2 = alpha * Scalar(4.0) * epsilon * pow(sigma,Scalar(6.0));
	
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
	MY_BOOST_CHECK_SMALL(force_arrays.fy[0], tol_small);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[0], tol_small);
	MY_BOOST_CHECK_CLOSE(force_arrays.virial[0], -0.15773330233059, tol);

	// particle 1 should be pulled right
	MY_BOOST_CHECK_CLOSE(force_arrays.fx[1], 1.18299976747949, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fy[1], tol_small);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[1], tol_small);
	MY_BOOST_CHECK_CLOSE(force_arrays.virial[1], -0.15773330233059, tol);
	
	// particle 2 should be pulled down
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[2], -1.77449965121923, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fx[2], tol_small);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[2], tol_small);
	MY_BOOST_CHECK_CLOSE(force_arrays.virial[2], -0.23659995349591, tol);

	// particle 3 should be pulled up
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[3], 1.77449965121923, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fx[3], tol_small);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[3], tol_small);
	MY_BOOST_CHECK_CLOSE(force_arrays.virial[3], -0.23659995349591, tol);
	
	// particle 4 should be pulled back
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[4], -2.95749941869871, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fx[4], tol_small);
	MY_BOOST_CHECK_SMALL(force_arrays.fy[4], tol_small);
	MY_BOOST_CHECK_CLOSE(force_arrays.virial[4], -0.39433325582651, tol);

	// particle 3 should be pulled forward
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[5], 2.95749941869871, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fx[5], tol_small);
	MY_BOOST_CHECK_SMALL(force_arrays.fy[5], tol_small);
	MY_BOOST_CHECK_CLOSE(force_arrays.virial[5], -0.39433325582651, tol);
	}
	
//! Unit test a comparison between 2 LJForceComputes on a "real" system
void lj_force_comparison_test(ljforce_creator lj_creator1, ljforce_creator lj_creator2, ExecutionConfiguration exec_conf)
	{
	#ifdef CUDA
	g_gpu_error_checking = true;
	#endif
	
	const unsigned int N = 5000;
	
	// create a random particle system to sum forces on
	RandomInitializer rand_init(N, Scalar(0.2), Scalar(0.9), "A");
	shared_ptr<ParticleData> pdata(new ParticleData(rand_init, exec_conf));
	shared_ptr<BinnedNeighborList> nlist(new BinnedNeighborList(pdata, Scalar(3.0), Scalar(0.8)));
	
	shared_ptr<LJForceCompute> fc1 = lj_creator1(pdata, nlist, Scalar(3.0));
	shared_ptr<LJForceCompute> fc2 = lj_creator2(pdata, nlist, Scalar(3.0));
		
	// setup some values for alpha and sigma
	Scalar epsilon = Scalar(1.0);
	Scalar sigma = Scalar(1.2);
	Scalar alpha = Scalar(0.45);
	Scalar lj1 = Scalar(4.0) * epsilon * pow(sigma,Scalar(12.0));
	Scalar lj2 = alpha * Scalar(4.0) * epsilon * pow(sigma,Scalar(6.0));
	
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
		BOOST_CHECK_CLOSE(arrays1.pe[i], arrays2.pe[i], tol);
		BOOST_CHECK_CLOSE(arrays1.virial[i], arrays2.virial[i], tol);
		}
	}
	
//! Test the ability of the lj force compute to compute forces with different shift modes
void lj_force_shift_test(ljforce_creator lj_creator, ExecutionConfiguration exec_conf)
	{
	#ifdef CUDA
	g_gpu_error_checking = true;
	#endif
	
	// this 2-particle test is just to get a plot of the potential and force vs r cut
	shared_ptr<ParticleData> pdata_2(new ParticleData(2, BoxDim(1000.0), 1, 0, 0, 0, 0, exec_conf));
	ParticleDataArrays arrays = pdata_2->acquireReadWrite();
	arrays.x[0] = arrays.y[0] = arrays.z[0] = 0.0;
	arrays.x[1] = Scalar(2.8); arrays.y[1] = arrays.z[1] = 0.0;
	pdata_2->release();
	shared_ptr<NeighborList> nlist_2(new NeighborList(pdata_2, Scalar(3.0), Scalar(0.8)));
	shared_ptr<LJForceCompute> fc_no_shift = lj_creator(pdata_2, nlist_2, Scalar(3.0));
	fc_no_shift->setShiftMode(LJForceCompute::no_shift);
	shared_ptr<LJForceCompute> fc_shift = lj_creator(pdata_2, nlist_2, Scalar(3.0));
	fc_shift->setShiftMode(LJForceCompute::shift);
	shared_ptr<LJForceCompute> fc_xplor = lj_creator(pdata_2, nlist_2, Scalar(3.0));
	fc_xplor->setShiftMode(LJForceCompute::xplor);
	fc_xplor->setXplorFraction(Scalar(2.0/3.0));
	
	nlist_2->setStorageMode(NeighborList::full);

	// setup a standard epsilon and sigma
	Scalar epsilon = Scalar(1.0);
	Scalar sigma = Scalar(1.0);
	Scalar alpha = Scalar(1.0);
	Scalar lj1 = Scalar(4.0) * epsilon * pow(sigma,Scalar(12.0));
	Scalar lj2 = alpha * Scalar(4.0) * epsilon * pow(sigma,Scalar(6.0));
	fc_no_shift->setParams(0,0,lj1,lj2);
	fc_shift->setParams(0,0,lj1,lj2);
	fc_xplor->setParams(0,0,lj1,lj2);
	
	fc_no_shift->compute(0);
	fc_shift->compute(0);
	fc_xplor->compute(0);

	ForceDataArrays force_arrays_no_shift = fc_no_shift->acquire();
	ForceDataArrays force_arrays_shift = fc_shift->acquire();
	ForceDataArrays force_arrays_xplor = fc_xplor->acquire();

	MY_BOOST_CHECK_CLOSE(force_arrays_no_shift.fx[0], 0.017713272731914, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays_no_shift.pe[0], -0.0041417095577326, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays_no_shift.fx[1], -0.017713272731914, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays_no_shift.pe[1], -0.0041417095577326, tol);

	// shifted just has pe shifted by a given amount
	MY_BOOST_CHECK_CLOSE(force_arrays_shift.fx[0], 0.017713272731914, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays_shift.pe[0], -0.0014019886856134, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays_shift.fx[1], -0.017713272731914, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays_shift.pe[1], -0.0014019886856134, tol);

	// xplor has slight tweaks
	MY_BOOST_CHECK_CLOSE(force_arrays_xplor.fx[0], 0.012335911924312, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays_xplor.pe[0], -0.001130667359194/2.0, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays_xplor.fx[1], -0.012335911924312, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays_xplor.pe[1], -0.001130667359194/2.0, tol);
	
	// check again, prior to r_on to make sure xplor isn't doing something weird
	arrays = pdata_2->acquireReadWrite();
	arrays.x[0] = arrays.y[0] = arrays.z[0] = 0.0;
	arrays.x[1] = Scalar(1.5); arrays.y[1] = arrays.z[1] = 0.0;
	pdata_2->release();
	
	fc_no_shift->compute(1);
	fc_shift->compute(1);
	fc_xplor->compute(1);

	force_arrays_no_shift = fc_no_shift->acquire();
	force_arrays_shift = fc_shift->acquire();
	force_arrays_xplor = fc_xplor->acquire();

	MY_BOOST_CHECK_CLOSE(force_arrays_no_shift.fx[0], 1.1580288310461, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays_no_shift.pe[0], -0.16016829713928, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays_no_shift.fx[1], -1.1580288310461, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays_no_shift.pe[1], -0.16016829713928, tol);

	// shifted just has pe shifted by a given amount
	MY_BOOST_CHECK_CLOSE(force_arrays_shift.fx[0], 1.1580288310461, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays_shift.pe[0], -0.15742857626716, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays_shift.fx[1], -1.1580288310461, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays_shift.pe[1], -0.15742857626716, tol);

	// xplor has slight tweaks
	MY_BOOST_CHECK_CLOSE(force_arrays_xplor.fx[0], 1.1580288310461, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays_xplor.pe[0], -0.16016829713928, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays_xplor.fx[1], -1.1580288310461, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays_xplor.pe[1], -0.16016829713928, tol);
	
	// check once again to verify that nothing fish happens past r_cut
	arrays = pdata_2->acquireReadWrite();
	arrays.x[0] = arrays.y[0] = arrays.z[0] = 0.0;
	arrays.x[1] = Scalar(3.1); arrays.y[1] = arrays.z[1] = 0.0;
	pdata_2->release();
	
	fc_no_shift->compute(2);
	fc_shift->compute(2);
	fc_xplor->compute(2);

	force_arrays_no_shift = fc_no_shift->acquire();
	force_arrays_shift = fc_shift->acquire();
	force_arrays_xplor = fc_xplor->acquire();

	MY_BOOST_CHECK_SMALL(force_arrays_no_shift.fx[0], tol_small);
	MY_BOOST_CHECK_SMALL(force_arrays_no_shift.pe[0], tol_small);
	MY_BOOST_CHECK_SMALL(force_arrays_no_shift.fx[1], tol_small);
	MY_BOOST_CHECK_SMALL(force_arrays_no_shift.pe[1], tol_small);

	// shifted just has pe shifted by a given amount
	MY_BOOST_CHECK_SMALL(force_arrays_shift.fx[0], tol_small);
	MY_BOOST_CHECK_SMALL(force_arrays_shift.pe[0], tol_small);
	MY_BOOST_CHECK_SMALL(force_arrays_shift.fx[1], tol_small);
	MY_BOOST_CHECK_SMALL(force_arrays_shift.pe[1], tol_small);

	// xplor has slight tweaks
	MY_BOOST_CHECK_SMALL(force_arrays_xplor.fx[0], tol_small);
	MY_BOOST_CHECK_SMALL(force_arrays_xplor.pe[0], tol_small);
	MY_BOOST_CHECK_SMALL(force_arrays_xplor.fx[1], tol_small);
	MY_BOOST_CHECK_SMALL(force_arrays_xplor.pe[1], tol_small);
	}

//! LJForceCompute creator for unit tests
shared_ptr<LJForceCompute> base_class_lj_creator(shared_ptr<ParticleData> pdata, shared_ptr<NeighborList> nlist, Scalar r_cut)
	{
	return shared_ptr<LJForceCompute>(new LJForceCompute(pdata, nlist, r_cut));
	}

#ifdef ENABLE_CUDA
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
	lj_force_particle_test(lj_creator_base, ExecutionConfiguration(ExecutionConfiguration::CPU, 0));
	}
	
//! boost test case for periodic test on CPU
BOOST_AUTO_TEST_CASE( LJForce_periodic )
	{
	ljforce_creator lj_creator_base = bind(base_class_lj_creator, _1, _2, _3);
	lj_force_periodic_test(lj_creator_base, ExecutionConfiguration(ExecutionConfiguration::CPU, 0));
	}

//! boost test case for particle test on CPU
BOOST_AUTO_TEST_CASE( LJForce_shift )
	{
	ljforce_creator lj_creator_base = bind(base_class_lj_creator, _1, _2, _3);
	lj_force_shift_test(lj_creator_base, ExecutionConfiguration(ExecutionConfiguration::CPU, 0));
	}
	
# ifdef ENABLE_CUDA
//! boost test case for particle test on GPU
BOOST_AUTO_TEST_CASE( LJForceGPU_particle )
	{
	ljforce_creator lj_creator_gpu = bind(gpu_lj_creator, _1, _2, _3);
	lj_force_particle_test(lj_creator_gpu, ExecutionConfiguration(ExecutionConfiguration::GPU, ExecutionConfiguration::getDefaultGPU()));
	}

//! boost test case for periodic test on the GPU
BOOST_AUTO_TEST_CASE( LJForceGPU_periodic )
	{
	ljforce_creator lj_creator_gpu = bind(gpu_lj_creator, _1, _2, _3);
	lj_force_periodic_test(lj_creator_gpu, ExecutionConfiguration(ExecutionConfiguration::GPU, ExecutionConfiguration::getDefaultGPU()));
	}

//! boost test case for shift test on GPU
BOOST_AUTO_TEST_CASE( LJForceGPU_shift )
	{
	ljforce_creator lj_creator_gpu = bind(gpu_lj_creator, _1, _2, _3);
	lj_force_shift_test(lj_creator_gpu, ExecutionConfiguration(ExecutionConfiguration::GPU, ExecutionConfiguration::getDefaultGPU()));
	}

//! boost test case for comparing GPU output to base class output
BOOST_AUTO_TEST_CASE( LJForceGPU_compare )
	{
	ljforce_creator lj_creator_gpu = bind(gpu_lj_creator, _1, _2, _3);
	ljforce_creator lj_creator_base = bind(base_class_lj_creator, _1, _2, _3);
	lj_force_comparison_test(lj_creator_base, lj_creator_gpu, ExecutionConfiguration(ExecutionConfiguration::GPU, ExecutionConfiguration::getDefaultGPU()));
	}
	
//! boost test case for comparing multi-GPU output to base class output
BOOST_AUTO_TEST_CASE( LJForceMultiGPU_compare )
	{
	vector<unsigned int> gpu_list;
	gpu_list.push_back(ExecutionConfiguration::getDefaultGPU());
	gpu_list.push_back(ExecutionConfiguration::getDefaultGPU());
	gpu_list.push_back(ExecutionConfiguration::getDefaultGPU());
	gpu_list.push_back(ExecutionConfiguration::getDefaultGPU());
	ExecutionConfiguration exec_conf(ExecutionConfiguration::GPU, gpu_list);

	ljforce_creator lj_creator_gpu = bind(gpu_lj_creator, _1, _2, _3);
	ljforce_creator lj_creator_base = bind(base_class_lj_creator, _1, _2, _3);
	lj_force_comparison_test(lj_creator_base, lj_creator_gpu, exec_conf);
	}
#endif

/*BOOST_AUTO_TEST_CASE(potential_writer)
	{
	#ifdef CUDA
	g_gpu_error_checking = true;
	#endif
	
	// this 2-particle test is just to get a plot of the potential and force vs r cut
	shared_ptr<ParticleData> pdata_2(new ParticleData(2, BoxDim(1000.0), 1, 0, ExecutionConfiguration()));
	ParticleDataArrays arrays = pdata_2->acquireReadWrite();
	arrays.x[0] = arrays.y[0] = arrays.z[0] = 0.0;
	arrays.x[1] = Scalar(0.9); arrays.y[1] = arrays.z[1] = 0.0;
	pdata_2->release();
	shared_ptr<NeighborList> nlist_2(new NeighborList(pdata_2, Scalar(3.0), Scalar(0.8)));
	shared_ptr<LJForceCompute> fc(new LJForceCompute(pdata_2, nlist_2, Scalar(3.0)));
	// nlist_2->setStorageMode(NeighborList::full);
	fc->setShiftMode(LJForceCompute::xplor);

	// setup a standard epsilon and sigma
	Scalar epsilon = Scalar(1.0);
	Scalar sigma = Scalar(1.0);
	Scalar alpha = Scalar(1.0);
	Scalar lj1 = Scalar(4.0) * epsilon * pow(sigma,Scalar(12.0));
	Scalar lj2 = alpha * Scalar(4.0) * epsilon * pow(sigma,Scalar(6.0));
	fc->setParams(0,0,lj1,lj2);
	
	ofstream f("lj_dat.m");
	f << "lj = [";
	unsigned int count = 0;	
	for (float r = 0.96; r <= 3.5; r+= 0.001)
		{
		// set the distance
		ParticleDataArrays arrays = pdata_2->acquireReadWrite();
		arrays.x[0] = arrays.y[0] = arrays.z[0] = 0.0;
		arrays.x[1] = Scalar(r); arrays.y[1] = arrays.z[1] = 0.0;
		pdata_2->release();
		
		// compute the forces
		fc->compute(count);
		count++;
	
		ForceDataArrays force_arrays = fc->acquire();
		f << r << " " << force_arrays.fx[0] << " " << fc->calcEnergySum() << " ; " << endl;	
		}
	f << "];" << endl;
	f.close();
	}*/

#ifdef WIN32
#pragma warning( pop )
#endif
