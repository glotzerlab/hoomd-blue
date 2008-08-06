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

//! Name the boost unit test module
#define BOOST_TEST_MODULE BondForceTests
#include "boost_utf_configure.h"

#include <boost/test/floating_point_comparison.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>

#include "HarmonicBondForceCompute.h"
#include "ConstForceCompute.h"
#ifdef USE_CUDA
#include "BondForceComputeGPU.h"
#endif

#include "Initializers.h"

using namespace std;
using namespace boost;

/*! \file bond_force_test.cc
	\brief Implements unit tests for BondForceCompute and child classes
	\ingroup unit_tests
*/

//! Helper macro for testing if two numbers are close
#define MY_BOOST_CHECK_CLOSE(a,b,c) BOOST_CHECK_CLOSE(a,Scalar(b),Scalar(c))

//! Global tolerance for floating point comparisons
#ifdef SINGLE_PRECISION
const Scalar tol = Scalar(1e-1);
#else
const Scalar tol = 1e-6;
#endif

//! Typedef to make using the boost::function factory easier
typedef boost::function<shared_ptr<HarmonicBondForceCompute>  (shared_ptr<ParticleData> pdata)> bondforce_creator;

//! Perform some simple functionality tests of any BondForceCompute
/*! \param bf_creator Function that creates a BondForceCompute
	\note With the creator as a parameter, the same code can be used to test any derived child
		of the BondForceCompute
*/
void bond_force_basic_tests(bondforce_creator bf_creator)
	{
	/////////////////////////////////////////////////////////
	// start with the simplest possible test: 2 particles in a huge box with only one bond type
	shared_ptr<ParticleData> pdata_2(new ParticleData(2, BoxDim(1000.0), 1, 1));
	ParticleDataArrays arrays = pdata_2->acquireReadWrite();
	arrays.x[0] = arrays.y[0] = arrays.z[0] = 0.0;
	arrays.x[1] = Scalar(0.9);
	arrays.y[1] = arrays.z[1] = 0.0;
	pdata_2->release();

	// create the bond force compute to check
	shared_ptr<HarmonicBondForceCompute> fc_2 = bf_creator(pdata_2);
	fc_2->setParams(0, 1.5, 0.75);

	// compute the force and check the results
	fc_2->compute(0);
	ForceDataArrays force_arrays = fc_2->acquire();
	// check that the force is correct, it should be 0 since we haven't created any bonds yet
	MY_BOOST_CHECK_CLOSE(force_arrays.fx[0], 0.0, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[0], 0.0, tol);
	
	// add a bond and check again
	pdata_2->getBondData()->addBond(Bond(0, 0,1));
	fc_2->compute(1);
	
	// this time there should be a force
	force_arrays = fc_2->acquire();
	MY_BOOST_CHECK_CLOSE(force_arrays.fx[0], 0.225, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[0], 0.0084375, tol);
		
	// check that the two forces are negatives of each other
	MY_BOOST_CHECK_CLOSE(force_arrays.fx[0], -force_arrays.fx[1], tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[0], -force_arrays.fy[1], tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[0], -force_arrays.fz[1], tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[0], force_arrays.pe[1], tol);

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
	MY_BOOST_CHECK_CLOSE(force_arrays.fx[0], -0.225, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fx[1], 0.225, tol);

	////////////////////////////////////////////////////////////////////
	// now, lets do a more thorough test and include boundary conditions
	// there are way too many permutations to test here, so I will simply
	// test +x, -x, +y, -y, +z, and -z independantly
	// build a 6 particle system with particles across each boundary
	// also test more than one type of bond
	shared_ptr<ParticleData> pdata_6(new ParticleData(6, BoxDim(20.0, 40.0, 60.0), 1, 3));
	arrays = pdata_6->acquireReadWrite();
	arrays.x[0] = Scalar(-9.6); arrays.y[0] = 0; arrays.z[0] = 0.0;
	arrays.x[1] =  Scalar(9.6); arrays.y[1] = 0; arrays.z[1] = 0.0;
	arrays.x[2] = 0; arrays.y[2] = Scalar(-19.6); arrays.z[2] = 0.0;
	arrays.x[3] = 0; arrays.y[3] = Scalar(19.6); arrays.z[3] = 0.0;
	arrays.x[4] = 0; arrays.y[4] = 0; arrays.z[4] = Scalar(-29.6);
	arrays.x[5] = 0; arrays.y[5] = 0; arrays.z[5] =  Scalar(29.6);
	pdata_6->release();
	
	shared_ptr<HarmonicBondForceCompute> fc_6 = bf_creator(pdata_6);
	fc_6->setParams(0, 1.5, 0.75);
	fc_6->setParams(1, 2.0*1.5, 0.75);
	fc_6->setParams(2, 1.5, 0.5);
	
	pdata_6->getBondData()->addBond(Bond(0, 0,1));
	pdata_6->getBondData()->addBond(Bond(1, 2,3));
	pdata_6->getBondData()->addBond(Bond(2, 4,5));
	
	fc_6->compute(0);
	// check that the forces are correctly computed
	force_arrays = fc_6->acquire();
	MY_BOOST_CHECK_CLOSE(force_arrays.fx[0], -0.075, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[0], 0, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[0], 0, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[0], 9.375e-4, tol);

	MY_BOOST_CHECK_CLOSE(force_arrays.fx[1], 0.075, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[1], 0, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[1], 0, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[1], 9.375e-4, tol);

	MY_BOOST_CHECK_CLOSE(force_arrays.fx[2], 0, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[2], -0.075 * 2.0, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[2], 0, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[2], 9.375e-4 * 2.0, tol);	

	MY_BOOST_CHECK_CLOSE(force_arrays.fx[3], 0, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[3], 0.075 * 2.0, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[3], 0, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[3], 9.375e-4 * 2.0, tol);	

	MY_BOOST_CHECK_CLOSE(force_arrays.fx[4], 0, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[4], 0, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[4], -0.45, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[4], 0.03375, tol);

	MY_BOOST_CHECK_CLOSE(force_arrays.fx[5], 0, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[5], 0, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[5], 0.45, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[5], 0.03375, tol);

	// one more test: this one will test two things:
	// 1) That the forces are computed correctly even if the particles are rearranged in memory
	// and 2) That two forces can add to the same particle
	shared_ptr<ParticleData> pdata_4(new ParticleData(4, BoxDim(100.0, 100.0, 100.0), 1, 1));
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
	shared_ptr<HarmonicBondForceCompute> fc_4 = bf_creator(pdata_4);
	fc_4->setParams(0, 1.5, 1.75);
	// only add bonds on the left, top, and bottom of the square
	pdata_4->getBondData()->addBond(Bond(0, 2,3));
	pdata_4->getBondData()->addBond(Bond(0, 2,0));
	pdata_4->getBondData()->addBond(Bond(0, 0,1));

	fc_4->compute(0);
	force_arrays = fc_4->acquire();
	// the right two particles shoul only have a force pulling them right
	MY_BOOST_CHECK_CLOSE(force_arrays.fx[1], 1.125, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[1], 0, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[1], 0, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[1], 0.2109375, tol);

	MY_BOOST_CHECK_CLOSE(force_arrays.fx[3], 1.125, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[3], 0, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[3], 0, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[3], 0.2109375, tol);

	// the bottom left particle should have a force pulling down and to the left
	MY_BOOST_CHECK_CLOSE(force_arrays.fx[0], -1.125, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[0], -1.125, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[0], 0, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[0], 0.421875, tol);

	// and the top left particle should have a force pulling up and to the left
	MY_BOOST_CHECK_CLOSE(force_arrays.fx[2], -1.125, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[2], 1.125, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[2], 0, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[2], 0.421875, tol);
	}
	
//! Compares the output of two HarmonicBondForceComputes
void bond_force_comparison_tests(bondforce_creator bf_creator1, bondforce_creator bf_creator2)
	{
	const unsigned int M = 10;
	const unsigned int N = M*M*M;
	
	// create a particle system to sum forces on
	// use a simple cubic array of particles so that random bonds
	// don't result in huge forces on a random particle arrangement
	SimpleCubicInitializer sc_init(M, 1.5, "A");
	shared_ptr<ParticleData> pdata(new ParticleData(sc_init));
	
	shared_ptr<HarmonicBondForceCompute> fc1 = bf_creator1(pdata);
	shared_ptr<HarmonicBondForceCompute> fc2 = bf_creator2(pdata);
	fc1->setParams(0, 300.0, 1.75);
	fc2->setParams(0, 300.0, 1.75);

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
				pdata->getBondData()->addBond(Bond(0, i*M*M + j*M + k, i*M*M + j*M + k + 1));
				}

		
	// compute the forces
	fc1->compute(0);
	fc2->compute(0);
	
	// verify that the forces are identical (within roundoff errors)
	ForceDataArrays arrays1 = fc1->acquire();
	ForceDataArrays arrays2 = fc2->acquire();

	Scalar rough_tol = Scalar(2.0);

	for (unsigned int i = 0; i < N; i++)
		{
		BOOST_CHECK_CLOSE(arrays1.fx[i], arrays2.fx[i], rough_tol);
		BOOST_CHECK_CLOSE(arrays1.fy[i], arrays2.fy[i], rough_tol);
		BOOST_CHECK_CLOSE(arrays1.fz[i], arrays2.fz[i], rough_tol);
		BOOST_CHECK_CLOSE(arrays1.pe[i], arrays2.pe[i], rough_tol);
		}
	}
	
//! Check ConstForceCompute to see that it operates properly
void const_force_test()
	{
	// Generate a simple test particle data
	shared_ptr<ParticleData> pdata_2(new ParticleData(2, BoxDim(1000.0)));
	ParticleDataArrays arrays = pdata_2->acquireReadWrite();
	arrays.x[0] = arrays.y[0] = arrays.z[0] = 0.0;
	arrays.x[1] = Scalar(0.9);
	arrays.y[1] = arrays.z[1] = 0.0;
	pdata_2->release();

	// Create the ConstForceCompute and check that it works properly
	ConstForceCompute fc(pdata_2, Scalar(-1.3), Scalar(2.5), Scalar(45.67));
	ForceDataArrays force_arrays = fc.acquire();
	MY_BOOST_CHECK_CLOSE(force_arrays.fx[0], -1.3, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[0], 2.5, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[0], 45.67, tol);

	MY_BOOST_CHECK_CLOSE(force_arrays.fx[1], -1.3, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[1], 2.5, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[1], 45.67, tol);

	// check the setforce method
	fc.setForce(Scalar(67.54), Scalar(22.1), Scalar(-1.4));
	force_arrays = fc.acquire();
	MY_BOOST_CHECK_CLOSE(force_arrays.fx[0], 67.54, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[0], 22.1, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[0], -1.4, tol);

	MY_BOOST_CHECK_CLOSE(force_arrays.fx[1], 67.54, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[1], 22.1, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[1], -1.4, tol);
	}

//! HarmonicBondForceCompute creator for bond_force_basic_tests()
shared_ptr<HarmonicBondForceCompute> base_class_bf_creator(shared_ptr<ParticleData> pdata)
	{
	return shared_ptr<HarmonicBondForceCompute>(new HarmonicBondForceCompute(pdata));
	}
	
#ifdef USE_CUDA
//! BondForceCompute creator for bond_force_basic_tests()
shared_ptr<HarmonicBondForceCompute> gpu_bf_creator(shared_ptr<ParticleData> pdata)
	{
	return shared_ptr<BondForceCompute>(new BondForceComputeGPU(pdata));
	}
#endif

//! boost test case for bond forces on the CPU
BOOST_AUTO_TEST_CASE( BondForceCompute_basic )
	{
	bondforce_creator bf_creator = bind(base_class_bf_creator, _1);
	bond_force_basic_tests(bf_creator);
	}

#ifdef USE_CUDA
//! boost test case for bond forces on the GPU
BOOST_AUTO_TEST_CASE( BondForceComputeGPU_basic )
	{
	bondforce_creator bf_creator = bind(gpu_bf_creator, _1);
	bond_force_basic_tests(bf_creator);
	}
	
//! boost test case for comparing bond GPU and CPU BondForceComputes
BOOST_AUTO_TEST_CASE( BondForceComputeGPU_compare )
	{
	bondforce_creator bf_creator_gpu = bind(gpu_bf_creator, _1);
	bondforce_creator bf_creator = bind(base_class_bf_creator, _1);
	bond_force_comparison_tests(bf_creator, bf_creator_gpu);
	}
	
#endif

//! boost test case for constant forces
BOOST_AUTO_TEST_CASE( ConstForceCompute_basic )
	{
	const_force_test();
	}

