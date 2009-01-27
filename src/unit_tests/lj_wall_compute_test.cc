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

//! Name the unit test module
#define BOOST_TEST_MODULE LJWallForceTests
#include "boost_utf_configure.h"

#include <boost/test/floating_point_comparison.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include "LJWallForceCompute.h"
#include "WallData.h"

#include <math.h>

using namespace std;
using namespace boost;

/*! \file lj_wall_compute_test.cc
	\brief Implements unit tests for LJWallForceCompute and descendants
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

//! Typedef'd LJWallForceCompute factory
typedef boost::function<shared_ptr<LJWallForceCompute> (shared_ptr<SystemDefinition> sysdef, Scalar r_cut)> ljwallforce_creator;

//! Test the ability of the lj wall force compute to actually calculate forces
void ljwall_force_particle_test(ljwallforce_creator ljwall_creator, ExecutionConfiguration exec_conf)
	{
	#ifdef CUDA
	g_gpu_error_checking = true;
	#endif
	
	// this 3 particle test will check proper wall force computation among all 3 axes
	shared_ptr<SystemDefinition> sysdef_3(new SystemDefinition(3, BoxDim(1000.0), 1, 0, exec_conf));
	shared_ptr<ParticleData> pdata_3 = sysdef_3->getParticleData();
	
	ParticleDataArrays arrays = pdata_3->acquireReadWrite();
	arrays.x[0] = 0.0; arrays.y[0] = Scalar(1.2); arrays.z[0] = 0.0;	// particle to test wall at pos 0,0,0
	arrays.x[1] = Scalar(12.2); arrays.y[1] = Scalar(-10.0); arrays.z[1] = 0.0;	// particle to test wall at pos 10,0,0
	arrays.x[2] = 0.0; arrays.y[2] = Scalar(10.0); arrays.z[2] = Scalar(-12.9);	// particle to test wall at pos 0,0,-10
	pdata_3->release();
	
	// create the wall force compute with a default cuttoff of 1.0 => all forces should be 0 for the first round
	shared_ptr<LJWallForceCompute> fc_3 = ljwall_creator(sysdef_3, Scalar(1.0));
	
	// pick some parameters
	Scalar epsilon = Scalar(1.15);
	Scalar sigma = Scalar(1.0);
	Scalar alpha = Scalar(1.0);
	Scalar lj1 = Scalar(4.0) * epsilon * pow(sigma,Scalar(12.0));
	Scalar lj2 = alpha * Scalar(4.0) * epsilon * pow(sigma,Scalar(6.0));
	fc_3->setParams(0,lj1,lj2);
	
	// compute the forces
	fc_3->compute(0);
	
	// there are no walls, so all forces should be zero
	ForceDataArrays force_arrays = fc_3->acquire();
	MY_BOOST_CHECK_SMALL(force_arrays.fx[0], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fy[0], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[0], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.pe[0], tol);

	MY_BOOST_CHECK_SMALL(force_arrays.fx[1], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fy[1], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[1], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.pe[1], tol);

	MY_BOOST_CHECK_SMALL(force_arrays.fx[2], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fy[2], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[2], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.pe[2], tol);
	
	// add the walls
	sysdef_3->getWallData()->addWall(Wall(0.0, 0.0, 0.0, 0.0, 1.0, 0.0));
	sysdef_3->getWallData()->addWall(Wall(10.0, 0.0, 0.0, 1.0, 0.0, 0.0));
	sysdef_3->getWallData()->addWall(Wall(0.0, 0.0, -10.0, 0.0, 0.0, 1.0));
	
	// compute the forces again
	fc_3->compute(1);
	
	// they should still be zero
	force_arrays = fc_3->acquire();
	MY_BOOST_CHECK_SMALL(force_arrays.fx[0], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fy[0], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[0], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.pe[0], tol);

	MY_BOOST_CHECK_SMALL(force_arrays.fx[1], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fy[1], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[1], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.pe[1], tol);	

	MY_BOOST_CHECK_SMALL(force_arrays.fx[2], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fy[2], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[2], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.pe[2], tol);

	// increase the cuttoff to check the actual force computation
	fc_3->setRCut(3.0);
	fc_3->compute(2);
	force_arrays = fc_3->acquire();
	MY_BOOST_CHECK_SMALL(force_arrays.fx[0], tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fy[0], -2.54344734, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[0], tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[0], -1.0246100807205, tol);

	MY_BOOST_CHECK_CLOSE(force_arrays.fx[1], -0.108697879, tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fy[1], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fz[1], tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[1], -0.04021378505, tol);

	MY_BOOST_CHECK_SMALL(force_arrays.fx[2], tol);
	MY_BOOST_CHECK_SMALL(force_arrays.fy[2], tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.fz[2], 0.0159463169, tol);
	MY_BOOST_CHECK_CLOSE(force_arrays.pe[2], -0.0077203876329103, tol);
	}

//! LJWallForceCompute creator for unit tests
shared_ptr<LJWallForceCompute> base_class_ljwall_creator(shared_ptr<SystemDefinition> sysdef, Scalar r_cut)
	{
	return shared_ptr<LJWallForceCompute>(new LJWallForceCompute(sysdef, r_cut));
	}

//! boost test case for particle test on CPU
BOOST_AUTO_TEST_CASE( LJWallForce_particle )
	{
	ljwallforce_creator ljwall_creator_base = bind(base_class_ljwall_creator, _1, _2);
	ljwall_force_particle_test(ljwall_creator_base, ExecutionConfiguration(ExecutionConfiguration::CPU, 0));
	}

#ifdef WIN32
#pragma warning( pop )
#endif
