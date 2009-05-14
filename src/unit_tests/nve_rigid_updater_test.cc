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

// $Id: nve_updater_test.cc 1622 2009-01-28 22:51:01Z joaander $
// $URL: http://svn2.assembla.com/svn/hoomd/trunk/src/unit_tests/nve_updater_test.cc $

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <iostream>

//! name the boost unit test module
#define BOOST_TEST_MODULE NVERigidUpdaterTests
#include "boost_utf_configure.h"

#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

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

/*! \file nve_rigid_updater_test.cc
	\brief Implements unit tests for NVERigidUpdater 
	\ingroup unit_tests
*/


//! Tolerance for floating point comparisons
#ifdef SINGLE_PRECISION
const Scalar tol = Scalar(1e-2);
#else
const Scalar tol = 1e-3;
#endif

//! Typedef'd NVEUpdator class factory
typedef boost::function<shared_ptr<NVEUpdater> (shared_ptr<SystemDefinition> sysdef, Scalar deltaT)> nveup_creator;

void nve_updater_integrate_tests(nveup_creator nve_creator, ExecutionConfiguration exec_conf)
{
	#ifdef ENABLE_CUDA
	g_gpu_error_checking = true;
	#endif
	
	// check that the nve updater can actually integrate particle positions and velocities correctly
	// start with a 2 particle system to keep things simple: also put everything in a huge box so boundary conditions
	// don't come into play
	shared_ptr<SystemDefinition> sysdef(new SystemDefinition(10, BoxDim(1000.0), 1, 0, exec_conf));
	shared_ptr<ParticleData> pdata = sysdef->getParticleData();
	
	ParticleDataArrays arrays = pdata->acquireReadWrite();
	
	// setup a simple initial state
	arrays.x[0] = Scalar(-1.0); arrays.y[0] = 0.0; arrays.z[0] = 0.0;
	arrays.vx[0] = Scalar(-0.5); arrays.body[0] = 0;
	arrays.x[1] =  Scalar(-1.0); arrays.y[1] = 1.0; arrays.z[1] = 0.0;
	arrays.vx[1] = Scalar(0.2); arrays.body[1] = 0;
	arrays.x[2] = Scalar(-1.0); arrays.y[2] = 2.0; arrays.z[2] = 0.0;
	arrays.vy[2] = Scalar(-0.1); arrays.body[2] = 0;
	arrays.x[3] = Scalar(-1.0); arrays.y[3] = 3.0; arrays.z[3] = 0.0;
	arrays.vy[3] = Scalar(0.3);  arrays.body[3] = 0;
	arrays.x[4] = Scalar(-1.0); arrays.y[4] = 4.0; arrays.z[4] = 0.0;
	arrays.vz[4] = Scalar(-0.2); arrays.body[4] = 0;
	
	arrays.x[5] = 0.0; arrays.y[5] = Scalar(0.0); arrays.z[5] = 0.0;
	arrays.vx[5] = Scalar(0.2); arrays.body[5] = 1;
	arrays.x[6] = 0.0; arrays.y[6] = Scalar(1.0); arrays.z[6] = 0.0;
	arrays.vy[6] = Scalar(0.8); arrays.body[6] = 1;
	arrays.x[7] = 0.0; arrays.y[7] = Scalar(2.0); arrays.z[7] = 0.0;
	arrays.vy[7] = Scalar(-0.6); arrays.body[7] = 1;
	arrays.x[8] = 0.0; arrays.y[8] = Scalar(3.0); arrays.z[8] = 0.0;
	arrays.vz[8] = Scalar(0.7); arrays.body[8] = 1;
	arrays.x[9] = 0.0; arrays.y[9] = Scalar(4.0); arrays.z[9] = 0.0;
	arrays.vy[9] = Scalar(-0.5); arrays.body[9] = 1;
	
	pdata->release();
	
	Scalar deltaT = Scalar(0.0001);
	shared_ptr<NVEUpdater> nve_up = nve_creator(sysdef, deltaT);
	shared_ptr<NeighborList> nlist(new NeighborList(sysdef, Scalar(3.0), Scalar(0.8)));
	shared_ptr<LJForceCompute> fc(new LJForceCompute(sysdef, nlist, Scalar(3.0)));
	
	// setup some values for alpha and sigma
	Scalar epsilon = Scalar(1.0);
	Scalar sigma = Scalar(1.0);
	Scalar alpha = Scalar(1.0);
	Scalar lj1 = Scalar(4.0) * epsilon * pow(sigma,Scalar(12.0));
	Scalar lj2 = alpha * Scalar(4.0) * epsilon * pow(sigma,Scalar(6.0));
	
	// specify the force parameters
	fc->setParams(0,0,lj1,lj2);

	
	nve_up->addForceCompute(fc);
	
	
	sysdef->init();
	for (int i = 0; i < 500; i++)
		{
		if (i%100 == 0) cout << "step " << i << "\n";
		nve_up->update(i);
		}

	shared_ptr<RigidData> rdata = sysdef->getRigidData();
	ArrayHandle<Scalar4> com_handle(rdata->getCOM(), access_location::host, access_mode::read);
	unsigned int n_bodies = rdata->getNumBodies();
	for (unsigned int i = 0; i < n_bodies; i++) 
		cout << com_handle.data[i].x << "\t" << com_handle.data[i].y << "\t" << com_handle.data[i].z << "\n";

	
/*	arrays = pdata->acquireReadWrite();
	for (unsigned int i = 0; i < arrays.nparticles; i++) 
		cout << arrays.ax[i] << "\t" << arrays.ay[i] << "\t" << arrays.az[i] << "\n";
	pdata->release();
*/
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


#ifdef ENABLE_CUDA
//! boost test case for base class integration tests
BOOST_AUTO_TEST_CASE( NVEUpdaterGPU_integrate_tests )
{
	nveup_creator nve_creator_gpu = bind(gpu_nve_creator, _1, _2);
	nve_updater_integrate_tests(nve_creator_gpu, ExecutionConfiguration(ExecutionConfiguration::GPU, ExecutionConfiguration::getDefaultGPU()));
}

#endif

#ifdef WIN32
#pragma warning( pop )
#endif
