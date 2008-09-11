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
#include "NPTUpdater.h"
#ifdef USE_CUDA
#include "NPTUpdaterGPU.h"
#endif

#include "BinnedNeighborList.h"
#include "Initializers.h"
#include "LJForceCompute.h"

#include <math.h>

using namespace std;
using namespace boost;

/*! \file npt_updater_test.cc
	\brief Implements unit tests for NPTpdater and descendants
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

//! Typedef'd NPTUpdator class factory
typedef boost::function<shared_ptr<NPTUpdater> (shared_ptr<ParticleData> pdata, Scalar deltaT, Scalar tau, Scalar tauP, Scalar T, Scalar P) > nptup_creator;
	

//! Compares the output from NPTUpdater, averages pressure and temeprature and comares them with 
//! given pressures and temperatures.  
void npt_updater_test(nptup_creator npt_creator)
	{
	const unsigned int N = 108;
	Scalar T = 1.0;
	Scalar P = 1.0;

	// create two identical random particle systems to simulate
	RandomInitializer rand_init(N, Scalar(0.2), Scalar(0.9), "A");
	rand_init.setSeed(12345);
	shared_ptr<ParticleData> pdata(new ParticleData(rand_init));

	shared_ptr<BinnedNeighborList> nlist(new BinnedNeighborList(pdata, Scalar(2.5), Scalar(0.8)));

	shared_ptr<LJForceCompute> fc(new LJForceCompute(pdata, nlist, Scalar(2.5)));

		
	// setup some values for alpha and sigma
	Scalar epsilon = Scalar(1.0);
	Scalar sigma = Scalar(1.0);
	Scalar alpha = Scalar(1.0);
	Scalar lj1 = Scalar(4.0) * epsilon * pow(sigma,Scalar(12.0));
	Scalar lj2 = alpha * Scalar(4.0) * epsilon * pow(sigma,Scalar(6.0));
	
	// specify the force parameters
	fc->setParams(0,0,lj1,lj2);


	shared_ptr<NPTUpdater> npt = npt_creator(pdata, Scalar(0.001),Scalar(1.0),Scalar(1.0),T,P);


	npt->addForceCompute(fc);


	// step for a 10,000 timesteps to relax pessure and tempreratue
	// before computing averages
	for (int i = 0; i < 10000; i++)
		npt->update(i);

	// now do the averaging for next 100k steps
	Scalar avrT = 0.0;
	Scalar avrP = 0.0;
	for (int i = 10001; i < 110000; i++)
	       {
                 avrT += npt->computeTemperature();
		 avrP += npt->computePressure();
		 npt->update(i);
	       }

	avrT /= 100000.0;
	avrP /= 100000.0;
	Scalar rough_tol = 5.0;
	MY_BOOST_CHECK_CLOSE(T, avrT, rough_tol);
	MY_BOOST_CHECK_CLOSE(P, avrP, rough_tol);
	
	}
	
//! NPTUpdater factory for the unit tests
shared_ptr<NPTUpdater> base_class_npt_creator(shared_ptr<ParticleData> pdata, Scalar deltaT, Scalar tau, Scalar tauP, Scalar T, Scalar P)
	{
	  return shared_ptr<NPTUpdater>(new NPTUpdater(pdata, deltaT,tau,tauP,T,P));
	}
	
#ifdef USE_CUDA
//! NPTUpdaterGPU factory for the unit tests
shared_ptr<NPTUpdater> gpu_nve_creator(shared_ptr<ParticleData> pdata, Scalar deltaT, Scalar tau, Scalar tauP, Scalar T, Scalar P)
	{
	  return shared_ptr<NVEUpdater>(new NVEUpdaterGPU(pdata, deltaT, tau, tauP, T, P));
	}
#endif
	
	
//! boost test case for base class integration tests
BOOST_AUTO_TEST_CASE( NPTUpdater_tests )
	{
	  nptup_creator npt_creator = bind(base_class_npt_creator, _1, _2,_3,_4,_5,_6);
	npt_updater_test(npt_creator);
	}
	
	
#ifdef USE_CUDA

// Need to add some stuff in here when NPT finaly gets implemented on GPU

#endif

#ifdef WIN32
#pragma warning( pop )
#endif
