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

//! name the boost unit test module
#define BOOST_TEST_MODULE NVEUpdaterTests
#include "boost_utf_configure.h"

#include <boost/test/floating_point_comparison.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include "ConstForceCompute.h"
#include "NPTUpdater.h"
#ifdef ENABLE_CUDA
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
typedef boost::function<shared_ptr<NPTUpdater> (shared_ptr<SystemDefinition> sysdef,
												Scalar deltaT,
												Scalar tau,
												Scalar tauP,
												Scalar T,
												Scalar P) > nptup_creator;


//! Basic functionality test of a generic NPTUpdater
void npt_updater_test(nptup_creator npt_creator, ExecutionConfiguration exec_conf)
    {
#ifdef CUDA
    g_gpu_error_checking = true;
#endif
    
    const unsigned int N = 1000;
    Scalar T = 2.0;
    Scalar P = 1.0;
    
    // create two identical random particle systems to simulate
    RandomInitializer rand_init(N, Scalar(0.2), Scalar(0.9), "A");
    rand_init.setSeed(12345);
    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(rand_init, exec_conf));
    shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    
    shared_ptr<BinnedNeighborList> nlist(new BinnedNeighborList(sysdef, Scalar(2.5), Scalar(0.8)));
    
    shared_ptr<LJForceCompute> fc(new LJForceCompute(sysdef, nlist, Scalar(2.5)));
    
    
    // setup some values for alpha and sigma
    Scalar epsilon = Scalar(1.0);
    Scalar sigma = Scalar(1.0);
    Scalar alpha = Scalar(1.0);
    Scalar lj1 = Scalar(4.0) * epsilon * pow(sigma,Scalar(12.0));
    Scalar lj2 = alpha * Scalar(4.0) * epsilon * pow(sigma,Scalar(6.0));
    
    // specify the force parameters
    fc->setParams(0,0,lj1,lj2);
    
    
    shared_ptr<NPTUpdater> npt = npt_creator(sysdef, Scalar(0.001),Scalar(1.0),Scalar(1.0),T,P);
    
    
    npt->addForceCompute(fc);
    
    
    // step for a 10,000 timesteps to relax pessure and tempreratue
    // before computing averages
    for (int i = 0; i < 10000; i++)
        {
        npt->update(i);
        }
        
    // now do the averaging for next 100k steps
    Scalar avrT = 0.0;
    Scalar avrP = 0.0;
    for (int i = 10001; i < 50000; i++)
        {
        avrT += npt->computeTemperature(i);
        avrP += npt->computePressure(i);
        npt->update(i);
        }
        
    avrT /= 40000.0;
    avrP /= 40000.0;
    Scalar rough_tol = 2.0;
    MY_BOOST_CHECK_CLOSE(T, avrT, rough_tol);
    MY_BOOST_CHECK_CLOSE(P, avrP, rough_tol);
    
    }

//! Compares the output from one NVEUpdater to another
void npt_updater_compare_test(nptup_creator npt_creator1, nptup_creator npt_creator2, ExecutionConfiguration exec_conf)
    {
#ifdef CUDA
    g_gpu_error_checking = true;
#endif
    
    const unsigned int N = 1000;
    Scalar T = 2.0;
    Scalar P = 1.0;
    
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
    
    shared_ptr<NPTUpdater> npt1 = npt_creator1(sysdef1, Scalar(0.005),Scalar(1.0),Scalar(1.0),T,P);
    shared_ptr<NPTUpdater> npt2 = npt_creator2(sysdef2, Scalar(0.005),Scalar(1.0),Scalar(1.0),T,P);
    
    npt1->addForceCompute(fc1);
    npt2->addForceCompute(fc2);
    
    for (int i = 0; i < 10000; i++)
        {
        npt1->update(i);
        npt2->update(i);
        }
        
    // now do the averaging for next 100k steps
    Scalar avrT1 = 0.0;
    Scalar avrT2 = 0.0;
    Scalar avrP1 = 0.0;
    Scalar avrP2 = 0.0;
    for (int i = 10001; i < 50000; i++)
        {
        avrT1 += npt1->computeTemperature(i);
        avrT2 += npt2->computeTemperature(i);
        avrP1 += npt1->computePressure(i);
        avrP2 += npt2->computePressure(i);
        npt1->update(i);
        npt2->update(i);
        }
        
    avrT1 /= 40000.0;
    avrT2 /= 40000.0;
    avrP1 /= 40000.0;
    avrP2 /= 40000.0;
    Scalar rough_tol = 1.0;
    MY_BOOST_CHECK_CLOSE(avrT1, avrT2, rough_tol);
    MY_BOOST_CHECK_CLOSE(avrP1, avrP2, rough_tol);
    
    
    }

//! NPTUpdater factory for the unit tests
shared_ptr<NPTUpdater> base_class_npt_creator(shared_ptr<SystemDefinition> sysdef,
											  Scalar deltaT,
											  Scalar tau,
											  Scalar tauP,
											  Scalar T,
											  Scalar P)
    {
    boost::shared_ptr<Variant> T_variant(new VariantConst(T));
    boost::shared_ptr<Variant> P_variant(new VariantConst(P));
    return shared_ptr<NPTUpdater>(new NPTUpdater(sysdef, deltaT,tau,tauP,T_variant,P_variant));
    }

#ifdef ENABLE_CUDA
//! NPTUpdaterGPU factory for the unit tests
shared_ptr<NPTUpdater> gpu_npt_creator(shared_ptr<SystemDefinition> sysdef,
									   Scalar deltaT,
									   Scalar tau,
									   Scalar tauP,
									   Scalar T,
									   Scalar P)
    {
    boost::shared_ptr<Variant> T_variant(new VariantConst(T));
    boost::shared_ptr<Variant> P_variant(new VariantConst(P));
    return shared_ptr<NPTUpdater>(new NPTUpdaterGPU(sysdef, deltaT, tau, tauP, T_variant, P_variant));
    }
#endif


//! boost test case for base class integration tests
BOOST_AUTO_TEST_CASE( NPTUpdater_tests )
    {
    nptup_creator npt_creator = bind(base_class_npt_creator, _1, _2,_3,_4,_5,_6);
    npt_updater_test(npt_creator, ExecutionConfiguration(ExecutionConfiguration::CPU));
    }


#ifdef ENABLE_CUDA

//! boost test case for base class integration tests
BOOST_AUTO_TEST_CASE( NPTUpdaterGPU_tests )
    {
    nptup_creator npt_creator = bind(gpu_npt_creator, _1, _2,_3,_4,_5,_6);
    npt_updater_test(npt_creator, ExecutionConfiguration(ExecutionConfiguration::GPU));
    }

//! boost test case for comparing the GPU integrator to the CPU one
BOOST_AUTO_TEST_CASE( NPTUpdaterGPU_comparison_tests)
    {
    nptup_creator npt_creator_gpu = bind(gpu_npt_creator, _1, _2, _3,_4,_5,_6);
    nptup_creator npt_creator = bind(base_class_npt_creator, _1, _2, _3,_4,_5,_6);
    npt_updater_compare_test(npt_creator, npt_creator_gpu, ExecutionConfiguration(ExecutionConfiguration::GPU));
    }
#endif

#ifdef WIN32
#pragma warning( pop )
#endif
