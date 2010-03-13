/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
// Maintainer: joaander

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <iostream>

#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include "TwoStepNPT.h"
#ifdef ENABLE_CUDA
#include "TwoStepNPTGPU.h"
#endif
#include "IntegratorTwoStep.h"

#include "BinnedNeighborList.h"
#include "Initializers.h"
#include "AllPairPotentials.h"

#include <math.h>

using namespace std;
using namespace boost;

/*! \file npt_updater_test.cc
    \brief Implements unit tests for NPTpdater and descendants
    \ingroup unit_tests
*/

//! name the boost unit test module
#define BOOST_TEST_MODULE TwoStepNPTTests
#include "boost_utf_configure.h"

//! Typedef'd NPTUpdator class factory
typedef boost::function<shared_ptr<TwoStepNPT> (shared_ptr<SystemDefinition> sysdef,
                                                shared_ptr<ParticleGroup> group,
                                                Scalar tau,
                                                Scalar tauP,
                                                Scalar T,
                                                Scalar P) > twostepnpt_creator;


//! Basic functionality test of a generic TwoStepNPT
void npt_updater_test(twostepnpt_creator npt_creator, ExecutionConfiguration exec_conf)
    {
#ifdef ENABLE_CUDA
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
    shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef, 0, pdata->getN()-1));
    shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_all));
    
    shared_ptr<BinnedNeighborList> nlist(new BinnedNeighborList(sysdef, Scalar(2.5), Scalar(0.8)));
    
    shared_ptr<PotentialPairLJ> fc(new PotentialPairLJ(sysdef, nlist, ""));
    fc->setRcut(0, 0, Scalar(2.5));
    
    // setup some values for alpha and sigma
    Scalar epsilon = Scalar(1.0);
    Scalar sigma = Scalar(1.0);
    Scalar alpha = Scalar(1.0);
    Scalar lj1 = Scalar(4.0) * epsilon * pow(sigma,Scalar(12.0));
    Scalar lj2 = alpha * Scalar(4.0) * epsilon * pow(sigma,Scalar(6.0));
    
    // specify the force parameters
    fc->setParams(0,0,make_scalar2(lj1,lj2));
    
    
    shared_ptr<TwoStepNPT> two_step_npt = npt_creator(sysdef, group_all, Scalar(1.0),Scalar(1.0),T,P);
    shared_ptr<IntegratorTwoStep> npt(new IntegratorTwoStep(sysdef, Scalar(0.001)));
    npt->addIntegrationMethod(two_step_npt);
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

//! NPTUpdater factory for the unit tests
shared_ptr<TwoStepNPT> base_class_npt_creator(shared_ptr<SystemDefinition> sysdef,
                                              shared_ptr<ParticleGroup> group,
                                              Scalar tau,
                                              Scalar tauP,
                                              Scalar T,
                                              Scalar P)
    {
    boost::shared_ptr<Variant> T_variant(new VariantConst(T));
    boost::shared_ptr<Variant> P_variant(new VariantConst(P));
    return shared_ptr<TwoStepNPT>(new TwoStepNPT(sysdef, group,tau,tauP,T_variant,P_variant));
    }

#ifdef ENABLE_CUDA
//! NPTUpdaterGPU factory for the unit tests
shared_ptr<TwoStepNPT> gpu_npt_creator(shared_ptr<SystemDefinition> sysdef,
                                       shared_ptr<ParticleGroup> group,
                                       Scalar tau,
                                       Scalar tauP,
                                       Scalar T,
                                       Scalar P)
    {
    boost::shared_ptr<Variant> T_variant(new VariantConst(T));
    boost::shared_ptr<Variant> P_variant(new VariantConst(P));
    return shared_ptr<TwoStepNPT>(new TwoStepNPTGPU(sysdef, group, tau, tauP, T_variant, P_variant));
    }
#endif


//! boost test case for base class integration tests
BOOST_AUTO_TEST_CASE( TwoStepNPT_tests )
    {
    twostepnpt_creator npt_creator = bind(base_class_npt_creator, _1, _2,_3,_4,_5,_6);
    npt_updater_test(npt_creator, ExecutionConfiguration(ExecutionConfiguration::CPU));
    }


#ifdef ENABLE_CUDA

//! boost test case for base class integration tests
BOOST_AUTO_TEST_CASE( TwoStepNPTGPU_tests )
    {
    twostepnpt_creator npt_creator = bind(gpu_npt_creator, _1, _2,_3,_4,_5,_6);
    npt_updater_test(npt_creator, ExecutionConfiguration(ExecutionConfiguration::GPU));
    }

#endif

#ifdef WIN32
#pragma warning( pop )
#endif

