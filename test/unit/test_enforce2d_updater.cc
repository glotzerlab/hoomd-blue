/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
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


#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <iostream>

#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include "Enforce2DUpdater.h"
#include "AllPairPotentials.h"
#include "HOOMDInitializer.h"
#include "NeighborListBinned.h"
#include "TwoStepNVT.h"
#include "ComputeThermo.h"

#ifdef ENABLE_CUDA
#include "Enforce2DUpdaterGPU.h"
#endif

#include "IntegratorTwoStep.h"

#include "HOOMDDumpWriter.h"
#include "saruprng.h"

#include <math.h>

using namespace std;
using namespace boost;

//! label the boost test module
#define BOOST_TEST_MODULE Enforce2DUpdaterTests
#include "boost_utf_configure.h"

/*! \file enforce2d_updater_test.cc
    \brief Unit tests for the Enforce2DUpdater class
    \ingroup unit_tests
*/

//! Typedef'd Enforce2DUpdater factory
typedef boost::function<shared_ptr<Enforce2DUpdater> (shared_ptr<SystemDefinition> sysdef)> enforce2d_creator;

//! boost test case to verify proper operation of Enforce2DUpdater
void enforce2d_basic_test(enforce2d_creator creator, boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    BoxDim box(20.0, 20.0, 1.0);
    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(100, box, 1, 0, 0, 0, 0, exec_conf));

    sysdef->setNDimensions(2);
    shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef, 0, pdata->getN()-1));
    shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_all));
        
    Saru saru(11, 21, 33);

    ParticleDataArrays arrays = pdata->acquireReadWrite();

    // setup a simple initial state
    Scalar tiny = 1e-3;
    for (unsigned int i=0; i<10; i++)
        for (unsigned int j=0; j<10; j++) 
            {
            unsigned int k = i*10 + j;
            arrays.x[k] = 2*i-10.0 + tiny;
            arrays.y[k] = 2*j-10.0 + tiny;
            arrays.z[k] = 0.0;
            arrays.vx[k] = saru.f(-1.0, 1.0);
            arrays.vy[k] = saru.f(-1.0, 1.0);
            arrays.vz[k] = 0.0;
            }
        
    pdata->release();
    
    boost::shared_ptr<Variant> T(new VariantConst(1.0));
    shared_ptr<ComputeThermo> thermo(new ComputeThermo(sysdef, group_all));
    thermo->setNDOF(2*group_all->getNumMembers()-2);
    shared_ptr<TwoStepNVT> two_step_nvt(new TwoStepNVT(sysdef, group_all, thermo, 0.5, T));
        
    Scalar deltaT = Scalar(0.005);
    shared_ptr<IntegratorTwoStep> nve_up(new IntegratorTwoStep(sysdef, deltaT));
    nve_up->addIntegrationMethod(two_step_nvt);
    
    shared_ptr<NeighborListBinned> nlist(new NeighborListBinned(sysdef, Scalar(2.5), Scalar(0.3)));
    nlist->setStorageMode(NeighborList::half);

    shared_ptr<PotentialPairLJ> fc(new PotentialPairLJ(sysdef, nlist));

    // setup some values for alpha and sigma
    Scalar epsilon = Scalar(1.0);
    Scalar sigma = Scalar(1.0);
    Scalar alpha = Scalar(1.0);
    Scalar lj1 = Scalar(4.0) * epsilon * pow(sigma,Scalar(12.0));
    Scalar lj2 = alpha * Scalar(4.0) * epsilon * pow(sigma,Scalar(6.0));
    
    // specify the force parameters
    fc->setParams(0,0,make_scalar2(lj1,lj2));
    fc->setRcut(0,0,Scalar(2.5));
    fc->setShiftMode(PotentialPairLJ::shift);

    nve_up->addForceCompute(fc);
    nve_up->prepRun(0);

    // verify that the atoms leave the xy plane if no contstraints are present
    // and random forces are added (due to roundoff error in a long simulation)s
    unsigned int np = pdata->getN();
    for (int t = 0; t < 1000; t++)
        {
        if (t%100 == 0) {
            arrays = pdata->acquireReadWrite();
            for (unsigned int i=0; i<np; i++)
                {
                arrays.az[i] += saru.f(-0.001, 0.002);
                arrays.vz[i] += saru.f(-0.002, 0.001);
                }
            pdata->release();        
        }
        nve_up->update(t);
        }

    Scalar total_deviation = Scalar(0.0);
    arrays = pdata->acquireReadWrite();
    for (unsigned int i=0; i<np; i++)
        total_deviation += fabs(arrays.z[i]);

    //make sure the deviation is large (should be >> tol)
    BOOST_CHECK(total_deviation > tol);
        
    // re-initialize the initial state
    for (unsigned int i=0; i<10; i++)
        for (unsigned int j=0; j<10; j++) 
            {
            unsigned int k = i*10 + j;
            arrays.x[k] = 2*i-10.0 + tiny;
            arrays.y[k] = 2*j-10.0 + tiny;
            arrays.z[k] = 0.0;
            arrays.vx[k] = saru.f(-1.0, 1.0);
            arrays.vy[k] = saru.f(-1.0, 1.0);
            arrays.vz[k] = 0.0;
            }
            
    pdata->release();
    pdata->notifyParticleSort();

    shared_ptr<Enforce2DUpdater> enforce2d = creator(sysdef);
     
    // verify that the atoms never leave the xy plane if contstraint is present:
    for (int t = 0; t < 1000; t++)
        {
        if (t%100 == 0) {
            arrays = pdata->acquireReadWrite();
            for (unsigned int i=0; i<np; i++)
                {
                arrays.az[i] += saru.f(-0.01, 0.02);
                arrays.vz[i] += saru.f(-0.1, 0.2);
                }
            pdata->release();        
        }
        enforce2d->update(t);
        nve_up->update(t);
        }

    total_deviation = Scalar(0.0);
    arrays = pdata->acquireReadWrite();
    for (unsigned int i=0; i<np; i++)
        {
        total_deviation += fabs(arrays.z[i]);
        }
    pdata->release();

    MY_BOOST_CHECK_CLOSE(total_deviation, 0.0, tol);

    }

//! Enforce2DUpdater creator for unit tests
shared_ptr<Enforce2DUpdater> base_class_enforce2d_creator(shared_ptr<SystemDefinition> sysdef)
    {
    return shared_ptr<Enforce2DUpdater>(new Enforce2DUpdater(sysdef));
    }

#ifdef ENABLE_CUDA
//! Enforce2DUpdaterGPU creator for unit tests
shared_ptr<Enforce2DUpdater> gpu_enforce2d_creator(shared_ptr<SystemDefinition> sysdef)
    {
    return shared_ptr<Enforce2DUpdater>(new Enforce2DUpdaterGPU(sysdef));
    }
#endif

//! boost test case for basic enforce2d tests
BOOST_AUTO_TEST_CASE( Enforce2DUpdater_basic )
    {
    enforce2d_creator creator = bind(base_class_enforce2d_creator, _1);
   enforce2d_basic_test(creator, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
    
#ifdef ENABLE_CUDA
//! boost test case for basic enforce2d tests
BOOST_AUTO_TEST_CASE( Enforce2DUpdaterGPU_basic )
    {
    enforce2d_creator creator = bind(gpu_enforce2d_creator, _1);
    enforce2d_basic_test(creator, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
#endif
    
#ifdef WIN32
#pragma warning( pop )
#endif

