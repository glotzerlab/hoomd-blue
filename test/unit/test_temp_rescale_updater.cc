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

#include <boost/shared_ptr.hpp>

#include "ComputeThermo.h"
#include "TempRescaleUpdater.h"

#ifdef ENABLE_CUDA
#include "ComputeThermoGPU.h"
#endif

#include <math.h>

using namespace std;
using namespace boost;

//! label the boost test module
#define BOOST_TEST_MODULE TempRescaleUpdaterTests
#include "boost_utf_configure.h"

/*! \file temp_rescale_updater_test.cc
    \brief Unit tests for the ComputeThermo and TempRescaleUpdater classes.
    \ingroup unit_tests
*/

//! boost test case to verify proper operation of ComputeThermo
BOOST_AUTO_TEST_CASE( ComputeThermo_basic )
    {
    // verify that we can constructe a TempCompute properly
    // create a simple particle data to test with
    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(2, BoxDim(1000.0), 4));
    shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    
    ParticleDataArrays arrays = pdata->acquireReadWrite();
    arrays.x[0] = arrays.y[0] = arrays.z[0] = 0.0;
    arrays.vx[0] = 1.0; arrays.vy[0] = 2.0; arrays.vz[0] = 3.0;
    arrays.x[1] = arrays.y[1] = arrays.z[1] = 1.0;
    arrays.vx[1] = 4.0; arrays.vy[1] = 5.0; arrays.vz[1] = 6.0;
    pdata->release();
    
    // construct a TempCompute and see that everything is set properly
    shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef, 0, pdata->getN()-1));
    shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_all));
    shared_ptr<ComputeThermo> tc(new ComputeThermo(sysdef, group_all));
    
    // check that we can actually compute temperature
    tc->setNDOF(3*pdata->getN());
    tc->compute(0);
    MY_BOOST_CHECK_CLOSE(tc->getTemperature(), 15.1666666666666666666667, tol);
    }

#ifdef ENABLE_CUDA
//! boost test case to verify proper operation of ComputeThermoGPU
BOOST_AUTO_TEST_CASE( ComputeThermoGPU_basic )
    {
    // verify that we can constructe a TempCompute properly
    // create a simple particle data to test with
    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(2, BoxDim(1000.0), 4));
    shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    
    ParticleDataArrays arrays = pdata->acquireReadWrite();
    arrays.x[0] = arrays.y[0] = arrays.z[0] = 0.0;
    arrays.vx[0] = 1.0; arrays.vy[0] = 2.0; arrays.vz[0] = 3.0;
    arrays.x[1] = arrays.y[1] = arrays.z[1] = 1.0;
    arrays.vx[1] = 4.0; arrays.vy[1] = 5.0; arrays.vz[1] = 6.0;
    pdata->release();
    
    // construct a TempCompute and see that everything is set properly
    shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef, 0, pdata->getN()-1));
    shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_all));
    shared_ptr<ComputeThermoGPU> tc(new ComputeThermoGPU(sysdef, group_all));
    
    // check that we can actually compute temperature
    tc->setNDOF(3*pdata->getN());
    tc->compute(0);
    MY_BOOST_CHECK_CLOSE(tc->getTemperature(), 15.1666666666666666666667, tol);
    }
#endif

//! boost test case to verify proper operation of TempRescaleUpdater
BOOST_AUTO_TEST_CASE( TempRescaleUpdater_basic )
    {
    // create a simple particle data to test with
    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(2, BoxDim(1000.0), 4));
    shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    
    ParticleDataArrays arrays = pdata->acquireReadWrite();
    arrays.x[0] = arrays.y[0] = arrays.z[0] = 0.0;
    arrays.vx[0] = 1.0; arrays.vy[0] = 2.0; arrays.vz[0] = 3.0;
    arrays.x[1] = arrays.y[1] = arrays.z[1] = 1.0;
    arrays.vx[1] = 4.0; arrays.vy[1] = 5.0; arrays.vz[1] = 6.0;
    pdata->release();
    
    // construct a Computethermo for the updater
    shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef, 0, pdata->getN()-1));
    shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_all));
    shared_ptr<ComputeThermo> tc(new ComputeThermo(sysdef, group_all));
    
    
    // variant T for the rescaler
    shared_ptr<VariantConst> T_variant(new VariantConst(1.2));
    
    // construct the updater and make sure everything is set properly
    shared_ptr<TempRescaleUpdater> rescaler(new TempRescaleUpdater(sysdef, tc, T_variant));
    
    // run the updater and check the new temperature
    rescaler->update(0);
    tc->compute(1);
    MY_BOOST_CHECK_CLOSE(tc->getTemperature(), 1.2, tol);
    
    // check that the setT method works
    shared_ptr<VariantConst> T_variant2(new VariantConst(2.0));
    rescaler->setT(T_variant2);
    rescaler->update(1);
    tc->compute(2);
    MY_BOOST_CHECK_CLOSE(tc->getTemperature(), 2.0, tol);
    }

#ifdef WIN32
#pragma warning( pop )
#endif

