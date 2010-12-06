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
// Maintainer: phillicl

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <iostream>

#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include "IntegratorTwoStep.h"
#include "TwoStepBDNVT.h"
#include "ConstraintSphere.h"
#ifdef ENABLE_CUDA
#include "ConstraintSphereGPU.h"
#endif

#include <math.h>

using namespace std;
using namespace boost;

/*! \file constraint_sphere_test.cc
    \brief Implements unit tests for ConstraintSphere and descendants
    \ingroup unit_tests
*/

//! name the boost unit test module
#define BOOST_TEST_MODULE ConstraintSphereTests
#include "boost_utf_configure.h"

//! Typedef'd class factory
typedef boost::function<shared_ptr<ConstraintSphere> (shared_ptr<SystemDefinition> sysdef,
                                                      shared_ptr<ParticleGroup> group,
                                                      Scalar3 P,
                                                      Scalar r)> cs_creator_t;

//! Run a BD simulation on 6 particles and validate the constraints
void constraint_sphere_tests(cs_creator_t cs_creator, boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    Scalar3 P = make_scalar3(1.0f, 2.0f, 3.0f);
    Scalar r = 10.0f;
    
    // Build a 6 particle system with all particles starting at the 6 "corners" of a sphere centered
    // at P with radius r. Use a huge box so boundary conditions don't come into play
    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(6, BoxDim(1000000.0), 1, 0, 0, 0, 0, exec_conf));
    shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef, 0, pdata->getN()-1));
    shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_all));
    
    ParticleDataArrays arrays = pdata->acquireReadWrite();
    
    // set the particle initial positions
    arrays.x[0] = P.x - r;
    arrays.x[1] = P.x + r;
    arrays.y[2] = P.y - r;
    arrays.y[3] = P.y + r;
    arrays.z[4] = P.z - r;
    arrays.z[5] = P.z + r;

    pdata->release();
    
    Scalar deltaT = Scalar(0.01);
    Scalar Temp = Scalar(2.0);
    
    // run the particles in a BD simulation with a constraint force applied and verify that the constraint is always
    // satisfied
    shared_ptr<VariantConst> T_variant(new VariantConst(Temp));
    shared_ptr<TwoStepBDNVT> two_step_bdnvt(new TwoStepBDNVT(sysdef, group_all, T_variant, 123, 0));
    shared_ptr<IntegratorTwoStep> bdnvt_up(new IntegratorTwoStep(sysdef, deltaT));
    bdnvt_up->addIntegrationMethod(two_step_bdnvt);
    
    boost::shared_ptr<ConstraintSphere> cs = cs_creator(sysdef, group_all, P, r);
    bdnvt_up->addForceConstraint(cs);
    bdnvt_up->prepRun(0);
    
    for (int i = 0; i < 1000; i++)
        {
        bdnvt_up->update(i);
        
        arrays = pdata->acquireReadWrite();
        
        for (unsigned int j = 0; j < 6; j++)
            {
            Scalar3 V;
            V.x = arrays.x[j] - P.x;
            V.y = arrays.y[j] - P.y;
            V.z = arrays.z[j] - P.z;
            
            Scalar current_r = sqrt(V.x*V.x + V.y*V.y + V.z*V.z);
            MY_BOOST_CHECK_CLOSE(current_r, r, loose_tol);
            }
        
        pdata->release();
        }
    }

//! ConstraintSphere factory for the unit tests
shared_ptr<ConstraintSphere> base_class_cs_creator(shared_ptr<SystemDefinition> sysdef,
                                                   shared_ptr<ParticleGroup> group,
                                                   Scalar3 P,
                                                   Scalar r)
    {
    return shared_ptr<ConstraintSphere>(new ConstraintSphere(sysdef, group, P, r));
    }

#ifdef ENABLE_CUDA
//! ConstraintSphereGPU factory for the unit tests
shared_ptr<ConstraintSphere> gpu_cs_creator(shared_ptr<SystemDefinition> sysdef,
                                                   shared_ptr<ParticleGroup> group,
                                                   Scalar3 P,
                                                   Scalar r)
    {
    return shared_ptr<ConstraintSphere>(new ConstraintSphereGPU(sysdef, group, P, r));
    }
#endif

//! Basic test for the base class
BOOST_AUTO_TEST_CASE( BDUpdater_tests )
    {
    cs_creator_t cs_creator = bind(base_class_cs_creator, _1, _2, _3, _4);
    constraint_sphere_tests(cs_creator, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

#ifdef ENABLE_CUDA
//! Basic test for the GPU class
BOOST_AUTO_TEST_CASE( BDUpdaterGPU_tests )
    {
    cs_creator_t cs_creator = bind(gpu_cs_creator, _1, _2, _3, _4);
    constraint_sphere_tests(cs_creator, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
#endif


#ifdef WIN32
#pragma warning( pop )
#endif

