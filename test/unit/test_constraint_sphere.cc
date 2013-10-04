/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

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

    {
    ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);

    // set the particle initial positions
    h_pos.data[0].x = P.x - r;
    h_pos.data[1].x = P.x + r;
    h_pos.data[2].y = P.y - r;
    h_pos.data[3].y = P.y + r;
    h_pos.data[4].z = P.z - r;
    h_pos.data[5].z = P.z + r;
    }

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


        for (unsigned int j = 0; j < 6; j++)
            {
            Scalar3 V;
            Scalar3 pos = pdata->getPosition(j);
            V.x = pos.x - P.x;
            V.y = pos.y - P.y;
            V.z = pos.z - P.z;

            Scalar current_r = sqrt(V.x*V.x + V.y*V.y + V.z*V.z);
            MY_BOOST_CHECK_CLOSE(current_r, r, loose_tol);
            }

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
