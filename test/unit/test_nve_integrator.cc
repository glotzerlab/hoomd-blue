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

#include "ConstForceCompute.h"
#include "TwoStepNVE.h"
#ifdef ENABLE_CUDA
#include "TwoStepNVEGPU.h"
#endif

#include "IntegratorTwoStep.h"

#include "AllPairPotentials.h"
#include "NeighborList.h"
#include "Initializers.h"

#include <math.h>

using namespace std;
using namespace boost;

/*! \file nve_updater_test.cc
    \brief Implements unit tests for TwoStepNVE and descendants
    \ingroup unit_tests
*/

//! name the boost unit test module
#define BOOST_TEST_MODULE NVEUpdaterTests
#include "boost_utf_configure.h"

//! Typedef'd NVEUpdator class factory
typedef boost::function<shared_ptr<TwoStepNVE> (shared_ptr<SystemDefinition> sysdef,
                                                shared_ptr<ParticleGroup> group)> twostepnve_creator;

//! Integrate 1 particle through time and compare to an analytical solution
void nve_updater_integrate_tests(twostepnve_creator nve_creator, boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // check that the nve updater can actually integrate particle positions and velocities correctly
    // start with a 2 particle system to keep things simple: also put everything in a huge box so boundary conditions
    // don't come into play
    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(2, BoxDim(1000.0), 4, 0, 0, 0, 0, exec_conf));
    shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef, 0, pdata->getN()-1));
    shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_all));
    
    {
    ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(pdata->getVelocities(), access_location::host, access_mode::readwrite);
    // setup a simple initial state
    h_pos.data[0].x = 0.0;
    h_pos.data[0].y = 1.0;
    h_pos.data[0].z = 2.0;
    h_vel.data[0].x = 3.0;
    h_vel.data[0].y = 2.0;
    h_vel.data[0].z = 1.0;

    h_pos.data[1].x = 10.0;
    h_pos.data[1].y = 11.0;
    h_pos.data[1].z = 12.0;
    h_vel.data[1].x = 13.0;
    h_vel.data[1].y = 12.0;
    h_vel.data[1].z = 11.0;
    }
    
    Scalar deltaT = Scalar(0.0001);
    shared_ptr<TwoStepNVE> two_step_nve = nve_creator(sysdef, group_all);
    shared_ptr<IntegratorTwoStep> nve_up(new IntegratorTwoStep(sysdef, deltaT));
    nve_up->addIntegrationMethod(two_step_nve);
    
    // also test the ability of the updater to add two force computes together properly
    shared_ptr<ConstForceCompute> fc1(new ConstForceCompute(sysdef, 1.5, 0.0, 0.0));
    nve_up->addForceCompute(fc1);
    shared_ptr<ConstForceCompute> fc2(new ConstForceCompute(sysdef, 0.0, 2.5, 0.0));
    nve_up->addForceCompute(fc2);
    
    nve_up->prepRun(0);
    
    // verify proper integration compared to x = x0 + v0 t + 1/2 a t^2, v = v0 + a t
    // roundoff errors prevent this from keeping within 0.1% error for long
    for (int i = 0; i < 500; i++)
        {
        {
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(), access_location::host, access_mode::read);
        
        Scalar t = Scalar(i) * deltaT;
        MY_BOOST_CHECK_CLOSE(h_pos.data[0].x, 0.0 + 3.0 * t + 1.0/2.0 * 1.5 * t*t, loose_tol);
        MY_BOOST_CHECK_CLOSE(h_vel.data[0].x, 3.0 + 1.5 * t, loose_tol);
        
        MY_BOOST_CHECK_CLOSE(h_pos.data[0].y, 1.0 + 2.0 * t + 1.0/2.0 * 2.5 * t*t, loose_tol);
        MY_BOOST_CHECK_CLOSE(h_vel.data[0].y, 2.0 + 2.5 * t, loose_tol);
        
        MY_BOOST_CHECK_CLOSE(h_pos.data[0].z, 2.0 + 1.0 * t + 1.0/2.0 * 0 * t*t, loose_tol);
        MY_BOOST_CHECK_CLOSE(h_vel.data[0].z, 1.0 + 0 * t, loose_tol);
        
        MY_BOOST_CHECK_CLOSE(h_pos.data[1].x, 10.0 + 13.0 * t + 1.0/2.0 * 1.5 * t*t, loose_tol);
        MY_BOOST_CHECK_CLOSE(h_vel.data[1].x, 13.0 + 1.5 * t, loose_tol);
        
        MY_BOOST_CHECK_CLOSE(h_pos.data[1].y, 11.0 + 12.0 * t + 1.0/2.0 * 2.5 * t*t, loose_tol);
        MY_BOOST_CHECK_CLOSE(h_vel.data[1].y, 12.0 + 2.5 * t, loose_tol);
        
        MY_BOOST_CHECK_CLOSE(h_pos.data[1].z, 12.0 + 11.0 * t + 1.0/2.0 * 0 * t*t, loose_tol);
        MY_BOOST_CHECK_CLOSE(h_vel.data[1].z, 11.0 + 0 * t, loose_tol);
        }

        nve_up->update(i);
        }
    }

//! Check that the particle movement limit works
void nve_updater_limit_tests(twostepnve_creator nve_creator, boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // create a simple 1 particle system
    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(1, BoxDim(1000.0), 1, 0, 0, 0, 0, exec_conf));
    shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef, 0, pdata->getN()-1));
    shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_all));
    
    {
    ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(pdata->getVelocities(), access_location::host, access_mode::readwrite);
    
    // setup a simple initial state
    h_pos.data[0].x = 0.0;
    h_pos.data[0].y = 1.0;
    h_pos.data[0].z = 2.0;
    h_vel.data[0].x = 0.0;
    h_vel.data[0].y = 0.0;
    h_vel.data[0].z = 0.0;
    }
    
    Scalar deltaT = Scalar(0.0001);
    shared_ptr<TwoStepNVE> two_step_nve = nve_creator(sysdef, group_all);
    shared_ptr<IntegratorTwoStep> nve_up(new IntegratorTwoStep(sysdef, deltaT));
    nve_up->addIntegrationMethod(two_step_nve);
    
    // set the limit
    Scalar limit = Scalar(0.1);
    two_step_nve->setLimit(limit);
    
    // create an insanely large force to test the limiting method
    shared_ptr<ConstForceCompute> fc1(new ConstForceCompute(sysdef, 1e9, 2e9, 3e9));
    nve_up->addForceCompute(fc1);
    
    // expected movement vectors
    Scalar dx = limit / sqrt(14.0);
    Scalar dy = 2.0 * limit / sqrt(14.0);
    Scalar dz = 3.0 * limit / sqrt(14.0);
    
    Scalar vx = limit / sqrt(14.0) / deltaT;
    Scalar vy = 2.0 * limit / sqrt(14.0) / deltaT;
    Scalar vz = 3.0 * limit / sqrt(14.0) / deltaT;
    
    nve_up->prepRun(0);
    
    // verify proper integration compared to x = x0 + dx * i
    nve_up->update(0);
    for (int i = 1; i < 500; i++)
        {
        {
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(), access_location::host, access_mode::read);
        
        MY_BOOST_CHECK_CLOSE(h_pos.data[0].x, 0.0 + dx * Scalar(i), tol);
        MY_BOOST_CHECK_CLOSE(h_vel.data[0].x, vx, tol);
        
        MY_BOOST_CHECK_CLOSE(h_pos.data[0].y, 1.0 + dy * Scalar(i), tol);
        MY_BOOST_CHECK_CLOSE(h_vel.data[0].y, vy, tol);
        
        MY_BOOST_CHECK_CLOSE(h_pos.data[0].z, 2.0 + dz * Scalar(i), tol);
        MY_BOOST_CHECK_CLOSE(h_vel.data[0].z, vz, tol);
        }
        
        nve_up->update(i);
        }
    }


//! Make a few particles jump across the boundary and verify that the updater works
void nve_updater_boundary_tests(twostepnve_creator nve_creator, boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    ////////////////////////////////////////////////////////////////////
    // now, lets do a more thorough test and include boundary conditions
    // there are way too many permutations to test here, so I will simply
    // test +x, -x, +y, -y, +z, and -z independantly
    // build a 6 particle system with particles set to move across each boundary
    shared_ptr<SystemDefinition> sysdef_6(new SystemDefinition(6, BoxDim(20.0, 40.0, 60.0), 1, 0, 0, 0, 0, exec_conf));
    shared_ptr<ParticleData> pdata_6 = sysdef_6->getParticleData();
    shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef_6, 0, pdata_6->getN()-1));
    shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef_6, selector_all));
    
    {
    ArrayHandle<Scalar4> h_pos(pdata_6->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(pdata_6->getVelocities(), access_location::host, access_mode::readwrite);
    h_pos.data[0].x = Scalar(-9.6); h_pos.data[0].y = 0; h_pos.data[0].z = 0.0;
    h_vel.data[0].x = Scalar(-0.5);
    h_pos.data[1].x =  Scalar(9.6); h_pos.data[1].y = 0; h_pos.data[1].z = 0.0;
    h_vel.data[1].x = Scalar(0.6);
    h_pos.data[2].x = 0; h_pos.data[2].y = Scalar(-19.6); h_pos.data[2].z = 0.0;
    h_vel.data[2].y = Scalar(-0.5);
    h_pos.data[3].x = 0; h_pos.data[3].y = Scalar(19.6); h_pos.data[3].z = 0.0;
    h_vel.data[3].y = Scalar(0.6);
    h_pos.data[4].x = 0; h_pos.data[4].y = 0; h_pos.data[4].z = Scalar(-29.6);
    h_vel.data[4].z = Scalar(-0.5);
    h_pos.data[5].x = 0; h_pos.data[5].y = 0; h_pos.data[5].z =  Scalar(29.6);
    h_vel.data[5].z = Scalar(0.6);
    }
    
    Scalar deltaT = 1.0;
    shared_ptr<TwoStepNVE> two_step_nve = nve_creator(sysdef_6, group_all);
    shared_ptr<IntegratorTwoStep> nve_up(new IntegratorTwoStep(sysdef_6, deltaT));
    nve_up->addIntegrationMethod(two_step_nve);
    
    // no forces on these particles
    shared_ptr<ConstForceCompute> fc1(new ConstForceCompute(sysdef_6, 0, 0.0, 0.0));
    nve_up->addForceCompute(fc1);
    
    nve_up->prepRun(0);
    
    // move the particles across the boundary
    nve_up->update(0);
    
    // check that they go to the proper final position
    {
    ArrayHandle<Scalar4> h_pos(pdata_6->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<int3> h_image(pdata_6->getImages(), access_location::host, access_mode::read);
    MY_BOOST_CHECK_CLOSE(h_pos.data[0].x, 9.9, tol);
    BOOST_CHECK_EQUAL(h_image.data[0].x, -1);
    MY_BOOST_CHECK_CLOSE(h_pos.data[1].x, -9.8, tol);
    BOOST_CHECK_EQUAL(h_image.data[1].x, 1);
    MY_BOOST_CHECK_CLOSE(h_pos.data[2].y, 19.9, tol);
    BOOST_CHECK_EQUAL(h_image.data[2].y, -1);
    MY_BOOST_CHECK_CLOSE(h_pos.data[3].y, -19.8, tol);
    BOOST_CHECK_EQUAL(h_image.data[3].y, 1);
    MY_BOOST_CHECK_CLOSE(h_pos.data[4].z, 29.9, tol);
    BOOST_CHECK_EQUAL(h_image.data[4].z, -1);
    MY_BOOST_CHECK_CLOSE(h_pos.data[5].z, -29.8, tol);
    BOOST_CHECK_EQUAL(h_image.data[5].z, 1);
    }
    }

//! Compares the output from one TwoStepNVE to another
void nve_updater_compare_test(twostepnve_creator nve_creator1,
                              twostepnve_creator nve_creator2,
                              boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    const unsigned int N = 1000;
    
    // create two identical random particle systems to simulate
    RandomInitializer rand_init(N, Scalar(0.2), Scalar(0.9), "A");
    rand_init.setSeed(12345);
    boost::shared_ptr<SnapshotSystemData> snap = rand_init.getSnapshot();
    shared_ptr<SystemDefinition> sysdef1(new SystemDefinition(snap, exec_conf));
    shared_ptr<ParticleData> pdata1 = sysdef1->getParticleData();
    shared_ptr<ParticleSelector> selector_all1(new ParticleSelectorTag(sysdef1, 0, pdata1->getN()-1));
    shared_ptr<ParticleGroup> group_all1(new ParticleGroup(sysdef1, selector_all1));

    shared_ptr<SystemDefinition> sysdef2(new SystemDefinition(snap, exec_conf));
    shared_ptr<ParticleData> pdata2 = sysdef2->getParticleData();
    shared_ptr<ParticleSelector> selector_all2(new ParticleSelectorTag(sysdef2, 0, pdata2->getN()-1));
    shared_ptr<ParticleGroup> group_all2(new ParticleGroup(sysdef2, selector_all2));
    
    shared_ptr<NeighborList> nlist1(new NeighborList(sysdef1, Scalar(3.0), Scalar(0.8)));
    shared_ptr<NeighborList> nlist2(new NeighborList(sysdef2, Scalar(3.0), Scalar(0.8)));
    
    shared_ptr<PotentialPairLJ> fc1(new PotentialPairLJ(sysdef1, nlist1));
    fc1->setRcut(0, 0, Scalar(3.0));
    shared_ptr<PotentialPairLJ> fc2(new PotentialPairLJ(sysdef2, nlist2));
    fc2->setRcut(0, 0, Scalar(3.0));

    // setup some values for alpha and sigma
    Scalar epsilon = Scalar(1.0);
    Scalar sigma = Scalar(1.2);
    Scalar alpha = Scalar(0.45);
    Scalar lj1 = Scalar(4.0) * epsilon * pow(sigma,Scalar(12.0));
    Scalar lj2 = alpha * Scalar(4.0) * epsilon * pow(sigma,Scalar(6.0));
    
    // specify the force parameters
    fc1->setParams(0,0,make_scalar2(lj1,lj2));
    fc2->setParams(0,0,make_scalar2(lj1,lj2));
    
    shared_ptr<TwoStepNVE> two_step_nve1 = nve_creator1(sysdef1, group_all1);
    shared_ptr<IntegratorTwoStep> nve1(new IntegratorTwoStep(sysdef1, Scalar(0.005)));
    nve1->addIntegrationMethod(two_step_nve1);

    shared_ptr<TwoStepNVE> two_step_nve2 = nve_creator2(sysdef2, group_all2);
    shared_ptr<IntegratorTwoStep> nve2(new IntegratorTwoStep(sysdef2, Scalar(0.005)));
    nve2->addIntegrationMethod(two_step_nve2);
    
    nve1->addForceCompute(fc1);
    nve2->addForceCompute(fc2);
    
    nve1->prepRun(0);
    nve2->prepRun(0);
    
    // step for only a few time steps and verify that they are the same
    // we can't do much more because these things are chaotic and diverge quickly
    for (int i = 0; i < 5; i++)
        {
        std::cout << "i = " << i << std::endl;
        {
        ArrayHandle<Scalar4> h_pos1(pdata1->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_vel1(pdata1->getVelocities(), access_location::host, access_mode::read);
        ArrayHandle<Scalar3> h_accel1(pdata1->getAccelerations(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_pos2(pdata2->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_vel2(pdata2->getVelocities(), access_location::host, access_mode::read);
        ArrayHandle<Scalar3> h_accel2(pdata2->getAccelerations(), access_location::host, access_mode::read);

        Scalar rough_tol = 2.0;
        //cout << arrays1.x[100] << " " << arrays2.x[100] << endl;
        
        // check position, velocity and acceleration
        for (unsigned int j = 0; j < N; j++)
            {
            MY_BOOST_CHECK_CLOSE(h_pos1.data[j].x, h_pos2.data[j].x, rough_tol);
            MY_BOOST_CHECK_CLOSE(h_pos1.data[j].y, h_pos2.data[j].y, rough_tol);
            MY_BOOST_CHECK_CLOSE(h_pos1.data[j].z, h_pos2.data[j].z, rough_tol);
            
            MY_BOOST_CHECK_CLOSE(h_vel1.data[j].x, h_vel2.data[j].x, rough_tol);
            MY_BOOST_CHECK_CLOSE(h_vel1.data[j].y, h_vel2.data[j].y, rough_tol);
            MY_BOOST_CHECK_CLOSE(h_vel1.data[j].z, h_vel2.data[j].z, rough_tol);
            
            MY_BOOST_CHECK_CLOSE(h_accel1.data[j].x, h_accel2.data[j].x, rough_tol);
            MY_BOOST_CHECK_CLOSE(h_accel1.data[j].y, h_accel2.data[j].y, rough_tol);
            MY_BOOST_CHECK_CLOSE(h_accel1.data[j].z, h_accel2.data[j].z, rough_tol);
            }
            
        }
        nve1->update(i);
        nve2->update(i);
        }
    }

//! TwoStepNVE factory for the unit tests
shared_ptr<TwoStepNVE> base_class_nve_creator(shared_ptr<SystemDefinition> sysdef, shared_ptr<ParticleGroup> group)
    {
    return shared_ptr<TwoStepNVE>(new TwoStepNVE(sysdef, group));
    }

#ifdef ENABLE_CUDA
//! TwoStepNVEGPU factory for the unit tests
shared_ptr<TwoStepNVE> gpu_nve_creator(shared_ptr<SystemDefinition> sysdef, shared_ptr<ParticleGroup> group)
    {
    return shared_ptr<TwoStepNVE>(new TwoStepNVEGPU(sysdef, group));
    }
#endif


//! boost test case for base class integration tests
BOOST_AUTO_TEST_CASE( TwoStepNVE_integrate_tests )
    {
    twostepnve_creator nve_creator = bind(base_class_nve_creator, _1, _2);
    nve_updater_integrate_tests(nve_creator, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

//! boost test case for base class limit tests
BOOST_AUTO_TEST_CASE( TwoStepNVE_limit_tests )
    {
    twostepnve_creator nve_creator = bind(base_class_nve_creator, _1, _2);
    nve_updater_limit_tests(nve_creator, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

//! boost test case for base class boundary tests
BOOST_AUTO_TEST_CASE( TwoStepNVE_boundary_tests )
    {
    twostepnve_creator nve_creator = bind(base_class_nve_creator, _1, _2);
    nve_updater_boundary_tests(nve_creator, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
//! Need work on NVEUpdaterGPU with rigid bodies to test these cases
#ifdef ENABLE_CUDA
//! boost test case for base class integration tests
BOOST_AUTO_TEST_CASE( TwoStepNVEGPU_integrate_tests )
    {
    twostepnve_creator nve_creator_gpu = bind(gpu_nve_creator, _1, _2);
    nve_updater_integrate_tests(nve_creator_gpu, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! boost test case for base class limit tests
BOOST_AUTO_TEST_CASE( TwoStepNVEGPU_limit_tests )
    {
    twostepnve_creator nve_creator = bind(gpu_nve_creator, _1, _2);
    nve_updater_limit_tests(nve_creator, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! boost test case for base class boundary tests
BOOST_AUTO_TEST_CASE( TwoStepNVEGPU_boundary_tests )
    {
    twostepnve_creator nve_creator_gpu = bind(gpu_nve_creator, _1, _2);
    nve_updater_boundary_tests(nve_creator_gpu, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! boost test case for comparing the GPU and CPU NVEUpdaters
BOOST_AUTO_TEST_CASE( TwoStepNVEGPU_comparison_tests)
    {
    twostepnve_creator nve_creator_gpu = bind(gpu_nve_creator, _1, _2);
    twostepnve_creator nve_creator = bind(base_class_nve_creator, _1, _2);
    nve_updater_compare_test(nve_creator, nve_creator_gpu, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

#endif

#ifdef WIN32
#pragma warning( pop )
#endif

