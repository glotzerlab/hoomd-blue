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
// Maintainer: akohlmey

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <iostream>

#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include "CGCMMForceCompute.h"
#ifdef ENABLE_CUDA
#include "CGCMMForceComputeGPU.h"
#endif

#include "BinnedNeighborList.h"
#include "Initializers.h"

#include <math.h>

using namespace std;
using namespace boost;

/*! \file cgcmm_force_test.cc
    \brief Implements unit tests for CGCMMForceCompute and descendants
    \ingroup unit_tests
*/

//! Name the unit test module
#define BOOST_TEST_MODULE CGCMMForceTests
#include "boost_utf_configure.h"

//! Typedef'd CGCMMForceCompute factory
typedef boost::function<shared_ptr<CGCMMForceCompute> (shared_ptr<SystemDefinition> sysdef, shared_ptr<NeighborList> nlist, Scalar r_cut)> cgcmmforce_creator;

//! Test the ability of the cgcmm LJ12-4 force compute to actually calucate forces
void cgcmm_force_particle124_test(cgcmmforce_creator cgcmm_creator, ExecutionConfiguration exec_conf)
    {
#ifdef ENABLE_CUDA
    g_gpu_error_checking = true;
#endif
    
    // this 3-particle test subtly checks several conditions
    // the particles are arranged on the x axis,  1   2   3
    // such that 2 is inside the cuttoff radius of 1 and 3, but 1 and 3 are outside the cuttoff
    // of course, the buffer will be set on the neighborlist so that 3 is included in it
    // thus, this case tests the ability of the force summer to sum more than one force on
    // a particle and ignore a particle outside the radius
    
    // periodic boundary conditions will be handeled in another test
    shared_ptr<SystemDefinition> sysdef_3(new SystemDefinition(3, BoxDim(1000.0), 1, 0, 0, 0, 0, exec_conf));
    shared_ptr<ParticleData> pdata_3 = sysdef_3->getParticleData();
    
    ParticleDataArrays arrays = pdata_3->acquireReadWrite();
    arrays.x[0] = arrays.y[0] = arrays.z[0] = 0.0;
    arrays.x[1] = Scalar(pow(3.0,1.0/8.0)); arrays.y[1] = arrays.z[1] = 0.0;
    arrays.x[2] = Scalar(2.0*pow(3.0,1.0/8.0)); arrays.y[2] = arrays.z[2] = 0.0;
    pdata_3->release();
    shared_ptr<NeighborList> nlist_3(new NeighborList(sysdef_3, Scalar(1.3), Scalar(3.0)));
    shared_ptr<CGCMMForceCompute> fc_3 = cgcmm_creator(sysdef_3, nlist_3, Scalar(1.3));
    
    // first test: setup a sigma of 1.0 so that all forces will be 0
    Scalar epsilon = Scalar(1.15);
    Scalar sigma = Scalar(1.0);
    Scalar alpha = Scalar(1.0);
    Scalar lj1 = Scalar(2.598076) * epsilon * pow(sigma,Scalar(12.0));
    Scalar lj2 = Scalar(0.0);
    Scalar lj3 = Scalar(0.0);
    Scalar lj4 = -alpha * Scalar(2.598076) * epsilon * pow(sigma,Scalar(4.0));
    //Scalar lj1 = Scalar(0.0);
    //Scalar lj2 = alpha * Scalar(6.75) * epsilon * pow(sigma,Scalar(6.0));
    //Scalar lj3 = Scalar(6.75) * epsilon * pow(sigma,Scalar(9.0));
    //Scalar lj4 = Scalar(0.0);
    
    fc_3->setParams(0,0,lj1,lj2,lj3,lj4);
    
    // compute the forces
    fc_3->compute(0);
    
    ForceDataArrays force_arrays = fc_3->acquire();
    MY_BOOST_CHECK_SMALL(force_arrays.fx[0], tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fy[0], tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[0], tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[0], -0.575, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.virial[0], tol);
    
    MY_BOOST_CHECK_SMALL(force_arrays.fx[1], tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fy[1], tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[1], tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[1], -1.15, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.virial[1], tol);
    
    MY_BOOST_CHECK_SMALL(force_arrays.fx[2], tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fy[2], tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[2], tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[2], -0.575, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.virial[2], tol);
    
    // now change sigma and alpha so we can check that it is computing the right force
    sigma = Scalar(1.2); // < bigger sigma should push particle 0 left and particle 2 right
    alpha = Scalar(0.45);
    lj1 = Scalar(2.598076) * epsilon * pow(sigma,Scalar(12.0));
    lj2 = Scalar(0.0);
    lj3 = Scalar(0.0);
    lj4 = -alpha * Scalar(2.598076) * epsilon * pow(sigma,Scalar(4.0));
    fc_3->setParams(0,0,lj1,lj2,lj3,lj4);
    fc_3->compute(1);
    
    force_arrays = fc_3->acquire();
    MY_BOOST_CHECK_CLOSE(force_arrays.fx[0], -48.0146523, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fy[0], tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[0], tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[0], 1.758563, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.virial[0], 9.18042374, tol);
    
    // center particle should still be a 0 force by symmetry
    MY_BOOST_CHECK_SMALL(force_arrays.fx[1], tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fy[1], 1e-5);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[1], 1e-5);
    // there is still an energy and virial, though
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[1], 3.517125, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.virial[1], 18.3608475, tol);
    
    MY_BOOST_CHECK_CLOSE(force_arrays.fx[2], 48.0146561, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fy[2], tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[2], tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[2], 1.758563, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.virial[2], 9.18042374, tol);
    
    // swap the order of particles 0 ans 2 in memory to check that the force compute handles this properly
    arrays = pdata_3->acquireReadWrite();
    arrays.x[2] = arrays.y[2] = arrays.z[2] = 0.0;
    arrays.x[0] = Scalar(2.0*pow(3.0,1.0/8.0)); arrays.y[0] = arrays.z[0] = 0.0;
    
    arrays.tag[0] = 2;
    arrays.tag[2] = 0;
    arrays.rtag[0] = 2;
    arrays.rtag[2] = 0;
    pdata_3->release();
    
    // notify the particle data that we changed the order
    pdata_3->notifyParticleSort();
    
    // recompute the forces at the same timestep, they should be updated
    fc_3->compute(1);
    force_arrays = fc_3->acquire();
    MY_BOOST_CHECK_CLOSE(force_arrays.fx[0], 48.0146523, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.fx[2], -48.0146523, tol);
    }

//! Test the ability of the cgcmm LJ9-6 force compute to actually calucate forces
void cgcmm_force_particle96_test(cgcmmforce_creator cgcmm_creator, ExecutionConfiguration exec_conf)
    {
#ifdef ENABLE_CUDA
    g_gpu_error_checking = true;
#endif
    
    // this 3-particle test subtly checks several conditions
    // the particles are arranged on the x axis,  1   2   3
    // such that 2 is inside the cuttoff radius of 1 and 3, but 1 and 3 are outside the cuttoff
    // of course, the buffer will be set on the neighborlist so that 3 is included in it
    // thus, this case tests the ability of the force summer to sum more than one force on
    // a particle and ignore a particle outside the radius
    
    // periodic boundary conditions will be handeled in another test
    shared_ptr<SystemDefinition> sysdef_3(new SystemDefinition(3, BoxDim(1000.0), 1, 0, 0, 0, 0, exec_conf));
    shared_ptr<ParticleData> pdata_3 = sysdef_3->getParticleData();
    
    ParticleDataArrays arrays = pdata_3->acquireReadWrite();
    arrays.x[0] = arrays.y[0] = arrays.z[0] = 0.0;
    arrays.x[1] = Scalar(pow(1.5,1.0/3.0)); arrays.y[1] = arrays.z[1] = 0.0;
    arrays.x[2] = Scalar(2.0*pow(1.5,1.0/3.0)); arrays.y[2] = arrays.z[2] = 0.0;
    pdata_3->release();
    shared_ptr<NeighborList> nlist_3(new NeighborList(sysdef_3, Scalar(1.3), Scalar(3.0)));
    shared_ptr<CGCMMForceCompute> fc_3 = cgcmm_creator(sysdef_3, nlist_3, Scalar(1.3));
    
    // first test: setup a sigma of 1.0 so that all forces will be 0
    Scalar epsilon = Scalar(1.15);
    Scalar sigma = Scalar(1.0);
    Scalar alpha = Scalar(1.0);
    Scalar lj1 = Scalar(0.0);
    Scalar lj2 = Scalar(6.75) * epsilon * pow(sigma,Scalar(9.0));
    Scalar lj3 = -alpha * Scalar(6.75) * epsilon * pow(sigma,Scalar(6.0));
    Scalar lj4 = Scalar(0.0);
    
    fc_3->setParams(0,0,lj1,lj2,lj3,lj4);
    
    // compute the forces
    fc_3->compute(0);
    
    ForceDataArrays force_arrays = fc_3->acquire();
    MY_BOOST_CHECK_SMALL(force_arrays.fx[0], tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fy[0], tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[0], tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[0], -0.575, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.virial[0], tol);
    
    MY_BOOST_CHECK_SMALL(force_arrays.fx[1], tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fy[1], tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[1], tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[1], -1.15, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.virial[1], tol);
    
    MY_BOOST_CHECK_SMALL(force_arrays.fx[2], tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fy[2], tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[2], tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[2], -0.575, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.virial[2], tol);
    
    // now change sigma and alpha so we can check that it is computing the right force
    sigma = Scalar(1.2); // < bigger sigma should push particle 0 left and particle 2 right
    alpha = Scalar(0.45);
    lj1 = Scalar(0.0);
    lj2 = Scalar(6.75) * epsilon * pow(sigma,Scalar(9.0));
    lj3 = -alpha * Scalar(6.75) * epsilon * pow(sigma,Scalar(6.0));
    lj4 = Scalar(0.0);
    fc_3->setParams(0,0,lj1,lj2,lj3,lj4);
    fc_3->compute(1);
    
    force_arrays = fc_3->acquire();
    MY_BOOST_CHECK_CLOSE(force_arrays.fx[0], -69.00675, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fy[0], tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[0], tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[0], 3.615877, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.virial[0], 13.1655083, tol);
    
    // center particle should still be a 0 force by symmetry
    MY_BOOST_CHECK_SMALL(force_arrays.fx[1], tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fy[1], 1e-5);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[1], 1e-5);
    // there is still an energy and virial, though
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[1], 7.231755, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.virial[1], 26.3310165, tol);
    
    MY_BOOST_CHECK_CLOSE(force_arrays.fx[2], 69.00675, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fy[2], tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[2], tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[2], 3.615877, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.virial[2], 13.1655083, tol);
    
    // swap the order of particles 0 ans 2 in memory to check that the force compute handles this properly
    arrays = pdata_3->acquireReadWrite();
    arrays.x[2] = arrays.y[2] = arrays.z[2] = 0.0;
    arrays.x[0] = Scalar(2.0*pow(1.5,1.0/3.0)); arrays.y[0] = arrays.z[0] = 0.0;
    
    arrays.tag[0] = 2;
    arrays.tag[2] = 0;
    arrays.rtag[0] = 2;
    arrays.rtag[2] = 0;
    pdata_3->release();
    
    // notify the particle data that we changed the order
    pdata_3->notifyParticleSort();
    
    // recompute the forces at the same timestep, they should be updated
    fc_3->compute(1);
    force_arrays = fc_3->acquire();
    MY_BOOST_CHECK_CLOSE(force_arrays.fx[0], 69.00675, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.fx[2], -69.00675, tol);
    }

//! Tests the ability of a CGCMMForceCompute to handle periodic boundary conditions
void cgcmm_force_periodic_test(cgcmmforce_creator cgcmm_creator, ExecutionConfiguration exec_conf)
    {
#ifdef ENABLE_CUDA
    g_gpu_error_checking = true;
#endif
    
    ////////////////////////////////////////////////////////////////////
    // now, lets do a more thorough test and include boundary conditions
    // there are way too many permutations to test here, so I will simply
    // test +x, -x, +y, -y, +z, and -z independantly
    // build a 6 particle system with particles across each boundary
    // also test the ability of the force compute to use different particle types
    
    shared_ptr<SystemDefinition> sysdef_6(new SystemDefinition(6, BoxDim(20.0, 40.0, 60.0), 3, 0, 0, 0, 0, exec_conf));
    shared_ptr<ParticleData> pdata_6 = sysdef_6->getParticleData();
    
    ParticleDataArrays arrays = pdata_6->acquireReadWrite();
    arrays.x[0] = Scalar(-9.6); arrays.y[0] = 0; arrays.z[0] = 0.0;
    arrays.x[1] =  Scalar(9.6); arrays.y[1] = 0; arrays.z[1] = 0.0;
    arrays.x[2] = 0; arrays.y[2] = Scalar(-19.6); arrays.z[2] = 0.0;
    arrays.x[3] = 0; arrays.y[3] = Scalar(19.6); arrays.z[3] = 0.0;
    arrays.x[4] = 0; arrays.y[4] = 0; arrays.z[4] = Scalar(-29.6);
    arrays.x[5] = 0; arrays.y[5] = 0; arrays.z[5] =  Scalar(29.6);
    
    arrays.type[0] = 0;
    arrays.type[1] = 1;
    arrays.type[2] = 2;
    arrays.type[3] = 0;
    arrays.type[4] = 2;
    arrays.type[5] = 1;
    pdata_6->release();
    
    shared_ptr<NeighborList> nlist_6(new NeighborList(sysdef_6, Scalar(1.3), Scalar(3.0)));
    shared_ptr<CGCMMForceCompute> fc_6 = cgcmm_creator(sysdef_6, nlist_6, Scalar(1.3));
    
    // choose a small sigma so that all interactions are attractive
    Scalar epsilon = Scalar(1.0);
    Scalar sigma = Scalar(0.5);
    Scalar alpha = Scalar(0.45);
    Scalar lj1 = Scalar(4.0) * epsilon * pow(sigma,Scalar(12.0));
    Scalar lj2 = Scalar(0.0);
    Scalar lj3 = -alpha * Scalar(4.0) * epsilon * pow(sigma,Scalar(6.0));
    Scalar lj4 = Scalar(0.0);
    
    // make life easy: just change epsilon for the different pairs
    fc_6->setParams(0,0,lj1,lj2,lj3,lj4);
    fc_6->setParams(0,1,Scalar(2.0)*lj1,Scalar(2.0)*lj2,Scalar(2.0)*lj3,Scalar(2.0)*lj4);
    fc_6->setParams(0,2,Scalar(3.0)*lj1,Scalar(3.0)*lj2,Scalar(3.0)*lj3,Scalar(3.0)*lj4);
    fc_6->setParams(1,1,Scalar(4.0)*lj1,Scalar(4.0)*lj2,Scalar(4.0)*lj3,Scalar(4.0)*lj4);
    fc_6->setParams(1,2,Scalar(5.0)*lj1,Scalar(5.0)*lj2,Scalar(5.0)*lj3,Scalar(5.0)*lj4);
    fc_6->setParams(2,2,Scalar(6.0)*lj1,Scalar(6.0)*lj2,Scalar(6.0)*lj3,Scalar(6.0)*lj4);
    
    fc_6->compute(0);
    
    ForceDataArrays force_arrays = fc_6->acquire();
    // particle 0 should be pulled left
    MY_BOOST_CHECK_CLOSE(force_arrays.fx[0], -1.18299976747949, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fy[0], tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[0], tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.virial[0], -0.15773330233059, tol);
    
    // particle 1 should be pulled right
    MY_BOOST_CHECK_CLOSE(force_arrays.fx[1], 1.18299976747949, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fy[1], 1e-5);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[1], 1e-5);
    MY_BOOST_CHECK_CLOSE(force_arrays.virial[1], -0.15773330233059, tol);
    
    // particle 2 should be pulled down
    MY_BOOST_CHECK_CLOSE(force_arrays.fy[2], -1.77449965121923, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fx[2], tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[2], tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.virial[2], -0.23659995349591, tol);
    
    // particle 3 should be pulled up
    MY_BOOST_CHECK_CLOSE(force_arrays.fy[3], 1.77449965121923, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fx[3], 1e-5);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[3], 1e-5);
    MY_BOOST_CHECK_CLOSE(force_arrays.virial[3], -0.23659995349591, tol);
    
    // particle 4 should be pulled back
    MY_BOOST_CHECK_CLOSE(force_arrays.fz[4], -2.95749941869871, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fx[4], tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fy[4], tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.virial[4], -0.39433325582651, tol);
    
    // particle 3 should be pulled forward
    MY_BOOST_CHECK_CLOSE(force_arrays.fz[5], 2.95749941869871, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fx[5], 1e-5);
    MY_BOOST_CHECK_SMALL(force_arrays.fy[5], 1e-5);
    MY_BOOST_CHECK_CLOSE(force_arrays.virial[5], -0.39433325582651, tol);
    }

//! Unit test a comparison between 2 CGCMMForceComputes on a "real" system
void cgcmm_force_comparison_test(cgcmmforce_creator cgcmm_creator1, cgcmmforce_creator cgcmm_creator2, ExecutionConfiguration exec_conf)
    {
#ifdef ENABLE_CUDA
    g_gpu_error_checking = true;
#endif
    
    const unsigned int N = 5000;
    
    // create a random particle system to sum forces on
    RandomInitializer rand_init(N, Scalar(0.2), Scalar(0.9), "A");
    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(rand_init, exec_conf));
    shared_ptr<BinnedNeighborList> nlist(new BinnedNeighborList(sysdef, Scalar(3.0), Scalar(0.8)));
    
    shared_ptr<CGCMMForceCompute> fc1 = cgcmm_creator1(sysdef, nlist, Scalar(3.0));
    shared_ptr<CGCMMForceCompute> fc2 = cgcmm_creator2(sysdef, nlist, Scalar(3.0));
    
    // setup some values for alpha and sigma
    Scalar epsilon = Scalar(1.0);
    Scalar sigma = Scalar(1.2);
    Scalar alpha = Scalar(0.45);
    Scalar lj1 = Scalar(4.0) * epsilon * pow(sigma,Scalar(12.0));
    Scalar lj2 = Scalar(0.0);
    Scalar lj3 = -alpha * Scalar(4.0) * epsilon * pow(sigma,Scalar(6.0));
    Scalar lj4 = Scalar(0.0);
    
    // specify the force parameters
    fc1->setParams(0,0,lj1,lj2,lj3,lj4);
    fc2->setParams(0,0,lj1,lj2,lj3,lj4);
    
    // compute the forces
    fc1->compute(0);
    fc2->compute(0);
    
    // verify that the forces are identical (within roundoff errors)
    ForceDataArrays arrays1 = fc1->acquire();
    ForceDataArrays arrays2 = fc2->acquire();
    
    // compare average deviation between the two computes
    double deltaf2 = 0.0;
    double deltape2 = 0.0;
    double deltav2 = 0.0;
        
    for (unsigned int i = 0; i < N; i++)
        {
        deltaf2 += double(arrays1.fx[i] - arrays2.fx[i]) * double(arrays1.fx[i] - arrays2.fx[i]);
        deltaf2 += double(arrays1.fy[i] - arrays2.fy[i]) * double(arrays1.fy[i] - arrays2.fy[i]);
        deltaf2 += double(arrays1.fz[i] - arrays2.fz[i]) * double(arrays1.fz[i] - arrays2.fz[i]);
        deltape2 += double(arrays1.pe[i] - arrays2.pe[i]) * double(arrays1.pe[i] - arrays2.pe[i]);
        deltav2 += double(arrays1.virial[i] - arrays2.virial[i]) * double(arrays1.virial[i] - arrays2.virial[i]);

        // also check that each individual calculation is somewhat close
        BOOST_CHECK_CLOSE(arrays1.fx[i], arrays2.fx[i], loose_tol);
        BOOST_CHECK_CLOSE(arrays1.fy[i], arrays2.fy[i], loose_tol);
        BOOST_CHECK_CLOSE(arrays1.fz[i], arrays2.fz[i], loose_tol);
        BOOST_CHECK_CLOSE(arrays1.pe[i], arrays2.pe[i], loose_tol);
        BOOST_CHECK_CLOSE(arrays1.virial[i], arrays2.virial[i], loose_tol);
        }
    deltaf2 /= double(sysdef->getParticleData()->getN());
    deltape2 /= double(sysdef->getParticleData()->getN());
    deltav2 /= double(sysdef->getParticleData()->getN());
    BOOST_CHECK_SMALL(deltaf2, double(tol_small));
    BOOST_CHECK_SMALL(deltape2, double(tol_small));
    BOOST_CHECK_SMALL(deltav2, double(tol_small));
    }

//! CGCMMForceCompute creator for unit tests
shared_ptr<CGCMMForceCompute> base_class_cgcmm_creator(shared_ptr<SystemDefinition> sysdef, shared_ptr<NeighborList> nlist, Scalar r_cut)
    {
    return shared_ptr<CGCMMForceCompute>(new CGCMMForceCompute(sysdef, nlist, r_cut));
    }

#ifdef ENABLE_CUDA
//! CGCMMForceComputeGPU creator for unit tests
shared_ptr<CGCMMForceCompute> gpu_cgcmm_creator(shared_ptr<SystemDefinition> sysdef, shared_ptr<NeighborList> nlist, Scalar r_cut)
    {
    nlist->setStorageMode(NeighborList::full);
    shared_ptr<CGCMMForceComputeGPU> cgcmm(new CGCMMForceComputeGPU(sysdef, nlist, r_cut));
    // the default block size kills valgrind :) reduce it
    cgcmm->setBlockSize(64);
    return cgcmm;
    }
#endif

//! boost test case for particle test on CPU
BOOST_AUTO_TEST_CASE( CGCMMForce_particle124 )
    {
    cgcmmforce_creator cgcmm_creator_base = bind(base_class_cgcmm_creator, _1, _2, _3);
    cgcmm_force_particle124_test(cgcmm_creator_base, ExecutionConfiguration(ExecutionConfiguration::CPU));
    }

//! boost test case for particle test on CPU
BOOST_AUTO_TEST_CASE( CGCMMForce_particle96 )
    {
    cgcmmforce_creator cgcmm_creator_base = bind(base_class_cgcmm_creator, _1, _2, _3);
    cgcmm_force_particle96_test(cgcmm_creator_base, ExecutionConfiguration(ExecutionConfiguration::CPU));
    }

//! boost test case for periodic test on CPU
BOOST_AUTO_TEST_CASE( CGCMMForce_periodic )
    {
    cgcmmforce_creator cgcmm_creator_base = bind(base_class_cgcmm_creator, _1, _2, _3);
    cgcmm_force_periodic_test(cgcmm_creator_base, ExecutionConfiguration(ExecutionConfiguration::CPU));
    }

# ifdef ENABLE_CUDA
//! boost test case for particle test on GPU - threaded
BOOST_AUTO_TEST_CASE( CGCMMForceGPU_particle124 )
    {
    cgcmmforce_creator cgcmm_creator_gpu = bind(gpu_cgcmm_creator, _1, _2, _3);
    cgcmm_force_particle124_test(cgcmm_creator_gpu, ExecutionConfiguration(ExecutionConfiguration::GPU));
    }

//! boost test case for particle test on GPU - threaded
BOOST_AUTO_TEST_CASE( CGCMMForceGPU_particle96 )
    {
    cgcmmforce_creator cgcmm_creator_gpu = bind(gpu_cgcmm_creator, _1, _2, _3);
    cgcmm_force_particle96_test(cgcmm_creator_gpu, ExecutionConfiguration(ExecutionConfiguration::GPU));
    }

//! boost test case for periodic test on the GPU
BOOST_AUTO_TEST_CASE( CGCMMForceGPU_periodic )
    {
    cgcmmforce_creator cgcmm_creator_gpu = bind(gpu_cgcmm_creator, _1, _2, _3);
    cgcmm_force_periodic_test(cgcmm_creator_gpu, ExecutionConfiguration(ExecutionConfiguration::GPU));
    }

//! boost test case for comparing GPU output to base class output
BOOST_AUTO_TEST_CASE( CGCMMForceGPU_compare )
    {
    cgcmmforce_creator cgcmm_creator_gpu = bind(gpu_cgcmm_creator, _1, _2, _3);
    cgcmmforce_creator cgcmm_creator_base = bind(base_class_cgcmm_creator, _1, _2, _3);
    cgcmm_force_comparison_test(cgcmm_creator_base, cgcmm_creator_gpu, ExecutionConfiguration(ExecutionConfiguration::GPU));
    }

#endif

#ifdef WIN32
#pragma warning( pop )
#endif

