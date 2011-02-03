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
#include <fstream>

#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include "AllPairPotentials.h"

#include "NeighborListBinned.h"
#include "Initializers.h"

#include <math.h>

using namespace std;
using namespace boost;

/*! \file gaussian_force_test.cc
    \brief Implements unit tests for PotentialPairGauss and descendants
    \ingroup unit_tests
*/

//! Name the unit test module
#define BOOST_TEST_MODULE PotentialPairGaussTests
#include "boost_utf_configure.h"

//! Typedef'd PotentialPairGauss factory
typedef boost::function<shared_ptr<PotentialPairGauss> (shared_ptr<SystemDefinition> sysdef,
                                                        shared_ptr<NeighborList> nlist)> gaussforce_creator;

//! Test the ability of the gauss force compute to actually calucate forces
void gauss_force_particle_test(gaussforce_creator gauss_creator, boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
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
    arrays.x[1] = Scalar(1.0); arrays.y[1] = arrays.z[1] = 0.0;
    arrays.x[2] = Scalar(2.0); arrays.y[2] = arrays.z[2] = 0.0;
    pdata_3->release();
    shared_ptr<NeighborList> nlist_3(new NeighborList(sysdef_3, Scalar(1.3), Scalar(3.0)));
    shared_ptr<PotentialPairGauss> fc_3 = gauss_creator(sysdef_3, nlist_3);
    fc_3->setRcut(0, 0, Scalar(1.3));
    
    // first test: choose a basic sigma
    Scalar epsilon = Scalar(1.15);
    Scalar sigma = Scalar(0.5);
    fc_3->setParams(0,0,make_scalar2(epsilon,sigma));
    
    // compute the forces
    fc_3->compute(0);
    
    ForceDataArrays force_arrays = fc_3->acquire();
    MY_BOOST_CHECK_CLOSE(force_arrays.fx[0], -0.622542302888418, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fy[0], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[0], tol_small);
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[0], 0.155635575722105/2.0, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.virial[0], 0.103757050481403, tol);
    
    MY_BOOST_CHECK_SMALL(force_arrays.fx[1], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.fy[1], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[1], tol_small);
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[1], 0.155635575722105, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.virial[1], 0.103757050481403*2, tol);
    
    MY_BOOST_CHECK_CLOSE(force_arrays.fx[2], 0.622542302888418, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fy[2], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[2], tol_small);
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[2], 0.155635575722105/2.0, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.virial[2], 0.103757050481403, tol);
    
    // swap the order of particles 0 ans 2 in memory to check that the force compute handles this properly
    arrays = pdata_3->acquireReadWrite();
    arrays.x[2] = arrays.y[2] = arrays.z[2] = 0.0;
    arrays.x[0] = Scalar(2.0); arrays.y[0] = arrays.z[0] = 0.0;
    
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
    MY_BOOST_CHECK_CLOSE(force_arrays.fx[0], 0.622542302888418, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.fx[2], -0.622542302888418, tol);
    }

//! Tests the ability of a PotentialPairGauss to handle periodic boundary conditions
void gauss_force_periodic_test(gaussforce_creator gauss_creator, boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
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
    shared_ptr<PotentialPairGauss> fc_6 = gauss_creator(sysdef_6, nlist_6);
    fc_6->setRcut(0, 0, Scalar(1.3));
    fc_6->setRcut(0, 1, Scalar(1.3));
    fc_6->setRcut(0, 2, Scalar(1.3));
    fc_6->setRcut(1, 1, Scalar(1.3));
    fc_6->setRcut(1, 2, Scalar(1.3));
    fc_6->setRcut(2, 2, Scalar(1.3));
    
    // choose a small sigma so that all interactions are attractive
    Scalar epsilon = Scalar(1.0);
    Scalar sigma = Scalar(0.5);
    
    // make life easy: just change epsilon for the different pairs
    fc_6->setParams(0,0,make_scalar2(epsilon,sigma));
    fc_6->setParams(0,1,make_scalar2(Scalar(2.0)*epsilon,sigma));
    fc_6->setParams(0,2,make_scalar2(Scalar(3.0)*epsilon,sigma));
    fc_6->setParams(1,1,make_scalar2(Scalar(4.0)*epsilon,sigma));
    fc_6->setParams(1,2,make_scalar2(Scalar(5.0)*epsilon,sigma));
    fc_6->setParams(2,2,make_scalar2(Scalar(6.0)*epsilon,sigma));
    
    fc_6->compute(0);
    
    ForceDataArrays force_arrays = fc_6->acquire();
    // particle 0 should be pushed right
    MY_BOOST_CHECK_CLOSE(force_arrays.fx[0], 2.224298403625553*0.8, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fy[0], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[0], tol_small);
    MY_BOOST_CHECK_CLOSE(force_arrays.virial[0], 0.296573120483407*0.8, tol);
    
    // particle 1 should be pushed left
    MY_BOOST_CHECK_CLOSE(force_arrays.fx[1], -2.224298403625553*0.8, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fy[1], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[1], tol_small);
    MY_BOOST_CHECK_CLOSE(force_arrays.virial[1], 0.296573120483407*0.8, tol);
    
    // particle 2 should be pushed up
    MY_BOOST_CHECK_CLOSE(force_arrays.fy[2], 3.336447605438329*0.8, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fx[2], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[2], tol_small);
    MY_BOOST_CHECK_CLOSE(force_arrays.virial[2], 0.444859680725111*0.8, tol);
    
    // particle 3 should be pushed down
    MY_BOOST_CHECK_CLOSE(force_arrays.fy[3], -3.336447605438329*0.8, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fx[3], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[3], tol_small);
    MY_BOOST_CHECK_CLOSE(force_arrays.virial[3], 0.444859680725111*0.8, tol);
    
    // particle 4 should be pushed forward
    MY_BOOST_CHECK_CLOSE(force_arrays.fz[4], 5.560746009063882*0.8, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fx[4], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.fy[4], tol_small);
    MY_BOOST_CHECK_CLOSE(force_arrays.virial[4], 0.741432801208518*0.8, tol);
    
    // particle 3 should be pushed back
    MY_BOOST_CHECK_CLOSE(force_arrays.fz[5], -5.560746009063882*0.8, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fx[5], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.fy[5], tol_small);
    MY_BOOST_CHECK_CLOSE(force_arrays.virial[5], 0.741432801208518*0.8, tol);
    }

//! Unit test a comparison between 2 LJForceComputes on a "real" system
void gauss_force_comparison_test(gaussforce_creator gauss_creator1,
                                 gaussforce_creator gauss_creator2,
                                 boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    const unsigned int N = 5000;
    
    // create a random particle system to sum forces on
    RandomInitializer rand_init(N, Scalar(0.2), Scalar(0.9), "A");
    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(rand_init, exec_conf));
    shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    
    shared_ptr<NeighborListBinned> nlist(new NeighborListBinned(sysdef, Scalar(3.0), Scalar(0.8)));
    
    shared_ptr<PotentialPairGauss> fc1 = gauss_creator1(sysdef, nlist);
    shared_ptr<PotentialPairGauss> fc2 = gauss_creator2(sysdef, nlist);
    fc1->setRcut(0, 0, Scalar(3.0));
    fc2->setRcut(0, 0, Scalar(3.0));
    
    // setup some values for epsilon and sigma
    Scalar epsilon = Scalar(1.0);
    Scalar sigma = Scalar(1.2);
    
    // specify the force parameters
    fc1->setParams(0,0,make_scalar2(epsilon,sigma));
    fc2->setParams(0,0,make_scalar2(epsilon,sigma));
    
    // compute the forces
    fc1->compute(0);
    fc2->compute(0);
    
    // verify that the forces are identical (within roundoff errors)
    ForceDataArrays force_arrays1 = fc1->acquire();
    ForceDataArrays force_arrays2 = fc2->acquire();
    
    // compare average deviation between the two computes
    double deltaf2 = 0.0;
    double deltape2 = 0.0;
    double deltav2 = 0.0;
        
    for (unsigned int i = 0; i < N; i++)
        {
        deltaf2 += double(force_arrays1.fx[i] - force_arrays2.fx[i]) * double(force_arrays1.fx[i] - force_arrays2.fx[i]);
        deltaf2 += double(force_arrays1.fy[i] - force_arrays2.fy[i]) * double(force_arrays1.fy[i] - force_arrays2.fy[i]);
        deltaf2 += double(force_arrays1.fz[i] - force_arrays2.fz[i]) * double(force_arrays1.fz[i] - force_arrays2.fz[i]);
        deltape2 += double(force_arrays1.pe[i] - force_arrays2.pe[i]) * double(force_arrays1.pe[i] - force_arrays2.pe[i]);
        deltav2 += double(force_arrays1.virial[i] - force_arrays2.virial[i]) * double(force_arrays1.virial[i] - force_arrays2.virial[i]);

        // also check that each individual calculation is somewhat close
        BOOST_CHECK_CLOSE(force_arrays1.fx[i], force_arrays2.fx[i], loose_tol);
        BOOST_CHECK_CLOSE(force_arrays1.fy[i], force_arrays2.fy[i], loose_tol);
        BOOST_CHECK_CLOSE(force_arrays1.fz[i], force_arrays2.fz[i], loose_tol);
        BOOST_CHECK_CLOSE(force_arrays1.pe[i], force_arrays2.pe[i], loose_tol);
        BOOST_CHECK_CLOSE(force_arrays1.virial[i], force_arrays2.virial[i], loose_tol);
        }
    deltaf2 /= double(pdata->getN());
    deltape2 /= double(pdata->getN());
    deltav2 /= double(pdata->getN());
    BOOST_CHECK_SMALL(deltaf2, double(tol_small));
    BOOST_CHECK_SMALL(deltape2, double(tol_small));
    BOOST_CHECK_SMALL(deltav2, double(tol_small));
    }

//! Test the ability of the gauss force compute to compute forces with different shift modes
void gauss_force_shift_test(gaussforce_creator gauss_creator, boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // this 2-particle test is just to get a plot of the potential and force vs r cut
    shared_ptr<SystemDefinition> sysdef_2(new SystemDefinition(2, BoxDim(1000.0), 1, 0, 0, 0, 0, exec_conf));
    shared_ptr<ParticleData> pdata_2 = sysdef_2->getParticleData();
    
    ParticleDataArrays arrays = pdata_2->acquireReadWrite();
    arrays.x[0] = arrays.y[0] = arrays.z[0] = 0.0;
    arrays.x[1] = Scalar(2.8); arrays.y[1] = arrays.z[1] = 0.0;
    pdata_2->release();
    shared_ptr<NeighborList> nlist_2(new NeighborList(sysdef_2, Scalar(3.0), Scalar(0.8)));
    shared_ptr<PotentialPairGauss> fc_no_shift = gauss_creator(sysdef_2, nlist_2);
    fc_no_shift->setShiftMode(PotentialPairGauss::no_shift);
    fc_no_shift->setRcut(0, 0, Scalar(3.0));
    shared_ptr<PotentialPairGauss> fc_shift = gauss_creator(sysdef_2, nlist_2);
    fc_shift->setShiftMode(PotentialPairGauss::shift);
    fc_shift->setRcut(0, 0, Scalar(3.0));

    nlist_2->setStorageMode(NeighborList::full);
    
    // setup a standard epsilon and sigma
    Scalar epsilon = Scalar(1.0);
    Scalar sigma = Scalar(1.0);
    fc_no_shift->setParams(0,0,make_scalar2(epsilon,sigma));
    fc_shift->setParams(0,0,make_scalar2(epsilon,sigma));
    
    fc_no_shift->compute(0);
    fc_shift->compute(0);
    
    ForceDataArrays force_arrays_no_shift = fc_no_shift->acquire();
    
    MY_BOOST_CHECK_CLOSE(force_arrays_no_shift.fx[0], -0.055555065284237, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays_no_shift.pe[0], 0.019841094744370/2.0, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays_no_shift.fx[1], 0.055555065284237, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays_no_shift.pe[1], 0.019841094744370/2.0, tol);
    
    ForceDataArrays force_arrays_shift = fc_shift->acquire();

    // shifted just has pe shifted by a given amount
    MY_BOOST_CHECK_CLOSE(force_arrays_shift.fx[0], -0.055555065284237, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays_shift.pe[0], 0.008732098206128/2.0, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays_shift.fx[1], 0.055555065284237, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays_shift.pe[1], 0.008732098206128/2.0, tol);
    
    // check once again to verify that nothing fish happens past r_cut
    arrays = pdata_2->acquireReadWrite();
    arrays.x[0] = arrays.y[0] = arrays.z[0] = 0.0;
    arrays.x[1] = Scalar(3.1); arrays.y[1] = arrays.z[1] = 0.0;
    pdata_2->release();
    
    fc_no_shift->compute(2);
    fc_shift->compute(2);
    
    force_arrays_no_shift = fc_no_shift->acquire();
    force_arrays_shift = fc_shift->acquire();
    
    MY_BOOST_CHECK_SMALL(force_arrays_no_shift.fx[0], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays_no_shift.pe[0], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays_no_shift.fx[1], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays_no_shift.pe[1], tol_small);
    
    // shifted just has pe shifted by a given amount
    MY_BOOST_CHECK_SMALL(force_arrays_shift.fx[0], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays_shift.pe[0], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays_shift.fx[1], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays_shift.pe[1], tol_small);
    }

//! LJForceCompute creator for unit tests
shared_ptr<PotentialPairGauss> base_class_gauss_creator(shared_ptr<SystemDefinition> sysdef,
                                                        shared_ptr<NeighborList> nlist)
    {
    return shared_ptr<PotentialPairGauss>(new PotentialPairGauss(sysdef, nlist));
    }

#ifdef ENABLE_CUDA
//! PotentialPairGaussGPU creator for unit tests
shared_ptr<PotentialPairGaussGPU> gpu_gauss_creator(shared_ptr<SystemDefinition> sysdef,
                                                    shared_ptr<NeighborList> nlist)
    {
    nlist->setStorageMode(NeighborList::full);
    shared_ptr<PotentialPairGaussGPU> gauss(new PotentialPairGaussGPU(sysdef, nlist));
    // the default block size kills valgrind :) reduce it
    gauss->setBlockSize(64);
    return gauss;
    }
#endif

//! boost test case for particle test on CPU
BOOST_AUTO_TEST_CASE( GaussForce_particle )
    {
    gaussforce_creator gauss_creator_base = bind(base_class_gauss_creator, _1, _2);
    gauss_force_particle_test(gauss_creator_base, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

//! boost test case for periodic test on CPU
BOOST_AUTO_TEST_CASE( GaussForce_periodic )
    {
    gaussforce_creator gauss_creator_base = bind(base_class_gauss_creator, _1, _2);
    gauss_force_periodic_test(gauss_creator_base, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

//! boost test case for particle test on CPU
BOOST_AUTO_TEST_CASE( GaussForce_shift )
    {
    gaussforce_creator gauss_creator_base = bind(base_class_gauss_creator, _1, _2);
    gauss_force_shift_test(gauss_creator_base, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

# ifdef ENABLE_CUDA
//! boost test case for particle test on GPU
BOOST_AUTO_TEST_CASE( GaussForceGPU_particle )
    {
    gaussforce_creator gauss_creator_gpu = bind(gpu_gauss_creator, _1, _2);
    gauss_force_particle_test(gauss_creator_gpu, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! boost test case for periodic test on the GPU
BOOST_AUTO_TEST_CASE( GaussForceGPU_periodic )
    {
    gaussforce_creator gauss_creator_gpu = bind(gpu_gauss_creator, _1, _2);
    gauss_force_periodic_test(gauss_creator_gpu, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! boost test case for shift test on GPU
BOOST_AUTO_TEST_CASE( GaussForceGPU_shift )
    {
    gaussforce_creator gauss_creator_gpu = bind(gpu_gauss_creator, _1, _2);
    gauss_force_shift_test(gauss_creator_gpu, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! boost test case for comparing GPU output to base class output
BOOST_AUTO_TEST_CASE( GaussForceGPU_compare )
    {
    gaussforce_creator gauss_creator_gpu = bind(gpu_gauss_creator, _1, _2);
    gaussforce_creator gauss_creator_base = bind(base_class_gauss_creator, _1, _2);
    gauss_force_comparison_test(gauss_creator_base, gauss_creator_gpu, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

#endif

#ifdef WIN32
#pragma warning( pop )
#endif

