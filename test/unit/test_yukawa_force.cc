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

#include "BinnedNeighborList.h"
#include "Initializers.h"

#include <math.h>

using namespace std;
using namespace boost;

/*! \file yukawa_force_test.cc
    \brief Implements unit tests for PotentialPairYukawa and descendants
    \ingroup unit_tests
*/

//! Name the unit test module
#define BOOST_TEST_MODULE PotentialPairYukawaTests
#include "boost_utf_configure.h"

//! Typedef'd PotentialPairYukawa factory
typedef boost::function<shared_ptr<PotentialPairYukawa> (shared_ptr<SystemDefinition> sysdef,
                                                         shared_ptr<NeighborList> nlist)> yukawaforce_creator;

//! Test the ability of the yukawa force compute to actually calucate forces
void yukawa_force_particle_test(yukawaforce_creator yukawa_creator, boost::shared_ptr<ExecutionConfiguration> exec_conf)
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
    shared_ptr<PotentialPairYukawa> fc_3 = yukawa_creator(sysdef_3, nlist_3);
    fc_3->setRcut(0, 0, Scalar(1.3));
    
    // first test: choose a basic set of values
    Scalar epsilon = Scalar(1.15);
    Scalar kappa = Scalar(0.6);
    fc_3->setParams(0,0,make_scalar2(epsilon,kappa));
    
    // compute the forces
    fc_3->compute(0);
    
    ForceDataArrays force_arrays = fc_3->acquire();
    MY_BOOST_CHECK_CLOSE(force_arrays.fx[0], -1.009813410413, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fy[0], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[0], tol_small);
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[0], 0.63113338150813/2.0, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.virial[0], 0.16830223506884, tol);
    
    MY_BOOST_CHECK_SMALL(force_arrays.fx[1], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.fy[1], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[1], tol_small);
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[1], 0.63113338150813, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.virial[1], 0.16830223506884*2.0, tol);
    
    MY_BOOST_CHECK_CLOSE(force_arrays.fx[2], 1.009813410413, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fy[2], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[2], tol_small);
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[2], 0.63113338150813/2.0, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.virial[2], 0.16830223506884, tol);
    
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
    MY_BOOST_CHECK_CLOSE(force_arrays.fx[0], 1.009813410413, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.fx[2], -1.009813410413, tol);
    }

//! Unit test a comparison between 2 PotentialPairYukawa's on a "real" system
void yukawa_force_comparison_test(yukawaforce_creator yukawa_creator1,
                                  yukawaforce_creator yukawa_creator2,
                                  boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    const unsigned int N = 5000;
    
    // create a random particle system to sum forces on
    RandomInitializer rand_init(N, Scalar(0.1), Scalar(1.0), "A");
    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(rand_init, exec_conf));
    shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    
    shared_ptr<BinnedNeighborList> nlist(new BinnedNeighborList(sysdef, Scalar(3.0), Scalar(0.8)));
    
    shared_ptr<PotentialPairYukawa> fc1 = yukawa_creator1(sysdef, nlist);
    shared_ptr<PotentialPairYukawa> fc2 = yukawa_creator2(sysdef, nlist);
    fc1->setRcut(0, 0, Scalar(3.0));
    fc2->setRcut(0, 0, Scalar(3.0));
    
    // setup some values for epsilon and sigma
    Scalar epsilon = Scalar(5.0);
    Scalar kappa = Scalar(6.0);
    
    // specify the force parameters
    fc1->setParams(0,0,make_scalar2(epsilon,kappa));
    fc2->setParams(0,0,make_scalar2(epsilon,kappa));
    
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
    deltaf2 /= double(pdata->getN());
    deltape2 /= double(pdata->getN());
    deltav2 /= double(pdata->getN());
    BOOST_CHECK_SMALL(deltaf2, double(tol_small));
    BOOST_CHECK_SMALL(deltape2, double(tol_small));
    BOOST_CHECK_SMALL(deltav2, double(tol_small));
    }

//! PotentialPairYukawa creator for unit tests
shared_ptr<PotentialPairYukawa> base_class_yukawa_creator(shared_ptr<SystemDefinition> sysdef,
                                                          shared_ptr<NeighborList> nlist)
    {
    return shared_ptr<PotentialPairYukawa>(new PotentialPairYukawa(sysdef, nlist));
    }

#ifdef ENABLE_CUDA
//! PotentialPairYukawaGPU creator for unit tests
shared_ptr<PotentialPairYukawaGPU> gpu_yukawa_creator(shared_ptr<SystemDefinition> sysdef,
                                                      shared_ptr<NeighborList> nlist)
    {
    nlist->setStorageMode(NeighborList::full);
    shared_ptr<PotentialPairYukawaGPU> yukawa(new PotentialPairYukawaGPU(sysdef, nlist));
    // the default block size kills valgrind :) reduce it
    yukawa->setBlockSize(64);
    return yukawa;
    }
#endif

//! boost test case for particle test on CPU
BOOST_AUTO_TEST_CASE( YukawaForce_particle )
    {
    yukawaforce_creator yukawa_creator_base = bind(base_class_yukawa_creator, _1, _2);
    yukawa_force_particle_test(yukawa_creator_base, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

# ifdef ENABLE_CUDA
//! boost test case for particle test on GPU
BOOST_AUTO_TEST_CASE( YukawaForceGPU_particle )
    {
    yukawaforce_creator yukawa_creator_gpu = bind(gpu_yukawa_creator, _1, _2);
    yukawa_force_particle_test(yukawa_creator_gpu, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! boost test case for comparing GPU output to base class output
BOOST_AUTO_TEST_CASE( YukawaForceGPU_compare )
    {
    yukawaforce_creator yukawa_creator_gpu = bind(gpu_yukawa_creator, _1, _2);
    yukawaforce_creator yukawa_creator_base = bind(base_class_yukawa_creator, _1, _2);
    yukawa_force_comparison_test(yukawa_creator_base,
                                 yukawa_creator_gpu,
                                 boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

#endif

#ifdef WIN32
#pragma warning( pop )
#endif

