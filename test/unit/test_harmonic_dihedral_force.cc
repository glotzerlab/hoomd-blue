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

// Maintainer: akohlmey

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <iostream>

#include <boost/bind.hpp>
#include <boost/function.hpp>

#include "HarmonicDihedralForceCompute.h"
#include "ConstForceCompute.h"
#ifdef ENABLE_CUDA
#include "HarmonicDihedralForceComputeGPU.h"
#endif

#include <stdio.h>

#include "Initializers.h"

using namespace std;
using namespace boost;

//! Name the boost unit test module
#define BOOST_TEST_MODULE DihedralForceTests
#include "boost_utf_configure.h"

//! Typedef to make using the boost::function factory easier
typedef boost::function<shared_ptr<HarmonicDihedralForceCompute>  (shared_ptr<SystemDefinition> sysdef)> dihedralforce_creator;

//! Perform some simple functionality tests of any BondForceCompute
void dihedral_force_basic_tests(dihedralforce_creator tf_creator, boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    /////////////////////////////////////////////////////////
    // start with the simplest possible test: 4 particles in a huge box with only one dihedral type !!!! NO DIHEDRALS
    shared_ptr<SystemDefinition> sysdef_4(new SystemDefinition(4, BoxDim(1000.0), 1, 0, 0, 1, 0, exec_conf));
    shared_ptr<ParticleData> pdata_4 = sysdef_4->getParticleData();
    
    ParticleDataArrays arrays = pdata_4->acquireReadWrite();
    arrays.x[0] = Scalar(10.0); // put atom a at (10,1,2)
    arrays.y[0] = Scalar(1.0);
    arrays.z[0] = Scalar(2.0);
    
    arrays.x[1] = arrays.y[1] = arrays.z[1] = Scalar(1.0); // put atom b at (1,1,1)
    
    
    arrays.x[2] = Scalar(6.0); // put atom c at (6,-7,8)
    arrays.y[2] = Scalar(-7.0);
    arrays.z[2] = Scalar(8.0);
    
    arrays.x[3] = Scalar(9.0); // put atom d at (9,50,11)
    arrays.y[3] = Scalar(50.0);
    arrays.z[3] = Scalar(11.0);
    
    /*
        printf(" Particle 1: x = %f  y = %f  z = %f \n", arrays.x[0], arrays.y[0], arrays.z[0]);
        printf(" Particle 2: x = %f  y = %f  z = %f \n", arrays.x[1], arrays.y[1], arrays.z[1]);
        printf(" Particle 3: x = %f  y = %f  z = %f \n", arrays.x[2], arrays.y[2], arrays.z[2]);
        printf(" Particle 4: x = %f  y = %f  z = %f \n", arrays.x[3], arrays.y[3], arrays.z[3]);
        printf("\n");
    */
    pdata_4->release();
    
    // create the dihedral force compute to check
    shared_ptr<HarmonicDihedralForceCompute> fc_4 = tf_creator(sysdef_4);
    fc_4->setParams(0, Scalar(30.0), -1, 3); // type=0, K=30.0,sign=-1,multiplicity=3
    
    // compute the force and check the results
    fc_4->compute(0);
    ForceDataArrays force_arrays = fc_4->acquire();
    
    // check that the force is correct, it should be 0 since we haven't created any dihedrals yet
    MY_BOOST_CHECK_SMALL(force_arrays.fx[0], tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fy[0], tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[0], tol);
    MY_BOOST_CHECK_SMALL(force_arrays.pe[0], tol);
    MY_BOOST_CHECK_SMALL(force_arrays.virial[0], tol);
    
    // add an dihedrals and check again
    sysdef_4->getDihedralData()->addDihedral(Dihedral(0,0,1,2,3)); // add type 0 dihedral bewtween atoms 0-1-2-3
    fc_4->compute(1);
    
    
    // this time there should be a force
    force_arrays = fc_4->acquire();
    MY_BOOST_CHECK_CLOSE(force_arrays.fx[0], -0.5*0.118121, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.fy[0], 0.5*0.856380, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.fz[0], 0.5*1.063092, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[0], 0.5*14.945559, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.virial[0], tol);
    
    MY_BOOST_CHECK_CLOSE(force_arrays.fx[1], -0.5*0.522868, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.fy[1], -0.5*0.525225, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.fz[1], -0.5*0.226780, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[1], 0.5*14.945559, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.virial[1], tol);
    
    MY_BOOST_CHECK_CLOSE(force_arrays.fx[2], 0.5*0.445003, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.fy[2], -0.5*0.328375, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.fz[2], -0.5*0.693145, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[2], 0.5*14.945559, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.virial[2], tol);
    
    MY_BOOST_CHECK_CLOSE(force_arrays.fx[3], 0.5*0.195986, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.fy[3], -0.5*0.002780, loose_tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.fz[3], -0.5*0.143167, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[3], 0.5*14.945559, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.virial[3], tol);
    
    /*
     FORCE 1: fx = -0.118121  fy = 0.856380  fz = 1.063092
     FORCE 2: fx = -0.522868  fy = -0.525225  fz = -0.226780
     FORCE 3: fx = 0.445003  fy = -0.328375  fz = -0.693145
     FORCE 4: fx = 0.195986  fy = -0.002780  fz = -0.143167
    
    */
    
    // rearrange the two particles in memory and see if they are properly updated
    arrays = pdata_4->acquireReadWrite();
    
    arrays.x[1] = Scalar(10.0); // put atom b at (10,1,2)
    arrays.y[1] = Scalar(1.0);
    arrays.z[1] = Scalar(2.0);
    
    arrays.x[0] = arrays.y[0] = arrays.z[0] = Scalar(1.0); // put atom a at (1,1,1)
    
    arrays.tag[0] = 1;
    arrays.tag[1] = 0;
    arrays.rtag[0] = 1;
    arrays.rtag[1] = 0;
    pdata_4->release();
    
    // notify that we made the sort
    pdata_4->notifyParticleSort();
    // recompute at the same timestep, the forces should still be updated
    fc_4->compute(1);
    
    force_arrays = fc_4->acquire();
    
    MY_BOOST_CHECK_CLOSE(force_arrays.fx[1], -0.5*0.118121, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.fy[1], 0.5*0.856380, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.fz[1], 0.5*1.063092, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[1], 0.5*14.945559, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.virial[1], tol);
    
    MY_BOOST_CHECK_CLOSE(force_arrays.fx[0], -0.5*0.522868, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.fy[0], -0.5*0.525225, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.fz[0], -0.5*0.226780, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[0], 0.5*14.945559, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.virial[0], tol);
    
    
    
    ////////////////////////////////////////////////////////////////////
    // now, lets do a more thorough test and include boundary conditions
    // there are way too many permutations to test here, so I will simply
    // test +x, -x, +y, -y, +z, and -z independantly
    // build a 8 particle system with particles across each boundary
    // also test more than one type of dihedral
    shared_ptr<SystemDefinition> sysdef_8(new SystemDefinition(8, BoxDim(60.0, 70.0, 80.0), 1, 0, 0, 2, 0, exec_conf));
    shared_ptr<ParticleData> pdata_8 = sysdef_8->getParticleData();
    
    arrays = pdata_8->acquireReadWrite();
    arrays.x[0] = Scalar(-9.6); arrays.y[0] = -9.0; arrays.z[0] = 0.0;
    arrays.x[1] =  Scalar(9.6); arrays.y[1] = 1.0; arrays.z[1] = 0.0;
    arrays.x[2] = 0; arrays.y[2] = Scalar(-19.6); arrays.z[2] = 0.0;
    arrays.x[3] = 0; arrays.y[3] = Scalar(19.6); arrays.z[3] = 10.0;
    arrays.x[4] = 0; arrays.y[4] = 0; arrays.z[4] = Scalar(-29.6);
    arrays.x[5] = 0; arrays.y[5] = 0; arrays.z[5] =  Scalar(29.6);
    arrays.x[6] = 3; arrays.y[6] = 3; arrays.z[6] =  Scalar(29.6);
    arrays.x[7] = 3; arrays.y[7] = 0; arrays.z[7] =  Scalar(31.0);
    pdata_8->release();
    
    shared_ptr<HarmonicDihedralForceCompute> fc_8 = tf_creator(sysdef_8);
    fc_8->setParams(0, 50.0, -1, 3);
    fc_8->setParams(1, 30.0,  1, 4);
    
    sysdef_8->getDihedralData()->addDihedral(Dihedral(0, 0,1,2,3));
    sysdef_8->getDihedralData()->addDihedral(Dihedral(1, 4,5,6,7));
    
    fc_8->compute(0);
    // check that the forces are correctly computed
    force_arrays = fc_8->acquire();
    
    MY_BOOST_CHECK_SMALL(force_arrays.fx[0], tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fy[0], tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.fz[0], 0.5*10.504562,tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[0], 0.5*7.687906, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.virial[0], tol);
    
    MY_BOOST_CHECK_CLOSE(force_arrays.fx[1], -0.5*5.724548, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.fy[1], 0.5*2.667751,tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.fz[1], 0.5*5.650116,tol); // 5.650116
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[1], 0.5*7.687906, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.virial[1], tol);
    
    MY_BOOST_CHECK_CLOSE(force_arrays.fx[2], 0.5*10.384811,tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.fy[2], -0.5*4.839524, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.fz[2], -0.5*22.843740,tol); //
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[2], 0.5*7.687906, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.virial[2], tol);
    
    MY_BOOST_CHECK_CLOSE(force_arrays.fx[3], -0.5*4.660264,tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.fy[3], 0.5*2.171773, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.fz[3], 0.5*6.689062,tol); //
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[3], 0.5*7.687906, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.virial[3], tol);
    
    MY_BOOST_CHECK_CLOSE(force_arrays.fx[4], 0.5*2.949620,tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.fy[4], -0.5*2.949620,tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[4], tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[4], 0.5*2.318964, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.virial[4], tol);
    
    MY_BOOST_CHECK_CLOSE(force_arrays.fx[5], 0.5*3.698439,tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.fy[5], -0.5*3.698439,tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.fz[5], -0.5*14.245843,tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[5], 0.5*2.318964, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.virial[5], tol);
    
    MY_BOOST_CHECK_CLOSE(force_arrays.fx[6], 0.5*6.648058,tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.fy[6], -0.5*6.648058,tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.fz[6], -0.5*14.245843,tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[6], 0.5*2.318964, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.virial[6], tol);
    
    MY_BOOST_CHECK_CLOSE(force_arrays.fx[7], -0.5*13.296117,tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.fy[7], 0.5*13.296117,tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.fz[7], 0.5*28.491686,tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[7], 0.5*2.318964, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.virial[7], tol);
    
    
    // one more test: this one will test two things:
    // 1) That the forces are computed correctly even if the particles are rearranged in memory
    // and 2) That two forces can add to the same particle
    shared_ptr<SystemDefinition> sysdef_5(new SystemDefinition(5, BoxDim(100.0, 100.0, 100.0), 1, 0, 0, 1, 0, exec_conf));
    shared_ptr<ParticleData> pdata_5 = sysdef_5->getParticleData();
    
    arrays = pdata_5->acquireReadWrite();
    
    arrays.x[0] = Scalar(-9.6); arrays.y[0] = -9.0; arrays.z[0] = 0.0;
    arrays.x[1] =  Scalar(9.6); arrays.y[1] = 1.0; arrays.z[1] = 0.0;
    arrays.x[2] = 0; arrays.y[2] = Scalar(-19.6); arrays.z[2] = 0.0;
    arrays.x[3] = 0; arrays.y[3] = Scalar(19.6); arrays.z[3] = 10.0;
    arrays.x[4] = 0; arrays.y[4] = 0; arrays.z[4] = Scalar(-29.6);
    
    arrays.tag[0] = 2;
    arrays.tag[1] = 3;
    arrays.tag[2] = 0;
    arrays.tag[3] = 1;
    arrays.rtag[arrays.tag[0]] = 0;
    arrays.rtag[arrays.tag[1]] = 1;
    arrays.rtag[arrays.tag[2]] = 2;
    arrays.rtag[arrays.tag[3]] = 3;
    pdata_5->release();
    
    // build the dihedral force compute and try it out
    shared_ptr<HarmonicDihedralForceCompute> fc_5 = tf_creator(sysdef_5);
    fc_5->setParams(0, 15.0, -1, 4);
    
    sysdef_5->getDihedralData()->addDihedral(Dihedral(0, 0,1,2,3));
    sysdef_5->getDihedralData()->addDihedral(Dihedral(0, 1,2,3,4));
    
    fc_5->compute(0);
    force_arrays = fc_5->acquire();
    
    MY_BOOST_CHECK_CLOSE(force_arrays.fx[0], -0.5*1.400442, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.fy[0], -0.5*1.251086,tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.fz[0], 0.5*3.152951,tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[0], 0.5*7.944149, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.virial[0], tol);
    
    MY_BOOST_CHECK_CLOSE(force_arrays.fx[1], 0.5*1.719594, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.fy[1], -0.5*3.301620,tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.fz[1], 0.5*5.293722,tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[1], 0.5*7.944149, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.virial[1], tol);
    
    MY_BOOST_CHECK_CLOSE(force_arrays.fx[2], 0.5*1.153410, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.fy[2], 0.5*1.044598,tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.fz[2], -0.5*4.094823,tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[2], 0.5*5.176867, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.virial[2], tol);
    
    MY_BOOST_CHECK_CLOSE(force_arrays.fx[3], -0.5*0.581728, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.fy[3], 0.5*1.797707,tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.fz[3], -0.5*4.582985,tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[3], 0.5*7.944149, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.virial[3], tol);
    
    MY_BOOST_CHECK_CLOSE(force_arrays.fx[4], -0.5*0.890834, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.fy[4], 0.5*1.710401,tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.fz[4], 0.5*0.231135,tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[4], 0.5*2.767281, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.virial[4], tol);
    
    }





//! Compares the output of two HarmonicDihedralForceComputes
void dihedral_force_comparison_tests(dihedralforce_creator tf_creator1,
                                     dihedralforce_creator tf_creator2,
                                     boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    const unsigned int N = 1000;
    
    // create a particle system to sum forces on
    // just randomly place particles. We don't really care how huge the bond forces get: this is just a unit test
    RandomInitializer rand_init(N, Scalar(0.2), Scalar(0.9), "A");
    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(rand_init, exec_conf));
    
    shared_ptr<HarmonicDihedralForceCompute> fc1 = tf_creator1(sysdef);
    shared_ptr<HarmonicDihedralForceCompute> fc2 = tf_creator2(sysdef);
    fc1->setParams(0, Scalar(3.0), -1, 3);
    fc2->setParams(0, Scalar(3.0), -1, 3);
    
    // add dihedrals
    for (unsigned int i = 0; i < N-3; i++)
        {
        sysdef->getDihedralData()->addDihedral(Dihedral(0, i, i+1,i+2, i+3));
        }
        
    // compute the forces
    fc1->compute(0);
    fc2->compute(0);
    
    // verify that the forces are identical (within roundoff errors)
    ForceDataArrays force_arrays1 = fc1->acquire();
    ForceDataArrays force_arrays2 = fc2->acquire();
    
    Scalar rough_tol = Scalar(3.0);
    
    for (unsigned int i = 0; i < N; i++)
        {
        BOOST_CHECK_CLOSE(force_arrays1.fx[i], force_arrays2.fx[i], rough_tol);
        BOOST_CHECK_CLOSE(force_arrays1.fy[i], force_arrays2.fy[i], rough_tol);
        BOOST_CHECK_CLOSE(force_arrays1.fz[i], force_arrays2.fz[i], rough_tol);
        BOOST_CHECK_CLOSE(force_arrays1.pe[i], force_arrays2.pe[i], rough_tol);
        BOOST_CHECK_SMALL(force_arrays1.virial[i], rough_tol);
        BOOST_CHECK_SMALL(force_arrays2.virial[i], rough_tol);
        }
        
    }





//! HarmonicDihedralForceCompute creator for dihedral_force_basic_tests()
shared_ptr<HarmonicDihedralForceCompute> base_class_tf_creator(shared_ptr<SystemDefinition> sysdef)
    {
    return shared_ptr<HarmonicDihedralForceCompute>(new HarmonicDihedralForceCompute(sysdef));
    }

#ifdef ENABLE_CUDA
//! DihedralForceCompute creator for bond_force_basic_tests()
shared_ptr<HarmonicDihedralForceCompute> gpu_tf_creator(shared_ptr<SystemDefinition> sysdef)
    {
    return shared_ptr<HarmonicDihedralForceCompute>(new HarmonicDihedralForceComputeGPU(sysdef));
    }
#endif

//! boost test case for dihedral forces on the CPU
BOOST_AUTO_TEST_CASE( HarmonicDihedralForceCompute_basic )
    {
    printf(" IN BOOST_AUTO_TEST_CASE: CPU \n");
    dihedralforce_creator tf_creator = bind(base_class_tf_creator, _1);
    dihedral_force_basic_tests(tf_creator, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

#ifdef ENABLE_CUDA
//! boost test case for dihedral forces on the GPU
BOOST_AUTO_TEST_CASE( HarmonicDihedralForceComputeGPU_basic )
    {
    printf(" IN BOOST_AUTO_TEST_CASE: GPU \n");
    dihedralforce_creator tf_creator = bind(gpu_tf_creator, _1);
    dihedral_force_basic_tests(tf_creator, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! boost test case for comparing bond GPU and CPU BondForceComputes
BOOST_AUTO_TEST_CASE( HarmonicDihedralForceComputeGPU_compare )
    {
    dihedralforce_creator tf_creator_gpu = bind(gpu_tf_creator, _1);
    dihedralforce_creator tf_creator = bind(base_class_tf_creator, _1);
    dihedral_force_comparison_tests(tf_creator, tf_creator_gpu, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! boost test case for comparing calculation on the CPU to multi-gpu ones
BOOST_AUTO_TEST_CASE( HarmonicDihedralForce_MultiGPU_compare)
    {
    boost::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::GPU));
    
    dihedralforce_creator tf_creator_gpu = bind(gpu_tf_creator, _1);
    dihedralforce_creator tf_creator = bind(base_class_tf_creator, _1);
    dihedral_force_comparison_tests(tf_creator, tf_creator_gpu, exec_conf);
    }
#endif

