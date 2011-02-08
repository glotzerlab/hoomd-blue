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

#include "HarmonicImproperForceCompute.h"
#include "ConstForceCompute.h"
#ifdef ENABLE_CUDA
#include "HarmonicImproperForceComputeGPU.h"
#endif

#include <stdio.h>

#include "Initializers.h"

using namespace std;
using namespace boost;

//! Name the boost unit test module
#define BOOST_TEST_MODULE ImproperForceTests
#include "boost_utf_configure.h"

//! Typedef to make using the boost::function factory easier
typedef boost::function<shared_ptr<HarmonicImproperForceCompute>  (shared_ptr<SystemDefinition> sysdef)> improperforce_creator;

//! Perform some simple functionality tests of any BondForceCompute
void improper_force_basic_tests(improperforce_creator tf_creator, boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    /////////////////////////////////////////////////////////
    // start with the simplest possible test: 4 particles in a huge box with only one improper type !!!! NO IMPROPERS
    shared_ptr<SystemDefinition> sysdef_4(new SystemDefinition(4, BoxDim(1000.0), 1, 0, 0, 0, 1, exec_conf));
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
    
    // create the improper force compute to check
    shared_ptr<HarmonicImproperForceCompute> fc_4 = tf_creator(sysdef_4);
    fc_4->setParams(0, Scalar(2.0), Scalar(1.570796)); // type=0, K=2.0,chi=pi/2
    
    // compute the force and check the results
    fc_4->compute(0);
    
    {
    GPUArray<Scalar4>& force_array_1 =  fc_4->getForceArray();
    GPUArray<Scalar>& virial_array_1 =  fc_4->getVirialArray();
    ArrayHandle<Scalar4> h_force_1(force_array_1,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_1(virial_array_1,access_location::host,access_mode::read);
    
    // check that the force is correct, it should be 0 since we haven't created any impropers yet
    MY_BOOST_CHECK_SMALL(h_force_1.data[0].x, tol);
    MY_BOOST_CHECK_SMALL(h_force_1.data[0].y, tol);
    MY_BOOST_CHECK_SMALL(h_force_1.data[0].z, tol);
    MY_BOOST_CHECK_SMALL(h_force_1.data[0].w, tol);
    MY_BOOST_CHECK_SMALL(h_virial_1.data[0], tol);
    }

    // add an impropers and check again
    sysdef_4->getImproperData()->addDihedral(Dihedral(0,0,1,2,3)); // add type 0 improper bewtween atoms 0-1-2-3
    fc_4->compute(1);
    /*
     FORCE 1: fx = 0.024609  fy = -0.178418  fz = -0.221484
     FORCE 2: fx = 0.108934  fy = 0.109425  fz = 0.047247
     FORCE 3: fx = -0.092712  fy = 0.068413  fz = 0.144409
     FORCE 4: fx = -0.040832  fy = 0.000579  fz = 0.029827
     Energy: 1 = 0.158927  2 = 0.158927  3 = 0.158927 4 = 0.158927
    
    */
    
    {
    // this time there should be a force
    GPUArray<Scalar4>& force_array_2 =  fc_4->getForceArray();
    GPUArray<Scalar>& virial_array_2 =  fc_4->getVirialArray();
    ArrayHandle<Scalar4> h_force_2(force_array_2,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_2(virial_array_2,access_location::host,access_mode::read);
    MY_BOOST_CHECK_CLOSE(h_force_2.data[0].x, 0.5*0.0246093274, tol);
    MY_BOOST_CHECK_CLOSE(h_force_2.data[0].y, -0.5*0.178418, tol);
    MY_BOOST_CHECK_CLOSE(h_force_2.data[0].z, -0.5*0.221484, tol);
    MY_BOOST_CHECK_CLOSE(h_force_2.data[0].w, 0.5*0.158927, tol);
    MY_BOOST_CHECK_SMALL(h_virial_2.data[0], tol);
    
    MY_BOOST_CHECK_CLOSE(h_force_2.data[1].x, 0.5*0.108934, tol);
    MY_BOOST_CHECK_CLOSE(h_force_2.data[1].y, 0.5*0.109425 , tol);
    MY_BOOST_CHECK_CLOSE(h_force_2.data[1].z, 0.5*0.047247, tol);
    MY_BOOST_CHECK_CLOSE(h_force_2.data[1].w, 0.5*0.158927, tol);
    MY_BOOST_CHECK_SMALL(h_virial_2.data[1], tol);
    
    MY_BOOST_CHECK_CLOSE(h_force_2.data[2].x, -0.5*0.092712, tol);
    MY_BOOST_CHECK_CLOSE(h_force_2.data[2].y, 0.5*0.068413, tol);
    MY_BOOST_CHECK_CLOSE(h_force_2.data[2].z, 0.5*0.144409, tol);
    MY_BOOST_CHECK_CLOSE(h_force_2.data[2].w, 0.5*0.158927, tol);
    MY_BOOST_CHECK_SMALL(h_virial_2.data[2], tol);
    
    MY_BOOST_CHECK_CLOSE(h_force_2.data[3].x, -0.5*0.040832, tol);
    MY_BOOST_CHECK_CLOSE(h_force_2.data[3].y, 0.5*0.000579173, tol);
    MY_BOOST_CHECK_CLOSE(h_force_2.data[3].z, 0.5*0.029827416, tol);
    MY_BOOST_CHECK_CLOSE(h_force_2.data[3].w, 0.5*0.158927, tol);
    MY_BOOST_CHECK_SMALL(h_virial_2.data[3], tol);
    }

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
    
    {
    GPUArray<Scalar4>& force_array_3 =  fc_4->getForceArray();
    GPUArray<Scalar>& virial_array_3 =  fc_4->getVirialArray();
    ArrayHandle<Scalar4> h_force_3(force_array_3,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_3(virial_array_3,access_location::host,access_mode::read);
    
    MY_BOOST_CHECK_CLOSE(h_force_3.data[1].x, 0.5*0.0246093274, tol);
    MY_BOOST_CHECK_CLOSE(h_force_3.data[1].y, -0.5*0.178418, tol);
    MY_BOOST_CHECK_CLOSE(h_force_3.data[1].z, -0.5*0.221484, tol);
    MY_BOOST_CHECK_CLOSE(h_force_3.data[1].w, 0.5*0.158927, tol);
    MY_BOOST_CHECK_SMALL(h_virial_3.data[1], tol);
    
    MY_BOOST_CHECK_CLOSE(h_force_3.data[0].x, 0.5*0.108934, tol);
    MY_BOOST_CHECK_CLOSE(h_force_3.data[0].y, 0.5*0.109425 , tol);
    MY_BOOST_CHECK_CLOSE(h_force_3.data[0].z, 0.5*0.047247, tol);
    MY_BOOST_CHECK_CLOSE(h_force_3.data[0].w, 0.5*0.158927, tol);
    MY_BOOST_CHECK_SMALL(h_virial_3.data[0], tol);
    }
    
    ////////////////////////////////////////////////////////////////////
    // now, lets do a more thorough test and include boundary conditions
    // there are way too many permutations to test here, so I will simply
    // test +x, -x, +y, -y, +z, and -z independantly
    // build a 8 particle system with particles across each boundary
    // also test more than one type of impropers
    shared_ptr<SystemDefinition> sysdef_8(new SystemDefinition(8, BoxDim(60.0, 70.0, 80.0), 1, 0, 0, 0, 2, exec_conf));
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
    
    shared_ptr<HarmonicImproperForceCompute> fc_8 = tf_creator(sysdef_8);
    fc_8->setParams(0, Scalar(2.0), Scalar(1.578));
    fc_8->setParams(1, Scalar(4.0), Scalar(1.444));
    
    sysdef_8->getImproperData()->addDihedral(Dihedral(0, 0,1,2,3));
    sysdef_8->getImproperData()->addDihedral(Dihedral(1, 4,5,6,7));
    
    fc_8->compute(0);

    {
    // check that the forces are correctly computed
    GPUArray<Scalar4>& force_array_4 =  fc_8->getForceArray();
    GPUArray<Scalar>& virial_array_4 =  fc_8->getVirialArray();
    ArrayHandle<Scalar4> h_force_4(force_array_4,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_4(virial_array_4,access_location::host,access_mode::read);
    /*
     FORCE 1: fx = -0.000000  fy = 0.000000  fz = 0.275672
     FORCE 2: fx = -0.150230  fy = 0.070010  fz = 0.148276
     FORCE 3: fx = 0.272530  fy = -0.127004  fz = -0.599490
     FORCE 4: fx = -0.122300  fy = 0.056994  fz = 0.175541
     Energy: 1 = 0.412477  2 = 0.412477  3 = 0.412477 4 = 0.412477
    
     Virial: 1 = -0.000001  2 = -0.000001  3 = -0.000001 4 = -0.000001
    */
    
    MY_BOOST_CHECK_SMALL(h_force_4.data[0].x, tol);
    MY_BOOST_CHECK_SMALL(h_force_4.data[0].y, tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[0].z, 0.5*0.275672,tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[0].w, 0.5*0.412477, tol);
    MY_BOOST_CHECK_SMALL(h_virial_4.data[0], tol);
    
    MY_BOOST_CHECK_CLOSE(h_force_4.data[1].x, -0.5*0.150230, tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[1].y, 0.5*0.070010,tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[1].z, 0.5*0.148276,tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[1].w, 0.5*0.412477, tol);
    MY_BOOST_CHECK_SMALL(h_virial_4.data[1], tol);
    
    MY_BOOST_CHECK_CLOSE(h_force_4.data[2].x, 0.5*0.272530,tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[2].y, -0.5*0.127004, tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[2].z, -0.5*0.599490,tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[2].w, 0.5*0.412477, tol);
    MY_BOOST_CHECK_SMALL(h_virial_4.data[2], tol);
    
    MY_BOOST_CHECK_CLOSE(h_force_4.data[3].x, -0.5*0.122300,tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[3].y, 0.5*0.056994, tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[3].z, 0.5*0.175541,tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[3].w, 0.5*0.412477, tol);
    MY_BOOST_CHECK_SMALL(h_virial_4.data[3], tol);
    
    /*
     FORCE 1: fx = -0.000000  fy = 0.000000  fz = 0.275672
     FORCE 2: fx = -0.150230  fy = 0.070010  fz = 0.148276
     FORCE 3: fx = 0.272530  fy = -0.127004  fz = -0.599490
     FORCE 4: fx = -0.122300  fy = 0.056994  fz = 0.175541
     FORCE 5: fx = -0.124166  fy = 0.124166  fz = -0.000000
     FORCE 6: fx = -0.155688  fy = 0.155688  fz = 0.599688
     FORCE 7: fx = -0.279854  fy = 0.279854  fz = 0.599688
     FORCE 8: fx = 0.559709  fy = -0.559709  fz = -1.199376
     Energy: 1 = 0.412477  2 = 0.412477  3 = 0.412477 4 = 0.412477
     Energy: 5 = 0.208441  6 = 0.208441  7 = 0.208441 8 = 0.208441
    
    */
    MY_BOOST_CHECK_CLOSE(h_force_4.data[4].x, -0.5*0.124166,tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[4].y, 0.5*0.124166,tol);
    MY_BOOST_CHECK_SMALL(h_force_4.data[4].z, tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[4].w, 0.5*0.208441, tol);
    MY_BOOST_CHECK_SMALL(h_virial_4.data[4], tol);
    
    MY_BOOST_CHECK_CLOSE(h_force_4.data[5].x, -0.5*0.155688,tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[5].y, 0.5*0.155688,tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[5].z, 0.5*0.599688,tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[5].w, 0.5*0.208441, tol);
    MY_BOOST_CHECK_SMALL(h_virial_4.data[5], tol);
    
    MY_BOOST_CHECK_CLOSE(h_force_4.data[6].x, -0.5*0.279854,tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[6].y, 0.5*0.279854,tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[6].z, 0.5*0.599688,tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[6].w, 0.5*0.208441, tol);
    MY_BOOST_CHECK_SMALL(h_virial_4.data[6], tol);
    
    MY_BOOST_CHECK_CLOSE(h_force_4.data[7].x, 0.5*0.559709,tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[7].y, -0.5*0.559709,tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[7].z, -0.5*1.199376,tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[7].w, 0.5*0.208441, tol);
    MY_BOOST_CHECK_SMALL(h_virial_4.data[7], tol);
    }
    
    // one more test: this one will test two things:
    // 1) That the forces are computed correctly even if the particles are rearranged in memory
    // and 2) That two forces can add to the same particle
    shared_ptr<SystemDefinition> sysdef_5(new SystemDefinition(5, BoxDim(100.0, 100.0, 100.0), 1, 0, 0, 0, 1, exec_conf));
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
    
    // build the improper force compute and try it out
    shared_ptr<HarmonicImproperForceCompute> fc_5 = tf_creator(sysdef_5);
    fc_5->setParams(0, Scalar(5.0), Scalar(1.33333));
    
    sysdef_5->getImproperData()->addDihedral(Dihedral(0, 0,1,2,3));
    sysdef_5->getImproperData()->addDihedral(Dihedral(0, 1,2,3,4));
    
    fc_5->compute(0);
    
    {
    GPUArray<Scalar4>& force_array_5 =  fc_5->getForceArray();
    GPUArray<Scalar>& virial_array_5 =  fc_5->getVirialArray();
    ArrayHandle<Scalar4> h_force_5(force_array_5,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_5(virial_array_5,access_location::host,access_mode::read);
    
    MY_BOOST_CHECK_CLOSE(h_force_5.data[0].x, 0.5*0.304428, tol);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[0].y, 0.5*0.0141169504,loose_tol);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[0].z, -0.5*0.504949928,tol);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[0].w, 0.5*1.285859, tol);
    MY_BOOST_CHECK_SMALL(h_virial_5.data[0], tol);
    
    MY_BOOST_CHECK_CLOSE(h_force_5.data[1].x, -0.5*0.00688943266, loose_tol);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[1].y, 0.5*0.013229,loose_tol);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[1].z, -0.5*0.274493,loose_tol);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[1].w, 0.5*1.285859, tol);
    MY_BOOST_CHECK_SMALL(h_virial_5.data[1], tol);
    
    /*
     FORCE 1: fx = 0.304428  fy = 0.014121  fz = -0.504956
     FORCE 2: fx = -0.006890  fy = 0.013229  fz = -0.274493
     FORCE 3: fx = -0.175244  fy = -0.158713  fz = 0.622154
     FORCE 4: fx = -0.035541  fy = -0.035200  fz = 0.134787
     FORCE 5: fx = -0.086752  fy = 0.166564  fz = 0.022509
     Energy: 1 = 1.285859  2 = 1.285859  3 = 0.888413 4 = 1.285859
    
     Energy: 5 = 0.397447
    
    */
    MY_BOOST_CHECK_CLOSE(h_force_5.data[2].x, -0.5*0.175244, loose_tol);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[2].y, -0.5*0.158713,loose_tol);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[2].z, 0.5*0.622154,loose_tol);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[2].w, 0.5*0.888413, tol);
    MY_BOOST_CHECK_SMALL(h_virial_5.data[2], tol);
    
    MY_BOOST_CHECK_CLOSE(h_force_5.data[3].x, -0.5*0.035541, loose_tol);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[3].y, -0.5*0.035200,loose_tol);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[3].z, 0.5*0.134787,loose_tol);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[3].w, 0.5*1.285859, loose_tol);
    MY_BOOST_CHECK_SMALL(h_virial_5.data[3], tol);
    
    MY_BOOST_CHECK_CLOSE(h_force_5.data[4].x, -0.5*0.086752, tol);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[4].y, 0.5*0.166564,tol);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[4].z, 0.5*0.022509,loose_tol);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[4].w, 0.5*0.397447, tol);
    MY_BOOST_CHECK_SMALL(h_virial_5.data[4], tol);
    }
    }





//! Compares the output of two HarmonicImproperForceComputes
void improper_force_comparison_tests(improperforce_creator tf_creator1,
                                     improperforce_creator tf_creator2,
                                     boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // INTERESTING NOTE: the code will depending on the number of ramdom particles
    // even 1000 will make the code blow up, 500 is used for safety... hope it works!
    const unsigned int N = 500;
    
    // create a particle system to sum forces on
    // just randomly place particles. We don't really care how huge the bond forces get: this is just a unit test
    RandomInitializer rand_init(N, Scalar(0.2), Scalar(0.9), "A");
    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(rand_init, exec_conf));
    
    shared_ptr<HarmonicImproperForceCompute> fc1 = tf_creator1(sysdef);
    shared_ptr<HarmonicImproperForceCompute> fc2 = tf_creator2(sysdef);
    fc1->setParams(0, Scalar(2.0), Scalar(3.0));
    fc2->setParams(0, Scalar(2.0), Scalar(3.0));
    
    // add improperrs
    for (unsigned int i = 0; i < N-3; i++)
        {
        sysdef->getImproperData()->addDihedral(Dihedral(0, i, i+1,i+2, i+3));
        }
        
    // compute the forces
    fc1->compute(0);
    fc2->compute(0);
    
    {
    // verify that the forces are identical (within roundoff errors)
    GPUArray<Scalar4>& force_array_6 =  fc1->getForceArray();
    GPUArray<Scalar>& virial_array_6 =  fc1->getVirialArray();
    ArrayHandle<Scalar4> h_force_6(force_array_6,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_6(virial_array_6,access_location::host,access_mode::read);
    GPUArray<Scalar4>& force_array_7 =  fc2->getForceArray();
    GPUArray<Scalar>& virial_array_7 =  fc2->getVirialArray();
    ArrayHandle<Scalar4> h_force_7(force_array_7,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_7(virial_array_7,access_location::host,access_mode::read);
    
    // compare average deviation between the two computes
    double deltaf2 = 0.0;
    double deltape2 = 0.0;
        
    for (unsigned int i = 0; i < N; i++)
        {
        deltaf2 += double(h_force_7.data[i].x - h_force_6.data[i].x) * double(h_force_7.data[i].x - h_force_6.data[i].x);
        deltaf2 += double(h_force_7.data[i].y - h_force_6.data[i].y) * double(h_force_7.data[i].y - h_force_6.data[i].y);
        deltaf2 += double(h_force_7.data[i].z - h_force_6.data[i].z) * double(h_force_7.data[i].z - h_force_6.data[i].z);
        deltape2 += double(h_force_7.data[i].w - h_force_6.data[i].w) * double(h_force_7.data[i].w - h_force_6.data[i].w);

        // also check that each individual calculation is somewhat close
        }
    deltaf2 /= double(sysdef->getParticleData()->getN());
    deltape2 /= double(sysdef->getParticleData()->getN());
    BOOST_CHECK_SMALL(deltaf2, double(tol_small));
    BOOST_CHECK_SMALL(deltape2, double(tol_small));
    }
    }

//! HarmonicImproperForceCompute creator for improper_force_basic_tests()
shared_ptr<HarmonicImproperForceCompute> base_class_tf_creator(shared_ptr<SystemDefinition> sysdef)
    {
    return shared_ptr<HarmonicImproperForceCompute>(new HarmonicImproperForceCompute(sysdef));
    }

#ifdef ENABLE_CUDA
//! ImproperForceCompute creator for bond_force_basic_tests()
shared_ptr<HarmonicImproperForceCompute> gpu_tf_creator(shared_ptr<SystemDefinition> sysdef)
    {
    return shared_ptr<HarmonicImproperForceCompute>(new HarmonicImproperForceComputeGPU(sysdef));
    }
#endif

//! boost test case for improper forces on the CPU
BOOST_AUTO_TEST_CASE( HarmonicImproperForceCompute_basic )
    {
    printf(" IN BOOST_AUTO_TEST_CASE: CPU \n");
    improperforce_creator tf_creator = bind(base_class_tf_creator, _1);
    improper_force_basic_tests(tf_creator, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

#ifdef ENABLE_CUDA
//! boost test case for improper forces on the GPU
BOOST_AUTO_TEST_CASE( HarmonicImproperForceComputeGPU_basic )
    {
    printf(" IN BOOST_AUTO_TEST_CASE: GPU \n");
    improperforce_creator tf_creator = bind(gpu_tf_creator, _1);
    improper_force_basic_tests(tf_creator, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! boost test case for comparing bond GPU and CPU BondForceComputes
BOOST_AUTO_TEST_CASE( HarmonicImproperForceComputeGPU_compare )
    {
    improperforce_creator tf_creator_gpu = bind(gpu_tf_creator, _1);
    improperforce_creator tf_creator = bind(base_class_tf_creator, _1);
    improper_force_comparison_tests(tf_creator, tf_creator_gpu, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

#endif

