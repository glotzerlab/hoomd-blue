// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// this include is necessary to get MPI included before anything else to support intel MPI
#include "hoomd/ExecutionConfiguration.h"

#include <iostream>

#include <functional>

#include "hoomd/md/HarmonicImproperForceCompute.h"
#include "hoomd/ConstForceCompute.h"
#ifdef ENABLE_CUDA
#include "hoomd/md/HarmonicImproperForceComputeGPU.h"
#endif

#include <stdio.h>

#include "hoomd/Initializers.h"
#include "hoomd/SnapshotSystemData.h"

using namespace std;
using namespace std::placeholders;

#include "hoomd/test/upp11_config.h"
HOOMD_UP_MAIN();

//! Typedef to make using the std::function factory easier
typedef std::function<std::shared_ptr<HarmonicImproperForceCompute>  (std::shared_ptr<SystemDefinition> sysdef)> improperforce_creator;

//! Perform some simple functionality tests of any BondForceCompute
void improper_force_basic_tests(improperforce_creator tf_creator, std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    /////////////////////////////////////////////////////////
    // start with the simplest possible test: 4 particles in a huge box with only one improper type !!!! NO IMPROPERS
    std::shared_ptr<SystemDefinition> sysdef_4(new SystemDefinition(4, BoxDim(1000.0), 1, 0, 0, 0, 1, exec_conf));
    std::shared_ptr<ParticleData> pdata_4 = sysdef_4->getParticleData();

    {
    ArrayHandle<Scalar4> h_pos(pdata_4->getPositions(), access_location::host, access_mode::readwrite);

    h_pos.data[0].x = Scalar(10.0); // put atom a at (10,1,2)
    h_pos.data[0].y = Scalar(1.0);
    h_pos.data[0].z = Scalar(2.0);

    h_pos.data[1].x = h_pos.data[1].y = h_pos.data[1].z = Scalar(1.0); // put atom b at (1,1,1)


    h_pos.data[2].x = Scalar(6.0); // put atom c at (6,-7,8)
    h_pos.data[2].y = Scalar(-7.0);
    h_pos.data[2].z = Scalar(8.0);

    h_pos.data[3].x = Scalar(9.0); // put atom d at (9,50,11)
    h_pos.data[3].y = Scalar(50.0);
    h_pos.data[3].z = Scalar(11.0);
    }

    /*
        printf(" Particle 1: x = %f  y = %f  z = %f \n", h_pos.data[0].x, h_pos.data[0].y, h_pos.data[0].z);
        printf(" Particle 2: x = %f  y = %f  z = %f \n", h_pos.data[1].x, h_pos.data[1].y, h_pos.data[1].z);
        printf(" Particle 3: x = %f  y = %f  z = %f \n", h_pos.data[2].x, h_pos.data[2].y, h_pos.data[2].z);
        printf(" Particle 4: x = %f  y = %f  z = %f \n", h_pos.data[3].x, h_pos.data[3].y, h_pos.data[3].z);
        printf("\n");
    */

    // create the improper force compute to check
    std::shared_ptr<HarmonicImproperForceCompute> fc_4 = tf_creator(sysdef_4);
    fc_4->setParams(0, Scalar(2.0), Scalar(1.570796)); // type=0, K=2.0,chi=pi/2

    // compute the force and check the results
    fc_4->compute(0);

    {
    GlobalArray<Scalar4>& force_array_1 =  fc_4->getForceArray();
    GlobalArray<Scalar>& virial_array_1 =  fc_4->getVirialArray();
    unsigned int pitch = virial_array_1.getPitch();
    ArrayHandle<Scalar4> h_force_1(force_array_1,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_1(virial_array_1,access_location::host,access_mode::read);

    // check that the force is correct, it should be 0 since we haven't created any impropers yet
    MY_CHECK_SMALL(h_force_1.data[0].x, tol);
    MY_CHECK_SMALL(h_force_1.data[0].y, tol);
    MY_CHECK_SMALL(h_force_1.data[0].z, tol);
    MY_CHECK_SMALL(h_force_1.data[0].w, tol);
    MY_CHECK_SMALL(h_virial_1.data[0*pitch+0], tol);
    MY_CHECK_SMALL(h_virial_1.data[1*pitch+0], tol);
    MY_CHECK_SMALL(h_virial_1.data[2*pitch+0], tol);
    MY_CHECK_SMALL(h_virial_1.data[3*pitch+0], tol);
    MY_CHECK_SMALL(h_virial_1.data[4*pitch+0], tol);
    MY_CHECK_SMALL(h_virial_1.data[5*pitch+0], tol);
    }

    // add an impropers and check again
    sysdef_4->getImproperData()->addBondedGroup(Dihedral(0,0,1,2,3)); // add type 0 improper between atoms 0-1-2-3
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
    GlobalArray<Scalar4>& force_array_2 =  fc_4->getForceArray();
    GlobalArray<Scalar>& virial_array_2 =  fc_4->getVirialArray();
    unsigned int pitch = virial_array_2.getPitch();
    ArrayHandle<Scalar4> h_force_2(force_array_2,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_2(virial_array_2,access_location::host,access_mode::read);
    MY_CHECK_CLOSE(h_force_2.data[0].x, 0.5*0.0246093274, tol);
    MY_CHECK_CLOSE(h_force_2.data[0].y, -0.5*0.178418, tol);
    MY_CHECK_CLOSE(h_force_2.data[0].z, -0.5*0.221484, tol);
    MY_CHECK_CLOSE(h_force_2.data[0].w, 0.5*0.158927, tol);
    MY_CHECK_SMALL(h_virial_2.data[0*pitch+0]
                        +h_virial_2.data[3*pitch+0]
                        +h_virial_2.data[5*pitch+0], tol);

    MY_CHECK_CLOSE(h_force_2.data[1].x, 0.5*0.108934, tol);
    MY_CHECK_CLOSE(h_force_2.data[1].y, 0.5*0.109425 , tol);
    MY_CHECK_CLOSE(h_force_2.data[1].z, 0.5*0.047247, tol);
    MY_CHECK_CLOSE(h_force_2.data[1].w, 0.5*0.158927, tol);
    MY_CHECK_SMALL(h_virial_2.data[0*pitch+1]
                        +h_virial_2.data[3*pitch+1]
                        +h_virial_2.data[5*pitch+1], tol);

    MY_CHECK_CLOSE(h_force_2.data[2].x, -0.5*0.092712, tol);
    MY_CHECK_CLOSE(h_force_2.data[2].y, 0.5*0.068413, tol);
    MY_CHECK_CLOSE(h_force_2.data[2].z, 0.5*0.144409, tol);
    MY_CHECK_CLOSE(h_force_2.data[2].w, 0.5*0.158927, tol);
    MY_CHECK_SMALL(h_virial_2.data[0*pitch+2]
                        +h_virial_2.data[3*pitch+2]
                        +h_virial_2.data[5*pitch+2], tol);

    MY_CHECK_CLOSE(h_force_2.data[3].x, -0.5*0.040832, tol);
    MY_CHECK_CLOSE(h_force_2.data[3].y, 0.5*0.000579173, tol);
    MY_CHECK_CLOSE(h_force_2.data[3].z, 0.5*0.029827416, tol);
    MY_CHECK_CLOSE(h_force_2.data[3].w, 0.5*0.158927, tol);
    MY_CHECK_SMALL(h_virial_2.data[0*pitch+3]
                        +h_virial_2.data[3*pitch+3]
                        +h_virial_2.data[5*pitch+3], tol);
    }

    // rearrange the two particles in memory and see if they are properly updated
    {
    ArrayHandle<Scalar4> h_pos(pdata_4->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_tag(pdata_4->getTags(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_rtag(pdata_4->getRTags(), access_location::host, access_mode::readwrite);

    h_pos.data[1].x = Scalar(10.0); // put atom b at (10,1,2)
    h_pos.data[1].y = Scalar(1.0);
    h_pos.data[1].z = Scalar(2.0);

    h_pos.data[0].x = h_pos.data[0].y = h_pos.data[0].z = Scalar(1.0); // put atom a at (1,1,1)

    h_tag.data[0] = 1;
    h_tag.data[1] = 0;
    h_rtag.data[0] = 1;
    h_rtag.data[1] = 0;
    }

    // notify that we made the sort
    pdata_4->notifyParticleSort();
    // recompute at the same timestep, the forces should still be updated
    fc_4->compute(1);

    {
    GlobalArray<Scalar4>& force_array_3 =  fc_4->getForceArray();
    GlobalArray<Scalar>& virial_array_3 =  fc_4->getVirialArray();
    unsigned int pitch = virial_array_3.getPitch();
    ArrayHandle<Scalar4> h_force_3(force_array_3,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_3(virial_array_3,access_location::host,access_mode::read);

    MY_CHECK_CLOSE(h_force_3.data[1].x, 0.5*0.0246093274, tol);
    MY_CHECK_CLOSE(h_force_3.data[1].y, -0.5*0.178418, tol);
    MY_CHECK_CLOSE(h_force_3.data[1].z, -0.5*0.221484, tol);
    MY_CHECK_CLOSE(h_force_3.data[1].w, 0.5*0.158927, tol);
    MY_CHECK_SMALL(h_virial_3.data[0*pitch+1]
                        +h_virial_3.data[3*pitch+1]
                        +h_virial_3.data[5*pitch+1], tol);

    MY_CHECK_CLOSE(h_force_3.data[0].x, 0.5*0.108934, tol);
    MY_CHECK_CLOSE(h_force_3.data[0].y, 0.5*0.109425 , tol);
    MY_CHECK_CLOSE(h_force_3.data[0].z, 0.5*0.047247, tol);
    MY_CHECK_CLOSE(h_force_3.data[0].w, 0.5*0.158927, tol);
    MY_CHECK_SMALL(h_virial_3.data[0*pitch+0]
                        +h_virial_3.data[3*pitch+0]
                        +h_virial_3.data[5*pitch+0], tol);
    }

    ////////////////////////////////////////////////////////////////////
    // now, lets do a more thorough test and include boundary conditions
    // there are way too many permutations to test here, so I will simply
    // test +x, -x, +y, -y, +z, and -z independently
    // build a 8 particle system with particles across each boundary
    // also test more than one type of impropers
    std::shared_ptr<SystemDefinition> sysdef_8(new SystemDefinition(8, BoxDim(60.0, 70.0, 80.0), 1, 0, 0, 0, 2, exec_conf));
    std::shared_ptr<ParticleData> pdata_8 = sysdef_8->getParticleData();

    {
    ArrayHandle<Scalar4> h_pos(pdata_8->getPositions(), access_location::host, access_mode::readwrite);
    h_pos.data[0].x = Scalar(-9.6); h_pos.data[0].y = -9.0; h_pos.data[0].z = 0.0;
    h_pos.data[1].x =  Scalar(9.6); h_pos.data[1].y = 1.0; h_pos.data[1].z = 0.0;
    h_pos.data[2].x = 0; h_pos.data[2].y = Scalar(-19.6); h_pos.data[2].z = 0.0;
    h_pos.data[3].x = 0; h_pos.data[3].y = Scalar(19.6); h_pos.data[3].z = 10.0;
    h_pos.data[4].x = 0; h_pos.data[4].y = 0; h_pos.data[4].z = Scalar(-29.6);
    h_pos.data[5].x = 0; h_pos.data[5].y = 0; h_pos.data[5].z =  Scalar(29.6);
    h_pos.data[6].x = 3; h_pos.data[6].y = 3; h_pos.data[6].z =  Scalar(29.6);
    h_pos.data[7].x = 3; h_pos.data[7].y = 0; h_pos.data[7].z =  Scalar(31.0);
    }

    std::shared_ptr<HarmonicImproperForceCompute> fc_8 = tf_creator(sysdef_8);
    fc_8->setParams(0, Scalar(2.0), Scalar(1.578));
    fc_8->setParams(1, Scalar(4.0), Scalar(1.444));

    sysdef_8->getImproperData()->addBondedGroup(Dihedral(0, 0,1,2,3));
    sysdef_8->getImproperData()->addBondedGroup(Dihedral(1, 4,5,6,7));

    fc_8->compute(0);

    {
    // check that the forces are correctly computed
    GlobalArray<Scalar4>& force_array_4 =  fc_8->getForceArray();
    GlobalArray<Scalar>& virial_array_4 =  fc_8->getVirialArray();
    unsigned int pitch = virial_array_4.getPitch();
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

    MY_CHECK_SMALL(h_force_4.data[0].x, tol);
    MY_CHECK_SMALL(h_force_4.data[0].y, tol);
    MY_CHECK_CLOSE(h_force_4.data[0].z, 0.5*0.275672,tol);
    MY_CHECK_CLOSE(h_force_4.data[0].w, 0.5*0.412477, tol);
    MY_CHECK_SMALL(h_virial_4.data[0*pitch+0]
                        +h_virial_4.data[3*pitch+0]
                        +h_virial_4.data[5*pitch+0], tol);

    MY_CHECK_CLOSE(h_force_4.data[1].x, -0.5*0.150230, tol);
    MY_CHECK_CLOSE(h_force_4.data[1].y, 0.5*0.070010,tol);
    MY_CHECK_CLOSE(h_force_4.data[1].z, 0.5*0.148276,tol);
    MY_CHECK_CLOSE(h_force_4.data[1].w, 0.5*0.412477, tol);
    MY_CHECK_SMALL(h_virial_4.data[0*pitch+1]
                        +h_virial_4.data[3*pitch+1]
                        +h_virial_4.data[5*pitch+1], tol);

    MY_CHECK_CLOSE(h_force_4.data[2].x, 0.5*0.272530,tol);
    MY_CHECK_CLOSE(h_force_4.data[2].y, -0.5*0.127004, tol);
    MY_CHECK_CLOSE(h_force_4.data[2].z, -0.5*0.599490,tol);
    MY_CHECK_CLOSE(h_force_4.data[2].w, 0.5*0.412477, tol);
    MY_CHECK_SMALL(h_virial_4.data[0*pitch+2]
                        +h_virial_4.data[3*pitch+2]
                        +h_virial_4.data[5*pitch+2], tol);

    MY_CHECK_CLOSE(h_force_4.data[3].x, -0.5*0.122300,tol);
    MY_CHECK_CLOSE(h_force_4.data[3].y, 0.5*0.056994, tol);
    MY_CHECK_CLOSE(h_force_4.data[3].z, 0.5*0.175541,tol);
    MY_CHECK_CLOSE(h_force_4.data[3].w, 0.5*0.412477, tol);
    MY_CHECK_SMALL(h_virial_4.data[0*pitch+3]
                        +h_virial_4.data[3*pitch+3]
                        +h_virial_4.data[5*pitch+3], tol);

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
    MY_CHECK_CLOSE(h_force_4.data[4].x, -0.5*0.124166,tol);
    MY_CHECK_CLOSE(h_force_4.data[4].y, 0.5*0.124166,tol);
    MY_CHECK_SMALL(h_force_4.data[4].z, tol);
    MY_CHECK_CLOSE(h_force_4.data[4].w, 0.5*0.208441, tol);
    MY_CHECK_SMALL(h_virial_4.data[0*pitch+4]
                        +h_virial_4.data[3*pitch+4]
                        +h_virial_4.data[5*pitch+4], tol);

    MY_CHECK_CLOSE(h_force_4.data[5].x, -0.5*0.155688,tol);
    MY_CHECK_CLOSE(h_force_4.data[5].y, 0.5*0.155688,tol);
    MY_CHECK_CLOSE(h_force_4.data[5].z, 0.5*0.599688,tol);
    MY_CHECK_CLOSE(h_force_4.data[5].w, 0.5*0.208441, tol);
    MY_CHECK_SMALL(h_virial_4.data[0*pitch+5]
                        +h_virial_4.data[3*pitch+5]
                        +h_virial_4.data[5*pitch+5], tol);

    MY_CHECK_CLOSE(h_force_4.data[6].x, -0.5*0.279854,tol);
    MY_CHECK_CLOSE(h_force_4.data[6].y, 0.5*0.279854,tol);
    MY_CHECK_CLOSE(h_force_4.data[6].z, 0.5*0.599688,tol);
    MY_CHECK_CLOSE(h_force_4.data[6].w, 0.5*0.208441, tol);
    MY_CHECK_SMALL(h_virial_4.data[0*pitch+6]
                        +h_virial_4.data[3*pitch+6]
                        +h_virial_4.data[5*pitch+6], tol);

    MY_CHECK_CLOSE(h_force_4.data[7].x, 0.5*0.559709,tol);
    MY_CHECK_CLOSE(h_force_4.data[7].y, -0.5*0.559709,tol);
    MY_CHECK_CLOSE(h_force_4.data[7].z, -0.5*1.199376,tol);
    MY_CHECK_CLOSE(h_force_4.data[7].w, 0.5*0.208441, tol);
    MY_CHECK_SMALL(h_virial_4.data[0*pitch+7]
                        +h_virial_4.data[3*pitch+7]
                        +h_virial_4.data[5*pitch+7], tol);

    }

    // one more test: this one will test two things:
    // 1) That the forces are computed correctly even if the particles are rearranged in memory
    // and 2) That two forces can add to the same particle
    std::shared_ptr<SystemDefinition> sysdef_5(new SystemDefinition(5, BoxDim(100.0, 100.0, 100.0), 1, 0, 0, 0, 1, exec_conf));
    std::shared_ptr<ParticleData> pdata_5 = sysdef_5->getParticleData();

    {
    ArrayHandle<Scalar4> h_pos(pdata_5->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_tag(pdata_5->getTags(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_rtag(pdata_5->getRTags(), access_location::host, access_mode::readwrite);

    h_pos.data[0].x = Scalar(-9.6); h_pos.data[0].y = -9.0; h_pos.data[0].z = 0.0;
    h_pos.data[1].x =  Scalar(9.6); h_pos.data[1].y = 1.0; h_pos.data[1].z = 0.0;
    h_pos.data[2].x = 0; h_pos.data[2].y = Scalar(-19.6); h_pos.data[2].z = 0.0;
    h_pos.data[3].x = 0; h_pos.data[3].y = Scalar(19.6); h_pos.data[3].z = 10.0;
    h_pos.data[4].x = 0; h_pos.data[4].y = 0; h_pos.data[4].z = Scalar(-29.6);

    h_tag.data[0] = 2;
    h_tag.data[1] = 3;
    h_tag.data[2] = 0;
    h_tag.data[3] = 1;
    h_rtag.data[h_tag.data[0]] = 0;
    h_rtag.data[h_tag.data[1]] = 1;
    h_rtag.data[h_tag.data[2]] = 2;
    h_rtag.data[h_tag.data[3]] = 3;
    }

    // build the improper force compute and try it out
    std::shared_ptr<HarmonicImproperForceCompute> fc_5 = tf_creator(sysdef_5);
    fc_5->setParams(0, Scalar(5.0), Scalar(1.33333));

    sysdef_5->getImproperData()->addBondedGroup(Dihedral(0, 0,1,2,3));
    sysdef_5->getImproperData()->addBondedGroup(Dihedral(0, 1,2,3,4));

    fc_5->compute(0);

    {
    GlobalArray<Scalar4>& force_array_5 =  fc_5->getForceArray();
    GlobalArray<Scalar>& virial_array_5 =  fc_5->getVirialArray();
    unsigned int pitch = virial_array_5.getPitch();
    ArrayHandle<Scalar4> h_force_5(force_array_5,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_5(virial_array_5,access_location::host,access_mode::read);

    MY_CHECK_CLOSE(h_force_5.data[0].x, 0.5*0.304428, tol);
    MY_CHECK_CLOSE(h_force_5.data[0].y, 0.5*0.0141169504,loose_tol);
    MY_CHECK_CLOSE(h_force_5.data[0].z, -0.5*0.504949928,tol);
    MY_CHECK_CLOSE(h_force_5.data[0].w, 0.5*1.285859, tol);
    MY_CHECK_SMALL(h_virial_5.data[0*pitch+0]
                        +h_virial_5.data[3*pitch+0]
                        +h_virial_5.data[5*pitch+0], tol);

    MY_CHECK_CLOSE(h_force_5.data[1].x, -0.5*0.00688943266, loose_tol);
    MY_CHECK_CLOSE(h_force_5.data[1].y, 0.5*0.013229,loose_tol);
    MY_CHECK_CLOSE(h_force_5.data[1].z, -0.5*0.274493,loose_tol);
    MY_CHECK_CLOSE(h_force_5.data[1].w, 0.5*1.285859, tol);
    MY_CHECK_SMALL(h_virial_5.data[0*pitch+1]
                        +h_virial_5.data[3*pitch+1]
                        +h_virial_5.data[5*pitch+1], tol);

    /*
     FORCE 1: fx = 0.304428  fy = 0.014121  fz = -0.504956
     FORCE 2: fx = -0.006890  fy = 0.013229  fz = -0.274493
     FORCE 3: fx = -0.175244  fy = -0.158713  fz = 0.622154
     FORCE 4: fx = -0.035541  fy = -0.035200  fz = 0.134787
     FORCE 5: fx = -0.086752  fy = 0.166564  fz = 0.022509
     Energy: 1 = 1.285859  2 = 1.285859  3 = 0.888413 4 = 1.285859

     Energy: 5 = 0.397447

    */
    MY_CHECK_CLOSE(h_force_5.data[2].x, -0.5*0.175244, loose_tol);
    MY_CHECK_CLOSE(h_force_5.data[2].y, -0.5*0.158713,loose_tol);
    MY_CHECK_CLOSE(h_force_5.data[2].z, 0.5*0.622154,loose_tol);
    MY_CHECK_CLOSE(h_force_5.data[2].w, 0.5*0.888413, tol);
    MY_CHECK_SMALL(h_virial_5.data[0*pitch+2]
                        +h_virial_5.data[3*pitch+2]
                        +h_virial_5.data[5*pitch+2], tol);

    MY_CHECK_CLOSE(h_force_5.data[3].x, -0.5*0.035541, loose_tol);
    MY_CHECK_CLOSE(h_force_5.data[3].y, -0.5*0.035200,loose_tol);
    MY_CHECK_CLOSE(h_force_5.data[3].z, 0.5*0.134787,loose_tol);
    MY_CHECK_CLOSE(h_force_5.data[3].w, 0.5*1.285859, loose_tol);
    MY_CHECK_SMALL(h_virial_5.data[0*pitch+3]
                        +h_virial_5.data[3*pitch+3]
                        +h_virial_5.data[5*pitch+3], tol);

    MY_CHECK_CLOSE(h_force_5.data[4].x, -0.5*0.086752, tol);
    MY_CHECK_CLOSE(h_force_5.data[4].y, 0.5*0.166564,tol);
    MY_CHECK_CLOSE(h_force_5.data[4].z, 0.5*0.022509,loose_tol);
    MY_CHECK_CLOSE(h_force_5.data[4].w, 0.5*0.397447, tol);
    MY_CHECK_SMALL(h_virial_5.data[0*pitch+4]
                        +h_virial_5.data[3*pitch+4]
                        +h_virial_5.data[5*pitch+4], tol);
    }
    }





//! Compares the output of two HarmonicImproperForceComputes
void improper_force_comparison_tests(improperforce_creator tf_creator1,
                                     improperforce_creator tf_creator2,
                                     std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // INTERESTING NOTE: the code will depending on the number of random particles
    // even 1000 will make the code blow up, 500 is used for safety... hope it works!
    const unsigned int N = 500;

    // create a particle system to sum forces on
    // just randomly place particles. We don't really care how huge the bond forces get: this is just a unit test
    RandomInitializer rand_init(N, Scalar(0.2), Scalar(0.9), "A");
    std::shared_ptr< SnapshotSystemData<Scalar> > snap = rand_init.getSnapshot();
    snap->improper_data.type_mapping.push_back("A");
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));

    std::shared_ptr<HarmonicImproperForceCompute> fc1 = tf_creator1(sysdef);
    std::shared_ptr<HarmonicImproperForceCompute> fc2 = tf_creator2(sysdef);
    fc1->setParams(0, Scalar(2.0), Scalar(3.0));
    fc2->setParams(0, Scalar(2.0), Scalar(3.0));

    // add impropers
    for (unsigned int i = 0; i < N-3; i++)
        {
        sysdef->getImproperData()->addBondedGroup(Dihedral(0, i, i+1,i+2, i+3));
        }

    // compute the forces
    fc1->compute(0);
    fc2->compute(0);

    {
    // verify that the forces are identical (within roundoff errors)
    GlobalArray<Scalar4>& force_array_6 =  fc1->getForceArray();
    GlobalArray<Scalar>& virial_array_6 =  fc1->getVirialArray();
    ArrayHandle<Scalar4> h_force_6(force_array_6,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_6(virial_array_6,access_location::host,access_mode::read);
    GlobalArray<Scalar4>& force_array_7 =  fc2->getForceArray();
    GlobalArray<Scalar>& virial_array_7 =  fc2->getVirialArray();
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
    CHECK_SMALL(deltaf2, double(tol_small));
    CHECK_SMALL(deltape2, double(tol_small));
    }
    }

//! HarmonicImproperForceCompute creator for improper_force_basic_tests()
std::shared_ptr<HarmonicImproperForceCompute> base_class_tf_creator(std::shared_ptr<SystemDefinition> sysdef)
    {
    return std::shared_ptr<HarmonicImproperForceCompute>(new HarmonicImproperForceCompute(sysdef));
    }

#ifdef ENABLE_CUDA
//! ImproperForceCompute creator for bond_force_basic_tests()
std::shared_ptr<HarmonicImproperForceCompute> gpu_tf_creator(std::shared_ptr<SystemDefinition> sysdef)
    {
    return std::shared_ptr<HarmonicImproperForceCompute>(new HarmonicImproperForceComputeGPU(sysdef));
    }
#endif

//! test case for improper forces on the CPU
UP_TEST( HarmonicImproperForceCompute_basic )
    {
    printf(" IN UP_TEST: CPU \n");
    improperforce_creator tf_creator = bind(base_class_tf_creator, _1);
    improper_force_basic_tests(tf_creator, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

#ifdef ENABLE_CUDA
//! test case for improper forces on the GPU
UP_TEST( HarmonicImproperForceComputeGPU_basic )
    {
    printf(" IN UP_TEST: GPU \n");
    improperforce_creator tf_creator = bind(gpu_tf_creator, _1);
    improper_force_basic_tests(tf_creator, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! test case for comparing bond GPU and CPU BondForceComputes
UP_TEST( HarmonicImproperForceComputeGPU_compare )
    {
    improperforce_creator tf_creator_gpu = bind(gpu_tf_creator, _1);
    improperforce_creator tf_creator = bind(base_class_tf_creator, _1);
    improper_force_comparison_tests(tf_creator, tf_creator_gpu, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

#endif
