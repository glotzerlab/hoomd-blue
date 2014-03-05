/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2014 The Regents of
the University of Michigan All rights reserved.

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

#include "AllBondPotentials.h"
#include "ConstForceCompute.h"
#include "SnapshotSystemData.h"

#include "Initializers.h"

using namespace std;
using namespace boost;

/*! \file harmonic_bond_force_test.cc
    \brief Implements unit tests for PotentialBondHarmonic and
           PotentialBondHarmonicGPU
    \ingroup unit_tests
*/

//! Name the boost unit test module
#define BOOST_TEST_MODULE BondForceTests
#include "boost_utf_configure.h"

//! Typedef to make using the boost::function factory easier
typedef boost::function<boost::shared_ptr<PotentialBondHarmonic>  (boost::shared_ptr<SystemDefinition> sysdef)> bondforce_creator;

//! Perform some simple functionality tests of any BondForceCompute
void bond_force_basic_tests(bondforce_creator bf_creator, boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    /////////////////////////////////////////////////////////
    // start with the simplest possible test: 2 particles in a huge box with only one bond type
    boost::shared_ptr<SystemDefinition> sysdef_2(new SystemDefinition(2, BoxDim(1000.0), 1, 1, 0, 0, 0, exec_conf));
    boost::shared_ptr<ParticleData> pdata_2 = sysdef_2->getParticleData();

    pdata_2->setPosition(0,make_scalar3(0.0,0.0,0.0));
    pdata_2->setPosition(1,make_scalar3(0.9,0.0,0.0));
    pdata_2->setFlags(~PDataFlags(0));

    // create the bond force compute to check
    boost::shared_ptr<PotentialBondHarmonic> fc_2 = bf_creator(sysdef_2);
    fc_2->setParams(0, make_scalar2(1.5, 0.75));

    // compute the force and check the results
    fc_2->compute(0);
    GPUArray<Scalar4>& force_array_1 =  fc_2->getForceArray();
    GPUArray<Scalar>& virial_array_1 =  fc_2->getVirialArray();

    {
    unsigned int pitch = virial_array_1.getPitch();
    ArrayHandle<Scalar4> h_force_1(force_array_1,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_1(virial_array_1,access_location::host,access_mode::read);
    // check that the force is correct, it should be 0 since we haven't created any bonds yet
    MY_BOOST_CHECK_SMALL(h_force_1.data[0].x, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_1.data[0].y, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_1.data[0].z, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_1.data[0].w, tol_small);
    MY_BOOST_CHECK_SMALL(h_virial_1.data[0*pitch+0], tol_small);
    MY_BOOST_CHECK_SMALL(h_virial_1.data[1*pitch+0], tol_small);
    MY_BOOST_CHECK_SMALL(h_virial_1.data[2*pitch+0], tol_small);
    MY_BOOST_CHECK_SMALL(h_virial_1.data[3*pitch+0], tol_small);
    MY_BOOST_CHECK_SMALL(h_virial_1.data[4*pitch+0], tol_small);
    MY_BOOST_CHECK_SMALL(h_virial_1.data[5*pitch+0], tol_small);
    }

    // add a bond and check again
    sysdef_2->getBondData()->addBondedGroup(Bond(0, 0,1));
    fc_2->compute(1);

    {
    // this time there should be a force
    GPUArray<Scalar4>& force_array_2 =  fc_2->getForceArray();
    GPUArray<Scalar>& virial_array_2 =  fc_2->getVirialArray();
    unsigned int pitch = virial_array_2.getPitch();
    ArrayHandle<Scalar4> h_force_2(force_array_2,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_2(virial_array_2,access_location::host,access_mode::read);
    MY_BOOST_CHECK_CLOSE(h_force_2.data[0].x, 0.225, tol);
    MY_BOOST_CHECK_SMALL(h_force_2.data[0].y, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_2.data[0].z, tol_small);
    MY_BOOST_CHECK_CLOSE(h_force_2.data[0].w, 0.0084375, tol);
    MY_BOOST_CHECK_CLOSE(Scalar(1./3.)*(h_virial_2.data[0*pitch+0]
                                       +h_virial_2.data[3*pitch+0]
                                       +h_virial_2.data[5*pitch+0]), -0.03375, tol);

    // check that the two forces are negatives of each other
    MY_BOOST_CHECK_CLOSE(h_force_2.data[0].x, -h_force_2.data[1].x, tol);
    MY_BOOST_CHECK_CLOSE(h_force_2.data[0].y, -h_force_2.data[1].y, tol);
    MY_BOOST_CHECK_CLOSE(h_force_2.data[0].z, -h_force_2.data[1].z, tol);
    MY_BOOST_CHECK_CLOSE(h_force_2.data[0].w, h_force_2.data[1].w, tol);
    MY_BOOST_CHECK_CLOSE(Scalar(1./3.)*(h_virial_2.data[0*pitch+1]
                                       +h_virial_2.data[3*pitch+1]
                                       +h_virial_2.data[5*pitch+1]), -0.03375, tol);
    }

    // rearrange the two particles in memory and see if they are properly updated
    {
    ArrayHandle<Scalar4> h_pos(pdata_2->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_tag(pdata_2->getTags(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_rtag(pdata_2->getRTags(), access_location::host, access_mode::readwrite);

    h_pos.data[0].x = Scalar(0.9);
    h_pos.data[1].x = Scalar(0.0);
    h_tag.data[0] = 1;
    h_tag.data[1] = 0;
    h_rtag.data[0] = 1;
    h_rtag.data[1] = 0;
    }

    // notify that we made the sort
    pdata_2->notifyParticleSort();
    // recompute at the same timestep, the forces should still be updated
    fc_2->compute(1);

    {
    // this time there should be a force
    GPUArray<Scalar4>& force_array_3 =  fc_2->getForceArray();
    GPUArray<Scalar>& virial_array_3 =  fc_2->getVirialArray();
    ArrayHandle<Scalar4> h_force_3(force_array_3,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_3(virial_array_3,access_location::host,access_mode::read);
    MY_BOOST_CHECK_CLOSE(h_force_3.data[0].x, -0.225, tol);
    MY_BOOST_CHECK_CLOSE(h_force_3.data[1].x, 0.225, tol);
    }

    // check r=r_0 behavior
    pdata_2->setPosition(0,make_scalar3(0.0,0.0,0.0));
    pdata_2->setPosition(1,make_scalar3(0.75,0.0,0.0));

    fc_2->compute(2);

    {
    // the force should be zero
    GPUArray<Scalar4>& force_array_4 =  fc_2->getForceArray();
    GPUArray<Scalar>& virial_array_4 =  fc_2->getVirialArray();
    ArrayHandle<Scalar4> h_force_4(force_array_4,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_4(virial_array_4,access_location::host,access_mode::read);
    MY_BOOST_CHECK_SMALL(h_force_4.data[0].x, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_4.data[1].x, tol_small);
    }

    ////////////////////////////////////////////////////////////////////
    // now, lets do a more thorough test and include boundary conditions
    // there are way too many permutations to test here, so I will simply
    // test +x, -x, +y, -y, +z, and -z independantly
    // build a 6 particle system with particles across each boundary
    // also test more than one type of bond
    boost::shared_ptr<SystemDefinition> sysdef_6(new SystemDefinition(6, BoxDim(20.0, 40.0, 60.0), 1, 3, 0, 0, 0, exec_conf));
    boost::shared_ptr<ParticleData> pdata_6 = sysdef_6->getParticleData();
    pdata_6->setFlags(~PDataFlags(0));

    pdata_6->setPosition(0, make_scalar3(-9.6,0.0,0.0));
    pdata_6->setPosition(1, make_scalar3(9.6, 0.0,0.0));
    pdata_6->setPosition(2, make_scalar3(0.0,-19.6,0.0));
    pdata_6->setPosition(3, make_scalar3(0.0,19.6,0.0));
    pdata_6->setPosition(4, make_scalar3(0.0,0.0,-29.6));
    pdata_6->setPosition(5, make_scalar3(0.0,0.0,29.6));

    boost::shared_ptr<PotentialBondHarmonic> fc_6 = bf_creator(sysdef_6);
    fc_6->setParams(0, make_scalar2( 1.5, 0.75));
    fc_6->setParams(1, make_scalar2(2.0*1.5, 0.75));
    fc_6->setParams(2, make_scalar2(1.5, 0.5));

    sysdef_6->getBondData()->addBondedGroup(Bond(0, 0,1));
    sysdef_6->getBondData()->addBondedGroup(Bond(1, 2,3));
    sysdef_6->getBondData()->addBondedGroup(Bond(2, 4,5));

    fc_6->compute(0);

    {
    // check that the forces are correctly computed
    GPUArray<Scalar4>& force_array_5 =  fc_6->getForceArray();
    GPUArray<Scalar>& virial_array_5 =  fc_6->getVirialArray();
    unsigned int pitch = virial_array_5.getPitch();
    ArrayHandle<Scalar4> h_force_5(force_array_5,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_5(virial_array_5,access_location::host,access_mode::read);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[0].x, -0.075, tol);
    MY_BOOST_CHECK_SMALL(h_force_5.data[0].y, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_5.data[0].z, tol_small);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[0].w, 9.375e-4, tol);
    MY_BOOST_CHECK_CLOSE(Scalar(1./3.)*(h_virial_5.data[0*pitch+0]
                                       +h_virial_5.data[3*pitch+0]
                                       +h_virial_5.data[5*pitch+0]), -0.01, tol);

    MY_BOOST_CHECK_CLOSE(h_force_5.data[1].x, 0.075, tol);
    MY_BOOST_CHECK_SMALL(h_force_5.data[1].y, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_5.data[1].z, tol_small);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[1].w, 9.375e-4, tol);
    MY_BOOST_CHECK_CLOSE(Scalar(1./3.)*(h_virial_5.data[0*pitch+1]
                                       +h_virial_5.data[3*pitch+1]
                                       +h_virial_5.data[5*pitch+1]), -0.01, tol);

    MY_BOOST_CHECK_SMALL(h_force_5.data[2].x, tol_small);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[2].y, -0.075 * 2.0, tol);
    MY_BOOST_CHECK_SMALL(h_force_5.data[2].z, tol_small);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[2].w, 9.375e-4 * 2.0, tol);
    MY_BOOST_CHECK_CLOSE(Scalar(1./3.)*(h_virial_5.data[0*pitch+2]
                                       +h_virial_5.data[3*pitch+2]
                                       +h_virial_5.data[5*pitch+2]), -0.02, tol);

    MY_BOOST_CHECK_SMALL(h_force_5.data[3].x, tol_small);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[3].y, 0.075 * 2.0, tol);
    MY_BOOST_CHECK_SMALL(h_force_5.data[3].z, tol_small);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[3].w, 9.375e-4 * 2.0, tol);
    MY_BOOST_CHECK_CLOSE(Scalar(1./3.)*(h_virial_5.data[0*pitch+3]
                                       +h_virial_5.data[3*pitch+3]
                                       +h_virial_5.data[5*pitch+3]), -0.02, tol);

    MY_BOOST_CHECK_SMALL(h_force_5.data[4].x, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_5.data[4].y, tol_small);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[4].z, -0.45, tol);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[4].w, 0.03375, tol);
    MY_BOOST_CHECK_CLOSE(Scalar(1./3.)*(h_virial_5.data[0*pitch+4]
                                       +h_virial_5.data[3*pitch+4]
                                       +h_virial_5.data[5*pitch+4]), -0.06, tol);

    MY_BOOST_CHECK_SMALL(h_force_5.data[5].x, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_5.data[5].y, tol_small);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[5].z, 0.45, tol);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[5].w, 0.03375, tol);
    MY_BOOST_CHECK_CLOSE(Scalar(1./3.)*(h_virial_5.data[0*pitch+5]
                                       +h_virial_5.data[3*pitch+5]
                                       +h_virial_5.data[5*pitch+5]), -0.06, tol);
    }

    // one more test: this one will test two things:
    // 1) That the forces are computed correctly even if the particles are rearranged in memory
    // and 2) That two forces can add to the same particle
    boost::shared_ptr<SystemDefinition> sysdef_4(new SystemDefinition(4, BoxDim(100.0, 100.0, 100.0), 1, 1, 0, 0, 0, exec_conf));
    boost::shared_ptr<ParticleData> pdata_4 = sysdef_4->getParticleData();
    pdata_4->setFlags(~PDataFlags(0));

    {
    ArrayHandle<Scalar4> h_pos(pdata_4->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_tag(pdata_4->getTags(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_rtag(pdata_4->getRTags(), access_location::host, access_mode::readwrite);

    // make a square of particles
    h_pos.data[0].x = 0.0; h_pos.data[0].y = 0.0; h_pos.data[0].z = 0.0;
    h_pos.data[1].x = 1.0; h_pos.data[1].y = 0; h_pos.data[1].z = 0.0;
    h_pos.data[2].x = 0; h_pos.data[2].y = 1.0; h_pos.data[2].z = 0.0;
    h_pos.data[3].x = 1.0; h_pos.data[3].y = 1.0; h_pos.data[3].z = 0.0;

    h_tag.data[0] = 2;
    h_tag.data[1] = 3;
    h_tag.data[2] = 0;
    h_tag.data[3] = 1;
    h_rtag.data[h_tag.data[0]] = 0;
    h_rtag.data[h_tag.data[1]] = 1;
    h_rtag.data[h_tag.data[2]] = 2;
    h_rtag.data[h_tag.data[3]] = 3;
    }

    // build the bond force compute and try it out
    boost::shared_ptr<PotentialBondHarmonic> fc_4 = bf_creator(sysdef_4);
    fc_4->setParams(0, make_scalar2(1.5, 1.75));
    // only add bonds on the left, top, and bottom of the square
    sysdef_4->getBondData()->addBondedGroup(Bond(0, 2,3));
    sysdef_4->getBondData()->addBondedGroup(Bond(0, 2,0));
    sysdef_4->getBondData()->addBondedGroup(Bond(0, 0,1));

    fc_4->compute(0);

    {
    GPUArray<Scalar4>& force_array_6 =  fc_4->getForceArray();
    GPUArray<Scalar>& virial_array_6 =  fc_4->getVirialArray();
    unsigned int pitch = virial_array_6.getPitch();
    ArrayHandle<Scalar4> h_force_6(force_array_6,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_6(virial_array_6,access_location::host,access_mode::read);
    // the right two particles shoul only have a force pulling them right
    MY_BOOST_CHECK_CLOSE(h_force_6.data[1].x, 1.125, tol);
    MY_BOOST_CHECK_SMALL(h_force_6.data[1].y, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_6.data[1].z, tol_small);
    MY_BOOST_CHECK_CLOSE(h_force_6.data[1].w, 0.2109375, tol);
    MY_BOOST_CHECK_CLOSE(Scalar(1./3.)*(h_virial_6.data[0*pitch+1]
                                       +h_virial_6.data[3*pitch+1]
                                       +h_virial_6.data[5*pitch+1]), 0.1875, tol);

    MY_BOOST_CHECK_CLOSE(h_force_6.data[3].x, 1.125, tol);
    MY_BOOST_CHECK_SMALL(h_force_6.data[3].y, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_6.data[3].z, tol_small);
    MY_BOOST_CHECK_CLOSE(h_force_6.data[3].w, 0.2109375, tol);
    MY_BOOST_CHECK_CLOSE(Scalar(1./3.)*(h_virial_6.data[0*pitch+3]
                                       +h_virial_6.data[3*pitch+3]
                                       +h_virial_6.data[5*pitch+3]), 0.1875, tol);

    // the bottom left particle should have a force pulling down and to the left
    MY_BOOST_CHECK_CLOSE(h_force_6.data[0].x, -1.125, tol);
    MY_BOOST_CHECK_CLOSE(h_force_6.data[0].y, -1.125, tol);
    MY_BOOST_CHECK_SMALL(h_force_6.data[0].z, tol_small);
    MY_BOOST_CHECK_CLOSE(h_force_6.data[0].w, 0.421875, tol);
    MY_BOOST_CHECK_CLOSE(Scalar(1./3.)*(h_virial_6.data[0*pitch+0]
                                       +h_virial_6.data[3*pitch+0]
                                       +h_virial_6.data[5*pitch+0]), 0.375, tol);

    // and the top left particle should have a force pulling up and to the left
    MY_BOOST_CHECK_CLOSE(h_force_6.data[2].x, -1.125, tol);
    MY_BOOST_CHECK_CLOSE(h_force_6.data[2].y, 1.125, tol);
    MY_BOOST_CHECK_SMALL(h_force_6.data[2].z, tol_small);
    MY_BOOST_CHECK_CLOSE(h_force_6.data[2].w, 0.421875, tol);
    MY_BOOST_CHECK_CLOSE(Scalar(1./3.)*(h_virial_6.data[0*pitch+2]
                                       +h_virial_6.data[3*pitch+2]
                                       +h_virial_6.data[5*pitch+2]), 0.375, tol);
    }
    }

//! Compares the output of two PotentialBondHarmonics
void bond_force_comparison_tests(bondforce_creator bf_creator1, bondforce_creator bf_creator2, boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    const unsigned int N = 1000;

    // create a particle system to sum forces on
    // just randomly place particles. We don't really care how huge the bond forces get: this is just a unit test
    RandomInitializer rand_init(N, Scalar(0.2), Scalar(0.9), "A");
    boost::shared_ptr<SnapshotSystemData> snap = rand_init.getSnapshot();
    snap->bond_data.type_mapping.push_back("A");
    boost::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));
    boost::shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    pdata->setFlags(~PDataFlags(0));

    boost::shared_ptr<PotentialBondHarmonic> fc1 = bf_creator1(sysdef);
    boost::shared_ptr<PotentialBondHarmonic> fc2 = bf_creator2(sysdef);
    fc1->setParams(0, make_scalar2(Scalar(300.0), Scalar(1.6)));
    fc2->setParams(0, make_scalar2(Scalar(300.0), Scalar(1.6)));

    // add bonds
    for (unsigned int i = 0; i < N-1; i++)
        {
        sysdef->getBondData()->addBondedGroup(Bond(0, i, i+1));
        }

    // compute the forces
    fc1->compute(0);
    fc2->compute(0);

    // verify that the forces are identical (within roundoff errors)
    {
    GPUArray<Scalar4>& force_array_7 =  fc1->getForceArray();
    GPUArray<Scalar>& virial_array_7 =  fc1->getVirialArray();
    unsigned int pitch = virial_array_7.getPitch();
    ArrayHandle<Scalar4> h_force_7(force_array_7,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_7(virial_array_7,access_location::host,access_mode::read);
    GPUArray<Scalar4>& force_array_8 =  fc2->getForceArray();
    GPUArray<Scalar>& virial_array_8 =  fc2->getVirialArray();
    ArrayHandle<Scalar4> h_force_8(force_array_8,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_8(virial_array_8,access_location::host,access_mode::read);

    // compare average deviation between the two computes
    double deltaf2 = 0.0;
    double deltape2 = 0.0;
    double deltav2[6];
    for (unsigned int i = 0; i < 6;  i++)
        deltav2[i] = 0.0;

    for (unsigned int i = 0; i < N; i++)
        {
        deltaf2 += double(h_force_8.data[i].x - h_force_7.data[i].x) * double(h_force_8.data[i].x - h_force_7.data[i].x);
        deltaf2 += double(h_force_8.data[i].y - h_force_7.data[i].y) * double(h_force_8.data[i].y - h_force_7.data[i].y);
        deltaf2 += double(h_force_8.data[i].z - h_force_7.data[i].z) * double(h_force_8.data[i].z - h_force_7.data[i].z);
        deltape2 += double(h_force_8.data[i].w - h_force_7.data[i].w) * double(h_force_8.data[i].w - h_force_7.data[i].w);
        for (unsigned int j = 0; j < 6; j++)
            deltav2[j] += double(h_virial_8.data[j*pitch+i] - h_virial_7.data[j*pitch+i]) * double(h_virial_8.data[j*pitch+i] - h_virial_7.data[j*pitch+i]);

        // also check that each individual calculation is somewhat close
        }
    deltaf2 /= double(pdata->getN());
    deltape2 /= double(pdata->getN());
    for (unsigned int j = 0; j < 6; j++)
        deltav2[j] /= double(pdata->getN());
    BOOST_CHECK_SMALL(deltaf2, double(tol_small));
    BOOST_CHECK_SMALL(deltape2, double(tol_small));
    BOOST_CHECK_SMALL(deltav2[0], double(tol_small));
    BOOST_CHECK_SMALL(deltav2[1], double(tol_small));
    BOOST_CHECK_SMALL(deltav2[2], double(tol_small));
    BOOST_CHECK_SMALL(deltav2[3], double(tol_small));
    BOOST_CHECK_SMALL(deltav2[4], double(tol_small));
    BOOST_CHECK_SMALL(deltav2[5], double(tol_small));
    }
    }

//! Check ConstForceCompute to see that it operates properly
void const_force_test(boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // Generate a simple test particle data
    boost::shared_ptr<SystemDefinition> sysdef_2(new SystemDefinition(2, BoxDim(1000.0), 1, 0, 0, 0, 0, exec_conf));
    boost::shared_ptr<ParticleData> pdata_2 = sysdef_2->getParticleData();
    pdata_2->setFlags(~PDataFlags(0));

    pdata_2->setPosition(0,make_scalar3(0.0,0.0,0.0));
    pdata_2->setPosition(1,make_scalar3(0.9,0.0,0.0));

    // Create the ConstForceCompute and check that it works properly
    ConstForceCompute fc(sysdef_2, Scalar(-1.3), Scalar(2.5), Scalar(45.67));
    {
    GPUArray<Scalar4>& force_array_9 =  fc.getForceArray();
    GPUArray<Scalar>& virial_array_9 =  fc.getVirialArray();
    unsigned int pitch = virial_array_9.getPitch();
    ArrayHandle<Scalar4> h_force_9(force_array_9,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_9(virial_array_9,access_location::host,access_mode::read);
    MY_BOOST_CHECK_CLOSE(h_force_9.data[0].x, -1.3, tol);
    MY_BOOST_CHECK_CLOSE(h_force_9.data[0].y, 2.5, tol);
    MY_BOOST_CHECK_CLOSE(h_force_9.data[0].z, 45.67, tol);
    MY_BOOST_CHECK_SMALL(h_force_9.data[0].w, tol_small);
    MY_BOOST_CHECK_SMALL(h_virial_9.data[0*pitch+0]
                        +h_virial_9.data[3*pitch+0]
                        +h_virial_9.data[5*pitch+0], tol_small);

    MY_BOOST_CHECK_CLOSE(h_force_9.data[1].x, -1.3, tol);
    MY_BOOST_CHECK_CLOSE(h_force_9.data[1].y, 2.5, tol);
    MY_BOOST_CHECK_CLOSE(h_force_9.data[1].z, 45.67, tol);
    MY_BOOST_CHECK_SMALL(h_force_9.data[1].w, tol_small);
    MY_BOOST_CHECK_SMALL(h_virial_9.data[0*pitch+1]
                        +h_virial_9.data[3*pitch+1]
                        +h_virial_9.data[5*pitch+1], tol_small);
    }

    // check the setforce method
    fc.setForce(Scalar(67.54), Scalar(22.1), Scalar(-1.4));
    {
    GPUArray<Scalar4>& force_array_10 =  fc.getForceArray();
    GPUArray<Scalar>& virial_array_10 =  fc.getVirialArray();
    unsigned int pitch = virial_array_10.getPitch();
    ArrayHandle<Scalar4> h_force_10(force_array_10,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_10(virial_array_10,access_location::host,access_mode::read);
    MY_BOOST_CHECK_CLOSE(h_force_10.data[0].x, 67.54, tol);
    MY_BOOST_CHECK_CLOSE(h_force_10.data[0].y, 22.1, tol);
    MY_BOOST_CHECK_CLOSE(h_force_10.data[0].z, -1.4, tol);
    MY_BOOST_CHECK_SMALL(h_force_10.data[0].w, tol_small);
    MY_BOOST_CHECK_SMALL(h_virial_10.data[0*pitch+0]
                        +h_virial_10.data[3*pitch+0]
                        +h_virial_10.data[5*pitch+0], tol_small);

    MY_BOOST_CHECK_CLOSE(h_force_10.data[1].x, 67.54, tol);
    MY_BOOST_CHECK_CLOSE(h_force_10.data[1].y, 22.1, tol);
    MY_BOOST_CHECK_CLOSE(h_force_10.data[1].z, -1.4, tol);
    MY_BOOST_CHECK_SMALL(h_force_10.data[1].w, tol_small);
    MY_BOOST_CHECK_SMALL(h_virial_10.data[0*pitch+1]
                        +h_virial_10.data[3*pitch+1]
                        +h_virial_10.data[5*pitch+1], tol_small);
    }
    }

//! PotentialBondHarmonic creator for bond_force_basic_tests()
boost::shared_ptr<PotentialBondHarmonic> base_class_bf_creator(boost::shared_ptr<SystemDefinition> sysdef)
    {
    return boost::shared_ptr<PotentialBondHarmonic>(new PotentialBondHarmonic(sysdef));
    }

#ifdef ENABLE_CUDA
//! PotentialBondHarmonic creator for bond_force_basic_tests()
boost::shared_ptr<PotentialBondHarmonic> gpu_bf_creator(boost::shared_ptr<SystemDefinition> sysdef)
    {
    return boost::shared_ptr<PotentialBondHarmonic>(new PotentialBondHarmonicGPU(sysdef));
    }
#endif

//! boost test case for bond forces on the CPU
BOOST_AUTO_TEST_CASE( PotentialBondHarmonic_basic )
    {
    bondforce_creator bf_creator = bind(base_class_bf_creator, _1);
    bond_force_basic_tests(bf_creator, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

#ifdef ENABLE_CUDA
//! boost test case for bond forces on the GPU
BOOST_AUTO_TEST_CASE( PotentialBondHarmonicGPU_basic )
    {
    bondforce_creator bf_creator = bind(gpu_bf_creator, _1);
    bond_force_basic_tests(bf_creator, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! boost test case for comparing bond GPU and CPU BondForceComputes
BOOST_AUTO_TEST_CASE( PotentialBondHarmonicGPU_compare )
    {
    bondforce_creator bf_creator_gpu = bind(gpu_bf_creator, _1);
    bondforce_creator bf_creator = bind(base_class_bf_creator, _1);
    bond_force_comparison_tests(bf_creator, bf_creator_gpu, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

#endif

//! boost test case for constant forces
BOOST_AUTO_TEST_CASE( ConstForceCompute_basic )
    {
    const_force_test(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

#ifdef WIN32
#pragma warning( pop )
#endif
