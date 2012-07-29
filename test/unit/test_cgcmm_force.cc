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

#include "CGCMMForceCompute.h"
#ifdef ENABLE_CUDA
#include "CGCMMForceComputeGPU.h"
#endif

#include "NeighborListBinned.h"
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
void cgcmm_force_particle124_test(cgcmmforce_creator cgcmm_creator, boost::shared_ptr<ExecutionConfiguration> exec_conf)
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

    pdata_3->setPosition(0,make_scalar3(0.0,0.0,0.0));
    pdata_3->setPosition(1,make_scalar3(pow(3.0,1.0/8.0),0.0,0.0));
    pdata_3->setPosition(2,make_scalar3(2.0*pow(3.0,1.0/8.0),0.0,0.0));

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
    
    {
    GPUArray<Scalar4>& force_array_1 =  fc_3->getForceArray();
    GPUArray<Scalar>& virial_array_1 =  fc_3->getVirialArray();
    unsigned int pitch = fc_3->getVirialArray().getPitch();
    ArrayHandle<Scalar4> h_force_1(force_array_1,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_1(virial_array_1,access_location::host,access_mode::read);
    MY_BOOST_CHECK_SMALL(h_force_1.data[0].x, tol);
    MY_BOOST_CHECK_SMALL(h_force_1.data[0].y, tol);
    MY_BOOST_CHECK_SMALL(h_force_1.data[0].z, tol);
    MY_BOOST_CHECK_CLOSE(h_force_1.data[0].w, -0.575, tol);
    MY_BOOST_CHECK_SMALL(h_virial_1.data[0*pitch+0]
                        +h_virial_1.data[3*pitch+0]
                        +h_virial_1.data[5*pitch+0], tol);
    
    MY_BOOST_CHECK_SMALL(h_force_1.data[1].x, tol);
    MY_BOOST_CHECK_SMALL(h_force_1.data[1].y, tol);
    MY_BOOST_CHECK_SMALL(h_force_1.data[1].z, tol);
    MY_BOOST_CHECK_CLOSE(h_force_1.data[1].w, -1.15, tol);
    MY_BOOST_CHECK_SMALL(h_virial_1.data[0*pitch+1]
                        +h_virial_1.data[3*pitch+1]
                        +h_virial_1.data[5*pitch+1], tol);
    
    MY_BOOST_CHECK_SMALL(h_force_1.data[2].x, tol);
    MY_BOOST_CHECK_SMALL(h_force_1.data[2].y, tol);
    MY_BOOST_CHECK_SMALL(h_force_1.data[2].z, tol);
    MY_BOOST_CHECK_CLOSE(h_force_1.data[2].w, -0.575, tol);
    MY_BOOST_CHECK_SMALL(h_virial_1.data[0*pitch+2]
                        +h_virial_1.data[3*pitch+2]
                        +h_virial_1.data[5*pitch+2], tol);
    }

    // now change sigma and alpha so we can check that it is computing the right force
    sigma = Scalar(1.2); // < bigger sigma should push particle 0 left and particle 2 right
    alpha = Scalar(0.45);
    lj1 = Scalar(2.598076) * epsilon * pow(sigma,Scalar(12.0));
    lj2 = Scalar(0.0);
    lj3 = Scalar(0.0);
    lj4 = -alpha * Scalar(2.598076) * epsilon * pow(sigma,Scalar(4.0));
    fc_3->setParams(0,0,lj1,lj2,lj3,lj4);
    fc_3->compute(1);
    
    {
    GPUArray<Scalar4>& force_array_2 =  fc_3->getForceArray();
    GPUArray<Scalar>& virial_array_2 =  fc_3->getVirialArray();
    ArrayHandle<Scalar4> h_force_2(force_array_2,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_2(virial_array_2,access_location::host,access_mode::read);
    unsigned int pitch = virial_array_2.getPitch();
    MY_BOOST_CHECK_CLOSE(h_force_2.data[0].x, -48.0146523, tol);
    MY_BOOST_CHECK_SMALL(h_force_2.data[0].y, tol);
    MY_BOOST_CHECK_SMALL(h_force_2.data[0].z, tol);
    MY_BOOST_CHECK_CLOSE(h_force_2.data[0].w, 1.758563, tol);
    MY_BOOST_CHECK_CLOSE(Scalar(1./3.)*(h_virial_2.data[0*pitch+0]
                                 +h_virial_2.data[3*pitch+0]
                                 +h_virial_2.data[5*pitch+0]), 9.18042374, tol);
    
    // center particle should still be a 0 force by symmetry
    MY_BOOST_CHECK_SMALL(h_force_2.data[1].x, tol);
    MY_BOOST_CHECK_SMALL(h_force_2.data[1].y, 1e-5);
    MY_BOOST_CHECK_SMALL(h_force_2.data[1].z, 1e-5);
    // there is still an energy and virial, though
    MY_BOOST_CHECK_CLOSE(h_force_2.data[1].w, 3.517125, tol);
    MY_BOOST_CHECK_CLOSE(Scalar(1./3.)*(h_virial_2.data[0*pitch+1]
                                 +h_virial_2.data[3*pitch+1]
                                 +h_virial_2.data[5*pitch+1]),18.3608475, tol);
    
    MY_BOOST_CHECK_CLOSE(h_force_2.data[2].x, 48.0146561, tol);
    MY_BOOST_CHECK_SMALL(h_force_2.data[2].y, tol);
    MY_BOOST_CHECK_SMALL(h_force_2.data[2].z, tol);
    MY_BOOST_CHECK_CLOSE(h_force_2.data[2].w, 1.758563, tol);
    MY_BOOST_CHECK_CLOSE(Scalar(1./3.)*(h_virial_2.data[0*pitch+2]
                                 +h_virial_2.data[3*pitch+2]
                                 +h_virial_2.data[5*pitch+2]),9.18042374, tol);
    }

    // swap the order of particles 0 ans 2 in memory to check that the force compute handles this properly
    {
    ArrayHandle<Scalar4> h_pos(pdata_3->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_tag(pdata_3->getTags(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_rtag(pdata_3->getRTags(), access_location::host, access_mode::readwrite);

    h_pos.data[2].x = h_pos.data[2].y = h_pos.data[2].z = 0.0;
    h_pos.data[0].x = Scalar(2.0*pow(3.0,1.0/8.0)); h_pos.data[0].y = h_pos.data[0].z = 0.0;

    h_tag.data[0] = 2;
    h_tag.data[2] = 0;
    h_rtag.data[0] = 2;
    h_rtag.data[2] = 0;
    }
    
    // notify the particle data that we changed the order
    pdata_3->notifyParticleSort();
    
    // recompute the forces at the same timestep, they should be updated
    fc_3->compute(1);
    
    {
    GPUArray<Scalar4>& force_array_3 =  fc_3->getForceArray();
    GPUArray<Scalar>& virial_array_3 =  fc_3->getVirialArray();
    ArrayHandle<Scalar4> h_force_3(force_array_3,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_3(virial_array_3,access_location::host,access_mode::read);
    MY_BOOST_CHECK_CLOSE(h_force_3.data[0].x, 48.0146523, tol);
    MY_BOOST_CHECK_CLOSE(h_force_3.data[2].x, -48.0146523, tol);
    }
    }

//! Test the ability of the cgcmm LJ9-6 force compute to actually calucate forces
void cgcmm_force_particle96_test(cgcmmforce_creator cgcmm_creator, boost::shared_ptr<ExecutionConfiguration> exec_conf)
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
    
    {
    ArrayHandle<Scalar4> h_pos(pdata_3->getPositions(), access_location::host, access_mode::readwrite);
    h_pos.data[0].x = h_pos.data[0].y = h_pos.data[0].z = 0.0;
    h_pos.data[1].x = Scalar(pow(1.5,1.0/3.0)); h_pos.data[1].y = h_pos.data[1].z = 0.0;
    h_pos.data[2].x = Scalar(2.0*pow(1.5,1.0/3.0)); h_pos.data[2].y = h_pos.data[2].z = 0.0;
    }
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
    
    {
    GPUArray<Scalar4>& force_array_4 =  fc_3->getForceArray();
    GPUArray<Scalar>& virial_array_4 =  fc_3->getVirialArray();
    unsigned int pitch = virial_array_4.getPitch();
    ArrayHandle<Scalar4> h_force_4(force_array_4,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_4(virial_array_4,access_location::host,access_mode::read);
    MY_BOOST_CHECK_SMALL(h_force_4.data[0].x, tol);
    MY_BOOST_CHECK_SMALL(h_force_4.data[0].y, tol);
    MY_BOOST_CHECK_SMALL(h_force_4.data[0].z, tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[0].w, -0.575, tol);
    MY_BOOST_CHECK_SMALL(h_virial_4.data[0*pitch+0]
                        +h_virial_4.data[3*pitch+0]
                        +h_virial_4.data[5*pitch+0], tol);
    
    MY_BOOST_CHECK_SMALL(h_force_4.data[1].x, tol);
    MY_BOOST_CHECK_SMALL(h_force_4.data[1].y, tol);
    MY_BOOST_CHECK_SMALL(h_force_4.data[1].z, tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[1].w, -1.15, tol);
    MY_BOOST_CHECK_SMALL(h_virial_4.data[0*pitch+1]
                        +h_virial_4.data[3*pitch+1]
                        +h_virial_4.data[5*pitch+1], tol);
    
    MY_BOOST_CHECK_SMALL(h_force_4.data[2].x, tol);
    MY_BOOST_CHECK_SMALL(h_force_4.data[2].y, tol);
    MY_BOOST_CHECK_SMALL(h_force_4.data[2].z, tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[2].w, -0.575, tol);
    MY_BOOST_CHECK_SMALL(h_virial_4.data[0*pitch+2]
                        +h_virial_4.data[3*pitch+2]
                        +h_virial_4.data[5*pitch+2], tol);

    }

    // now change sigma and alpha so we can check that it is computing the right force
    sigma = Scalar(1.2); // < bigger sigma should push particle 0 left and particle 2 right
    alpha = Scalar(0.45);
    lj1 = Scalar(0.0);
    lj2 = Scalar(6.75) * epsilon * pow(sigma,Scalar(9.0));
    lj3 = -alpha * Scalar(6.75) * epsilon * pow(sigma,Scalar(6.0));
    lj4 = Scalar(0.0);
    fc_3->setParams(0,0,lj1,lj2,lj3,lj4);
    fc_3->compute(1);
    
    {
    GPUArray<Scalar4>& force_array_5 =  fc_3->getForceArray();
    GPUArray<Scalar>& virial_array_5 =  fc_3->getVirialArray();
    unsigned int pitch = virial_array_5.getPitch();
    ArrayHandle<Scalar4> h_force_5(force_array_5,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_5(virial_array_5,access_location::host,access_mode::read);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[0].x, -69.00675, tol);
    MY_BOOST_CHECK_SMALL(h_force_5.data[0].y, tol);
    MY_BOOST_CHECK_SMALL(h_force_5.data[0].z, tol);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[0].w, 3.615877, tol);
    MY_BOOST_CHECK_CLOSE(Scalar(1./3.)*(h_virial_5.data[0*pitch+0]
                        +h_virial_5.data[3*pitch+0]
                        +h_virial_5.data[5*pitch+0]), 13.1655083,tol);
    
    // center particle should still be a 0 force by symmetry
    MY_BOOST_CHECK_SMALL(h_force_5.data[1].x, tol);
    MY_BOOST_CHECK_SMALL(h_force_5.data[1].y, 1e-5);
    MY_BOOST_CHECK_SMALL(h_force_5.data[1].z, 1e-5);
    // there is still an energy and virial, though
    MY_BOOST_CHECK_CLOSE(h_force_5.data[1].w, 7.231755, tol);
    MY_BOOST_CHECK_CLOSE(Scalar(1./3.)*(h_virial_5.data[0*pitch+1]
                        +h_virial_5.data[3*pitch+1]
                        +h_virial_5.data[5*pitch+1]), 26.3310165,tol);
    
    MY_BOOST_CHECK_CLOSE(h_force_5.data[2].x, 69.00675, tol);
    MY_BOOST_CHECK_SMALL(h_force_5.data[2].y, tol);
    MY_BOOST_CHECK_SMALL(h_force_5.data[2].z, tol);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[2].w, 3.615877, tol);
    MY_BOOST_CHECK_CLOSE(Scalar(1./3.)*(h_virial_5.data[0*pitch+2]
                        +h_virial_5.data[3*pitch+2]
                        +h_virial_5.data[5*pitch+2]), 13.1655083,tol);
    }

    // swap the order of particles 0 ans 2 in memory to check that the force compute handles this properly
    {
    ArrayHandle<Scalar4> h_pos(pdata_3->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_tag(pdata_3->getTags(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_rtag(pdata_3->getRTags(), access_location::host, access_mode::readwrite);

    h_pos.data[2].x = h_pos.data[2].y = h_pos.data[2].z = 0.0;
    h_pos.data[0].x = Scalar(2.0*pow(1.5,1.0/3.0)); h_pos.data[0].y = h_pos.data[0].z = 0.0;
    
    h_tag.data[0] = 2;
    h_tag.data[2] = 0;
    h_rtag.data[0] = 2;
    h_rtag.data[2] = 0;
    }
    
    // notify the particle data that we changed the order
    pdata_3->notifyParticleSort();
    
    // recompute the forces at the same timestep, they should be updated
    fc_3->compute(1);
    
    {
    GPUArray<Scalar4>& force_array_6 =  fc_3->getForceArray();
    GPUArray<Scalar>& virial_array_6 =  fc_3->getVirialArray();
    ArrayHandle<Scalar4> h_force_6(force_array_6,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_6(virial_array_6,access_location::host,access_mode::read);
    MY_BOOST_CHECK_CLOSE(h_force_6.data[0].x, 69.00675, tol);
    MY_BOOST_CHECK_CLOSE(h_force_6.data[2].x, -69.00675, tol);
    }
    }

//! Tests the ability of a CGCMMForceCompute to handle periodic boundary conditions
void cgcmm_force_periodic_test(cgcmmforce_creator cgcmm_creator, boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    ////////////////////////////////////////////////////////////////////
    // now, lets do a more thorough test and include boundary conditions
    // there are way too many permutations to test here, so I will simply
    // test +x, -x, +y, -y, +z, and -z independantly
    // build a 6 particle system with particles across each boundary
    // also test the ability of the force compute to use different particle types
    
    shared_ptr<SystemDefinition> sysdef_6(new SystemDefinition(6, BoxDim(20.0, 40.0, 60.0), 3, 0, 0, 0, 0, exec_conf));
    shared_ptr<ParticleData> pdata_6 = sysdef_6->getParticleData();
    
    pdata_6->setPosition(0,make_scalar3(-9.6,0.0,0.0));
    pdata_6->setPosition(1,make_scalar3(9.6,0.0,0.0));
    pdata_6->setPosition(2,make_scalar3(0.0,-19.6,0.0));
    pdata_6->setPosition(3,make_scalar3(0.0,19.6,0.0));
    pdata_6->setPosition(4,make_scalar3(0.0,0.0,-29.6));
    pdata_6->setPosition(5,make_scalar3(0.0,0.0,29.6));

    pdata_6->setType(0,0);
    pdata_6->setType(1,1);
    pdata_6->setType(2,2);
    pdata_6->setType(3,0);
    pdata_6->setType(4,2);
    pdata_6->setType(5,1);
    
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
    
    {
    GPUArray<Scalar4>& force_array_7 =  fc_6->getForceArray();
    GPUArray<Scalar>& virial_array_7 =  fc_6->getVirialArray();
    unsigned int pitch = virial_array_7.getPitch();
    ArrayHandle<Scalar4> h_force_7(force_array_7,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_7(virial_array_7,access_location::host,access_mode::read);
    // particle 0 should be pulled left
    MY_BOOST_CHECK_CLOSE(h_force_7.data[0].x, -1.18299976747949, tol);
    MY_BOOST_CHECK_SMALL(h_force_7.data[0].y, tol);
    MY_BOOST_CHECK_SMALL(h_force_7.data[0].z, tol);
    MY_BOOST_CHECK_CLOSE(Scalar(1./3.)*(h_virial_7.data[0*pitch+0]
                                 +h_virial_7.data[3*pitch+0]
                                 +h_virial_7.data[5*pitch+0]), -0.15773330233059, tol);
    
    // particle 1 should be pulled right
    MY_BOOST_CHECK_CLOSE(h_force_7.data[1].x, 1.18299976747949, tol);
    MY_BOOST_CHECK_SMALL(h_force_7.data[1].y, 1e-5);
    MY_BOOST_CHECK_SMALL(h_force_7.data[1].z, 1e-5);
    MY_BOOST_CHECK_CLOSE(Scalar(1./3.)*(h_virial_7.data[0*pitch+1]
                                 +h_virial_7.data[3*pitch+1]
                                 +h_virial_7.data[5*pitch+1]), -0.15773330233059, tol);
    
    // particle 2 should be pulled down
    MY_BOOST_CHECK_CLOSE(h_force_7.data[2].y, -1.77449965121923, tol);
    MY_BOOST_CHECK_SMALL(h_force_7.data[2].x, tol);
    MY_BOOST_CHECK_SMALL(h_force_7.data[2].z, tol);
    MY_BOOST_CHECK_CLOSE(Scalar(1./3.)*(h_virial_7.data[0*pitch+2]
                                 +h_virial_7.data[3*pitch+2]
                                 +h_virial_7.data[5*pitch+2]), -0.23659995349591, tol);
    
    // particle 3 should be pulled up
    MY_BOOST_CHECK_CLOSE(h_force_7.data[3].y, 1.77449965121923, tol);
    MY_BOOST_CHECK_SMALL(h_force_7.data[3].x, 1e-5);
    MY_BOOST_CHECK_SMALL(h_force_7.data[3].z, 1e-5);
    MY_BOOST_CHECK_CLOSE(Scalar(1./3.)*(h_virial_7.data[0*pitch+3]
                                 +h_virial_7.data[3*pitch+3]
                                 +h_virial_7.data[5*pitch+3]), -0.23659995349591, tol);
    
    // particle 4 should be pulled back
    MY_BOOST_CHECK_CLOSE(h_force_7.data[4].z, -2.95749941869871, tol);
    MY_BOOST_CHECK_SMALL(h_force_7.data[4].x, tol);
    MY_BOOST_CHECK_SMALL(h_force_7.data[4].y, tol);
    MY_BOOST_CHECK_CLOSE(Scalar(1./3.)*(h_virial_7.data[0*pitch+4]
                                 +h_virial_7.data[3*pitch+4]
                                 +h_virial_7.data[5*pitch+4]),  -0.39433325582651, tol);
    
    // particle 3 should be pulled forward
    MY_BOOST_CHECK_CLOSE(h_force_7.data[5].z, 2.95749941869871, tol);
    MY_BOOST_CHECK_SMALL(h_force_7.data[5].x, 1e-5);
    MY_BOOST_CHECK_SMALL(h_force_7.data[5].y, 1e-5);
    MY_BOOST_CHECK_CLOSE(Scalar(1./3.)*(h_virial_7.data[0*pitch+5]
                                 +h_virial_7.data[3*pitch+5]
                                 +h_virial_7.data[5*pitch+5]),  -0.39433325582651, tol);
    }
    }

//! Unit test a comparison between 2 CGCMMForceComputes on a "real" system
void cgcmm_force_comparison_test(cgcmmforce_creator cgcmm_creator1, cgcmmforce_creator cgcmm_creator2, boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    const unsigned int N = 5000;
    
    // create a random particle system to sum forces on
    RandomInitializer rand_init(N, Scalar(0.2), Scalar(0.9), "A");
    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(rand_init, exec_conf));
    shared_ptr<NeighborListBinned> nlist(new NeighborListBinned(sysdef, Scalar(3.0), Scalar(0.8)));
    
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
    
    {
    // verify that the forces are identical (within roundoff errors)
    GPUArray<Scalar4>& force_array_8 =  fc1->getForceArray();
    GPUArray<Scalar>& virial_array_8 =  fc1->getVirialArray();
    unsigned int pitch = virial_array_8.getPitch();
    ArrayHandle<Scalar4> h_force_8(force_array_8,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_8(virial_array_8,access_location::host,access_mode::read);
    GPUArray<Scalar4>& force_array_9 =  fc2->getForceArray();
    GPUArray<Scalar>& virial_array_9 =  fc2->getVirialArray();
    ArrayHandle<Scalar4> h_force_9(force_array_9,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_9(virial_array_9,access_location::host,access_mode::read);
    
    // compare average deviation between the two computes
    double deltaf2 = 0.0;
    double deltape2 = 0.0;
    double deltav2[6];
    for (unsigned int i = 0; i < 6; i++)
        deltav2[i] = 0.0;
        
    for (unsigned int i = 0; i < N; i++)
        {
        deltaf2 += double(h_force_9.data[i].x - h_force_8.data[i].x) * double(h_force_9.data[i].x - h_force_8.data[i].x);
        deltaf2 += double(h_force_9.data[i].y - h_force_8.data[i].y) * double(h_force_9.data[i].y - h_force_8.data[i].y);
        deltaf2 += double(h_force_9.data[i].z - h_force_8.data[i].z) * double(h_force_9.data[i].z - h_force_8.data[i].z);
        deltape2 += double(h_force_9.data[i].w - h_force_8.data[i].w) * double(h_force_9.data[i].w - h_force_8.data[i].w);
        for (int j = 0; j < 6; j++)
            deltav2[j] += double(h_virial_9.data[j*pitch+i] - h_virial_8.data[j*pitch+i]) * double(h_virial_9.data[j*pitch+i] - h_virial_8.data[j*pitch+i]);

        // also check that each individual calculation is somewhat close
        }
    deltaf2 /= double(sysdef->getParticleData()->getN());
    deltape2 /= double(sysdef->getParticleData()->getN());
    for (unsigned int j = 0; j < 6; j++)
        deltav2[j] /= double(sysdef->getParticleData()->getN());
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
    cgcmm_force_particle124_test(cgcmm_creator_base, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

//! boost test case for particle test on CPU
BOOST_AUTO_TEST_CASE( CGCMMForce_particle96 )
    {
    cgcmmforce_creator cgcmm_creator_base = bind(base_class_cgcmm_creator, _1, _2, _3);
    cgcmm_force_particle96_test(cgcmm_creator_base, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

//! boost test case for periodic test on CPU
BOOST_AUTO_TEST_CASE( CGCMMForce_periodic )
    {
    cgcmmforce_creator cgcmm_creator_base = bind(base_class_cgcmm_creator, _1, _2, _3);
    cgcmm_force_periodic_test(cgcmm_creator_base, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

# ifdef ENABLE_CUDA
//! boost test case for particle test on GPU - threaded
BOOST_AUTO_TEST_CASE( CGCMMForceGPU_particle124 )
    {
    cgcmmforce_creator cgcmm_creator_gpu = bind(gpu_cgcmm_creator, _1, _2, _3);
    cgcmm_force_particle124_test(cgcmm_creator_gpu, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! boost test case for particle test on GPU - threaded
BOOST_AUTO_TEST_CASE( CGCMMForceGPU_particle96 )
    {
    cgcmmforce_creator cgcmm_creator_gpu = bind(gpu_cgcmm_creator, _1, _2, _3);
    cgcmm_force_particle96_test(cgcmm_creator_gpu, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! boost test case for periodic test on the GPU
BOOST_AUTO_TEST_CASE( CGCMMForceGPU_periodic )
    {
    cgcmmforce_creator cgcmm_creator_gpu = bind(gpu_cgcmm_creator, _1, _2, _3);
    cgcmm_force_periodic_test(cgcmm_creator_gpu, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! boost test case for comparing GPU output to base class output
BOOST_AUTO_TEST_CASE( CGCMMForceGPU_compare )
    {
    cgcmmforce_creator cgcmm_creator_gpu = bind(gpu_cgcmm_creator, _1, _2, _3);
    cgcmmforce_creator cgcmm_creator_base = bind(base_class_cgcmm_creator, _1, _2, _3);
    cgcmm_force_comparison_test(cgcmm_creator_base, cgcmm_creator_gpu, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

#endif

#ifdef WIN32
#pragma warning( pop )
#endif

