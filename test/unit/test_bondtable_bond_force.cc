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

#include <fstream>

#include <boost/bind.hpp>
#include <boost/function.hpp>

#include "BondTablePotential.h"

#ifdef ENABLE_CUDA
#include "BondTablePotentialGPU.h"
#endif

#include "Initializers.h"

using namespace std;
using namespace boost;

/*! \file harmonic_bond_force_test.cc
    \brief Implements unit tests for BondTablePotential and
           BondTablePotentialGPU
    \ingroup unit_tests
*/

//! Name the boost unit test module
#define BOOST_TEST_MODULE BondTableForceTests
#include "boost_utf_configure.h"

//! Typedef to make using the boost::function factory easier
typedef boost::function<boost::shared_ptr<BondTablePotential>  (boost::shared_ptr<SystemDefinition> sysdef, unsigned int width)> bondforce_creator;

//! Perform some simple functionality tests of any BondTableForceCompute
void bond_force_basic_tests(bondforce_creator bf_creator, boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    /////////////////////////////////////////////////////////
    // start with the simplest possible test: 2 particles in a huge box with only one bond type
    boost::shared_ptr<SystemDefinition> sysdef_2(new SystemDefinition(2, BoxDim(1000.0), 1, 1, 0, 0, 0, exec_conf));
    boost::shared_ptr<ParticleData> pdata_2 = sysdef_2->getParticleData();

    pdata_2->setPosition(0,make_scalar3(0.0,0.0,0.0));
    pdata_2->setPosition(1,make_scalar3(1.0,0.0,0.0));

    // create the bond force compute to check
    boost::shared_ptr<BondTablePotential> fc_2 = bf_creator(sysdef_2,3);



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

    // specify a table to interpolate
    vector<Scalar> V, F;
    V.push_back(10.0);  F.push_back(1.0);
    V.push_back(21.0);  F.push_back(6.0);
    V.push_back(5.0);   F.push_back(2.0);
    fc_2->setTable(0, V, F, 2.0, 4.0);

    // now go to rmin and check for the correct force value
    {
    ArrayHandle<Scalar4> h_pos(pdata_2->getPositions(), access_location::host, access_mode::readwrite);
    h_pos.data[1].x = Scalar(2.0);
    }

    fc_2->compute(1);

    {
    GPUArray<Scalar4>& force_array_3 =  fc_2->getForceArray();
    GPUArray<Scalar>& virial_array_3 =  fc_2->getVirialArray();
    unsigned int pitch = virial_array_3.getPitch();
    ArrayHandle<Scalar4> h_force_3(force_array_3,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_3(virial_array_3,access_location::host,access_mode::read);

    MY_BOOST_CHECK_CLOSE(h_force_3.data[0].x, -1.0, tol);
    MY_BOOST_CHECK_SMALL(h_force_3.data[0].y, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_3.data[0].z, tol_small);
    MY_BOOST_CHECK_CLOSE(h_force_3.data[0].w, 5.0, tol);
    MY_BOOST_CHECK_CLOSE(Scalar(1./3.)*(h_virial_3.data[0*pitch+0]
                                       +h_virial_3.data[3*pitch+0]
                                       +h_virial_3.data[5*pitch+0]), (1.0 / 6.0) * 2.0, tol);

    MY_BOOST_CHECK_CLOSE(h_force_3.data[1].x, 1.0, tol);
    MY_BOOST_CHECK_SMALL(h_force_3.data[1].y, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_3.data[1].z, tol_small);
    MY_BOOST_CHECK_CLOSE(h_force_3.data[1].w, 5.0, tol);
    MY_BOOST_CHECK_CLOSE(Scalar(1./3.)*(h_virial_3.data[0*pitch+1]
                                       +h_virial_3.data[3*pitch+1]
                                       +h_virial_3.data[5*pitch+1]), (1.0 / 6.0) * 2.0, tol);
    }

    // go halfway in-between two points
    {
    ArrayHandle<Scalar4> h_pos(pdata_2->getPositions(), access_location::host, access_mode::readwrite);
    h_pos.data[1].y = Scalar(3.5);
    h_pos.data[1].x = Scalar(0.0);
    }

    // check the forces
    fc_2->compute(2);

    {
    GPUArray<Scalar4>& force_array_4 =  fc_2->getForceArray();
    GPUArray<Scalar>& virial_array_4 =  fc_2->getVirialArray();
    unsigned int pitch = virial_array_4.getPitch();
    ArrayHandle<Scalar4> h_force_4(force_array_4,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_4(virial_array_4,access_location::host,access_mode::read);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[0].y, -4.0, tol);
    MY_BOOST_CHECK_SMALL(h_force_4.data[0].x, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_4.data[0].z, tol_small);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[0].w, 13.0/2.0, tol);
    MY_BOOST_CHECK_CLOSE(Scalar(1./3.)*(h_virial_4.data[0*pitch+0]
                                       +h_virial_4.data[3*pitch+0]
                                       +h_virial_4.data[5*pitch+0]), (1.0 / 6.0) * 4.0 * 3.5, tol);

    MY_BOOST_CHECK_CLOSE(h_force_4.data[1].y, 4.0, tol);
    MY_BOOST_CHECK_SMALL(h_force_4.data[1].x, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_4.data[1].z, tol_small);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[1].w, 13.0 / 2.0, tol);
    MY_BOOST_CHECK_CLOSE(Scalar(1./3.)*(h_virial_4.data[0*pitch+1]
                                       +h_virial_4.data[3*pitch+1]
                                       +h_virial_4.data[5*pitch+1]), (1.0 / 6.0) * 4.0 * 3.5, tol);
    }
    }


//! checks to see if BondTablePotential correctly handles multiple types
void bond_force_type_test(bondforce_creator bf_creator, boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // start with the simplest possible test: 3 particles in a huge box with two bond types
    boost::shared_ptr<SystemDefinition> sysdef_2(new SystemDefinition(3, BoxDim(1000.0), 1, 2, 0, 0, 0, exec_conf));
    boost::shared_ptr<ParticleData> pdata_2 = sysdef_2->getParticleData();

    {
    pdata_2->setPosition(0,make_scalar3(0.0,0.0,0.0));
    pdata_2->setPosition(1,make_scalar3(1.0,0.0,0.0));
    pdata_2->setPosition(2,make_scalar3(1.0,1.0,0.0));
    }

    // create the bond force compute to check
    boost::shared_ptr<BondTablePotential> fc_2 = bf_creator(sysdef_2,3);

    // specify a table to interpolate
    vector<Scalar> V, F;
    V.push_back(10.0);  F.push_back(1.0);
    V.push_back(20.0);  F.push_back(6.0);
    V.push_back(5.0);   F.push_back(2.0);
    fc_2->setTable(0, V, F, 1.0, 2.0);

    // specify a second table to interpolate
    V.clear(); F.clear();
    V.push_back(20.0);  F.push_back(2.0);
    V.push_back(40.0);  F.push_back(12.0);
    V.push_back(10.0);   F.push_back(4.0);
    fc_2->setTable(1, V, F, 0.0, 2.0);

    // add a bond
    sysdef_2->getBondData()->addBondedGroup(Bond(0, 0,1));

    // add a second bond
    sysdef_2->getBondData()->addBondedGroup(Bond(1, 1,2));

    // compute and check
    fc_2->compute(0);

    {
    GPUArray<Scalar4>& force_array_6 =  fc_2->getForceArray();
    GPUArray<Scalar>& virial_array_6 =  fc_2->getVirialArray();
    unsigned int pitch = virial_array_6.getPitch();
    ArrayHandle<Scalar4> h_force_6(force_array_6,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_6(virial_array_6,access_location::host,access_mode::read);

    MY_BOOST_CHECK_CLOSE(h_force_6.data[0].x, -1.0, tol);
    MY_BOOST_CHECK_SMALL(h_force_6.data[0].y, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_6.data[0].z, tol_small);
    MY_BOOST_CHECK_CLOSE(h_force_6.data[0].w, 5.0, tol);
    MY_BOOST_CHECK_CLOSE(Scalar(1./3.)*(h_virial_6.data[0*pitch+0]
                                       +h_virial_6.data[3*pitch+0]
                                       +h_virial_6.data[5*pitch+0]), (1.0)*1.0/6.0, tol);

    MY_BOOST_CHECK_CLOSE(h_force_6.data[1].x, 1.0, tol);
    MY_BOOST_CHECK_CLOSE(h_force_6.data[1].y, -12.0, tol);
    MY_BOOST_CHECK_SMALL(h_force_6.data[1].z, tol_small);
    MY_BOOST_CHECK_CLOSE(h_force_6.data[1].w, 20.0 + 5.0, tol);
    MY_BOOST_CHECK_CLOSE(Scalar(1./3.)*(h_virial_6.data[0*pitch+1]
                                       +h_virial_6.data[3*pitch+1]
                                       +h_virial_6.data[5*pitch+1]), (1*1.0 + 12.0 * 1.0)*1.0/6.0, tol);

    MY_BOOST_CHECK_SMALL(h_force_6.data[2].x, tol_small);
    MY_BOOST_CHECK_CLOSE(h_force_6.data[2].y, 12.0, tol);
    MY_BOOST_CHECK_SMALL(h_force_6.data[2].z, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_6.data[2].w, 120.0);
    MY_BOOST_CHECK_CLOSE(Scalar(1./3.)*(h_virial_6.data[0*pitch+2]
                                       +h_virial_6.data[3*pitch+2]
                                       +h_virial_6.data[5*pitch+2]), (12*1.0)*1.0/6.0, tol);

    }
     }



//! BondTablePotential creator for bond_force_basic_tests()
boost::shared_ptr<BondTablePotential> base_class_bf_creator(boost::shared_ptr<SystemDefinition> sysdef, unsigned int width)
    {
    return boost::shared_ptr<BondTablePotential>(new BondTablePotential(sysdef, width));
    }

#ifdef ENABLE_CUDA
//! BondTablePotential creator for bond_force_basic_tests()
boost::shared_ptr<BondTablePotential> gpu_bf_creator(boost::shared_ptr<SystemDefinition> sysdef, unsigned int width)
    {
    return boost::shared_ptr<BondTablePotential>(new BondTablePotentialGPU(sysdef, width));
    }
#endif

//! boost test case for bond forces on the CPU
BOOST_AUTO_TEST_CASE( BondTablePotential_basic )
    {
    bondforce_creator bf_creator = bind(base_class_bf_creator, _1, _2);
    bond_force_basic_tests(bf_creator, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

//! boost test case for bond force type on the CPU
BOOST_AUTO_TEST_CASE( BondTablePotential_type )
    {
    bondforce_creator bf_creator = bind(base_class_bf_creator, _1, _2);
    bond_force_type_test(bf_creator, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }


#ifdef ENABLE_CUDA
//! boost test case for bond forces on the GPU
BOOST_AUTO_TEST_CASE( BondTablePotentialGPU_basic )
    {
    bondforce_creator bf_creator = bind(gpu_bf_creator, _1, _2);
    bond_force_basic_tests(bf_creator, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! boost test case for bond force type on the GPU
BOOST_AUTO_TEST_CASE( BondTablePotentialGPU_type )
    {
    bondforce_creator bf_creator = bind(gpu_bf_creator, _1, _2);
    bond_force_type_test(bf_creator, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

#endif



#ifdef WIN32
#pragma warning( pop )
#endif
