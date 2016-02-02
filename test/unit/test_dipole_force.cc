/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2016 The Regents of
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

// this include is necessary to get MPI included before anything else to support intel MPI
#include "ExecutionConfiguration.h"

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <iostream>
#include <fstream>

#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include "AllAnisoPairPotentials.h"

#include "NeighborListTree.h"
#include "Initializers.h"

#include <math.h>

using namespace std;
using namespace boost;

/*! \file test_dipole_force.cc
    \brief Implements unit tests for AnisoPotentialPairDipole and AnisoPotentialPairDipoleGPU
    \ingroup unit_tests
*/

//! Name the unit test module
#define BOOST_TEST_MODULE PotentialPairDipoleTests
#include "boost_utf_configure.h"

typedef boost::function<boost::shared_ptr<AnisoPotentialPairDipole> (boost::shared_ptr<SystemDefinition> sysdef,
                                                     boost::shared_ptr<NeighborList> nlist)> dipoleforce_creator;

//! Test the ability of the Gay Berne force compute to actually calucate forces
void dipole_force_particle_test(dipoleforce_creator dipole_creator, boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    boost::shared_ptr<SystemDefinition> sysdef_2(new SystemDefinition(2, BoxDim(1000.0), 1, 0, 0, 0, 0, exec_conf));
    boost::shared_ptr<ParticleData> pdata_2 = sysdef_2->getParticleData();
    pdata_2->setFlags(~PDataFlags(0));

    {
    ArrayHandle<Scalar4> h_pos(pdata_2->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_charge(pdata_2->getCharges(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_orientation(pdata_2->getOrientationArray(), access_location::host, access_mode::readwrite);

    h_pos.data[0].x = h_pos.data[0].y = h_pos.data[0].z = 0.0;
    h_charge.data[0] = 2.0;
    h_pos.data[1].x = .8; h_pos.data[1].y = .45; h_pos.data[1].z = .9;
    h_charge.data[1] = 1.0;
    h_pos.data[0].w = h_pos.data[1].w = __int_as_scalar(0);

    // default orientation has dipole in (1, 0, 0) direction
    h_orientation.data[0] = make_scalar4(1,0,0,0);
    // rotate particle 1 by 2*pi/3 about the (1, 1, 0) axis
    h_orientation.data[1] = make_scalar4(cos(2*M_PI/6), sin(2*M_PI/6)/sqrt(2), sin(2*M_PI/6)/sqrt(2), 0);
    }
    boost::shared_ptr<NeighborList> nlist_2(new NeighborListTree(sysdef_2, Scalar(6.0), Scalar(6.5)));
    boost::shared_ptr<AnisoPotentialPairDipole> fc_2 = dipole_creator(sysdef_2, nlist_2);
    fc_2->setRcut(0, 0, Scalar(6.0));

    // Compare with lammps dipole potential, which fixes A=1 and kappa=0
    fc_2->setParams(0, 0, make_scalar3(0.6, 1, 0));

    // compute the forces
    fc_2->compute(0);

    {
    GPUArray<Scalar4>& force_array_1 =  fc_2->getForceArray();
    GPUArray<Scalar>& virial_array_1 =  fc_2->getVirialArray();
    GPUArray<Scalar4>& torque_array_1 =  fc_2->getTorqueArray();
    ArrayHandle<Scalar4> h_force_1(force_array_1,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_1(virial_array_1,access_location::host,access_mode::read);
    ArrayHandle<Scalar4> h_torque_1(torque_array_1,access_location::host,access_mode::read);
    MY_BOOST_CHECK_CLOSE(h_force_1.data[0].x, -1.07832, tol);
    MY_BOOST_CHECK_CLOSE(h_force_1.data[0].y, -1.26201, tol);
    MY_BOOST_CHECK_CLOSE(h_force_1.data[0].z, -0.810835, tol);
    MY_BOOST_CHECK_CLOSE(h_force_1.data[0].w, 0.917602, tol);
    MY_BOOST_CHECK_CLOSE(h_torque_1.data[0].x, 0, tol);
    MY_BOOST_CHECK_CLOSE(h_torque_1.data[0].y, 0.154201, tol);
    MY_BOOST_CHECK_CLOSE(h_torque_1.data[0].z, -0.256091, tol);

    MY_BOOST_CHECK_CLOSE(h_force_1.data[1].x, 1.07832, tol);
    MY_BOOST_CHECK_CLOSE(h_force_1.data[1].y, 1.26201, tol);
    MY_BOOST_CHECK_CLOSE(h_force_1.data[1].z, 0.810835, tol);
    MY_BOOST_CHECK_CLOSE(h_force_1.data[1].w, 0.917602, tol);
    MY_BOOST_CHECK_CLOSE(h_torque_1.data[1].x, 0.770933, tol);
    MY_BOOST_CHECK_CLOSE(h_torque_1.data[1].y, -0.476021, tol);
    MY_BOOST_CHECK_CLOSE(h_torque_1.data[1].z, -0.268273, tol);
    }
    }

//! LJForceCompute creator for unit tests
boost::shared_ptr<AnisoPotentialPairDipole> base_class_dipole_creator(boost::shared_ptr<SystemDefinition> sysdef,
                                                  boost::shared_ptr<NeighborList> nlist)
    {
    return boost::shared_ptr<AnisoPotentialPairDipole>(new AnisoPotentialPairDipole(sysdef, nlist));
    }

#ifdef ENABLE_CUDA
//! LJForceComputeGPU creator for unit tests
boost::shared_ptr<AnisoPotentialPairDipoleGPU> gpu_dipole_creator(boost::shared_ptr<SystemDefinition> sysdef,
                                          boost::shared_ptr<NeighborList> nlist)
    {
    nlist->setStorageMode(NeighborList::full);
    return boost::shared_ptr<AnisoPotentialPairDipoleGPU>(new AnisoPotentialPairDipoleGPU(sysdef, nlist));
    }
#endif

//! boost test case for particle test on CPU
BOOST_AUTO_TEST_CASE( AnisoPotentialPairDipole_particle )
    {
    dipoleforce_creator dipole_creator_base = bind(base_class_dipole_creator, _1, _2);
    dipole_force_particle_test(dipole_creator_base, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

#ifdef ENABLE_CUDA
//! boost test case for particle test on GPU
BOOST_AUTO_TEST_CASE( AnisoPotentialPairDipoleGPU_particle )
    {
    dipoleforce_creator dipole_creator_gpu = bind(gpu_dipole_creator, _1, _2);
    dipole_force_particle_test(dipole_creator_gpu, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
#endif
