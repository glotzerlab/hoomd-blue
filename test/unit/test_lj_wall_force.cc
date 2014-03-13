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
#include <boost/shared_ptr.hpp>

#include "LJWallForceCompute.h"
#include "WallData.h"

#include <math.h>

using namespace std;
using namespace boost;

/*! \file lj_wall_compute_test.cc
    \brief Implements unit tests for LJWallForceCompute and descendants
    \ingroup unit_tests
*/

//! Name the unit test module
#define BOOST_TEST_MODULE LJWallForceTests
#include "boost_utf_configure.h"

//! Typedef'd LJWallForceCompute factory
typedef boost::function<boost::shared_ptr<LJWallForceCompute> (boost::shared_ptr<SystemDefinition> sysdef,
                                                        Scalar r_cut)> ljwallforce_creator;

//! Test the ability of the lj wall force compute to actually calculate forces
void ljwall_force_particle_test(ljwallforce_creator ljwall_creator, boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // this 3 particle test will check proper wall force computation among all 3 axes
    boost::shared_ptr<SystemDefinition> sysdef_3(new SystemDefinition(3, BoxDim(1000.0), 1, 0, 0, 0, 0, exec_conf));
    boost::shared_ptr<ParticleData> pdata_3 = sysdef_3->getParticleData();

    {
    ArrayHandle<Scalar4> h_pos(pdata_3->getPositions(), access_location::host, access_mode::readwrite);
    h_pos.data[0].x = 0.0; h_pos.data[0].y = Scalar(1.2); h_pos.data[0].z = 0.0;    // particle to test wall at pos 0,0,0
    h_pos.data[1].x = Scalar(12.2); h_pos.data[1].y = Scalar(-10.0); h_pos.data[1].z = 0.0; // particle to test wall at pos 10,0,0
    h_pos.data[2].x = 0.0; h_pos.data[2].y = Scalar(10.0); h_pos.data[2].z = Scalar(-12.9); // particle to test wall at pos 0,0,-10
    }

    // create the wall force compute with a default cuttoff of 1.0 => all forces should be 0 for the first round
    boost::shared_ptr<LJWallForceCompute> fc_3 = ljwall_creator(sysdef_3, Scalar(1.0));

    // pick some parameters
    Scalar epsilon = Scalar(1.15);
    Scalar sigma = Scalar(1.0);
    Scalar alpha = Scalar(1.0);
    Scalar lj1 = Scalar(4.0) * epsilon * pow(sigma,Scalar(12.0));
    Scalar lj2 = alpha * Scalar(4.0) * epsilon * pow(sigma,Scalar(6.0));
    fc_3->setParams(0,lj1,lj2);

    // compute the forces
    fc_3->compute(0);

    {
    // there are no walls, so all forces should be zero
    GPUArray<Scalar4>& force_array_1 =  fc_3->getForceArray();
    GPUArray<Scalar>& virial_array_1 =  fc_3->getVirialArray();
    ArrayHandle<Scalar4> h_force_1(force_array_1,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_1(virial_array_1,access_location::host,access_mode::read);
    MY_BOOST_CHECK_SMALL(h_force_1.data[0].x, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_1.data[0].y, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_1.data[0].z, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_1.data[0].w, tol_small);

    MY_BOOST_CHECK_SMALL(h_force_1.data[1].x, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_1.data[1].y, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_1.data[1].z, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_1.data[1].w, tol_small);

    MY_BOOST_CHECK_SMALL(h_force_1.data[2].x, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_1.data[2].y, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_1.data[2].z, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_1.data[2].w, tol_small);
    }

    // add the walls
    sysdef_3->getWallData()->addWall(Wall(0.0, 0.0, 0.0, 0.0, 1.0, 0.0));
    sysdef_3->getWallData()->addWall(Wall(10.0, 0.0, 0.0, 1.0, 0.0, 0.0));
    sysdef_3->getWallData()->addWall(Wall(0.0, 0.0, -10.0, 0.0, 0.0, 1.0));

    // compute the forces again
    fc_3->compute(1);

    {
    // they should still be zero
    GPUArray<Scalar4>& force_array_2 =  fc_3->getForceArray();
    GPUArray<Scalar>& virial_array_2 =  fc_3->getVirialArray();
    ArrayHandle<Scalar4> h_force_2(force_array_2,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_2(virial_array_2,access_location::host,access_mode::read);
    MY_BOOST_CHECK_SMALL(h_force_2.data[0].x, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_2.data[0].y, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_2.data[0].z, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_2.data[0].w, tol_small);

    MY_BOOST_CHECK_SMALL(h_force_2.data[1].x, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_2.data[1].y, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_2.data[1].z, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_2.data[1].w, tol_small);

    MY_BOOST_CHECK_SMALL(h_force_2.data[2].x, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_2.data[2].y, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_2.data[2].z, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_2.data[2].w, tol_small);
    }

    // increase the cuttoff to check the actual force computation
    fc_3->setRCut(3.0);
    fc_3->compute(2);

    {
    GPUArray<Scalar4>& force_array_3 =  fc_3->getForceArray();
    GPUArray<Scalar>& virial_array_3 =  fc_3->getVirialArray();
    ArrayHandle<Scalar4> h_force_3(force_array_3,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_3(virial_array_3,access_location::host,access_mode::read);
    MY_BOOST_CHECK_SMALL(h_force_3.data[0].x, tol_small);
    MY_BOOST_CHECK_CLOSE(h_force_3.data[0].y, -2.54344734, tol);
    MY_BOOST_CHECK_SMALL(h_force_3.data[0].z, tol_small);
    MY_BOOST_CHECK_CLOSE(h_force_3.data[0].w, -1.0246100807205, tol);

    MY_BOOST_CHECK_CLOSE(h_force_3.data[1].x, -0.108697879, tol);
    MY_BOOST_CHECK_SMALL(h_force_3.data[1].y, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_3.data[1].z, tol_small);
    MY_BOOST_CHECK_CLOSE(h_force_3.data[1].w, -0.04021378505, tol);

    MY_BOOST_CHECK_SMALL(h_force_3.data[2].x, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_3.data[2].y, tol_small);
    MY_BOOST_CHECK_CLOSE(h_force_3.data[2].z, 0.0159463169, tol);
    MY_BOOST_CHECK_CLOSE(h_force_3.data[2].w, -0.0077203876329103, tol);
    }
    }

//! LJWallForceCompute creator for unit tests
boost::shared_ptr<LJWallForceCompute> base_class_ljwall_creator(boost::shared_ptr<SystemDefinition> sysdef, Scalar r_cut)
    {
    return boost::shared_ptr<LJWallForceCompute>(new LJWallForceCompute(sysdef, r_cut));
    }

//! boost test case for particle test on CPU
BOOST_AUTO_TEST_CASE( LJWallForce_particle )
    {
    ljwallforce_creator ljwall_creator_base = bind(base_class_ljwall_creator, _1, _2);
    ljwall_force_particle_test(ljwall_creator_base, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

#ifdef WIN32
#pragma warning( pop )
#endif
