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
// Maintainer: baschult

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <iostream>

#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include "ConstExternalFieldDipoleForceCompute.h"

#include <math.h>
#include "QuaternionMath.h"

#ifndef PI
#define PI 3.141592653589793
#endif

using namespace std;
using namespace boost;

/*! \file lj_wall_compute_test.cc
    \brief Implements unit tests for LJWallForceCompute and descendants
    \ingroup unit_tests
*/

//! Name the unit test module
#define BOOST_TEST_MODULE ConstExternalFieldDipoleForceTest
#include "boost_utf_configure.h"

//! Typedef'd ConstExternalFieldDipoleForceCompute factory
typedef boost::function<shared_ptr<ConstExternalFieldDipoleForceCompute> (shared_ptr<SystemDefinition> sysdef,
                                                        Scalar fx,Scalar fy, Scalar fz, Scalar p)> cefd_force_creator;

//! Test the ability of the lj wall force compute to actually calculate forces
void cefd_force_particle_test(cefd_force_creator cefd_creator, boost::shared_ptr<ExecutionConfiguration> exec_conf)
    { 
    // this 3 particle test will check proper wall force computation among all 3 axes
    shared_ptr<SystemDefinition> sysdef_3(new SystemDefinition(3, BoxDim(1000.0), 1, 0, 0, 0, 0, exec_conf));
    shared_ptr<ParticleData> pdata_3 = sysdef_3->getParticleData();
    
    pdata_3->setPosition(0,make_scalar3(0.0,1.2,0.0));
    pdata_3->setPosition(1,make_scalar3(12.2,-10.0,0.0));
    pdata_3->setPosition(2,make_scalar3(0.0,10.0,-12.9));

    {   
    ArrayHandle<Scalar4> h_orientation(pdata_3->getOrientationArray());
    eulerToQuat(0.0,0.5*PI, 0.0, h_orientation.data[0]); //dipole along x
    eulerToQuat(0.5*PI,0, 0.0, h_orientation.data[1]); //dipole along y
    eulerToQuat(2*PI, 2*PI, 2*PI, h_orientation.data[2]); //dipole along z
    }
    // create the wall force compute with a default cutoff of 1.0 => all forces should be 0 for the first round
    shared_ptr<ConstExternalFieldDipoleForceCompute> fc_3 = cefd_creator(sysdef_3, Scalar(0.0),Scalar(0.0),Scalar(0.0),Scalar(0.0));
    
    // pick some parameters
    Scalar field_x = 0.0;
    Scalar field_y = 0.0;
    Scalar field_z = 0.0;
    Scalar       p = 1.0;
    fc_3->setParams(field_x,field_y,field_z,p);
    
    // compute the forces
    fc_3->compute(0);
    
    {
    // there are no walls, so all forces should be zero
    GPUArray<Scalar4>& force_array_1 =  fc_3->getForceArray();
    GPUArray<Scalar>& virial_array_1 =  fc_3->getVirialArray();
    GPUArray<Scalar4>& torque_array_1 = fc_3->getTorqueArray();
    ArrayHandle<Scalar4> h_force_1(force_array_1,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_1(virial_array_1,access_location::host,access_mode::read);
    ArrayHandle<Scalar4> h_torque_1(torque_array_1,access_location::host,access_mode::read);

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

    MY_BOOST_CHECK_SMALL(h_torque_1.data[0].x, tol_small);
    MY_BOOST_CHECK_SMALL(h_torque_1.data[0].y, tol_small);
    MY_BOOST_CHECK_SMALL(h_torque_1.data[0].z, tol_small);
    MY_BOOST_CHECK_SMALL(h_torque_1.data[0].w, tol_small);
    
    MY_BOOST_CHECK_SMALL(h_torque_1.data[1].x, tol_small);
    MY_BOOST_CHECK_SMALL(h_torque_1.data[1].y, tol_small);
    MY_BOOST_CHECK_SMALL(h_torque_1.data[1].z, tol_small);
    MY_BOOST_CHECK_SMALL(h_torque_1.data[1].w, tol_small);
    
    MY_BOOST_CHECK_SMALL(h_torque_1.data[2].x, tol_small);
    MY_BOOST_CHECK_SMALL(h_torque_1.data[2].y, tol_small);
    MY_BOOST_CHECK_SMALL(h_torque_1.data[2].z, tol_small);
    MY_BOOST_CHECK_SMALL(h_torque_1.data[2].w, tol_small);
    }

    //turn on the field
    field_x = 0.1;
    field_y = 1.0;
    field_z = 10.0;
    p = 1.0;
    fc_3->setParams(field_x,field_y,field_z,p);
    
    // compute the forces again
    fc_3->compute(1);
    
    {
    // they should still be zero
    GPUArray<Scalar4>& force_array_2 =  fc_3->getForceArray();
    GPUArray<Scalar>& virial_array_2 =  fc_3->getVirialArray();
    GPUArray<Scalar4>& torque_array_2 =  fc_3->getTorqueArray();
    ArrayHandle<Scalar4> h_force_2(force_array_2,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_2(virial_array_2,access_location::host,access_mode::read);
    ArrayHandle<Scalar4> h_torque_2(torque_array_2,access_location::host,access_mode::read);

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

    MY_BOOST_CHECK_SMALL(h_torque_2.data[0].x, tol_small);
    MY_BOOST_CHECK_CLOSE(h_torque_2.data[0].y, -field_z, tol);
    MY_BOOST_CHECK_CLOSE(h_torque_2.data[0].z, field_y,tol);
    MY_BOOST_CHECK_SMALL(h_torque_2.data[0].w, tol_small);
     
    MY_BOOST_CHECK_CLOSE(h_torque_2.data[1].x, -field_z, tol);
    MY_BOOST_CHECK_SMALL(h_torque_2.data[1].y, tol_small);
    MY_BOOST_CHECK_CLOSE(h_torque_2.data[1].z, field_x,tol);
    MY_BOOST_CHECK_SMALL(h_torque_2.data[1].w, tol_small);
    
    MY_BOOST_CHECK_CLOSE(h_torque_2.data[2].x, -field_y, tol);
    MY_BOOST_CHECK_CLOSE(h_torque_2.data[2].y, field_x, tol);
    MY_BOOST_CHECK_SMALL(h_torque_2.data[2].z, tol_small);
    MY_BOOST_CHECK_SMALL(h_torque_2.data[2].w, tol_small);
    }

   
    }

//! ConstExternalFieldDipoleForceCompute creator for unit tests
shared_ptr<ConstExternalFieldDipoleForceCompute> base_class_cefd_creator(shared_ptr<SystemDefinition> sysdef,Scalar fx, Scalar fy, Scalar fz, Scalar p)
    {
    return shared_ptr<ConstExternalFieldDipoleForceCompute>(new ConstExternalFieldDipoleForceCompute(sysdef, fx, fy, fz, p));
    }

//! boost test case for particle test on CPU
BOOST_AUTO_TEST_CASE( cefd_particle )
    {
    cefd_force_creator cefd_creator_base = bind(base_class_cefd_creator, _1, _2,_3,_4,_5);
    cefd_force_particle_test(cefd_creator_base, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

#ifdef WIN32
#pragma warning( pop )
#endif

