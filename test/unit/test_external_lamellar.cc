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
#include <fstream>

#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include "AllExternalPotentials.h"

#include "Initializers.h"

#include <math.h>

using namespace std;
using namespace boost;

/*! \file lj_force_test.cc
    \brief Implements unit tests for PotentialPairLJ and PotentialPairLJGPU and descendants
    \ingroup unit_tests
*/

//! Name the unit test module
#define BOOST_TEST_MODULE PotentialExternalLamellarTests
#include "boost_utf_configure.h"

//! Typedef'd LJForceCompute factory
typedef boost::function<shared_ptr<PotentialExternalLamellar> (shared_ptr<SystemDefinition> sysdef)> lamellarforce_creator;

//! Test the ability of the lj force compute to actually calucate forces
void lamellar_force_particle_test(lamellarforce_creator lamellar_creator, boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // this 3-particle test subtly checks several conditions
    // the particles are arranged on the x axis,  1   2   3
    // types of the particles : 0, 1, 0

    // periodic boundary conditions will be handeled in another test
    shared_ptr<SystemDefinition> sysdef_3(new SystemDefinition(3, BoxDim(5.0), 2, 0, 0, 0, 0, exec_conf));
    shared_ptr<ParticleData> pdata_3 = sysdef_3->getParticleData();

    pdata_3->setPosition(0,make_scalar3(1.7,0.0,0.0));
    pdata_3->setPosition(1,make_scalar3(2.0,0.0,0.0));
    pdata_3->setPosition(2,make_scalar3(3.5,0.0,0.0));
    pdata_3->setType(1,1);
    shared_ptr<PotentialExternalLamellar> fc_3 = lamellar_creator(sysdef_3);

    // first test: setup a sigma of 1.0 so that all forces will be 0
    unsigned int index = 0;
    Scalar orderParameter = 0.5;
    Scalar interfaceWidth = 0.5;
    unsigned int periodicity = 2;
    fc_3->setParams(0,make_scalar4(__int_as_scalar(index),orderParameter,interfaceWidth,__int_as_scalar(periodicity)));
    fc_3->setParams(1,make_scalar4(__int_as_scalar(index),-orderParameter,interfaceWidth,__int_as_scalar(periodicity)));

    // compute the forces
    fc_3->compute(0);

    {
    GPUArray<Scalar4>& force_array_1 =  fc_3->getForceArray();
    GPUArray<Scalar>& virial_array_1 =  fc_3->getVirialArray();
    unsigned int pitch = virial_array_1.getPitch();
    ArrayHandle<Scalar4> h_force_1(force_array_1,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_1(virial_array_1,access_location::host,access_mode::read);
    MY_BOOST_CHECK_CLOSE(h_force_1.data[0].x, -0.180137, tol);
    MY_BOOST_CHECK_SMALL(h_force_1.data[0].y, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_1.data[0].z, tol_small);
    MY_BOOST_CHECK_CLOSE(h_force_1.data[0].w, -0.0338307, tol);
    MY_BOOST_CHECK_SMALL(h_virial_1.data[0*pitch+0]
                        +h_virial_1.data[3*pitch+0]
                        +h_virial_1.data[5*pitch+0], tol_small);


    MY_BOOST_CHECK_CLOSE(h_force_1.data[1].x, 0.189752, tol);
    MY_BOOST_CHECK_SMALL(h_force_1.data[1].y, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_1.data[1].z, tol_small);
    MY_BOOST_CHECK_CLOSE(h_force_1.data[1].w, -0.024571, tol);
    MY_BOOST_CHECK_SMALL(h_virial_1.data[0*pitch+1]
                        +h_virial_1.data[3*pitch+1]
                        +h_virial_1.data[5*pitch+1], tol_small);

    MY_BOOST_CHECK_CLOSE(h_force_1.data[2].x, 0.115629, tol);
    MY_BOOST_CHECK_SMALL(h_force_1.data[2].y, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_1.data[2].z, tol_small);
    MY_BOOST_CHECK_CLOSE(h_force_1.data[2].w, -0.0640261, tol);
    MY_BOOST_CHECK_SMALL(h_virial_1.data[0*pitch+2]
                        +h_virial_1.data[3*pitch+2]
                        +h_virial_1.data[5*pitch+2], tol_small);
    }
    }

#if 0
//! Unit test a comparison between 2 LamellarForceComputes on a "real" system
void lamellar_force_comparison_test(lamellarforce_creator lamellar_creator1, lamellarforce_creator lamellar_creator2, boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    const unsigned int N = 5000;

    // create a random particle system to sum forces on
    RandomInitializer rand_init(N, Scalar(0.2), Scalar(0.9), "A");
    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(rand_init, exec_conf));
    shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    shared_ptr<PotentialExternalLamellar> fc1 = lamellar_creator1(sysdef);
    shared_ptr<PotentialExternalLamellar> fc2 = lamellar_creator2(sysdef);

    unsigned int index = 0;
    Scalar orderParameter = 0.5;
    Scalar interfaceWidth = 0.5;
    unsigned int periodicity = 2;
    fc1->setParams(make_scalar4(__int_as_scalar(index),orderParameter,interfaceWidth,__int_as_scalar(periodicity)));
    fc2->setParams(make_scalar4(__int_as_scalar(index),orderParameter,interfaceWidth,__int_as_scalar(periodicity)));

    // compute the forces
    fc1->compute(0);
    fc2->compute(0);

    {
    // verify that the forces are identical (within roundoff errors)
    GPUArray<Scalar4>& force_array_5 =  fc1->getForceArray();
    GPUArray<Scalar>& virial_array_5 =  fc1->getVirialArray();
    unsigned int pitch = virial_array_5.getPitch();
    ArrayHandle<Scalar4> h_force_5(force_array_5,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_5(virial_array_5,access_location::host,access_mode::read);
    GPUArray<Scalar4>& force_array_6 =  fc2->getForceArray();
    GPUArray<Scalar>& virial_array_6 =  fc2->getVirialArray();
    ArrayHandle<Scalar4> h_force_6(force_array_6,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_6(virial_array_6,access_location::host,access_mode::read);

    // compare average deviation between the two computes
    double deltaf2 = 0.0;
    double deltape2 = 0.0;
    double deltav2[6];
    for (unsigned int i = 0; i < 6; i++)
        deltav2[i] = 0.0;

    for (unsigned int i = 0; i < N; i++)
        {
        deltaf2 += double(h_force_6.data[i].x - h_force_5.data[i].x) * double(h_force_6.data[i].x - h_force_5.data[i].x);
        deltaf2 += double(h_force_6.data[i].y - h_force_5.data[i].y) * double(h_force_6.data[i].y - h_force_5.data[i].y);
        deltaf2 += double(h_force_6.data[i].z - h_force_5.data[i].z) * double(h_force_6.data[i].z - h_force_5.data[i].z);
        deltape2 += double(h_force_6.data[i].w - h_force_5.data[i].w) * double(h_force_6.data[i].w - h_force_5.data[i].w);
        for (unsigned int j = 0; j < 6; j++)
            deltav2[j] += double(h_virial_6.data[j*pitch+i] - h_virial_5.data[j*pitch+i]) * double(h_virial_6.data[j*pitch+i] - h_virial_5.data[j*pitch+i]);

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
#endif

//! LJForceCompute creator for unit tests
shared_ptr<PotentialExternalLamellar> base_class_lamellar_creator(shared_ptr<SystemDefinition> sysdef)
    {
    return shared_ptr<PotentialExternalLamellar>(new PotentialExternalLamellar(sysdef));
    }

#ifdef ENABLE_CUDA
//! LJForceComputeGPU creator for unit tests
shared_ptr<PotentialExternalLamellar> gpu_lamellar_creator(shared_ptr<SystemDefinition> sysdef)
    {
    shared_ptr<PotentialExternalLamellarGPU> lamellar(new PotentialExternalLamellarGPU(sysdef));
    // the default block size kills valgrind :) reduce it
//    lj->setBlockSize(64);
    return lamellar;
    }
#endif

//! boost test case for particle test on CPU
BOOST_AUTO_TEST_CASE( PotentialExternalLamellar_particle )
    {
    lamellarforce_creator lamellar_creator_base = bind(base_class_lamellar_creator, _1);
    lamellar_force_particle_test(lamellar_creator_base, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

# ifdef ENABLE_CUDA
//! boost test case for particle test on GPU
BOOST_AUTO_TEST_CASE( PotentialExternalLamellaGPU_particle )
    {
    lamellarforce_creator lamellar_creator_gpu = bind(gpu_lamellar_creator, _1);
    lamellar_force_particle_test(lamellar_creator_gpu, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

/*
//! boost test case for comparing GPU output to base class output
BOOST_AUTO_TEST_CASE( LJForceGPU_compare )
    {
    ljforce_creator lj_creator_gpu = bind(gpu_lj_creator, _1, _2);
    ljforce_creator lj_creator_base = bind(base_class_lj_creator, _1, _2);
    lj_force_comparison_test(lj_creator_base, lj_creator_gpu, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
*/

#endif
#ifdef WIN32
#pragma warning( pop )
#endif
