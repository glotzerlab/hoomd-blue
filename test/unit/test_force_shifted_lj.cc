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
#include <fstream>

#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include "AllPairPotentials.h"

#include "NeighborListBinned.h"
#include "Initializers.h"

#include <math.h>

using namespace std;
using namespace boost;

/*! \file fslj_force_test.cc
    \brief Implements unit tests for PotentialPairForceShiftedLJ and PotentialPairForceShiftedLJGPU and descendants
    \ingroup unit_tests
*/

//! Name the unit test module
#define BOOST_TEST_MODULE PotentialPairForceShiftedLJTests
#include "boost_utf_configure.h"

//! Typedef'd LJForceCompute factory
typedef boost::function<boost::shared_ptr<PotentialPairForceShiftedLJ> (boost::shared_ptr<SystemDefinition> sysdef,
                                                     boost::shared_ptr<NeighborList> nlist)> ljforce_creator;

//! Test the ability of the lj force compute to actually calucate forces
void fslj_force_particle_test(ljforce_creator lj_creator, boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    boost::shared_ptr<SystemDefinition> sysdef_2(new SystemDefinition(2, BoxDim(1000.0), 1, 0, 0, 0, 0, exec_conf));
    boost::shared_ptr<ParticleData> pdata_2 = sysdef_2->getParticleData();
    pdata_2->setFlags(~PDataFlags(0));

    pdata_2->setPosition(0, make_scalar3(0.0,0.0,0.0));
    pdata_2->setPosition(1, make_scalar3(1.3,0.0,0.0));

    boost::shared_ptr<NeighborList> nlist_2(new NeighborList(sysdef_2, Scalar(1.5), Scalar(0.4)));
    boost::shared_ptr<PotentialPairForceShiftedLJ> fc_no_shift = lj_creator(sysdef_2, nlist_2);
    fc_no_shift->setRcut(0, 0, Scalar(1.5));
    fc_no_shift->setShiftMode(PotentialPairForceShiftedLJ::no_shift);

    boost::shared_ptr<PotentialPairForceShiftedLJ> fc_shift = lj_creator(sysdef_2, nlist_2);
    fc_shift->setRcut(0, 0, Scalar(1.5));
    fc_shift->setShiftMode(PotentialPairForceShiftedLJ::shift);

    // setup a standard epsilon and sigma
    Scalar epsilon = Scalar(1.0);
    Scalar sigma = Scalar(1.0);
    Scalar alpha = Scalar(1.0);
    Scalar lj1 = Scalar(4.0) * epsilon * pow(sigma,Scalar(12.0));
    Scalar lj2 = alpha * Scalar(4.0) * epsilon * pow(sigma,Scalar(6.0));
    fc_no_shift->setParams(0,0,make_scalar2(lj1,lj2));
    fc_shift->setParams(0,0,make_scalar2(lj1,lj2));

    fc_no_shift->compute(0);
    fc_shift->compute(0);


    {
    GPUArray<Scalar4>& force_array_1 =  fc_no_shift->getForceArray();
    GPUArray<Scalar>& virial_array_1 =  fc_no_shift->getVirialArray();
    ArrayHandle<Scalar4> h_force_1(force_array_1,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_1(virial_array_1,access_location::host,access_mode::read);

    MY_BOOST_CHECK_CLOSE(h_force_1.data[0].x,  1.0819510987449876 , tol);
    MY_BOOST_CHECK_CLOSE(h_force_1.data[0].w, -0.21270557412540803, tol);
    MY_BOOST_CHECK_CLOSE(h_force_1.data[1].x, -1.0819510987449876, tol);
    MY_BOOST_CHECK_CLOSE(h_force_1.data[1].w, -0.21270557412540803, tol);
    }

    {
    GPUArray<Scalar4>& force_array_2 =  fc_shift->getForceArray();
    GPUArray<Scalar>& virial_array_2 =  fc_shift->getVirialArray();
    ArrayHandle<Scalar4> h_force_2(force_array_2,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_2(virial_array_2,access_location::host,access_mode::read);
    MY_BOOST_CHECK_CLOSE(h_force_2.data[0].x,  1.0819510987449876 , tol);
    MY_BOOST_CHECK_CLOSE(h_force_2.data[0].w, -0.05253727698612069, tol);
    MY_BOOST_CHECK_CLOSE(h_force_2.data[1].x, -1.0819510987449876, tol);
    MY_BOOST_CHECK_CLOSE(h_force_2.data[1].w, -0.05253727698612069, tol);
    }

    }

//! LJForceCompute creator for unit tests
boost::shared_ptr<PotentialPairForceShiftedLJ> base_class_lj_creator(boost::shared_ptr<SystemDefinition> sysdef,
                                                  boost::shared_ptr<NeighborList> nlist)
    {
    return boost::shared_ptr<PotentialPairForceShiftedLJ>(new PotentialPairForceShiftedLJ(sysdef, nlist));
    }

#ifdef ENABLE_CUDA
//! LJForceComputeGPU creator for unit tests
boost::shared_ptr<PotentialPairForceShiftedLJGPU> gpu_lj_creator(boost::shared_ptr<SystemDefinition> sysdef,
                                          boost::shared_ptr<NeighborList> nlist)
    {
    nlist->setStorageMode(NeighborList::full);
    boost::shared_ptr<PotentialPairForceShiftedLJGPU> lj(new PotentialPairForceShiftedLJGPU(sysdef, nlist));
    return lj;
    }
#endif

//! boost test case for particle test on CPU
BOOST_AUTO_TEST_CASE( PotentialPairForceShiftedLJ_particle )
    {
    ljforce_creator lj_creator_base = bind(base_class_lj_creator, _1, _2);
    fslj_force_particle_test(lj_creator_base, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

# ifdef ENABLE_CUDA
//! boost test case for particle test on GPU
BOOST_AUTO_TEST_CASE( LJForceGPU_particle )
    {
    ljforce_creator lj_creator_gpu = bind(gpu_lj_creator, _1, _2);
    fslj_force_particle_test(lj_creator_gpu, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
#endif

#ifdef WIN32
#pragma warning( pop )
#endif
