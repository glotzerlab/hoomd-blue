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



#include <boost/python.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include <iostream>
#include <fstream>
                                                                       
#include "PPPMForceCompute.h"
#ifdef ENABLE_CUDA
#include "PPPMForceComputeGPU.h"
#endif 

#include "NeighborListBinned.h"
#include "Initializers.h"

#include <math.h>

using namespace std;
using namespace boost;
using namespace boost::python;

/*! \file pppm_force_test.cc
    \brief Implements unit tests for PPPMForceCompute and PPPMForceComputeGPU and descendants
    \ingroup unit_tests
*/

//! Name the unit test module
#define BOOST_TEST_MODULE PPPMTest
#include "boost_utf_configure.h"

//! Typedef'd PPPMForceCompute factory
 
typedef boost::function<shared_ptr<PPPMForceCompute> (shared_ptr<SystemDefinition> sysdef,
                                                      shared_ptr<NeighborList> nlist, 
                                                      shared_ptr<ParticleGroup> group)> pppmforce_creator;
 
//! Test the ability of the lj force compute to actually calucate forces
void pppm_force_particle_test(pppmforce_creator pppm_creator, boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // this is a 2-particle of charge 1 and -1 
    // due to the complexity of FFTs, the correct resutls are not analytically computed
    // but instead taken from a known working implementation of the PPPM method
    // The box lengths and grid points are different in each direction
    
    shared_ptr<SystemDefinition> sysdef_2(new SystemDefinition(2, BoxDim(6.0, 10.0, 14.0), 1, 0, 0, 0, 0, exec_conf));
    shared_ptr<ParticleData> pdata_2 = sysdef_2->getParticleData();
    
    shared_ptr<NeighborList> nlist_2(new NeighborList(sysdef_2, Scalar(1.0), Scalar(1.0)));
    shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef_2, 0, 1));
    shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef_2, selector_all));

    {
    ArrayHandle<Scalar4> h_pos(pdata_2->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_charge(pdata_2->getCharges(), access_location::host, access_mode::readwrite);

    h_pos.data[0].x = h_pos.data[0].y = h_pos.data[0].z = 1.0;
    h_charge.data[0] = 1.0;
    h_pos.data[1].x = h_pos.data[1].y = h_pos.data[1].z = 2.0;
    h_charge.data[1] = -1.0;

    }

    shared_ptr<PPPMForceCompute> fc_2 = pppm_creator(sysdef_2, nlist_2, group_all);
    

    // first test: setup a sigma of 1.0 so that all forces will be 0
    int Nx = 10;
    int Ny = 15; 
    int Nz = 24;
    int order = 5;
    Scalar kappa = 1.0;
    Scalar rcut = 1.0;
    fc_2->setParams(Nx, Ny, Nz, order, kappa, rcut);
    
    // compute the forces
    fc_2->compute(0);
    
    ArrayHandle<Scalar4> h_force(fc_2->getForceArray(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_virial(fc_2->getVirialArray(), access_location::host, access_mode::read);
    unsigned int pitch = fc_2->getVirialArray().getPitch();

    MY_BOOST_CHECK_CLOSE(h_force.data[0].x, 0.151335f, tol_small);
    MY_BOOST_CHECK_CLOSE(h_force.data[0].y, 0.172246f, tol_small);
    MY_BOOST_CHECK_CLOSE(h_force.data[0].z, 0.179186f, tol_small);
    MY_BOOST_CHECK_SMALL(h_force.data[0].w, tol_small);
    MY_BOOST_CHECK_SMALL(h_virial.data[0*pitch+0]
                        +h_virial.data[3*pitch+0]
                        +h_virial.data[5*pitch+0], tol_small);
    
    MY_BOOST_CHECK_CLOSE(h_force.data[1].x, -0.151335f, tol_small);
    MY_BOOST_CHECK_CLOSE(h_force.data[1].y, -0.172246f, tol_small);
    MY_BOOST_CHECK_CLOSE(h_force.data[1].z, -0.179186f, tol_small);
    MY_BOOST_CHECK_SMALL(h_force.data[1].w, tol_small);
    MY_BOOST_CHECK_SMALL(h_virial.data[0*pitch+1]
                        +h_virial.data[3*pitch+1]
                        +h_virial.data[5*pitch+1], tol_small);

    }

//! PPPMForceCompute creator for unit tests
shared_ptr<PPPMForceCompute> base_class_pppm_creator(shared_ptr<SystemDefinition> sysdef,
                                                     shared_ptr<NeighborList> nlist,
                                                     shared_ptr<ParticleGroup> group)
    {
    return shared_ptr<PPPMForceCompute>(new PPPMForceCompute(sysdef, nlist, group));
    }

#ifdef ENABLE_CUDA
//! PPPMForceComputeGPU creator for unit tests
shared_ptr<PPPMForceCompute> gpu_pppm_creator(shared_ptr<SystemDefinition> sysdef,
                                              shared_ptr<NeighborList> nlist,
                                              shared_ptr<ParticleGroup> group)
    {
    nlist->setStorageMode(NeighborList::full);
    return shared_ptr<PPPMForceComputeGPU> (new PPPMForceComputeGPU(sysdef, nlist, group));
    }
#endif


//! boost test case for particle test on CPU
BOOST_AUTO_TEST_CASE( PPPMForceCompute_basic )
    {
    pppmforce_creator pppm_creator = bind(base_class_pppm_creator, _1, _2, _3);
    pppm_force_particle_test(pppm_creator, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

#ifdef ENABLE_CUDA
//! boost test case for bond forces on the GPU
BOOST_AUTO_TEST_CASE( PPPMForceComputeGPU_basic )
    {
    pppmforce_creator pppm_creator = bind(gpu_pppm_creator, _1, _2, _3);
    pppm_force_particle_test(pppm_creator, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
#endif

#ifdef WIN32
#pragma warning( pop )
#endif

