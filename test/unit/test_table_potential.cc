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

#include <fstream>

#include "TablePotential.h"
#include "NeighborList.h"
#ifdef ENABLE_CUDA
#include "TablePotentialGPU.h"
#endif

using namespace std;
using namespace boost;

//! Name the unit test module
#define BOOST_TEST_MODULE TablePotentialTests
#include "boost_utf_configure.h"

/*! \file table_potential.cc
    \brief Implements unit tests for TablePotential and descendants
    \ingroup unit_tests
*/

//! Typedef'd TablePotential factory
typedef boost::function<shared_ptr<TablePotential> (shared_ptr<SystemDefinition> sysdef,
                                                    shared_ptr<NeighborList> nlist,
                                                    unsigned int width)> table_potential_creator;

//! performs some really basic checks on the TablePotential class
void table_potential_basic_test(table_potential_creator table_creator, boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // perform a basic test to see of the potential and force can be interpolated between two particles
    shared_ptr<SystemDefinition> sysdef_2(new SystemDefinition(2, BoxDim(1000.0), 1, 0, 0, 0, 0, exec_conf));
    shared_ptr<ParticleData> pdata_2 = sysdef_2->getParticleData();
    
    ParticleDataArrays arrays = pdata_2->acquireReadWrite();
    arrays.x[0] = arrays.y[0] = arrays.z[0] = 0.0;
    arrays.x[1] = Scalar(1.0); arrays.y[1] = arrays.z[1] = 0.0;
    pdata_2->release();
    
    shared_ptr<NeighborList> nlist_2(new NeighborList(sysdef_2, Scalar(7.0), Scalar(0.8)));
    shared_ptr<TablePotential> fc_2 = table_creator(sysdef_2, nlist_2, 3);
    
    // first check for proper initialization by seeing if the force and potential come out to be 0
    fc_2->compute(0);
    
    {
    GPUArray<Scalar4>& force_array_1 =  fc_2->getForceArray();
    GPUArray<Scalar>& virial_array_1 =  fc_2->getVirialArray();
    unsigned int pitch = virial_array_1.getPitch();
    ArrayHandle<Scalar4> h_force_1(force_array_1,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_1(virial_array_1,access_location::host,access_mode::read);
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
    
    MY_BOOST_CHECK_SMALL(h_force_1.data[1].x, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_1.data[1].y, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_1.data[1].z, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_1.data[1].w, tol_small);
    MY_BOOST_CHECK_SMALL(h_virial_1.data[0*pitch+1], tol_small);
    MY_BOOST_CHECK_SMALL(h_virial_1.data[1*pitch+1], tol_small);
    MY_BOOST_CHECK_SMALL(h_virial_1.data[2*pitch+1], tol_small);
    MY_BOOST_CHECK_SMALL(h_virial_1.data[3*pitch+1], tol_small);
    MY_BOOST_CHECK_SMALL(h_virial_1.data[4*pitch+1], tol_small);
    MY_BOOST_CHECK_SMALL(h_virial_1.data[5*pitch+1], tol_small);
    }

    // specify a table to interpolate
    vector<float> V, F;
    V.push_back(10.0);  F.push_back(1.0);
    V.push_back(21.0);  F.push_back(6.0);
    V.push_back(5.0);   F.push_back(2.0);
    fc_2->setTable(0, 0, V, F, 2.0, 4.0);
    
    // compute the forces again and check that they are still 0
    fc_2->compute(1);
    
    {
    GPUArray<Scalar4>& force_array_2 =  fc_2->getForceArray();
    GPUArray<Scalar>& virial_array_2 =  fc_2->getVirialArray();
    unsigned int pitch = virial_array_2.getPitch();
    ArrayHandle<Scalar4> h_force_2(force_array_2,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_2(virial_array_2,access_location::host,access_mode::read);
    MY_BOOST_CHECK_SMALL(h_force_2.data[0].x, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_2.data[0].y, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_2.data[0].z, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_2.data[0].w, tol_small);
    MY_BOOST_CHECK_SMALL(h_virial_2.data[0*pitch+0], tol_small);
    MY_BOOST_CHECK_SMALL(h_virial_2.data[1*pitch+0], tol_small);
    MY_BOOST_CHECK_SMALL(h_virial_2.data[2*pitch+0], tol_small);
    MY_BOOST_CHECK_SMALL(h_virial_2.data[3*pitch+0], tol_small);
    MY_BOOST_CHECK_SMALL(h_virial_2.data[4*pitch+0], tol_small);
    MY_BOOST_CHECK_SMALL(h_virial_2.data[5*pitch+0], tol_small);
    
    MY_BOOST_CHECK_SMALL(h_force_2.data[1].x, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_2.data[1].y, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_2.data[1].z, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_2.data[1].w, tol_small);
    MY_BOOST_CHECK_SMALL(h_virial_2.data[0*pitch+1], tol_small);
    MY_BOOST_CHECK_SMALL(h_virial_2.data[1*pitch+1], tol_small);
    MY_BOOST_CHECK_SMALL(h_virial_2.data[2*pitch+1], tol_small);
    MY_BOOST_CHECK_SMALL(h_virial_2.data[3*pitch+1], tol_small);
    MY_BOOST_CHECK_SMALL(h_virial_2.data[4*pitch+1], tol_small);
    MY_BOOST_CHECK_SMALL(h_virial_2.data[5*pitch+1], tol_small);

    }

    // now go to rmin and check for the correct force value
    arrays = pdata_2->acquireReadWrite();
    arrays.x[1] = Scalar(2.0);
    pdata_2->release();
    
    fc_2->compute(2);
    
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
    arrays = pdata_2->acquireReadWrite();
    arrays.y[1] = Scalar(3.5);
    arrays.x[1] = Scalar(0.0);
    pdata_2->release();
    
    // check the forces
    fc_2->compute(3);
    
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

    // and now check for when r > rmax
    arrays = pdata_2->acquireReadWrite();
    arrays.z[1] = Scalar(4.0);
    pdata_2->release();
    
    // compute and check
    fc_2->compute(4);
    
    {
    GPUArray<Scalar4>& force_array_5 =  fc_2->getForceArray();
    GPUArray<Scalar>& virial_array_5 =  fc_2->getVirialArray();
    unsigned int pitch = virial_array_5.getPitch();
    ArrayHandle<Scalar4> h_force_5(force_array_5,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_5(virial_array_5,access_location::host,access_mode::read);
    MY_BOOST_CHECK_SMALL(h_force_5.data[0].x, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_5.data[0].y, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_5.data[0].z, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_5.data[0].w, tol_small);
    MY_BOOST_CHECK_SMALL(h_virial_5.data[0*pitch+0], tol_small);
    MY_BOOST_CHECK_SMALL(h_virial_5.data[1*pitch+0], tol_small);
    MY_BOOST_CHECK_SMALL(h_virial_5.data[2*pitch+0], tol_small);
    MY_BOOST_CHECK_SMALL(h_virial_5.data[3*pitch+0], tol_small);
    MY_BOOST_CHECK_SMALL(h_virial_5.data[4*pitch+0], tol_small);
    MY_BOOST_CHECK_SMALL(h_virial_5.data[5*pitch+0], tol_small);
    
    MY_BOOST_CHECK_SMALL(h_force_5.data[1].x, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_5.data[1].y, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_5.data[1].z, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_5.data[1].w, tol_small);
    MY_BOOST_CHECK_SMALL(h_virial_5.data[0*pitch+1], tol_small);
    MY_BOOST_CHECK_SMALL(h_virial_5.data[1*pitch+1], tol_small);
    MY_BOOST_CHECK_SMALL(h_virial_5.data[2*pitch+1], tol_small);
    MY_BOOST_CHECK_SMALL(h_virial_5.data[3*pitch+1], tol_small);
    MY_BOOST_CHECK_SMALL(h_virial_5.data[4*pitch+1], tol_small);
    MY_BOOST_CHECK_SMALL(h_virial_5.data[5*pitch+1], tol_small);
    }
    }

//! checks to see if TablePotential correctly handles multiple types
void table_potential_type_test(table_potential_creator table_creator, boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // perform a basic test to see of the potential and force can be interpolated between two particles
    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(4, BoxDim(1000.0), 2, 0, 0, 0, 0, exec_conf));
    shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    
    ParticleDataArrays arrays = pdata->acquireReadWrite();
    arrays.x[0] = arrays.y[0] = arrays.z[0] = 0.0; arrays.type[0] = 0;
    arrays.x[1] = Scalar(1.5); arrays.y[1] = arrays.z[1] = 0.0; arrays.type[1] = 1;
    arrays.x[2] = 0.0; arrays.y[2] = Scalar(1.5); arrays.z[2] = 0.0; arrays.type[2] = 0;
    arrays.x[3] = Scalar(1.5); arrays.y[3] = Scalar(1.5); arrays.z[3] = 0.0; arrays.type[3] = 1;
    pdata->release();
    
    shared_ptr<NeighborList> nlist(new NeighborList(sysdef, Scalar(2.0), Scalar(0.8)));
    shared_ptr<TablePotential> fc = table_creator(sysdef, nlist, 3);
    
    // specify a table to interpolate
    vector<float> V, F;
    V.push_back(10.0);  F.push_back(1.0);
    V.push_back(20.0);  F.push_back(6.0);
    V.push_back(5.0);   F.push_back(2.0);
    fc->setTable(0, 0, V, F, 1.0, 2.0);
    
    // next type pair
    V.clear(); F.clear();
    V.push_back(20.0);  F.push_back(2.0);
    V.push_back(40.0);  F.push_back(12.0);
    V.push_back(10.0);   F.push_back(4.0);
    fc->setTable(0, 1, V, F, 0.0, 2.0);
    
    // next type pair
    V.clear(); F.clear();
    V.push_back(5.0);  F.push_back(0.5);
    V.push_back(10.0);  F.push_back(3.0);
    V.push_back(2.5);   F.push_back(2.0);
    fc->setTable(1, 1, V, F, 1.0, 2.0);
    
    // compute and check
    fc->compute(0);
    
    {
    GPUArray<Scalar4>& force_array_6 =  fc->getForceArray();
    GPUArray<Scalar>& virial_array_6 =  fc->getVirialArray();
    unsigned int pitch = virial_array_6.getPitch();
    ArrayHandle<Scalar4> h_force_6(force_array_6,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_6(virial_array_6,access_location::host,access_mode::read);
    MY_BOOST_CHECK_CLOSE(h_force_6.data[0].x, -8.0, tol);
    MY_BOOST_CHECK_CLOSE(h_force_6.data[0].y, -6.0, tol);
    MY_BOOST_CHECK_SMALL(h_force_6.data[0].z, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_6.data[0].w, 10.0+25.0);
    MY_BOOST_CHECK_CLOSE(Scalar(1./3.)*(h_virial_6.data[0*pitch+0]
                                       +h_virial_6.data[3*pitch+0]
                                       +h_virial_6.data[5*pitch+0]), (8*1.5+6*1.5)*1.0/6.0, tol);
    
    MY_BOOST_CHECK_CLOSE(h_force_6.data[1].x, 8.0, tol);
    MY_BOOST_CHECK_CLOSE(h_force_6.data[1].y, -3.0, tol);
    MY_BOOST_CHECK_SMALL(h_force_6.data[1].z, tol_small);
    MY_BOOST_CHECK_CLOSE(h_force_6.data[1].w, 25.0/2.0 + 5.0, tol);
    MY_BOOST_CHECK_CLOSE(Scalar(1./3.)*(h_virial_6.data[0*pitch+1]
                                       +h_virial_6.data[3*pitch+1]
                                       +h_virial_6.data[5*pitch+1]), (8*1.5 + 3.0 * 1.5)*1.0/6.0, tol);
    
    MY_BOOST_CHECK_CLOSE(h_force_6.data[2].x, -8.0, tol);
    MY_BOOST_CHECK_CLOSE(h_force_6.data[2].y, 6.0, tol);
    MY_BOOST_CHECK_SMALL(h_force_6.data[2].z, tol_small);
    MY_BOOST_CHECK_SMALL(h_force_6.data[2].w, 10.0+25.0);
    MY_BOOST_CHECK_CLOSE(Scalar(1./3.)*(h_virial_6.data[0*pitch+2]
                                       +h_virial_6.data[3*pitch+2]
                                       +h_virial_6.data[5*pitch+2]), (8*1.5+6*1.5)*1.0/6.0, tol);
    
    MY_BOOST_CHECK_CLOSE(h_force_6.data[3].x, 8.0, tol);
    MY_BOOST_CHECK_CLOSE(h_force_6.data[3].y, 3.0, tol);
    MY_BOOST_CHECK_SMALL(h_force_6.data[3].z, tol_small);
    MY_BOOST_CHECK_CLOSE(h_force_6.data[3].w, 25.0/2.0 + 5.0, tol);
    MY_BOOST_CHECK_CLOSE(Scalar(1./3.)*(h_virial_6.data[0*pitch+3]
                                       +h_virial_6.data[3*pitch+3]
                                       +h_virial_6.data[5*pitch+3]), (8*1.5 + 3.0*1.5)*1.0/6.0, tol);
    }
     }

//! TablePotential creator for unit tests
shared_ptr<TablePotential> base_class_table_creator(shared_ptr<SystemDefinition> sysdef,
                                                    shared_ptr<NeighborList> nlist,
                                                    unsigned int width)
    {
    return shared_ptr<TablePotential>(new TablePotential(sysdef, nlist, width));
    }

#ifdef ENABLE_CUDA
//! TablePotentialGPU creator for unit tests
shared_ptr<TablePotential> gpu_table_creator(shared_ptr<SystemDefinition> sysdef,
                                             shared_ptr<NeighborList> nlist,
                                             unsigned int width)
    {
    nlist->setStorageMode(NeighborList::full);
    shared_ptr<TablePotentialGPU> table(new TablePotentialGPU(sysdef, nlist, width));
    // the default block size kills valgrind :) reduce it
    table->setBlockSize(64);
    return table;
    }
#endif


//! boost test case for basic test on CPU
BOOST_AUTO_TEST_CASE( TablePotential_basic )
    {
    table_potential_creator table_creator_base = bind(base_class_table_creator, _1, _2, _3);
    table_potential_basic_test(table_creator_base, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

//! boost test case for type test on CPU
BOOST_AUTO_TEST_CASE( TablePotential_type )
    {
    table_potential_creator table_creator_base = bind(base_class_table_creator, _1, _2, _3);
    table_potential_type_test(table_creator_base, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

#ifdef ENABLE_CUDA
//! boost test case for basic test on GPU
BOOST_AUTO_TEST_CASE( TablePotentialGPU_basic )
    {
    table_potential_creator table_creator_gpu = bind(gpu_table_creator, _1, _2, _3);
    table_potential_basic_test(table_creator_gpu, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! boost test case for type test on GPU
BOOST_AUTO_TEST_CASE( TablePotentialGPU_type )
    {
    table_potential_creator table_creator_gpu = bind(gpu_table_creator, _1, _2, _3);
    table_potential_type_test(table_creator_gpu, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
#endif

#ifdef WIN32
#pragma warning( pop )
#endif

