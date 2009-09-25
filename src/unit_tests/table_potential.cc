/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
// Maintainer: joaander

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

//! Name the unit test module
#define BOOST_TEST_MODULE TablePotentialTests
#include "boost_utf_configure.h"

#include <fstream>

#include "TablePotential.h"
#include "BinnedNeighborList.h"
#ifdef ENABLE_CUDA
#include "TablePotentialGPU.h"
#endif

using namespace std;
using namespace boost;

/*! \file table_potential_test.cc
    \brief Implements unit tests for TablePotential and descendants
    \ingroup unit_tests
*/

//! Helper macro for testing if two numbers are close
#define MY_BOOST_CHECK_CLOSE(a,b,c) BOOST_CHECK_CLOSE(a,Scalar(b),Scalar(c))
//! Helper macro for testing if a number is small
#define MY_BOOST_CHECK_SMALL(a,c) BOOST_CHECK_SMALL(a,Scalar(c))

//! Tolerance in percent to use for comparing various LJForceComputes to each other
#ifdef SINGLE_PRECISION
const Scalar tol = Scalar(4);
#else
const Scalar tol = 1e-6;
#endif
//! Global tolerance for check_small comparisons
const Scalar tol_small = 1e-4;

//! Typedef'd TablePotential factory
typedef boost::function<shared_ptr<TablePotential> (shared_ptr<SystemDefinition> sysdef, 
                                                    shared_ptr<NeighborList> nlist, 
                                                    unsigned int width)> table_potential_creator;

//! performs some really basic checks on the TablePotential class
void table_potential_basic_test(table_potential_creator table_creator, ExecutionConfiguration exec_conf)
    {
    #ifdef CUDA
    g_gpu_error_checking = true;
    #endif
    
    // perform a basic test to see of the potential and force can be interpolated between two particles
    shared_ptr<SystemDefinition> sysdef_2(new SystemDefinition(2, BoxDim(1000.0), 1, 0, 0, 0, 0, ExecutionConfiguration()));
    shared_ptr<ParticleData> pdata_2 = sysdef_2->getParticleData();
    
    ParticleDataArrays arrays = pdata_2->acquireReadWrite();
    arrays.x[0] = arrays.y[0] = arrays.z[0] = 0.0;
    arrays.x[1] = Scalar(1.0); arrays.y[1] = arrays.z[1] = 0.0;
    pdata_2->release();
    
    shared_ptr<NeighborList> nlist_2(new NeighborList(sysdef_2, Scalar(7.0), Scalar(0.8)));
    shared_ptr<TablePotential> fc_2 = table_creator(sysdef_2, nlist_2, 3);
    
    // first check for proper initialization by seeing if the force and potential come out to be 0
    fc_2->compute(0);
    
    ForceDataArrays force_arrays = fc_2->acquire();
    MY_BOOST_CHECK_SMALL(force_arrays.fx[0], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.fy[0], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[0], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.pe[0], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.virial[0], tol_small);

    MY_BOOST_CHECK_SMALL(force_arrays.fx[1], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.fy[1], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[1], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.pe[1], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.virial[1], tol_small);

    // specify a table to interpolate
    vector<Scalar> V, F;
    V.push_back(10.0);  F.push_back(1.0);
    V.push_back(21.0);  F.push_back(6.0);
    V.push_back(5.0);   F.push_back(2.0);
    fc_2->setTable(0, 0, V, F, 2.0, 4.0);
    
    // compute the forces again and check that they are still 0            
    fc_2->compute(1);
    
    force_arrays = fc_2->acquire();
    MY_BOOST_CHECK_SMALL(force_arrays.fx[0], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.fy[0], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[0], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.pe[0], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.virial[0], tol_small);

    MY_BOOST_CHECK_SMALL(force_arrays.fx[1], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.fy[1], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[1], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.pe[1], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.virial[1], tol_small);
    
    // now go to rmin and check for the correct force value
    arrays = pdata_2->acquireReadWrite();
    arrays.x[1] = Scalar(2.0);
    pdata_2->release();

    fc_2->compute(2);
    
    force_arrays = fc_2->acquire();
    MY_BOOST_CHECK_CLOSE(force_arrays.fx[0], -1.0, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fy[0], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[0], tol_small);
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[0], 5.0, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.virial[0], (1.0 / 6.0) * 2.0, tol);

    MY_BOOST_CHECK_CLOSE(force_arrays.fx[1], 1.0, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fy[1], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[1], tol_small);
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[1], 5.0, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.virial[1], (1.0 / 6.0) * 2.0, tol);
    
    // go halfway in-between two points
    arrays = pdata_2->acquireReadWrite();
    arrays.y[1] = Scalar(3.5);
    arrays.x[1] = Scalar(0.0);
    pdata_2->release();    
    
    // check the forces
    fc_2->compute(3);
    force_arrays = fc_2->acquire();
    MY_BOOST_CHECK_CLOSE(force_arrays.fy[0], -4.0, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fx[0], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[0], tol_small);
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[0], 13.0/2.0, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.virial[0], (1.0 / 6.0) * 4.0 * 3.5, tol);

    MY_BOOST_CHECK_CLOSE(force_arrays.fy[1], 4.0, tol);
    MY_BOOST_CHECK_SMALL(force_arrays.fx[1], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[1], tol_small);
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[1], 13.0 / 2.0, tol);
    MY_BOOST_CHECK_CLOSE(force_arrays.virial[1], (1.0 / 6.0) * 4.0 * 3.5, tol);
    
    // and now check for when r > rmax
    arrays = pdata_2->acquireReadWrite();
    arrays.z[1] = Scalar(4.0);
    pdata_2->release();
    
    // compute and check
    fc_2->compute(4);
    
    force_arrays = fc_2->acquire();
    MY_BOOST_CHECK_SMALL(force_arrays.fx[0], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.fy[0], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[0], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.pe[0], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.virial[0], tol_small);

    MY_BOOST_CHECK_SMALL(force_arrays.fx[1], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.fy[1], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[1], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.pe[1], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.virial[1], tol_small);
    }
 
 //! checks to see if TablePotential correctly handles multiple types
/*void table_potential_type_test(table_potential_creator table_creator, ExecutionConfiguration exec_conf)
    {
    #ifdef CUDA
    g_gpu_error_checking = true;
    #endif
    
    // perform a basic test to see of the potential and force can be interpolated between two particles
    shared_ptr<SystemDefinition> sysdef_3(new SystemDefinition(2, BoxDim(1000.0), 2, 0, 0, 0, 0, ExecutionConfiguration()));
    shared_ptr<ParticleData> pdata_3 = sysdef_2->getParticleData();
    
    ParticleDataArrays arrays = pdata_3->acquireReadWrite();
    arrays.x[0] = arrays.y[0] = arrays.z[0] = 0.0;
    arrays.x[1] = Scalar(1.0); arrays.y[1] = arrays.z[1] = 0.0;
    pdata_2->release();
    
    shared_ptr<NeighborList> nlist_2(new NeighborList(sysdef_2, Scalar(7.0), Scalar(0.8)));
    shared_ptr<TablePotential> fc_2 = table_creator(sysdef_2, nlist_2, 3);
    
    // first check for proper initialization by seeing if the force and potential come out to be 0
    fc_2->compute(0);
    
    ForceDataArrays force_arrays = fc_2->acquire();
    MY_BOOST_CHECK_SMALL(force_arrays.fx[0], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.fy[0], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[0], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.pe[0], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.virial[0], tol_small);

    MY_BOOST_CHECK_SMALL(force_arrays.fx[1], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.fy[1], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[1], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.pe[1], tol_small);
    MY_BOOST_CHECK_SMALL(force_arrays.virial[1], tol_small);

    // specify a table to interpolate
    vector<Scalar> V, F;
    V.push_back(10.0);  F.push_back(1.0);
    V.push_back(21.0);  F.push_back(6.0);
    V.push_back(5.0);   F.push_back(2.0);
    fc_2->setTable(0, 0, V, F, 2.0, 4.0);
    
    }*/
          
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
    table_potential_basic_test(table_creator_base, ExecutionConfiguration(ExecutionConfiguration::CPU));
    }

#ifdef ENABLE_CUDA
//! boost test case for basic test on CPU
BOOST_AUTO_TEST_CASE( TablePotentialGPU_basic )
    {
    table_potential_creator table_creator_gpu = bind(gpu_table_creator, _1, _2, _3);
    table_potential_basic_test(table_creator_gpu, ExecutionConfiguration(ExecutionConfiguration::GPU));
    }
#endif


/*BOOST_AUTO_TEST_CASE(potential_writer)
    {
    #ifdef CUDA
    g_gpu_error_checking = true;
    #endif
    
    // this 2-particle test is just to get a plot of the potential and force vs r cut
    shared_ptr<SystemDefinition> sysdef_2(new SystemDefinition(2, BoxDim(1000.0), 1, 0, 0, 0, 0, ExecutionConfiguration()));
    shared_ptr<ParticleData> pdata_2 = sysdef_2->getParticleData();
    
    ParticleDataArrays arrays = pdata_2->acquireReadWrite();
    arrays.x[0] = arrays.y[0] = arrays.z[0] = 0.0;
    arrays.x[1] = Scalar(0.9); arrays.y[1] = arrays.z[1] = 0.0;
    pdata_2->release();
    shared_ptr<NeighborList> nlist_2(new NeighborList(sysdef_2, Scalar(7.0), Scalar(0.8)));
    nlist_2->setStorageMode(NeighborList::full);
    shared_ptr<TablePotential> fc(new TablePotentialGPU(sysdef_2, nlist_2, 1000));

    // provide a basic potential and "force"
    vector<Scalar> V, F;
    // 5 point test
//    V.push_back(10.0);  F.push_back(-10.0/1.0);
//    V.push_back(15.0);  F.push_back(-15.0/2.0);
//    V.push_back(5.0);   F.push_back(-5.0/3.0);
//    V.push_back(8.0);   F.push_back(-8.0/4.0);
//    V.push_back(18.0);  F.push_back(-18.0/5.0);

    // 1000 point lj test
//    Scalar delta_r = (5.0 - 0.5) / (999);
//    for (unsigned int i = 0; i < 1000; i++)
//        {
//        Scalar r = 0.5 + delta_r * Scalar(i);
//        V.push_back(4.0 * (pow(1.0 / r, 12) - pow(1.0 / r, 6)));
//        F.push_back(4.0 * (12.0 * pow(1.0 / r, 14) - 6 * pow(1.0 / r, 8)));
//        }
        
    // 1000 point gaussian test
    Scalar delta_r = (5.0) / (999);
    for (unsigned int i = 0; i < 1000; i++)
        {
        Scalar r = delta_r * Scalar(i);
        V.push_back(1.5 * expf(-r / 0.5));
        if (r == 0.0)
            F.push_back(0);
        else
            F.push_back(1.5 / 0.5 * expf(-r / 0.5) / r);
        }
    
    fc->setTable(0, 0, V, F, 0.0, 5.0);
    
    ofstream f("table_dat.m");
    f << "table = [";
    unsigned int count = 0;	
    for (float r = 0.0; r <= 5.0; r+= 0.001)
        {
        // set the distance
        ParticleDataArrays arrays = pdata_2->acquireReadWrite();
        arrays.x[0] = arrays.y[0] = arrays.z[0] = 0.0;
        arrays.x[1] = Scalar(r); arrays.y[1] = arrays.z[1] = 0.0;
        pdata_2->release();
        
        // compute the forces
        fc->compute(count);
        count++;
    
        ForceDataArrays force_arrays = fc->acquire();
        f << r << " " << force_arrays.fx[0] << " " << fc->calcEnergySum() << " ; " << endl;	
        }
    f << "];" << endl;
    f.close();
    }*/
    
#ifdef WIN32
#pragma warning( pop )
#endif