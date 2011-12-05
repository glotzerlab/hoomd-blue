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


/*! \file pdata_test.cc
    \brief Unit tests for BoxDim, ParticleData, SimpleCubicInitializer, and RandomInitializer classes.
    \ingroup unit_tests
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <iostream>

#include <boost/bind.hpp>

#include "ParticleData.h"
#include "Initializers.h"

using namespace std;
using namespace boost;

//! Name the boost unit test module
#define BOOST_TEST_MODULE ParticleDataTests
#include "boost_utf_configure.h"

//! Perform some basic tests on the boxdim structure
BOOST_AUTO_TEST_CASE( BoxDim_test )
    {
    Scalar tol = Scalar(1e-6);
    
    // test default constructor
    BoxDim a;
    MY_BOOST_CHECK_CLOSE(a.xlo,0.0, tol);
    MY_BOOST_CHECK_CLOSE(a.ylo,0.0, tol);
    MY_BOOST_CHECK_CLOSE(a.zlo,0.0, tol);
    MY_BOOST_CHECK_CLOSE(a.xhi,0.0, tol);
    MY_BOOST_CHECK_CLOSE(a.yhi,0.0, tol);
    MY_BOOST_CHECK_CLOSE(a.zhi,0.0, tol);
    
    BoxDim b(10.0);
    MY_BOOST_CHECK_CLOSE(b.xlo,-5.0, tol);
    MY_BOOST_CHECK_CLOSE(b.ylo,-5.0, tol);
    MY_BOOST_CHECK_CLOSE(b.zlo,-5.0, tol);
    MY_BOOST_CHECK_CLOSE(b.xhi,5.0, tol);
    MY_BOOST_CHECK_CLOSE(b.yhi,5.0, tol);
    MY_BOOST_CHECK_CLOSE(b.zhi,5.0, tol);
    
    BoxDim c(10.0, 30.0, 50.0);
    MY_BOOST_CHECK_CLOSE(c.xlo,-5.0, tol);
    MY_BOOST_CHECK_CLOSE(c.ylo,-15.0, tol);
    MY_BOOST_CHECK_CLOSE(c.zlo,-25.0, tol);
    MY_BOOST_CHECK_CLOSE(c.xhi,5.0, tol);
    MY_BOOST_CHECK_CLOSE(c.yhi,15.0, tol);
    MY_BOOST_CHECK_CLOSE(c.zhi,25.0, tol);
    
    // test for assignment and copy constructor
    BoxDim d(c);
    MY_BOOST_CHECK_CLOSE(d.xlo,-5.0, tol);
    MY_BOOST_CHECK_CLOSE(d.ylo,-15.0, tol);
    MY_BOOST_CHECK_CLOSE(d.zlo,-25.0, tol);
    MY_BOOST_CHECK_CLOSE(d.xhi,5.0, tol);
    MY_BOOST_CHECK_CLOSE(d.yhi,15.0, tol);
    MY_BOOST_CHECK_CLOSE(d.zhi,25.0, tol);
    
    BoxDim e;
    e = c;
    MY_BOOST_CHECK_CLOSE(e.xlo,-5.0, tol);
    MY_BOOST_CHECK_CLOSE(e.ylo,-15.0, tol);
    MY_BOOST_CHECK_CLOSE(e.zlo,-25.0, tol);
    MY_BOOST_CHECK_CLOSE(e.xhi,5.0, tol);
    MY_BOOST_CHECK_CLOSE(e.yhi,15.0, tol);
    MY_BOOST_CHECK_CLOSE(e.zhi,25.0, tol);
    
    b = b;
    MY_BOOST_CHECK_CLOSE(b.xlo,-5.0, tol);
    MY_BOOST_CHECK_CLOSE(b.ylo,-5.0, tol);
    MY_BOOST_CHECK_CLOSE(b.zlo,-5.0, tol);
    MY_BOOST_CHECK_CLOSE(b.xhi,5.0, tol);
    MY_BOOST_CHECK_CLOSE(b.yhi,5.0, tol);
    MY_BOOST_CHECK_CLOSE(b.zhi,5.0, tol);
    }

//! Test operation of the particle data class
BOOST_AUTO_TEST_CASE( ParticleData_test )
    {
    BoxDim box(10.0, 30.0, 50.0);
    boost::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::CPU));
    ParticleData a(1, box, 1, exec_conf);
    
    Scalar tol = Scalar(1e-6);
    
    // make sure the box is working
    const BoxDim& c = a.getBox();
    MY_BOOST_CHECK_CLOSE(c.xlo,-5.0, tol);
    MY_BOOST_CHECK_CLOSE(c.ylo,-15.0, tol);
    MY_BOOST_CHECK_CLOSE(c.zlo,-25.0, tol);
    MY_BOOST_CHECK_CLOSE(c.xhi,5.0, tol);
    MY_BOOST_CHECK_CLOSE(c.yhi,15.0, tol);
    MY_BOOST_CHECK_CLOSE(c.zhi,25.0, tol);
    
    BoxDim box2(5.0, 5.0, 5.0);
    a.setBox(box2);
    const BoxDim& d = a.getBox();
    MY_BOOST_CHECK_CLOSE(d.xlo,-2.5, tol);
    MY_BOOST_CHECK_CLOSE(d.ylo,-2.5, tol);
    MY_BOOST_CHECK_CLOSE(d.zlo,-2.5, tol);
    MY_BOOST_CHECK_CLOSE(d.xhi,2.5, tol);
    MY_BOOST_CHECK_CLOSE(d.yhi,2.5, tol);
    MY_BOOST_CHECK_CLOSE(d.zhi,2.5, tol);
    
    // make sure that getN is working
    BOOST_CHECK(a.getN() == 1);
    
    // Test the ability to acquire data
    ParticleDataArrays arrays = a.acquireReadWrite();
    // begin by verifying that the defaults the class adversizes are set
    BOOST_CHECK(arrays.nparticles == 1);
    MY_BOOST_CHECK_CLOSE(arrays.x[0], 0.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays.y[0], 0.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays.z[0], 0.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays.vx[0], 0.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays.vy[0], 0.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays.vz[0], 0.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays.ax[0], 0.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays.ay[0], 0.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays.az[0], 0.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays.charge[0], 0.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays.mass[0], 1.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays.diameter[0], 1.0, tol);
    BOOST_CHECK_EQUAL(arrays.ix[0], 0);
    BOOST_CHECK_EQUAL(arrays.iy[0], 0);
    BOOST_CHECK_EQUAL(arrays.iz[0], 0);
    BOOST_CHECK(arrays.type[0] == 0);
    BOOST_CHECK(arrays.rtag[0] == 0);
    BOOST_CHECK(arrays.tag[0] == 0);
    BOOST_CHECK(arrays.body[0] == NO_BODY);
    
    // set some new values for testing
    arrays.x[0] = 1.0;
    arrays.y[0] = 2.0;
    arrays.z[0] = -2.0;
    arrays.vx[0] = 11.0;
    arrays.vy[0] = 12.0;
    arrays.vz[0] = 13.0;
    arrays.ax[0] = 21.0;
    arrays.ay[0] = 22.0;
    arrays.az[0] = 23.0;
    arrays.charge[0] = 24.0;
    arrays.mass[0] = 25.0;
    arrays.diameter[0] = 26.0;
    arrays.ix[0] =  27;
    arrays.iy[0] = 28;
    arrays.iz[0] = 29;
    arrays.type[0] = 1;
    arrays.body[0] = 0;
    
    a.release();
    
    // make sure when the data is re-acquired, the values read properly
    ParticleDataArraysConst arrays_const = a.acquireReadOnly();
    BOOST_CHECK(arrays_const.nparticles == 1);
    MY_BOOST_CHECK_CLOSE(arrays_const.x[0], 1.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays_const.y[0], 2.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays_const.z[0], -2.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays_const.vx[0], 11.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays_const.vy[0], 12.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays_const.vz[0], 13.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays_const.ax[0], 21.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays_const.ay[0], 22.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays_const.az[0], 23.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays_const.charge[0], 24.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays_const.mass[0], 25.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays_const.diameter[0], 26.0, tol);
    BOOST_CHECK_EQUAL(arrays_const.ix[0], 27);
    BOOST_CHECK_EQUAL(arrays_const.iy[0], 28);
    BOOST_CHECK_EQUAL(arrays_const.iz[0], 29);
    BOOST_CHECK(arrays_const.type[0] == 1);
    BOOST_CHECK(arrays_const.rtag[0] == 0);
    BOOST_CHECK(arrays_const.tag[0] == 0);
    BOOST_CHECK(arrays_const.body[0] == 0);
    
    a.release();
    
    // finally, lets check a larger ParticleData for correctness of the initialization
    const unsigned int N = 1000;
    ParticleData b(N, box, 1, exec_conf);
    arrays_const = b.acquireReadOnly();
    BOOST_CHECK(arrays_const.nparticles == N);
    for (unsigned int i = 0; i < N; i++)
        {
        MY_BOOST_CHECK_CLOSE(arrays_const.x[i], 0.0, tol);
        MY_BOOST_CHECK_CLOSE(arrays_const.y[i], 0.0, tol);
        MY_BOOST_CHECK_CLOSE(arrays_const.z[i], 0.0, tol);
        MY_BOOST_CHECK_CLOSE(arrays_const.vx[i], 0.0, tol);
        MY_BOOST_CHECK_CLOSE(arrays_const.vy[i], 0.0, tol);
        MY_BOOST_CHECK_CLOSE(arrays_const.vz[i], 0.0, tol);
        MY_BOOST_CHECK_CLOSE(arrays_const.ax[i], 0.0, tol);
        MY_BOOST_CHECK_CLOSE(arrays_const.ay[i], 0.0, tol);
        MY_BOOST_CHECK_CLOSE(arrays_const.az[i], 0.0, tol);
        MY_BOOST_CHECK_CLOSE(arrays_const.charge[i], 0.0, tol);
        MY_BOOST_CHECK_CLOSE(arrays_const.mass[i], 1.0, tol);
        MY_BOOST_CHECK_CLOSE(arrays_const.diameter[i], 1.0, tol);
        BOOST_CHECK_EQUAL(arrays_const.ix[i], 0);
        BOOST_CHECK_EQUAL(arrays_const.iy[i], 0);
        BOOST_CHECK_EQUAL(arrays_const.iz[i], 0);
        BOOST_CHECK(arrays_const.type[i] == 0);
        BOOST_CHECK(arrays_const.rtag[i] == i);
        BOOST_CHECK(arrays_const.tag[i] == i);
        BOOST_CHECK(arrays_const.body[i] == NO_BODY);
        }
        
    b.release();
    }

#ifdef ENABLE_CUDA
//! Tests the ability of the ParticleData class to copy data between CPU <-> GPU
BOOST_AUTO_TEST_CASE( ParticleData_gpu_tests )
    {
    Scalar tol = Scalar(1e-6);
    
    // This set of tests will actually check that the ParticleData class is working
    // It would be a pain in the ass to test every possible state change in going from
    // the data being on the CPU to -on the GPU to on both, etc.... so we will just check
    // basic functionality here. Any subtle bugs will just have to show up when
    // unit tests are done that compare simulation runs on the cpu to those on the GPU
    boost::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::GPU));
    BoxDim box(10.0,30.0,50.0);
    int N = 500;
    ParticleData pdata(N, box, 1, exec_conf);
    ParticleDataArrays arrays = pdata.acquireReadWrite();
    for (int i = 0; i < N; i++)
        {
        arrays.x[i] = float(i)/100.0f;
        arrays.y[i] = float(i)/75.0f;
        arrays.z[i] = float(i)/50.0f;
        
        arrays.ax[i] = float(i);
        arrays.ay[i] = float(i) * 2.0f;
        arrays.az[i] = float(i) * 3.0f;
        
        arrays.charge[i] = float(i) * 4.0f;
        arrays.mass[i] = float(i) * 5.0f;
        arrays.diameter[i] = float(i) * 6.0f;
        
        arrays.ix[i] = i*7;
        arrays.iy[i] = i*8;
        arrays.iz[i] = i*9;
        
        arrays.type[i] = i;
        
        arrays.body[i] = i % 10;
        }
    pdata.release();
    // try accessing the data on the GPU
    gpu_pdata_arrays d_pdata = pdata.acquireReadWriteGPU();
    gpu_pdata_texread_test(d_pdata);
    CHECK_CUDA_ERROR();
    pdata.release();
    
    pdata.acquireReadOnly();
    for (unsigned int i = 0; i < (unsigned int)N; i++)
        {
        // check to make sure that the position copied back OK
        MY_BOOST_CHECK_CLOSE(arrays.x[i], float(i)/100.0f, tol);
        MY_BOOST_CHECK_CLOSE(arrays.y[i], float(i)/75.0f, tol);
        MY_BOOST_CHECK_CLOSE(arrays.z[i], float(i)/50.0f, tol);
        
        // check to make sure that the texture read worked and read back ok
        BOOST_CHECK(arrays.vx[i] == arrays.x[i]);
        BOOST_CHECK(arrays.vy[i] == arrays.y[i]);
        BOOST_CHECK(arrays.vz[i] == arrays.z[i]);
        
        // check to make sure that the accel was copied back ok
        MY_BOOST_CHECK_CLOSE(arrays.ax[i], float(i), tol);
        MY_BOOST_CHECK_CLOSE(arrays.ay[i], float(i) * 2.0f, tol);
        MY_BOOST_CHECK_CLOSE(arrays.az[i], float(i) * 3.0f, tol);
        
        // check the charge, mass and diameter
        MY_BOOST_CHECK_CLOSE(arrays.charge[i], float(i) * 4.0f, tol);
        MY_BOOST_CHECK_CLOSE(arrays.mass[i], float(i) * 5.0f, tol);
        MY_BOOST_CHECK_CLOSE(arrays.diameter[i], float(i) * 6.0f, tol);
        
        // check the image flag
        BOOST_CHECK_EQUAL(arrays.ix[i], (int)i*7);
        BOOST_CHECK_EQUAL(arrays.iy[i], (int)i*8);
        BOOST_CHECK_EQUAL(arrays.iz[i], (int)i*9);
        
        BOOST_CHECK(arrays.type[i] == i);
        BOOST_CHECK(arrays.body[i] == i % 10);
        }
    pdata.release();
    }

#endif

//! Test operation of the simple cubic initializer class
BOOST_AUTO_TEST_CASE( SimpleCubic_test )
    {
    Scalar tol = Scalar(1e-6);
    
    // make a simple one-particle box
    boost::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::CPU));
    SimpleCubicInitializer one(1, 2.0, "ABC");
    ParticleData one_data(one, exec_conf);
    ParticleDataArraysConst arrays_const = one_data.acquireReadOnly();
    BOOST_CHECK(arrays_const.nparticles == 1);
    MY_BOOST_CHECK_CLOSE(arrays_const.x[0], -1.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays_const.y[0], -1.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays_const.z[0], -1.0, tol);
    one_data.release();
    
    BOOST_CHECK_EQUAL(one_data.getNameByType(0), "ABC");
    BOOST_CHECK_EQUAL(one_data.getTypeByName("ABC"), (unsigned int)0);
    
    // now try an 8-particle one
    SimpleCubicInitializer eight(2, 2.0, "A");
    ParticleData eight_data(eight, exec_conf);
    
    arrays_const = eight_data.acquireReadOnly();
    BOOST_CHECK(arrays_const.nparticles == 8);
    MY_BOOST_CHECK_CLOSE(arrays_const.x[0], -2.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays_const.y[0], -2.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays_const.z[0], -2.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays_const.x[1], 0.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays_const.y[1], -2.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays_const.z[1], -2.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays_const.x[2], -2.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays_const.y[2], 0.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays_const.z[2], -2.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays_const.x[3], 0.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays_const.y[3], 0.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays_const.z[3], -2.0, tol);
    
    MY_BOOST_CHECK_CLOSE(arrays_const.x[4], -2.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays_const.y[4], -2.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays_const.z[4], 0.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays_const.x[5], 0.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays_const.y[5], -2.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays_const.z[5], 0.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays_const.x[6], -2.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays_const.y[6], 0.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays_const.z[6], 0.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays_const.x[7], 0.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays_const.y[7], 0.0, tol);
    MY_BOOST_CHECK_CLOSE(arrays_const.z[7], 0.0, tol);
    eight_data.release();
    }

//! Tests the RandomParticleInitializer class
BOOST_AUTO_TEST_CASE( Random_test )
    {
    // create a fairly dense system with a minimum distance of 0.8
    boost::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::CPU));
    Scalar min_dist = Scalar(0.8);
    RandomInitializer rand_init(500, Scalar(0.4), min_dist, "ABC");
    ParticleData pdata(rand_init, exec_conf);
    
    BOOST_CHECK_EQUAL(pdata.getNameByType(0), "ABC");
    BOOST_CHECK_EQUAL(pdata.getTypeByName("ABC"), (unsigned int)0);
    
    ParticleDataArraysConst arrays = pdata.acquireReadOnly();
    
    // check that the distances between particles are OK
    BoxDim box = pdata.getBox();
    Scalar L = box.xhi - box.xlo;
    for (unsigned int i = 0; i < arrays.nparticles; i++)
        {
        BOOST_CHECK(arrays.x[i] <= box.xhi && arrays.x[i] >= box.xlo);
        BOOST_CHECK(arrays.y[i] <= box.yhi && arrays.y[i] >= box.ylo);
        BOOST_CHECK(arrays.z[i] <= box.zhi && arrays.z[i] >= box.zlo);
        
        for (unsigned int j = 0; j < arrays.nparticles; j++)
            {
            if (i == j)
                continue;
                
            Scalar dx = arrays.x[j] - arrays.x[i];
            Scalar dy = arrays.y[j] - arrays.y[i];
            Scalar dz = arrays.z[j] - arrays.z[i];
            
            if (dx < -L/Scalar(2.0))
                dx += L;
            if (dx > L/Scalar(2.0))
                dx -= L;
                
            if (dy < -L/Scalar(2.0))
                dy += L;
            if (dy > L/Scalar(2.0))
                dy -= L;
                
            if (dz < -L/Scalar(2.0))
                dz += L;
            if (dz > L/Scalar(2.0))
                dz -= L;
                
            Scalar dr2 = dx*dx + dy*dy + dz*dz;
            BOOST_CHECK(dr2 >= min_dist*min_dist);
            }
        }
        
    pdata.release();
    }

/*#include "RandomGenerator.h"
#include "MOL2DumpWriter.h"
BOOST_AUTO_TEST_CASE( Generator_test )
    {
    vector<string> types;
    for (int i = 0; i < 6; i++)
        types.push_back("A");
    for (int i = 0; i < 7; i++)
        types.push_back("B");
    for (int i = 0; i < 6; i++)
        types.push_back("A");

    vector<string> types2;
    for (int i = 0; i < 7; i++)
        types2.push_back("B");

    boost::shared_ptr<PolymerParticleGenerator> poly(new PolymerParticleGenerator(1.2, types, 100));
    boost::shared_ptr<PolymerParticleGenerator> poly2(new PolymerParticleGenerator(1.2, types2, 100));
    BoxDim box(40);
    RandomGenerator generator(box, 1);
    generator.setSeparationRadius("A", 0.5);
    generator.setSeparationRadius("B", 0.5);
    generator.addGenerator(20, poly);
    generator.addGenerator(20, poly2);

    generator.generate();

    boost::shared_ptr<ParticleData> pdata(new ParticleData(generator));
    MOL2DumpWriter dump(pdata, string("test.mol2"));
    dump.analyze(0);
    }*/

#ifdef WIN32
#pragma warning( pop )
#endif

