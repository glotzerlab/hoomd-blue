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

// $Id: xml_reader_writer_test.cc 2148 2009-10-07 20:05:29Z joaander $
// $URL: https://codeblue.umich.edu/hoomd-blue/svn/trunk/src/unit_tests/xml_reader_writer_test.cc $
// Maintainer: joaander

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

//! Name the unit test module
#define BOOST_TEST_MODULE XMLReaderWriterTest
#include "boost_utf_configure.h"

#include <boost/test/floating_point_comparison.hpp>

#include <math.h>
#include "HOOMDBinaryDumpWriter.h"
#include "HOOMDBinaryInitializer.h"
#include "BondData.h"
#include "AngleData.h"

#include <iostream>
#include <sstream>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/convenience.hpp>
using namespace boost::filesystem;
#include <boost/shared_ptr.hpp>
using namespace boost;

#include <fstream>
using namespace std;

//! Need a simple define for checking two close values whether they are double or single
#define MY_BOOST_CHECK_CLOSE(a,b,c) BOOST_CHECK_CLOSE(a,Scalar(b),Scalar(c))

//! Tolerance for floating point comparisons
#ifdef SINGLE_PRECISION
const Scalar tol = Scalar(1e-3);
#else
const Scalar tol = 1e-3;
#endif

/*! \file xml_reader_writer_test.cc
    \brief Unit tests for HOOMDDumpWriter and HOOMDumpReader
    \ingroup unit_tests
*/

//! Performs low level tests of HOOMDDumpWriter
BOOST_AUTO_TEST_CASE( HOOMDBinaryReaderWriterBasicTests )
    {
#ifdef CUDA
    g_gpu_error_checking = true;
#endif
    
    // start by creating a single particle system: see it the correct file is written
    Scalar Lx(2.5), Ly(4.5), Lz(12.1);
    
    BoxDim box(Lx,Ly, Lz);
    int n_atom = 4;
    int n_types = 2;
    int n_bond_types = 2;
    int n_angle_types = 1;
    int n_dihedral_types = 1;
    int n_improper_types = 1;
    
    shared_ptr<SystemDefinition> sysdef1(new SystemDefinition(n_atom, box, n_types, n_bond_types, n_angle_types, n_dihedral_types, n_improper_types));
    shared_ptr<ParticleData> pdata1 = sysdef1->getParticleData();
    
    // set recognizable values for the particle
    const ParticleDataArrays array = pdata1->acquireReadWrite();
    Scalar x0(1.1), y1(2.1234567890123456), z3(-5.76);
    array.x[0] = x0;
    array.y[1] = y1;
    array.z[3] = z3;
    
    int ix3 = -1, iy1=-5, iz2=6;
    array.ix[3] = ix3;
    array.iy[1] = iy1;
    array.iz[2] = iz2;
    
    Scalar vx1(-1.4567), vy3(-10.0987654321098765), vz1(56.78);
    array.vx[1] = vx1;
    array.vy[3] = vy3;
    array.vz[1] = vz1;
    
    Scalar mass2(1.8);
    array.mass[2] = mass2;
    
    Scalar diameter3(3.8);
    array.diameter[3] = diameter3;
    
    int type1 = 1;
    array.type[1] = type1;
    
    pdata1->release();
    
    shared_ptr<IntegratorData> idata = sysdef1->getIntegratorData();
    // add some integrator states
    std::string name1 = "nvt", name2 = "langevin";
    Scalar var1(1.2), var2(0.1234), var3(1234324);
    IntegratorVariables iv0, iv1;
    iv0.type = name1;
    iv1.type = name2;
    iv0.variable.resize(2);
    iv0.variable[0] = var1;
    iv0.variable[1] = var2;
    iv1.variable.resize(1, var3);
    idata->registerIntegrator(0);
    idata->setIntegratorVariables(0, iv0);
    idata->registerIntegrator(1);
    idata->setIntegratorVariables(1, iv1);
    
    // add a couple walls for fun
    sysdef1->getWallData()->addWall(Wall(1,0,0, 0,1,0));
    sysdef1->getWallData()->addWall(Wall(0,1,0, 0,0,1));
    sysdef1->getWallData()->addWall(Wall(0,0,1, 1,0,0));
    
    // add a few bonds too
    sysdef1->getBondData()->addBond(Bond(0, 0, 1));
    sysdef1->getBondData()->addBond(Bond(1, 1, 0));
    
    // and angles as well
    sysdef1->getAngleData()->addAngle(Angle(0, 0, 1, 2));
    sysdef1->getAngleData()->addAngle(Angle(0, 1, 2, 0));
    
    // and a dihedral
    sysdef1->getDihedralData()->addDihedral(Dihedral(0, 0, 1, 2, 3));
    
    // and an improper
    sysdef1->getImproperData()->addDihedral(Dihedral(0, 3, 2, 1, 0));
    
    // create the writer
    shared_ptr<HOOMDBinaryDumpWriter> writer(new HOOMDBinaryDumpWriter(sysdef1, "test"));
    
    remove_all("test.0000000000.bin");
    BOOST_REQUIRE(!exists("test.0000000000.bin"));
    
    // write the first output
    writer->analyze(0);
        
    // make sure the file was created
    BOOST_REQUIRE(exists("test.0000000000.bin"));

    HOOMDBinaryInitializer init("test.0000000000.bin");
    shared_ptr<SystemDefinition> sysdef2(new SystemDefinition(init));
    shared_ptr<ParticleData> pdata2 = sysdef2->getParticleData();
    
    BOOST_CHECK_EQUAL(init.getTimeStep(), (unsigned int)0);
    BOOST_CHECK_EQUAL(pdata1->getN(), (unsigned int)n_atom);
    BOOST_CHECK_EQUAL(pdata2->getN(), (unsigned int)n_atom);
    BOOST_CHECK_EQUAL(pdata1->getNTypes(), (unsigned int)n_types);
    BOOST_CHECK_EQUAL(pdata2->getNTypes(), (unsigned int)n_types);

    MY_BOOST_CHECK_CLOSE(pdata1->getBox().xhi - pdata1->getBox().xlo, Lx, tol);
    MY_BOOST_CHECK_CLOSE(pdata1->getBox().yhi - pdata1->getBox().ylo, Ly, tol);
    MY_BOOST_CHECK_CLOSE(pdata1->getBox().zhi - pdata1->getBox().zlo, Lz, tol);
    MY_BOOST_CHECK_CLOSE(pdata2->getBox().xhi - pdata1->getBox().xlo, Lx, tol);
    MY_BOOST_CHECK_CLOSE(pdata2->getBox().yhi - pdata1->getBox().ylo, Ly, tol);
    MY_BOOST_CHECK_CLOSE(pdata2->getBox().zhi - pdata1->getBox().zlo, Lz, tol);
    
    const ParticleDataArrays array1 = pdata1->acquireReadWrite();
    BOOST_CHECK_EQUAL(array1.x[0], x0);
    BOOST_CHECK_EQUAL(array1.y[1], y1);
    BOOST_CHECK_EQUAL(array1.z[3], z3);
    BOOST_CHECK_EQUAL(array1.vx[1], vx1);
    BOOST_CHECK_EQUAL(array1.vy[3], vy3);
    BOOST_CHECK_EQUAL(array1.vz[1], vz1);

    BOOST_CHECK_EQUAL(array1.type[1], (unsigned int)type1);

    pdata1->release();

    const ParticleDataArrays array2 = pdata2->acquireReadWrite();
    BOOST_CHECK_EQUAL(array2.x[0], x0);
    BOOST_CHECK_EQUAL(array2.y[1], y1);
    BOOST_CHECK_EQUAL(array2.z[3], z3);
    BOOST_CHECK_EQUAL(array2.vx[1], vx1);
    BOOST_CHECK_EQUAL(array2.vy[3], vy3);
    BOOST_CHECK_EQUAL(array2.vz[1], vz1);

    BOOST_CHECK_EQUAL(array2.type[1], (unsigned int)type1);

    pdata2->release();
    
    //integrator variables check
    IntegratorVariables iv1_0, iv2_0, iv1_1, iv2_1;
    iv1_0 = sysdef1->getIntegratorData()->getIntegratorVariables(0);
    iv1_1 = sysdef1->getIntegratorData()->getIntegratorVariables(1);
    iv2_0 = sysdef2->getIntegratorData()->getIntegratorVariables(0);
    iv2_1 = sysdef2->getIntegratorData()->getIntegratorVariables(1);

    BOOST_CHECK_EQUAL(sysdef1->getIntegratorData()->getNumIntegrators(), (unsigned int) 2);
    BOOST_CHECK_EQUAL(sysdef2->getIntegratorData()->getNumIntegrators(), (unsigned int) 2);
    BOOST_CHECK_EQUAL(iv1_0.type, name1);
    BOOST_CHECK_EQUAL(iv2_0.type, name1);
    BOOST_CHECK_EQUAL(iv1_1.type, name2);
    BOOST_CHECK_EQUAL(iv2_1.type, name2);
    BOOST_CHECK_EQUAL(iv1_0.variable[0], var1);
    BOOST_CHECK_EQUAL(iv1_0.variable[1], var2);
    BOOST_CHECK_EQUAL(iv1_1.variable[0], var3);
    BOOST_CHECK_EQUAL(iv2_0.variable[0], var1);
    BOOST_CHECK_EQUAL(iv2_0.variable[1], var2);
    BOOST_CHECK_EQUAL(iv2_1.variable[0], var3);
    
    //
    // create the writer
    shared_ptr<HOOMDBinaryDumpWriter> writer2(new HOOMDBinaryDumpWriter(sysdef1, "test"));
    
    remove_all("test.0000000010.bin");
    BOOST_REQUIRE(!exists("test.0000000010.bin"));
    
    // write the first output
    writer->analyze(10);
        
    // make sure the file was created
    BOOST_REQUIRE(exists("test.0000000010.bin"));

    HOOMDBinaryInitializer init3("test.0000000010.bin");
    shared_ptr<SystemDefinition> sysdef3(new SystemDefinition(init3));
    shared_ptr<ParticleData> pdata3 = sysdef3->getParticleData();
    
    BOOST_CHECK_EQUAL(init3.getTimeStep(), (unsigned int)10);

    const ParticleDataArrays array3 = pdata3->acquireReadWrite();
    BOOST_CHECK_EQUAL(array3.mass[2], mass2);
    BOOST_CHECK_EQUAL(array.diameter[3], diameter3);
    BOOST_CHECK_EQUAL(array3.ix[3], ix3);
    BOOST_CHECK_EQUAL(array3.iy[1], iy1);
    BOOST_CHECK_EQUAL(array3.iz[2], iz2);

    }
    
#ifdef WIN32
#pragma warning( pop )
#endif

