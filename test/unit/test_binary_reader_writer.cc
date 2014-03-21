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

#include <math.h>
#include "HOOMDBinaryDumpWriter.h"
#include "HOOMDBinaryInitializer.h"
#include "BondedGroupData.h"

#include <iostream>
#include <sstream>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/convenience.hpp>
using namespace boost::filesystem;
#include <boost/shared_ptr.hpp>
using namespace boost;

#include <fstream>
using namespace std;

//! Name the unit test module
#define BOOST_TEST_MODULE BinaryReaderWriterTest
#include "boost_utf_configure.h"

/*! \file xml_reader_writer_test.cc
    \brief Unit tests for HOOMDDumpWriter and HOOMDumpReader
    \ingroup unit_tests
*/

//! Performs low level tests of HOOMDDumpWriter
BOOST_AUTO_TEST_CASE( HOOMDBinaryReaderWriterBasicTests )
    {
    // start by creating a single particle system: see it the correct file is written
    Scalar Lx(2.5), Ly(4.5), Lz(12.1);

    BoxDim box(Lx,Ly, Lz);
    int n_atom = 4;
    int n_types = 2;
    int n_bond_types = 2;
    int n_angle_types = 1;
    int n_dihedral_types = 1;
    int n_improper_types = 1;

    boost::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::CPU));
    boost::shared_ptr<SystemDefinition> sysdef1(new SystemDefinition(n_atom, box, n_types, n_bond_types, n_angle_types, n_dihedral_types, n_improper_types, exec_conf));
    boost::shared_ptr<ParticleData> pdata1 = sysdef1->getParticleData();

    // set recognizable values for the particle
    Scalar x0(1.1), y1(2.1234567890123456), z3(-5.76);
    int ix3 = -1, iy1=-5, iz2=6;
    Scalar vx1(-1.4567), vy3(-10.0987654321098765), vz1(56.78);
    Scalar mass2(1.8);
    Scalar diameter3(3.8);
    int type1 = 1;
    {
    ArrayHandle<Scalar4> h_pos(pdata1->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(pdata1->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<int3> h_image(pdata1->getImages(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_diameter(pdata1->getDiameters(), access_location::host, access_mode::readwrite);

    h_pos.data[0].x = x0;
    h_pos.data[1].y = y1;
    h_pos.data[3].z = z3;

    h_image.data[3].x = ix3;
    h_image.data[1].y = iy1;
    h_image.data[2].z = iz2;

    h_vel.data[1].x = vx1;
    h_vel.data[3].y = vy3;
    h_vel.data[1].z = vz1;

    h_vel.data[2].w = mass2;

    h_diameter.data[3] = diameter3;

    h_pos.data[1].w = __int_as_scalar(type1);
    }

    boost::shared_ptr<IntegratorData> idata = sysdef1->getIntegratorData();
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
    unsigned int i0 = idata->registerIntegrator();
    idata->setIntegratorVariables(i0, iv0);
    unsigned int i1 = idata->registerIntegrator();
    idata->setIntegratorVariables(i1, iv1);

    // add a couple walls for fun
    sysdef1->getWallData()->addWall(Wall(1,0,0, 0,1,0));
    sysdef1->getWallData()->addWall(Wall(0,1,0, 0,0,1));
    sysdef1->getWallData()->addWall(Wall(0,0,1, 1,0,0));

    // add a few bonds too
    sysdef1->getBondData()->addBondedGroup(Bond(0, 0, 1));
    sysdef1->getBondData()->addBondedGroup(Bond(1, 1, 0));

    // and angles as well
    sysdef1->getAngleData()->addBondedGroup(Angle(0, 0, 1, 2));
    sysdef1->getAngleData()->addBondedGroup(Angle(0, 1, 2, 0));

    // and a dihedral
    sysdef1->getDihedralData()->addBondedGroup(Dihedral(0, 0, 1, 2, 3));

    // and an improper
    sysdef1->getImproperData()->addBondedGroup(Dihedral(0, 3, 2, 1, 0));

    // create the writer
    boost::shared_ptr<HOOMDBinaryDumpWriter> writer(new HOOMDBinaryDumpWriter(sysdef1, "test"));

    remove_all("test.0000000000.bin");
    BOOST_REQUIRE(!exists("test.0000000000.bin"));

    // write the first output
    writer->analyze(0);

    // make sure the file was created
    BOOST_REQUIRE(exists("test.0000000000.bin"));

    HOOMDBinaryInitializer init(exec_conf, "test.0000000000.bin");
    boost::shared_ptr<SnapshotSystemData> snapshot;
    snapshot = init.getSnapshot();
    boost::shared_ptr<SystemDefinition> sysdef2(new SystemDefinition(snapshot, exec_conf));
    boost::shared_ptr<ParticleData> pdata2 = sysdef2->getParticleData();

    BOOST_CHECK_EQUAL(init.getTimeStep(), (unsigned int)0);
    BOOST_CHECK_EQUAL(pdata1->getN(), (unsigned int)n_atom);
    BOOST_CHECK_EQUAL(pdata2->getN(), (unsigned int)n_atom);
    BOOST_CHECK_EQUAL(pdata1->getNTypes(), (unsigned int)n_types);
    BOOST_CHECK_EQUAL(pdata2->getNTypes(), (unsigned int)n_types);

    MY_BOOST_CHECK_CLOSE(pdata1->getBox().getL().x, Lx, tol);
    MY_BOOST_CHECK_CLOSE(pdata1->getBox().getL().y, Ly, tol);
    MY_BOOST_CHECK_CLOSE(pdata1->getBox().getL().z, Lz, tol);
    MY_BOOST_CHECK_CLOSE(pdata2->getBox().getL().x, Lx, tol);
    MY_BOOST_CHECK_CLOSE(pdata2->getBox().getL().y, Ly, tol);
    MY_BOOST_CHECK_CLOSE(pdata2->getBox().getL().z, Lz, tol);

    {
    ArrayHandle<Scalar4> h_pos(pdata1->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_vel(pdata1->getVelocities(), access_location::host, access_mode::read);
    BOOST_CHECK_EQUAL(h_pos.data[0].x, x0);
    BOOST_CHECK_EQUAL(h_pos.data[1].y, y1);
    BOOST_CHECK_EQUAL(h_pos.data[3].z, z3);
    BOOST_CHECK_EQUAL(h_vel.data[1].x, vx1);
    BOOST_CHECK_EQUAL(h_vel.data[3].y, vy3);
    BOOST_CHECK_EQUAL(h_vel.data[1].z, vz1);

    BOOST_CHECK_EQUAL((unsigned int)__scalar_as_int(h_pos.data[1].w), (unsigned int)type1);

    }
    {
    ArrayHandle<Scalar4> h_pos(pdata2->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_vel(pdata2->getVelocities(), access_location::host, access_mode::read);
    BOOST_CHECK_EQUAL(h_pos.data[0].x, x0);
    BOOST_CHECK_EQUAL(h_pos.data[1].y, y1);
    BOOST_CHECK_EQUAL(h_pos.data[3].z, z3);
    BOOST_CHECK_EQUAL(h_vel.data[1].x, vx1);
    BOOST_CHECK_EQUAL(h_vel.data[3].y, vy3);
    BOOST_CHECK_EQUAL(h_vel.data[1].z, vz1);

    BOOST_CHECK_EQUAL((unsigned int)__scalar_as_int(h_pos.data[1].w), (unsigned int)type1);
    }

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
    boost::shared_ptr<HOOMDBinaryDumpWriter> writer2(new HOOMDBinaryDumpWriter(sysdef1, "test"));

    remove_all("test.0000000010.bin");
    BOOST_REQUIRE(!exists("test.0000000010.bin"));

    // write the first output
    writer->analyze(10);

    // make sure the file was created
    BOOST_REQUIRE(exists("test.0000000010.bin"));

    HOOMDBinaryInitializer init3(exec_conf,"test.0000000010.bin");
    boost::shared_ptr<SnapshotSystemData> snapshot2;
    snapshot2 = init3.getSnapshot();
    boost::shared_ptr<SystemDefinition> sysdef3(new SystemDefinition(snapshot2, exec_conf));
    boost::shared_ptr<ParticleData> pdata3 = sysdef3->getParticleData();

    BOOST_CHECK_EQUAL(init3.getTimeStep(), (unsigned int)10);

    {
    ArrayHandle<Scalar4> h_vel(pdata3->getVelocities(), access_location::host, access_mode::read);
    ArrayHandle<int3> h_image(pdata3->getImages(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_diameter(pdata3->getDiameters(), access_location::host, access_mode::read);

    BOOST_CHECK_EQUAL(h_vel.data[2].w, mass2);
    BOOST_CHECK_EQUAL(h_diameter.data[3], diameter3);
    BOOST_CHECK_EQUAL(h_image.data[3].x, ix3);
    BOOST_CHECK_EQUAL(h_image.data[1].y, iy1);
    BOOST_CHECK_EQUAL(h_image.data[2].z, iz2);
    }

    remove_all("test.0000000000.bin");
    remove_all("test.0000000010.bin");
    }

#ifdef WIN32
#pragma warning( pop )
#endif
