// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// this include is necessary to get MPI included before anything else to support intel MPI
#include "hoomd/ExecutionConfiguration.h"

#include <math.h>
#include "hoomd/deprecated/HOOMDDumpWriter.h"
#include "hoomd/deprecated/HOOMDInitializer.h"
#include "hoomd/ParticleGroup.h"
#include "hoomd/BondedGroupData.h"
#include "hoomd/Filesystem.h"

#include <iostream>
#include <sstream>
#include <memory>

#include <fstream>
using namespace std;

#include "hoomd/test/upp11_config.h"
HOOMD_UP_MAIN();

/*! \file xml_reader_writer_test.cc
    \brief Unit tests for HOOMDDumpWriter and HOOMDumpReader
    \ingroup unit_tests
*/

//! Performs low level tests of HOOMDDumpWriter
UP_TEST( HOOMDDumpWriterBasicTests )
    {
    // temporary directory for files
    std::string tmp_path = ".";

    Scalar3 I;

    // start by creating a single particle system: see it the correct file is written
    BoxDim box(Scalar(35), Scalar(55), Scalar(125));

    // set some tilt factors
    box.setTiltFactors(Scalar(1.0),Scalar(0.5),Scalar(0.25));
    int n_types = 5;
    int n_bond_types = 0;
    int n_angle_types = 0;
    int n_dihedral_types = 0;
    int n_improper_types = 0;

    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(5, box, n_types, n_bond_types, n_angle_types, n_dihedral_types, n_improper_types));
    std::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    // set recognizable values for the particle
        {
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar3> h_accel(pdata->getAccelerations(), access_location::host, access_mode::readwrite);
        ArrayHandle<int3> h_image(pdata->getImages(), access_location::host, access_mode::readwrite);
        ArrayHandle<unsigned int> h_body(pdata->getBodies(), access_location::host, access_mode::readwrite);
        ArrayHandle<unsigned int> h_tag(pdata->getTags(), access_location::host, access_mode::readwrite);
        ArrayHandle<unsigned int> h_rtag(pdata->getRTags(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> h_charge(pdata->getCharges(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> h_diameter(pdata->getDiameters(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> h_orient(pdata->getOrientationArray(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar3> h_moments(pdata->getMomentsOfInertiaArray(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> h_angmom(pdata->getAngularMomentumArray(), access_location::host, access_mode::readwrite);

        // Dummy particle 0 that we will skip with the particle selector
        h_pos.data[0].x = Scalar(2.5);
        h_pos.data[0].y = Scalar(3.5);
        h_pos.data[0].z = Scalar(-4.5);

        h_image.data[0].x = -2;
        h_image.data[0].y = -6;
        h_image.data[0].z = 5;

        h_vel.data[0].x = Scalar(-2.5);
        h_vel.data[0].y = Scalar(-11.5);
        h_vel.data[0].z = Scalar(57.5);

        h_accel.data[0].x = Scalar(1.0);
        h_accel.data[0].y = Scalar(2.0);
        h_accel.data[0].z = Scalar(3.0);

        h_vel.data[0].w = Scalar(10.5); //mass

        h_charge.data[0] = 1.0;

        h_diameter.data[0] = Scalar(4.5);

        h_pos.data[0].w = __int_as_scalar(2); //type

        h_body.data[0] = 1;

        h_orient.data[0] = make_scalar4(0,1,2,3);
        I = make_scalar3(7, 9, 1);
        h_moments.data[0] = I;
        h_angmom.data[0] = make_scalar4(0,-1,-2,-3);

        // Group particle 1
        h_pos.data[1].x = Scalar(1.5);
        h_pos.data[1].y = Scalar(2.5);
        h_pos.data[1].z = Scalar(-5.5);

        h_image.data[1].x = -1;
        h_image.data[1].y = -5;
        h_image.data[1].z = 6;

        h_vel.data[1].x = Scalar(-1.5);
        h_vel.data[1].y = Scalar(-10.5);
        h_vel.data[1].z = Scalar(56.5);

        h_accel.data[1].x = Scalar(-1.0);
        h_accel.data[1].y = Scalar(-2.0);
        h_accel.data[1].z = Scalar(-3.0);

        h_vel.data[1].w = Scalar(1.5); //mass

        h_charge.data[1] = -1.0;

        h_diameter.data[1] = Scalar(3.5);

        h_pos.data[1].w = __int_as_scalar(3); //type

        h_body.data[1] = NO_BODY;

        h_orient.data[1] = make_scalar4(3,2,1,0);
        I = make_scalar3(0, 1, 2);
        h_moments.data[1] = I;
        h_angmom.data[1] = make_scalar4(0,1,2,3);

        // Group particle 2
        h_pos.data[2].x = Scalar(1.5);
        h_pos.data[2].y = Scalar(2.5);
        h_pos.data[2].z = Scalar(-3.5);

        h_image.data[2].x = 10;
        h_image.data[2].y = 500;
        h_image.data[2].z = 900;

        h_vel.data[2].x = Scalar(-1.5);
        h_vel.data[2].y = Scalar(-10.5);
        h_vel.data[2].z = Scalar(5.5);

        h_accel.data[2].x = Scalar(-1.5);
        h_accel.data[2].y = Scalar(-2.5);
        h_accel.data[2].z = Scalar(-3.5);

        h_vel.data[2].w = Scalar(2.5); /// mass

        h_charge.data[2] = 1.5;

        h_diameter.data[2] = Scalar(4.5);

        h_pos.data[2].w = __int_as_scalar(0);

        h_body.data[2] = 1;

        h_orient.data[2] = make_scalar4(4,5,6,7);
        I = make_scalar3(5, 4, 3);
        h_moments.data[2] = I;
        h_angmom.data[2] = make_scalar4(9,8,7,6);

        // Group particle 3
        h_pos.data[3].x = Scalar(-1.5);
        h_pos.data[3].y = Scalar(2.5);
        h_pos.data[3].z = Scalar(3.5);

        h_image.data[3].x = 10;
        h_image.data[3].y = 500;
        h_image.data[3].z = 900;

        h_vel.data[3].x = Scalar(-1.5);
        h_vel.data[3].y = Scalar(-10.5);
        h_vel.data[3].z = Scalar(5.5);

        h_accel.data[3].x = Scalar(4.5);
        h_accel.data[3].y = Scalar(5.5);
        h_accel.data[3].z = Scalar(6.5);

        h_vel.data[3].w = Scalar(2.5);

        h_charge.data[3] = -1.5;

        h_diameter.data[3] = Scalar(4.5);

        h_pos.data[3].w = __int_as_scalar(1);

        h_body.data[3] = 1;

        h_orient.data[3] = make_scalar4(7,6,5,4);
        I = make_scalar3(1, 11, 21);
        h_moments.data[3] = I;
        h_angmom.data[3] = make_scalar4(1, 2, 3, 4);

        // Group particle 4
        h_pos.data[4].x = Scalar(-1.5);
        h_pos.data[4].y = Scalar(2.5);
        h_pos.data[4].z = Scalar(3.5);

        h_image.data[4].x = 105;
        h_image.data[4].y = 5005;
        h_image.data[4].z = 9005;

        h_vel.data[4].x = Scalar(-1.5);
        h_vel.data[4].y = Scalar(-10.5);
        h_vel.data[4].z = Scalar(5.5);

        h_accel.data[4].x = Scalar(-7.5);
        h_accel.data[4].y = Scalar(-8.5);
        h_accel.data[4].z = Scalar(-9.5);

        h_vel.data[4].w = Scalar(2.5);

        h_charge.data[4] = 2.0;

        h_diameter.data[4] = Scalar(4.5);

        h_pos.data[4].w = __int_as_scalar(1);

        h_body.data[4] = 0;

        h_orient.data[4] = make_scalar4(-1,-2,-3,-4);
        I = make_scalar3(51,41,31);
        h_moments.data[4] = I;
        h_angmom.data[4] = make_scalar4(51,41,31,21);
        }

    // create the writer
    std::shared_ptr<ParticleSelector> select_tag(new ParticleSelectorTag(sysdef,1,4));
    std::shared_ptr<ParticleGroup> group_tag(new ParticleGroup(sysdef, select_tag, true));
    std::shared_ptr<HOOMDDumpWriter> writer(new HOOMDDumpWriter(sysdef, tmp_path+"/test", group_tag));

    writer->setOutputPosition(false);

    // 0. file header
        {
        // make sure the first output file is deleted

        // write the first output
        writer->analyze(0);

        // make sure the file was created
        UP_ASSERT(filesystem::exists(tmp_path+"/test.0000000000.xml"));

        // check the output line by line
        ifstream f((tmp_path+"/test.0000000000.xml").c_str());
        string line;
        getline(f, line);
        UP_ASSERT_EQUAL(line, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "<hoomd_xml version=\"1.7\">");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line,  "<configuration time_step=\"0\" dimensions=\"3\" natoms=\"4\" >");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line,  "<box lx=\"35\" ly=\"55\" lz=\"125\" xy=\"1\" xz=\"0.5\" yz=\"0.25\"/>");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line,  "</configuration>");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line,  "</hoomd_xml>");
        UP_ASSERT(!f.bad());
        f.close();
        unlink((tmp_path+"/test.0000000000.xml").c_str());
        }

    // 1. position
        {
        writer->setOutputPosition(true);

        // write the file
        writer->analyze(10);

        // make sure the file was created
        UP_ASSERT(filesystem::exists(tmp_path+"/test.0000000010.xml"));

        // assume that the first lines tested in the first case are still OK and skip them
        ifstream f((tmp_path+"/test.0000000010.xml").c_str());
        string line;
        getline(f, line); // <?xml
        getline(f, line); // <HOOMD_xml
        getline(f, line); // <Configuration
        getline(f, line); // <Box

        getline(f, line);
        UP_ASSERT_EQUAL(line, "<position num=\"4\">");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "1.5 2.5 -5.5");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "1.5 2.5 -3.5");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "-1.5 2.5 3.5");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "-1.5 2.5 3.5");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "</position>");

        getline(f, line); // </configuration
        getline(f, line); // </HOOMD_xml
        f.close();
        unlink((tmp_path+"/test.0000000010.xml").c_str());
        }

    // 2. image
        {
        writer->setOutputPosition(false);
        writer->setOutputImage(true);

        // write the file
        writer->analyze(20);

        // assume that the first lines tested in the first case are still OK and skip them
        ifstream f((tmp_path+"/test.0000000020.xml").c_str());
        string line;
        getline(f, line); // <?xml
        getline(f, line); // <HOOMD_xml
        getline(f, line); // <Configuration
        getline(f, line); // <Box

        getline(f, line);
        UP_ASSERT_EQUAL(line, "<image num=\"4\">");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "-1 -5 6");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "10 500 900");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "10 500 900");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "105 5005 9005");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "</image>");
        f.close();
        unlink((tmp_path+"/test.0000000020.xml").c_str());
        }

    // 3. velocity
        {
        writer->setOutputImage(false);
        writer->setOutputVelocity(true);

        // write the file
        writer->analyze(30);

        // assume that the first lines tested in the first case are still OK and skip them
        ifstream f((tmp_path+"/test.0000000030.xml").c_str());
        string line;
        getline(f, line); // <?xml
        getline(f, line); // <HOOMD_xml
        getline(f, line); // <Configuration
        getline(f, line); // <Box

        getline(f, line);
        UP_ASSERT_EQUAL(line, "<velocity num=\"4\">");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "-1.5 -10.5 56.5");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "-1.5 -10.5 5.5");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "-1.5 -10.5 5.5");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "-1.5 -10.5 5.5");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "</velocity>");
        f.close();
        unlink((tmp_path+"/test.0000000030.xml").c_str());
        }

    // 4. acceleration
        {
        writer->setOutputVelocity(false);
        writer->setOutputAccel(true);

        // write the file
        writer->analyze(40);

        // assume that the first lines tested in the first case are still OK and skip them
        ifstream f((tmp_path+"/test.0000000040.xml").c_str());
        string line;
        getline(f, line); // <?xml
        getline(f, line); // <HOOMD_xml
        getline(f, line); // <Configuration
        getline(f, line); // <Box

        getline(f, line);
        UP_ASSERT_EQUAL(line, "<acceleration num=\"4\">");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "-1 -2 -3");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "-1.5 -2.5 -3.5");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "4.5 5.5 6.5");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "-7.5 -8.5 -9.5");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "</acceleration>");
        f.close();
        unlink((tmp_path+"/test.0000000040.xml").c_str());
        }

    // 5. mass
        {
        writer->setOutputAccel(false);
        writer->setOutputMass(true);

        // write the file
        writer->analyze(50);

        // assume that the first lines tested in the first case are still OK and skip them
        ifstream f((tmp_path+"/test.0000000050.xml").c_str());
        string line;
        getline(f, line); // <?xml
        getline(f, line); // <HOOMD_xml
        getline(f, line); // <Configuration
        getline(f, line); // <Box

        getline(f, line);
        UP_ASSERT_EQUAL(line, "<mass num=\"4\">");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "1.5");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "2.5");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "2.5");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "2.5");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "</mass>");
        f.close();
        unlink((tmp_path+"/test.0000000050.xml").c_str());
        }

    // 6. charge
        {
        writer->setOutputMass(false);
        writer->setOutputCharge(true);

        // write the file
        writer->analyze(60);

        // assume that the first lines tested in the first case are still OK and skip them
        ifstream f((tmp_path+"/test.0000000060.xml").c_str());
        string line;
        getline(f, line); // <?xml
        getline(f, line); // <HOOMD_xml
        getline(f, line); // <Configuration
        getline(f, line); // <Box

        getline(f, line);
        UP_ASSERT_EQUAL(line, "<charge num=\"4\">");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "-1");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "1.5");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "-1.5");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "2");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "</charge>");
        f.close();
        unlink((tmp_path+"/test.0000000060.xml").c_str());
        }

    // 7. diameter
        {
        writer->setOutputCharge(false);
        writer->setOutputDiameter(true);

        // write the file
        writer->analyze(70);

        // assume that the first lines tested in the first case are still OK and skip them
        ifstream f((tmp_path+"/test.0000000070.xml").c_str());
        string line;
        getline(f, line); // <?xml
        getline(f, line); // <HOOMD_xml
        getline(f, line); // <Configuration
        getline(f, line); // <Box

        getline(f, line);
        UP_ASSERT_EQUAL(line, "<diameter num=\"4\">");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "3.5");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "4.5");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "4.5");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "4.5");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "</diameter>");
        f.close();
        unlink((tmp_path+"/test.0000000070.xml").c_str());
        }

    // 8. type
        {
        writer->setOutputDiameter(false);
        writer->setOutputType(true);

        // write the file
        writer->analyze(80);

        // assume that the first lines tested in the first case are still OK and skip them
        ifstream f((tmp_path+"/test.0000000080.xml").c_str());
        string line;
        getline(f, line); // <?xml
        getline(f, line); // <HOOMD_xml
        getline(f, line); // <Configuration
        getline(f, line); // <Box

        getline(f, line);
        UP_ASSERT_EQUAL(line, "<type num=\"4\">");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "D");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "A");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "B");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "B");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "</type>");
        f.close();
        unlink((tmp_path+"/test.0000000080.xml").c_str());
        }

    // 9. body
        {
        writer->setOutputType(false);
        writer->setOutputBody(true);

        // write the file
        writer->analyze(90);

        // assume that the first lines tested in the first case are still OK and skip them
        ifstream f((tmp_path+"/test.0000000090.xml").c_str());
        string line;
        getline(f, line); // <?xml
        getline(f, line); // <HOOMD_xml
        getline(f, line); // <Configuration
        getline(f, line); // <Box

        getline(f, line);
        UP_ASSERT_EQUAL(line, "<body num=\"4\">");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "-1");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "1");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "1");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "0");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "</body>");
        f.close();
        unlink((tmp_path+"/test.0000000090.xml").c_str());
        }

    // 10. orientation
        {
        writer->setOutputBody(false);
        writer->setOutputOrientation(true);

        // write the file
        writer->analyze(100);

        // assume that the first lines tested in the first case are still OK and skip them
        ifstream f((tmp_path+"/test.0000000100.xml").c_str());
        string line;
        getline(f, line); // <?xml
        getline(f, line); // <HOOMD_xml
        getline(f, line); // <Configuration
        getline(f, line); // <Box

        getline(f, line);
        UP_ASSERT_EQUAL(line, "<orientation num=\"4\">");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "3 2 1 0");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "4 5 6 7");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "7 6 5 4");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "-1 -2 -3 -4");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "</orientation>");
        f.close();
        unlink((tmp_path+"/test.0000000100.xml").c_str());
        }

    // 11. moment of inertia
        {
        writer->setOutputOrientation(false);
        writer->setOutputMomentInertia(true);

        // write the file
        writer->analyze(110);

        // assume that the first lines tested in the first case are still OK and skip them
        ifstream f((tmp_path+"/test.0000000110.xml").c_str());
        string line;
        getline(f, line); // <?xml
        getline(f, line); // <HOOMD_xml
        getline(f, line); // <Configuration
        getline(f, line); // <Box

        getline(f, line);
        UP_ASSERT_EQUAL(line, "<moment_inertia num=\"4\">");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "0 1 2");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "5 4 3");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "1 11 21");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "51 41 31");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "</moment_inertia>");
        f.close();
        unlink((tmp_path+"/test.0000000110.xml").c_str());
        }

    // 12. angular momentum
        {
        writer->setOutputMomentInertia(false);
        writer->setOutputAngularMomentum(true);

        // write the file
        writer->analyze(120);

        // assume that the first lines tested in the first case are still OK and skip them
        ifstream f((tmp_path+"/test.0000000120.xml").c_str());
        string line;
        getline(f, line); // <?xml
        getline(f, line); // <HOOMD_xml
        getline(f, line); // <Configuration
        getline(f, line); // <Box

        getline(f, line);
        UP_ASSERT_EQUAL(line, "<angmom num=\"4\">");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "0 1 2 3");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "9 8 7 6");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "1 2 3 4");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "51 41 31 21");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "</angmom>");
        f.close();
        unlink((tmp_path+"/test.0000000120.xml").c_str());
        }
    }

//! Tests the ability of HOOMDDumpWriter to write out the system topology
UP_TEST( HOOMDDumpWriter_topology_test)
    {
    // temporary directory for files
    std::string tmp_path = ".";

    // start by creating a single particle system: see it the correct file is written
    BoxDim box(Scalar(35), Scalar(55), Scalar(125));

    // set some tilt factors
    box.setTiltFactors(Scalar(1.0),Scalar(0.5),Scalar(0.25));
    int n_types = 1;
    int n_bond_types = 2;
    int n_angle_types = 1;
    int n_dihedral_types = 1;
    int n_improper_types = 1;

    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(4, box, n_types, n_bond_types, n_angle_types, n_dihedral_types, n_improper_types));

    // add a few bonds too
    sysdef->getBondData()->addBondedGroup(Bond(0, 0, 1));
    sysdef->getBondData()->addBondedGroup(Bond(1, 1, 0));

    // and angles as well
    sysdef->getAngleData()->addBondedGroup(Angle(0, 0, 1, 2));
    sysdef->getAngleData()->addBondedGroup(Angle(0, 1, 2, 0));

    // and a dihedral
    sysdef->getDihedralData()->addBondedGroup(Dihedral(0, 0, 1, 2, 3));

    // and an improper
    sysdef->getImproperData()->addBondedGroup(Dihedral(0, 3, 2, 1, 0));

    // and two constraints
    sysdef->getConstraintData()->addBondedGroup(Constraint(Scalar(1.5),0,1));
    sysdef->getConstraintData()->addBondedGroup(Constraint(Scalar(2.5),1,2));

    // create the writer

    // 1. if the group is not all, then the xml file should be empty
        {
        std::shared_ptr<ParticleSelector> select_tag(new ParticleSelectorTag(sysdef,1,2));
        std::shared_ptr<ParticleGroup> group_tag(new ParticleGroup(sysdef, select_tag, true));
        std::shared_ptr<HOOMDDumpWriter> writer(new HOOMDDumpWriter(sysdef, tmp_path+"/test", group_tag));
        writer->setOutputPosition(false);
        writer->setOutputBond(true);

        writer->analyze(0);

        // make sure the file was created
        UP_ASSERT(filesystem::exists(tmp_path+"/test.0000000000.xml"));

        // check the output line by line
        ifstream f((tmp_path+"/test.0000000000.xml").c_str());
        string line;
        getline(f, line);
        UP_ASSERT_EQUAL(line, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "<hoomd_xml version=\"1.7\">");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line,  "<configuration time_step=\"0\" dimensions=\"3\" natoms=\"2\" >");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line,  "<box lx=\"35\" ly=\"55\" lz=\"125\" xy=\"1\" xz=\"0.5\" yz=\"0.25\"/>");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line,  "</configuration>");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line,  "</hoomd_xml>");
        UP_ASSERT(!f.bad());
        f.close();
        unlink((tmp_path+"/test.0000000000.xml").c_str());
        }

    std::shared_ptr<ParticleSelectorAll> select_all(new ParticleSelectorAll(sysdef));
    std::shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, select_all, true));
    std::shared_ptr<HOOMDDumpWriter> writer(new HOOMDDumpWriter(sysdef, tmp_path+"/test", group_all));
    writer->setOutputPosition(false);
    // 1. bonds
        {
        writer->setOutputBond(true);

        // write the file
        writer->analyze(10);

        // assume that the first lines tested in the first case are still OK and skip them
        ifstream f((tmp_path+"/test.0000000010.xml").c_str());
        string line;
        getline(f, line); // <?xml
        getline(f, line); // <HOOMD_xml
        getline(f, line); // <Configuration
        getline(f, line); // <Box

        getline(f, line);
        UP_ASSERT_EQUAL(line, "<bond num=\"2\">");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "bondA 0 1");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "bondB 1 0");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "</bond>");
        f.close();
        unlink((tmp_path+"/test.0000000010.xml").c_str());
        }

    // 2. angles
        {
        writer->setOutputBond(false);
        writer->setOutputAngle(true);

        // write the file
        writer->analyze(20);

        // assume that the first lines tested in the first case are still OK and skip them
        ifstream f((tmp_path+"/test.0000000020.xml").c_str());
        string line;
        getline(f, line); // <?xml
        getline(f, line); // <HOOMD_xml
        getline(f, line); // <Configuration
        getline(f, line); // <Box

        getline(f, line);
        UP_ASSERT_EQUAL(line, "<angle num=\"2\">");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "angleA 0 1 2");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "angleA 1 2 0");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "</angle>");
        f.close();
        unlink((tmp_path+"/test.0000000020.xml").c_str());
        }

    // 3. dihedrals
        {
        writer->setOutputAngle(false);
        writer->setOutputDihedral(true);

        // write the file
        writer->analyze(30);

        // assume that the first lines tested in the first case are still OK and skip them
        ifstream f((tmp_path+"/test.0000000030.xml").c_str());
        string line;
        getline(f, line); // <?xml
        getline(f, line); // <HOOMD_xml
        getline(f, line); // <Configuration
        getline(f, line); // <Box

        getline(f, line);
        UP_ASSERT_EQUAL(line, "<dihedral num=\"1\">");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "dihedralA 0 1 2 3");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "</dihedral>");
        f.close();
        unlink((tmp_path+"/test.0000000030.xml").c_str());
        }

    // 4. impropers
        {
        writer->setOutputDihedral(false);
        writer->setOutputImproper(true);

        // write the file
        writer->analyze(40);

        // assume that the first lines tested in the first case are still OK and skip them
        ifstream f((tmp_path+"/test.0000000040.xml").c_str());
        string line;
        getline(f, line); // <?xml
        getline(f, line); // <HOOMD_xml
        getline(f, line); // <Configuration
        getline(f, line); // <Box

        getline(f, line);
        UP_ASSERT_EQUAL(line, "<improper num=\"1\">");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "improperA 3 2 1 0");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "</improper>");
        f.close();
        unlink((tmp_path+"/test.0000000040.xml").c_str());
        }

    // 5. constraints
        {
        writer->setOutputImproper(false);
        writer->setOutputConstraint(true);

        // write the file
        writer->analyze(50);

        // assume that the first lines tested in the first case are still OK and skip them
        ifstream f((tmp_path+"/test.0000000050.xml").c_str());
        string line;
        getline(f, line); // <?xml
        getline(f, line); // <HOOMD_xml
        getline(f, line); // <Configuration
        getline(f, line); // <Box

        getline(f, line);
        UP_ASSERT_EQUAL(line, "<constraint num=\"2\">");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "0 1 1.5");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "1 2 2.5");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "</constraint>");
        f.close();
        unlink((tmp_path+"/test.0000000050.xml").c_str());
        }
    }

//! Tests the ability of HOOMDDumpWriter to handle tagged and reordered particles
UP_TEST( HOOMDDumpWriter_tag_test )
    {
    // temporary directory for files
    std::string tmp_path = ".";

    // start by creating a single particle system: see it the correct file is written
    BoxDim box(Scalar(100.5), Scalar(120.5), Scalar(130.5));
    int n_types = 10;
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(6, box, n_types));
    std::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    // this is the shuffle order of the particles
    unsigned int tags[6] = { 5, 2, 3, 1, 0, 4 };
    unsigned int rtags[6] = { 4, 3, 1, 2, 5, 0 };

    {
    // set recognizable values for the particle
    ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(pdata->getAccelerations(), access_location::host, access_mode::readwrite);
    ArrayHandle<int3> h_image(pdata->getImages(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_body(pdata->getBodies(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_tag(pdata->getTags(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_rtag(pdata->getRTags(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_charge(pdata->getCharges(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_diameter(pdata->getDiameters(), access_location::host, access_mode::readwrite);

    for (int i = 0; i < 6; i++)
        {
        h_tag.data[i] = tags[i];
        unsigned int tag = tags[i];

        h_pos.data[i].x = Scalar(tag)+Scalar(0.5);
        h_pos.data[i].y = Scalar(tag)+Scalar(1.5);
        h_pos.data[i].z = Scalar(tag)+Scalar(2.5);

        h_image.data[i].x = tag - 10;
        h_image.data[i].y = tag - 11;
        h_image.data[i].z = tag + 50;

        h_vel.data[i].x = Scalar(tag)*Scalar(10.0);
        h_vel.data[i].y = Scalar(tag)*Scalar(11.0);
        h_vel.data[i].z = Scalar(tag)*Scalar(12.0);

        h_pos.data[i].w =__int_as_scalar(tag + 2);
        h_rtag.data[i] = rtags[i];
        }
    }

    // create the writer
    std::shared_ptr<ParticleSelectorAll> select_all(new ParticleSelectorAll(sysdef));
    std::shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, select_all, true));
    std::shared_ptr<HOOMDDumpWriter> writer(new HOOMDDumpWriter(sysdef, tmp_path+"/test", group_all));

    // write the file with all outputs enabled
    writer->setOutputPosition(true);
    writer->setOutputVelocity(true);
    writer->setOutputType(true);
    writer->setOutputImage(true);

    // now the big mess: check the file line by line
        {
        // write the first output
        writer->analyze(100);

        // make sure the file was created
        UP_ASSERT(filesystem::exists(tmp_path+"/test.0000000100.xml"));

        // check the output line by line
        ifstream f((tmp_path+"/test.0000000100.xml").c_str());
        string line;
        getline(f, line);
        UP_ASSERT_EQUAL(line, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "<hoomd_xml version=\"1.7\">");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line,  "<configuration time_step=\"100\" dimensions=\"3\" natoms=\"6\" >");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line,  "<box lx=\"100.5\" ly=\"120.5\" lz=\"130.5\" xy=\"0\" xz=\"0\" yz=\"0\"/>");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "<position num=\"6\">");
        UP_ASSERT(!f.bad());

        // check all the positions
        getline(f, line);
        UP_ASSERT_EQUAL(line, "0.5 1.5 2.5");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "1.5 2.5 3.5");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "2.5 3.5 4.5");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "3.5 4.5 5.5");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "4.5 5.5 6.5");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "5.5 6.5 7.5");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line,  "</position>");
        UP_ASSERT(!f.bad());

        // check all the images
        getline(f, line);
        UP_ASSERT_EQUAL(line, "<image num=\"6\">");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "-10 -11 50");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "-9 -10 51");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "-8 -9 52");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "-7 -8 53");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "-6 -7 54");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "-5 -6 55");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line,  "</image>");
        UP_ASSERT(!f.bad());

        // check all velocities
        getline(f, line);
        UP_ASSERT_EQUAL(line, "<velocity num=\"6\">");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "0 0 0");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "10 11 12");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "20 22 24");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "30 33 36");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "40 44 48");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "50 55 60");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "</velocity>");

        // check all types
        getline(f, line);
        UP_ASSERT_EQUAL(line, "<type num=\"6\">");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "C");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "D");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "E");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "F");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "G");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "H");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line, "</type>");

        getline(f, line);
        UP_ASSERT_EQUAL(line,  "</configuration>");
        UP_ASSERT(!f.bad());

        getline(f, line);
        UP_ASSERT_EQUAL(line,  "</hoomd_xml>");
        UP_ASSERT(!f.bad());
        f.close();
        unlink((tmp_path+"/test.0000000100.xml").c_str());
        }
    }

//! Test basic functionality of HOOMDInitializer
UP_TEST( HOOMDInitializer_basic_tests )
    {
    // temporary directory for files
    std::string tmp_path = ".";

    // create a test input file
    ofstream f((tmp_path+"/test_input.xml").c_str());
    f << "<?xml version =\"1.0\" encoding =\"UTF-8\" ?>\n\
<hoomd_xml version=\"1.7\">\n\
<configuration time_step=\"150000000\" dimensions=\"2\">\n\
<box lx=\"20.05\" ly= \"32.12345\" lz=\"45.098\" xy=\".12\" xz=\".23\" yz=\".34\"/>\n\
<position >\n\
1.4 2.567890 3.45\n\
2.4 3.567890 4.45\n\
3.4 4.567890 5.45\n\
4.4 5.567890 6.45\n\
5.4 6.567890 7.45\n\
6.4 7.567890 8.45\n\
</position>\n\
<image>\n\
10 20 30\n\
11 21 31\n\
12 22 32\n\
13 23 33\n\
14 24 34\n\
15 25 35\n\
</image>\n\
<velocity>\n\
10.12 12.1567 1.056\n\
20.12 22.1567 2.056\n\
30.12 32.1567 3.056\n\
40.12 42.1567 4.056\n\
50.12 52.1567 5.056\n\
60.12 62.1567 6.056\n\
</velocity>\n\
<mass>\n\
1.0\n\
2.0\n\
3.0\n\
4.0\n\
5.0\n\
6.0\n\
</mass>\n\
<diameter>\n\
7.0\n\
8.0\n\
9.0\n\
10.0\n\
11.0\n\
12.0\n\
</diameter>\n\
<body>\n\
-1\n\
0\n\
1\n\
2\n\
3\n\
4\n\
</body>\n\
<type>\n\
5\n\
4\n\
3\n\
2\n\
1\n\
0\n\
</type>\n\
<charge>\n\
0.0\n\
10.0\n\
20.0\n\
30.0\n\
40.0\n\
50.0\n\
</charge>\n\
<moment_inertia>\n\
0 1 2 \n\
10 11 12\n\
20 21 22\n\
30 31 32\n\
40 41 42\n\
50 51 52\n\
</moment_inertia>\n\
<angmom>\n\
1 10 100 1000\n\
2 20 200 2000\n\
3 30 300 3000\n\
4 40 400 4000\n\
5 50 500 5000\n\
6 60 600 6000\n\
</angmom>\n\
<bond>\n\
bond_a 0 1\n\
bond_b 1 2\n\
bond_a 2 3\n\
bond_c 3 4\n\
</bond>\n\
<angle>\n\
angle_a 0 1 2\n\
angle_b 1 2 3\n\
angle_a 2 3 4\n\
</angle>\n\
<dihedral>\n\
di_a 0 1 2 3\n\
di_b 1 2 3 4\n\
</dihedral>\n\
<improper>\n\
im_a 3 2 1 0\n\
im_b 5 4 3 2\n\
</improper>\n\
<constraint>\n\
0 1 1.5\n\
1 2 2.5\n\
</constraint>\n\
</configuration>\n\
</hoomd_xml>" << endl;
    f.close();

    // now that we have created a test file, load it up into a pdata
    std::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::CPU));
    HOOMDInitializer init(exec_conf,tmp_path+"/test_input.xml");
    std::shared_ptr< SnapshotSystemData<Scalar> > snapshot;
    snapshot = init.getSnapshot();
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snapshot));
    std::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    // verify all parameters
    UP_ASSERT_EQUAL(init.getTimeStep(), (unsigned int)150000000);
    UP_ASSERT_EQUAL(sysdef->getNDimensions(), (unsigned int)2);
    UP_ASSERT_EQUAL(pdata->getN(), (unsigned int)6);
    UP_ASSERT_EQUAL(pdata->getNTypes(), (unsigned int)6);
    MY_CHECK_CLOSE(pdata->getGlobalBox().getL().x, 20.05, tol);
    MY_CHECK_CLOSE(pdata->getGlobalBox().getL().y, 32.12345, tol);
    MY_CHECK_CLOSE(pdata->getGlobalBox().getL().z, 45.098, tol);
    MY_CHECK_CLOSE(pdata->getGlobalBox().getTiltFactorXY(), 0.12, tol);
    MY_CHECK_CLOSE(pdata->getGlobalBox().getTiltFactorXZ(), 0.23, tol);
    MY_CHECK_CLOSE(pdata->getGlobalBox().getTiltFactorYZ(), 0.34, tol);

    {
    ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_vel(pdata->getVelocities(), access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_accel(pdata->getAccelerations(), access_location::host, access_mode::read);
    ArrayHandle<int3> h_image(pdata->getImages(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_body(pdata->getBodies(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(pdata->getRTags(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_charge(pdata->getCharges(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_diameter(pdata->getDiameters(), access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_moments(pdata->getMomentsOfInertiaArray(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_angmom(pdata->getAngularMomentumArray(), access_location::host, access_mode::readwrite);

    for (int i = 0; i < 6; i++)
        {
        MY_CHECK_CLOSE(h_pos.data[i].x, Scalar(i) + Scalar(1.4), tol);
        MY_CHECK_CLOSE(h_pos.data[i].y, Scalar(i) + Scalar(2.567890), tol);
        MY_CHECK_CLOSE(h_pos.data[i].z, Scalar(i) + Scalar(3.45), tol);

        UP_ASSERT_EQUAL(h_image.data[i].x, 10 + i);
        UP_ASSERT_EQUAL(h_image.data[i].y, 20 + i);
        UP_ASSERT_EQUAL(h_image.data[i].z, 30 + i);

        MY_CHECK_CLOSE(h_vel.data[i].x, Scalar(i+1)*Scalar(10.0) + Scalar(0.12), tol);
        MY_CHECK_CLOSE(h_vel.data[i].y, Scalar(i+1)*Scalar(10.0) + Scalar(2.1567), tol);
        MY_CHECK_CLOSE(h_vel.data[i].z, Scalar(i+1) + Scalar(0.056), tol);

        MY_CHECK_CLOSE(h_vel.data[i].w, Scalar(i+1), tol); // mass

        MY_CHECK_CLOSE(h_diameter.data[i], Scalar(i+7), tol);

        MY_CHECK_CLOSE(h_charge.data[i], Scalar(i)*Scalar(10.0), tol);

        UP_ASSERT_EQUAL(h_body.data[i], (unsigned int)(i-1));

        // checking that the type is correct becomes tricky because types are identified by
        // string
        ostringstream type_name;
        type_name << 5-i;   // the expected type is the integer 5-i
        UP_ASSERT_EQUAL((unsigned int)__scalar_as_int(h_pos.data[i].w), pdata->getTypeByName(type_name.str()));
        UP_ASSERT_EQUAL(h_tag.data[i], (unsigned int)i);
        UP_ASSERT_EQUAL(h_rtag.data[i], (unsigned int)i);

        // check the moment_inertia values
        Scalar3 I;
        I = h_moments.data[i];
        MY_CHECK_CLOSE(I.x, i*10, tol);
        MY_CHECK_CLOSE(I.y, i*10+1, tol);
        MY_CHECK_CLOSE(I.z, i*10+2, tol);

        // check the angular momentum values
        Scalar4 M;
        M = h_angmom.data[i];
        MY_CHECK_CLOSE(M.x, i+1, tol);
        MY_CHECK_CLOSE(M.y, (i+1)*10, tol);
        MY_CHECK_CLOSE(M.z, (i+1)*100, tol);
        MY_CHECK_CLOSE(M.w, (i+1)*1000, tol);
        }
    }

    // check the bonds
    std::shared_ptr<BondData> bond_data = sysdef->getBondData();

    // 4 bonds should have been read in
    UP_ASSERT_EQUAL(bond_data->getN(), (unsigned int)4);

    // check that the types have been named properly
    UP_ASSERT_EQUAL(bond_data->getNTypes(), (unsigned int)3);
    UP_ASSERT_EQUAL(bond_data->getTypeByName("bond_a"), (unsigned int)0);
    UP_ASSERT_EQUAL(bond_data->getTypeByName("bond_b"), (unsigned int)1);
    UP_ASSERT_EQUAL(bond_data->getTypeByName("bond_c"), (unsigned int)2);

    UP_ASSERT_EQUAL(bond_data->getNameByType(0), string("bond_a"));
    UP_ASSERT_EQUAL(bond_data->getNameByType(1), string("bond_b"));
    UP_ASSERT_EQUAL(bond_data->getNameByType(2), string("bond_c"));

    // verify each bond
    Bond b = bond_data-> getGroupByTag(0);
    UP_ASSERT_EQUAL(b.a, (unsigned int)0);
    UP_ASSERT_EQUAL(b.b, (unsigned int)1);
    UP_ASSERT_EQUAL(b.type, (unsigned int)0);

    b = bond_data-> getGroupByTag(1);
    UP_ASSERT_EQUAL(b.a, (unsigned int)1);
    UP_ASSERT_EQUAL(b.b, (unsigned int)2);
    UP_ASSERT_EQUAL(b.type, (unsigned int)1);

    b = bond_data-> getGroupByTag(2);
    UP_ASSERT_EQUAL(b.a, (unsigned int)2);
    UP_ASSERT_EQUAL(b.b, (unsigned int)3);
    UP_ASSERT_EQUAL(b.type, (unsigned int)0);

    b = bond_data-> getGroupByTag(3);
    UP_ASSERT_EQUAL(b.a, (unsigned int)3);
    UP_ASSERT_EQUAL(b.b, (unsigned int)4);
    UP_ASSERT_EQUAL(b.type, (unsigned int)2);

    // check the angles
    std::shared_ptr<AngleData> angle_data = sysdef->getAngleData();

    // 3 angles should have been read in
    UP_ASSERT_EQUAL(angle_data->getN(), (unsigned int)3);

    // check that the types have been named properly
    UP_ASSERT_EQUAL(angle_data->getNTypes(), (unsigned int)2);
    UP_ASSERT_EQUAL(angle_data->getTypeByName("angle_a"), (unsigned int)0);
    UP_ASSERT_EQUAL(angle_data->getTypeByName("angle_b"), (unsigned int)1);

    UP_ASSERT_EQUAL(angle_data->getNameByType(0), string("angle_a"));
    UP_ASSERT_EQUAL(angle_data->getNameByType(1), string("angle_b"));

    // verify each angle
    Angle a = angle_data->getGroupByTag(0);
    UP_ASSERT_EQUAL(a.a, (unsigned int)0);
    UP_ASSERT_EQUAL(a.b, (unsigned int)1);
    UP_ASSERT_EQUAL(a.c, (unsigned int)2);
    UP_ASSERT_EQUAL(a.type, (unsigned int)0);

    a = angle_data->getGroupByTag(1);
    UP_ASSERT_EQUAL(a.a, (unsigned int)1);
    UP_ASSERT_EQUAL(a.b, (unsigned int)2);
    UP_ASSERT_EQUAL(a.c, (unsigned int)3);
    UP_ASSERT_EQUAL(a.type, (unsigned int)1);

    a = angle_data->getGroupByTag(2);
    UP_ASSERT_EQUAL(a.a, (unsigned int)2);
    UP_ASSERT_EQUAL(a.b, (unsigned int)3);
    UP_ASSERT_EQUAL(a.c, (unsigned int)4);
    UP_ASSERT_EQUAL(a.type, (unsigned int)0);

    // check the dihedrals
    std::shared_ptr<DihedralData> dihedral_data = sysdef->getDihedralData();

    // 2 dihedrals should have been read in
    UP_ASSERT_EQUAL(dihedral_data->getN(), (unsigned int)2);

    // check that the types have been named properly
    UP_ASSERT_EQUAL(dihedral_data->getNTypes(), (unsigned int)2);
    UP_ASSERT_EQUAL(dihedral_data->getTypeByName("di_a"), (unsigned int)0);
    UP_ASSERT_EQUAL(dihedral_data->getTypeByName("di_b"), (unsigned int)1);

    UP_ASSERT_EQUAL(dihedral_data->getNameByType(0), string("di_a"));
    UP_ASSERT_EQUAL(dihedral_data->getNameByType(1), string("di_b"));

    // verify each dihedral
    Dihedral d = dihedral_data->getGroupByTag(0);
    UP_ASSERT_EQUAL(d.a, (unsigned int)0);
    UP_ASSERT_EQUAL(d.b, (unsigned int)1);
    UP_ASSERT_EQUAL(d.c, (unsigned int)2);
    UP_ASSERT_EQUAL(d.d, (unsigned int)3);
    UP_ASSERT_EQUAL(d.type, (unsigned int)0);

    d = dihedral_data->getGroupByTag(1);
    UP_ASSERT_EQUAL(d.a, (unsigned int)1);
    UP_ASSERT_EQUAL(d.b, (unsigned int)2);
    UP_ASSERT_EQUAL(d.c, (unsigned int)3);
    UP_ASSERT_EQUAL(d.d, (unsigned int)4);
    UP_ASSERT_EQUAL(d.type, (unsigned int)1);


    // check the impropers
    std::shared_ptr<ImproperData> improper_data = sysdef->getImproperData();

    // 2 dihedrals should have been read in
    UP_ASSERT_EQUAL(improper_data->getN(), (unsigned int)2);

    // check that the types have been named properly
    UP_ASSERT_EQUAL(improper_data->getNTypes(), (unsigned int)2);
    UP_ASSERT_EQUAL(improper_data->getTypeByName("im_a"), (unsigned int)0);
    UP_ASSERT_EQUAL(improper_data->getTypeByName("im_b"), (unsigned int)1);

    UP_ASSERT_EQUAL(improper_data->getNameByType(0), string("im_a"));
    UP_ASSERT_EQUAL(improper_data->getNameByType(1), string("im_b"));

    // verify each dihedral
    d = improper_data->getGroupByTag(0);
    UP_ASSERT_EQUAL(d.a, (unsigned int)3);
    UP_ASSERT_EQUAL(d.b, (unsigned int)2);
    UP_ASSERT_EQUAL(d.c, (unsigned int)1);
    UP_ASSERT_EQUAL(d.d, (unsigned int)0);
    UP_ASSERT_EQUAL(d.type, (unsigned int)0);

    d = improper_data->getGroupByTag(1);
    UP_ASSERT_EQUAL(d.a, (unsigned int)5);
    UP_ASSERT_EQUAL(d.b, (unsigned int)4);
    UP_ASSERT_EQUAL(d.c, (unsigned int)3);
    UP_ASSERT_EQUAL(d.d, (unsigned int)2);
    UP_ASSERT_EQUAL(d.type, (unsigned int)1);

    // check the constraints
    std::shared_ptr<ConstraintData> constraint_data = sysdef->getConstraintData();

    // 2 dihedrals should have been read in
    UP_ASSERT_EQUAL(constraint_data->getNGlobal(), (unsigned int)2);

    // verify each dihedral
    Constraint c = constraint_data->getGroupByTag(0);
    UP_ASSERT_EQUAL(c.a, (unsigned int)0);
    UP_ASSERT_EQUAL(c.b, (unsigned int)1);
    UP_ASSERT_EQUAL(c.d, Scalar(1.5));

    // verify each dihedral
    c = constraint_data->getGroupByTag(1);
    UP_ASSERT_EQUAL(c.a, (unsigned int)1);
    UP_ASSERT_EQUAL(c.b, (unsigned int)2);
    UP_ASSERT_EQUAL(c.d, Scalar(2.5));


    // clean up after ourselves
    unlink((tmp_path+"/test_input.xml").c_str());
    }
