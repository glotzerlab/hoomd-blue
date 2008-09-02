/*
Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
Source Software License
Copyright (c) 2008 Ames Laboratory Iowa State University
All rights reserved.

Redistribution and use of HOOMD, in source and binary forms, with or
without modification, are permitted, provided that the following
conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names HOOMD's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$

//! Name the unit test module
#define BOOST_TEST_MODULE XMLReaderWriterTest
#include "boost_utf_configure.h"

#include <boost/test/floating_point_comparison.hpp>

#include <math.h>
#include "HOOMDDumpWriter.h"
#include "HOOMDInitializer.h"
#include "BondData.h"

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
BOOST_AUTO_TEST_CASE( HOOMDDumpWriterBasicTests )
	{
	// start by creating a single particle system: see it the correct file is written
	BoxDim box(Scalar(2.5), Scalar(4.5), Scalar(12.1));
	int n_types = 5;
	int n_bond_types = 2;
	shared_ptr<ParticleData> pdata(new ParticleData(2, box, n_types, n_bond_types));
	// set recognizable values for the particle
	const ParticleDataArrays array = pdata->acquireReadWrite();
	array.x[0] = 1.1;
	array.y[0] = 2.1234567890123456;
	array.z[0] = -5.76;
	
	array.vx[0] = -1.4567;
	array.vy[0] = -10.0987654321098765;
	array.vz[0] = 56.78;
	
	array.type[0] = 3;
	
	array.x[1] = 1.2;
	array.y[1] = 2.1;
	array.z[1] = -3.4;
	
	array.vx[1] = -1.5;
	array.vy[1] = -10.6;
	array.vz[1] = 5.7;
	
	array.type[1] = 0;
	pdata->release();
	
	// add a couple walls for fun
	pdata->getWallData()->addWall(Wall(1,0,0, 0,1,0));
	pdata->getWallData()->addWall(Wall(0,1,0, 0,0,1));
	pdata->getWallData()->addWall(Wall(0,0,1, 1,0,0));
	
	// add a few bonds too
	pdata->getBondData()->addBond(Bond(0, 0, 1));
	pdata->getBondData()->addBond(Bond(1, 1, 0));
	
	// create the writer
	shared_ptr<HOOMDDumpWriter> writer(new HOOMDDumpWriter(pdata, "test"));
	
	// first file written will have all outputs disabled
	writer->setOutputPosition(false);
	writer->setOutputVelocity(false);
	writer->setOutputType(false);

	// first test
		{
		// make sure the first output file is deleted
		remove_all("test.0000000000.xml");
		BOOST_REQUIRE(!exists("test.0000000000.xml"));
		
		// write the first output
		writer->analyze(0);
		
		// make sure the file was created
		BOOST_REQUIRE(exists("test.0000000000.xml"));
		
		// check the output line by line
		ifstream f("test.0000000000.xml");
		string line;
		getline(f, line);
		BOOST_CHECK_EQUAL(line, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
		BOOST_REQUIRE(!f.bad());
		
		getline(f, line);
		BOOST_CHECK_EQUAL(line, "<hoomd_xml>");
		BOOST_REQUIRE(!f.bad());
		
		getline(f, line);
		BOOST_CHECK_EQUAL(line,  "<configuration time_step=\"0\">");
		BOOST_REQUIRE(!f.bad());
		
		getline(f, line);
		BOOST_CHECK_EQUAL(line,  "<box units=\"sigma\"  Lx=\"2.5\" Ly=\"4.5\" Lz=\"12.1\"/>");
		BOOST_REQUIRE(!f.bad());
		
		getline(f, line);
		BOOST_CHECK_EQUAL(line,  "</configuration>");
		BOOST_REQUIRE(!f.bad());

		getline(f, line);
		BOOST_CHECK_EQUAL(line,  "</hoomd_xml>");
		BOOST_REQUIRE(!f.bad());
		f.close();
		}
		
	// second test: test position
		{
		writer->setOutputPosition(true);
		
		// make sure the first output file is deleted
		remove_all("test.0000000010.xml");
		BOOST_REQUIRE(!exists("test.0000000010.xml"));	
		
		// write the file
		writer->analyze(10);
		
		// make sure the file was created
		BOOST_REQUIRE(exists("test.0000000010.xml"));
		
		// assume that the first lines tested in the first case are still OK and skip them
		ifstream f("test.0000000010.xml");
		string line;
		getline(f, line); // <?xml
		getline(f, line); // <HOOMD_xml
		getline(f, line); // <Configuration
		getline(f, line); // <Box
		
		getline(f, line);
		BOOST_CHECK_EQUAL(line, "<position units=\"sigma\">");
		BOOST_REQUIRE(!f.bad());
		
		getline(f, line);
		BOOST_CHECK_EQUAL(line, "1.1 2.12346 -5.76");
		BOOST_REQUIRE(!f.bad());
		
		getline(f, line);
		BOOST_CHECK_EQUAL(line, "1.2 2.1 -3.4");
		BOOST_REQUIRE(!f.bad());
		
		getline(f, line);
		BOOST_CHECK_EQUAL(line, "</position>");
		
		getline(f, line); // </configuration
		getline(f, line); // </HOOMD_xml
		f.close();
		}
		
	// third test: test velocity
		{
		writer->setOutputPosition(false);
		writer->setOutputVelocity(true);
		
		// make sure the first output file is deleted
		remove_all("test.0000000020.xml");
		BOOST_REQUIRE(!exists("test.0000000020.xml"));	
		
		// write the file
		writer->analyze(20);
		
		// assume that the first lines tested in the first case are still OK and skip them
		ifstream f("test.0000000020.xml");
		string line;
		getline(f, line); // <?xml
		getline(f, line); // <HOOMD_xml
		getline(f, line); // <Configuration
		getline(f, line); // <Box
		
		getline(f, line);
		BOOST_CHECK_EQUAL(line, "<velocity units=\"sigma/tau\">");
		BOOST_REQUIRE(!f.bad());
		
		getline(f, line);
		BOOST_CHECK_EQUAL(line, "-1.4567 -10.0988 56.78");
		BOOST_REQUIRE(!f.bad());
		
		getline(f, line);
		BOOST_CHECK_EQUAL(line, "-1.5 -10.6 5.7");
		BOOST_REQUIRE(!f.bad());
	
		getline(f, line);
		BOOST_CHECK_EQUAL(line, "</velocity>");
		f.close();
		}
		
	// fourth test: the type array
		{
		writer->setOutputPosition(false);
		writer->setOutputVelocity(false);
		writer->setOutputType(true);
		
		// make sure the first output file is deleted
		remove_all("test.0000000030.xml");
		BOOST_REQUIRE(!exists("test.0000000030.xml"));	
		
		// write the file
		writer->analyze(30);
		
		// assume that the first lines tested in the first case are still OK and skip them
		ifstream f("test.0000000030.xml");
		string line;
		getline(f, line); // <?xml
		getline(f, line); // <HOOMD_xml
		getline(f, line); // <Configuration
		getline(f, line); // <Box
		
		getline(f, line);
		BOOST_CHECK_EQUAL(line, "<type>");
		BOOST_REQUIRE(!f.bad());
		
		getline(f, line);
		BOOST_CHECK_EQUAL(line, "D");
		BOOST_REQUIRE(!f.bad());
		
		getline(f, line);
		BOOST_CHECK_EQUAL(line, "A");
		BOOST_REQUIRE(!f.bad());
		
		getline(f, line);
		BOOST_CHECK_EQUAL(line, "</type>");
		f.close();
		}

	// fifth test: the wall array
		{
		writer->setOutputPosition(false);
		writer->setOutputVelocity(false);
		writer->setOutputType(false);
		writer->setOutputWall(true);
		
		// make sure the first output file is deleted
		remove_all("test.0000000040.xml");
		BOOST_REQUIRE(!exists("test.0000000040.xml"));	
		
		// write the file
		writer->analyze(40);
		
		// assume that the first lines tested in the first case are still OK and skip them
		ifstream f("test.0000000040.xml");
		string line;
		getline(f, line); // <?xml
		getline(f, line); // <HOOMD_xml
		getline(f, line); // <Configuration
		getline(f, line); // <Box
		
		getline(f, line);
		BOOST_CHECK_EQUAL(line, "<wall>");
		BOOST_REQUIRE(!f.bad());
		
		getline(f, line);
		BOOST_CHECK_EQUAL(line, "<coord ox=\"1\" oy=\"0\" oz=\"0\" nx=\"0\" ny=\"1\" nz=\"0\" />");
		BOOST_REQUIRE(!f.bad());
		
		getline(f, line);
		BOOST_CHECK_EQUAL(line, "<coord ox=\"0\" oy=\"1\" oz=\"0\" nx=\"0\" ny=\"0\" nz=\"1\" />");
		BOOST_REQUIRE(!f.bad());
		
		getline(f, line);
		BOOST_CHECK_EQUAL(line, "<coord ox=\"0\" oy=\"0\" oz=\"1\" nx=\"1\" ny=\"0\" nz=\"0\" />");
		BOOST_REQUIRE(!f.bad());
		
		getline(f, line);
		BOOST_CHECK_EQUAL(line, "</wall>");
		f.close();
		}

	// sixth test: the bond array
		{
		writer->setOutputPosition(false);
		writer->setOutputVelocity(false);
		writer->setOutputType(false);
		writer->setOutputWall(false);
		writer->setOutputBond(true);
		
		// make sure the first output file is deleted
		remove_all("test.0000000050.xml");
		BOOST_REQUIRE(!exists("test.0000000050.xml"));
		
		// write the file
		writer->analyze(50);
		
		// assume that the first lines tested in the first case are still OK and skip them
		ifstream f("test.0000000050.xml");
		string line;
		getline(f, line); // <?xml
		getline(f, line); // <HOOMD_xml
		getline(f, line); // <Configuration
		getline(f, line); // <Box
		
		getline(f, line);
		BOOST_CHECK_EQUAL(line, "<bond>");
		BOOST_REQUIRE(!f.bad());
		
		getline(f, line);
		BOOST_CHECK_EQUAL(line, "bondA 0 1");
		BOOST_REQUIRE(!f.bad());
		
		getline(f, line);
		BOOST_CHECK_EQUAL(line, "bondB 1 0");
		BOOST_REQUIRE(!f.bad());
		
		getline(f, line);
		BOOST_CHECK_EQUAL(line, "</bond>");
		f.close();
		}


	remove_all("test.0000000000.xml");
	remove_all("test.0000000010.xml");
	remove_all("test.0000000020.xml");
	remove_all("test.0000000030.xml");
	remove_all("test.0000000040.xml");
	remove_all("test.0000000050.xml");
	}

//! Tests the ability of HOOMDDumpWriter to handle tagged and reordered particles
BOOST_AUTO_TEST_CASE( HOOMDDumpWriter_tag_test )
	{
	// start by creating a single particle system: see it the correct file is written
	BoxDim box(Scalar(100.5), Scalar(120.5), Scalar(130.5));
	int n_types = 10;
	shared_ptr<ParticleData> pdata(new ParticleData(6, box, n_types));
	
	// this is the shuffle order of the particles
	unsigned int tags[6] = { 5, 2, 3, 1, 0, 4 };
	unsigned int rtags[6] = { 4, 3, 1, 2, 5, 0 };
	
	// set recognizable values for the particle
	const ParticleDataArrays array = pdata->acquireReadWrite();
	for (int i = 0; i < 6; i++)
		{
		array.tag[i] = tags[i];
		unsigned int tag = tags[i];
		
		array.x[i] = Scalar(tag)+Scalar(0.1);
		array.y[i] = Scalar(tag)+Scalar(1.1);
		array.z[i] = Scalar(tag)+Scalar(2.1);

		array.vx[i] = Scalar(tag)*Scalar(10.0);
		array.vy[i] = Scalar(tag)*Scalar(11.0);
		array.vz[i] = Scalar(tag)*Scalar(12.0);
		
		array.type[i] = tag + 2;
		array.rtag[i] = rtags[i];
		}
	pdata->release();
	
	// create the writer
	shared_ptr<HOOMDDumpWriter> writer(new HOOMDDumpWriter(pdata, "test"));
	
	// write the file with all outputs enabled
	writer->setOutputPosition(true);
	writer->setOutputVelocity(true);
	writer->setOutputType(true);
	
	// now the big mess: check the file line by line		
		{
		// make sure the first output file is deleted
		remove_all("test.0000000100.xml");
		BOOST_REQUIRE(!exists("test.0000000100.xml"));
		
		// write the first output
		writer->analyze(100);
		
		// make sure the file was created
		BOOST_REQUIRE(exists("test.0000000100.xml"));
		
		// check the output line by line
		ifstream f("test.0000000100.xml");
		string line;
		getline(f, line);
		BOOST_CHECK_EQUAL(line, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
		BOOST_REQUIRE(!f.bad());
		
		getline(f, line);
		BOOST_CHECK_EQUAL(line, "<hoomd_xml>");
		BOOST_REQUIRE(!f.bad());
		
		getline(f, line);
		BOOST_CHECK_EQUAL(line,  "<configuration time_step=\"100\">");
		BOOST_REQUIRE(!f.bad());
		
		getline(f, line);
		BOOST_CHECK_EQUAL(line,  "<box units=\"sigma\"  Lx=\"100.5\" Ly=\"120.5\" Lz=\"130.5\"/>");
		BOOST_REQUIRE(!f.bad());
		
		getline(f, line);
		BOOST_CHECK_EQUAL(line, "<position units=\"sigma\">");
		BOOST_REQUIRE(!f.bad());
		
		// check all the positions
		getline(f, line);
		BOOST_CHECK_EQUAL(line, "0.1 1.1 2.1");
		BOOST_REQUIRE(!f.bad());
		
		getline(f, line);
		BOOST_CHECK_EQUAL(line, "1.1 2.1 3.1");
		BOOST_REQUIRE(!f.bad());
		
		getline(f, line);
		BOOST_CHECK_EQUAL(line, "2.1 3.1 4.1");
		BOOST_REQUIRE(!f.bad());
		
		getline(f, line);
		BOOST_CHECK_EQUAL(line, "3.1 4.1 5.1");
		BOOST_REQUIRE(!f.bad());
		
		getline(f, line);			
		BOOST_CHECK_EQUAL(line, "4.1 5.1 6.1");
		BOOST_REQUIRE(!f.bad());
		
		getline(f, line);
		BOOST_CHECK_EQUAL(line, "5.1 6.1 7.1");
		BOOST_REQUIRE(!f.bad());
		
		getline(f, line);
		BOOST_CHECK_EQUAL(line,  "</position>");
		BOOST_REQUIRE(!f.bad());
		
		// check all velocities
		getline(f, line);
		BOOST_CHECK_EQUAL(line, "<velocity units=\"sigma/tau\">");
		BOOST_REQUIRE(!f.bad());
		
		getline(f, line);
		BOOST_CHECK_EQUAL(line, "0 0 0");
		BOOST_REQUIRE(!f.bad());

		getline(f, line);
		BOOST_CHECK_EQUAL(line, "10 11 12");
		BOOST_REQUIRE(!f.bad());

		getline(f, line);
		BOOST_CHECK_EQUAL(line, "20 22 24");
		BOOST_REQUIRE(!f.bad());
		
		getline(f, line);
		BOOST_CHECK_EQUAL(line, "30 33 36");
		BOOST_REQUIRE(!f.bad());

		getline(f, line);
		BOOST_CHECK_EQUAL(line, "40 44 48");
		BOOST_REQUIRE(!f.bad());
		
		getline(f, line);
		BOOST_CHECK_EQUAL(line, "50 55 60");
		BOOST_REQUIRE(!f.bad());

		getline(f, line);
		BOOST_CHECK_EQUAL(line, "</velocity>");
		
		// check all types
		getline(f, line);
		BOOST_CHECK_EQUAL(line, "<type>");
		BOOST_REQUIRE(!f.bad());
		
		getline(f, line);
		BOOST_CHECK_EQUAL(line, "C");
		BOOST_REQUIRE(!f.bad());
		
		getline(f, line);
		BOOST_CHECK_EQUAL(line, "D");
		BOOST_REQUIRE(!f.bad());
		
		getline(f, line);
		BOOST_CHECK_EQUAL(line, "E");
		BOOST_REQUIRE(!f.bad());
		
		getline(f, line);
		BOOST_CHECK_EQUAL(line, "F");
		BOOST_REQUIRE(!f.bad());
		
		getline(f, line);
		BOOST_CHECK_EQUAL(line, "G");
		BOOST_REQUIRE(!f.bad());
		
		getline(f, line);
		BOOST_CHECK_EQUAL(line, "H");
		BOOST_REQUIRE(!f.bad());
		
		getline(f, line);
		BOOST_CHECK_EQUAL(line, "</type>");	

		getline(f, line);
		BOOST_CHECK_EQUAL(line,  "</configuration>");
		BOOST_REQUIRE(!f.bad());

		getline(f, line);
		BOOST_CHECK_EQUAL(line,  "</hoomd_xml>");
		BOOST_REQUIRE(!f.bad());
		f.close();
		remove_all("test.0000000100.xml");
		}
	}

//! Test basic functionality of HOOMDInitializer
BOOST_AUTO_TEST_CASE( HOOMDInitializer_basic_tests )
	{
	// create a test input file
	ofstream f("test_input.xml");
f << "<?xml version =\"1.0\" encoding =\"UTF-8\" ?>\n\
<hoomd_xml>\n\
<configuration time_step=\"150000000\">\n\
<box Units =\"sigma\"  Lx=\"20.05\" Ly= \"32.12345\" Lz=\"45.098\" />\n\
<position units =\"sigma\" >\n\
1.4 2.567890 3.45\n\
2.4 3.567890 4.45\n\
3.4 4.567890 5.45\n\
4.4 5.567890 6.45\n\
5.4 6.567890 7.45\n\
6.4 7.567890 8.45\n\
</position>\n\
<velocity units =\"sigma/tau\">\n\
10.12 12.1567 1.056\n\
20.12 22.1567 2.056\n\
30.12 32.1567 3.056\n\
40.12 42.1567 4.056\n\
50.12 52.1567 5.056\n\
60.12 62.1567 6.056\n\
</velocity>\n\
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
<wall>\n\
<coord ox=\"1.0\" oy=\"2.0\" oz=\"3.0\" nx=\"4.0\" ny=\"5.0\" nz=\"6.0\"/>\n\
<coord ox=\"7.0\" oy=\"8.0\" oz=\"9.0\" nx=\"10.0\" ny=\"11.0\" nz=\"-12.0\"/>\n\
</wall>\n\
<bond>\n\
bond_a 0 1\n\
bond_b 1 2\n\
bond_a 2 3\n\
bond_c 3 4\n\
</bond>\n\
</configuration>\n\
</hoomd_xml>" << endl;
	f.close();

	// now that we have created a test file, load it up into a pdata
	HOOMDInitializer init("test_input.xml");
	shared_ptr<ParticleData> pdata(new ParticleData(init));
	
	// verify all parameters
	BOOST_CHECK_EQUAL(init.getTimeStep(), (unsigned int)150000000);
	BOOST_CHECK_EQUAL(pdata->getN(), (unsigned int)6);
	BOOST_CHECK_EQUAL(pdata->getNTypes(), (unsigned int)6);
	MY_BOOST_CHECK_CLOSE(pdata->getBox().xhi - pdata->getBox().xlo, 20.05, tol);
	MY_BOOST_CHECK_CLOSE(pdata->getBox().yhi - pdata->getBox().ylo, 32.12345, tol);
	MY_BOOST_CHECK_CLOSE(pdata->getBox().zhi - pdata->getBox().zlo, 45.098, tol);
	
	ParticleDataArraysConst arrays = pdata->acquireReadOnly();
	for (int i = 0; i < 6; i++)
		{
		MY_BOOST_CHECK_CLOSE(arrays.x[i], Scalar(i) + Scalar(1.4), tol);
		MY_BOOST_CHECK_CLOSE(arrays.y[i], Scalar(i) + Scalar(2.567890), tol);
		MY_BOOST_CHECK_CLOSE(arrays.z[i], Scalar(i) + Scalar(3.45), tol);
		
		MY_BOOST_CHECK_CLOSE(arrays.vx[i], Scalar(i+1)*Scalar(10.0) + Scalar(0.12), tol);
		MY_BOOST_CHECK_CLOSE(arrays.vy[i], Scalar(i+1)*Scalar(10.0) + Scalar(2.1567), tol);
		MY_BOOST_CHECK_CLOSE(arrays.vz[i], Scalar(i+1) + Scalar(0.056), tol);

		MY_BOOST_CHECK_CLOSE(arrays.charge[i], Scalar(i)*Scalar(10.0), tol);
		
		// checking that the type is correct becomes tricky because types are identified by 
		// string
		ostringstream type_name;
		type_name << 5-i;	// the expected type is the integer 5-i
		BOOST_CHECK_EQUAL(arrays.type[i], pdata->getTypeByName(type_name.str()));
		BOOST_CHECK_EQUAL(arrays.tag[i], (unsigned int)i);
		BOOST_CHECK_EQUAL(arrays.rtag[i], (unsigned int)i);
		}
	pdata->release();

	// check the walls
	BOOST_REQUIRE_EQUAL(pdata->getWallData()->getNumWalls(), (unsigned int)2);
	Wall wall1 = pdata->getWallData()->getWall(0);
	MY_BOOST_CHECK_CLOSE(wall1.origin_x, 1.0, tol);
	MY_BOOST_CHECK_CLOSE(wall1.origin_y, 2.0, tol);
	MY_BOOST_CHECK_CLOSE(wall1.origin_z, 3.0, tol);
	// normals are made unit length when loaded, so these values differ from the ones in the file
	MY_BOOST_CHECK_CLOSE(wall1.normal_x, 0.455842306, tol);
	MY_BOOST_CHECK_CLOSE(wall1.normal_y, 0.569802882, tol);
	MY_BOOST_CHECK_CLOSE(wall1.normal_z, 0.683763459, tol);

	Wall wall2 = pdata->getWallData()->getWall(1);
	MY_BOOST_CHECK_CLOSE(wall2.origin_x, 7.0, tol);
	MY_BOOST_CHECK_CLOSE(wall2.origin_y, 8.0, tol);
	MY_BOOST_CHECK_CLOSE(wall2.origin_z, 9.0, tol);
	// normals are made unit length when loaded, so these values differ from the ones in the file
	MY_BOOST_CHECK_CLOSE(wall2.normal_x, 0.523423923, tol);
	MY_BOOST_CHECK_CLOSE(wall2.normal_y, 0.575766315, tol);
	MY_BOOST_CHECK_CLOSE(wall2.normal_z, -0.628108707, tol);

	// check the bonds
	boost::shared_ptr<BondData> bond_data = pdata->getBondData();
	
	// 4 bonds should have been read in
	BOOST_REQUIRE_EQUAL(bond_data->getNumBonds(), (unsigned int)4);
	
	// check that the types have been named properly
	BOOST_REQUIRE_EQUAL(bond_data->getNBondTypes(), (unsigned int)3);
	BOOST_CHECK_EQUAL(bond_data->getTypeByName("bond_a"), (unsigned int)0);
	BOOST_CHECK_EQUAL(bond_data->getTypeByName("bond_b"), (unsigned int)1);
	BOOST_CHECK_EQUAL(bond_data->getTypeByName("bond_c"), (unsigned int)2);
	
	BOOST_CHECK_EQUAL(bond_data->getNameByType(0), string("bond_a"));
	BOOST_CHECK_EQUAL(bond_data->getNameByType(1), string("bond_b"));
	BOOST_CHECK_EQUAL(bond_data->getNameByType(2), string("bond_c"));
	
	// verify each bond
	Bond b = bond_data->getBond(0);
	BOOST_CHECK_EQUAL(b.a, (unsigned int)0);
	BOOST_CHECK_EQUAL(b.b, (unsigned int)1);
	BOOST_CHECK_EQUAL(b.type, (unsigned int)0);
	
	b = bond_data->getBond(1);
	BOOST_CHECK_EQUAL(b.a, (unsigned int)1);
	BOOST_CHECK_EQUAL(b.b, (unsigned int)2);
	BOOST_CHECK_EQUAL(b.type, (unsigned int)1);
	
	b = bond_data->getBond(2);
	BOOST_CHECK_EQUAL(b.a, (unsigned int)2);
	BOOST_CHECK_EQUAL(b.b, (unsigned int)3);
	BOOST_CHECK_EQUAL(b.type, (unsigned int)0);
	
	b = bond_data->getBond(3);
	BOOST_CHECK_EQUAL(b.a, (unsigned int)3);
	BOOST_CHECK_EQUAL(b.b, (unsigned int)4);
	BOOST_CHECK_EQUAL(b.type, (unsigned int)2);
	
	// clean up after ourselves
	remove_all("test_input.xml");
	}
