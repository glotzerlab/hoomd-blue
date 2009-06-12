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
// Maintainer: joaander

/*! \file HOOMDInitializer.cc
	\brief Defines the HOOMDInitializer class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 4267 )
#endif

#include "HOOMDInitializer.h"

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <algorithm>

using namespace std;

#include <boost/python.hpp>
using namespace boost::python;

using namespace boost;

/*! \param fname File name with the data to load
	The file will be read and parsed fully during the constructor call.
*/
HOOMDInitializer::HOOMDInitializer(const std::string &fname)
 	{
	// initialize member variables
	m_timestep = 0;
	m_box_read = false;

	// initialize the parser map
	m_parser_map["box"] = bind(&HOOMDInitializer::parseBoxNode, this, _1);
	m_parser_map["position"] = bind(&HOOMDInitializer::parsePositionNode, this, _1);
	m_parser_map["image"] = bind(&HOOMDInitializer::parseImageNode, this, _1);
	m_parser_map["velocity"] = bind(&HOOMDInitializer::parseVelocityNode, this, _1);
	m_parser_map["mass"] = bind(&HOOMDInitializer::parseMassNode, this, _1);
	m_parser_map["diameter"] = bind(&HOOMDInitializer::parseDiameterNode, this, _1);
	m_parser_map["type"] = bind(&HOOMDInitializer::parseTypeNode, this, _1);
	m_parser_map["bond"] = bind(&HOOMDInitializer::parseBondNode, this, _1);
	m_parser_map["angle"] = bind(&HOOMDInitializer::parseAngleNode, this, _1);
	m_parser_map["dihedral"] = bind(&HOOMDInitializer::parseDihedralNode, this, _1);
	m_parser_map["improper"] = bind(&HOOMDInitializer::parseImproperNode, this, _1);
	m_parser_map["charge"] = bind(&HOOMDInitializer::parseChargeNode, this, _1);
	m_parser_map["wall"] = bind(&HOOMDInitializer::parseWallNode, this, _1);

	// read in the file
	readFile(fname);
	}

/* XXX: shouldn't the following methods be put into 
 * the header so that they get inlined? */

/*! \returns Numer of particles parsed from the XML file
*/
unsigned int HOOMDInitializer::getNumParticles() const
	{
	assert(m_pos_array.size() > 0);
	return (unsigned int)m_pos_array.size();
	}

/*! \returns Numer of particle types parsed from the XML file
*/
unsigned int HOOMDInitializer::getNumParticleTypes() const
	{
	assert(m_type_mapping.size() > 0);
	return (unsigned int)m_type_mapping.size();
	}

/*! \returns Box dimensions parsed from the XML file
*/
BoxDim HOOMDInitializer::getBox() const
	{		
	return m_box;
	}

/*! \returns Time step parsed from the XML file
*/
unsigned int HOOMDInitializer::getTimeStep() const
	{
	return m_timestep;
	}

/* change internal timestep number. */
void HOOMDInitializer::setTimeStep(unsigned int ts) 
   {
       m_timestep = ts;
   }

/*! \param pdata The particle data 
	
	initArrays takes the internally stored copy of the particle data and copies it
	into the provided particle data arrays for storage in ParticleData.
*/
void HOOMDInitializer::initArrays(const ParticleDataArrays &pdata) const
	{
	assert(m_pos_array.size() > 0 && m_pos_array.size() == pdata.nparticles);

	// loop through all the particles and set them up
	for (unsigned int i = 0; i < m_pos_array.size(); i++)
		{
		pdata.x[i] = m_pos_array[i].x;
		pdata.y[i] = m_pos_array[i].y;
		pdata.z[i] = m_pos_array[i].z;

		pdata.tag[i] = i;
		pdata.rtag[i] = i;
		}
		
	if (m_image_array.size() != 0)
		{
		assert(m_image_array.size() == m_pos_array.size());

		for (unsigned int i = 0; i < m_pos_array.size(); i++)
			{
			pdata.ix[i] = m_image_array[i].x;
			pdata.iy[i] = m_image_array[i].y;
			pdata.iz[i] = m_image_array[i].z;
			}
		}
	
	if (m_vel_array.size() != 0)
		{
		assert(m_vel_array.size() == m_pos_array.size());

		for (unsigned int i = 0; i < m_pos_array.size(); i++)
			{
			pdata.vx[i] = m_vel_array[i].x;
			pdata.vy[i] = m_vel_array[i].y;
			pdata.vz[i] = m_vel_array[i].z;
			}
		}
		
	if (m_mass_array.size() != 0)
		{
		assert(m_mass_array.size() == m_pos_array.size());
		
		for (unsigned int i = 0; i < m_pos_array.size(); i++)
			pdata.mass[i] = m_mass_array[i];
		}
		
	if (m_diameter_array.size() != 0)
		{
		assert(m_diameter_array.size() == m_pos_array.size());
		
		for (unsigned int i = 0; i < m_pos_array.size(); i++)
			pdata.diameter[i] = m_diameter_array[i];
		}

	if (m_charge_array.size() != 0)
		{
		assert(m_charge_array.size() == m_pos_array.size());
		
		for (unsigned int i = 0; i < m_pos_array.size(); i++)
			pdata.charge[i] = m_charge_array[i];
		}

	if (m_type_array.size() != 0)
		{
		assert(m_type_array.size() == m_pos_array.size());
		
		for (unsigned int i = 0; i < m_pos_array.size(); i++)
			pdata.type[i] = m_type_array[i];
		}
	}

/*! \param wall_data WallData to initialize with the data read from the file
*/
void HOOMDInitializer::initWallData(boost::shared_ptr<WallData> wall_data) const
	{
	// copy the walls over from our internal list
	for (unsigned int i = 0; i < m_walls.size(); i++)
		wall_data->addWall(m_walls[i]);
	}

/*!	\param fname File name of the hoomd_xml file to read in
	\post Internal data arrays and members are filled out from which futre calls
	like initArrays will use to intialize the ParticleData
	
	This function implements the main parser loop. It reads in XML nodes from the
	file one by one and passes them of to parsers registered in \c m_parser_map.
*/
void HOOMDInitializer::readFile(const string &fname)
	{
	// Create a Root Node and a child node
	XMLNode root_node;

	// Open the file and read the root element "hoomd_xml"
	cout<< "Reading " << fname << "..." << endl;
	XMLResults results;
	root_node = XMLNode::parseFile(fname.c_str(),"hoomd_xml", &results);

    // handle errors
    if (results.error != eXMLErrorNone)
		{
        // create message
        if (results.error==eXMLErrorFirstTagNotFound)
			{
			cerr << endl << "***Error! Root node of " << fname << " is not <hoomd_xml>" << endl << endl;
			throw runtime_error("Error reading xml file");
			}
		
		ostringstream error_message;
		error_message << XMLNode::getError(results.error) << " in file " << fname << " at line " << results.nLine << " col " << results.nColumn;
		cerr << endl << "***Error! " << error_message.str() << endl << endl;
		throw runtime_error("Error reading xml file");
		}
		
	string xml_version;
	if (root_node.isAttributeSet("version"))
		{
		xml_version = root_node.getAttribute("version");
		}
	else
		{
		cout << "Notice: No version specified in hoomd_xml root node: assuming 1.0" << endl;
		xml_version = string("1.0");
		}
		
	// right now, the version tag doesn't do anything: just warn if it is not 1.0 or 1.1
	if ((xml_version != string("1.0")) && (xml_version != string("1.1")))
		cout << endl << "***Warning! hoomd_xml file with version other than 1.0 or 1.1 specified, I don't know how to read this. Continuing anyways." << endl << endl;

	// the file was parsed successfully by the XML reader. Extract the information now
	// start by checking the number of configurations in the file
	int num_configurations = root_node.nChildNode("configuration");
	if (num_configurations == 0)
		{
		cerr << endl << "***Error! No <configuration> specified in the XML file" << endl << endl;
		throw runtime_error("Error reading xml file");
		}
	if (num_configurations > 1)
		{
		cerr << endl << "***Error! Sorry, the input XML file must have only one configuration" << endl << endl;
		throw runtime_error("Error reading xml file");
		}
	
	// extract the only configuration node
	XMLNode configuration_node = root_node.getChildNode("configuration");
	// extract the time step
	if (configuration_node.isAttributeSet("time_step"))
		{
		m_timestep = atoi(configuration_node.getAttribute("time_step"));
		}

	// loop through all child nodes of the configuration
	for (int cur_node=0; cur_node < configuration_node.nChildNode(); cur_node++)
		{
		// extract the name and call the appropriate node parser, if it exists
		XMLNode node = configuration_node.getChildNode(cur_node);
		string name = node.getName();
		transform(name.begin(), name.end(), name.begin(), ::tolower);
		
		std::map< std::string, boost::function< void (const XMLNode&) > >::iterator parser;
		parser = m_parser_map.find(name);
		if (parser != m_parser_map.end())
			parser->second(node);
		else
			cout << "Notice: Parser for node <" << name << "> not defined, ignoring" << endl;
		}
	
	// check for required items in the file
	if (!m_box_read)
		{
		cerr << endl << "***Error! A <box> node is required to define the dimensions of the simulation box" << endl << endl;
		throw runtime_error("Error extracting data from hoomd_xml file");
		}
	if (m_pos_array.size() == 0)
		{
		cerr << endl << "***Error! No particles defined in <position> node" << endl << endl;
		throw runtime_error("Error extracting data from hoomd_xml file");
		}

	// check for potential user errors
	if (m_vel_array.size() != 0 && m_vel_array.size() != m_pos_array.size())
		{
		cerr << endl << "***Error! " << m_vel_array.size() << " velocities != " << m_pos_array.size() << " positions" << endl << endl;
		throw runtime_error("Error extracting data from hoomd_xml file");
		}
	if (m_mass_array.size() != 0 && m_mass_array.size() != m_pos_array.size())
		{
		cerr << endl << "***Error! " << m_mass_array.size() << " masses != " << m_pos_array.size() << " positions" << endl << endl;
		throw runtime_error("Error extracting data from hoomd_xml file");
		}
	if (m_diameter_array.size() != 0 && m_diameter_array.size() != m_pos_array.size())
		{
		cerr << endl << "***Error! " << m_diameter_array.size() << " diameters != " << m_pos_array.size() << " positions" << endl << endl;
		throw runtime_error("Error extracting data from hoomd_xml file");
		}
	if (m_image_array.size() != 0 && m_image_array.size() != m_pos_array.size())
		{
		cerr << endl << "***Error! " << m_image_array.size() << " images != " << m_pos_array.size() << " positions" << endl << endl;
		throw runtime_error("Error extracting data from hoomd_xml file");
		}
	if (m_type_array.size() != 0 && m_type_array.size() != m_pos_array.size())
		{
		cerr << endl << "***Error! " << m_type_array.size() << " type values != " << m_pos_array.size() << " positions" << endl << endl;
		throw runtime_error("Error extracting data from hoomd_xml file");
		}
	if (m_charge_array.size() != 0 && m_charge_array.size() != m_pos_array.size())
		{
		cerr << endl << "***Error! " << m_charge_array.size() << " charge values != " << m_pos_array.size() << " positions" << endl << endl;
		throw runtime_error("Error extracting data from hoomd_xml file");
		}

	// notify the user of what we have accomplished
	cout << "--- hoomd_xml file read summary" << endl;
	cout << getNumParticles() << " positions at timestep " << m_timestep << endl;
	if (m_image_array.size() > 0)
		cout << m_image_array.size() << " images" << endl;	
	if (m_vel_array.size() > 0)
		cout << m_vel_array.size() << " velocities" << endl;
	if (m_mass_array.size() > 0)
		cout << m_mass_array.size() << " masses" << endl;
	if (m_diameter_array.size() > 0)
		cout << m_diameter_array.size() << " diameters" << endl;
	cout << getNumParticleTypes() <<  " particle types" << endl;
	if (m_bonds.size() > 0)
		cout << m_bonds.size() << " bonds" << endl;
	if (m_angles.size() > 0)
		cout << m_angles.size() << " angles" << endl;
	if (m_dihedrals.size() > 0)
		cout << m_dihedrals.size() << " dihedrals" << endl;
	if (m_impropers.size() > 0)
		cout << m_impropers.size() << " impropers" << endl;
	if (m_charge_array.size() > 0)
		cout << m_charge_array.size() << " charges" << endl;
	if (m_walls.size() > 0)
		cout << m_walls.size() << " walls" << endl;
	}

/*! \param node XMLNode passed from the top level parser in readFile
	This function extracts all of the information in the attributes of the \b box node
*/
void HOOMDInitializer::parseBoxNode(const XMLNode &node)
	{
	// first, verify that this is the box node
	string name = node.getName();
	transform(name.begin(), name.end(), name.begin(), ::tolower);	
	assert(name == string("box"));
	
	// temporary values for extracting attributes as Scalars
	Scalar Lx,Ly,Lz;
	istringstream temp;
	
	// use string streams to extract Lx, Ly, Lz
	// throw exceptions if these attributes are not set
	if (!node.isAttributeSet("lx"))
		{
		cerr << endl << "***Error! lx not set in <box> node" << endl << endl;
		throw runtime_error("Error extracting data from hoomd_xml file");
		}
	temp.str(node.getAttribute("lx"));
	temp >> Lx;
	temp.clear();

	if (!node.isAttributeSet("ly"))
		{
		cerr << endl << "***Error! ly not set in <box> node" << endl << endl;
		throw runtime_error("Error extracting data from hoomd_xml file");
		}
	temp.str(node.getAttribute("ly"));
	temp >> Ly;
	temp.clear();

	if (!node.isAttributeSet("lz")) 
		{
		cerr << endl << "***Error! lz not set in <box> node" << endl << endl;
		throw runtime_error("Error extracting data from hoomd_xml file");
		}
	temp.str(node.getAttribute("lz"));
	temp >> Lz;
	temp.clear();

	// initialize the BoxDim and set the flag telling that we read the <box> node
	m_box = BoxDim(Lx,Ly,Lz);
	m_box_read = true;
	}

/*! \param node XMLNode passed from the top level parser in readFile
	This function extracts all of the data in a \b position node and fills out m_pos_array. The number
	of particles in the array is determined dynamically.
*/
void HOOMDInitializer::parsePositionNode(const XMLNode &node)
	{
	// check that this is actually a position node
	string name = node.getName();
	transform(name.begin(), name.end(), name.begin(), ::tolower);	
	assert(name == string("position"));

	// units is currently unused, but will be someday: warn the user if they forget it
	//if (!node.isAttributeSet("units")) cout << "Warning! units not specified in <position> node" << endl;

	// extract the data from the node
	istringstream parser;
	if (node.getText())
		{
		parser.str(node.getText());
		while (parser.good())
			{
			Scalar x,y,z;
			parser >> x >> y >> z;
			m_pos_array.push_back(vec(x,y,z));
			}
		}
	else
		{
		cout << "***Warning! Found position node with no text. Possible typo." << endl;
		}
	}

/*! \param node XMLNode passed from the top level parser in readFile
	This function extracts all of the data in a \b image node and fills out m_pos_array. The number
	of particles in the array is determined dynamically.
*/
void HOOMDInitializer::parseImageNode(const XMLNode& node)
	{
	string name = node.getName();
	transform(name.begin(), name.end(), name.begin(), ::tolower);
	assert(name == string("image"));

	// extract the data from the node
	istringstream parser;
	if (node.getText())
		{
		parser.str(node.getText());
		while (parser.good())
			{
			int x,y,z;
			parser >> x >> y >> z;
			m_image_array.push_back(vec_int(x,y,z));
			}
		}
	else
		{
		cout << "***Warning! Found image node with no text. Possible typo." << endl;
		}
	}

/*! \param node XMLNode passed from the top level parser in readFile
	This function extracts all of the data in a \b velocity node and fills out m_vel_array. The number
	of particles in the array is determined dynamically.
*/
void HOOMDInitializer::parseVelocityNode(const XMLNode &node)
	{
	// check that this is actually a velocity node
	string name = node.getName();
	transform(name.begin(), name.end(), name.begin(), ::tolower);	
	assert(name == string("velocity"));

	// units is currently unused, but will be someday: warn the user if they forget it
	// if (!node.isAttributeSet("units")) cout << "Warning! units not specified in <velocity> node" << endl;

	// extract the data from the node
	istringstream parser;
	if (node.getText())
		{
		parser.str(node.getText());
		while (parser.good())
			{
			Scalar x,y,z;
			parser >> x >> y >> z;
			m_vel_array.push_back(vec(x,y,z));
			}
		}
	else
		{
		cout << "***Warning! Found velocity node with no text. Possible typo." << endl;
		}
	}
	
/*! \param node XMLNode passed from the top level parser in readFile
	This function extracts all of the data in a \b mass node and fills out m_mass_array. The number
	of particles in the array is determined dynamically.
*/
void HOOMDInitializer::parseMassNode(const XMLNode &node)
	{
	// check that this is actually a velocity node
	string name = node.getName();
	transform(name.begin(), name.end(), name.begin(), ::tolower);	
	assert(name == string("mass"));

	// units is currently unused, but will be someday: warn the user if they forget it
	// if (!node.isAttributeSet("units")) cout << "Warning! units not specified in <velocity> node" << endl;

	// extract the data from the node
	istringstream parser;
	if (node.getText())
		{
		parser.str(node.getText());
		while (parser.good())
			{
			Scalar mass;
			parser >> mass;
			m_mass_array.push_back(mass);
			}
		}
	else
		{
		cout << "***Warning! Found mass node with no text. Possible typo." << endl;
		}
	}
	
/*! \param node XMLNode passed from the top level parser in readFile
	This function extracts all of the data in a \b diameter node and fills out m_diameter_array. The number
	of particles in the array is determined dynamically.
*/
void HOOMDInitializer::parseDiameterNode(const XMLNode &node)
	{
	// check that this is actually a velocity node
	string name = node.getName();
	transform(name.begin(), name.end(), name.begin(), ::tolower);	
	assert(name == string("diameter"));

	// units is currently unused, but will be someday: warn the user if they forget it
	// if (!node.isAttributeSet("units")) cout << "Warning! units not specified in <velocity> node" << endl;

	// extract the data from the node
	istringstream parser;
	if (node.getText())
		{
		parser.str(node.getText());
		while (parser.good())
			{
			Scalar diameter;
			parser >> diameter;
			m_diameter_array.push_back(diameter);
			}
		}
	else
		{
		cout << "***Warning! Found diameter node with no text. Possible typo." << endl;
		}
	}

/*! \param node XMLNode passed from the top level parser in readFile
	This function extracts all of the data in a \b type node and fills out m_type_array. The number
	of particles in the array is determined dynamically.
*/
void HOOMDInitializer::parseTypeNode(const XMLNode &node)
	{
	// check that this is actually a type node
	string name = node.getName();
	transform(name.begin(), name.end(), name.begin(), ::tolower);	
	assert(name == string("type"));

	// extract the data from the node
	istringstream parser;
	if (node.getText())
		{
		parser.str(node.getText());
		while (parser.good())
			{
			// dynamically determine the particle types
			string type;
			parser >> type;
			
			m_type_array.push_back(getTypeId(type));
			}
		}
	else
		{
		cout << "***Warning! Found type node with no text. Possible typo." << endl;
		}
	}

/*! \param node XMLNode passed from the top level parser in readFile
	This function extracts all of the data in a \b bond node and fills out m_bonds. The number
	of bonds in the array is determined dynamically.
*/
void HOOMDInitializer::parseBondNode(const XMLNode &node)
	{
	// check that this is actually a bond node
	string name = node.getName();
	transform(name.begin(), name.end(), name.begin(), ::tolower);	
	assert(name == string("bond"));

	// extract the data from the node
	istringstream parser;
	if (node.getText())
		{
		parser.str(node.getText());
		while (parser.good())
			{
			string type_name;
			unsigned int a, b;
			parser >> type_name >> a >> b;
			m_bonds.push_back(Bond(getBondTypeId(type_name), a, b));
			}
		}
	else
		{
		cout << "***Warning! Found bond node with no text. Possible typo." << endl;
		}
	}

void HOOMDInitializer::parseAngleNode(const XMLNode &node)
	{
	// check that this is actually a angle node
	string name = node.getName();
	transform(name.begin(), name.end(), name.begin(), ::tolower);	
	assert(name == string("angle"));

	// extract the data from the node
	istringstream parser;
	if (node.getText())
		{
		parser.str(node.getText());
		while (parser.good())
			{
			string type_name;
			unsigned int a, b, c;
			parser >> type_name >> a >> b >> c;
			m_angles.push_back(Angle(getAngleTypeId(type_name), a, b, c));
			}
		}
	else
		{
		cout << "***Warning! Found angle node with no text. Possible typo." << endl;
		}
	}

/*! \param node XMLNode passed from the top level parser in readFile
	This function extracts all of the data in a \b dihedral node and fills out m_dihedrals. The number
	of dihedrals in the array is determined dynamically.
*/
void HOOMDInitializer::parseDihedralNode(const XMLNode &node)
	{
	// check that this is actually a dihedral node
	string name = node.getName();
	transform(name.begin(), name.end(), name.begin(), ::tolower);	
	assert(name == string("dihedral"));

	// extract the data from the node
	istringstream parser;
	if (node.getText())
		{
		parser.str(node.getText());
		while (parser.good())
			{
			string type_name;
			unsigned int a, b, c, d;
			parser >> type_name >> a >> b >> c >> d;
			m_dihedrals.push_back(Dihedral(getDihedralTypeId(type_name), a, b, c, d));
			}
		}
	else
		{
		cout << "***Warning! Found dihedral node with no text. Possible typo." << endl;
		}
	}

/*! \param node XMLNode passed from the top level parser in readFile
	This function extracts all of the data in a \b dihedral node and fills out m_dihedrals. The number
	of dihedrals in the array is determined dynamically.
*/
void HOOMDInitializer::parseImproperNode(const XMLNode &node)
	{
	// check that this is actually a improper node
	string name = node.getName();
	transform(name.begin(), name.end(), name.begin(), ::tolower);	
	assert(name == string("improper"));

	// extract the data from the node
	istringstream parser;
	if (node.getText())
		{
		parser.str(node.getText());
		while (parser.good())
			{
			string type_name;
			unsigned int a, b, c, d;
			parser >> type_name >> a >> b >> c >> d;
			m_impropers.push_back(Improper(getImproperTypeId(type_name), a, b, c, d));
			}
		}
	else
		{
		cout << "***Warning! Found improper node with no text. Possible typo." << endl;
		}
	}

/*! \param node XMLNode passed from the top level parser in readFile
	This function extracts all of the data in a \b charge node and fills out m_charge_array. The number
	of particles in the array is determined dynamically.
*/
void HOOMDInitializer::parseChargeNode(const XMLNode &node)
	{
	// check that this is actually a charge node
	string name = node.getName();
	transform(name.begin(), name.end(), name.begin(), ::tolower);	
	assert(name == string("charge"));

	// extract the data from the node
	istringstream parser;
	if (node.getText())
		{
		parser.str(node.getText());
		while (parser.good())
			{
			Scalar charge;
			parser >> charge;
			
			m_charge_array.push_back(charge);
			}
		}
	else
		{
		cout << "***Warning! Found charge node with no text. Possible typo." << endl;
		}
	}

/*! \param node XMLNode passed from the top level parser in readFile
	This function extracts all of the data in a \b wall node and fills out m_walls. The number
	of walls is dtermined dynamically.
*/
void HOOMDInitializer::parseWallNode(const XMLNode& node)
	{
	// check that this is actually a wall node
	string name = node.getName();
	transform(name.begin(), name.end(), name.begin(), ::tolower);	
	assert(name == string("wall"));

	for (int cur_node=0; cur_node < node.nChildNode(); cur_node++)
		{
		// check to make sure this is a node type we understand
		XMLNode child_node = node.getChildNode(cur_node);
		if (string(child_node.getName()) != string("coord"))
			{
			cout << "Notice: Ignoring <" << child_node.getName() << "> node in <wall> node";
			}
		else
			{
			// extract x,y,z, nx, ny, nz
			Scalar ox,oy,oz,nx,ny,nz;
			if (!child_node.isAttributeSet("ox"))
				{
				cerr << endl << "***Error! ox not set in <coord> node" << endl << endl;
				throw runtime_error("Error extracting data from hoomd_xml file");
				}
			ox = (Scalar)atof(child_node.getAttribute("ox"));

			if (!child_node.isAttributeSet("oy"))
				{
				cerr << endl << "***Error! oy not set in <coord> node" << endl << endl;
				throw runtime_error("Error extracting data from hoomd_xml file");
				}
			oy = (Scalar)atof(child_node.getAttribute("oy"));

			if (!child_node.isAttributeSet("oz"))
				{
				cerr << endl << "***Error! oz not set in <coord> node" << endl << endl;
				throw runtime_error("Error extracting data from hoomd_xml file");
				}
			oz = (Scalar)atof(child_node.getAttribute("oz"));

			if (!child_node.isAttributeSet("nx"))
				{
				cerr << endl << "***Error! nx not set in <coord> node" << endl << endl;
				throw runtime_error("Error extracting data from hoomd_xml file");
				}
			nx = (Scalar)atof(child_node.getAttribute("nx"));

			if (!child_node.isAttributeSet("ny"))
				{
				cerr << endl << "***Error! ny not set in <coord> node" << endl << endl;
				throw runtime_error("Error extracting data from hoomd_xml file");
				}
			ny = (Scalar)atof(child_node.getAttribute("ny"));

			if (!child_node.isAttributeSet("nz"))
				{
				cerr << endl << "***Error! nz not set in <coord> node" << endl << endl;
				throw runtime_error("Error extracting data from hoomd_xml file");
				}
			nz = (Scalar)atof(child_node.getAttribute("nz"));

			m_walls.push_back(Wall(ox,oy,oz,nx,ny,nz));
			}
		}
	}

/*! \param name Name to get type id of
	If \a name has already been added, this returns the type index of that name.
	If \a name has not yet been added, it is added to the list and the new id is returned.
*/
unsigned int HOOMDInitializer::getTypeId(const std::string& name)
	{
	// search for the type mapping
	for (unsigned int i = 0; i < m_type_mapping.size(); i++)
		{
		if (m_type_mapping[i] == name)
			return i;
		}
	// add a new one if it is not found
	m_type_mapping.push_back(name);
	return (unsigned int)m_type_mapping.size()-1;
	}

/*! \param name Name to get type id of
	If \a name has already been added, this returns the type index of that name.
	If \a name has not yet been added, it is added to the list and the new id is returned.
*/
unsigned int HOOMDInitializer::getBondTypeId(const std::string& name)
	{
	// search for the type mapping
	for (unsigned int i = 0; i < m_bond_type_mapping.size(); i++)
		{
		if (m_bond_type_mapping[i] == name)
			return i;
		}
	// add a new one if it is not found
	m_bond_type_mapping.push_back(name);
	return (unsigned int)m_bond_type_mapping.size()-1;
	}

/*! \param name Name to get type id of
	If \a name has already been added, this returns the type index of that name.
	If \a name has not yet been added, it is added to the list and the new id is returned.
*/
unsigned int HOOMDInitializer::getAngleTypeId(const std::string& name)
	{
	// search for the type mapping
	for (unsigned int i = 0; i < m_angle_type_mapping.size(); i++)
		{
		if (m_angle_type_mapping[i] == name)
			return i;
		}
	// add a new one if it is not found
	m_angle_type_mapping.push_back(name);
	return (unsigned int)m_angle_type_mapping.size()-1;
	}

/*! \param name Name to get type id of
	If \a name has already been added, this returns the type index of that name.
	If \a name has not yet been added, it is added to the list and the new id is returned.
*/
unsigned int HOOMDInitializer::getDihedralTypeId(const std::string& name)
	{
	// search for the type mapping
	for (unsigned int i = 0; i < m_dihedral_type_mapping.size(); i++)
		{
		if (m_dihedral_type_mapping[i] == name)
			return i;
		}
	// add a new one if it is not found
	m_dihedral_type_mapping.push_back(name);
	return (unsigned int)m_dihedral_type_mapping.size()-1;
	}


/*! \param name Name to get type id of
	If \a name has already been added, this returns the type index of that name.
	If \a name has not yet been added, it is added to the list and the new id is returned.
*/
unsigned int HOOMDInitializer::getImproperTypeId(const std::string& name)
	{
	// search for the type mapping
	for (unsigned int i = 0; i < m_improper_type_mapping.size(); i++)
		{
		if (m_improper_type_mapping[i] == name)
			return i;
		}
	// add a new one if it is not found
	m_improper_type_mapping.push_back(name);
	return (unsigned int)m_improper_type_mapping.size()-1;
	}

/*! \return Number of bond types determined from the XML file
*/
unsigned int HOOMDInitializer::getNumBondTypes() const
	{
	return (unsigned int)m_bond_type_mapping.size();
	}
		
/*! \return Number of angle types determined from the XML file
*/
unsigned int HOOMDInitializer::getNumAngleTypes() const
	{
	return (unsigned int)m_angle_type_mapping.size();
	}

/*! \return Number of dihedral types determined from the XML file
*/
unsigned int HOOMDInitializer::getNumDihedralTypes() const
	{
	return (unsigned int)m_dihedral_type_mapping.size();
	}

/*! \return Number of improper types determined from the XML file
*/
unsigned int HOOMDInitializer::getNumImproperTypes() const
	{
	return (unsigned int)m_improper_type_mapping.size();
	}

/*! \param bond_data Shared pointer to the BondData to be initialized
	Adds all bonds found in the XML file to the BondData
*/
void HOOMDInitializer::initBondData(boost::shared_ptr<BondData> bond_data) const
	{
	// loop through all the bonds and add a bond for each
	for (unsigned int i = 0; i < m_bonds.size(); i++)	
		bond_data->addBond(m_bonds[i]);
	
	bond_data->setBondTypeMapping(m_bond_type_mapping);
	}
	
/*! \param angle_data Shared pointer to the AngleData to be initialized
	Adds all angles found in the XML file to the AngleData
*/
void HOOMDInitializer::initAngleData(boost::shared_ptr<AngleData> angle_data) const
	{
	// loop through all the angles and add an angle for each
	for (unsigned int i = 0; i < m_angles.size(); i++)	
		angle_data->addAngle(m_angles[i]);
	
	angle_data->setAngleTypeMapping(m_angle_type_mapping);
	}

/*! \param dihedral_data Shared pointer to the DihedralData to be initialized
	Adds all dihedrals found in the XML file to the DihedralData
*/
void HOOMDInitializer::initDihedralData(boost::shared_ptr<DihedralData> dihedral_data) const
	{
	// loop through all the dihedrals and add an dihedral for each
	for (unsigned int i = 0; i < m_dihedrals.size(); i++)	
		dihedral_data->addDihedral(m_dihedrals[i]);
	
	dihedral_data->setDihedralTypeMapping(m_dihedral_type_mapping);
	}

/*! \param improper_data Shared pointer to the ImproperData to be initialized
	Adds all impropers found in the XML file to the ImproperData
*/
void HOOMDInitializer::initImproperData(boost::shared_ptr<ImproperData> improper_data) const
	{
	// loop through all the impropers and add an improper for each
	for (unsigned int i = 0; i < m_impropers.size(); i++)	
		improper_data->addImproper(m_impropers[i]);
	
	improper_data->setImproperTypeMapping(m_improper_type_mapping);
	}

/*! \returns A mapping of type ids to type names deteremined from the XML input file
*/
std::vector<std::string> HOOMDInitializer::getTypeMapping() const
	{
	return m_type_mapping;
	}

void export_HOOMDInitializer()
	{
	class_< HOOMDInitializer, bases<ParticleDataInitializer> >("HOOMDInitializer", init<const string&>())
		// virtual methods from ParticleDataInitializer are inherited
		.def("getTimeStep", &HOOMDInitializer::getTimeStep)
		.def("setTimeStep", &HOOMDInitializer::setTimeStep)
		;
	}

#ifdef WIN32
#pragma warning( pop )
#endif
