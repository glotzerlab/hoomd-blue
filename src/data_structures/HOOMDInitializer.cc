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

#include "HOOMDInitializer.h"

#include <fstream>
#include <stdexcept>
#include <sstream>

using namespace std;

#ifdef USE_PYTHON
#include <boost/python.hpp>
using namespace boost::python;
#endif

using namespace boost;

/*! \param fname File name with the data to load
*/
HOOMDInitializer::HOOMDInitializer(const std::string &fname)
 	{
	// initialize member variables
	m_nparticle_types = 0;
	m_timestep = 0;
	m_box_read = false;

	// initialize the parser map
	m_parser_map["box"] = bind(&HOOMDInitializer::parseBoxNode, this, _1);
	m_parser_map["position"] = bind(&HOOMDInitializer::parsePositionNode, this, _1);
	m_parser_map["velocity"] = bind(&HOOMDInitializer::parseVelocityNode, this, _1);
	m_parser_map["type"] = bind(&HOOMDInitializer::parseTypeNode, this, _1);
	m_parser_map["bond"] = bind(&HOOMDInitializer::parseBondNode, this, _1);

	// read in the file
	readFile(fname);
	}
		
unsigned int HOOMDInitializer::getNumParticles() const
	{
	assert(m_pos_array.size() > 0);
	return m_pos_array.size();
	}
		
unsigned int HOOMDInitializer::getNumParticleTypes() const
	{
	assert(m_nparticle_types > 0);
	return m_nparticle_types;
	}

BoxDim HOOMDInitializer::getBox() const
	{		
	return m_box;
	}

unsigned int HOOMDInitializer::getTimeStep() const
	{
	return m_timestep;
	}


/* ! \param pdata The particle data 

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
	
	if (m_type_array.size() != 0)
		{
		assert(m_type_array.size() == m_pos_array.size());
		
		for (unsigned int i = 0; i < m_pos_array.size(); i++)
			pdata.type[i] = m_type_array[i];
		}
	}

/*!	\param fname File name of the hoomd_xml file to read in
	\post Internal data arrays and members are filled out from which futre calls
	like initArrays will use to intialize the ParticleData
*/
void HOOMDInitializer::readFile(const string &fname)
	{
	// Create a Root Node and a child node
	XMLNode root_node;

	// Open the file and read the root element "hoomd_xml"
	cout<< "Reading " << fname << "..." << endl;
	XMLResults results;
	root_node = XMLNode::parseFile(fname.c_str(),"HOOMD_xml", &results);

    // handle errors
    if (results.error != eXMLErrorNone)
		{
        // create message
        if (results.error==eXMLErrorFirstTagNotFound) 
			throw runtime_error(string("Root node of ") + fname + " is not <hoomd_xml>");
		
		ostringstream error_message;
		error_message << XMLNode::getError(results.error) << " in file " << fname << " at line " << results.nLine << " col " << results.nColumn;
		throw runtime_error(error_message.str());
		}

	// the file was parsed successfully by the XML reader. Extract the information now
	// start by checking the number of configurations in the file
	int num_configurations = root_node.nChildNode("configuration");
	if (num_configurations == 0)
		throw runtime_error("No <configuration> specified in the XML file");
	if (num_configurations > 1)
		throw runtime_error("Sorry, the input XML file must have only one configuration");
	
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
		std::map< std::string, boost::function< void (const XMLNode&) > >::iterator parser;
		parser = m_parser_map.find(name);
		if (parser != m_parser_map.end())
			parser->second(node);
		else
			cout << "Parser for node <" << name << "> not defined, ignoring" << endl;
		}
	
	// check for required items in the file
	if (!m_box_read)
		{
		cout << "A <box> node is required to define the dimensions of the simulation box" << endl;
		throw runtime_error("Error extracting data from hoomd_xml file");
		}
	if (m_pos_array.size() == 0)
		{
		cout << "No particles defined in <position> node" << endl;
		throw runtime_error("Error extracting data from hoomd_xml file");
		}

	// check for potential user errors
	if (m_vel_array.size() != 0 && m_vel_array.size() != m_pos_array.size())
		{
		cout << "Error " << m_vel_array.size() << " velocities != " << m_pos_array.size() << " positions" << endl;
		throw runtime_error("Error extracting data from hoomd_xml file");
		}
	if (m_type_array.size() != 0 && m_type_array.size() != m_pos_array.size())
		{
		cout << "Error " << m_type_array.size() << " type values != " << m_pos_array.size() << " positions" << endl;
		throw runtime_error("Error extracting data from hoomd_xml file");
		}

	// notify the user of what we have accomplished
	cout << "Read " << getNumParticles() << " positions";
	if (m_vel_array.size() > 0)
		cout << ", " << m_vel_array.size() << " velocities";
	cout << ", with " << getNumParticleTypes() <<  " particle types";
	if (m_bonds.size() > 0)
		cout << " and " << m_bonds.size() << " bonds";
	cout << "." << endl;
	}

/*! \param node XMLNode passed from the top level parser in readFile
	This function extracts all of the information in the attributes of the <box> node
*/
void HOOMDInitializer::parseBoxNode(const XMLNode &node)
	{
	// first, verify that this is the box node
	assert(string(node.getName()) == string("box"));
	
	// temporary values for extracting attributes as Scalars
	Scalar Lx,Ly,Lz;
	istringstream temp;
	
	// use string streams to extract Lx, Ly, Lz
	// throw exceptions if these attributes are not set
	if (!node.isAttributeSet("Lx"))
		{
		cout << "Lx not set in <box> node" << endl;
		throw runtime_error("Error extracting data from hoomd_xml file");
		}
	temp.str(node.getAttribute("Lx"));
	temp >> Lx;
	temp.clear();

	if (!node.isAttributeSet("Ly"))
		{
		cout << "Ly not set in <box> node" << endl;
		throw runtime_error("Error extracting data from hoomd_xml file");
		}
	temp.str(node.getAttribute("Ly"));
	temp >> Ly;
	temp.clear();

	if (!node.isAttributeSet("Lz")) 
		{
		cout << "Lz not set in <box> node" << endl;
		throw runtime_error("Error extracting data from hoomd_xml file");
		}
	temp.str(node.getAttribute("Lz"));
	temp >> Lz;
	temp.clear();

	// initialize the BoxDim and set the flag telling that we read the <box> node
	m_box = BoxDim(Lx,Ly,Lz);
	m_box_read = true;
	}

/* \param node XMLNode passed from the top level parser in readFile
	This function extracts all of the data in a <position> node and fills out m_pos_array. The number
	of particles in the array is determined dynamically.
*/
void HOOMDInitializer::parsePositionNode(const XMLNode &node)
	{
	// check that this is actually a position node
	assert(string(node.getName()) == string("position"));

	// units is currently unused, but will be someday: warn the user if they forget it
	if (!node.isAttributeSet("units")) cout << "Warning: units not specified in <position> node" << endl;

	// extract the data from the node
	istringstream parser;
	parser.str(node.getText());
	while (parser.good())
		{
		Scalar x,y,z;
		parser >> x >> y >> z;
		m_pos_array.push_back(vec(x,y,z));
		}
	}

/* \param node XMLNode passed from the top level parser in readFile
	This function extracts all of the data in a <velocity> node and fills out m_vel_array. The number
	of particles in the array is determined dynamically.
*/
void HOOMDInitializer::parseVelocityNode(const XMLNode &node)
	{
	// check that this is actually a position node
	assert(string(node.getName()) == string("velocity"));

	// units is currently unused, but will be someday: warn the user if they forget it
	if (!node.isAttributeSet("units")) cout << "Warning: units not specified in <velocity> node" << endl;

	// extract the data from the node
	istringstream parser;
	parser.str(node.getText());
	while (parser.good())
		{
		Scalar x,y,z;
		parser >> x >> y >> z;
		m_vel_array.push_back(vec(x,y,z));
		}
	}

/* \param node XMLNode passed from the top level parser in readFile
	This function extracts all of the data in a <type> node and fills out m_type_array. The number
	of particles in the array is determined dynamically.
*/
void HOOMDInitializer::parseTypeNode(const XMLNode &node)
	{
	// check that this is actually a position node
	assert(string(node.getName()) == string("type"));

	// extract the data from the node
	istringstream parser;
	parser.str(node.getText());
	while (parser.good())
		{
		unsigned int type;
		parser >> type;
		
		m_type_array.push_back(type);

		// dynamically determine the number of particle types
		if (type+1 > m_nparticle_types)
			m_nparticle_types = type+1;
		}
	}

/* \param node XMLNode passed from the top level parser in readFile
	This function extracts all of the data in a <bond> node and fills out m_bonds. The number
	of bonds in the array is determined dynamically.
*/
void HOOMDInitializer::parseBondNode(const XMLNode &node)
	{
	// check that this is actually a position node
	assert(string(node.getName()) == string("bond"));

	// extract the data from the node
	istringstream parser;
	parser.str(node.getText());
	while (parser.good())
		{
		unsigned int a, b;
		parser >> a >> b;
		m_bonds.push_back(bond(a, b));
		}
	}

/* \param node XMLNode passed from the top level parser in readFile
	This function extracts all of the data in a <charge> node and fills out m_charge_array. The number
	of particles in the array is determined dynamically.
*/
void HOOMDInitializer::parseChargeNode(const XMLNode &node)
	{
	// check that this is actually a position node
	assert(string(node.getName()) == string("charge"));

	// extract the data from the node
	istringstream parser;
	parser.str(node.getText());
	while (parser.good())
		{
		Scalar charge;
		parser >> charge;
		
		m_charge_array.push_back(charge);
		}
	}

		
void HOOMDInitializer::setupNeighborListExclusions(boost::shared_ptr<NeighborList> nlist)
	{
	// loop through all the bonds and add an exclusion for each
	for (unsigned int i = 0; i < m_bonds.size(); i++)
		nlist->addExclusion(m_bonds[i].tag_a, m_bonds[i].tag_b);
	}
	
void HOOMDInitializer::setupBonds(boost::shared_ptr<BondForceCompute> fc_bond)
	{
	// loop through all the bonds and add a bond for each
	for (unsigned int i = 0; i < m_bonds.size(); i++)	
		fc_bond->addBond(m_bonds[i].tag_a, m_bonds[i].tag_b);
	}

#ifdef USE_PYTHON
void export_HOOMDInitializer()
	{
	class_< HOOMDInitializer, bases<ParticleDataInitializer> >("HOOMDInitializer", init<const string&>())
		// virtual methods from ParticleDataInitializer are inherited
		.def("setupNeighborListExclusions", &HOOMDInitializer::setupNeighborListExclusions)
		.def("setupBonds", &HOOMDInitializer::setupBonds) 
		;
	}
#endif
