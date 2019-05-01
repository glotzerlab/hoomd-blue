// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file HOOMDInitializer.cc
    \brief Defines the HOOMDInitializer class
*/
#include "HOOMDInitializer.h"
#include "hoomd/SnapshotSystemData.h"
#include "hoomd/ExecutionConfiguration.h"

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <algorithm>

using namespace std;
namespace py = pybind11;

/*! \param fname File name with the data to load
    The file will be read and parsed fully during the constructor call.
*/
HOOMDInitializer::HOOMDInitializer(std::shared_ptr<const ExecutionConfiguration> exec_conf,
    const std::string &fname,
    bool wrap_coordinates)
    : m_timestep(0),
      m_exec_conf(exec_conf),
      m_wrap(wrap_coordinates)
    {
    // we only execute on rank 0
    if (m_exec_conf->getRank()) return;

    // initialize member variables
    m_box_read = false;
    m_num_dimensions = 3;

    // initialize the parser map
    m_parser_map["box"] = std::bind(&HOOMDInitializer::parseBoxNode, this, std::placeholders::_1);
    m_parser_map["position"] = std::bind(&HOOMDInitializer::parsePositionNode, this, std::placeholders::_1);
    m_parser_map["image"] = std::bind(&HOOMDInitializer::parseImageNode, this, std::placeholders::_1);
    m_parser_map["velocity"] = std::bind(&HOOMDInitializer::parseVelocityNode, this, std::placeholders::_1);
    m_parser_map["mass"] = std::bind(&HOOMDInitializer::parseMassNode, this, std::placeholders::_1);
    m_parser_map["diameter"] = std::bind(&HOOMDInitializer::parseDiameterNode, this, std::placeholders::_1);
    m_parser_map["type"] = std::bind(&HOOMDInitializer::parseTypeNode, this, std::placeholders::_1);
    m_parser_map["body"] = std::bind(&HOOMDInitializer::parseBodyNode, this, std::placeholders::_1);
    m_parser_map["bond"] = std::bind(&HOOMDInitializer::parseBondNode, this, std::placeholders::_1);
    m_parser_map["angle"] = std::bind(&HOOMDInitializer::parseAngleNode, this, std::placeholders::_1);
    m_parser_map["dihedral"] = std::bind(&HOOMDInitializer::parseDihedralNode, this, std::placeholders::_1);
    m_parser_map["improper"] = std::bind(&HOOMDInitializer::parseImproperNode, this, std::placeholders::_1);
    m_parser_map["pair"] = std::bind(&HOOMDInitializer::parsePairNode, this, std::placeholders::_1);
    m_parser_map["constraint"] = std::bind(&HOOMDInitializer::parseConstraintsNode, this, std::placeholders::_1);
    m_parser_map["charge"] = std::bind(&HOOMDInitializer::parseChargeNode, this, std::placeholders::_1);
    m_parser_map["orientation"] = std::bind(&HOOMDInitializer::parseOrientationNode, this, std::placeholders::_1);
    m_parser_map["moment_inertia"] = std::bind(&HOOMDInitializer::parseMomentInertiaNode, this, std::placeholders::_1);
    m_parser_map["angmom"] = std::bind(&HOOMDInitializer::parseAngularMomentumNode, this, std::placeholders::_1);

    // read in the file
    readFile(fname);
    }

/* XXX: shouldn't the following methods be put into
 * the header so that they get inlined? */

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

/*! initializes a snapshot with the internally stored copy of the system data */
std::shared_ptr< SnapshotSystemData<Scalar> > HOOMDInitializer::getSnapshot() const
    {
    std::shared_ptr< SnapshotSystemData<Scalar> > snapshot(new SnapshotSystemData<Scalar>());

    // we only execute on rank 0
    if (m_exec_conf->getRank()) return snapshot;

    // initialize dimensions
    snapshot->dimensions = m_num_dimensions;

    // initialize box dimensions
    snapshot->global_box = m_box;

    /*
     * Initialize particle data
     */
    assert(m_pos_array.size() > 0);

    SnapshotParticleData<Scalar>& pdata = snapshot->particle_data;

    // allocate memory in snapshot
    pdata.resize(m_pos_array.size());

    // loop through all the particles and set them up
    for (unsigned int i = 0; i < m_pos_array.size(); i++)
        {
        pdata.pos[i] = vec3<Scalar>(m_pos_array[i].x, m_pos_array[i].y, m_pos_array[i].z);
        }

    if (m_image_array.size() != 0)
        {
        assert(m_image_array.size() == m_pos_array.size());

        for (unsigned int i = 0; i < m_pos_array.size(); i++)
            pdata.image[i] = make_int3(m_image_array[i].x, m_image_array[i].y, m_image_array[i].z);
        }

    if (m_wrap)
        {
        // wrap coordinates into box
        for (unsigned int i = 0; i < m_pos_array.size(); i++)
            {
            m_box.wrap(pdata.pos[i],pdata.image[i]);
            }
        }

    if (m_vel_array.size() != 0)
        {
        assert(m_vel_array.size() == m_pos_array.size());

        for (unsigned int i = 0; i < m_pos_array.size(); i++)
            pdata.vel[i] = vec3<Scalar>(m_vel_array[i].x, m_vel_array[i].y, m_vel_array[i].z);
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

    if (m_body_array.size() != 0)
        {
        assert(m_body_array.size() == m_pos_array.size());

        for (unsigned int i = 0; i < m_pos_array.size(); i++)
            pdata.body[i] = m_body_array[i];
        }

    if (m_type_mapping.size()) pdata.type_mapping = m_type_mapping;

    // Initialize moments of inertia
    if (m_moment_inertia.size())
        {
        for (unsigned int i = 0; i < m_pos_array.size(); i++)
            {
            pdata.inertia[i] = vec3<Scalar>(m_moment_inertia[i]);
            }
        }

    // Initialize orientations
    if (m_orientation.size())
        {
        for (unsigned int i = 0; i < m_pos_array.size(); i++)
            {
            pdata.orientation[i] = quat<Scalar>(m_orientation[i]);
            }
        }

    // Initialize angular momenta
    if (m_angmom.size())
        {
        for (unsigned int i = 0; i < m_pos_array.size(); i++)
            {
            pdata.angmom[i] = quat<Scalar>(m_angmom[i]);
            }
        }

    /*
     * Initialize bond data
     */
    BondData::Snapshot& bdata = snapshot->bond_data;

    // allocate memory in snapshot
    bdata.resize(m_bonds.size());

    // loop through all the bonds and add a bond for each
    for (unsigned int i = 0; i < m_bonds.size(); i++)
        {
        bdata.groups[i] = m_bonds[i];
        bdata.type_id[i] = m_bond_types[i];
        }

    bdata.type_mapping = m_bond_type_mapping;

    /*
     * Initialize angle data
     */
    AngleData::Snapshot& adata = snapshot->angle_data;

    // allocate memory in snapshot
    adata.resize(m_angles.size());

    // loop through all the angles and add an angle for each
    for (unsigned int i = 0; i < m_angles.size(); i++)
        {
        adata.groups[i] = m_angles[i];
        adata.type_id[i] = m_angle_types[i];
        }

    adata.type_mapping = m_angle_type_mapping;

    /*
     * Initialize dihedral data
     */
    DihedralData::Snapshot& ddata = snapshot->dihedral_data;

    // allocate memory
    ddata.resize(m_dihedrals.size());

    // loop through all the dihedrals and add an dihedral for each
    for (unsigned int i = 0; i < m_dihedrals.size(); i++)
        {
        ddata.groups[i] = m_dihedrals[i];
        ddata.type_id[i] = m_dihedral_types[i];
        }

    ddata.type_mapping = m_dihedral_type_mapping;

    /*
     * Initialize improper data
     */
    ImproperData::Snapshot& idata = snapshot->improper_data;

    // allocate memory
    idata.resize(m_impropers.size());

    // loop through all the dihedrals and add an dihedral for each
    for (unsigned int i = 0; i < m_impropers.size(); i++)
        {
        idata.groups[i] = m_impropers[i];
        idata.type_id[i] = m_improper_types[i];
        }

    idata.type_mapping = m_improper_type_mapping;

    /*
     * Initialize constraint data
     */
    ConstraintData::Snapshot& cdata = snapshot->constraint_data;

    // allocate memory
    cdata.resize(m_constraints.size());

    // loop through all the constraints and add a constraint for each
    for (unsigned int i = 0; i < m_constraints.size(); i++)
        {
        cdata.groups[i] = m_constraints[i];
        cdata.val[i] = m_constraint_distances[i];
        }

    // initialize with empty vector
    cdata.type_mapping = std::vector<std::string>();

    /*
     * Initialize pair data
     */
    PairData::Snapshot& pair_data = snapshot->pair_data;

    // allocate memory
    pair_data.resize(m_pairs.size());

    // loop through all the pairs and add a special pair for each
    for (unsigned int i = 0; i < m_pairs.size(); i++)
        {
        pair_data.groups[i] = m_pairs[i];
        pair_data.type_id[i] = m_pair_types[i];
        }

    pair_data.type_mapping = m_pair_type_mapping;

    return snapshot;
    }

/*! \param fname File name of the hoomd_xml file to read in
    \post Internal data arrays and members are filled out from which future calls
    like getSnapshot() will use to initialize the ParticleData

    This function implements the main parser loop. It reads in XML nodes from the
    file one by one and passes them of to parsers registered in \c m_parser_map.
*/
void HOOMDInitializer::readFile(const string &fname)
    {
    // Create a Root Node and a child node
    XMLNode root_node;

    // Open the file and read the root element "hoomd_xml"
    m_exec_conf->msg->notice(2) << "Reading " << fname << "..." << endl;
    XMLResults results;
    root_node = XMLNode::parseFile(fname.c_str(),"hoomd_xml", &results);

    // handle errors
    if (results.error != eXMLErrorNone)
        {
        // create message
        if (results.error==eXMLErrorFirstTagNotFound)
            {
            m_exec_conf->msg->error() << endl << "Root node of " << fname << " is not <hoomd_xml>" << endl << endl;
            throw runtime_error("Error reading xml file");
            }

        ostringstream error_message;
        error_message << XMLNode::getError(results.error) << " in file "
        << fname << " at line " << results.nLine << " col "
        << results.nColumn;
        m_exec_conf->msg->error() << endl << error_message.str() << endl << endl;
        throw runtime_error("Error reading xml file");
        }

    if (root_node.isAttributeSet("version"))
        {
        m_xml_version = root_node.getAttribute("version");
        }
    else
        {
        m_exec_conf->msg->notice(2) << "No version specified in hoomd_xml root node: assuming 1.0" << endl;
        m_xml_version = string("1.0");
        }

    // right now, the version tag doesn't do anything: just warn if it is not a valid version
    vector<string> valid_versions;
    valid_versions.push_back("1.0");
    valid_versions.push_back("1.1");
    valid_versions.push_back("1.2");
    valid_versions.push_back("1.3");
    valid_versions.push_back("1.4");
    valid_versions.push_back("1.5");
    valid_versions.push_back("1.6");
    valid_versions.push_back("1.7");
    valid_versions.push_back("1.8");
    bool valid = false;
    vector<string>::iterator i;
    for (i = valid_versions.begin(); i != valid_versions.end(); ++i)
        {
        if (m_xml_version == *i)
            {
            valid = true;
            break;
            }
        }
    if (!valid)
        m_exec_conf->msg->warning() << endl
             << "hoomd_xml file with version not in the range 1.0-1.7  specified,"
             << " I don't know how to read this. Continuing anyways." << endl << endl;

    // the file was parsed successfully by the XML reader. Extract the information now
    // start by checking the number of configurations in the file
    int num_configurations = root_node.nChildNode("configuration");
    if (num_configurations == 0)
        {
        m_exec_conf->msg->error() << endl << "No <configuration> specified in the XML file" << endl << endl;
        throw runtime_error("Error reading xml file");
        }
    if (num_configurations > 1)
        {
        m_exec_conf->msg->error() << endl << "Sorry, the input XML file must have only one configuration" << endl << endl;
        throw runtime_error("Error reading xml file");
        }

    // extract the only configuration node
    XMLNode configuration_node = root_node.getChildNode("configuration");
    // extract the time step
    if (configuration_node.isAttributeSet("time_step"))
        {
        m_timestep = atoi(configuration_node.getAttribute("time_step"));
        }

    // extract the number of dimensions, or default to 3
    if (configuration_node.isAttributeSet("dimensions"))
        {
        m_num_dimensions = atoi(configuration_node.getAttribute("dimensions"));
        }
    else
        m_num_dimensions = 3;

    // loop through all child nodes of the configuration
    for (int cur_node=0; cur_node < configuration_node.nChildNode(); cur_node++)
        {
        // extract the name and call the appropriate node parser, if it exists
        XMLNode node = configuration_node.getChildNode(cur_node);
        string name = node.getName();
        transform(name.begin(), name.end(), name.begin(), ::tolower);

        std::map< std::string, std::function< void (const XMLNode&) > >::iterator parser;
        parser = m_parser_map.find(name);
        if (parser != m_parser_map.end())
            parser->second(node);
        else
            m_exec_conf->msg->notice(2) << "Parser for node <" << name << "> not defined, ignoring" << endl;
        }

    // check for required items in the file
    if (!m_box_read)
        {
        m_exec_conf->msg->error() << endl
             << "A <box> node is required to define the dimensions of the simulation box"
             << endl << endl;
        throw runtime_error("Error extracting data from hoomd_xml file");
        }
    if (m_pos_array.size() == 0)
        {
        m_exec_conf->msg->error() << endl << "No particles defined in <position> node" << endl << endl;
        throw runtime_error("Error extracting data from hoomd_xml file");
        }
    if (m_type_array.size() == 0)
        {
        m_exec_conf->msg->error() << endl << "No particles defined in <type> node" << endl << endl;
        throw runtime_error("Error extracting data from hoomd_xml file");
        }

    // check for potential user errors
    if (m_vel_array.size() != 0 && m_vel_array.size() != m_pos_array.size())
        {
        m_exec_conf->msg->error() << endl << m_vel_array.size() << " velocities != " << m_pos_array.size()
             << " positions" << endl << endl;
        throw runtime_error("Error extracting data from hoomd_xml file");
        }
    if (m_mass_array.size() != 0 && m_mass_array.size() != m_pos_array.size())
        {
        m_exec_conf->msg->error() << endl << m_mass_array.size() << " masses != " << m_pos_array.size()
             << " positions" << endl << endl;
        throw runtime_error("Error extracting data from hoomd_xml file");
        }
    if (m_diameter_array.size() != 0 && m_diameter_array.size() != m_pos_array.size())
        {
        m_exec_conf->msg->error() << endl << m_diameter_array.size() << " diameters != " << m_pos_array.size()
             << " positions" << endl << endl;
        throw runtime_error("Error extracting data from hoomd_xml file");
        }
    if (m_image_array.size() != 0 && m_image_array.size() != m_pos_array.size())
        {
        m_exec_conf->msg->error() << endl << m_image_array.size() << " images != " << m_pos_array.size()
             << " positions" << endl << endl;
        throw runtime_error("Error extracting data from hoomd_xml file");
        }
    if (m_type_array.size() != m_pos_array.size())
        {
        m_exec_conf->msg->error() << endl << m_type_array.size() << " type values != " << m_pos_array.size()
             << " positions" << endl << endl;
        throw runtime_error("Error extracting data from hoomd_xml file");
        }
    if (m_charge_array.size() != 0 && m_charge_array.size() != m_pos_array.size())
        {
        m_exec_conf->msg->error() << endl << m_charge_array.size() << " charge values != " << m_pos_array.size()
             << " positions" << endl << endl;
        throw runtime_error("Error extracting data from hoomd_xml file");
        }
    if (m_body_array.size() != 0 && m_body_array.size() != m_pos_array.size())
        {
        m_exec_conf->msg->error() << endl << m_body_array.size() << " body values != " << m_pos_array.size()
             << " positions" << endl << endl;
        throw runtime_error("Error extracting data from hoomd_xml file");
        }
    if (m_orientation.size() != 0 && m_orientation.size() != m_pos_array.size())
        {
        m_exec_conf->msg->error() << endl << m_orientation.size() << " orientation values != " << m_pos_array.size()
             << " positions" << endl << endl;
        throw runtime_error("Error extracting data from hoomd_xml file");
        }
    if (m_moment_inertia.size() != 0 && m_moment_inertia.size() != m_pos_array.size())
        {
        m_exec_conf->msg->error() << endl << m_moment_inertia.size() << " moment_inertia values != " << m_pos_array.size()
             << " positions" << endl << endl;
        throw runtime_error("Error extracting data from hoomd_xml file");
        }
    if (m_angmom.size() != 0 && m_angmom.size() != m_pos_array.size())
        {
        m_exec_conf->msg->error() << endl << m_angmom.size() << " angmom values != " << m_pos_array.size()
             << " positions" << endl << endl;
        throw runtime_error("Error extracting data from hoomd_xml file");
        }


    // notify the user of what we have accomplished
    m_exec_conf->msg->notice(2) << "--- hoomd_xml file read summary" << endl;
    m_exec_conf->msg->notice(2) << m_pos_array.size() << " positions at timestep " << m_timestep << endl;
    if (m_image_array.size() > 0)
        m_exec_conf->msg->notice(2) << m_image_array.size() << " images" << endl;
    if (m_vel_array.size() > 0)
        m_exec_conf->msg->notice(2) << m_vel_array.size() << " velocities" << endl;
    if (m_mass_array.size() > 0)
        m_exec_conf->msg->notice(2) << m_mass_array.size() << " masses" << endl;
    if (m_diameter_array.size() > 0)
        m_exec_conf->msg->notice(2) << m_diameter_array.size() << " diameters" << endl;
    m_exec_conf->msg->notice(2) << m_type_mapping.size() <<  " particle types" << endl;
    if (m_body_array.size() > 0)
        m_exec_conf->msg->notice(2) << m_body_array.size() << " particle body values" << endl;
    if (m_bonds.size() > 0)
        m_exec_conf->msg->notice(2) << m_bonds.size() << " bonds" << endl;
    if (m_angles.size() > 0)
        m_exec_conf->msg->notice(2) << m_angles.size() << " angles" << endl;
    if (m_dihedrals.size() > 0)
        m_exec_conf->msg->notice(2) << m_dihedrals.size() << " dihedrals" << endl;
    if (m_impropers.size() > 0)
        m_exec_conf->msg->notice(2) << m_impropers.size() << " impropers" << endl;
    if (m_constraints.size() > 0)
        m_exec_conf->msg->notice(2) << m_constraints.size() << " constraints" << endl;
    if (m_pairs.size() > 0)
        m_exec_conf->msg->notice(2) << m_pairs.size() << " special pairs" << endl;
    if (m_charge_array.size() > 0)
        m_exec_conf->msg->notice(2) << m_charge_array.size() << " charges" << endl;
    if (m_orientation.size() > 0)
        m_exec_conf->msg->notice(2) << m_orientation.size() << " orientations" << endl;
    if (m_moment_inertia.size() > 0)
        m_exec_conf->msg->notice(2) << m_moment_inertia.size() << " moments of inertia" << endl;
    if (m_angmom.size() > 0)
        m_exec_conf->msg->notice(2) << m_angmom.size() << " angular moments" << endl;
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
    Scalar xy(0.0), xz(0.0), yz(0.0);
    istringstream temp;

    // use string streams to extract Lx, Ly, Lz
    // throw exceptions if these attributes are not set
    if (!node.isAttributeSet("lx"))
        {
        m_exec_conf->msg->error() << endl << "lx not set in <box> node" << endl << endl;
        throw runtime_error("Error extracting data from hoomd_xml file");
        }
    temp.str(node.getAttribute("lx"));
    temp >> Lx;
    temp.clear();

    if (!node.isAttributeSet("ly"))
        {
        m_exec_conf->msg->error() << endl << "ly not set in <box> node" << endl << endl;
        throw runtime_error("Error extracting data from hoomd_xml file");
        }
    temp.str(node.getAttribute("ly"));
    temp >> Ly;
    temp.clear();

    if (!node.isAttributeSet("lz"))
        {
        m_exec_conf->msg->error() << endl << "lz not set in <box> node" << endl << endl;
        throw runtime_error("Error extracting data from hoomd_xml file");
        }
    temp.str(node.getAttribute("lz"));
    temp >> Lz;
    temp.clear();

    // If no tilt factors are provided, they default to zero
    if (node.isAttributeSet("xy"))
        {
        temp.str(node.getAttribute("xy"));
        temp >> xy;
        temp.clear();
        }

    if (node.isAttributeSet("xz"))
        {
        temp.str(node.getAttribute("xz"));
        temp >> xz;
        temp.clear();
        }

    if (node.isAttributeSet("yz"))
        {
        temp.str(node.getAttribute("yz"));
        temp >> yz;
        temp.clear();
        }

    // initialize the BoxDim and set the flag telling that we read the <box> node
    m_box = BoxDim(Lx,Ly,Lz);
    m_box.setTiltFactors(xy,xz,yz);
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

    // extract the data from the node
    string all_text;
    for (int i = 0; i < node.nText(); i++)
        all_text += string(node.getText(i)) + string("\n");

    istringstream parser;
    parser.str(all_text);
    while (parser.good())
        {
        Scalar x,y,z;
        parser >> x >> y >> z;
        if (parser.good())
            m_pos_array.push_back(vec(x,y,z));
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
    string all_text;
    for (int i = 0; i < node.nText(); i++)
        all_text += string(node.getText(i)) + string("\n");

    istringstream parser;
    parser.str(all_text);
    while (parser.good())
        {
        int x,y,z;
        parser >> x >> y >> z;
        if (parser.good())
            m_image_array.push_back(vec_int(x,y,z));
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

    // extract the data from the node
    string all_text;
    for (int i = 0; i < node.nText(); i++)
        all_text += string(node.getText(i)) + string("\n");

    istringstream parser;
    parser.str(all_text);
    while (parser.good())
        {
        Scalar x,y,z;
        parser >> x >> y >> z;
        if (parser.good())
            m_vel_array.push_back(vec(x,y,z));
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

    // extract the data from the node
    string all_text;
    for (int i = 0; i < node.nText(); i++)
        all_text += string(node.getText(i)) + string("\n");

    istringstream parser;
    parser.str(all_text);
    while (parser.good())
        {
        Scalar mass;
        parser >> mass;
        if (parser.good())
            m_mass_array.push_back(mass);
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

    // extract the data from the node
    string all_text;
    for (int i = 0; i < node.nText(); i++)
        all_text += string(node.getText(i)) + string("\n");

    istringstream parser;
    parser.str(all_text);
    while (parser.good())
        {
        Scalar diameter;
        parser >> diameter;
        if (parser.good())
            m_diameter_array.push_back(diameter);
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
    string all_text;
    for (int i = 0; i < node.nText(); i++)
        all_text += string(node.getText(i)) + string("\n");

    istringstream parser;
    parser.str(all_text);
    while (parser.good())
        {
        // dynamically determine the particle types
        string type;
        parser >> type;
        if (parser.good())
            m_type_array.push_back(getTypeId(type));
        }
    }

/*! \param node XMLNode passed from the top level parser in readFile
    This function extracts all of the data in a \b body node and fills out m_body_array. The number
    of particles in the array is determined dynamically.
*/
void HOOMDInitializer::parseBodyNode(const XMLNode &node)
    {
    // check that this is actually a type node
    string name = node.getName();
    transform(name.begin(), name.end(), name.begin(), ::tolower);
    assert(name == string("body"));

    // extract the data from the node
    string all_text;
    for (int i = 0; i < node.nText(); i++)
        all_text += string(node.getText(i)) + string("\n");

    istringstream parser;
    parser.str(all_text);
    while (parser.good())
        {
        // handle -1 as NO_BODY
        int body;
        parser >> body;

        if (parser.good())
            {
            if (body == -1)
                m_body_array.push_back(NO_BODY);
            else
                m_body_array.push_back(body);
            }
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
    string all_text;
    for (int i = 0; i < node.nText(); i++)
        all_text += string(node.getText(i)) + string("\n");

    istringstream parser;
    parser.str(all_text);
    while (parser.good())
        {
        string type_name;
        unsigned int a, b;
        parser >> type_name >> a >> b;
        if (parser.good())
            {
            BondData::members_t bond;
            bond.tag[0] = a; bond.tag[1] = b;
            m_bonds.push_back(bond);
            m_bond_types.push_back(getBondTypeId(type_name));
            }
        }
    }

void HOOMDInitializer::parseAngleNode(const XMLNode &node)
    {
    // check that this is actually a angle node
    string name = node.getName();
    transform(name.begin(), name.end(), name.begin(), ::tolower);
    assert(name == string("angle"));

    // extract the data from the node
    string all_text;
    for (int i = 0; i < node.nText(); i++)
        all_text += string(node.getText(i)) + string("\n");

    istringstream parser;
    parser.str(all_text);
    while (parser.good())
        {
        string type_name;
        unsigned int a, b, c;
        parser >> type_name >> a >> b >> c;
        if (parser.good())
            {
            AngleData::members_t angle;
            angle.tag[0] = a; angle.tag[1] = b; angle.tag[2] = c;
            m_angles.push_back(angle);
            m_angle_types.push_back(getAngleTypeId(type_name));
            }
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
    string all_text;
    for (int i = 0; i < node.nText(); i++)
        all_text += string(node.getText(i)) + string("\n");

    istringstream parser;
    parser.str(all_text);
    while (parser.good())
        {
        string type_name;
        unsigned int a, b, c, d;
        parser >> type_name >> a >> b >> c >> d;
        if (parser.good())
            {
            DihedralData::members_t dihedral;
            dihedral.tag[0] = a; dihedral.tag[1] = b; dihedral.tag[2] = c; dihedral.tag[3] = d;
            m_dihedrals.push_back(dihedral);
            m_dihedral_types.push_back(getDihedralTypeId(type_name));
            }
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
    string all_text;
    for (int i = 0; i < node.nText(); i++)
        all_text += string(node.getText(i)) + string("\n");

    istringstream parser;
    parser.str(all_text);
    while (parser.good())
        {
        string type_name;
        unsigned int a, b, c, d;
        parser >> type_name >> a >> b >> c >> d;
        if (parser.good())
            {
            ImproperData::members_t improper;
            improper.tag[0] = a; improper.tag[1] = b; improper.tag[2] = c; improper.tag[3] = d;
            m_impropers.push_back(improper);
            m_improper_types.push_back(getImproperTypeId(type_name));
            }
        }
    }

/*! \param node XMLNode passed from the top level parser in readFile
    This function extracts all of the data in a \b dihedral node and fills out m_dihedrals. The number
    of dihedrals in the array is determined dynamically.
*/
void HOOMDInitializer::parsePairNode(const XMLNode &node)
    {
    // check that this is actually a pair node
    string name = node.getName();
    transform(name.begin(), name.end(), name.begin(), ::tolower);
    assert(name == string("pair"));

    // extract the data from the node
    string all_text;
    for (int i = 0; i < node.nText(); i++)
        all_text += string(node.getText(i)) + string("\n");

    istringstream parser;
    parser.str(all_text);
    while (parser.good())
        {
        string type_name;
        unsigned int a, b;
        parser >> type_name >> a >> b;
        if (parser.good())
            {
            PairData::members_t pair;
            pair.tag[0] = a; pair.tag[1] = b;
            m_pairs.push_back(pair);
            m_pair_types.push_back(getPairTypeId(type_name));
            }
        }
    }


/*! \param node XMLNode passed from the top level parser in readFile
    This function extracts all of the data in a \b constraint node and fills out m_constraints. The number
    of constraints in the array is determined dynamically.
*/
void HOOMDInitializer::parseConstraintsNode(const XMLNode &node)
    {
    // check that this is actually a constraint node
    string name = node.getName();
    transform(name.begin(), name.end(), name.begin(), ::tolower);
    assert(name == string("constraint"));

    // extract the data from the node
    string all_text;
    for (int i = 0; i < node.nText(); i++)
        all_text += string(node.getText(i)) + string("\n");

    istringstream parser;
    parser.str(all_text);
    while (parser.good())
        {
        unsigned int a, b;
        Scalar d;
        parser >> a >> b >> d;
        if (parser.good())
            {
            ConstraintData::members_t constraint;
            constraint.tag[0] = a; constraint.tag[1] = b;
            m_constraints.push_back(constraint);
            m_constraint_distances.push_back(d);
            }
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
    string all_text;
    for (int i = 0; i < node.nText(); i++)
        all_text += string(node.getText(i)) + string("\n");

    istringstream parser;
    parser.str(all_text);
    while (parser.good())
        {
        Scalar charge;
        parser >> charge;
        if (parser.good())
            m_charge_array.push_back(charge);
        }
    }

/*! \param node XMLNode passed from the top level parser in readFile
    This function extracts all of the data in a \b orientation node and fills out m_orientation. The number
    of particles in the array is determined dynamically.
*/
void HOOMDInitializer::parseOrientationNode(const XMLNode &node)
    {
    // check that this is actually a charge node
    string name = node.getName();
    transform(name.begin(), name.end(), name.begin(), ::tolower);
    assert(name == string("orientation"));

    // extract the data from the node
    string all_text;
    for (int i = 0; i < node.nText(); i++)
        all_text += string(node.getText(i)) + string("\n");

    istringstream parser;
    parser.str(all_text);
    while (parser.good())
        {
        Scalar ox, oy, oz, ow;
        parser >> ox >> oy >> oz >> ow;
        if (parser.good())
            m_orientation.push_back(make_scalar4(ox, oy, oz, ow));
        }
    }

/*! \param node XMLNode passed from the top level parser in readFile
    This function extracts all of the data in a \b angmom node and fills out m_angmom. The number
    of particles in the array is determined dynamically.
*/
void HOOMDInitializer::parseAngularMomentumNode(const XMLNode &node)
    {
    // check that this is actually a charge node
    string name = node.getName();
    transform(name.begin(), name.end(), name.begin(), ::tolower);
    assert(name == string("angmom"));

    // extract the data from the node
    string all_text;
    for (int i = 0; i < node.nText(); i++)
        all_text += string(node.getText(i)) + string("\n");

    istringstream parser;
    parser.str(all_text);
    while (parser.good())
        {
        Scalar mx, my, mz, mw;
        parser >> mx >> my >> mz >> mw;
        if (parser.good())
            m_angmom.push_back(make_scalar4(mx, my, mz, mw));
        }
    }


/*! \param node XMLNode passed from the top level parser in readFile
    This function extracts all of the data in a \b moment_inertia node and fills out m_moment_inertia. The number
    of particles in the array is determined dynamically.
*/
void HOOMDInitializer::parseMomentInertiaNode(const XMLNode &node)
    {
    // check that this is actually a charge node
    string name = node.getName();
    transform(name.begin(), name.end(), name.begin(), ::tolower);
    assert(name == string("moment_inertia"));

    // extract the data from the node
    string all_text;
    for (int i = 0; i < node.nText(); i++)
        all_text += string(node.getText(i)) + string("\n");

    istringstream parser;
    parser.str(all_text);
    bool read_offdiagonal = (m_xml_version == "1.4" || m_xml_version == "1.5");

    if (read_offdiagonal)
        {
        m_exec_conf->msg->warning() << "Ignoring off-diagonal moments of inertia in this XML file version < 1.6"
            << std::endl;
        }

    while (parser.good())
        {
        Scalar3 I;
        Scalar tmp;
        parser >> I.x;
        if (read_offdiagonal)
            {
            parser >> tmp;
            parser >> tmp;
            }
        parser >> I.y;
        if (read_offdiagonal) parser >> tmp;
        parser >> I.z;

        if (parser.good())
            m_moment_inertia.push_back(I);
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

/*! \param name Name to get type id of
    If \a name has already been added, this returns the type index of that name.
    If \a name has not yet been added, it is added to the list and the new id is returned.
*/
unsigned int HOOMDInitializer::getPairTypeId(const std::string& name)
    {
    // search for the type mapping
    for (unsigned int i = 0; i < m_pair_type_mapping.size(); i++)
        {
        if (m_pair_type_mapping[i] == name)
            return i;
        }
    // add a new one if it is not found
    m_pair_type_mapping.push_back(name);
    return (unsigned int)m_pair_type_mapping.size()-1;
    }


void export_HOOMDInitializer(py::module& m)
    {
    py::class_< HOOMDInitializer >(m,"HOOMDInitializer")
    .def(py::init<std::shared_ptr<const ExecutionConfiguration>, const string&>())
    .def(py::init<std::shared_ptr<const ExecutionConfiguration>, const string&, bool>())
    .def("getTimeStep", &HOOMDInitializer::getTimeStep)
    .def("setTimeStep", &HOOMDInitializer::setTimeStep)
    .def("getSnapshot", &HOOMDInitializer::getSnapshot)
    ;
    }
