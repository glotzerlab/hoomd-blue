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

// $Id: HOOMDBinaryInitializer.cc 2148 2009-10-07 20:05:29Z joaander $
// $URL: https://codeblue.umich.edu/hoomd-blue/svn/trunk/src/data_structures/HOOMDBinaryInitializer.cc $
// Maintainer: joaander

/*! \file HOOMDBinaryInitializer.cc
    \brief Defines the HOOMDBinaryInitializer class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 4267 )
#endif

#include "HOOMDBinaryInitializer.h"

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
HOOMDBinaryInitializer::HOOMDBinaryInitializer(const std::string &fname)
    {
    // initialize member variables
    m_timestep = 0;
    m_box_read = false;
        
    // read in the file
    readFile(fname);
    }

/* XXX: shouldn't the following methods be put into
 * the header so that they get inlined? */

/*! \returns Numer of particles parsed from the XML file
*/
unsigned int HOOMDBinaryInitializer::getNumParticles() const
    {
    assert(m_pos_array.size() > 0);
    return (unsigned int)m_pos_array.size();
    }

/*! \returns Numer of particle types parsed from the XML file
*/
unsigned int HOOMDBinaryInitializer::getNumParticleTypes() const
    {
    assert(m_type_mapping.size() > 0);
    return (unsigned int)m_type_mapping.size();
    }

/*! \returns Box dimensions parsed from the XML file
*/
BoxDim HOOMDBinaryInitializer::getBox() const
    {
    return m_box;
    }

/*! \returns Time step parsed from the XML file
*/
unsigned int HOOMDBinaryInitializer::getTimeStep() const
    {
    return m_timestep;
    }

/* change internal timestep number. */
void HOOMDBinaryInitializer::setTimeStep(unsigned int ts)
    {
    m_timestep = ts;
    }

/*! \param pdata The particle data

    initArrays takes the internally stored copy of the particle data and copies it
    into the provided particle data arrays for storage in ParticleData.
*/
void HOOMDBinaryInitializer::initArrays(const ParticleDataArrays &pdata) const
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
void HOOMDBinaryInitializer::initWallData(boost::shared_ptr<WallData> wall_data) const
    {
    // copy the walls over from our internal list
    for (unsigned int i = 0; i < m_walls.size(); i++)
        wall_data->addWall(m_walls[i]);
    }

/*! \param fname File name of the hoomd_xml file to read in
    \post Internal data arrays and members are filled out from which futre calls
    like initArrays will use to intialize the ParticleData

    This function implements the main parser loop. It reads in XML nodes from the
    file one by one and passes them of to parsers registered in \c m_parser_map.
*/
void HOOMDBinaryInitializer::readFile(const string &fname)
    {
    // Open the file
    cout<< "Reading " << fname << "..." << endl;
    ifstream f(fname.c_str(), ios::in|ios::binary);
    
    // handle errors
    if (f.fail())
        {
        cerr << endl << "***Error! Error opening " << fname << endl << endl;
        throw runtime_error("Error reading binary file");
        }
    
    int version = 1;
    int file_version;
    f.read((char*)&file_version, sizeof(int));
    
    // right now, the version tag doesn't do anything: just warn if they don't match
    if (version != file_version)
        cout << endl
             << "***Warning! hoomd binary file does not match the current version,"
             << " I don't know how to read this. Continuing anyways." << endl << endl;
    
    //parse timestep
    int timestep;
    f.read((char*)&timestep, sizeof(int));	
    m_timestep = timestep;
    
    //parse box
    Scalar Lx,Ly,Lz;
    f.read((char*)&Lx, sizeof(Scalar));	
    f.read((char*)&Ly, sizeof(Scalar));	
    f.read((char*)&Lz, sizeof(Scalar));	
    m_box = BoxDim(Lx,Ly,Lz);
    m_box_read = true;
        
    //parse positions
    bool positions_flag = false;
    f.read((char*)&positions_flag, sizeof(bool));
    if (positions_flag) 
        {
        unsigned int np = 0;
        f.read((char*)&np, sizeof(unsigned int));
        for (unsigned int j = 0; j < np; j++)
            {
            Scalar x, y, z;
            f.read((char*)&x, sizeof(Scalar));	
            f.read((char*)&y, sizeof(Scalar));	
            f.read((char*)&z, sizeof(Scalar));	
            m_pos_array.push_back(vec(x,y,z));
            }
        }

    //parse images
    bool image_flag = false;
    f.read((char*)&image_flag, sizeof(bool));
    if (image_flag) 
        {
        unsigned int np = 0;
        f.read((char*)&np, sizeof(unsigned int));	
        for (unsigned int j = 0; j < np; j++)
            {
            int x, y, z;
            f.read((char*)&x, sizeof(int));	
            f.read((char*)&y, sizeof(int));	
            f.read((char*)&z, sizeof(int));	
            m_image_array.push_back(vec_int(x,y,z));
            }
        }

    //parse velocities
    bool velocities_flag = false;
    f.read((char*)&velocities_flag, sizeof(bool));
    if (velocities_flag) 
        {
        unsigned int np = 0;
        f.read((char*)&np, sizeof(unsigned int));	
        for (unsigned int j = 0; j < np; j++)
            {
            Scalar vx, vy, vz;
            f.read((char*)&vx, sizeof(Scalar));	
            f.read((char*)&vy, sizeof(Scalar));	
            f.read((char*)&vz, sizeof(Scalar));	
            m_vel_array.push_back(vec(vx,vy,vz));
            }
        }

    //parse accelerations
    bool accelerations_flag = false;
    f.read((char*)&accelerations_flag, sizeof(bool));
    if (accelerations_flag) 
        {
        unsigned int np = 0;
        f.read((char*)&np, sizeof(unsigned int));	
        for (unsigned int j = 0; j < np; j++)
            {
            Scalar ax, ay, az;
            f.read((char*)&ax, sizeof(Scalar));	
            f.read((char*)&ay, sizeof(Scalar));	
            f.read((char*)&az, sizeof(Scalar));	
            m_vel_array.push_back(vec(ax,ay,az));
            }
        }

    //parse masses
    bool masses_flag = false;
    f.read((char*)&masses_flag, sizeof(bool));
    if (masses_flag) 
        {
        unsigned int np = 0;
        f.read((char*)&np, sizeof(unsigned int));	
        for (unsigned int j = 0; j < np; j++)
            {
            Scalar mass;
            f.read((char*)&mass, sizeof(Scalar));	
            m_mass_array.push_back(mass);
            }
        }

    //parse diameters
    bool diameters_flag = false;
    f.read((char*)&diameters_flag, sizeof(bool));
    if (diameters_flag) 
        {
        unsigned int np = 0;
        f.read((char*)&np, sizeof(unsigned int));	
        for (unsigned int j = 0; j < np; j++)
            {
            Scalar diameter;
            f.read((char*)&diameter, sizeof(Scalar));	
            m_diameter_array.push_back(diameter);
            }
        }

    //parse types
    bool types_flag = false;
    f.read((char*)&types_flag, sizeof(bool));
    if (types_flag) 
        {
        unsigned int np = 0;
        f.read((char*)&np, sizeof(unsigned int));	
        for (unsigned int j = 0; j < np; j++)
            {
            unsigned int len;
            f.read((char*)&len, sizeof(unsigned int));
            char type_cstr[len+1];
            f.read(type_cstr, len*sizeof(char));
            type_cstr[len] = '\0';
            string type = type_cstr;
            m_type_array.push_back(getTypeId(type));
            }
        }
    
    //parse bonds
    bool bonds_flag = false;
    f.read((char*)&bonds_flag, sizeof(bool));
    if (bonds_flag) 
        {
        unsigned int nb = 0;
        f.read((char*)&nb, sizeof(unsigned int));	
        for (unsigned int j = 0; j < nb; j++)
            {
            unsigned int len;
            f.read((char*)&len, sizeof(unsigned int));
            char bondtype_cstr[len+1];
            f.read(bondtype_cstr, len*sizeof(char));
            bondtype_cstr[len] = '\0';
            string type_name = bondtype_cstr;
            
            unsigned int a, b;
            f.read((char*)&a, sizeof(unsigned int));
            f.read((char*)&b, sizeof(unsigned int));
            
            m_bonds.push_back(Bond(getBondTypeId(type_name), a, b));
            }
        }

    //parse angles
    bool angles_flag = false;
    f.read((char*)&angles_flag, sizeof(bool));
    if (angles_flag) 
        {
        unsigned int na = 0;
        f.read((char*)&na, sizeof(unsigned int));	
        for (unsigned int j = 0; j < na; j++)
            {
            unsigned int len;
            f.read((char*)&len, sizeof(unsigned int));
            char angletype_cstr[len+1];
            f.read(angletype_cstr, len*sizeof(char));
            angletype_cstr[len] = '\0';
            string type_name = angletype_cstr;
            
            unsigned int a, b, c;
            f.read((char*)&a, sizeof(unsigned int));
            f.read((char*)&b, sizeof(unsigned int));
            f.read((char*)&c, sizeof(unsigned int));
            
            m_angles.push_back(Angle(getAngleTypeId(type_name), a, b, c));
            }
        }

    //parse dihedrals
    bool dihedrals_flag = false;
    f.read((char*)&dihedrals_flag, sizeof(bool));
    if (dihedrals_flag) 
        {
        unsigned int nd = 0;
        f.read((char*)&nd, sizeof(unsigned int));	
        for (unsigned int j = 0; j < nd; j++)
            {
            unsigned int len;
            f.read((char*)&len, sizeof(unsigned int));
            char dihedraltype_cstr[len+1];
            f.read(dihedraltype_cstr, len*sizeof(char));
            dihedraltype_cstr[len] = '\0';
            string type_name = dihedraltype_cstr;
            
            unsigned int a, b, c, d;
            f.read((char*)&a, sizeof(unsigned int));
            f.read((char*)&b, sizeof(unsigned int));
            f.read((char*)&c, sizeof(unsigned int));
            f.read((char*)&d, sizeof(unsigned int));
            
            m_dihedrals.push_back(Dihedral(getDihedralTypeId(type_name), a, b, c, d));
            }
        }

    //parse impropers
    bool impropers_flag = false;
    f.read((char*)&impropers_flag, sizeof(bool));
    if (impropers_flag) 
        {
        unsigned int nd = 0;
        f.read((char*)&nd, sizeof(unsigned int));	
        for (unsigned int j = 0; j < nd; j++)
            {
            unsigned int len;
            f.read((char*)&len, sizeof(unsigned int));
            char impropertype_cstr[len+1];
            f.read(impropertype_cstr, len*sizeof(char));
            impropertype_cstr[len] = '\0';
            string type_name = impropertype_cstr;
            
            unsigned int a, b, c, d;
            f.read((char*)&a, sizeof(unsigned int));
            f.read((char*)&b, sizeof(unsigned int));
            f.read((char*)&c, sizeof(unsigned int));
            f.read((char*)&d, sizeof(unsigned int));
            
            m_impropers.push_back(Dihedral(getImproperTypeId(type_name), a, b, c, d));
            }
        }
    
    //charges?
    
    //parse walls
    bool wall_flag = false;
    f.read((char*)&wall_flag, sizeof(bool));
    if (wall_flag)
        {
        unsigned int nw = 0;
        f.read((char*)&nw, sizeof(unsigned int));	
        for (unsigned int j = 0; j < nw; j++)
            {
            Scalar ox, oy, oz, nx, ny, nz;
            f.read((char*)&(ox), sizeof(Scalar));	
            f.read((char*)&(oy), sizeof(Scalar));	
            f.read((char*)&(oz), sizeof(Scalar));	
            f.read((char*)&(nx), sizeof(Scalar));	
            f.read((char*)&(ny), sizeof(Scalar));	
            f.read((char*)&(nz), sizeof(Scalar));	
            m_walls.push_back(Wall(ox,oy,oz,nx,ny,nz));
            }
        }
    
    // check for required items in the file
    if (!m_box_read)
        {
        cerr << endl
             << "***Error! A <box> node is required to define the dimensions of the simulation box"
             << endl << endl;
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
        cerr << endl << "***Error! " << m_vel_array.size() << " velocities != " << m_pos_array.size()
             << " positions" << endl << endl;
        throw runtime_error("Error extracting data from hoomd_xml file");
        }
    if (m_mass_array.size() != 0 && m_mass_array.size() != m_pos_array.size())
        {
        cerr << endl << "***Error! " << m_mass_array.size() << " masses != " << m_pos_array.size()
             << " positions" << endl << endl;
        throw runtime_error("Error extracting data from hoomd_xml file");
        }
    if (m_diameter_array.size() != 0 && m_diameter_array.size() != m_pos_array.size())
        {
        cerr << endl << "***Error! " << m_diameter_array.size() << " diameters != " << m_pos_array.size()
             << " positions" << endl << endl;
        throw runtime_error("Error extracting data from hoomd_xml file");
        }
    if (m_image_array.size() != 0 && m_image_array.size() != m_pos_array.size())
        {
        cerr << endl << "***Error! " << m_image_array.size() << " images != " << m_pos_array.size()
             << " positions" << endl << endl;
        throw runtime_error("Error extracting data from hoomd_xml file");
        }
    if (m_type_array.size() != 0 && m_type_array.size() != m_pos_array.size())
        {
        cerr << endl << "***Error! " << m_type_array.size() << " type values != " << m_pos_array.size()
             << " positions" << endl << endl;
        throw runtime_error("Error extracting data from hoomd_xml file");
        }
    if (m_charge_array.size() != 0 && m_charge_array.size() != m_pos_array.size())
        {
        cerr << endl << "***Error! " << m_charge_array.size() << " charge values != " << m_pos_array.size()
             << " positions" << endl << endl;
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

/*! \param name Name to get type id of
    If \a name has already been added, this returns the type index of that name.
    If \a name has not yet been added, it is added to the list and the new id is returned.
*/
unsigned int HOOMDBinaryInitializer::getTypeId(const std::string& name)
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
unsigned int HOOMDBinaryInitializer::getBondTypeId(const std::string& name)
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
unsigned int HOOMDBinaryInitializer::getAngleTypeId(const std::string& name)
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
unsigned int HOOMDBinaryInitializer::getDihedralTypeId(const std::string& name)
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
unsigned int HOOMDBinaryInitializer::getImproperTypeId(const std::string& name)
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
unsigned int HOOMDBinaryInitializer::getNumBondTypes() const
    {
    return (unsigned int)m_bond_type_mapping.size();
    }

/*! \return Number of angle types determined from the XML file
*/
unsigned int HOOMDBinaryInitializer::getNumAngleTypes() const
    {
    return (unsigned int)m_angle_type_mapping.size();
    }

/*! \return Number of dihedral types determined from the XML file
*/
unsigned int HOOMDBinaryInitializer::getNumDihedralTypes() const
    {
    return (unsigned int)m_dihedral_type_mapping.size();
    }

/*! \return Number of improper types determined from the XML file
*/
unsigned int HOOMDBinaryInitializer::getNumImproperTypes() const
    {
    return (unsigned int)m_improper_type_mapping.size();
    }

/*! \param bond_data Shared pointer to the BondData to be initialized
    Adds all bonds found in the XML file to the BondData
*/
void HOOMDBinaryInitializer::initBondData(boost::shared_ptr<BondData> bond_data) const
    {
    // loop through all the bonds and add a bond for each
    for (unsigned int i = 0; i < m_bonds.size(); i++)
        bond_data->addBond(m_bonds[i]);
        
    bond_data->setBondTypeMapping(m_bond_type_mapping);
    }

/*! \param angle_data Shared pointer to the AngleData to be initialized
    Adds all angles found in the XML file to the AngleData
*/
void HOOMDBinaryInitializer::initAngleData(boost::shared_ptr<AngleData> angle_data) const
    {
    // loop through all the angles and add an angle for each
    for (unsigned int i = 0; i < m_angles.size(); i++)
        angle_data->addAngle(m_angles[i]);
        
    angle_data->setAngleTypeMapping(m_angle_type_mapping);
    }

/*! \param dihedral_data Shared pointer to the DihedralData to be initialized
    Adds all dihedrals found in the XML file to the DihedralData
*/
void HOOMDBinaryInitializer::initDihedralData(boost::shared_ptr<DihedralData> dihedral_data) const
    {
    // loop through all the dihedrals and add an dihedral for each
    for (unsigned int i = 0; i < m_dihedrals.size(); i++)
        dihedral_data->addDihedral(m_dihedrals[i]);
        
    dihedral_data->setDihedralTypeMapping(m_dihedral_type_mapping);
    }

/*! \param improper_data Shared pointer to the ImproperData to be initialized
    Adds all impropers found in the XML file to the ImproperData
*/
void HOOMDBinaryInitializer::initImproperData(boost::shared_ptr<DihedralData> improper_data) const
    {
    // loop through all the impropers and add an improper for each
    for (unsigned int i = 0; i < m_impropers.size(); i++)
        improper_data->addDihedral(m_impropers[i]);
        
    improper_data->setDihedralTypeMapping(m_improper_type_mapping);
    }

/*! \returns A mapping of type ids to type names deteremined from the XML input file
*/
std::vector<std::string> HOOMDBinaryInitializer::getTypeMapping() const
    {
    return m_type_mapping;
    }

void export_HOOMDBinaryInitializer()
    {
    class_< HOOMDBinaryInitializer, bases<ParticleDataInitializer> >("HOOMDBinaryInitializer", init<const string&>())
    // virtual methods from ParticleDataInitializer are inherited
    .def("getTimeStep", &HOOMDBinaryInitializer::getTimeStep)
    .def("setTimeStep", &HOOMDBinaryInitializer::setTimeStep)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

