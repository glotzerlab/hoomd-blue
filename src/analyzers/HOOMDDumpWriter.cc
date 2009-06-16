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

/*! \file HOOMDDumpWriter.cc
	\brief Defines the HOOMDDumpWriter class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include <sstream>
#include <fstream>
#include <stdexcept>
#include <iomanip>
#include <boost/shared_ptr.hpp>

#include "HOOMDDumpWriter.h"
#include "BondData.h"
#include "AngleData.h"
#include "DihedralData.h"
#include "WallData.h"

using namespace std;
using namespace boost;

/*! \param pdata Particle data to read when dumping files
	\param base_fname The base name of the file xml file to output the information

	\note .timestep.xml will be apended to the end of \a base_fname when analyze() is called.
*/
HOOMDDumpWriter::HOOMDDumpWriter(boost::shared_ptr<ParticleData> pdata, std::string base_fname)
	: Analyzer(pdata), m_base_fname(base_fname), m_output_position(true), m_output_image(false),
	  m_output_velocity(false), m_output_mass(false), m_output_diameter(false), m_output_type(false),
	  m_output_bond(false), m_output_angle(false), m_output_wall(false), m_output_dihedral(false),
	  m_output_improper(false)
	{
	}

/*! \param enable Set to true to enable the writing of particle positions to the files in analyze()
*/
void HOOMDDumpWriter::setOutputPosition(bool enable)
	{
	m_output_position = enable;
	}
	
/*! \param enable Set to true to enable the writing of particle images to the files in analyze()
*/
void HOOMDDumpWriter::setOutputImage(bool enable)
	{
	m_output_image = enable;
	}

/*!\param enable Set to true to output particle velocities to the XML file on the next call to analyze()
*/
void HOOMDDumpWriter::setOutputVelocity(bool enable)
	{
	m_output_velocity = enable;
	}

/*!\param enable Set to true to output particle masses to the XML file on the next call to analyze()
*/
void HOOMDDumpWriter::setOutputMass(bool enable)
	{
	m_output_mass = enable;
	}
	
/*!\param enable Set to true to output particle diameters to the XML file on the next call to analyze()
*/	
void HOOMDDumpWriter::setOutputDiameter(bool enable)
	{
	m_output_diameter = enable;
	}
	
/*! \param enable Set to true to output particle types to the XML file on the next call to analyze()
*/
void HOOMDDumpWriter::setOutputType(bool enable)
	{
	m_output_type = enable;
	}
/*! \param enable Set to true to output bonds to the XML file on the next call to analyze()
*/
void HOOMDDumpWriter::setOutputBond(bool enable)
	{
	m_output_bond = enable;
	}
/*! \param enable Set to true to output angles to the XML file on the next call to analyze()
*/
void HOOMDDumpWriter::setOutputAngle(bool enable)
	{
	m_output_angle = enable;
	}
/*! \param enable Set to true to output walls to the XML file on the next call to analyze()
*/
void HOOMDDumpWriter::setOutputWall(bool enable)
	{
	m_output_wall = enable;
	}
/*! \param enable Set to true to output dihedrals to the XML file on the next call to analyze()
*/
void HOOMDDumpWriter::setOutputDihedral(bool enable)
	{
	m_output_dihedral = enable;
	}
/*! \param enable Set to true to output impropers to the XML file on the next call to analyze()
*/
void HOOMDDumpWriter::setOutputImproper(bool enable)
	{
	m_output_improper = enable;
	}


/*! \param fname File name to write
	\param timestep Current time step of the simulation
*/
void HOOMDDumpWriter::writeFile(std::string fname, unsigned int timestep)
	{
	// open the file for writing
	ofstream f(fname.c_str());
	
	if (!f.good())
		{
		cerr << endl << "***Error! Unable to open dump file for writing: " << fname << endl << endl;
		throw runtime_error("Error writting hoomd_xml dump file");
		}

	// acquire the particle data
	ParticleDataArraysConst arrays = m_pdata->acquireReadOnly();
	BoxDim box = m_pdata->getBox();
	Scalar Lx,Ly,Lz;
	Lx=Scalar(box.xhi-box.xlo);
	Ly=Scalar(box.yhi-box.ylo);
	Lz=Scalar(box.zhi-box.zlo);
	
	f << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" <<endl;
	f << "<hoomd_xml version=\"1.1\">" << endl;
	f << "<configuration time_step=\"" << timestep << "\">" << endl;
	
	f << "<box units=\"sigma\" " << " lx=\""<< Lx << "\" ly=\""<< Ly << "\" lz=\""<< Lz << "\"/>" << endl;

	// If the position flag is true output the position of all particles to the file 
	if (m_output_position)
		{
		f << "<position units=\"sigma\" num=\"" << m_pdata->getN() << "\">" << endl;
		for (unsigned int j = 0; j < arrays.nparticles; j++)
			{
			// use the rtag data to output the particles in the order they were read in
			int i;
			i= arrays.rtag[j];
				
			Scalar x = (arrays.x[i]);
			Scalar y = (arrays.y[i]);
			Scalar z = (arrays.z[i]);
			
			f << x << " " << y << " "<< z << endl;

			if (!f.good())
				{
				cerr << endl << "***Error! Unexpected error writing HOOMD dump file" << endl << endl;
				throw runtime_error("Error writting HOOMD dump file");
				}
			}
		f <<"</position>"<<endl;
		}
		
	// If the image flag is true, output the image of each particle to the file
	if (m_output_image)
		{
		f << "<image num=\"" << m_pdata->getN() << "\">" << endl;
		for (unsigned int j = 0; j < arrays.nparticles; j++)
			{
			// use the rtag data to output the particles in the order they were read in
			int i;
			i= arrays.rtag[j];
				
			int x = (arrays.ix[i]);
			int y = (arrays.iy[i]);
			int z = (arrays.iz[i]);
			
			f << x << " " << y << " "<< z << endl;

			if (!f.good())
				{
				cerr << endl << "***Error! Unexpected error writing HOOMD dump file" << endl << endl;
				throw runtime_error("Error writting HOOMD dump file");
				}
			}
		f <<"</image>"<<endl;
		}
		
	// If the velocity flag is true output the velocity of all particles to the file
	if (m_output_velocity)
		{
		f <<"<velocity units=\"sigma/tau\" num=\"" << m_pdata->getN() << "\">" << endl;

		for (unsigned int j = 0; j < arrays.nparticles; j++)
			{
			// use the rtag data to output the particles in the order they were read in
			int i;
			i= arrays.rtag[j];				
		
			Scalar vx = arrays.vx[i];
			Scalar vy = arrays.vy[i];
			Scalar vz = arrays.vz[i];
			f << vx << " " << vy << " " << vz << endl;
			if (!f.good())
				{
				cerr << endl << "***Error! Unexpected error writing HOOMD dump file" << endl << endl;
				throw runtime_error("Error writting HOOMD dump file");
				}
			}

		f <<"</velocity>" <<endl;
		}
		
	// If the mass flag is true output the mass of all particles to the file
	if (m_output_mass)
		{
		f <<"<mass num=\"" << m_pdata->getN() << "\">" << endl;

		for (unsigned int j = 0; j < arrays.nparticles; j++)
			{
			// use the rtag data to output the particles in the order they were read in
			int i;
			i= arrays.rtag[j];
		
			Scalar mass = arrays.mass[i];
			f << mass << endl;
			if (!f.good())
				{
				cerr << endl << "***Error! Unexpected error writing HOOMD dump file" << endl << endl;
				throw runtime_error("Error writting HOOMD dump file");
				}
			}

		f <<"</mass>" <<endl;
		}
		
	// If the diameter flag is true output the mass of all particles to the file
	if (m_output_diameter)
		{
		f <<"<diameter units=\"sigma\" num=\"" << m_pdata->getN() << "\">" << endl;

		for (unsigned int j = 0; j < arrays.nparticles; j++)
			{
			// use the rtag data to output the particles in the order they were read in
			int i;
			i= arrays.rtag[j];
		
			Scalar diameter = arrays.diameter[i];
			f << diameter << endl;
			if (!f.good())
				{
				cerr << endl << "***Error! Unexpected error writing HOOMD dump file" << endl << endl;
				throw runtime_error("Error writting HOOMD dump file");
				}
			}

		f <<"</diameter>" <<endl;
		}

	// If the Type flag is true output the types of all particles to an xml file
	if	(m_output_type)
		{
		f <<"<type num=\"" << m_pdata->getN() << "\">" <<endl;
		for (unsigned int j = 0; j < arrays.nparticles; j++)
			{
			int i;
			i= arrays.rtag[j];
			f << m_pdata->getNameByType(arrays.type[i]) << endl;
			}
		f <<"</type>" <<endl;
		}
		
	// if the bond flag is true, output the bonds to the xml file
	if (m_output_bond)
		{
		f << "<bond num=\"" << m_pdata->getBondData()->getNumBonds() << "\">" << endl;
		shared_ptr<BondData> bond_data = m_pdata->getBondData();
		
		// loop over all bonds and write them out
		for (unsigned int i = 0; i < bond_data->getNumBonds(); i++)
			{
			Bond bond = bond_data->getBond(i);
			f << bond_data->getNameByType(bond.type) << " " << bond.a << " " << bond.b << endl;
			}
		
		f << "</bond>" << endl;
		}
		
	// if the angle flag is true, output the angles to the xml file
	if (m_output_angle)
		{
		f << "<angle num=\"" << m_pdata->getAngleData()->getNumAngles() << "\">" << endl;
		shared_ptr<AngleData> angle_data = m_pdata->getAngleData();
		
		// loop over all angles and write them out
		for (unsigned int i = 0; i < angle_data->getNumAngles(); i++)
			{
			Angle angle = angle_data->getAngle(i);
			f << angle_data->getNameByType(angle.type) << " " << angle.a  << " " << angle.b << " " << angle.c << endl;
			}
		
		f << "</angle>" << endl;
		}
	
	// if dihedral is true, write out dihedrals to the xml file
	if (m_output_dihedral)
		{
		f << "<dihedral num=\"" << m_pdata->getDihedralData()->getNumDihedrals() << "\">" << endl;
		shared_ptr<DihedralData> dihedral_data = m_pdata->getDihedralData();
		
		// loop over all angles and write them out
		for (unsigned int i = 0; i < dihedral_data->getNumDihedrals(); i++)
			{
			Dihedral dihedral = dihedral_data->getDihedral(i);
			f << dihedral_data->getNameByType(dihedral.type) << " " << dihedral.a  << " " << dihedral.b << " "
			  << dihedral.c << " " << dihedral.d << endl;
			}
		
		f << "</dihedral>" << endl;
		}
		
	// if improper is true, write out impropers to the xml file
	if (m_output_improper)
		{
		f << "<improper num=\"" << m_pdata->getImproperData()->getNumDihedrals() << "\">" << endl;
		shared_ptr<DihedralData> improper_data = m_pdata->getImproperData();
		
		// loop over all angles and write them out
		for (unsigned int i = 0; i < improper_data->getNumDihedrals(); i++)
			{
			Dihedral dihedral = improper_data->getDihedral(i);
			f << improper_data->getNameByType(dihedral.type) << " " << dihedral.a  << " " << dihedral.b << " "
			  << dihedral.c << " " << dihedral.d << endl;
			}
		
		f << "</improper>" << endl;
		}
		
	// if the wall flag is true, output the walls to the xml file
	if (m_output_wall)
		{
		f << "<wall>" << endl;
		shared_ptr<WallData> wall_data = m_pdata->getWallData();
		
		// loop over all walls and write them out
		for (unsigned int i = 0; i < wall_data->getNumWalls(); i++)
			{
			Wall wall = wall_data->getWall(i);
			f << "<coord ox=\"" << wall.origin_x << "\" oy=\"" << wall.origin_y << "\" oz=\"" << wall.origin_z <<
				"\" nx=\"" << wall.normal_x << "\" ny=\"" << wall.normal_y << "\" nz=\"" << wall.normal_z << "\" />" << endl;
			}
		f << "</wall>" << endl;
		}
			
	f << "</configuration>" << endl;
	f << "</hoomd_xml>" <<endl;

	if (!f.good())
		{
		cerr << endl << "***Error! Unexpected error writing HOOMD dump file" << endl << endl;
		throw runtime_error("Error writting HOOMD dump file");
		}

	f.close();
	m_pdata->release();
	
	}

/*! \param timestep Current time step of the simulation
	Writes a snapshot of the current state of the ParticleData to a hoomd_xml file.
*/
void HOOMDDumpWriter::analyze(unsigned int timestep)
	{
	ostringstream full_fname;
	string filetype = ".xml";
	
	// Generate a filename with the timestep padded to ten zeros
	full_fname << m_base_fname << "." << setfill('0') << setw(10) << timestep << filetype;
	writeFile(full_fname.str(), timestep);
	}

void export_HOOMDDumpWriter()
	{
	class_<HOOMDDumpWriter, boost::shared_ptr<HOOMDDumpWriter>, bases<Analyzer>, boost::noncopyable>
		("HOOMDDumpWriter", init< boost::shared_ptr<ParticleData>, std::string >())
		.def("setOutputPosition", &HOOMDDumpWriter::setOutputPosition)
		.def("setOutputImage", &HOOMDDumpWriter::setOutputImage)
		.def("setOutputVelocity", &HOOMDDumpWriter::setOutputVelocity)
		.def("setOutputMass", &HOOMDDumpWriter::setOutputMass)
		.def("setOutputDiameter", &HOOMDDumpWriter::setOutputDiameter)
		.def("setOutputType", &HOOMDDumpWriter::setOutputType)
		.def("setOutputBond", &HOOMDDumpWriter::setOutputBond)
		.def("setOutputAngle", &HOOMDDumpWriter::setOutputAngle)
		.def("setOutputWall", &HOOMDDumpWriter::setOutputWall)
		.def("writeFile", &HOOMDDumpWriter::writeFile)
		;
	}
	
#ifdef WIN32
#pragma warning( pop )
#endif
