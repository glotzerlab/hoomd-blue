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

/*! \file HOOMDDumpWriter.cc
	\brief Defines the HOOMDDumpWriter class
*/

#ifdef USE_PYTHON
#include <boost/python.hpp>
using namespace boost::python;
#endif

#include <sstream>
#include <fstream>
#include <stdexcept>
#include <iomanip>
#include <boost/shared_ptr.hpp>

#include "HOOMDDumpWriter.h"

using namespace std;

/*! \param pdata Particle data to read when dumping files
	\param base_fname The base name of the file xml file to output the information

	\note .timestep.xml will be apended to the end of \a base_fname when analyze() is called.
*/
HOOMDDumpWriter::HOOMDDumpWriter(boost::shared_ptr<ParticleData> pdata, std::string base_fname)
	: Analyzer(pdata), m_base_fname(base_fname), m_output_position(true), m_output_velocity(false), m_output_type(false)
	{
	}

/*! \param enable Set to true to enable the writing of particle positions to the files in analyze()
*/
void HOOMDDumpWriter::setOutputPosition(bool enable)
	{
	m_output_position = enable;
	}

/*!  
	\param enable Set to true to output particle velocities to the XML file on the next call to analyze()

*/
void HOOMDDumpWriter::setOutputVelocity(bool enable)
	{
	m_output_velocity = enable;
	}
/*! \param enable Set to true to output particle types to the XML file on the next call to analyze()
*/
void HOOMDDumpWriter::setOutputType(bool enable)
	{
	m_output_type = enable;
	}

/*! \param timestep Current time step of the simulation
	Writes a snapshot of the current state of the ParticleData to a hoomd_xml file.
*/
void HOOMDDumpWriter::analyze(unsigned int timestep)
	{
	ostringstream full_fname;
	Scalar Lx,Ly,Lz;
	string filetype = ".xml";
	
	// Generate a filename with the timestep padded to ten zeros
	full_fname << m_base_fname << "." << setfill('0') << setw(10) << timestep << filetype;

	// open the file for writing
	ofstream f(full_fname.str().c_str());
	
	if (!f.good())
		{
		cerr << "Unable to open dump file for writing: " << full_fname.str() << endl;
		throw runtime_error("Error writting hoomd_xml dump file");
		}

	// acquire the particle data
	ParticleDataArraysConst arrays = m_pdata->acquireReadOnly();
	BoxDim box = m_pdata->getBox();
	Lx=Scalar(box.xhi-box.xlo);
	Ly=Scalar(box.yhi-box.ylo);
	Lz=Scalar(box.zhi-box.zlo);
	
	f << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" <<endl;
	f << "<hoomd_xml>" << endl;
	f << "<configuration time_step=\"" << timestep << "\">" << endl;
	
	f << "<box units=\"sigma\" " << " Lx=\""<< Lx << "\" Ly=\""<< Ly << "\" Lz=\""<< Lz << "\"/>" << endl;

	// If the position flag is true output the position of all particles to the file 
	if (m_output_position)
		{
		f << "<position units=\"sigma\">" << endl;
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
				cerr << "Unexpected error writing HOOMD dump file" << endl;
				throw runtime_error("Error writting HOOMD dump file");
				}
			}
		f <<"</position>"<<endl;
		}

	// If the velocity flag is true output the velocity of all particles to the file
	if (m_output_velocity)
		{
		f <<"<velocity units=\"sigma/tau\">" << endl;

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
				cerr << "Unexpected error writing HOOMD dump file" << endl;
				throw runtime_error("Error writting HOOMD dump file");
				}
			}

		f <<"</velocity>" <<endl;
		}

	// If the Type flag is true output the types of all particles to an xml file
	if	(m_output_type)
		{
		f <<"<type>" <<endl;
		for (unsigned int j = 0; j < arrays.nparticles; j++)
			{
			int i;
			i= arrays.rtag[j];
			f << arrays.type[i] << endl;
			}
		f <<"</type>" <<endl;
		}
	f << "</configuration>" << endl;
	f << "</hoomd_xml>" <<endl;

	if (!f.good())
		{
		cerr << "Unexpected error writing HOOMD dump file" << endl;
		throw runtime_error("Error writting HOOMD dump file");
		}

	f.close();
	m_pdata->release();
	}

#ifdef USE_PYTHON
void export_HOOMDDumpWriter()
	{
	class_<HOOMDDumpWriter, boost::shared_ptr<HOOMDDumpWriter>, bases<Analyzer>, boost::noncopyable>
		("HOOMDDumpWriter", init< boost::shared_ptr<ParticleData>, std::string >())
		.def("setOutputPosition", &HOOMDDumpWriter::setOutputPosition)
		.def("setOutputVelocity", &HOOMDDumpWriter::setOutputVelocity)
		.def("setOutputType", &HOOMDDumpWriter::setOutputType)
		;
	}
#endif
	
