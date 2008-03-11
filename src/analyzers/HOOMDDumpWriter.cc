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
#if BOOST_VERSION >= 103400
using namespace boost::iostreams;
#endif

/*! \param pdata Particle data to read when dumping files
	\param base_fname The base name of the file xml file to output the information
	\param compression_flag  Flag to enable/disable compression

	\note .timestep.xml will be apended to the end of \a base_fname when analyze() is called.
*/
HOOMDDumpWriter::HOOMDDumpWriter(boost::shared_ptr<ParticleData> pdata, std::string base_fname, bool compression_flag)
	: Analyzer(pdata), m_base_fname(base_fname),m_compression_flag(compression_flag)
	{
	}

/*! 
	\param position_flag Set to true to output particle positions to the XML file on the next call to analyze()

*/
void HOOMDDumpWriter::setPositionFlag(bool position_flag)
{
	m_position_flag = position_flag;

}
/*!  
	\param velocity_flag Set to true to output particle velocities to the XML file on the next call to analyze()

*/
void HOOMDDumpWriter::setVelocityFlag (bool velocity_flag)
{
	m_velocity_flag = velocity_flag;

}
/*!   
	\param type_flag Set to true to output particle types to the XML file on the next call to analyze()

*/
void HOOMDDumpWriter::setTypeFlag(bool type_flag)
{
	m_type_flag = type_flag;

}


/*! \param timestep Current time step of the simulation

	To output all the information about the particle to an XML file at that particular time step.
	
	The folowing data are outputed unconditionally 
	-Time step
	-Number of particles at that time step
	-Number of particle types
	-Length, Width and Height of the simulation box

	The following following data are outputed on user request 
	(see setPositionFlag() setVelocityFlag() and setTypeFlag())
	-Position of  each particle
	-Velocity of each particle
	-Type of each particle
*/
void HOOMDDumpWriter::analyze(unsigned int timestep)
	{
	ostringstream full_fname;
	ostringstream temp;
	//int Lx,Ly,Lz;
	Scalar Lx,Ly,Lz;
	string filetype;

	if (m_compression_flag == true)
		filetype=".gz";
	else if (m_compression_flag == false )
		filetype=".xml";
	
	// Ten zero padded along with the timestep
	full_fname << m_base_fname << "." << setfill('0') << setw(10) << timestep << filetype;

	// open the file for writing
	ofstream f(full_fname.str().c_str());
	
	if (!f.good())
		{
		cerr << "Unable to open dump file for writing: " << full_fname.str() << endl;
		throw runtime_error("Error writting HOOMD dump file");
		}

	// acquire the particle data
	ParticleDataArraysConst arrays = m_pdata->acquireReadOnly();
	BoxDim box = m_pdata->getBox();
	Lx=Scalar(box.xhi-box.xlo);
	Ly=Scalar(box.yhi-box.ylo);
	Lz=Scalar(box.zhi-box.zlo);
	
	f << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" <<endl;
	f << "<HOOMD_xml>" << endl;
		//f << " <Configuration time_step=\""<< timestep <<"\" N=\""<< arrays.nparticles <<"\" NTypes=\""<< m_ntypes <<"\" />" << endl;
		f << "<Configuration time_step=\""<< timestep <<"\" N=\""<< arrays.nparticles <<"\" NTypes=\""<< m_pdata->getNTypes() <<"\"/>" << endl;
		
		f << "<Box Units=\"sigma\" " << " Lx=\""<< Lx << "\" Ly=\""<< Ly << "\" Lz=\""<< Lz << "\"/>" << endl;
	
		// If the position flag is true output the position of all particles to an xml file 
		if(m_position_flag == true)
		{
			f << "<Position units=\"sigma\">" << endl;

				for (unsigned int j = 0; j < arrays.nparticles; j++)
				{
					int i;
					i= arrays.rtag[j];
						
					// Fred bug Fixed 
					Scalar x = (arrays.x[i]); // - box.xlo) / (box.xhi - box.xlo);
					Scalar y = (arrays.y[i]); // - box.ylo) / (box.yhi - box.ylo);
					Scalar z = (arrays.z[i]); // - box.zlo) / (box.zhi - box.zlo);
					
					f << x <<" "<< y <<" "<< z <<endl;

					if (!f.good())
					{
						cerr << "Unexpected error writing HOOMD dump file" << endl;
						throw runtime_error("Error writting HOOMD dump file");
					}
							
				 }
	   
		  f <<"</Position>"<<endl;
	
		}
		// If the velocity flag is true output the velocity of all particles to an xml file
		if(m_velocity_flag == true)
		{

		 f <<"<Velocity units=\"sigma/tau\">"<<endl;
	
			for (unsigned int j = 0; j < arrays.nparticles; j++)
			{
		
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
			f <<"</Velocity>" <<endl;
		
		  }
		// If the Type flag is true output the types of all particles to an xml file
		if(m_type_flag == true)
		{
			f <<"<Type>" <<endl;
				for (unsigned int j = 0; j < arrays.nparticles; j++)
				{
			
					int i;
					i= arrays.rtag[j];
					f << arrays.type[i]  <<endl;
					
				}
				
			f <<"</Type>" <<endl;
		}
	
		f << "</HOOMD_xml>" <<endl;

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
		("HOOMDDumpWriter", init< boost::shared_ptr<ParticleData>, std::string, bool >())
		.def("setPositionFlag", &HOOMDDumpWriter::setPositionFlag)
		.def("setVelocityFlag", &HOOMDDumpWriter::setVelocityFlag)
		.def("setTypeFlag", &HOOMDDumpWriter::setTypeFlag)
		;
	}
#endif
	
