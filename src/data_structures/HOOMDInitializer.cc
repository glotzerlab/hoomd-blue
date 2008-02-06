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

/*! \param fname File name with the data to load
*/
HOOMDInitializer::HOOMDInitializer(const std::string &fname)
 	{
	m_particles = NULL;
	m_bonds = NULL;
	m_nparticle_types = 0;
	loud = false;
	readFile(fname);
	}
		
HOOMDInitializer::~HOOMDInitializer()
	{
	delete[] m_particles;
	delete[] m_bonds;
	}

unsigned int HOOMDInitializer::getNumParticles() const
	{
	return m_N;
	}
		
unsigned int HOOMDInitializer::getNumParticleTypes() const
	{
	return m_nparticle_types;
	}

BoxDim HOOMDInitializer::getBox() const
	{		
	return m_box;
	}

unsigned int HOOMDInitializer::getTimeStep() const
{
	return m_timestep ;
}


/* ! \param pdata The particle data 

*/
void HOOMDInitializer::initArrays(const ParticleDataArrays &pdata) const
	{
	// loop through all the particles and set them up
	for (unsigned int i = 0; i < m_N; i++)
		{
		pdata.x[i] = m_particles[i].x;
		pdata.y[i] = m_particles[i].y;
		pdata.z[i] = m_particles[i].z;
		
		pdata.vx[i] = m_particles[i].vx;
		pdata.vy[i] = m_particles[i].vy;
		pdata.vz[i] = m_particles[i].vz;
		
		pdata.type[i] = m_particles[i].type;
		pdata.tag[i] = i;
		pdata.rtag[i] = i;
		}
	}

/*!	\param fname Name of the XML file to input the data of the particles
	
*/
void HOOMDInitializer::readFile(const string &fname){

	 // Create a Root Node and a child node
	 XMLNode xMainNode;
	 XMLNode xNode;

	 // Open the file and search for the Root element "HOOMD_xml" 
	 xMainNode=XMLNode::openFileHelper(fname.c_str(),"HOOMD_xml");
	
	 cout<< endl << endl;
	 cout<< "Reading " << fname.c_str() << endl;
	 int num_children;
  	 num_children = xMainNode.nChildNode();
	 bool position_flag=false;
	 bool configuration_flag=false;
	 bool box_flag=false;
	 bool bond_flag = false;

	for(int j=0;j<num_children;j++)
	{
		xNode = xMainNode.getChildNode(j);
		std::string name = xNode.getName();
		unsigned int i;
		istringstream f;
		

		// Find the Child Node "Configuration"  to extract the time_step 
		if(name=="Configuration")
		{
			configuration_flag = true;
			if(!xNode.isEmpty())
			{	
				cout << endl;
					
				if (loud) cout <<" number of elements for this node config is "<<xNode.nAttribute()<<endl;

				if (loud) cout <<"CONFIGURATION DETAILS "<<endl;
				
				cout << "Reading Time step..." << endl;
				
				if (loud) cout << "Time step           -  " << xNode.getAttribute("time_step") << endl;
				m_timestep = atoi(xNode.getAttribute("time_step"));
				
				if(xNode.isAttributeSet("N"))
				{
				
					cout << "Reading number of particles..." << endl;
					cout << "	Number of particles read -  " <<xNode.getAttribute("N")<<endl;
					m_N = atoi(xNode.getAttribute("N")) ;
				
					// Allocate memory for all particles 
					m_particles = new particle[m_N];
				}
				else
				{
					cout<<"Number of particles used in the simulation is not specified in the XML file "<<endl;
					throw runtime_error("Error in reading Number of particles from XML file ");
				}
				
				cout << "Reading types of particles..." << endl;

				if (loud) cout << "Types of particles  -  " << xNode.getAttribute("NTypes")<<endl;
				m_nparticle_types  = atoi(xNode.getAttribute("NTypes"))  ;
				
				
				if(xNode.isAttributeSet("NBonds"))
				{
					bond_flag=true;
					
					cout << "Reading Number of bonds..." << endl;

					if (loud) cout << "Number of bonds     - " << xNode.getAttribute("NBonds")<<endl<<endl;
					m_nbonds  = atoi(xNode.getAttribute("NBonds"))  ;
					// Allocate memory for bonds 
					m_bonds = new bond[m_nbonds];
				}
				else
				{
					cout << "Total Number of Bond forces in the simulation is not present in the XML file "<<endl<<endl;
				}

			}
			else
			{
				cerr<<"The Input XML files does not have Configuration Details like Number of particles, Types of particles and time step "<<endl;
				throw runtime_error("Error in reading Configuration Details of particles");
			}
		}
		
		// Get the Simulation Box details from the XML file 	
		else if(name=="Box")
		{
			box_flag = true;
			if(!xNode.isEmpty())
			{ 
				Scalar Lx,Ly,Lz;
				istringstream temp;
								
				cout << "Reading Box dimentions..." << endl;
				
				if (loud) cout << "SIMULATION BOX DIMENSIONS " << endl;
				if (loud) cout <<"Box = "<< xNode.getAttribute("Lx")<<" x "<< xNode.getAttribute("Ly")<<" x "<<xNode.getAttribute("Lz")<<"  "<<xNode.getAttribute("Units")<< endl<<endl ;
				//Lx = Scalar ((xNode.getAttribute("Lx")));
				//Ly = Scalar ((xNode.getAttribute("Ly")));
				//Lz = Scalar ((xNode.getAttribute("Lz")));
				
				//Fred Fixed Bug 
				temp.str(xNode.getAttribute("Lx"));
				temp >> Lx;
				temp.clear();

				temp.str(xNode.getAttribute("Ly"));
				temp >> Ly;
				temp.clear();


				temp.str(xNode.getAttribute("Lz"));
				temp >> Lz;
				temp.clear();
		
				m_box = BoxDim(Lx,Ly,Lz);
			}
			else 
			{
				cerr<<"The Input XMl files does not have Simulation Box details like LX , LY , LZ "<<endl;
				throw runtime_error("Error in reading Simulation Box Details of particles");
			}
		}
		
		// Check "Position" node to extract the x,y,z coordinates of all particles 
		else if(name=="Position")
		{
			position_flag = true;
			if(!xNode.isEmpty())
			{ 
				cout << "Reading Position details..." << endl ; 
				if (loud) cout << "unit                - " << xNode.getAttribute("Units")<<endl;
				f.str(xNode.getText());
				for (i = 0 ; i< m_N;i++)
				{
					if (!f.good())
					{
						cerr << "Unexpected error in reading Position of particles" << endl;
						throw runtime_error("Error in reading position of particles");
					}
				f >> m_particles[i].x >> m_particles[i].y >> m_particles[i].z ; 
				}
			if (loud) cout<<"** Retrieved information on POSITION of all particles ** "<< endl<<endl ;
			}
			else
			{
				cerr<<"The Input XMl files does not have Position details of particles like X , Y , Z  coordinates "<<endl;
				throw runtime_error("Error in reading position details of particles");
			}
		}

		// Check if there is a child node "Velocity" to extract the vx,vy,vz of all particles
		else if(name=="Velocity")
		{
			if(!xNode.isEmpty())
			{ 
				f.clear();
				cout << "Reading Velocity details ... " << endl ;
				if (loud) cout << "unit                - " << xNode.getAttribute("Units")<<endl;
				f.str(xNode.getText());
				for (i = 0 ; i< m_N;i++)
				{
					if (!f.good())
					{
						cerr << "Unexpected error in reading Velocity of particles" << endl;
						throw runtime_error("Error in reading velocity of particles");
					}
					f >> m_particles[i].vx >> m_particles[i].vy >> m_particles[i].vz ;	
				}
				if (loud) cout<<"** Retrieved information on VELOCITY of all particles ** "<< endl<<endl;
			}
			/*
			else
			{	
				cerr<<"The Input XMl files does not have velocity details of particles like vx, vy , vz  "<<endl;
				throw runtime_error("Error in reading velocity details of particles");
			}
			*/
		}

		// Check if there is a child node "Bonds" 
		else if(name=="Bonds" && bond_flag==true)
		{
			if(!xNode.isEmpty())
			{ 
				f.clear();
				cout << "Reading Bond details ... " << endl ;
				if (loud) cout << "unit                - " << xNode.getAttribute("Units")<<endl;
				f.str(xNode.getText());
				for (i = 0 ; i< m_nbonds;i++)
				{
					if (!f.good())
					{
						cerr << "Unexpected error in reading Bond details  of particles" << endl;
						throw runtime_error("Error in reading Bond detals of particles");
					}
					
					f >> m_bonds[i].tag_a >> m_bonds[i].tag_b;	
				}
				if (loud) cout<<"** Retrieved information on Bonds of all particles ** "<< endl<<endl;
			}
			/*
			else
			{	
				cout<<"The Input XMl files does have either Number of bonds or Bond details of particles "<<endl;
				//throw runtime_error("Error in reading Bond detail of particles");
			}
			*/
			
		}


	
		// Check if there is a child node "Type" to extract the x,y,z coordinates of all particles 
		else if(name=="Type")
		{
			if(!xNode.isEmpty())
			{ 
				f.clear();
				cout << "Reading Type details ... " << endl ;
				f.str(xNode.getText());
				for (i = 0 ; i< m_N;i++)
				{
					if (!f.good())
					{
						cerr << "Unexpected error in reading Types of particles" << endl;
						throw runtime_error("Error in reading types of particles");
					}
					f >> m_particles[i].type ;
				}
				if (loud) cout<<"** Retrieved information on TYPES of all particles ** "<< endl<<endl;
			}
			else
			{
				cout << "TYPE DETAILS" << endl ;
				for (i = 0 ; i< m_N;i++)
				{
					m_particles[i].type=0; // Set the values of type of particles to zero ( default ) 
				}
				cout << "TYPES of all particles set to default value - zero "<< endl<<endl;
			}
		}
		else
		{	
			cout<< endl;
			cout<< "Ingnoring "<<xNode.getName()<<"..."<<endl<<endl;
						
		}
		
		
	}
	
	if(!configuration_flag)
	{
		cout << endl;
		//cerr<<"Mandatory Details missing in the XML file"<<endl;
		cerr <<"The Input XMl files does not have configuration details of particles "<<endl;
		throw runtime_error("Error in reading configuration details of particles");
		
	}

	if(!box_flag)
	{
		cout << endl;
		cerr << "The Input XMl files does not have box dimenstion details "<<endl;
		throw runtime_error("Error in reading Box dimension details of particles");
	}

	
	

	if(position_flag == false)
	{
		cout << endl;
		cerr << "The Input XMl files does not have Position details of particles like X , Y , Z  coordinates "<<endl;
		throw runtime_error("Error in reading position details of particles");
	}


	// bonds are optional
	/*if(bond_flag ==false)
	{
		cout<<endl;
		cerr<<"The Input XMl files does not have all the bond details needed "<<endl;
		throw runtime_error("Error in reading Bond details of particles");

	}*/

	

	cout << endl;
	cout << " ===  END === " << endl;

}


		
void HOOMDInitializer::setupNeighborListExclusions(boost::shared_ptr<NeighborList> nlist)
	{
	// loop through all the bonds and add an exclusion for each
	for (unsigned int i = 0; i < m_nbonds; i++)
		nlist->addExclusion(m_bonds[i].tag_a, m_bonds[i].tag_b);
	}
	
void HOOMDInitializer::setupBonds(boost::shared_ptr<BondForceCompute> fc_bond)
	{
	// loop through all the bonds and add a bond for each
	for (unsigned int i = 0; i < m_nbonds; i++)	
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
