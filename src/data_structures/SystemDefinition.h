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

/*! \file SystemDefinition.h
 	\brief Defines the SystemDefinition class
 */

#include "ParticleData.h"
#include "BondData.h"
#include "WallData.h"
#include "RigidData.h"
#include "AngleData.h"
#include "DihedralData.h"

#include <boost/shared_ptr.hpp>

#ifndef __SYSTEM_DEFINITION_H__
#define __SYSTEM_DEFINITION_H__

//! Container class for all data needed to define the MD system
/*! SystemDefinition is a big bucket where all of the data defining the MD system goes.
	Everything is stored as a shared pointer for quick and easy access from within C++
	and python without worrying about data management.
	
	<b>Background and intended usage</b>
	
	The most fundamental data structure stored in SystemDefinition is the ParticleData.
	It stores essential data on a per particle basis (position, velocity, type, mass, etc...)
	as well as defining the number of particles in the system and the simulation box. Many other
	data structures in SystemDefinition also refer to particles and store other data related to
	them (i.e. BondData lists bonds between particles). These will need access to information such
	as the number of particles in the system or potentially some of the per-particle data stored
	in ParticleData. To facilitate this, ParticleData will always be initialized \b fist and its
	shared pointer can then be passed to any future data structure in SystemDefinition that needs 
	such a reference.
	
	More generally, any data structure class in SystemDefinition can potentially reference any other,
	simply by giving the shared pointer to the referenced class to the constructor of the one that
	needs to refer to it. Note that using this setup, there can be no circular references. This is a 
	\b good \b thing ^TM, as it promotes good separation and isolation of the various classes responsibilities.
	
	In rare circumstances, a references back really is required (i.e. notification of referring classes when
	ParticleData resorts particles). Any event based notifications of such should be managed with boost::signals.
	Any ongoing references where two data structure classes are so interwoven that they must constantly refer to
	each other should be avoided (consider merging them into one class).
	
	<b>Initializing</b>
	
	A default constructed SystemDefinition is full of NULL shared pointers. Such is intended to be assigned to
	by one created by a SystemInitializer.
	
	Several other default constructors are provided, mainly to provide backward compatibility to unit tests that
	relied on the simple initialization constructors provided by ParticleData.
	
	\ingroup data_structs
*/
class SystemDefinition
	{
	public:
		//! Constructs a NULL SystemDefinition
		SystemDefinition();
		//! Conctructs a SystemDefinition with a simply initialized ParticleData
		SystemDefinition(unsigned int N, const BoxDim &box, unsigned int n_types=1, unsigned int n_bond_types=0, unsigned int n_angle_types=0, unsigned int n_dihedral_types=0, unsigned int n_improper_types=0, const ExecutionConfiguration& exec_conf=ExecutionConfiguration());
		//! Construct from an initializer (temporary measure until the initializer setup is rewritten)
		SystemDefinition(const ParticleDataInitializer& init, const ExecutionConfiguration&  exec_conf=ExecutionConfiguration());
		
		//! Initialize data
		int init();

		//! Write restart file
		void writeRestart(unsigned int timestep);

		//! Read restart file
		void readRestart(const std::string& file_name);
		
		//! Get the particle data
		boost::shared_ptr<ParticleData> getParticleData() const { return m_particle_data; }
		//! Get the bond data
		boost::shared_ptr<BondData> getBondData() const { return m_bond_data; }
		//! Get the wall data
		boost::shared_ptr<WallData> getWallData() const { return m_wall_data; }
		//! Get the rigid body data
		boost::shared_ptr<RigidData> getRigidData() const { return m_rigid_data; }
		//! Access the angle data defined for the simulation
		boost::shared_ptr<AngleData> getAngleData() { return m_angle_data; }
		//! Access the dihedral data defined for the simulation
		boost::shared_ptr<DihedralData> getDihedralData() { return m_dihedral_data; }
		//! Access the improper data defined for the simulation
		boost::shared_ptr<DihedralData> getImproperData() { return m_improper_data; }
		
	private:
		boost::shared_ptr<ParticleData> m_particle_data;	//!< Particle data for the system
		boost::shared_ptr<BondData> m_bond_data;			//!< Bond data for the system
		boost::shared_ptr<WallData> m_wall_data;			//!< Wall data for the system
		boost::shared_ptr<RigidData> m_rigid_data;			//!< Rigid bodies data for the system
		boost::shared_ptr<AngleData> m_angle_data;			//!< Angle data for the system
		boost::shared_ptr<DihedralData> m_dihedral_data;	//!< Dihedral data for the system
		boost::shared_ptr<DihedralData> m_improper_data;	//!< Improper data for the system
	};

//! Exports SystemDefinition to python
void export_SystemDefinition();

#endif
