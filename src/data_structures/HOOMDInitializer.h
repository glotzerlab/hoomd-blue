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

// $Id: HOOMDInitializer.h 506 2007-08-10 15:46:09Z joaander $
// $URL: svn+ssh://fstanley@photon.hopto.org/hoomd/trunk/src/data_structures/HOOMDInitializer.h $

#include "ParticleData.h"
#include "NeighborList.h"
#include "BondForceCompute.h"
#include "xmlParser.h"

#include <string>

#ifndef __HOOMD_INITIALIZER_H__
#define __HOOMD_INITIALIZER_H__

//! Initializes particle data from a Hoomd input file
/*! \ingroup data_structs
*/
class HOOMDInitializer : public ParticleDataInitializer
	{
	public:
		//! Loads in the file and parses the data
		HOOMDInitializer(const std::string &fname);
		
		//! Frees memory
		virtual ~HOOMDInitializer();
		
		//! Returns the number of particles to be initialized
		virtual unsigned int getNumParticles() const;
		
		//! Returns the number of particle types to be initialized
		virtual unsigned int getNumParticleTypes() const;
		
		//! Returns the timestep of the simulation
		virtual unsigned int getTimeStep() const;

		//! Returns the box the particles will sit in
		virtual BoxDim getBox() const;		
		
		//! Initializes the particle data arrays
		virtual void initArrays(const ParticleDataArrays &pdata) const;
		
		//! Adds a neighbor list exclusion for each bond read from the input file
		void setupNeighborListExclusions(boost::shared_ptr<NeighborList> nlist);
		
		//! Calls BondForceCompute::addBond for each bond read from the input file
		void setupBonds(boost::shared_ptr<BondForceCompute> fc_bond);

		//! Makes the ininializer output more descriptive
		void setLoud(bool verb) {loud = verb;}
	private:
		//! Helper function to read the input file
		void readFile(const std::string &fname);
		 
		BoxDim m_box;	//!< Simulation box read from the file

		bool loud; //!< Whether or not to print very descriptive output
		
		struct particle //!< particle data
			{
			particle() : x(0.0), y(0.0), z(0.0), vx(0.0), vy(0.0), vz(0.0), type(0)
				{
				}
			Scalar x;	//!< Particle X -coordinates 
			Scalar y;	//!< Particle Y -coordinates 
			Scalar z;	//!< Particle Z -coordinates 
			Scalar vx;	//!< Velocity of the particle on x axis 
			Scalar vy;	//!< Velocity of the particle on y axis 
			Scalar vz;	//!< Velocity of the particle on z axis 
			unsigned int type;     //!< Particle Type
			};
		
		particle *m_particles;	//!< Particle data read in from the file
					
		struct bond				//!< bond on the particles
			{
			unsigned int tag_a;		//!< First particle in the bond
			unsigned int tag_b;		//!< Second particle in the bond
			};

		bond *m_bonds;	//!< Bonds read in from the file
	
		unsigned int m_nparticle_types; //!< Types of particles in the Simulation Box
		unsigned int m_N;				//!< Number of particles in the Simulation  box
		unsigned int m_nbonds;			//!< Number of bonds
		unsigned int m_timestep;			//!< The time stamp 
	};
	
#ifdef USE_PYTHON
//! Exports HOOMDInitializer to python
void export_HOOMDInitializer();
#endif

#endif

		
	
