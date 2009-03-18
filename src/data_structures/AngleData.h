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

/*! \file AngleData.h
 	\brief Declares AngleData and related classes
 */
 
#ifndef __ANGLEDATA_H__
#define __ANGLEDATA_H__

#include <vector>
#include <boost/shared_ptr.hpp>
#include <boost/signal.hpp>
#include <boost/utility.hpp>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include "AngleData.cuh"
#endif

// forward declaration of ParticleData to avoid circular references
class ParticleData;

//! Stores an angle between two particles
/*! Each angle is given an integer \c type from 0 to \c NAngleTypes-1 and the \em tags
	of the three angled particles.
	\ingroup data_structs
*/
struct Angle
	{
	//! Constructs an angle
	/*! \param angle_type Type index of the angle
		\param tag_a Tag of the first particle in the angle
		\param tag_b Tag of the second particle in the angle
		\param tag_c Tag of the third particle in the angle
	*/
	Angle(unsigned int angle_type, unsigned int tag_a, unsigned int tag_b, unsigned int tag_c) : type(angle_type), a(tag_a), b(tag_b), c(tag_c) { }
	unsigned int type;	//!< The type index of the angle
	unsigned int a;		//!< The tag of the first particle in the angle
	unsigned int b;		//!< The tag of the second particle in the angle
	unsigned int c;		//!< The tag of the third particle in the angle
	};
	
//! Stores all angles in the simulation and mangages the GPU angle data structure
/*! AngleData tracks every angle defined in the simulation. On the CPU, angles are stored just
	as a simple vector of Angle structs. On the GPU, the list of angles is decomposed into a 
	table with every column listing the angles of a single particle: see 
	gpu_angletable_array for more info.
	
	A ParticleData instance owns a single AngleData which classes such as AngleForceCompute 
	can access for their needs.
	
	Angles can be dynamically added, although doing this on a per-timestep basis can 
	slow performance significantly. For simplicity and convinence, however, the number
	of angle types cannot change after initialization.
	\ingroup data_structs
*/
class AngleData : boost::noncopyable
	{
	public:
		//! Constructs an empty list with no angles
		AngleData(ParticleData* pdata, unsigned int n_angle_types);
		
		//! Destructor
		~AngleData();
		
		//! Add an angle to the list
		void addAngle(const Angle& angle);
		
		//! Get the number of angles
		/*! \return Number of angles present
		*/
		const unsigned int getNumAngles() const { return (unsigned int)m_angles.size(); }
		
		//! Get a given an angle
		/*! \param i Angle to access
		*/
		const Angle& getAngle(unsigned int i) const
			{ assert(i < m_angles.size()); return m_angles[i]; }
		
		//! Get the number of angle types
		/*! \return Number of angle types in the list of angles
		*/
		const unsigned int getNAngleTypes() const { return m_n_angle_types; }
		
		//! Set the type mapping
		void setAngleTypeMapping(const std::vector<std::string>& angle_type_mapping);
		
		//! Gets the particle type index given a name
		unsigned int getTypeByName(const std::string &name);
		
		//! Gets the name of a given particle type index
		std::string getNameByType(unsigned int type);
		
		# ifdef ENABLE_CUDA
		//! Access the angles on the GPU
		std::vector<gpu_angletable_array>& acquireGPU();

		#endif

		
	private:
		const unsigned int m_n_angle_types;				//!< Number of angle types
		bool m_angles_dirty;								//!< True if the angle list has been changed
		ParticleData* m_pdata;							//!< Particle Data these angles belong to
		std::vector<Angle> m_angles;						//!< List of angles on the CPU
		std::vector<std::string> m_angle_type_mapping;	//!< Mapping between angle type indices and names
		
		boost::signals::connection m_sort_connection;	//!< Connection to the resort signal from ParticleData
		
		//! Helper function to set the dirty flag when particles are resorted
		/*! setDirty() just sets the \c m_angles_dirty flag when partciles are sorted or an angle is added.
			The flag is used to test if the data structure needs updating on the GPU.
		*/
		void setDirty() { m_angles_dirty = true; }
			
		#ifdef ENABLE_CUDA
		std::vector<gpu_angletable_array> m_gpu_angledata;	//!< List of angles on the GPU
		uint4 *m_host_angles;				//!< Host copy of the angle list
		unsigned int *m_host_n_angles;		//!< Host copy of the number of angles

                /*! \enum angleABC tells if the Angle is on the a,b,or c atom
                */
		enum angleABC
			{
			a_atom = 0,	//!< atom is the a particle in an a-b-c triplet
			b_atom = 1,	//!< atom is the b particle in an a-b-c triplet
			c_atom = 2	//!< atom is the c particle in an a-b-c triplet
			};
		
		//! Helper function to update the angle table on the device
		void updateAngleTable();
		
		//! Helper function to reallocate the angle table on the device
		void reallocateAngleTable(int height);
		
		//! Helper function to allocate the angle table
		void allocateAngleTable(int height);
		
		//! Helper function to free the angle table
		void freeAngleTable();
		
		//! Copies the angle table to the device
		void copyAngleTable();
		
		#endif
	};

//! Exports AngleData to python
void export_AngleData();

#endif
