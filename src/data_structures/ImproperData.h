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
// Maintainer: dnlebard

/*! \file ImproperData.h
 	\brief Declares ImproperData and related classes
 */
 
#ifndef __IMPROPERDATA_H__
#define __IMPROPERDATA_H__

#include <vector>
#include <boost/shared_ptr.hpp>
#include <boost/signal.hpp>
#include <boost/utility.hpp>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include "ImproperData.cuh"
#endif

// forward declaration of ParticleData to avoid circular references
class ParticleData;

//! Stores an improper between four particles
/*! Each improper is given an integer \c type from 0 to \c NImproperTypes-1 and the \em tags
	of the three improper particles.
	\ingroup data_structs
*/
struct Improper
	{
	//! Constructs an improper
	/*! \param improper_type Type index of the improper
		\param tag_a Tag of the first particle in the improper
		\param tag_b Tag of the second particle in the improper
		\param tag_c Tag of the third particle in the improper
		\param tag_d Tag of the forth particle in the improper
	*/
	Improper(unsigned int improper_type, unsigned int tag_a, unsigned int tag_b, unsigned int tag_c, unsigned int tag_d) : type(improper_type), a(tag_a), b(tag_b), c(tag_c), d(tag_d) { }
	unsigned int type;	//!< The type index of the improper
	unsigned int a;		//!< The tag of the first particle in the improper
	unsigned int b;		//!< The tag of the second particle in the improper
	unsigned int c;		//!< The tag of the third particle in the improper
	unsigned int d;		//!< The tag of the forth particle in the improper
	};
	
//! Stores all impropers in the simulation and mangages the GPU improper data structure
/*! ImproperData tracks every improper defined in the simulation. On the CPU, impropers are stored just
	as a simple vector of Improper structs. On the GPU, the list of impropers is decomposed into a 
	table with every column listing the impropers of a single particle: see 
	gpu_impropertable_array for more info.
	
	A ParticleData instance owns a single ImproperData which classes such as ImproperForceCompute 
	can access for their needs.
	
	Impropers can be dynamically added, although doing this on a per-timestep basis can 
	slow performance significantly. For simplicity and convinence, however, the number
	of improper types cannot change after initialization.
	\ingroup data_structs
*/
class ImproperData : boost::noncopyable
	{
	public:
		//! Constructs an empty list with no impropers
		ImproperData(boost::shared_ptr<ParticleData> pdata, unsigned int n_improper_types = 0);
		
		//! Destructor
		~ImproperData();
		
		//! Add an improper to the list
		void addImproper(const Improper& improper);
		
		//! Get the number of impropers
		/*! \return Number of impropers present
		*/
		const unsigned int getNumImpropers() const { return (unsigned int)m_impropers.size(); }
		
		//! Get access to a improper
		/*! \param i Improper to access
		*/
		const Improper& getImproper(unsigned int i) const
			{ assert(i < m_impropers.size()); return m_impropers[i]; }
		
		//! Get the number of improper types
		/*! \return Number of improper types in the list of impropers
		*/
		const unsigned int getNImproperTypes() const { return m_n_improper_types; }
		
		//! Set the type mapping
		void setImproperTypeMapping(const std::vector<std::string>& improper_type_mapping);
		
		//! Gets the particle type index given a name
		unsigned int getTypeByName(const std::string &name);
		
		//! Gets the name of a given particle type index
		std::string getNameByType(unsigned int type);
		
		# ifdef ENABLE_CUDA
		//! Access the impropers on the GPU
		std::vector<gpu_impropertable_array>& acquireGPU();

		#endif

		
	private:
		const unsigned int m_n_improper_types;				//!< Number of improper types
		bool m_impropers_dirty;								//!< True if the improper list has been changed
		boost::shared_ptr<ParticleData> m_pdata;							//!< Particle Data these impropers belong to
		std::vector<Improper> m_impropers;						//!< List of impropers on the CPU
		std::vector<std::string> m_improper_type_mapping;	//!< Mapping between improper type indices and names
		
		boost::signals::connection m_sort_connection;	//!< Connection to the resort signal from ParticleData
		
		//! Helper function to set the dirty flag when particles are resorted
		/*! setDirty() just sets the \c m_impropers_dirty flag when partciles are sorted or an improper is added.
			The flag is used to test if the data structure needs updating on the GPU.
		*/
		void setDirty() { m_impropers_dirty = true; }
			
		#ifdef ENABLE_CUDA
		std::vector<gpu_impropertable_array> m_gpu_improperdata;	//!< List of impropers on the GPU
		uint4 *m_host_impropers;				//!< Host copy of the improper list (3atoms of a,b,c, or d, plus the type)
		uint1 *m_host_impropersABCD;			//!< Host copy of the improperABCD list
		unsigned int *m_host_n_impropers;		//!< Host copy of the number of impropers

                /*! \enum improperABCD tells if the Improper is on the a,b,c, or d atom
                */
		enum improperABCD
			{
			a_atom = 0,	//!< atom is the a particle in an a-b-c-d quartet
			b_atom = 1,	//!< atom is the b particle in an a-b-c-d quartet
			c_atom = 2,	//!< atom is the c particle in an a-b-c-d quartet
			d_atom = 3	//!< atom is the d particle in an a-b-c-d quartet
			};
		
		//! Helper function to update the improper table on the device
		void updateImproperTable();
		
		//! Helper function to reallocate the improper table on the device
		void reallocateImproperTable(int height);
		
		//! Helper function to allocate the improper table
		void allocateImproperTable(int height);
		
		//! Helper function to free the improper table
		void freeImproperTable();
		
		//! Copies the improper table to the device
		void copyImproperTable();
		
		#endif
	};

//! Exports ImproperData to python
void export_ImproperData();

#endif
