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

/*! \file ParticleData.h
	\brief Defines the ParticleData class and associated utilities
*/

#ifndef __PARTICLE_DATA_H__
#define __PARTICLE_DATA_H__

#include <boost/shared_ptr.hpp>

// The requirements state that we need to handle both single and double precision through
// a define
#ifdef SINGLE_PRECISION
//! Floating point type (single precision)
typedef float Scalar;
#else
//! Floating point type (double precision)
typedef double Scalar;
#endif

#include <stdlib.h>

#ifdef USE_CUDA
#include "gpu_pdata.h"
#endif

#include "Profiler.h"

// windows doesn't understand __restrict__, it is __restrict instead
#ifdef WIN32
#define __restrict__ __restrict
#endif

/*! \ingroup hoomd_lib
	@{
*/

/*! \defgroup data_structs Data structures
	\brief All classes that are related to the fundamental data 
		structures for storing particles.
		
	\details See \ref page_dev_info for more information
*/

/*! @}
*/

//! Stores box dimensions
/*! All particles in the ParticleData structure are inside of a box. This struct defines
	that box. Inside is defined as x >= xlo && x < xhi, and similarly for y and z. 
	\note Requirements state that xhi = -xlo, and the same goes for y and z
	\ingroup data_structs
*/
struct BoxDim
	{
	Scalar xlo;	//!< Minimum x coord of the box
	Scalar xhi;	//!< Maximum x coord of the box
	Scalar ylo;	//!< Minimum y coord of the box
	Scalar yhi;	//!< Maximum y coord of the box
	Scalar zlo;	//!< Minimum z coord of the box
	Scalar zhi;	//!< Maximum z coord of the box
	
	//! Constructs a useless box
	BoxDim();	
	//! Constructs a box from -Len/2 to Len/2
	BoxDim(Scalar Len);
	//! Constructs a box from -Len_x/2 to Len_x/2 for each dimension x
	BoxDim(Scalar Len_x, Scalar Len_y, Scalar Len_z);
	};
	
//! Structure of arrays containing the particle data
/*! Once acquired, the user of the ParticleData gets access to the data arrays
	through this structure. 
	Assumptions that the user of this data structure can make:
		- None of the arrays alias each other
		- No other code will be accessing these arrays in parallel (data arrays can
			only be acquired once at a time).
		- All particles are inside the box (see BoxDim)
		- Each rtag element refers uniquely to a single particle (care should be taken
			by the user not to to clobber this)
	More importantly, there are some assumptions that cannot be made:
		- Data may be in a different order next time the arrays are acquired
		- Pointers may be in a different location in memory next time the arrays are acquired
		- Anything updating these arrays CANNOT move particles outside the box
		
	\note Most of the data structures store properties for each particle. For example
		  x[i] is the c-coordinate of particle i. The one exception is the rtag array.
		  Instead of rtag[i] being the tag of particle i, rtag[tag] is the index i itself.
		  
	Values in the type array can range from 0 to ParticleData->getNTypes()-1.
	\ingroup data_structs
*/
struct ParticleDataArrays
	{
	//! Zeros pointers
	ParticleDataArrays();
		
	unsigned int nparticles;	//!< Number of particles in the arrays
	Scalar * __restrict__ x;	//!< array of x-coordinates
	Scalar * __restrict__ y;	//!< array of y-coordinates
	Scalar * __restrict__ z;	//!< array of z-coordinates
	Scalar * __restrict__ vx;	//!< array of x-component of velocities
	Scalar * __restrict__ vy;	//!< array of y-component of velocities
	Scalar * __restrict__ vz;	//!< array of z-component of velocities
	Scalar * __restrict__ ax;	//!< array of x-component of acceleration
	Scalar * __restrict__ ay;	//!< array of y-component of acceleration
	Scalar * __restrict__ az;	//!< array of z-component of acceleration
	
	unsigned int * __restrict__ type; //!< Type index of each particle
	unsigned int * __restrict__ rtag; //!< Reverse-lookup tag.
	unsigned int * __restrict__ tag;  //!< Forward-lookup tag.
	};

//! Read only arrays
/*! This is the same as ParticleDataArrays, but has const pointers to prevent
 	code with read-only access to write to the data arrays
 	\ingroup data_structs
 */
struct ParticleDataArraysConst
	{
	//! Zeros pointers
	ParticleDataArraysConst();		
		
	unsigned int nparticles;	//!< Number of particles in the arrays
	Scalar const * __restrict__ x;	//!< array of x-coordinates
	Scalar const * __restrict__ y;	//!< array of y-coordinates
	Scalar const * __restrict__ z;	//!< array of z-coordinates
	Scalar const * __restrict__ vx;	//!< array of x-component of velocities
	Scalar const * __restrict__ vy;	//!< array of y-component of velocities
	Scalar const * __restrict__ vz;	//!< array of z-component of velocities
	Scalar const * __restrict__ ax;	//!< array of x-component of acceleration
	Scalar const * __restrict__ ay;	//!< array of y-component of acceleration
	Scalar const * __restrict__ az;	//!< array of z-component of acceleration
	
	unsigned int const * __restrict__ type; //!< Type index of each particle
	unsigned int const * __restrict__ rtag; //!< Reverse-lookup tag.
	unsigned int const * __restrict__ tag;  //!< Forward-lookup tag. 
	};	
	
//! similar to ParticleData, only for walls
struct WallDataArrays {

	//!The number of walls contained in the struct
	unsigned int numWalls;

	//!The x coordinate for the origin of the planes
	Scalar * __restrict__ ox;
	//!The y coordinate for the origin of the planes
	Scalar * __restrict__ oy;
	//!The z coordinate for the origin of the planes
	Scalar * __restrict__ oz;
				
	//!The x direction for the normal of the planes
	Scalar * __restrict__ nx;
	//!The y direction for the normal of the planes
	Scalar * __restrict__ ny;
	//!The z direction for the normal of the planes
	Scalar * __restrict__ nz;
				
	//! constructs useless walls
	WallDataArrays();
				
	//! constructs walls along the boundries of a Box
	WallDataArrays(BoxDim);
};
	
class WallData {

	public:
		WallData() : m_walls() {}
		WallData(BoxDim box) : m_walls(box) {}
		~WallData() {}
		WallDataArrays getWallArrays() { return m_walls; }
		void addWall(Scalar ox_p, Scalar oy_p, Scalar oz_p, Scalar nx_p, Scalar ny_p, Scalar nz_p);
	private:
		WallDataArrays m_walls;
};

//! Abstract interface for initializing a ParticleData
/*! A ParticleDataInitializer should only be used with the appropriate constructor
	of ParticleData(). That constructure calls the methods of this class to determine
	the number of particles, number of particle types, the simulation box, and then 
	initializes itself. Then initArrays() is called on a set of acquired
	ParticleDataArrays which the initializer is to fill out.
	
	\note This class is an abstract interface with pure virtual functions. Derived
	classes must implement these methods.
	\ingroup data_structs
	*/ 
class ParticleDataInitializer
	{
	public:
		//! Empty constructor
		ParticleDataInitializer() { }
		//! Empty Destructor
		virtual ~ParticleDataInitializer() { }
		
		//! Returns the number of particles to be initialized
		virtual unsigned int getNumParticles() const = 0;

		//! Returns the number of particles types to be initialized
		virtual unsigned int getNumParticleTypes() const = 0;
		
		//! Returns the box the particles will sit in
		virtual BoxDim getBox() const = 0;
		
		//! Initializes the particle data arrays
		virtual void initArrays(const ParticleDataArrays &pdata) const = 0;
	};
	
//! Manages all of the data arrays for the particles
/*! ParticleData stores and manages particle coordinates, velocities, accelerations, type,
	and tag information. This data must be available both via the CPU and GPU memories. 
	All copying of data back and forth from the GPU is accomplished transparently. To access
	the particle data for read-only purposes: call acquireReadOnly() for CPU access or 
	acquireReadOnlyGPU() for GPU access. Similarly, if any values in the data are to be 
	changed, data pointers can be gotten with: acquireReadWrite() for the CPU and 
	acquireReadWriteGPU() for the GPU. A single ParticleData cannot be acquired multiple
	times without releasing it. Call release() to do so. An assert() will fail in debug 
	builds if the ParticleData is acquired more than once without being released.
	
	For performance reasons, data is stored as simple arrays. Once ParticleDataArrays 
	(or the const counterpart) has been acquired, the coordinates of the particle with
	<em>index</em> \c i can be accessed with <code>arrays.x[i]</code>, <code>arrays.y[i]</code>,
	and <code>arrays.z[i]</code> where \c i runs from 0 to <code>arrays.nparticles</code>.
	
	Velocities can similarly be accessed through the members vx,vy, and vz
	
	\warning Particles can and will be rearranged in the arrays throughout a simulation.
	So, a particle that was once at index 5 may be at index 123 the next time the data
	is acquired. Individual particles can be tracked through all these changes by their tag.
	<code>arrays.tag[i]</code> identifies the tag of the particle that currently has index
	\c i, and the index of a particle with tag \c tag can be read from <code>arrays.rtag[tag]</code>.
	
	In order to help other classes deal with particles changing indices, any class that 
	changes the order must call setSortedTimestep(). Any class interested in determining
	when the last resort was done can find out by calling getSortedTimestep().
	
	\note When writing to the particle data, particles must not be moved outside the box.
	In debug builds, release() will fail an assertion if this is done.
	\ingroup data_structs
*/
class ParticleData
	{
	public:
		//! Construct with N particles in the given box
		ParticleData(unsigned int N, const BoxDim &box, unsigned int n_types=1);
		//! Construct from an initializer
		ParticleData(const ParticleDataInitializer& init);
		//! Destructor
		virtual ~ParticleData();		
			
		//! Get the simulation box
		const BoxDim& getBox() const;
		//! Set the simulation box
		void setBox(const BoxDim &box);

		WallData getWalls() { return m_wallData; }
		
		//! Get the number of particles
		/*! \return Number of particles in the box
		*/
		unsigned int getN() const
			{
			return m_arrays.nparticles;
			}
		
		//! Get the number of particle types
		/*! \return Number of particle types
			\note Particle types are indexed from 0 to NTypes-1
		*/
		unsigned int getNTypes() const
			{
			return m_ntypes;
			}
		
		//! Acquire read access to the particle data
		const ParticleDataArraysConst& acquireReadOnly();
		//! Acquire read/write access to the particle data
		const ParticleDataArrays& acquireReadWrite();
		
		#ifdef USE_CUDA
		//! Acquire read access to the particle data on the GPU
		gpu_pdata_arrays acquireReadOnlyGPU();
		//! Acquire read/write access to the particle data on the GPU
		gpu_pdata_arrays acquireReadWriteGPU();
		
		//! Get the box for the GPU
		/*! \returns Box dimensions suitable for passing to the GPU code
		*/
		const gpu_boxsize& getBoxGPU() { return m_gpu_box; }		
		#endif
		
		//! Release the acquired data
		void release();

		//! Set the profiler to profile CPU<-->GPU memory copies
		/*! \param prof Pointer to the profiler to use. Set to NULL to deactivate profiling
		*/		 			
		void setProfiler(boost::shared_ptr<Profiler> prof) { m_prof=prof; }
	
		//! Gets the last time step particles were resorted
		/*! \returns Last time step when the particles were resorted
		 */ 
		unsigned int getLastSortedTstep() const
			{ 
			return m_last_sorted_tstep; 
			}
			
		//! Sets the last sorted time step
		/*! \param t Time step to set
		 	Anything that rearranges the order of the particles in the arrays 
		 	<em>must</em> call this method.
		 	\note This API is likely to removed before too long and replaced with
		 		signals and slots. 
		 */
		void setLastSortedTstep(unsigned int t) { m_last_sorted_tstep = t; }
		
	protected:
		BoxDim m_box;		//!< The simulation box
		void *m_data;		//!< Raw data allocated
		size_t m_nbytes;	//!< Number of bytes allocated
		unsigned int m_ntypes; //!< Number of particle types
		
		bool m_acquired;		//!< Flag to track if data has been acquired
		
		unsigned int m_last_sorted_tstep;	//!< Tracks the last timestep this particle data was sorted
		
		ParticleDataArrays m_arrays;	//!< Pointers into m_data for particle access
		ParticleDataArraysConst m_arrays_const;	//!< Pointers into m_data for const particle access
		boost::shared_ptr<Profiler> m_prof;		//!< Pointer to the profiler. NULL if there is no profiler.
		
		WallData m_wallData;	//!< Data about the walls in a simulation
		
		#ifdef USE_CUDA
		
		//! Simple type for identifying where the most up to date particle data is
		enum DataLocation
			{
			cpu,	//!< Particle data was last modified on the CPU
			cpugpu,	//!< CPU and GPU contain identical data
			gpu		//!< Particle data was last modified on the GPU
			};
		
		DataLocation m_data_location;	//!< Where the most recently modified particle data lives
		gpu_pdata_arrays m_gpu_pdata;	//!< Stores the pointers to memory on the GPU
		gpu_boxsize m_gpu_box;		//!< Mirror structure of m_box for the GPU
		float *m_d_staging;				//!< Staging array (device memory) where uninterleaved data is coped to/from.
		unsigned int m_uninterleave_pitch;	//!< Remember the pitch between x,y,z,type in the uninterleaved data
		unsigned int m_single_xarray_bytes;	//!< Remember the number of bytes allocated for a single float array
				
		//! Helper function to move data from the host to the device
		void hostToDeviceCopy();
		//! Helper function to move data from the device to the host
		void deviceToHostCopy();

		#endif
		
		//! Helper function to allocate CPU data
		void allocate(unsigned int N);
		//! Deallocates data
		void deallocate();
		//! Helper function to check that particles are in the box
		bool inBox();
	};
	
#ifdef USE_PYTHON
//! Exports the BoxDim class to python
void export_BoxDim();
//! Exports ParticleDataInitializer to python
void export_ParticleDataInitializer();
//! Exports ParticleData to python
void export_ParticleData();
#endif

#endif
