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

/*! \file ParticleData.cc
 	\brief Contains all code for BoxDim, ParticleData, and ParticleDataArrays.
 */

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <iostream>
#include <cassert>
#include <stdlib.h>
#include <stdexcept>
#include <sstream>
#include <iomanip>

using namespace std;

#include <boost/python.hpp>
using namespace boost::python;

#include "ParticleData.h"
#include "Profiler.h"
#include "AngleData.h"
#include "DihedralData.h"

#include <boost/bind.hpp>

using namespace boost::signals;
using namespace boost;

///////////////////////////////////////////////////////////////////////////
// BoxDim constructors

/*! \post All dimensions are 0.0
*/
BoxDim::BoxDim()
	{	
	xlo = xhi = ylo = yhi = zlo = zhi = 0.0;
	}

/*! \param Len Length of one side of the box
 	\post Box ranges from \c -Len/2 to \c +Len/2 in all 3 dimensions
 */
BoxDim::BoxDim(Scalar Len)
	{
	// sanity check
	assert(Len > 0);
	
	// assign values
	xlo = ylo = zlo = -Len/Scalar(2.0);
	xhi = zhi = yhi = Len/Scalar(2.0);
	}

/*! \param Len_x Length of the x dimension of the box
 	\param Len_y Length of the x dimension of the box
 	\param Len_z Length of the x dimension of the box
 */	
BoxDim::BoxDim(Scalar Len_x, Scalar Len_y, Scalar Len_z)
	{
	// sanity check
	assert(Len_x > 0 && Len_y > 0 && Len_z > 0);
	
	// assign values
	xlo = -Len_x/Scalar(2.0);
	xhi = Len_x/Scalar(2.0);
	
	ylo = -Len_y/Scalar(2.0);
	yhi = Len_y/Scalar(2.0);
	
	zlo = -Len_z/Scalar(2.0);
	zhi = Len_z/Scalar(2.0);
	}

////////////////////////////////////////////////////////////////////////////
// ParticleDataArrays constructors
/*! \post All pointers are NULL
*/
ParticleDataArrays::ParticleDataArrays() : nparticles(0), x(NULL), y(NULL), z(NULL),
	vx(NULL), vy(NULL), vz(NULL), ax(NULL), ay(NULL), az(NULL), charge(NULL), mass(NULL), diameter(NULL), ix(NULL),
	iy(NULL), iz(NULL), body(NULL), type(NULL), rtag(NULL)
	{
	}

/*! \post All pointers are NULL
*/
ParticleDataArraysConst::ParticleDataArraysConst() : nparticles(0), x(NULL), y(NULL), z(NULL),
	vx(NULL), vy(NULL), vz(NULL), ax(NULL), ay(NULL), az(NULL), charge(NULL), mass(NULL), diameter(NULL), ix(NULL),
	iy(NULL), iz(NULL), body(NULL), type(NULL), rtag(NULL)
	{
	}	
	

////////////////////////////////////////////////////////////////////////////
// ParticleData members

/*! \param N Number of particles to allocate memory for
	\param n_types Number of particle types that will exist in the data arrays
	\param box Box the particles live in
        \param n_angle_types Number of angles to create
        \param n_dihedral_types Number of dihedrals to create
        \param n_improper_types Number of impropers to create
	\param exec_conf ExecutionConfiguration to use when executing code on the GPU
	
	\post \c x,\c y,\c z,\c vx,\c vy,\c vz,\c ax,\c ay, and \c az are allocated and initialized to 0.0
	\post \c charge is allocated and initialized to a value of 0.0
	\post \c diameter is allocated and initialized to a value of 1.0
	\post \c mass is allocated and initialized to a value of 1.0
	\post \c ix, \c iy, \c iz are allocated and initialized to values of 0.0
	\post \c rtag is allocated and given the default initialization rtag[i] = i
	\post \c tag is allocated and given the default initialization tag[i] = i
	\post \c type is allocated and given the default value of type[i] = 0
	\post \c body is allocated and given the devault value of type[i] = NO_BODY
	\post Arrays are not currently acquired
	
	Type mappings assign particle types "A", "B", "C", ....
*/ 
ParticleData::ParticleData(unsigned int N, const BoxDim &box, unsigned int n_types, const ExecutionConfiguration& exec_conf)	
	: m_box(box), m_exec_conf(exec_conf), m_data(NULL), m_nbytes(0), m_ntypes(n_types), m_acquired(false)
	{
	// check the input for errors
	if (m_ntypes == 0)
		{
		cerr << endl << "***Error! Number of particle types must be greater than 0." << endl << endl;
		throw std::runtime_error("Error initializing ParticleData");
		}
		
	#ifdef ENABLE_CUDA
	m_gpu_pdata.resize(m_exec_conf.gpu.size());
	m_exec_conf.tagAll(__FILE__, __LINE__);
	#endif
	
	// allocate memory
	allocate(N);

	// sanity check
	assert(m_arrays.x != NULL && m_arrays.y != NULL && m_arrays.z != NULL);
	assert(m_arrays.vx != NULL && m_arrays.vy != NULL && m_arrays.vz != NULL);
	assert(m_arrays.ax != NULL && m_arrays.ay != NULL && m_arrays.az != NULL);
	assert(m_arrays.ix != NULL && m_arrays.iy != NULL && m_arrays.iz != NULL);
	assert(m_arrays.mass != NULL && m_arrays.diameter != NULL);
	assert(m_arrays.type != NULL && m_arrays.rtag != NULL && m_arrays.tag != NULL && m_arrays.charge != NULL);
	assert(m_arrays.body != NULL);
	
	// set default values
	for (unsigned int i = 0; i < N; i++)
		{
		m_arrays.x[i] = m_arrays.y[i] = m_arrays.z[i] = 0.0;
		m_arrays.vx[i] = m_arrays.vy[i] = m_arrays.vz[i] = 0.0;
		m_arrays.ax[i] = m_arrays.ay[i] = m_arrays.az[i] = 0.0;
		m_arrays.charge[i] = 0.0;
		m_arrays.mass[i] = 1.0;
		m_arrays.diameter[i] = 1.0;
		m_arrays.ix[i] = m_arrays.iy[i] = m_arrays.iz[i] = 0;
		
		m_arrays.body[i] = NO_BODY;
		m_arrays.type[i] = 0;
		m_arrays.rtag[i] = i;
		m_arrays.tag[i] = i;
		}
	
	// default constructed shared ptr is null as desired
	m_prof = boost::shared_ptr<Profiler>();

	// setup the type mappings
	for (unsigned int i = 0; i < m_ntypes; i++)
		{
		char name[2];
		name[0] = 'A' + i;
		name[1] = '\0';
		m_type_mapping.push_back(string(name));
		}
		
	// if this is a GPU build, initialize the graphics card mirror data structures
	#ifdef ENABLE_CUDA
	if (!m_exec_conf.gpu.empty())
		{
		hostToDeviceCopy();
		// now the data is copied to both the cpu and gpu and is unmodified, set the initial state
		m_data_location = cpugpu;
	
		// setup the box
		m_gpu_box.Lx = m_box.xhi - m_box.xlo;
		m_gpu_box.Ly = m_box.yhi - m_box.ylo;
		m_gpu_box.Lz = m_box.zhi - m_box.zlo;
		m_gpu_box.Lxinv = 1.0f / m_gpu_box.Lx;
		m_gpu_box.Lyinv = 1.0f / m_gpu_box.Ly;
		m_gpu_box.Lzinv = 1.0f / m_gpu_box.Lz;
		}
	else
		{
		m_data_location = cpu;
		}

	#endif
	}
	
/*! Calls the initializer's members to determine the number of particles, box size and then
	uses it to fill out the position and velocity data.
	\param init Initializer to use
	\param exec_conf Execution configuration to run on
*/
ParticleData::ParticleData(const ParticleDataInitializer& init, const ExecutionConfiguration& exec_conf) : m_exec_conf(exec_conf), m_data(NULL), m_nbytes(0), m_ntypes(0), m_acquired(false)
	{
	m_ntypes = init.getNumParticleTypes();
	// check the input for errors
	if (m_ntypes == 0)
		{
		cerr << endl << "***Error! Number of particle types must be greater than 0." << endl << endl;
		throw std::runtime_error("Error initializing ParticleData");
		}
		
	#ifdef ENABLE_CUDA
	m_gpu_pdata.resize(m_exec_conf.gpu.size());
	m_exec_conf.tagAll(__FILE__, __LINE__);
	#endif
	
	// allocate memory
	allocate(init.getNumParticles());
	
	// sanity check
	assert(m_arrays.x != NULL && m_arrays.y != NULL && m_arrays.z != NULL);
	assert(m_arrays.vx != NULL && m_arrays.vy != NULL && m_arrays.vz != NULL);
	assert(m_arrays.ax != NULL && m_arrays.ay != NULL && m_arrays.az != NULL);
	assert(m_arrays.ix != NULL && m_arrays.iy != NULL && m_arrays.iz != NULL);
	assert(m_arrays.mass != NULL && m_arrays.diameter != NULL);
	assert(m_arrays.type != NULL && m_arrays.rtag != NULL && m_arrays.tag != NULL && m_arrays.charge != NULL);
	assert(m_arrays.body != NULL);
	
	// set default values
	for (unsigned int i = 0; i < m_arrays.nparticles; i++)
		{
		m_arrays.x[i] = m_arrays.y[i] = m_arrays.z[i] = 0.0;
		m_arrays.vx[i] = m_arrays.vy[i] = m_arrays.vz[i] = 0.0;
		m_arrays.ax[i] = m_arrays.ay[i] = m_arrays.az[i] = 0.0;
		m_arrays.charge[i] = 0.0;
		m_arrays.mass[i] = 1.0;
		m_arrays.diameter[i] = 1.0;
		m_arrays.ix[i] = m_arrays.iy[i] = m_arrays.iz[i] = 0;
		
		m_arrays.body[i] = NO_BODY;
		m_arrays.type[i] = 0;
		m_arrays.rtag[i] = i;
		m_arrays.tag[i] = i;
		}
	
	// need to set m_data_location before any call to setBox
	#ifdef ENABLE_CUDA
	m_data_location = cpu;
	#endif
	
	setBox(init.getBox());
	init.initArrays(m_arrays);
	
	// it is an error for particles to be initialized outside of their box
	if (!inBox(false))
		{
		cerr << endl << "***Error! Not all particles were found inside the given box" << endl << endl;
		throw runtime_error("Error initializing ParticleData");
		}
		
	// assign the type mapping
	m_type_mapping	= init.getTypeMapping();
		
	// default constructed shared ptr is null as desired
	m_prof = boost::shared_ptr<Profiler>();

	// if this is a GPU build, initialize the graphics card mirror data structure
	#ifdef ENABLE_CUDA
	if (!m_exec_conf.gpu.empty())
		{
		hostToDeviceCopy();
		// now the data is copied to both the cpu and gpu and is unmodified, set the initial state
		m_data_location = cpugpu;
		}
	else
		{
		m_data_location = cpu;
		}
	#endif
	}

/*! Frees all allocated memory
 */
ParticleData::~ParticleData()
	{
	deallocate();
	}
			
/*! \return Simulation box dimensions
 */
const BoxDim & ParticleData::getBox() const
	{
	return m_box;
	}
		
/*! \param box New box to set
	\note ParticleData does NOT enforce any boundary conditions. When a new box is set,
		it is the responsibility of the caller to ensure that all particles lie within 
		the new box.
*/
void ParticleData::setBox(const BoxDim &box)
	{
	m_box = box;
	assert(inBox(true));
		
	#ifdef ENABLE_CUDA
	// setup the box
	m_gpu_box.Lx = m_box.xhi - m_box.xlo;
	m_gpu_box.Ly = m_box.yhi - m_box.ylo;
	m_gpu_box.Lz = m_box.zhi - m_box.zlo;
	m_gpu_box.Lxinv = 1.0f / m_gpu_box.Lx;
	m_gpu_box.Lyinv = 1.0f / m_gpu_box.Ly;
	m_gpu_box.Lzinv = 1.0f / m_gpu_box.Lz;
	#endif

	m_boxchange_signal();
	}

	
/*! Access to the particle data is granted only when acquired. The data may be living
	in the graphics card memory, so accesses may be expensive as they involve copying
	over the PCI-Express connection. Access should only be acquired when needed and as few 
	times as possible. 
	
	This method gives read-only access to the particle data. If the data is not to be 
	written to, this is preferred as it will avoid copying the data back to the 
	graphics card.
	
	\note There are NO sophisticated mulithreaded syncronization routines. Only one
		thread may acquire the particle data at a time, and it may not acquire it twice
		without releasing it first.
		
	\note Debug builds enforce the acquire/release pairing with asserts.
	
	\return Pointers that can be used to access the particle data on the CPU
		
	\sa release()
*/ 
const ParticleDataArraysConst & ParticleData::acquireReadOnly()
	{
	// sanity check
	assert(m_data);
	assert(!m_acquired);
	assert(inBox(true));
	m_acquired = true;
	
	#ifdef ENABLE_CUDA
	
	// this is the complicated graphics card version, need to do some work
	// switch based on the current location of the data
	switch (m_data_location)
		{
		case cpu:
			// if the data is solely on the cpu, life is easy, return the data arrays
			// and stay in the same state
			return m_arrays_const;
			break;
		case cpugpu:
			// if the data is up to date on both the cpu and gpu, life is easy, return
			// the data arrays and stay in the same state
			return m_arrays_const;
			break;
		case gpu:
			// if the data resides on the gpu, it needs to be copied back to the cpu
			// this changes to the cpugpu state since the data is now fully up to date on 
			// both
			deviceToHostCopy();
			m_data_location = cpugpu;
			return m_arrays_const;
			break;
		default:
			// anything other than the above is an undefined state!
			assert(false);
			return m_arrays_const;	
			break;				
		}

	#else
	// this is just a simple CPU implementation, no graphics card involved.
	// So, just return the data arrays
	return m_arrays_const;
	#endif
	}

/*! Acquire access to the particle data, allowing both read and write access.
	\sa acquireReadOnly
	\sa release
*/
const ParticleDataArrays & ParticleData::acquireReadWrite()
	{
	// sanity check
	assert(m_data);
	assert(!m_acquired);
	assert(inBox(true));
	m_acquired = true;
	
	#ifdef ENABLE_CUDA
	
	// this is the complicated graphics card version, need to do some work
	// switch based on the current location of the data
	switch (m_data_location)
		{
		case cpu:
			// if the data is solely on the cpu, life is easy, return the data arrays
			// and stay in the same state
			return m_arrays;
			break;
		case cpugpu:
			// if the data is up to date on both the cpu and gpu, it is about to be modified
			// on the cpu, so change states to that and then return the data
			m_data_location = cpu;
			return m_arrays;
			break;
		case gpu:
			// if the data resides on the gpu, it needs to be copied back to the cpu
			// this changes to the cpu state since the data is about to be modified 
			// on the cpu
			deviceToHostCopy();
			m_data_location = cpu;
			return m_arrays;
			break;
		default:
			// anything other than the above is an undefined state!
			assert(false);
			return m_arrays;
			break;
		}
	#else
	// this is just a simple CPU implementation, no graphics card involved.
	// So, just return the data arrays
	return m_arrays;
	#endif
	}
	
#ifdef ENABLE_CUDA

/*! Acquire access to the particle data, for read only access on the gpu.
	
 	\return Pointers to GPU device memory where the particle data can be accessed 
	\sa acquireReadOnly
	\sa release
*/
std::vector<gpu_pdata_arrays>& ParticleData::acquireReadOnlyGPU()
	{
	// sanity check
	assert(!m_acquired);
	m_acquired = true;
	
	if (m_exec_conf.gpu.size() == 0)
		{
		cerr << endl << "***Error! Reqesting GPU pdata, but no GPU in the Execution Configuration" << endl << endl;
		throw runtime_error("Error acquiring GPU data");
		}
		
	// this is the complicated graphics card version, need to do some work
	// switch based on the current location of the data
	switch (m_data_location)
		{
		case cpu:
			// if the data is on the cpu, we need to copy it over to the gpu
			hostToDeviceCopy();
			
			// now we are in the cpugpu state
			m_data_location = cpugpu;
			return m_gpu_pdata;
			break;
		case cpugpu:
			// state remains the same
			return m_gpu_pdata;
			break;
		case gpu:
			// state remains the same
			return m_gpu_pdata;
			break;
		default:
			// anything other than the above is an undefined state!
			assert(false);
			return m_gpu_pdata;	
			break;				
		}
	}

/*! Acquire access to the particle data, for read/write access on the gpu. 

 	\return Pointers to GPU device memory where the particle data can be accessed 
	\sa acquireReadOnly
	\sa release
*/
std::vector<gpu_pdata_arrays>& ParticleData::acquireReadWriteGPU()
	{
	// sanity check
	assert(!m_acquired);
	m_acquired = true;
	
	if (m_exec_conf.gpu.size() == 0)
		{
		cerr << endl << "Reqesting GPU pdata, but no GPU in the Execution Configuration" << endl << endl;
		throw runtime_error("Error acquiring GPU data");
		}
	
	// flag that we have done a read/write on the GPU
	m_readwrite_gpu = true;
	
	// this is the complicated graphics card version, need to do some work
	// switch based on the current location of the data
	switch (m_data_location)
		{
		case cpu:
			// if the data is on the cpu, we need to copy it over to the gpu and 
			hostToDeviceCopy();
						
			// now we are in the gpu state
			m_data_location = gpu;
			return m_gpu_pdata;
			break;
		case cpugpu:
			// state goes to gpu
			m_data_location = gpu;
			return m_gpu_pdata;
			break;
		case gpu:
			// state remains the same
			return m_gpu_pdata;
			break;
		default:
			// anything other than the above is an undefined state!
			assert(false);
			return m_gpu_pdata;	
			break;
		}
	}	
		

#endif
	

/*! When a caller is done with the data from either acquireReadOnly or acquireReadWrite, it
	needs to release it so that it can be acquired again further down the line. 
	
	After calling realease, the caller has no buisiness dereferencing the pointers 
	it got from acquire any more. Data may be moved around in memory when it is not 
	acquired.
	
	\warning No caller should move particles outside the simulation box. In debug builds, release()
	checks to make sure this is the case.
*/
void ParticleData::release()
	{
	// sanity checks
	assert(m_acquired);
	
	// this is just a simple CPU implementation, no graphics card involved.
	// all memory mods were direct into the pointers, so just flip the acquired bit
	m_acquired = false;
	}
	
/*! \param func Function to call when the particles are resorted
	\return Connection to manage the signal/slot connection
	Calls are performed by using boost::signals. The function passed in
	\a func will be called every time the ParticleData is notified of a particle
	sort via notifyParticleSort().
	\note If the caller class is destroyed, it needs to disconnect the signal connection
	via \b con.disconnect where \b con is the return value of this function.
*/
boost::signals::connection ParticleData::connectParticleSort(const boost::function<void ()> &func)
	{
	return m_sort_signal.connect(func);
	}

/*! \b ANY time particles are rearranged in memory, this function must be called.
	\note The call must be made after calling release()
*/
void ParticleData::notifyParticleSort()
	{
	m_sort_signal();
	}

/*! \param func Function to call when the box size changes
	\return Connection to manage the signal/slot connection
	Calls are performed by using boost::signals. The function passed in
	\a func will be called every time the the box size is changed via setBox()
	\note If the caller class is destroyed, it needs to disconnect the signal connection
	via \b con.disconnect where \b con is the return value of this function.
*/
boost::signals::connection ParticleData::connectBoxChange(const boost::function<void ()> &func)
	{
	return m_boxchange_signal.connect(func);
	}
	
/*! \param name Type name to get the index of
	\return Type index of the corresponding type name
	\note Throws an exception if the type name is not found
*/
unsigned int ParticleData::getTypeByName(const std::string &name)
	{
	assert(m_type_mapping.size() == m_ntypes);
	// search for the name
	for (unsigned int i = 0; i < m_type_mapping.size(); i++)
		{
		if (m_type_mapping[i] == name)
			return i;
		}
		
	cerr << endl << "***Error! Type " << name << " not found!" << endl;
	throw runtime_error("Error mapping type name");	
	return 0;
	}
		
/*! \param type Type index to get the name of
	\returns Type name of the requested type
	\note Type indices must range from 0 to getNTypes or this method throws an exception.
*/
std::string ParticleData::getNameByType(unsigned int type)
	{
	assert(m_type_mapping.size() == m_ntypes);
	// check for an invalid request
	if (type >= m_ntypes)
		{
		cerr << endl << "***Error! Requesting type name for non-existant type " << type << endl << endl;
		throw runtime_error("Error mapping type name");
		}
		
	// return the name
	return m_type_mapping[type];
	}

	
/*! \param N Number of particles to allocate memory for
	\pre No memory is allocated and the pointers in m_arrays point nowhere
	\post All memory is allocated and the pointers in m_arrays are set properly
	\note As per the requirements, this method is implemented to allocate all of the
		many arrays in one big chunk.
	\note For efficiency copying to/from the GPU, arrays are 256 byte aligned and x,y,z,type are 
		next to each other in memory in that order. Similarly, vx,vy,vz are next to 
		each other and ax,ay,az are next to each other. The pitch between starts of 
		these arrays is stored in m_uninterleave_pitch. 
*/ 
void ParticleData::allocate(unsigned int N)
	{
	// check the input
	if (N == 0)
		{
		cerr << endl << "***Error! ParticleData is being asked to allocate 0 particles.... this makes no sense whatsoever" << endl << endl;
		throw runtime_error("Error allocating ParticleData");
		}
	
	m_nbytes = 0;
	
	/////////////////////////////////////////////////////
	// Count bytes needed by the main data structures
	#ifdef ENABLE_CUDA
	// 256 byte aligned version for GPU
	
	// start by adding up the number of bytes needed for the Scalar arrays, rounding up by 16
	unsigned int single_xarray_bytes = sizeof(Scalar) * N;
	if ((single_xarray_bytes & 255) != 0)
		single_xarray_bytes += 256 - (single_xarray_bytes & 255);
	
	// total all bytes from scalar arrays
	m_nbytes += single_xarray_bytes * 12;
	
	// now add up the number of bytes for the int arrays, rounding up to 16 bytes
	unsigned int single_iarray_bytes = sizeof(unsigned int) * N;
	if ((single_iarray_bytes & 255) != 0)
		single_iarray_bytes += 256 - (single_iarray_bytes & 255);
	
	m_nbytes += single_iarray_bytes * 7;
	
	#else
	
	// allocation on the CPU
	// start by adding up the number of bytes needed for the Scalar arrays
	unsigned int single_xarray_bytes = sizeof(Scalar) * N;
	
	// total all bytes from scalar arrays
	m_nbytes += single_xarray_bytes * 12;
	
	// now add up the number of bytes for the int arrays
	unsigned int single_iarray_bytes = sizeof(unsigned int) * N;
	
	m_nbytes += single_iarray_bytes * 7;
	#endif
	
	//////////////////////////////////////////////////////
	// allocate the memory on the CPU, use pinned memory if compiling for the GPU
	#ifdef ENABLE_CUDA
	m_exec_conf.tagAll(__FILE__, __LINE__);
	if (!m_exec_conf.gpu.empty())
		{
		m_exec_conf.gpu[0]->call(bind(cudaHostAlloc, &m_data, m_nbytes, cudaHostAllocPortable));
		}
	else
		{
		m_data = malloc(m_nbytes);
		}
	#else
	m_data = malloc(m_nbytes);
	#endif
	
	// Now that m_data is allocated, we need to play some pointer games to assign
	// the x,y,z, etc... pointers
	char *cur_byte = (char *)m_data;
	m_arrays_const.x = m_arrays.x = (Scalar *)cur_byte;  cur_byte += single_xarray_bytes;
	m_arrays_const.y = m_arrays.y = (Scalar *)cur_byte;  cur_byte += single_xarray_bytes;
	m_arrays_const.z = m_arrays.z = (Scalar *)cur_byte;  cur_byte += single_xarray_bytes;
	m_arrays_const.type = m_arrays.type = (unsigned int *)cur_byte;  cur_byte += single_iarray_bytes;
	m_arrays_const.vx = m_arrays.vx = (Scalar *)cur_byte;  cur_byte += single_xarray_bytes;
	m_arrays_const.vy = m_arrays.vy = (Scalar *)cur_byte;  cur_byte += single_xarray_bytes;
	m_arrays_const.vz = m_arrays.vz = (Scalar *)cur_byte;  cur_byte += single_xarray_bytes;	
	m_arrays_const.ax = m_arrays.ax = (Scalar *)cur_byte;  cur_byte += single_xarray_bytes;
	m_arrays_const.ay = m_arrays.ay = (Scalar *)cur_byte;  cur_byte += single_xarray_bytes;
	m_arrays_const.az = m_arrays.az = (Scalar *)cur_byte;  cur_byte += single_xarray_bytes;
	m_arrays_const.charge = m_arrays.charge = (Scalar *)cur_byte;  cur_byte += single_xarray_bytes;
	m_arrays_const.mass = m_arrays.mass = (Scalar *)cur_byte;  cur_byte += single_xarray_bytes;
	m_arrays_const.diameter = m_arrays.diameter = (Scalar *)cur_byte;  cur_byte += single_xarray_bytes;
	m_arrays_const.ix = m_arrays.ix = (int *)cur_byte;  cur_byte += single_iarray_bytes;
	m_arrays_const.iy = m_arrays.iy = (int *)cur_byte;  cur_byte += single_iarray_bytes;
	m_arrays_const.iz = m_arrays.iz = (int *)cur_byte;  cur_byte += single_iarray_bytes;
	m_arrays_const.rtag = m_arrays.rtag = (unsigned int *)cur_byte;  cur_byte += single_iarray_bytes;
	m_arrays_const.tag = m_arrays.tag = (unsigned int *)cur_byte;  cur_byte += single_iarray_bytes;
	m_arrays_const.body = m_arrays.body = (unsigned int *)cur_byte;  cur_byte += single_iarray_bytes;
	m_arrays_const.nparticles = m_arrays.nparticles = N;
	
	// sanity check
	assert(cur_byte == ((char*)m_data) + m_nbytes);
	
	#ifdef ENABLE_CUDA
	//////////////////////////////////////////////////////////
	// allocate memory on each GPU
	
	// calculate pitch values for swizzling memory on the GPU
	m_uninterleave_pitch = single_xarray_bytes/4;
	m_single_xarray_bytes = single_xarray_bytes;
	
	// setup list of staging arrays
	m_d_staging.resize(m_exec_conf.gpu.size());
	
	for (unsigned int cur_gpu = 0; cur_gpu < m_exec_conf.gpu.size(); cur_gpu++)
		{
		m_gpu_pdata[cur_gpu].N = N;
		m_exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
	
		m_exec_conf.gpu[cur_gpu]->call(bind(cudaMalloc, (void **)((void *)&m_gpu_pdata[cur_gpu].pos), single_xarray_bytes * 4));
		m_exec_conf.gpu[cur_gpu]->call(bind(cudaMalloc, (void **)((void *)&m_gpu_pdata[cur_gpu].vel), single_xarray_bytes * 4) );
		m_exec_conf.gpu[cur_gpu]->call(bind(cudaMalloc, (void **)((void *)&m_gpu_pdata[cur_gpu].accel), single_xarray_bytes * 4) );
		m_exec_conf.gpu[cur_gpu]->call(bind(cudaMalloc, (void **)((void *)&m_gpu_pdata[cur_gpu].charge), single_xarray_bytes) );
		m_exec_conf.gpu[cur_gpu]->call(bind(cudaMalloc, (void **)((void *)&m_gpu_pdata[cur_gpu].mass), single_xarray_bytes) );
		m_exec_conf.gpu[cur_gpu]->call(bind(cudaMalloc, (void **)((void *)&m_gpu_pdata[cur_gpu].diameter), single_xarray_bytes) );
		m_exec_conf.gpu[cur_gpu]->call(bind(cudaMalloc, (void **)((void *)&m_gpu_pdata[cur_gpu].image), single_xarray_bytes * 4) );
		m_exec_conf.gpu[cur_gpu]->call(bind(cudaMalloc, (void **)((void *)&m_gpu_pdata[cur_gpu].tag), sizeof(unsigned int)*N) );
		m_exec_conf.gpu[cur_gpu]->call(bind(cudaMalloc, (void **)((void *)&m_gpu_pdata[cur_gpu].rtag), sizeof(unsigned int)*N));
		m_exec_conf.gpu[cur_gpu]->call(bind(cudaMalloc, (void **)((void *)&m_gpu_pdata[cur_gpu].body), sizeof(unsigned int)*N));
		
		// allocate temporary holding area for uninterleaved data
		m_exec_conf.gpu[cur_gpu]->call(bind(cudaMalloc, (void **)((void *)&m_d_staging[cur_gpu]), single_xarray_bytes*4));
		}
		
	// allocate host staging location
	if (!m_exec_conf.gpu.empty())
		m_exec_conf.gpu[0]->call(bind(cudaHostAlloc, (void **)((void *)&m_h_staging), sizeof(float4)*N, cudaHostAllocPortable));
		
	// assign which particles are local to which GPU
	if (!m_exec_conf.gpu.empty())
		{
		// special case: only 1 GPU
		if (m_exec_conf.gpu.size() == 1)
			{
			m_gpu_pdata[0].local_beg = 0;
			m_gpu_pdata[0].local_num = N;
			}
		else
			{
			// we need at least as many particles as GPUs or bad things will happen
			if (N < m_exec_conf.gpu.size()*32)
				{
				cerr << endl << "***Error! Allocating fewer than 32 particles per GPU" << endl << endl;
				throw runtime_error("Error allocating ParticleData");
				}
			
			int particles_per_gpu = N / (int)m_exec_conf.gpu.size();
		
			// round particles per gpu to a multiple of 32 for coalescing
			if ((particles_per_gpu & 31) != 0)
				particles_per_gpu += 32 - (particles_per_gpu & 31);
			
			// fill out first GPU
			int start = 0;
			m_gpu_pdata[0].local_beg = start;
			m_gpu_pdata[0].local_num = particles_per_gpu;
			start += particles_per_gpu;
			cout << "Notice: GPU " << 0 << " assigned particles " << m_gpu_pdata[0].local_beg << "-" << m_gpu_pdata[0].local_beg + m_gpu_pdata[0].local_num - 1 << endl;
			
			// middle GPUs
			for (unsigned int cur_gpu = 1; cur_gpu < m_exec_conf.gpu.size()-1; cur_gpu++)
				{
				m_gpu_pdata[cur_gpu].local_beg = start;
				m_gpu_pdata[cur_gpu].local_num = particles_per_gpu;
				start += particles_per_gpu;
				cout << "Notice: GPU " << cur_gpu << " assigned particles " << m_gpu_pdata[cur_gpu].local_beg << "-" << m_gpu_pdata[cur_gpu].local_beg + m_gpu_pdata[cur_gpu].local_num - 1<< endl;
				}
			
			// last gpu
			int cur_gpu = (int)m_exec_conf.gpu.size()-1;
			m_gpu_pdata[cur_gpu].local_beg = start;
			m_gpu_pdata[cur_gpu].local_num = N - start;
			cout << "Notice: GPU " << cur_gpu << " assigned particles " << m_gpu_pdata[cur_gpu].local_beg << "-" << m_gpu_pdata[cur_gpu].local_beg + m_gpu_pdata[cur_gpu].local_num - 1<< endl;
			}
		}
		
	#endif
	}
	
/*! \pre Memory has been allocated
	\post Memory is deallocated and pointers are set to NULL
*/
void ParticleData::deallocate()
	{
	assert(m_data);
		
	// free the data
	#ifdef ENABLE_CUDA
	if (!m_exec_conf.gpu.empty())
		{	
		m_exec_conf.gpu[0]->call(bind(cudaFreeHost, m_data));
		m_exec_conf.gpu[0]->call(bind(cudaFreeHost, m_h_staging));
		
		for (unsigned int cur_gpu = 0; cur_gpu < m_exec_conf.gpu.size(); cur_gpu++)
			{
			m_exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
			m_exec_conf.gpu[cur_gpu]->call(bind(cudaFree, m_gpu_pdata[cur_gpu].pos));
			m_exec_conf.gpu[cur_gpu]->call(bind(cudaFree, m_gpu_pdata[cur_gpu].vel));
			m_exec_conf.gpu[cur_gpu]->call(bind(cudaFree, m_gpu_pdata[cur_gpu].accel));
			m_exec_conf.gpu[cur_gpu]->call(bind(cudaFree, m_gpu_pdata[cur_gpu].charge));
			m_exec_conf.gpu[cur_gpu]->call(bind(cudaFree, m_gpu_pdata[cur_gpu].mass));
			m_exec_conf.gpu[cur_gpu]->call(bind(cudaFree, m_gpu_pdata[cur_gpu].diameter));
			m_exec_conf.gpu[cur_gpu]->call(bind(cudaFree, m_gpu_pdata[cur_gpu].image));
			m_exec_conf.gpu[cur_gpu]->call(bind(cudaFree, m_gpu_pdata[cur_gpu].tag));
			m_exec_conf.gpu[cur_gpu]->call(bind(cudaFree, m_gpu_pdata[cur_gpu].rtag));
			m_exec_conf.gpu[cur_gpu]->call(bind(cudaFree, m_gpu_pdata[cur_gpu].body));
			
			m_exec_conf.gpu[cur_gpu]->call(bind(cudaFree, m_d_staging[cur_gpu]));
			
			// zero pointers
			m_gpu_pdata[cur_gpu].pos = NULL;
			m_gpu_pdata[cur_gpu].vel = NULL;
			m_gpu_pdata[cur_gpu].accel = NULL;
			m_gpu_pdata[cur_gpu].charge = NULL;
			m_gpu_pdata[cur_gpu].mass = NULL;
			m_gpu_pdata[cur_gpu].diameter = NULL;
			m_gpu_pdata[cur_gpu].image = NULL;
			m_gpu_pdata[cur_gpu].tag = NULL;
			m_gpu_pdata[cur_gpu].rtag = NULL;
			m_gpu_pdata[cur_gpu].body = NULL;
			m_d_staging[cur_gpu] = NULL;
			}
		}
	else
		{
		free(m_data);
		}
	
	#else
	free(m_data);
	#endif
	
	// zero the pointers
	m_data = 0;
	m_arrays_const.x = m_arrays.x = 0;
	m_arrays_const.y = m_arrays.y = 0;
	m_arrays_const.z = m_arrays.z = 0;
	m_arrays_const.vx = m_arrays.vx = 0;
	m_arrays_const.vy = m_arrays.vy = 0;
	m_arrays_const.vz = m_arrays.vz = 0;	
	m_arrays_const.ax = m_arrays.ax = 0;
	m_arrays_const.ay = m_arrays.ay = 0;
	m_arrays_const.az = m_arrays.az = 0;	
	m_arrays_const.charge = m_arrays.charge = 0;	
	m_arrays_const.mass = m_arrays.mass = 0;	
	m_arrays_const.diameter = m_arrays.diameter = 0;	
	m_arrays_const.ix = m_arrays.ix = 0;	
	m_arrays_const.iy = m_arrays.iy = 0;
	m_arrays_const.iz = m_arrays.iz = 0;
	m_arrays_const.type = m_arrays.type = 0;
	m_arrays_const.rtag = m_arrays.rtag = 0;
	m_arrays_const.tag = m_arrays.tag = 0;
	m_arrays_const.body = m_arrays.body = 0;
	}
	
/*! \param need_aquire set to true if the inBox check should aquire the data before checking
	\return true If and only if all particles are in the simulation box
	\note This function is only called in debug builds
*/
bool ParticleData::inBox(bool need_aquire)
	{
	#ifdef ENABLE_CUDA
	if (need_aquire && m_data_location == gpu)
		deviceToHostCopy();
	#endif
	for (unsigned int i = 0; i < m_arrays.nparticles; i++)
		{
		if (m_arrays.x[i] < m_box.xlo-Scalar(1e-5) || m_arrays.x[i] > m_box.xhi+Scalar(1e-5))
			{
			cout << "pos " << i << ":" << setprecision(12) << m_arrays.x[i] << " " << m_arrays.y[i] << " " << m_arrays.z[i] << endl;
			cout << "lo: " << m_box.xlo << " " << m_box.ylo << " " << m_box.zlo << endl;
			cout << "hi: " << m_box.xhi << " " << m_box.yhi << " " << m_box.zhi << endl;
			return false;
			}
		if (m_arrays.y[i] < m_box.ylo-Scalar(1e-5) || m_arrays.y[i] > m_box.yhi+Scalar(1e-5))
			{
			cout << "pos " << i << ":" << setprecision(12) << m_arrays.x[i] << " " << m_arrays.y[i] << " " << m_arrays.z[i] << endl;
			cout << "lo: " << m_box.xlo << " " << m_box.ylo << " " << m_box.zlo << endl;
			cout << "hi: " << m_box.xhi << " " << m_box.yhi << " " << m_box.zhi << endl;
			return false;
			}
		if (m_arrays.z[i] < m_box.zlo-Scalar(1e-5) || m_arrays.z[i] > m_box.zhi+Scalar(1e-5))
			{
			cout << "pos " << i << ":" << setprecision(12) << m_arrays.x[i] << " " << m_arrays.y[i] << " " << m_arrays.z[i] << endl;
			cout << "lo: " << m_box.xlo << " " << m_box.ylo << " " << m_box.zlo << endl;
			cout << "hi: " << m_box.xhi << " " << m_box.yhi << " " << m_box.zhi << endl;			
			return false;
			}
		}
	return true;
	}

#ifdef ENABLE_CUDA
/*! \post Particle data is copied from the GPU to the CPU
*/
void ParticleData::hostToDeviceCopy()
	{
	// we should never be called unless 1 or more gpus is in the exec conf... verify
	assert(m_exec_conf.gpu.size() >= 1);
	
	// commenting profiling: enable when benchmarking suspected slow portions of the code. This isn't needed all the time
	if (m_prof) m_prof->push(m_exec_conf, "PDATA C2G");
		
	const int N = m_arrays.nparticles;	
	
	// because of the way memory was allocated, we can copy lots of pieces in huge chunks
	// the number of bytes copied and where depend highly on the order of allocation 
	// inside allocate
	
	for (unsigned int cur_gpu = 0; cur_gpu < m_exec_conf.gpu.size(); cur_gpu++)
		{
		m_exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
		// copy position data to the staging area
		m_exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, m_d_staging[cur_gpu], m_arrays.x, m_single_xarray_bytes*4, cudaMemcpyHostToDevice));
		// interleave the data
		m_exec_conf.gpu[cur_gpu]->call(bind(gpu_interleave_float4, m_gpu_pdata[cur_gpu].pos, m_d_staging[cur_gpu], N, m_uninterleave_pitch));
	
		m_exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
		// copy velocity data to the staging area
		m_exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, m_d_staging[cur_gpu], m_arrays.vx, m_single_xarray_bytes*3, cudaMemcpyHostToDevice));
		//interleave the data
		m_exec_conf.gpu[cur_gpu]->call(bind(gpu_interleave_float4, m_gpu_pdata[cur_gpu].vel, m_d_staging[cur_gpu], N, m_uninterleave_pitch));
		
		m_exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
		// copy acceleration data to the staging area
		m_exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, m_d_staging[cur_gpu], m_arrays.ax, m_single_xarray_bytes*3, cudaMemcpyHostToDevice));
		//interleave the data
		m_exec_conf.gpu[cur_gpu]->call(bind(gpu_interleave_float4, m_gpu_pdata[cur_gpu].accel, m_d_staging[cur_gpu], N, m_uninterleave_pitch));
		
		m_exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
		// copy charge
		m_exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, m_gpu_pdata[cur_gpu].charge, m_arrays.charge, m_single_xarray_bytes, cudaMemcpyHostToDevice));
	
		m_exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
		// copy mass
		m_exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, m_gpu_pdata[cur_gpu].mass, m_arrays.mass, m_single_xarray_bytes, cudaMemcpyHostToDevice));

		m_exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
		// copy diameter
		m_exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, m_gpu_pdata[cur_gpu].diameter, m_arrays.diameter, m_single_xarray_bytes, cudaMemcpyHostToDevice));
		
		m_exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
		// copy image
		m_exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, m_d_staging[cur_gpu], m_arrays.ix, m_single_xarray_bytes*3, cudaMemcpyHostToDevice));
		//interleave the data
		m_exec_conf.gpu[cur_gpu]->call(bind(gpu_interleave_float4, (float4*)m_gpu_pdata[cur_gpu].image, m_d_staging[cur_gpu], N, m_uninterleave_pitch));

		m_exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
		// copy the tag and rtag data
		m_exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, m_gpu_pdata[cur_gpu].tag, m_arrays.tag, sizeof(unsigned int)*N, cudaMemcpyHostToDevice)); 
		m_exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, m_gpu_pdata[cur_gpu].rtag, m_arrays.rtag, sizeof(unsigned int)*N, cudaMemcpyHostToDevice));
		m_exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, m_gpu_pdata[cur_gpu].body, m_arrays.body, sizeof(unsigned int)*N, cudaMemcpyHostToDevice));
		}
	
	if (m_prof) m_prof->pop(m_exec_conf, 0, m_exec_conf.gpu.size() * m_single_xarray_bytes*4 + m_single_xarray_bytes*3*2 + sizeof(unsigned int)*N * 3);
	}

//! Basic union for coverting ints <-> floats
union floatint
	{
	float f;		//!< float to read/write
	int i;	//!< int to read/write at the same memory location
	};

/*! Particle data is copied from the GPU to the CPU
*/
void ParticleData::deviceToHostCopy()
	{
	// commenting profiling: enable when benchmarking suspected slow portions of the code. This isn't needed all the time
	if (m_prof) m_prof->push(m_exec_conf, "PDATA G2C");
	const int N = m_arrays.nparticles;
	
	// we should never be called unless 1 or more gpus is in the exec conf... verify
	assert(m_exec_conf.gpu.size() >= 1);

	// because of the way memory was allocated, we can copy lots of pieces in huge chunks
	// the number of bytes copied and where depend highly on the order of allocation 
	// inside allocate

	for (unsigned int cur_gpu = 0; cur_gpu < m_exec_conf.gpu.size(); cur_gpu++)
		{
		m_exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
		int local_beg = m_gpu_pdata[cur_gpu].local_beg;
		int local_num = m_gpu_pdata[cur_gpu].local_num;
		
		m_exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
		// copy position
		m_exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, m_h_staging, m_gpu_pdata[cur_gpu].pos + local_beg, local_num*sizeof(float4), cudaMemcpyDeviceToHost));
		// fill out position/type
		for (int i = 0; i < local_num; i++)
			{
			m_arrays.x[local_beg + i] = m_h_staging[i].x;
			m_arrays.y[local_beg + i] = m_h_staging[i].y;
			m_arrays.z[local_beg + i] = m_h_staging[i].z;
			floatint fi;
			fi.f = m_h_staging[i].w;
			m_arrays.type[local_beg + i] = (unsigned int)fi.i;
			}
			
		m_exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
		// copy velocity
		m_exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, m_h_staging, m_gpu_pdata[cur_gpu].vel + local_beg, local_num*sizeof(float4), cudaMemcpyDeviceToHost));
		// fill out velocity
		for (int i = 0; i < local_num; i++)
			{
			m_arrays.vx[local_beg + i] = m_h_staging[i].x;
			m_arrays.vy[local_beg + i] = m_h_staging[i].y;
			m_arrays.vz[local_beg + i] = m_h_staging[i].z;
			}
		
		m_exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
		// copy acceleration
		m_exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, m_h_staging, m_gpu_pdata[cur_gpu].accel + local_beg, local_num*sizeof(float4), cudaMemcpyDeviceToHost));
		// fill out acceleartion
		for (int i = 0; i < local_num; i++)
			{
			m_arrays.ax[local_beg + i] = m_h_staging[i].x;
			m_arrays.ay[local_beg + i] = m_h_staging[i].y;
			m_arrays.az[local_beg + i] = m_h_staging[i].z;
			}

		m_exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
		// copy charge
		m_exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, m_arrays.charge + local_beg, m_gpu_pdata[cur_gpu].charge + local_beg, local_num*sizeof(float), cudaMemcpyDeviceToHost));
	
		m_exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
		// copy mass
		m_exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, m_arrays.mass + local_beg, m_gpu_pdata[cur_gpu].mass + local_beg, local_num*sizeof(float), cudaMemcpyDeviceToHost));
		
		m_exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
		// copy diameter
		m_exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, m_arrays.diameter + local_beg, m_gpu_pdata[cur_gpu].diameter + local_beg, local_num*sizeof(float), cudaMemcpyDeviceToHost));

		m_exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
		// copy image
		m_exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, m_h_staging, m_gpu_pdata[cur_gpu].image + local_beg, local_num*sizeof(uint4), cudaMemcpyDeviceToHost));
		// fill out position/type
		for (int i = 0; i < local_num; i++)
			{
			floatint fi;
			fi.f = m_h_staging[i].x;
			m_arrays.ix[local_beg + i] = fi.i;
			
			fi.f = m_h_staging[i].y;			
			m_arrays.iy[local_beg + i] = fi.i;
			
			fi.f = m_h_staging[i].z;
			m_arrays.iz[local_beg + i] = fi.i;
			}
	
		m_exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
		// copy the tag and rtag data
		m_exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, m_arrays.tag + local_beg, m_gpu_pdata[cur_gpu].tag + local_beg, sizeof(unsigned int)*local_num, cudaMemcpyDeviceToHost)); 
		m_exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, m_arrays.rtag + local_beg, m_gpu_pdata[cur_gpu].rtag + local_beg, sizeof(unsigned int)*local_num, cudaMemcpyDeviceToHost));
		m_exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, m_arrays.body + local_beg, m_gpu_pdata[cur_gpu].body + local_beg, sizeof(unsigned int)*local_num, cudaMemcpyDeviceToHost));
		}

	if (m_prof) m_prof->pop(m_exec_conf, 0, m_single_xarray_bytes*4 + m_single_xarray_bytes*3*2 + sizeof(unsigned int)*N * 3);
	}
	
unsigned int ParticleData::getLocalBeg(unsigned int gpu)
	{
	assert(gpu < m_exec_conf.gpu.size());
	return m_gpu_pdata[gpu].local_beg;
	}

unsigned int ParticleData::getLocalNum(unsigned int gpu)
	{
	assert(gpu < m_exec_conf.gpu.size());
	return m_gpu_pdata[gpu].local_num;
	}
	
void ParticleData::communicatePosition()
	{
	assert(!m_acquired);
	assert(m_exec_conf.gpu.size() != 0);
	
	// if there is only one GPU, we can skip this
	if (m_exec_conf.gpu.size() == 1)
		return;
	
	if (m_prof) m_prof->push(m_exec_conf, "PDATA Communicate");
	
	// count the number of bytes transferred
	unsigned int num_bytes = 0;
	
	// copy position data from all GPUs to the staging area
	for (unsigned int cur_gpu = 0; cur_gpu < m_exec_conf.gpu.size(); cur_gpu++)
		{
		m_exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
		int local_beg = m_gpu_pdata[cur_gpu].local_beg;
		int local_num = m_gpu_pdata[cur_gpu].local_num;
		
		m_exec_conf.gpu[cur_gpu]->callAsync(bind(cudaMemcpy, m_h_staging+local_beg, m_gpu_pdata[cur_gpu].pos + local_beg, local_num*sizeof(float4), cudaMemcpyDeviceToHost));
		
		num_bytes += local_num*sizeof(float4);
		}
		
	m_exec_conf.syncAll();
		
	// copy the full data back to all GPUs
	for (unsigned int cur_gpu = 0; cur_gpu < m_exec_conf.gpu.size(); cur_gpu++)
		{
		m_exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
		unsigned int local_beg = m_gpu_pdata[cur_gpu].local_beg;
		unsigned int local_num = m_gpu_pdata[cur_gpu].local_num;
		unsigned int local_end = local_beg + local_num;
		
		if (local_beg != 0)
			{
			m_exec_conf.gpu[cur_gpu]->callAsync(bind(cudaMemcpy, m_gpu_pdata[cur_gpu].pos, m_h_staging, (local_beg)*sizeof(float4), cudaMemcpyHostToDevice));
			
			num_bytes += (local_beg)*sizeof(float4);
			}
			
		if (local_end != getN())
			{
			m_exec_conf.gpu[cur_gpu]->callAsync(bind(cudaMemcpy, m_gpu_pdata[cur_gpu].pos + local_end, m_h_staging + local_end, (getN() - local_end)*sizeof(float4), cudaMemcpyHostToDevice));
			
			num_bytes += (getN() - local_end)*sizeof(float4);
			}
		}
		
	m_exec_conf.syncAll();
		
	if (m_prof) m_prof->pop(m_exec_conf, 0, num_bytes);
	}
	
#endif

//! Helper for python __str__ for BoxDim
/*! Formats the box dim into a nice string
	\param box Box to format
*/
string print_boxdim(BoxDim *box)
	{
	assert(box);
	// turn the box dim into a nicely formatted string
	ostringstream s;
	s << "x: (" << box->xlo << "," << box->xhi << ") / y: (" << box->ylo << "," << box->yhi << ") / z: ("
		<< box->zlo << "," << box->zhi << ")";
	return s.str();
	}

void export_BoxDim()
	{
	class_<BoxDim>("BoxDim")
		.def(init<Scalar>())
		.def(init<Scalar, Scalar, Scalar>())
		.def_readwrite("xlo", &BoxDim::xlo)
		.def_readwrite("xhi", &BoxDim::xhi)
		.def_readwrite("ylo", &BoxDim::ylo)
		.def_readwrite("yhi", &BoxDim::yhi)	
		.def_readwrite("zlo", &BoxDim::zlo)
		.def_readwrite("zhi", &BoxDim::zhi)
		.def("__str__", &print_boxdim)
		;
	}

//! Wrapper class needed for exposing virtual functions to python
class ParticleDataInitializerWrap : public ParticleDataInitializer, public wrapper<ParticleDataInitializer>
	{
	public:
		//! Calls the overidden ParticleDataInitializer::getNumParticles()
		unsigned int getNumParticles() const
			{
			return this->get_override("getNumParticles")();
			}
		
		//! Calls the overidden ParticleDataInitializer::getNumParticleTypes()
		unsigned int getNumParticleTypes() const
			{
			return this->get_override("getNumParticleTypes")();
			}
			
		//! Calls the overidden ParticleDataInitializer::getBox()
		BoxDim getBox() const
			{
			return this->get_override("getBox")();
			}
			
		//! Calls the overidden ParticleDataInitialzier::initArrays()
		/*! \param pdata Arrays data structure to pass on to the overridden method
		*/
		void initArrays(const ParticleDataArrays &pdata) const
			{
			this->get_override("initArrays")(pdata);
			}
			
		//! Calls the overidden ParticleDataInitializer::getTypeMapping()
		std::vector<std::string> getTypeMapping() const
			{
			return this->get_override("getTypeMapping")();
			}
	};
	

void export_ParticleDataInitializer()
	{
	class_<ParticleDataInitializerWrap, boost::noncopyable>("ParticleDataInitializer")
		.def("getNumParticles", pure_virtual(&ParticleDataInitializer::getNumParticles))
		.def("getNumParticleTypes", pure_virtual(&ParticleDataInitializer::getNumParticleTypes))
		.def("getBox", pure_virtual(&ParticleDataInitializer::getBox))
		.def("initArrays", pure_virtual(&ParticleDataInitializer::initArrays)) 
		//^^-- needs a python definition of ParticleDataArrays to be of any use
		;
	}
	

//! Helper for python __str__ for ParticleData
/*! Gives a synopsis of a ParticleData in a string
	\param pdata Particle data to format parameters from
*/
string print_ParticleData(ParticleData *pdata)
	{
	assert(pdata);
	ostringstream s;
	s << "ParticleData: " << pdata->getN() << " particles";
	return s.str();
	}

void export_ParticleData()
	{
	class_<ParticleData, boost::shared_ptr<ParticleData>, boost::noncopyable>("ParticleData", init<unsigned int, const BoxDim&, unsigned int>())
		.def(init<const ParticleDataInitializer&>())
		.def(init<const ParticleDataInitializer&, const ExecutionConfiguration&>())
		.def(init<unsigned int, const BoxDim&, unsigned int, const ExecutionConfiguration&>()) 
		.def("getBox", &ParticleData::getBox, return_value_policy<copy_const_reference>())
		.def("setBox", &ParticleData::setBox)
		.def("getN", &ParticleData::getN)
		.def("getNTypes", &ParticleData::getNTypes)
		.def("getMaximumDiameter", &ParticleData::getMaxDiameter)		
		.def("getNameByType", &ParticleData::getNameByType)
		.def("getTypeByName", &ParticleData::getTypeByName)
		.def("acquireReadOnly", &ParticleData::acquireReadOnly, return_value_policy<copy_const_reference>())
		.def("acquireReadWrite", &ParticleData::acquireReadWrite, return_value_policy<copy_const_reference>())
		.def("release", &ParticleData::release)
		.def("setProfiler", &ParticleData::setProfiler)
		.def("getExecConf", &ParticleData::getExecConf, return_internal_reference<>())
		.def("__str__", &print_ParticleData)
		;
	}

#ifdef WIN32
#pragma warning( pop )
#endif
