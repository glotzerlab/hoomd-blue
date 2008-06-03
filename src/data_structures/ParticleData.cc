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

/*! \file ParticleData.cc
 	\brief Contains all code for BoxDim, ParticleData, and ParticleDataArrays.
 */

#include <iostream>
#include <cassert>
#include <stdlib.h>
#include <stdexcept>
#include <sstream>
#include <iomanip>

using namespace std;

#ifdef USE_PYTHON
#include <boost/python.hpp>
using namespace boost::python;
#endif

#include "ParticleData.h"
#include "WallData.h"

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
	vx(NULL), vy(NULL), vz(NULL), ax(NULL), ay(NULL), az(NULL), charge(NULL), type(NULL), rtag(NULL)
	{
	}

/*! \post All pointers are NULL
*/
ParticleDataArraysConst::ParticleDataArraysConst() : nparticles(0), x(NULL), y(NULL), z(NULL),
	vx(NULL), vy(NULL), vz(NULL), ax(NULL), ay(NULL), az(NULL), charge(NULL), type(NULL), rtag(NULL)
	{
	}	
	

////////////////////////////////////////////////////////////////////////////
// ParticleData members

/*! \param N Number of particles to allocate memory for
	\param n_types Number of particle types that will exist in the data arrays
	\param box Box the particles live in
	\post \c x,\c y,\c z,\c vx,\c vy,\c vz,\c ax,\c ay, and \c az are allocated and initialized to 0.0
	\post \c rtag is allocated and given the default initialization rtag[i] = i
	\post \c tag is allocated and given the default initialization tag[i] = i
	\post \c type is allocated and given the default value of type[i] = 0
	\post Arrays are not currently acquired
*/ 
ParticleData::ParticleData(unsigned int N, const BoxDim &box, unsigned int n_types, ExecutionConfiguration exec_conf)	
	: m_box(box), m_exec_conf(exec_conf), m_data(NULL), m_nbytes(0), m_ntypes(n_types), m_acquired(false)
	{
	// check the input for errors
	if (m_ntypes == 0)
		{
		throw runtime_error("Number of particle types must be greater than 0.");
		}
		
	#ifdef USE_CUDA
	if (m_exec_conf.gpu.size() != 0 && m_exec_conf.gpu.size() > 1)
		{
		cout << "More than one GPU is not currently supported";
		throw std::runtime_error("Error initializing ParticleData");
		}
	#endif
	
	// allocate memory
	allocate(N);

	// sanity check
	assert(m_arrays.x != NULL && m_arrays.y != NULL && m_arrays.z != NULL);
	assert(m_arrays.vx != NULL && m_arrays.vy != NULL && m_arrays.vz != NULL);
	assert(m_arrays.ax != NULL && m_arrays.ay != NULL && m_arrays.az != NULL);
	assert(m_arrays.type != NULL && m_arrays.rtag != NULL && m_arrays.tag != NULL && m_arrays.charge != NULL);
	
	// set default values
	for (unsigned int i = 0; i < N; i++)
		{
		m_arrays.x[i] = m_arrays.y[i] = m_arrays.z[i] = 0.0;
		m_arrays.vx[i] = m_arrays.vy[i] = m_arrays.vz[i] = 0.0;
		m_arrays.ax[i] = m_arrays.ay[i] = m_arrays.az[i] = 0.0;
		m_arrays.charge[i] = 0.0;
		m_arrays.type[i] = 0;
		m_arrays.rtag[i] = i;
		m_arrays.tag[i] = i;
		}
	
	// default constructed shared ptr is null as desired
	m_prof = boost::shared_ptr<Profiler>();

	// allocate walls
	m_wallData = shared_ptr<WallData>(new WallData());

	// if this is a GPU build, initialize the graphics card mirror data structures
	#ifdef USE_CUDA
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
*/
ParticleData::ParticleData(const ParticleDataInitializer& init, ExecutionConfiguration exec_conf) : m_exec_conf(exec_conf), m_data(NULL), m_nbytes(0), m_ntypes(0), m_acquired(false)
	{
	m_ntypes = init.getNumParticleTypes();
	// check the input for errors
	if (m_ntypes == 0)
		{
		throw runtime_error("Number of particle types must be greater than 0.");
		}
		
	#ifdef USE_CUDA
	if (m_exec_conf.gpu.size() != 0 && m_exec_conf.gpu.size() > 1)
		{
		cout << "More than one GPU is not currently supported";
		throw std::runtime_error("Error initializing ParticleData");
		}
	#endif
	
	// allocate memory
	allocate(init.getNumParticles());
	
	// sanity check
	assert(m_arrays.x != NULL && m_arrays.y != NULL && m_arrays.z != NULL);
	assert(m_arrays.vx != NULL && m_arrays.vy != NULL && m_arrays.vz != NULL);
	assert(m_arrays.ax != NULL && m_arrays.ay != NULL && m_arrays.az != NULL);
	assert(m_arrays.type != NULL && m_arrays.rtag != NULL && m_arrays.tag != NULL && m_arrays.charge != NULL);
	
	// set default values
	for (unsigned int i = 0; i < m_arrays.nparticles; i++)
		{
		m_arrays.x[i] = m_arrays.y[i] = m_arrays.z[i] = 0.0;
		m_arrays.vx[i] = m_arrays.vy[i] = m_arrays.vz[i] = 0.0;
		m_arrays.ax[i] = m_arrays.ay[i] = m_arrays.az[i] = 0.0;
		m_arrays.charge[i] = 0.0;
		m_arrays.type[i] = 0;
		m_arrays.rtag[i] = i;
		m_arrays.tag[i] = i;
		}

	// allocate walls
	m_wallData = shared_ptr<WallData>(new WallData());
		
	setBox(init.getBox());
	init.initArrays(m_arrays);
	init.initWallData(m_wallData);
	
	// it is an error for particles to be initialized outside of their box
	if (!inBox())
		{
		cout << "Not all particles were found inside the given box" << endl;
		throw runtime_error("Error initializing ParticleData");
		}
	
	// default constructed shared ptr is null as desired
	m_prof = boost::shared_ptr<Profiler>();

	// if this is a GPU build, initialize the graphics card mirror data structure
	#ifdef USE_CUDA
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
	assert(inBox());
		
	#ifdef USE_CUDA
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
	m_acquired = true;
	
	#ifdef USE_CUDA
	
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
	m_acquired = true;
	
	#ifdef USE_CUDA
	
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
	
#ifdef USE_CUDA

/*! Acquire access to the particle data, for read only access on the gpu.
	
 	\return Pointers to GPU device memory where the particle data can be accessed 
	\sa acquireReadOnly
	\sa release
*/
gpu_pdata_arrays ParticleData::acquireReadOnlyGPU()
	{
	// sanity check
	assert(!m_acquired);
	m_acquired = true;
	
	if (m_exec_conf.gpu.size() == 0)
		{
		cout << "Reqesting GPU pdata, but no GPU in the Execution Configuration";
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
gpu_pdata_arrays ParticleData::acquireReadWriteGPU()
	{
	// sanity check
	assert(!m_acquired);
	m_acquired = true;
	
	if (m_exec_conf.gpu.size() == 0)
		{
		cout << "Reqesting GPU pdata, but no GPU in the Execution Configuration";
		throw runtime_error("Error acquiring GPU data");
		}
	
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
	assert(inBox());
		
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
		throw runtime_error("ParticleData is being asked to allocate 0 particles.... this makes no sense whatsoever");
	
	m_nbytes = 0;
	
	/////////////////////////////////////////////////////
	// Count bytes needed by the main data structures
	#ifdef USE_CUDA
	// 256 byte aligned version for GPU
	
	// start by adding up the number of bytes needed for the Scalar arrays, rounding up by 16
	unsigned int single_xarray_bytes = sizeof(Scalar) * N;
	if ((single_xarray_bytes & 255) != 0)
		single_xarray_bytes += 256 - (single_xarray_bytes & 255);
	
	// total all bytes from scalar arrays
	m_nbytes += single_xarray_bytes * 10;
	
	// now add up the number of bytes for the int arrays, rounding up to 16 bytes
	unsigned int single_iarray_bytes = sizeof(unsigned int) * N;
	if ((single_iarray_bytes & 255) != 0)
		single_iarray_bytes += 256 - (single_iarray_bytes & 255);
	
	m_nbytes += single_iarray_bytes * 3;
	
	#else
	
	// allocation on the GPU
	// start by adding up the number of bytes needed for the Scalar arrays
	unsigned int single_xarray_bytes = sizeof(Scalar) * N;
	
	// total all bytes from scalar arrays
	m_nbytes += single_xarray_bytes * 10;
	
	// now add up the number of bytes for the int arrays
	unsigned int single_iarray_bytes = sizeof(unsigned int) * N;
	
	m_nbytes += single_iarray_bytes * 3;
	#endif
	
	//////////////////////////////////////////////////////
	// allocate the memory on the CPU, use pinned memory if compiling for the GPU
	#ifdef USE_CUDA
	if (!m_exec_conf.gpu.empty())
		{
		m_exec_conf.gpu[0]->call(bind(cudaMallocHost, &m_data, m_nbytes));
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
	m_arrays_const.rtag = m_arrays.rtag = (unsigned int *)cur_byte;  cur_byte += single_iarray_bytes;
	m_arrays_const.tag = m_arrays.tag = (unsigned int *)cur_byte;  cur_byte += single_iarray_bytes;
	m_arrays_const.nparticles = m_arrays.nparticles = N;
	
	// sanity check
	assert(cur_byte == ((char*)m_data) + m_nbytes);
	
	#ifdef USE_CUDA
	//////////////////////////////////////////////////////////
	// allocate memory on the GPU
	if (!m_exec_conf.gpu.empty())
		{
		m_gpu_pdata.N = N;
		m_uninterleave_pitch = single_xarray_bytes/4;
		m_single_xarray_bytes = single_xarray_bytes;
	
		m_exec_conf.gpu[0]->call(bind(cudaMalloc, (void **)((void *)&m_gpu_pdata.pos), single_xarray_bytes * 4));
		m_exec_conf.gpu[0]->call(bind(cudaMalloc, (void **)((void *)&m_gpu_pdata.vel), single_xarray_bytes * 4) );
		m_exec_conf.gpu[0]->call(bind(cudaMalloc, (void **)((void *)&m_gpu_pdata.accel), single_xarray_bytes * 4) );
		m_exec_conf.gpu[0]->call(bind(cudaMalloc, (void **)((void *)&m_gpu_pdata.charge), single_xarray_bytes) );
		m_exec_conf.gpu[0]->call(bind(cudaMalloc, (void **)((void *)&m_gpu_pdata.tag), sizeof(unsigned int)*N) );
		m_exec_conf.gpu[0]->call(bind(cudaMalloc, (void **)((void *)&m_gpu_pdata.rtag), sizeof(unsigned int)*N));
		
		// allocate temporary holding area for uninterleaved data
		m_exec_conf.gpu[0]->call(bind(cudaMalloc, (void **)((void *)&m_d_staging), single_xarray_bytes*4));
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
	#ifdef USE_CUDA
	if (!m_exec_conf.gpu.empty())
		{	
		m_exec_conf.gpu[0]->call(bind(cudaFreeHost, m_data));
		
		m_exec_conf.gpu[0]->call(bind(cudaFree, m_gpu_pdata.pos));
		m_exec_conf.gpu[0]->call(bind(cudaFree, m_gpu_pdata.vel));
		m_exec_conf.gpu[0]->call(bind(cudaFree, m_gpu_pdata.accel));
		m_exec_conf.gpu[0]->call(bind(cudaFree, m_gpu_pdata.charge));
		m_exec_conf.gpu[0]->call(bind(cudaFree, m_gpu_pdata.tag));
		m_exec_conf.gpu[0]->call(bind(cudaFree, m_gpu_pdata.rtag));
		
		m_exec_conf.gpu[0]->call(bind(cudaFree, m_d_staging));
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
	m_arrays_const.type = m_arrays.type = 0;
	m_arrays_const.rtag = m_arrays.rtag = 0;
	m_arrays_const.tag = m_arrays.tag = 0;
	
	#ifdef USE_CUDA
	m_gpu_pdata.pos = NULL;
	m_gpu_pdata.vel = NULL;
	m_gpu_pdata.accel = NULL;
	m_gpu_pdata.charge = NULL;
	m_gpu_pdata.tag = NULL;
	m_gpu_pdata.rtag = NULL;
	#endif
	}
	
/*! \return true If and only if all particles are in the simulation box
	\note This function is only called in debug builds
*/
bool ParticleData::inBox()
	{
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

#ifdef USE_CUDA
/*! \post Particle data is copied from the GPU to the CPU
*/
void ParticleData::hostToDeviceCopy()
	{
	// we should never be called unless 1 gpu is in the exec conf... verify
	assert(m_exec_conf.gpu.size() == 1);
	
	if (m_prof) m_prof->push("PDATA C2G");
		
	const int N = m_arrays.nparticles;	
	
	// because of the way memory was allocated, we can copy lots of pieces in huge chunks
	// the number of bytes copied and where depend highly on the order of allocation 
	// inside allocate
	
	// copy position data to the staging area
	m_exec_conf.gpu[0]->call(bind(cudaMemcpy, m_d_staging, m_arrays.x, m_single_xarray_bytes*4, cudaMemcpyHostToDevice));
	// interleave the data
	m_exec_conf.gpu[0]->call(bind(gpu_interleave_float4, m_gpu_pdata.pos, m_d_staging, N, m_uninterleave_pitch));

	// copy velocity data to the staging area
	m_exec_conf.gpu[0]->call(bind(cudaMemcpy, m_d_staging, m_arrays.vx, m_single_xarray_bytes*3, cudaMemcpyHostToDevice));
	//interleave the data
	m_exec_conf.gpu[0]->call(bind(gpu_interleave_float4, m_gpu_pdata.vel, m_d_staging, N, m_uninterleave_pitch));
	
	// copy acceleration data to the staging area
	m_exec_conf.gpu[0]->call(bind(cudaMemcpy, m_d_staging, m_arrays.ax, m_single_xarray_bytes*3, cudaMemcpyHostToDevice));
	//interleave the data
	m_exec_conf.gpu[0]->call(bind(gpu_interleave_float4, m_gpu_pdata.accel, m_d_staging, N, m_uninterleave_pitch));
	
	// copy charge
	m_exec_conf.gpu[0]->call(bind(cudaMemcpy, m_gpu_pdata.charge, m_arrays.charge, m_single_xarray_bytes, cudaMemcpyHostToDevice));

	// copy the tag and rtag data
	m_exec_conf.gpu[0]->call(bind(cudaMemcpy, m_gpu_pdata.tag, m_arrays.tag, sizeof(unsigned int)*N, cudaMemcpyHostToDevice)); 
	m_exec_conf.gpu[0]->call(bind(cudaMemcpy, m_gpu_pdata.rtag, m_arrays.rtag, sizeof(unsigned int)*N, cudaMemcpyHostToDevice));
	
	if (m_prof)
		{
		m_exec_conf.gpu[0]->call(bind(cudaThreadSynchronize)); 
		m_prof->pop(0, m_single_xarray_bytes*4 + m_single_xarray_bytes*3*2 + sizeof(unsigned int)*N * 2);
		}
	}
		
/*! Particle data is copied from the CPU to the GPU
*/
void ParticleData::deviceToHostCopy()
	{
	if (m_prof) m_prof->push("PDATA G2C");
	
	// we should never be called unless 1 gpu is in the exec conf... verify
	assert(m_exec_conf.gpu.size() == 1);

	const int N = m_arrays.nparticles;

	// because of the way memory was allocated, we can copy lots of pieces in huge chunks
	// the number of bytes copied and where depend highly on the order of allocation 
	// inside allocate

	// uninterleave the data
	m_exec_conf.gpu[0]->call(bind(gpu_uninterleave_float4, m_d_staging, m_gpu_pdata.pos, N, m_uninterleave_pitch));
	// copy position data from the staging area
	m_exec_conf.gpu[0]->call(bind(cudaMemcpy, m_arrays.x, m_d_staging, m_single_xarray_bytes*4, cudaMemcpyDeviceToHost));

	// uninterleave the data
	m_exec_conf.gpu[0]->call(bind(gpu_uninterleave_float4, m_d_staging, m_gpu_pdata.vel, N, m_uninterleave_pitch));
	// copy velocity data from the staging area
	m_exec_conf.gpu[0]->call(bind(cudaMemcpy, m_arrays.vx, m_d_staging, m_single_xarray_bytes*3, cudaMemcpyDeviceToHost));

	// uninterleave the data
	m_exec_conf.gpu[0]->call(bind(gpu_uninterleave_float4, m_d_staging, m_gpu_pdata.accel, N, m_uninterleave_pitch));
	// copy velocity data from the staging area
	m_exec_conf.gpu[0]->call(bind(cudaMemcpy, m_arrays.ax, m_d_staging, m_single_xarray_bytes*3, cudaMemcpyDeviceToHost));

	// copy charge
	m_exec_conf.gpu[0]->call(bind(cudaMemcpy, m_arrays.charge, m_gpu_pdata.charge, m_single_xarray_bytes, cudaMemcpyDeviceToHost));
	
	// copy the tag and rtag data
	m_exec_conf.gpu[0]->call(bind(cudaMemcpy, m_arrays.tag, m_gpu_pdata.tag, sizeof(unsigned int)*N, cudaMemcpyDeviceToHost)); 
	m_exec_conf.gpu[0]->call(bind(cudaMemcpy, m_arrays.rtag, m_gpu_pdata.rtag, sizeof(unsigned int)*N, cudaMemcpyDeviceToHost));

	if (m_prof) 
		{
		m_exec_conf.gpu[0]->call(bind(cudaThreadSynchronize));
		m_prof->pop(0, m_single_xarray_bytes*4 + m_single_xarray_bytes*3*2 + sizeof(unsigned int)*N * 2);
		}
	}
#endif
	
#ifdef USE_PYTHON

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

//! Wrapper class needed for exposing virtual functions to c++
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
		.def("getBox", &ParticleData::getBox, return_value_policy<copy_const_reference>())
		.def("setBox", &ParticleData::setBox)
		.def("getN", &ParticleData::getN)
		.def("getNTypes", &ParticleData::getNTypes)
		.def("acquireReadOnly", &ParticleData::acquireReadOnly, return_value_policy<copy_const_reference>())
		.def("acquireReadWrite", &ParticleData::acquireReadWrite, return_value_policy<copy_const_reference>())
		#ifdef USE_CUDA
		.def("acquireReadOnlyGPU", &ParticleData::acquireReadOnlyGPU)
		.def("acquireReadWriteGPU", &ParticleData::acquireReadWriteGPU)
		.def("getBoxGPU", &ParticleData::getBoxGPU, return_value_policy<copy_const_reference>())
		#endif
		.def("release", &ParticleData::release)
		.def("setProfiler", &ParticleData::setProfiler)
		.def("__str__", &print_ParticleData)
		;
	}
#endif
