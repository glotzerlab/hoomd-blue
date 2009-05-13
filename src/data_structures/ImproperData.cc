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

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include "ImproperData.h"
#include "ParticleData.h"

#include <boost/python.hpp>
using namespace boost::python;

#include <boost/bind.hpp>
using namespace boost;

#include <iostream>
#include <stdexcept>
using namespace std;

/*! \file ImproperData.cc
 	\brief Defines ImproperData.
 */

/*! \param pdata ParticleData these improper refer into
	\param n_improper_types Number of improper types in the list
	
	Taking in pdata as a pointer instead of a shared pointer is sloppy, but there really isn't an alternative
	due to the way ParticleData is constructed. Things will be fixed in a later version with a reorganization
	of the various data structures. For now, be careful not to destroy the ParticleData and keep the ImproperData hanging
	around.
*/
ImproperData::ImproperData(ParticleData* pdata, unsigned int n_improper_types) : m_n_improper_types(n_improper_types), m_impropers_dirty(false), m_pdata(pdata)
	{
	assert(pdata);
	
	// attach to the signal for notifications of particle sorts
	m_sort_connection = m_pdata->connectParticleSort(bind(&ImproperData::setDirty, this));
	
	// offer a default type mapping
	for (unsigned int i = 0; i < n_improper_types; i++)
		{
		char suffix[2];
		suffix[0] = 'A' + i;
		suffix[1] = '\0';

		string name = string("improper") + string(suffix);
		m_improper_type_mapping.push_back(name);
		}

	#ifdef ENABLE_CUDA
	// get the execution configuration
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
	
	// init pointers
	m_host_impropers = NULL;
	m_host_n_impropers = NULL;
        m_host_impropersABCD = NULL;
	m_gpu_improperdata.resize(exec_conf.gpu.size());
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		{
		m_gpu_improperdata[cur_gpu].impropers = NULL;
                m_gpu_improperdata[cur_gpu].improperABCD = NULL;
		m_gpu_improperdata[cur_gpu].n_impropers = NULL;
		m_gpu_improperdata[cur_gpu].height = 0;
		m_gpu_improperdata[cur_gpu].pitch = 0;
		}
	
	// allocate memory on the GPU if there is a GPU in the execution configuration
	if (exec_conf.gpu.size() >= 1)
		{
		allocateImproperTable(1);
		}
	#endif
	}

ImproperData::~ImproperData()
	{
	m_sort_connection.disconnect();

	#ifdef ENABLE_CUDA
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
	if (!exec_conf.gpu.empty())
		{
		freeImproperTable();
		}
	#endif
	}

/*! \post An improper between particles specified in \a improper is created. 
	
	\note Each improper should only be specified once! There are no checks to prevent one from being 
	specified more than once, and doing so would result in twice the force and twice the energy.
	
	\note If an improper is added with \c type=49, then there must be at least 50 improper types (0-49) total,
	even if not all values are used. So impropers should be added with small contiguous types.
	\param improper The Improper to add to the list
 */	
void ImproperData::addImproper(const Improper& improper)
	{

	// check for some silly errors a user could make 	
	if (improper.a >= m_pdata->getN() || improper.b >= m_pdata->getN() || improper.c >= m_pdata->getN()  || improper.d >= m_pdata->getN())
		{
		cerr << endl << "***Error! Particle tag out of bounds when attempting to add improper: " << improper.a << "," << improper.b << "," << improper.c << "," << improper.d << endl << endl;
		throw runtime_error("Error adding improper");
		}
	
	if (improper.a == improper.b || improper.a == improper.c || improper.b == improper.c || improper.a == improper.d || improper.b == improper.d || improper.c == improper.d )
		{
		cerr << endl << "***Error! Particle cannot included in an improper twice! " << improper.a << "," << improper.b << "," << improper.c << "," << improper.d << endl << endl;
		throw runtime_error("Error adding improper");
		}
	
	// check that the type is within bouds
	if (improper.type+1 > m_n_improper_types)
		{
		cerr << endl << "***Error! Invalid improper type! " << improper.type << ", the number of types is " << m_n_improper_types << endl << endl;
		throw runtime_error("Error adding improper");
		}

	m_impropers.push_back(improper);
	m_impropers_dirty = true;
	}
	
/*! \param improper_type_mapping Mapping array to set
	\c improper_type_mapping[type] should be set to the name of the improper type with index \c type.
	The vector \b must have \c n_improper_types elements in it.
*/
void ImproperData::setImproperTypeMapping(const std::vector<std::string>& improper_type_mapping)
	{
	assert(improper_type_mapping.size() == m_n_improper_types);
	m_improper_type_mapping = improper_type_mapping;
	}
	

/*! \param name Type name to get the index of
	\return Type index of the corresponding type name
	\note Throws an exception if the type name is not found
*/
unsigned int ImproperData::getTypeByName(const std::string &name)
	{
	// search for the name
	for (unsigned int i = 0; i < m_improper_type_mapping.size(); i++)
		{
		if (m_improper_type_mapping[i] == name)
			return i;
		}
		
	cerr << endl << "***Error! Improper type " << name << " not found!" << endl;
	throw runtime_error("Error mapping type name");	
	return 0;
	}
		
/*! \param type Type index to get the name of
	\returns Type name of the requested type
	\note Type indices must range from 0 to getNTypes or this method throws an exception.
*/
std::string ImproperData::getNameByType(unsigned int type)
	{
	// check for an invalid request
	if (type >= m_n_improper_types)
		{
		cerr << endl << "***Error! Requesting type name for non-existant type " << type << endl << endl;
		throw runtime_error("Error mapping type name");
		}
		
	// return the name
	return m_improper_type_mapping[type];
	}
	
#ifdef ENABLE_CUDA

/*! Updates the improper data on the GPU if needed and returns the data structure needed to access it.
*/
std::vector<gpu_impropertable_array>& ImproperData::acquireGPU()
	{
	if (m_impropers_dirty)
		{
		updateImproperTable();
		m_impropers_dirty = false;
		}
	return m_gpu_improperdata;
	}


/*! \post The improper tag data added via addImproper() is translated to impropers based
	on particle index for use in the GPU kernel. This new improper table is then uploaded
	to the device.
*/
void ImproperData::updateImproperTable()
	{

	assert(m_host_n_impropers);
	assert(m_host_impropers);
	assert(m_host_impropersABCD);
	
	// count the number of impropers per particle
	// start by initializing the host n_impropers values to 0
	memset(m_host_n_impropers, 0, sizeof(unsigned int) * m_pdata->getN());

	// loop through the particles and count the number of impropers based on each particle index
	ParticleDataArraysConst arrays = m_pdata->acquireReadOnly();
	for (unsigned int cur_improper = 0; cur_improper < m_impropers.size(); cur_improper++)
		{
		unsigned int tag1 = m_impropers[cur_improper].a; //
		unsigned int tag2 = m_impropers[cur_improper].b;
		unsigned int tag3 = m_impropers[cur_improper].c; //
		unsigned int tag4 = m_impropers[cur_improper].d; //
		int idx1 = arrays.rtag[tag1]; //
		int idx2 = arrays.rtag[tag2];
		int idx3 = arrays.rtag[tag3]; //
		int idx4 = arrays.rtag[tag4]; //
		
		m_host_n_impropers[idx1]++; //
		m_host_n_impropers[idx2]++;
		m_host_n_impropers[idx3]++; //
		m_host_n_impropers[idx4]++; //
		}
		
	// find the maximum number of impropers
	unsigned int num_impropers_max = 0;
	for (unsigned int i = 0; i < arrays.nparticles; i++)
		{
		if (m_host_n_impropers[i] > num_impropers_max)
			num_impropers_max = m_host_n_impropers[i];
		}
		
	// re allocate memory if needed
	if (num_impropers_max > m_gpu_improperdata[0].height)
		{
		reallocateImproperTable(num_impropers_max);
		}
		
	// now, update the actual table
	// zero the number of impropers counter (again)
	memset(m_host_n_impropers, 0, sizeof(unsigned int) * m_pdata->getN());
	
	// loop through all impropers and add them to each column in the list
	unsigned int pitch = m_pdata->getN(); //removed 'unsigned'
	for (unsigned int cur_improper = 0; cur_improper < m_impropers.size(); cur_improper++)
		{
		unsigned int tag1 = m_impropers[cur_improper].a;
		unsigned int tag2 = m_impropers[cur_improper].b;
		unsigned int tag3 = m_impropers[cur_improper].c;
		unsigned int tag4 = m_impropers[cur_improper].d;
		unsigned int type = m_impropers[cur_improper].type;
		int idx1 = arrays.rtag[tag1];
		int idx2 = arrays.rtag[tag2];
		int idx3 = arrays.rtag[tag3];
		int idx4 = arrays.rtag[tag4];
		improperABCD improper_type_abcd;

		// get the number of impropers for the b in a-b-c triplet
		int num1 = m_host_n_impropers[idx1]; //
                int num2 = m_host_n_impropers[idx2];
		int num3 = m_host_n_impropers[idx3]; //
		int num4 = m_host_n_impropers[idx4]; //
		
		// add a new improper to the table, provided each one is a "b" from an a-b-c triplet 
                // store in the texture as .x=a=idx1, .y=c=idx2, etc. comes from the gpu
                // or the cpu, generally from the idx2 index
                improper_type_abcd = a_atom;
		m_host_impropers[num1*pitch + idx1] = make_uint4(idx2, idx3, idx4, type); //
                m_host_impropersABCD[num1*pitch + idx1] = make_uint1(improper_type_abcd);

                improper_type_abcd = b_atom;
		m_host_impropers[num2*pitch + idx2] = make_uint4(idx1, idx3, idx4, type);
                m_host_impropersABCD[num2*pitch + idx2] = make_uint1(improper_type_abcd);

                improper_type_abcd = c_atom;
		m_host_impropers[num3*pitch + idx3] = make_uint4(idx1, idx2, idx4, type); //
                m_host_impropersABCD[num3*pitch + idx3] = make_uint1(improper_type_abcd);

                improper_type_abcd = d_atom;
		m_host_impropers[num4*pitch + idx4] = make_uint4(idx1, idx2, idx3, type); //
                m_host_impropersABCD[num4*pitch + idx4] = make_uint1(improper_type_abcd);
		
		// increment the number of impropers
		m_host_n_impropers[idx1]++; //
		m_host_n_impropers[idx2]++;
		m_host_n_impropers[idx3]++; //
		m_host_n_impropers[idx4]++; //
		}
	
	m_pdata->release();
	
	// copy the improper table to the device
	copyImproperTable();
	}
	
/*! \param height New height for the improper table
	\post Reallocates memory on the device making room for up to 
		\a height impropers per particle.
	\note updateImproperTable() needs to be called after so that the
		data in the improper table will be correct.
*/
void ImproperData::reallocateImproperTable(int height)
	{
	freeImproperTable();
	allocateImproperTable(height);
	}
	
/*! \param height Height for the improper table
*/
void ImproperData::allocateImproperTable(int height)
	{
	// make sure the arrays have been deallocated
	assert(m_host_impropers == NULL);
	assert(m_host_impropersABCD == NULL);
	assert(m_host_n_impropers == NULL);
	
	unsigned int N = m_pdata->getN();
	
	// get the execution configuration
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
	
	// allocate device memory
	exec_conf.tagAll(__FILE__, __LINE__);
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		exec_conf.gpu[cur_gpu]->call(bind(&gpu_impropertable_array::allocate, &m_gpu_improperdata[cur_gpu], m_pdata->getLocalNum(cur_gpu), height));
	
	
	// allocate and zero host memory
	exec_conf.gpu[0]->call(bind(cudaMallocHost, (void**)((void*)&m_host_n_impropers), N*sizeof(int)));
	memset((void*)m_host_n_impropers, 0, N*sizeof(int));
	
	exec_conf.gpu[0]->call(bind(cudaMallocHost, (void**)((void*)&m_host_impropers), N * height * sizeof(uint4)));
	memset((void*)m_host_impropers, 0, N*height*sizeof(uint4));

	exec_conf.gpu[0]->call(bind(cudaMallocHost, (void**)((void*)&m_host_impropersABCD), N * height * sizeof(uint1)));
	memset((void*)m_host_impropersABCD, 0, N*height*sizeof(uint1));

	}

void ImproperData::freeImproperTable()
	{
	// get the execution configuration
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();	
	
	// free device memory
	exec_conf.tagAll(__FILE__, __LINE__);
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		exec_conf.gpu[cur_gpu]->call(bind(&gpu_impropertable_array::deallocate, &m_gpu_improperdata[cur_gpu]));
	
	// free host memory
	exec_conf.gpu[0]->call(bind(cudaFreeHost, m_host_impropers));
	m_host_impropers = NULL;
	// free host memory
	exec_conf.gpu[0]->call(bind(cudaFreeHost, m_host_impropersABCD));
	m_host_impropersABCD = NULL;
	exec_conf.gpu[0]->call(bind(cudaFreeHost, m_host_n_impropers));
	m_host_n_impropers = NULL;
	}

//! Copies the improper table to the device
void ImproperData::copyImproperTable()
	{
	// get the execution configuration
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();	
	
	exec_conf.tagAll(__FILE__, __LINE__);
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		{
		// we need to copy the table row by row since cudaMemcpy2D has severe pitch limitations
		for (unsigned int row = 0; row < m_gpu_improperdata[0].height; row++)
			{
			exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, m_gpu_improperdata[cur_gpu].impropers + m_gpu_improperdata[cur_gpu].pitch*row, 
				m_host_impropers + row * m_pdata->getN() + m_pdata->getLocalBeg(cur_gpu),
				sizeof(uint4) * m_pdata->getLocalNum(cur_gpu),
				cudaMemcpyHostToDevice));

exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, m_gpu_improperdata[cur_gpu].improperABCD + m_gpu_improperdata[cur_gpu].pitch*row, 
				m_host_impropersABCD + row * m_pdata->getN() + m_pdata->getLocalBeg(cur_gpu),
				sizeof(uint1) * m_pdata->getLocalNum(cur_gpu),
				cudaMemcpyHostToDevice));
			}
				
		exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, m_gpu_improperdata[cur_gpu].n_impropers, 
				m_host_n_impropers + m_pdata->getLocalBeg(cur_gpu),
				sizeof(unsigned int) * m_pdata->getLocalNum(cur_gpu),
				cudaMemcpyHostToDevice));
		}
	}
#endif

void export_ImproperData()
	{
	class_<ImproperData, boost::shared_ptr<ImproperData>, boost::noncopyable>("ImproperData", init<ParticleData *, unsigned int>())
		.def("getNumImpropers", &ImproperData::getNumImpropers)
		.def("getNImproperTypes", &ImproperData::getNImproperTypes)
		.def("getTypeByName", &ImproperData::getTypeByName)
		.def("getNameByType", &ImproperData::getNameByType)
		;
	
	}

#ifdef WIN32
#pragma warning( pop )
#endif

