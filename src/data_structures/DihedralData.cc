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

#include "DihedralData.h"
#include "ParticleData.h"

#include <boost/python.hpp>
using namespace boost::python;

#include <boost/bind.hpp>
using namespace boost;

#include <iostream>
#include <stdexcept>
using namespace std;

/*! \file DihedralData.cc
 	\brief Defines DihedralData.
 */

/*! \param pdata ParticleData these dihedrals refer into
	\param n_dihedral_types Number of dihedral types in the list
	
	Taking in pdata as a pointer instead of a shared pointer is sloppy, but there really isn't an alternative
	due to the way ParticleData is constructed. Things will be fixed in a later version with a reorganization
	of the various data structures. For now, be careful not to destroy the ParticleData and keep the DihedralData hanging
	around.
*/
DihedralData::DihedralData(boost::shared_ptr<ParticleData> pdata, unsigned int n_dihedral_types) : m_n_dihedral_types(n_dihedral_types), m_dihedrals_dirty(false), m_pdata(pdata)
	{
	assert(pdata);
	
	// attach to the signal for notifications of particle sorts
	m_sort_connection = m_pdata->connectParticleSort(bind(&DihedralData::setDirty, this));
	
	// offer a default type mapping
	for (unsigned int i = 0; i < n_dihedral_types; i++)
		{
		char suffix[2];
		suffix[0] = 'A' + i;
		suffix[1] = '\0';

		string name = string("dihedral") + string(suffix);
		m_dihedral_type_mapping.push_back(name);
		}

	#ifdef ENABLE_CUDA
	// get the execution configuration
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
	
	// init pointers
	m_host_dihedrals = NULL;
	m_host_n_dihedrals = NULL;
        m_host_dihedralsABCD = NULL;
	m_gpu_dihedraldata.resize(exec_conf.gpu.size());
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		{
		m_gpu_dihedraldata[cur_gpu].dihedrals = NULL;
		m_gpu_dihedraldata[cur_gpu].dihedralABCD = NULL;
		m_gpu_dihedraldata[cur_gpu].n_dihedrals = NULL;
		m_gpu_dihedraldata[cur_gpu].height = 0;
		m_gpu_dihedraldata[cur_gpu].pitch = 0;
		}
	
	// allocate memory on the GPU if there is a GPU in the execution configuration
	if (exec_conf.gpu.size() >= 1)
		{
		allocateDihedralTable(1);
		}
	#endif
	}

DihedralData::~DihedralData()
	{
	m_sort_connection.disconnect();

	#ifdef ENABLE_CUDA
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
	if (!exec_conf.gpu.empty())
		{
		freeDihedralTable();
		}
	#endif
	}

/*! \post An dihedral between particles specified in \a dihedral is created. 
	
	\note Each dihedral should only be specified once! There are no checks to prevent one from being 
	specified more than once, and doing so would result in twice the force and twice the energy.
	
	\note If an dihedral is added with \c type=49, then there must be at least 50 dihedral types (0-49) total,
	even if not all values are used. So dihedrals should be added with small contiguous types.
	\param dihedral The Dihedral to add to the list
 */	
void DihedralData::addDihedral(const Dihedral& dihedral)
	{

	// check for some silly errors a user could make 	
	if (dihedral.a >= m_pdata->getN() || dihedral.b >= m_pdata->getN() || dihedral.c >= m_pdata->getN()  || dihedral.d >= m_pdata->getN())
		{
		cerr << endl << "***Error! Particle tag out of bounds when attempting to add dihedral: " << dihedral.a << "," << dihedral.b << "," << dihedral.c << endl << endl;
		throw runtime_error("Error adding dihedral");
		}
	
	if (dihedral.a == dihedral.b || dihedral.a == dihedral.c || dihedral.b == dihedral.c || dihedral.a == dihedral.d || dihedral.b == dihedral.d || dihedral.c == dihedral.d )
		{
		cerr << endl << "***Error! Particle cannot included in an dihedral twice! " << dihedral.a << "," << dihedral.b << "," << dihedral.c << endl << endl;
		throw runtime_error("Error adding dihedral");
		}
	
	// check that the type is within bouds
	if (dihedral.type+1 > m_n_dihedral_types)
		{
		cerr << endl << "***Error! Invalid dihedral type! " << dihedral.type << ", the number of types is " << m_n_dihedral_types << endl << endl;
		throw runtime_error("Error adding dihedral");
		}

	m_dihedrals.push_back(dihedral);
	m_dihedrals_dirty = true;
	}
	
/*! \param dihedral_type_mapping Mapping array to set
	\c dihedral_type_mapping[type] should be set to the name of the dihedral type with index \c type.
	The vector \b must have \c n_dihedral_types elements in it.
*/
void DihedralData::setDihedralTypeMapping(const std::vector<std::string>& dihedral_type_mapping)
	{
	assert(dihedral_type_mapping.size() == m_n_dihedral_types);
	m_dihedral_type_mapping = dihedral_type_mapping;
	}
	

/*! \param name Type name to get the index of
	\return Type index of the corresponding type name
	\note Throws an exception if the type name is not found
*/
unsigned int DihedralData::getTypeByName(const std::string &name)
	{
	// search for the name
	for (unsigned int i = 0; i < m_dihedral_type_mapping.size(); i++)
		{
		if (m_dihedral_type_mapping[i] == name)
			return i;
		}
		
	cerr << endl << "***Error! Dihedral type " << name << " not found!" << endl;
	throw runtime_error("Error mapping type name");	
	return 0;
	}
		
/*! \param type Type index to get the name of
	\returns Type name of the requested type
	\note Type indices must range from 0 to getNTypes or this method throws an exception.
*/
std::string DihedralData::getNameByType(unsigned int type)
	{
	// check for an invalid request
	if (type >= m_n_dihedral_types)
		{
		cerr << endl << "***Error! Requesting type name for non-existant type " << type << endl << endl;
		throw runtime_error("Error mapping type name");
		}
		
	// return the name
	return m_dihedral_type_mapping[type];
	}
	
#ifdef ENABLE_CUDA

/*! Updates the dihedral data on the GPU if needed and returns the data structure needed to access it.
*/
std::vector<gpu_dihedraltable_array>& DihedralData::acquireGPU()
	{
	if (m_dihedrals_dirty)
		{
		updateDihedralTable();
		m_dihedrals_dirty = false;
		}
	return m_gpu_dihedraldata;
	}


/*! \post The dihedral tag data added via addDihedral() is translated to dihedrals based
	on particle index for use in the GPU kernel. This new dihedral table is then uploaded
	to the device.
*/
void DihedralData::updateDihedralTable()
	{

	assert(m_host_n_dihedrals);
	assert(m_host_dihedrals);
	assert(m_host_dihedralsABCD);
	
	// count the number of dihedrals per particle
	// start by initializing the host n_dihedrals values to 0
	memset(m_host_n_dihedrals, 0, sizeof(unsigned int) * m_pdata->getN());

	// loop through the particles and count the number of dihedrals based on each particle index
        // however, only the b atom in the a-b-c dihedral is included in the count.
	ParticleDataArraysConst arrays = m_pdata->acquireReadOnly();
	for (unsigned int cur_dihedral = 0; cur_dihedral < m_dihedrals.size(); cur_dihedral++)
		{
		unsigned int tag1 = m_dihedrals[cur_dihedral].a; //
		unsigned int tag2 = m_dihedrals[cur_dihedral].b;
		unsigned int tag3 = m_dihedrals[cur_dihedral].c; //
		unsigned int tag4 = m_dihedrals[cur_dihedral].d; //
		int idx1 = arrays.rtag[tag1]; //
		int idx2 = arrays.rtag[tag2];
		int idx3 = arrays.rtag[tag3]; //
		int idx4 = arrays.rtag[tag4]; //
		
		m_host_n_dihedrals[idx1]++; //
		m_host_n_dihedrals[idx2]++;
		m_host_n_dihedrals[idx3]++; //
		m_host_n_dihedrals[idx4]++; //
		}
		
	// find the maximum number of dihedrals
	unsigned int num_dihedrals_max = 0;
	for (unsigned int i = 0; i < arrays.nparticles; i++)
		{
		if (m_host_n_dihedrals[i] > num_dihedrals_max)
			num_dihedrals_max = m_host_n_dihedrals[i];
		}
		
	// re allocate memory if needed
	if (num_dihedrals_max > m_gpu_dihedraldata[0].height)
		{
		reallocateDihedralTable(num_dihedrals_max);
		}
		
	// now, update the actual table
	// zero the number of dihedrals counter (again)
	memset(m_host_n_dihedrals, 0, sizeof(unsigned int) * m_pdata->getN());
	
	// loop through all dihedrals and add them to each column in the list
	unsigned int pitch = m_pdata->getN(); //removed 'unsigned'
	for (unsigned int cur_dihedral = 0; cur_dihedral < m_dihedrals.size(); cur_dihedral++)
		{
		unsigned int tag1 = m_dihedrals[cur_dihedral].a;
		unsigned int tag2 = m_dihedrals[cur_dihedral].b;
		unsigned int tag3 = m_dihedrals[cur_dihedral].c;
		unsigned int tag4 = m_dihedrals[cur_dihedral].d;
		unsigned int type = m_dihedrals[cur_dihedral].type;
		int idx1 = arrays.rtag[tag1];
		int idx2 = arrays.rtag[tag2];
		int idx3 = arrays.rtag[tag3];
		int idx4 = arrays.rtag[tag4];
		dihedralABCD dihedral_type_abcd;

		// get the number of dihedrals for the b in a-b-c triplet
		int num1 = m_host_n_dihedrals[idx1]; //
                int num2 = m_host_n_dihedrals[idx2];
		int num3 = m_host_n_dihedrals[idx3]; //
		int num4 = m_host_n_dihedrals[idx4]; //
		
		// add a new dihedral to the table, provided each one is a "b" from an a-b-c triplet 
		// store in the texture as .x=a=idx1, .y=c=idx2, and b comes from the gpu
		// or the cpu, generally from the idx2 index
		dihedral_type_abcd = a_atom;
		m_host_dihedrals[num1*pitch + idx1] = make_uint4(idx2, idx3, idx4, type); //
		m_host_dihedralsABCD[num1*pitch + idx1] = make_uint1(dihedral_type_abcd);

		dihedral_type_abcd = b_atom;
		m_host_dihedrals[num2*pitch + idx2] = make_uint4(idx1, idx3, idx4, type);
		m_host_dihedralsABCD[num2*pitch + idx2] = make_uint1(dihedral_type_abcd);

		dihedral_type_abcd = c_atom;
		m_host_dihedrals[num3*pitch + idx3] = make_uint4(idx1, idx2, idx4, type); //
		m_host_dihedralsABCD[num3*pitch + idx3] = make_uint1(dihedral_type_abcd);

		dihedral_type_abcd = d_atom;
		m_host_dihedrals[num4*pitch + idx4] = make_uint4(idx1, idx2, idx3, type); //
		m_host_dihedralsABCD[num4*pitch + idx4] = make_uint1(dihedral_type_abcd);
		
		// increment the number of dihedrals
		m_host_n_dihedrals[idx1]++; //
		m_host_n_dihedrals[idx2]++;
		m_host_n_dihedrals[idx3]++; //
		m_host_n_dihedrals[idx4]++; //
		}
	
	m_pdata->release();
	
	// copy the dihedral table to the device
	copyDihedralTable();
	}
	
/*! \param height New height for the dihedral table
	\post Reallocates memory on the device making room for up to 
		\a height dihedrals per particle.
	\note updateDihedralTable() needs to be called after so that the
		data in the dihedral table will be correct.
*/
void DihedralData::reallocateDihedralTable(int height)
	{
	freeDihedralTable();
	allocateDihedralTable(height);
	}
	
/*! \param height Height for the dihedral table
*/
void DihedralData::allocateDihedralTable(int height)
	{
	// make sure the arrays have been deallocated
	assert(m_host_dihedrals == NULL);
	assert(m_host_dihedralsABCD == NULL);
	assert(m_host_n_dihedrals == NULL);
	
	unsigned int N = m_pdata->getN();
	
	// get the execution configuration
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
	
	// allocate device memory
	exec_conf.tagAll(__FILE__, __LINE__);
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		exec_conf.gpu[cur_gpu]->call(bind(&gpu_dihedraltable_array::allocate, &m_gpu_dihedraldata[cur_gpu], m_pdata->getLocalNum(cur_gpu), height));
	
	
	// allocate and zero host memory
	exec_conf.gpu[0]->call(bind(cudaMallocHost, (void**)((void*)&m_host_n_dihedrals), N*sizeof(int)));
	memset((void*)m_host_n_dihedrals, 0, N*sizeof(int));
	
	exec_conf.gpu[0]->call(bind(cudaMallocHost, (void**)((void*)&m_host_dihedrals), N * height * sizeof(uint4)));
	memset((void*)m_host_dihedrals, 0, N*height*sizeof(uint4));

	exec_conf.gpu[0]->call(bind(cudaMallocHost, (void**)((void*)&m_host_dihedralsABCD), N * height * sizeof(uint1)));
	memset((void*)m_host_dihedralsABCD, 0, N*height*sizeof(uint1));

	}

void DihedralData::freeDihedralTable()
	{
	// get the execution configuration
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();	
	
	// free device memory
	exec_conf.tagAll(__FILE__, __LINE__);
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		exec_conf.gpu[cur_gpu]->call(bind(&gpu_dihedraltable_array::deallocate, &m_gpu_dihedraldata[cur_gpu]));
	
	// free host memory
	exec_conf.gpu[0]->call(bind(cudaFreeHost, m_host_dihedrals));
	m_host_dihedrals = NULL;
	// free host memory
	exec_conf.gpu[0]->call(bind(cudaFreeHost, m_host_dihedralsABCD));
	m_host_dihedralsABCD = NULL;
	exec_conf.gpu[0]->call(bind(cudaFreeHost, m_host_n_dihedrals));
	m_host_n_dihedrals = NULL;
	}

//! Copies the dihedral table to the device
void DihedralData::copyDihedralTable()
	{
	// get the execution configuration
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();	
	
	exec_conf.tagAll(__FILE__, __LINE__);
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		{
		// we need to copy the table row by row since cudaMemcpy2D has severe pitch limitations
		for (unsigned int row = 0; row < m_gpu_dihedraldata[0].height; row++)
			{
			exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, m_gpu_dihedraldata[cur_gpu].dihedrals + m_gpu_dihedraldata[cur_gpu].pitch*row, 
				m_host_dihedrals + row * m_pdata->getN() + m_pdata->getLocalBeg(cur_gpu),
				sizeof(uint4) * m_pdata->getLocalNum(cur_gpu),
				cudaMemcpyHostToDevice));

exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, m_gpu_dihedraldata[cur_gpu].dihedralABCD + m_gpu_dihedraldata[cur_gpu].pitch*row, 
				m_host_dihedralsABCD + row * m_pdata->getN() + m_pdata->getLocalBeg(cur_gpu),
				sizeof(uint1) * m_pdata->getLocalNum(cur_gpu),
				cudaMemcpyHostToDevice));
			}
				
		exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, m_gpu_dihedraldata[cur_gpu].n_dihedrals, 
				m_host_n_dihedrals + m_pdata->getLocalBeg(cur_gpu),
				sizeof(unsigned int) * m_pdata->getLocalNum(cur_gpu),
				cudaMemcpyHostToDevice));
		}
	}
#endif

void export_DihedralData()
	{
	class_<DihedralData, boost::shared_ptr<DihedralData>, boost::noncopyable>("DihedralData", init<boost::shared_ptr<ParticleData>, unsigned int>())
		.def("getNumDihedrals", &DihedralData::getNumDihedrals)
		.def("getNDihedralTypes", &DihedralData::getNDihedralTypes)
		.def("getTypeByName", &DihedralData::getTypeByName)
		.def("getNameByType", &DihedralData::getNameByType)
		;
	
	}

#ifdef WIN32
#pragma warning( pop )
#endif

