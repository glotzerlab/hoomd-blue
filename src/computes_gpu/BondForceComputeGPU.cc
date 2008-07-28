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

/*! \file BondForceComputeGPU.cc
	\brief Declares the BondForceComputeGPU class
*/

#include "BondForceComputeGPU.h"

#ifdef USE_PYTHON
#include <boost/python.hpp>
using namespace boost::python;
#endif

#include <boost/bind.hpp>
using namespace boost;

using namespace std;

/*! \param pdata ParticleData to compute bond forces on
	\param K Stiffness parameter for the force computation
	\param r_0 Equilibrium length for the force computation
*/
BondForceComputeGPU::BondForceComputeGPU(boost::shared_ptr<ParticleData> pdata, Scalar K, Scalar r_0)
	: BondForceCompute(pdata, K, r_0), m_dirty(true), m_block_size(64), m_bondlist(NULL)
	{
	// init pointers
	m_gpu_bondtable.list = NULL;
	m_gpu_bondtable.height = 0;
	m_gpu_bondtable.pitch = 0;
	
	// check the execution configuration
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
	// only one GPU is currently supported
	if (exec_conf.gpu.size() == 0)
		{
		cout << "Creating a BondForceComputeGPU with no GPU in the execution configuration" << endl;
		throw std::runtime_error("Error initializing BondForceComputeGPU");
		}
	if (exec_conf.gpu.size() != 1)
		{
		cout << "More than one GPU is not currently supported";
		throw std::runtime_error("Error initializing BondForceComputeGPU");
		}
	
	// allocate the initial bond table with height 1
	allocateBondTable(1);
		
	// attach to the signal for notifications of particle sorts
	m_sort_connection = m_pdata->connectParticleSort(bind(&BondForceComputeGPU::setDirty, this));
	}
	
BondForceComputeGPU::~BondForceComputeGPU()
	{
	freeBondTable();
	m_sort_connection.disconnect();
	}
		
/*! \sa BondForceCompute::addBond()
	\param tag1 Tag of the first particle in the bond
	\param tag2 Tag of the second particle in the bond
	
	Adding bonds is not meant to be an action performed during the simulation.
	The copy of the data to the GPU every time would be a significant overhead.
*/
void BondForceComputeGPU::addBond(unsigned int tag1, unsigned int tag2)
	{
	BondForceCompute::addBond(tag1, tag2);
	m_dirty = true;
	}
	
/*! \post The bond tag data added via addBond() is translated to bonds based
	on particle index for use in the GPU kernel. This new bond table is then uploaded
	to the device.
*/
void BondForceComputeGPU::updateBondTable()
	{
	// start by finding the number of bonds per particle
	vector< unsigned int > num_bonds(m_pdata->getN(), 0);
	
	ParticleDataArraysConst arrays = m_pdata->acquireReadOnly();
	
	for (unsigned int cur_bond = 0; cur_bond < m_bonds.size(); cur_bond++)
		{
		unsigned int tag1 = m_bonds[cur_bond].m_tag1;
		unsigned int tag2 = m_bonds[cur_bond].m_tag2;
		int idx1 = arrays.rtag[tag1];
		int idx2 = arrays.rtag[tag2];
		
		num_bonds[idx1]++;
		num_bonds[idx2]++;
		}
		
	// find the maximum number of bonds
	unsigned int num_bonds_max = 0;
	for (unsigned int i = 0; i < arrays.nparticles; i++)
		{
		if (num_bonds[i] > num_bonds_max)
			num_bonds_max = num_bonds[i];
		}
	
	// re allocate memory if needed
	if (num_bonds_max+1 > m_gpu_bondtable.height)
		{
		reallocateBondTable(num_bonds_max+1);
		}
		
	// now, update the actual table
	// zero the number of bonds counter
	for (unsigned int i = 0; i < arrays.nparticles; i++)
		m_bondlist[i] = 0;

	int pitch = m_gpu_bondtable.pitch;
	for (unsigned int cur_bond = 0; cur_bond < m_bonds.size(); cur_bond++)
		{
		unsigned int tag1 = m_bonds[cur_bond].m_tag1;
		unsigned int tag2 = m_bonds[cur_bond].m_tag2;
		int idx1 = arrays.rtag[tag1];
		int idx2 = arrays.rtag[tag2];
		
		// get the number of bonds for each particle
		int num1 = m_bondlist[idx1];
		int num2 = m_bondlist[idx2];
		
		// add the new bonds to the table
		m_bondlist[(num1 + 1)*pitch + idx1] = idx2;
		m_bondlist[(num2 + 1)*pitch + idx2] = idx1;
		
		// increment the number of bonds
		m_bondlist[idx1]++;
		m_bondlist[idx2]++;
		}
	
	// copy the bond table to the device
	copyBondTable();
	
	m_pdata->release();
	}
	
/*! \param height New height for the bond table
	\post Reallocates memory on the device making room for up to 
		\a height bonds per particle.
	\note updateBondTable() needs to be called after so that the
		data in the bond table will be correct.
*/
void BondForceComputeGPU::reallocateBondTable(int height)
	{
	freeBondTable();
	allocateBondTable(height);
	}
	
/*! \param height New height for the bond table
*/
void BondForceComputeGPU::allocateBondTable(int height)
	{
	// make sure the arrays have been deallocated
	assert(m_bondlist == NULL);
	assert(m_gpu_bondtable.list == NULL);
	
	unsigned int N = m_pdata->getN();
	size_t pitch;
	
	// get the execution configuration
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
	
	// allocate and zero device memory
	exec_conf.gpu[0]->call(bind(cudaMallocPitch, (void**)((void*)&m_gpu_bondtable.list), &pitch, N*sizeof(unsigned int), height));
	
	// want pitch in elements, not bytes
	m_gpu_bondtable.pitch = (int)pitch / sizeof(int);
	m_gpu_bondtable.height = height;
	exec_conf.gpu[0]->call(bind(cudaMemset, (void*)m_gpu_bondtable.list, 0, pitch * height));
	
	// allocate and zero host memory
	exec_conf.gpu[0]->call(bind(cudaMallocHost, (void**)((void*)&m_bondlist), pitch * height));
	memset((void*)m_bondlist, 0, pitch*height);
	}

void BondForceComputeGPU::freeBondTable()
	{
	// get the execution configuration
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();	
	
	exec_conf.gpu[0]->call(bind(cudaFree, m_gpu_bondtable.list));
	m_gpu_bondtable.list = NULL;
	exec_conf.gpu[0]->call(bind(cudaFreeHost, m_bondlist));
	m_bondlist = NULL;
	}
		
//! Copies the bond table to the device
void BondForceComputeGPU::copyBondTable()
	{
	// get the execution configuration
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();	
	
	exec_conf.gpu[0]->call(bind(cudaMemcpy, m_gpu_bondtable.list, m_bondlist, 
			sizeof(unsigned int) * m_gpu_bondtable.height * m_gpu_bondtable.pitch, 
			cudaMemcpyHostToDevice) );
	}
	
/*! Internal method for computing the forces on the GPU. 
	\post The force data on the GPU is written with the calculated forces
	
	\param timestep Current time step of the simulation
*/
void BondForceComputeGPU::computeForces(unsigned int timestep)
	{
	// start the profile
	if (m_prof)
		{
		m_prof->push("Bond.GPU");
		m_prof->push("Table copy");
		}
		
	// the bond table on the GPU needs to be updated if we are dirty or the particles have been resorted
	if (m_dirty)
		{
		updateBondTable();
		m_dirty = false;
		}
	
	if (m_prof)
		{
		m_prof->pop();
		m_prof->push("Compute");
		}
		
	// get the execution configuration
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();

	// the bond table is up to date: we are good to go. Call the kernel
	gpu_pdata_arrays pdata = m_pdata->acquireReadOnlyGPU();
	gpu_boxsize box = m_pdata->getBoxGPU();
	
	exec_conf.gpu[0]->call(bind(gpu_bondforce_sum, m_d_forces, &pdata, &box, &m_gpu_bondtable, m_K, m_r_0, m_block_size));
		
	// the force data is now only up to date on the gpu
	m_data_location = gpu;
	
	if (m_prof)
		{
		exec_conf.gpu[0]->call(bind(cudaThreadSynchronize));
		m_prof->pop(34 * m_bonds.size()*2, 20*m_pdata->getN() + 20*m_bonds.size()*2);
		}
		
	m_pdata->release();
	
	if (m_prof)
		m_prof->pop();
	}
	
#ifdef USE_PYTHON
void export_BondForceComputeGPU()
	{
	class_<BondForceComputeGPU, boost::shared_ptr<BondForceComputeGPU>, bases<BondForceCompute>, boost::noncopyable >
		("BondForceComputeGPU", init< boost::shared_ptr<ParticleData>, Scalar, Scalar >())
		.def("setBlockSize", &BondForceComputeGPU::setBlockSize)
		;
	}
#endif

