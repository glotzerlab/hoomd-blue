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

// $Id: FENEBondForceComputeGPU.cc 1131 2008-08-31 20:28:43Z phillicl $
// $URL: https://svn2.assembla.com/svn/hoomd/tags/hoomd-0.7.0/src/computes_gpu/FENEBondForceComputeGPU.cc $
// Maintainer: phillicl

/*! \file FENEBondForceComputeGPU.cc
	\brief Defines the FENEBondForceComputeGPU class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#include "FENEBondForceComputeGPU.h"

#include <boost/python.hpp>
using namespace boost::python;

#include <boost/bind.hpp>
using namespace boost;

using namespace std;

/*! \param sysdef System to compute bond forces on
*/
FENEBondForceComputeGPU::FENEBondForceComputeGPU(boost::shared_ptr<SystemDefinition> sysdef)
	: FENEBondForceCompute(sysdef), m_block_size(64)
	{
	// only one GPU is currently supported
	if (exec_conf.gpu.size() == 0)
		{
		cerr << endl << "***Error! Creating a FENEBondForceComputeGPU with no GPU in the execution configuration" << endl << endl;
		throw std::runtime_error("Error initializing FENEBondForceComputeGPU");
		}
	
	// allocate and zero device memory for K, R0 parameters
	m_gpu_params.resize(exec_conf.gpu.size());
	
	exec_conf.tagAll(__FILE__, __LINE__);
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		{
		exec_conf.gpu[cur_gpu]->call(bind(cudaMalloc, (void**)((void*)&m_gpu_params[cur_gpu]), m_bond_data->getNBondTypes()*sizeof(float4)));
		exec_conf.gpu[cur_gpu]->call(bind(cudaMemset, (void*)m_gpu_params[cur_gpu], 0, m_bond_data->getNBondTypes()*sizeof(float4)));
		}
	
	// allocate host memory for GPU parameters
	m_host_params = new float4[m_bond_data->getNBondTypes()];
	memset(m_host_params, 0, m_bond_data->getNBondTypes()*sizeof(float4));
	
	// allocate device memory for the radius error check parameters
	m_checkr.resize(exec_conf.gpu.size());
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		{
		exec_conf.gpu[cur_gpu]->call(bind(cudaMalloc, (void**)((void*)&m_checkr[cur_gpu]), sizeof(int)));
		exec_conf.gpu[cur_gpu]->call(bind(cudaMemset, (void*)m_checkr[cur_gpu], 0, sizeof(int)));
		}
	}
	
FENEBondForceComputeGPU::~FENEBondForceComputeGPU()
	{
	// free memory on the GPU
	exec_conf.tagAll(__FILE__, __LINE__);
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		{
		exec_conf.gpu[cur_gpu]->call(bind(cudaFree, (void*)m_gpu_params[cur_gpu]));
		m_gpu_params[cur_gpu] = NULL;
		
		exec_conf.gpu[cur_gpu]->call(bind(cudaFree, (void*)m_checkr[cur_gpu]));
		m_checkr[cur_gpu] = NULL;
		}
		
	// free memory on the CPU
	delete[] m_host_params;
	m_host_params = NULL;
	}

/*! \param type Type of the bond to set parameters for
	\param K Stiffness parameter for the force computation
	\param r_0 Equilibrium length for the force computation
	\param sigma  Particle diameter
	\param epsilon Determines hardness of the particles in the WCA part of the interaction (usually set to 1/T)
s	
	Sets parameters for the potential of a particular bond type and updates the 
	parameters on the GPU.
*/
void FENEBondForceComputeGPU::setParams(unsigned int type, Scalar K, Scalar r_0, Scalar sigma, Scalar epsilon)
	{
	FENEBondForceCompute::setParams(type, K, r_0, sigma, epsilon);
	
	// update the local copy of the memory
	m_host_params[type] = make_float4(K, r_0, sigma, epsilon);
	
	// copy the parameters to the GPU
	exec_conf.tagAll(__FILE__, __LINE__);
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, m_gpu_params[cur_gpu], m_host_params, m_bond_data->getNBondTypes()*sizeof(float4), cudaMemcpyHostToDevice));
	}

/*! Internal method for computing the forces on the GPU.
	\post The force data on the GPU is written with the calculated forces
	
	\param timestep Current time step of the simulation
	
	Calls gpu_compute_fene_bond_forces to do the dirty work.
*/
void FENEBondForceComputeGPU::computeForces(unsigned int timestep)
	{
	// start the profile
	if (m_prof) m_prof->push(exec_conf, "FENE");
	
	vector<gpu_bondtable_array>& gpu_bondtable = m_bond_data->acquireGPU();
	
	// the bond table is up to date: we are good to go. Call the kernel
	vector<gpu_pdata_arrays>& pdata = m_pdata->acquireReadOnlyGPU();
	gpu_boxsize box = m_pdata->getBoxGPU();
	
	// hackish method for tracking exceedsR0 over multiple GPUs
	unsigned int exceedsR0[8];
	if (exec_conf.gpu.size() > 8)
		{
		cerr << endl << "***Error! FENEBondForceGPU cannot run on more than 8 GPUs... sorry" << endl << endl;
		throw std::runtime_error("Error running FENEBondForceComputeGPU");
		}
			
	// run the kernel on each GPU in parallel	
	exec_conf.tagAll(__FILE__, __LINE__);
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		{
		exceedsR0[cur_gpu] = 0;
		exec_conf.gpu[cur_gpu]->callAsync(bind(gpu_compute_fene_bond_forces, m_gpu_forces[cur_gpu].d_data, pdata[cur_gpu], box, gpu_bondtable[cur_gpu], m_gpu_params[cur_gpu], m_checkr[cur_gpu], m_bond_data->getNBondTypes(), m_block_size, exceedsR0[cur_gpu]));
		}
	exec_conf.syncAll();
	
	//check the fene bondlength violation condition
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		{
		if (exceedsR0[cur_gpu])
			{
			cerr << endl << "***Error! FENE bond length exceeds maximum permitted" << endl << endl;
			throw std::runtime_error("Error in fene bond calculation");
			}
		}
		
	// the force data is now only up to date on the gpu
	m_data_location = gpu;
	
	m_pdata->release();
	
	int64_t mem_transfer = m_pdata->getN() * 4+16+20 + m_bond_data->getNumBonds() * 2 * (8+16+8);
	int64_t flops = m_bond_data->getNumBonds() * 2 * (3+12+8+6+7+3+7);
	if (m_prof) m_prof->pop(exec_conf, flops, mem_transfer);
	}
	
void export_FENEBondForceComputeGPU()
	{
	class_<FENEBondForceComputeGPU, boost::shared_ptr<FENEBondForceComputeGPU>, bases<FENEBondForceCompute>, boost::noncopyable >
		("FENEBondForceComputeGPU", init< boost::shared_ptr<SystemDefinition> >())
		.def("setBlockSize", &FENEBondForceComputeGPU::setBlockSize)
		;
	}

#ifdef WIN32
#pragma warning( pop )
#endif
