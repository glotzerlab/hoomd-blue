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

/*! \file HarmonicBondForceComputeGPU.cc
	\brief Defines the HarmonicBondForceComputeGPU class
*/

#include "HarmonicBondForceComputeGPU.h"

#ifdef USE_PYTHON
#include <boost/python.hpp>
using namespace boost::python;
#endif

#include <boost/bind.hpp>
using namespace boost;

using namespace std;

/*! \param pdata ParticleData to compute bond forces on
*/
HarmonicBondForceComputeGPU::HarmonicBondForceComputeGPU(boost::shared_ptr<ParticleData> pdata)
	: HarmonicBondForceCompute(pdata), m_block_size(64)
	{
	// check the execution configuration
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
	// only one GPU is currently supported
	if (exec_conf.gpu.size() == 0)
		{
		cerr << endl << "***Error! Creating a BondForceComputeGPU with no GPU in the execution configuration" << endl << endl;
		throw std::runtime_error("Error initializing BondForceComputeGPU");
		}
	if (exec_conf.gpu.size() != 1)
		{
		cerr << endl << "***Error! More than one GPU is not currently supported" << endl << endl;
		throw std::runtime_error("Error initializing BondForceComputeGPU");
		}
	
	// allocate and zero device memory
	exec_conf.gpu[0]->setTag(__FILE__, __LINE__);
	exec_conf.gpu[0]->call(bind(cudaMalloc, (void**)((void*)&m_gpu_params), m_bond_data->getNBondTypes()*sizeof(float2)));
	exec_conf.gpu[0]->call(bind(cudaMemset, (void*)m_gpu_params, 0, m_bond_data->getNBondTypes()*sizeof(float2)));
	
	m_host_params = new float2[m_bond_data->getNBondTypes()];
	memset(m_host_params, 0, m_bond_data->getNBondTypes()*sizeof(float2));
	}
	
HarmonicBondForceComputeGPU::~HarmonicBondForceComputeGPU()
	{
	// free memory on the GPU
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
	exec_conf.gpu[0]->setTag(__FILE__, __LINE__);
	exec_conf.gpu[0]->call(bind(cudaFree, (void*)m_gpu_params));
	m_gpu_params = NULL;
	
	// free memory on the CPU
	delete[] m_host_params;
	m_host_params = NULL;
	}

/*! \param type Type of the bond to set parameters for
	\param K Stiffness parameter for the force computation
	\param r_0 Equilibrium length for the force computation
	
	Sets parameters for the potential of a particular bond type and updates the 
	parameters on the GPU.
*/
void HarmonicBondForceComputeGPU::setParams(unsigned int type, Scalar K, Scalar r_0)
	{
	HarmonicBondForceCompute::setParams(type, K, r_0);
	
	// update the local copy of the memory
	m_host_params[type] = make_float2(K, r_0);
	
	// copy the parameters to the GPU
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
	exec_conf.gpu[0]->setTag(__FILE__, __LINE__);
	exec_conf.gpu[0]->call(bind(cudaMemcpy, m_gpu_params, m_host_params, m_bond_data->getNBondTypes()*sizeof(float2), cudaMemcpyHostToDevice));
	}

/*! Internal method for computing the forces on the GPU. 
	\post The force data on the GPU is written with the calculated forces
	
	\param timestep Current time step of the simulation
*/
void HarmonicBondForceComputeGPU::computeForces(unsigned int timestep)
	{
	// start the profile
	if (m_prof)
		{
		m_prof->push("Bond.GPU");
		m_prof->push("Table copy");
		}
		
	gpu_bondtable_array gpu_bondtable = m_bond_data->acquireGPU();
	
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
	
	exec_conf.gpu[0]->setTag(__FILE__, __LINE__);
	exec_conf.gpu[0]->call(bind(gpu_bondforce_sum, m_d_forces, &pdata, &box, &gpu_bondtable, m_gpu_params, m_bond_data->getNBondTypes(), m_block_size));
		
	// the force data is now only up to date on the gpu
	m_data_location = gpu;
	
	if (m_prof)
		{
		exec_conf.gpu[0]->call(bind(cudaThreadSynchronize));
		m_prof->pop(34 * m_bond_data->getNumBonds()*2, 20*m_pdata->getN() + 20*m_bond_data->getNumBonds()*2);
		}
		
	m_pdata->release();
	
	if (m_prof)
		m_prof->pop();
	}
	
#ifdef USE_PYTHON
void export_HarmonicBondForceComputeGPU()
	{
	class_<HarmonicBondForceComputeGPU, boost::shared_ptr<HarmonicBondForceComputeGPU>, bases<HarmonicBondForceCompute>, boost::noncopyable >
		("HarmonicBondForceComputeGPU", init< boost::shared_ptr<ParticleData> >())
		.def("setBlockSize", &HarmonicBondForceComputeGPU::setBlockSize)
		;
	}
#endif

