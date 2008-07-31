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

/*! \file NVTUpdaterGPU.cc
	\brief Defines the NVTUpdaterGPU class
*/

#include "NVTUpdaterGPU.h"
#include "gpu_updaters.h"

#ifdef USE_PYTHON
#include <boost/python.hpp>
using namespace boost::python;
#endif

#include <boost/bind.hpp>
using namespace boost;

using namespace std;

/*! \param pdata Particle data to update
	\param deltaT Time step to use
	\param tau NVT period
	\param T Temperature set point
*/
NVTUpdaterGPU::NVTUpdaterGPU(boost::shared_ptr<ParticleData> pdata, Scalar deltaT, Scalar tau, Scalar T) : NVTUpdater(pdata, deltaT, tau, T)
	{
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
	// at least one GPU is needed
	if (exec_conf.gpu.size() == 0)
		{
		cout << "Creating a NVTUpdaterGPU with no GPU in the execution configuration" << endl;
		throw std::runtime_error("Error initializing NVTUpdaterGPU");
		}
		
	allocateNVTData(128);
	}

NVTUpdaterGPU::~NVTUpdaterGPU()
	{
	freeNVTData();
	}

/*! \param block_size block size to allocate data for
*/
void NVTUpdaterGPU::allocateNVTData(int block_size)
	{
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
	
	d_nvt_data.block_size = block_size;
	d_nvt_data.NBlocks = m_pdata->getN() / block_size + 1;
	
	exec_conf.gpu[0]->setTag(__FILE__, __LINE__);
	exec_conf.gpu[0]->call(bind(cudaMalloc, (void**)((void*)&d_nvt_data.partial_Ksum), d_nvt_data.NBlocks * sizeof(float)));
	exec_conf.gpu[0]->call(bind(cudaMalloc, (void**)((void*)&d_nvt_data.Ksum), sizeof(float)));
	exec_conf.gpu[0]->call(bind(cudaMalloc, (void**)((void*)&d_nvt_data.Xi), sizeof(float)));
	exec_conf.gpu[0]->call(bind(cudaMalloc, (void**)((void*)&d_nvt_data.Xi_dbl), sizeof(float)));

	// initialize Xi to 1.0
	float Xi = 1.0;
	exec_conf.gpu[0]->call(bind(cudaMemcpy, d_nvt_data.Xi, &Xi, sizeof(float), cudaMemcpyHostToDevice));
	}

void NVTUpdaterGPU::freeNVTData()
	{
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
	
	exec_conf.gpu[0]->setTag(__FILE__, __LINE__);
	exec_conf.gpu[0]->call(bind(cudaFree, d_nvt_data.partial_Ksum));
	d_nvt_data.partial_Ksum = NULL;
	exec_conf.gpu[0]->call(bind(cudaFree, d_nvt_data.Ksum));
	d_nvt_data.Ksum = NULL;
	exec_conf.gpu[0]->call(bind(cudaFree, d_nvt_data.Xi));
	d_nvt_data.Xi = NULL;
	exec_conf.gpu[0]->call(bind(cudaFree, d_nvt_data.Xi_dbl));
	d_nvt_data.Xi_dbl = NULL;
	}

/*! \param timestep Current time step of the simulation
*/
void NVTUpdaterGPU::update(unsigned int timestep)
	{
	assert(m_pdata);
	
	// if we haven't been called before, then the accelerations	have not been set and we need to calculate them
	if (!m_accel_set)
		{
		m_accel_set = true;
		// use the option of computeAccelerationsGPU to populate pdata.accel so the first step is
		// is calculated correctly
		computeAccelerationsGPU(timestep, "NVT.GPU", true);
		}

	if (m_prof)
		m_prof->push("NVT.GPU");
		
	// access the particle data arrays
	gpu_pdata_arrays d_pdata = m_pdata->acquireReadWriteGPU();
	gpu_boxsize box = m_pdata->getBoxGPU(); 

	if (m_prof)
		m_prof->push("Half-step 1");
		
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
	exec_conf.gpu[0]->setTag(__FILE__, __LINE__);
	exec_conf.gpu[0]->call(bind(nvt_pre_step, &d_pdata, &box, &d_nvt_data, m_deltaT));
	
	if (m_prof)
		{
		exec_conf.gpu[0]->call(bind(cudaThreadSynchronize));
		m_prof->pop(36*m_pdata->getN(), 80 * m_pdata->getN());
		}
	
	// release the particle data arrays so that they can be accessed to add up the accelerations
	m_pdata->release();
	
	// functions that computeAccelerations calls profile themselves, so suspend
	// the profiling for now
	if (m_prof)
		m_prof->pop();
	
	// for the next half of the step, we need the accelerations at t+deltaT
	computeAccelerationsGPU(timestep+1, "NVT.GPU", false);
	
	if (m_prof)
		{
		m_prof->push("NVT.GPU");
		m_prof->push("Reducing");
		}
		
	exec_conf.gpu[0]->setTag(__FILE__, __LINE__);
	exec_conf.gpu[0]->call(bind(nvt_reduce_ksum, &d_nvt_data));
	
	if (m_prof)
		{
		exec_conf.gpu[0]->call(bind(cudaThreadSynchronize));
		m_prof->pop();
		}
	
	if (m_prof)
		m_prof->push("Half-step 2");
	
	// get the particle data arrays again so we can update the 2nd half of the step
	d_pdata = m_pdata->acquireReadWriteGPU();
	exec_conf.gpu[0]->setTag(__FILE__, __LINE__);
	exec_conf.gpu[0]->call(bind(nvt_step, &d_pdata, &d_nvt_data, m_d_force_data_ptrs, (int)m_forces.size(), m_deltaT, m_tau, m_T));
	m_pdata->release();
	
	// and now the acceleration at timestep+1 is precalculated for the first half of the next step
	if (m_prof)
		{
		exec_conf.gpu[0]->call(bind(cudaThreadSynchronize));
		m_prof->pop(15 * m_pdata->getN(), m_pdata->getN() * 16 * (3 + m_forces.size()));	
		m_prof->pop();
		}
		
	exec_conf.gpu[0]->sync();
	}
	
#ifdef USE_PYTHON
void export_NVTUpdaterGPU()
	{
	class_<NVTUpdaterGPU, boost::shared_ptr<NVTUpdaterGPU>, bases<NVTUpdater>, boost::noncopyable>
		("NVTUpdaterGPU", init< boost::shared_ptr<ParticleData>, Scalar, Scalar, Scalar >())
		;
	}
#endif

