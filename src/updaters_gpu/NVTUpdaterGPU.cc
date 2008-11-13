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

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include "NVTUpdaterGPU.h"

#include <boost/python.hpp>
using namespace boost::python;

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
	// at least one GPU is needed
	if (exec_conf.gpu.size() == 0)
		{
		cerr << endl << "***Error! Creating a NVTUpdaterGPU with no GPU in the execution configuration" << endl << endl;
		throw std::runtime_error("Error initializing NVTUpdaterGPU");
		}
	
	d_nvt_data.resize(exec_conf.gpu.size());
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
	// allocate the memory for the partial m*v^2 sums on each GPU	
	exec_conf.tagAll(__FILE__, __LINE__);
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		{
		d_nvt_data[cur_gpu].block_size = block_size;
		d_nvt_data[cur_gpu].NBlocks = m_pdata->getLocalNum(cur_gpu) / block_size + 1;
		
		exec_conf.gpu[cur_gpu]->call(bind(cudaMalloc, (void**)((void*)&d_nvt_data[cur_gpu].partial_Ksum), d_nvt_data[cur_gpu].NBlocks * sizeof(float)));
		exec_conf.gpu[cur_gpu]->call(bind(cudaMalloc, (void**)((void*)&d_nvt_data[cur_gpu].Ksum), sizeof(float)));
		}
	}

void NVTUpdaterGPU::freeNVTData()
	{
	// free the memory for the partial m*v^2 sums on each GPU	
	exec_conf.tagAll(__FILE__, __LINE__);
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		{
		exec_conf.gpu[cur_gpu]->call(bind(cudaFree, d_nvt_data[cur_gpu].partial_Ksum));
		d_nvt_data[cur_gpu].partial_Ksum = NULL;
		exec_conf.gpu[cur_gpu]->call(bind(cudaFree, d_nvt_data[cur_gpu].Ksum));
		d_nvt_data[cur_gpu].Ksum = NULL;
		}
	}

/*! \param timestep Current time step of the simulation

	Calls gpu_nvt_pre_step, gpu_nvt_reduce_ksum, and gpu_nvt_step to do the dirty work.
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
		computeAccelerationsGPU(timestep, "NVT", true);
		}

	if (m_prof) m_prof->push(exec_conf, "NVT");
		
	// access the particle data arrays
	vector<gpu_pdata_arrays>& d_pdata = m_pdata->acquireReadWriteGPU();
	gpu_boxsize box = m_pdata->getBoxGPU();

	if (m_prof) m_prof->push(exec_conf, "Half-step 1");
		
	// launch the pre-step kernel on all GPUs in parallel
	exec_conf.tagAll(__FILE__, __LINE__);
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		exec_conf.gpu[cur_gpu]->callAsync(bind(gpu_nvt_pre_step, d_pdata[cur_gpu], box, d_nvt_data[cur_gpu], m_Xi, m_deltaT));

	exec_conf.syncAll();
	
	if (m_prof) m_prof->pop(exec_conf, 36*m_pdata->getN(), 80 * m_pdata->getN());
	
	// release the particle data arrays so that they can be accessed to add up the accelerations
	m_pdata->release();
	
	// communicate the updated positions among the GPUs
	m_pdata->communicatePosition();
	
	// functions that computeAccelerations calls profile themselves, so suspend
	// the profiling for now
	if (m_prof) m_prof->pop(exec_conf);
	
	// for the next half of the step, we need the accelerations at t+deltaT
	computeAccelerationsGPU(timestep+1, "NVT", false);
	
	if (m_prof)
		{
		m_prof->push(exec_conf, "NVT");
		m_prof->push(exec_conf, "Reducing");
		}
		
	// reduce the Ksum values on all GPUs in parallel
	exec_conf.tagAll(__FILE__, __LINE__);
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		exec_conf.gpu[cur_gpu]->callAsync(bind(gpu_nvt_reduce_ksum, d_nvt_data[cur_gpu]));

	exec_conf.syncAll();
		
	// copy the values from the GPUs to the CPU and complete the sum
	float Ksum_total = 0.0f;
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		{
		float Ksum_tmp;
		exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, &Ksum_tmp, d_nvt_data[cur_gpu].Ksum, sizeof(float), cudaMemcpyDeviceToHost));
		Ksum_total += Ksum_tmp;
		}
		
	if (m_prof) m_prof->pop(exec_conf);
	if (m_prof) m_prof->push(exec_conf, "Half-step 2");
	
	// need previous xi to update eta
	Scalar xi_prev = m_Xi;	
	
	// update Xi
	m_curr_T = Ksum_total / (3.0f * float(m_pdata->getN()));
	m_Xi += m_deltaT / (m_tau*m_tau) * (m_curr_T / m_T - 1.0f);

	// update eta
	m_eta += m_deltaT / Scalar(2.0) * (m_Xi + xi_prev);
	
	// get the particle data arrays again so we can update the 2nd half of the step
	d_pdata = m_pdata->acquireReadWriteGPU();
	
	exec_conf.tagAll(__FILE__, __LINE__);
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		exec_conf.gpu[cur_gpu]->callAsync(bind(gpu_nvt_step, d_pdata[cur_gpu], d_nvt_data[cur_gpu], m_d_force_data_ptrs[cur_gpu], (int)m_forces.size(), m_Xi, m_deltaT));
	exec_conf.syncAll();
	
	m_pdata->release();
	
	// and now the acceleration at timestep+1 is precalculated for the first half of the next step
	if (m_prof)
		{
		m_prof->pop(exec_conf, 15 * m_pdata->getN(), m_pdata->getN() * 16 * (3 + m_forces.size()));
		m_prof->pop();
		}
	}
	
void export_NVTUpdaterGPU()
	{
	class_<NVTUpdaterGPU, boost::shared_ptr<NVTUpdaterGPU>, bases<NVTUpdater>, boost::noncopyable>
		("NVTUpdaterGPU", init< boost::shared_ptr<ParticleData>, Scalar, Scalar, Scalar >())
		;
	}

#ifdef WIN32
#pragma warning( pop )
#endif

