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

/*! \file NPTUpdaterGPU.cc
	\brief Defines the NPTUpdaterGPU class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include "NPTUpdaterGPU.h"

#include <boost/python.hpp>
using namespace boost::python;

#include <boost/bind.hpp>
using namespace boost;

using namespace std;

/*! \param pdata Particle data to update
	\param deltaT Time step to use
	\param tau Nose-Hoover period
	\param tauP barostat period
	\param T Temperature set point
	\param P Pressure set point
*/
NPTUpdaterGPU::NPTUpdaterGPU(boost::shared_ptr<ParticleData> pdata, Scalar deltaT, Scalar tau, Scalar tauP, Scalar T, Scalar P) : NPTUpdater(pdata, deltaT, tau, tauP, T, P)
	{
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();

	m_d_virial_data_ptrs.resize(exec_conf.gpu.size());
	// allocate and initialize force data pointers (if running on a GPU)
	if (!exec_conf.gpu.empty())
		{
		for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
			{
			exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
			exec_conf.gpu[cur_gpu]->call(bind(cudaMalloc, (void **)((void *)&m_d_virial_data_ptrs[cur_gpu]), sizeof(float*)*32));
			exec_conf.gpu[cur_gpu]->call(bind(cudaMemset, (void*)m_d_virial_data_ptrs[cur_gpu], 0, sizeof(float*)*32));
			}
		}

	// at least one GPU is needed
	if (exec_conf.gpu.size() == 0)
		{
		cerr << endl << "***Error! Creating a NPTUpdaterGPU with no GPU in the execution configuration." << endl << endl;
		throw std::runtime_error("Error initializing NPTUpdaterGPU.");
		}
	
	d_npt_data.resize(exec_conf.gpu.size());
	allocateNPTData(128);
	}

NPTUpdaterGPU::~NPTUpdaterGPU()	
	{
	freeNPTData();
	}

/*! \param block_size block size to allocate data for
*/
void NPTUpdaterGPU::allocateNPTData(int block_size)
	{
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
	vector<gpu_pdata_arrays>& d_pdata = m_pdata->acquireReadWriteGPU();
	int local_num;

	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		{
		d_npt_data[cur_gpu].block_size = block_size;
		d_npt_data[cur_gpu].NBlocks = m_pdata->getLocalNum(cur_gpu) / block_size + 1;
		
		exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
		exec_conf.gpu[cur_gpu]->call(bind(cudaMalloc, (void**)((void*)&d_npt_data[cur_gpu].partial_Ksum), d_npt_data[cur_gpu].NBlocks * sizeof(float)));
		exec_conf.gpu[cur_gpu]->call(bind(cudaMalloc, (void**)((void*)&d_npt_data[cur_gpu].Ksum), sizeof(float)));
		exec_conf.gpu[cur_gpu]->call(bind(cudaMalloc, (void**)((void*)&d_npt_data[cur_gpu].partial_Wsum), d_npt_data[cur_gpu].NBlocks * sizeof(float)));
		exec_conf.gpu[cur_gpu]->call(bind(cudaMalloc, (void**)((void*)&d_npt_data[cur_gpu].Wsum), sizeof(float)));
		local_num = d_pdata[cur_gpu].local_num;
		exec_conf.gpu[cur_gpu]->call(bind(cudaMalloc, (void**)((void*)&d_npt_data[cur_gpu].virial), local_num * sizeof(float)));

		}
	m_pdata->release();

	}

// frees memory

void NPTUpdaterGPU::freeNPTData()
	{
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();

	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		{
		exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
		exec_conf.gpu[cur_gpu]->call(bind(cudaFree, d_npt_data[cur_gpu].partial_Ksum));
		d_npt_data[cur_gpu].partial_Ksum = NULL;
		exec_conf.gpu[cur_gpu]->call(bind(cudaFree, d_npt_data[cur_gpu].Ksum));
		d_npt_data[cur_gpu].Ksum = NULL;
		exec_conf.gpu[cur_gpu]->call(bind(cudaFree, d_npt_data[cur_gpu].partial_Wsum));
		d_npt_data[cur_gpu].partial_Wsum = NULL;
		exec_conf.gpu[cur_gpu]->call(bind(cudaFree, d_npt_data[cur_gpu].Wsum));
		d_npt_data[cur_gpu].Wsum = NULL;
		exec_conf.gpu[cur_gpu]->call(bind(cudaFree, d_npt_data[cur_gpu].virial));
		d_npt_data[cur_gpu].virial = NULL;
		exec_conf.gpu[cur_gpu]->call(bind(cudaFree, (void *)m_d_virial_data_ptrs[cur_gpu]));
		}
	}

/*! Calls parent Integrator::addForceCompute before setting up virial compute

    \param fc ForceCompute to add	

*/
void NPTUpdaterGPU::addForceCompute(boost::shared_ptr<ForceCompute> fc)
       {
	 Integrator::addForceCompute(fc);
	 // add stuff for virials
	#ifdef ENABLE_CUDA
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
	if (!exec_conf.gpu.empty())
		{
		for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
			{
			exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
			// reinitialize the memory on the device
		
			// fill out the memory on the host
			// this only needs to be done once since the output of acquireGPU is
			// guaranteed not to change later
			float *h_virial_data_ptrs[32];
			for (int i = 0; i < 32; i++)
				h_virial_data_ptrs[i] = NULL;
			
			for (unsigned int i = 0; i < m_forces.size(); i++)
				h_virial_data_ptrs[i] = m_forces[i]->acquireGPU()[cur_gpu].d_data.virial;
			
			exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, (void*)m_d_virial_data_ptrs[cur_gpu], (void*)h_virial_data_ptrs, sizeof(float*)*32, cudaMemcpyHostToDevice));
			}
		}
	#endif
	}

/*! Call removeForceComputes() to completely wipe out the list of force computes
	that the integrator uses to sum forces.
	Removes virial compute.
*/
void NPTUpdaterGPU::removeForceComputes()
	{
	#ifdef ENABLE_CUDA
	 
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
	if (!exec_conf.gpu.empty())
		{
		for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
			{
			exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
		 
			// reinitialize the memory on the device
			float *h_virial_data_ptrs[32];
			for (int i = 0; i < 32; i++)
				h_virial_data_ptrs[i] = NULL;
			
			exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, (void*)m_d_virial_data_ptrs[cur_gpu], (void*)h_virial_data_ptrs, sizeof(float*)*32, cudaMemcpyHostToDevice));
			}
		}
	
	#endif
	Integrator::removeForceComputes();
	}


/*! \param timestep Current time step of the simulation
*/
void NPTUpdaterGPU::update(unsigned int timestep)
	{
	assert(m_pdata);
	int N = m_pdata->getN();
		
	// if we haven't been called before, then the accelerations
	// have not been set and we need to calculate them
	if (!m_accel_set)
		{
		m_accel_set = true;
		// use the option of computeAccelerationsGPU to populate pdata.accel so the first step is
		// is calculated correctly
		computeAccelerationsGPU(timestep, "NPT", true);
		
		m_curr_T = computeTemperature(timestep);  // Compute temperature for the first time step
		m_curr_P = computePressure(timestep);     // Compute pressure for the first time step
		}

	if (m_prof) m_prof->push(exec_conf, "NPT");
		
	// access the particle data arrays
	vector<gpu_pdata_arrays>& d_pdata = m_pdata->acquireReadWriteGPU();
	gpu_boxsize box = m_pdata->getBoxGPU();

	if (m_prof) m_prof->push(exec_conf, "Half-step 1");
		
	// advance thermostat (m_Xi) half a time step
	m_Xi += (1.0f/2.0f)/(m_tau*m_tau)*(m_curr_T/m_T - 1.0f)*m_deltaT;

	// advance barostat (m_Eta) half time step
	m_Eta += (1.0f/2.0f)/(m_tauP*m_tauP)*m_V/(N*m_T)*(m_curr_P - m_P)*m_deltaT;

	
	// perform first half of the time step; propagate velocities for 1/2*deltaT and
        // positions for full deltaT
	
	exec_conf.tagAll(__FILE__, __LINE__);
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		{
		exec_conf.gpu[cur_gpu]->callAsync(bind(gpu_npt_pre_step, d_pdata[cur_gpu], box, d_npt_data[cur_gpu], m_Xi, m_Eta, m_deltaT));
		}
	exec_conf.syncAll();
	
	if (m_prof) m_prof->pop(exec_conf, 36*m_pdata->getN(), 80 * m_pdata->getN());

	// advance volume 

	m_V *= exp(3.0f*m_Eta*m_deltaT);

	// rescale box length
	float box_len_scale = exp(m_Eta*m_deltaT);
	m_Lx *= box_len_scale;
	m_Ly *= box_len_scale;
	m_Lz *= box_len_scale;

	// release the particle data arrays so that they can be accessed to add up the accelerations
	m_pdata->release();

	// rescale simulation box
	m_pdata->setBox(BoxDim(m_Lx, m_Ly, m_Lz));
	
	// communicate the updated positions among the GPUs
	m_pdata->communicatePosition();
	
	// functions that computeAccelerations calls profile themselves, so suspend
	// the profiling for now
	if (m_prof) m_prof->pop(exec_conf);
	
	// for the next half of the step, we need the accelerations at t+deltaT
	computeAccelerationsGPU(timestep+1, "NPT", false);
	
	if (m_prof) m_prof->push(exec_conf, "NPT");

	// compute temperature for the next half time step
	m_curr_T = computeTemperature(timestep+1);
	// compute pressure for the next half time step
	m_curr_P = computePressure(timestep+1);
	
	if (m_prof) m_prof->push(exec_conf, "Half-step 2");
	
	// get the particle data arrays again so we can update the 2nd half of the step
	d_pdata = m_pdata->acquireReadWriteGPU();
	
	// 2nd half time step; propagate velocities from t+1/2*deltaT to t+deltaT
	exec_conf.tagAll(__FILE__, __LINE__);
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		exec_conf.gpu[cur_gpu]->callAsync(bind(gpu_npt_step, d_pdata[cur_gpu], d_npt_data[cur_gpu], m_d_force_data_ptrs[cur_gpu], (int)m_forces.size(), m_Xi, m_Eta, m_deltaT));
	exec_conf.syncAll();
		
	m_pdata->release();
	
	// and now the acceleration at timestep+1 is precalculated for the first half of the next step
	if (m_prof)
		{
		m_prof->pop(exec_conf, 15 * m_pdata->getN(), m_pdata->getN() * 16 * (3 + m_forces.size()));
		m_prof->pop();
		}
	
	// Update barostat variable m_Eta to t+deltaT
	m_Eta += (1.0f/2.0f)/(m_tauP*m_tauP)*m_V/(N*m_T)*(m_curr_P - m_P)*m_deltaT;

	// Update thermostat variable m_Xi to t+deltaT
	m_Xi += (1.0f/2.0f)/(m_tau*m_tau)*(m_curr_T/m_T - 1.0f)*m_deltaT;
	}

/*! \param timestep Current time step of the simulation
	Calculates current temperature of the system
    \returns current temperature of the system
 */
Scalar NPTUpdaterGPU::computeTemperature(unsigned int timestep)
	{
        // acquire the particle data on the GPU
	vector<gpu_pdata_arrays>& d_pdata = m_pdata->acquireReadWriteGPU();
	float g = 3.0f*m_pdata->getN(); // Number of degrees of freedom g = 3*N
	
	if (m_prof) m_prof->push(exec_conf, "Compute Temp");
		
	// compute the Ksum values on each GPU in parallel
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		{
		exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
		exec_conf.gpu[cur_gpu]->callAsync(bind(gpu_npt_temperature, d_npt_data[cur_gpu], d_pdata[cur_gpu]));
		exec_conf.gpu[cur_gpu]->callAsync(bind(gpu_npt_reduce_ksum, d_npt_data[cur_gpu]));
		}
	exec_conf.syncAll();
	
	// copy the values from the GPU to the CPU and complete the sum
	float Ksum_total = 0.0f;
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
	    {
		float Ksum_tmp;
		exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, &Ksum_tmp, d_npt_data[cur_gpu].Ksum, sizeof(float), cudaMemcpyDeviceToHost));
		Ksum_total += Ksum_tmp;
		}

	m_pdata->release();
		
	if (m_prof) m_prof->pop(exec_conf);
	
	return Ksum_total / g;
	}

/*! \param timestep Current time step of the simulation
	Calculates current pressure of the system
    \returns current pressure of the system
 */
Scalar NPTUpdaterGPU::computePressure(unsigned int timestep)
	{
	if (m_prof) m_prof->push("Compute Press");
	
	// Number of particles
	unsigned int N = m_pdata->getN();
	
	// acquire the particle data on the GPU 
	vector<gpu_pdata_arrays>& d_pdata = m_pdata->acquireReadWriteGPU();
	
	exec_conf.tagAll(__FILE__, __LINE__);
	
	// sum up virials and then total the Wsum on each GPU in parallel
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		{
		exec_conf.gpu[cur_gpu]->callAsync(bind(gpu_integrator_sum_virials, d_npt_data[cur_gpu], d_pdata[cur_gpu], m_d_virial_data_ptrs[cur_gpu], (int)m_forces.size()));
		
		exec_conf.gpu[cur_gpu]->callAsync(bind(gpu_npt_pressure, d_npt_data[cur_gpu], d_pdata[cur_gpu]));
		
		exec_conf.gpu[cur_gpu]->callAsync(bind(gpu_npt_reduce_wsum, d_npt_data[cur_gpu]));
		}
	exec_conf.syncAll();
	
	// done with the particle data
	m_pdata->release();
	
	// copy the values from the GPU to the CPU and complete the sum
	float Wsum_total = 0.0f;
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		{
		float Wsum_tmp;
		exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, &Wsum_tmp, d_npt_data[cur_gpu].Wsum, sizeof(float), cudaMemcpyDeviceToHost));
		
		Wsum_total += Wsum_tmp;
		}
		
	if (m_prof) m_prof->pop();

	return (N * m_curr_T + Wsum_total)/m_V; // return presssure PV = N*T + W
	}
	
void export_NPTUpdaterGPU()
	{
	class_<NPTUpdaterGPU, boost::shared_ptr<NPTUpdaterGPU>, bases<NPTUpdater>, boost::noncopyable>
	  ("NPTUpdaterGPU", init< boost::shared_ptr<ParticleData>, Scalar, Scalar, Scalar, Scalar, Scalar >())
		;
	}

#ifdef WIN32
#pragma warning( pop )
#endif

