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

/*! \file Integrator.cc
	\brief Defines the Integrator base class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#ifdef USE_PYTHON
#include <boost/python.hpp>
using namespace boost::python;
#endif

#include "Integrator.h"

#include <boost/bind.hpp>
using namespace boost;

#ifdef USE_CUDA
#include "gpu_integrator.h"
#endif

using namespace std;

/*! \param pdata Particle data to update
	\param deltaT Time step to use
*/
Integrator::Integrator(boost::shared_ptr<ParticleData> pdata, Scalar deltaT) : Updater(pdata), m_deltaT(deltaT)
	{
	if (m_deltaT <= 0.0)
		cout << "***Warning! A timestep of less than 0.0 was specified to an integrator" << endl;

	#ifdef USE_CUDA
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();

	m_d_force_data_ptrs.resize(exec_conf.gpu.size());

	// allocate and initialize force data pointers (if running on a GPU)
	if (!exec_conf.gpu.empty())
		{
		for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
			{
			exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
			exec_conf.gpu[cur_gpu]->call(bind(cudaMalloc, (void **)((void *)&m_d_force_data_ptrs[cur_gpu]), sizeof(float4*)*32));
			exec_conf.gpu[cur_gpu]->call(bind(cudaMemset, (void*)m_d_force_data_ptrs[cur_gpu], 0, sizeof(float4*)*32));
			}
		}
	#endif
	}

Integrator::~Integrator()
	{
	#ifdef USE_CUDA
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
	if (!exec_conf.gpu.empty())
		{
		for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
			{
			exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
			exec_conf.gpu[cur_gpu]->call(bind(cudaFree, (void *)m_d_force_data_ptrs[cur_gpu]));
			}
		}
	#endif
	}

/*! \param fc ForceCompute to add	
*/
void Integrator::addForceCompute(boost::shared_ptr<ForceCompute> fc)
	{
	assert(fc);
	m_forces.push_back(fc);
	
	#ifdef USE_CUDA
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
	if (!exec_conf.gpu.empty())
		{
		for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
			{
			exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
			// reinitialize the memory on the device
		
			// fill out the memory on the host
			// this only needs to be done once since the output of acquireGPU is
			// guarunteed not to change later
			float4 *h_force_data_ptrs[32];
			for (int i = 0; i < 32; i++)
				h_force_data_ptrs[i] = NULL;
			
			for (unsigned int i = 0; i < m_forces.size(); i++)
				h_force_data_ptrs[i] = m_forces[i]->acquireGPU()[cur_gpu];
			
			exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, (void*)m_d_force_data_ptrs[cur_gpu], (void*)h_force_data_ptrs, sizeof(float4*)*32, cudaMemcpyHostToDevice));
			}
		}
	#endif
	}
	
/*! Call removeForceComputes() to completely wipe out the list of force computes
	that the integrator uses to sum forces.
*/
void Integrator::removeForceComputes()
	{
	m_forces.clear();
	
	#ifdef USE_CUDA
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
	if (!exec_conf.gpu.empty())
		{
		for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
			{
			exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
		
			// reinitialize the memory on the device
			float4 *h_force_data_ptrs[32];
			for (int i = 0; i < 32; i++)
				h_force_data_ptrs[i] = NULL;
			
			exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, (void*)m_d_force_data_ptrs[cur_gpu], (void*)h_force_data_ptrs, sizeof(float4*)*32, cudaMemcpyHostToDevice));
			}
		}
	#endif
	}
	
/*! \param deltaT New time step to set
*/
void Integrator::setDeltaT(Scalar deltaT)
	{
	if (m_deltaT <= 0.0)
		cout << "***Warning! A timestep of less than 0.0 was specified to an integrator" << endl;
	m_deltaT = deltaT;
	}

/*! \param timestep Current timestep
	\param profiler_name Name of the profiler element to continue timing under
	\post \c arrays.ax, \c arrays.ay, and \c arrays.az are set based on the forces computed by the ForceComputes
*/
void Integrator::computeAccelerations(unsigned int timestep, const std::string& profiler_name)
	{
	// this code is written in reduced units, so m=1. I set it here just in case the code is ever
	// modified to support other masses
	Scalar minv = 1.0;
	
	// compute the forces
	for (unsigned int i = 0; i < m_forces.size(); i++)
		{
		assert(m_forces[i]);
		m_forces[i]->compute(timestep);
		}

	if (m_prof)
		{
		m_prof->push(profiler_name);
		m_prof->push("Sum accel");
		}
		
	// now, get our own access to the arrays and add up the accelerations
	ParticleDataArrays arrays = m_pdata->acquireReadWrite();

	// start by zeroing the acceleration arrays
	memset((void *)arrays.ax, 0, sizeof(Scalar)*arrays.nparticles);
	memset((void *)arrays.ay, 0, sizeof(Scalar)*arrays.nparticles);
	memset((void *)arrays.az, 0, sizeof(Scalar)*arrays.nparticles);
	
	// now, add up the accelerations
	for (unsigned int i = 0; i < m_forces.size(); i++)
		{
		assert(m_forces[i]);
		ForceDataArrays force_arrays = m_forces[i]->acquire();
		
		for (unsigned int j = 0; j < arrays.nparticles; j++)
			{
			arrays.ax[j] += force_arrays.fx[j]*minv;
			arrays.ay[j] += force_arrays.fy[j]*minv;
			arrays.az[j] += force_arrays.fz[j]*minv;
			}
		}

	m_pdata->release();
	
	if (m_prof)
		{
		m_prof->pop(6*m_pdata->getN()*m_forces.size(), sizeof(Scalar)*3*m_pdata->getN()*(1+2*m_forces.size()));
		m_prof->pop();
		}
	}	

#ifdef USE_CUDA

/*! \param timestep Current timestep
	\param profiler_name Name of the profiler element to continue timing under
	\param sum_accel If set to true, forces will be summed into pdata.accel

	\post \c arrays.ax, \c arrays.ay, and \c arrays.az on the GPU are set based on the forces computed by the ForceComputes

	\note Setting sum_accel to true is convenient, but incurs an extra kernel call's overhead in a 
		performance hit. This is measured to be ~2% in real simulations. If at all possible,
		design the integrator to use sum_accel=false and perform the sum in the integrator using
		integrator_sum_forces_inline()
*/
void Integrator::computeAccelerationsGPU(unsigned int timestep, const std::string& profiler_name, bool sum_accel)
	{
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
	if (exec_conf.gpu.empty())
		{
		cerr << endl << "***Error! Integrator asked to compute GPU accelerations but there is no GPU in the execution configuration" << endl << endl;
		throw runtime_error("Error computing accelerations");
		}
	
	// compute the forces
	for (unsigned int i = 0; i < m_forces.size(); i++)
		{
		assert(m_forces[i]);
		m_forces[i]->compute(timestep);
		
		// acquire each computation on the GPU as we go
		m_forces[i]->acquireGPU();
		}

	// only perform the sum if requested
	if (sum_accel)
		{
		if (m_prof)
			{
			m_prof->push(profiler_name);
			m_prof->push("Sum accel");
			}
		
		// acquire the particle data on the GPU and add the forces into the acceleration
		vector<gpu_pdata_arrays>& d_pdata = m_pdata->acquireReadWriteGPU();

		// sum up all the forces
		for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
			{
			exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
			exec_conf.gpu[cur_gpu]->callAsync(bind(integrator_sum_forces, &d_pdata[cur_gpu], m_d_force_data_ptrs[cur_gpu], (int)m_forces.size()));
			}
			
		for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
			exec_conf.gpu[cur_gpu]->sync();
			
		// done
		m_pdata->release();
		
		if (m_prof)
			{
			exec_conf.gpu[0]->call(bind(cudaThreadSynchronize));
			m_prof->pop(6*m_pdata->getN()*m_forces.size(), sizeof(Scalar)*4*m_pdata->getN()*(1+m_forces.size()));
			m_prof->pop();
			}
		}
	}

#endif
		
/*! The base class integrator actually does nothing in update()
	\param timestep Current time step of the simulation
*/
void Integrator::update(unsigned int timestep)
	{
	}
	
#ifdef USE_PYTHON
void export_Integrator()
	{
	class_<Integrator, boost::shared_ptr<Integrator>, bases<Updater>, boost::noncopyable>
		("Integrator", init< boost::shared_ptr<ParticleData>, Scalar >())
		.def("addForceCompute", &Integrator::addForceCompute)
		.def("removeForceComputes", &Integrator::removeForceComputes)
		.def("setDeltaT", &Integrator::setDeltaT)
		;
	}
#endif

#ifdef WIN32
#pragma warning( pop )
#endif
