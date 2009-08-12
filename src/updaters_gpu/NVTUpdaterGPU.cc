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
// Maintainer: joaander

/*! \file NVTUpdaterGPU.cc
	\brief Defines the NVTUpdaterGPU class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include "NVTUpdaterGPU.h"
#include "NVTRigidUpdaterGPU.cuh"
#include "NVTRigidUpdater.h"
#include "RigidData.h"

#include <boost/python.hpp>
using namespace boost::python;

#include <boost/bind.hpp>
using namespace boost;

using namespace std;

#ifdef ENABLE_CUDA
#include "gpu_settings.h"
#endif

/*! \param sysdef System data to update
	\param deltaT Time step to use
	\param tau NVT period
	\param T Temperature set point
*/
NVTUpdaterGPU::NVTUpdaterGPU(boost::shared_ptr<SystemDefinition> sysdef, Scalar deltaT, Scalar tau, boost::shared_ptr<Variant> T) : NVTUpdater(sysdef, deltaT, tau, T)
	{
	// at least one GPU is needed
	if (exec_conf.gpu.size() == 0)
		{
		cerr << endl << "***Error! Creating a NVTUpdaterGPU with no GPU in the execution configuration" << endl << endl;
		throw std::runtime_error("Error initializing NVTUpdaterGPU");
		}
	
	d_nvt_data.resize(exec_conf.gpu.size());
	d_nvt_rigid_data.resize(1);		// Rigid bodies now available on one GPU
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
		
		exec_conf.gpu[cur_gpu]->call(bind(cudaMallocHack, (void**)((void*)&d_nvt_data[cur_gpu].partial_Ksum), d_nvt_data[cur_gpu].NBlocks * sizeof(float)));
		exec_conf.gpu[cur_gpu]->call(bind(cudaMallocHack, (void**)((void*)&d_nvt_data[cur_gpu].Ksum), sizeof(float)));
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
	
	if (m_rigid_updater)
		{
		exec_conf.gpu[0]->call(bind(cudaFree, d_nvt_rigid_data[0].partial_Ksum_t));
		d_nvt_rigid_data[0].partial_Ksum_t = NULL;
		exec_conf.gpu[0]->call(bind(cudaFree, d_nvt_rigid_data[0].partial_Ksum_r));
		d_nvt_rigid_data[0].partial_Ksum_r = NULL;
		exec_conf.gpu[0]->call(bind(cudaFree, d_nvt_rigid_data[0].Ksum_t));
		d_nvt_rigid_data[0].Ksum_t = NULL;
		exec_conf.gpu[0]->call(bind(cudaFree, d_nvt_rigid_data[0].Ksum_r));
		d_nvt_rigid_data[0].Ksum_r = NULL;
		}
	}

/*! \param timestep Current time step of the simulation

	Calls gpu_nvt_pre_step, gpu_nvt_reduce_ksum, and gpu_nvt_step to do the dirty work.
*/
void NVTUpdaterGPU::update(unsigned int timestep)
	{
	assert(m_pdata);
	
	// get the rigid data from SystemDefinition
	boost::shared_ptr<RigidData> rigid_data = m_sysdef->getRigidData();
	
	// if there is any rigid body
	unsigned int n_bodies = rigid_data->getNumBodies();
	if (n_bodies > 0 && !m_rigid_updater) 
		m_rigid_updater = boost::shared_ptr<NVTRigidUpdater> (new NVTRigidUpdater(m_sysdef, m_deltaT, m_T));

	// if we haven't been called before, then the accelerations	have not been set and we need to calculate them
	if (!m_accel_set)
		{
		m_accel_set = true;
		// use the option of computeAccelerationsGPU to populate pdata.accel so the first step is
		// is calculated correctly
		computeAccelerationsGPU(timestep, "NVT", true);

		// compute the initial net forces, torques and angular momenta 
		if (m_rigid_updater) 
			{
			m_rigid_updater->setup();
			
			// Only one GPU
			d_nvt_rigid_data[0].n_bodies = n_bodies;
			
			exec_conf.gpu[0]->call(bind(cudaMallocHack, (void**)((void*)&d_nvt_rigid_data[0].partial_Ksum_t), n_bodies * sizeof(float)));
			exec_conf.gpu[0]->call(bind(cudaMallocHack, (void**)((void*)&d_nvt_rigid_data[0].partial_Ksum_r), n_bodies * sizeof(float)));
			exec_conf.gpu[0]->call(bind(cudaMallocHack, (void**)((void*)&d_nvt_rigid_data[0].Ksum_t), sizeof(float)));
			exec_conf.gpu[0]->call(bind(cudaMallocHack, (void**)((void*)&d_nvt_rigid_data[0].Ksum_r), sizeof(float)));
			}
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

	// pre-step kernel for rigid bodies
	if (m_rigid_updater && exec_conf.gpu.size() == 1) // only one GPU for the moment
		{
		ArrayHandle<Scalar> body_mass_handle(rigid_data->getBodyMass(), access_location::device, access_mode::read);
		ArrayHandle<Scalar4> moment_inertia_handle(rigid_data->getMomentInertia(), access_location::device, access_mode::read);
		ArrayHandle<Scalar4> com_handle(rigid_data->getCOM(), access_location::device, access_mode::readwrite);
		ArrayHandle<Scalar4> vel_handle(rigid_data->getVel(), access_location::device, access_mode::readwrite);
		ArrayHandle<Scalar4> angvel_handle(rigid_data->getAngVel(), access_location::device, access_mode::readwrite);
		ArrayHandle<Scalar4> angmom_handle(rigid_data->getAngMom(), access_location::device, access_mode::readwrite);
		ArrayHandle<Scalar4> orientation_handle(rigid_data->getOrientation(), access_location::device, access_mode::readwrite);
		ArrayHandle<Scalar4> ex_space_handle(rigid_data->getExSpace(), access_location::device, access_mode::readwrite);
		ArrayHandle<Scalar4> ey_space_handle(rigid_data->getEySpace(), access_location::device, access_mode::readwrite);
		ArrayHandle<Scalar4> ez_space_handle(rigid_data->getEzSpace(), access_location::device, access_mode::readwrite);
		ArrayHandle<int> body_imagex_handle(rigid_data->getBodyImagex(), access_location::device, access_mode::readwrite);
		ArrayHandle<int> body_imagey_handle(rigid_data->getBodyImagey(), access_location::device, access_mode::readwrite);
		ArrayHandle<int> body_imagez_handle(rigid_data->getBodyImagez(), access_location::device, access_mode::readwrite);
		ArrayHandle<Scalar4> particle_pos_handle(rigid_data->getParticlePos(), access_location::device, access_mode::read);
		ArrayHandle<unsigned int> particle_indices_handle(rigid_data->getParticleIndices(), access_location::device, access_mode::read);
		ArrayHandle<Scalar4> force_handle(rigid_data->getForce(), access_location::device, access_mode::read);
		ArrayHandle<Scalar4> torque_handle(rigid_data->getTorque(), access_location::device, access_mode::read);
		
		gpu_rigid_data_arrays d_rdata;
		d_rdata.n_bodies = rigid_data->getNumBodies();
		d_rdata.nmax = rigid_data->getNmax();
		d_rdata.local_beg = 0;
		d_rdata.local_num = d_rdata.n_bodies;
		d_rdata.body_mass = body_mass_handle.data;
		d_rdata.moment_inertia = moment_inertia_handle.data;
		d_rdata.com = com_handle.data;
		d_rdata.vel = vel_handle.data;
		d_rdata.angvel = angvel_handle.data;
		d_rdata.angmom = angmom_handle.data;
		d_rdata.orientation = orientation_handle.data;
		d_rdata.ex_space = ex_space_handle.data;
		d_rdata.ey_space = ey_space_handle.data;
		d_rdata.ez_space = ez_space_handle.data;
		d_rdata.body_imagex = body_imagex_handle.data;
		d_rdata.body_imagey = body_imagey_handle.data;
		d_rdata.body_imagez = body_imagez_handle.data;
		d_rdata.particle_pos = particle_pos_handle.data;
		d_rdata.particle_indices = particle_indices_handle.data;
		d_rdata.force = force_handle.data;
		d_rdata.torque = torque_handle.data;
		
		ArrayHandle<Scalar> eta_dot_t_handle(m_rigid_updater->getEtaDotT(), access_location::host, access_mode::read);
		ArrayHandle<Scalar> eta_dot_r_handle(m_rigid_updater->getEtaDotR(), access_location::host, access_mode::read);
		ArrayHandle<Scalar4> conjqm_handle(m_rigid_updater->getConjqm(), access_location::device, access_mode::readwrite);
	
		d_nvt_rigid_data[0].eta_dot_t0 = eta_dot_t_handle.data[0];
		d_nvt_rigid_data[0].eta_dot_r0 = eta_dot_r_handle.data[0];
		d_nvt_rigid_data[0].conjqm = conjqm_handle.data;

		exec_conf.gpu[0]->callAsync(bind(gpu_nvt_rigid_body_pre_step, d_pdata[0], d_rdata, box, d_nvt_rigid_data[0], m_deltaT));
		}

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
	m_curr_T = Ksum_total / m_dof;
	m_Xi += m_deltaT / (m_tau*m_tau) * (m_curr_T / m_T->getValue(timestep) - 1.0f);

	// update eta
	m_eta += m_deltaT / Scalar(2.0) * (m_Xi + xi_prev);
	
	// Update rigid body thermostats here
	float Ksum_t, Ksum_r;
	exec_conf.gpu[0]->callAsync(bind(gpu_nvt_rigid_reduce_ksum, d_nvt_rigid_data[0]));
		
	exec_conf.gpu[0]->call(bind(cudaMemcpy, &Ksum_t, d_nvt_rigid_data[0].Ksum_t, sizeof(float), cudaMemcpyDeviceToHost));
	exec_conf.gpu[0]->call(bind(cudaMemcpy, &Ksum_r, d_nvt_rigid_data[0].Ksum_r, sizeof(float), cudaMemcpyDeviceToHost));
	
	m_rigid_updater->updateThermostats(Ksum_t, Ksum_r, timestep);

	// get the particle data arrays again so we can update the 2nd half of the step
	d_pdata = m_pdata->acquireReadWriteGPU();
		

	exec_conf.tagAll(__FILE__, __LINE__);
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		exec_conf.gpu[cur_gpu]->callAsync(bind(gpu_nvt_step, d_pdata[cur_gpu], d_nvt_data[cur_gpu], m_d_force_data_ptrs[cur_gpu], (int)m_forces.size(), m_Xi, m_deltaT));

	// post-step kernel for rigid bodies
	if (m_rigid_updater && exec_conf.gpu.size() == 1) // only one GPU for the moment
		{
			
		ArrayHandle<Scalar> body_mass_handle(rigid_data->getBodyMass(), access_location::device, access_mode::read);
		ArrayHandle<Scalar4> moment_inertia_handle(rigid_data->getMomentInertia(), access_location::device, access_mode::read);
		ArrayHandle<Scalar4> com_handle(rigid_data->getCOM(), access_location::device, access_mode::read);
		ArrayHandle<Scalar4> vel_handle(rigid_data->getVel(), access_location::device, access_mode::readwrite);
		ArrayHandle<Scalar4> angvel_handle(rigid_data->getAngVel(), access_location::device, access_mode::readwrite);
		ArrayHandle<Scalar4> angmom_handle(rigid_data->getAngMom(), access_location::device, access_mode::readwrite);
		ArrayHandle<Scalar4> orientation_handle(rigid_data->getOrientation(), access_location::device, access_mode::read);
		ArrayHandle<Scalar4> ex_space_handle(rigid_data->getExSpace(), access_location::device, access_mode::read);
		ArrayHandle<Scalar4> ey_space_handle(rigid_data->getEySpace(), access_location::device, access_mode::read);
		ArrayHandle<Scalar4> ez_space_handle(rigid_data->getEzSpace(), access_location::device, access_mode::read);
		ArrayHandle<Scalar4> particle_pos_handle(rigid_data->getParticlePos(), access_location::device, access_mode::read);
		ArrayHandle<unsigned int> particle_indices_handle(rigid_data->getParticleIndices(), access_location::device, access_mode::read);
		ArrayHandle<Scalar4> force_handle(rigid_data->getForce(), access_location::device, access_mode::readwrite);
		ArrayHandle<Scalar4> torque_handle(rigid_data->getTorque(), access_location::device, access_mode::readwrite);
		
		gpu_rigid_data_arrays d_rdata;
		d_rdata.n_bodies = rigid_data->getNumBodies();
		d_rdata.nmax = rigid_data->getNmax();
		d_rdata.local_beg = 0;
		d_rdata.local_num = d_rdata.n_bodies;
		d_rdata.body_mass = body_mass_handle.data;
		d_rdata.moment_inertia = moment_inertia_handle.data;
		d_rdata.com = com_handle.data;
		d_rdata.vel = vel_handle.data;
		d_rdata.angvel = angvel_handle.data;
		d_rdata.angmom = angmom_handle.data;
		d_rdata.ex_space = ex_space_handle.data;
		d_rdata.ey_space = ey_space_handle.data;
		d_rdata.ez_space = ez_space_handle.data;
		d_rdata.orientation = orientation_handle.data;
		d_rdata.particle_pos = particle_pos_handle.data;
		d_rdata.particle_indices = particle_indices_handle.data;
		d_rdata.force = force_handle.data;
		d_rdata.torque = torque_handle.data;

		ArrayHandle<Scalar> eta_dot_t_handle(m_rigid_updater->getEtaDotT(), access_location::host, access_mode::read);
		ArrayHandle<Scalar> eta_dot_r_handle(m_rigid_updater->getEtaDotR(), access_location::host, access_mode::read);
		ArrayHandle<Scalar4> conjqm_handle(m_rigid_updater->getConjqm(), access_location::device, access_mode::readwrite);

		d_nvt_rigid_data[0].eta_dot_t0 = eta_dot_t_handle.data[0];
		d_nvt_rigid_data[0].eta_dot_r0 = eta_dot_r_handle.data[0];
		d_nvt_rigid_data[0].conjqm = conjqm_handle.data;

		exec_conf.gpu[0]->callAsync(bind(gpu_nvt_rigid_body_step, d_pdata[0], d_rdata, m_d_force_data_ptrs[0], (int)m_forces.size(), box, d_nvt_rigid_data[0], m_deltaT));
		}

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
		("NVTUpdaterGPU", init< boost::shared_ptr<SystemDefinition>, Scalar, Scalar, boost::shared_ptr<Variant> >())
		;
	}

#ifdef WIN32
#pragma warning( pop )
#endif

