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

/*! \file NVEUpdaterGPU.cc
	\brief Defines the NVEUpdaterGPU class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include "NVEUpdaterGPU.h"
#include "NVEUpdaterGPU.cuh"
#include "NVERigidUpdaterGPU.cuh"

#include "NVERigidUpdater.h"
#include "RigidData.h"

#include <boost/bind.hpp>
using namespace boost;

#include <boost/python.hpp>
using namespace boost::python;

using namespace std;

/*! \param sysdef System to update
	\param deltaT Time step to use
*/
NVEUpdaterGPU::NVEUpdaterGPU(boost::shared_ptr<SystemDefinition> sysdef, Scalar deltaT) : NVEUpdater(sysdef, deltaT)
	{
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
	// at least one GPU is needed
	if (exec_conf.gpu.size() == 0)
		{
		cerr << endl << "***Error! Creating a NVEUpdaterGPU with no GPU in the execution configuration" << endl << endl;
		throw std::runtime_error("Error initializing NVEUpdaterGPU");
		}
	}

/*! \param timestep Current time step of the simulation

	Calls gpu_nve_pre_step and gpu_nve_step to do the dirty work.
*/
void NVEUpdaterGPU::update(unsigned int timestep)
{
	assert(m_pdata);

	// get the rigid data from SystemDefinition
	boost::shared_ptr<RigidData> rigid_data = m_sysdef->getRigidData();
	
	// if there is any rigid body and the flag has not yet been set
	static bool has_rigid_bodies = false;
	if (rigid_data->getNumBodies() > 0 && has_rigid_bodies == false) 
		{
		m_rigid_updater = boost::shared_ptr<NVERigidUpdater> (new NVERigidUpdater(m_sysdef, m_deltaT));
		// set the flag
		has_rigid_bodies = true;
		}
	
	// if we haven't been called before, then the accelerations	have not been set and we need to calculate them
	if (!m_accel_set)
		{
		m_accel_set = true;
		// use the option of computeAccelerationsGPU to populate pdata.accel so the first step is
		// is calculated correctly
		computeAccelerationsGPU(timestep, "NVE", true);
			
		// rigid body setup: compute initial body forces and torques: do the similar thing as to NVEUpdater?? call NVERigidUpdater::setup?
		if (has_rigid_bodies) m_rigid_updater->setup();
		}

	if (m_prof)
		m_prof->push(exec_conf, "NVE");
		
	// access the particle data arrays
	vector<gpu_pdata_arrays>& d_pdata = m_pdata->acquireReadWriteGPU();
	gpu_boxsize box = m_pdata->getBoxGPU();
		
	if (m_prof) m_prof->push(exec_conf, "Half-step 1");
	
	// call the pre-step kernel on all GPUs in parallel
	exec_conf.tagAll(__FILE__, __LINE__);
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		exec_conf.gpu[cur_gpu]->call(bind(gpu_nve_pre_step, d_pdata[cur_gpu], box, m_deltaT, m_limit, m_limit_val));

	// pre-step kernel for rigid bodies
	if (has_rigid_bodies && exec_conf.gpu.size() == 1) // only one GPU for the moment
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
		
		exec_conf.gpu[0]->call(bind(gpu_nve_rigid_body_pre_step, d_pdata[0], d_rdata, box, m_deltaT, m_limit, m_limit_val));
		}
	
	exec_conf.syncAll();
	
	uint64_t mem_transfer = m_pdata->getN() * (16+32+16+48);
	uint64_t flops = m_pdata->getN() * (15+3+9+15);
	if (m_prof) m_prof->pop(exec_conf, flops, mem_transfer);
	
	// release the particle data arrays so that they can be accessed to add up the accelerations
	m_pdata->release();
	
	// communicate the updated positions among the GPUs
	m_pdata->communicatePosition();
	
	// functions that computeAccelerations calls profile themselves, so suspend
	// the profiling for now
	if (m_prof) m_prof->pop(exec_conf);
	
	// for the next half of the step, we need the accelerations at t+deltaT
	computeAccelerationsGPU(timestep+1, "NVE", false);
	
	if (m_prof) m_prof->push(exec_conf, "NVE");
	if (m_prof) m_prof->push(exec_conf, "Half-step 2");
	
	// get the particle data arrays again so we can update the 2nd half of the step
	d_pdata = m_pdata->acquireReadWriteGPU();
	
	// call the post-step kernel on all GPUs in parallel
	exec_conf.tagAll(__FILE__, __LINE__);
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		exec_conf.gpu[cur_gpu]->call(bind(gpu_nve_step, d_pdata[cur_gpu], m_d_force_data_ptrs[cur_gpu], (int)m_forces.size(), m_deltaT, m_limit, m_limit_val));
	
	// post-step kernel for rigid bodies
	if (has_rigid_bodies && exec_conf.gpu.size() == 1) // only one GPU for the moment
		{
		ArrayHandle<Scalar> body_mass_handle(rigid_data->getBodyMass(), access_location::device, access_mode::read);
		ArrayHandle<Scalar4> vel_handle(rigid_data->getVel(), access_location::device, access_mode::readwrite);
		ArrayHandle<Scalar4> angmom_handle(rigid_data->getAngMom(), access_location::device, access_mode::readwrite);
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
		d_rdata.vel = vel_handle.data;	
		d_rdata.angmom = angmom_handle.data;	
		d_rdata.ex_space = ex_space_handle.data;	
		d_rdata.ey_space = ey_space_handle.data;	
		d_rdata.ez_space = ez_space_handle.data;			
		d_rdata.particle_pos = particle_pos_handle.data;	  
		d_rdata.particle_indices = particle_indices_handle.data; 
		d_rdata.force = force_handle.data;	
		d_rdata.torque = torque_handle.data;	
		
		exec_conf.gpu[0]->call(bind(gpu_nve_rigid_body_step, d_pdata[0], d_rdata, m_d_force_data_ptrs[0], (int)m_forces.size(), m_deltaT, m_limit, m_limit_val));

		}
	
	exec_conf.syncAll();
	m_pdata->release();
	
	// and now the acceleration at timestep+1 is precalculated for the first half of the next step
	if (m_prof)
		{
		mem_transfer = m_pdata->getN() * (16*m_forces.size() + 4 + 16 + 32);
		flops = m_pdata->getN() * (3*m_forces.size() + 3 + 6);
		m_prof->pop(exec_conf, flops, mem_transfer);
		m_prof->pop();
		}
	}
	
void export_NVEUpdaterGPU()
	{
	class_<NVEUpdaterGPU, boost::shared_ptr<NVEUpdaterGPU>, bases<NVEUpdater>, boost::noncopyable>
		("NVEUpdaterGPU", init< boost::shared_ptr<SystemDefinition>, Scalar >())
		;
	}

#ifdef WIN32
#pragma warning( pop )
#endif

