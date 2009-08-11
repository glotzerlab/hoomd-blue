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

// $Id: BD_NVTUpdaterGPU.cc 1234 2008-09-11 16:29:13Z philliclBD_NVT $
// $URL: http://svn2.assembla.com/svn/hoomd/trunk/src/updaters_gpu/BD_NVTUpdaterGPU.cc $
// Maintainer: phillicl

/*! \file BD_NVTUpdaterGPU.cc
	\brief Defines the BD_NVTUpdaterGPU class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include "BD_NVTUpdaterGPU.h"
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
	\param Temp Temperature to set
	\param seed Random seed to use for the random force compuataion
*/
BD_NVTUpdaterGPU::BD_NVTUpdaterGPU(boost::shared_ptr<SystemDefinition> sysdef, Scalar deltaT, boost::shared_ptr<Variant> Temp, unsigned int seed, bool use_diam) : BD_NVTUpdater(sysdef, deltaT, Temp, seed, use_diam)
	{
	// at least one GPU is needed
	if (exec_conf.gpu.size() == 0)
		{
		cerr << endl << "***Error! Creating a BD_NVTUpdaterGPU with no GPU in the execution configuration" << endl << endl;
		throw std::runtime_error("Error initializing BD_NVTUpdaterGPU");
		}
	}

/*! \param timestep Current time step of the simulation
*/
void BD_NVTUpdaterGPU::update(unsigned int timestep)
	{
	assert(m_pdata);
	
	// get the rigid data from SystemDefinition
	boost::shared_ptr<RigidData> rigid_data = m_sysdef->getRigidData();
		
	// if there is any rigid body
	unsigned int n_bodies = rigid_data->getNumBodies();
	if (n_bodies > 0 && !m_rigid_updater) 
		m_rigid_updater = boost::shared_ptr<NVERigidUpdater> (new NVERigidUpdater(m_sysdef, m_deltaT));
		
	// hack to get correct profiling
	m_bdfc->setProfiler(m_prof);

	if (!m_bath) addStochasticBath();

	// if we haven't been called before, then the accelerations	have not been set and we need to calculate them
	if (!m_accel_set)
		{
		m_accel_set = true;
		// use the option of computeAccelerationsGPU to populate pdata.accel so the first step is
		// is calculated correctly
		computeAccelerationsGPU(timestep, "BD_NVT", true);
			
		// compute the initial net forces, torques and angular momenta 
		if (m_rigid_updater) m_rigid_updater->setup();
		}

	if (m_prof)
		m_prof->push(exec_conf, "BD_NVT");
	
	// access the particle data arrays
	vector<gpu_pdata_arrays>& d_pdata = m_pdata->acquireReadWriteGPU();
	gpu_boxsize box = m_pdata->getBoxGPU();

	if (m_prof) m_prof->push(exec_conf, "Half-step 1");
		
	// call the pre-step kernel on all GPUs in parallel
	exec_conf.tagAll(__FILE__, __LINE__);
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		exec_conf.gpu[cur_gpu]->call(bind(gpu_nve_pre_step, d_pdata[cur_gpu], box, m_deltaT, m_limit, m_limit_val));
	
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
			
			exec_conf.gpu[0]->call(bind(gpu_nve_rigid_body_pre_step, d_pdata[0], d_rdata, box, m_deltaT, m_limit, m_limit_val));
		}
		
	exec_conf.syncAll();
	
	uint64_t mem_transfer = m_pdata->getN() * (16+32+32);
	uint64_t flops = m_pdata->getN() * (15+3+9+12);
	if (m_prof) m_prof->pop(exec_conf, flops, mem_transfer);
	
	// release the particle data arrays so that they can be accessed to add up the accelerations
	m_pdata->release();
	
	// communicate the updated positions among the GPUs
	m_pdata->communicatePosition();
	
	// functions that computeAccelerations calls profile themselves, so suspend
	// the profiling for now
	if (m_prof) m_prof->pop(exec_conf);

	// for the next half of the step, we need the accelerations at t+deltaT
	computeAccelerationsGPU(timestep+1, "BD_NVT", false);
	
	if (m_prof) m_prof->push(exec_conf, "BD_NVT");
	if (m_prof) m_prof->push(exec_conf, "Half-step 2");
	
	// get the particle data arrays again so we can update the 2nd half of the step
	d_pdata = m_pdata->acquireReadWriteGPU();
	
	// call the post-step kernel on all GPUs in parallel
	exec_conf.tagAll(__FILE__, __LINE__);
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)	
		exec_conf.gpu[cur_gpu]->call(bind(gpu_nve_step, d_pdata[cur_gpu], m_d_force_data_ptrs[cur_gpu], (int)m_forces.size(), m_deltaT, m_limit, m_limit_val));
	
	// post-step kernel for rigid bodies
	if (m_rigid_updater && exec_conf.gpu.size() == 1) // only one GPU for the moment
		{
			
			ArrayHandle<Scalar> body_mass_handle(rigid_data->getBodyMass(), access_location::device, access_mode::read);
			ArrayHandle<Scalar4> moment_inertia_handle(rigid_data->getMomentInertia(), access_location::device, access_mode::read);
			ArrayHandle<Scalar4> com_handle(rigid_data->getCOM(), access_location::device, access_mode::read);
			ArrayHandle<Scalar4> vel_handle(rigid_data->getVel(), access_location::device, access_mode::readwrite);
			ArrayHandle<Scalar4> angvel_handle(rigid_data->getAngVel(), access_location::device, access_mode::readwrite);
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
			d_rdata.moment_inertia = moment_inertia_handle.data;
			d_rdata.com = com_handle.data;
			d_rdata.vel = vel_handle.data;
			d_rdata.angvel = angvel_handle.data;
			d_rdata.angmom = angmom_handle.data;
			d_rdata.ex_space = ex_space_handle.data;
			d_rdata.ey_space = ey_space_handle.data;
			d_rdata.ez_space = ez_space_handle.data;
			d_rdata.particle_pos = particle_pos_handle.data;
			d_rdata.particle_indices = particle_indices_handle.data;
			d_rdata.force = force_handle.data;
			d_rdata.torque = torque_handle.data;
			
			exec_conf.gpu[0]->call(bind(gpu_nve_rigid_body_step, d_pdata[0], d_rdata, m_d_force_data_ptrs[0], (int)m_forces.size(), box, m_deltaT, m_limit, m_limit_val));
			
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
	
void export_BD_NVTUpdaterGPU()
	{
	class_<BD_NVTUpdaterGPU, boost::shared_ptr<BD_NVTUpdaterGPU>, bases<BD_NVTUpdater>, boost::noncopyable>
		("BD_NVTUpdaterGPU", init< boost::shared_ptr<SystemDefinition>, Scalar, boost::shared_ptr<Variant>, unsigned int, bool >())
		;
	}

#ifdef WIN32
#pragma warning( pop )
#endif

