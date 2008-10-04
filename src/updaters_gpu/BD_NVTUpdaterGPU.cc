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

/*! \file BD_NVTUpdaterGPU.cc
	\brief Defines the BD_NVTUpdaterGPU class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include "BD_NVTUpdaterGPU.h"
#include "gpu_updaters.h"

#include <boost/bind.hpp>
using namespace boost;

#ifdef USE_PYTHON
#include <boost/python.hpp>
using namespace boost::python;
#endif

using namespace std;

/*! \param pdata Particle data to update
	\param deltaT Time step to use
*/
BD_NVTUpdaterGPU::BD_NVTUpdaterGPU(boost::shared_ptr<ParticleData> pdata, Scalar deltaT, Scalar Temp, unsigned int seed) : BD_NVTUpdater(pdata, deltaT, Temp, seed)
	{
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
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
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();

	// if we haven't been called before, then the accelerations	have not been set and we need to calculate them
	if (!m_accel_set)
		{
		m_accel_set = true;
		// use the option of computeAccelerationsGPU to populate pdata.accel so the first step is
		// is calculated correctly
		computeBDAccelerationsGPU(timestep, "BD_NVT", true);
		}

	if (m_prof)
		m_prof->push(exec_conf, "BD_NVT");
		
	// access the particle data arrays
	vector<gpu_pdata_arrays>& d_pdata = m_pdata->acquireReadWriteGPU();
	gpu_boxsize box = m_pdata->getBoxGPU();

	if (m_prof) m_prof->push(exec_conf, "Half-step 1");
		
	exec_conf.gpu[0]->setTag(__FILE__, __LINE__);
	exec_conf.gpu[0]->call(bind(nve_pre_step, &d_pdata[0], &box, m_deltaT, m_limit, m_limit_val));
	
	uint64_t mem_transfer = m_pdata->getN() * (16+32+32);
	uint64_t flops = m_pdata->getN() * (15+3+9+12);
	if (m_prof) m_prof->pop(exec_conf, flops, mem_transfer);
	
	// release the particle data arrays so that they can be accessed to add up the accelerations
	m_pdata->release();
	
	// functions that computeAccelerations calls profile themselves, so suspend
	// the profiling for now
	if (m_prof) m_prof->pop(exec_conf);
	
	// for the next half of the step, we need the accelerations at t+deltaT
	computeBDAccelerationsGPU(timestep+1, "BD_NVT.GPU", false);
	
	if (m_prof) m_prof->push(exec_conf, "BD_NVT");
	if (m_prof) m_prof->push(exec_conf, "Half-step 2");
	
	// get the particle data arrays again so we can update the 2nd half of the step
	d_pdata = m_pdata->acquireReadWriteGPU();
	exec_conf.gpu[0]->setTag(__FILE__, __LINE__);
	exec_conf.gpu[0]->call(bind(nve_step, &d_pdata[0], m_d_force_data_ptrs[0], (int)m_forces.size() + 1, m_deltaT, m_limit, m_limit_val));
	m_pdata->release();
	
	// and now the acceleration at timestep+1 is precalculated for the first half of the next step
	if (m_prof)
		{
		mem_transfer = m_pdata->getN() * (16*m_forces.size() + 16 + 32);
		flops = m_pdata->getN() * (3*m_forces.size() + 6);
		m_prof->pop(exec_conf, flops, mem_transfer);
		m_prof->pop();
		}
	}
	
#ifdef USE_PYTHON
void export_BD_NVTUpdaterGPU()
	{
	class_<BD_NVTUpdaterGPU, boost::shared_ptr<BD_NVTUpdaterGPU>, bases<BD_NVTUpdater>, boost::noncopyable>
		("BD_NVTUpdaterGPU", init< boost::shared_ptr<ParticleData>, Scalar, Scalar, unsigned int >())
		;
	}
#endif

#ifdef WIN32
#pragma warning( pop )
#endif

