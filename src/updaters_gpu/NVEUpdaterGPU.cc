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

#include "NVEUpdaterGPU.h"
#include "gpu_updaters.h"

#ifdef USE_PYTHON
#include <boost/python.hpp>
using namespace boost::python;
#endif

using namespace std;

/*! \param pdata Particle data to update
	\param deltaT Time step to use
*/
NVEUpdaterGPU::NVEUpdaterGPU(boost::shared_ptr<ParticleData> pdata, Scalar deltaT) : NVEUpdater(pdata, deltaT)
	{
	}

/*! \param timestep Current time step of the simulation
*/
void NVEUpdaterGPU::update(unsigned int timestep)
	{
	assert(m_pdata);

	// if we haven't been called before, then the accelerations	have not been set and we need to calculate them
	if (!m_accel_set)
		{
		m_accel_set = true;
		// use the option of computeAccelerationsGPU to populate pdata.accel so the first step is
		// is calculated correctly
		computeAccelerationsGPU(timestep, "NVE.GPU", true);
		}

	if (m_prof)
		m_prof->push("NVE.GPU");
		
	// access the particle data arrays
	gpu_pdata_arrays d_pdata = m_pdata->acquireReadWriteGPU();
	gpu_boxsize box = m_pdata->getBoxGPU();

	if (m_prof)
		m_prof->push("Half-step 1");
		
	nve_pre_step(&d_pdata, &box, m_deltaT);
	
	if (m_prof)
		{
		cudaThreadSynchronize();
		m_prof->pop(m_pdata->getN() * 36, m_pdata->getN() * 16 * 5);
		}
	
	// release the particle data arrays so that they can be accessed to add up the accelerations
	m_pdata->release();
	
	// functions that computeAccelerations calls profile themselves, so suspend
	// the profiling for now
	if (m_prof)
		m_prof->pop();
	
	// for the next half of the step, we need the accelerations at t+deltaT
	computeAccelerationsGPU(timestep+1, "NVE.GPU", false);
	
	if (m_prof)
		m_prof->push("NVE.GPU");
			
	if (m_prof)
		m_prof->push("Half-step 2");
	
	// get the particle data arrays again so we can update the 2nd half of the step
	d_pdata = m_pdata->acquireReadWriteGPU();
	nve_step(&d_pdata, m_d_force_data_ptrs, m_forces.size(), m_deltaT);
	m_pdata->release();
	
	// and now the acceleration at timestep+1 is precalculated for the first half of the next step
	if (m_prof)
		{
		cudaThreadSynchronize();
		m_prof->pop(m_pdata->getN() * 9, m_pdata->getN() * 16 * (3 + m_forces.size()));	
		m_prof->pop();
		}
	}
	
#ifdef USE_PYTHON
void export_NVEUpdaterGPU()
	{
	class_<NVEUpdaterGPU, boost::shared_ptr<NVEUpdaterGPU>, bases<NVEUpdater>, boost::noncopyable>
		("NVEUpdaterGPU", init< boost::shared_ptr<ParticleData>, Scalar >())
		;
	}
#endif

