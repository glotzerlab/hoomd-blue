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

/*! \file NVTUpdater.cc
	\brief Defines the NVTUpdater class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#ifdef USE_PYTHON
#include <boost/python.hpp>
using namespace boost::python;
#endif

#include "NVTUpdater.h"
#include <math.h>

using namespace std;

/*! \param pdata Particle data to update
	\param deltaT Time step to use
	\param tau NVT period
	\param T Temperature set point
*/
NVTUpdater::NVTUpdater(boost::shared_ptr<ParticleData> pdata, Scalar deltaT, Scalar tau, Scalar T) : Integrator(pdata, deltaT), m_tau(tau), m_T(T), m_accel_set(false)
	{
	if (m_tau <= 0.0)
		cout << "***Warning! tau set less than 0.0 in NVTUpdater" << endl;
	if (m_T <= 0.0)
		cout << "***Warning! T set less than 0.0 in NVTUpdater" << endl;
	m_Xi = 1.0;
	}

/*! \param timestep Current time step of the simulation
*/
void NVTUpdater::update(unsigned int timestep)
	{
	assert(m_pdata);
	static bool gave_warning = false;

	if (m_forces.size() == 0 && !gave_warning)
		{
		cout << "***Warning! No forces defined in NVTUpdater, continuing anyways" << endl;
		gave_warning = true;
		}
	
	// if we haven't been called before, then the accelerations	have not been set and we need to calculate them
	if (!m_accel_set)
		{
		m_accel_set = true;
		computeAccelerations(timestep, "NVT");
		}

	if (m_prof)
		{
		m_prof->push("NVT");
		m_prof->push("Half-step 1");
		}
		
	// access the particle data arrays
	ParticleDataArrays arrays = m_pdata->acquireReadWrite();
	assert(arrays.x != NULL && arrays.y != NULL && arrays.z != NULL);
	assert(arrays.vx != NULL && arrays.vy != NULL && arrays.vz != NULL);
	assert(arrays.ax != NULL && arrays.ay != NULL && arrays.az != NULL);
	
	// now we can get on with the Nose-Hoover
	// first half step: update the velocities and positions
	// NOTE: Mass = 1.0
	// sum up K while we are doing the loop
	Scalar Ksum = 0.0;
	for (unsigned int j = 0; j < arrays.nparticles; j++)
		{
		Scalar denominv = Scalar(1.0) / (Scalar(1.0) + m_deltaT/Scalar(2.0) * m_Xi);
	
		arrays.vx[j] = (arrays.vx[j] + Scalar(1.0/2.0)*arrays.ax[j]*m_deltaT) * denominv;	
		arrays.x[j] += m_deltaT * arrays.vx[j];
		
		arrays.vy[j] = (arrays.vy[j] + Scalar(1.0/2.0)*arrays.ay[j]*m_deltaT) * denominv;
		arrays.y[j] += m_deltaT * arrays.vy[j];
		
		arrays.vz[j] = (arrays.vz[j] + Scalar(1.0/2.0)*arrays.az[j]*m_deltaT) * denominv;
		arrays.z[j] += m_deltaT * arrays.vz[j];

		Ksum += arrays.vx[j]*arrays.vx[j] + arrays.vy[j]*arrays.vy[j] + arrays.vz[j]*arrays.vz[j];
		}
	
	// update Xi
	Scalar g = Scalar(3*m_pdata->getN());
	Scalar T_current = Ksum / g;
	m_Xi += m_deltaT / (m_tau*m_tau) * (T_current/m_T - Scalar(1.0));
		
	// We aren't done yet! Need to fix the periodic boundary conditions
	// this implementation only works if the particles go a wee bit outside the box, which is all that should ever happen under normal circumstances
	// get a local copy of the simulation box too
	const BoxDim& box = m_pdata->getBox();
	// sanity check
	assert(box.xhi > box.xlo && box.yhi > box.ylo && box.zhi > box.zlo);	
	
	// precalculate box lenghts
	Scalar Lx = box.xhi - box.xlo;
	Scalar Ly = box.yhi - box.ylo;
	Scalar Lz = box.zhi - box.zlo;

	for (unsigned int j = 0; j < arrays.nparticles; j++)
		{
		// wrap the particle around the box
		if (arrays.x[j] >= box.xhi)
			arrays.x[j] -= Lx;
		else
		if (arrays.x[j] < box.xlo)
			arrays.x[j] += Lx;
			
		if (arrays.y[j] >= box.yhi)
			arrays.y[j] -= Ly;
		else
		if (arrays.y[j] < box.ylo)
			arrays.y[j] += Ly;
			
		if (arrays.z[j] >= box.zhi)
			arrays.z[j] -= Lz;
		else
		if (arrays.z[j] < box.zlo)
			arrays.z[j] += Lz;
		}

	
	// release the particle data arrays so that they can be accessed to add up the accelerations
	m_pdata->release();
	
	// functions that computeAccelerations calls profile themselves, so suspend
	// the profiling for now
	if (m_prof)
		{
		m_prof->pop();
		m_prof->pop();
		}
	
	// for the next half of the step, we need the accelerations at t+deltaT
	computeAccelerations(timestep+1, "NVT");
	
	if (m_prof)
		{
		m_prof->push("NVT");
		m_prof->push("Half-step 2");
		}
	
	// get the particle data arrays again so we can update the 2nd half of the step
	arrays = m_pdata->acquireReadWrite();
	
	// v(t+deltaT) = v(t+deltaT/2) + 1/2 * a(t+deltaT)*deltaT
	for (unsigned int j = 0; j < arrays.nparticles; j++)
		{
		arrays.vx[j] += Scalar(1.0/2.0) * m_deltaT * (arrays.ax[j] - m_Xi * arrays.vx[j]);
		arrays.vy[j] += Scalar(1.0/2.0) * m_deltaT * (arrays.ay[j] - m_Xi * arrays.vy[j]);
		arrays.vz[j] += Scalar(1.0/2.0) * m_deltaT * (arrays.az[j] - m_Xi * arrays.vz[j]);
		}

	m_pdata->release();
	
	// and now the acceleration at timestep+1 is precalculated for the first half of the next step
	if (m_prof)
		{
		m_prof->pop();
		m_prof->pop();
		}
	}
	
#ifdef USE_PYTHON
void export_NVTUpdater()
	{
	class_<NVTUpdater, boost::shared_ptr<NVTUpdater>, bases<Integrator>, boost::noncopyable>
		("NVTUpdater", init< boost::shared_ptr<ParticleData>, Scalar, Scalar, Scalar >())
		.def("setT", &NVTUpdater::setT)
		.def("setTau", &NVTUpdater::setTau)
		;
		
	}
#endif

#ifdef WIN32
#pragma warning( pop )
#endif