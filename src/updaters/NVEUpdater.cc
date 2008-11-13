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

/*! \file NVEUpdater.cc
	\brief Defines the NVEUpdater class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "NVEUpdater.h"
#include <math.h>

using namespace std;

/*! \param pdata Particle data to update
	\param deltaT Time step to use
*/
NVEUpdater::NVEUpdater(boost::shared_ptr<ParticleData> pdata, Scalar deltaT) : Integrator(pdata, deltaT), m_accel_set(false), m_limit(false), m_limit_val(1.0)
	{
	}

/*! \param limit Distance to limit particle movement each time step

	Once the limit is set, future calls to update() will never move a particle 
	a distance larger than the limit in a single time step
*/
void NVEUpdater::setLimit(Scalar limit)
	{
	assert(limit > 0.0);
	
	m_limit = true;
	m_limit_val = limit;
	}
		
/*! Disables the limit, allowing particles to move normally
*/
void NVEUpdater::removeLimit()
	{
	m_limit = false;
	}
	
/*! NVEUpdater provides
	- \c nve_kinetic_energy
*/
std::vector< std::string > NVEUpdater::getProvidedLogQuantities()
	{
	vector<string> list;
	list.push_back("nve_kinetic_energy");
	return list;
	}
	
Scalar NVEUpdater::getLogValue(const std::string& quantity, unsigned int timestep)
	{
	if (quantity == string("nve_kinetic_energy"))
		{
		const ParticleDataArraysConst arrays = m_pdata->acquireReadOnly();
		
		// always perform the sum in double precision for better accuracy
		// this is cheating and is really just a temporary hack to get logging up and running
		// the potential accuracy loss in simulations needs to be evaluated here and a proper
		// summation algorithm put in place
		double ke_total = 0.0;
		for (unsigned int i=0; i < m_pdata->getN(); i++)
			{
			ke_total += 0.5 * ((double)arrays.vx[i] * (double)arrays.vx[i] + (double)arrays.vy[i] * (double)arrays.vy[i] + (double)arrays.vz[i] * (double)arrays.vz[i]);
			}
	
		m_pdata->release();	
		return Scalar(ke_total);
		}
	else
		{
		cerr << endl << "***Error! " << quantity << " is not a valid log quantity for NVEUpdater" << endl;
		throw runtime_error("Error getting log value");
		}
	}	

/*! Uses velocity verlet
	\param timestep Current time step of the simulation
	
	\pre Associated ParticleData is initialized, and particle positions and velocities 
		are set for time timestep
	\post Forces and accelerations are computed and particle's positions, velocities 
		and accelartions are updated to their values at timestep+1.
*/
void NVEUpdater::update(unsigned int timestep)
	{
	assert(m_pdata);
	static bool gave_warning = false;

	if (m_forces.size() == 0 && !gave_warning)
		{
		cout << "Notice: No forces defined in NVEUpdater, Continuing anyways" << endl;
		gave_warning = true;
		}

	// if we haven't been called before, then the accelerations	have not been set and we need to calculate them
	if (!m_accel_set)
		{
		m_accel_set = true;
		computeAccelerations(timestep, "NVE");
		}

	if (m_prof)
		{
		m_prof->push("NVE");
		m_prof->push("Half-step 1");
		}
		
	// access the particle data arrays
	ParticleDataArrays arrays = m_pdata->acquireReadWrite();
	assert(arrays.x != NULL && arrays.y != NULL && arrays.z != NULL);
	assert(arrays.vx != NULL && arrays.vy != NULL && arrays.vz != NULL);
	assert(arrays.ax != NULL && arrays.ay != NULL && arrays.az != NULL);
	
	// now we can get on with the velocity verlet
	// r(t+deltaT) = r(t) + v(t)*deltaT + (1/2)a(t)*deltaT^2
	// v(t+deltaT/2) = v(t) + (1/2)a*deltaT
	for (unsigned int j = 0; j < arrays.nparticles; j++)
		{
		Scalar dx = arrays.vx[j]*m_deltaT + Scalar(1.0/2.0)*arrays.ax[j]*m_deltaT*m_deltaT;
		Scalar dy = arrays.vy[j]*m_deltaT + Scalar(1.0/2.0)*arrays.ay[j]*m_deltaT*m_deltaT;
		Scalar dz = arrays.vz[j]*m_deltaT + Scalar(1.0/2.0)*arrays.az[j]*m_deltaT*m_deltaT;
		
		// limit the movement of the particles
		if (m_limit)
			{
			Scalar len = sqrt(dx*dx + dy*dy + dz*dz);
			if (len > m_limit_val)
				{
				dx = dx / len * m_limit_val;
				dy = dy / len * m_limit_val;
				dz = dz / len * m_limit_val;
				}
			}
		
		arrays.x[j] += dx;
		arrays.y[j] += dy;
		arrays.z[j] += dz;
		
		arrays.vx[j] += Scalar(1.0/2.0)*arrays.ax[j]*m_deltaT;
		arrays.vy[j] += Scalar(1.0/2.0)*arrays.ay[j]*m_deltaT;
		arrays.vz[j] += Scalar(1.0/2.0)*arrays.az[j]*m_deltaT;
		}
		
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
	computeAccelerations(timestep+1, "NVE");
	
	if (m_prof)
		{
		m_prof->push("NVE");
		m_prof->push("Half-step 2");
		}
	
	// get the particle data arrays again so we can update the 2nd half of the step
	arrays = m_pdata->acquireReadWrite();
	
	// v(t+deltaT) = v(t+deltaT/2) + 1/2 * a(t+deltaT)*deltaT
	for (unsigned int j = 0; j < arrays.nparticles; j++)
		{
		arrays.vx[j] += Scalar(1.0/2.0)*arrays.ax[j]*m_deltaT;
		arrays.vy[j] += Scalar(1.0/2.0)*arrays.ay[j]*m_deltaT;
		arrays.vz[j] += Scalar(1.0/2.0)*arrays.az[j]*m_deltaT;
		
		// limit the movement of the particles
		if (m_limit)
			{
			Scalar vel = sqrt(arrays.vx[j]*arrays.vx[j] + arrays.vy[j]*arrays.vy[j] + arrays.vz[j]*arrays.vz[j]);
			if ( (vel*m_deltaT) > m_limit_val)
				{
				arrays.vx[j] = arrays.vx[j] / vel * m_limit_val / m_deltaT;
				arrays.vy[j] = arrays.vy[j] / vel * m_limit_val / m_deltaT;
				arrays.vz[j] = arrays.vz[j] / vel * m_limit_val / m_deltaT;
				}
			}
		}

	m_pdata->release();
	
	// and now the acceleration at timestep+1 is precalculated for the first half of the next step
	if (m_prof)
		{
		m_prof->pop();
		m_prof->pop();
		}
	}
	
void export_NVEUpdater()
	{
	class_<NVEUpdater, boost::shared_ptr<NVEUpdater>, bases<Integrator>, boost::noncopyable>
		("NVEUpdater", init< boost::shared_ptr<ParticleData>, Scalar >())
		.def("setLimit", &NVEUpdater::setLimit)
		.def("removeLimit", &NVEUpdater::removeLimit);
	}

#ifdef WIN32
#pragma warning( pop )
#endif
