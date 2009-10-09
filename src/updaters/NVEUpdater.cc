/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
// Maintainer: joaander

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
#include "NVERigidUpdater.h"
#include <math.h>

using namespace std;

/*! \param sysdef System to update
    \param deltaT Time step to use
*/
NVEUpdater::NVEUpdater(boost::shared_ptr<SystemDefinition> sysdef, Scalar deltaT) : Integrator(sysdef, deltaT),
        m_accel_set(false), m_limit(false), m_limit_val(1.0)
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

/*! \param quantity Name of the quantity to log
    \param timestep Current time step of the simulation

    NVEUpdater calculates the conserved quantity as the sum of the kinetic and potential energies.
    All other quantity requests are passed up to Integrator::getLogValue().
*/
Scalar NVEUpdater::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == string("conserved_quantity"))
        {
        return computeKineticEnergy(timestep) + computePotentialEnergy(timestep);
        }
    else
        {
        // pass it on up to the base class
        return Integrator::getLogValue(quantity, timestep);
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

	// get the rigid data from SystemDefinition
	boost::shared_ptr<RigidData> rigid_data = m_sysdef->getRigidData();

	// if there is any rigid body
	unsigned int n_bodies = rigid_data->getNumBodies();
	if (n_bodies > 0 && !m_rigid_updater) 
		{			
		// allocate the rigid updater
		m_rigid_updater = boost::shared_ptr<NVERigidUpdater> (new NVERigidUpdater(m_sysdef, m_deltaT));
		assert(m_rigid_updater);
		}
		
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
		
		// compute the initial body velocities, forces, torques and angular momenta
		if (m_rigid_updater) m_rigid_updater->setup();
			
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
		// no need to update particles in rigid bodies
		if (arrays.body[j] != NO_BODY) continue;
			
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
		// no need to update particles in rigid bodies
		if (arrays.body[j] != NO_BODY) continue;
			
		// wrap the particle around the box
		if (arrays.x[j] >= box.xhi)
			{
			arrays.x[j] -= Lx;
			arrays.ix[j]++;
			}
		else
		if (arrays.x[j] < box.xlo)
			{
			arrays.x[j] += Lx;
			arrays.ix[j]--;
			}
			
		if (arrays.y[j] >= box.yhi)
			{
			arrays.y[j] -= Ly;
			arrays.iy[j]++;
			}
		else
		if (arrays.y[j] < box.ylo)
			{
			arrays.y[j] += Ly;
			arrays.iy[j]--;
			}
			
		if (arrays.z[j] >= box.zhi)
			{
			arrays.z[j] -= Lz;
			arrays.iz[j]++;
			}
		else
		if (arrays.z[j] < box.zlo)
			{
			arrays.z[j] += Lz;
			arrays.iz[j]--;
			}
		}
	
	// release the particle data arrays 
	m_pdata->release();
		
	// rigid body 1st step integration	
	if (m_rigid_updater) m_rigid_updater->initialIntegrate(timestep);
		
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
		// no need to update particles in rigid bodies
		if (arrays.body[j] != NO_BODY) continue;
			
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

	// release the particle data arrays	
	m_pdata->release();
		
	// rigid body 2nd step integration (net forces and torques are computed within)
	if (m_rigid_updater) m_rigid_updater->finalIntegrate(timestep);

	
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
    ("NVEUpdater", init< boost::shared_ptr<SystemDefinition>, Scalar >())
    .def("setLimit", &NVEUpdater::setLimit)
    .def("removeLimit", &NVEUpdater::removeLimit);
    }

#ifdef WIN32
#pragma warning( pop )
#endif

