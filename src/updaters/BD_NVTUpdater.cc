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

// $Id: BD_NVTUpdater.cc 1206 2008-09-04 18:00:45Z phillicl $
// $URL: http://svn2.assembla.com/svn/hoomd/trunk/src/updaters/BD_NVTUpdater.cc $

/*! \file BD_NVTUpdater.cc
	\brief Defines the BD_NVTUpdater class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "BD_NVTUpdater.h"
#include <math.h>

#include <boost/bind.hpp>

#ifdef USE_CUDA
#include "gpu_integrator.h"
#endif


using namespace std;

/*! \param pdata Particle data to update
	\param deltaT Time step to use
*/
BD_NVTUpdater::BD_NVTUpdater(boost::shared_ptr<ParticleData> pdata, Scalar deltaT, Scalar Temp, unsigned int seed) : Integrator(pdata, deltaT), 
	m_accel_set(false), m_limit(false), m_limit_val(1.0), m_T(Temp)
	{


	cout << "Adding on Stochastic Bath with deltaT = " << deltaT << " and Temp  = " << m_T << endl; 	

	
	// check the execution configuration
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
	if (exec_conf.exec_mode == ExecutionConfiguration::CPU )
		using_gpu = false;
	else using_gpu = true;
	cout << exec_conf.exec_mode << endl;
	cout << "Using GPU " << using_gpu << endl;
 
     
	#ifdef USE_CUDA
	if (using_gpu) m_bdfc_gpu =  boost::shared_ptr<StochasticForceComputeGPU> (new StochasticForceComputeGPU(pdata, deltaT, m_T, seed));
	#endif
	
	m_bdfc = boost::shared_ptr<StochasticForceCompute>(new StochasticForceCompute(pdata, deltaT, m_T, seed));
    
	
	
	}

/*! \param Temp Temperature of the Stochastic Bath
*/
	
void BD_NVTUpdater::setT(Scalar Temp) 
	{	
	 m_T = Temp;
	 
	#ifdef USE_CUDA
	if (using_gpu)
	m_bdfc_gpu->setT(m_T);
	#endif
	
	m_bdfc->setT(m_T); 
	}	

/*! Once the limit is set, future calls to update() will never move a particle 
	a distance larger than limit in a single time step
*/
void BD_NVTUpdater::setLimit(Scalar limit)
	{
	assert(limit > 0.0);
	
	m_limit = true;
	m_limit_val = limit;
	}
		
/*! Disables the limit, allowing particles to move normally
*/
void BD_NVTUpdater::removeLimit()
	{
	m_limit = false;
	}
	
/*! BD_NVTUpdater provides
	- \c nve_kinetic_energy
*/
std::vector< std::string > BD_NVTUpdater::getProvidedLogQuantities()
	{
	vector<string> list;
	list.push_back("nvt_kinetic_energy");
	list.push_back("temperature");
	return list;
	}
	
Scalar BD_NVTUpdater::getLogValue(const std::string& quantity)
	{
	if (quantity == string("nvt_kinetic_energy"))
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
		
	else if (quantity == string("temperature"))
		{
		const ParticleDataArraysConst arrays = m_pdata->acquireReadOnly();
		
		// always perform the sum in double precision for better accuracy
		// this is cheating and is really just a temporary hack to get logging up and running
		// the potential accuracy loss in simulations needs to be evaluated here and a proper
		// summation algorithm put in place
		
		//Also note that KE does not currently take into account mass.  Fix this eventually
		double ke_total = 0.0;
		int nparticles = m_pdata->getN();
		for (unsigned int i=0; i < m_pdata->getN(); i++)
			{
			ke_total += 0.5 * ((double)arrays.vx[i] * (double)arrays.vx[i] + (double)arrays.vy[i] * (double)arrays.vy[i] + (double)arrays.vz[i] * (double)arrays.vz[i]);
			}
	
		m_pdata->release();	
		return Scalar(2.0/3.0)*Scalar(ke_total)/nparticles;
		}
				
	else
		{
		cerr << endl << "***Error! " << quantity << " is not a valid log quantity for BD_NVTUpdater" << endl;
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
void BD_NVTUpdater::update(unsigned int timestep)
	{
	assert(m_pdata);
	static bool gave_warning = false;

	if (m_forces.size() == 0 && !gave_warning)
		{
		cout << "Notice: No forces defined in BD_NVTUpdater, Continuing anyways" << endl;
		gave_warning = true;
		}

	// if we haven't been called before, then the accelerations	have not been set and we need to calculate them
	if (!m_accel_set)
		{
		m_accel_set = true;
		computeBDAccelerations(timestep, "BD_NVT");
		}

	if (m_prof)
		{
		m_prof->push("BD_NVT");
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
	
	// functions that computeBDAccelerations calls profile themselves, so suspend
	// the profiling for now
	if (m_prof)
		{
		m_prof->pop();
		m_prof->pop();
		}

	// for the next half of the step, we need the accelerations at t+deltaT
	computeBDAccelerations(timestep+1, "BD_NVT");
	
	if (m_prof)
		{
		m_prof->push("BD_NVT");
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

/*! \param timestep Current timestep
	\param profiler_name Name of the profiler element to continue timing under
	\post \c arrays.ax, \c arrays.ay, and \c arrays.az are set based on the forces computed by the ForceComputes
*/
void BD_NVTUpdater::computeBDAccelerations(unsigned int timestep, const std::string& profiler_name)
	{
	// this code is written in reduced units, so m=1. I set it here just in case the code is ever
	// modified to support other masses
	Scalar minv = 1.0;
	
	//handle the stochastic forcecompute
	assert(m_bdfc);
	m_bdfc->compute(timestep);		
	
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
	
	// account for the stochastic bath impact on accelerations.
	assert(m_bdfc);
	ForceDataArrays force_arrays = m_bdfc->acquire();
	for (unsigned int j = 0; j < arrays.nparticles; j++)
		{
		arrays.ax[j] += force_arrays.fx[j]*minv;
		arrays.ay[j] += force_arrays.fy[j]*minv;
		arrays.az[j] += force_arrays.fz[j]*minv;
		}
	
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


void BD_NVTUpdater::computeBDAccelerationsGPU(unsigned int timestep, const std::string& profiler_name, bool sum_accel)
	{
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
	if (exec_conf.gpu.empty())
		{
		cerr << endl << "***Error! Integrator asked to compute GPU accelerations but there is no GPU in the execution configuration" << endl << endl;
		throw runtime_error("Error computing accelerations");
		}
	
		//handle the gpu stochastic forcecompute
		assert(m_bdfc_gpu);
		m_bdfc_gpu->compute(timestep);		
		m_bdfc_gpu->acquireGPU();

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
			m_prof->push(exec_conf, "Sum accel");
			}
		
		// acquire the particle data on the GPU and add the forces into the acceleration
		vector<gpu_pdata_arrays>& d_pdata = m_pdata->acquireReadWriteGPU();

		// sum up all the forces
		for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
			{
			exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
			exec_conf.gpu[cur_gpu]->callAsync(boost::bind(integrator_sum_forces, &d_pdata[cur_gpu], m_d_force_data_ptrs[cur_gpu], (int)m_forces.size() + 1));
			}
			
		exec_conf.syncAll();
			
		// done
		m_pdata->release();

//NEED TO HANDLE PROFILING CORRECTLY (i.e. add Stochastic part)		
		if (m_prof)
			{
			m_prof->pop(exec_conf, 6*m_pdata->getN()*m_forces.size(), sizeof(Scalar)*4*m_pdata->getN()*(1+m_forces.size()));
			m_prof->pop();
			}
		}
	}

#endif
	
void export_BD_NVTUpdater()
	{
	class_<BD_NVTUpdater, boost::shared_ptr<BD_NVTUpdater>, bases<Integrator>, boost::noncopyable>
		("BD_NVTUpdater", init< boost::shared_ptr<ParticleData>, Scalar, Scalar, unsigned int >())
		.def("setGamma", &BD_NVTUpdater::setGamma)
		.def("setT", &BD_NVTUpdater::setT)
		.def("setLimit", &BD_NVTUpdater::setLimit)
		.def("removeLimit", &BD_NVTUpdater::removeLimit);
	}

#ifdef WIN32
#pragma warning( pop )
#endif
