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
	\param Temp Temperature to set
	\param seed Random seed to use for the random force compuataion
*/
BD_NVTUpdater::BD_NVTUpdater(boost::shared_ptr<ParticleData> pdata, Scalar deltaT, Scalar Temp, unsigned int seed) : NVEUpdater(pdata, deltaT), 
	m_accel_set(false), m_limit(false), m_limit_val(1.0), m_T(Temp), m_deltaT(deltaT), m_seed(seed), m_bath(false)
	{


	cout << "Adding on Stochastic Bath with deltaT = " << deltaT << " and Temp  = " << m_T << endl; 	

	
	// check the execution configuration
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
	if (exec_conf.exec_mode == ExecutionConfiguration::CPU )
		using_gpu = false;
	else using_gpu = true;

	m_bdfc = boost::shared_ptr<StochasticForceCompute>(new StochasticForceCompute(m_pdata, m_deltaT, m_T, m_seed));  
	#ifdef USE_CUDA
	if (using_gpu) m_bdfc_gpu =  boost::shared_ptr<StochasticForceComputeGPU> (new StochasticForceComputeGPU(m_pdata, m_deltaT, m_T, m_seed));
	#endif    
	
	addStochasticBath();

	}

void BD_NVTUpdater::addStochasticBath()
	{
	if (m_bath)	cout << "Stochastic Bath Already Added" << endl;
	else {

		#ifdef USE_CUDA
		if (using_gpu) {
			this->addForceCompute(m_bdfc_gpu);	
			}	
		#endif 
		if (!using_gpu) {
			this->addForceCompute(m_bdfc);	
			}

		m_bath_index = (unsigned int)m_forces.size() - 1;	
		
		boost::shared_ptr<StochasticForceCompute> stochastic_force(boost::shared_dynamic_cast<StochasticForceCompute>(m_forces[m_bath_index]));	
		assert(stochastic_force);
	    
		m_bath = true;
		}
		
	}


/*! \param Temp Temperature of the Stochastic Bath
*/
	
void BD_NVTUpdater::setT(Scalar Temp) 
	{	
	 m_T = Temp;
	boost::shared_ptr<StochasticForceCompute> stochastic_force(boost::shared_dynamic_cast<StochasticForceCompute>(m_forces[m_bath_index]));	
	stochastic_force->setT(m_T); 
	}	

/*! \param limit Distance to limit particle movement each time step

	Once the limit is set, future calls to update() will never move a particle 
	a distance larger than the limit in a single time step
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

/*! Disables the ForceComputes
*/
void BD_NVTUpdater::removeForceComputes()
	{
	m_bath = false;
	Integrator::removeForceComputes();
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
	if (!m_bath) addStochasticBath();
	NVEUpdater::update(timestep);
	}

//! Exports the BD_NVTUpdater class to python	
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
