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

// $Id: BD_NVTUpdater.cc 1206 2008-09-04 18:00:45Z joaander $
// $URL: http://svn2.assembla.com/svn/hoomd/trunk/src/updaters/BD_NVTUpdater.cc $

/*! \file BD_NVTUpdater.cc
	\brief Defines the BD_NVTUpdater class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#ifdef USE_PYTHON
#include <boost/python.hpp>
using namespace boost::python;
#endif

#include "BD_NVTUpdater.h"
#include <math.h>

using namespace std;

/*! \param pdata Particle data to update
	\param deltaT Time step to use
	\param Temp Temperature of Stochastic Bath (and simulation)
*/
BD_NVTUpdater::BD_NVTUpdater(boost::shared_ptr<ParticleData> pdata, Scalar deltaT, Scalar Temp) : NVEUpdater(pdata, deltaT), m_T(Temp)
	{
	// Add the Stochastic Force.  The default gamma for all particle types is 1.0;
	
	//The commands below need to be different if CUDA is being used!!
	boost::shared_ptr<StochasticForceCompute> m_bdfc(new StochasticForceCompute(pdata, deltaT, m_T));
	cout << "Adding on Stochastic Bath with deltaT = " << deltaT << " and Temp  = " << m_T << endl; 
	this->addForceCompute(m_bdfc);
	
	}

/*! \param Temp Temperature of the Stochastic Bath
*/
	
void BD_NVTUpdater::setT(Scalar Temp) 
	{	
	 m_T = Temp;
	 this->getForceCompute(0)->setT(m_T); 
	}	
	
	
/*! BD_NVTUpdater provides
	- \c nvt_kinetic_energy
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

	
#ifdef USE_PYTHON
void export_BD_NVTUpdater()
	{
	class_<BD_NVTUpdater, boost::shared_ptr<BD_NVTUpdater>, bases<NVEUpdater>, boost::noncopyable>
		("BD_NVTUpdater", init< boost::shared_ptr<ParticleData>, Scalar, Scalar >())
		.def("setT", &BD_NVTUpdater::setT);  //This should get changed into something that permits gammas to be set for different particle types
		
	}
#endif

#ifdef WIN32
#pragma warning( pop )
#endif
