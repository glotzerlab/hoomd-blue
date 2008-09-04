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

/*! \file TempRescaleUpdater.cc
	\brief Defines the TempRescaleUpdater class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#ifdef USE_PYTHON
#include <boost/python.hpp>
using namespace boost::python;
#endif

#include "TempRescaleUpdater.h"

#include <iostream>
#include <math.h>
#include <stdexcept>

using namespace std;

/*! \param pdata Particle data to set temperature on
	\param tc TempCompute to compute the temperature with
	\param tset Temperature set point
*/
TempRescaleUpdater::TempRescaleUpdater(boost::shared_ptr<ParticleData> pdata, boost::shared_ptr<TempCompute> tc, Scalar tset)
	: Updater(pdata), m_tc(tc), m_tset(tset)
	{
	assert(m_pdata);
	assert(tc);
	if (m_tset < 0.0)
		{
		cerr << endl << "***Error! TempRescaleUpdater: Cannot set a negative temperature" << endl << endl;
		throw runtime_error("Error initializing TempRescaleUpdater");
		}
	}
	
	
/*! Perform the proper velocity rescaling
	\param timestep Current time step of the simulation
*/
void TempRescaleUpdater::update(unsigned int timestep)
	{
	// find the current temperature

	assert(m_tc);
	m_tc->compute(timestep);
	Scalar cur_temp = m_tc->getTemp();

	if (m_prof)
		{
		m_prof->push("TempRescale");
		m_prof->push("Update");
		}

	if (cur_temp < 1e-3)
		{
		cout << "Notice: TempRescaleUpdater cannot scale a 0 temperature to anything but 0, skipping this step" << endl;
		}
	else
		{
		// calculate a fraction to scale the velocities by
		Scalar fraction = sqrt(m_tset / cur_temp);
		
		// scale the particles velocities
		assert(m_pdata);
		ParticleDataArrays arrays = m_pdata->acquireReadWrite();
		
		for (unsigned int i = 0; i < arrays.nparticles; i++)
			{
			arrays.vx[i] *= fraction;
			arrays.vy[i] *= fraction;
			arrays.vz[i] *= fraction;
			}
		
		m_pdata->release();
		}
	
	if (m_prof)
		{
		m_prof->pop();
		m_prof->pop();
		}
	}
	
/*! \param tset New temperature set point
	\note The new set point doesn't take effect until the next call to update()
*/
void TempRescaleUpdater::setT(Scalar tset)
	{
	m_tset = tset;
	
	if (m_tset < 0.0)
		{
		cerr << endl << "***Error! TempRescaleUpdater: Cannot set a negative temperature" << endl << endl;
		throw runtime_error("Error initializing TempRescaleUpdater");
		}
	}

#ifdef USE_PYTHON
void export_TempRescaleUpdater()
	{
	class_<TempRescaleUpdater, boost::shared_ptr<TempRescaleUpdater>, bases<Updater>, boost::noncopyable>
		("TempRescaleUpdater", init< boost::shared_ptr<ParticleData>, boost::shared_ptr<TempCompute>, Scalar >())
		.def("setT", &TempRescaleUpdater::setT)
		;
	}
#endif

#ifdef WIN32
#pragma warning( pop )
#endif

