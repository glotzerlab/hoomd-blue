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

/*! \file TempCompute.cc
	\brief Contains code for the TempCompute class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include "TempCompute.h"

#ifdef USE_PYTHON
#include <boost/python.hpp>
using namespace boost::python;
#endif

#include <iostream>
using namespace std;


/*! \param pdata Particle Data to compute temperature of
*/
TempCompute::TempCompute(boost::shared_ptr<ParticleData> pdata) : Compute(pdata), m_temp(0.0)
	{
	assert(m_pdata);
	m_dof = m_pdata->getN() * 3;
	}

/*! Calls computeTemp if the temperature needs updating
	\param timestep Current time step of the simulation
*/
void TempCompute::compute(unsigned int timestep)
	{
	if (!shouldCompute(timestep))
		return;
	
	computeTemp();
	}
		

/*! Computes the temperature by computing the kinetic energy and multiplying by the appropriate factor.
	\note This is computed in reduced units
*/
void TempCompute::computeTemp()
	{
	if (m_prof) m_prof->push("Temp");
	
	assert(m_pdata);
	assert(m_dof != 0);
	
	// access the particle data
	const ParticleDataArraysConst& arrays = m_pdata->acquireReadOnly(); 

	// total up kinetic energy
	Scalar K = 0.0;
	for (unsigned int i = 0; i < arrays.nparticles; i++)
		{
		// note m=1 in reduced units		
		K += Scalar(1.0/2.0) * (arrays.vx[i] * arrays.vx[i] + arrays.vy[i] * arrays.vy[i] + arrays.vz[i]*arrays.vz[i]);
		}
		
	// K = 1/2 * k_b * T * dof
	// => T = K * 2 / dof / k_b
	m_temp = K * Scalar(2.0) / Scalar(m_dof);
	
	m_pdata->release();
	
	if (m_prof) m_prof->pop();
	}

#ifdef USE_PYTHON
void export_TempCompute()
	{
	class_<TempCompute, boost::shared_ptr<TempCompute>, bases<Compute>, boost::noncopyable >
		("TempCompute", init< boost::shared_ptr<ParticleData> >())
		.def("setDOF", &TempCompute::setDOF)
		.def("getTemp", &TempCompute::getTemp)
		;
	}
#endif

#ifdef WIN32
#pragma warning( pop )
#endif
