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

#ifdef USE_PYTHON
#include <boost/python.hpp>
using namespace boost::python;
#endif

#include "ConstForceCompute.h"

using namespace std;

/*! \file ConstForceCompute.cc
	\brief Contains code for the ConstForceCompute class
*/

/*! \param pdata Particle data to compute forces on
	\param fx x-component of the force
	\param fy y-component of the force
	\param fz z-component of the force
	\note This class doesn't actually do anything with the particle data. It just returns a constant force
*/
ConstForceCompute::ConstForceCompute(boost::shared_ptr<ParticleData> pdata, Scalar fx, Scalar fy, Scalar fz)
	: ForceCompute(pdata)
	{
	setForce(fx,fy,fz);
	}
	
/*! \param fx x-component of the force
	\param fy y-component of the force
	\param fz z-component of the force
*/
void ConstForceCompute::setForce(Scalar fx, Scalar fy, Scalar fz)
	{
	assert(m_fx != NULL && m_fy != NULL && m_fz != NULL && m_pdata != NULL);
	
	// setting the force is simple, just fill out every element of the force array
	for (unsigned int i = 0; i < m_pdata->getN(); i++)
		{
		m_fx[i] = fx;
		m_fy[i] = fy;
		m_fz[i] = fz;
		m_pe[i] = 0;
		}

	#ifdef USE_CUDA
	// data now only exists on the CPU
	m_data_location = cpu;
	#endif
	}
	
/*! Actually, this function does nothing. Since the data arrays were already filled out by setForce(),
	we don't need to do a thing here :)
	\param timestep Current timestep
*/
void ConstForceCompute::computeForces(unsigned int timestep)
	{
	}


#ifdef USE_PYTHON
void export_ConstForceCompute()
	{
	class_< ConstForceCompute, boost::shared_ptr<ConstForceCompute>, bases<ForceCompute>, boost::noncopyable >
		("ConstForceCompute", init< boost::shared_ptr<ParticleData>, Scalar, Scalar, Scalar >())
		.def("setForce", &ConstForceCompute::setForce)
		;
	}
#endif
