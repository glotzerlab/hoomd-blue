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
// Maintainer: joaander

/*! \file BoxResizeUpdater.cc
	\brief Defines the BoxResizeUpdater class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "BoxResizeUpdater.h"

#include <math.h>
#include <iostream>
#include <stdexcept>

// windows feels that rintf should not exist.....
#ifdef WIN32
//! replacement for rint in windows
double rint(double x)
	{
	return floor(x+.5);
	}
	
//! replacement for rint in windows
double rintf(float x)
	{
	return floorf(x+.5f);
	}
#endif


using namespace std;

/*! \param pdata Particle data to set the box size on
	\param Lx length of the x dimension over time
	\param Ly length of the y dimension over time
	\param Lz length of the z dimension over time
	
	The default setting is to scale particle positions along with the box.
*/
BoxResizeUpdater::BoxResizeUpdater(boost::shared_ptr<ParticleData> pdata, boost::shared_ptr<Variant> Lx, boost::shared_ptr<Variant> Ly, boost::shared_ptr<Variant> Lz)
	: Updater(pdata), m_Lx(Lx), m_Ly(Ly), m_Lz(Lz), m_scale_particles(true)
	{
	assert(m_pdata);
	assert(m_Lx);
	assert(m_Ly);
	assert(m_Lz);
	}

/*! \param scale_particles Set to true to scale particles with the box. Set to false to leave particle positions alone
	when scaling the box.
*/
void BoxResizeUpdater::setParams(bool scale_particles)
	{
	m_scale_particles = scale_particles;
	}
	
/*! Perform the needed calculations to scale the box size
	\param timestep Current time step of the simulation
*/
void BoxResizeUpdater::update(unsigned int timestep)
	{
	if (m_prof) m_prof->push("BoxResize");

	// first, compute what the current box size should be
	Scalar Lx = m_Lx->getValue(timestep);
	Scalar Ly = m_Ly->getValue(timestep);
	Scalar Lz = m_Lz->getValue(timestep);

	// check if the current box size is the same
	BoxDim curBox = m_pdata->getBox();
	bool no_change = fabs((Lx - curBox.xhi - curBox.xlo) / Lx) < 1e-5 && 
					fabs((Ly - curBox.yhi - curBox.ylo) / Ly) < 1e-5 &&
					fabs((Lz - curBox.zhi - curBox.zlo) / Lz) < 1e-5;
					
	// only change the box if there is a change in the box size
	if (!no_change)
		{	
		// scale the particle positions (if we have been asked to)
		if (m_scale_particles)
			{
			Scalar sx = Lx / (curBox.xhi - curBox.xlo);
			Scalar sy = Ly / (curBox.yhi - curBox.ylo);
			Scalar sz = Lz / (curBox.zhi - curBox.zlo);			
			
			// move the particles to be inside the new box
			ParticleDataArrays arrays = m_pdata->acquireReadWrite();
		
			for (unsigned int i = 0; i < arrays.nparticles; i++)
				{
				arrays.x[i] *= sx;
				arrays.y[i] *= sy;
				arrays.z[i] *= sz;
				}
				
			m_pdata->release();
			}
		else if (Lx < (curBox.xhi - curBox.xlo) || Ly < (curBox.yhi - curBox.ylo) || Lz < (curBox.zhi - curBox.zlo))
			{
			// otherwise, we need to ensure that the particles are still in the box if it is smaller
			// move the particles to be inside the new box
			ParticleDataArrays arrays = m_pdata->acquireReadWrite();
		
			for (unsigned int i = 0; i < arrays.nparticles; i++)
				{
				arrays.x[i] -= Lx * rintf(arrays.x[i] / Lx);
				arrays.y[i] -= Ly * rintf(arrays.y[i] / Ly);
				arrays.z[i] -= Lz * rintf(arrays.z[i] / Lz);
				}			
		
			m_pdata->release();
			}
			
		// set the new box
		m_pdata->setBox(BoxDim(Lx, Ly, Lz));
		}
	
	if (m_prof) m_prof->pop();
	}
	
void export_BoxResizeUpdater()
	{
	class_<BoxResizeUpdater, boost::shared_ptr<BoxResizeUpdater>, bases<Updater>, boost::noncopyable>
		("BoxResizeUpdater", init< boost::shared_ptr<ParticleData>, boost::shared_ptr<Variant>, boost::shared_ptr<Variant>, boost::shared_ptr<Variant> >())
		.def("setParams", &BoxResizeUpdater::setParams);
	}

#ifdef WIN32
#pragma warning( pop )
#endif
