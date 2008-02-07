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

#include <stdlib.h>

// windows is stupid and needs this to define pi
#define _USE_MATH_DEFINES
#include <math.h>

#include <cassert>
#include <stdexcept>

using namespace std;

#ifdef USE_PYTHON
#include <boost/python.hpp>
using namespace boost::python;
#endif

#include "Initializers.h"


////////////////////////////////////////////////////////////////////////////////
// Simple Cubic Initializer

/*! An \c M x \c M x \c M array of particles is going to be created \c spacing units
	apart from each other.
	\param M Number of particles along a side of the cube
	\param spacing Separation between particles
*/
SimpleCubicInitializer::SimpleCubicInitializer(unsigned int M, Scalar spacing) : m_M(M), m_spacing(spacing), box(M * spacing)
	{
	}

/*! \return Number of particles that will be created
*/
unsigned int SimpleCubicInitializer::getNumParticles() const
	{
	return m_M * m_M * m_M;
	}

unsigned int SimpleCubicInitializer::getNumParticleTypes() const
	{
	return 1;
	}		

/*! \return Box dimensions that just fit the particles on their lattice
*/
BoxDim SimpleCubicInitializer::getBox() const
	{
	return box;
	}
		
		
/*! Initializes the particles
	\param pdata Particle data arrays to write the cubic crystal into
*/
void SimpleCubicInitializer::initArrays(const ParticleDataArrays &pdata) const
	{
	assert(pdata.nparticles > 0);
	assert(pdata.x != NULL);
	assert(pdata.y != NULL);
	assert(pdata.z != NULL);
	
		
	// just do a simple triple for loop to fill the space
	unsigned int c = 0;
	for (unsigned int k = 0; k < m_M; k++)
		{
		for (unsigned int j = 0; j < m_M; j++)
			{
			for (unsigned int i = 0; i < m_M; i++)
				{
				pdata.x[c] = i * m_spacing + box.xlo;
				pdata.y[c] = j * m_spacing + box.ylo;
				pdata.z[c] = k * m_spacing + box.zlo;
				c++;
				}
			}
		}
		
	// sanity check, make sure c didn't overrun the array
	assert(c == pdata.nparticles);
	}



/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

/*! \param N number of particles to create
	\param phi_p Packing fraction of particles in the box
	\param min_dist Minimum distance two particles will be placed apart
	\note assumes particles have a diameter of 1
*/
RandomInitializer::RandomInitializer(unsigned int N, Scalar phi_p, Scalar min_dist) : m_N(N), m_phi_p(phi_p), m_min_dist(min_dist)
	{
	// sanity checks
	if (N == 0)
		throw runtime_error("RandomInitializer: Cannot generate 0 particles");
	if (phi_p <= 0)
		throw runtime_error("RandomInitializer: phi_p <= 0 doesn't make sense");
	if (min_dist < 0)
		throw runtime_error("RandomInitializer: min_dist <= 0 doesn't make sense");
	
	Scalar L = pow(Scalar(M_PI/6.0)*Scalar(N) / phi_p, Scalar(1.0/3.0));
	m_box = BoxDim(L);
	}

/*! \return Number of particles that will be initialized
*/
unsigned int RandomInitializer::getNumParticles() const
	{
	return m_N;
	}
		
/*! \return Number of particle types that will be initialized
*/
unsigned int RandomInitializer::getNumParticleTypes() const
	{
	return 1;
	}

/*! \return Box particles will be placed in
*/
BoxDim RandomInitializer::getBox() const
	{
	return m_box;
	}

/*! \param seed Random seed to set
	Two RandomInitializers with the same random seen should produce the same
	particle positions.

	\warning setSeed is guarunteed to work properly if and only if
	there are no methods that might call random() called between
	the setSeed and the construction of the ParticleData.
*/
void RandomInitializer::setSeed(unsigned int seed)
	{
	srand(seed);
	}
	
/*! \param pdata Particle data arrays to write particles in to
	\post \a N particles are randomly placed in the box
	\note An exception is thrown if too many tries are made to find a spot where
		min_dist can be satisfied.
*/
void RandomInitializer::initArrays(const ParticleDataArrays &pdata) const
	{
	assert(pdata.nparticles == m_N);
	
	Scalar L = m_box.xhi*Scalar(2.0);
	for (unsigned int i = 0; i < pdata.nparticles; i++)
		{
		// generate random particles until we find a suitable one meating the min_dist
		// criteria
		bool done = false;
		unsigned int tries = 0;
		Scalar x,y,z;
		while(!done)
			{
			//Hack to fix compilation error
			x = Scalar((rand())/Scalar(RAND_MAX) - 0.5)*L;
			y = Scalar((rand())/Scalar(RAND_MAX) - 0.5)*L;
			z = Scalar((rand())/Scalar(RAND_MAX) - 0.5)*L;
			// assume we are done unless we are not
			done = true;
			// only do the minimum distance check if the minimum distance is non-zero
			if (m_min_dist > 1e-6)
				{
				for (unsigned int j = 0; j < i; j++)
					{
					Scalar dx = pdata.x[j] - x;
					if (dx < -L/Scalar(2.0))
						dx += L;
					if (dx > L/Scalar(2.0))
						dx -= L;
					
					Scalar dy = pdata.y[j] - y;
					if (dy < -L/Scalar(2.0))
						dy += L;
					if (dy > L/Scalar(2.0))
						dy -= L;
					
					Scalar dz = pdata.z[j] - z;
					if (dz < -L/Scalar(2.0))
						dz += L;
					if (dz > L/Scalar(2.0))
						dz -= L;
						
					Scalar dr2 = dx*dx + dy*dy + dz*dz;
					if (dr2 <= m_min_dist * m_min_dist)
						done = false;
					}
				}
			tries++;
			if (tries > pdata.nparticles*100)
				throw runtime_error("RandomInitializer: Unable to find location for particle after trying many times"); 
			}
			
		pdata.x[i] = x;
		pdata.y[i] = y;
		pdata.z[i] = z;
		}
	}

/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

/*! \param N number of particles to create
	\param phi_p Packing fraction of particles in the box
	\param min_dist Minimum distance two particles will be placed apart
	\note assumes particles have a diameter of 1
*/
RandomInitializerWithWalls::RandomInitializerWithWalls(unsigned int N, Scalar phi_p, Scalar min_dist) : RandomInitializer(N, phi_p, min_dist)
	{
	// sanity checks
	if (N == 0)
		throw runtime_error("RandomInitializer: Cannot generate 0 particles");
	if (phi_p <= 0)
		throw runtime_error("RandomInitializer: phi_p <= 0 doesn't make sense");
	if (min_dist < 0)
		throw runtime_error("RandomInitializer: min_dist <= 0 doesn't make sense");
	
	Scalar L = pow(Scalar(M_PI/6.0)*Scalar(N) / phi_p, Scalar(1.0/3.0));
	m_box = BoxDim(L);
	m_walls = WallData(m_box);
	}
	
RandomInitializerWithWalls::~RandomInitializerWithWalls() {
	

}


WallData RandomInitializerWithWalls::getWalls() const
	{
	return m_walls;
	}



#ifdef USE_PYTHON
void export_SimpleCubicInitializer()
	{
	class_< SimpleCubicInitializer, bases<ParticleDataInitializer> >("SimpleCubicInitializer", init<unsigned int, Scalar>())
		;
	// no need to .def methods, they are all inherited
	}
	
void export_RandomInitializer()
	{
	class_< RandomInitializer, bases<ParticleDataInitializer> >("RandomInitializer", init<unsigned int, Scalar, Scalar>())
		;
	// no need to .def methods, they are all inherited
	}

void export_RandomInitializerWithWalls()
	{
	class_< RandomInitializerWithWalls, bases<ParticleDataInitializer> >("RandomInitializerWithWalls", init<unsigned int, Scalar, Scalar>())
		;
	// no need to .def methods, they are all inherited
	}
#endif
