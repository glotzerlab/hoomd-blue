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

// $Id: StochasticForceCompute.cc 1234 2008-09-11 16:29:13Z joaander $
// $URL: http://svn2.assembla.com/svn/hoomd/trunk/src/computes/StochasticForceCompute.cc $ 

/*! \file StochasticForceCompute.cc
	\brief Defines the StochasticForceCompute class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "StochasticForceCompute.h"
#include <stdexcept>
#include <math.h>

// May want to change this to the Boost Library random number generator... but lets start with something simple
#include <cstdlib>

using namespace std;

/*! \param pdata ParticleData to compute forces on
	\param Temp Temperature of the bath of random particles
	\param deltaT Length of the computation timestep
	\param seed	Seed for initializing the RNG
*/
StochasticForceCompute::StochasticForceCompute(boost::shared_ptr<ParticleData> pdata, Scalar deltaT, Scalar Temp, unsigned int seed):
ForceCompute(pdata), m_T(Temp), m_dt(deltaT), m_seed(seed)
	{
	if (m_T <= 0.0)
		{
		cerr << endl << "***Error! Negative or Zero Temperature in StochasticForceCompute doesn't make sense." << endl << endl;
		throw runtime_error("Error initializing StochasticForceCompute");
		}
		
	// initialize the number of types value
	m_ntypes = m_pdata->getNTypes();
	assert(m_ntypes > 0);
	
	// allocate memory for friction coefficients
	m_gamma = new Scalar[m_ntypes]; 
	
	// sanity check
	assert(m_gamma != NULL);
	
	// initialize the parameters to 1;
	//memset((void *)m_gamma, Scalar(1),sizeof(Scalar)*m_ntypes);
	for (unsigned int i = 0; i < m_ntypes; i++) m_gamma[i] = Scalar(1.0);

    // seed the RNG
	srand(seed);
	}

/*! Frees used memory
*/
StochasticForceCompute::~StochasticForceCompute()
	{
	delete[] m_gamma;
	m_gamma = NULL;
	}

/*! \param typ Particle type index to set parameters for
	\param gamma gamma parameter

	\note \a gamma represents the friction coefficient set for a particle.  
	It is the users choice what aspect of the particle determines 
	its friction coefficient, but diameter is often used
*/
void StochasticForceCompute::setParams(unsigned int typ, Scalar gamma)
	{
	if (typ >= m_pdata->getNTypes())
		{
		cerr << endl << "***Error! Trying to set gamma params for a non existant type! " << typ << endl << endl;
		throw runtime_error("Error setting params in StochasticForceCompute");
		}
	
	// set the parameters
	m_gamma[typ] = gamma;
	}
	
/*! StochasticForceCompute provides
	- \c Nothing
*/
std::vector< std::string > StochasticForceCompute::getProvidedLogQuantities()
	{
	vector<string> list;
   //return empty list.  This force provides no logged quantities
	return list;
	}
	

void StochasticForceCompute::computeForces(unsigned int timestep)
	{
	// start the profile for this compute
	if (m_prof) m_prof->push("Stochastic Force");	
	
	// get numparticle var for easier access
	unsigned int numParticles = m_pdata->getN();

	// access the particle data
	const ParticleDataArraysConst &particles=  m_pdata->acquireReadOnly();
	
	// here we go, main calc loop
	// loop over every particle in the sim, 
	// calculate forces and store them int m_fx,y,z
	for (unsigned int i = 0; i < numParticles; i++)
		{

        // No potential energy results from these dissipative forces
		Scalar pe = 0.0;  

		// Grab particle data from all the arrays for this loop
		Scalar vx = particles.vx[i];
		Scalar vy = particles.vy[i];
		Scalar vz = particles.vz[i];
		unsigned int type = particles.type[i];

		// Here need to acquire three random numbers
		Scalar rx = Scalar(2.0)*Scalar(rand())/Scalar(RAND_MAX) - Scalar(1.0);
		Scalar ry = Scalar(2.0)*Scalar(rand())/Scalar(RAND_MAX) - Scalar(1.0);
		Scalar rz = Scalar(2.0)*Scalar(rand())/Scalar(RAND_MAX) - Scalar(1.0);
			
	    // Calculate the coefficient  (How do I get dt?? - The World's most klugey method, the integrator must pass this along)
		// Note, this formulation assumes a unit value for the boltzmann constant, kb
		Scalar coeff_fric = sqrt(Scalar(6.0)*m_gamma[type]*m_T/m_dt);
			
		m_fx[i] = rx*coeff_fric - m_gamma[type]*vx;
		m_fy[i] = ry*coeff_fric - m_gamma[type]*vy;
		m_fz[i] = rz*coeff_fric - m_gamma[type]*vz;
		m_pe[i] = pe;
		}
	
	#ifdef USE_CUDA
	m_data_location = cpu;
	#endif
	m_pdata->release();
	
	if (m_prof) m_prof->pop();
	}

#ifdef WIN32
#pragma warning( pop )
#endif
