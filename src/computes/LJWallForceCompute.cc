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

/*! \file LJWallForceCompute.cc
	\brief Defines the LJWallForceCompute class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "LJWallForceCompute.h"
#include "WallData.h"
#include <stdexcept>

using namespace std;

/*! \param sysdef System to compute forces on
	\param r_cut Cuttoff distance beyond which the force is zero.
*/
LJWallForceCompute::LJWallForceCompute(boost::shared_ptr<SystemDefinition> sysdef, Scalar r_cut):
ForceCompute(sysdef), m_r_cut(r_cut)
	{
	if (r_cut < 0.0)
		{
		cerr << endl << "***Error! Negative r_cut in LJWallForceCompute doesn't make sense." << endl << endl;
		throw runtime_error("Error initializing LJWallForceCompute");
		}
		
	// initialize the number of types value
	unsigned int ntypes = m_pdata->getNTypes();
	assert(ntypes > 0);
	
	// allocate data for lj1 and lj2
	m_lj1 = new Scalar[ntypes];
	m_lj2 = new Scalar[ntypes];
	
	// sanity check
	assert(m_lj1 != NULL && m_lj2 != NULL);
	
	// initialize the parameters to 0;
	memset((void*)m_lj1, 0, sizeof(Scalar)*ntypes);
	memset((void*)m_lj2, 0, sizeof(Scalar)*ntypes);
	}

/*! Frees used memory
*/
LJWallForceCompute::~LJWallForceCompute()
	{
	delete[] m_lj1;
	delete[] m_lj2;
	m_lj1 = NULL;
	m_lj2 = NULL;
	}

/*! \param typ Particle type index to set parameters for
	\param lj1 lj1 parameter
	\param lj2 lj2 parameter

	\note \a lj1 are \a lj2 are low level parameters used in the calculation. In order to specify
	these for a normal lennard jones formula (with alpha), they should be set to the following.
	\a lj1 = 4.0 * epsilon * pow(sigma,12.0)
	\a lj2 = alpha * 4.0 * epsilon * pow(sigma,6.0);
*/
void LJWallForceCompute::setParams(unsigned int typ, Scalar lj1, Scalar lj2)
	{
	if (typ >= m_pdata->getNTypes())
		{
		cerr << endl << "***Error! Trying to set LJ params for a non existant type! " << typ << endl << endl;
		throw runtime_error("Error setting params in LJWallForceCompute");
		}
	
	// set the parameters
	m_lj1[typ] = lj1;
	m_lj2[typ] = lj2;
	}
	
/*! LJWallForceCompute provides
	- \c wall_lj_energy
*/
std::vector< std::string > LJWallForceCompute::getProvidedLogQuantities()
	{
	vector<string> list;
	list.push_back("wall_lj_energy");
	return list;
	}

Scalar LJWallForceCompute::getLogValue(const std::string& quantity, unsigned int timestep)
	{
	if (quantity == string("wall_lj_energy"))
		{
		compute(timestep);
		return calcEnergySum();
		}
	else
		{
		cerr << endl << "***Error! " << quantity << " is not a valid log quantity for LJWallForceCompute" << endl << endl;
		throw runtime_error("Error getting log value");
		}
	}

void LJWallForceCompute::computeForces(unsigned int timestep)
	{
	// start the profile for this compute
	if (m_prof) m_prof->push("LJ wall");	
	
	// get numparticle var for easier access
	unsigned int numParticles = m_pdata->getN();
	boost::shared_ptr<WallData> wall_data = m_sysdef->getWallData();
	unsigned int numWalls = wall_data->getNumWalls();

	// precalculate r_cut squqred
	Scalar r_cut_sq = m_r_cut * m_r_cut;

	// precalculate box lengths for use in the periodic imaging
	BoxDim box = m_pdata->getBox();
	Scalar Lx = box.xhi - box.xlo;
	Scalar Ly = box.yhi - box.ylo;
	Scalar Lz = box.zhi - box.zlo;

	// access the particle data
	const ParticleDataArraysConst &particles=  m_pdata->acquireReadOnly();
	
	// here we go, main calc loop
	// loop over every particle in the sim, 
	// calculate forces and store them int m_fx,y,z
	for (unsigned int i = 0; i < numParticles; i++)
		{
		// Initialize some force variables to be used as temporary
		// storage in each iteration, initialized to 0 from which the force will be computed
		Scalar fx = 0.0, fy = 0.0, fz = 0.0;
		Scalar pe = 0.0;

		// Grab particle data from all the arrays for this loop
		Scalar px = particles.x[i];
		Scalar py = particles.y[i];
		Scalar pz = particles.z[i];
		unsigned int type = particles.type[i];

		// for each wall that exists in the simulation
		// calculate the force that it exerts on a particle
		// the sum of the forces from each wall is the resulting force
		for (unsigned int cur_wall_idx = 0; cur_wall_idx < numWalls; cur_wall_idx++)
			{
			const Wall& cur_wall = wall_data->getWall(cur_wall_idx);
		
			// calculate distance from point to plane
			// http://mathworld.wolfram.com/Point-PlaneDistance.html
			Scalar distFromWall = cur_wall.normal_x * (px - cur_wall.origin_x) 
								+ cur_wall.normal_y * (py - cur_wall.origin_y) 
								+ cur_wall.normal_z * (pz - cur_wall.origin_z);

			// use the distance to create a vector pointing from the plane to the particle
			Scalar dx = cur_wall.normal_x * distFromWall;
			Scalar dy = cur_wall.normal_y * distFromWall;
			Scalar dz = cur_wall.normal_z * distFromWall;
			
			// continue with the evaluation of the LJ force copied from LJForceCompute
			// apply periodic boundary conditions
			if (dx >= box.xhi)
				dx -= Lx;
			else
			if (dx < box.xlo)
				dx += Lx;

			if (dy >= box.yhi)
				dy -= Ly;
			else
			if (dy < box.ylo)
				dy += Ly;
			
			if (dz >= box.zhi)
				dz -= Lz;
			else
			if (dz < box.zlo)
				dz += Lz;
			
			// start computing the force
			Scalar rsq = dx*dx + dy*dy + dz*dz;
			
			// only compute the force if the particles are closer than the cuttoff
			if (rsq < r_cut_sq)
				{
				// compute the force magnitude/r
				Scalar r2inv = Scalar(1.0)/rsq;
				Scalar r6inv = r2inv * r2inv * r2inv;
				Scalar forcelj = r6inv * (Scalar(12.0)*m_lj1[type]*r6inv - Scalar(6.0)*m_lj2[type]);
				Scalar fforce = forcelj * r2inv;
				Scalar tmp_eng = r6inv * (m_lj1[type]*r6inv - m_lj2[type]);
			
				// accumulate the force vector
				fx += dx*fforce;
				fy += dy*fforce;
				fz += dz*fforce;
				pe += tmp_eng;
				}
			}
		
		m_fx[i] = fx;
		m_fy[i] = fy;
		m_fz[i] = fz;
		m_pe[i] = pe;
		}
	
	#ifdef ENABLE_CUDA
	m_data_location = cpu;
	#endif
	m_pdata->release();
	
	if (m_prof) m_prof->pop();
	}

void export_LJWallForceCompute()
	{
	class_<LJWallForceCompute, boost::shared_ptr<LJWallForceCompute>, bases<ForceCompute>, boost::noncopyable >
		("LJWallForceCompute", init< boost::shared_ptr<SystemDefinition>, Scalar >())
		.def("setParams", &LJWallForceCompute::setParams)
		;
	}

#ifdef WIN32
#pragma warning( pop )
#endif
