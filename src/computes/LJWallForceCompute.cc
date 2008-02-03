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

#include "LJWallForceCompute.h"
#include <stdexcept>

using namespace std;

LJWallForceCompute::LJWallForceCompute(boost::shared_ptr<ParticleData> pdata, Scalar r_cut):
ForceCompute(pdata), m_r_cut(r_cut)
	{
	
	if (r_cut < 0.0) 
		throw runtime_error("Negative r_cut in LJWallForceCompute doesn't make sense.");
		
	// initialize the number of types value
	m_ntypes = m_pdata->getNTypes();
	assert(m_ntypes > 0);
	
	// allocate data for lj1 and lj2
	m_lj1 = new Scalar[m_ntypes];
	m_lj2 = new Scalar[m_ntypes];
	
	// sanity check
	assert(m_lj1 != NULL && m_lj2 != NULL);
	
	//Make walls
	// TODO: move wall data into particle data and make a class
	//m_pdata.setWalls(WallData(pdata->getBox()));
	
	// initialize the parameters to 0;
	memset((void*)m_lj1, 0, sizeof(Scalar)*m_ntypes);
	memset((void*)m_lj2, 0, sizeof(Scalar)*m_ntypes);
	}

LJWallForceCompute::~LJWallForceCompute(void)
	{
	delete[] m_lj1;
	delete[] m_lj2;
	m_lj1 = NULL;
	m_lj2 = NULL;

	}

//coppied from LJForceCompute
void LJWallForceCompute::setParams(unsigned int typ1, Scalar lj1, Scalar lj2)
	{
	if (typ1 >= m_ntypes)
	{
	cerr << "Trying to set LJ params for a non existant type! " << typ1 << endl;
	throw runtime_error("Invalid type specification in LJForceCompute::setParams");
	}
	
	// set lj1 in both symmetric positions in the matrix	
	m_lj1[typ1] = lj1;
	m_lj2[typ1] = lj2;
	}

void LJWallForceCompute::computeForces(unsigned int timestep){
	
	// get numparticle var for easier access
	unsigned int numParticles = m_pdata->getN();
	unsigned int numWalls = m_pdata->getWalls().getWallArrays().numWalls;

	// allocate space for forces to be stored
	//zero forces because most particles won't be close to a wall
	memset(m_fx, 0, sizeof(Scalar) * numParticles);
	memset(m_fy, 0, sizeof(Scalar) * numParticles);
	memset(m_fz, 0, sizeof(Scalar) * numParticles);

	//Initialize some force variables to be used as temporary
	//storage in each iteration
	Scalar fx, fy, fz;

	Scalar r_cut_sq = m_r_cut * m_r_cut;

	const ParticleDataArraysConst &particles=  m_pdata->acquireReadOnly();
	
	BoxDim box = m_pdata->getBox();

	// precalculate box lengths for use in the periodic imaging
	Scalar Lx = box.xhi - box.xlo;
	Scalar Ly = box.yhi - box.ylo;
	Scalar Lz = box.zhi - box.zlo;

	  //here we go, main calc loop
	 // loop over every particle in the sim, 
	//calculate forces and store them int m_fx,y,z
	for (unsigned int i = 0; i < numParticles; i++){

		//Grab particle data from all the arrays for this loop
		Scalar px = particles.x[i];
		Scalar py = particles.y[i];
		Scalar pz = particles.z[i];
		unsigned int typei = particles.type[i];


		  //for each wall that exists in the simulation
		 //calculate the force that it exerts on a particle
		//the sum of the forces from each wall is the resulting force
		for (unsigned int j = 0; j < numWalls; j++){
			
			// Grab wall data for the current wall
			Scalar nx = m_pdata->getWalls().getWallArrays().nx[j];
			Scalar ny = m_pdata->getWalls().getWallArrays().ny[j];
			Scalar nz = m_pdata->getWalls().getWallArrays().nz[j];
			Scalar ox = m_pdata->getWalls().getWallArrays().ox[j];
			Scalar oy = m_pdata->getWalls().getWallArrays().oy[j];
			Scalar oz = m_pdata->getWalls().getWallArrays().oz[j];
		
			// calculate distance from point to plane
			// http://mathworld.wolfram.com/Point-PlaneDistance.html
			Scalar distFromWall = nx * (px - ox) + ny * (py - oy) + nz * (pz - oz);

			// access the lj1 and lj2 rows for the current particle type
			Scalar * __restrict__ lj1_row = &(m_lj1[typei*m_ntypes]);
			Scalar * __restrict__ lj2_row = &(m_lj2[typei*m_ntypes]);
			
			// use the distance to create a vector pointing from the plane to the particle
			Scalar dx = nx * distFromWall;
			Scalar dy = ny * distFromWall;
			Scalar dz = nz * distFromWall;
			
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
				Scalar forcelj = r6inv * (lj1_row[typei]*r6inv - lj2_row[typei]);
				Scalar fforce = forcelj * r2inv;
			                                
				// calculate the force vector
				fx = dx*fforce;
				fy = dy*fforce;
				fz = dz*fforce;
				}
			else
				fx = fy = fz = Scalar(0.0);

			m_fx[i] += fx;
			m_fy[i] += fy;
			m_fz[i] += fz;
		}

	}
	
	#ifdef USE_CUDA
	m_data_location = cpu;
	#endif
	m_pdata->release();
}


