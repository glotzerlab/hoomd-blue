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

#include "LJForceCompute.h"
#include <stdexcept>

/*! \file LJForceCompute.cc
	\brief Contains code for the LJForceCompute class
*/

using namespace std;

/*! \param pdata Particle Data to compute forces on
 	\param nlist Neighborlist to use for computing the forces
	\param r_cut Cuttoff radius beyond which the force is 0
	\post memory is allocated and all parameters lj1 and lj2 are set to 0.0
*/
LJForceCompute::LJForceCompute(boost::shared_ptr<ParticleData> pdata, boost::shared_ptr<NeighborList> nlist, Scalar r_cut) 
	: ForceCompute(pdata), m_nlist(nlist), m_r_cut(r_cut)
	{
	assert(m_pdata);
	assert(m_nlist);
	
	if (r_cut < 0.0)
		throw runtime_error("Negative r_cut in LJForceCompute makes no sense");
	
	// initialize the number of types value
	m_ntypes = m_pdata->getNTypes();
	assert(m_ntypes > 0);
	
	// allocate data for lj1 and lj2
	m_lj1 = new Scalar[m_ntypes*m_ntypes];
	m_lj2 = new Scalar[m_ntypes*m_ntypes];
	
	// sanity check
	assert(m_lj1 != NULL && m_lj2 != NULL);
	
	// initialize the parameters to 0;
	memset((void*)m_lj1, 0, sizeof(Scalar)*m_ntypes*m_ntypes);
	memset((void*)m_lj2, 0, sizeof(Scalar)*m_ntypes*m_ntypes);
	}
	

LJForceCompute::~LJForceCompute()
	{
	// deallocate our memory
	delete[] m_lj1;
	delete[] m_lj2;
	m_lj1 = NULL;
	m_lj2 = NULL;
	}
		

/*! \post The parameters \a lj1 and \a lj2 are set for the pairs \a typ1, \a typ2 and \a typ2, \a typ1.
	\note \a lj? are low level parameters used in the calculation. In order to specify
	these for a normal lennard jones formula (with alpha), they should be set to the following.
	- \a lj1 = 4.0 * epsilon * pow(sigma,12.0)
	- \a lj2 = alpha * 4.0 * epsilon * pow(sigma,6.0);
	
	Setting the parameters for typ1,typ2 automatically sets the same parameters for typ2,typ1: there
	is no need to call this funciton for symmetric pairs. Any pairs that this function is not called
	for will have lj1 and lj2 set to 0.0.
	
	\param typ1 Specifies one type of the pair
	\param typ2 Specifies the second type of the pair
	\param lj1 First parameter used to calcluate forces
	\param lj2 Second parameter used to calculate forces
*/
void LJForceCompute::setParams(unsigned int typ1, unsigned int typ2, Scalar lj1, Scalar lj2)
	{
	if (typ1 >= m_ntypes || typ2 >= m_ntypes)
		{
		cerr << "Trying to set LJ params for a non existant type! " << typ1 << "," << typ2 << endl;
		throw runtime_error("Invalid type specification in LJForceCompute::setParams");
		}
	
	// set lj1 in both symmetric positions in the matrix	
	m_lj1[typ1*m_ntypes + typ2] = lj1;
	m_lj1[typ2*m_ntypes + typ1] = lj1;
	
	// set lj2 in both symmetric positions in the matrix
	m_lj2[typ1*m_ntypes + typ2] = lj2;
	m_lj2[typ2*m_ntypes + typ1] = lj2;
	}
	
/*! LJForceCompute provides
	- \c lj_energy
*/
std::vector< std::string > LJForceCompute::getProvidedLogQuantities()
	{
	vector<string> list;
	list.push_back("lj_energy");
	return list;
	}
	
Scalar LJForceCompute::getLogValue(const std::string& quantity)
	{
	if (quantity == string("lj_energy"))
		{
		return calcEnergySum();
		}
	else
		{
		cout << "Error! " << quantity << " is not a valid log quantity for LJForceCompute" << endl;
		throw runtime_error("Error getting log value");
		}
	}

/*! \post The lennard jones forces are computed for the given timestep. The neighborlist's
 	compute method is called to ensure that it is up to date.
	
	\param timestep specifies the current time step of the simulation
*/
void LJForceCompute::computeForces(unsigned int timestep)
	{
	// start by updating the neighborlist
	m_nlist->compute(timestep);
	
	// start the profile
	if (m_prof)
		m_prof->push("LJ.cpu");
	
	// depending on the neighborlist settings, we can take advantage of newton's third law
	// to reduce computations at the cost of memory access complexity: set that flag now
	bool third_law = m_nlist->getStorageMode() == NeighborList::half;
	
	// access the neighbor list
	const vector< vector< unsigned int > >& full_list = m_nlist->getList();

	// access the particle data
	const ParticleDataArraysConst& arrays = m_pdata->acquireReadOnly(); 
	// sanity check
	assert(arrays.x != NULL && arrays.y != NULL && arrays.z != NULL);
	
	// get a local copy of the simulation box too
	const BoxDim& box = m_pdata->getBox();
	// sanity check
	assert(box.xhi > box.xlo && box.yhi > box.ylo && box.zhi > box.zlo);	
	
	if (m_prof)
		m_prof->push("Compute");

	// create a temporary copy of r_cut sqaured
	Scalar r_cut_sq = m_r_cut * m_r_cut;	 
	
	// precalculate box lenghts for use in the periodic imaging
	Scalar Lx = box.xhi - box.xlo;
	Scalar Ly = box.yhi - box.ylo;
	Scalar Lz = box.zhi - box.zlo;
	
	// tally up the number of forces calculated
	int64_t n_calc = 0;
	int64_t n_force_calc = 0;
	
	// need to start from a zero force
	memset(m_fx, 0, sizeof(Scalar)*arrays.nparticles);
	memset(m_fy, 0, sizeof(Scalar)*arrays.nparticles);
	memset(m_fz, 0, sizeof(Scalar)*arrays.nparticles);
	memset(m_pe, 0, sizeof(Scalar)*arrays.nparticles);

	// for each particle
	for (unsigned int i = 0; i < arrays.nparticles; i++)
		{
		// access the particle's position
		Scalar xi = arrays.x[i];
		Scalar yi = arrays.y[i];
		Scalar zi = arrays.z[i];
		
		unsigned int typei = arrays.type[i];
		// sanity check
		assert(typei < m_pdata->getNTypes());
		
		// access the lj1 and lj2 rows for the current particle type
		Scalar * __restrict__ lj1_row = &(m_lj1[typei*m_ntypes]);
		Scalar * __restrict__ lj2_row = &(m_lj2[typei*m_ntypes]);

		Scalar fxi = 0.0;
		Scalar fyi = 0.0;
		Scalar fzi = 0.0;
		Scalar pei = 0.0;
		
		// loop over all of the neighbors of this particle
		const vector< unsigned int >& list = full_list[i];
		const unsigned int size = (unsigned int)list.size();
		
		for (unsigned int j = 0; j < size; j++)
			{
			// increment our calculation counter
			n_calc++;
			
			unsigned int k = list[j];
			// sanity check
			assert(k < m_pdata->getN());
				
			// calculate dr
			Scalar dx = xi - arrays.x[k];
			Scalar dy = yi - arrays.y[k];
			Scalar dz = zi - arrays.z[k];
			unsigned int typej = arrays.type[k];
			
			// sanity check
			assert(typej < m_pdata->getNTypes());
			
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
		
			Scalar fforce = 0.0;
			// only compute the force if the particles are closer than the cuttoff
			if (rsq < r_cut_sq)
				{
				// tally up how many forces we compute
				n_force_calc++;
					
				// compute the force magnitude/r
				Scalar r2inv = Scalar(1.0)/rsq;
				Scalar r6inv = r2inv * r2inv * r2inv;
				Scalar forcelj = r6inv * (Scalar(12.0)*lj1_row[typej]*r6inv - Scalar(6.0)*lj2_row[typej]);
				fforce = forcelj * r2inv;
				Scalar tmp_eng = r6inv * (lj1_row[typej]*r6inv - lj2_row[typej]);
				
				// add the force to the particle i
				fxi += dx*fforce;
				fyi += dy*fforce;
				fzi += dz*fforce;
				pei += Scalar(0.5)*tmp_eng;
				
				// add the force to particle j if we are using the third law
				if (third_law)
					{
					m_fx[k] -= dx*fforce;
					m_fy[k] -= dy*fforce;
					m_fz[k] -= dz*fforce;
					m_pe[k] += Scalar(0.5)*tmp_eng;
					}
				}
			
			}
		m_fx[i] += fxi;
		m_fy[i] += fyi;
		m_fz[i] += fzi;
		m_pe[i] += pei;
		}

	// and that is it.
	// FLOPS: 9+12 for each n_calc and an additional 11 for each n_full_calc
		// make that 14 if third_law is 1
	int64_t flops = (9+12)*n_calc + 16*n_force_calc;
	if (third_law)
		flops += 4*n_force_calc;
		
	// memory transferred: 3*sizeof(Scalar) + 2*sizeof(int) for each n_calc
	// plus 3*sizeof(Scalar) for each n_full_calc + another 3*sizeofScalar if third_law is 1
	// PLUS an additional 3*sizeof(Scalar) + sizeof(int) for each particle
	int64_t mem_transfer = 0;
	mem_transfer += (3*sizeof(Scalar) + 2*sizeof(int)) * (n_calc + arrays.nparticles);
	mem_transfer += 4*sizeof(Scalar)*n_force_calc;
	if (third_law)
		mem_transfer += 4*sizeof(Scalar);
	
	m_pdata->release();
	
	#ifdef USE_CUDA
	// the force data is now only up to date on the cpu
	m_data_location = cpu;
	#endif

	if (m_prof)
		{
		m_prof->pop(flops, mem_transfer);
		m_prof->pop();
		}
	}

#ifdef USE_PYTHON
void export_LJForceCompute()
	{
	class_<LJForceCompute, boost::shared_ptr<LJForceCompute>, bases<ForceCompute>, boost::noncopyable >
		("LJForceCompute", init< boost::shared_ptr<ParticleData>, boost::shared_ptr<NeighborList>, Scalar >())
		.def("setParams", &LJForceCompute::setParams)
		;
	}
#endif



