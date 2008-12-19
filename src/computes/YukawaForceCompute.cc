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

// $Id: YukawaForceCompute.cc 1325 2008-10-06 13:42:07Z phillicl $
// $URL: http://svn2.assembla.com/svn/hoomd/trunk/src/computes/YukawaForceCompute.cc $

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "YukawaForceCompute.h"
#include <stdexcept>

/*! \file YukawaForceCompute.cc
	\brief Contains code for the YukawaForceCompute class
*/

using namespace std;

/*! \param pdata Particle Data to compute forces on
 	\param nlist Neighborlist to use for computing the forces
	\param r_cut Cuttoff radius beyond which the force is 0
	\param kappa Coefficient to use in calculating the force
	\post memory is allocated and the parameter epsilon is set to 0.0
*/
YukawaForceCompute::YukawaForceCompute(boost::shared_ptr<ParticleData> pdata, boost::shared_ptr<NeighborList> nlist, Scalar r_cut, Scalar kappa) 
	: ForceCompute(pdata), m_nlist(nlist), m_r_cut(r_cut), m_kappa(kappa)
	{
	assert(m_pdata);
	assert(m_nlist);
	
	if (r_cut < 0.0)
		{
		cerr << endl << "***Error! Negative r_cut in YukawaForceCompute makes no sense" << endl << endl;
		throw runtime_error("Error initializing YukawaForceCompute");
		}
	
	// TODO - Right Here a calculation of the Energy being neglected in the cutoff and a warning about using Kappa values that are too small
	
	// initialize the number of types value
	m_ntypes = m_pdata->getNTypes();
	assert(m_ntypes > 0);
	
	// allocate data for epsilon
	m_epsilon = new Scalar[m_ntypes*m_ntypes];
	
	// sanity check
	assert(m_epsilon != NULL);
	
	// initialize the parameters to 0;
	memset((void*)m_epsilon, 0, sizeof(Scalar)*m_ntypes*m_ntypes);
	}
	

YukawaForceCompute::~YukawaForceCompute()
	{
	// deallocate our memory
	delete[] m_epsilon;
	m_epsilon = NULL;
	}
		

/*! \post The parameter \a epsilon is set for the pairs \a typ1, \a typ2 and \a typ2, \a typ1.
	\note \a epsilon are low level parameters used in the calculation. In order to specify
	these for a normal lennard jones formula (with alpha), they should be set to the following.
	- \a epsilon = 
	
	Setting the parameters for typ1,typ2 automatically sets the same parameters for typ2,typ1: there
	is no need to call this funciton for symmetric pairs. Any pairs that this function is not called
	for will have epsilon set to 0.0.
	
	\param typ1 Specifies one type of the pair
	\param typ2 Specifies the second type of the pair
	\param epsilon First parameter used to calcluate forces
*/
void YukawaForceCompute::setParams(unsigned int typ1, unsigned int typ2, Scalar epsilon)
	{
	if (typ1 >= m_ntypes || typ2 >= m_ntypes)
		{
		cerr << endl << "***Error! Trying to set Yukawa params for a non existant type! " << typ1 << "," << typ2 << endl << endl;
		throw runtime_error("Error setting parameters in YukawaForceCompute");
		}
	
	// set epsilon in both symmetric positions in the matrix	
	m_epsilon[typ1*m_ntypes + typ2] = epsilon;
	m_epsilon[typ2*m_ntypes + typ1] = epsilon;

	}
	
/*! YukawaForceCompute provides
	- \c yukawa_energy
*/
std::vector< std::string > YukawaForceCompute::getProvidedLogQuantities()
	{
	vector<string> list;
	list.push_back("pair_yukawa_energy");
	return list;
	}
	
Scalar YukawaForceCompute::getLogValue(const std::string& quantity, unsigned int timestep)
	{
	if (quantity == string("pair_yukawa_energy"))
		{
		compute(timestep);
		return calcEnergySum();
		}
	else
		{
		cerr << endl << "***Error! " << quantity << " is not a valid log quantity for YukawaForceCompute" << endl << endl;
		throw runtime_error("Error getting log value");
		}
	}

/*! \post The lennard jones forces are computed for the given timestep. The neighborlist's
 	compute method is called to ensure that it is up to date.
	
	\param timestep specifies the current time step of the simulation
*/
void YukawaForceCompute::computeForces(unsigned int timestep)
	{
	// start by updating the neighborlist
	m_nlist->compute(timestep);
	
	// start the profile for this compute
	if (m_prof) m_prof->push("Yukawa pair");
	
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
	
	// create a temporary copy of r_cut sqaured
	Scalar r_cut_sq = m_r_cut * m_r_cut;
	
	// precalculate box lenghts for use in the periodic imaging
	Scalar Lx = box.xhi - box.xlo;
	Scalar Ly = box.yhi - box.ylo;
	Scalar Lz = box.zhi - box.zlo;
	
	// tally up the number of forces calculated
	int64_t n_calc = 0;
	
	// need to start from a zero force, energy and virial
	// (MEM TRANSFER: 5*N scalars)
	memset(m_fx, 0, sizeof(Scalar)*arrays.nparticles);
	memset(m_fy, 0, sizeof(Scalar)*arrays.nparticles);
	memset(m_fz, 0, sizeof(Scalar)*arrays.nparticles);
	memset(m_pe, 0, sizeof(Scalar)*arrays.nparticles);
	memset(m_virial, 0, sizeof(Scalar)*arrays.nparticles);

	// for each particle
	for (unsigned int i = 0; i < arrays.nparticles; i++)
		{
		// access the particle's position and type (MEM TRANSFER: 4 scalars)
		Scalar xi = arrays.x[i];
		Scalar yi = arrays.y[i];
		Scalar zi = arrays.z[i];
		unsigned int typei = arrays.type[i];
		// sanity check
		assert(typei < m_pdata->getNTypes());
		
		// access the epsilon row for the current particle type
		Scalar * __restrict__ epsilon_row = &(m_epsilon[typei*m_ntypes]);

		// initialize current particle force, potential energy, and virial to 0
		Scalar fxi = 0.0;
		Scalar fyi = 0.0;
		Scalar fzi = 0.0;
		Scalar pei = 0.0;
		Scalar viriali = 0.0;
		
		// loop over all of the neighbors of this particle
		const vector< unsigned int >& list = full_list[i];
		const unsigned int size = (unsigned int)list.size();
		for (unsigned int j = 0; j < size; j++)
			{
			// increment our calculation counter
			n_calc++;
			
			// access the index of this neighbor (MEM TRANSFER: 1 scalar)
			unsigned int k = list[j];
			// sanity check
			assert(k < m_pdata->getN());
				
			// calculate dr (MEM TRANSFER: 3 scalars / FLOPS: 3)
			Scalar dx = xi - arrays.x[k];
			Scalar dy = yi - arrays.y[k];
			Scalar dz = zi - arrays.z[k];
			
			// access the type of the neighbor particle (MEM TRANSFER: 1 scalar
			unsigned int typej = arrays.type[k];
			// sanity check
			assert(typej < m_pdata->getNTypes());
			
			// apply periodic boundary conditions (FLOPS: 9 (worst case: first branch is missed, the 2nd is taken and the add is done)
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
			// calculate r squared (FLOPS: 5)
			Scalar rsq = dx*dx + dy*dy + dz*dz;
		
			// only compute the force if the particles are closer than the cuttoff (FLOPS: 1)
			if (rsq < r_cut_sq)
				{
				// compute the force magnitude/r in forcemag_divr (FLOPS: 9)
				Scalar r = sqrt(rsq);
				Scalar rinv = Scalar(1.0)/r;
				Scalar r2inv = Scalar(1.0)/rsq;
				Scalar forcemag_divr = epsilon_row[typej]*exp(-m_kappa*r)*r2inv*(m_kappa + rinv);
				
				// compute the pair energy and virial (FLOPS: 6)
				// note the sign in the virial calculation, this is because dx,dy,dz are \vec{r}_{ji} thus
				// there is no - in the 1/6 to compensate				
				Scalar pair_virial = Scalar(1.0/6.0) * rsq * forcemag_divr;
				Scalar pair_eng = Scalar(0.5) * epsilon_row[typej]*exp(-m_kappa*r)*rinv; // Plus Shift?
				
				// add the force, potential energy and virial to the particle i
				// (FLOPS: 8)
				fxi += dx*forcemag_divr;
				fyi += dy*forcemag_divr;
				fzi += dz*forcemag_divr;
				pei += pair_eng;
				viriali += pair_virial;
				
				// add the force to particle j if we are using the third law (MEM TRANSFER: 10 scalars / FLOPS: 8)
				if (third_law)
					{
					m_fx[k] -= dx*forcemag_divr;
					m_fy[k] -= dy*forcemag_divr;
					m_fz[k] -= dz*forcemag_divr;
					m_pe[k] += pair_eng;
					m_virial[k] += pair_virial;
					}
				}
			
			}
			
		// finally, increment the force, potential energy and virial for particle i
		// (MEM TRANSFER: 10 scalars / FLOPS: 5)
		m_fx[i] += fxi;
		m_fy[i] += fyi;
		m_fz[i] += fzi;
		m_pe[i] += pei;
		m_virial[i] += viriali;
		}

	m_pdata->release();
	
	#ifdef ENABLE_CUDA
	// the force data is now only up to date on the cpu
	m_data_location = cpu;
	#endif

	int64_t flops = m_pdata->getN() * 5 + n_calc * (3+5+9+1+9+6+8);
	if (third_law) flops += n_calc * 8;
	int64_t mem_transfer = m_pdata->getN() * (5+4+10)*sizeof(Scalar) + n_calc * (1+3+1)*sizeof(Scalar);
	if (third_law) mem_transfer += n_calc*10*sizeof(Scalar);
	if (m_prof) m_prof->pop(flops, mem_transfer);
	}

void export_YukawaForceCompute()
	{
	class_<YukawaForceCompute, boost::shared_ptr<YukawaForceCompute>, bases<ForceCompute>, boost::noncopyable >
		("YukawaForceCompute", init< boost::shared_ptr<ParticleData>, boost::shared_ptr<NeighborList>, Scalar, Scalar >())
		.def("setParams", &YukawaForceCompute::setParams)
		;
	}

#ifdef WIN32
#pragma warning( pop )
#endif
