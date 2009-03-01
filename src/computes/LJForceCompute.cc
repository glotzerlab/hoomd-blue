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

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "LJForceCompute.h"
#include <stdexcept>

/*! \file LJForceCompute.cc
	\brief Defines the LJForceCompute class
*/

using namespace std;

/*! \param sysdef System to compute forces on
 	\param nlist Neighborlist to use for computing the forces
	\param r_cut Cuttoff radius beyond which the force is 0
	\post memory is allocated and all parameters lj1 and lj2 are set to 0.0
*/
LJForceCompute::LJForceCompute(boost::shared_ptr<SystemDefinition> sysdef, boost::shared_ptr<NeighborList> nlist, Scalar r_cut) 
	: ForceCompute(sysdef), m_nlist(nlist), m_r_cut(r_cut), m_shift_mode(no_shift), m_xplor_fraction(Scalar(2.0/3.0)), m_slj(false)
	{
	assert(m_pdata);
	assert(m_nlist);
	
	if (r_cut < 0.0)
		{
		cerr << endl << "***Error! Negative r_cut in LJForceCompute makes no sense" << endl << endl;
		throw runtime_error("Error initializing LJForceCompute");
		}
	
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
		cerr << endl << "***Error! Trying to set LJ params for a non existant type! " << typ1 << "," << typ2 << endl << endl;
		throw runtime_error("Error setting parameters in LJForceCompute");
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
	list.push_back("pair_lj_energy");
	return list;
	}
	
Scalar LJForceCompute::getLogValue(const std::string& quantity, unsigned int timestep)
	{
	if (quantity == string("pair_lj_energy"))
		{
		compute(timestep);
		return calcEnergySum();
		}
	else
		{
		cerr << endl << "***Error! " << quantity << " is not a valid log quantity for LJForceCompute" << endl << endl;
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
	
	// start the profile for this compute
	if (m_prof) m_prof->push("LJ pair");
	
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
	
	// factor out loop invariants
	Scalar r_on_sq = m_xplor_fraction*m_xplor_fraction * r_cut_sq;
	Scalar rcut2inv = Scalar(1.0) / r_cut_sq;
	Scalar rcut6inv = rcut2inv * rcut2inv * rcut2inv;
	Scalar xplor_denom_inv = Scalar(1.0) / ((r_cut_sq - r_on_sq) * (r_cut_sq - r_on_sq) * (r_cut_sq - r_on_sq));
	
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
		Scalar alphai = 0.0;
		if (m_slj) alphai = arrays.diameter[i]/2.0 - 1.0/2.0;  //Sigma here is being set to 1.0
		unsigned int typei = arrays.type[i];
		// sanity check
		assert(typei < m_pdata->getNTypes());
		
		// access the lj1 and lj2 rows for the current particle type
		Scalar * __restrict__ lj1_row = &(m_lj1[typei*m_ntypes]);
		Scalar * __restrict__ lj2_row = &(m_lj2[typei*m_ntypes]);

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
			Scalar alphaj = 0.0;
			if (m_slj) alphaj = arrays.diameter[k]/2.0 - 1.0/2.0;  //Sigma here is being set to 1.0

			
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
			
			// Shift the distance if diameter-shifted LJ force being used
			Scalar r, radj;
			if (m_slj) {
				r = sqrt(rsq);
				radj = r - alphai -alphaj; 
				rsq = radj*radj;		// This is now the diameter adjusted potential distance for slj
				}
				
			// only compute the force if the particles are closer than the cuttoff (FLOPS: 1)
			if (rsq < r_cut_sq)
				{
				// compute the force magnitude/r in forcemag_divr (FLOPS: 9)
				Scalar r2inv = Scalar(1.0)/rsq;
				Scalar r6inv = r2inv * r2inv * r2inv;
				Scalar forcemag_divr; 
				if (m_slj) {
					Scalar radj_inv = Scalar(1.0)/radj;
					Scalar r_inv = Scalar(1.0)/r;
					forcemag_divr = radj_inv * r_inv * r6inv * (Scalar(12.0)*lj1_row[typej]*r6inv - Scalar(6.0)*lj2_row[typej]);
					}
				else forcemag_divr= r2inv * r6inv * (Scalar(12.0)*lj1_row[typej]*r6inv - Scalar(6.0)*lj2_row[typej]);
				
				
				// compute the pair energy (FLOPS: 4)
				Scalar pair_eng = Scalar(0.5) * r6inv * (lj1_row[typej]*r6inv - lj2_row[typej]);
				
				if (m_shift_mode == shift)
					{
					// shifting is enabled: shift the energy (FLOPS: 5)
					pair_eng -= Scalar(0.5) * rcut6inv * (lj1_row[typej]*rcut6inv - lj2_row[typej]);
					}
				else
				if (m_shift_mode == xplor)
					{
					if (rsq >= r_on_sq)
						{
						// Implement XPLOR smoothing (FLOPS: 16)
						Scalar old_pair_eng = pair_eng;
						Scalar old_forcemag_divr = forcemag_divr;
						
						Scalar rsq_minus_r_cut_sq = rsq - r_cut_sq;
						Scalar s = rsq_minus_r_cut_sq * rsq_minus_r_cut_sq * (r_cut_sq + Scalar(2.0) * rsq - Scalar(3.0) * r_on_sq) * xplor_denom_inv;
						Scalar ds_dr_divr = Scalar(12.0) * (rsq - r_on_sq) * rsq_minus_r_cut_sq * xplor_denom_inv;
						
						// make modifications to the old pair energy and force
						pair_eng = old_pair_eng * s;
						// note: I'm not sure why the minus sign needs to be there: my notes have a +. But this is verified correct
						// I think it might have something to do with the fact that I'm actually calculating \vec{r}_{ji} instead of {ij}
						forcemag_divr = s * old_forcemag_divr - ds_dr_divr * Scalar(2.0)*old_pair_eng;
						}
					}
					
				// compute the virial (FLOPS: 2)
				// note the sign in the virial calculation, this is because dx,dy,dz are \vec{r}_{ji} thus
				// there is no - in the 1/6 to compensate	
				Scalar pair_virial;
				if (!m_slj) pair_virial = Scalar(1.0/6.0) * rsq * forcemag_divr;
				else pair_virial = Scalar(1.0/6.0) * r * r * forcemag_divr;  //rsq has been "adjusted" for diameter, r has not!
				
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
	if (m_shift_mode == shift)
		flops += n_calc * 5;
	else
	if (m_shift_mode == xplor)
		flops += n_calc * 16;
	
	if (m_slj) flops += 10;  //At least I count 10 extra calculations, though I am counting a sqrt as only 1 FLOP.
	

	if (third_law) flops += n_calc * 8;
	int64_t mem_transfer = m_pdata->getN() * (5+4+10)*sizeof(Scalar) + n_calc * (1+3+1)*sizeof(Scalar);
	if (third_law) mem_transfer += n_calc*10*sizeof(Scalar);
	if (m_prof) m_prof->pop(flops, mem_transfer);
	}

void export_LJForceCompute()
	{
	scope in_lj = class_<LJForceCompute, boost::shared_ptr<LJForceCompute>, bases<ForceCompute>, boost::noncopyable >
		("LJForceCompute", init< boost::shared_ptr<SystemDefinition>, boost::shared_ptr<NeighborList>, Scalar >())
		.def("setParams", &LJForceCompute::setParams)
		.def("setXplorFraction", &LJForceCompute::setXplorFraction)
		.def("setShiftMode", &LJForceCompute::setShiftMode)
		.def("setSLJ", &LJForceCompute::setSLJ)
		;
		
	enum_<LJForceCompute::energyShiftMode>("energyShiftMode")
		.value("no_shift", LJForceCompute::no_shift)
		.value("shift", LJForceCompute::shift)
		.value("xplor", LJForceCompute::xplor)
		;
	}

#ifdef WIN32
#pragma warning( pop )
#endif
