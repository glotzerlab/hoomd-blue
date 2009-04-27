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

// $Id: FENEBondForceCompute.cc 1127 2008-08-31 19:54:39Z phillicl $
// $URL: https://svn2.assembla.com/svn/hoomd/tags/hoomd-0.7.0/src/computes/FENEBondForceCompute.cc $
// Maintainer: phillicl

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "FENEBondForceCompute.h"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <math.h>

using namespace std;

/*! \file FENEBondForceCompute.cc
	\brief Defines the FENEBondForceCompute class
*/

/*! \param pdata Particle data to compute forces on
	\post Memory is allocated, default parameters are set and forces are zeroed.
*/
FENEBondForceCompute::FENEBondForceCompute(boost::shared_ptr<ParticleData> pdata) :	ForceCompute(pdata),
	m_K(NULL), m_r_0(NULL), m_lj1(NULL), m_lj2(NULL), m_epsilon(NULL)
	{
	// access the bond data for later use
	m_bond_data = m_pdata->getBondData();
	
	// check for some silly errors a user could make 
	if (m_bond_data->getNBondTypes() == 0)
		{
		cout << endl << "***Error! No bond types specified" << endl << endl;
		throw runtime_error("Error initializing FENEBondForceCompute");
		}
		
	// allocate the parameters
	m_K = new Scalar[m_bond_data->getNBondTypes()];
	m_r_0 = new Scalar[m_bond_data->getNBondTypes()];
	m_lj1 = new Scalar[m_bond_data->getNBondTypes()];
	m_lj2 = new Scalar[m_bond_data->getNBondTypes()];
	m_epsilon = new Scalar[m_bond_data->getNBondTypes()];

	
	// initialize parameters
	memset(m_K, 0, sizeof(Scalar) * m_bond_data->getNBondTypes());
	memset(m_r_0, 0, sizeof(Scalar) * m_bond_data->getNBondTypes());
	for (unsigned int i = 0; i < m_bond_data->getNBondTypes(); i++) m_lj1[i]=Scalar(1.0); 
	for (unsigned int i = 0; i < m_bond_data->getNBondTypes(); i++) m_lj2[i]=Scalar(1.0); 
	memset(m_epsilon, 0, sizeof(Scalar) * m_bond_data->getNBondTypes());
	}
	
FENEBondForceCompute::~FENEBondForceCompute()
	{
	delete[] m_K;
	delete[] m_r_0;
	delete[] m_lj1;
	delete[] m_lj2;
	delete[] m_epsilon;
	}
	
/*! \param type Type of the bond to set parameters for
	\param K Stiffness parameter for the force computation
	\param r_0 maximum bond length for the force computation
	\param sigma Value of sigma in the force calculation
	\param epsilon Value of epsilon in the force calculation

	Sets parameters for the potential of a particular bond type
*/
void FENEBondForceCompute::setParams(unsigned int type, Scalar K, Scalar r_0, Scalar sigma, Scalar epsilon)
	{
	// make sure the type is valid
	if (type >= m_bond_data->getNBondTypes())
		{
		cout << endl << "***Error! Invalid bond type specified" << endl << endl;
		throw runtime_error("Error setting parameters in FENEBondForceCompute");
		}
	
	m_K[type] = K;
	m_r_0[type] = r_0;
	m_lj1[type] = 4*epsilon*pow(sigma,12);
	m_lj2[type] = 4*epsilon*pow(sigma,6);
	m_epsilon[type] = epsilon;
	
   //cout << 	"Setting FENE parameters K=" << K << ", r0=" << r_0 << ", sigma=" << sigma <<  ", lj1=" << m_lj1[type] << ", lj2=" << m_lj2[type] << ", epsilon=" << m_epsilon[type] << endl;

	// check for some silly errors a user could make 
	if (K <= 0)
		cout << "***Warning! K <= 0 specified for fene bond" << endl;
	if (r_0 <= 0)
		cout << "***Warning! r_0 <= 0 specified for fene bond" << endl;
	if (sigma <= 0)
		cout << "***Warning! sigma <= 0 specified for fene bond" << endl;
	if (epsilon <= 0)
		cout << "***Warning! epsilon <= 0 specified for fene bond" << endl;				
			}

/*! BondForceCompute provides
	- \c fene_energy
*/
std::vector< std::string > FENEBondForceCompute::getProvidedLogQuantities()
	{
	vector<string> list;
	list.push_back("bond_fene_energy");
	return list;
	}

/*! \param quantity Name of the quantity to get the log value of
	\param timestep Current time step of the simulation
*/	
Scalar FENEBondForceCompute::getLogValue(const std::string& quantity, unsigned int timestep)
	{
	if (quantity == string("bond_fene_energy"))
		{
		compute(timestep);
		return calcEnergySum();
		}
	else
		{
		cerr << endl << "***Error! " << quantity << " is not a valid log quantity for FENEBondForceCompute" << endl << endl;
		throw runtime_error("Error getting log value");
		}
	}	
	
/*! Actually perform the force computation
	\param timestep Current time step
 */
void FENEBondForceCompute::computeForces(unsigned int timestep)
 	{
	if (m_prof) m_prof->push("FENE");

 	assert(m_pdata);
 	// access the particle data arrays
	ParticleDataArraysConst arrays = m_pdata->acquireReadOnly();
	// there are enough other checks on the input data: but it doesn't hurt to be safe
	assert(m_fx);
	assert(m_fy);
	assert(m_fz);
	assert(m_pe);
	assert(arrays.x);
	assert(arrays.y);
	assert(arrays.z);

	// get a local copy of the simulation box too
	const BoxDim& box = m_pdata->getBox();
	// sanity check
	assert(box.xhi > box.xlo && box.yhi > box.ylo && box.zhi > box.zlo);

	// precalculate box lengths
	Scalar Lx = box.xhi - box.xlo;
	Scalar Ly = box.yhi - box.ylo;
	Scalar Lz = box.zhi - box.zlo;
	Scalar Lx2 = Lx / Scalar(2.0);
	Scalar Ly2 = Ly / Scalar(2.0);
	Scalar Lz2 = Lz / Scalar(2.0);
	
	// need to start from a zero force, potential energy and virial
	// (MEM TRANSFER: 5 Scalars)
	memset((void*)m_fx, 0, sizeof(Scalar) * m_pdata->getN());
	memset((void*)m_fy, 0, sizeof(Scalar) * m_pdata->getN());
	memset((void*)m_fz, 0, sizeof(Scalar) * m_pdata->getN());
	memset((void*)m_pe, 0, sizeof(Scalar) * m_pdata->getN());
	memset((void*)m_virial, 0, sizeof(Scalar) * m_pdata->getN());
	
	// for each of the bonds
	const unsigned int size = (unsigned int)m_bond_data->getNumBonds();
	for (unsigned int i = 0; i < size; i++)
		{
		// lookup the tag of each of the particles participating in the bond
		const Bond& bond = m_bond_data->getBond(i);
		assert(bond.a < m_pdata->getN());
		assert(bond.b < m_pdata->getN());
				
		// transform a and b into indicies into the particle data arrays
		// (MEM TRANSFER: 4 integers)
		unsigned int idx_a = arrays.rtag[bond.a];
		unsigned int idx_b = arrays.rtag[bond.b];
		assert(idx_a < m_pdata->getN());
		assert(idx_b < m_pdata->getN());

		// calculate d\vec{r}
		// (MEM TRANSFER: 6 Scalars / FLOPS: 3)
		Scalar dx = arrays.x[idx_b] - arrays.x[idx_a];
		Scalar dy = arrays.y[idx_b] - arrays.y[idx_a];
		Scalar dz = arrays.z[idx_b] - arrays.z[idx_a];

		// if the vector crosses the box, pull it back
		// (FLOPS: 9 (worst case: first branch is missed, the 2nd is taken and the add is done))
		if (dx >= Lx2)
			dx -= Lx;
		else
		if (dx < -Lx2)
			dx += Lx;
		
		if (dy >= Ly2)
			dy -= Ly;
		else
		if (dy < -Ly2)
			dy += Ly;
		
		if (dz >= Lz2)
			dz -= Lz;
		else
		if (dz < -Lz2)
			dz += Lz;

		// sanity check
		assert(dx >= box.xlo && dx < box.xhi);
		assert(dy >= box.ylo && dx < box.yhi);
		assert(dz >= box.zlo && dx < box.zhi);

		//ALL FLOPS NEED TO BE FIXED
		// on paper, the formula turns out to be: F = -K/(1-(r/r_0)^2) * \vec{r} + (12*lj1/r^12 - 6*lj2/r^6) *\vec{r}
		// FLOPS: 5
		Scalar rsq = dx*dx+dy*dy+dz*dz;
		
		// compute the force magnitude/r in forcemag_divr (FLOPS: 9)
		Scalar r2inv = Scalar(1.0)/rsq;
		Scalar r6inv = r2inv * r2inv * r2inv;

		Scalar WCAforcemag_divr;
		Scalar pair_eng;
		if (rsq < 1.01944064370214) {   //wcalimit squared (2^(1/6))^2
			WCAforcemag_divr = r2inv * r6inv * (Scalar(12.0)*m_lj1[bond.type]*r6inv - Scalar(6.0)*m_lj2[bond.type]);
			pair_eng = Scalar(0.5) * (r6inv * (m_lj1[bond.type]*r6inv - m_lj2[bond.type]) + m_epsilon[bond.type]);
			}
		else {
		    WCAforcemag_divr = 0;
			pair_eng = 0;
			}
			
		// Additional check for FENE spring
		assert(rsq < m_r_0[bond.type]*m_r_0[bond.type]);
		
		// calculate force and energy
		// MEM TRANSFER 2 Scalars: FLOPS: 13
		Scalar forcemag_divr = -m_K[bond.type] / (Scalar(1.0) - rsq /(m_r_0[bond.type]*m_r_0[bond.type])) + WCAforcemag_divr;  //FLOPS 4
		Scalar bond_eng = -Scalar(0.5) * Scalar(0.5) * m_K[bond.type] * (m_r_0[bond.type] * m_r_0[bond.type]) * log(Scalar(1.0) - rsq/(m_r_0[bond.type] * m_r_0[bond.type]));		
		
		// calculate virial (FLOPS: 2)
		Scalar bond_virial = Scalar(1.0/6.0) * rsq * forcemag_divr;
		
		// add the force to the particles
		// (MEM TRANSFER: 20 Scalars / FLOPS 16)
		m_fx[idx_b] += forcemag_divr * dx;
		m_fy[idx_b] += forcemag_divr * dy;
		m_fz[idx_b] += forcemag_divr * dz;
		m_pe[idx_b] += bond_eng + pair_eng;
		m_virial[idx_b] += bond_virial;
		
		m_fx[idx_a] -= forcemag_divr * dx;
		m_fy[idx_a] -= forcemag_divr * dy;
		m_fz[idx_a] -= forcemag_divr * dz;
		m_pe[idx_a] += bond_eng + pair_eng;
		m_virial[idx_a] += bond_virial;
		}

	m_pdata->release();

	#ifdef ENABLE_CUDA
	// the data is now only up to date on the CPU
	m_data_location = cpu;
	#endif

	if (m_prof) m_prof->pop(m_bond_data->getNumBonds() * (3+9+5+13+2+16), m_pdata->getN() * 5 * sizeof(Scalar) + m_bond_data->getNumBonds() * ( (4) * sizeof(unsigned int) + (6+2+20) ) );
	}
	
void export_FENEBondForceCompute()
	{
	class_<FENEBondForceCompute, boost::shared_ptr<FENEBondForceCompute>, bases<ForceCompute>, boost::noncopyable >
		("FENEBondForceCompute", init< boost::shared_ptr<ParticleData> >())
		.def("setParams", &FENEBondForceCompute::setParams)
		;
	}

#ifdef WIN32
#pragma warning( pop )
#endif
