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

#include "BondForceCompute.h"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <math.h>

using namespace std;

/*! \file BondForceCompute.cc
	\brief Contains code for the BondForceCompute class
*/

/*! \param pdata Particle data to compute forces on
	\param K Stiffness parameter for the force computation
	\param r_0 Equilibrium length for the force computation
	\post Memory is allocated, forces are zeroed, and no bonds are defined
*/
BondForceCompute::BondForceCompute(boost::shared_ptr<ParticleData> pdata, Scalar K, Scalar r_0) :	ForceCompute(pdata),
	m_K(K), m_r_0(r_0)
	{
	// check for some silly errors a user could make 
	if (m_K <= 0)
		cout << "Warning: K <= 0 specified for harmonic bond" << endl;
	if (m_r_0 <= 0)
		cout << "Warning: r_0 <= 0 specified for harmonic bond" << endl;
	}
	
/*! Sets new parameters for the potential
	\param K Stiffness parameter for the force computation
	\param r_0 Equilibrium length for the force computation
*/
void BondForceCompute::setParams(Scalar K, Scalar r_0)
	{
	m_K = K;
	m_r_0 = r_0;

	// check for some silly errors a user could make 
	if (m_K <= 0)
		cout << "Warning: K <= 0 specified for harmonic bond" << endl;
	if (m_r_0 <= 0)
		cout << "Warning: r_0 <= 0 specified for harmonic bond" << endl;
	}

/*! \post A bond between particle with tag \a tag1 and with tag \a tag2 are bonded. When
	compute() is next called, the harmonic force will be calculated between these two particles.
	\note Each bond should only be specified once! There are no checks to prevent one from being 
	specified more than once, and doing so would result in twice the force and twice the energy.
	For a bond between \c i and \c j, only call \c addBond(i,j). Do NOT additionally call 
	\c addBond(j,i). The first call is sufficient to include the forces on both particle i and j.
	
	\param tag1 Tag of the first particle in the bond
	\param tag2 Tag of the second particle in the bond
 */	
void BondForceCompute::addBond(unsigned int tag1, unsigned int tag2)
	{
	// check for some silly errors a user could make 	
	if (tag1 >= m_pdata->getN() || tag2 >= m_pdata->getN())
		{
		ostringstream err;
		err << "Particle tag out of bounds when attempting to add bond: " << tag1 << "," << tag2 << endl;
		throw runtime_error(err.str());
		} 
	if (tag1 == tag2)
		{
		ostringstream err;
		err << "Particle cannot be bonded to itself! " << tag1 << "," << tag2 << endl;
		throw runtime_error(err.str());
		}

	m_bonds.push_back(BondPair(tag1, tag2));
	}
	
/*! Actually perform the force computation
	\param timestep Current time step
 */
void BondForceCompute::computeForces(unsigned int timestep)
 	{
	if (m_prof)
		m_prof->push("Bond");

 	assert(m_pdata);
 	// access the particle data arrays
	ParticleDataArraysConst arrays = m_pdata->acquireReadOnly();
	// there are enough other checks on the input data: but it doesn't hurt to be safe
	assert(m_fx);
	assert(m_fy);
	assert(m_fz);
	assert(arrays.x);
	assert(arrays.y);
	assert(arrays.z);

	if (m_prof)
		m_prof->push("Compute");

	// get a local copy of the simulation box too
	const BoxDim& box = m_pdata->getBox();
	// sanity check
	assert(box.xhi > box.xlo && box.yhi > box.ylo && box.zhi > box.zlo);

	// precalculate box lenghts
	Scalar Lx = box.xhi - box.xlo;
	Scalar Ly = box.yhi - box.ylo;
	Scalar Lz = box.zhi - box.zlo;
	Scalar Lx2 = Lx / Scalar(2.0);
	Scalar Ly2 = Ly / Scalar(2.0);
	Scalar Lz2 = Lz / Scalar(2.0);
	
	// need to start from a zero force
	memset((void*)m_fx, 0, sizeof(Scalar) * m_pdata->getN());
	memset((void*)m_fy, 0, sizeof(Scalar) * m_pdata->getN());
	memset((void*)m_fz, 0, sizeof(Scalar) * m_pdata->getN());	
	
	// for each of the bonds
	const unsigned int size = m_bonds.size(); 
	for (unsigned int i = 0; i < size; i++)
		{
		// lookup the tag of each of the particles participating in the bond
		unsigned int a = m_bonds[i].m_tag1;
		unsigned int b = m_bonds[i].m_tag2;
		
		assert(a < m_pdata->getN());
		assert(b < m_pdata->getN());
				
		// transform a and b into indicies into the particle data arrays
		a = arrays.rtag[a];
		b = arrays.rtag[b];

		assert(a < m_pdata->getN());
		assert(b < m_pdata->getN());

		// calculate d\vec{r}		
		Scalar dx = arrays.x[b] - arrays.x[a];
		Scalar dy = arrays.y[b] - arrays.y[a];
		Scalar dz = arrays.z[b] - arrays.z[a];

		// if the vector crosses the box, pull it back
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

		// on paper, the formula turns out to be: F = 2*K*\vec{r} * (r_0/r - 1)
		// now calculate r
		Scalar r = sqrt(dx*dx+dy*dy+dz*dz);
		Scalar tmp = Scalar(2.0) * m_K * (m_r_0 / r - Scalar(1.0));
		
		// add the force to the particles
		m_fx[b] += tmp * dx;
		m_fy[b] += tmp * dy;
		m_fz[b] += tmp * dz;
		m_fx[a] -= tmp * dx;
		m_fy[a] -= tmp * dy;
		m_fz[a] -= tmp * dz;
		} 

	m_pdata->release();

	#ifdef USE_CUDA
	// the data is now only up to date on the CPU
	m_data_location = cpu;
	#endif

	// each bond computes ~23 FLOPS
	// and 4 index reads, then 6 position reads and 6 force writes
	if (m_prof)
		{
		m_prof->pop(23*m_bonds.size(), (4*sizeof(int)+12*sizeof(Scalar))*m_bonds.size());
		m_prof->pop();
		}
	}
	
#ifdef USE_PYTHON
void export_BondForceCompute()
	{
	class_<BondForceCompute, boost::shared_ptr<BondForceCompute>, bases<ForceCompute>, boost::noncopyable >
		("BondForceCompute", init< boost::shared_ptr<ParticleData>, Scalar, Scalar >())
		.def("setParams", &BondForceCompute::setParams)
		.def("addBond", &BondForceCompute::addBond)
		;
	}
#endif
