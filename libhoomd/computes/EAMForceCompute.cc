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

// $Id: EAMForceCompute.cc 1839 2009-05-04 20:39:55Z joaander $
// $URL: https://svn2.assembla.com/svn/hoomd/trunk/src/computes/EAMForceCompute.cc $
// Maintainer: joaander

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "EAMForceCompute.h"
#include <stdexcept>
#define min(a, b)  (((a) < (b)) ? (a) : (b)) 
#define max(a, b)  (((a) > (b)) ? (a) : (b)) 
/*! \file EAMForceCompute.cc
	\brief Defines the EAMForceCompute class
*/

using namespace std;

/*! \param pdata System to compute forces on
 	\param nlist Neighborlist to use for computing the forces
	\param r_cut Cuttoff radius beyond which the force is 0
	\post memory is allocated and all parameters lj1 and lj2 are set to 0.0
*/
EAMForceCompute::EAMForceCompute(boost::shared_ptr<ParticleData> pdata, boost::shared_ptr<NeighborList> nlist, Scalar r_cut, char *filename) 
	: ForceCompute(pdata), m_nlist(nlist), m_r_cut(r_cut)
	{
	assert(m_pdata);
	assert(m_nlist);
	
	if (r_cut < 0.0)
		{
		cerr << endl << "***Error! Negative r_cut in EAMForceCompute makes no sense" << endl << endl;
		throw runtime_error("Error initializing EAMForceCompute");
		}
	loadFile(filename);
	// initialize the number of types value
	m_ntypes = m_pdata->getNTypes();
	assert(m_ntypes > 0);
	

	}
	

EAMForceCompute::~EAMForceCompute()
	{

	}
void EAMForceCompute::loadFile(char *filename)
	{
	unsigned int tmp_int, type, i;
	double  tmp_mass, tmp;
	char tmp_str[5];
  	// open potential file
  	FILE *fp;
    fp = fopen(filename,"r");
    if (fp == NULL) 
		{
		cerr << endl << "***Error! Can not load EAM file" << endl << endl;
		throw runtime_error("Error loading file");
    	}
	for(i = 0; i < 3; i++) while(fgetc(fp) != '\n');
	fscanf(fp, "%d", &m_ntypes);
	//Считываем имена
	for(i = 0; i < m_ntypes; i++)
		{
		fscanf(fp, "%2s", tmp_str);
		names.push_back(tmp_str);
		}
	//Считываем параметры
	fscanf(fp,"%d", &nrho);
	fscanf(fp,"%lg", &tmp);
	drho = tmp;
	rdrho = (Scalar)(1.0 / drho);
	fscanf(fp,"%d", &nr);
	fscanf(fp,"%lg", &tmp);
	dr = tmp;
	rdr = (Scalar)(1.0 / dr);
	fscanf(fp,"%lg", &tmp);
	m_r_cut = tmp;
	//Считываем Функции погружения и электронные плотности.
	for(type = 0 ; type < m_ntypes; type++)
		{
		fscanf(fp, "%d %lg %lg %3s ", &tmp_int, &tmp_mass, &tmp, &tmp_str);
		mass.push_back(tmp_mass);

		for(i = 0 ; i < nrho; i++)
			{
			fscanf(fp, "%lg", &tmp);
			embeddingFunction.push_back((Scalar)tmp);
			}
		for(i = 0 ; i < nr; i++)
			{
			fscanf(fp, "%lg", &tmp);
			electronDensity.push_back((Scalar)tmp);
			}
	
		}
	//Считываем потенциалы взаимодействия.
	for(i = 0; i < (ceil(m_ntypes * m_ntypes / 2) + 1) * nr; i++)
		{
		fscanf(fp, "%lg", &tmp);
		pairPotential.push_back((Scalar)tmp);
		}

	fclose(fp);
	derivativeEmbeddingFunction.resize(m_ntypes * nrho);
	derivativeElectronDensity.resize(m_ntypes * nr);
	derivativePairPotential.resize((ceil(m_ntypes * m_ntypes / 2) + 1) * nr);
	//Вычисляем производные Функции погружения и электронной плотности.
	for(type = 0 ; type < m_ntypes; type++)
		{
		for(i = 0 ; i < nrho - 1; i++)
			{
			derivativeEmbeddingFunction[i + type * nrho] = 
				(embeddingFunction[i + 1 + type * nrho] - embeddingFunction[i + type * nrho]) / drho;
			}
		for(i = 0 ; i < nr - 1; i++)
			{	
			derivativeElectronDensity[i + type * nr] = 
				(electronDensity[i + 1 + type * nr] - electronDensity[i + type * nr]) / dr;
			}
	
		}
	//Вычисляем производные потенциалов взаимодействия.
	for(i = 0; i < (ceil(m_ntypes * m_ntypes / 2) + 1) * nr; i++)
		{
		if((i + 1)%nr == 0) continue;
		derivativePairPotential[i] = (pairPotential[i + 1] - pairPotential[i]) / dr;
		}
	
	}
std::vector< std::string > EAMForceCompute::getProvidedLogQuantities()
	{
	vector<string> list;
	list.push_back("pair_lj_energy");
	return list;
	}
	
Scalar EAMForceCompute::getLogValue(const std::string& quantity, unsigned int timestep)
	{
	if (quantity == string("pair_lj_energy"))
		{
		compute(timestep);
		return calcEnergySum();
		}
	else
		{
		cerr << endl << "***Error! " << quantity << " is not a valid log quantity for EAMForceCompute" << endl << endl;
		throw runtime_error("Error getting log value");
		}
	}

/*! \post The lennard jones forces are computed for the given timestep. The neighborlist's
 	compute method is called to ensure that it is up to date.
	
	\param timestep specifies the current time step of the simulation
*/
void EAMForceCompute::computeForces(unsigned int timestep)
	{
	// start by updating the neighborlist
	m_nlist->compute(timestep);
	
	// start the profile for this compute
	if (m_prof) m_prof->push("EAM pair");
	
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
	Scalar rho;
	vector<Scalar> atomElectronDensity;
	atomElectronDensity.resize(arrays.nparticles);
	vector<Scalar> atomDerivativeEmbeddingFunction;
	atomDerivativeEmbeddingFunction.resize(arrays.nparticles);
	vector<Scalar> atomEmbeddingFunction;
	atomEmbeddingFunction.resize(arrays.nparticles);
	for (unsigned int i = 0; i < arrays.nparticles; i++)
		{
		// access the particle's position and type (MEM TRANSFER: 4 scalars)
		Scalar xi = arrays.x[i];
		Scalar yi = arrays.y[i];
		Scalar zi = arrays.z[i];
		unsigned int typei = arrays.type[i];
		// sanity check
		assert(typei < m_pdata->getNTypes());
		


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
				 Scalar position_float = sqrt(rsq) * rdr;
				 Scalar position = position_float;
				 unsigned int r_index = (unsigned int)position;
				 position -= r_index;				 
				 atomElectronDensity[i] += electronDensity[r_index + nr * typej] + derivativeElectronDensity[r_index + nr * typej] * position * dr;	
				 if(third_law)
					{
					atomElectronDensity[k] += electronDensity[r_index + nr * typei] 
						+ derivativeElectronDensity[r_index + nr * typei] * position * dr;	
					}
				}
			}
		}
		
	for (unsigned int i = 0; i < arrays.nparticles; i++)
		{
		unsigned int typei = arrays.type[i];

		Scalar position = atomElectronDensity[i] * rdrho;
		unsigned int r_index = (unsigned int)position;
		position -= (Scalar)r_index;
		atomDerivativeEmbeddingFunction[i] = derivativeEmbeddingFunction[r_index + typei * nrho];
		
		m_pe[i] += embeddingFunction[r_index + typei * nrho] + derivativeEmbeddingFunction[r_index + typei * nrho] * position * drho;
		}
	for (unsigned int i = 0; i < arrays.nparticles; i++)
		{
		// access the particle's position and type (MEM TRANSFER: 4 scalars)
		Scalar xi = arrays.x[i];
		Scalar yi = arrays.y[i];
		Scalar zi = arrays.z[i];
		unsigned int typei = arrays.type[i];
		// sanity check
		assert(typei < m_pdata->getNTypes());
		


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
		
		
			if (rsq > r_cut_sq) continue;
			Scalar r = sqrt(rsq);
			Scalar inverseR = 1.0 / r;
			Scalar position = r * rdr;
			unsigned int r_index = (unsigned int)position;
			position = position - (Scalar)r_index;
			Scalar pair_eng = (pairPotential[r_index + (typei + typej) * nr] + 
				derivativePairPotential[r_index + (typei + typej) * nr] * position * dr) * inverseR;			
			Scalar derivativePhi = (derivativePairPotential[r_index + (typei + typej) * nr] - pair_eng) * inverseR;
			Scalar derivativeRhoI = derivativeElectronDensity[r_index + typei * nr];			
			Scalar derivativeRhoJ = derivativeElectronDensity[r_index + typej * nr];			
			Scalar fullDerivativePhi = atomDerivativeEmbeddingFunction[i] * derivativeRhoJ +
				atomDerivativeEmbeddingFunction[k] * derivativeRhoI + derivativePhi;
			Scalar pairForce = - fullDerivativePhi * inverseR;
			viriali += float(1.0/6.0) * rsq * pairForce;
			fxi += dx * pairForce;
			fyi += dy * pairForce;
			fzi += dz * pairForce;
			pei += pair_eng;
			if (third_law)
				{
				m_fx[k] -= dx * pairForce;
				m_fy[k] -= dy * pairForce;
				m_fz[k] -= dz * pairForce;
				}
			}
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

void export_EAMForceCompute()
	{
	scope in_eam = class_<EAMForceCompute, boost::shared_ptr<EAMForceCompute>, bases<ForceCompute>, boost::noncopyable >
		("EAMForceCompute", init< boost::shared_ptr<ParticleData>, boost::shared_ptr<NeighborList>, Scalar, char*>());
		
	}

#ifdef WIN32
#pragma warning( pop )
#endif
