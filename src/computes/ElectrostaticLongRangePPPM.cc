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

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif


// conditionally compile only if a fast fourier transform is defined
#ifdef USE_FFT

#include <iostream>
using namespace std;

#include "ElectrostaticLongRangePPPM.h"
#include <stdexcept>


/*! \file ElectrostaticLongRangePPPM.cc
	\brief Contains the code for the ElectrostaticLongRangePPPM class
*/

using namespace std;

/*!     \param pdata Particle Data to compute forces on
	\param alpha Split parameter of short vs long range electrostatics (see header file for more info) 
*/
ElectrostaticLongRangePPPM::ElectrostaticLongRangePPPM(boost::shared_ptr<ParticleData> pdata, unsigned int Mmesh_x,unsigned int Mmesh_y,unsigned int Mmesh_z,unsigned int P_order_a, Scalar alpha,bool third_law_m)
	:ForceCompute(pdata),N_mesh_x(Mmesh_x),N_mesh_y(Mmesh_y),N_mesh_z(Mmesh_z),P_order(P_order_a),m_alpha(alpha),third_law(third_law_m)
	{
	assert(m_pdata);
	
	if (alpha < 0.0)
		{
		cerr << endl << "***Error! Negative alpha in ElectrostaticShortRange makes no sense" << endl << endl;
		throw runtime_error("Error initializing ElectrostaticShortRange");
		}

	//Cast mesh sizes as a scalar
	S_mesh_x=static_cast<Scalar>(N_mesh_x);
	S_mesh_y=static_cast<Scalar>(N_mesh_y);
	S_mesh_z=static_cast<Scalar>(N_mesh_z);

        // get a copy of the simulation box too
	box = m_pdata->getBox();
	// sanity check
	assert(box.xhi > box.xlo && box.yhi > box.ylo && box.zhi > box.zlo);	
	
	// precalculate box lenghts for use in the periodic imaging
	Lx = box.xhi - box.xlo;
	Ly = box.yhi - box.ylo;
	Lz = box.zhi - box.zlo;
	
	//Compute lattice spacings along the different directions
	h_x=S_mesh_x/Lx;
	h_y=S_mesh_y/Ly;
	h_z=S_mesh_z/Lz;

	//allocate space for the density and the influence function
	rho_real=new Scalar**[N_mesh_z];
	rho_kspace=new Scalar**[N_mesh_z];
	G_Inf=new Scalar**[N_mesh_z];

	for(unsigned int i=0;i<N_mesh_z;i++){
		rho_real[i]=new Scalar*[N_mesh_y];
		rho_kspace[i]=new Scalar*[N_mesh_y];
		G_Inf[i]=new Scalar*[N_mesh_y];
	}

	for(unsigned int j=0;j<N_mesh_z;j++){
	for(unsigned int i=0;i<N_mesh_y;i++){
		rho_real[i][j]=new Scalar[N_mesh_x];
		rho_kspace[i][j]=new Scalar[N_mesh_x];
		G_Inf[i][j]=new Scalar[N_mesh_x];
	}
	}

	if(P_order%2) make_rho_helper=&ElectrostaticLongRangePPPM::make_rho_odd;
	else make_rho_helper=&ElectrostaticLongRangePPPM::make_rho_even;

}


ElectrostaticLongRangePPPM::~ElectrostaticLongRangePPPM()
	{
	
	for(unsigned int j=0;j<N_mesh_z;j++){
	for(unsigned int i=0;i<N_mesh_y;i++){
		delete[] rho_real[i][j];
		delete[] rho_kspace[i][j];
		delete[] G_Inf[i][j];
	}
	}

	for(unsigned int i=0;i<N_mesh_z;i++){
		delete[] rho_real[i];
		delete[] rho_kspace[i];
		delete[] G_Inf[i];
	}
		
	delete[] rho_real;
	delete[] rho_kspace;
	delete[] G_Inf;
}

void ElectrostaticLongRangePPPM::make_rho(void)
{
	(this->*make_rho_helper)();
}

void ElectrostaticLongRangePPPM::make_rho_even(void)
{
	
	// access the particle data
	const ParticleDataArraysConst& arrays = m_pdata->acquireReadOnly(); 
	// sanity check
	assert(arrays.x != NULL && arrays.y != NULL && arrays.z != NULL);
	for (unsigned int i = 0; i < arrays.nparticles; i++)
		{
		// access the particle's position and charge (MEM TRANSFER: 4 Scalars)
		Scalar xi = arrays.x[i];
		Scalar yi = arrays.y[i];
		Scalar zi = arrays.z[i];

		Scalar q_i=arrays.charge[i];

		//compute the two nearest points on the grid

		Scalar x_floor=floor((xi-box.xhi)/h_x);//MAKE SURE THIS NUMBER IS ALWAYS POSITIVE
		unsigned int ix_floor=static_cast<unsigned int>(x_floor);
		
		Scalar y_floor=floor((yi-box.yhi)/h_y);//MAKE SURE THIS NUMBER IS ALWAYS POSITIVE
		unsigned int iy_floor=static_cast<unsigned int>(y_floor);
		
		Scalar z_floor=floor((zi-box.zhi)/h_z);//MAKE SURE THIS NUMBER IS ALWAYS POSITIVE
		unsigned int iz_floor=static_cast<unsigned int>(z_floor);

	    }
}
void ElectrostaticLongRangePPPM::make_rho_odd(void)
{
}

void ElectrostaticLongRangePPPM::computeForces(unsigned int timestep)
	{
	// start the profile
	if (m_prof) m_prof->push("ElecLongRange");
	
	// access the particle data
	const ParticleDataArraysConst& arrays = m_pdata->acquireReadOnly(); 
	// sanity check
	assert(arrays.x != NULL && arrays.y != NULL && arrays.z != NULL);
	
	// tally up the number of forces calculated
	int64_t n_calc = 0;
	
	// need to start from a zero force, potential energy and virial
	// (MEM TRANSFER 5*N Scalars)
	memset(m_fx, 0, sizeof(Scalar)*arrays.nparticles);
	memset(m_fy, 0, sizeof(Scalar)*arrays.nparticles);
	memset(m_fz, 0, sizeof(Scalar)*arrays.nparticles);
	memset(m_pe, 0, sizeof(Scalar)*arrays.nparticles);
	memset(m_virial, 0, sizeof(Scalar)*arrays.nparticles);

	// for each particle
	for (unsigned int i = 0; i < arrays.nparticles; i++)
		{
		// access the particle's position and charge (MEM TRANSFER: 4 Scalars)
		Scalar xi = arrays.x[i];
		Scalar yi = arrays.y[i];
		Scalar zi = arrays.z[i];

		Scalar q_i=arrays.charge[i];
		
		// zero force, potential energy, and virial for the current particle
		Scalar fxi = 0.0;
		Scalar fyi = 0.0;
		Scalar fzi = 0.0;
		Scalar pei = 0.0;
		Scalar viriali = 0.0;
		
		
		// (FLOPS: 5 / MEM TRANSFER: 10 Scalars)
		m_fx[i] += fxi;
		m_fy[i] += fyi;
		m_fz[i] += fzi;
		m_pe[i] += pei;
		m_virial[i] += viriali;
		}
		
	m_pdata->release();
	
	#ifdef USE_CUDA
	// the force data is now only up to date on the cpu
	m_data_location = cpu;
	#endif
	
	int64_t flops = n_calc * (3+9+5+18+6+8+8);
	if (third_law)	flops += n_calc*8;
	int64_t mem_transfer = m_pdata->getN() * (5 + 4 + 10)*sizeof(Scalar) + n_calc * ( (3+6)*sizeof(Scalar) + (1)*sizeof(unsigned int));
	if (third_law) mem_transfer += n_calc*10*sizeof(Scalar);
	if (m_prof) m_prof->pop(flops, mem_transfer);
	}

#endif

#ifdef WIN32
#pragma warning( pop )
#endif
