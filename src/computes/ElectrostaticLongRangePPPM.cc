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

#include <math.h>

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
	assert(box.xhi > box.xlo &&  box.yhi > box.ylo && box.zhi > box.zlo);	
	
	// precalculate box lenghts for use in the periodic imaging
	Lx = box.xhi - box.xlo;
	Ly = box.yhi - box.ylo;
	Lz = box.zhi - box.zlo;
	
	//Compute lattice spacings along the different directions
	h_x= S_mesh_x/Lx;
	h_y= S_mesh_y/Ly;
	h_z= S_mesh_z/Lz;

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
	int P_half=static_cast<int>(P_order/2);

	//initialize rho to zero
    for(unsigned int i=0;i<N_mesh_x;i++){
	for(unsigned int j=0;j<N_mesh_y;j++){
	for(unsigned int k=0;k<N_mesh_z;k++){
		rho_real[i][j][k]=0.0;
	}
	}
	}

	// access the particle data
	const ParticleDataArraysConst& arrays = m_pdata->acquireReadOnly(); 
	// sanity check
	assert(arrays.x != NULL && arrays.y != NULL && arrays.z != NULL);

	//place the continuum charge on the grid
	for (unsigned int i = 0; i < arrays.nparticles; i++)
		{
		// access the particle's position and charge (MEM TRANSFER: 4 Scalars)
		Scalar xi = arrays.x[i];
		Scalar yi = arrays.y[i];
		Scalar zi = arrays.z[i];

		Scalar q_i= arrays.charge[i];

		//compute the two nearest points on the grid
	    
		Scalar x_floor= floor((xi-box.xhi)/h_x);//MAKE SURE THIS NUMBER IS ALWAYS POSITIVE   
		unsigned int ix_floor=static_cast<unsigned int>(x_floor);
		
		Scalar y_floor= floor((yi-box.yhi)/h_y);//MAKE SURE THIS NUMBER IS ALWAYS POSITIVE
		unsigned int iy_floor=static_cast<unsigned int>(y_floor);
		
		Scalar z_floor= floor((zi-box.zhi)/h_z);//MAKE SURE THIS NUMBER IS ALWAYS POSITIVE
		unsigned int iz_floor=static_cast<unsigned int>(z_floor);

		unsigned int ind_x=0;
		unsigned int ind_y=0;
		unsigned int ind_z=0;

		Scalar dx=xi-x_floor-0.5;
		Scalar dy=yi-y_floor-0.5;
		Scalar dz=zi-z_floor-0.5;
        
		//take into account boundary conditions
		for(int lx=-P_half+1;lx<=P_half;lx++){
				ind_x=ix_floor+lx+((N_mesh_x-ix_floor-lx)/N_mesh_x)*N_mesh_x-((ix_floor+lx)/N_mesh_x)*N_mesh_x;
        for(int ly=-P_half+1;ly<=P_half;ly++){
				ind_y=iy_floor+lx+((N_mesh_y-iy_floor-ly)/N_mesh_y)*N_mesh_y-((iy_floor+ly)/N_mesh_y)*N_mesh_y;
		for(int lz=-P_half+1;lz<=P_half;lz++){
			    ind_z=iz_floor+lz+((N_mesh_z-iz_floor-lx)/N_mesh_z)*N_mesh_z-((ix_floor+lx)/N_mesh_x)*N_mesh_x;
				rho_real[ind_x][ind_y][ind_z]+=q_i*Poly(lx,dx)*Poly(ly,dy)*Poly(lz,dz);
	    }
		}
		}
	}
		//The charge is now defined on the grid for P even
}
void ElectrostaticLongRangePPPM::make_rho_odd(void)
{
	
    int P_half=static_cast<int>(P_order/2);

	//initialize rho to zero
    for(unsigned int i=0;i<N_mesh_x;i++){
	for(unsigned int j=0;j<N_mesh_y;j++){
	for(unsigned int k=0;k<N_mesh_z;k++){
		rho_real[i][j][k]=0.0;
	}
	}
	}

	// access the particle data
	const ParticleDataArraysConst& arrays = m_pdata->acquireReadOnly(); 
	// sanity check
	assert(arrays.x != NULL && arrays.y != NULL && arrays.z != NULL);

	//place the continuum charge on the grid
	for (unsigned int i = 0; i < arrays.nparticles; i++)
		{
		// access the particle's position and charge (MEM TRANSFER: 4 Scalars)
		Scalar xi = arrays.x[i];
		Scalar yi = arrays.y[i];
		Scalar zi = arrays.z[i];

		Scalar q_i= arrays.charge[i];

		//compute the nearest point on the grid
		Scalar x_lat = (xi-box.xhi)/h_x; //MAKE SURE THIS NUMBER IS ALWAYS POSITIVE
		Scalar x_floor = floor(x_lat);
		unsigned int ix_lat=static_cast<unsigned int>(x_floor);
		if((x_lat-x_floor)>0.5) {      //boundary conditions are tricky here
			ix_lat+=1-((ix_lat+1)/N_mesh_x)*N_mesh_x;
			x_floor+=1.0;  //note that x_floor has now become x_ceil
		}
			
		Scalar y_lat=(yi-box.yhi)/h_y;  //MAKE SURE THIS NUMBER IS ALWAYS POSITIVE
		Scalar y_floor= floor(y_lat);
		unsigned int iy_lat=static_cast<unsigned int>(y_floor);
		if((y_lat-y_floor)>0.5){          //boundary conditions are tricky here
			iy_lat+=1-((iy_lat+1)/N_mesh_y)*N_mesh_y;
			y_floor+=1.0; //note that y_floor has now become y_ceil
		}
		
        Scalar z_lat=(zi-box.zhi)/h_z;  //MAKE SURE THIS NUMBER IS ALWAYS POSITIVE
		Scalar z_floor= floor(z_lat);
		unsigned int iz_lat=static_cast<unsigned int>(z_floor);
		if((z_lat-z_floor)>0.5){            //boundary conditions are tricky here
			iz_lat+=1-((iz_lat+1)/N_mesh_z)*N_mesh_z;
			z_floor+=1.0; //note that z_floor has now become z_ceil
		}

		unsigned int ind_x=0;
		unsigned int ind_y=0;
		unsigned int ind_z=0;

		Scalar dx=xi-x_floor;
		Scalar dy=yi-y_floor;
		Scalar dz=zi-z_floor;
        
		//take into account boundary conditions
		for(int lx=-P_half;lx<=P_half;lx++){
				ind_x=ix_lat+lx+((N_mesh_x-ix_lat-lx)/N_mesh_x)*N_mesh_x-((ix_lat+lx)/N_mesh_x)*N_mesh_x;
        for(int ly=-P_half;ly<=P_half;ly++){
				ind_y=iy_lat+lx+((N_mesh_y-iy_lat-ly)/N_mesh_y)*N_mesh_y-((iy_lat+ly)/N_mesh_y)*N_mesh_y;
		for(int lz=-P_half;lz<=P_half;lz++){
			    ind_z=iz_lat+lz+((N_mesh_z-iz_lat-lx)/N_mesh_z)*N_mesh_z-((ix_lat+lx)/N_mesh_x)*N_mesh_x;
				rho_real[ind_x][ind_y][ind_z]+=q_i*Poly(lx,dx)*Poly(ly,dy)*Poly(lz,dz);
	    }
		}
		}
	}

}

Scalar ElectrostaticLongRangePPPM::Poly(int l,Scalar x)
{
	return 1;
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
