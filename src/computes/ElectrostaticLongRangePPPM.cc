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

/*! \file ElectrostaticLongRangePPPM.cc
	\brief Contains the code for the ElectrostaticLongRangePPPM class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

// conditionally compile only if a fast fourier transform is defined
#ifdef USE_FFT

#ifdef WIN32
#define _USE_MATH_DEFINES
#endif

#include <math.h>

#include <boost/math/special_functions/sinc.hpp>

#include <iostream>
using namespace std;

#include "ElectrostaticLongRangePPPM.h"
#include <stdexcept>

using namespace std;

/*! \param pdata Particle Data to compute forces on
	\param alpha Split parameter of short vs long range electrostatics (see header file for more info) 
*/
ElectrostaticLongRangePPPM::ElectrostaticLongRangePPPM(boost::shared_ptr<ParticleData> pdata, unsigned int Mmesh_x,unsigned int Mmesh_y,unsigned int Mmesh_z,unsigned int P_order_a, Scalar alpha,boost::shared_ptr<FFTClass> FFTP,bool third_law_m)
	:ForceCompute(pdata),FFT(FFTP),N_mesh_x(Mmesh_x),N_mesh_y(Mmesh_y),N_mesh_z(Mmesh_z),P_order(P_order_a),m_alpha(alpha),third_law(third_law_m)
	{
	assert(m_pdata);
	
	if (alpha < 0.0)
		{
		cerr << endl << "***Error! Negative alpha in ElectrostaticLongRangePPPM makes no sense" << endl << endl;
		throw runtime_error("Error initializing ElectrostaticLongRange");
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
	rho_real=new CScalar**[N_mesh_z];
	rho_kspace=new CScalar**[N_mesh_z];
	G_Inf=new Scalar**[N_mesh_z];
	fx_kspace=new CScalar**[N_mesh_z];
    fy_kspace=new CScalar**[N_mesh_z];
    fz_kspace=new CScalar**[N_mesh_z];
    e_kspace=new CScalar**[N_mesh_z];
    v_kspace=new CScalar**[N_mesh_z];
    fx_real=new CScalar**[N_mesh_z];           
    fy_real=new CScalar**[N_mesh_z];          
    fz_real=new CScalar**[N_mesh_z];           
    e_real=new CScalar**[N_mesh_z];            
    v_real=new CScalar**[N_mesh_z];            


	for(unsigned int i=0;i<N_mesh_z;i++){
		rho_real[i]=new CScalar*[N_mesh_y];
		rho_kspace[i]=new CScalar*[N_mesh_y];
		G_Inf[i]=new Scalar*[N_mesh_y];
		fx_kspace[i]=new CScalar*[N_mesh_y];
		fy_kspace[i]=new CScalar*[N_mesh_y];
		fz_kspace[i]=new CScalar*[N_mesh_y];
		e_kspace[i]=new CScalar*[N_mesh_y];
		v_kspace[i]=new CScalar*[N_mesh_y];
		fx_real[i]=new CScalar*[N_mesh_y];
		fy_real[i]=new CScalar*[N_mesh_y];
		fz_real[i]=new CScalar*[N_mesh_y];
		e_real[i]=new CScalar*[N_mesh_y];
		v_real[i]=new CScalar*[N_mesh_y];
	}

	for(unsigned int j=0;j<N_mesh_z;j++){
	for(unsigned int i=0;i<N_mesh_y;i++){
		rho_real[i][j]=new CScalar[N_mesh_x];
		rho_kspace[i][j]=new CScalar[N_mesh_x];
		G_Inf[i][j]=new Scalar[N_mesh_x];
		fx_kspace[i][j]=new CScalar[N_mesh_x];
		fy_kspace[i][j]=new CScalar[N_mesh_x];
		fz_kspace[i][j]=new CScalar[N_mesh_x];
		e_kspace[i][j]=new CScalar[N_mesh_x];
		v_kspace[i][j]=new CScalar[N_mesh_x];
		fx_real[i][j]=new CScalar[N_mesh_x];
		fy_real[i][j]=new CScalar[N_mesh_x];
		fz_real[i][j]=new CScalar[N_mesh_x];
		e_real[i][j]=new CScalar[N_mesh_x];
		v_real[i][j]=new CScalar[N_mesh_x];
	}
	}
    
    // allocate space for polynomial needed to compute the influence function

	Denom_Coeff=new Scalar[P_order];
	
	// construct the polynomials needed to compute the denominator coefficients of the influence function 
	Denominator_Poly_G();

	// allocate space for polynomial need to compute the charge distribution on the grid
	P_coeff=new Scalar*[P_order];
	for(unsigned int i=0;i<P_order;i++)P_coeff[i]=new Scalar[P_order]; 

	// Compute the polynomial coefficients needed for the charge distribution
	ComputePolyCoeff();
		
	// the charge distribution on the grid is quite different if the number is odd or even so
	// we decide at run time whether to use the rho even or rho odd function

	if(P_order%2) {
		make_rho_helper=&ElectrostaticLongRangePPPM::make_rho_odd;
		back_interpolate_helper=&ElectrostaticLongRangePPPM::back_interpolate_odd;
	}
	else{
		make_rho_helper=&ElectrostaticLongRangePPPM::make_rho_even;
		back_interpolate_helper=&ElectrostaticLongRangePPPM::back_interpolate_even;
	}

	//compute the influence function

	Compute_G();

	//this finalizes the construction of the class
}

ElectrostaticLongRangePPPM::~ElectrostaticLongRangePPPM()
	{
	
	//deallocate influence function and rho in real and k-space

	for(unsigned int j=0;j<N_mesh_z;j++){
	for(unsigned int i=0;i<N_mesh_y;i++){
		delete[] rho_real[i][j];
		delete[] rho_kspace[i][j];
		delete[] G_Inf[i][j];
		delete[] fx_kspace[i][j];
		delete[] fy_kspace[i][j];
		delete[] fz_kspace[i][j];
		delete[] e_kspace[i][j];
		delete[] v_kspace[i][j];
		delete[] fx_real[i][j];
		delete[] fy_real[i][j];
		delete[] fz_real[i][j];
		delete[] e_real[i][j];
		delete[] v_real[i][j];
	}
	}

	for(unsigned int i=0;i<N_mesh_z;i++){
		delete[] rho_real[i];
		delete[] rho_kspace[i];
		delete[] G_Inf[i];
		delete[] fx_kspace[i];
		delete[] fy_kspace[i];
		delete[] fz_kspace[i];
		delete[] e_kspace[i];
		delete[] v_kspace[i];
		delete[] fx_real[i];
		delete[] fy_real[i];
		delete[] fz_real[i];
		delete[] e_real[i];
		delete[] v_real[i];
	}
		
	delete[] rho_real;
	delete[] rho_kspace;
	delete[] G_Inf;
	delete[] fx_kspace;
	delete[] fy_kspace;
	delete[] fz_kspace;
	delete[] e_kspace;
	delete[] v_kspace;
	delete[] fx_real;
	delete[] fy_real;
	delete[] fz_real;
	delete[] e_real;
	delete[] v_real;

	//deallocate polynomial coefficients
	
	for(unsigned int i=0;i<P_order;i++) delete[] P_coeff[i]; 

	delete[] P_coeff;	
	delete[] Denom_Coeff;
}

void ElectrostaticLongRangePPPM::make_rho(void)
{
	(this->*make_rho_helper)();
}

void ElectrostaticLongRangePPPM::back_interpolate(CScalar ***Grid,Scalar *Continuum)
{
	(this->*back_interpolate_helper)(Grid,Continuum);
}

void ElectrostaticLongRangePPPM::make_rho_even(void)
{
	int P_half=static_cast<int>(P_order/2);

	//initialize rho to zero
    for(unsigned int i=0;i<N_mesh_x;i++){
	for(unsigned int j=0;j<N_mesh_y;j++){
	for(unsigned int k=0;k<N_mesh_z;k++){
		(rho_real[i][j][k]).r=0.0;
		(rho_real[i][j][k]).i=0.0;
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
				(rho_real[ind_x][ind_y][ind_z]).r+=q_i*Poly(lx+P_half-1,dx)*Poly(ly+P_half+1,dy)*Poly(lz+P_half+1,dz);
	    }
		}
		}
	}
	 //The charge is now defined on the grid for P even
}
void ElectrostaticLongRangePPPM::back_interpolate_even(CScalar ***Grid,Scalar *Continuum)
{
	int P_half=static_cast<int>(P_order/2);

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
				Continuum[i]+=((Grid[ind_x][ind_y][ind_z]).r)*Poly(lx+P_half-1,dx)*Poly(ly+P_half+1,dy)*Poly(lz+P_half+1,dz);
	    }
		}
		}
	}
}
void ElectrostaticLongRangePPPM::make_rho_odd(void)
{
	
    int P_half=static_cast<int>(P_order/2);

	//initialize rho to zero
    for(unsigned int i=0;i<N_mesh_x;i++){
	for(unsigned int j=0;j<N_mesh_y;j++){
	for(unsigned int k=0;k<N_mesh_z;k++){
		(rho_real[i][j][k]).r=0.0;
		(rho_real[i][j][k]).i=0.0;
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
				(rho_real[ind_x][ind_y][ind_z]).r+=q_i*Poly(lx+P_half,dx)*Poly(ly+P_half,dy)*Poly(lz+P_half,dz);
	    }
		}
		}
	}

}

void ElectrostaticLongRangePPPM::back_interpolate_odd(CScalar ***Grid,Scalar *Continuum)
{ 
	
	int P_half=static_cast<int>(P_order/2);

	// access the particle data
	const ParticleDataArraysConst& arrays = m_pdata->acquireReadOnly(); 
	// sanity check
	assert(arrays.x != NULL && arrays.y != NULL && arrays.z != NULL);

	//place quantity Grid back into continuum
	for (unsigned int i = 0; i < arrays.nparticles; i++)
		{
		// access the particle's position and charge (MEM TRANSFER: 4 Scalars)
		Scalar xi = arrays.x[i];
		Scalar yi = arrays.y[i];
		Scalar zi = arrays.z[i];

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
				Continuum[i]+=((Grid[ind_x][ind_y][ind_z]).r)*Poly(lx+P_half,dx)*Poly(ly+P_half,dy)*Poly(lz+P_half,dz);
	    }
		}
		}
	}
}

Scalar ElectrostaticLongRangePPPM::Poly(int l,Scalar x)
{
	Scalar P_res=0;

	for(int i=static_cast<int>(P_order)-1;i>=0;i--){
		P_res+=P_coeff[l][i]+x*P_res;
	}

	return P_res;
}

void ElectrostaticLongRangePPPM::ComputePolyCoeff(void)
{
	//This piece of code is not pretty, but gets the job done
	//It is inspired on a similar routine that exists in LAMMPS

  Scalar s;

  double **a;
  double **b;

  a=new double*[2*P_order];
  b=new double*[P_order];

  for(unsigned int i=0;i<2*P_order;i++) a[i]= new double[P_order];
  
  for(unsigned int i=0;i<P_order;i++) b[i]= new double[P_order];
  

  for (int k = -static_cast<int>(P_order); k <= static_cast<int>(P_order); k++) 
    for (int l = 0; l < static_cast<int>(P_order); l++)
      a[l][k+P_order] = 0.0;
        
  a[0][P_order] = 1.0;
  for (int j = 1; j < static_cast<int>(P_order); j++) {
    for (int k = -j; k <= j; k += 2) {
      s = 0.0;
      for (int l = 0; l < j; l++) {
	a[l+1][k+P_order] = (a[l][k+1+P_order]-a[l][k-1+P_order]) / (l+1);
	s += pow(0.5,(double) l+1) * 
	  (a[l][k-1+P_order] + pow(-1.0,(double) l) * a[l][k+1+P_order]) / (l+1);
      }
      a[0][k+P_order] = s;
    }
  }

  int m = (1-static_cast<int>(P_order))/2;
  
  int i_add=0;
  if(!(P_order%2)) i_add=1;
  for (int k = -(static_cast<int>(P_order)-1); k < static_cast<int>(P_order); k+=2) {
	  for (int l = 0; l < static_cast<int>(P_order); l++){
	  b[(k+i_add)/2+P_order/2-i_add][l]=a[l][k+P_order];
	  }
    m++;
  }
   //The b coefficients are not in the right order, this is why we need to reverse them

	for(unsigned int j1=0;j1<P_order;j1++){
	  for(unsigned int j2=0;j2<P_order;j2++){
		  P_coeff[P_order-j1-1][j2]=b[j1][j2];
		  }
		}

  //clean up the mess, and it is a mess
  
  for(unsigned int i=0;i<2*P_order;i++) delete[] a[i];
  
  for(unsigned int i=0;i<P_order;i++) delete[] b[i];

  delete[] a;
  delete[] b;
  
}

void ElectrostaticLongRangePPPM::Compute_G(void)
{
	vector<Scalar> v_num;
	Scalar xsi,ysi,zsi;
	Scalar k_x,k_y,k_z;
	Scalar k_per_x,k_per_y,k_per_z,k_per_norm;

	for(unsigned int i=0;i<N_mesh_x;i++){
	for(unsigned int j=0;j<N_mesh_y;j++){
	for(unsigned int k=0;k<N_mesh_z;k++){
		G_Inf[i][j][k]=0.0;
	}
	}
	}
    
	for(unsigned int i=0;i<N_mesh_x;i++){
		k_x=2*i*M_PI/h_x;
		k_per_x=2*M_PI*static_cast<Scalar>(i-((2*i)/N_mesh_x)*N_mesh_x)/h_x; 
        xsi=sin(k_x);
	for(unsigned int j=0;j<N_mesh_y;j++){
		k_y=2*j*M_PI/h_y;
		k_per_y=2*M_PI*static_cast<Scalar>(j-((2*j)/N_mesh_y)*N_mesh_y)/h_y;
        ysi=sin(k_y);
	for(unsigned int k=0;k<N_mesh_z;k++){
		k_z=2*k*M_PI/h_z;
		k_per_z=2*M_PI*static_cast<Scalar>(k-((2*k)/N_mesh_z)*N_mesh_z)/h_z;
        zsi=sin(k_z);

		k_per_norm=k_per_x*k_per_x+k_per_y*k_per_y+k_per_z*k_per_z; // modulus of the derivative
		v_num=Numerator_G(k_x,k_y,k_z);	

		G_Inf[i][j][k]=(k_per_x*v_num[0]+k_per_y*v_num[1]+k_per_z*v_num[2])/(k_per_norm*Denominator_G(xsi,ysi,zsi));
	}
	}
	}


	// Influence function has been computed

}

Scalar ElectrostaticLongRangePPPM::Denominator_G(Scalar xsi,Scalar ysi,Scalar zsi)
{
	//Computation of the denominator of the influence function, excluding the derivative square term
	//                   inf           P_order-1
	//we need to compute Sum W(k+pi*j)^2=Sum b(l)*sin(k*h/2)^(2l) 
	//                   j=-inf          l=0
	//The coefficients b(l)=Denom_Coeff[l] are defined in another routine
	//the variable xsi,ysi,zsi are xsi=sin(k_x*h_x/2),ysi=sin(k_y*h_y/2),zsi=sin(k_z*h_z/2)
	
	Scalar s_x=0;
	Scalar s_y=0;
	Scalar s_z=0;

	for(unsigned int j=P_order-1;j>=0;j--){
		s_x=Denom_Coeff[j]+s_x*xsi;
		s_y=Denom_Coeff[j]+s_y*ysi;
		s_z=Denom_Coeff[j]+s_z*ysi;
	}

	return s_x*s_x*s_y*s_y*s_z*s_z;
}

vector<Scalar> ElectrostaticLongRangePPPM::Numerator_G(Scalar kx,Scalar ky,Scalar kz)
{
	//Arguments kx,ky,kz are the k-space vectors
	//Numerator of the influence function
	 
	vector<Scalar> DG(3);
	//define the return type

	for(int j=0;j<3;j++) DG[j]=0.0;

	//The number of terms to be added are calculated with more precision than e-10
	//This is way more than needed, but given that this function is precomputed it does
	//not cause any overhead in the calculation.

	int n_x=static_cast<int>(m_alpha*h_x*sqrt(10*log(10.0))/3.16)+1;
	int n_y=static_cast<int>(m_alpha*h_y*sqrt(10*log(10.0))/3.16)+1;
	int n_z=static_cast<int>(m_alpha*h_z*sqrt(10*log(10.0))/3.16)+1;

	//identify the zero mode

	bool is_mode_k_zero=false;

	if( (fabs(kx)<1/Lx)&&(fabs(ky)<1/Ly)&&(fabs(kz)<1/Lz)) is_mode_k_zero=true;
    
	//we will use double precision and then cast it to Scalar

	double kx_n,ky_n,kz_n,k_mod;
	double skx,sky,skz,sk_all;
	double D_exp;
	double D_cum_x=0.0;
	double D_cum_y=0.0;
	double D_cum_z=0.0;

	
	for(int j_x=-n_x;j_x<=n_x;j_x++){
		kx_n=kx+2*j_x*M_PI/h_x;
		skx=pow(boost::math::sinc_pi(kx_n*h_x/2.0),static_cast<int>(2*P_order));
		for(int j_y=-n_y;j_y<=n_y;j_y++){
			ky_n=ky+2*j_y*M_PI/h_y;
			sky=pow(boost::math::sinc_pi(ky_n*h_y/2.0),static_cast<int>(2*P_order));
			for(int j_z=-n_z;j_z<=n_z;j_z++){
						kz_n=kz+2*j_z*M_PI/h_z;
						skz=pow(boost::math::sinc_pi(kz_n*h_z/2.0),static_cast<int>(2*P_order));
	
						//the zero mode (k_x^2+k_y^2+k_z^2=0) requires special treatment
						if((is_mode_k_zero)&&(!(j_z==0))&&(!(j_y==0))&&(!(j_x==0))){

						sk_all=skx*sky*skz;
						k_mod=kx_n*kx_n+ky_n*ky_n+kz_n*kz_n;
						
						D_exp=4*M_PI*exp(-k_mod/(4*m_alpha*m_alpha));

						D_cum_x+=D_exp*sk_all/k_mod;
						D_cum_y+=D_exp*sk_all/k_mod;
						D_cum_z+=D_exp*sk_all/k_mod;
						}

										}
									}
								}

						DG[0]=static_cast<Scalar>(D_cum_x);
						DG[1]=static_cast<Scalar>(D_cum_y);
						DG[2]=static_cast<Scalar>(D_cum_z);

	return DG;
}

void ElectrostaticLongRangePPPM::Denominator_Poly_G(void)
{
	//The coefficients Denom_Coeff are computed from the recurrence relation 
	//
	//
	for(int l=0;l<static_cast<int>(P_order);l++) Denom_Coeff[l]=0.0;
	Denom_Coeff[0]=1;

	for(int j=1;j<static_cast<int>(P_order);j++){
		for(int l=j;l>0;l++) Denom_Coeff[l]=4.0*(Denom_Coeff[l]*(l-j)*(l-j-0.5)-Denom_Coeff[l-1]*(l-j-1)*(l-j-1));//CHECK THIS
		Denom_Coeff[0]=4.0*j*(j+0.5)*Denom_Coeff[0];//CHECK THIS
	}

	//There is a 1/(2P_order-1)! coefficient to be added

	unsigned int ifact=1;
	for(unsigned int j=1;j<2*P_order;j++) ifact*=j;
	Scalar g_coeff=1/static_cast<Scalar>(ifact);

	for(unsigned int j=0;j<P_order;j++) Denom_Coeff[j]*=g_coeff;
}

const Scalar & ElectrostaticLongRangePPPM::Influence_function(unsigned int ix,unsigned int iy,unsigned int iz) const
{
	if((ix<N_mesh_x)&&(iy<N_mesh_y)&&(iz<N_mesh_z)){
		return G_Inf[ix][iy][iz];
	}
	else
	{
		cerr << endl << "***Error! attempting to access a non existing value of the influence function" << endl << endl;
		throw runtime_error("Error in Influence_function member function of ElectrostaticLongRangePPPM class ");
	}
}

const CScalar & ElectrostaticLongRangePPPM::Show_rho_real(unsigned int ix,unsigned int iy,unsigned int iz) const
{
	if((ix<N_mesh_x)&&(iy<N_mesh_y)&&(iz<N_mesh_z)){
		return rho_real[ix][iy][iz];
	}
	else
	{
		cerr << endl << "***Error! attempting to access a non existing value of the mesh density" << endl << endl;
		throw runtime_error("Error in Show_rho_real member function of ElectrostaticLongRangePPPM class ");
	}
}

const Scalar & ElectrostaticLongRangePPPM::Poly_coeff_Grid(unsigned int i,unsigned int j)
{
	if((i<P_order)&&(j<P_order)) return P_coeff[i][j];
	else
	{
		cerr << endl << "***Error! attempting to access a non existing value of the Polynomial coefficient" << endl << endl;
		throw runtime_error("Error in Poly_coeff_Grid member function of ElectrostaticLongRangePPPM class ");
	}
}

unsigned int ElectrostaticLongRangePPPM::N_mesh_x_axis(void) const
{	
	return N_mesh_x;
}

unsigned int ElectrostaticLongRangePPPM::N_mesh_y_axis(void) const
{	
	return N_mesh_y;
}

unsigned int ElectrostaticLongRangePPPM::N_mesh_z_axis(void) const
{	
	return N_mesh_z;
}

void ElectrostaticLongRangePPPM::computeForces(unsigned int timestep)
	{
	// start the profile
	if (m_prof) m_prof->push("ElecLongRange");
	
	// tally up the number of forces calculated
	int64_t n_calc = 0;

	// access the particle data
	const ParticleDataArraysConst& arrays = m_pdata->acquireReadOnly(); 
	// sanity check
	assert(arrays.x != NULL && arrays.y != NULL && arrays.z != NULL);
	
	// need to start from a zero force, energy and virial
	// (MEM TRANSFER: 5*N scalars)
	memset(m_fx, 0, sizeof(Scalar)*arrays.nparticles);
	memset(m_fy, 0, sizeof(Scalar)*arrays.nparticles);
	memset(m_fz, 0, sizeof(Scalar)*arrays.nparticles);
	memset(m_pe, 0, sizeof(Scalar)*arrays.nparticles);
	memset(m_virial, 0, sizeof(Scalar)*arrays.nparticles);
	
	//assign the charge to the grid
	make_rho();

	//compute rho in k space
	FFT->cmplx_fft(N_mesh_x,N_mesh_y,N_mesh_z,rho_real,rho_kspace,-1);

	//compute energy in k-space

	for(unsigned int i=0;i<N_mesh_x;i++){
	for(unsigned int j=0;j<N_mesh_y;j++){
	for(unsigned int k=0;k<N_mesh_z;k++){
		(e_kspace[i][j][k]).r=G_Inf[i][j][k]*((rho_kspace[i][j][k]).r);
		(e_kspace[i][j][k]).i=G_Inf[i][j][k]*((rho_kspace[i][j][k]).i);
	}
	}
	}

    //compute forces in k-space

	Scalar k_per_x,k_per_y,k_per_z;

	for(unsigned int i=0;i<N_mesh_x;i++){
	k_per_x=2*M_PI*static_cast<Scalar>(i-((2*i)/N_mesh_x)*N_mesh_x)/h_x;
	for(unsigned int j=0;j<N_mesh_y;j++){
	k_per_y=2*M_PI*static_cast<Scalar>(j-((2*j)/N_mesh_y)*N_mesh_y)/h_y;
	for(unsigned int k=0;k<N_mesh_z;k++){
	k_per_z=2*M_PI*static_cast<Scalar>(k-((2*k)/N_mesh_z)*N_mesh_z)/h_z;
		(fx_kspace[i][j][k]).r=k_per_x*((e_kspace[i][j][k]).r);
		(fx_kspace[i][j][k]).i=k_per_x*((e_kspace[i][j][k]).i);
		(fy_kspace[i][j][k]).r=k_per_y*((e_kspace[i][j][k]).r);
		(fy_kspace[i][j][k]).i=k_per_y*((e_kspace[i][j][k]).i);
		(fz_kspace[i][j][k]).r=k_per_z*((e_kspace[i][j][k]).r);
		(fz_kspace[i][j][k]).i=k_per_z*((e_kspace[i][j][k]).i);
	}
	}
	}
	
	//compute the energies and forces on the grid
	
	FFT->cmplx_fft(N_mesh_x,N_mesh_y,N_mesh_z,e_kspace,e_real,+1);
	FFT->cmplx_fft(N_mesh_x,N_mesh_y,N_mesh_z,fx_kspace,fx_real,+1);
	FFT->cmplx_fft(N_mesh_x,N_mesh_y,N_mesh_z,fy_kspace,fy_real,+1);
	FFT->cmplx_fft(N_mesh_x,N_mesh_y,N_mesh_z,fz_kspace,fz_real,+1);

	//Back interpolate to obtain the forces

	back_interpolate(e_real,m_pe);
	back_interpolate(fx_real,m_fx);
	back_interpolate(fy_real,m_fy);
	back_interpolate(fz_real,m_fz);

	//the previous formula computes the electric field, need to multiply by the charge

	for (unsigned int i = 0; i < arrays.nparticles; i++)
		{
			Scalar q_i= arrays.charge[i];
			m_fx[i]*=q_i;
			m_fy[i]*=q_i;
			m_fz[i]*=q_i;
			m_pe[i]*=q_i;
		}

	// and this is it, forces, energies and virial(TO DO) are calculated

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
