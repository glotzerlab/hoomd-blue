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
#define _USE_MATH_DEFINES
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <iostream>

//! Name the unit test module
#define BOOST_TEST_MODULE ElectrostaticLongRangePPPMTests
#include "boost_utf_configure.h"

#include <boost/test/floating_point_comparison.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include "ParticleData.h"
#include "IndexTransform.h"

#include <math.h>

using namespace std;
using namespace boost;


//You need a fft defined in order to pass this text
#ifdef USE_FFTW

#include "ElectrostaticLongRangePPPM.h"
#include "FFTClass.h"
#include "FftwWrapper.h"

/*! \file ElectrostaticLongRange_PPPM_test.cc
	\brief Implements unit tests for ElectrostaticLongRangePPPM and descendants
	\ingroup unit_tests
*/

//! Helper macro for testing if two numbers are close
#define MY_BOOST_CHECK_CLOSE(a,b,c) BOOST_CHECK_CLOSE(a,Scalar(b),Scalar(c))
//! Helper macro for testing if a number is small
#define MY_BOOST_CHECK_SMALL(a,c) BOOST_CHECK_SMALL(a,Scalar(c))

//! Tolerance in percent to use for comparing various ElectrostaticLongRangePPPM to each other
const Scalar tol = Scalar(1);
//! minimum charge grid point worth computing
#define TOL 1e-7
//! minimum force worth computing
const Scalar MIN_force=Scalar(1.0e-9); 

//! Typedef ElectrostaticLongRangePPPM factory
typedef function<shared_ptr<ElectrostaticLongRangePPPM> (shared_ptr<ParticleData> pdata,unsigned int Mmesh_x,unsigned int Mmesh_y,unsigned int Mmesh_z, unsigned int P_order_a, Scalar alpha, shared_ptr<FFTClass> FFTP,bool third_law_m)> LongRangePPPM_creator;
	
//! Test the ability of the Short Range Electrostatic force compute to actually calculate forces
/*! \param LongRangePPPM_object_n1 I have no idea: the write of this code needs to document it better
	\note With the creator as a parameter, the same code can be used to test any derived child
		of ElectrostaticLongRangePPPM
*/
void LongRangePPPM_PositionGrid_even(LongRangePPPM_creator LongRangePPPM_object_n1)
	{
	cout << "Testing charge distribution on the grid in class ElectrostaticLongRangePPPM" << endl;
	// Simple test to check that the charge is defined correctly on the grid
	shared_ptr<ParticleData> pdata_6(new ParticleData(6, BoxDim(20.0,40.0,60.0), 1));

	ParticleDataArrays arrays = pdata_6->acquireReadWrite();

	// six charges are located near the edge of the box
	arrays.x[0]=Scalar(-9.6);arrays.y[0]=Scalar(0.1);arrays.z[0]=Scalar(0.0);arrays.charge[0]=1.0;
    arrays.x[1]=Scalar(9.7);arrays.y[1]=Scalar(0.0);arrays.z[1]=Scalar(-0.2);arrays.charge[1]=-2.0;
	arrays.x[2]=Scalar(0.5);arrays.y[2]=Scalar(-19.6);arrays.z[2]=Scalar(0.0);arrays.charge[2]=1.0;
    arrays.x[3]=Scalar(-0.5);arrays.y[3]=Scalar(19.1);arrays.z[3]=Scalar(0.0);arrays.charge[3]=3.0;
    arrays.x[4]=Scalar(0.7);arrays.y[4]=Scalar(0.0);arrays.z[4]=Scalar(-29.4);arrays.charge[4]=-1.0;
    arrays.x[5]=Scalar(0.0);arrays.y[5]=Scalar(0.0);arrays.z[5]=Scalar(29.9);arrays.charge[5]=-2.0;

	// allow for acquiring data in the future
	pdata_6->release();

    // Define mesh parameters as well as order of the distribution, etc.. 
	unsigned int Nmesh_x=20;
	unsigned int Nmesh_y=8;
	unsigned int Nmesh_z=12; 
	unsigned int P_order=6; 
	Scalar alpha=4.0;
	shared_ptr<FftwWrapper> FFTW(new  FftwWrapper(Nmesh_x,Nmesh_y,Nmesh_z));
	bool third_law=false;

	shared_ptr<ElectrostaticLongRangePPPM> PPPM_6=LongRangePPPM_object_n1(pdata_6,Nmesh_x,Nmesh_y,Nmesh_z,P_order,alpha,FFTW,third_law);
	// An ElectrostaticLongRangePPPM object with specified value of grid parameters, alpha, and fft routine instantiated
	
	// now let us check that the charges are correctly distributed

	//First check that the polynomials used to spread the charge on the grid are what they should be
	//This may eliminate unpleasant bugs

	//The values of the coefficents are taken from Appendix E in the Deserno and Holm paper
		
	Scalar **Exact=new Scalar*[P_order];
	for(unsigned int i=0;i<P_order;i++) Exact[i]=new Scalar[P_order];

	Exact[0][0]=1.0;Exact[0][1]=-10.0;Exact[0][2]=40.0;Exact[0][3]=-80.0;Exact[0][4]=80.0;Exact[0][5]=-32.0;
	Exact[1][0]=237.0;Exact[1][1]=-750.0;Exact[1][2]=840.0;Exact[1][3]=-240.0;Exact[1][4]=-240.0;Exact[1][5]=160.0;
	Exact[2][0]=1682.0;Exact[2][1]=-1540.0;Exact[2][2]=-880.0;Exact[2][3]=1120.0;Exact[2][4]=160.0;Exact[2][5]=-320.0;
	Exact[3][0]=1682.0;Exact[3][1]=1540.0;Exact[3][2]=-880.0;Exact[3][3]=-1120.0;Exact[3][4]=160.0;Exact[3][5]=320.0;
	Exact[4][0]=237.0;Exact[4][1]=750.0;Exact[4][2]=840.0;Exact[4][3]=240.0;Exact[4][4]=-240.0;Exact[4][5]=-160.0;
	Exact[5][0]=1.0;Exact[5][1]=10.0;Exact[5][2]=40.0;Exact[5][3]=80.0;Exact[5][4]=80.0;Exact[5][5]=32.0;
	
	for(unsigned int i=0;i<P_order;i++){
		for(unsigned int j=0;j<P_order;j++){
				Exact[i][j]*=static_cast<Scalar>(1/3840.0);
		}
	}

	for(unsigned int i=0;i<P_order;i++){
		for(unsigned int j=0;j<P_order;j++){
			    //high accuracy test 0.001%
				MY_BOOST_CHECK_CLOSE(Exact[i][j],PPPM_6->Poly_coeff_Grid(i,j),0.001*tol);
		}
	}
	
    //Check passed, Polynomial coeffs are good, now let us compute the charges on the grid

	// First define a matrix of real numbers
	Scalar *rho_by_hand=new Scalar[Nmesh_x*Nmesh_y*Nmesh_z];
	//Define a class to transform indices
	IndexTransform T;   
	T.SetD3to1D(Nmesh_x,Nmesh_y,Nmesh_z);
	unsigned int ind;

	// Initialize to zero
	for(unsigned int i=0;i<Nmesh_x;i++){
	for(unsigned int j=0;j<Nmesh_y;j++){
	for(unsigned int k=0;k<Nmesh_z;k++){
		ind=T.D3To1D(i,j,k);
		rho_by_hand[ind]=0.0;
	}
	}
	}

	//Distribute the first point (-9.6,0.1,0) on the grid defined above with (20,8,12) points
	{
	unsigned int i,j,k;
	int i_h;
	Scalar xs=0.0;
	Scalar ys=0.0;
	Scalar zs=0.0;
  
	for(unsigned int l=0;l<P_order;l++){
		i_h=-static_cast<int>(P_order/2)+1+static_cast<int>(l);
		i=static_cast<unsigned int>(i_h+((Nmesh_x-i_h-1)/Nmesh_x)*Nmesh_x);
		xs=0.0;
		for(int ii=static_cast<int>(P_order)-1;ii>=0;ii--) xs+=Exact[l][ii]-0.1*xs;
		for(unsigned int m=0;m<P_order;m++){
			j=Nmesh_y/2-P_order/2+1+m;
			ys=0.0;
			for(int ii=static_cast<int>(P_order)-1;ii>=0;ii--) ys+=Exact[m][ii]-0.48*ys;
				for(unsigned int n=0;n<P_order;n++){
				k=Nmesh_z/2-P_order/2+1+n;
				zs=0.0;
				for(int ii=static_cast<int>(P_order)-1;ii>=0;ii--) zs+=Exact[n][ii]-0.5*zs;
				ind=T.D3To1D(i,j,k);
				rho_by_hand[ind]+=xs*ys*zs;
						}
				}
		}
	}
    
    //Distribute the second point (9.7,0,-0.2) on the grid defined above with (20,8,12) points
	{
	unsigned int i,j,k;
	int i_h;
	Scalar xs=0.0;
	Scalar ys=0.0;
	Scalar zs=0.0;
	for(unsigned int l=0;l<P_order;l++){
		i_h=Nmesh_x-static_cast<int>(P_order/2)+static_cast<int>(l);
		i=static_cast<unsigned int>(i_h-(i_h/Nmesh_x)*Nmesh_x);
		xs=0.0;
		for(int ii=static_cast<int>(P_order)-1;ii>=0;ii--) xs+=Exact[l][ii]+0.2*xs;
		for(unsigned int m=0;m<P_order;m++){
			j=Nmesh_y/2-P_order/2+1+m;
			ys=0.0;
			for(int ii=static_cast<int>(P_order)-1;ii>=0;ii--) ys+=Exact[m][ii]-0.5*ys;
			for(unsigned int n=0;n<P_order;n++){
				k=Nmesh_z/2-P_order/2+n;
				zs=0.0;
				for(int ii=static_cast<int>(P_order)-1;ii>=0;ii--) zs+=Exact[n][ii]+0.46*zs;
				ind=T.D3To1D(i,j,k);
				rho_by_hand[ind]+=-2*xs*ys*zs;
				// second point has charge -2
						}
				}
		}
	}

	//Distribute the third point (0.5,-19.6,0) on the grid defined above with (20,8,12) points
	{
	unsigned int i,j,k;
	int i_h;
	Scalar xs=0.0;
	Scalar ys=0.0;
	Scalar zs=0.0;
	for(unsigned int l=0;l<P_order;l++){
		i=Nmesh_x/2-P_order/2+1+l;
		xs=0.0;
		for(int ii=static_cast<int>(P_order)-1;ii>=0;ii--) xs+=Exact[l][ii]-0.0*xs;
		for(unsigned int m=0;m<P_order;m++){
			i_h=-static_cast<int>(P_order/2)+1+static_cast<int>(m);
			j=static_cast<unsigned int>(i_h+((Nmesh_y-i_h-1)/Nmesh_y)*Nmesh_y);
			ys=0.0;
				for(int ii=static_cast<int>(P_order)-1;ii>=0;ii--) ys+=Exact[m][ii]-0.42*ys;
			for(unsigned int n=0;n<P_order;n++){
				k=Nmesh_z/2-P_order/2+1+n;
				zs=0.0;
				for(int ii=static_cast<int>(P_order)-1;ii>=0;ii--) zs+=Exact[n][ii]-0.5*zs;
			ind=T.D3To1D(i,j,k);
			rho_by_hand[ind]+=xs*ys*zs;
						}
				}
		}
	}

	//Distribute the fourth point (-0.5,19.1,0) on the grid defined above with (20,8,12) points
	{
	unsigned int i,j,k;
	int i_h;
	Scalar xs=0.0;
	Scalar ys=0.0;
	Scalar zs=0.0;
	for(unsigned int l=0;l<P_order;l++){
		i=Nmesh_x/2-P_order/2+l;
		xs=0.0;
		for(int ii=static_cast<int>(P_order)-1;ii>=0;ii--) xs+=Exact[l][ii]-0.0*xs;
		for(unsigned int m=0;m<P_order;m++){
			i_h=Nmesh_y-static_cast<int>(P_order/2)+m;
			j=static_cast<unsigned int>(i_h-(i_h/Nmesh_y)*Nmesh_y);
			ys=0.0;
				for(int ii=static_cast<int>(P_order)-1;ii>=0;ii--) ys+=Exact[m][ii]+0.32*ys;
			for(unsigned int n=0;n<P_order;n++){
				k=Nmesh_z/2-P_order/2+1+n;
				zs=0.0;
				for(int ii=static_cast<int>(P_order)-1;ii>=0;ii--) zs+=Exact[n][ii]-0.5*zs;
			ind=T.D3To1D(i,j,k);
			rho_by_hand[ind]+=3.0*xs*ys*zs;
			//This point has charge +3
						}
				}
		}
	}

    //Distribute the fifth point (0.7,0,-29.4) on the grid defined above with (20,8,12) points
	{
	unsigned int i,j,k;
	int i_h;
	Scalar xs=0.0;
	Scalar ys=0.0;
	Scalar zs=0.0;
	for(unsigned int l=0;l<P_order;l++){
		i=Nmesh_x/2-P_order/2+1+l;
		xs=0.0;
		for(int ii=static_cast<int>(P_order)-1;ii>=0;ii--) xs+=Exact[l][ii]+0.2*xs;
		for(unsigned int m=0;m<P_order;m++){
			j=Nmesh_y/2-P_order/2+1+m;
			ys=0.0;
				for(int ii=static_cast<int>(P_order)-1;ii>=0;ii--) ys+=Exact[m][ii]-0.5*ys;
			for(unsigned int n=0;n<P_order;n++){
				i_h=-static_cast<int>(P_order/2)+1+static_cast<int>(n);
				k=static_cast<unsigned int>(i_h+((Nmesh_z-i_h-1)/Nmesh_z)*Nmesh_z);
				zs=0.0;
				for(int ii=static_cast<int>(P_order)-1;ii>=0;ii--) zs+=Exact[n][ii]-0.38*zs;
			ind=T.D3To1D(i,j,k);
			rho_by_hand[ind]+=-xs*ys*zs;
			//This point has charge -1
						}
				}
		}
	}
	
	//Distribute the sixth point (0,0,29.9) on the grid defined above with (20,8,12) points
	
	{
	unsigned int i,j,k;
	int i_h;
	Scalar xs=0.0;
	Scalar ys=0.0;
	Scalar zs=0.0;
	for(unsigned int l=0;l<P_order;l++){
		i=Nmesh_x/2-P_order/2+1+l;
		xs=0.0;
		for(int ii=static_cast<int>(P_order)-1;ii>=0;ii--) xs+=Exact[l][ii]-0.5*xs;
		for(unsigned int m=0;m<P_order;m++){
			j=Nmesh_y/2-P_order/2+1+m;
			ys=0.0;
				for(int ii=static_cast<int>(P_order)-1;ii>=0;ii--) ys+=Exact[m][ii]-0.5*ys;
			for(unsigned int n=0;n<P_order;n++){
				i_h=Nmesh_z-static_cast<int>(P_order/2)+n;
				k=static_cast<unsigned int>(i_h-(i_h/Nmesh_z)*Nmesh_z);
				zs=0.0;
				for(int ii=static_cast<int>(P_order)-1;ii>=0;ii--) zs+=Exact[n][ii]+0.48*zs;
			ind=T.D3To1D(i,j,k);
			rho_by_hand[ind]+=-2.0*xs*ys*zs;
			//This point has charge -2
						}
				}
		}
	}
	
	//Now let us compare with rho as computed by PPPM class

	PPPM_6->make_rho();
	Scalar val;
	Scalar total_charge_mesh=0.0;
	Scalar add2=0.0;

	for(unsigned int i=0;i<Nmesh_x;i++){
	for(unsigned int j=0;j<Nmesh_y;j++){
	for(unsigned int k=0;k<Nmesh_z;k++){
		ind=T.D3To1D(i,j,k);
		val=(PPPM_6->Show_rho_real(i,j,k)).r;
		total_charge_mesh+=val;
		if(fabs(val)>TOL)
			MY_BOOST_CHECK_CLOSE(val,rho_by_hand[ind],0.1*tol);
	}
	}
	}

	Scalar total_charge=0.0;

	for (unsigned int i = 0; i < arrays.nparticles; i++)
		total_charge+=arrays.charge[i];

	//Check that the charge in the mesh adds to the total charge

	MY_BOOST_CHECK_SMALL(total_charge-total_charge_mesh,100*TOL);

	delete[] rho_by_hand;

	for(unsigned int i=P_order;i>0;--i) delete[] Exact[i-1];
	delete[] Exact;
}

// If the order of assignment is even or odd the class uses two different algorithms
// which are decided at runtime, the case of P odd needs to be tested as well

//! test odd
void LongRangePPPM_PositionGrid_odd(LongRangePPPM_creator LongRangePPPM_object_n2)
	{
	shared_ptr<ParticleData> pdata_6(new ParticleData(6, BoxDim(20.0,40.0,60.0), 1));

	ParticleDataArrays arrays = pdata_6->acquireReadWrite();

	// six charges are located near the edge of the box
	arrays.x[0]=Scalar(-9.6);arrays.y[0]=Scalar(0.1);arrays.z[0]=Scalar(0.0);arrays.charge[0]=1.0;
    arrays.x[1]=Scalar(9.7);arrays.y[1]=Scalar(0.0);arrays.z[1]=Scalar(-0.2);arrays.charge[1]=-2.0;
	arrays.x[2]=Scalar(0.5);arrays.y[2]=Scalar(-19.6);arrays.z[2]=Scalar(0.0);arrays.charge[2]=1.0;
    arrays.x[3]=Scalar(-0.5);arrays.y[3]=Scalar(19.1);arrays.z[3]=Scalar(0.0);arrays.charge[3]=3.0;
    arrays.x[4]=Scalar(0.7);arrays.y[4]=Scalar(0.0);arrays.z[4]=Scalar(-29.4);arrays.charge[4]=-1.0;
    arrays.x[5]=Scalar(0.0);arrays.y[5]=Scalar(0.0);arrays.z[5]=Scalar(29.9);arrays.charge[5]=-2.0;

	// allow for acquiring data in the future
	pdata_6->release();

    // Define mesh parameters as well as order of the distribution, etc.. 
	unsigned int Nmesh_x=20;
	unsigned int Nmesh_y=8;
	unsigned int Nmesh_z=12; 
	unsigned int P_order=3; 
	Scalar alpha=4.0;
	shared_ptr<FftwWrapper> FFTW(new  FftwWrapper(Nmesh_x,Nmesh_y,Nmesh_z));
	bool third_law=false;

	shared_ptr<ElectrostaticLongRangePPPM> PPPM_6=LongRangePPPM_object_n2(pdata_6,Nmesh_x,Nmesh_y,Nmesh_z,P_order,alpha,FFTW,third_law);
	// An ElectrostaticLongRangePPPM object with specified value of grid parameters, alpha, and fft routine instantiated
	// now let us check that the charges are correctly distributed

	//First check that the polynomials used to spread the charge on the grid are what they should be
	//This may eliminate unpleasant bugs

	//The values of the coefficents are taken from Appendix E in the Deserno and Holm paper
		
	Scalar **Exact=new Scalar*[P_order];
	for(unsigned int i=0;i<P_order;i++) Exact[i]=new Scalar[P_order];

	Exact[0][0]=1.0;Exact[0][1]=-4.0;Exact[0][2]=4.0;
	Exact[1][0]=6.0;Exact[1][1]=0.0;Exact[1][2]=-8.0;
	Exact[2][0]=1.0;Exact[2][1]=4.0;Exact[2][2]=4.0;
	
	for(unsigned int i=0;i<P_order;i++){
	for(unsigned int j=0;j<P_order;j++){
				Exact[i][j]*=static_cast<Scalar>(1/8.0);
		}
	}

	

	for(unsigned int i=0;i<P_order;i++){
		for(unsigned int j=0;j<P_order;j++){
			    //high accuracy test 0.001%
			    if(Exact[i][j]>TOL)
				MY_BOOST_CHECK_CLOSE(Exact[i][j],PPPM_6->Poly_coeff_Grid(i,j),0.001*tol);
		}
	}
	
    //Check passed, Polynomial coeffs are good, now let us compute the charges on the grid

	// First define a matrix of real numbers
	Scalar *rho_by_hand=new Scalar[Nmesh_x*Nmesh_y*Nmesh_z];
	//Define a class to transform indices
	IndexTransform T;   
	T.SetD3to1D(Nmesh_x,Nmesh_y,Nmesh_z);
	unsigned int ind;

	// Initialize to zero
	for(unsigned int i=0;i<Nmesh_x;i++){
	for(unsigned int j=0;j<Nmesh_y;j++){
	for(unsigned int k=0;k<Nmesh_z;k++){
		ind=T.D3To1D(i,j,k);
		rho_by_hand[ind]=0.0;
	}
	}
	}

	//Distribute the first point (-9.6,0.1,0) on the grid defined above with (20,8,12) points
	{
	unsigned int i,j,k;
	int i_h;
	Scalar xs=0.0;
	Scalar ys=0.0;
	Scalar zs=0.0;
  
	for(unsigned int l=0;l<P_order;l++){
		i_h=-static_cast<int>(P_order/2)+static_cast<int>(l);
		i=static_cast<unsigned int>(i_h+((Nmesh_x-i_h-1)/Nmesh_x)*Nmesh_x);
		xs=0.0;
		for(int ii=static_cast<int>(P_order)-1;ii>=0;ii--) xs+=Exact[l][ii]+0.4*xs;
		for(unsigned int m=0;m<P_order;m++){
			j=Nmesh_y/2-P_order/2+m;
			ys=0.0;
			for(int ii=static_cast<int>(P_order)-1;ii>=0;ii--) ys+=Exact[m][ii]+0.02*ys;
				for(unsigned int n=0;n<P_order;n++){
				k=Nmesh_z/2-P_order/2+n;
				zs=0.0;
				for(int ii=static_cast<int>(P_order)-1;ii>=0;ii--) zs+=Exact[n][ii]-0.0*zs;
				ind=T.D3To1D(i,j,k);
				rho_by_hand[ind]+=xs*ys*zs;
						}
				}
		}
	}
    
    //Distribute the second point (9.7,0,-0.2) on the grid defined above with (20,8,12) points
	{
	unsigned int i,j,k;
	int i_h;
	Scalar xs=0.0;
	Scalar ys=0.0;
	Scalar zs=0.0;
	for(unsigned int l=0;l<P_order;l++){
		i_h=Nmesh_x-static_cast<int>(P_order/2)+static_cast<int>(l);
		i=static_cast<unsigned int>(i_h-(i_h/Nmesh_x)*Nmesh_x);
		xs=0.0;
		for(int ii=static_cast<int>(P_order)-1;ii>=0;ii--) xs+=Exact[l][ii]-0.3*xs;
		for(unsigned int m=0;m<P_order;m++){
			j=Nmesh_y/2-P_order/2+m;
			ys=0.0;
			for(int ii=static_cast<int>(P_order)-1;ii>=0;ii--) ys+=Exact[m][ii]-0.0*ys;
			for(unsigned int n=0;n<P_order;n++){
				k=Nmesh_z/2-P_order/2+n;
				zs=0.0;
				for(int ii=static_cast<int>(P_order)-1;ii>=0;ii--) zs+=Exact[n][ii]-0.04*zs;
				ind=T.D3To1D(i,j,k);
				rho_by_hand[ind]+=-2*xs*ys*zs;
				// second point has charge -2
						}
				}
		}
	}
    
	
	//Distribute the third point (0.5,-19.6,0) on the grid defined above with (20,8,12) points
	{
	unsigned int i,j,k;
	int i_h;
	Scalar xs=0.0;
	Scalar ys=0.0;
	Scalar zs=0.0;
	for(unsigned int l=0;l<P_order;l++){
		i=Nmesh_x/2-P_order/2+l;
		xs=0.0;
		for(int ii=static_cast<int>(P_order)-1;ii>=0;ii--) xs+=Exact[l][ii]+0.5*xs;
		for(unsigned int m=0;m<P_order;m++){
			i_h=-static_cast<int>(P_order/2)+static_cast<int>(m);
			j=static_cast<unsigned int>(i_h+((Nmesh_y-i_h-1)/Nmesh_y)*Nmesh_y);
			ys=0.0;
				for(int ii=static_cast<int>(P_order)-1;ii>=0;ii--) ys+=Exact[m][ii]+0.08*ys;
			for(unsigned int n=0;n<P_order;n++){
				k=Nmesh_z/2-P_order/2+n;
				zs=0.0;
				for(int ii=static_cast<int>(P_order)-1;ii>=0;ii--) zs+=Exact[n][ii]-0.0*zs;
			ind=T.D3To1D(i,j,k);
			rho_by_hand[ind]+=xs*ys*zs;
						}
				}
		}
	}
	
	
	//Distribute the fourth point (-0.5,19.1,0) on the grid defined above with (20,8,12) points
	{
	unsigned int i,j,k;
	int i_h;
	Scalar xs=0.0;
	Scalar ys=0.0;
	Scalar zs=0.0;
	for(unsigned int l=0;l<P_order;l++){
		i=Nmesh_x/2-P_order/2-1+l;
		xs=0.0;
		for(int ii=static_cast<int>(P_order)-1;ii>=0;ii--) xs+=Exact[l][ii]+0.5*xs;
		for(unsigned int m=0;m<P_order;m++){
			i_h=Nmesh_y-static_cast<int>(P_order/2)+m;
			j=static_cast<unsigned int>(i_h-(i_h/Nmesh_y)*Nmesh_y);
			ys=0.0;
				for(int ii=static_cast<int>(P_order)-1;ii>=0;ii--) ys+=Exact[m][ii]-0.18*ys;
			for(unsigned int n=0;n<P_order;n++){
				k=Nmesh_z/2-P_order/2+n;
				zs=0.0;
				for(int ii=static_cast<int>(P_order)-1;ii>=0;ii--) zs+=Exact[n][ii]-0.0*zs;
			ind=T.D3To1D(i,j,k);
			rho_by_hand[ind]+=3.0*xs*ys*zs;
			//This point has charge +3
						}
				}
		}
	}

	
    //Distribute the fifth point (0.7,0,-29.4) on the grid defined above with (20,8,12) points
	{
	unsigned int i,j,k;
	int i_h;
	Scalar xs=0.0;
	Scalar ys=0.0;
	Scalar zs=0.0;
	for(unsigned int l=0;l<P_order;l++){
		i=Nmesh_x/2-P_order/2+1+l;
		xs=0.0;
		for(int ii=static_cast<int>(P_order)-1;ii>=0;ii--) xs+=Exact[l][ii]-0.3*xs;
		for(unsigned int m=0;m<P_order;m++){
			j=Nmesh_y/2-P_order/2+m;
			ys=0.0;
				for(int ii=static_cast<int>(P_order)-1;ii>=0;ii--) ys+=Exact[m][ii]-0.0*ys;
			for(unsigned int n=0;n<P_order;n++){
				i_h=-static_cast<int>(P_order/2)+static_cast<int>(n);
				k=static_cast<unsigned int>(i_h+((Nmesh_z-i_h-1)/Nmesh_z)*Nmesh_z);
				zs=0.0;
				for(int ii=static_cast<int>(P_order)-1;ii>=0;ii--) zs+=Exact[n][ii]+0.12*zs;
			ind=T.D3To1D(i,j,k);
			rho_by_hand[ind]+=-xs*ys*zs;
			//This point has charge -1
						}
				}
		}
	}
	
	//Distribute the sixth point (0,0,29.9) on the grid defined above with (20,8,12) points
	
	{
	unsigned int i,j,k;
	int i_h;
	Scalar xs=0.0;
	Scalar ys=0.0;
	Scalar zs=0.0;
	for(unsigned int l=0;l<P_order;l++){
		i=Nmesh_x/2-P_order/2+l;
		xs=0.0;
		for(int ii=static_cast<int>(P_order)-1;ii>=0;ii--) xs+=Exact[l][ii]-0.0*xs;
		for(unsigned int m=0;m<P_order;m++){
			j=Nmesh_y/2-P_order/2+m;
			ys=0.0;
				for(int ii=static_cast<int>(P_order)-1;ii>=0;ii--) ys+=Exact[m][ii]-0.0*ys;
			for(unsigned int n=0;n<P_order;n++){
				i_h=Nmesh_z-static_cast<int>(P_order/2)+n;
				k=static_cast<unsigned int>(i_h-(i_h/Nmesh_z)*Nmesh_z);
				zs=0.0;
				for(int ii=static_cast<int>(P_order)-1;ii>=0;ii--) zs+=Exact[n][ii]-0.02*zs;
			ind=T.D3To1D(i,j,k);
			rho_by_hand[ind]+=-2.0*xs*ys*zs;
			//This point has charge -2
						}
				}
		}
	}
	
	//Now let us compare with rho as computed by PPPM class
	PPPM_6->make_rho();
	
	Scalar val;
	Scalar total_charge_mesh=0.0;
	Scalar add2=0.0;

	for(unsigned int i=0;i<Nmesh_x;i++){
	for(unsigned int j=0;j<Nmesh_y;j++){
	for(unsigned int k=0;k<Nmesh_z;k++){
		ind=T.D3To1D(i,j,k);
		val=(PPPM_6->Show_rho_real(i,j,k)).r;
		total_charge_mesh+=val;
		if(fabs(val)>TOL)
			MY_BOOST_CHECK_CLOSE(val,rho_by_hand[ind],0.1*tol);
	}
	}
	}

	Scalar total_charge=0.0;

	for (unsigned int i = 0; i < arrays.nparticles; i++)
		total_charge+=arrays.charge[i];

	//Check that the charge in the mesh adds to the total charge

	MY_BOOST_CHECK_SMALL(total_charge-total_charge_mesh,100*TOL);

	delete[] rho_by_hand;

	for(unsigned int i=P_order;i>0;--i) delete[] Exact[i-1];
	delete[] Exact;
}

//! the author has decided to not document this function for inexplicable reasons
void LongRangePPPM_InfluenceFunction(LongRangePPPM_creator LongRangePPPM_object_IF)
	{
	// Unit test for the influence function
	shared_ptr<ParticleData> pdata_1(new ParticleData(1, BoxDim(20.0,20.0,20.0), 1));

	ParticleDataArrays arrays = pdata_1->acquireReadWrite();

	// six charges are located near the edge of the box
	arrays.x[0]=Scalar(-3.6);arrays.y[0]=Scalar(0.1);arrays.z[0]=Scalar(2.0);arrays.charge[0]=1.0;

	// allow for acquiring data in the future
	pdata_1->release();

    // Define mesh parameters as well as order of the distribution, etc.. 
	unsigned int Nmesh_x=20;
	unsigned int Nmesh_y=20;
	unsigned int Nmesh_z=20; 
	unsigned int P_order=6; 
	Scalar alpha=1.0;
	shared_ptr<FftwWrapper> FFTW(new  FftwWrapper(Nmesh_x,Nmesh_y,Nmesh_z));
	bool third_law=false;

	shared_ptr<ElectrostaticLongRangePPPM> PPPM_1=LongRangePPPM_object_IF(pdata_1,Nmesh_x,Nmesh_y,Nmesh_z,P_order,alpha,FFTW,third_law);

	//First we make sure that the analytical expression used to compute the denominator
	//of the influence function is what it should be

	Scalar Denom_by_hand;
	Scalar k_val;
	Scalar sin_k_val;
	Scalar sin_pow;
	Scalar k_val_pow;
	Scalar Denom_Exact;

	for(unsigned int n=1;n<Nmesh_x/2;n++){
		k_val=n*M_PI/static_cast<Scalar>(Nmesh_x);
		sin_k_val=sin(k_val);
		Denom_by_hand=0.0;
		Denom_Exact=0.0;
		sin_pow=1;
		for(int ii=0;ii<static_cast<int>(2*P_order);ii++) sin_pow*=sin_k_val;
		for(int jj=static_cast<int>(P_order-1);jj>=0;jj--) Denom_Exact=PPPM_1->Poly_coeff_Denom_Influence_Function(jj)+Denom_Exact*sin_k_val*sin_k_val;
		
		for(int m=-500;m<600;m++)//This value putts an extremely conservative cutoff to the sum
		{
			k_val_pow=1;
			for(int ii=0;ii<static_cast<int>(2*P_order);ii++) k_val_pow*=(k_val+M_PI*m);
			Denom_by_hand+=1/k_val_pow;
		}
			Denom_by_hand*=sin_pow;
			MY_BOOST_CHECK_CLOSE(Denom_by_hand,Denom_Exact,0.01*tol);
			//compare the denominator of the influence function with the approximate result
	    }
	
		//Now let us compare the rest of the influence function
}
//! ElectrostaticShortRange creator for unit tests
shared_ptr<ElectrostaticLongRangePPPM> base_class_PPPM_creator(shared_ptr<ParticleData> pdata,unsigned int Nmesh_x,unsigned int Nmesh_y,unsigned int Nmesh_z, unsigned int P_order, Scalar alpha,shared_ptr<FFTClass> FFTW,bool third_law_m)
	{
	return shared_ptr<ElectrostaticLongRangePPPM>(new ElectrostaticLongRangePPPM(pdata, Nmesh_x, Nmesh_y, Nmesh_z, P_order,alpha,FFTW,third_law_m));
	}
	
//! boost test case for particle test on CPU
/* BOOST_AUTO_TEST_CASE(LongRangePPPM_PositionGrid_Even_test)
{
	LongRangePPPM_creator LongRangePPPM_creator_base = bind(base_class_PPPM_creator, _1, _2, _3, _4, _5,_6,_7,_8);
	LongRangePPPM_PositionGrid_even(LongRangePPPM_creator_base);
}
BOOST_AUTO_TEST_CASE(LongRangePPPM_PositionGrid_Odd_test)
{
	LongRangePPPM_creator LongRangePPPM_creator_base = bind(base_class_PPPM_creator, _1, _2, _3, _4, _5,_6,_7,_8);
	LongRangePPPM_PositionGrid_odd(LongRangePPPM_creator_base);
}
*/

BOOST_AUTO_TEST_CASE(LongRangePPPM_InfluenceFunction_Test)
{
	LongRangePPPM_creator LongRangePPPM_creator_base = bind(base_class_PPPM_creator, _1, _2, _3, _4, _5,_6,_7,_8);
	LongRangePPPM_InfluenceFunction(LongRangePPPM_creator_base);
}



#else

// We can't have the unit test passing if the code wasn't even compiled!
BOOST_AUTO_TEST_CASE(dummy_test)
	{
	BOOST_FAIL("ElectrostaticLongRange_PPPM not compiled");
	}
#endif

#ifdef WIN32
#pragma warning( pop )
#endif
