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
#pragma warning( disable : 4103 4244 )
#define _USE_MATH_DEFINES
#endif

#include <iostream>

//! Name the unit test module
#define BOOST_TEST_MODULE Fourier_transform_test
#include "boost_utf_configure.h"

#include <boost/test/floating_point_comparison.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include "ParticleData.h"

#include <math.h>

using namespace std;
using namespace boost;


#ifdef USE_FFTW

#include "FftwWrapper.h"
#include "IndexTransform.h"

/*! \file Fourier_transform_test.cc
	\brief Implements a simple unit test for fft using fftw
	\ingroup unit_tests
*/

//! Define the minimum value to be considered
#define TOL 1e-5

//! Helper macro for testing if two numbers are close
#define MY_BOOST_CHECK_CLOSE(a,b,c) BOOST_CHECK_CLOSE(a,Scalar(b),Scalar(c))

//! Tolerance in percent to use for comparing the accuracy of the fourier transform
const Scalar tol = Scalar(1);

//! Typedef'd fftw factory
typedef boost::function<shared_ptr<FftwWrapper> (unsigned int N_x,unsigned int N_y,unsigned int N_z)> fftw_creator;
	
//! Test the fftw wrapper class for HOOMD
/*! \param fftw_test Creator class to create the fftw wrapper
	\note Only fftw is implemented, but the same test can be applied to any other fft
*/
void fftw_accuracy_test(fftw_creator fftw_test)
	{
	cout << "Testing the fftw implementation on HOOMD" << endl;

    double Exact_Conf_real(double x,double y,double l);
	double Exact_Conf_Imag(double x,double y,double l);

	unsigned int N_x=35;
	unsigned int N_y=27;
	unsigned int N_z=64;

	Scalar l_x,l_y,l_z;
	
	l_x=Scalar(0.02);
	l_y=Scalar(0.05);
	l_z=Scalar(1);

	IndexTransform T;
	T.SetD3to1D(N_x,N_y,N_z);
	unsigned int ind;

	CScalar *rho_real;
	CScalar *rho_kspace;
	CScalar *Exact_kspace;
	CScalar *rho_real_init;
	
	rho_real=new CScalar[N_x*N_y*N_z];
	rho_kspace=new CScalar[N_x*N_y*N_z];
	Exact_kspace=new CScalar[N_x*N_y*N_z];
	rho_real_init=new CScalar[N_x*N_y*N_z];

	// Create a fftw object with N_x,N_y,N_z points 
	shared_ptr<FftwWrapper> fftw_1=fftw_test(N_x,N_y,N_z);

	//we will do the fourier transform of an exponential

	for(unsigned int i=0;i<N_x;i++){
			for(unsigned int j=0;j<N_y;j++){
				for(unsigned int k=0;k<N_z;k++){
					ind=T.D3To1D(i,j,k);
					(rho_real[ind]).r=exp(-l_x*i-l_y*j-l_z*k);
					(rho_real[ind]).i=0.0;
					(rho_real_init[ind]).r=(rho_real[ind]).r;
					(rho_real_init[ind]).i=(rho_real[ind]).i;
					}
				}
			}

	//Compute the FFT
	fftw_1->cmplx_fft(N_x,N_y,N_z,rho_real,rho_kspace,1);

	//Calculate the exact analytical result

	Scalar a_x,a_y,a_z;
	Scalar b_x,b_y,b_z;

	for(unsigned int i=0;i<N_x;i++){
			for(unsigned int j=0;j<N_y;j++){
				for(unsigned int k=0;k<N_z;k++){
					ind=T.D3To1D(i,j,k);
					a_x=static_cast<Scalar>(Exact_Conf_real(N_x,i,l_x));
					a_y=static_cast<Scalar>(Exact_Conf_real(N_y,j,l_y));
					a_z=static_cast<Scalar>(Exact_Conf_real(N_z,k,l_z));
					b_x=static_cast<Scalar>(Exact_Conf_Imag(N_x,i,l_x));
					b_y=static_cast<Scalar>(Exact_Conf_Imag(N_y,j,l_y));
					b_z=static_cast<Scalar>(Exact_Conf_Imag(N_z,k,l_z));
					(Exact_kspace[ind]).r=a_x*a_y*a_z-b_x*b_y*a_z-a_x*b_y*b_z-b_x*a_y*b_z;
					(Exact_kspace[ind]).i=a_x*a_y*b_z+a_x*b_y*a_z+b_x*a_y*a_z-b_x*b_y*b_z;
					}
				}
			}
	
	//compare the exact result with the calculated result

	for(unsigned int i=0;i<N_x;i++){
			for(unsigned int j=0;j<N_y;j++){
				for(unsigned int k=0;k<N_z;k++){
					ind=T.D3To1D(i,j,k);
					a_x=(Exact_kspace[ind]).r;
					b_x=(rho_kspace[ind]).r;
					a_y=(Exact_kspace[ind]).i;
					b_y=(rho_kspace[ind]).i;
					if(!((fabs(a_x)<TOL)&&(fabs(b_x)<TOL)))
						if(fabs(a_x)<0.001)
							MY_BOOST_CHECK_CLOSE(a_x,b_x,10*tol);
						else MY_BOOST_CHECK_CLOSE(a_x,b_x,tol);
					if(!((fabs(a_y)<TOL)&&(fabs(b_y)<TOL)))
						if(fabs(a_y)<0.001)
							MY_BOOST_CHECK_CLOSE(a_y,b_y,10*tol);
						else MY_BOOST_CHECK_CLOSE(a_y,b_y,tol);
						//The reason for this somewhat convoluted expression is that
						//occasionally fftw is not accurate to the 5th decimal place
					}
				}
			}

		//Next test, do the inverse fft ...
	
		fftw_1->cmplx_fft(N_x,N_y,N_z,rho_kspace,rho_real,-1);

		Scalar vol=static_cast<Scalar>(N_x*N_y*N_z);

		//and make sure that it is what should be
		
		for(unsigned int i=0;i<N_x;i++){
			for(unsigned int j=0;j<N_y;j++){
				for(unsigned int k=0;k<N_z;k++){
					ind=T.D3To1D(i,j,k);
				    a_x=((rho_real[ind]).r)/vol;
					b_x=(rho_real_init[ind]).r;
					a_y=((rho_real[ind]).i)/vol;
					b_y=(rho_real_init[ind]).i;
					if(fabs(a_x)>TOL)
					MY_BOOST_CHECK_CLOSE(a_x,b_x,tol);
					if(fabs(a_y)>TOL)
					MY_BOOST_CHECK_CLOSE(a_y,b_y,tol);
					}
				}
			}
		
		// and this is it, now prevent memory leaks

	delete[] rho_real;
	delete[] rho_kspace;
	delete[] Exact_kspace;
	delete[] rho_real_init;

	}

//! ElectrostaticShortRange creator for unit tests
shared_ptr<FftwWrapper> base_class_FftwWrapper_creator(unsigned int N_x,unsigned int N_y,unsigned int N_z)
	{
	return shared_ptr<FftwWrapper>(new FftwWrapper(N_x,N_y,N_z));
	}
	
//! boost test case for FFTW on the CPU
BOOST_AUTO_TEST_CASE(fftw_test_accuracy)
	{
	fftw_creator fftw_creator_base = bind(base_class_FftwWrapper_creator, _1, _2, _3);
	fftw_accuracy_test(fftw_creator_base);
	}


//! these functions are the exact analytical solution of real and imaginary part
double Exact_Conf_real(double x,double y,double l)
{
  return (1-exp(-l*x))*(1-exp(-l)*cos(2*M_PI*y/x))/(1+exp(-2*l)-2*exp(-l)*cos(2*M_PI*y/x));	
}
//! these functions are the exact analytical solution of real and imaginary part
double Exact_Conf_Imag(double x,double y,double l)
{
  return -(1-exp(-l*x))*exp(-l)*sin(2*M_PI*y/x)/(1+exp(-2*l)-2*exp(-l)*cos(2*M_PI*y/x));	
}

#else

// We can't have the unit test passing if the code wasn't even compiled!
BOOST_AUTO_TEST_CASE(dummy_test)
	{
	BOOST_FAIL("Fourier_transform_test not compiled");
	}

#endif


#ifdef WIN32
#pragma warning( pop )
#endif

