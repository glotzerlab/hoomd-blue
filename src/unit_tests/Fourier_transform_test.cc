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
	\brief Implements two simple unit tests for fft
	\ingroup unit_tests
*/

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

    double Exact_Conf_real(double x,double y);
	double Exact_Conf_Imag(double x,double y);

	unsigned int N_x=32;
	unsigned int N_y=1;
	unsigned int N_z=1;

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

	for(unsigned int k=0;k<N_z;k++){
			for(unsigned int j=0;j<N_y;j++){
				for(unsigned int i=0;i<N_x;i++){
					ind=T.D3To1D(i,j,k);
					(rho_real[ind]).r=exp(-static_cast<Scalar>(i+j+k));
					(rho_real[ind]).i=0.0;
					(rho_real_init[ind]).r=(rho_real[ind]).r;
					(rho_real_init[ind]).i=(rho_real[ind]).i;
					}
				}
			}

	//Compute the FT
	fftw_1->cmplx_fft(N_x,N_y,N_z,rho_real,rho_kspace,-1);

	//Calculate the exact analytical result

	for(unsigned int k=0;k<N_z;k++){
			for(unsigned int j=0;j<N_y;j++){
				for(unsigned int i=0;i<N_x;i++){
					ind=T.D3To1D(i,j,k);
					(Exact_kspace[ind]).r=static_cast<Scalar>(Exact_Conf_real(N_z,k)*Exact_Conf_real(N_y,j)*Exact_Conf_real(N_x,i));
					(Exact_kspace[ind]).i=static_cast<Scalar>(Exact_Conf_Imag(N_z,k)*Exact_Conf_Imag(N_y,j)*Exact_Conf_Imag(N_x,i));
					}
				}
			}
	
	//compare the exact result with the calculated result

	for(unsigned int k=0;k<N_z;k++){
			for(unsigned int j=0;j<N_y;j++){
				for(unsigned int i=0;i<N_x;i++){
					ind=T.D3To1D(i,j,k);
					cout << " exact = " << (Exact_kspace[ind]).i << " calc = " << (rho_kspace[ind]).i << endl;
					MY_BOOST_CHECK_CLOSE((Exact_kspace[ind]).r,(rho_kspace[ind]).r,tol);
					MY_BOOST_CHECK_CLOSE((Exact_kspace[ind]).i,(rho_kspace[ind]).i,tol);
					}
				}
			}

		//Next test, do the inverse fft ...
	
		fftw_1->cmplx_fft(N_x,N_y,N_z,rho_kspace,rho_real,1);

		Scalar vol=static_cast<Scalar>(N_x*N_y*N_z);

		//and make sure that it is what should be
		
		for(unsigned int k=0;k<N_z;k++){
			for(unsigned int j=0;j<N_y;j++){
				for(unsigned int i=0;i<N_x;i++){
					ind=T.D3To1D(i,j,k);
					MY_BOOST_CHECK_CLOSE(vol*((rho_real[ind]).r),(rho_real_init[ind]).r,tol);
					MY_BOOST_CHECK_CLOSE(vol*((rho_real[ind]).i),(rho_real_init[ind]).i,tol);
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
double Exact_Conf_real(double x,double y)
{
  return (1-exp(-x))*(1-exp(-1.0)*cos(2*M_PI*y/x))/(1+exp(-2.0)-2*exp(-1.0)*cos(2*M_PI*y/x));	
}
//! these functions are the exact analytical solution of real and imaginary part
double Exact_Conf_Imag(double x,double y)
{
  return -(1-exp(-x))*exp(-1.0)*sin(2*M_PI*y/x)/(1+exp(-2.0)-2*exp(-1.0)*cos(2*M_PI*y/x));	
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

