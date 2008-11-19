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
//! minimum force worth computing
const Scalar MIN_force=Scalar(1.0e-9); 

//! Typedef ElectrostaticLongRangePPPM factory
typedef function<shared_ptr<ElectrostaticLongRangePPPM> (shared_ptr<ParticleData> pdata,unsigned int Mmesh_x,unsigned int Mmesh_y,unsigned int Mmesh_z, unsigned int P_order_a, Scalar alpha, shared_ptr<FFTClass> FFTP,bool third_law_m)> LongRangePPPM_creator;
	
//! Test the ability of the Short Range Electrostatic force compute to actually calculate forces
/*! \param LongRangePPPM_object_n1 I have no idea: the write of this code needs to document it better
	\note With the creator as a parameter, the same code can be used to test any derived child
		of ElectrostaticLongRangePPPM
*/
void LongRangePPPM_PositionGrid(LongRangePPPM_creator LongRangePPPM_object_n1)
	{
	cout << "Testing charge distribution on the grid in class ElectrostaticLongRangePPPM" << endl;
	// Simple test to check that the charge is defined correctly on the grid
	shared_ptr<ParticleData> pdata_6(new ParticleData(6, BoxDim(20.0,40.0,60.0), 1));
	ParticleDataArrays arrays = pdata_6->acquireReadWrite();

	// six charges are located near the edge of the box
	arrays.x[0]=Scalar(-9.6);arrays.y[0]=Scalar(0.0);arrays.z[0]=Scalar(0.0);arrays.charge[0]=1.0;
    arrays.x[1]=Scalar(9.6);arrays.y[1]=Scalar(0.0);arrays.z[1]=Scalar(0.0);arrays.charge[1]=-2.0;
	arrays.x[2]=Scalar(0.0);arrays.y[2]=Scalar(-19.5);arrays.z[2]=Scalar(0.0);arrays.charge[2]=1.0;
    arrays.x[3]=Scalar(0.0);arrays.y[3]=Scalar(19.5);arrays.z[3]=Scalar(0.0);arrays.charge[3]=3.0;
    arrays.x[4]=Scalar(0.0);arrays.y[4]=Scalar(0.0);arrays.z[4]=Scalar(-29.4);arrays.charge[4]=-1.0;
    arrays.x[5]=Scalar(0.0);arrays.y[5]=Scalar(0.0);arrays.z[5]=Scalar(29.4);arrays.charge[5]=-2.0;

	// allow for acquiring data in the future
	pdata_6->release();

    // Define mesh parameters as well as order of the distribution, etc.. 
	unsigned int Nmesh_x=40;
	unsigned int Nmesh_y=80;
	unsigned int Nmesh_z=120; 
	unsigned int P_order=6; 
	Scalar alpha=4.0;
	shared_ptr<FftwWrapper> FFTW(new  FftwWrapper(Nmesh_x,Nmesh_y,Nmesh_z));
	bool third_law=false;

	//shared_ptr<ElectrostaticLongRangePPPM> PPPM_6=LongRangePPPM_object_n1(pdata_6,Nmesh_x,Nmesh_y,Nmesh_z,P_order,alpha,FFTW,third_law);
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
	Exact[3][0]=1682.0;Exact[3][1]=1540.0;Exact[3][2]=880.0;Exact[3][3]=-1120.0;Exact[3][4]=-160.0;Exact[3][5]=320.0;
	Exact[4][0]=237.0;Exact[4][1]=750.0;Exact[4][2]=840.0;Exact[4][3]=240.0;Exact[4][4]=-240.0;Exact[4][5]=-160.0;
	Exact[5][0]=1.0;Exact[5][1]=10.0;Exact[5][2]=40.0;Exact[5][3]=80.0;Exact[5][4]=80.0;Exact[5][5]=32.0;
	
	for(unsigned int i=0;i<P_order;i++){
		for(unsigned int j=0;j<P_order;j++){
				Exact[i][j]*=static_cast<Scalar>(1/3840.0);
		}
	}

	for(unsigned int i=0;i<P_order;i++){
		for(unsigned int j=0;j<P_order;j++){
				//MY_BOOST_CHECK_CLOSE(Exact[i][j],PPPM_6->Poly_coeff_Grid(i,j),tol);
		}
	}
	
	//Check passed, now let us compute the charges on the grid
	for(unsigned int i=0;i<P_order;i++) delete[] Exact[i];
	delete [] Exact;
}

//! ElectrostaticShortRange creator for unit tests
shared_ptr<ElectrostaticLongRangePPPM> base_class_PPPM_creator(shared_ptr<ParticleData> pdata,unsigned int Nmesh_x,unsigned int Nmesh_y,unsigned int Nmesh_z, unsigned int P_order, Scalar alpha,shared_ptr<FFTClass> FFTW,bool third_law_m)
	{
	return shared_ptr<ElectrostaticLongRangePPPM>(new ElectrostaticLongRangePPPM(pdata, Nmesh_x, Nmesh_y, Nmesh_z, P_order,alpha,FFTW,third_law_m));
	}
	
//! boost test case for particle test on CPU
BOOST_AUTO_TEST_CASE(LongRangePPPM_PositionGrid_test)
{
	LongRangePPPM_creator LongRangePPPM_creator_base = bind(base_class_PPPM_creator, _1, _2, _3, _4, _5,_6,_7,_8);
	LongRangePPPM_PositionGrid(LongRangePPPM_creator_base);
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
