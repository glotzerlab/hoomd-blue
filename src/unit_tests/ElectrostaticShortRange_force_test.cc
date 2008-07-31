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


#include <iostream>

//! Name the unit test module
#define BOOST_TEST_MODULE ShortRangeElectrostaticForceTests
#include "boost_utf_configure.h"

#include <boost/test/floating_point_comparison.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/math/special_functions/erf.hpp>

#include "ElectrostaticShortRange.h"

#include "BinnedNeighborList.h"
#include "Initializers.h"

#include <math.h>

#define EWALD_F  1.128379167
using namespace std;
using namespace boost;

/*! \file ElectrostaticShortRange_force_test.cc
	\brief Implements unit tests for ElectrostaticShortRange and descendants
	\ingroup unit_tests
*/

//! Helper macro for testing if two numbers are close
#define MY_BOOST_CHECK_CLOSE(a,b,c) BOOST_CHECK_CLOSE(a,Scalar(b),Scalar(c))
//! Helper macro for testing if a number is small
#define MY_BOOST_CHECK_SMALL(a,c) BOOST_CHECK_SMALL(a,Scalar(c))

//! Tolerance in percent to use for comparing various ElectrostaticShortRange to each other
#ifdef SINGLE_PRECISION
const Scalar tol = Scalar(1);
#else
const Scalar tol = 1e-6;
#endif

//! Typedef'd ElectrostaticShortRange factory
typedef boost::function<shared_ptr<ElectrostaticShortRange> (shared_ptr<ParticleData> pdata, shared_ptr<NeighborList> nlist, Scalar r_cut, Scalar alpha, Scalar delta, Scalar min_value)> ElectrostaticShortRange_force_creator;
	
//! Test the ability of the Short Range Electrostatic force compute to actually calculate forces
/*! 
	\note With the creator as a parameter, the same code can be used to test any derived child
		of ElectrostaticShortRange
*/
void ElectrostaticShortRange_force_accuracy_test(ElectrostaticShortRange_force_creator Elstatics_ShortRange_creator)
	{
	cout << "Short Range Electrostatic force starting" << endl;
	// Simple test to check the accuracy of the look up table
	shared_ptr<ParticleData> pdata_2(new ParticleData(2, BoxDim(1000.0), 1));
	ParticleDataArrays arrays = pdata_2->acquireReadWrite();
	arrays.x[0] = arrays.y[0] = arrays.z[0] = 0.0; arrays.charge[0]=1.0;
	// A positively charged particle is located at the origin
	arrays.x[1] = 1.0; arrays.y[1] = arrays.z[1] = 0.0; arrays.charge[1]=1.0;
	// Another positive charge is located at distance 1 in the x axis
	pdata_2->release();
	shared_ptr<NeighborList> nlist_2(new NeighborList(pdata_2, Scalar(3.0), Scalar(5.0)));
	// The cut-off is set to 3 while the buffer size is 5
	Scalar r_cut=3.0;
	Scalar alpha=1.0;
	Scalar delta=0.1;
	Scalar min_value=0.41;
	shared_ptr<ElectrostaticShortRange> fc_2=Elstatics_ShortRange_creator(pdata_2,nlist_2,r_cut,alpha,delta,min_value);
	// An ElectrostaticShortRange object with cut_off 3.0, alpha 1.0 and delta=0.05 is instantiated
	// now let us check how much the force differs from the exact calculation for 20 points;
	
	for(int j=0;j<10;j++){
    
	arrays.x[1]=min_value+sqrt(static_cast<double>(j));
	
	cout << "Particle axis at " << arrays.x[1] << endl;
	
	Scalar dx = arrays.x[1]-arrays.x[0];
	Scalar dy = arrays.y[1]-arrays.y[0];
	Scalar dz = arrays.z[1]-arrays.z[0];
	
	Scalar rsq = sqrt(dx*dx + dy*dy + dz*dz);
	Scalar al_rsq=alpha*rsq;
    
	Scalar fExactx=-dx*(EWALD_F*alpha*exp(-al_rsq*al_rsq)+boost::math::erfc(al_rsq)/rsq)/pow(rsq,2);
    Scalar fExacty=-dy*(EWALD_F*alpha*exp(-al_rsq*al_rsq)+boost::math::erfc(al_rsq)/rsq)/pow(rsq,2);
	Scalar fExactz=-dz*(EWALD_F*alpha*exp(-al_rsq*al_rsq)+boost::math::erfc(al_rsq)/rsq)/pow(rsq,2);
	Scalar fExactE=0.5*boost::math::erfc(al_rsq)/rsq;
	
	fc_2->compute(j);

	ForceDataArrays force_arrays=fc_2->acquire();

	
	
	cout<< "Force x p 1 " << force_arrays.fx[0] << " Exact value " << fExactx << endl;
	cout<< "Force y p 1 " << force_arrays.fy[0] << " Exact value " << fExacty << endl;
	cout<< "Force z p 1 " << force_arrays.fz[0] << " Exact value " << fExactz << endl;
    cout<< "Energy  p 1 " << force_arrays.pe[0] << " Exact value " << fExactE << endl;
	
	cout<< "Force x p 2 " << force_arrays.fx[1] << " Exact value " << -fExactx << endl;
	cout<< "Force y p 2 " << force_arrays.fy[1] << " Exact value " << -fExacty << endl;
	cout<< "Force z p 2 " << force_arrays.fz[1] << " Exact value " << -fExactz << endl;
	cout<< "Energy  p 2 " << force_arrays.pe[1] << " Exact value " << fExactE << endl;
	
						}
	}


//! ElectrostaticShortRange creator for unit tests
shared_ptr<ElectrostaticShortRange> base_class_ShortRangeElectrostatic_creator(shared_ptr<ParticleData> pdata, shared_ptr<NeighborList> nlist, Scalar r_cut, Scalar alpha, Scalar delta, Scalar min_value)
	{
	return shared_ptr<ElectrostaticShortRange>(new ElectrostaticShortRange(pdata, nlist, r_cut, alpha, delta,min_value));
	}
	
//! boost test case for particle test on CPU
BOOST_AUTO_TEST_CASE(ElectrostaticShortRange_force_accuracy)
	{
	ElectrostaticShortRange_force_creator ElectrostaticShortRange_creator_base = bind(base_class_ShortRangeElectrostatic_creator, _1, _2, _3, _4, _5,_6);
	ElectrostaticShortRange_force_accuracy_test(ElectrostaticShortRange_creator_base);
}


#undef EWALD_F  
