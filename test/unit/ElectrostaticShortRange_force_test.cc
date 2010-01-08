/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <iostream>

// conditionally compile in only if boost is 1.35 or later
#include <boost/version.hpp>
#if (BOOST_VERSION >= 103500)

#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/math/special_functions/erf.hpp>

#include "ElectrostaticShortRange.h"

#include "BinnedNeighborList.h"
#include "Initializers.h"

#include <math.h>

//! This number is just 2/sqrt(Pi)
#define EWALD_F  1.128379167
using namespace std;
using namespace boost;

/*! \file ElectrostaticShortRange_force_test.cc
    \brief Implements unit tests for ElectrostaticShortRange and descendants
    \ingroup unit_tests
*/

//! Name the unit test module
#define BOOST_TEST_MODULE ElectrostaticShortRangeTests
#include "boost_utf_configure.h"
//! minimum force worth computing
const Scalar MIN_force=Scalar(1.0e-9);

//! Typedef'd ElectrostaticShortRange factory
typedef boost::function<shared_ptr<ElectrostaticShortRange> (shared_ptr<SystemDefinition> sysdef,
                                                             shared_ptr<NeighborList> nlist,
                                                             Scalar r_cut,
                                                             Scalar alpha,
                                                             Scalar delta,
                                                             Scalar min_value)> ElectrostaticShortRange_force_creator;

//! Test the ability of the Short Range Electrostatic force compute to actually calculate forces
void ElectrostaticShortRange_force_accuracy_test(ElectrostaticShortRange_force_creator Elstatics_ShortRange_creator,
                                                 ExecutionConfiguration exec_conf)
    {
    cout << "Testing the accuracy of the look up table in ElectrostaticShortRange" << endl;
    // Simple test to check the accuracy of the look up table
    shared_ptr<SystemDefinition> sysdef_2(new SystemDefinition(2, BoxDim(1000.0), 1, 0, 0, 0, 0, exec_conf));
    shared_ptr<ParticleData> pdata_2 = sysdef_2->getParticleData();
    ParticleDataArrays arrays = pdata_2->acquireReadWrite();
    arrays.x[0] = arrays.y[0] = arrays.z[0] = 0.0; arrays.charge[0]=1.0;
    // A positively charged particle is located at the origin
    arrays.x[1] = 1.0; arrays.y[1] = arrays.z[1] = 0.0; arrays.charge[1]=1.0;
    // Another positive charge is located at distance 1 in the x axis
    pdata_2->release();
    shared_ptr<NeighborList> nlist_2(new NeighborList(sysdef_2, Scalar(3.0), Scalar(5.0)));
    // The cut-off is set to 3 while the buffer size is 5
    Scalar r_cut=Scalar(3.0);
    
    Scalar delta=Scalar(0.1);
    Scalar min_value=Scalar(0.41);
    
    for (int k1=0; k1<6; k1++)
        {
        
        Scalar alpha=Scalar(0.1+k1);
        //Test different values of alpha as well
        shared_ptr<ElectrostaticShortRange> fc_2=Elstatics_ShortRange_creator(sysdef_2,nlist_2,r_cut,alpha,delta,min_value);
        // An ElectrostaticShortRange object with specified value of cut_off, alpha, delta and min_value is instantiated
        // now let us check how much the force differs from the exact calculation for N_p**3 points within the cut_off;
        
        int N_p=7;
        Scalar delta_test= (r_cut-min_value)/(sqrt(r_cut)*N_p);
        int j_count=0;
        
        for (int jx=0; jx<N_p; jx++)
            {
            for (int jy=0; jy<N_p; jy++)
                {
                for (int jz=0; jz<N_p; jz++)
                    {
                    
                    arrays.x[1]=min_value+Scalar((static_cast<double>(jx))*delta_test);
                    arrays.y[1]=min_value+Scalar((static_cast<double>(jy))*delta_test);
                    arrays.z[1]=min_value+Scalar((static_cast<double>(jz))*delta_test);
                    
                    Scalar dx = arrays.x[1]-arrays.x[0];
                    Scalar dy = arrays.y[1]-arrays.y[0];
                    Scalar dz = arrays.z[1]-arrays.z[0];
                    
                    Scalar rsq = sqrt(dx*dx + dy*dy + dz*dz);
                    Scalar al_rsq=alpha*rsq;
                    
                    Scalar erfc_al=boost::math::erfc(al_rsq);
                    
                    Scalar fExactx=-dx*(Scalar(EWALD_F)*alpha*exp(-al_rsq*al_rsq)+erfc_al/rsq)/pow(rsq,2);
                    Scalar fExacty=-dy*(Scalar(EWALD_F)*alpha*exp(-al_rsq*al_rsq)+erfc_al/rsq)/pow(rsq,2);
                    Scalar fExactz=-dz*(Scalar(EWALD_F)*alpha*exp(-al_rsq*al_rsq)+erfc_al/rsq)/pow(rsq,2);
                    Scalar fExactE=Scalar(0.5)*erfc_al/rsq;
                    
                    fc_2->compute(j_count);
                    j_count++;
                    
                    ForceDataArrays force_arrays=fc_2->acquire();
                    
                    if (fabs(force_arrays.fx[0])>MIN_force)
                        {
                        MY_BOOST_CHECK_CLOSE(force_arrays.fx[0],fExactx,tol);
                        MY_BOOST_CHECK_CLOSE(force_arrays.fx[1],-fExactx,tol);
                        }
                    if (fabs(force_arrays.fy[0])>MIN_force)
                        {
                        MY_BOOST_CHECK_CLOSE(force_arrays.fy[0],fExacty,tol);
                        MY_BOOST_CHECK_CLOSE(force_arrays.fy[1],-fExacty,tol);
                        }
                    if (fabs(force_arrays.fz[0])>MIN_force)
                        {
                        MY_BOOST_CHECK_CLOSE(force_arrays.fz[0],fExactz,tol);
                        MY_BOOST_CHECK_CLOSE(force_arrays.fz[1],-fExactz,tol);
                        }
                    if (fabs(force_arrays.pe[0])>MIN_force)
                        {
                        MY_BOOST_CHECK_CLOSE(force_arrays.pe[0],fExactE,tol);
                        MY_BOOST_CHECK_CLOSE(force_arrays.pe[1],fExactE,tol);
                        }
                    }
                }
            }
        }
    }

//! Tests periodic boundary conditions
void ElectrostaticShortRange_periodic_test(ElectrostaticShortRange_force_creator Elstatics_ShortRange_creator,
                                           ExecutionConfiguration exec_conf)
    {
    cout << "Testing periodic conditions in the calculation of ElectrostaticShortRange" << endl;
    // Here we are going to place particles next to the boundary of the box and see that
    // periodic boudnary conditions work as expected
    // simuilar test as in lj_force_test
    
    shared_ptr<SystemDefinition> sysdef_6(new SystemDefinition(6,BoxDim(20.0,40.0,60.0),1, 0, 0, 0, 0, exec_conf));
    shared_ptr<ParticleData> pdata_6 = sysdef_6->getParticleData();
    ParticleDataArrays arrays=pdata_6->acquireReadWrite();
    
    arrays.x[0]=Scalar(-9.6); arrays.y[0]=Scalar(0.0); arrays.z[0]=Scalar(0.0); arrays.charge[0]=1.0;
    arrays.x[1]=Scalar(9.6); arrays.y[1]=Scalar(0.0); arrays.z[1]=Scalar(0.0); arrays.charge[1]=1.0;
    arrays.x[2]=Scalar(0.0); arrays.y[2]=Scalar(-19.5); arrays.z[2]=Scalar(0.0); arrays.charge[2]=1.0;
    arrays.x[3]=Scalar(0.0); arrays.y[3]=Scalar(19.5); arrays.z[3]=Scalar(0.0); arrays.charge[3]=1.0;
    arrays.x[4]=Scalar(0.0); arrays.y[4]=Scalar(0.0); arrays.z[4]=Scalar(-29.4); arrays.charge[4]=1.0;
    arrays.x[5]=Scalar(0.0); arrays.y[5]=Scalar(0.0); arrays.z[5]=Scalar(29.4); arrays.charge[5]=1.0;
    
    pdata_6->release();
    
    Scalar r_cut=Scalar(3.0);
    Scalar alpha=Scalar(1.0);
    
    shared_ptr<NeighborList> nlist_6(new NeighborList(sysdef_6,r_cut,Scalar(5.0)));
    shared_ptr<ElectrostaticShortRange> fc_6=Elstatics_ShortRange_creator(sysdef_6,nlist_6,r_cut,alpha,Scalar(0.2),Scalar(0.3));
    
    fc_6->compute(0);
    ForceDataArrays force_arrays=fc_6->acquire();
    
    //particle 0 is repelled from x=-10, hence it is pulled right
    MY_BOOST_CHECK_CLOSE(force_arrays.fx[0],1.146699475,1e-3);
    MY_BOOST_CHECK_SMALL(force_arrays.fy[0],1e-5);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[0],1e-5);
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[0],0.16118689,1e-3);
    
    //particle 1 is repelled from x=10, hence it is pulled left
    
    MY_BOOST_CHECK_CLOSE(force_arrays.fx[1],-1.14669475,1e-3);
    MY_BOOST_CHECK_SMALL(force_arrays.fy[1],1e-5);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[1],1e-5);
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[1],0.16118689,1e-3);
    
    //particle 2 is repelled from y=-20, so it is pulled up
    
    MY_BOOST_CHECK_SMALL(force_arrays.fx[2],1e-5);
    MY_BOOST_CHECK_CLOSE(force_arrays.fy[2],0.5724067,1e-3);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[2],1e-5);
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[2],0.078649603,1e-3);
    
    //particle 3 is repelled from y=20, so it is pulled down
    
    MY_BOOST_CHECK_SMALL(force_arrays.fx[3],1e-5);
    MY_BOOST_CHECK_CLOSE(force_arrays.fy[3],-0.5724067,1e-3);
    MY_BOOST_CHECK_SMALL(force_arrays.fz[3],1e-5);
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[3],0.078649603,1e-3);
    
    //particle 4 is repelled from z=-30, so it is pulled up
    
    MY_BOOST_CHECK_SMALL(force_arrays.fx[4],1e-5);
    MY_BOOST_CHECK_SMALL(force_arrays.fy[4],1e-5);
    MY_BOOST_CHECK_CLOSE(force_arrays.fz[4],0.2850689,1e-3);
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[4],0.037369176,1e-3);
    
    //particle 5 is repelled from z=30, so it is pulled down
    
    MY_BOOST_CHECK_SMALL(force_arrays.fx[5],1e-5);
    MY_BOOST_CHECK_SMALL(force_arrays.fy[5],1e-5);
    MY_BOOST_CHECK_CLOSE(force_arrays.fz[5],-0.2850689,1e-3);
    MY_BOOST_CHECK_CLOSE(force_arrays.pe[5],0.037369176,1e-3);
    }

//! ElectrostaticShortRange creator for unit tests
shared_ptr<ElectrostaticShortRange>
base_class_ShortRangeElectrostatic_creator(shared_ptr<SystemDefinition> sysdef,
                                           shared_ptr<NeighborList> nlist,
                                           Scalar r_cut,
                                           Scalar alpha,
                                           Scalar delta,
                                           Scalar min_value)
    {
    return shared_ptr<ElectrostaticShortRange>(new ElectrostaticShortRange(sysdef, nlist, r_cut, alpha, delta,min_value));
    }

//! boost test case for particle test on CPU
BOOST_AUTO_TEST_CASE(ElectrostaticShortRange_force_accuracy)
    {
    ElectrostaticShortRange_force_creator ElectrostaticShortRange_creator_base = 
        bind(base_class_ShortRangeElectrostatic_creator, _1, _2, _3, _4, _5,_6);
    ElectrostaticShortRange_force_accuracy_test(ElectrostaticShortRange_creator_base,
                                                ExecutionConfiguration(ExecutionConfiguration::CPU));
    }

//! boost test periodic boundary conditions
BOOST_AUTO_TEST_CASE(ElectrostaticShortRange_force_periodic)
    {
    ElectrostaticShortRange_force_creator ElectrostaticShortRange_creator_base = 
        bind(base_class_ShortRangeElectrostatic_creator, _1, _2, _3, _4, _5,_6);
    ElectrostaticShortRange_periodic_test(ElectrostaticShortRange_creator_base,
                                          ExecutionConfiguration(ExecutionConfiguration::CPU));
    }



#undef EWALD_F
#else
// We can't have the unit test passing if the code wasn't even compiled!
BOOST_AUTO_TEST_CASE(dummy_test)
    {
    BOOST_FAIL("ElectrostaticShortRange not compiled");
    }

#endif

#ifdef WIN32
#pragma warning( pop )
#endif

