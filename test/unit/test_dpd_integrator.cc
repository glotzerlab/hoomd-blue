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

// $Id: test_lj_force.cc 3302 2010-08-04 20:15:44Z joaander $
// $URL: https://codeblue.umich.edu/hoomd-blue/svn/branches/dpd/test/unit/test_lj_force.cc $
// Maintainer: joaander

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <iostream>
#include <fstream>

#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include "ComputeThermo.h"
#include "AllPairPotentials.h"

#include "TwoStepNVE.h"
#ifdef ENABLE_CUDA
#include "TwoStepNVEGPU.h"
#endif

#include "BinnedNeighborList.h"
#include "Initializers.h"
#include "IntegratorTwoStep.h"


#include <math.h>

using namespace std;
using namespace boost;

/*! \file dpd_integrator_test.cc
    \brief Implements unit tests for PotentialPairDPDThermo
    \ingroup unit_tests
*/

//! Name the unit test module
#define BOOST_TEST_MODULE PotentialPairDPDThermo
#include "boost_utf_configure.h"



template <class PP_DPD>
void dpd_temperature_test(boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(1000, BoxDim(5.0), 1, 0, 0, 0, 0, exec_conf));   
    shared_ptr<ParticleData> pdata = sysdef->getParticleData(); 
    shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef, 0, pdata->getN()-1));
    shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_all));
    
    ParticleDataArrays arrays = pdata->acquireReadWrite();
    
    // setup a simple initial dense state
    for (int j = 0; j < 1000; j++)
        {
        arrays.x[j] = -2.0 + 0.3*(j %10);
        arrays.y[j] = -2.0 + 0.3*(j/10 %10);
        arrays.z[j] = -2.0 + 0.3*(j/100);
        arrays.vx[j] = 0.0;
        arrays.vy[j] = 0.0;
        arrays.vz[j] = 0.0;
        }
        
    pdata->release();
    
    Scalar deltaT = Scalar(0.02);
    Scalar Temp = Scalar(2.0);
    shared_ptr<VariantConst> T_variant(new VariantConst(Temp));    
    
    cout << endl << "Test 1" << endl;
    cout << "Creating an dpd gas of 1000 particles" << endl;
    cout << "Temperature set at " << Temp << endl;
    
    shared_ptr<TwoStepNVE> two_step_nve(new TwoStepNVE(sysdef,group_all));
    shared_ptr<ComputeThermo> thermo(new ComputeThermo(sysdef, group_all));
    thermo->setNDOF(3*1000);
    shared_ptr<IntegratorTwoStep> nve_up(new IntegratorTwoStep(sysdef, deltaT));
    nve_up->addIntegrationMethod(two_step_nve);
    
    
    // Construction of the Force Compute
    shared_ptr<NeighborList> nlist(new NeighborList(sysdef, Scalar(1.0), Scalar(0.8)));   
    nlist->setStorageMode(NeighborList::full);     
    shared_ptr<PotentialPairDPDThermoDPD> dpd_thermo(new PP_DPD(sysdef,nlist));
    dpd_thermo->setSeed(12345);
    dpd_thermo->setT(T_variant);
    dpd_thermo->setParams(0,0,make_scalar2(30,4.5));
    dpd_thermo->setRcut(0, 0, Scalar(1.0));
    nve_up->addForceCompute(dpd_thermo);


    Scalar(AvgT) = 0.0;
    for (unsigned int i = 0; i < 600; i++)
        {
        // Sample the Temperature
        if (i > 0 && i % 100 == 0)
            {
            thermo->compute(i);
            AvgT += thermo->getTemperature();
            //cout << "Temp " << thermo->getTemperature() << endl;
            
            }
            
        nve_up->update(i);
        }
    AvgT /= 5;         
    cout << "Average Temperature " << AvgT << endl;
    MY_BOOST_CHECK_CLOSE(AvgT, 2.0, 5);

   // Calculate Momentum
    Scalar(Mom_x) = 0;
    Scalar(Mom_y) = 0;
    Scalar(Mom_z) = 0;
    
    arrays = pdata->acquireReadWrite();
    
    // get momentum
    for (int j = 0; j < 1000; j++)
        {
        Mom_x += arrays.vx[j] ;
        Mom_y += arrays.vy[j];
        Mom_z += arrays.vz[j];
        }
        
    pdata->release();   
    
    MY_BOOST_CHECK_SMALL(Mom_x, 1e-3);
    MY_BOOST_CHECK_SMALL(Mom_y, 1e-3);
    MY_BOOST_CHECK_SMALL(Mom_z, 1e-3);
    
 
    
               
    }
    
BOOST_AUTO_TEST_CASE( DPD_Temp_Test )
	    {       
	   dpd_temperature_test< PotentialPairDPDThermo<EvaluatorPairDPDThermo> >(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
	    }    

#ifdef ENABLE_CUDA
BOOST_AUTO_TEST_CASE( DPD_GPU_Temp_Test )
	    {       
	   dpd_temperature_test< PotentialPairDPDThermoGPU<EvaluatorPairDPDThermo, gpu_compute_dpdthermodpd_forces > >(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
	    }   
#endif

#ifdef WIN32
#pragma warning( pop )
#endif

