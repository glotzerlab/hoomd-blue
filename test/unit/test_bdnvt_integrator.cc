/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

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

#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include "IntegratorTwoStep.h"
#include "TwoStepBDNVT.h"
#ifdef ENABLE_CUDA
#include "TwoStepBDNVTGPU.h"
#endif

#include "NeighborListBinned.h"
#include "Initializers.h"
#include "AllPairPotentials.h"

#include <math.h>

using namespace std;
using namespace boost;

/*! \file bdnvt_updater_test.cc
    \brief Implements unit tests for TwoStepBDNVT and descendants
    \ingroup unit_tests
*/

//! name the boost unit test module
#define BOOST_TEST_MODULE BD_NVTUpdaterTests
#include "boost_utf_configure.h"

//! Typedef'd NVEUpdator class factory
typedef boost::function<shared_ptr<TwoStepBDNVT> (shared_ptr<SystemDefinition> sysdef,
                                                  shared_ptr<ParticleGroup> group,
                                                  Scalar T,
                                                  unsigned int seed,
                                                  bool gamma_diam)> twostepbdnvt_creator;

//! Apply the Stochastic BD Bath to 1000 particles ideal gas
void bd_updater_tests(twostepbdnvt_creator bdnvt_creator, boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // check that a Brownian Dynamics integrator results in a correct diffusion coefficient
    // and correct average temperature.  Change the temperature and gamma and show this produces
    // a correct temperature and diffuction coefficent
    // Build a 1000 particle system with all the particles started at the origin, but with no interaction:
    //also put everything in a huge box so boundary conditions don't come into play
    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(1000, BoxDim(1000000.0), 4, 0, 0, 0, 0, exec_conf));
    shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef, 0, pdata->getN()-1));
    shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_all));

    {
    ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(pdata->getVelocities(), access_location::host, access_mode::readwrite);

    // setup a simple initial state
    for (int j = 0; j < 1000; j++)
        {
        h_pos.data[j].x = 0.0;
        h_pos.data[j].y = 0.0;
        h_pos.data[j].z = 0.0;
        h_vel.data[j].x = 0.0;
        h_vel.data[j].y = 0.0;
        h_vel.data[j].z = 0.0;
        }
        
    }
    Scalar deltaT = Scalar(0.01);
    Scalar Temp = Scalar(2.0);
    
    cout << endl << "Test 1" << endl;
    cout << "Creating an ideal gas of 1000 particles" << endl;
    cout << "Temperature set at " << Temp << endl;
    
    shared_ptr<TwoStepBDNVT> two_step_bdnvt = bdnvt_creator(sysdef, group_all, Temp, 123, 0);
    shared_ptr<IntegratorTwoStep> bdnvt_up(new IntegratorTwoStep(sysdef, deltaT));
    bdnvt_up->addIntegrationMethod(two_step_bdnvt);
    bdnvt_up->prepRun(0);
    
    int i;
    Scalar AvgT = Scalar(0);
    Scalar VarianceE = Scalar(0);
    Scalar KE;
    
    for (i = 0; i < 50000; i++)
        {
        if (i % 100 == 0)
            {
            ArrayHandle< Scalar4 > h_vel(pdata->getVelocities(), access_location::host, access_mode::read);
            KE = Scalar(0);
            for (int j = 0; j < 1000; j++) 
                KE += Scalar(0.5) * (h_vel.data[j].x*h_vel.data[j].x +h_vel.data[j].y*h_vel.data[j].y + h_vel.data[j].z*h_vel.data[j].z);
            // mass = 1.0;  k = 1.0;
            //VarianceE += pow((KE-Scalar(1.5)*1000*Temp),2);
            AvgT += Scalar(2.0)*KE/(3*1000);
            //cout << "Average Temperature at time step " << i << " is " << AvgT << endl;
            }
            
        bdnvt_up->update(i);
        }
    AvgT /= Scalar(50000.0/100.0);
    VarianceE /= Scalar(50000.0/100.0);

    {
    ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::read);
    Scalar MSD = Scalar(0);
    Scalar t = Scalar(i) * deltaT;
    
    for (int j = 0; j < 1000; j++) MSD += (h_pos.data[j].x*h_pos.data[j].x + h_pos.data[j].y*h_pos.data[j].y + h_pos.data[j].z*h_pos.data[j].z);
    Scalar D = MSD/(6*t*1000);
    
    cout << "Calculating Diffusion Coefficient " << D << endl;
    cout << "Average Temperature " << AvgT << endl;
    
    // Turning off Variance Check as requires too many calculations to converge for a unit test... 
    // but have demonstrated that it works
    //cout << "Energy Variance " << VarianceE <<  " " << 3.0/2.0*Temp*Temp*1000 << endl;
    
    //Dividing a very large number by a very large number... not great accuracy!
    MY_BOOST_CHECK_CLOSE(D, 2.0, 3);
    MY_BOOST_CHECK_CLOSE(AvgT, 2.0, 2.5);
    }
    
    // Resetting the Temperature to 1.0
    shared_ptr<VariantConst> T_variant(new VariantConst(1.0));
    cout << "Temperature set at " << T_variant->getValue(0) << endl;
    two_step_bdnvt->setT(T_variant);
    
    //Restoring the position of the particles to the origin for simplicity of calculating diffusion
    {
    ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);

    for (int j = 0; j < 1000; j++)
        {
        h_pos.data[j].x = 0.0;
        h_pos.data[j].y = 0.0;
        h_pos.data[j].z = 0.0;
        }
    }
    
    AvgT = Scalar(0);
    for (i = 0; i < 50000; i++)
        {
        if (i % 100 == 0)
            {
            ArrayHandle< Scalar4 > h_vel(pdata->getVelocities(), access_location::host, access_mode::read);
            KE = Scalar(0);
            for (int j = 0; j < 1000; j++) 
                KE += Scalar(0.5)*(h_vel.data[j].x*h_vel.data[j].x + h_vel.data[j].y*h_vel.data[j].y + h_vel.data[j].z*h_vel.data[j].z);
            // mass = 1.0;  k = 1.0;
            AvgT += Scalar(2.0)*KE/(3*1000);
            }
            
        bdnvt_up->update(i);
        }
    AvgT /= Scalar(50000.0/100.0);

    {
    ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::read);
    Scalar MSD = Scalar(0);
    Scalar t = Scalar(i) * deltaT;
    
    for (int j = 0; j < 1000; j++) MSD += (h_pos.data[j].x*h_pos.data[j].x + h_pos.data[j].y*h_pos.data[j].y + h_pos.data[j].z*h_pos.data[j].z);

    Scalar D = MSD/(6*t*1000);
    
    cout << "Calculating Diffusion Coefficient " << D << endl;
    cout << "Average Temperature " << AvgT << endl;
    
    //Dividing a very large number by a very large number... not great accuracy!
    MY_BOOST_CHECK_CLOSE(D, 1.0, 5);
    MY_BOOST_CHECK_CLOSE(AvgT, 1.0, 1);
    }
    
    // Setting Gamma to 0.5
    cout << "Gamma set at 0.5" << endl;
    two_step_bdnvt->setGamma(0, Scalar(0.5));
    
    //Restoring the position of the particles to the origin for simplicity of calculating diffusion
    {
    ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);

    for (int j = 0; j < 1000; j++)
        {
        h_pos.data[j].x = 0.0;
        h_pos.data[j].y = 0.0;
        h_pos.data[j].z = 0.0;
        }
    }
    
    AvgT = Scalar(0);
    for (i = 0; i < 50000; i++)
        {
        if (i % 100 == 0)
            {
            ArrayHandle< Scalar4 > h_vel(pdata->getVelocities(), access_location::host, access_mode::read);
            KE = Scalar(0);
            for (int j = 0; j < 1000; j++) 
                KE += Scalar(0.5)*(h_vel.data[j].x*h_vel.data[j].x + h_vel.data[j].y*h_vel.data[j].y + h_vel.data[j].z*h_vel.data[j].z);
            // mass = 1.0;  k = 1.0;
            AvgT += Scalar(2.0)*KE/(3*1000);
            }
        bdnvt_up->update(i);
        }
    AvgT /= Scalar(50000.0/100.0);

    {
    ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::read);
    Scalar MSD = Scalar(0);
    Scalar t = Scalar(i) * deltaT;
    
    for (int j = 0; j < 1000; j++) MSD += (h_pos.data[j].x*h_pos.data[j].x + h_pos.data[j].y*h_pos.data[j].y + h_pos.data[j].z*h_pos.data[j].z);
    Scalar D = MSD/(6*t*1000);
    
    cout << "Calculating Diffusion Coefficient " << D << endl;
    cout << "Average Temperature " << AvgT << endl;
    
    //Dividing a very large number by a very large number... not great accuracy!
    MY_BOOST_CHECK_CLOSE(D, 2.0, 5);
    MY_BOOST_CHECK_CLOSE(AvgT, 1.0, 1);
    }
    }

//! Apply the Stochastic BD Bath to 1000 particles ideal gas with gamma set by diameters
void bd_updater_diamtests(twostepbdnvt_creator bdnvt_creator, boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    
    cout << endl << "Test 2" << endl;
    cout << "Test setting diameter" << endl;

    // check that a Brownian Dynamics integrator results in a correct diffusion coefficient
    // and correct average temperature.  Change the temperature and diameters and show this produces
    // a correct temperature and diffuction coefficent
    // Build a 1000 particle system with all the particles started at the origin, but with no interaction:
    //also put everything in a huge box so boundary conditions don't come into play
    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(1000, BoxDim(1000000.0), 4, 0, 0, 0, 0, exec_conf));
    shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef, 0, pdata->getN()-1));
    shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_all));

    {
    ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(pdata->getVelocities(), access_location::host, access_mode::readwrite);
    
    // setup a simple initial state
    for (int j = 0; j < 1000; j++)
        {
        h_pos.data[j].x = 0.0;
        h_pos.data[j].y = 0.0;
        h_pos.data[j].z = 0.0;
        h_vel.data[j].x = 0.0;
        h_vel.data[j].y = 0.0;
        h_vel.data[j].z = 0.0;
        }
        
    }
    
    Scalar deltaT = Scalar(0.01);
    Scalar Temp = Scalar(1.0);
    cout << "Temperature set at " << Temp << endl;
    shared_ptr<TwoStepBDNVT> two_step_bdnvt = bdnvt_creator(sysdef, group_all, Temp, 123, 1);
    shared_ptr<IntegratorTwoStep> bdnvt_up(new IntegratorTwoStep(sysdef, deltaT));
    bdnvt_up->addIntegrationMethod(two_step_bdnvt);
    bdnvt_up->prepRun(0);
    
    Scalar AvgT = Scalar(0);
    int i;
    Scalar KE;
    for (i = 0; i < 50000; i++)
        {
        if (i % 100 == 0)
            {
            ArrayHandle< Scalar4 > h_vel(pdata->getVelocities(), access_location::host, access_mode::read);
            KE = Scalar(0);
            for (int j = 0; j < 1000; j++) 
                KE += Scalar(0.5) * (h_vel.data[j].x*h_vel.data[j].x +h_vel.data[j].y*h_vel.data[j].y + h_vel.data[j].z*h_vel.data[j].z);
            // mass = 1.0;  k = 1.0;
            AvgT += Scalar(2.0)*KE/(3*1000);
            }
            
        bdnvt_up->update(i);
        }
    AvgT /= Scalar(50000.0/100.0);

    {
    ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::read);
    Scalar MSD = Scalar(0);
    Scalar t = Scalar(i) * deltaT;
    
    for (int j = 0; j < 1000; j++) MSD += (h_pos.data[j].x*h_pos.data[j].x + h_pos.data[j].y*h_pos.data[j].y + h_pos.data[j].z*h_pos.data[j].z);
    Scalar D = MSD/(6*t*1000);
    
    cout << "Calculating Diffusion Coefficient " << D << endl;
    cout << "Average Temperature " << AvgT << endl;
    
    //Dividing a very large number by a very large number... not great accuracy!
    MY_BOOST_CHECK_CLOSE(D, 1.0, 5);
    MY_BOOST_CHECK_CLOSE(AvgT, 1.0, 1);
    }

    // Setting Diameters to 0.5
    cout << "Diameters set at 0.5" << endl;
    {
    ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_diameter(pdata->getDiameters(), access_location::host, access_mode::readwrite);
    for (int j = 0; j < 1000; j++)
        {
        h_diameter.data[j] = 0.5;
        }
    //Restoring the position of the particles to the origin for simplicity of calculating diffusion
    for (int j = 0; j < 1000; j++)
        {
        h_pos.data[j].x = 0.0;
        h_pos.data[j].y = 0.0;
        h_pos.data[j].z = 0.0;
        }
    }
    
    AvgT = Scalar(0);
    for (i = 0; i < 50000; i++)
        {
        if (i % 100 == 0)
            {
            ArrayHandle< Scalar4 > h_vel(pdata->getVelocities(), access_location::host, access_mode::read);
            KE = Scalar(0);
            for (int j = 0; j < 1000; j++) 
                KE += Scalar(0.5) * (h_vel.data[j].x*h_vel.data[j].x +h_vel.data[j].y*h_vel.data[j].y + h_vel.data[j].z*h_vel.data[j].z);
            // mass = 1.0;  k = 1.0;
            AvgT += Scalar(2.0)*KE/(3*1000);
            }
        bdnvt_up->update(i);
        }
    AvgT /= Scalar(50000.0/100.0);
    
    {
    ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::read);

    Scalar MSD = Scalar(0);
    Scalar t = Scalar(i) * deltaT;
    
    for (int j = 0; j < 1000; j++) MSD += (h_pos.data[j].x*h_pos.data[j].x + h_pos.data[j].y*h_pos.data[j].y + h_pos.data[j].z*h_pos.data[j].z);

    Scalar D = MSD/(6*t*1000);
    
    cout << "Calculating Diffusion Coefficient " << D << endl;
    cout << "Average Temperature " << AvgT << endl;
    
    //Dividing a very large number by a very large number... not great accuracy!
    MY_BOOST_CHECK_CLOSE(D, 2.0, 5);
    MY_BOOST_CHECK_CLOSE(AvgT, 1.0, 1);
    }
    }

//! Apply the Stochastic BD Bath to 1000 particles ideal gas
void bd_twoparticles_updater_tests(twostepbdnvt_creator bdnvt_creator, boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // check that a Brownian Dynamics integrator results in a correct diffusion coefficients
    // and correct average temperature when applied to a population of two different particle types
    // Build a 1000 particle system with all the particles started at the origin, but with no interaction:
    //also put everything in a huge box so boundary conditions don't come into play
    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(1000, BoxDim(1000000.0), 4, 0, 0, 0, 0, exec_conf));
    shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef, 0, pdata->getN()-1));
    shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_all));

    {
    ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(pdata->getVelocities(), access_location::host, access_mode::readwrite);
    
    // setup a simple initial state
    for (int j = 0; j < 1000; j++)
        {
        h_pos.data[j].x = 0.0;
        h_pos.data[j].y = 0.0;
        h_pos.data[j].z = 0.0;
        h_vel.data[j].x = 0.0;
        h_vel.data[j].y = 0.0;
        h_vel.data[j].z = 0.0;
        if (j < 500) h_pos.data[j].w = __int_as_scalar(0); //type = 0
        else h_pos.data[j].w = __int_as_scalar(1); // type = 1
        }
        
    }
    
    Scalar deltaT = Scalar(0.01);
    Scalar Temp = Scalar(1.0);
    
    
    cout << endl << "Test 3" << endl;
    cout << "Creating an ideal gas of 1000 particles" << endl;
    cout << "Temperature set at " << Temp << endl;
    
    shared_ptr<TwoStepBDNVT> two_step_bdnvt = bdnvt_creator(sysdef, group_all, Temp, 268, 0);
    shared_ptr<IntegratorTwoStep> bdnvt_up(new IntegratorTwoStep(sysdef, deltaT));
    bdnvt_up->addIntegrationMethod(two_step_bdnvt);
    bdnvt_up->prepRun(0);
    
    int i;
    Scalar AvgT = Scalar(0);
    Scalar KE;
    
    // Splitting the Particles in half and giving the two population different gammas..
    cout << "Two Particle Types: Gamma set at 1.0 and 2.0 respectively" << endl;
    two_step_bdnvt->setGamma(0, Scalar(1.0));
    two_step_bdnvt->setGamma(1, Scalar(2.0));
    
    AvgT = Scalar(0);
    for (i = 0; i < 50000; i++)
        {
        if (i % 100 == 0)
            {
            ArrayHandle< Scalar4 > h_vel(pdata->getVelocities(), access_location::host, access_mode::read);
            KE = Scalar(0);
            for (int j = 0; j < 1000; j++) 
                KE += Scalar(0.5) * (h_vel.data[j].x*h_vel.data[j].x +h_vel.data[j].y*h_vel.data[j].y + h_vel.data[j].z*h_vel.data[j].z);
            // mass = 1.0;  k = 1.0;
            AvgT += Scalar(2.0)*KE/(3*1000);
            }
        bdnvt_up->update(i);
        }
    AvgT /= Scalar(50000.0/100.0);

    {
    ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::read);

    Scalar MSD1 = Scalar(0);
    Scalar MSD2 = Scalar(0);
    Scalar t = Scalar(i) * deltaT;
    
    for (int j = 0; j < 500; j++) MSD1 += (h_pos.data[j].x*h_pos.data[j].x + h_pos.data[j].y*h_pos.data[j].y + h_pos.data[j].z*h_pos.data[j].z);
    Scalar D1 = MSD1/(6*t*500);
    for (int j = 500; j < 1000; j++) MSD2 += (h_pos.data[j].x*h_pos.data[j].x + h_pos.data[j].y*h_pos.data[j].y + h_pos.data[j].z*h_pos.data[j].z);
    Scalar D2 = MSD2/(6*t*500);
    
    cout << "Calculating Diffusion Coefficient 1 and 2 " << endl << D1 << endl << D2 << endl;
    cout << "Average Temperature " << AvgT << endl;
    
    //Dividing a very large number by a very large number... not great accuracy!
    MY_BOOST_CHECK_CLOSE(D1, 1.0, 8);
    MY_BOOST_CHECK_CLOSE(D2, 0.5, 5);
    MY_BOOST_CHECK_CLOSE(AvgT, 1.0, 1);
    }
    }

//! Apply the Stochastic BD Bath to 1000 LJ Particles
void bd_updater_lj_tests(twostepbdnvt_creator bdnvt_creator, boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // check that a stochastic force applied on top of NVE integrator for a 1000 LJ particles stilll produces the correct average temperature
    // Build a 1000 particle system with particles scattered on the x, y, and z axes.
    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(1000, BoxDim(1000000.0), 4, 0, 0, 0, 0, exec_conf));
    shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef, 0, pdata->getN()-1));
    shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_all));

    {
    ArrayHandle<Scalar4> h_pos(pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(pdata->getVelocities(), access_location::host, access_mode::readwrite);
    
    // setup a simple initial state
    for (int j = 0; j < 1000; j++)
        {
        h_pos.data[j].x = (j % 100)*pow(Scalar(-1.0),Scalar(j));
        h_pos.data[j].y = (j/100)*pow(Scalar(-1.0),Scalar(j));
        h_pos.data[j].z = pow(Scalar(-1.0),Scalar(j));
        h_vel.data[j].x = 0.0;
        h_vel.data[j].y = 0.0;
        h_vel.data[j].z = 0.0;
        }
        
    }
    
    Scalar deltaT = Scalar(0.01);
    Scalar Temp = Scalar(2.0);
    cout << endl << "Test 4" << endl;
    cout << "Creating 1000 LJ particles" << endl;
    cout << "Temperature set at " << Temp << endl;
    
    shared_ptr<TwoStepBDNVT> two_step_bdnvt = bdnvt_creator(sysdef, group_all, Temp, 358, 0);
    shared_ptr<IntegratorTwoStep> bdnvt_up(new IntegratorTwoStep(sysdef, deltaT));
    bdnvt_up->addIntegrationMethod(two_step_bdnvt);
    
    shared_ptr<NeighborList> nlist(new NeighborList(sysdef, Scalar(1.3), Scalar(3.0)));
    shared_ptr<PotentialPairLJ> fc3(new PotentialPairLJ(sysdef, nlist));
    fc3->setRcut(0, 0, Scalar(1.3));
    
    Scalar epsilon = Scalar(1.15);
    Scalar sigma = Scalar(1.0);
    Scalar alpha = Scalar(1.0);
    Scalar lj1 = Scalar(4.0) * epsilon * pow(sigma,Scalar(12.0));
    Scalar lj2 = alpha * Scalar(4.0) * epsilon * pow(sigma,Scalar(6.0));
    fc3->setParams(0,0,make_scalar2(lj1,lj2));
    bdnvt_up->addForceCompute(fc3);
    
    int i;
    Scalar AvgT = Scalar(0);
    //Scalar VarianceE = Scalar(0);
    Scalar KE;
    
    bdnvt_up->prepRun(0);
    
    for (i = 0; i < 50000; i++)
        {
        if (i % 10 == 0)
            {
            ArrayHandle< Scalar4 > h_vel(pdata->getVelocities(), access_location::host, access_mode::read);
            KE = Scalar(0);
            for (int j = 0; j < 1000; j++) 
                KE += Scalar(0.5) * (h_vel.data[j].x*h_vel.data[j].x +h_vel.data[j].y*h_vel.data[j].y + h_vel.data[j].z*h_vel.data[j].z);
            // mass = 1.0;  k = 1.0;
            AvgT += Scalar(2.0)*KE/(3*1000);
            }
            
        bdnvt_up->update(i);
        }
    AvgT /= Scalar(50000.0/10.0);
    
    cout << "Average Temperature " << AvgT << endl;
    
    MY_BOOST_CHECK_CLOSE(AvgT, 2.0, 1);
    
    // Resetting the Temperature to 1.0
    shared_ptr<VariantConst> T_variant(new VariantConst(1.0));
    cout << "Temperature set at " << T_variant->getValue(0) << endl;
    two_step_bdnvt->setT(T_variant);
    
    AvgT = Scalar(0);
    for (i = 0; i < 50000; i++)
        {
        if (i % 10 == 0)
            {
            ArrayHandle< Scalar4 > h_vel(pdata->getVelocities(), access_location::host, access_mode::read);
            KE = Scalar(0);
            for (int j = 0; j < 1000; j++) 
                KE += Scalar(0.5) * (h_vel.data[j].x*h_vel.data[j].x +h_vel.data[j].y*h_vel.data[j].y + h_vel.data[j].z*h_vel.data[j].z);
            // mass = 1.0;  k = 1.0;
            AvgT += Scalar(2.0)*KE/(3*1000);
            }
            
        bdnvt_up->update(i);
        }
    AvgT /= Scalar(50000.0/10.0);
    
    cout << "Average Temperature " << AvgT << endl;
    
    MY_BOOST_CHECK_CLOSE(AvgT, 1.0, 1);
    
    }


//! BD_NVTUpdater factory for the unit tests
shared_ptr<TwoStepBDNVT> base_class_bdnvt_creator(shared_ptr<SystemDefinition> sysdef,
                                                  shared_ptr<ParticleGroup> group,
                                                  Scalar Temp,
                                                  unsigned int seed,
                                                  bool use_diam)
    {
    shared_ptr<VariantConst> T_variant(new VariantConst(Temp));
    return shared_ptr<TwoStepBDNVT>(new TwoStepBDNVT(sysdef, group, T_variant, seed, use_diam));
    }

#ifdef ENABLE_CUDA
//! BD_NVTUpdaterGPU factory for the unit tests
shared_ptr<TwoStepBDNVT> gpu_bdnvt_creator(shared_ptr<SystemDefinition> sysdef,
                                            shared_ptr<ParticleGroup> group,
                                            Scalar Temp,
                                            unsigned int seed,
                                            bool use_diam)
    {
    shared_ptr<VariantConst> T_variant(new VariantConst(Temp));
    return shared_ptr<TwoStepBDNVT>(new TwoStepBDNVTGPU(sysdef, group, T_variant, seed, use_diam));
    }
#endif

//! Basic test for the base class
BOOST_AUTO_TEST_CASE( BDUpdater_tests )
    {
    twostepbdnvt_creator bdnvt_creator = bind(base_class_bdnvt_creator, _1, _2, _3, _4, _5);
    bd_updater_tests(bdnvt_creator, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

//! Diameter test for the base class
BOOST_AUTO_TEST_CASE( BDUpdater_diamtests )
    {
    twostepbdnvt_creator bdnvt_creator = bind(base_class_bdnvt_creator, _1, _2, _3, _4, _5);
    bd_updater_diamtests(bdnvt_creator, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

//! two particle test for the base class
BOOST_AUTO_TEST_CASE( BDUpdater_twoparticles_tests )
    {
    twostepbdnvt_creator bdnvt_creator = bind(base_class_bdnvt_creator, _1, _2, _3, _4, _5);
    bd_twoparticles_updater_tests(bdnvt_creator, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

//! extended LJ-liquid test for the base class
BOOST_AUTO_TEST_CASE( BDUpdater_LJ_tests )
    {
    twostepbdnvt_creator bdnvt_creator = bind(base_class_bdnvt_creator, _1, _2, _3, _4, _5);
    bd_updater_lj_tests(bdnvt_creator, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

#ifdef ENABLE_CUDA
//! Basic test for the GPU class
BOOST_AUTO_TEST_CASE( BDUpdaterGPU_tests )
    {
    twostepbdnvt_creator bdnvt_creator_gpu = bind(gpu_bdnvt_creator, _1, _2, _3, _4, _5);
    bd_updater_tests(bdnvt_creator_gpu, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! Diameter Setting test for the GPU class
BOOST_AUTO_TEST_CASE( BDUpdaterGPU_diamtests )
    {
    twostepbdnvt_creator bdnvt_creator_gpu = bind(gpu_bdnvt_creator, _1, _2, _3, _4, _5);
    bd_updater_diamtests(bdnvt_creator_gpu, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! two particle test for the GPU class
BOOST_AUTO_TEST_CASE( BDUpdaterGPU_twoparticles_tests )
    {
    twostepbdnvt_creator bdnvt_creator_gpu = bind(gpu_bdnvt_creator, _1, _2, _3, _4, _5);
    bd_twoparticles_updater_tests(bdnvt_creator_gpu, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! extended LJ-liquid test for the GPU class
BOOST_AUTO_TEST_CASE( BDUpdaterGPU_LJ_tests )
    {
    twostepbdnvt_creator bdnvt_creator_gpu = bind(gpu_bdnvt_creator, _1, _2, _3, _4, _5);
    bd_updater_lj_tests(bdnvt_creator_gpu, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
#endif


#ifdef WIN32
#pragma warning( pop )
#endif

