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

// $Id$
// $URL$

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <iostream>

//! name the boost unit test module
#define BOOST_TEST_MODULE NVERigidUpdaterTests
#include "boost_utf_configure.h"

#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include "TwoStepNVERigid.h"
#ifdef ENABLE_CUDA
#include "TwoStepNVERigidGPU.h"
#endif

#include "IntegratorTwoStep.h"

#include "BinnedNeighborList.h"
#include "Initializers.h"
#include "LJForceCompute.h"

#ifdef ENABLE_CUDA
#include "BinnedNeighborListGPU.h"
#include "LJForceComputeGPU.h"
#endif

#include "saruprng.h"
#include <math.h>
#include <time.h>

using namespace std;
using namespace boost;

/*! \file nve_rigid_updater_test.cc
    \brief Implements unit tests for NVERigidUpdater
    \ingroup unit_tests
*/


//! Tolerance for floating point comparisons
#ifdef SINGLE_PRECISION
const Scalar tol = Scalar(1e-2);
#else
const Scalar tol = 1e-3;
#endif

//! Typedef'd TwoStepNVERigid class factory
typedef boost::function<shared_ptr<TwoStepNVERigid> (shared_ptr<SystemDefinition> sysdef,
                                                    shared_ptr<ParticleGroup> group)> nveup_creator;

void nve_updater_integrate_tests(nveup_creator nve_creator, const ExecutionConfiguration &exec_conf)
    {
#ifdef ENABLE_CUDA
    g_gpu_error_checking = true;
#endif
    
    // check that the nve updater can actually integrate particle positions and velocities correctly
    // start with a 2 particle system to keep things simple: also put everything in a huge box so boundary conditions
    // don't come into play
    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(10, BoxDim(1000.0), 1, 0, 0, 0, 0, exec_conf));
    shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef, 0, pdata->getN()-1));
    shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_all));
    
    ParticleDataArrays arrays = pdata->acquireReadWrite();
    
    // setup a simple initial state
    arrays.x[0] = Scalar(-1.0); arrays.y[0] = 0.0; arrays.z[0] = 0.0;
    arrays.vx[0] = Scalar(-0.5); arrays.body[0] = 0;
    arrays.x[1] =  Scalar(-1.0); arrays.y[1] = 1.0; arrays.z[1] = 0.0;
    arrays.vx[1] = Scalar(0.2); arrays.body[1] = 0;
    arrays.x[2] = Scalar(-1.0); arrays.y[2] = 2.0; arrays.z[2] = 0.0;
    arrays.vy[2] = Scalar(-0.1); arrays.body[2] = 0;
    arrays.x[3] = Scalar(-1.0); arrays.y[3] = 3.0; arrays.z[3] = 0.0;
    arrays.vy[3] = Scalar(0.3);  arrays.body[3] = 0;
    arrays.x[4] = Scalar(-1.0); arrays.y[4] = 4.0; arrays.z[4] = 0.0;
    arrays.vz[4] = Scalar(-0.2); arrays.body[4] = 0;
    
    arrays.x[5] = 0.0; arrays.y[5] = Scalar(0.0); arrays.z[5] = 0.0;
    arrays.vx[5] = Scalar(0.2); arrays.body[5] = 1;
    arrays.x[6] = 0.0; arrays.y[6] = Scalar(1.0); arrays.z[6] = 0.0;
    arrays.vy[6] = Scalar(0.8); arrays.body[6] = 1;
    arrays.x[7] = 0.0; arrays.y[7] = Scalar(2.0); arrays.z[7] = 0.0;
    arrays.vy[7] = Scalar(-0.6); arrays.body[7] = 1;
    arrays.x[8] = 0.0; arrays.y[8] = Scalar(3.0); arrays.z[8] = 0.0;
    arrays.vz[8] = Scalar(0.7); arrays.body[8] = 1;
    arrays.x[9] = 0.0; arrays.y[9] = Scalar(4.0); arrays.z[9] = 0.0;
    arrays.vy[9] = Scalar(-0.5); arrays.body[9] = 1;
    
    pdata->release();
    
    Scalar deltaT = Scalar(0.001);
    shared_ptr<TwoStepNVERigid> two_step_nve = nve_creator(sysdef, group_all);
    shared_ptr<IntegratorTwoStep> nve_up(new IntegratorTwoStep(sysdef, deltaT));
    nve_up->addIntegrationMethod(two_step_nve);
    
    shared_ptr<NeighborList> nlist(new NeighborList(sysdef, Scalar(3.0), Scalar(0.8)));
    shared_ptr<LJForceCompute> fc(new LJForceCompute(sysdef, nlist, Scalar(3.0)));
    
    // setup some values for alpha and sigma
    Scalar epsilon = Scalar(1.0);
    Scalar sigma = Scalar(1.0);
    Scalar alpha = Scalar(1.0);
    Scalar lj1 = Scalar(4.0) * epsilon * pow(sigma, Scalar(12.0));
    Scalar lj2 = alpha * Scalar(4.0) * epsilon * pow(sigma, Scalar(6.0));
    
    // specify the force parameters
    fc->setParams(0,0,lj1,lj2);
    
    nve_up->addForceCompute(fc);
    
    unsigned int steps = 1000;
    unsigned int sampling = 100;
    
    sysdef->init();
    
    shared_ptr<RigidData> rdata = sysdef->getRigidData();
    unsigned int nbodies = rdata->getNumBodies();
    cout << "Number of particles = " << arrays.nparticles << "; Number of rigid bodies = " << nbodies << "\n";
    
    for (unsigned int i = 0; i < steps; i++)
        {
        if (i % sampling == 0) cout << "Step " << i << "\n";
        
        nve_up->update(i);
        }
        
    ArrayHandle<Scalar4> com_handle(rdata->getCOM(), access_location::host, access_mode::read);
    cout << "Rigid body center of mass:\n";
    for (unsigned int i = 0; i < nbodies; i++)
        cout << i << "\t " << com_handle.data[i].x << "\t" << com_handle.data[i].y << "\t" << com_handle.data[i].z << "\n";
        
    // Output coordinates
    arrays = pdata->acquireReadWrite();
    
    FILE *fp = fopen("test_integrate.xyz", "w");
    BoxDim box = pdata->getBox();
    Scalar Lx = box.xhi - box.xlo;
    Scalar Ly = box.yhi - box.ylo;
    Scalar Lz = box.zhi - box.zlo;
    fprintf(fp, "%d\n%f\t%f\t%f\n", arrays.nparticles, Lx, Ly, Lz);
    for (unsigned int i = 0; i < arrays.nparticles; i++)
        fprintf(fp, "N\t%f\t%f\t%f\n", arrays.x[i], arrays.y[i], arrays.z[i]);
        
    fclose(fp);
    pdata->release();
    
    }


void nve_updater_energy_tests(nveup_creator nve_creator, const ExecutionConfiguration& exec_conf)
    {
#ifdef ENABLE_CUDA
    g_gpu_error_checking = true;
#endif
    
    // check that the nve updater can actually integrate particle positions and velocities correctly
    // start with a 2 particle system to keep things simple: also put everything in a huge box so boundary conditions
    // don't come into play
    unsigned int nbodies = 1000; // 12000, 24000, 36000, 48000, 60000
    unsigned int nparticlesperbody = 5;
    unsigned int N = nbodies * nparticlesperbody;
    Scalar box_length = 80.0; // 80, 90, 100, 120, 150
    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(N, BoxDim(box_length), 1, 0, 0, 0, 0, exec_conf));
    shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef, 0, pdata->getN()-1));
    shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_all));

    BoxDim box = pdata->getBox();
    
    // setup a simple initial state
    unsigned int ibody = 0;
    unsigned int iparticle = 0;
    Scalar x0 = box.xlo + 0.01;
    Scalar y0 = box.ylo + 0.01;
    Scalar z0 = box.zlo + 0.01;
    Scalar xspacing = 6.0f;
    Scalar yspacing = 2.0f;
    Scalar zspacing = 2.0f;
    
    unsigned int seed = 10483;
    boost::shared_ptr<Saru> random = boost::shared_ptr<Saru>(new Saru(seed));
    Scalar temperature = 1.0;
    Scalar KE = Scalar(0.0);
    
    ParticleDataArrays arrays = pdata->acquireReadWrite();
    
    // initialize bodies in a cubic lattice with some velocity profile
    for (unsigned int i = 0; i < nbodies; i++)
        {
        for (unsigned int j = 0; j < nparticlesperbody; j++)
            {
            
            arrays.x[iparticle] = x0 + 1.0 * j;
            arrays.y[iparticle] = y0 + 0.0;
            arrays.z[iparticle] = z0 + 0.0;
            
            /*
            if (j == 0)
            {
                arrays.x[iparticle] = x0 + 0.0;
                arrays.y[iparticle] = y0 + 0.0;
                arrays.z[iparticle] = z0 + 0.0;
            }
            else if (j == 1)
            {
                arrays.x[iparticle] = x0 + 1.0;
                                arrays.y[iparticle] = y0 + 0.0;
                                arrays.z[iparticle] = z0 + 0.0;
            
            }
            else if (j == 2)
            {
                arrays.x[iparticle] = x0 + 2.0;
                                arrays.y[iparticle] = y0 + 0.0;
                                arrays.z[iparticle] = z0 + 0.0;
            
            }
            else if (j == 3)
            {
                arrays.x[iparticle] = x0 + 2.76;
                                arrays.y[iparticle] = y0 + 0.642;
                                arrays.z[iparticle] = z0 + 0.0;
            
            }
            else if (j == 4)
            {
                arrays.x[iparticle] = x0 + 3.52;
                                arrays.y[iparticle] = y0 + 1.284;
                                arrays.z[iparticle] = z0 + 0.0;
            
            }
            */
            
            /*
            if (j == 0)
                {
                arrays.x[iparticle] = x0 + 0.577;
                arrays.y[iparticle] = y0 + 0.577;
                arrays.z[iparticle] = z0 + 0.577;
                }
            else if (j == 1)
                {
                arrays.x[iparticle] = x0 + 1.154;
                arrays.y[iparticle] = y0 + 1.154;
                arrays.z[iparticle] = z0 + 1.154;
                
                }
            else if (j == 2)
                {
                arrays.x[iparticle] = x0 + 0.0;
                arrays.y[iparticle] = y0 + 0.0;
                arrays.z[iparticle] = z0 + 1.154;
                
                }
            else if (j == 3)
                {
                arrays.x[iparticle] = x0 + 0.0;
                arrays.y[iparticle] = y0 + 1.154;
                arrays.z[iparticle] = z0 + 0.0;
                
                }
            else if (j == 4)
                {
                arrays.x[iparticle] = x0 + 1.154;
                arrays.y[iparticle] = y0 + 0.0;
                arrays.z[iparticle] = z0 + 0.0;
                }
            */    
                
            arrays.vx[iparticle] = random->d();
            arrays.vy[iparticle] = random->d();
            arrays.vz[iparticle] = random->d();
            
            KE += Scalar(0.5) * (arrays.vx[iparticle]*arrays.vx[iparticle] + arrays.vy[iparticle]*arrays.vy[iparticle] + arrays.vz[iparticle]*arrays.vz[iparticle]);
            
            arrays.body[iparticle] = ibody;
            
            iparticle++;
            }
            
        x0 += xspacing;
        if (x0 + xspacing >= box.xhi)
            {
            x0 = box.xlo + 0.01;
            
            y0 += yspacing;
            if (y0 + yspacing >= box.yhi)
                {
                y0 = box.ylo + 0.01;
                
                z0 += zspacing;
                if (z0 + zspacing >= box.zhi)
                    z0 = box.zlo + 0.01;
                }
            }
            
        ibody++;
        }
        
    assert(iparticle == N);
    
    FILE *fp = fopen("initial.xyz", "w");
    Scalar Lx = box.xhi - box.xlo;
    Scalar Ly = box.yhi - box.ylo;
    Scalar Lz = box.zhi - box.zlo;
    fprintf(fp, "%d\n%f\t%f\t%f\n", arrays.nparticles, Lx, Ly, Lz);
    for (unsigned int i = 0; i < arrays.nparticles; i++)
        fprintf(fp, "N\t%f\t%f\t%f\n", arrays.x[i], arrays.y[i], arrays.z[i]);
    fclose(fp);
    
    pdata->release();
    
    Scalar deltaT = Scalar(0.001);
    shared_ptr<TwoStepNVERigid> two_step_nve = nve_creator(sysdef, group_all);
    shared_ptr<IntegratorTwoStep> nve_up(new IntegratorTwoStep(sysdef, deltaT));
    nve_up->addIntegrationMethod(two_step_nve);
    
    shared_ptr<BinnedNeighborListGPU> nlist(new BinnedNeighborListGPU(sysdef, Scalar(2.5), Scalar(0.3)));
    shared_ptr<LJForceComputeGPU> fc(new LJForceComputeGPU(sysdef, nlist, Scalar(2.5)));
    
    // setup some values for alpha and sigma
    Scalar epsilon = Scalar(1.0);
    Scalar sigma = Scalar(1.0);
    Scalar alpha = Scalar(1.0);
    Scalar lj1 = Scalar(4.0) * epsilon * pow(sigma, Scalar(12.0));
    Scalar lj2 = alpha * Scalar(4.0) * epsilon * pow(sigma, Scalar(6.0));
    
    // specify the force parameters
    fc->setParams(0,0,lj1,lj2);
    
    nve_up->addForceCompute(fc);
    
    sysdef->init();
    
    Scalar PE;
    unsigned int steps = 10000;
    unsigned int sampling = 10000;
    
    shared_ptr<RigidData> rdata = sysdef->getRigidData();
    unsigned int nrigid_dof = rdata->getNumDOF();
    
    // Rescale particle velocities to desired temperature:
    Scalar current_temp = 2.0 * KE / nrigid_dof;
    Scalar factor = sqrt(temperature / current_temp);
    
    arrays = pdata->acquireReadWrite();
    for (unsigned int j = 0; j < N; j++)
        {
        arrays.vx[j] *= factor;
        arrays.vy[j] *= factor;
        arrays.vz[j] *= factor;
        }
        
    pdata->release();
    
    cout << "Number of particles = " << N << "; Number of rigid bodies = " << rdata->getNumBodies() << "\n";
    cout << "Step\tTemp\tPotEng\tKinEng\tTotalE\n";
    
    clock_t start = clock();
    
    for (unsigned int i = 0; i <= steps; i++)
        {
        
        nve_up->update(i);
        
        if (i % sampling == 0)
            {
            arrays = pdata->acquireReadWrite();
            KE = Scalar(0.0);
            for (unsigned int j = 0; j < N; j++)
                KE += Scalar(0.5) * (arrays.vx[j]*arrays.vx[j] + arrays.vy[j]*arrays.vy[j] + arrays.vz[j]*arrays.vz[j]);
            PE = fc->calcEnergySum();
            
            current_temp = 2.0 * KE / nrigid_dof;
            printf("%8d\t%12.8g\t%12.8g\t%12.8g\t%12.8g\n", i, current_temp, PE / N, KE / N, (PE + KE) / N);
            
            pdata->release();
            }
        }
        
    clock_t end = clock();
    double elapsed = (double)(end - start) / (double)CLOCKS_PER_SEC;
    printf("Elapased time: %f sec or %f TPS\n", elapsed, (double)steps / elapsed);
    
    // Output coordinates
    arrays = pdata->acquireReadWrite();
    
    fp = fopen("test_energy.xyz", "w");
    Lx = box.xhi - box.xlo;
    Ly = box.yhi - box.ylo;
    Lz = box.zhi - box.zlo;
    fprintf(fp, "%d\n%f\t%f\t%f\n", arrays.nparticles, Lx, Ly, Lz);
    for (unsigned int i = 0; i < arrays.nparticles; i++)
        fprintf(fp, "N\t%f\t%f\t%f\n", arrays.x[i], arrays.y[i], arrays.z[i]);
        
    fclose(fp);
    pdata->release();
    
    }

//! TwoStepNVERigid factory for the unit tests
shared_ptr<TwoStepNVERigid> base_class_nve_creator(shared_ptr<SystemDefinition> sysdef, shared_ptr<ParticleGroup> group)
    {
    return shared_ptr<TwoStepNVERigid>(new TwoStepNVERigid(sysdef, group));
    }

#ifdef ENABLE_CUDA
//! TwoStepNVERigidGPU factory for the unit tests
shared_ptr<TwoStepNVERigid> gpu_nve_creator(shared_ptr<SystemDefinition> sysdef, shared_ptr<ParticleGroup> group)
    {
    return shared_ptr<TwoStepNVERigid>(new TwoStepNVERigidGPU(sysdef, group));
    }
#endif

//! boost test case for base class integration tests
/*
BOOST_AUTO_TEST_CASE( NVERigidUpdater_energy_tests )
{
    printf("\nTesting energy conservation on CPU...\n");
    nveup_creator nve_creator = bind(base_class_nve_creator, _1, _2);
    nve_updater_energy_tests(nve_creator, ExecutionConfiguration(ExecutionConfiguration::CPU));
}
*/
#ifdef ENABLE_CUDA

//! boost test case for base class integration tests
BOOST_AUTO_TEST_CASE( NVERigidUpdaterGPU_energy_tests )
    {
    printf("\nTesting energy conservation on GPU...\n");
    nveup_creator nve_creator_gpu = bind(gpu_nve_creator, _1, _2);
    nve_updater_energy_tests(nve_creator_gpu, ExecutionConfiguration(ExecutionConfiguration::GPU));
    }

#endif

#ifdef WIN32
#pragma warning( pop )
#endif

