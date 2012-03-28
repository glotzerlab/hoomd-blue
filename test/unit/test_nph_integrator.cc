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

#include "ComputeThermo.h"
#include "TwoStepNPH.h"
#ifdef ENABLE_CUDA
#include "TwoStepNPHGPU.h"
#include "ComputeThermoGPU.h"
#endif
#include "IntegratorTwoStep.h"

#include "NeighborListBinned.h"
#include "Initializers.h"
#include "AllPairPotentials.h"

#include "saruprng.h"

#include <math.h>

using namespace std;
using namespace boost;

/*! \file nph_updater_test.cc
    \brief Implements unit tests for NPHpdater and descendants
    \ingroup unit_tests
*/

//! name the boost unit test module
#define BOOST_TEST_MODULE TwoStepNPHTests
#include "boost_utf_configure.h"

//! Typedef'd NPHUpdator class factory
typedef boost::function<shared_ptr<TwoStepNPH> (shared_ptr<SystemDefinition> sysdef,
                                                shared_ptr<ParticleGroup> group,
                                                Scalar W,
                                                Scalar P,
                                                TwoStepNPH::integrationMode mode) > twostepnph_creator;

//! Helper function to get gaussian random numbers
Scalar inline gaussianRand(Saru& saru, Scalar sigma);

//! Basic functionality test of a generic TwoStepNPH
void nph_updater_test(twostepnph_creator nph_creator, boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    const unsigned int N = 1000;
    Scalar P = 1.0;
    Scalar T0 = 1.0;
    Scalar deltaT = 0.001;

    TwoStepNPH::integrationMode mode = TwoStepNPH::cubic;
    Scalar W = .001;

/* the anisotropic integration modes work, too, but since the test particle system is isotropic,
   it does not constitute a very good test case for the anisotropic fluctuations */

//   W = 1.0;
//   mode = TwoStepNPH::orthorhombic;

//   W = 1.0;
//   mode = TwoStepNPH::tetragonal;

    // create two identical random particle systems to simulate
    RandomInitializer rand_init(N, Scalar(0.2), Scalar(0.9), "A");
    rand_init.setSeed(12345);
    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(rand_init, exec_conf));
    shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    // give the particles velocities according to a Maxwell-Boltzmann distribution
    Saru saru(54321);

    // total up the system momentum
    Scalar3 total_momentum = make_scalar3(0.0, 0.0, 0.0);
    unsigned int nparticles= pdata->getN();

    // generate the gaussian velocity distribution
    for (unsigned int idx = 0; idx < nparticles; idx++)
    {
        // generate gaussian velocities
        Scalar mass = pdata->getMass(idx);
        Scalar sigma = T0 / mass;
        Scalar vx = gaussianRand(saru, sigma);
        Scalar vy = gaussianRand(saru, sigma);
        Scalar vz = gaussianRand(saru, sigma);

        // total up the system momentum
        total_momentum.x += vx * mass;
        total_momentum.y += vy * mass;
        total_momentum.z += vz * mass;

        // assign the velocities
        pdata->setVelocity(idx,make_scalar3(vx,vy,vz));
    }

    // loop through the particles again and remove the system momentum
    total_momentum.x /= nparticles;
    total_momentum.y /= nparticles;
    total_momentum.z /= nparticles;
    {
    ArrayHandle<Scalar4> h_vel(pdata->getVelocities(), access_location::host, access_mode::readwrite);
    for (unsigned int idx = 0; idx < nparticles; idx++)
    {
        Scalar mass = h_vel.data[idx].w;
        h_vel.data[idx].x -= total_momentum.x / mass;
        h_vel.data[idx].y -= total_momentum.y / mass;
        h_vel.data[idx].z -= total_momentum.z / mass;
    }

    }


    // enable the energy computation
    PDataFlags flags;
    flags[pdata_flag::pressure_tensor] = 1;
    flags[pdata_flag::isotropic_virial] = 1;
    // only for output of enthalpy
    flags[pdata_flag::potential_energy] = 1;
    pdata->setFlags(flags);

    shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef, 0, pdata->getN()-1));
    shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_all));

    shared_ptr<NeighborListBinned> nlist(new NeighborListBinned(sysdef, Scalar(2.5), Scalar(0.8)));

    shared_ptr<PotentialPairLJ> fc(new PotentialPairLJ(sysdef, nlist));
    fc->setRcut(0, 0, Scalar(pow(Scalar(2.0),Scalar(1./6.))));

    // setup some values for alpha and sigma
    Scalar epsilon = Scalar(1.0);
    Scalar sigma = Scalar(1.0);
    Scalar alpha = Scalar(1.0);
    Scalar lj1 = Scalar(4.0) * epsilon * pow(sigma,Scalar(12.0));
    Scalar lj2 = alpha * Scalar(4.0) * epsilon * pow(sigma,Scalar(6.0));

    // specify the force parameters
    fc->setParams(0,0,make_scalar2(lj1,lj2));
    // If we want accurate calculation of potential energy, we need to apply the
    // energy shift
    fc->setShiftMode(PotentialPairLJ::shift);

    shared_ptr<TwoStepNPH> two_step_nph = nph_creator(sysdef, group_all, W,P, mode);
    shared_ptr<IntegratorTwoStep> nph(new IntegratorTwoStep(sysdef, Scalar(deltaT)));
    nph->addIntegrationMethod(two_step_nph);
    nph->addForceCompute(fc);
    nph->prepRun(0);

    // step for a 10,000 timesteps to relax pessure and tempreratue
    // before computing averages

    for (int i = 0; i < 10000; i++)
        {
        nph->update(i);
        }

    shared_ptr<ComputeThermo> compute_thermo(new ComputeThermo(sysdef, group_all, "name"));
    compute_thermo->setNDOF(3*N-3);

    // now do the averaging for next 100k steps
    Scalar avrPxx = 0.0;
    Scalar avrPyy = 0.0;
    Scalar avrPzz = 0.0;
    int count = 0;

    compute_thermo->compute(0);
    BoxDim box = pdata->getBox();
    Scalar3 L = box.getL();
    Scalar volume = L.x*L.y*L.z;
    Scalar enthalpy =  compute_thermo->getKineticEnergy() + compute_thermo->getPotentialEnergy() + P * volume;
    Scalar barostat_energy = nph->getLogValue("nph_barostat_energy", 0);
    Scalar H_ref = enthalpy + barostat_energy; // the conserved quantity

    for (int i = 10001; i < 50000; i++)
        {
        if (i % 100 == 0)
            {
            compute_thermo->compute(i);
            PressureTensor P_current = compute_thermo->getPressureTensor();
            avrPxx += P_current.xx;
            avrPyy += P_current.yy;
            avrPzz += P_current.zz;

            count++;
            }
        nph->update(i);
        }

    compute_thermo->compute(count+1);
    box = pdata->getBox();
    L = box.getL();
    volume = L.x*L.y*L.z;
    enthalpy =  compute_thermo->getKineticEnergy() + compute_thermo->getPotentialEnergy() + P * volume;
    barostat_energy = nph->getLogValue("nph_barostat_energy", count+1);
    Scalar H_final = enthalpy + barostat_energy;
    // check conserved quantity
    Scalar tol = 0.01;
    MY_BOOST_CHECK_CLOSE(H_ref,H_final,tol);

    avrPxx /= Scalar(count);
    avrPyy /= Scalar(count);
    avrPzz /= Scalar(count);
    Scalar avrP= Scalar(1./3.)*(avrPxx+avrPyy+avrPzz);
    Scalar rough_tol = 2.0;
    MY_BOOST_CHECK_CLOSE(P, avrP, rough_tol);

    }

//! Helper function to get gaussian random numbers
Scalar inline gaussianRand(Saru& saru, Scalar sigma)
{
    Scalar x1 = saru.d();
    Scalar x2 = saru.d();
    Scalar z = sqrt(-2.0 * log(x1)) * cos(2 * M_PI * x2);
    z = z * sigma;
    return z;
}

//! NPHUpdater factory for the unit tests
shared_ptr<TwoStepNPH> base_class_nph_creator(shared_ptr<SystemDefinition> sysdef,
                                              shared_ptr<ParticleGroup> group,
                                              Scalar W,
                                              Scalar P,
                                              TwoStepNPH::integrationMode mode)
    {
    boost::shared_ptr<Variant> P_variant(new VariantConst(P));
    boost::shared_ptr<ComputeThermo> thermo(new ComputeThermo(sysdef, group));
    // for the tests, we can assume that group is the all group
    return shared_ptr<TwoStepNPH>(new TwoStepNPH(sysdef, group, thermo, W, P_variant,mode,""));
    }

#ifdef ENABLE_CUDA
//! NPHIntegratorGPU factory for the unit tests
shared_ptr<TwoStepNPH> gpu_nph_creator(shared_ptr<SystemDefinition> sysdef,
                                              shared_ptr<ParticleGroup> group,
                                              Scalar W,
                                              Scalar P,
                                              TwoStepNPH::integrationMode mode)
    {
    boost::shared_ptr<Variant> P_variant(new VariantConst(P));
    boost::shared_ptr<ComputeThermo> thermo(new ComputeThermoGPU(sysdef, group));
    // for the tests, we can assume that group is the all group
    return shared_ptr<TwoStepNPH>(new TwoStepNPHGPU(sysdef, group, thermo, W, P_variant,mode,""));
    }
#endif

//! boost test case for base class integration tests
BOOST_AUTO_TEST_CASE( TwoStepNPH_tests )
    {
    twostepnph_creator nph_creator = bind(base_class_nph_creator, _1, _2,_3,_4,_5);
    nph_updater_test(nph_creator, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

#ifdef ENABLE_CUDA

//! boost test case for GPU integration tests
BOOST_AUTO_TEST_CASE( TwoStepNPHGPU_tests )
    {
    twostepnph_creator nph_creator = bind(gpu_nph_creator, _1, _2,_3,_4,_5);
    nph_updater_test(nph_creator, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

#endif
#ifdef WIN32
#pragma warning( pop )
#endif

