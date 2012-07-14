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
#include "TwoStepNPTMTK.h"
#ifdef ENABLE_CUDA
#include "TwoStepNPTMTKGPU.h"
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

/*! \file test_npt_mtk_integrator.cc
    \brief Implements unit tests for NPTMTKpdater and descendants
    \ingroup unit_tests
*/

//! name the boost unit test module
#define BOOST_TEST_MODULE TwoStepNPTMTKTests
#include "boost_utf_configure.h"

//! Typedef'd NPTMTKUpdator class factory
typedef boost::function<shared_ptr<TwoStepNPTMTK> (shared_ptr<SystemDefinition> sysdef,
                                                shared_ptr<ParticleGroup> group,
                                                boost::shared_ptr<ComputeThermo> thermo_group,
                                                Scalar tau,
                                                Scalar tauP,
                                                Scalar T,
                                                Scalar P,
                                                TwoStepNPTMTK::integrationMode mode) > twostep_npt_mtk_creator;

//! Basic functionality test of a generic TwoStepNPTMTK
void npt_mtk_updater_test(twostep_npt_mtk_creator npt_mtk_creator, boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    const unsigned int N = 10000;
    Scalar P = 1.0;
    Scalar T0 = 2.0;
    Scalar deltaT = 0.001;

    TwoStepNPTMTK::integrationMode mode = TwoStepNPTMTK::cubic;
    Scalar tau = .1;
    Scalar tauP = .1;

/* the anisotropic integration modes work, too, but since the test particle system is isotropic,
   it has a degenerate box shape and thus does not constitute a good test case for these
   integration modes */

//   mode = TwoStepNPTMTK::orthorhombic;
//   mode = TwoStepNPTMTK::tetragonal;

    // create two identical random particle systems to simulate
    RandomInitializer rand_init(N, Scalar(0.2), Scalar(0.9), "A");
    rand_init.setSeed(12345);
    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(rand_init, exec_conf));
    shared_ptr<ParticleData> pdata = sysdef->getParticleData();

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

    shared_ptr<ComputeThermo> thermo_group(new ComputeThermo(sysdef, group_all, "name"));
    thermo_group->setNDOF(3*N);

    shared_ptr<TwoStepNPTMTK> two_step_npt_mtk = npt_mtk_creator(sysdef, group_all, thermo_group, tau, tauP,T0,P, mode);
    shared_ptr<IntegratorTwoStep> npt_mtk(new IntegratorTwoStep(sysdef, Scalar(deltaT)));
    npt_mtk->addIntegrationMethod(two_step_npt_mtk);
    npt_mtk->addForceCompute(fc);
    npt_mtk->prepRun(0);

    // step for a 10,000 timesteps to relax pessure and tempreratue
    // before computing averages

    std::cout << "Equilibrating 10,000 steps... " << std::endl;
    for (int i = 0; i < 10000; i++)
        {
        if (i % 1000 == 0)
            std::cout << i << std::endl;
        npt_mtk->update(i);
        }

    // now do the averaging for next 100k steps
    Scalar avrPxx = 0.0;
    Scalar avrPyy = 0.0;
    Scalar avrPzz = 0.0;
    Scalar avrT = 0.0;
    int count = 0;

    thermo_group->compute(0);
    BoxDim box = pdata->getBox();
    Scalar3 L = box.getL();
    Scalar volume = L.x*L.y*L.z;
    Scalar enthalpy =  thermo_group->getKineticEnergy() + thermo_group->getPotentialEnergy() + P * volume;
    Scalar barostat_energy = npt_mtk->getLogValue("npt_mtk_barostat_energy", 0);
    Scalar thermostat_energy = npt_mtk->getLogValue("npt_mtk_thermostat_energy", 0);
    Scalar H_ref = enthalpy + barostat_energy + thermostat_energy; // the conserved quantity

    std::cout << "Measuring up to 50,000 steps... " << std::endl;
    for (int i = 10000; i < 50000; i++)
        {
        if (i % 1000 == 0)
            std::cout << i << std::endl;
        if (i % 100 == 0)
            {
            thermo_group->compute(i);
            PressureTensor P_current = thermo_group->getPressureTensor();
            avrPxx += P_current.xx;
            avrPyy += P_current.yy;
            avrPzz += P_current.zz;

            avrT += thermo_group->getTemperature();
            count++;

            //box = pdata->getBox();
            //L = box.getL();
            //std::cout << "L == (" << L.x << ", " << L.y << ", " << L.z << ")" << std::endl;
            }
        npt_mtk->update(i);
        }

    thermo_group->compute(count+1);
    box = pdata->getBox();
    L = box.getL();
    volume = L.x*L.y*L.z;
    enthalpy =  thermo_group->getKineticEnergy() + thermo_group->getPotentialEnergy() + P * volume;
    barostat_energy = npt_mtk->getLogValue("npt_mtk_barostat_energy", count+1);
    thermostat_energy = npt_mtk->getLogValue("npt_mtk_thermostat_energy", count+1);
    Scalar H_final = enthalpy + barostat_energy + thermostat_energy;
    // check conserved quantity
    Scalar tol = 0.02;
    MY_BOOST_CHECK_CLOSE(H_ref,H_final,tol);

    avrPxx /= Scalar(count);
    avrPyy /= Scalar(count);
    avrPzz /= Scalar(count);
    avrT /= Scalar(count);
    Scalar avrP= Scalar(1./3.)*(avrPxx+avrPyy+avrPzz);
    Scalar rough_tol = 2.0;
    MY_BOOST_CHECK_CLOSE(P, avrP, rough_tol);

    MY_BOOST_CHECK_CLOSE(T0, avrT, rough_tol);

    }

//! IntegratorTwoStepNPTMTK factory for the unit tests
shared_ptr<TwoStepNPTMTK> base_class_npt_mtk_creator(shared_ptr<SystemDefinition> sysdef,
                                              shared_ptr<ParticleGroup> group,
                                              boost::shared_ptr<ComputeThermo> thermo_group,
                                              Scalar tau,
                                              Scalar tauP,
                                              Scalar T,
                                              Scalar P,
                                              TwoStepNPTMTK::integrationMode mode)
    {
    boost::shared_ptr<Variant> P_variant(new VariantConst(P));
    boost::shared_ptr<Variant> T_variant(new VariantConst(T));
    // for the tests, we can assume that group is the all group
    return shared_ptr<TwoStepNPTMTK>(new TwoStepNPTMTK(sysdef, group, thermo_group, tau, tauP,T_variant, P_variant,mode));
    }

#ifdef ENABLE_CUDA
//! NPTMTKIntegratorGPU factory for the unit tests
shared_ptr<TwoStepNPTMTK> gpu_npt_mtk_creator(shared_ptr<SystemDefinition> sysdef,
                                              shared_ptr<ParticleGroup> group,
                                              boost::shared_ptr<ComputeThermo> thermo_group,
                                              Scalar tau,
                                              Scalar tauP,
                                              Scalar T,
                                              Scalar P,
                                              TwoStepNPTMTK::integrationMode mode)
    {
    boost::shared_ptr<Variant> P_variant(new VariantConst(P));
    boost::shared_ptr<Variant> T_variant(new VariantConst(T));
    // for the tests, we can assume that group is the all group
    return shared_ptr<TwoStepNPTMTK>(new TwoStepNPTMTKGPU(sysdef, group, thermo_group, tau, tauP, T_variant, P_variant,mode));
    }
#endif

//! boost test case for base class integration tests
BOOST_AUTO_TEST_CASE( TwoStepNPTMTK_tests )
    {
    twostep_npt_mtk_creator npt_mtk_creator = bind(base_class_npt_mtk_creator, _1, _2,_3,_4,_5, _6, _7, _8);
    npt_mtk_updater_test(npt_mtk_creator, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

#ifdef ENABLE_CUDA

//! boost test case for GPU integration tests
BOOST_AUTO_TEST_CASE( TwoStepNPTMTKGPU_tests )
    {
    twostep_npt_mtk_creator npt_mtk_creator = bind(gpu_npt_mtk_creator, _1, _2,_3,_4,_5,_6, _7, _8);
    npt_mtk_updater_test(npt_mtk_creator, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

#endif

#ifdef WIN32
#pragma warning( pop )
#endif
