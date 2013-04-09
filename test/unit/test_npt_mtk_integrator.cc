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

#include "CellList.h"
#include "NeighborList.h"
#include "NeighborListBinned.h"
#include "Initializers.h"
#include "AllPairPotentials.h"

#ifdef ENABLE_CUDA
#include "NeighborListGPUBinned.h"
#include "CellListGPU.h"
#endif

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
                                                TwoStepNPTMTK::couplingMode mode,
                                                unsigned int flags) > twostep_npt_mtk_creator;

//! Basic functionality test of a generic TwoStepNPTMTK
void npt_mtk_updater_test(twostep_npt_mtk_creator npt_mtk_creator, boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // we use a tightly packed cubic LJ crystal for testing,
    // because this one has a sufficient shear elasticity
    // to avoid that the box gets too tilted during triclinic NPT
    const unsigned int L = 20; // number of particles along one box edge
    Scalar P = 142.5; // use a REALLY high value of pressure to keep the system in solid state
    Scalar T0 = .9;
    Scalar deltaT = 0.001;

    const TwoStepNPTMTK::couplingMode coupling_modes[] = {TwoStepNPTMTK::couple_xyz,
                                             TwoStepNPTMTK::couple_none,
                                             TwoStepNPTMTK::couple_yz,
                                             TwoStepNPTMTK::couple_none};
    const unsigned int orthorhombic = TwoStepNPTMTK::baro_x | TwoStepNPTMTK::baro_y |TwoStepNPTMTK::baro_z;
    const unsigned int all = orthorhombic | TwoStepNPTMTK::baro_xy | TwoStepNPTMTK::baro_xz |TwoStepNPTMTK::baro_yz;

    const unsigned int npt_flags[] = {orthorhombic, orthorhombic, orthorhombic, all};
    const std::string mode_name[] = {"cubic", "orthorhombic", "tetragonal", "triclinic"};
    unsigned int n_modes = 4;
 
    Scalar tau = .1;
    Scalar tauP = .1;


    // create two identical random particle systems to simulate
    SimpleCubicInitializer cubic_init(L, Scalar(0.89), "A");
    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(cubic_init, exec_conf));
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

    shared_ptr<NeighborList> nlist;
    shared_ptr<PotentialPairLJ> fc;
    shared_ptr<CellList> cl;
    #ifdef ENABLE_CUDA
    if (exec_conf->isCUDAEnabled())
        {
        cl = shared_ptr<CellList>( new CellListGPU(sysdef) );
        nlist = shared_ptr<NeighborList>( new NeighborListGPUBinned(sysdef, Scalar(2.5), Scalar(0.8),cl));
        fc = shared_ptr<PotentialPairLJ>( new PotentialPairLJGPU(sysdef, nlist));
        }
    else
    #endif
        {
        cl = shared_ptr<CellList>( new CellList(sysdef) );
        nlist = shared_ptr<NeighborList>(new NeighborListBinned(sysdef, Scalar(2.5), Scalar(0.8),cl));
        fc = shared_ptr<PotentialPairLJ>( new PotentialPairLJ(sysdef, nlist));
        }

    fc->setRcut(0, 0, Scalar(2.5));

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
    thermo_group->setNDOF(3*pdata->getN());
    shared_ptr<IntegratorTwoStep> npt_mtk(new IntegratorTwoStep(sysdef, Scalar(deltaT)));
    npt_mtk->addForceCompute(fc);

    // successively integrate the system using different methods
    unsigned int offs=0;
    for (unsigned int i_mode = 0; i_mode < n_modes; i_mode++)
        {
        TwoStepNPTMTK::couplingMode mode = coupling_modes[i_mode];
        unsigned int flags = npt_flags[i_mode];
        std::cout << "Testing NPT with mode " << mode_name[i_mode] << std::endl;
        shared_ptr<TwoStepNPTMTK> two_step_npt_mtk = npt_mtk_creator(sysdef, group_all, thermo_group, tau, tauP,T0,P, mode, flags);
        npt_mtk->removeAllIntegrationMethods();
        npt_mtk->addIntegrationMethod(two_step_npt_mtk);
        npt_mtk->prepRun(0);

        // step for a 10,000 timesteps to relax pessure and tempreratue
        // before computing averages

        std::cout << "Equilibrating 10,000 steps... " << std::endl;
        unsigned int timestep;
        for (unsigned int i = 0; i < 10000; i++)
            {
            timestep = offs + i;
            if (i % 1000 == 0)
                std::cout << i << std::endl;
            npt_mtk->update(timestep);
            }

        // now do the averaging for next 40k steps
        Scalar avrPxx(0.0);
        Scalar avrPxy(0.0);
        Scalar avrPxz(0.0);
        Scalar avrPyy(0.0);
        Scalar avrPyz(0.0);
        Scalar avrPzz(0.0);
        Scalar avrT(0.0);
        int count = 0;

        thermo_group->compute(0);
        BoxDim box = pdata->getBox();
        Scalar volume = box.getVolume();
        Scalar enthalpy =  thermo_group->getKineticEnergy() + thermo_group->getPotentialEnergy() + P * volume;
        Scalar barostat_energy = npt_mtk->getLogValue("npt_barostat_energy", 0);
        Scalar thermostat_energy = npt_mtk->getLogValue("npt_thermostat_energy", 0);
        Scalar H_ref = enthalpy + barostat_energy + thermostat_energy; // the conserved quantity

        std::cout << "Measuring up to 50,000 steps... " << std::endl;
        for (unsigned int i = 10000; i < 50000; i++)
            {
            timestep = offs + i;
            if (i % 1000 == 0)
                std::cout << i << std::endl;
            if (i% 100 == 0)
                {
                thermo_group->compute(timestep);
                PressureTensor P_current = thermo_group->getPressureTensor();
                avrPxx += P_current.xx;
                avrPxy += P_current.xy;
                avrPxz += P_current.xz;
                avrPyy += P_current.yy;
                avrPyz += P_current.yz;
                avrPzz += P_current.zz;

                avrT += thermo_group->getTemperature();
                count++;

                /*
                box = pdata->getBox();
                Scalar3 L = box.getL();
                std::cout << "L == (" << L.x << ", " << L.y << ", " << L.z << ")" << std::endl;
                */
                }
            npt_mtk->update(timestep);
            }

        thermo_group->compute(timestep+1);
        box = pdata->getBox();
        volume = box.getVolume();
        enthalpy =  thermo_group->getKineticEnergy() + thermo_group->getPotentialEnergy() + P * volume;
        barostat_energy = npt_mtk->getLogValue("npt_barostat_energy", count+1);
        thermostat_energy = npt_mtk->getLogValue("npt_thermostat_energy", count+1);
        Scalar H_final = enthalpy + barostat_energy + thermostat_energy;

        // check conserved quantity, required accuracy 2*10^-4
        Scalar tol = 0.02;
        MY_BOOST_CHECK_CLOSE(H_ref,H_final,tol);

        avrPxx /= Scalar(count);
        avrPxy /= Scalar(count);
        avrPxz /= Scalar(count);
        avrPyy /= Scalar(count);
        avrPyz /= Scalar(count);
        avrPzz /= Scalar(count);
        avrT /= Scalar(count);
        Scalar avrP= Scalar(1./3.)*(avrPxx+avrPyy+avrPzz);
        Scalar rough_tol = 2.0;
        if (i_mode == 0) // cubic
            MY_BOOST_CHECK_CLOSE(avrP, P, rough_tol);
        else if (i_mode == 1) // orthorhombic
            {
            MY_BOOST_CHECK_CLOSE(avrPxx, P, rough_tol);
            MY_BOOST_CHECK_CLOSE(avrPyy, P, rough_tol);
            MY_BOOST_CHECK_CLOSE(avrPzz, P, rough_tol);
            }
        else if (i_mode == 2) // tetragonal
            {
            MY_BOOST_CHECK_CLOSE(avrPxx, P, rough_tol);
            MY_BOOST_CHECK_CLOSE(Scalar(1.0/2.0)*(avrPyy+avrPzz), avrP, rough_tol);
            }
       else if (mode == 3) // triclinic
            {
            MY_BOOST_CHECK_CLOSE(avrPxx, P, rough_tol);
            MY_BOOST_CHECK_CLOSE(avrPyy, P, rough_tol);
            MY_BOOST_CHECK_CLOSE(avrPzz, P, rough_tol);
            MY_BOOST_CHECK_SMALL(avrPxy,rough_tol);
            MY_BOOST_CHECK_SMALL(avrPxz,rough_tol);
            MY_BOOST_CHECK_SMALL(avrPyz,rough_tol);
            }
        MY_BOOST_CHECK_CLOSE(T0, avrT, rough_tol);
        offs+=timestep;
        }
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

//! Test ability to integrate in the NPH ensemble
void nph_integration_test(twostep_npt_mtk_creator nph_creator, boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    const unsigned int N = 1000;
    Scalar P = 1.0;
    Scalar T0 = 1.0;
    Scalar deltaT = 0.001;

    Scalar tauP(1.0);

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

    shared_ptr<NeighborList> nlist;
    shared_ptr<PotentialPairLJ> fc;
    shared_ptr<CellList> cl;
    #ifdef ENABLE_CUDA
    if (exec_conf->isCUDAEnabled())
        {
        cl = shared_ptr<CellList>( new CellListGPU(sysdef) );
        nlist = shared_ptr<NeighborList>( new NeighborListGPUBinned(sysdef, Scalar(2.5), Scalar(0.8),cl));
        fc = shared_ptr<PotentialPairLJ>( new PotentialPairLJGPU(sysdef, nlist));
        }
    else
    #endif
        {
        cl = shared_ptr<CellList>( new CellList(sysdef) );
        nlist = shared_ptr<NeighborList>(new NeighborListBinned(sysdef, Scalar(2.5), Scalar(0.8),cl));
        fc = shared_ptr<PotentialPairLJ>( new PotentialPairLJ(sysdef, nlist));
        }

 
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

    shared_ptr<ComputeThermo> compute_thermo(new ComputeThermo(sysdef, group_all, "name"));
    compute_thermo->setNDOF(3*N-3);


    // set up integration without thermostat
    shared_ptr<TwoStepNPTMTK> two_step_npt = nph_creator(sysdef,
                                                         group_all,
                                                         compute_thermo,
                                                         Scalar(1.0),
                                                         tauP,
                                                         Scalar(1.0),
                                                         P,
                                                         TwoStepNPTMTK::couple_xyz,
                                                         TwoStepNPTMTK::baro_x | TwoStepNPTMTK::baro_y | TwoStepNPTMTK::baro_z);
    shared_ptr<IntegratorTwoStep> nph(new IntegratorTwoStep(sysdef, Scalar(deltaT)));
    nph->addIntegrationMethod(two_step_npt);
    nph->addForceCompute(fc);
    nph->prepRun(0);

    // step for a 10,000 timesteps to relax pessure and tempreratue
    // before computing averages

    for (int i = 0; i < 10000; i++)
        {
        nph->update(i);
        }

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
    Scalar barostat_energy = nph->getLogValue("npt_barostat_energy", 0);
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
    barostat_energy = nph->getLogValue("npt_barostat_energy", count+1);
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

//! IntegratorTwoStepNPTMTK factory for the unit tests
shared_ptr<TwoStepNPTMTK> base_class_npt_mtk_creator(shared_ptr<SystemDefinition> sysdef,
                                              shared_ptr<ParticleGroup> group,
                                              boost::shared_ptr<ComputeThermo> thermo_group,
                                              Scalar tau,
                                              Scalar tauP,
                                              Scalar T,
                                              Scalar P,
                                              TwoStepNPTMTK::couplingMode mode,
                                              unsigned int flags)
    {
    boost::shared_ptr<Variant> P_variant(new VariantConst(P));
    boost::shared_ptr<Variant> T_variant(new VariantConst(T));
    // for the tests, we can assume that group is the all group
    return shared_ptr<TwoStepNPTMTK>(new TwoStepNPTMTK(sysdef, group, thermo_group, tau, tauP,T_variant, P_variant,mode,flags,false));
    }

shared_ptr<TwoStepNPTMTK> base_class_nph_creator(shared_ptr<SystemDefinition> sysdef,
                                              shared_ptr<ParticleGroup> group,
                                              boost::shared_ptr<ComputeThermo> thermo_group,
                                              Scalar tau,
                                              Scalar tauP,
                                              Scalar T,
                                              Scalar P,
                                              TwoStepNPTMTK::couplingMode mode,
                                              unsigned int flags)
    {
    boost::shared_ptr<Variant> P_variant(new VariantConst(P));
    boost::shared_ptr<Variant> T_variant(new VariantConst(T));
    // for the tests, we can assume that group is the all group
    return shared_ptr<TwoStepNPTMTK>(new TwoStepNPTMTK(sysdef, group, thermo_group, tau, tauP,T_variant, P_variant,mode,flags,true));
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
                                              TwoStepNPTMTK::couplingMode mode,
                                              unsigned int flags)
    {
    boost::shared_ptr<Variant> P_variant(new VariantConst(P));
    boost::shared_ptr<Variant> T_variant(new VariantConst(T));
    // for the tests, we can assume that group is the all group
    return shared_ptr<TwoStepNPTMTK>(new TwoStepNPTMTKGPU(sysdef, group, thermo_group, tau, tauP, T_variant, P_variant,mode,flags,false));
    }

shared_ptr<TwoStepNPTMTK> gpu_nph_creator(shared_ptr<SystemDefinition> sysdef,
                                              shared_ptr<ParticleGroup> group,
                                              boost::shared_ptr<ComputeThermo> thermo_group,
                                              Scalar tau,
                                              Scalar tauP,
                                              Scalar T,
                                              Scalar P,
                                              TwoStepNPTMTK::couplingMode mode,
                                              unsigned int flags)
    {
    boost::shared_ptr<Variant> P_variant(new VariantConst(P));
    boost::shared_ptr<Variant> T_variant(new VariantConst(T));
    // for the tests, we can assume that group is the all group
    return shared_ptr<TwoStepNPTMTK>(new TwoStepNPTMTKGPU(sysdef, group, thermo_group, tau, tauP, T_variant, P_variant,mode,flags,true));
    }
#endif

#ifdef ENABLE_CUDA

//! boost test case for GPU integration tests
BOOST_AUTO_TEST_CASE( TwoStepNPTMTKGPU_tests )
    {
    twostep_npt_mtk_creator npt_mtk_creator = bind(gpu_npt_mtk_creator, _1, _2,_3,_4,_5,_6, _7, _8,_9);
    npt_mtk_updater_test(npt_mtk_creator, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

BOOST_AUTO_TEST_CASE( TwoStepNPTMTKGPU_cubic_NPH)
    {
    twostep_npt_mtk_creator npt_mtk_creator = bind(gpu_nph_creator, _1, _2,_3,_4,_5,_6, _7, _8, _9);
    nph_integration_test(npt_mtk_creator, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

#endif

//! boost test case for base class integration tests
BOOST_AUTO_TEST_CASE( TwoStepNPTMTK_tests )
    {
    twostep_npt_mtk_creator npt_mtk_creator = bind(base_class_npt_mtk_creator, _1, _2,_3,_4,_5, _6, _7, _8,_9);
    npt_mtk_updater_test(npt_mtk_creator, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

//! boost test case for NPH integration
BOOST_AUTO_TEST_CASE( TwoStepNPTMTK_cubic_NPH )
    {
    twostep_npt_mtk_creator npt_mtk_creator = bind(base_class_nph_creator, _1, _2,_3,_4,_5, _6, _7, _8,_9);
    nph_integration_test(npt_mtk_creator, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }


#ifdef WIN32
#pragma warning( pop )
#endif
