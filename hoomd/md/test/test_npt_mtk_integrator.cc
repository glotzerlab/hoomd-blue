// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include <iostream>

#include <functional>
#include <memory>

#include "hoomd/ComputeThermo.h"
#include "hoomd/md/TwoStepNPTMTK.h"
#ifdef ENABLE_CUDA
#include "hoomd/md/TwoStepNPTMTKGPU.h"
#include "hoomd/ComputeThermoGPU.h"
#endif
#include "hoomd/md/IntegratorTwoStep.h"

#include "hoomd/CellList.h"
#include "hoomd/md/NeighborList.h"
#include "hoomd/md/NeighborListBinned.h"
#include "hoomd/Initializers.h"
#include "hoomd/deprecated/RandomGenerator.h"
#include "hoomd/md/AllPairPotentials.h"
#include "hoomd/md/AllAnisoPairPotentials.h"

#ifdef ENABLE_CUDA
#include "hoomd/md/NeighborListGPUBinned.h"
#include "hoomd/CellListGPU.h"
#endif

#include "hoomd/RandomNumbers.h"
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"
#include "hoomd/extern/pybind/include/pybind11/embed.h"
namespace py = pybind11;

#include "hoomd/Variant.h"

using namespace hoomd;

#include <math.h>

using namespace std;
using namespace std::placeholders;

/*! \file test_npt_mtk_integrator.cc
    \brief Implements unit tests for NPTMTKpdater and descendants
    \ingroup unit_tests
*/

#include "hoomd/test/upp11_config.h"
HOOMD_UP_MAIN();

PYBIND11_EMBEDDED_MODULE(variant, m)
    {
    export_Variant(m);
    }


typedef struct
    {
    std::shared_ptr<SystemDefinition> sysdef;
    std::shared_ptr<ParticleGroup> group;
    std::shared_ptr<ComputeThermo> thermo_group;
    std::shared_ptr<ComputeThermo> thermo_group_t;
    Scalar tau;
    Scalar tauP;
    Scalar T;
    Scalar P;
    TwoStepNPTMTK::couplingMode mode;
    unsigned int flags;
    } args_t;

//! Typedef'd NPTMTKUpdater class factory
typedef std::function<std::shared_ptr<TwoStepNPTMTK> (args_t args) > twostep_npt_mtk_creator;

//! Basic functionality test of a generic TwoStepNPTMTK
void npt_mtk_updater_test(twostep_npt_mtk_creator npt_mtk_creator, std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // we use a tightly packed cubic LJ crystal for testing,
    // because this one has a sufficient shear elasticity
    // to avoid that the box gets too tilted during triclinic NPT
    const unsigned int L = 10; // number of particles along one box edge
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
    std::shared_ptr< SnapshotSystemData<Scalar> > snap = cubic_init.getSnapshot();
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));
    std::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    // enable the energy computation
    PDataFlags flags;
    flags[pdata_flag::pressure_tensor] = 1;
    flags[pdata_flag::isotropic_virial] = 1;
    // only for output of enthalpy
    flags[pdata_flag::potential_energy] = 1;
    pdata->setFlags(flags);

    std::shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef, 0, pdata->getN()-1));
    std::shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_all));

    std::shared_ptr<NeighborList> nlist;
    std::shared_ptr<PotentialPairLJ> fc;
    std::shared_ptr<CellList> cl;
    #ifdef ENABLE_CUDA
    if (exec_conf->isCUDAEnabled())
        {
        cl = std::shared_ptr<CellList>( new CellListGPU(sysdef) );
        nlist = std::shared_ptr<NeighborList>( new NeighborListGPUBinned(sysdef, Scalar(2.5), Scalar(0.4),cl));
        fc = std::shared_ptr<PotentialPairLJ>( new PotentialPairLJGPU(sysdef, nlist));
        }
    else
    #endif
        {
        cl = std::shared_ptr<CellList>( new CellList(sysdef) );
        nlist = std::shared_ptr<NeighborList>(new NeighborListBinned(sysdef, Scalar(2.5), Scalar(0.4),cl));
        fc = std::shared_ptr<PotentialPairLJ>( new PotentialPairLJ(sysdef, nlist));
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

    std::shared_ptr<ComputeThermo> thermo_group;
    std::shared_ptr<ComputeThermo> thermo_group_t;
    #ifdef ENABLE_CUDA
    if (exec_conf->isCUDAEnabled())
        {
        thermo_group = std::shared_ptr<ComputeThermo>(new ComputeThermoGPU(sysdef, group_all, "name"));
        thermo_group_t = std::shared_ptr<ComputeThermo>(new ComputeThermoGPU(sysdef, group_all, "name_t"));
        }
    else
    #endif
        {
        thermo_group = std::shared_ptr<ComputeThermo>((new ComputeThermo(sysdef, group_all, "name")));
        thermo_group_t = std::shared_ptr<ComputeThermo>((new ComputeThermo(sysdef, group_all, "name_t")));
        }

    thermo_group->setNDOF(3*pdata->getN()-3);
    thermo_group_t->setNDOF(3*pdata->getN()-3);
    std::shared_ptr<IntegratorTwoStep> npt_mtk(new IntegratorTwoStep(sysdef, Scalar(deltaT)));
    npt_mtk->addForceCompute(fc);

    // successively integrate the system using different methods
    unsigned int offs=0;
    for (unsigned int i_mode = 0; i_mode < n_modes; i_mode++)
        {
        TwoStepNPTMTK::couplingMode mode = coupling_modes[i_mode];
        unsigned int flags = npt_flags[i_mode];
        std::cout << "Testing NPT with mode " << mode_name[i_mode] << std::endl;

        args_t args;
        args.sysdef=sysdef;
        args.group = group_all;
        args.thermo_group = thermo_group;
        args.thermo_group_t = thermo_group_t;
        args.tau = tau;
        args.tauP = tauP;
        args.T = T0;
        args.P = P;
        args.mode = mode;
        args.flags = flags;

        std::shared_ptr<TwoStepNPTMTK> two_step_npt_mtk = npt_mtk_creator(args);
        npt_mtk->removeAllIntegrationMethods();
        npt_mtk->addIntegrationMethod(two_step_npt_mtk);
        npt_mtk->prepRun(0);

        // step for a 10,000 timesteps to relax pressure and temperature
        // before computing averages

        std::cout << "Equilibrating 10,000 steps... " << std::endl;
        unsigned int timestep;
        for (unsigned int i = 0; i < 10000; i++)
            {
            timestep = offs + i;
            if (i % 1000 == 0)
                {
                std::cout << i << std::endl;
                BoxDim box = pdata->getBox();
                Scalar3 npd = box.getNearestPlaneDistance();
                std::cout << "Box L: " << box.getL().x << " " << box.getL().y << " " << box.getL().z
                          << " t: " << box.getTiltFactorXY() << " " << box.getTiltFactorXZ() << " " << box.getTiltFactorYZ()
                          << " npd: " << npd.x << " " << npd.y << " " << npd.z << std::endl;
                }
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

        bool flag = false;
        thermo_group_t->compute(0);
        BoxDim box = pdata->getBox();
        Scalar volume = box.getVolume();
        Scalar enthalpy =  thermo_group_t->getKineticEnergy() + thermo_group_t->getPotentialEnergy() + P * volume;
        Scalar barostat_energy = npt_mtk->getLogValue("npt_barostat_energy", flag);
        Scalar thermostat_energy = npt_mtk->getLogValue("npt_thermostat_energy", flag);
        Scalar H_ref = enthalpy + barostat_energy + thermostat_energy; // the conserved quantity

        // 0.02 % accuracy for conserved quantity
        Scalar H_tol = 0.02;

        std::cout << "Measuring up to 20,000 steps... " << std::endl;
        for (unsigned int i = 10000; i < 20000; i++)
            {
            timestep = offs + i;

            npt_mtk->update(timestep);

            if (i % 1000 == 0)
                {
                std::cout << i << std::endl;
                }

            if (i% 100 == 0)
                {
                thermo_group_t->compute(timestep+1);
                PressureTensor P_current = thermo_group_t->getPressureTensor();
                avrPxx += P_current.xx;
                avrPxy += P_current.xy;
                avrPxz += P_current.xz;
                avrPyy += P_current.yy;
                avrPyz += P_current.yz;
                avrPzz += P_current.zz;

                avrT += thermo_group_t->getTemperature();
                count++;

                box = pdata->getBox();
                volume = box.getVolume();
                enthalpy =  thermo_group_t->getKineticEnergy() + thermo_group_t->getPotentialEnergy() + P * volume;
                barostat_energy = npt_mtk->getLogValue("npt_barostat_energy",flag);
                thermostat_energy = npt_mtk->getLogValue("npt_thermostat_energy",flag);
                Scalar H = enthalpy + barostat_energy + thermostat_energy;
                MY_CHECK_CLOSE(H_ref,H,H_tol);

                /*
                box = pdata->getBox();
                Scalar3 L = box.getL();
                std::cout << "L == (" << L.x << ", " << L.y << ", " << L.z << ")" << std::endl;
                */
                }
            }

        thermo_group_t->compute(timestep+1);
        box = pdata->getBox();
        volume = box.getVolume();
        enthalpy =  thermo_group_t->getKineticEnergy() + thermo_group_t->getPotentialEnergy() + P * volume;
        barostat_energy = npt_mtk->getLogValue("npt_barostat_energy", flag);
        thermostat_energy = npt_mtk->getLogValue("npt_thermostat_energy", flag);
        Scalar H_final = enthalpy + barostat_energy + thermostat_energy;

        // check conserved quantity, required accuracy 2*10^-4
        MY_CHECK_CLOSE(H_ref,H_final,H_tol);

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
            MY_CHECK_CLOSE(avrP, P, rough_tol);
        else if (i_mode == 1) // orthorhombic
            {
            MY_CHECK_CLOSE(avrPxx, P, rough_tol);
            MY_CHECK_CLOSE(avrPyy, P, rough_tol);
            MY_CHECK_CLOSE(avrPzz, P, rough_tol);
            }
        else if (i_mode == 2) // tetragonal
            {
            MY_CHECK_CLOSE(avrPxx, P, rough_tol);
            MY_CHECK_CLOSE(Scalar(1.0/2.0)*(avrPyy+avrPzz), avrP, rough_tol);
            }
       else if (mode == 3) // triclinic
            {
            MY_CHECK_CLOSE(avrPxx, P, rough_tol);
            MY_CHECK_CLOSE(avrPyy, P, rough_tol);
            MY_CHECK_CLOSE(avrPzz, P, rough_tol);
            MY_CHECK_SMALL(avrPxy,rough_tol);
            MY_CHECK_SMALL(avrPxz,rough_tol);
            MY_CHECK_SMALL(avrPyz,rough_tol);
            }
        MY_CHECK_CLOSE(T0, avrT, rough_tol);
        offs+=timestep;
        }
    }

//! Test ability to integrate in the NPH ensemble
void nph_integration_test(twostep_npt_mtk_creator nph_creator, std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    const unsigned int N = 1000;
    Scalar P = 1.0;
    Scalar T0 = 1.0;
    Scalar deltaT = 0.001;

    Scalar tauP(1.0);

    // create two identical random particle systems to simulate
    RandomInitializer rand_init(N, Scalar(0.2), Scalar(0.9), "A");
    rand_init.setSeed(12345);
    std::shared_ptr< SnapshotSystemData<Scalar> > snap = rand_init.getSnapshot();
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));
    std::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    // give the particles velocities according to a Maxwell-Boltzmann distribution
    RandomGenerator rng(54321);

    // total up the system momentum
    Scalar3 total_momentum = make_scalar3(0.0, 0.0, 0.0);
    unsigned int nparticles= pdata->getN();

    // generate the gaussian velocity distribution
    for (unsigned int idx = 0; idx < nparticles; idx++)
    {
        // generate gaussian velocities
        Scalar mass = pdata->getMass(idx);
        Scalar sigma = T0 / mass;
        NormalDistribution<Scalar> normal(sigma);
        Scalar vx = normal(rng);
        Scalar vy = normal(rng);
        Scalar vz = normal(rng);

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

    std::shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef, 0, pdata->getN()-1));
    std::shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_all));

    std::shared_ptr<NeighborList> nlist;
    std::shared_ptr<PotentialPairLJ> fc;
    std::shared_ptr<CellList> cl;
    #ifdef ENABLE_CUDA
    if (exec_conf->isCUDAEnabled())
        {
        cl = std::shared_ptr<CellList>( new CellListGPU(sysdef) );
        nlist = std::shared_ptr<NeighborList>( new NeighborListGPUBinned(sysdef, Scalar(2.5), Scalar(0.8),cl));
        fc = std::shared_ptr<PotentialPairLJ>( new PotentialPairLJGPU(sysdef, nlist));
        }
    else
    #endif
        {
        cl = std::shared_ptr<CellList>( new CellList(sysdef) );
        nlist = std::shared_ptr<NeighborList>(new NeighborListBinned(sysdef, Scalar(2.5), Scalar(0.8),cl));
        fc = std::shared_ptr<PotentialPairLJ>( new PotentialPairLJ(sysdef, nlist));
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

    std::shared_ptr<ComputeThermo> compute_thermo(new ComputeThermo(sysdef, group_all, "name"));
    compute_thermo->setNDOF(3*N-3);

    std::shared_ptr<ComputeThermo> compute_thermo_t(new ComputeThermo(sysdef, group_all, "name"));
    compute_thermo_t->setNDOF(3*N-3);

    // set up integration without thermostat
    args_t args;
    args.sysdef=sysdef;
    args.group = group_all;
    args.thermo_group = compute_thermo;
    args.thermo_group_t = compute_thermo_t;
    args.tau = 1.0;
    args.tauP = tauP;
    args.T = 1.0;
    args.P = P;
    args.mode = TwoStepNPTMTK::couple_xyz;
    args.flags = TwoStepNPTMTK::baro_x | TwoStepNPTMTK::baro_y | TwoStepNPTMTK::baro_z;

    std::shared_ptr<TwoStepNPTMTK> two_step_npt = nph_creator(args);
    std::shared_ptr<IntegratorTwoStep> nph(new IntegratorTwoStep(sysdef, Scalar(deltaT)));
    nph->addIntegrationMethod(two_step_npt);
    nph->addForceCompute(fc);
    nph->prepRun(0);

    // step for a 10,000 timesteps to relax pressure and temperature
    // before computing averages

    for (int i = 0; i < 10000; i++)
        {
        nph->update(i);
        }

    // now do the averaging for next 10k steps
    Scalar avrPxx = 0.0;
    Scalar avrPyy = 0.0;
    Scalar avrPzz = 0.0;
    int count = 0;

    bool flag = false;
    compute_thermo_t->compute(0);
    BoxDim box = pdata->getBox();
    Scalar3 L = box.getL();
    Scalar volume = L.x*L.y*L.z;
    Scalar enthalpy =  compute_thermo_t->getKineticEnergy() + compute_thermo_t->getPotentialEnergy() + P * volume;
    Scalar barostat_energy = nph->getLogValue("npt_barostat_energy", flag);
    Scalar H_ref = enthalpy + barostat_energy; // the conserved quantity

    for (int i = 10001; i < 20000; i++)
        {
        if (i % 100 == 0)
            {
            compute_thermo_t->compute(i);
            PressureTensor P_current = compute_thermo_t->getPressureTensor();
            avrPxx += P_current.xx;
            avrPyy += P_current.yy;
            avrPzz += P_current.zz;

            count++;
            }
        nph->update(i);
        }

    compute_thermo_t->compute(count+1);
    box = pdata->getBox();
    L = box.getL();
    volume = L.x*L.y*L.z;
    enthalpy =  compute_thermo_t->getKineticEnergy() + compute_thermo_t->getPotentialEnergy() + P * volume;
    barostat_energy = nph->getLogValue("npt_barostat_energy", flag);
    Scalar H_final = enthalpy + barostat_energy;
    // check conserved quantity
    Scalar tol = 0.01;
    MY_CHECK_CLOSE(H_ref,H_final,tol);

    avrPxx /= Scalar(count);
    avrPyy /= Scalar(count);
    avrPzz /= Scalar(count);
    Scalar avrP= Scalar(1./3.)*(avrPxx+avrPyy+avrPzz);
    Scalar rough_tol = 2.0;
    MY_CHECK_CLOSE(P, avrP, rough_tol);
    }

//! Basic functionality test of a generic TwoStepNPTMTK for anisotropic pair potential
void npt_mtk_updater_aniso(twostep_npt_mtk_creator npt_mtk_creator, std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    Scalar phi_p = 0.2;
    unsigned int N = 1000;
    Scalar L = pow(M_PI/6.0/phi_p*Scalar(N),1.0/3.0);
    BoxDim box_g(L);

    Scalar P = 1.2;
    Scalar T0 = .5;
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

    SimpleCubicInitializer cubic_init(10, Scalar(0.89), "A");
    std::shared_ptr< SnapshotSystemData<Scalar> > snap = cubic_init.getSnapshot();

    // have to set moment of inertia to actually test aniso integration
    for(unsigned int i(0); i < snap->particle_data.size; ++i)
        snap->particle_data.inertia[i] = vec3<Scalar>(1.0, 1.0, 1.0);

    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));
    std::shared_ptr<ParticleData> pdata = sysdef->getParticleData();

    // enable the energy computation
    PDataFlags flags;
    flags[pdata_flag::pressure_tensor] = 1;
    flags[pdata_flag::isotropic_virial] = 1;
    flags[pdata_flag::potential_energy] = 1;
    flags[pdata_flag::rotational_kinetic_energy] = 1;
    pdata->setFlags(flags);

    std::shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef, 0, pdata->getN()-1));
    std::shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_all));

    std::shared_ptr<NeighborList> nlist;

    // set up Gay-Berne
    std::shared_ptr<AnisoPotentialPairGB> fc;
    std::shared_ptr<CellList> cl;

    Scalar r_cut = 2.5;
    Scalar r_buff = 0.3;

    #ifdef ENABLE_CUDA
    if (exec_conf->isCUDAEnabled())
        {
        cl = std::shared_ptr<CellList>( new CellListGPU(sysdef) );
        nlist = std::shared_ptr<NeighborList>( new NeighborListGPUBinned(sysdef, r_cut, r_buff,cl));
        fc = std::shared_ptr<AnisoPotentialPairGBGPU>(new AnisoPotentialPairGBGPU(sysdef, nlist));
        }
    else
    #endif
        {
        cl = std::shared_ptr<CellList>( new CellList(sysdef) );
        nlist = std::shared_ptr<NeighborList>(new NeighborListBinned(sysdef, r_cut, r_buff, cl));
        fc = std::shared_ptr<AnisoPotentialPairGB>(new AnisoPotentialPairGB(sysdef, nlist));
        }

    fc->setRcut(0, 0, r_cut);

    // setup some values for alpha and sigma
    Scalar epsilon = Scalar(1.0);
    Scalar lperp = Scalar(0.3);
    Scalar lpar = Scalar(0.5);
    fc->setParams(0,0,make_scalar3(epsilon,lperp,lpar));

    // If we want accurate calculation of potential energy, we need to apply the
    // energy shift
    fc->setShiftMode(AnisoPotentialPairGB::shift);

    std::shared_ptr<ComputeThermo> thermo_group;
    std::shared_ptr<ComputeThermo> thermo_group_t;
    #ifdef ENABLE_CUDA
    if (exec_conf->isCUDAEnabled())
        {
        thermo_group = std::shared_ptr<ComputeThermo>(new ComputeThermoGPU(sysdef, group_all, "name"));
        thermo_group_t = std::shared_ptr<ComputeThermo>(new ComputeThermoGPU(sysdef, group_all, "name_t"));
        }
    else
    #endif
        {
        thermo_group = std::shared_ptr<ComputeThermo>((new ComputeThermo(sysdef, group_all, "name")));
        thermo_group_t = std::shared_ptr<ComputeThermo>((new ComputeThermo(sysdef, group_all, "name_t")));
        }

    thermo_group->setNDOF(3*pdata->getN()-3);
    thermo_group_t->setNDOF(3*pdata->getN()-3);

    std::shared_ptr<IntegratorTwoStep> npt_mtk(new IntegratorTwoStep(sysdef, Scalar(deltaT)));
    npt_mtk->addForceCompute(fc);

    // successively integrate the system using different methods
    unsigned int offs=0;
    for (unsigned int i_mode = 0; i_mode < n_modes; i_mode++)
        {
        TwoStepNPTMTK::couplingMode mode = coupling_modes[i_mode];
        unsigned int flags = npt_flags[i_mode];
        std::cout << "Testing NPT with Gay-Berne in mode " << mode_name[i_mode] << std::endl;

        args_t args;
        args.sysdef=sysdef;
        args.group = group_all;
        args.thermo_group = thermo_group;
        args.thermo_group_t = thermo_group_t;
        args.tau = tau;
        args.tauP = tauP;
        args.T = T0;
        args.P = P;
        args.mode = mode;
        args.flags = flags;

        std::shared_ptr<TwoStepNPTMTK> two_step_npt_mtk = npt_mtk_creator(args);
        npt_mtk->removeAllIntegrationMethods();
        npt_mtk->addIntegrationMethod(two_step_npt_mtk);

        unsigned int ndof_rot = npt_mtk->getRotationalNDOF(group_all);
        thermo_group->setRotationalNDOF(ndof_rot);
        thermo_group_t->setRotationalNDOF(ndof_rot);

        npt_mtk->prepRun(0);

        // step for a 10,000 timesteps to relax pressure and temperature
        // before computing averages

        unsigned int n_equil_steps = 1500;
        std::cout << "Equilibrating " << n_equil_steps << " steps... " << std::endl;
        unsigned int timestep;
        for (unsigned int i = 0; i < n_equil_steps; i++)
            {
            timestep = offs + i;
            if (i % 1000 == 0)
                {
                std::cout << i << std::endl;
                BoxDim box = pdata->getBox();
                Scalar3 npd = box.getNearestPlaneDistance();
                std::cout << "Box L: " << box.getL().x << " " << box.getL().y << " " << box.getL().z
                          << " t: " << box.getTiltFactorXY() << " " << box.getTiltFactorXZ() << " " << box.getTiltFactorYZ()
                          << " npd: " << npd.x << " " << npd.y << " " << npd.z << std::endl;
                }
            npt_mtk->update(timestep);
            }

        // now do the averaging
        Scalar avrPxx(0.0);
        Scalar avrPxy(0.0);
        Scalar avrPxz(0.0);
        Scalar avrPyy(0.0);
        Scalar avrPyz(0.0);
        Scalar avrPzz(0.0);
        Scalar avrT(0.0);
        int count = 0;

        bool flag = false;
        thermo_group_t->compute(0);
        BoxDim box = pdata->getBox();
        Scalar volume = box.getVolume();
        Scalar enthalpy =  thermo_group_t->getKineticEnergy() + thermo_group_t->getPotentialEnergy() + P * volume;
        Scalar barostat_energy = npt_mtk->getLogValue("npt_barostat_energy", flag);
        Scalar thermostat_energy = npt_mtk->getLogValue("npt_thermostat_energy", flag);
        Scalar H_ref = enthalpy + barostat_energy + thermostat_energy; // the conserved quantity

        // 0.25 % accuracy for conserved quantity
        Scalar H_tol = 0.25;

        unsigned int n_measure_steps = 10000;
        std::cout << "Measuring over " << n_measure_steps << " steps... " << std::endl;
        for (unsigned int i = n_equil_steps; i < n_equil_steps+n_measure_steps; i++)
            {
            timestep = offs + i;
            npt_mtk->update(timestep);
            if (i % 10000 == 0)
                {
                std::cout << i << std::endl;
                }
            if (i% 100 == 0)
                {
                thermo_group_t->compute(timestep+1);
                PressureTensor P_current = thermo_group_t->getPressureTensor();
                avrPxx += P_current.xx;
                avrPxy += P_current.xy;
                avrPxz += P_current.xz;
                avrPyy += P_current.yy;
                avrPyz += P_current.yz;
                avrPzz += P_current.zz;

                avrT += thermo_group_t->getTemperature();
                count++;

                box = pdata->getBox();
                volume = box.getVolume();
                Scalar ke = thermo_group_t->getKineticEnergy();
                Scalar rotational_ke = thermo_group_t->getRotationalKineticEnergy();
                Scalar pe = thermo_group_t->getPotentialEnergy();
                enthalpy =  ke + pe + P * volume;
                barostat_energy = npt_mtk->getLogValue("npt_barostat_energy",flag);
                thermostat_energy = npt_mtk->getLogValue("npt_thermostat_energy",flag);
                Scalar H = enthalpy + barostat_energy + thermostat_energy;
                std::cout << "KE: " << ke << " PE: " << pe << " PV: " << P*volume << std::endl;
                std::cout << "baro: " << barostat_energy << " thermo: " << thermostat_energy << " rot KE: " << rotational_ke << std::endl;
                MY_CHECK_CLOSE(H_ref,H,H_tol);

                /*
                box = pdata->getBox();
                Scalar3 L = box.getL();
                std::cout << "L == (" << L.x << ", " << L.y << ", " << L.z << ")" << std::endl;
                */
                }
            }

        thermo_group_t->compute(timestep+1);
        box = pdata->getBox();
        volume = box.getVolume();
        enthalpy =  thermo_group_t->getKineticEnergy() + thermo_group_t->getPotentialEnergy() + P * volume;
        barostat_energy = npt_mtk->getLogValue("npt_barostat_energy", flag);
        thermostat_energy = npt_mtk->getLogValue("npt_thermostat_energy", flag);
        Scalar H_final = enthalpy + barostat_energy + thermostat_energy;

        // check conserved quantity, required accuracy 2*10^-4
        MY_CHECK_CLOSE(H_ref,H_final,H_tol);

        avrPxx /= Scalar(count);
        avrPxy /= Scalar(count);
        avrPxz /= Scalar(count);
        avrPyy /= Scalar(count);
        avrPyz /= Scalar(count);
        avrPzz /= Scalar(count);
        avrT /= Scalar(count);
        Scalar avrP= Scalar(1./3.)*(avrPxx+avrPyy+avrPzz);
        Scalar rough_tol = 5.0;
        if (i_mode == 0) // cubic
            MY_CHECK_CLOSE(avrP, P, rough_tol);
        else if (i_mode == 1) // orthorhombic
            {
            MY_CHECK_CLOSE(avrPxx, P, rough_tol);
            MY_CHECK_CLOSE(avrPyy, P, rough_tol);
            MY_CHECK_CLOSE(avrPzz, P, rough_tol);
            }
        else if (i_mode == 2) // tetragonal
            {
            MY_CHECK_CLOSE(avrPxx, P, rough_tol);
            MY_CHECK_CLOSE(Scalar(1.0/2.0)*(avrPyy+avrPzz), avrP, rough_tol);
            }
       else if (mode == 3) // triclinic
            {
            MY_CHECK_CLOSE(avrPxx, P, rough_tol);
            MY_CHECK_CLOSE(avrPyy, P, rough_tol);
            MY_CHECK_CLOSE(avrPzz, P, rough_tol);
            MY_CHECK_SMALL(avrPxy,rough_tol);
            MY_CHECK_SMALL(avrPxz,rough_tol);
            MY_CHECK_SMALL(avrPyz,rough_tol);
            }
        MY_CHECK_CLOSE(T0, avrT, rough_tol);
        offs+=timestep;
        }
    }

//! IntegratorTwoStepNPTMTK factory for the unit tests
std::shared_ptr<TwoStepNPTMTK> base_class_npt_mtk_creator(args_t args)
    {
    std::shared_ptr<Variant> P_variant(new VariantConst(args.P));
    std::shared_ptr<Variant> zero_variant(new VariantConst(0.0));
    // necessary to create python objects
    py::scoped_interpreter guard{};
    py::module::import("variant");
    py::list S;
    S.append(P_variant);
    S.append(P_variant);
    S.append(P_variant);
    S.append(zero_variant);
    S.append(zero_variant);
    S.append(zero_variant);

    std::shared_ptr<Variant> T_variant(new VariantConst(args.T));
    // for the tests, we can assume that group is the all group
    return std::shared_ptr<TwoStepNPTMTK>(new TwoStepNPTMTK(args.sysdef,
        args.group,
        args.thermo_group,
        args.thermo_group_t,
        args.tau,
        args.tauP,
        T_variant,
        S,
        args.mode,
        args.flags,
        false));
    }

std::shared_ptr<TwoStepNPTMTK> base_class_nph_creator(args_t args)
    {
    std::shared_ptr<Variant> P_variant(new VariantConst(args.P));
    std::shared_ptr<Variant> zero_variant(new VariantConst(0.0));
    // necessary to create python objects
    py::scoped_interpreter guard{};
    py::module::import("variant");
    py::list S;
    S.append(P_variant);
    S.append(P_variant);
    S.append(P_variant);
    S.append(zero_variant);
    S.append(zero_variant);
    S.append(zero_variant);
    std::cout << py::len(S) << std::endl;

    std::shared_ptr<Variant> T_variant(new VariantConst(args.T));
    // for the tests, we can assume that group is the all group
    return std::shared_ptr<TwoStepNPTMTK>(new TwoStepNPTMTK(args.sysdef,
        args.group,
        args.thermo_group,
        args.thermo_group_t,
        args.tau,
        args.tauP,
        T_variant,
        S,
        args.mode,
        args.flags,true));
    }

#ifdef ENABLE_CUDA
//! NPTMTKIntegratorGPU factory for the unit tests
std::shared_ptr<TwoStepNPTMTK> gpu_npt_mtk_creator(args_t args)
    {
    std::shared_ptr<Variant> P_variant(new VariantConst(args.P));
    std::shared_ptr<Variant> zero_variant(new VariantConst(0.0));
    // necessary to create python objects
    py::scoped_interpreter guard{};
    py::module::import("variant");
    py::list S;
    S.append(P_variant);
    S.append(P_variant);
    S.append(P_variant);
    S.append(zero_variant);
    S.append(zero_variant);
    S.append(zero_variant);
    std::shared_ptr<Variant> T_variant(new VariantConst(args.T));
    // for the tests, we can assume that group is the all group
    return std::shared_ptr<TwoStepNPTMTK>(new TwoStepNPTMTKGPU(args.sysdef, args.group, args.thermo_group, args.thermo_group_t,
        args.tau, args.tauP, T_variant, S,args.mode,args.flags,false));
    }

std::shared_ptr<TwoStepNPTMTK> gpu_nph_creator(args_t args)
    {
    std::shared_ptr<Variant> P_variant(new VariantConst(args.P));
    std::shared_ptr<Variant> zero_variant(new VariantConst(0.0));
    // necessary to create python objects
    py::scoped_interpreter guard{};
    py::module::import("variant");
    py::list S;
    S.append(P_variant);
    S.append(P_variant);
    S.append(P_variant);
    S.append(zero_variant);
    S.append(zero_variant);
    S.append(zero_variant);

    std::shared_ptr<Variant> T_variant(new VariantConst(args.T));
    return std::shared_ptr<TwoStepNPTMTK>(new TwoStepNPTMTKGPU(args.sysdef, args.group, args.thermo_group, args.thermo_group_t,
        args.tau, args.tauP, T_variant, S, args.mode,args.flags,true));
    }
#endif

//! test case for base class integration tests
UP_TEST( TwoStepNPTMTK_tests )
    {
    twostep_npt_mtk_creator npt_mtk_creator = bind(base_class_npt_mtk_creator, _1);
    std::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::CPU));
    npt_mtk_updater_test(npt_mtk_creator, exec_conf);
    }

//! test case for base class integration tests
UP_TEST( TwoStepNPTMTK_aniso )
    {
    twostep_npt_mtk_creator npt_mtk_creator = bind(base_class_npt_mtk_creator, _1);
    std::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::CPU));
    npt_mtk_updater_aniso(npt_mtk_creator, exec_conf);
    }

//! test case for NPH integration
UP_TEST( TwoStepNPTMTK_cubic_NPH )
    {
    twostep_npt_mtk_creator npt_mtk_creator = bind(base_class_nph_creator, _1);
    nph_integration_test(npt_mtk_creator, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

#ifdef ENABLE_CUDA
//! test case for GPU integration tests
UP_TEST( TwoStepNPTMTKGPU_tests )
    {
    twostep_npt_mtk_creator npt_mtk_creator = bind(gpu_npt_mtk_creator, _1);
    npt_mtk_updater_test(npt_mtk_creator, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! test case for GPU integration tests
UP_TEST( TwoStepNPTMTKGPU_aniso )
    {
    twostep_npt_mtk_creator npt_mtk_creator = bind(gpu_npt_mtk_creator, _1);
    npt_mtk_updater_aniso(npt_mtk_creator, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

UP_TEST( TwoStepNPTMTKGPU_cubic_NPH)
    {
    twostep_npt_mtk_creator npt_mtk_creator = bind(gpu_nph_creator, _1);
    nph_integration_test(npt_mtk_creator, std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
#endif

#ifdef WIN32
#pragma warning( pop )
#endif
