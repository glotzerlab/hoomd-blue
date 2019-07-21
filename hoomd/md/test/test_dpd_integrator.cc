// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.




#include <iostream>
#include <fstream>

#include <memory>

#include "hoomd/ComputeThermo.h"
#include "hoomd/md/AllPairPotentials.h"

#include "hoomd/md/TwoStepNVE.h"
#ifdef ENABLE_CUDA
#include "hoomd/md/TwoStepNVEGPU.h"
#endif

#include "hoomd/md/NeighborListTree.h"
#include "hoomd/Initializers.h"
#include "hoomd/md/IntegratorTwoStep.h"


#include <math.h>

using namespace std;

/*! \file dpd_integrator_test.cc
    \brief Implements unit tests for PotentialPairDPDThermo
    \ingroup unit_tests
*/

#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN();





template <class PP_DPD>
void dpd_conservative_force_test(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(2, BoxDim(50.0), 1, 0, 0, 0, 0, exec_conf));
    std::shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    std::shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef, 0, pdata->getN()-1));
    std::shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_all));

    // setup a simple initial system
    pdata->setPosition(0,make_scalar3(0.0,0.0,0.0));
    pdata->setVelocity(0,make_scalar3(0.0,0.0,0.0));
    pdata->setPosition(1,make_scalar3(0.1,0.0,0.0));
    pdata->setVelocity(1,make_scalar3(0.0,0.0,0.0));

    // Construction of the Force Compute
    std::shared_ptr<NeighborListTree> nlist(new NeighborListTree(sysdef, Scalar(2.0), Scalar(0.8)));
    nlist->setStorageMode(NeighborList::full);
    std::shared_ptr<PotentialPairDPD> dpdc(new PP_DPD(sysdef,nlist));
    dpdc->setParams(0,0,make_scalar2(30,0));
    dpdc->setRcut(0, 0, Scalar(2.0));

    // compute the forces
    dpdc->compute(0);

    GlobalArray<Scalar4>& force_array_1 =  dpdc->getForceArray();
    GlobalArray<Scalar>& virial_array_1 =  dpdc->getVirialArray();
    ArrayHandle<Scalar4> h_force_1(force_array_1,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_1(virial_array_1,access_location::host,access_mode::read);
    MY_CHECK_CLOSE(h_force_1.data[0].x, -28.5, tol);
    MY_CHECK_CLOSE(h_force_1.data[0].y, 0, tol);
    MY_CHECK_CLOSE(h_force_1.data[0].z, 0, tol);
    MY_CHECK_CLOSE(h_force_1.data[0].w, 13.5375, tol);
    }

UP_TEST( DPD_ForceConservative_Test )
    {
    dpd_conservative_force_test< PotentialPair<EvaluatorPairDPDThermo> >(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

#ifdef ENABLE_CUDA
UP_TEST( DPD_GPU_ForceConservative_Test )
    {
    dpd_conservative_force_test< PotentialPairGPU<EvaluatorPairDPDThermo, gpu_compute_dpdthermo_forces > >(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
#endif

template <class PP_DPD>
void dpd_temperature_test(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(1000, BoxDim(5.0), 1, 0, 0, 0, 0, exec_conf));
    std::shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    std::shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef, 0, pdata->getN()-1));
    std::shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_all));

    // setup a simple initial dense state
    for (int j = 0; j < 1000; j++)
        {
        pdata->setPosition(j,make_scalar3(-2.0 + 0.3*(j %10),
                                         -2.0 + 0.3*(j/10 %10),
                                          -2.0 + 0.3*(j/100)));
        pdata->setVelocity(j,make_scalar3(0.0,0.0,0.0));
        }

    Scalar deltaT = Scalar(0.02);
    Scalar Temp = Scalar(2.0);
    std::shared_ptr<VariantConst> T_variant(new VariantConst(Temp));

    cout << endl << "Test 1" << endl;
    cout << "Creating an dpd gas of 1000 particles" << endl;
    cout << "Temperature set at " << Temp << endl;

    std::shared_ptr<TwoStepNVE> two_step_nve(new TwoStepNVE(sysdef,group_all));
    std::shared_ptr<ComputeThermo> thermo(new ComputeThermo(sysdef, group_all));
    thermo->setNDOF(3*1000);
    std::shared_ptr<IntegratorTwoStep> nve_up(new IntegratorTwoStep(sysdef, deltaT));
    nve_up->addIntegrationMethod(two_step_nve);


    // Construction of the Force Compute
    std::shared_ptr<NeighborListTree> nlist(new NeighborListTree(sysdef, Scalar(1.0), Scalar(0.8)));
    nlist->setStorageMode(NeighborList::full);
    std::shared_ptr<PotentialPairDPDThermoDPD> dpd_thermo(new PP_DPD(sysdef,nlist));
    dpd_thermo->setSeed(12345);
    dpd_thermo->setT(T_variant);
    dpd_thermo->setParams(0,0,make_scalar2(30,4.5));
    dpd_thermo->setRcut(0, 0, Scalar(1.0));
    nve_up->addForceCompute(dpd_thermo);
    nve_up->prepRun(0);

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
    MY_CHECK_CLOSE(AvgT, 2.0, 5);

   // Calculate Momentum
    Scalar(Mom_x) = 0;
    Scalar(Mom_y) = 0;
    Scalar(Mom_z) = 0;

    // get momentum
    for (int j = 0; j < 1000; j++)
        {
        Scalar3 vel = pdata->getVelocity(j);
        Mom_x += vel.x;
        Mom_y += vel.y;
        Mom_z += vel.z;
        }

    MY_CHECK_SMALL(Mom_x, 1e-3);
    MY_CHECK_SMALL(Mom_y, 1e-3);
    MY_CHECK_SMALL(Mom_z, 1e-3);




    }

UP_TEST( DPD_Temp_Test )
    {
    dpd_temperature_test< PotentialPairDPDThermo<EvaluatorPairDPDThermo> >(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

#ifdef ENABLE_CUDA
UP_TEST( DPD_GPU_Temp_Test )
    {
    dpd_temperature_test< PotentialPairDPDThermoGPU<EvaluatorPairDPDThermo, gpu_compute_dpdthermodpd_forces > >(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
#endif
