/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2014 The Regents of
the University of Michigan All rights reserved.

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


#include <iostream>

#include <boost/bind.hpp>
#include <boost/function.hpp>

#include "ConstForceCompute.h"
#include "TwoStepNVTMTK.h"
#include "ComputeThermo.h"
#ifdef ENABLE_CUDA
#include "TwoStepNVTMTKGPU.h"
#endif
#include "IntegratorTwoStep.h"

#include "AllPairPotentials.h"
#include "NeighborListBinned.h"
#include "Initializers.h"
#include "RandomGenerator.h"

#include <math.h>

using namespace std;
using namespace boost;

/*! \file nvt_updater_test.cc
    \brief Implements unit tests for NVTUpdater and descendants
    \ingroup unit_tests
*/

//! name the boost unit test module
#define BOOST_TEST_MODULE NVTUpdaterTests
#include "boost_utf_configure.h"


//! Typedef'd NVTUpdator class factory
typedef boost::function<boost::shared_ptr<TwoStepNVTMTK> (boost::shared_ptr<SystemDefinition> sysdef,
                                                 boost::shared_ptr<ParticleGroup> group,
                                                 boost::shared_ptr<ComputeThermo> thermo,
                                                 Scalar Q,
                                                 Scalar T)> twostepnvt_creator;

//! NVTUpdater creator
boost::shared_ptr<TwoStepNVTMTK> base_class_nvt_creator(boost::shared_ptr<SystemDefinition> sysdef,
                                              boost::shared_ptr<ParticleGroup> group,
                                              boost::shared_ptr<ComputeThermo> thermo,
                                              Scalar Q,
                                              Scalar T)
    {
    boost::shared_ptr<VariantConst> T_variant(new VariantConst(T));
    return boost::shared_ptr<TwoStepNVTMTK>(new TwoStepNVTMTK(sysdef, group, thermo, Q, T_variant));
    }

#ifdef ENABLE_CUDA
//! NVTUpdaterGPU factory for the unit tests
boost::shared_ptr<TwoStepNVTMTK> gpu_nvt_creator(boost::shared_ptr<SystemDefinition> sysdef,
                                       boost::shared_ptr<ParticleGroup> group,
                                       boost::shared_ptr<ComputeThermo> thermo,
                                       Scalar Q,
                                       Scalar T)
    {
    boost::shared_ptr<VariantConst> T_variant(new VariantConst(T));
    return boost::shared_ptr<TwoStepNVTMTK>(new TwoStepNVTMTKGPU(sysdef, group, thermo, Q, T_variant));
    }
#endif

void test_nvt_mtk_integrator(boost::shared_ptr<ExecutionConfiguration> exec_conf, twostepnvt_creator nvt_creator)
{
    // initialize random particle system
    Scalar phi_p = 0.2;
    unsigned int N = 2000;
    Scalar L = pow(M_PI/6.0/phi_p*Scalar(N),1.0/3.0);
    BoxDim box_g(L);
    RandomGenerator rand_init(exec_conf, box_g, 12345, 3);
    std::vector<string> types;
    types.push_back("A");
    std::vector<unsigned int> bonds;
    std::vector<string> bond_types;
    rand_init.addGenerator((int)N, boost::shared_ptr<PolymerParticleGenerator>(new PolymerParticleGenerator(exec_conf, 1.0, types, bonds, bonds, bond_types, 100, 3)));
    rand_init.setSeparationRadius("A", .4);

    rand_init.generate();

    boost::shared_ptr<SnapshotSystemData> snap;
    snap = rand_init.getSnapshot();

    boost::shared_ptr<SystemDefinition> sysdef_1(new SystemDefinition(snap, exec_conf));
    boost::shared_ptr<ParticleData> pdata_1 = sysdef_1->getParticleData();
    boost::shared_ptr<ParticleSelector> selector_all_1(new ParticleSelectorTag(sysdef_1, 0, pdata_1->getNGlobal()-1));
    boost::shared_ptr<ParticleGroup> group_all_1(new ParticleGroup(sysdef_1, selector_all_1));

    Scalar r_cut = Scalar(3.0);
    Scalar r_buff = Scalar(0.8);
    boost::shared_ptr<NeighborList> nlist_1(new NeighborListBinned(sysdef_1, r_cut, r_buff));

    nlist_1->setStorageMode(NeighborList::full);
    boost::shared_ptr<PotentialPairLJ> fc_1 = boost::shared_ptr<PotentialPairLJ>(new PotentialPairLJ(sysdef_1, nlist_1));

    fc_1->setRcut(0, 0, r_cut);

    // setup some values for alpha and sigma
    Scalar epsilon = Scalar(1.0);
    Scalar sigma = Scalar(1.0);
    Scalar alpha = Scalar(1.0);
    Scalar lj1 = Scalar(4.0) * epsilon * pow(sigma,Scalar(12.0));
    Scalar lj2 = alpha * Scalar(4.0) * epsilon * pow(sigma,Scalar(6.0));
    fc_1->setParams(0,0,make_scalar2(lj1,lj2));

    Scalar deltaT = Scalar(0.005);
    Scalar Q = Scalar(2.0);
    Scalar T_ref = Scalar(1.5/3.0);
    Scalar tau = sqrt(Q / (Scalar(3.0) * T_ref));
    boost::shared_ptr<IntegratorTwoStep> nvt_1(new IntegratorTwoStep(sysdef_1, deltaT));
    boost::shared_ptr<ComputeThermo> thermo_1 = boost::shared_ptr<ComputeThermo>(new ComputeThermo(sysdef_1,group_all_1));

    boost::shared_ptr<TwoStepNVTMTK> two_step_nvt_1 = nvt_creator(sysdef_1, group_all_1, thermo_1, tau, T_ref);
;
    nvt_1->addIntegrationMethod(two_step_nvt_1);
    nvt_1->addForceCompute(fc_1);

    unsigned int ndof = nvt_1->getNDOF(group_all_1);
    thermo_1->setNDOF(ndof);

    nvt_1->prepRun(0);

    PDataFlags flags;
    flags[pdata_flag::potential_energy] = 1;
    pdata_1->setFlags(flags);

    // equilibrate
    std::cout << "Equilibrating for 10,000 time steps..." << std::endl;
    int i =0;

    for (i=0; i< 10000; i++)
        {
        // get conserved quantity
        nvt_1->update(i);
        if (i % 1000 == 0)
            std::cout << i << std::endl;
        }

    Scalar T_tol = .1;
    Scalar H_tol = .5;

    // conserved quantity
    thermo_1->compute(i);
    Scalar H_ini = thermo_1->getKineticEnergy() + thermo_1->getPotentialEnergy();
    H_ini += nvt_1->getLogValue("nvt_mtk_reservoir_energy", 0);

    std::cout << "Measuring temperature and conserved quantity for another 10,000 time steps..." << std::endl;
    Scalar avg_T(0.0);
    int n_measure_steps = 10000;
    for (i=10000; i< 10000+n_measure_steps; i++)
        {
        // get conserved quantity
        nvt_1->update(i);

        if (i % 1000 == 0)
            std::cout << i << std::endl;

        thermo_1->compute(i+1);

        avg_T += thermo_1->getTemperature();

        Scalar H = thermo_1->getKineticEnergy() + thermo_1->getPotentialEnergy();
        H += nvt_1->getLogValue("nvt_mtk_reservoir_energy", i+1);
        MY_BOOST_CHECK_CLOSE(H_ini,H, H_tol);
        }

    avg_T /= n_measure_steps;
    MY_BOOST_CHECK_CLOSE(T_ref, avg_T, T_tol);
    }

//! Compares the output from one NVTUpdater to another
void nvt_updater_compare_test(twostepnvt_creator nvt_creator1, twostepnvt_creator nvt_creator2, boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    const unsigned int N = 1000;

    // create two identical random particle systems to simulate
    RandomInitializer rand_init(N, Scalar(0.2), Scalar(0.9), "A");
    boost::shared_ptr<SnapshotSystemData> snap;
    rand_init.setSeed(12345);
    snap = rand_init.getSnapshot();

    boost::shared_ptr<SystemDefinition> sysdef1(new SystemDefinition(snap, exec_conf));
    boost::shared_ptr<ParticleData> pdata1 = sysdef1->getParticleData();
    boost::shared_ptr<ParticleSelector> selector_all1(new ParticleSelectorTag(sysdef1, 0, pdata1->getN()-1));
    boost::shared_ptr<ParticleGroup> group_all1(new ParticleGroup(sysdef1, selector_all1));

    boost::shared_ptr<SystemDefinition> sysdef2(new SystemDefinition(snap, exec_conf));
    boost::shared_ptr<ParticleData> pdata2 = sysdef2->getParticleData();
    boost::shared_ptr<ParticleSelector> selector_all2(new ParticleSelectorTag(sysdef2, 0, pdata2->getN()-1));
    boost::shared_ptr<ParticleGroup> group_all2(new ParticleGroup(sysdef2, selector_all2));

    boost::shared_ptr<NeighborList> nlist1(new NeighborList(sysdef1, Scalar(3.0), Scalar(0.8)));
    boost::shared_ptr<NeighborList> nlist2(new NeighborList(sysdef2, Scalar(3.0), Scalar(0.8)));

    boost::shared_ptr<PotentialPairLJ> fc1(new PotentialPairLJ(sysdef1, nlist1));
    fc1->setRcut(0, 0, Scalar(3.0));
    boost::shared_ptr<PotentialPairLJ> fc2(new PotentialPairLJ(sysdef2, nlist2));
    fc2->setRcut(0, 0, Scalar(3.0));


    // setup some values for alpha and sigma
    Scalar epsilon = Scalar(1.0);
    Scalar sigma = Scalar(1.2);
    Scalar alpha = Scalar(0.45);
    Scalar lj1 = Scalar(4.0) * epsilon * pow(sigma,Scalar(12.0));
    Scalar lj2 = alpha * Scalar(4.0) * epsilon * pow(sigma,Scalar(6.0));

    // specify the force parameters
    fc1->setParams(0,0,make_scalar2(lj1,lj2));
    fc2->setParams(0,0,make_scalar2(lj1,lj2));

    boost::shared_ptr<IntegratorTwoStep> nvt1(new IntegratorTwoStep(sysdef1, Scalar(0.005)));
    boost::shared_ptr<ComputeThermo> thermo1(new ComputeThermo(sysdef1, group_all1));
    thermo1->setNDOF(3*N-3);
    boost::shared_ptr<TwoStepNVTMTK> two_step_nvt1 = nvt_creator1(sysdef1, group_all1, thermo1, Scalar(0.5), Scalar(1.2));
    nvt1->addIntegrationMethod(two_step_nvt1);

    boost::shared_ptr<IntegratorTwoStep> nvt2(new IntegratorTwoStep(sysdef2, Scalar(0.005)));
    boost::shared_ptr<ComputeThermo> thermo2(new ComputeThermo(sysdef2, group_all2));
    thermo2->setNDOF(3*N-3);
    boost::shared_ptr<TwoStepNVTMTK> two_step_nvt2 = nvt_creator2(sysdef2, group_all2, thermo2, Scalar(0.5), Scalar(1.2));
    nvt2->addIntegrationMethod(two_step_nvt2);

    nvt1->addForceCompute(fc1);
    nvt2->addForceCompute(fc2);

    nvt1->prepRun(0);
    nvt2->prepRun(0);

    // step for 3 time steps and verify that they are the same
    // we can't do much more because these things are chaotic and diverge quickly
    for (int i = 0; i < 5; i++)
        {
        {
        ArrayHandle<Scalar4> h_pos1(pdata1->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_vel1(pdata1->getVelocities(), access_location::host, access_mode::read);
        ArrayHandle<Scalar3> h_accel1(pdata1->getAccelerations(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_pos2(pdata2->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_vel2(pdata2->getVelocities(), access_location::host, access_mode::read);
        ArrayHandle<Scalar3> h_accel2(pdata2->getAccelerations(), access_location::host, access_mode::read);

        //cout << arrays1.x[100] << " " << arrays2.x[100] << endl;
        Scalar rough_tol = 2.0;

        // check position, velocity and acceleration
        for (unsigned int j = 0; j < N; j++)
            {
            MY_BOOST_CHECK_CLOSE(h_pos1.data[j].x, h_pos2.data[j].x, rough_tol);
            MY_BOOST_CHECK_CLOSE(h_pos1.data[j].y, h_pos2.data[j].y, rough_tol);
            MY_BOOST_CHECK_CLOSE(h_pos1.data[j].z, h_pos2.data[j].z, rough_tol);

            MY_BOOST_CHECK_CLOSE(h_vel1.data[j].x, h_vel2.data[j].x, rough_tol);
            MY_BOOST_CHECK_CLOSE(h_vel1.data[j].y, h_vel2.data[j].y, rough_tol);
            MY_BOOST_CHECK_CLOSE(h_vel1.data[j].z, h_vel2.data[j].z, rough_tol);

            MY_BOOST_CHECK_CLOSE(h_accel1.data[j].x, h_accel2.data[j].x, rough_tol);
            MY_BOOST_CHECK_CLOSE(h_accel1.data[j].y, h_accel2.data[j].y, rough_tol);
            MY_BOOST_CHECK_CLOSE(h_accel1.data[j].z, h_accel2.data[j].z, rough_tol);
            }

        }
        nvt1->update(i);
        nvt2->update(i);
        }
    }

//! Performs a basic equilibration test of TwoStepNVTMTK
BOOST_AUTO_TEST_CASE( TwoStepNVTMTK_basic_test )
    {
    test_nvt_mtk_integrator(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)),bind(base_class_nvt_creator, _1, _2, _3, _4, _5));
    }


#ifdef ENABLE_CUDA
//! Performs a basic equilibration test of TwoStepNVTMTKGPU
BOOST_AUTO_TEST_CASE( TwoStepNVTMTKGPU_basic_test )
    {
    test_nvt_mtk_integrator(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)),bind(gpu_nvt_creator, _1, _2, _3, _4, _5));
    }

//! boost test case for comparing the GPU and CPU NVTUpdaters
BOOST_AUTO_TEST_CASE( TwoStepNVTMTKGPU_comparison_tests)
    {
    twostepnvt_creator nvt_creator_gpu = bind(gpu_nvt_creator, _1, _2, _3, _4, _5);
    twostepnvt_creator nvt_creator = bind(base_class_nvt_creator, _1, _2, _3, _4, _5);
    nvt_updater_compare_test(nvt_creator, nvt_creator_gpu, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

#endif

