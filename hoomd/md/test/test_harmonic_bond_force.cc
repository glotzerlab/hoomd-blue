// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// this include is necessary to get MPI included before anything else to support intel MPI
#include "hoomd/ExecutionConfiguration.h"

#include <iostream>

#include <functional>

#include "hoomd/SnapshotSystemData.h"
#include "hoomd/md/EvaluatorBondHarmonic.h"
#include "hoomd/md/PotentialBond.h"

#ifdef ENABLE_HIP
#include "hoomd/md/PotentialBondGPU.h"
#endif

#include "hoomd/Initializers.h"

using namespace std;
using namespace std::placeholders;
using namespace hoomd;
using namespace hoomd::md;

/*! \file harmonic_bond_force_test.cc
    \brief Implements unit tests for PotentialBondHarmonic and
           PotentialBondHarmonicGPU
    \ingroup unit_tests
*/

typedef class PotentialBond<EvaluatorBondHarmonic, BondData> PotentialBondHarmonic;

#ifdef ENABLE_HIP
typedef class PotentialBondGPU<EvaluatorBondHarmonic, BondData> PotentialBondHarmonicGPU;
#endif

#include "hoomd/test/upp11_config.h"
HOOMD_UP_MAIN();

//! Typedef to make using the std::function factory easier
typedef std::function<std::shared_ptr<PotentialBondHarmonic>(
    std::shared_ptr<SystemDefinition> sysdef)>
    bondforce_creator;

//! Perform some simple functionality tests of any BondForceCompute
void bond_force_basic_tests(bondforce_creator bf_creator,
                            std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    /////////////////////////////////////////////////////////
    // start with the simplest possible test: 2 particles in a huge box with only one bond type
    std::shared_ptr<SystemDefinition> sysdef_2(
        new SystemDefinition(2, BoxDim(1000.0), 1, 1, 0, 0, 0, exec_conf));
    std::shared_ptr<ParticleData> pdata_2 = sysdef_2->getParticleData();

    pdata_2->setPosition(0, make_scalar3(0.0, 0.0, 0.0));
    pdata_2->setPosition(1, make_scalar3(0.9, 0.0, 0.0));
    pdata_2->setFlags(~PDataFlags(0));

    // create the bond force compute to check
    std::shared_ptr<PotentialBondHarmonic> fc_2 = bf_creator(sysdef_2);
    fc_2->setParams(0, harmonic_params(1.5, 0.75));

    // compute the force and check the results
    fc_2->compute(0);
    const GlobalArray<Scalar4>& force_array_1 = fc_2->getForceArray();
    const GlobalArray<Scalar>& virial_array_1 = fc_2->getVirialArray();

        {
        size_t pitch = virial_array_1.getPitch();
        ArrayHandle<Scalar4> h_force_1(force_array_1, access_location::host, access_mode::read);
        ArrayHandle<Scalar> h_virial_1(virial_array_1, access_location::host, access_mode::read);
        // check that the force is correct, it should be 0 since we haven't created any bonds yet
        MY_CHECK_SMALL(h_force_1.data[0].x, tol_small);
        MY_CHECK_SMALL(h_force_1.data[0].y, tol_small);
        MY_CHECK_SMALL(h_force_1.data[0].z, tol_small);
        MY_CHECK_SMALL(h_force_1.data[0].w, tol_small);
        MY_CHECK_SMALL(h_virial_1.data[0 * pitch + 0], tol_small);
        MY_CHECK_SMALL(h_virial_1.data[1 * pitch + 0], tol_small);
        MY_CHECK_SMALL(h_virial_1.data[2 * pitch + 0], tol_small);
        MY_CHECK_SMALL(h_virial_1.data[3 * pitch + 0], tol_small);
        MY_CHECK_SMALL(h_virial_1.data[4 * pitch + 0], tol_small);
        MY_CHECK_SMALL(h_virial_1.data[5 * pitch + 0], tol_small);
        }

    // add a bond and check again
    sysdef_2->getBondData()->addBondedGroup(Bond(0, 0, 1));
    fc_2->compute(1);

        {
        // this time there should be a force
        const GlobalArray<Scalar4>& force_array_2 = fc_2->getForceArray();
        const GlobalArray<Scalar>& virial_array_2 = fc_2->getVirialArray();
        size_t pitch = virial_array_2.getPitch();
        ArrayHandle<Scalar4> h_force_2(force_array_2, access_location::host, access_mode::read);
        ArrayHandle<Scalar> h_virial_2(virial_array_2, access_location::host, access_mode::read);
        MY_CHECK_CLOSE(h_force_2.data[0].x, 0.225, tol);
        MY_CHECK_SMALL(h_force_2.data[0].y, tol_small);
        MY_CHECK_SMALL(h_force_2.data[0].z, tol_small);
        MY_CHECK_CLOSE(h_force_2.data[0].w, 0.0084375, tol);
        MY_CHECK_CLOSE(Scalar(1. / 3.)
                           * (h_virial_2.data[0 * pitch + 0] + h_virial_2.data[3 * pitch + 0]
                              + h_virial_2.data[5 * pitch + 0]),
                       -0.03375,
                       tol);

        // check that the two forces are negatives of each other
        MY_CHECK_CLOSE(h_force_2.data[0].x, -h_force_2.data[1].x, tol);
        MY_CHECK_CLOSE(h_force_2.data[0].y, -h_force_2.data[1].y, tol);
        MY_CHECK_CLOSE(h_force_2.data[0].z, -h_force_2.data[1].z, tol);
        MY_CHECK_CLOSE(h_force_2.data[0].w, h_force_2.data[1].w, tol);
        MY_CHECK_CLOSE(Scalar(1. / 3.)
                           * (h_virial_2.data[0 * pitch + 1] + h_virial_2.data[3 * pitch + 1]
                              + h_virial_2.data[5 * pitch + 1]),
                       -0.03375,
                       tol);
        }

        // rearrange the two particles in memory and see if they are properly updated
        {
        ArrayHandle<Scalar4> h_pos(pdata_2->getPositions(),
                                   access_location::host,
                                   access_mode::readwrite);
        ArrayHandle<unsigned int> h_tag(pdata_2->getTags(),
                                        access_location::host,
                                        access_mode::readwrite);
        ArrayHandle<unsigned int> h_rtag(pdata_2->getRTags(),
                                         access_location::host,
                                         access_mode::readwrite);

        h_pos.data[0].x = Scalar(0.9);
        h_pos.data[1].x = Scalar(0.0);
        h_tag.data[0] = 1;
        h_tag.data[1] = 0;
        h_rtag.data[0] = 1;
        h_rtag.data[1] = 0;
        }

    // notify that we made the sort
    pdata_2->notifyParticleSort();
    // recompute at the same timestep, the forces should still be updated
    fc_2->compute(1);

        {
        // this time there should be a force
        const GlobalArray<Scalar4>& force_array_3 = fc_2->getForceArray();
        const GlobalArray<Scalar>& virial_array_3 = fc_2->getVirialArray();
        ArrayHandle<Scalar4> h_force_3(force_array_3, access_location::host, access_mode::read);
        ArrayHandle<Scalar> h_virial_3(virial_array_3, access_location::host, access_mode::read);
        MY_CHECK_CLOSE(h_force_3.data[0].x, -0.225, tol);
        MY_CHECK_CLOSE(h_force_3.data[1].x, 0.225, tol);
        }

    // check r=r_0 behavior
    pdata_2->setPosition(0, make_scalar3(0.0, 0.0, 0.0));
    pdata_2->setPosition(1, make_scalar3(0.75, 0.0, 0.0));

    fc_2->compute(2);

        {
        // the force should be zero
        const GlobalArray<Scalar4>& force_array_4 = fc_2->getForceArray();
        const GlobalArray<Scalar>& virial_array_4 = fc_2->getVirialArray();
        ArrayHandle<Scalar4> h_force_4(force_array_4, access_location::host, access_mode::read);
        ArrayHandle<Scalar> h_virial_4(virial_array_4, access_location::host, access_mode::read);
        MY_CHECK_SMALL(h_force_4.data[0].x, tol_small);
        MY_CHECK_SMALL(h_force_4.data[1].x, tol_small);
        }

    ////////////////////////////////////////////////////////////////////
    // now, lets do a more thorough test and include boundary conditions
    // there are way too many permutations to test here, so I will simply
    // test +x, -x, +y, -y, +z, and -z independently
    // build a 6 particle system with particles across each boundary
    // also test more than one type of bond
    std::shared_ptr<SystemDefinition> sysdef_6(
        new SystemDefinition(6, BoxDim(20.0, 40.0, 60.0), 1, 3, 0, 0, 0, exec_conf));
    std::shared_ptr<ParticleData> pdata_6 = sysdef_6->getParticleData();
    pdata_6->setFlags(~PDataFlags(0));

    pdata_6->setPosition(0, make_scalar3(-9.6, 0.0, 0.0));
    pdata_6->setPosition(1, make_scalar3(9.6, 0.0, 0.0));
    pdata_6->setPosition(2, make_scalar3(0.0, -19.6, 0.0));
    pdata_6->setPosition(3, make_scalar3(0.0, 19.6, 0.0));
    pdata_6->setPosition(4, make_scalar3(0.0, 0.0, -29.6));
    pdata_6->setPosition(5, make_scalar3(0.0, 0.0, 29.6));

    std::shared_ptr<PotentialBondHarmonic> fc_6 = bf_creator(sysdef_6);
    fc_6->setParams(0, harmonic_params(1.5, 0.75));
    fc_6->setParams(1, harmonic_params(2.0 * 1.5, 0.75));
    fc_6->setParams(2, harmonic_params(1.5, 0.5));

    sysdef_6->getBondData()->addBondedGroup(Bond(0, 0, 1));
    sysdef_6->getBondData()->addBondedGroup(Bond(1, 2, 3));
    sysdef_6->getBondData()->addBondedGroup(Bond(2, 4, 5));

    fc_6->compute(0);

        {
        // check that the forces are correctly computed
        const GlobalArray<Scalar4>& force_array_5 = fc_6->getForceArray();
        const GlobalArray<Scalar>& virial_array_5 = fc_6->getVirialArray();
        size_t pitch = virial_array_5.getPitch();
        ArrayHandle<Scalar4> h_force_5(force_array_5, access_location::host, access_mode::read);
        ArrayHandle<Scalar> h_virial_5(virial_array_5, access_location::host, access_mode::read);
        MY_CHECK_CLOSE(h_force_5.data[0].x, -0.075, tol);
        MY_CHECK_SMALL(h_force_5.data[0].y, tol_small);
        MY_CHECK_SMALL(h_force_5.data[0].z, tol_small);
        MY_CHECK_CLOSE(h_force_5.data[0].w, 9.375e-4, tol);
        MY_CHECK_CLOSE(Scalar(1. / 3.)
                           * (h_virial_5.data[0 * pitch + 0] + h_virial_5.data[3 * pitch + 0]
                              + h_virial_5.data[5 * pitch + 0]),
                       -0.01,
                       tol);

        MY_CHECK_CLOSE(h_force_5.data[1].x, 0.075, tol);
        MY_CHECK_SMALL(h_force_5.data[1].y, tol_small);
        MY_CHECK_SMALL(h_force_5.data[1].z, tol_small);
        MY_CHECK_CLOSE(h_force_5.data[1].w, 9.375e-4, tol);
        MY_CHECK_CLOSE(Scalar(1. / 3.)
                           * (h_virial_5.data[0 * pitch + 1] + h_virial_5.data[3 * pitch + 1]
                              + h_virial_5.data[5 * pitch + 1]),
                       -0.01,
                       tol);

        MY_CHECK_SMALL(h_force_5.data[2].x, tol_small);
        MY_CHECK_CLOSE(h_force_5.data[2].y, -0.075 * 2.0, tol);
        MY_CHECK_SMALL(h_force_5.data[2].z, tol_small);
        MY_CHECK_CLOSE(h_force_5.data[2].w, 9.375e-4 * 2.0, tol);
        MY_CHECK_CLOSE(Scalar(1. / 3.)
                           * (h_virial_5.data[0 * pitch + 2] + h_virial_5.data[3 * pitch + 2]
                              + h_virial_5.data[5 * pitch + 2]),
                       -0.02,
                       tol);

        MY_CHECK_SMALL(h_force_5.data[3].x, tol_small);
        MY_CHECK_CLOSE(h_force_5.data[3].y, 0.075 * 2.0, tol);
        MY_CHECK_SMALL(h_force_5.data[3].z, tol_small);
        MY_CHECK_CLOSE(h_force_5.data[3].w, 9.375e-4 * 2.0, tol);
        MY_CHECK_CLOSE(Scalar(1. / 3.)
                           * (h_virial_5.data[0 * pitch + 3] + h_virial_5.data[3 * pitch + 3]
                              + h_virial_5.data[5 * pitch + 3]),
                       -0.02,
                       tol);

        MY_CHECK_SMALL(h_force_5.data[4].x, tol_small);
        MY_CHECK_SMALL(h_force_5.data[4].y, tol_small);
        MY_CHECK_CLOSE(h_force_5.data[4].z, -0.45, tol);
        MY_CHECK_CLOSE(h_force_5.data[4].w, 0.03375, tol);
        MY_CHECK_CLOSE(Scalar(1. / 3.)
                           * (h_virial_5.data[0 * pitch + 4] + h_virial_5.data[3 * pitch + 4]
                              + h_virial_5.data[5 * pitch + 4]),
                       -0.06,
                       tol);

        MY_CHECK_SMALL(h_force_5.data[5].x, tol_small);
        MY_CHECK_SMALL(h_force_5.data[5].y, tol_small);
        MY_CHECK_CLOSE(h_force_5.data[5].z, 0.45, tol);
        MY_CHECK_CLOSE(h_force_5.data[5].w, 0.03375, tol);
        MY_CHECK_CLOSE(Scalar(1. / 3.)
                           * (h_virial_5.data[0 * pitch + 5] + h_virial_5.data[3 * pitch + 5]
                              + h_virial_5.data[5 * pitch + 5]),
                       -0.06,
                       tol);
        }

    // one more test: this one will test two things:
    // 1) That the forces are computed correctly even if the particles are rearranged in memory
    // and 2) That two forces can add to the same particle
    std::shared_ptr<SystemDefinition> sysdef_4(
        new SystemDefinition(4, BoxDim(100.0, 100.0, 100.0), 1, 1, 0, 0, 0, exec_conf));
    std::shared_ptr<ParticleData> pdata_4 = sysdef_4->getParticleData();
    pdata_4->setFlags(~PDataFlags(0));

        {
        ArrayHandle<Scalar4> h_pos(pdata_4->getPositions(),
                                   access_location::host,
                                   access_mode::readwrite);
        ArrayHandle<unsigned int> h_tag(pdata_4->getTags(),
                                        access_location::host,
                                        access_mode::readwrite);
        ArrayHandle<unsigned int> h_rtag(pdata_4->getRTags(),
                                         access_location::host,
                                         access_mode::readwrite);

        // make a square of particles
        h_pos.data[0].x = 0.0;
        h_pos.data[0].y = 0.0;
        h_pos.data[0].z = 0.0;
        h_pos.data[1].x = 1.0;
        h_pos.data[1].y = 0;
        h_pos.data[1].z = 0.0;
        h_pos.data[2].x = 0;
        h_pos.data[2].y = 1.0;
        h_pos.data[2].z = 0.0;
        h_pos.data[3].x = 1.0;
        h_pos.data[3].y = 1.0;
        h_pos.data[3].z = 0.0;

        h_tag.data[0] = 2;
        h_tag.data[1] = 3;
        h_tag.data[2] = 0;
        h_tag.data[3] = 1;
        h_rtag.data[h_tag.data[0]] = 0;
        h_rtag.data[h_tag.data[1]] = 1;
        h_rtag.data[h_tag.data[2]] = 2;
        h_rtag.data[h_tag.data[3]] = 3;
        }

    // build the bond force compute and try it out
    std::shared_ptr<PotentialBondHarmonic> fc_4 = bf_creator(sysdef_4);
    fc_4->setParams(0, harmonic_params(1.5, 1.75));
    // only add bonds on the left, top, and bottom of the square
    sysdef_4->getBondData()->addBondedGroup(Bond(0, 2, 3));
    sysdef_4->getBondData()->addBondedGroup(Bond(0, 2, 0));
    sysdef_4->getBondData()->addBondedGroup(Bond(0, 0, 1));

    fc_4->compute(0);

        {
        const GlobalArray<Scalar4>& force_array_6 = fc_4->getForceArray();
        const GlobalArray<Scalar>& virial_array_6 = fc_4->getVirialArray();
        size_t pitch = virial_array_6.getPitch();
        ArrayHandle<Scalar4> h_force_6(force_array_6, access_location::host, access_mode::read);
        ArrayHandle<Scalar> h_virial_6(virial_array_6, access_location::host, access_mode::read);
        // the right two particles shoul only have a force pulling them right
        MY_CHECK_CLOSE(h_force_6.data[1].x, 1.125, tol);
        MY_CHECK_SMALL(h_force_6.data[1].y, tol_small);
        MY_CHECK_SMALL(h_force_6.data[1].z, tol_small);
        MY_CHECK_CLOSE(h_force_6.data[1].w, 0.2109375, tol);
        MY_CHECK_CLOSE(Scalar(1. / 3.)
                           * (h_virial_6.data[0 * pitch + 1] + h_virial_6.data[3 * pitch + 1]
                              + h_virial_6.data[5 * pitch + 1]),
                       0.1875,
                       tol);

        MY_CHECK_CLOSE(h_force_6.data[3].x, 1.125, tol);
        MY_CHECK_SMALL(h_force_6.data[3].y, tol_small);
        MY_CHECK_SMALL(h_force_6.data[3].z, tol_small);
        MY_CHECK_CLOSE(h_force_6.data[3].w, 0.2109375, tol);
        MY_CHECK_CLOSE(Scalar(1. / 3.)
                           * (h_virial_6.data[0 * pitch + 3] + h_virial_6.data[3 * pitch + 3]
                              + h_virial_6.data[5 * pitch + 3]),
                       0.1875,
                       tol);

        // the bottom left particle should have a force pulling down and to the left
        MY_CHECK_CLOSE(h_force_6.data[0].x, -1.125, tol);
        MY_CHECK_CLOSE(h_force_6.data[0].y, -1.125, tol);
        MY_CHECK_SMALL(h_force_6.data[0].z, tol_small);
        MY_CHECK_CLOSE(h_force_6.data[0].w, 0.421875, tol);
        MY_CHECK_CLOSE(Scalar(1. / 3.)
                           * (h_virial_6.data[0 * pitch + 0] + h_virial_6.data[3 * pitch + 0]
                              + h_virial_6.data[5 * pitch + 0]),
                       0.375,
                       tol);

        // and the top left particle should have a force pulling up and to the left
        MY_CHECK_CLOSE(h_force_6.data[2].x, -1.125, tol);
        MY_CHECK_CLOSE(h_force_6.data[2].y, 1.125, tol);
        MY_CHECK_SMALL(h_force_6.data[2].z, tol_small);
        MY_CHECK_CLOSE(h_force_6.data[2].w, 0.421875, tol);
        MY_CHECK_CLOSE(Scalar(1. / 3.)
                           * (h_virial_6.data[0 * pitch + 2] + h_virial_6.data[3 * pitch + 2]
                              + h_virial_6.data[5 * pitch + 2]),
                       0.375,
                       tol);
        }
    }

//! Compares the output of two PotentialBondHarmonics
void bond_force_comparison_tests(bondforce_creator bf_creator1,
                                 bondforce_creator bf_creator2,
                                 std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    const unsigned int N = 1000;

    // create a particle system to sum forces on
    // just randomly place particles. We don't really care how huge the bond forces get: this is
    // just a unit test
    RandomInitializer rand_init(N, Scalar(0.2), Scalar(0.9), "A");
    std::shared_ptr<SnapshotSystemData<Scalar>> snap = rand_init.getSnapshot();
    snap->bond_data.type_mapping.push_back("A");
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));
    std::shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    pdata->setFlags(~PDataFlags(0));

    std::shared_ptr<PotentialBondHarmonic> fc1 = bf_creator1(sysdef);
    std::shared_ptr<PotentialBondHarmonic> fc2 = bf_creator2(sysdef);
    fc1->setParams(0, harmonic_params(Scalar(300.0), Scalar(1.6)));
    fc2->setParams(0, harmonic_params(Scalar(300.0), Scalar(1.6)));

    // add bonds
    for (unsigned int i = 0; i < N - 1; i++)
        {
        sysdef->getBondData()->addBondedGroup(Bond(0, i, i + 1));
        }

    // compute the forces
    fc1->compute(0);
    fc2->compute(0);

        // verify that the forces are identical (within roundoff errors)
        {
        const GlobalArray<Scalar4>& force_array_7 = fc1->getForceArray();
        const GlobalArray<Scalar>& virial_array_7 = fc1->getVirialArray();
        size_t pitch = virial_array_7.getPitch();
        ArrayHandle<Scalar4> h_force_7(force_array_7, access_location::host, access_mode::read);
        ArrayHandle<Scalar> h_virial_7(virial_array_7, access_location::host, access_mode::read);
        const GlobalArray<Scalar4>& force_array_8 = fc2->getForceArray();
        const GlobalArray<Scalar>& virial_array_8 = fc2->getVirialArray();
        ArrayHandle<Scalar4> h_force_8(force_array_8, access_location::host, access_mode::read);
        ArrayHandle<Scalar> h_virial_8(virial_array_8, access_location::host, access_mode::read);

        // compare average deviation between the two computes
        double deltaf2 = 0.0;
        double deltape2 = 0.0;
        double deltav2[6];
        for (unsigned int i = 0; i < 6; i++)
            deltav2[i] = 0.0;

        for (unsigned int i = 0; i < N; i++)
            {
            deltaf2 += double(h_force_8.data[i].x - h_force_7.data[i].x)
                       * double(h_force_8.data[i].x - h_force_7.data[i].x);
            deltaf2 += double(h_force_8.data[i].y - h_force_7.data[i].y)
                       * double(h_force_8.data[i].y - h_force_7.data[i].y);
            deltaf2 += double(h_force_8.data[i].z - h_force_7.data[i].z)
                       * double(h_force_8.data[i].z - h_force_7.data[i].z);
            deltape2 += double(h_force_8.data[i].w - h_force_7.data[i].w)
                        * double(h_force_8.data[i].w - h_force_7.data[i].w);
            for (unsigned int j = 0; j < 6; j++)
                deltav2[j]
                    += double(h_virial_8.data[j * pitch + i] - h_virial_7.data[j * pitch + i])
                       * double(h_virial_8.data[j * pitch + i] - h_virial_7.data[j * pitch + i]);

            // also check that each individual calculation is somewhat close
            }
        deltaf2 /= double(pdata->getN());
        deltape2 /= double(pdata->getN());
        for (unsigned int j = 0; j < 6; j++)
            deltav2[j] /= double(pdata->getN());
        CHECK_SMALL(deltaf2, double(tol_small));
        CHECK_SMALL(deltape2, double(tol_small));
        CHECK_SMALL(deltav2[0], double(tol_small));
        CHECK_SMALL(deltav2[1], double(tol_small));
        CHECK_SMALL(deltav2[2], double(tol_small));
        CHECK_SMALL(deltav2[3], double(tol_small));
        CHECK_SMALL(deltav2[4], double(tol_small));
        CHECK_SMALL(deltav2[5], double(tol_small));
        }
    }

//! PotentialBondHarmonic creator for bond_force_basic_tests()
std::shared_ptr<PotentialBondHarmonic>
base_class_bf_creator(std::shared_ptr<SystemDefinition> sysdef)
    {
    return std::shared_ptr<PotentialBondHarmonic>(new PotentialBondHarmonic(sysdef));
    }

#ifdef ENABLE_HIP
//! PotentialBondHarmonic creator for bond_force_basic_tests()
std::shared_ptr<PotentialBondHarmonic> gpu_bf_creator(std::shared_ptr<SystemDefinition> sysdef)
    {
    return std::shared_ptr<PotentialBondHarmonic>(new PotentialBondHarmonicGPU(sysdef));
    }
#endif

//! test case for bond forces on the CPU
UP_TEST(PotentialBondHarmonic_basic)
    {
    bondforce_creator bf_creator = bind(base_class_bf_creator, _1);
    bond_force_basic_tests(bf_creator,
                           std::shared_ptr<ExecutionConfiguration>(
                               new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

#ifdef ENABLE_HIP
//! test case for bond forces on the GPU
UP_TEST(PotentialBondHarmonicGPU_basic)
    {
    bondforce_creator bf_creator = bind(gpu_bf_creator, _1);
    bond_force_basic_tests(bf_creator,
                           std::shared_ptr<ExecutionConfiguration>(
                               new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! test case for comparing bond GPU and CPU BondForceComputes
UP_TEST(PotentialBondHarmonicGPU_compare)
    {
    bondforce_creator bf_creator_gpu = bind(gpu_bf_creator, _1);
    bondforce_creator bf_creator = bind(base_class_bf_creator, _1);
    bond_force_comparison_tests(bf_creator,
                                bf_creator_gpu,
                                std::shared_ptr<ExecutionConfiguration>(
                                    new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

#endif
