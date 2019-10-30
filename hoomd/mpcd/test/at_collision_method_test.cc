// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

#include "utils.h"
#include "hoomd/mpcd/ATCollisionMethod.h"
#ifdef ENABLE_CUDA
#include "hoomd/mpcd/ATCollisionMethodGPU.h"
#endif // ENABLE_CUDA

#include "hoomd/SnapshotSystemData.h"
#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN()

//! Test for basic setup and functionality of the SRD collision method
template<class CM>
void at_collision_method_basic_test(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    std::shared_ptr< SnapshotSystemData<Scalar> > snap( new SnapshotSystemData<Scalar>() );
    snap->global_box = BoxDim(2.0);
    snap->particle_data.type_mapping.push_back("A");
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));

    // 4 particle system
    auto mpcd_sys_snap = std::make_shared<mpcd::SystemDataSnapshot>(sysdef);
        {
        auto mpcd_snap = mpcd_sys_snap->particles;
        mpcd_snap->resize(4);

        mpcd_snap->position[0] = vec3<Scalar>(-0.6, -0.6, -0.6);
        mpcd_snap->position[1] = vec3<Scalar>(-0.6, -0.6, -0.6);
        mpcd_snap->position[2] = vec3<Scalar>(0.5, 0.5, 0.5);
        mpcd_snap->position[3] = vec3<Scalar>(0.5, 0.5, 0.5);

        mpcd_snap->velocity[0] = vec3<Scalar>(2.0, 0.0, 0.0);
        mpcd_snap->velocity[1] = vec3<Scalar>(1.0, 0.0, 0.0);
        mpcd_snap->velocity[2] = vec3<Scalar>(5.0, -2.0, 3.0);
        mpcd_snap->velocity[3] = vec3<Scalar>(-1.0, 2.0, -5.0);
        }
    // Save original momentum for comparison as well
    const Scalar3 orig_mom = make_scalar3(7.0, 0.0, -2.0);
    const Scalar orig_energy = 36.5;
    const Scalar orig_temp = 9.75;

    // initialize system and collision method
    auto mpcd_sys = std::make_shared<mpcd::SystemData>(mpcd_sys_snap);
    std::shared_ptr<mpcd::ParticleData> pdata_4 = mpcd_sys->getParticleData();

    // thermos and temperature variant
    auto thermo = std::make_shared<mpcd::CellThermoCompute>(mpcd_sys);
    AllThermoRequest thermo_req(thermo);

    auto rand_thermo = std::make_shared<mpcd::CellThermoCompute>(mpcd_sys);
    std::shared_ptr<::Variant> T = std::make_shared<::VariantConst>(1.5);

    std::shared_ptr<mpcd::ATCollisionMethod> collide = std::make_shared<CM>(mpcd_sys, 0, 2, 1, 42, thermo, rand_thermo, T);
    collide->enableGridShifting(false);

    // nothing should happen on the first step
    UP_ASSERT(!collide->peekCollide(0));
    collide->collide(0);
        {
        // all net properties should still match inputs
        thermo->compute(0);
        const Scalar3 mom = thermo->getNetMomentum();
        CHECK_CLOSE(mom.x, orig_mom.x, tol_small);
        CHECK_CLOSE(mom.y, orig_mom.y, tol_small);
        CHECK_CLOSE(mom.z, orig_mom.z, tol_small);

        const Scalar energy = thermo->getNetEnergy();
        CHECK_CLOSE(energy, orig_energy, tol_small);

        const Scalar temp = thermo->getTemperature();
        CHECK_CLOSE(temp, orig_temp, tol_small);
        }

    // update should happen on the second step
    UP_ASSERT(collide->peekCollide(1));
    collide->collide(1);

    // ensure that momentum was conserved
    thermo->compute(2);
        {
        const Scalar3 mom = thermo->getNetMomentum();
        CHECK_CLOSE(mom.x, orig_mom.x, tol_small);
        CHECK_SMALL(mom.y, tol_small);
        CHECK_CLOSE(mom.z, orig_mom.z, tol_small);
        }

    // perform the collision many times, and ensure that the average temperature is correct
    const unsigned int num_sample = 50000;
    double Tavg = 0.0;
    for (unsigned int timestep=2; timestep < 2+num_sample; ++timestep)
        {
        thermo->compute(timestep);
        Tavg += thermo->getTemperature();

        collide->collide(timestep);
        }
    Tavg /= num_sample;
    CHECK_CLOSE(Tavg, 1.5, 0.02);
    }

//! Test that embedding a particle produces valid values (not a rigorous test of physics of embedding)
template<class CM>
void at_collision_method_embed_test(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    std::shared_ptr< SnapshotSystemData<Scalar> > snap( new SnapshotSystemData<Scalar>() );
    snap->global_box = BoxDim(2.0);
    snap->particle_data.type_mapping.push_back("A");
        {
        SnapshotParticleData<Scalar>& pdata_snap = snap->particle_data;
        pdata_snap.resize(1);
        pdata_snap.pos[0] = vec3<Scalar>(-0.6, -0.6, -0.6);
        pdata_snap.vel[0] = vec3<Scalar>(1.0, 2.0, 3.0);
        pdata_snap.mass[0] = 2.0;
        }
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));

    // 4 particle system
    auto mpcd_sys_snap = std::make_shared<mpcd::SystemDataSnapshot>(sysdef);
        {
        auto mpcd_snap = mpcd_sys_snap->particles;
        mpcd_snap->resize(4);

        mpcd_snap->position[0] = vec3<Scalar>(-0.6, -0.6, -0.6);
        mpcd_snap->position[1] = vec3<Scalar>(-0.6, -0.6, -0.6);
        mpcd_snap->position[2] = vec3<Scalar>(0.5, 0.5, 0.5);
        mpcd_snap->position[3] = vec3<Scalar>(0.5, 0.5, 0.5);

        mpcd_snap->velocity[0] = vec3<Scalar>(2.0, 0.0, 0.0);
        mpcd_snap->velocity[1] = vec3<Scalar>(1.0, 0.0, 0.0);
        mpcd_snap->velocity[2] = vec3<Scalar>(5.0, -2.0, 3.0);
        mpcd_snap->velocity[3] = vec3<Scalar>(-1.0, 2.0, -5.0);
        }

    // initialize system and collision method
    auto mpcd_sys = std::make_shared<mpcd::SystemData>(mpcd_sys_snap);
    std::shared_ptr<mpcd::ParticleData> pdata_4 = mpcd_sys->getParticleData();

    // thermos and temperature variant
    auto thermo = std::make_shared<mpcd::CellThermoCompute>(mpcd_sys);
    AllThermoRequest thermo_req(thermo);

    auto rand_thermo = std::make_shared<mpcd::CellThermoCompute>(mpcd_sys);
    std::shared_ptr<::Variant> T = std::make_shared<::VariantConst>(1.5);

    std::shared_ptr<mpcd::ATCollisionMethod> collide = std::make_shared<CM>(mpcd_sys, 0, 1, -1, 42, thermo, rand_thermo, T);
    collide->enableGridShifting(false);

    // embed the particle group into the mpcd system
    std::shared_ptr<ParticleSelector> selector_one(new ParticleSelectorAll(sysdef));
    std::shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_one));
    collide->setEmbeddedGroup(group_all);

    // Save original momentum for comparison as well
    thermo->compute(0);
    const Scalar3 orig_mom = thermo->getNetMomentum();
    collide->collide(0);
        {
        // velocity should be different now, but the mass should stay the same
        ArrayHandle<Scalar4> h_vel(sysdef->getParticleData()->getVelocities(), access_location::host, access_mode::read);
        UP_ASSERT(h_vel.data[0].x != 1.0);
        UP_ASSERT(h_vel.data[0].y != 2.0);
        UP_ASSERT(h_vel.data[0].z != 3.0);
        CHECK_CLOSE(h_vel.data[0].w, 2.0, tol_small);
        }

    // compute properties after collision
    thermo->compute(1);
    Scalar3 mom = thermo->getNetMomentum();
    CHECK_CLOSE(orig_mom.x, mom.x, tol_small);
    CHECK_CLOSE(orig_mom.y, mom.y, tol_small);
    CHECK_CLOSE(orig_mom.z, mom.z, tol_small);
    }

//! basic test case for MPCD ATCollisionMethod class
UP_TEST( at_collision_method_basic )
    {
    at_collision_method_basic_test<mpcd::ATCollisionMethod>(std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU));
    }
//! test embedding of particles into the MPCD ATCollisionMethod class
UP_TEST( at_collision_method_embed )
    {
    at_collision_method_embed_test<mpcd::ATCollisionMethod>(std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU));
    }
#ifdef ENABLE_CUDA
//! basic test case for MPCD ATCollisionMethodGPU class
UP_TEST( at_collision_method_basic_gpu )
    {
    at_collision_method_basic_test<mpcd::ATCollisionMethodGPU>(std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU));
    }
//! test embedding of particles into the MPCD ATCollisionMethodGPU class
UP_TEST( at_collision_method_embed_gpu )
    {
    at_collision_method_embed_test<mpcd::ATCollisionMethodGPU>(std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU));
    }
#endif // ENABLE_CUDA
