// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

#include "utils.h"
#include "hoomd/mpcd/CellList.h"
#include "hoomd/mpcd/CellThermoCompute.h"
#ifdef ENABLE_CUDA
#include "hoomd/mpcd/CellThermoComputeGPU.h"
#endif // ENABLE_CUDA

#include "hoomd/SnapshotSystemData.h"
#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN()

//! Test for correct calculation of cell thermo properties for MPCD particles
template<class CT>
void cell_thermo_basic_test(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    std::shared_ptr< SnapshotSystemData<Scalar> > snap( new SnapshotSystemData<Scalar>() );
    snap->global_box = BoxDim(2.0);
    snap->particle_data.type_mapping.push_back("A");
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));

    // place each particle in a different cell, doubling the first cell
    auto mpcd_sys_snap = std::make_shared<mpcd::SystemDataSnapshot>(sysdef);
        {
        auto mpcd_snap = mpcd_sys_snap->particles;
        mpcd_snap->resize(5);

        mpcd_snap->position[0] = vec3<Scalar>(-0.5, -0.5, -0.5);
        mpcd_snap->position[1] = vec3<Scalar>(-0.5, -0.5, -0.5);
        mpcd_snap->position[2] = vec3<Scalar>( 0.5,  0.5,  0.5);
        mpcd_snap->position[3] = vec3<Scalar>( 0.5,  0.5,  0.5);
        mpcd_snap->position[4] = vec3<Scalar>(-0.5,  0.5,  0.5);

        mpcd_snap->velocity[0] = vec3<Scalar>(2.0, 0.0, 0.0);
        mpcd_snap->velocity[1] = vec3<Scalar>(1.0, 0.0, 0.0);
        mpcd_snap->velocity[2] = vec3<Scalar>(0.0, -3.0, 0.0);
        mpcd_snap->velocity[3] = vec3<Scalar>(0.0, 0.0, -5.0);
        mpcd_snap->velocity[4] = vec3<Scalar>(1.0, -1.0, 4.0);
        }
    auto mpcd_sys = std::make_shared<mpcd::SystemData>(mpcd_sys_snap);
    std::shared_ptr<mpcd::ParticleData> pdata_5 = mpcd_sys->getParticleData();

    std::shared_ptr<mpcd::CellList> cl = mpcd_sys->getCellList();
    std::shared_ptr<CT> thermo = std::make_shared<CT>(mpcd_sys);
    AllThermoRequest thermo_req(thermo);
    thermo->compute(0);
        {
        const Index3D ci = cl->getCellIndexer();
        ArrayHandle<double4> h_avg_vel(thermo->getCellVelocities(), access_location::host, access_mode::read);
        ArrayHandle<double3> h_cell_energy(thermo->getCellEnergies(), access_location::host, access_mode::read);

        // Two particle cell (0,0,0)
        // Average velocity, mass
        CHECK_CLOSE(h_avg_vel.data[ci(0,0,0)].x, 1.5, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0,0,0)].y, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0,0,0)].z, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0,0,0)].w, 2.0, tol);
        // energy, temperature (relative to COM), np, flag
        CHECK_CLOSE(h_cell_energy.data[ci(0,0,0)].x, 2.5, tol);
        CHECK_CLOSE(h_cell_energy.data[ci(0,0,0)].y, 2.0*0.5*0.5/3.0, tol);
        UP_ASSERT_EQUAL(__double_as_int(h_cell_energy.data[ci(0,0,0)].z), 2);

        // Two particle cell (1,1,1)
        // Average velocity, mass
        CHECK_CLOSE(h_avg_vel.data[ci(1,1,1)].x, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1,1,1)].y, -1.5, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1,1,1)].z, -2.5, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1,1,1)].w, 2.0, tol);
        // energy, temperature (relative to COM), np, flag
        CHECK_CLOSE(h_cell_energy.data[ci(1,1,1)].x, 17.0, tol);
        CHECK_CLOSE(h_cell_energy.data[ci(1,1,1)].y, 2.0*(1.5*1.5+2.5*2.5)/3.0, tol);
        UP_ASSERT_EQUAL(__double_as_int(h_cell_energy.data[ci(1,1,1)].z), 2);

        // One particle cell (0,1,1)
        // Average velocity, mass
        CHECK_CLOSE(h_avg_vel.data[ci(0,1,1)].x, 1.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0,1,1)].y, -1.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0,1,1)].z, 4.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0,1,1)].w, 1.0, tol);
        // Has kinetic energy, but temperature should be zero
        CHECK_CLOSE(h_cell_energy.data[ci(0,1,1)].x, 9.0, tol);
        CHECK_CLOSE(h_cell_energy.data[ci(0,1,1)].y, 0.0, tol);
        UP_ASSERT_EQUAL(__double_as_int(h_cell_energy.data[ci(0,1,1)].z), 1);

        // Check the net stats of the system
        CHECK_CLOSE(thermo->getNetMomentum().x, 4.0, tol);
        CHECK_CLOSE(thermo->getNetMomentum().y, -4.0, tol);
        CHECK_CLOSE(thermo->getNetMomentum().z, -1.0, tol);
        CHECK_CLOSE(thermo->getNetEnergy(), 28.5, tol);
        CHECK_CLOSE(thermo->getTemperature(), 0.5*(2.0*0.5*0.5/3.0 + 2.0*(1.5*1.5+2.5*2.5)/3.0), tol);
        }

    // increase the mass and make sure that energies depend on mass, but velocities don't
    pdata_5->setMass(4.0);
    thermo->compute(1);
        {
        const Index3D ci = cl->getCellIndexer();
        ArrayHandle<double4> h_avg_vel(thermo->getCellVelocities(), access_location::host, access_mode::read);
        ArrayHandle<double3> h_cell_energy(thermo->getCellEnergies(), access_location::host, access_mode::read);

        // Two particle cell (0,0,0)
        // Average velocity, mass
        CHECK_CLOSE(h_avg_vel.data[ci(0,0,0)].x, 1.5, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0,0,0)].y, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0,0,0)].z, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0,0,0)].w, 8.0, tol);
        // energy, temperature (relative to COM), np, flag
        CHECK_CLOSE(h_cell_energy.data[ci(0,0,0)].x, 4.0*2.5, tol);
        CHECK_CLOSE(h_cell_energy.data[ci(0,0,0)].y, 4.0*2.0*0.5*0.5/3.0, tol);
        UP_ASSERT_EQUAL(__double_as_int(h_cell_energy.data[ci(0,0,0)].z), 2);

        // Two particle cell (1,1,1)
        // Average velocity, mass
        CHECK_CLOSE(h_avg_vel.data[ci(1,1,1)].x, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1,1,1)].y, -1.5, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1,1,1)].z, -2.5, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1,1,1)].w, 8.0, tol);
        // energy, temperature (relative to COM), np, flag
        CHECK_CLOSE(h_cell_energy.data[ci(1,1,1)].x, 4.0*17.0, tol);
        CHECK_CLOSE(h_cell_energy.data[ci(1,1,1)].y, 4.0*2.0*(1.5*1.5+2.5*2.5)/3.0, tol);
        UP_ASSERT_EQUAL(__double_as_int(h_cell_energy.data[ci(1,1,1)].z), 2);

        // One particle cell (0,1,1)
        // Average velocity, mass
        CHECK_CLOSE(h_avg_vel.data[ci(0,1,1)].x, 1.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0,1,1)].y, -1.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0,1,1)].z, 4.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0,1,1)].w, 4.0, tol);
        // Has kinetic energy, but temperature should be zero
        CHECK_CLOSE(h_cell_energy.data[ci(0,1,1)].x, 4.0*9.0, tol);
        CHECK_CLOSE(h_cell_energy.data[ci(0,1,1)].y, 0.0, tol);
        UP_ASSERT_EQUAL(__double_as_int(h_cell_energy.data[ci(0,1,1)].z), 1);

        // Check the net stats of the system
        CHECK_CLOSE(thermo->getNetMomentum().x, 4.0*4.0, tol);
        CHECK_CLOSE(thermo->getNetMomentum().y, 4.0*-4.0, tol);
        CHECK_CLOSE(thermo->getNetMomentum().z, 4.0*-1.0, tol);
        CHECK_CLOSE(thermo->getNetEnergy(), 4.0*28.5, tol);
        CHECK_CLOSE(thermo->getTemperature(), 4.0*0.5*(2.0*0.5*0.5/3.0 + 2.0*(1.5*1.5+2.5*2.5)/3.0), tol);
        }

    // switch a particle into a different cell, and make sure the DOF are reduced accordingly
    pdata_5->setMass(1.0);
        {
        ArrayHandle<Scalar4> h_pos(pdata_5->getPositions(), access_location::host, access_mode::readwrite);
        h_pos.data[2] = make_scalar4(-0.5, -0.5, -0.5, 0.0);
        }
    thermo->compute(2);
        {
        const Index3D ci = cl->getCellIndexer();
        ArrayHandle<double4> h_avg_vel(thermo->getCellVelocities(), access_location::host, access_mode::read);
        ArrayHandle<double3> h_cell_energy(thermo->getCellEnergies(), access_location::host, access_mode::read);

        // Three particle cell (0,0,0)
        // Average velocity, mass
        CHECK_CLOSE(h_avg_vel.data[ci(0,0,0)].x, 1.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0,0,0)].y, -1.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0,0,0)].z, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0,0,0)].w, 3.0, tol);
        // energy, temperature (relative to COM), np, flag
        CHECK_CLOSE(h_cell_energy.data[ci(0,0,0)].x, 7.0, tol);
        CHECK_CLOSE(h_cell_energy.data[ci(0,0,0)].y, (2*1.0*1.0+2*1.0*1.0+2.0*2.0)/6.0, tol);
        UP_ASSERT_EQUAL(__double_as_int(h_cell_energy.data[ci(0,0,0)].z), 3);

        // One particle cell (1,1,1)
        // Average velocity, mass
        CHECK_CLOSE(h_avg_vel.data[ci(1,1,1)].x, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1,1,1)].y, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1,1,1)].z, -5.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1,1,1)].w, 1.0, tol);
        // energy, temperature (relative to COM), np, flag
        CHECK_CLOSE(h_cell_energy.data[ci(1,1,1)].x, 12.5, tol);
        CHECK_CLOSE(h_cell_energy.data[ci(1,1,1)].y, 0.0, tol);
        UP_ASSERT_EQUAL(__double_as_int(h_cell_energy.data[ci(1,1,1)].z), 1);

        // One particle cell (0,1,1)
        // Average velocity, mass
        CHECK_CLOSE(h_avg_vel.data[ci(0,1,1)].x, 1.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0,1,1)].y, -1.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0,1,1)].z, 4.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0,1,1)].w, 1.0, tol);
        // Has kinetic energy, but temperature should be zero
        CHECK_CLOSE(h_cell_energy.data[ci(0,1,1)].x, 9.0, tol);
        CHECK_CLOSE(h_cell_energy.data[ci(0,1,1)].y, 0.0, tol);
        UP_ASSERT_EQUAL(__double_as_int(h_cell_energy.data[ci(0,1,1)].z), 1);

        // Check the net stats of the system, only average temperature should change now
        CHECK_CLOSE(thermo->getNetMomentum().x, 4.0, tol);
        CHECK_CLOSE(thermo->getNetMomentum().y, -4.0, tol);
        CHECK_CLOSE(thermo->getNetMomentum().z, -1.0, tol);
        CHECK_CLOSE(thermo->getNetEnergy(), 28.5, tol);
        CHECK_CLOSE(thermo->getTemperature(), (2*1.0*1.0+2*1.0*1.0+2.0*2.0)/6.0, tol);
        }
    }

//! Test for correct calculation of cell thermo properties with embedded particles
template<class CT>
void cell_thermo_embed_test(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    std::shared_ptr< SnapshotSystemData<Scalar> > snap( new SnapshotSystemData<Scalar>() );
    snap->global_box = BoxDim(2.0);
        {
        SnapshotParticleData<Scalar>& pdata_snap = snap->particle_data;
        pdata_snap.type_mapping.push_back("A");
        pdata_snap.resize(4);

        pdata_snap.pos[0] = vec3<Scalar>(-0.5, -0.5, -0.5);
        pdata_snap.pos[1] = vec3<Scalar>(-0.5, -0.5,  0.5);
        pdata_snap.pos[2] = vec3<Scalar>( 0.5, -0.5,  0.5);
        pdata_snap.pos[3] = vec3<Scalar>( 0.5,  0.5,  0.5);

        pdata_snap.mass[0] = 3.0;
        pdata_snap.mass[1] = 2.0;
        pdata_snap.mass[2] = 4.0;
        pdata_snap.mass[3] = 5.0;

        pdata_snap.vel[0] = vec3<Scalar>(-2.0, 0.0, 0.0);
        pdata_snap.vel[1] = vec3<Scalar>(1.0, 0.0, 0.0);
        pdata_snap.vel[2] = vec3<Scalar>(0.0, -3.0, 0.0);
        pdata_snap.vel[3] = vec3<Scalar>(0.0, 0.0, -5.0);
        }
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));

    auto mpcd_sys_snap = std::make_shared<mpcd::SystemDataSnapshot>(sysdef);
        {
        // MPCD particles have same velocity, different masses though
        auto mpcd_snap = mpcd_sys_snap->particles;
        mpcd_snap->resize(4);

        mpcd_snap->position[0] = vec3<Scalar>(-0.5, -0.5, -0.5);
        mpcd_snap->position[1] = vec3<Scalar>(-0.5, -0.5,  0.5);
        mpcd_snap->position[2] = vec3<Scalar>( 0.5, -0.5,  0.5);
        mpcd_snap->position[3] = vec3<Scalar>( 0.5,  0.5,  0.5);

        mpcd_snap->velocity[0] = vec3<Scalar>(2.0, 0.0, 0.0);
        mpcd_snap->velocity[1] = vec3<Scalar>(1.0, 0.0, 0.0);
        mpcd_snap->velocity[2] = vec3<Scalar>(0.0, -3.0, 0.0);
        mpcd_snap->velocity[3] = vec3<Scalar>(0.0, 0.0, -5.0);
        }
    auto mpcd_sys = std::make_shared<mpcd::SystemData>(mpcd_sys_snap);

    std::shared_ptr<ParticleData> embed_pdata = sysdef->getParticleData();
    std::shared_ptr<ParticleSelector> selector(new ParticleSelectorAll(sysdef));
    std::shared_ptr<ParticleGroup> group(new ParticleGroup(sysdef, selector));

    std::shared_ptr<mpcd::CellList> cl = mpcd_sys->getCellList();
    cl->setEmbeddedGroup(group);
    std::shared_ptr<CT> thermo = std::make_shared<CT>(mpcd_sys);
    AllThermoRequest thermo_req(thermo);
    thermo->compute(0);
        {
        const Index3D ci = cl->getCellIndexer();
        ArrayHandle<double4> h_avg_vel(thermo->getCellVelocities(), access_location::host, access_mode::read);
        ArrayHandle<double3> h_cell_energy(thermo->getCellEnergies(), access_location::host, access_mode::read);

        // Cell (0,0,0)
        // Average velocity, mass
        CHECK_CLOSE(h_avg_vel.data[ci(0,0,0)].x,-1.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0,0,0)].y, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0,0,0)].z, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0,0,0)].w, 4.0, tol);
        // energy, temperature (relative to COM), np, flag
        CHECK_CLOSE(h_cell_energy.data[ci(0,0,0)].x, 8.0, tol);
        CHECK_CLOSE(h_cell_energy.data[ci(0,0,0)].y, 4.0, tol);
        UP_ASSERT_EQUAL(__double_as_int(h_cell_energy.data[ci(0,0,0)].z), 2);

        // Cell (0,0,1)
        // Average velocity, mass
        CHECK_CLOSE(h_avg_vel.data[ci(0,0,1)].x, 1.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0,0,1)].y, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0,0,1)].z, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0,0,1)].w, 3.0, tol);
        // energy, temperature (relative to COM), np, flag
        CHECK_CLOSE(h_cell_energy.data[ci(0,0,1)].x, 1.5, tol);
        CHECK_CLOSE(h_cell_energy.data[ci(0,0,1)].y, 0.0, tol);
        UP_ASSERT_EQUAL(__double_as_int(h_cell_energy.data[ci(0,0,1)].z), 2);

        // Cell (1,0,1)
        // Average velocity, mass
        CHECK_CLOSE(h_avg_vel.data[ci(1,0,1)].x, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1,0,1)].y, -3.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1,0,1)].z, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1,0,1)].w, 5.0, tol);
        // Has kinetic energy, but temperature should be zero
        CHECK_CLOSE(h_cell_energy.data[ci(1,0,1)].x, 22.5, tol);
        CHECK_CLOSE(h_cell_energy.data[ci(1,0,1)].y, 0.0, tol);
        UP_ASSERT_EQUAL(__double_as_int(h_cell_energy.data[ci(1,0,1)].z), 2);

        // Cell (1,1,1)
        // Average velocity, mass
        CHECK_CLOSE(h_avg_vel.data[ci(1,1,1)].x, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1,1,1)].y, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1,1,1)].z, -5.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1,1,1)].w, 6.0, tol);
        // energy, temperature (relative to COM), np, flag
        CHECK_CLOSE(h_cell_energy.data[ci(1,1,1)].x, 75.0, tol);
        CHECK_CLOSE(h_cell_energy.data[ci(1,1,1)].y, 0.0, tol);
        UP_ASSERT_EQUAL(__double_as_int(h_cell_energy.data[ci(1,1,1)].z), 2);

        // Check the net stats of the system
        CHECK_CLOSE(thermo->getNetMomentum().x, -1.0, tol);
        CHECK_CLOSE(thermo->getNetMomentum().y, -15.0, tol);
        CHECK_CLOSE(thermo->getNetMomentum().z, -30.0, tol);
        CHECK_CLOSE(thermo->getNetEnergy(), 107.0, tol);
        CHECK_CLOSE(thermo->getTemperature(), (4.0 + 0.0 + 0.0 + 0.0)/4., tol);
        }
    }

UP_TEST( mpcd_cell_thermo_basic )
    {
    cell_thermo_basic_test<mpcd::CellThermoCompute>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
UP_TEST( mpcd_cell_thermo_embed )
    {
    cell_thermo_embed_test<mpcd::CellThermoCompute>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

#ifdef ENABLE_CUDA
UP_TEST( mpcd_cell_thermo_basic_gpu )
    {
    cell_thermo_basic_test<mpcd::CellThermoComputeGPU>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
UP_TEST( mpcd_cell_thermo_embed_gpu )
    {
    cell_thermo_embed_test<mpcd::CellThermoComputeGPU>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
#endif // ENABLE_CUDA
