// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/mpcd/CellList.h"
#include "hoomd/mpcd/CellThermoCompute.h"
#include "utils.h"
#ifdef ENABLE_HIP
#include "hoomd/mpcd/CellThermoComputeGPU.h"
#endif // ENABLE_HIP

#include "hoomd/SnapshotSystemData.h"
#include "hoomd/filter/ParticleFilterAll.h"
#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN()

using namespace hoomd;

//! Test for correct calculation of cell thermo properties for MPCD particles
template<class CT> void cell_thermo_basic_test(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    std::shared_ptr<SnapshotSystemData<Scalar>> snap(new SnapshotSystemData<Scalar>());
    snap->global_box = std::make_shared<BoxDim>(2.0);
    snap->particle_data.type_mapping.push_back("A");
    // place each particle in a different cell, doubling the first cell
    snap->mpcd_data.resize(5);
    snap->mpcd_data.type_mapping.push_back("A");
    snap->mpcd_data.position[0] = vec3<Scalar>(-0.5, -0.5, -0.5);
    snap->mpcd_data.position[1] = vec3<Scalar>(-0.5, -0.5, -0.5);
    snap->mpcd_data.position[2] = vec3<Scalar>(0.5, 0.5, 0.5);
    snap->mpcd_data.position[3] = vec3<Scalar>(0.5, 0.5, 0.5);
    snap->mpcd_data.position[4] = vec3<Scalar>(-0.5, 0.5, 0.5);

    snap->mpcd_data.velocity[0] = vec3<Scalar>(2.0, 0.0, 0.0);
    snap->mpcd_data.velocity[1] = vec3<Scalar>(1.0, 0.0, 0.0);
    snap->mpcd_data.velocity[2] = vec3<Scalar>(0.0, -3.0, 0.0);
    snap->mpcd_data.velocity[3] = vec3<Scalar>(0.0, 0.0, -5.0);
    snap->mpcd_data.velocity[4] = vec3<Scalar>(1.0, -1.0, 4.0);
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));

    std::shared_ptr<mpcd::ParticleData> pdata_5 = sysdef->getMPCDParticleData();
    auto cl = std::make_shared<mpcd::CellList>(sysdef);
    std::shared_ptr<CT> thermo = std::make_shared<CT>(sysdef, cl);
    AllThermoRequest thermo_req(thermo);
    thermo->compute(0);
        {
        const Index3D ci = cl->getCellIndexer();
        ArrayHandle<double4> h_avg_vel(thermo->getCellVelocities(),
                                       access_location::host,
                                       access_mode::read);
        ArrayHandle<double3> h_cell_energy(thermo->getCellEnergies(),
                                           access_location::host,
                                           access_mode::read);

        // Two particle cell (0,0,0)
        // Average velocity, mass
        CHECK_CLOSE(h_avg_vel.data[ci(0, 0, 0)].x, 1.5, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 0, 0)].y, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 0, 0)].z, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 0, 0)].w, 2.0, tol);
        // energy, temperature (relative to COM), np, flag
        CHECK_CLOSE(h_cell_energy.data[ci(0, 0, 0)].x, 2.5, tol);
        CHECK_CLOSE(h_cell_energy.data[ci(0, 0, 0)].y, 2.0 * 0.5 * 0.5 / 3.0, tol);
        UP_ASSERT_EQUAL(__double_as_int(h_cell_energy.data[ci(0, 0, 0)].z), 2);

        // Two particle cell (1,1,1)
        // Average velocity, mass
        CHECK_CLOSE(h_avg_vel.data[ci(1, 1, 1)].x, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1, 1, 1)].y, -1.5, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1, 1, 1)].z, -2.5, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1, 1, 1)].w, 2.0, tol);
        // energy, temperature (relative to COM), np, flag
        CHECK_CLOSE(h_cell_energy.data[ci(1, 1, 1)].x, 17.0, tol);
        CHECK_CLOSE(h_cell_energy.data[ci(1, 1, 1)].y, 2.0 * (1.5 * 1.5 + 2.5 * 2.5) / 3.0, tol);
        UP_ASSERT_EQUAL(__double_as_int(h_cell_energy.data[ci(1, 1, 1)].z), 2);

        // One particle cell (0,1,1)
        // Average velocity, mass
        CHECK_CLOSE(h_avg_vel.data[ci(0, 1, 1)].x, 1.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 1, 1)].y, -1.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 1, 1)].z, 4.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 1, 1)].w, 1.0, tol);
        // Has kinetic energy, but temperature should be zero
        CHECK_CLOSE(h_cell_energy.data[ci(0, 1, 1)].x, 9.0, tol);
        CHECK_CLOSE(h_cell_energy.data[ci(0, 1, 1)].y, 0.0, tol);
        UP_ASSERT_EQUAL(__double_as_int(h_cell_energy.data[ci(0, 1, 1)].z), 1);

        // Check the net stats of the system
        CHECK_CLOSE(thermo->getNetMomentum().x, 4.0, tol);
        CHECK_CLOSE(thermo->getNetMomentum().y, -4.0, tol);
        CHECK_CLOSE(thermo->getNetMomentum().z, -1.0, tol);
        CHECK_CLOSE(thermo->getNetEnergy(), 28.5, tol);
        CHECK_CLOSE(thermo->getTemperature(),
                    0.5 * (2.0 * 0.5 * 0.5 / 3.0 + 2.0 * (1.5 * 1.5 + 2.5 * 2.5) / 3.0),
                    tol);
        }

    // increase the mass and make sure that energies depend on mass, but velocities don't
    pdata_5->setMass(4.0);
    thermo->compute(1);
        {
        const Index3D ci = cl->getCellIndexer();
        ArrayHandle<double4> h_avg_vel(thermo->getCellVelocities(),
                                       access_location::host,
                                       access_mode::read);
        ArrayHandle<double3> h_cell_energy(thermo->getCellEnergies(),
                                           access_location::host,
                                           access_mode::read);

        // Two particle cell (0,0,0)
        // Average velocity, mass
        CHECK_CLOSE(h_avg_vel.data[ci(0, 0, 0)].x, 1.5, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 0, 0)].y, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 0, 0)].z, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 0, 0)].w, 8.0, tol);
        // energy, temperature (relative to COM), np, flag
        CHECK_CLOSE(h_cell_energy.data[ci(0, 0, 0)].x, 4.0 * 2.5, tol);
        CHECK_CLOSE(h_cell_energy.data[ci(0, 0, 0)].y, 4.0 * 2.0 * 0.5 * 0.5 / 3.0, tol);
        UP_ASSERT_EQUAL(__double_as_int(h_cell_energy.data[ci(0, 0, 0)].z), 2);

        // Two particle cell (1,1,1)
        // Average velocity, mass
        CHECK_CLOSE(h_avg_vel.data[ci(1, 1, 1)].x, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1, 1, 1)].y, -1.5, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1, 1, 1)].z, -2.5, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1, 1, 1)].w, 8.0, tol);
        // energy, temperature (relative to COM), np, flag
        CHECK_CLOSE(h_cell_energy.data[ci(1, 1, 1)].x, 4.0 * 17.0, tol);
        CHECK_CLOSE(h_cell_energy.data[ci(1, 1, 1)].y,
                    4.0 * 2.0 * (1.5 * 1.5 + 2.5 * 2.5) / 3.0,
                    tol);
        UP_ASSERT_EQUAL(__double_as_int(h_cell_energy.data[ci(1, 1, 1)].z), 2);

        // One particle cell (0,1,1)
        // Average velocity, mass
        CHECK_CLOSE(h_avg_vel.data[ci(0, 1, 1)].x, 1.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 1, 1)].y, -1.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 1, 1)].z, 4.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 1, 1)].w, 4.0, tol);
        // Has kinetic energy, but temperature should be zero
        CHECK_CLOSE(h_cell_energy.data[ci(0, 1, 1)].x, 4.0 * 9.0, tol);
        CHECK_CLOSE(h_cell_energy.data[ci(0, 1, 1)].y, 0.0, tol);
        UP_ASSERT_EQUAL(__double_as_int(h_cell_energy.data[ci(0, 1, 1)].z), 1);

        // Check the net stats of the system
        CHECK_CLOSE(thermo->getNetMomentum().x, 4.0 * 4.0, tol);
        CHECK_CLOSE(thermo->getNetMomentum().y, 4.0 * -4.0, tol);
        CHECK_CLOSE(thermo->getNetMomentum().z, 4.0 * -1.0, tol);
        CHECK_CLOSE(thermo->getNetEnergy(), 4.0 * 28.5, tol);
        CHECK_CLOSE(thermo->getTemperature(),
                    4.0 * 0.5 * (2.0 * 0.5 * 0.5 / 3.0 + 2.0 * (1.5 * 1.5 + 2.5 * 2.5) / 3.0),
                    tol);
        }

    // switch a particle into a different cell, and make sure the DOF are reduced accordingly
    pdata_5->setMass(1.0);
        {
        ArrayHandle<Scalar4> h_pos(pdata_5->getPositions(),
                                   access_location::host,
                                   access_mode::readwrite);
        h_pos.data[2] = make_scalar4(-0.5, -0.5, -0.5, 0.0);
        }
    thermo->compute(2);
        {
        const Index3D ci = cl->getCellIndexer();
        ArrayHandle<double4> h_avg_vel(thermo->getCellVelocities(),
                                       access_location::host,
                                       access_mode::read);
        ArrayHandle<double3> h_cell_energy(thermo->getCellEnergies(),
                                           access_location::host,
                                           access_mode::read);

        // Three particle cell (0,0,0)
        // Average velocity, mass
        CHECK_CLOSE(h_avg_vel.data[ci(0, 0, 0)].x, 1.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 0, 0)].y, -1.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 0, 0)].z, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 0, 0)].w, 3.0, tol);
        // energy, temperature (relative to COM), np, flag
        CHECK_CLOSE(h_cell_energy.data[ci(0, 0, 0)].x, 7.0, tol);
        CHECK_CLOSE(h_cell_energy.data[ci(0, 0, 0)].y,
                    (2 * 1.0 * 1.0 + 2 * 1.0 * 1.0 + 2.0 * 2.0) / 6.0,
                    tol);
        UP_ASSERT_EQUAL(__double_as_int(h_cell_energy.data[ci(0, 0, 0)].z), 3);

        // One particle cell (1,1,1)
        // Average velocity, mass
        CHECK_CLOSE(h_avg_vel.data[ci(1, 1, 1)].x, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1, 1, 1)].y, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1, 1, 1)].z, -5.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1, 1, 1)].w, 1.0, tol);
        // energy, temperature (relative to COM), np, flag
        CHECK_CLOSE(h_cell_energy.data[ci(1, 1, 1)].x, 12.5, tol);
        CHECK_CLOSE(h_cell_energy.data[ci(1, 1, 1)].y, 0.0, tol);
        UP_ASSERT_EQUAL(__double_as_int(h_cell_energy.data[ci(1, 1, 1)].z), 1);

        // One particle cell (0,1,1)
        // Average velocity, mass
        CHECK_CLOSE(h_avg_vel.data[ci(0, 1, 1)].x, 1.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 1, 1)].y, -1.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 1, 1)].z, 4.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 1, 1)].w, 1.0, tol);
        // Has kinetic energy, but temperature should be zero
        CHECK_CLOSE(h_cell_energy.data[ci(0, 1, 1)].x, 9.0, tol);
        CHECK_CLOSE(h_cell_energy.data[ci(0, 1, 1)].y, 0.0, tol);
        UP_ASSERT_EQUAL(__double_as_int(h_cell_energy.data[ci(0, 1, 1)].z), 1);

        // Check the net stats of the system, only average temperature should change now
        CHECK_CLOSE(thermo->getNetMomentum().x, 4.0, tol);
        CHECK_CLOSE(thermo->getNetMomentum().y, -4.0, tol);
        CHECK_CLOSE(thermo->getNetMomentum().z, -1.0, tol);
        CHECK_CLOSE(thermo->getNetEnergy(), 28.5, tol);
        CHECK_CLOSE(thermo->getTemperature(),
                    (2 * 1.0 * 1.0 + 2 * 1.0 * 1.0 + 2.0 * 2.0) / 6.0,
                    tol);
        }
    }

//! Test for correct calculation of cell thermo properties with embedded particles
template<class CT> void cell_thermo_embed_test(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    std::shared_ptr<SnapshotSystemData<Scalar>> snap(new SnapshotSystemData<Scalar>());
    snap->global_box = std::make_shared<BoxDim>(2.0);
        {
        SnapshotParticleData<Scalar>& pdata_snap = snap->particle_data;
        pdata_snap.type_mapping.push_back("A");
        pdata_snap.resize(4);

        pdata_snap.pos[0] = vec3<Scalar>(-0.5, -0.5, -0.5);
        pdata_snap.pos[1] = vec3<Scalar>(-0.5, -0.5, 0.5);
        pdata_snap.pos[2] = vec3<Scalar>(0.5, -0.5, 0.5);
        pdata_snap.pos[3] = vec3<Scalar>(0.5, 0.5, 0.5);

        pdata_snap.mass[0] = 3.0;
        pdata_snap.mass[1] = 2.0;
        pdata_snap.mass[2] = 4.0;
        pdata_snap.mass[3] = 5.0;

        pdata_snap.vel[0] = vec3<Scalar>(-2.0, 0.0, 0.0);
        pdata_snap.vel[1] = vec3<Scalar>(1.0, 0.0, 0.0);
        pdata_snap.vel[2] = vec3<Scalar>(0.0, -3.0, 0.0);
        pdata_snap.vel[3] = vec3<Scalar>(0.0, 0.0, -5.0);
        }
    snap->mpcd_data.resize(4);
    snap->mpcd_data.type_mapping.push_back("A");
    snap->mpcd_data.position[0] = vec3<Scalar>(-0.5, -0.5, -0.5);
    snap->mpcd_data.position[1] = vec3<Scalar>(-0.5, -0.5, 0.5);
    snap->mpcd_data.position[2] = vec3<Scalar>(0.5, -0.5, 0.5);
    snap->mpcd_data.position[3] = vec3<Scalar>(0.5, 0.5, 0.5);

    snap->mpcd_data.velocity[0] = vec3<Scalar>(2.0, 0.0, 0.0);
    snap->mpcd_data.velocity[1] = vec3<Scalar>(1.0, 0.0, 0.0);
    snap->mpcd_data.velocity[2] = vec3<Scalar>(0.0, -3.0, 0.0);
    snap->mpcd_data.velocity[3] = vec3<Scalar>(0.0, 0.0, -5.0);
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));

    std::shared_ptr<ParticleData> embed_pdata = sysdef->getParticleData();
    std::shared_ptr<ParticleFilter> selector(new ParticleFilterAll());
    std::shared_ptr<ParticleGroup> group(new ParticleGroup(sysdef, selector));

    auto cl = std::make_shared<mpcd::CellList>(sysdef);
    cl->setEmbeddedGroup(group);
    std::shared_ptr<CT> thermo = std::make_shared<CT>(sysdef, cl);
    AllThermoRequest thermo_req(thermo);
    thermo->compute(0);
        {
        const Index3D ci = cl->getCellIndexer();
        ArrayHandle<double4> h_avg_vel(thermo->getCellVelocities(),
                                       access_location::host,
                                       access_mode::read);
        ArrayHandle<double3> h_cell_energy(thermo->getCellEnergies(),
                                           access_location::host,
                                           access_mode::read);

        // Cell (0,0,0)
        // Average velocity, mass
        CHECK_CLOSE(h_avg_vel.data[ci(0, 0, 0)].x, -1.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 0, 0)].y, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 0, 0)].z, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 0, 0)].w, 4.0, tol);
        // energy, temperature (relative to COM), np, flag
        CHECK_CLOSE(h_cell_energy.data[ci(0, 0, 0)].x, 8.0, tol);
        CHECK_CLOSE(h_cell_energy.data[ci(0, 0, 0)].y, 4.0, tol);
        UP_ASSERT_EQUAL(__double_as_int(h_cell_energy.data[ci(0, 0, 0)].z), 2);

        // Cell (0,0,1)
        // Average velocity, mass
        CHECK_CLOSE(h_avg_vel.data[ci(0, 0, 1)].x, 1.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 0, 1)].y, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 0, 1)].z, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(0, 0, 1)].w, 3.0, tol);
        // energy, temperature (relative to COM), np, flag
        CHECK_CLOSE(h_cell_energy.data[ci(0, 0, 1)].x, 1.5, tol);
        CHECK_CLOSE(h_cell_energy.data[ci(0, 0, 1)].y, 0.0, tol);
        UP_ASSERT_EQUAL(__double_as_int(h_cell_energy.data[ci(0, 0, 1)].z), 2);

        // Cell (1,0,1)
        // Average velocity, mass
        CHECK_CLOSE(h_avg_vel.data[ci(1, 0, 1)].x, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1, 0, 1)].y, -3.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1, 0, 1)].z, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1, 0, 1)].w, 5.0, tol);
        // Has kinetic energy, but temperature should be zero
        CHECK_CLOSE(h_cell_energy.data[ci(1, 0, 1)].x, 22.5, tol);
        CHECK_CLOSE(h_cell_energy.data[ci(1, 0, 1)].y, 0.0, tol);
        UP_ASSERT_EQUAL(__double_as_int(h_cell_energy.data[ci(1, 0, 1)].z), 2);

        // Cell (1,1,1)
        // Average velocity, mass
        CHECK_CLOSE(h_avg_vel.data[ci(1, 1, 1)].x, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1, 1, 1)].y, 0.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1, 1, 1)].z, -5.0, tol);
        CHECK_CLOSE(h_avg_vel.data[ci(1, 1, 1)].w, 6.0, tol);
        // energy, temperature (relative to COM), np, flag
        CHECK_CLOSE(h_cell_energy.data[ci(1, 1, 1)].x, 75.0, tol);
        CHECK_CLOSE(h_cell_energy.data[ci(1, 1, 1)].y, 0.0, tol);
        UP_ASSERT_EQUAL(__double_as_int(h_cell_energy.data[ci(1, 1, 1)].z), 2);

        // Check the net stats of the system
        CHECK_CLOSE(thermo->getNetMomentum().x, -1.0, tol);
        CHECK_CLOSE(thermo->getNetMomentum().y, -15.0, tol);
        CHECK_CLOSE(thermo->getNetMomentum().z, -30.0, tol);
        CHECK_CLOSE(thermo->getNetEnergy(), 107.0, tol);
        CHECK_CLOSE(thermo->getTemperature(), (4.0 + 0.0 + 0.0 + 0.0) / 4., tol);
        }
    }

UP_TEST(mpcd_cell_thermo_basic)
    {
    cell_thermo_basic_test<mpcd::CellThermoCompute>(std::shared_ptr<ExecutionConfiguration>(
        new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }
UP_TEST(mpcd_cell_thermo_embed)
    {
    cell_thermo_embed_test<mpcd::CellThermoCompute>(std::shared_ptr<ExecutionConfiguration>(
        new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

#ifdef ENABLE_HIP
UP_TEST(mpcd_cell_thermo_basic_gpu)
    {
    cell_thermo_basic_test<mpcd::CellThermoComputeGPU>(std::shared_ptr<ExecutionConfiguration>(
        new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
UP_TEST(mpcd_cell_thermo_embed_gpu)
    {
    cell_thermo_embed_test<mpcd::CellThermoComputeGPU>(std::shared_ptr<ExecutionConfiguration>(
        new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
#endif // ENABLE_HIP
