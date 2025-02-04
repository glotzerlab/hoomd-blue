// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/mpcd/CellList.h"
#include "hoomd/mpcd/CellThermoCompute.h"
#include "utils.h"
#ifdef ENABLE_HIP
#include "hoomd/mpcd/CellThermoComputeGPU.h"
#endif // ENABLE_HIP

#include "hoomd/Communicator.h"
#include "hoomd/SnapshotSystemData.h"
#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN()

using namespace hoomd;

//! Test for correct calculation of MPCD grid dimensions
template<class CT> void cell_thermo_basic_test(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    UP_ASSERT_EQUAL(exec_conf->getNRanks(), 8);

    std::shared_ptr<SnapshotSystemData<Scalar>> snap(new SnapshotSystemData<Scalar>());
    snap->global_box = std::make_shared<BoxDim>(5.0);
    snap->particle_data.type_mapping.push_back("A");
    // place each particle all in the same cell
    snap->mpcd_data.resize(9);
    snap->mpcd_data.type_mapping.push_back("A");
    // start all particles in the middle of their domains (no overlapping comms)
    snap->mpcd_data.position[0] = vec3<Scalar>(-1.0, -1.0, -1.0);
    snap->mpcd_data.position[1] = vec3<Scalar>(1.0, -1.0, -1.0);
    snap->mpcd_data.position[2] = vec3<Scalar>(-1.0, 1.0, -1.0);
    snap->mpcd_data.position[3] = vec3<Scalar>(1.0, 1.0, -1.0);
    snap->mpcd_data.position[4] = vec3<Scalar>(-1.0, -1.0, 1.0);
    snap->mpcd_data.position[5] = vec3<Scalar>(1.0, -1.0, 1.0);
    snap->mpcd_data.position[6] = vec3<Scalar>(-1.0, 1.0, 1.0);
    snap->mpcd_data.position[7] = vec3<Scalar>(1.0, 1.0, 1.0);
    // put an extra particle on rank 0 so that at least one temp is defined
    snap->mpcd_data.position[8] = vec3<Scalar>(-1.0, -1.0, -1.0);

    snap->mpcd_data.velocity[0] = vec3<Scalar>(-1.0, -1.0, -1.0);
    snap->mpcd_data.velocity[1] = vec3<Scalar>(1.0, -1.0, -1.0);
    snap->mpcd_data.velocity[2] = vec3<Scalar>(-1.0, 1.0, -1.0);
    snap->mpcd_data.velocity[3] = vec3<Scalar>(1.0, 1.0, -1.0);
    snap->mpcd_data.velocity[4] = vec3<Scalar>(-1.0, -1.0, 1.0);
    snap->mpcd_data.velocity[5] = vec3<Scalar>(1.0, -1.0, 1.0);
    snap->mpcd_data.velocity[6] = vec3<Scalar>(-1.0, 1.0, 1.0);
    snap->mpcd_data.velocity[7] = vec3<Scalar>(1.0, 1.0, 1.0);
    snap->mpcd_data.velocity[8] = vec3<Scalar>(1.0, 1.0, 1.0);

    std::shared_ptr<DomainDecomposition> decomposition(
        new DomainDecomposition(exec_conf, snap->global_box->getL(), 2, 2, 2));
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf, decomposition));
    std::shared_ptr<Communicator> pdata_comm(new Communicator(sysdef, decomposition));
    sysdef->setCommunicator(pdata_comm);

    std::shared_ptr<mpcd::ParticleData> pdata = sysdef->getMPCDParticleData();
    auto cl = std::make_shared<mpcd::CellList>(sysdef, 1.0, false);

    std::shared_ptr<CT> thermo = std::make_shared<CT>(sysdef, cl);
    AllThermoRequest thermo_req(thermo);
    thermo->compute(0);
        {
        // check per-cell stats
        const Index3D& ci = cl->getCellIndexer();
        ArrayHandle<double4> h_cell_vel(thermo->getCellVelocities(),
                                        access_location::host,
                                        access_mode::read);
        ArrayHandle<double3> h_cell_energy(thermo->getCellEnergies(),
                                           access_location::host,
                                           access_mode::read);
        switch (exec_conf->getRank())
            {
        case 0:
            CHECK_CLOSE(h_cell_vel.data[ci(2, 2, 2)].x, 0.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(2, 2, 2)].y, 0.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(2, 2, 2)].z, 0.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(2, 2, 2)].w, 2.0, tol);

            CHECK_CLOSE(h_cell_energy.data[ci(2, 2, 2)].x, 3.0, tol);
            CHECK_CLOSE(h_cell_energy.data[ci(2, 2, 2)].y, 2.0, tol);
            UP_ASSERT_EQUAL(__double_as_int(h_cell_energy.data[ci(2, 2, 2)].z), 2);
            break;
        case 1:
            CHECK_CLOSE(h_cell_vel.data[ci(1, 2, 2)].x, 1.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(1, 2, 2)].y, -1.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(1, 2, 2)].z, -1.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(1, 2, 2)].w, 1.0, tol);

            CHECK_CLOSE(h_cell_energy.data[ci(1, 2, 2)].x, 1.5, tol);
            CHECK_CLOSE(h_cell_energy.data[ci(1, 2, 2)].y, 0.0, tol);
            UP_ASSERT_EQUAL(__double_as_int(h_cell_energy.data[ci(1, 2, 2)].z), 1);
            break;
        case 2:
            CHECK_CLOSE(h_cell_vel.data[ci(2, 1, 2)].x, -1.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(2, 1, 2)].y, 1.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(2, 1, 2)].z, -1.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(2, 1, 2)].w, 1.0, tol);

            CHECK_CLOSE(h_cell_energy.data[ci(2, 1, 2)].x, 1.5, tol);
            CHECK_CLOSE(h_cell_energy.data[ci(2, 1, 2)].y, 0.0, tol);
            UP_ASSERT_EQUAL(__double_as_int(h_cell_energy.data[ci(2, 1, 2)].z), 1);
            break;
        case 3:
            CHECK_CLOSE(h_cell_vel.data[ci(1, 1, 2)].x, 1.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(1, 1, 2)].y, 1.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(1, 1, 2)].z, -1.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(1, 1, 2)].w, 1.0, tol);

            CHECK_CLOSE(h_cell_energy.data[ci(1, 1, 2)].x, 1.5, tol);
            CHECK_CLOSE(h_cell_energy.data[ci(1, 1, 2)].y, 0.0, tol);
            UP_ASSERT_EQUAL(__double_as_int(h_cell_energy.data[ci(1, 1, 2)].z), 1);
            break;
        case 4:
            CHECK_CLOSE(h_cell_vel.data[ci(2, 2, 1)].x, -1.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(2, 2, 1)].y, -1.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(2, 2, 1)].z, 1.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(2, 2, 1)].w, 1.0, tol);

            CHECK_CLOSE(h_cell_energy.data[ci(2, 2, 1)].x, 1.5, tol);
            CHECK_CLOSE(h_cell_energy.data[ci(2, 2, 1)].y, 0.0, tol);
            UP_ASSERT_EQUAL(__double_as_int(h_cell_energy.data[ci(2, 2, 1)].z), 1);
            break;
        case 5:
            CHECK_CLOSE(h_cell_vel.data[ci(1, 2, 1)].x, 1.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(1, 2, 1)].y, -1.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(1, 2, 1)].z, 1.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(1, 2, 1)].w, 1.0, tol);

            CHECK_CLOSE(h_cell_energy.data[ci(1, 2, 1)].x, 1.5, tol);
            CHECK_CLOSE(h_cell_energy.data[ci(1, 2, 1)].y, 0.0, tol);
            UP_ASSERT_EQUAL(__double_as_int(h_cell_energy.data[ci(1, 2, 1)].z), 1);
            break;
        case 6:
            CHECK_CLOSE(h_cell_vel.data[ci(2, 1, 1)].x, -1.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(2, 1, 1)].y, 1.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(2, 1, 1)].z, 1.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(2, 1, 1)].w, 1.0, tol);

            CHECK_CLOSE(h_cell_energy.data[ci(2, 1, 1)].x, 1.5, tol);
            CHECK_CLOSE(h_cell_energy.data[ci(2, 1, 1)].y, 0.0, tol);
            UP_ASSERT_EQUAL(__double_as_int(h_cell_energy.data[ci(2, 1, 1)].z), 1);
            break;
        case 7:
            CHECK_CLOSE(h_cell_vel.data[ci(1, 1, 1)].x, 1.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(1, 1, 1)].y, 1.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(1, 1, 1)].z, 1.0, tol);
            CHECK_CLOSE(h_cell_vel.data[ci(1, 1, 1)].w, 1.0, tol);

            CHECK_CLOSE(h_cell_energy.data[ci(1, 1, 1)].x, 1.5, tol);
            CHECK_CLOSE(h_cell_energy.data[ci(1, 1, 1)].y, 0.0, tol);
            UP_ASSERT_EQUAL(__double_as_int(h_cell_energy.data[ci(1, 1, 1)].z), 1);
            break;
            }

        // Check the net stats of the system
        CHECK_CLOSE(thermo->getNetMomentum().x, 1.0, tol);
        CHECK_CLOSE(thermo->getNetMomentum().y, 1.0, tol);
        CHECK_CLOSE(thermo->getNetMomentum().z, 1.0, tol);
        CHECK_CLOSE(thermo->getNetEnergy(), 13.5, tol);
        CHECK_CLOSE(thermo->getTemperature(), 2.0, tol);
        }

        // scale all particles so that they move into one common cell
        {
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(),
                                   access_location::host,
                                   access_mode::readwrite);
        for (unsigned int i = 0; i < pdata->getN(); ++i)
            {
            h_pos.data[i].x *= 0.25;
            h_pos.data[i].y *= 0.25;
            h_pos.data[i].z *= 0.25;
            }
        }
    thermo->compute(1);
        {
        // all ranks should share the same cell stats now, just need to extract the correct indexing
        const int3 local_cell = cl->getLocalCell(make_int3(2, 2, 2));
        const Index3D& ci = cl->getCellIndexer();
        const unsigned int local_idx = ci(local_cell.x, local_cell.y, local_cell.z);

        ArrayHandle<double4> h_cell_vel(thermo->getCellVelocities(),
                                        access_location::host,
                                        access_mode::read);
        ArrayHandle<double3> h_cell_energy(thermo->getCellEnergies(),
                                           access_location::host,
                                           access_mode::read);
        CHECK_CLOSE(h_cell_vel.data[local_idx].x, 1.0 / 9.0, tol);
        CHECK_CLOSE(h_cell_vel.data[local_idx].y, 1.0 / 9.0, tol);
        CHECK_CLOSE(h_cell_vel.data[local_idx].z, 1.0 / 9.0, tol);
        CHECK_CLOSE(h_cell_vel.data[local_idx].w, 9.0, tol);
        /*
         * Temperature in the middle cell is relative to (1.0/9.0, 1.0/9.0, 1.0/9.0).
         * The center of mass KE is 0.5 * (1.^2+1.^2+1.^2)/9. = (1/2) * (3/9)
         * The total cell KE is 9 * 0.5 * (1^2+1^2+1^2) = (1/2) * (27). So,
         *
         * T = (KE - KE_CM) * (2./(3.*(DOF-1))) = (27 * 9 - 3)/(9 * 3 * 8) = 10 / 9
         */
        CHECK_CLOSE(h_cell_energy.data[local_idx].x, 13.5, tol);
        CHECK_CLOSE(h_cell_energy.data[local_idx].y, 10. / 9., tol);
        CHECK_CLOSE(__double_as_int(h_cell_energy.data[local_idx].z), 9, tol);

        // Check the net stats of the system
        CHECK_CLOSE(thermo->getNetMomentum().x, 1.0, tol);
        CHECK_CLOSE(thermo->getNetMomentum().y, 1.0, tol);
        CHECK_CLOSE(thermo->getNetMomentum().z, 1.0, tol);
        CHECK_CLOSE(thermo->getNetEnergy(), 13.5, tol);
        CHECK_CLOSE(thermo->getTemperature(), 10. / 9., tol);
        }
    }

UP_TEST(mpcd_cell_thermo_basic)
    {
    cell_thermo_basic_test<mpcd::CellThermoCompute>(std::shared_ptr<ExecutionConfiguration>(
        new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

#ifdef ENABLE_HIP
UP_TEST(mpcd_cell_thermo_basic_gpu)
    {
    cell_thermo_basic_test<mpcd::CellThermoComputeGPU>(std::shared_ptr<ExecutionConfiguration>(
        new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
#endif
