// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/Communicator.h"
#include "hoomd/mpcd/CellCommunicator.h"
#include "hoomd/mpcd/CellList.h"
#include "hoomd/mpcd/CellThermoTypes.h"
#include "hoomd/mpcd/CommunicatorUtilities.h"
#include "hoomd/mpcd/ReductionOperators.h"

#include "hoomd/SnapshotSystemData.h"
#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN()

using namespace hoomd;

//! Test for correct calculation of MPCD grid dimensions
void cell_communicator_reduce_test(std::shared_ptr<ExecutionConfiguration> exec_conf,
                                   bool mpi_x,
                                   bool mpi_y,
                                   bool mpi_z)
    {
    if (exec_conf->getPartition() != 0)
        return;

    std::shared_ptr<SnapshotSystemData<Scalar>> snap(new SnapshotSystemData<Scalar>());
    snap->global_box = std::make_shared<BoxDim>(5.0);
    snap->particle_data.type_mapping.push_back("A");
    snap->particle_data.resize(0);

    // setup test in mpi
    std::vector<Scalar> fx, fy, fz;
    unsigned int n_req_ranks = 1;
    if (mpi_x)
        {
        n_req_ranks *= 2;
        fx.push_back(0.5);
        }
    if (mpi_y)
        {
        n_req_ranks *= 2;
        fy.push_back(0.45);
        }
    if (mpi_z)
        {
        n_req_ranks *= 2;
        fz.push_back(0.55);
        }
    UP_ASSERT_EQUAL(exec_conf->getNRanks(), n_req_ranks);
    std::shared_ptr<DomainDecomposition> decomposition(
        new DomainDecomposition(exec_conf, snap->global_box->getL(), fx, fy, fz));
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf, decomposition));
    std::shared_ptr<Communicator> pdata_comm(new Communicator(sysdef, decomposition));
    sysdef->setCommunicator(pdata_comm);

    auto cl = std::make_shared<mpcd::CellList>(sysdef);
    cl->computeDimensions();

    // Fill in a dummy cell property array, which is just the global index of each cell
    // we use the 1-indexed cell (rather than standard 0) so that we can confirm sums are all done
    // correctly
    const Index3D& ci = cl->getCellIndexer();
    GPUArray<double3> props(ci.getNumElements(), exec_conf);
    GPUArray<double3> ref_props(ci.getNumElements(), exec_conf);
        {
        ArrayHandle<double3> h_props(props, access_location::host, access_mode::overwrite);
        ArrayHandle<double3> h_ref_props(ref_props, access_location::host, access_mode::overwrite);
        for (unsigned int k = 0; k < ci.getD(); ++k)
            {
            for (unsigned int j = 0; j < ci.getH(); ++j)
                {
                for (unsigned int i = 0; i < ci.getW(); ++i)
                    {
                    int3 global_cell = cl->getGlobalCell(make_int3(i, j, k));
                    global_cell.x += 1;
                    global_cell.y += 1;
                    global_cell.z += 1;

                    h_props.data[ci(i, j, k)] = make_double3(global_cell.x,
                                                             global_cell.y,
                                                             __int_as_double(global_cell.z));
                    h_ref_props.data[ci(i, j, k)] = make_double3(global_cell.x,
                                                                 global_cell.y,
                                                                 __int_as_double(global_cell.z));
                    }
                }
            }
        }

    // on summing, all communicated cells should simply increase by a multiple of the ranks they
    // overlap
    mpcd::CellCommunicator comm(sysdef, cl);
    comm.communicate(props, mpcd::detail::CellEnergyPackOp());
    auto num_comm_cells = cl->getNComm();
        {
        ArrayHandle<double3> h_props(props, access_location::host, access_mode::read);
        ArrayHandle<double3> h_ref_props(ref_props, access_location::host, access_mode::read);
        for (unsigned int k = 0; k < ci.getD(); ++k)
            {
            for (unsigned int j = 0; j < ci.getH(); ++j)
                {
                for (unsigned int i = 0; i < ci.getW(); ++i)
                    {
                    // count the number of ranks this cell overlaps, which gives the multiplier
                    // relative to the reference values
                    unsigned int noverlap = 1;
                    if (i < num_comm_cells[static_cast<unsigned int>(mpcd::detail::face::west)]
                        || i >= ci.getW()
                                    - num_comm_cells[static_cast<unsigned int>(
                                        mpcd::detail::face::east)])
                        {
                        noverlap *= 2;
                        }
                    if (j < num_comm_cells[static_cast<unsigned int>(mpcd::detail::face::south)]
                        || j >= ci.getH()
                                    - num_comm_cells[static_cast<unsigned int>(
                                        mpcd::detail::face::north)])
                        {
                        noverlap *= 2;
                        }
                    if (k < num_comm_cells[static_cast<unsigned int>(mpcd::detail::face::down)]
                        || k >= ci.getD()
                                    - num_comm_cells[static_cast<unsigned int>(
                                        mpcd::detail::face::up)])
                        {
                        noverlap *= 2;
                        }

                    UP_ASSERT_EQUAL(h_props.data[ci(i, j, k)].x,
                                    h_ref_props.data[ci(i, j, k)].x * noverlap);
                    UP_ASSERT_EQUAL(
                        h_props.data[ci(i, j, k)].y,
                        h_ref_props.data[ci(i, j, k)].y); // energy packing doesn't touch y element
                    UP_ASSERT_EQUAL(__double_as_int(h_props.data[ci(i, j, k)].z),
                                    __double_as_int(h_ref_props.data[ci(i, j, k)].z) * noverlap);
                    }
                }
            }
        }
    }

//! Test for error handling in overdecomposed systems
void cell_communicator_overdecompose_test(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    UP_ASSERT_EQUAL(exec_conf->getNRanks(), 8);

    std::shared_ptr<SnapshotSystemData<Scalar>> snap(new SnapshotSystemData<Scalar>());
    snap->global_box = std::make_shared<BoxDim>(6.0);
    snap->particle_data.type_mapping.push_back("A");
    snap->particle_data.resize(0);
    std::shared_ptr<DomainDecomposition> decomposition(
        new DomainDecomposition(exec_conf, snap->global_box->getL(), 2, 2, 2));
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf, decomposition));
    std::shared_ptr<Communicator> pdata_comm(new Communicator(sysdef, decomposition));
    sysdef->setCommunicator(pdata_comm);

    auto cl = std::make_shared<mpcd::CellList>(sysdef);
    cl->computeDimensions();

    // Don't really care what's in this array, just want to make sure errors get thrown
    // appropriately
    const Index3D& ci = cl->getCellIndexer();
    GPUArray<double4> props(ci.getNumElements(), exec_conf);
        {
        ArrayHandle<double4> h_props(props, access_location::host, access_mode::overwrite);
        memset(h_props.data, 0, ci.getNumElements() * sizeof(double4));
        }

    mpcd::CellCommunicator comm(sysdef, cl);
    mpcd::detail::CellVelocityPackOp pack_op;

    // initially, reduction should succeed
    comm.communicate(props, pack_op);

    // add a communication cell
    cl->setNExtraCells(1);
    cl->compute(1);

    // should throw an exception since the prop dims don't match the cell dims
    UP_ASSERT_EXCEPTION(std::runtime_error, [&] { comm.communicate(props, pack_op); });

    // should succeed on resizing of props
    props.resize(cl->getNCells());
    comm.communicate(props, pack_op);

    // add another communication cell, which overdecomposes the system
    cl->setNExtraCells(2);
    cl->compute(2);
    props.resize(cl->getNCells());
    UP_ASSERT_EXCEPTION(std::runtime_error, [&] { comm.communicate(props, pack_op); });

    // cut the cell size down, which should make it so that the system can be decomposed again
    cl->setCellSize(0.5);
    cl->compute(3);
    props.resize(cl->getNCells());
    comm.communicate(props, pack_op);

    // shrink the box size to the minimum that can be decomposed
    cl->setNExtraCells(0);
    cl->setCellSize(2.0);
    cl->compute(4);
    props.resize(cl->getNCells());
    comm.communicate(props, pack_op);

    // now shrink further to ensure failure
    cl->setCellSize(3.0);
    cl->compute(5);
    props.resize(cl->getNCells());
    UP_ASSERT_EXCEPTION(std::runtime_error, [&] { comm.communicate(props, pack_op); });
    }

//! dimension test case for MPCD CellList class
UP_TEST(mpcd_cell_communicator)
    {
    if (!exec_conf_cpu)
        {
        exec_conf_cpu = std::shared_ptr<ExecutionConfiguration>(
            new ExecutionConfiguration(ExecutionConfiguration::CPU));
        }

        // mpi in 1d
        {
        exec_conf_cpu->getMPIConfig()->splitPartitions(2);
        cell_communicator_reduce_test(exec_conf_cpu, true, false, false);
        cell_communicator_reduce_test(exec_conf_cpu, false, true, false);
        cell_communicator_reduce_test(exec_conf_cpu, false, false, true);
        }
        // mpi in 2d
        {
        exec_conf_cpu->getMPIConfig()->splitPartitions(4);
        cell_communicator_reduce_test(exec_conf_cpu, true, true, false);
        cell_communicator_reduce_test(exec_conf_cpu, true, false, true);
        cell_communicator_reduce_test(exec_conf_cpu, false, true, true);
        }
        // mpi in 3d
        {
        exec_conf_cpu->getMPIConfig()->splitPartitions(8);
        cell_communicator_reduce_test(exec_conf_cpu, true, true, true);
        }
    }

//! error handling test for overdecomposed boxes
UP_TEST(mpcd_cell_communicator_overdecompose)
    {
    if (!exec_conf_cpu)
        exec_conf_cpu = std::shared_ptr<ExecutionConfiguration>(
            new ExecutionConfiguration(ExecutionConfiguration::CPU));
    cell_communicator_overdecompose_test(exec_conf_cpu);
    }

#ifdef ENABLE_HIP
//! dimension test case for MPCD CellList class
UP_TEST(mpcd_cell_communicator_gpu)
    {
    if (!exec_conf_gpu)
        {
        exec_conf_gpu = std::shared_ptr<ExecutionConfiguration>(
            new ExecutionConfiguration(ExecutionConfiguration::GPU));
        }

        // mpi in 1d
        {
        exec_conf_gpu->getMPIConfig()->splitPartitions(2);
        cell_communicator_reduce_test(exec_conf_gpu, true, false, false);
        cell_communicator_reduce_test(exec_conf_gpu, false, true, false);
        cell_communicator_reduce_test(exec_conf_gpu, false, false, true);
        }
        // mpi in 2d
        {
        exec_conf_gpu->getMPIConfig()->splitPartitions(4);
        cell_communicator_reduce_test(exec_conf_gpu, true, true, false);
        cell_communicator_reduce_test(exec_conf_gpu, true, false, true);
        cell_communicator_reduce_test(exec_conf_gpu, false, true, true);
        }
        // mpi in 3d
        {
        exec_conf_gpu->getMPIConfig()->splitPartitions(8);
        cell_communicator_reduce_test(exec_conf_gpu, true, true, true);
        }
    }
#endif // ENABLE_HIP
