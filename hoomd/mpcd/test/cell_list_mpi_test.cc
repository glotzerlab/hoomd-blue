// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/mpcd/CellList.h"
#ifdef ENABLE_HIP
#include "hoomd/mpcd/CellListGPU.h"
#endif // ENABLE_HIP

#include "hoomd/Communicator.h"
#include "hoomd/SnapshotSystemData.h"
#include "hoomd/mpcd/Communicator.h"
#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN()

using namespace hoomd;

//! Test for correct calculation of MPCD grid dimensions
/*!
 * \param exec_conf Execution configuration
 * \param mpi_x Flag if expecting MPI in x dimension
 * \param mpi_y Flag if expecting MPI in y dimension
 * \param mpi_z Flag if expecting MPI in z dimension
 *
 * \tparam CL CellList class to use, should be consistent with \a exec_conf mode
 */
template<class CL>
void celllist_dimension_test(std::shared_ptr<ExecutionConfiguration> exec_conf,
                             bool mpi_x,
                             bool mpi_y,
                             bool mpi_z)
    {
    // only run tests on first partition
    if (exec_conf->getPartition() != 0)
        return;

    std::shared_ptr<SnapshotSystemData<Scalar>> snap(new SnapshotSystemData<Scalar>());
    snap->global_box = std::make_shared<BoxDim>(5.0);
    snap->particle_data.type_mapping.push_back("A");
    snap->mpcd_data.resize(1);
    snap->mpcd_data.type_mapping.push_back("A");

    // configure domain decompsition
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

        {
        const Index3D& di = decomposition->getDomainIndexer();
        UP_ASSERT_EQUAL(di.getW(), (mpi_x) ? 2 : 1);
        UP_ASSERT_EQUAL(di.getH(), (mpi_y) ? 2 : 1);
        UP_ASSERT_EQUAL(di.getD(), (mpi_z) ? 2 : 1);
        }

    // initialize mpcd system
    auto pdata_1 = sysdef->getMPCDParticleData();
    std::shared_ptr<mpcd::CellList> cl(new CL(sysdef));

    // compute the cell list
    cl->computeDimensions();

        {
        // check domain origins
        const int3 origin = cl->getOriginIndex();
        const uint3 pos = decomposition->getGridPos();
        if (mpi_x)
            {
            // exactly halfway -> -1 and 2
            if (pos.x)
                {
                UP_ASSERT_EQUAL(origin.x, 2);
                }
            else
                {
                UP_ASSERT_EQUAL(origin.x, -1);
                }
            }
        else
            {
            UP_ASSERT_EQUAL(origin.x, 0);
            }

        if (mpi_y)
            {
            // biased to lower edge -> -1 and 1
            if (pos.y)
                {
                UP_ASSERT_EQUAL(origin.y, 1);
                }
            else
                {
                UP_ASSERT_EQUAL(origin.y, -1);
                }
            }
        else
            {
            UP_ASSERT_EQUAL(origin.y, 0);
            }

        // biased to upper edge -> -1 and 2
        if (mpi_z)
            {
            if (pos.z)
                {
                UP_ASSERT_EQUAL(origin.z, 2);
                }
            else
                {
                UP_ASSERT_EQUAL(origin.z, -1);
                }
            }
        else
            {
            UP_ASSERT_EQUAL(origin.z, 0);
            }

        // check domain sizes
        const uint3 dim = cl->getDim();
        if (mpi_x)
            {
            // split evenly in x -> both domains are same size
            UP_ASSERT_EQUAL(dim.x, 4);
            }
        else
            {
            UP_ASSERT_EQUAL(dim.x, 5);
            }

        if (mpi_y)
            {
            // biased to lower edge -> upper domains need extra cell
            if (pos.y)
                {
                UP_ASSERT_EQUAL(dim.y, 5);
                }
            else
                {
                UP_ASSERT_EQUAL(dim.y, 4);
                }
            }
        else
            {
            UP_ASSERT_EQUAL(dim.y, 5);
            }

        if (mpi_z)
            {
            // biased to upper edge -> lower domains need extra cell
            if (pos.z)
                {
                UP_ASSERT_EQUAL(dim.z, 4);
                }
            else
                {
                UP_ASSERT_EQUAL(dim.z, 5);
                }
            }
        else
            {
            UP_ASSERT_EQUAL(dim.z, 5);
            }

        std::array<unsigned int, 6> num_comm = cl->getNComm();
        if (mpi_x)
            {
            if (pos.x)
                {
                UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::east)], 2);
                UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::west)], 1);
                }
            else
                {
                UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::east)], 1);
                UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::west)], 2);
                }
            }
        else
            {
            UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::east)], 0);
            UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::west)], 0);
            }

        if (mpi_y)
            {
            UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::north)], 2);
            UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::south)], 2);
            }
        else
            {
            UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::north)], 0);
            UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::south)], 0);
            }

        if (mpi_z)
            {
            UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::up)], 2);
            UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::down)], 2);
            }
        else
            {
            UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::up)], 0);
            UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::down)], 0);
            }

        // check for grid shifting errors
        UP_ASSERT_EXCEPTION(std::runtime_error,
                            [&] { cl->setGridShift(make_scalar3(-0.51, -0.51, -0.51)); });
        UP_ASSERT_EXCEPTION(std::runtime_error,
                            [&] { cl->setGridShift(make_scalar3(0.51, 0.51, 0.51)); });

        // check coverage box
        const BoxDim coverage = cl->getCoverageBox();
        if (mpi_x)
            {
            if (pos.x)
                {
                CHECK_CLOSE(coverage.getLo().x, 0.0, tol);
                CHECK_CLOSE(coverage.getHi().x, 3.0, tol);
                }
            else
                {
                CHECK_CLOSE(coverage.getLo().x, -3.0, tol);
                CHECK_CLOSE(coverage.getHi().x, 0.0, tol);
                }
            }
        else
            {
            CHECK_CLOSE(coverage.getLo().x, -2.5, tol);
            CHECK_CLOSE(coverage.getHi().x, 2.5, tol);
            }

        if (mpi_y)
            {
            if (pos.y)
                {
                CHECK_CLOSE(coverage.getLo().y, -1.0, tol);
                CHECK_CLOSE(coverage.getHi().y, 3.0, tol);
                }
            else
                {
                CHECK_CLOSE(coverage.getLo().y, -3.0, tol);
                CHECK_CLOSE(coverage.getHi().y, 0.0, tol);
                }
            }
        else
            {
            CHECK_CLOSE(coverage.getLo().y, -2.5, tol);
            CHECK_CLOSE(coverage.getHi().y, 2.5, tol);
            }

        if (mpi_z)
            {
            if (pos.z)
                {
                CHECK_CLOSE(coverage.getLo().z, 0.0, tol);
                CHECK_CLOSE(coverage.getHi().z, 3.0, tol);
                }
            else
                {
                CHECK_CLOSE(coverage.getLo().z, -3.0, tol);
                CHECK_CLOSE(coverage.getHi().z, 1.0, tol);
                }
            }
        else
            {
            CHECK_CLOSE(coverage.getLo().z, -2.5, tol);
            CHECK_CLOSE(coverage.getHi().z, 2.5, tol);
            }
        }

    /*******************/
    // Change the cell size, and ensure everything stays up to date
    cl->setCellSize(0.5);
    cl->computeDimensions();
        {
        // check domain origins
        const int3 origin = cl->getOriginIndex();
        const uint3 pos = decomposition->getGridPos();
        if (mpi_x)
            {
            // halfway is now exactly on a domain boundary
            if (pos.x)
                {
                UP_ASSERT_EQUAL(origin.x, 4);
                }
            else
                {
                UP_ASSERT_EQUAL(origin.x, -1);
                }
            }
        else
            {
            UP_ASSERT_EQUAL(origin.x, 0);
            }

        if (mpi_y)
            {
            // this edge falls halfway in the middle of cell 4 now
            if (pos.y)
                {
                UP_ASSERT_EQUAL(origin.y, 4);
                }
            else
                {
                UP_ASSERT_EQUAL(origin.y, -1);
                }
            }
        else
            {
            UP_ASSERT_EQUAL(origin.y, 0);
            }

        if (mpi_z)
            {
            // this edge falls halfway in the middle of cell 5 now
            if (pos.z)
                {
                UP_ASSERT_EQUAL(origin.z, 5);
                }
            else
                {
                UP_ASSERT_EQUAL(origin.z, -1);
                }
            }
        else
            {
            UP_ASSERT_EQUAL(origin.z, 0);
            }

        // check domain sizes
        const uint3 dim = cl->getDim();
        if (mpi_x)
            {
            // split evenly in x -> both domains are same size
            UP_ASSERT_EQUAL(dim.x, 7);
            }
        else
            {
            UP_ASSERT_EQUAL(dim.x, 10);
            }

        if (mpi_y)
            {
            // biased to lower edge -> upper domains need extra cell
            if (pos.y)
                {
                UP_ASSERT_EQUAL(dim.y, 7);
                }
            else
                {
                UP_ASSERT_EQUAL(dim.y, 6);
                }
            }
        else
            {
            UP_ASSERT_EQUAL(dim.y, 10);
            }

        if (mpi_z)
            {
            // biased to upper edge -> lower domains need extra cell
            if (pos.z)
                {
                UP_ASSERT_EQUAL(dim.z, 6);
                }
            else
                {
                UP_ASSERT_EQUAL(dim.z, 7);
                }
            }
        else
            {
            UP_ASSERT_EQUAL(dim.z, 10);
            }

        std::array<unsigned int, 6> num_comm = cl->getNComm();
        if (mpi_x)
            {
            UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::east)], 2);
            UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::west)], 2);
            }
        else
            {
            UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::east)], 0);
            UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::west)], 0);
            }

        if (mpi_y)
            {
            if (pos.y)
                {
                UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::north)], 2);
                UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::south)], 1);
                }
            else
                {
                UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::north)], 1);
                UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::south)], 2);
                }
            }
        else
            {
            UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::north)], 0);
            UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::south)], 0);
            }

        if (mpi_z)
            {
            if (pos.z)
                {
                UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::up)], 2);
                UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::down)], 1);
                }
            else
                {
                UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::up)], 1);
                UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::down)], 2);
                }
            }
        else
            {
            UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::up)], 0);
            UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::down)], 0);
            }

        UP_ASSERT_EXCEPTION(std::runtime_error,
                            [&] { cl->setGridShift(make_scalar3(-0.3, -0.3, -0.3)); });
        UP_ASSERT_EXCEPTION(std::runtime_error,
                            [&] { cl->setGridShift(make_scalar3(0.3, 0.3, 0.3)); });

        const BoxDim coverage = cl->getCoverageBox();
        if (mpi_x)
            {
            if (pos.x)
                {
                CHECK_CLOSE(coverage.getLo().x, -0.25, tol);
                CHECK_CLOSE(coverage.getHi().x, 2.75, tol);
                }
            else
                {
                CHECK_CLOSE(coverage.getLo().x, -2.75, tol);
                CHECK_CLOSE(coverage.getHi().x, 0.25, tol);
                }
            }
        else
            {
            CHECK_CLOSE(coverage.getLo().x, -2.5, tol);
            CHECK_CLOSE(coverage.getHi().x, 2.5, tol);
            }

        if (mpi_y)
            {
            if (pos.y)
                {
                CHECK_CLOSE(coverage.getLo().y, -0.25, tol);
                CHECK_CLOSE(coverage.getHi().y, 2.75, tol);
                }
            else
                {
                CHECK_CLOSE(coverage.getLo().y, -2.75, tol);
                CHECK_CLOSE(coverage.getHi().y, -0.25, tol);
                }
            }
        else
            {
            CHECK_CLOSE(coverage.getLo().y, -2.5, tol);
            CHECK_CLOSE(coverage.getHi().y, 2.5, tol);
            }

        if (mpi_z)
            {
            if (pos.z)
                {
                CHECK_CLOSE(coverage.getLo().z, 0.25, tol);
                CHECK_CLOSE(coverage.getHi().z, 2.75, tol);
                }
            else
                {
                CHECK_CLOSE(coverage.getLo().z, -2.75, tol);
                CHECK_CLOSE(coverage.getHi().z, 0.25, tol);
                }
            }
        else
            {
            CHECK_CLOSE(coverage.getLo().z, -2.5, tol);
            CHECK_CLOSE(coverage.getHi().z, 2.5, tol);
            }
        }

    /*******************/
    // Increase the number of communication cells. This will trigger an increase in the size of the
    // diffusion layer
    cl->setNExtraCells(1);
    cl->computeDimensions();
        {
        // all origins should be shifted down by 1 cell
        const int3 origin = cl->getOriginIndex();
        const uint3 pos = decomposition->getGridPos();
        if (mpi_x)
            {
            if (pos.x)
                {
                UP_ASSERT_EQUAL(origin.x, 3);
                }
            else
                {
                UP_ASSERT_EQUAL(origin.x, -2);
                }
            }
        else
            {
            UP_ASSERT_EQUAL(origin.x, 0);
            }

        if (mpi_y)
            {
            if (pos.y)
                {
                UP_ASSERT_EQUAL(origin.y, 3);
                }
            else
                {
                UP_ASSERT_EQUAL(origin.y, -2);
                }
            }
        else
            {
            UP_ASSERT_EQUAL(origin.y, 0);
            }

        if (mpi_z)
            {
            if (pos.z)
                {
                UP_ASSERT_EQUAL(origin.z, 4);
                }
            else
                {
                UP_ASSERT_EQUAL(origin.z, -2);
                }
            }
        else
            {
            UP_ASSERT_EQUAL(origin.z, 0);
            }

        // all dims should be increased by 2
        const uint3 dim = cl->getDim();
        if (mpi_x)
            {
            UP_ASSERT_EQUAL(dim.x, 9);
            }
        else
            {
            UP_ASSERT_EQUAL(dim.x, 10);
            }

        if (mpi_y)
            {
            if (pos.y)
                {
                UP_ASSERT_EQUAL(dim.y, 9);
                }
            else
                {
                UP_ASSERT_EQUAL(dim.y, 8);
                }
            }
        else
            {
            UP_ASSERT_EQUAL(dim.y, 10);
            }

        if (mpi_z)
            {
            if (pos.z)
                {
                UP_ASSERT_EQUAL(dim.z, 8);
                }
            else
                {
                UP_ASSERT_EQUAL(dim.z, 9);
                }
            }
        else
            {
            UP_ASSERT_EQUAL(dim.z, 10);
            }

        // all comms should be increased by 1
        std::array<unsigned int, 6> num_comm = cl->getNComm();
        if (mpi_x)
            {
            UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::east)], 3);
            UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::west)], 3);
            }
        else
            {
            UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::east)], 0);
            UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::west)], 0);
            }

        if (mpi_y)
            {
            if (pos.y)
                {
                UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::north)], 3);
                UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::south)], 2);
                }
            else
                {
                UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::north)], 2);
                UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::south)], 3);
                }
            }
        else
            {
            UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::north)], 0);
            UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::south)], 0);
            }

        if (mpi_z)
            {
            if (pos.z)
                {
                UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::up)], 3);
                UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::down)], 2);
                }
            else
                {
                UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::up)], 2);
                UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::down)], 3);
                }
            }
        else
            {
            UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::up)], 0);
            UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::down)], 0);
            }

        UP_ASSERT_EXCEPTION(std::runtime_error,
                            [&] { cl->setGridShift(make_scalar3(-0.3, -0.3, -0.3)); });
        UP_ASSERT_EXCEPTION(std::runtime_error,
                            [&] { cl->setGridShift(make_scalar3(0.3, 0.3, 0.3)); });

        const BoxDim coverage = cl->getCoverageBox();
        if (mpi_x)
            {
            if (pos.x)
                {
                CHECK_CLOSE(coverage.getLo().x, -0.75, tol);
                CHECK_CLOSE(coverage.getHi().x, 3.25, tol);
                }
            else
                {
                CHECK_CLOSE(coverage.getLo().x, -3.25, tol);
                CHECK_CLOSE(coverage.getHi().x, 0.75, tol);
                }
            }
        else
            {
            CHECK_CLOSE(coverage.getLo().x, -2.5, tol);
            CHECK_CLOSE(coverage.getHi().x, 2.5, tol);
            }

        if (mpi_y)
            {
            if (pos.y)
                {
                CHECK_CLOSE(coverage.getLo().y, -0.75, tol);
                CHECK_CLOSE(coverage.getHi().y, 3.25, tol);
                }
            else
                {
                CHECK_CLOSE(coverage.getLo().y, -3.25, tol);
                CHECK_CLOSE(coverage.getHi().y, 0.25, tol);
                }
            }
        else
            {
            CHECK_CLOSE(coverage.getLo().y, -2.5, tol);
            CHECK_CLOSE(coverage.getHi().y, 2.5, tol);
            }

        if (mpi_z)
            {
            if (pos.z)
                {
                CHECK_CLOSE(coverage.getLo().z, -0.25, tol);
                CHECK_CLOSE(coverage.getHi().z, 3.25, tol);
                }
            else
                {
                CHECK_CLOSE(coverage.getLo().z, -3.25, tol);
                CHECK_CLOSE(coverage.getHi().z, 0.75, tol);
                }
            }
        else
            {
            CHECK_CLOSE(coverage.getLo().z, -2.5, tol);
            CHECK_CLOSE(coverage.getHi().z, 2.5, tol);
            }
        }
    }

//! Test for correct cell listing of a basic system
template<class CL> void celllist_basic_test(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    UP_ASSERT_EQUAL(exec_conf->getNRanks(), 8);

    std::shared_ptr<SnapshotSystemData<Scalar>> snap(new SnapshotSystemData<Scalar>());
    snap->global_box = std::make_shared<BoxDim>(6.0);
    snap->particle_data.type_mapping.push_back("A");
    // place each particle in the same cell, but on different ranks
    /*
     * The +/- halves of the box owned by each domain are:
     *    x y z
     * 0: - - -
     * 1: + - -
     * 2: - + -
     * 3: + + -
     * 4: - - +
     * 5: + - +
     * 6: - + +
     * 7: + + +
     */
    snap->mpcd_data.resize(8);
    snap->mpcd_data.type_mapping.push_back("A");
    snap->mpcd_data.position[0] = vec3<Scalar>(-0.1, -0.1, -0.1);
    snap->mpcd_data.position[1] = vec3<Scalar>(0.1, -0.1, -0.1);
    snap->mpcd_data.position[2] = vec3<Scalar>(-0.1, 0.1, -0.1);
    snap->mpcd_data.position[3] = vec3<Scalar>(0.1, 0.1, -0.1);
    snap->mpcd_data.position[4] = vec3<Scalar>(-0.1, -0.1, 0.1);
    snap->mpcd_data.position[5] = vec3<Scalar>(0.1, -0.1, 0.1);
    snap->mpcd_data.position[6] = vec3<Scalar>(-0.1, 0.1, 0.1);
    snap->mpcd_data.position[7] = vec3<Scalar>(0.1, 0.1, 0.1);

    std::shared_ptr<DomainDecomposition> decomposition(
        new DomainDecomposition(exec_conf, snap->global_box->getL(), 2, 2, 2));
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf, decomposition));
    std::shared_ptr<Communicator> pdata_comm(new Communicator(sysdef, decomposition));
    sysdef->setCommunicator(pdata_comm);

    // initialize mpcd system
    std::shared_ptr<mpcd::ParticleData> pdata = sysdef->getMPCDParticleData();
    std::shared_ptr<mpcd::CellList> cl(new CL(sysdef));
    cl->compute(0);
    const unsigned int my_rank = exec_conf->getRank();
        {
        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(),
                                            access_location::host,
                                            access_mode::read);
        ArrayHandle<unsigned int> h_cell_list(cl->getCellList(),
                                              access_location::host,
                                              access_mode::read);
        Index3D ci = cl->getCellIndexer();
        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::read);

        switch (my_rank)
            {
        case 0:
            // global index is (2,2,2), with origin (-1,-1,-1)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(3, 3, 3)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(3, 3, 3));
            break;
        case 1:
            // global index is (3,2,2), with origin (2,-1,-1)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(1, 3, 3)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(1, 3, 3));
            break;
        case 2:
            // global index is (2,3,2), with origin (-1,2,-1)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(3, 1, 3)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(3, 1, 3));
            break;
        case 3:
            // global index is (3,3,2), with origin (2,2,-1)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(1, 1, 3)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(1, 1, 3));
            break;
        case 4:
            // global index is (2,2,3), with origin (-1,-1,2)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(3, 3, 1)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(3, 3, 1));
            break;
        case 5:
            // global index is (3,2,3), with origin (2,-1,2)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(1, 3, 1)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(1, 3, 1));
            break;
        case 6:
            // global index is (2,3,3), with origin (-1,2,2)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(3, 1, 1)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(3, 1, 1));
            break;
        case 7:
            // global index is (3,3,3), with origin (2,2,2)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(1, 1, 1)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(1, 1, 1));
            break;
            };
        }

    // apply a grid shift so that particles move into the same cell (3,3,3)
    cl->setGridShift(make_scalar3(-0.5, -0.5, -0.5));
    cl->compute(1);
        {
        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(),
                                            access_location::host,
                                            access_mode::read);
        ArrayHandle<unsigned int> h_cell_list(cl->getCellList(),
                                              access_location::host,
                                              access_mode::read);
        Index3D ci = cl->getCellIndexer();
        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::read);

        switch (my_rank)
            {
        case 0:
            // global index is (3,3,3), with origin (-1,-1,-1)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(4, 4, 4)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(4, 4, 4));
            break;
        case 1:
            // global index is (3,3,3), with origin (2,-1,-1)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(1, 4, 4)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(1, 4, 4));
            break;
        case 2:
            // global index is (3,3,3), with origin (-1,2,-1)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(4, 1, 4)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(4, 1, 4));
            break;
        case 3:
            // global index is (3,3,3), with origin (2,2,-1)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(1, 1, 4)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(1, 1, 4));
            break;
        case 4:
            // global index is (3,3,3), with origin (-1,-1,2)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(4, 4, 1)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(4, 4, 1));
            break;
        case 5:
            // global index is (3,3,3), with origin (2,-1,2)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(1, 4, 1)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(1, 4, 1));
            break;
        case 6:
            // global index is (3,3,3), with origin (-1,2,2)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(4, 1, 1)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(4, 1, 1));
            break;
        case 7:
            // global index is (3,3,3), with origin (2,2,2)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(1, 1, 1)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(1, 1, 1));
            break;
            };
        }

    // apply a grid shift so that particles move into the same cell (2,2,2)
    cl->setGridShift(make_scalar3(0.5, 0.5, 0.5));
    cl->compute(2);
        {
        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(),
                                            access_location::host,
                                            access_mode::read);
        ArrayHandle<unsigned int> h_cell_list(cl->getCellList(),
                                              access_location::host,
                                              access_mode::read);
        Index3D ci = cl->getCellIndexer();
        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::read);

        switch (my_rank)
            {
        case 0:
            // global index is (2,2,2), with origin (-1,-1,-1)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(3, 3, 3)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(3, 3, 3));
            break;
        case 1:
            // global index is (2,2,2), with origin (2,-1,-1)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(0, 3, 3)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(0, 3, 3));
            break;
        case 2:
            // global index is (2,2,2), with origin (-1,2,-1)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(3, 0, 3)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(3, 0, 3));
            break;
        case 3:
            // global index is (2,2,2), with origin (2,2,-1)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(0, 0, 3)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(0, 0, 3));
            break;
        case 4:
            // global index is (2,2,2), with origin (-1,-1,2)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(3, 3, 0)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(3, 3, 0));
            break;
        case 5:
            // global index is (2,2,2), with origin (2,-1,2)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(0, 3, 0)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(0, 3, 0));
            break;
        case 6:
            // global index is (2,2,2), with origin (-1,2,2)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(3, 0, 0)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(3, 0, 0));
            break;
        case 7:
            // global index is (2,2,2), with origin (2,2,2)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(0, 0, 0)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(0, 0, 0));
            break;
            };
        }
    }

//! Test for correct cell listing of a system with particles on the edges
template<class CL> void celllist_edge_test(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    UP_ASSERT_EQUAL(exec_conf->getNRanks(), 8);

    std::shared_ptr<SnapshotSystemData<Scalar>> snap(new SnapshotSystemData<Scalar>());
    snap->global_box = std::make_shared<BoxDim>(5.0);
    snap->particle_data.type_mapping.push_back("A");
    // dummy initialize one particle to every domain, we will move them outside the domains for
    // the tests
    /*
     * The +/- halves of the box owned by each domain are:
     *    x y z
     * 0: - - -
     * 1: + - -
     * 2: - + -
     * 3: + + -
     * 4: - - +
     * 5: + - +
     * 6: - + +
     * 7: + + +
     */
    snap->mpcd_data.resize(8);
    snap->mpcd_data.type_mapping.push_back("A");
    snap->mpcd_data.position[0] = vec3<Scalar>(-1.0, -1.0, -1.0);
    snap->mpcd_data.position[1] = vec3<Scalar>(1.0, -1.0, -1.0);
    snap->mpcd_data.position[2] = vec3<Scalar>(-1.0, 1.0, -1.0);
    snap->mpcd_data.position[3] = vec3<Scalar>(1.0, 1.0, -1.0);
    snap->mpcd_data.position[4] = vec3<Scalar>(-1.0, -1.0, 1.0);
    snap->mpcd_data.position[5] = vec3<Scalar>(1.0, -1.0, 1.0);
    snap->mpcd_data.position[6] = vec3<Scalar>(-1.0, 1.0, 1.0);
    snap->mpcd_data.position[7] = vec3<Scalar>(1.0, 1.0, 1.0);
    std::vector<Scalar> fx {0.5};
    std::vector<Scalar> fy {0.45};
    std::vector<Scalar> fz {0.55};
    std::shared_ptr<DomainDecomposition> decomposition(
        new DomainDecomposition(exec_conf, snap->global_box->getL(), fx, fy, fz));
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf, decomposition));
    std::shared_ptr<Communicator> pdata_comm(new Communicator(sysdef, decomposition));
    sysdef->setCommunicator(pdata_comm);

    std::shared_ptr<mpcd::ParticleData> pdata = sysdef->getMPCDParticleData();
    std::shared_ptr<mpcd::CellList> cl(new CL(sysdef));

    // move particles to edges of domains for testing
    const unsigned int my_rank = exec_conf->getRank();
        {
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(),
                                   access_location::host,
                                   access_mode::overwrite);
        switch (my_rank)
            {
        case 0:
            h_pos.data[0] = make_scalar4(-0.01, -0.01, -0.01, __int_as_scalar(0));
            break;
        case 1:
            h_pos.data[0] = make_scalar4(0.0, -0.01, -0.01, __int_as_scalar(0));
            break;
        case 2:
            h_pos.data[0] = make_scalar4(-0.01, 0.0, -0.01, __int_as_scalar(0));
            break;
        case 3:
            h_pos.data[0] = make_scalar4(0.0, 0.0, -0.01, __int_as_scalar(0));
            break;
        case 4:
            h_pos.data[0] = make_scalar4(-0.01, -0.01, 0.0, __int_as_scalar(0));
            break;
        case 5:
            h_pos.data[0] = make_scalar4(0.0, -0.01, 0.0, __int_as_scalar(0));
            break;
        case 6:
            h_pos.data[0] = make_scalar4(-0.01, 0.0, 0.0, __int_as_scalar(0));
            break;
        case 7:
            h_pos.data[0] = make_scalar4(0.0, 0.0, 0.0, __int_as_scalar(0));
            break;
            };
        }

    cl->compute(0);

        {
        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(),
                                            access_location::host,
                                            access_mode::read);
        ArrayHandle<unsigned int> h_cell_list(cl->getCellList(),
                                              access_location::host,
                                              access_mode::read);
        Index3D ci = cl->getCellIndexer();
        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::read);

        switch (my_rank)
            {
        case 0:
            // global index is (2,2,2), with origin (-1,-1,-1)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(3, 3, 3)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(3, 3, 3));
            break;
        case 1:
            // global index is (2,2,2), with origin (2,-1,-1)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(0, 3, 3)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(0, 3, 3));
            break;
        case 2:
            // global index is (2,2,2), with origin (-1,1,-1)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(3, 1, 3)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(3, 1, 3));
            break;
        case 3:
            // global index is (2,2,2), with origin (2,1,-1)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(0, 1, 3)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(0, 1, 3));
            break;
        case 4:
            // global index is (2,2,2), with origin (-1,-1,2)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(3, 3, 0)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(3, 3, 0));
            break;
        case 5:
            // global index is (2,2,2), with origin (2,-1,2)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(0, 3, 0)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(0, 3, 0));
            break;
        case 6:
            // global index is (2,2,2), with origin (-1,1,2)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(3, 1, 0)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(3, 1, 0));
            break;
        case 7:
            // global index is (2,2,2), with origin (2,1,2)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(0, 1, 0)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(0, 1, 0));
            break;
            };
        }

    // apply a grid shift, particles on left internal boundary will move up one cell
    // particles on right internal boundary will stay in the same cell
    cl->setGridShift(make_scalar3(-0.5, -0.5, -0.5));
    cl->compute(1);
        {
        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(),
                                            access_location::host,
                                            access_mode::read);
        ArrayHandle<unsigned int> h_cell_list(cl->getCellList(),
                                              access_location::host,
                                              access_mode::read);
        Index3D ci = cl->getCellIndexer();
        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::read);

        switch (my_rank)
            {
        case 0:
            // global index is (2,2,2), with origin (-1,-1,-1)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(3, 3, 3)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(3, 3, 3));
            break;
        case 1:
            // global index is (3,2,2), with origin (2,-1,-1)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(1, 3, 3)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(1, 3, 3));
            break;
        case 2:
            // global index is (2,3,2), with origin (-1,1,-1)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(3, 2, 3)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(3, 2, 3));
            break;
        case 3:
            // global index is (3,3,2), with origin (2,1,-1)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(1, 2, 3)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(1, 2, 3));
            break;
        case 4:
            // global index is (2,2,3), with origin (-1,-1,2)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(3, 3, 1)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(3, 3, 1));
            break;
        case 5:
            // global index is (3,2,3), with origin (2,-1,2)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(1, 3, 1)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(1, 3, 1));
            break;
        case 6:
            // global index is (2,3,3), with origin (-1,1,2)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(3, 2, 1)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(3, 2, 1));
            break;
        case 7:
            // global index is (3,3,3), with origin (2,1,2)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(1, 2, 1)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(1, 2, 1));
            break;
            };
        }

    // apply a grid shift, particles on left internal boundary will stay in cell
    // particles on right internal boundary will move down one cell
    cl->setGridShift(make_scalar3(0.5, 0.5, 0.5));
    cl->compute(2);
        {
        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(),
                                            access_location::host,
                                            access_mode::read);
        ArrayHandle<unsigned int> h_cell_list(cl->getCellList(),
                                              access_location::host,
                                              access_mode::read);
        Index3D ci = cl->getCellIndexer();
        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::read);

        switch (my_rank)
            {
        case 0:
            // global index is (1,1,1), with origin (-1,-1,-1)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(2, 2, 2)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(2, 2, 2));
            break;
        case 1:
            // global index is (2,1,1), with origin (2,-1,-1)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(0, 2, 2)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(0, 2, 2));
            break;
        case 2:
            // global index is (1,2,1), with origin (-1,1,-1)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(2, 1, 2)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(2, 1, 2));
            break;
        case 3:
            // global index is (2,2,1), with origin (2,1,-1)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(0, 1, 2)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(0, 1, 2));
            break;
        case 4:
            // global index is (1,1,2), with origin (-1,-1,2)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(2, 2, 0)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(2, 2, 0));
            break;
        case 5:
            // global index is (2,1,2), with origin (2,-1,2)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(0, 2, 0)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(0, 2, 0));
            break;
        case 6:
            // global index is (1,2,2), with origin (-1,1,2)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(2, 1, 0)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(2, 1, 0));
            break;
        case 7:
            // global index is (2,2,2), with origin (2,1,2)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(0, 1, 0)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(0, 1, 0));
            break;
            };
        }
    // now we move the particles to their exterior boundaries, and repeat the testing process
    // we are going to pad the cell list with an extra cell just to test that binning now
    cl->setNExtraCells(1);
        {
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(),
                                   access_location::host,
                                   access_mode::overwrite);
        switch (my_rank)
            {
        case 0:
            h_pos.data[0] = make_scalar4(-4.0, -4.0, -4.0, __int_as_scalar(0));
            break;
        case 1:
            h_pos.data[0] = make_scalar4(3.99, -4.0, -4.0, __int_as_scalar(0));
            break;
        case 2:
            h_pos.data[0] = make_scalar4(-4.0, 3.99, -4.0, __int_as_scalar(0));
            break;
        case 3:
            h_pos.data[0] = make_scalar4(3.99, 3.99, -4.0, __int_as_scalar(0));
            break;
        case 4:
            h_pos.data[0] = make_scalar4(-4.0, -4.0, 3.99, __int_as_scalar(0));
            break;
        case 5:
            h_pos.data[0] = make_scalar4(3.99, -4.0, 3.99, __int_as_scalar(0));
            break;
        case 6:
            h_pos.data[0] = make_scalar4(-4.0, 3.99, 3.99, __int_as_scalar(0));
            break;
        case 7:
            h_pos.data[0] = make_scalar4(3.99, 3.99, 3.99, __int_as_scalar(0));
            break;
            };
        }

    // reset the grid shift and recompute
    cl->setGridShift(make_scalar3(0, 0, 0));
    cl->compute(3);
        {
        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(),
                                            access_location::host,
                                            access_mode::read);
        ArrayHandle<unsigned int> h_cell_list(cl->getCellList(),
                                              access_location::host,
                                              access_mode::read);
        Index3D ci = cl->getCellIndexer();
        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::read);

        switch (my_rank)
            {
        case 0:
            // global index is (-2,-2,-2), with origin (-2,-2,-2)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(0, 0, 0)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(0, 0, 0));
            break;
        case 1:
            // global index is (6,-2,-2), with origin (1,-2,-2)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(5, 0, 0)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(5, 0, 0));
            break;
        case 2:
            // global index is (-2,6,-2), with origin (-2,0,-2)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(0, 6, 0)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(0, 6, 0));
            break;
        case 3:
            // global index is (6,6,-2), with origin (1,0,-2)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(5, 6, 0)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(5, 6, 0));
            break;
        case 4:
            // global index is (-2,-2,6), with origin (-2,-2,1)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(0, 0, 5)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(0, 0, 5));
            break;
        case 5:
            // global index is (6,-2,6), with origin (1,-2,1)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(5, 0, 5)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(5, 0, 5));
            break;
        case 6:
            // global index is (-2,6,6), with origin (-2,0,1)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(0, 6, 5)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(0, 6, 5));
            break;
        case 7:
            // global index is (6,6,6), with origin (1,0,1)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(5, 6, 5)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(5, 6, 5));
            break;
            };
        }

    // shift all particles to the right by 0.5
    // all particles on left bound will move up one cell, all particles on right bound stay
    cl->setGridShift(make_scalar3(-0.5, -0.5, -0.5));
    cl->compute(4);
        {
        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(),
                                            access_location::host,
                                            access_mode::read);
        ArrayHandle<unsigned int> h_cell_list(cl->getCellList(),
                                              access_location::host,
                                              access_mode::read);
        Index3D ci = cl->getCellIndexer();
        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::read);

        switch (my_rank)
            {
        case 0:
            // global index is (-1,-1,-1), with origin (-2,-2,-2)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(1, 1, 1)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(1, 1, 1));
            break;
        case 1:
            // global index is (6,-1,-1), with origin (1,-2,-2)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(5, 1, 1)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(5, 1, 1));
            break;
        case 2:
            // global index is (-1,6,-1), with origin (-2,0,-2)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(1, 6, 1)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(1, 6, 1));
            break;
        case 3:
            // global index is (6,6,-1), with origin (1,0,-2)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(5, 6, 1)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(5, 6, 1));
            break;
        case 4:
            // global index is (-1,-1,6), with origin (-2,-2,1)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(1, 1, 5)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(1, 1, 5));
            break;
        case 5:
            // global index is (6,-1,6), with origin (1,-2,1)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(5, 1, 5)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(5, 1, 5));
            break;
        case 6:
            // global index is (-1,6,6), with origin (-2,0,1)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(1, 6, 5)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(1, 6, 5));
            break;
        case 7:
            // global index is (6,6,6), with origin (1,0,1)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(5, 6, 5)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(5, 6, 5));
            break;
            };
        }

    // shift all particles left by 0.5
    // particles on the lower bound stay in the same bin, and particles on the right bound move down
    // one
    cl->setGridShift(make_scalar3(0.5, 0.5, 0.5));
    cl->compute(5);
        {
        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(),
                                            access_location::host,
                                            access_mode::read);
        ArrayHandle<unsigned int> h_cell_list(cl->getCellList(),
                                              access_location::host,
                                              access_mode::read);
        Index3D ci = cl->getCellIndexer();
        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::read);

        switch (my_rank)
            {
        case 0:
            // global index is (-2,-2,-2), with origin (-2,-2,-2)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(0, 0, 0)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(0, 0, 0));
            break;
        case 1:
            // global index is (5,-2,-2), with origin (1,-2,-2)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(4, 0, 0)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(4, 0, 0));
            break;
        case 2:
            // global index is (-2,5,-2), with origin (-2,0,-2)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(0, 5, 0)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(0, 5, 0));
            break;
        case 3:
            // global index is (5,5,-2), with origin (1,0,-2)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(4, 5, 0)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(4, 5, 0));
            break;
        case 4:
            // global index is (-2,-2,5), with origin (-2,-2,1)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(0, 0, 4)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(0, 0, 4));
            break;
        case 5:
            // global index is (5,-2,5), with origin (1,-2,1)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(4, 0, 4)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(4, 0, 4));
            break;
        case 6:
            // global index is (-2,5,5), with origin (-2,0,1)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(0, 5, 4)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(0, 5, 4));
            break;
        case 7:
            // global index is (5,5,5), with origin (1,0,1)
            UP_ASSERT_EQUAL(h_cell_np.data[ci(4, 5, 4)], 1);
            UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), ci(4, 5, 4));
            break;
            };
        }
    }

//! dimension test case for MPCD CellList class
UP_TEST(mpcd_cell_list_dimensions)
    {
        // mpi in 1d
        {
        std::shared_ptr<ExecutionConfiguration> exec_conf(
            new ExecutionConfiguration(ExecutionConfiguration::CPU, std::vector<int>()));
        exec_conf->getMPIConfig()->splitPartitions(2);
        celllist_dimension_test<mpcd::CellList>(exec_conf, true, false, false);
        celllist_dimension_test<mpcd::CellList>(exec_conf, false, true, false);
        celllist_dimension_test<mpcd::CellList>(exec_conf, false, false, true);
        }
        // mpi in 2d
        {
        std::shared_ptr<ExecutionConfiguration> exec_conf(
            new ExecutionConfiguration(ExecutionConfiguration::CPU, std::vector<int>()));
        exec_conf->getMPIConfig()->splitPartitions(4);
        celllist_dimension_test<mpcd::CellList>(exec_conf, true, true, false);
        celllist_dimension_test<mpcd::CellList>(exec_conf, true, false, true);
        celllist_dimension_test<mpcd::CellList>(exec_conf, false, true, true);
        }
        // mpi in 3d
        {
        std::shared_ptr<ExecutionConfiguration> exec_conf(
            new ExecutionConfiguration(ExecutionConfiguration::CPU, std::vector<int>()));
        exec_conf->getMPIConfig()->splitPartitions(8);
        celllist_dimension_test<mpcd::CellList>(exec_conf, true, true, true);
        }
    }

//! basic test case for MPCD CellList class
UP_TEST(mpcd_cell_list_basic_test)
    {
    celllist_basic_test<mpcd::CellList>(std::shared_ptr<ExecutionConfiguration>(
        new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

//! edge test case for MPCD CellList class
UP_TEST(mpcd_cell_list_edge_test)
    {
    celllist_edge_test<mpcd::CellList>(std::shared_ptr<ExecutionConfiguration>(
        new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

#ifdef ENABLE_HIP
//! dimension test case for MPCD CellListGPU class
UP_TEST(mpcd_cell_list_gpu_dimensions)
    {
        // mpi in 1d
        {
        std::shared_ptr<ExecutionConfiguration> exec_conf(
            new ExecutionConfiguration(ExecutionConfiguration::GPU, std::vector<int>()));
        exec_conf->getMPIConfig()->splitPartitions(2);
        celllist_dimension_test<mpcd::CellListGPU>(exec_conf, true, false, false);
        celllist_dimension_test<mpcd::CellListGPU>(exec_conf, false, true, false);
        celllist_dimension_test<mpcd::CellListGPU>(exec_conf, false, false, true);
        }
        // mpi in 2d
        {
        std::shared_ptr<ExecutionConfiguration> exec_conf(
            new ExecutionConfiguration(ExecutionConfiguration::GPU, std::vector<int>()));
        exec_conf->getMPIConfig()->splitPartitions(4);
        celllist_dimension_test<mpcd::CellListGPU>(exec_conf, true, true, false);
        celllist_dimension_test<mpcd::CellListGPU>(exec_conf, true, false, true);
        celllist_dimension_test<mpcd::CellListGPU>(exec_conf, false, true, true);
        }
        // mpi in 3d
        {
        std::shared_ptr<ExecutionConfiguration> exec_conf(
            new ExecutionConfiguration(ExecutionConfiguration::GPU, std::vector<int>()));
        exec_conf->getMPIConfig()->splitPartitions(8);
        celllist_dimension_test<mpcd::CellListGPU>(exec_conf, true, true, true);
        }
    }

//! basic test case for MPCD CellListGPU class
UP_TEST(mpcd_cell_list_gpu_basic_test)
    {
    celllist_basic_test<mpcd::CellListGPU>(std::shared_ptr<ExecutionConfiguration>(
        new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! edge test case for MPCD CellListGPU class
UP_TEST(mpcd_cell_list_gpu_edge_test)
    {
    celllist_edge_test<mpcd::CellListGPU>(std::shared_ptr<ExecutionConfiguration>(
        new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
#endif // ENABLE_HIP
