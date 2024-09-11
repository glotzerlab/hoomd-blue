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

#include "utils.h"

HOOMD_UP_MAIN()

using namespace hoomd;

void checkDomainBoundaries(std::shared_ptr<SystemDefinition> sysdef,
                           std::shared_ptr<mpcd::CellList> cl)
    {
    auto pdata = sysdef->getParticleData();
    auto exec_conf = pdata->getExecConf();
    MPI_Comm mpi_comm = exec_conf->getMPICommunicator();
    auto decomposition = pdata->getDomainDecomposition();

    auto num_comm = cl->getNComm();
    auto origin_idx = cl->getOriginIndex();
    auto cell_dim = cl->getDim();
    auto num_extra = cl->getNExtraCells();

    for (unsigned int dir = 0; dir < num_comm.size(); ++dir)
        {
        mpcd::detail::face d = static_cast<mpcd::detail::face>(dir);
        if (!cl->isCommunicating(d))
            continue;

        // receive in the opposite direction from which we send
        unsigned int send_neighbor = decomposition->getNeighborRank(dir);
        unsigned int recv_neighbor;
        if (dir % 2 == 0)
            recv_neighbor = decomposition->getNeighborRank(dir + 1);
        else
            recv_neighbor = decomposition->getNeighborRank(dir - 1);

        // first make sure each dimension is sending and receiving the same size data
        MPI_Request reqs[2];
        MPI_Status status[2];

        // check that the number received is the same as that being sent from neighbor
        unsigned int n_send = num_comm[dir];
        unsigned int n_expect_recv;
        if (dir % 2 == 0)
            n_expect_recv = num_comm[dir + 1];
        else
            n_expect_recv = num_comm[dir - 1];

        unsigned int n_recv;
        MPI_Isend(&n_send, 1, MPI_UNSIGNED, send_neighbor, 0, mpi_comm, &reqs[0]);
        MPI_Irecv(&n_recv, 1, MPI_UNSIGNED, recv_neighbor, 0, mpi_comm, &reqs[1]);
        MPI_Waitall(2, reqs, status);
        UP_ASSERT_EQUAL(n_recv, n_expect_recv);

        // check that the same cell ids are communicated
        std::vector<int> send_cells(n_send), recv_cells(n_recv);
        for (unsigned int i = 0; i < n_send; ++i)
            {
            if (d == mpcd::detail::face::east)
                {
                send_cells[i] = origin_idx.x + cell_dim.x - num_extra - n_send + i;
                }
            else if (d == mpcd::detail::face::west)
                {
                send_cells[i] = origin_idx.x + i;
                }
            else if (d == mpcd::detail::face::north)
                {
                send_cells[i] = origin_idx.y + cell_dim.y - num_extra - n_send + i;
                }
            else if (d == mpcd::detail::face::south)
                {
                send_cells[i] = origin_idx.y + i;
                }
            else if (d == mpcd::detail::face::up)
                {
                send_cells[i] = origin_idx.z + cell_dim.z - num_extra - n_send + i;
                }
            else if (d == mpcd::detail::face::down)
                {
                send_cells[i] = origin_idx.z + i;
                }
            }

        MPI_Isend(&send_cells[0], n_send, MPI_INT, send_neighbor, 1, mpi_comm, &reqs[0]);
        MPI_Irecv(&recv_cells[0], n_recv, MPI_INT, recv_neighbor, 1, mpi_comm, &reqs[1]);
        MPI_Waitall(2, reqs, status);

        for (unsigned int i = 0; i < n_recv; ++i)
            {
            // wrap the received cell back into the global box
            // only two of the entries will be valid, the others are dummies
            int3 recv_cell = make_int3(0, 0, 0);
            if (d == mpcd::detail::face::east || d == mpcd::detail::face::west)
                {
                recv_cell.x = recv_cells[i];
                }
            else if (d == mpcd::detail::face::north || d == mpcd::detail::face::south)
                {
                recv_cell.y = recv_cells[i];
                }
            else if (d == mpcd::detail::face::up || d == mpcd::detail::face::down)
                {
                recv_cell.z = recv_cells[i];
                }
            recv_cell = cl->wrapGlobalCell(recv_cell);

            // compute the expected cell to receive, also wrapped
            int3 expect_recv_cell = make_int3(0, 0, 0);
            if (d == mpcd::detail::face::east)
                {
                expect_recv_cell.x = origin_idx.x + i;
                }
            else if (d == mpcd::detail::face::west)
                {
                expect_recv_cell.x = origin_idx.x + cell_dim.x - num_extra - n_recv + i;
                }
            else if (d == mpcd::detail::face::north)
                {
                expect_recv_cell.y = origin_idx.y + i;
                }
            else if (d == mpcd::detail::face::south)
                {
                expect_recv_cell.y = origin_idx.y + cell_dim.y - num_extra - n_recv + i;
                }
            else if (d == mpcd::detail::face::up)
                {
                expect_recv_cell.z = origin_idx.z + i;
                }
            else if (d == mpcd::detail::face::down)
                {
                expect_recv_cell.z = origin_idx.z + cell_dim.z - num_extra - n_recv + i;
                }
            expect_recv_cell = cl->wrapGlobalCell(expect_recv_cell);

            UP_ASSERT_EQUAL(recv_cell.x, expect_recv_cell.x);
            UP_ASSERT_EQUAL(recv_cell.y, expect_recv_cell.y);
            UP_ASSERT_EQUAL(recv_cell.z, expect_recv_cell.z);
            }
        }
    }

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
                             bool mpi_z,
                             const Scalar3& tilt)
    {
    // only run tests on first partition
    if (exec_conf->getPartition() != 0)
        return;

    std::shared_ptr<SnapshotSystemData<Scalar>> snap(new SnapshotSystemData<Scalar>());
    snap->global_box = std::make_shared<BoxDim>(5.0);
    snap->global_box->setTiltFactors(tilt.x, tilt.y, tilt.z);
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
    std::shared_ptr<mpcd::CellList> cl(new CL(sysdef, make_uint3(5, 5, 5), false));

    // compute the cell list
    cl->computeDimensions();
    checkDomainBoundaries(sysdef, cl);

    const bool is_orthorhombic = (tilt == make_scalar3(0, 0, 0));
    if (is_orthorhombic)
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
            if (pos.x)
                {
                UP_ASSERT_EQUAL(dim.x, 4);
                }
            else
                {
                UP_ASSERT_EQUAL(dim.x, 4);
                }
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
        UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::east)],
                        (mpi_x) ? ((pos.x) ? 2 : 1) : 0);
        UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::west)],
                        (mpi_x) ? ((pos.x) ? 1 : 2) : 0);
        UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::north)],
                        (mpi_y) ? 2 : 0);
        UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::south)],
                        (mpi_y) ? 2 : 0);
        UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::up)],
                        (mpi_z) ? 2 : 0);
        UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::down)],
                        (mpi_z) ? 2 : 0);

        // check coverage box
        const BoxDim coverage = cl->getCoverageBox();
        if (mpi_x)
            {
            if (pos.x)
                {
                UP_ASSERT_SMALL(coverage.getLo().x, tol);
                UP_ASSERT_CLOSE(coverage.getHi().x, 3.0, tol);
                }
            else
                {
                UP_ASSERT_CLOSE(coverage.getLo().x, -3.0, tol);
                UP_ASSERT_SMALL(coverage.getHi().x, tol);
                }
            }
        else
            {
            UP_ASSERT_CLOSE(coverage.getLo().x, -2.5, tol);
            UP_ASSERT_CLOSE(coverage.getHi().x, 2.5, tol);
            }

        if (mpi_y)
            {
            if (pos.y)
                {
                UP_ASSERT_CLOSE(coverage.getLo().y, -1.0, tol);
                UP_ASSERT_CLOSE(coverage.getHi().y, 3.0, tol);
                }
            else
                {
                UP_ASSERT_CLOSE(coverage.getLo().y, -3.0, tol);
                UP_ASSERT_SMALL(coverage.getHi().y, tol);
                }
            }
        else
            {
            UP_ASSERT_CLOSE(coverage.getLo().y, -2.5, tol);
            UP_ASSERT_CLOSE(coverage.getHi().y, 2.5, tol);
            }

        if (mpi_z)
            {
            if (pos.z)
                {
                UP_ASSERT_SMALL(coverage.getLo().z, tol);
                UP_ASSERT_CLOSE(coverage.getHi().z, 3.0, tol);
                }
            else
                {
                UP_ASSERT_CLOSE(coverage.getLo().z, -3.0, tol);
                UP_ASSERT_CLOSE(coverage.getHi().z, 1.0, tol);
                }
            }
        else
            {
            UP_ASSERT_CLOSE(coverage.getLo().z, -2.5, tol);
            UP_ASSERT_CLOSE(coverage.getHi().z, 2.5, tol);
            }
        }

    /*******************/
    // Change the cell size, and ensure everything stays up to date
    cl->setGlobalDim(make_uint3(10, 10, 10));
    cl->computeDimensions();
    checkDomainBoundaries(sysdef, cl);
    if (is_orthorhombic)
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
                // floating point rounding makes this 8 not 7 (extra cell communicated on this
                // edge)
                UP_ASSERT_EQUAL(dim.z, 8);
                }
            }
        else
            {
            UP_ASSERT_EQUAL(dim.z, 10);
            }

        std::array<unsigned int, 6> num_comm = cl->getNComm();
        UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::east)],
                        (mpi_x) ? 2 : 0);
        UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::west)],
                        (mpi_x) ? 2 : 0);
        UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::north)],
                        (mpi_y) ? ((pos.y) ? 2 : 1) : 0);
        UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::south)],
                        (mpi_y) ? ((pos.y) ? 1 : 2) : 0);
        UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::up)],
                        (mpi_z) ? 2 : 0);
        UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::down)],
                        (mpi_z) ? 2 : 0);

        const BoxDim coverage = cl->getCoverageBox();
        if (mpi_x)
            {
            if (pos.x)
                {
                UP_ASSERT_CLOSE(coverage.getLo().x, -0.25, tol);
                UP_ASSERT_CLOSE(coverage.getHi().x, 2.75, tol);
                }
            else
                {
                UP_ASSERT_CLOSE(coverage.getLo().x, -2.75, tol);
                UP_ASSERT_CLOSE(coverage.getHi().x, 0.25, tol);
                }
            }
        else
            {
            UP_ASSERT_CLOSE(coverage.getLo().x, -2.5, tol);
            UP_ASSERT_CLOSE(coverage.getHi().x, 2.5, tol);
            }

        if (mpi_y)
            {
            if (pos.y)
                {
                UP_ASSERT_CLOSE(coverage.getLo().y, -0.25, tol);
                UP_ASSERT_CLOSE(coverage.getHi().y, 2.75, tol);
                }
            else
                {
                UP_ASSERT_CLOSE(coverage.getLo().y, -2.75, tol);
                UP_ASSERT_CLOSE(coverage.getHi().y, -0.25, tol);
                }
            }
        else
            {
            UP_ASSERT_CLOSE(coverage.getLo().y, -2.5, tol);
            UP_ASSERT_CLOSE(coverage.getHi().y, 2.5, tol);
            }

        if (mpi_z)
            {
            if (pos.z)
                {
                UP_ASSERT_CLOSE(coverage.getLo().z, 0.25, tol);
                UP_ASSERT_CLOSE(coverage.getHi().z, 2.75, tol);
                }
            else
                {
                UP_ASSERT_CLOSE(coverage.getLo().z, -2.75, tol);
                // floating point rounding makes this 0.75 not 0.25
                UP_ASSERT_CLOSE(coverage.getHi().z, 0.75, tol);
                }
            }
        else
            {
            UP_ASSERT_CLOSE(coverage.getLo().z, -2.5, tol);
            UP_ASSERT_CLOSE(coverage.getHi().z, 2.5, tol);
            }
        }

    /*******************/
    // Increase the number of communication cells. This will trigger an increase in the size of
    // the diffusion layer
    cl->setNExtraCells(1);
    cl->computeDimensions();
    checkDomainBoundaries(sysdef, cl);
    if (is_orthorhombic)
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
                // floating point rounding makes thes 10 not 9
                UP_ASSERT_EQUAL(dim.z, 10);
                }
            }
        else
            {
            UP_ASSERT_EQUAL(dim.z, 10);
            }

        // all comms should be increased by 1
        std::array<unsigned int, 6> num_comm = cl->getNComm();
        UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::east)],
                        (mpi_x) ? 3 : 0);
        UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::west)],
                        (mpi_x) ? 3 : 0);
        UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::north)],
                        (mpi_y) ? ((pos.y) ? 3 : 2) : 0);
        UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::south)],
                        (mpi_y) ? ((pos.y) ? 2 : 3) : 0);
        UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::up)],
                        (mpi_z) ? 3 : 0);
        UP_ASSERT_EQUAL(num_comm[static_cast<unsigned int>(mpcd::detail::face::down)],
                        (mpi_z) ? 3 : 0);

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
                // floating point rounding makes this 1.25 not 0.75
                CHECK_CLOSE(coverage.getHi().z, 1.25, tol);
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
template<class CL>
void celllist_basic_test(std::shared_ptr<ExecutionConfiguration> exec_conf,
                         const Scalar3& L,
                         const Scalar3& tilt)
    {
    UP_ASSERT_EQUAL(exec_conf->getNRanks(), 8);

    auto ref_box = std::make_shared<BoxDim>(6.0);
    auto box = std::make_shared<BoxDim>(L);
    box->setTiltFactors(tilt.x, tilt.y, tilt.z);

    std::shared_ptr<SnapshotSystemData<Scalar>> snap(new SnapshotSystemData<Scalar>());
    snap->global_box = box;
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
    snap->mpcd_data.position[0] = scale(vec3<Scalar>(-0.1, -0.1, -0.1), ref_box, box);
    snap->mpcd_data.position[1] = scale(vec3<Scalar>(0.1, -0.1, -0.1), ref_box, box);
    snap->mpcd_data.position[2] = scale(vec3<Scalar>(-0.1, 0.1, -0.1), ref_box, box);
    snap->mpcd_data.position[3] = scale(vec3<Scalar>(0.1, 0.1, -0.1), ref_box, box);
    snap->mpcd_data.position[4] = scale(vec3<Scalar>(-0.1, -0.1, 0.1), ref_box, box);
    snap->mpcd_data.position[5] = scale(vec3<Scalar>(0.1, -0.1, 0.1), ref_box, box);
    snap->mpcd_data.position[6] = scale(vec3<Scalar>(-0.1, 0.1, 0.1), ref_box, box);
    snap->mpcd_data.position[7] = scale(vec3<Scalar>(0.1, 0.1, 0.1), ref_box, box);

    std::shared_ptr<DomainDecomposition> decomposition(
        new DomainDecomposition(exec_conf, snap->global_box->getL(), 2, 2, 2));
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf, decomposition));
    std::shared_ptr<Communicator> pdata_comm(new Communicator(sysdef, decomposition));
    sysdef->setCommunicator(pdata_comm);

    // initialize mpcd system
    std::shared_ptr<mpcd::ParticleData> pdata = sysdef->getMPCDParticleData();
    std::shared_ptr<mpcd::CellList> cl(new CL(sysdef, make_uint3(6, 6, 6), false));
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
    const Scalar3 shift = (Scalar(0.5) / 6) * make_scalar3(1, 1, 1);
    cl->setGridShift(-shift);
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
    cl->setGridShift(shift);
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
template<class CL>
void celllist_edge_test(std::shared_ptr<ExecutionConfiguration> exec_conf,
                        const Scalar3& L,
                        const Scalar3& tilt)
    {
    UP_ASSERT_EQUAL(exec_conf->getNRanks(), 8);

    auto ref_box = std::make_shared<BoxDim>(5.0);
    auto box = std::make_shared<BoxDim>(L);
    box->setTiltFactors(tilt.x, tilt.y, tilt.z);
    bool is_orthorhombic = tilt.x == Scalar(0.0) && tilt.y == Scalar(0.0) && tilt.z == Scalar(0.0);

    std::shared_ptr<SnapshotSystemData<Scalar>> snap(new SnapshotSystemData<Scalar>());
    snap->global_box = box;
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
    snap->mpcd_data.position[0] = scale(vec3<Scalar>(-1.0, -1.0, -1.0), ref_box, box);
    snap->mpcd_data.position[1] = scale(vec3<Scalar>(1.0, -1.0, -1.0), ref_box, box);
    snap->mpcd_data.position[2] = scale(vec3<Scalar>(-1.0, 1.0, -1.0), ref_box, box);
    snap->mpcd_data.position[3] = scale(vec3<Scalar>(1.0, 1.0, -1.0), ref_box, box);
    snap->mpcd_data.position[4] = scale(vec3<Scalar>(-1.0, -1.0, 1.0), ref_box, box);
    snap->mpcd_data.position[5] = scale(vec3<Scalar>(1.0, -1.0, 1.0), ref_box, box);
    snap->mpcd_data.position[6] = scale(vec3<Scalar>(-1.0, 1.0, 1.0), ref_box, box);
    snap->mpcd_data.position[7] = scale(vec3<Scalar>(1.0, 1.0, 1.0), ref_box, box);
    std::vector<Scalar> fx {0.5};
    std::vector<Scalar> fy {0.45};
    std::vector<Scalar> fz {0.55};
    std::shared_ptr<DomainDecomposition> decomposition(
        new DomainDecomposition(exec_conf, snap->global_box->getL(), fx, fy, fz));
    std::shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf, decomposition));
    std::shared_ptr<Communicator> pdata_comm(new Communicator(sysdef, decomposition));
    sysdef->setCommunicator(pdata_comm);

    std::shared_ptr<mpcd::ParticleData> pdata = sysdef->getMPCDParticleData();
    std::shared_ptr<mpcd::CellList> cl(new CL(sysdef, make_uint3(5, 5, 5), false));

    // move particles to edges of domains for testing
    const unsigned int my_rank = exec_conf->getRank();
        {
        ArrayHandle<Scalar4> h_pos(pdata->getPositions(),
                                   access_location::host,
                                   access_mode::overwrite);
        switch (my_rank)
            {
        case 0:
            h_pos.data[0]
                = scale(make_scalar4(-0.01, -0.01, -0.01, __int_as_scalar(0)), ref_box, box);
            break;
        case 1:
            h_pos.data[0]
                = scale(make_scalar4(0.0, -0.01, -0.01, __int_as_scalar(0)), ref_box, box);
            break;
        case 2:
            h_pos.data[0]
                = scale(make_scalar4(-0.01, 0.0, -0.01, __int_as_scalar(0)), ref_box, box);
            break;
        case 3:
            h_pos.data[0] = scale(make_scalar4(0.0, 0.0, -0.01, __int_as_scalar(0)), ref_box, box);
            break;
        case 4:
            h_pos.data[0]
                = scale(make_scalar4(-0.01, -0.01, 0.0, __int_as_scalar(0)), ref_box, box);
            break;
        case 5:
            h_pos.data[0] = scale(make_scalar4(0.0, -0.01, 0.0, __int_as_scalar(0)), ref_box, box);
            break;
        case 6:
            h_pos.data[0] = scale(make_scalar4(-0.01, 0.0, 0.0, __int_as_scalar(0)), ref_box, box);
            break;
        case 7:
            h_pos.data[0] = scale(make_scalar4(0.0, 0.0, 0.0, __int_as_scalar(0)), ref_box, box);
            break;
            };
        }

    cl->compute(0);
    checkDomainBoundaries(sysdef, cl);
        {
        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(),
                                            access_location::host,
                                            access_mode::read);
        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::read);
        const unsigned int local_cell = make_local_cell(cl, 2, 2, 2);
        UP_ASSERT_EQUAL(h_cell_np.data[local_cell], 1);
        UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), local_cell);
        }

    // apply a grid shift, particles on left internal boundary will move up one cell
    // particles on right internal boundary will stay in the same cell
    const Scalar3 shift = (Scalar(0.5) / 5) * make_scalar3(1, 1, 1);
        {
        cl->setGridShift(-shift);
        cl->compute(1);
        checkDomainBoundaries(sysdef, cl);
        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(),
                                            access_location::host,
                                            access_mode::read);
        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::read);

        int3 cell;
        switch (my_rank)
            {
        case 0:
            cell = make_int3(2, 2, 2);
            break;
        case 1:
            cell = make_int3(3, 2, 2);
            break;
        case 2:
            cell = make_int3(2, 3, 2);
            break;
        case 3:
            cell = make_int3(3, 3, 2);
            break;
        case 4:
            cell = make_int3(2, 2, 3);
            break;
        case 5:
            cell = make_int3(3, 2, 3);
            break;
        case 6:
            cell = make_int3(2, 3, 3);
            break;
        case 7:
            cell = make_int3(3, 3, 3);
            break;
            };

        const unsigned int local_cell = make_local_cell(cl, cell.x, cell.y, cell.z);
        UP_ASSERT_EQUAL(h_cell_np.data[local_cell], 1);
        UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), local_cell);
        }

        // apply a grid shift, particles on left internal boundary will stay in cell
        // particles on right internal boundary will move down one cell
        // if (*box == *ref_box)
        {
        cl->setGridShift(shift);
        cl->compute(2);
        checkDomainBoundaries(sysdef, cl);

        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(),
                                            access_location::host,
                                            access_mode::read);
        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::read);

        int3 cell;
        switch (my_rank)
            {
        case 0:
            cell = make_int3(1, 1, 1);
            break;
        case 1:
            cell = make_int3(2, 1, 1);
            break;
        case 2:
            cell = make_int3(1, 2, 1);
            break;
        case 3:
            cell = make_int3(2, 2, 1);
            break;
        case 4:
            cell = make_int3(1, 1, 2);
            break;
        case 5:
            cell = make_int3(2, 1, 2);
            break;
        case 6:
            cell = make_int3(1, 2, 2);
            break;
        case 7:
            cell = make_int3(2, 2, 2);
            break;
            };

        const unsigned int local_cell = make_local_cell(cl, cell.x, cell.y, cell.z);
        UP_ASSERT_EQUAL(h_cell_np.data[local_cell], 1);
        UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), local_cell);
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
            h_pos.data[0] = scale(make_scalar4(-4.0, -4.0, -4.0, __int_as_scalar(0)), ref_box, box);
            break;
        case 1:
            h_pos.data[0] = scale(make_scalar4(3.99, -4.0, -4.0, __int_as_scalar(0)), ref_box, box);
            break;
        case 2:
            h_pos.data[0] = scale(make_scalar4(-4.0, 3.99, -4.0, __int_as_scalar(0)), ref_box, box);
            break;
        case 3:
            h_pos.data[0] = scale(make_scalar4(3.99, 3.99, -4.0, __int_as_scalar(0)), ref_box, box);
            break;
        case 4:
            h_pos.data[0] = scale(make_scalar4(-4.0, -4.0, 3.99, __int_as_scalar(0)), ref_box, box);
            break;
        case 5:
            h_pos.data[0] = scale(make_scalar4(3.99, -4.0, 3.99, __int_as_scalar(0)), ref_box, box);
            break;
        case 6:
            h_pos.data[0] = scale(make_scalar4(-4.0, 3.99, 3.99, __int_as_scalar(0)), ref_box, box);
            break;
        case 7:
            h_pos.data[0] = scale(make_scalar4(3.99, 3.99, 3.99, __int_as_scalar(0)), ref_box, box);
            break;
            };
        }

    // reset the grid shift and recompute
    cl->setGridShift(make_scalar3(0, 0, 0));
    cl->compute(3);
    checkDomainBoundaries(sysdef, cl);
        {
        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(),
                                            access_location::host,
                                            access_mode::read);
        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::read);

        int3 cell;
        switch (my_rank)
            {
        case 0:
            cell = make_int3(-2, -2, -2);
            break;
        case 1:
            cell = make_int3(6, -2, -2);
            break;
        case 2:
            cell = make_int3(-2, 6, -2);
            break;
        case 3:
            cell = make_int3(6, 6, -2);
            break;
        case 4:
            cell = make_int3(-2, -2, 6);
            break;
        case 5:
            cell = make_int3(6, -2, 6);
            break;
        case 6:
            cell = make_int3(-2, 6, 6);
            break;
        case 7:
            cell = make_int3(6, 6, 6);
            break;
            };

        const unsigned int local_cell = make_local_cell(cl, cell.x, cell.y, cell.z);
        UP_ASSERT_EQUAL(h_cell_np.data[local_cell], 1);
        UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), local_cell);
        }

    // shift all particles to the right by 0.5
    // all particles on left bound will move up one cell, all particles on right bound stay
    // this test has weird round off issues for triclinic, so only run for orthorhombic boxes
    if (is_orthorhombic)
        {
        cl->setGridShift(-shift);
        cl->compute(4);
        checkDomainBoundaries(sysdef, cl);

        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(),
                                            access_location::host,
                                            access_mode::read);
        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::read);

        int3 cell;
        switch (my_rank)
            {
        case 0:
            cell = make_int3(-2, -2, -2);
            break;
        case 1:
            cell = make_int3(6, -2, -2);
            break;
        case 2:
            cell = make_int3(-2, 6, -2);
            break;
        case 3:
            cell = make_int3(6, 6, -2);
            break;
        case 4:
            cell = make_int3(-2, -2, 6);
            break;
        case 5:
            cell = make_int3(6, -2, 6);
            break;
        case 6:
            cell = make_int3(-2, 6, 6);
            break;
        case 7:
            cell = make_int3(6, 6, 6);
            break;
            };

        const unsigned int local_cell = make_local_cell(cl, cell.x, cell.y, cell.z);
        UP_ASSERT_EQUAL(h_cell_np.data[local_cell], 1);
        UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), local_cell);
        }

        // shift all particles left by 0.5
        // particles on the lower bound stay in the same bin, and particles on the right bound move
        // down one
        {
        cl->setGridShift(shift);
        cl->compute(5);
        checkDomainBoundaries(sysdef, cl);

        ArrayHandle<unsigned int> h_cell_np(cl->getCellSizeArray(),
                                            access_location::host,
                                            access_mode::read);
        ArrayHandle<Scalar4> h_vel(pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::read);

        int3 cell;
        switch (my_rank)
            {
        case 0:
            cell = make_int3(-2, -2, -2);
            break;
        case 1:
            cell = make_int3(5, -2, -2);
            break;
        case 2:
            cell = make_int3(-2, 5, -2);
            break;
        case 3:
            cell = make_int3(5, 5, -2);
            break;
        case 4:
            cell = make_int3(-2, -2, 5);
            break;
        case 5:
            cell = make_int3(5, -2, 5);
            break;
        case 6:
            cell = make_int3(-2, 5, 5);
            break;
        case 7:
            cell = make_int3(5, 5, 5);
            break;
            };

        const unsigned int local_cell = make_local_cell(cl, cell.x, cell.y, cell.z);
        UP_ASSERT_EQUAL(h_cell_np.data[local_cell], 1);
        UP_ASSERT_EQUAL(__scalar_as_int(h_vel.data[0].w), local_cell);
        }
    }

//! dimension test case for MPCD CellList class
UP_TEST(mpcd_cell_list_dimensions)
    {
        // mpi in 1d
        {
        auto exec_conf = std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU,
                                                                  std::vector<int>());
        exec_conf->getMPIConfig()->splitPartitions(2);
        celllist_dimension_test<mpcd::CellList>(exec_conf,
                                                true,
                                                false,
                                                false,
                                                make_scalar3(0, 0, 0));
        celllist_dimension_test<mpcd::CellList>(exec_conf,
                                                false,
                                                true,
                                                false,
                                                make_scalar3(0, 0, 0));
        celllist_dimension_test<mpcd::CellList>(exec_conf,
                                                false,
                                                false,
                                                true,
                                                make_scalar3(0, 0, 0));
        }
        // mpi in 2d
        {
        auto exec_conf = std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU,
                                                                  std::vector<int>());
        exec_conf->getMPIConfig()->splitPartitions(4);
        celllist_dimension_test<mpcd::CellList>(exec_conf,
                                                true,
                                                true,
                                                false,
                                                make_scalar3(0, 0, 0));
        celllist_dimension_test<mpcd::CellList>(exec_conf,
                                                true,
                                                false,
                                                true,
                                                make_scalar3(0, 0, 0));
        celllist_dimension_test<mpcd::CellList>(exec_conf,
                                                false,
                                                true,
                                                true,
                                                make_scalar3(0, 0, 0));
        }
        // mpi in 3d
        {
        auto exec_conf = std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU,
                                                                  std::vector<int>());
        exec_conf->getMPIConfig()->splitPartitions(8);
        celllist_dimension_test<mpcd::CellList>(exec_conf, true, true, true, make_scalar3(0, 0, 0));
        }
    }

//! dimension test case for MPCD CellList class, triclinic
UP_TEST(mpcd_cell_list_dimensions_triclinic)
    {
        // mpi in 1d
        {
        auto exec_conf = std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU,
                                                                  std::vector<int>());
        exec_conf->getMPIConfig()->splitPartitions(2);
        celllist_dimension_test<mpcd::CellList>(exec_conf,
                                                true,
                                                false,
                                                false,
                                                make_scalar3(0.5, -0.75, 1.0));
        celllist_dimension_test<mpcd::CellList>(exec_conf,
                                                false,
                                                true,
                                                false,
                                                make_scalar3(0.5, -0.75, 1.0));
        celllist_dimension_test<mpcd::CellList>(exec_conf,
                                                false,
                                                false,
                                                true,
                                                make_scalar3(0.5, -0.75, 1.0));
        }
        // mpi in 2d
        {
        auto exec_conf = std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU,
                                                                  std::vector<int>());
        exec_conf->getMPIConfig()->splitPartitions(4);
        celllist_dimension_test<mpcd::CellList>(exec_conf,
                                                true,
                                                true,
                                                false,
                                                make_scalar3(0.5, -0.75, 1.0));
        celllist_dimension_test<mpcd::CellList>(exec_conf,
                                                true,
                                                false,
                                                true,
                                                make_scalar3(0.5, -0.75, 1.0));
        celllist_dimension_test<mpcd::CellList>(exec_conf,
                                                false,
                                                true,
                                                true,
                                                make_scalar3(0.5, -0.75, 1.0));
        }
        // mpi in 3d
        {
        auto exec_conf = std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU,
                                                                  std::vector<int>());
        exec_conf->getMPIConfig()->splitPartitions(8);
        celllist_dimension_test<mpcd::CellList>(exec_conf,
                                                true,
                                                true,
                                                true,
                                                make_scalar3(0.5, -0.75, 1.0));
        }
    }

//! basic test case for MPCD CellList class
UP_TEST(mpcd_cell_list_basic_test)
    {
    celllist_basic_test<mpcd::CellList>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU),
        make_scalar3(6.0, 6.0, 6.0),
        make_scalar3(0, 0, 0));
    }

//! basic test case for MPCD CellList class, noncubic
UP_TEST(mpcd_cell_list_basic_test_noncubic)
    {
    celllist_basic_test<mpcd::CellList>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU),
        make_scalar3(6.5, 7.0, 7.5),
        make_scalar3(0, 0, 0));
    }

//! basic test case for MPCD CellList class, triclinic
UP_TEST(mpcd_cell_list_basic_test_triclinic)
    {
    celllist_basic_test<mpcd::CellList>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU),
        make_scalar3(6.0, 6.0, 6.0),
        make_scalar3(0.5, -0.75, 1.0));
    }

//! edge test case for MPCD CellList class
UP_TEST(mpcd_cell_list_edge_test)
    {
    celllist_edge_test<mpcd::CellList>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU),
        make_scalar3(5.0, 5.0, 5.0),
        make_scalar3(0, 0, 0));
    }

//! edge test case for MPCD CellList class, noncubic
UP_TEST(mpcd_cell_list_edge_test_noncubic)
    {
    celllist_edge_test<mpcd::CellList>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU),
        make_scalar3(6.0, 6.5, 7.0),
        make_scalar3(0, 0, 0));
    }

//! edge test case for MPCD CellList class, triclinic
UP_TEST(mpcd_cell_list_edge_test_triclinic)
    {
    celllist_edge_test<mpcd::CellList>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::CPU),
        make_scalar3(5.0, 5.0, 5.0),
        make_scalar3(0.5, -0.75, 1.0));
    }

#ifdef ENABLE_HIP
//! dimension test case for MPCD CellListGPU class
UP_TEST(mpcd_cell_list_gpu_dimensions)
    {
        // mpi in 1d
        {
        auto exec_conf = std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU,
                                                                  std::vector<int>());
        exec_conf->getMPIConfig()->splitPartitions(2);
        celllist_dimension_test<mpcd::CellListGPU>(exec_conf,
                                                   true,
                                                   false,
                                                   false,
                                                   make_scalar3(0, 0, 0));
        celllist_dimension_test<mpcd::CellListGPU>(exec_conf,
                                                   false,
                                                   true,
                                                   false,
                                                   make_scalar3(0, 0, 0));
        celllist_dimension_test<mpcd::CellListGPU>(exec_conf,
                                                   false,
                                                   false,
                                                   true,
                                                   make_scalar3(0, 0, 0));
        }
        // mpi in 2d
        {
        auto exec_conf = std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU,
                                                                  std::vector<int>());
        exec_conf->getMPIConfig()->splitPartitions(4);
        celllist_dimension_test<mpcd::CellListGPU>(exec_conf,
                                                   true,
                                                   true,
                                                   false,
                                                   make_scalar3(0, 0, 0));
        celllist_dimension_test<mpcd::CellListGPU>(exec_conf,
                                                   true,
                                                   false,
                                                   true,
                                                   make_scalar3(0, 0, 0));
        celllist_dimension_test<mpcd::CellListGPU>(exec_conf,
                                                   false,
                                                   true,
                                                   true,
                                                   make_scalar3(0, 0, 0));
        }
        // mpi in 3d
        {
        auto exec_conf = std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU,
                                                                  std::vector<int>());
        exec_conf->getMPIConfig()->splitPartitions(8);
        celllist_dimension_test<mpcd::CellListGPU>(exec_conf,
                                                   true,
                                                   true,
                                                   true,
                                                   make_scalar3(0, 0, 0));
        }
    }

//! dimension test case for MPCD CellListGPU class
UP_TEST(mpcd_cell_list_gpu_dimensions_triclinic)
    {
        // mpi in 1d
        {
        auto exec_conf = std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU,
                                                                  std::vector<int>());
        exec_conf->getMPIConfig()->splitPartitions(2);
        celllist_dimension_test<mpcd::CellListGPU>(exec_conf,
                                                   true,
                                                   false,
                                                   false,
                                                   make_scalar3(0.5, -0.75, 1.0));
        celllist_dimension_test<mpcd::CellListGPU>(exec_conf,
                                                   false,
                                                   true,
                                                   false,
                                                   make_scalar3(0.5, -0.75, 1.0));
        celllist_dimension_test<mpcd::CellListGPU>(exec_conf,
                                                   false,
                                                   false,
                                                   true,
                                                   make_scalar3(0.5, -0.75, 1.0));
        }
        // mpi in 2d
        {
        auto exec_conf = std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU,
                                                                  std::vector<int>());
        exec_conf->getMPIConfig()->splitPartitions(4);
        celllist_dimension_test<mpcd::CellListGPU>(exec_conf,
                                                   true,
                                                   true,
                                                   false,
                                                   make_scalar3(0.5, -0.75, 1.0));
        celllist_dimension_test<mpcd::CellListGPU>(exec_conf,
                                                   true,
                                                   false,
                                                   true,
                                                   make_scalar3(0.5, -0.75, 1.0));
        celllist_dimension_test<mpcd::CellListGPU>(exec_conf,
                                                   false,
                                                   true,
                                                   true,
                                                   make_scalar3(0.5, -0.75, 1.0));
        }
        // mpi in 3d
        {
        auto exec_conf = std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU,
                                                                  std::vector<int>());
        exec_conf->getMPIConfig()->splitPartitions(8);
        celllist_dimension_test<mpcd::CellListGPU>(exec_conf,
                                                   true,
                                                   true,
                                                   true,
                                                   make_scalar3(0.5, -0.75, 1.0));
        }
    }

//! basic test case for MPCD CellListGPU class
UP_TEST(mpcd_cell_list_gpu_basic_test)
    {
    celllist_basic_test<mpcd::CellListGPU>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU),
        make_scalar3(6.0, 6.0, 6.0),
        make_scalar3(0, 0, 0));
    }

//! basic test case for MPCD CellListGPU class, noncubic
UP_TEST(mpcd_cell_list_gpu_basic_test_noncubic)
    {
    celllist_basic_test<mpcd::CellListGPU>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU),
        make_scalar3(6.5, 7.0, 7.5),
        make_scalar3(0, 0, 0));
    }

//! basic test case for MPCD CellListGPU class, triclinic
UP_TEST(mpcd_cell_list_gpu_basic_test_triclinic)
    {
    celllist_basic_test<mpcd::CellListGPU>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU),
        make_scalar3(6.0, 6.0, 6.0),
        make_scalar3(0.5, -0.75, 1.0));
    }

//! edge test case for MPCD CellListGPU class
UP_TEST(mpcd_cell_list_gpu_edge_test)
    {
    celllist_edge_test<mpcd::CellListGPU>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU),
        make_scalar3(5.0, 5.0, 5.0),
        make_scalar3(0, 0, 0));
    }

//! edge test case for MPCD CellListGPU class, noncubic
UP_TEST(mpcd_cell_list_gpu_edge_test_noncubic)
    {
    celllist_edge_test<mpcd::CellListGPU>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU),
        make_scalar3(6.0, 6.5, 7.0),
        make_scalar3(0, 0, 0));
    }

//! edge test case for MPCD CellListGPU class, triclinic
UP_TEST(mpcd_cell_list_gpu_edge_test_triclinic)
    {
    celllist_edge_test<mpcd::CellListGPU>(
        std::make_shared<ExecutionConfiguration>(ExecutionConfiguration::GPU),
        make_scalar3(5.0, 5.0, 5.0),
        make_scalar3(0.5, -0.75, 1.0));
    }
#endif // ENABLE_HIP
