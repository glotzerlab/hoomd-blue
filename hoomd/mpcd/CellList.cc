// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "CellList.h"

#ifdef ENABLE_MPI
#include "Communicator.h"
#include "hoomd/Communicator.h"
#endif // ENABLE_MPI

/*!
 * \file mpcd/CellList.cc
 * \brief Definition of mpcd::CellList
 */

namespace hoomd
    {
mpcd::CellList::CellList(std::shared_ptr<SystemDefinition> sysdef)
    : Compute(sysdef), m_mpcd_pdata(m_sysdef->getMPCDParticleData()), m_cell_size(1.0),
      m_cell_np_max(4), m_cell_np(m_exec_conf), m_cell_list(m_exec_conf),
      m_embed_cell_ids(m_exec_conf), m_conditions(m_exec_conf), m_needs_compute_dim(true),
      m_particles_sorted(false), m_virtual_change(false)
    {
    assert(m_mpcd_pdata);
    m_exec_conf->msg->notice(5) << "Constructing MPCD CellList" << std::endl;

    // by default, grid shifting is initialized to zeroes
    m_cell_dim = make_uint3(0, 0, 0);
    m_global_cell_dim = make_uint3(0, 0, 0);

    m_grid_shift = make_scalar3(0.0, 0.0, 0.0);
    m_max_grid_shift = 0.5 * m_cell_size;
    m_origin_idx = make_int3(0, 0, 0);

    resetConditions();

#ifdef ENABLE_MPI
    m_decomposition = m_pdata->getDomainDecomposition();
    m_num_extra = 0;
    m_cover_box = m_pdata->getBox();
#endif // ENABLE_MPI

    m_mpcd_pdata->getSortSignal().connect<mpcd::CellList, &mpcd::CellList::sort>(this);
    m_mpcd_pdata->getNumVirtualSignal().connect<mpcd::CellList, &mpcd::CellList::slotNumVirtual>(
        this);
    m_pdata->getParticleSortSignal().connect<mpcd::CellList, &mpcd::CellList::slotSorted>(this);
    m_pdata->getBoxChangeSignal().connect<mpcd::CellList, &mpcd::CellList::slotBoxChanged>(this);
    }

mpcd::CellList::~CellList()
    {
    m_exec_conf->msg->notice(5) << "Destroying MPCD CellList" << std::endl;
    m_mpcd_pdata->getSortSignal().disconnect<mpcd::CellList, &mpcd::CellList::sort>(this);
    m_mpcd_pdata->getNumVirtualSignal().disconnect<mpcd::CellList, &mpcd::CellList::slotNumVirtual>(
        this);
    m_pdata->getParticleSortSignal().disconnect<mpcd::CellList, &mpcd::CellList::slotSorted>(this);
    m_pdata->getBoxChangeSignal().disconnect<mpcd::CellList, &mpcd::CellList::slotBoxChanged>(this);
    }

void mpcd::CellList::compute(uint64_t timestep)
    {
    Compute::compute(timestep);

    if (m_virtual_change)
        {
        m_virtual_change = false;
        m_force_compute = true;
        }

    if (m_particles_sorted)
        {
        m_particles_sorted = false;
        m_force_compute = true;
        }

    if (m_needs_compute_dim)
        {
        computeDimensions();
        m_force_compute = true;
        }

    if (peekCompute(timestep))
        {
#ifdef ENABLE_MPI
        // exchange embedded particles if necessary
        if (m_sysdef->isDomainDecomposed() && needsEmbedMigrate(timestep))
            {
            auto comm = m_sysdef->getCommunicator().lock();
            if (!comm)
                {
                throw std::runtime_error("Embedded particle communicator needed but not set");
                }
            comm->forceMigrate();
            comm->communicate(timestep);
            }
#endif // ENABLE_MPI

        // resize to be able to hold the number of embedded particles
        if (m_embed_group)
            {
            m_embed_cell_ids.resize(m_embed_group->getNumMembers());
            }

        bool overflowed = false;
        do
            {
            buildCellList();

            overflowed = checkConditions();

            if (overflowed)
                {
                reallocate();
                resetConditions();
                }
            } while (overflowed);

        // we are finished building, explicitly mark everything (rather than using shouldCompute)
        m_first_compute = false;
        m_force_compute = false;
        m_last_computed = timestep;

        // signal to the ParticleData that the cell list cache is now valid
        m_mpcd_pdata->validateCellCache();
        }
    }

void mpcd::CellList::reallocate()
    {
    m_exec_conf->msg->notice(6) << "Allocating MPCD cell list, " << m_cell_np_max
                                << " particles in " << m_cell_indexer.getNumElements() << " cells."
                                << std::endl;
    m_cell_list_indexer = Index2D(m_cell_np_max, m_cell_indexer.getNumElements());
    m_cell_list.resize(m_cell_list_indexer.getNumElements());
    }

void mpcd::CellList::updateGlobalBox()
    {
    // triclinic boxes are not allowed
    const BoxDim& global_box = m_pdata->getGlobalBox();
    if (global_box.getTiltFactorXY() != Scalar(0.0) || global_box.getTiltFactorXZ() != Scalar(0.0)
        || global_box.getTiltFactorYZ() != Scalar(0.0))
        {
        m_exec_conf->msg->error() << "mpcd: box must be orthorhombic" << std::endl;
        throw std::runtime_error("Box must be orthorhombic");
        }

    // box must be evenly divisible by cell size
    const Scalar3 L = global_box.getL();
    m_global_cell_dim = make_uint3((unsigned int)round(L.x / m_cell_size),
                                   (unsigned int)round(L.y / m_cell_size),
                                   (unsigned int)round(L.z / m_cell_size));
    if (m_sysdef->getNDimensions() == 2)
        {
        if (m_global_cell_dim.z > 1)
            {
            m_exec_conf->msg->error()
                << "mpcd: In 2d simulations, box width must be smaller than cell size" << std::endl;
            throw std::runtime_error("Lz bigger than cell size in 2D!");
            }

        // force to be only one cell along z in 2d
        m_global_cell_dim.z = 1;
        }

    const double eps = 1e-5;
    if (fabs((double)L.x - m_global_cell_dim.x * (double)m_cell_size) > eps * m_cell_size
        || fabs((double)L.y - m_global_cell_dim.y * (double)m_cell_size) > eps * m_cell_size
        || (m_sysdef->getNDimensions() == 3
            && fabs((double)L.z - m_global_cell_dim.z * (double)m_cell_size) > eps * m_cell_size))
        {
        m_exec_conf->msg->error() << "mpcd: Box size must be even multiple of cell size"
                                  << std::endl;
        throw std::runtime_error("MPCD cell size must evenly divide box");
        }
    }

void mpcd::CellList::computeDimensions()
    {
    if (!m_needs_compute_dim)
        return;

    // first update / validate the global box
    updateGlobalBox();

#ifdef ENABLE_MPI
    uchar3 communicating = make_uchar3(0, 0, 0);
    if (m_decomposition)
        {
        const Index3D& di = m_decomposition->getDomainIndexer();
        communicating.x = (di.getW() > 1);
        communicating.y = (di.getH() > 1);
        communicating.z = (di.getD() > 1);
        }

    // Only do complicated sizing if some direction is being communicated
    if (communicating.x || communicating.y || communicating.z)
        {
        // Global simulation box for absolute position referencing
        const BoxDim& global_box = m_pdata->getGlobalBox();
        const Scalar3 global_lo = global_box.getLo();
        const Scalar3 global_L = global_box.getL();

        // if global box is valid (no triclinic skew), then local box also has no skew,
        // so assume orthorhombic in all subsequent calculations
        const BoxDim& box = m_pdata->getBox();

        // setup lo bin
        const Scalar3 delta_lo = box.getLo() - global_lo;
        int3 my_lo_bin = make_int3((int)std::floor((delta_lo.x - m_max_grid_shift) / m_cell_size),
                                   (int)std::floor((delta_lo.y - m_max_grid_shift) / m_cell_size),
                                   (int)std::floor((delta_lo.z - m_max_grid_shift) / m_cell_size));
        int3 lo_neigh_bin
            = make_int3((int)std::ceil((delta_lo.x + m_max_grid_shift) / m_cell_size),
                        (int)std::ceil((delta_lo.y + m_max_grid_shift) / m_cell_size),
                        (int)std::ceil((delta_lo.z + m_max_grid_shift) / m_cell_size));

        // setup hi bin
        const Scalar3 delta_hi = box.getHi() - global_lo;
        int3 my_hi_bin = make_int3((int)std::ceil((delta_hi.x + m_max_grid_shift) / m_cell_size),
                                   (int)std::ceil((delta_hi.y + m_max_grid_shift) / m_cell_size),
                                   (int)std::ceil((delta_hi.z + m_max_grid_shift) / m_cell_size));
        int3 hi_neigh_bin
            = make_int3((int)std::floor((delta_hi.x - m_max_grid_shift) / m_cell_size),
                        (int)std::floor((delta_hi.y - m_max_grid_shift) / m_cell_size),
                        (int)std::floor((delta_hi.z - m_max_grid_shift) / m_cell_size));

        // initially size the grid assuming one rank in each direction, and then resize based on
        // communication
        m_cell_dim = m_global_cell_dim;
        m_origin_idx = make_int3(0, 0, 0);
        std::fill(m_num_comm.begin(), m_num_comm.end(), 0);

        // Compute size of the box with diffusion layer
        Scalar3 cover_lo = box.getLo();
        Scalar3 cover_hi = box.getHi();
        uchar3 cover_periodic = box.getPeriodic();

        if (communicating.x)
            {
            // number of cells and cell origin, padding with extra cells in diffusion layer
            m_cell_dim.x = my_hi_bin.x - my_lo_bin.x + 2 * m_num_extra;
            m_origin_idx.x = my_lo_bin.x - m_num_extra;

            // number of communication cells along each direction
            m_num_comm[static_cast<unsigned int>(mpcd::detail::face::east)]
                = my_hi_bin.x - hi_neigh_bin.x + m_num_extra;
            m_num_comm[static_cast<unsigned int>(mpcd::detail::face::west)]
                = lo_neigh_bin.x - my_lo_bin.x + m_num_extra;

            // "safe" size of the diffusion layer
            cover_lo.x = m_origin_idx.x * m_cell_size + m_max_grid_shift - 0.5 * global_L.x;
            cover_hi.x = (m_origin_idx.x + m_cell_dim.x) * m_cell_size - m_max_grid_shift
                         - 0.5 * global_L.x;
            cover_periodic.x = 0;
            }

        if (communicating.y)
            {
            m_cell_dim.y = my_hi_bin.y - my_lo_bin.y + 2 * m_num_extra;
            m_origin_idx.y = my_lo_bin.y - m_num_extra;

            m_num_comm[static_cast<unsigned int>(mpcd::detail::face::north)]
                = my_hi_bin.y - hi_neigh_bin.y + m_num_extra;
            m_num_comm[static_cast<unsigned int>(mpcd::detail::face::south)]
                = lo_neigh_bin.y - my_lo_bin.y + m_num_extra;

            cover_lo.y = m_origin_idx.y * m_cell_size + m_max_grid_shift - 0.5 * global_L.y;
            cover_hi.y = (m_origin_idx.y + m_cell_dim.y) * m_cell_size - m_max_grid_shift
                         - 0.5 * global_L.y;
            cover_periodic.y = 0;
            }

        if (m_sysdef->getNDimensions() == 3 && communicating.z)
            {
            m_cell_dim.z = my_hi_bin.z - my_lo_bin.z + 2 * m_num_extra;
            m_origin_idx.z = my_lo_bin.z - m_num_extra;

            m_num_comm[static_cast<unsigned int>(mpcd::detail::face::up)]
                = my_hi_bin.z - hi_neigh_bin.z + m_num_extra;
            m_num_comm[static_cast<unsigned int>(mpcd::detail::face::down)]
                = lo_neigh_bin.z - my_lo_bin.z + m_num_extra;

            cover_lo.z = m_origin_idx.z * m_cell_size + m_max_grid_shift - 0.5 * global_L.z;
            cover_hi.z = (m_origin_idx.z + m_cell_dim.z) * m_cell_size - m_max_grid_shift
                         - 0.5 * global_L.z;
            cover_periodic.z = 0;
            }

        // set the box covered by this cell list
        m_cover_box = BoxDim(cover_lo, cover_hi, cover_periodic);

        checkDomainBoundaries();
        }
    else
#endif // ENABLE_MPI
        {
        m_cell_dim = m_global_cell_dim;
        m_origin_idx = make_int3(0, 0, 0);
        }

    // resize the cell indexers and per-cell counter
    m_global_cell_indexer = Index3D(m_global_cell_dim.x, m_global_cell_dim.y, m_global_cell_dim.z);
    m_cell_indexer = Index3D(m_cell_dim.x, m_cell_dim.y, m_cell_dim.z);
    m_cell_np.resize(m_cell_indexer.getNumElements());

    // reallocate per-cell memory
    reallocate();

    // dimensions are now current
    m_needs_compute_dim = false;
    notifySizeChange();
    }

#ifdef ENABLE_MPI
void mpcd::CellList::checkDomainBoundaries()
    {
    if (!m_decomposition)
        return;

    MPI_Comm mpi_comm = m_exec_conf->getMPICommunicator();

    for (unsigned int dir = 0; dir < m_num_comm.size(); ++dir)
        {
        mpcd::detail::face d = static_cast<mpcd::detail::face>(dir);
        if (!isCommunicating(d))
            continue;

        // receive in the opposite direction from which we send
        unsigned int send_neighbor = m_decomposition->getNeighborRank(dir);
        unsigned int recv_neighbor;
        if (dir % 2 == 0)
            recv_neighbor = m_decomposition->getNeighborRank(dir + 1);
        else
            recv_neighbor = m_decomposition->getNeighborRank(dir - 1);

        // first make sure each dimension is sending and receiving the same size data
        MPI_Request reqs[2];
        MPI_Status status[2];

        // check that the number received is the same as that being sent from neighbor
        unsigned int n_send = m_num_comm[dir];
        unsigned int n_expect_recv;
        if (dir % 2 == 0)
            n_expect_recv = m_num_comm[dir + 1];
        else
            n_expect_recv = m_num_comm[dir - 1];

        unsigned int n_recv;
        MPI_Isend(&n_send, 1, MPI_UNSIGNED, send_neighbor, 0, mpi_comm, &reqs[0]);
        MPI_Irecv(&n_recv, 1, MPI_UNSIGNED, recv_neighbor, 0, mpi_comm, &reqs[1]);
        MPI_Waitall(2, reqs, status);

        // check if any rank errored out
        unsigned int recv_error = 0;
        if (n_expect_recv != n_recv)
            {
            recv_error = 1;
            }
        unsigned int any_error = 0;
        MPI_Allreduce(&recv_error, &any_error, 1, MPI_UNSIGNED, MPI_SUM, mpi_comm);
        if (any_error)
            {
            if (recv_error)
                {
                m_exec_conf->msg->error() << "mpcd: expected to communicate " << n_expect_recv
                                          << " cells, but only receiving " << n_recv << std::endl;
                }
            throw std::runtime_error("Error setting up MPCD cell list");
            }

        // check that the same cell ids are communicated
        std::vector<int> send_cells(n_send), recv_cells(n_recv);
        for (unsigned int i = 0; i < n_send; ++i)
            {
            if (d == mpcd::detail::face::east)
                {
                send_cells[i] = m_origin_idx.x + m_cell_dim.x - m_num_extra - n_send + i;
                }
            else if (d == mpcd::detail::face::west)
                {
                send_cells[i] = m_origin_idx.x + i;
                }
            else if (d == mpcd::detail::face::north)
                {
                send_cells[i] = m_origin_idx.y + m_cell_dim.y - m_num_extra - n_send + i;
                }
            else if (d == mpcd::detail::face::south)
                {
                send_cells[i] = m_origin_idx.y + i;
                }
            else if (d == mpcd::detail::face::up)
                {
                send_cells[i] = m_origin_idx.z + m_cell_dim.z - m_num_extra - n_send + i;
                }
            else if (d == mpcd::detail::face::down)
                {
                send_cells[i] = m_origin_idx.z + i;
                }
            }

        MPI_Isend(&send_cells[0], n_send, MPI_INT, send_neighbor, 1, mpi_comm, &reqs[0]);
        MPI_Irecv(&recv_cells[0], n_recv, MPI_INT, recv_neighbor, 1, mpi_comm, &reqs[1]);
        MPI_Waitall(2, reqs, status);

        unsigned int overlap_error = 0;
        std::array<int, 2> err_pair {0, 0};
        for (unsigned int i = 0; i < n_recv && !overlap_error; ++i)
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
            recv_cell = wrapGlobalCell(recv_cell);

            // compute the expected cell to receive, also wrapped
            int3 expect_recv_cell = make_int3(0, 0, 0);
            if (d == mpcd::detail::face::east)
                {
                expect_recv_cell.x = m_origin_idx.x + i;
                }
            else if (d == mpcd::detail::face::west)
                {
                expect_recv_cell.x = m_origin_idx.x + m_cell_dim.x - m_num_extra - n_recv + i;
                }
            else if (d == mpcd::detail::face::north)
                {
                expect_recv_cell.y = m_origin_idx.y + i;
                }
            else if (d == mpcd::detail::face::south)
                {
                expect_recv_cell.y = m_origin_idx.y + m_cell_dim.y - m_num_extra - n_recv + i;
                }
            else if (d == mpcd::detail::face::up)
                {
                expect_recv_cell.z = m_origin_idx.z + i;
                }
            else if (d == mpcd::detail::face::down)
                {
                expect_recv_cell.z = m_origin_idx.z + m_cell_dim.z - m_num_extra - n_recv + i;
                }
            expect_recv_cell = wrapGlobalCell(expect_recv_cell);

            if (recv_cell.x != expect_recv_cell.x || recv_cell.y != expect_recv_cell.y
                || recv_cell.z != expect_recv_cell.z)
                {
                overlap_error = i;
                if (d == mpcd::detail::face::east || d == mpcd::detail::face::west)
                    {
                    err_pair[0] = recv_cell.x;
                    err_pair[1] = expect_recv_cell.x;
                    }
                else if (d == mpcd::detail::face::north || d == mpcd::detail::face::south)
                    {
                    err_pair[0] = recv_cell.y;
                    err_pair[1] = expect_recv_cell.y;
                    }
                else if (d == mpcd::detail::face::up || d == mpcd::detail::face::down)
                    {
                    err_pair[0] = recv_cell.z;
                    err_pair[1] = expect_recv_cell.z;
                    }
                }
            }

        // check if anyone reported an error, then race to see who gets to write it out
        any_error = 0;
        MPI_Allreduce(&overlap_error, &any_error, 1, MPI_UNSIGNED, MPI_SUM, mpi_comm);
        if (any_error)
            {
            if (overlap_error)
                {
                m_exec_conf->msg->error()
                    << "mpcd: communication grid does not overlap. " << "Expected to receive cell "
                    << err_pair[1] << " from rank " << recv_neighbor << ", but got cell "
                    << err_pair[0] << "." << std::endl;
                }
            throw std::runtime_error("Error setting up MPCD cell list");
            }
        }
    }

/*!
 * \param dir Direction of communication
 * \returns True if communication is occurring along \a dir
 *
 * The size of the domain indexer is checked along the direction of communication
 * to see if there are multiple ranks that must communicate.
 */
bool mpcd::CellList::isCommunicating(mpcd::detail::face dir)
    {
    if (!m_decomposition)
        return false;

    const Index3D& di = m_decomposition->getDomainIndexer();
    bool result = true;
    if ((dir == mpcd::detail::face::east || dir == mpcd::detail::face::west) && di.getW() == 1)
        result = false;
    else if ((dir == mpcd::detail::face::north || dir == mpcd::detail::face::south)
             && di.getH() == 1)
        result = false;
    else if ((dir == mpcd::detail::face::up || dir == mpcd::detail::face::down) && di.getD() == 1)
        result = false;

    return result;
    }
#endif // ENABLE_MPI

/*!
 * \param timestep Current simulation timestep
 */
void mpcd::CellList::buildCellList()
    {
    const BoxDim& box = m_pdata->getBox();
    const uchar3 periodic = box.getPeriodic();

    ArrayHandle<unsigned int> h_cell_list(m_cell_list,
                                          access_location::host,
                                          access_mode::overwrite);
    ArrayHandle<unsigned int> h_cell_np(m_cell_np, access_location::host, access_mode::overwrite);
    // zero the cell counter
    memset(h_cell_np.data, 0, sizeof(unsigned int) * m_cell_indexer.getNumElements());

    uint3 conditions = make_uint3(0, 0, 0);

    ArrayHandle<Scalar4> h_pos(m_mpcd_pdata->getPositions(),
                               access_location::host,
                               access_mode::read);
    ArrayHandle<Scalar4> h_vel(m_mpcd_pdata->getVelocities(),
                               access_location::host,
                               access_mode::readwrite);
    unsigned int N_mpcd = m_mpcd_pdata->getN() + m_mpcd_pdata->getNVirtual();
    unsigned int N_tot = N_mpcd;

    // we can't modify the velocity of embedded particles, so we only read their position
    std::unique_ptr<ArrayHandle<unsigned int>> h_embed_cell_ids;
    std::unique_ptr<ArrayHandle<Scalar4>> h_pos_embed;
    std::unique_ptr<ArrayHandle<unsigned int>> h_embed_member_idx;
    if (m_embed_group)
        {
        h_embed_cell_ids.reset(new ArrayHandle<unsigned int>(m_embed_cell_ids,
                                                             access_location::host,
                                                             access_mode::overwrite));
        h_pos_embed.reset(new ArrayHandle<Scalar4>(m_pdata->getPositions(),
                                                   access_location::host,
                                                   access_mode::read));
        h_embed_member_idx.reset(new ArrayHandle<unsigned int>(m_embed_group->getIndexArray(),
                                                               access_location::host,
                                                               access_mode::read));
        N_tot += m_embed_group->getNumMembers();
        }

    // total effective number of cells in the global box, optionally padded by
    // extra cells in MPI simulations
    uint3 n_global_cells = m_global_cell_dim;
#ifdef ENABLE_MPI
    if (isCommunicating(mpcd::detail::face::east))
        n_global_cells.x += 2 * m_num_extra;
    if (isCommunicating(mpcd::detail::face::north))
        n_global_cells.y += 2 * m_num_extra;
    if (isCommunicating(mpcd::detail::face::up))
        n_global_cells.z += 2 * m_num_extra;
#endif // ENABLE_MPI

    const Scalar3 global_lo = m_pdata->getGlobalBox().getLo();

    for (unsigned int cur_p = 0; cur_p < N_tot; ++cur_p)
        {
        Scalar4 postype_i;
        if (cur_p < N_mpcd)
            {
            postype_i = h_pos.data[cur_p];
            }
        else
            {
            postype_i = h_pos_embed->data[h_embed_member_idx->data[cur_p - N_mpcd]];
            }
        Scalar3 pos_i = make_scalar3(postype_i.x, postype_i.y, postype_i.z);

        if (std::isnan(pos_i.x) || std::isnan(pos_i.y) || std::isnan(pos_i.z))
            {
            conditions.y = cur_p + 1;
            continue;
            }

        // bin particle assuming orthorhombic box (already validated)
        const Scalar3 delta = (pos_i - m_grid_shift) - global_lo;
        int3 global_bin = make_int3((int)std::floor(delta.x / m_cell_size),
                                    (int)std::floor(delta.y / m_cell_size),
                                    (int)std::floor(delta.z / m_cell_size));

        // wrap cell back through the boundaries (grid shifting may send +/- 1 outside of range)
        // this is done using periodic from the "local" box, since this will be periodic
        // only when there is one rank along the dimension
        if (periodic.x)
            {
            if (global_bin.x == (int)n_global_cells.x)
                global_bin.x = 0;
            else if (global_bin.x == -1)
                global_bin.x = n_global_cells.x - 1;
            }
        if (periodic.y)
            {
            if (global_bin.y == (int)n_global_cells.y)
                global_bin.y = 0;
            else if (global_bin.y == -1)
                global_bin.y = n_global_cells.y - 1;
            }
        if (periodic.z)
            {
            if (global_bin.z == (int)n_global_cells.z)
                global_bin.z = 0;
            else if (global_bin.z == -1)
                global_bin.z = n_global_cells.z - 1;
            }

        // compute the local cell
        int3 bin = make_int3(global_bin.x - m_origin_idx.x,
                             global_bin.y - m_origin_idx.y,
                             global_bin.z - m_origin_idx.z);

        // validate and make sure no particles blew out of the box
        if ((bin.x < 0 || bin.x >= (int)m_cell_dim.x) || (bin.y < 0 || bin.y >= (int)m_cell_dim.y)
            || (bin.z < 0 || bin.z >= (int)m_cell_dim.z))
            {
            conditions.z = cur_p + 1;
            continue;
            }

        unsigned int bin_idx = m_cell_indexer(bin.x, bin.y, bin.z);
        unsigned int offset = h_cell_np.data[bin_idx];
        if (offset < m_cell_np_max)
            {
            h_cell_list.data[m_cell_list_indexer(offset, bin_idx)] = cur_p;
            }
        else
            {
            // overflow
            conditions.x = std::max(conditions.x, offset + 1);
            }

        // stash the current particle bin into the velocity array
        if (cur_p < N_mpcd)
            {
            h_vel.data[cur_p].w = __int_as_scalar(bin_idx);
            }
        else
            {
            h_embed_cell_ids->data[cur_p - N_mpcd] = bin_idx;
            }

        // increment the counter always
        ++h_cell_np.data[bin_idx];
        }

    // write out the conditions
    m_conditions.resetFlags(conditions);
    }

/*!
 * \param timestep Timestep that the sorting occurred
 * \param order Mapping of sorted particle indexes onto old particle indexes
 * \param rorder Mapping of old particle indexes onto sorted particle indexes
 */
void mpcd::CellList::sort(uint64_t timestep,
                          const GPUArray<unsigned int>& order,
                          const GPUArray<unsigned int>& rorder)
    {
    // no need to do any sorting if we can still be called at the current timestep
    if (peekCompute(timestep))
        return;

    // if mapping is not valid, signal that we need to force a recompute next time
    // that the cell list is needed. We don't call forceCompute() directly because this always
    // runs compute(), and we just want to defer to the next compute() call.
    if (rorder.isNull())
        {
        m_force_compute = true;
        return;
        }

    // iterate through particles in cell list, and update their indexes using reverse mapping
    ArrayHandle<unsigned int> h_rorder(rorder, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_cell_np(m_cell_np, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_cell_list(m_cell_list,
                                          access_location::host,
                                          access_mode::readwrite);
    const unsigned int N_mpcd = m_mpcd_pdata->getN();

    for (unsigned int idx = 0; idx < getNCells(); ++idx)
        {
        const unsigned int np = h_cell_np.data[idx];
        for (unsigned int offset = 0; offset < np; ++offset)
            {
            const unsigned int cl_idx = m_cell_list_indexer(offset, idx);
            const unsigned int pid = h_cell_list.data[cl_idx];
            // only update indexes of MPCD particles, not virtual or embedded particles
            if (pid < N_mpcd)
                {
                h_cell_list.data[cl_idx] = h_rorder.data[pid];
                }
            }
        }
    }

#ifdef ENABLE_MPI
bool mpcd::CellList::needsEmbedMigrate(uint64_t timestep)
    {
    // no migrate needed if no embedded particles
    if (!m_embed_group)
        return false;

    // ensure that the cell list has been sized first
    computeDimensions();

    // coverage box dimensions, assuming orthorhombic
    const Scalar3 lo = m_cover_box.getLo();
    const Scalar3 hi = m_cover_box.getHi();
    const uchar3 periodic = m_cover_box.getPeriodic();
    const unsigned int ndim = m_sysdef->getNDimensions();

    // particle data
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_group(m_embed_group->getIndexArray(),
                                      access_location::host,
                                      access_mode::read);
    const unsigned int N = m_embed_group->getNumMembers();

    // check if any particle lies outside of the box on this rank
    char migrate = 0;
    for (unsigned int i = 0; i < N && !migrate; ++i)
        {
        const unsigned int idx = h_group.data[i];
        const Scalar4 postype = h_pos.data[idx];
        const Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);

        if ((!periodic.x && (pos.x >= hi.x || pos.x < lo.x))
            || (!periodic.y && (pos.y >= hi.y || pos.y < lo.y))
            || (!periodic.z && ndim == 3 && (pos.z >= hi.z || pos.z < lo.z)))
            {
            migrate = 1;
            }
        }

    // reduce across all ranks
    MPI_Allreduce(MPI_IN_PLACE, &migrate, 1, MPI_CHAR, MPI_MAX, m_exec_conf->getMPICommunicator());

    return static_cast<bool>(migrate);
    }
#endif // ENABLE_MPI

bool mpcd::CellList::checkConditions()
    {
    bool result = false;

    uint3 conditions = m_conditions.readFlags();

    if (conditions.x > m_cell_np_max)
        {
        m_cell_np_max = conditions.x;
        result = true;
        }
    if (conditions.y)
        {
        unsigned int n = conditions.y - 1;
        if (n < m_mpcd_pdata->getN())
            m_exec_conf->msg->errorAllRanks()
                << "MPCD particle " << n << " has position NaN" << std::endl;
        else if (n < m_mpcd_pdata->getNVirtual())
            m_exec_conf->msg->errorAllRanks()
                << "MPCD virtual particle " << n << " has position NaN" << std::endl;
        else
            {
            ArrayHandle<unsigned int> h_embed_member_idx(m_embed_group->getIndexArray(),
                                                         access_location::host,
                                                         access_mode::read);
            m_exec_conf->msg->errorAllRanks()
                << "Embedded particle "
                << h_embed_member_idx.data[n - (m_mpcd_pdata->getN() + m_mpcd_pdata->getNVirtual())]
                << " has position NaN" << std::endl;
            }
        throw std::runtime_error("Error computing cell list");
        }
    if (conditions.z)
        {
        unsigned int n = conditions.z - 1;
        Scalar4 pos_empty_i;
        if (n < m_mpcd_pdata->getN() + m_mpcd_pdata->getNVirtual())
            {
            ArrayHandle<Scalar4> h_pos(m_mpcd_pdata->getPositions(),
                                       access_location::host,
                                       access_mode::read);
            pos_empty_i = h_pos.data[n];
            if (n < m_mpcd_pdata->getN())
                m_exec_conf->msg->errorAllRanks()
                    << "MPCD particle is no longer in the simulation box" << std::endl;
            else
                m_exec_conf->msg->errorAllRanks()
                    << "MPCD virtual particle is no longer in the simulation box" << std::endl;
            }
        else
            {
            ArrayHandle<Scalar4> h_pos_embed(m_pdata->getPositions(),
                                             access_location::host,
                                             access_mode::read);
            ArrayHandle<unsigned int> h_embed_member_idx(m_embed_group->getIndexArray(),
                                                         access_location::host,
                                                         access_mode::read);
            pos_empty_i
                = h_pos_embed
                      .data[h_embed_member_idx
                                .data[n - (m_mpcd_pdata->getN() + m_mpcd_pdata->getNVirtual())]];
            m_exec_conf->msg->errorAllRanks()
                << "Embedded particle is no longer in the simulation box" << std::endl;
            }

        Scalar3 pos = make_scalar3(pos_empty_i.x, pos_empty_i.y, pos_empty_i.z);
        m_exec_conf->msg->errorAllRanks()
            << "Cartesian coordinates: " << std::endl
            << "x: " << pos.x << " y: " << pos.y << " z: " << pos.z << std::endl
            << "Grid shift: " << std::endl
            << "x: " << m_grid_shift.x << " y: " << m_grid_shift.y << " z: " << m_grid_shift.z
            << std::endl;

        const BoxDim cover_box = getCoverageBox();
        Scalar3 lo = cover_box.getLo();
        Scalar3 hi = cover_box.getHi();
        uchar3 periodic = cover_box.getPeriodic();
        m_exec_conf->msg->errorAllRanks()
            << "Covered box lo: (" << lo.x << ", " << lo.y << ", " << lo.z << ")" << std::endl
            << "            hi: (" << hi.x << ", " << hi.y << ", " << hi.z << ")" << std::endl
            << "      periodic: (" << ((periodic.x) ? "1" : "0") << " "
            << ((periodic.y) ? "1" : "0") << " " << ((periodic.z) ? "1" : "0") << ")" << std::endl;
        throw std::runtime_error("Error computing cell list");
        }

    return result;
    }

void mpcd::CellList::resetConditions()
    {
    m_conditions.resetFlags(make_uint3(0, 0, 0));
    }

void mpcd::CellList::getCellStatistics() const
    {
    unsigned int min_np(0xffffffff), max_np(0);
    ArrayHandle<unsigned int> h_cell_np(m_cell_np, access_location::host, access_mode::read);
    for (unsigned int cur_cell = 0; cur_cell < m_cell_indexer.getNumElements(); ++cur_cell)
        {
        const unsigned int np = h_cell_np.data[cur_cell];
        if (np < min_np)
            min_np = np;
        if (np > max_np)
            max_np = np;
        }
    m_exec_conf->msg->notice(2) << "MPCD cell list stats:" << std::endl;
    m_exec_conf->msg->notice(2) << "Min: " << min_np << " Max: " << max_np << std::endl;
    }

/*!
 * \param global Global cell index to shift into the local box
 * \returns Local cell index
 *
 * \warning The returned cell index may lie outside the local grid. It is the
 *          caller's responsibility to check that the index is valid.
 */
const int3 mpcd::CellList::getLocalCell(const int3& global) const
    {
    int3 local = make_int3(global.x - m_origin_idx.x,
                           global.y - m_origin_idx.y,
                           global.z - m_origin_idx.z);

    return local;
    }

/*!
 * \param local Local cell index to shift into the global box
 * \returns Global cell index
 *
 * Local cell coordinates are wrapped around the global box so that a valid global
 * index is computed.
 */
const int3 mpcd::CellList::getGlobalCell(const int3& local) const
    {
    int3 global
        = make_int3(local.x + m_origin_idx.x, local.y + m_origin_idx.y, local.z + m_origin_idx.z);
    return wrapGlobalCell(global);
    }

/*!
 * \param cell Cell coordinates to wrap back into the global box
 *
 * \warning Only up to one global box size is wrapped. This method is intended
 *          to be used for wrapping cells off by only one or two from the global boundary.
 */
const int3 mpcd::CellList::wrapGlobalCell(const int3& cell) const
    {
    int3 wrap = cell;

    if (wrap.x >= (int)m_global_cell_dim.x)
        wrap.x -= m_global_cell_dim.x;
    else if (wrap.x < 0)
        wrap.x += m_global_cell_dim.x;

    if (wrap.y >= (int)m_global_cell_dim.y)
        wrap.y -= m_global_cell_dim.y;
    else if (wrap.y < 0)
        wrap.y += m_global_cell_dim.y;

    if (wrap.z >= (int)m_global_cell_dim.z)
        wrap.z -= m_global_cell_dim.z;
    else if (wrap.z < 0)
        wrap.z += m_global_cell_dim.z;

    return wrap;
    }

void mpcd::detail::export_CellList(pybind11::module& m)
    {
    pybind11::class_<mpcd::CellList, Compute, std::shared_ptr<mpcd::CellList>>(m, "CellList")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
        .def_property("cell_size", &mpcd::CellList::getCellSize, &mpcd::CellList::setCellSize)
        .def("setEmbeddedGroup", &mpcd::CellList::setEmbeddedGroup)
        .def("removeEmbeddedGroup", &mpcd::CellList::removeEmbeddedGroup);
    }

    } // end namespace hoomd
