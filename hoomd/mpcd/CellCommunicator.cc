// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/CellCommunicator.cc
 * \brief Definition of mpcd::CellCommunicator
 */

#ifdef ENABLE_MPI

#include "CellCommunicator.h"

// initialize with zero instances of the communicator
unsigned int mpcd::CellCommunicator::num_instances = 0;

/*!
 * \param sysdef System definition
 * \param cl MPCD cell list
 */
mpcd::CellCommunicator::CellCommunicator(std::shared_ptr<SystemDefinition> sysdef,
                                         std::shared_ptr<mpcd::CellList> cl)
    : m_id(num_instances++),
      m_sysdef(sysdef),
      m_pdata(sysdef->getParticleData()),
      m_exec_conf(m_pdata->getExecConf()),
      m_mpi_comm(m_exec_conf->getMPICommunicator()),
      m_decomposition(m_pdata->getDomainDecomposition()),
      m_cl(cl),
      m_communicating(false),
      m_send_buf(m_exec_conf),
      m_recv_buf(m_exec_conf),
      m_needs_init(true)
    {
    m_exec_conf->msg->notice(5) << "Constructing MPCD CellCommunicator" << std::endl;
    #ifdef ENABLE_CUDA
    if (m_exec_conf->isCUDAEnabled())
        {
        m_tuner_pack.reset(new Autotuner(32, 1024, 32, 5, 100000, "mpcd_cell_comm_pack_" + std::to_string(m_id), m_exec_conf));
        m_tuner_unpack.reset(new Autotuner(32, 1024, 32, 5, 100000, "mpcd_cell_comm_unpack_" + std::to_string(m_id), m_exec_conf));
        }
    #endif // ENABLE_CUDA

    m_cl->getSizeChangeSignal().connect<mpcd::CellCommunicator, &mpcd::CellCommunicator::slotInit>(this);
    }

mpcd::CellCommunicator::~CellCommunicator()
    {
    m_exec_conf->msg->notice(5) << "Destroying MPCD CellCommunicator" << std::endl;
    m_cl->getSizeChangeSignal().disconnect<mpcd::CellCommunicator, &mpcd::CellCommunicator::slotInit>(this);
    }

namespace mpcd
{
namespace detail
{
//! Unary operator to wrap global cell indexes into the local domain
struct LocalCellWrapOp
    {
    LocalCellWrapOp(std::shared_ptr<mpcd::CellList> cl_)
        : cl(cl_), ci(cl_->getCellIndexer()), gci(cl_->getGlobalCellIndexer())
        { }

    //! Transform the global 1D cell index into a local 1D cell index
    inline unsigned int operator()(unsigned int cell_idx)
        {
        // convert the 1D global cell index to a global cell tuple
        const uint3 cell = gci.getTriple(cell_idx);

        // convert the global cell tuple to a local cell tuple
        int3 local_cell = cl->getLocalCell(make_int3(cell.x, cell.y, cell.z));

        // wrap the local cell through the global boundaries, which should work for all reasonable cell comms.
        if (local_cell.x >= (int)gci.getW()) local_cell.x -= gci.getW();
        else if (local_cell.x < 0) local_cell.x += gci.getW();

        if (local_cell.y >= (int)gci.getH()) local_cell.y -= gci.getH();
        else if (local_cell.y < 0) local_cell.y += gci.getH();

        if (local_cell.z >= (int)gci.getD()) local_cell.z -= gci.getD();
        else if (local_cell.z < 0) local_cell.z += gci.getD();

        // convert the local cell tuple back to an index
        return ci(local_cell.x, local_cell.y, local_cell.z);
        }

    std::shared_ptr<mpcd::CellList> cl; //!< Cell list
    const Index3D ci;                   //!< Cell indexer
    const Index3D gci;                  //!< Global cell indexer
    };
} // end namespace detail
} // end namespace mpcd

void mpcd::CellCommunicator::initialize()
    {
    // obtain domain decomposition
    const Index3D& di = m_decomposition->getDomainIndexer();
    ArrayHandle<unsigned int> h_cart_ranks(m_decomposition->getCartRanks(), access_location::host, access_mode::read);
    const uint3 my_pos = m_decomposition->getGridPos();

    // use the cell list to compute the bounds
    const Index3D& ci = m_cl->getCellIndexer();
    const Index3D& global_ci = m_cl->getGlobalCellIndexer();
    auto num_comm_cells = m_cl->getNComm();
    const uint3 max_lo = make_uint3(num_comm_cells[static_cast<unsigned int>(mpcd::detail::face::west)],
                                    num_comm_cells[static_cast<unsigned int>(mpcd::detail::face::south)],
                                    num_comm_cells[static_cast<unsigned int>(mpcd::detail::face::down)]);
    const uint3 min_hi = make_uint3(ci.getW() - num_comm_cells[static_cast<unsigned int>(mpcd::detail::face::east)],
                                    ci.getH() - num_comm_cells[static_cast<unsigned int>(mpcd::detail::face::north)],
                                    ci.getD() - num_comm_cells[static_cast<unsigned int>(mpcd::detail::face::up)]);

    // check to make sure box is not overdecomposed
        {
        const unsigned int nextra = m_cl->getNExtraCells();
        unsigned int err = ((max_lo.x + nextra) > min_hi.x ||
                            (max_lo.y + nextra) > min_hi.y ||
                            (max_lo.z + nextra) > min_hi.z);
        MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_UNSIGNED, MPI_MAX, m_mpi_comm);
        if (err)
            {
            m_exec_conf->msg->error() << "mpcd: Simulation box is overdecomposed, decrease the number of ranks." << std::endl;
            throw std::runtime_error("Simulation box is overdecomposed for MPCD");
            }
        }

    // loop over all cells in the grid and determine where to send them
    std::multimap<unsigned int, unsigned int> send_map;
    std::set<unsigned int> neighbors;
    for (unsigned int k=0; k < ci.getD(); ++k)
        {
        for (unsigned int j=0; j < ci.getH(); ++j)
            {
            for (unsigned int i=0; i < ci.getW(); ++i)
                {
                // skip any cells interior to the grid, which will not be communicated
                // this is wasteful loop logic, but initialize will only be called rarely
                if (i >= max_lo.x && i < min_hi.x &&
                    j >= max_lo.y && j < min_hi.y &&
                    k >= max_lo.z && k < min_hi.z)
                    continue;

                // obtain the 1D global index of this cell
                const int3 global_cell = m_cl->getGlobalCell(make_int3(i,j,k));
                const unsigned int global_cell_idx = global_ci(global_cell.x, global_cell.y, global_cell.z);

                // check which direction the cell lies off rank in x,y,z
                std::vector<int> dx = {0};
                if (i < max_lo.x)
                    dx.push_back(-1);
                else if (i >= min_hi.x)
                    dx.push_back(1);

                std::vector<int> dy = {0};
                if (j < max_lo.y)
                    dy.push_back(-1);
                else if (j >= min_hi.y)
                    dy.push_back(1);

                std::vector<int> dz = {0};
                if (k < max_lo.z)
                    dz.push_back(-1);
                else if (k >= min_hi.z)
                    dz.push_back(1);

                // generate all permutations of these neighbors for the cell
                for (auto ddx = dx.begin(); ddx != dx.end(); ++ddx)
                    {
                    for (auto ddy = dy.begin(); ddy != dy.end(); ++ddy)
                        {
                        for (auto ddz = dz.begin(); ddz != dz.end(); ++ddz)
                            {
                            // skip self
                            if (*ddx == 0 && *ddy == 0 && *ddz == 0) continue;

                            // get neighbor rank tuple
                            int3 neigh = make_int3((int)my_pos.x + *ddx,
                                                   (int)my_pos.y + *ddy,
                                                   (int)my_pos.z + *ddz);

                            // wrap neighbor through the boundaries
                            if (neigh.x < 0)
                                neigh.x += di.getW();
                            else if (neigh.x >= (int)di.getW())
                                neigh.x -= di.getW();

                            if (neigh.y < 0)
                                neigh.y += di.getH();
                            else if (neigh.y >= (int)di.getH())
                                neigh.y -= di.getH();

                            if (neigh.z < 0)
                                neigh.z += di.getD();
                            else if (neigh.z >= (int)di.getD())
                                neigh.z -= di.getD();

                            // convert neighbor to a linear rank and push it into the unique neighbor set
                            const unsigned int neigh_rank = h_cart_ranks.data[di(neigh.x,neigh.y,neigh.z)];
                            neighbors.insert(neigh_rank);
                            send_map.insert(std::make_pair(neigh_rank, global_cell_idx));
                            } // ddz
                        } // ddy
                    } // ddx
                } // i
            } // j
        } // k

    // allocate send / receive index arrays
        {
        GPUArray<unsigned int> send_idx(send_map.size(), m_exec_conf);
        m_send_idx.swap(send_idx);
        }

    // fill the send indexes with the global values
    // flood the array of unique neighbors and count the number to send
        {
        ArrayHandle<unsigned int> h_send_idx(m_send_idx, access_location::host, access_mode::overwrite);
        unsigned int idx = 0;
        for (auto it = send_map.begin(); it != send_map.end(); ++it)
            {
            h_send_idx.data[idx++] = it->second;
            }

        m_neighbors.resize(neighbors.size());
        m_begin.resize(m_neighbors.size());
        m_num_send.resize(m_neighbors.size());
        idx = 0;
        for (auto it = neighbors.begin(); it != neighbors.end(); ++it)
            {
            auto lower = send_map.lower_bound(*it);
            auto upper = send_map.upper_bound(*it);

            m_neighbors[idx] = *it;
            m_begin[idx] = std::distance(send_map.begin(), lower);
            m_num_send[idx] = std::distance(lower, upper);
            ++idx;
            }
        }

    // send / receive the global cell indexes to be communicated with neighbors
    std::vector<unsigned int> recv_idx(m_send_idx.getNumElements());
        {
        ArrayHandle<unsigned int> h_send_idx(m_send_idx, access_location::host, access_mode::read);

        m_reqs.resize(2*m_neighbors.size());
        for (unsigned int idx=0; idx < m_neighbors.size(); ++idx)
            {
            const unsigned int offset = m_begin[idx];
            MPI_Isend(h_send_idx.data + offset, m_num_send[idx], MPI_INT, m_neighbors[idx], 0, m_mpi_comm, &m_reqs[2*idx]);
            MPI_Irecv(recv_idx.data() + offset, m_num_send[idx], MPI_INT, m_neighbors[idx], 0, m_mpi_comm, &m_reqs[2*idx+1]);
            }
        MPI_Waitall(m_reqs.size(), m_reqs.data(), MPI_STATUSES_IGNORE);
        }

    // transform all of the global cell indexes back into local cell indexes
        {
        ArrayHandle<unsigned int> h_send_idx(m_send_idx, access_location::host, access_mode::readwrite);

        mpcd::detail::LocalCellWrapOp wrapper(m_cl);
        std::transform(h_send_idx.data, h_send_idx.data + m_send_idx.getNumElements(), h_send_idx.data, wrapper);
        std::transform(recv_idx.begin(), recv_idx.end(), recv_idx.begin(), wrapper);
        }

    // map the received cells from a rank-basis to a cell-basis
        {
        std::multimap<unsigned int, unsigned int> cell_map;
        std::set<unsigned int> unique_cells;
        for (unsigned int idx=0; idx < recv_idx.size(); ++idx)
            {
            const unsigned int cell = recv_idx[idx];
            unique_cells.insert(cell);
            cell_map.insert(std::make_pair(cell, idx));
            }
        m_num_cells = unique_cells.size();

        /*
         * Allocate auxiliary memory for receiving cell reordering
         */
            {
            GPUArray<unsigned int> recv(recv_idx.size(), m_exec_conf);
            m_recv.swap(recv);

            GPUArray<unsigned int> cells(m_num_cells, m_exec_conf);
            m_cells.swap(cells);

            GPUArray<unsigned int> recv_begin(m_num_cells, m_exec_conf);
            m_recv_begin.swap(recv_begin);

            GPUArray<unsigned int> recv_end(m_num_cells, m_exec_conf);
            m_recv_end.swap(recv_end);
            }

        /*
         * Generate the compacted list of unique cells
         */
        ArrayHandle<unsigned int> h_cells(m_cells, access_location::host, access_mode::overwrite);
        unsigned int idx = 0;
        for (auto it = unique_cells.begin(); it != unique_cells.end(); ++it)
            {
            h_cells.data[idx++] = *it;
            }

        /*
         * Loop over the cell map to do run-length encoding on the keys. This
         * determines the range of data belonging to each received cell.
         */
        ArrayHandle<unsigned int> h_recv(m_recv, access_location::host, access_mode::overwrite);
        ArrayHandle<unsigned int> h_recv_begin(m_recv_begin, access_location::host, access_mode::overwrite);
        ArrayHandle<unsigned int> h_recv_end(m_recv_end, access_location::host, access_mode::overwrite);
        unsigned int last_cell = UINT_MAX;
        unsigned int cell_idx = 0;
        idx = 0;
        h_recv_begin.data[cell_idx] = idx;
        for (auto it = cell_map.begin(); it != cell_map.end(); ++it)
            {
            // record the sorted receive index
            h_recv.data[idx] = it->second;

            // if not very first pass and the current cell does not match the
            // last cell, then we are on a new cell, and need to demark an end / begin
            if (last_cell != UINT_MAX && it->first != last_cell)
                {
                h_recv_end.data[cell_idx] = idx;
                h_recv_begin.data[++cell_idx] = idx;
                }
            last_cell = it->first;

            ++idx;
            }
        h_recv_end.data[cell_idx] = idx;
        }
    }

#endif // ENABLE_MPI
