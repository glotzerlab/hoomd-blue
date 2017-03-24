// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/Communicator.cc
 * \brief Implements the mpcd::Communicator class
 */

#ifdef ENABLE_MPI

#include "Communicator.h"
#include "hoomd/SystemDefinition.h"

#include <algorithm>
#include "hoomd/extern/pybind/include/pybind11/stl.h"

using namespace std;
namespace py = pybind11;

// 27 neighbors is the maximum in 3 dimensions
const unsigned int mpcd::Communicator::neigh_max = 27;

/*!
 * \param sysdef System definition the communicator is associated with
 * \param decomposition Domain decomposition of the global box
 */
mpcd::Communicator::Communicator(std::shared_ptr<mpcd::SystemData> system_data)
          : m_mpcd_sys(system_data),
            m_sysdef(system_data->getSystemDefinition()),
            m_pdata(m_sysdef->getParticleData()),
            m_exec_conf(m_pdata->getExecConf()),
            m_mpcd_pdata(m_mpcd_sys->getParticleData()),
            m_mpi_comm(m_exec_conf->getMPICommunicator()),
            m_decomposition(m_pdata->getDomainDecomposition()),
            m_is_communicating(false),
            m_force_migrate(false),
            m_nneigh(0),
            m_n_unique_neigh(0),
            m_sendbuf(m_exec_conf),
            m_recvbuf(m_exec_conf)
    {
    // initialize array of neighbor processor ids
    assert(m_mpi_comm);
    assert(m_decomposition);

    m_exec_conf->msg->notice(5) << "Constructing MPCD Communicator" << endl;

    // allocate memory
    GPUArray<unsigned int> neighbors(neigh_max,m_exec_conf);
    m_neighbors.swap(neighbors);

    GPUArray<unsigned int> unique_neighbors(neigh_max,m_exec_conf);
    m_unique_neighbors.swap(unique_neighbors);

    // neighbor masks
    GPUArray<unsigned int> adj_mask(neigh_max, m_exec_conf);
    m_adj_mask.swap(adj_mask);

    initializeNeighborArrays();
    }

mpcd::Communicator::~Communicator()
    {
    m_exec_conf->msg->notice(5) << "Destroying MPCD Communicator" << std::endl;
    }

void mpcd::Communicator::initializeNeighborArrays()
    {
    Index3D di= m_decomposition->getDomainIndexer();

    uint3 mypos = m_decomposition->getGridPos();
    int l = mypos.x;
    int m = mypos.y;
    int n = mypos.z;

    ArrayHandle<unsigned int> h_neighbors(m_neighbors, access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_adj_mask(m_adj_mask, access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_cart_ranks(m_decomposition->getCartRanks(), access_location::host, access_mode::read);

    m_nneigh = 0;

    // loop over neighbors
    for (int ix=-1; ix <= 1; ix++)
        {
        int i = ix + l;
        if (i == (int)di.getW())
            i = 0;
        else if (i < 0)
            i += di.getW();

        // only if communicating along x-direction
        if (ix && di.getW() == 1) continue;

        for (int iy=-1; iy <= 1; iy++)
            {
            int j = iy + m;

            if (j == (int)di.getH())
                j = 0;
            else if (j < 0)
                j += di.getH();

            // only if communicating along y-direction
            if (iy && di.getH() == 1) continue;

            for (int iz=-1; iz <= 1; iz++)
                {
                int k = iz + n;

                if (k == (int)di.getD())
                    k = 0;
                else if (k < 0)
                    k += di.getD();

                // only if communicating along z-direction
                if (iz && di.getD() == 1) continue;

                // exclude ourselves
                if (!ix && !iy && !iz) continue;

                unsigned int dir = ((iz+1)*3+(iy+1))*3+(ix + 1);
                unsigned int mask = 1 << dir;

                unsigned int neighbor = h_cart_ranks.data[di(i,j,k)];
                h_neighbors.data[m_nneigh] = neighbor;
                h_adj_mask.data[m_nneigh] = mask;
                m_nneigh++;
                }
            }
        }

    ArrayHandle<unsigned int> h_unique_neighbors(m_unique_neighbors, access_location::host, access_mode::overwrite);

    // filter neighbors, combining adjacency masks
    std::map<unsigned int, unsigned int> neigh_map;
    for (unsigned int i = 0; i < m_nneigh; ++i)
        {
        unsigned int m = 0;

        for (unsigned int j = 0; j < m_nneigh; ++j)
            if (h_neighbors.data[j] == h_neighbors.data[i])
                m |= h_adj_mask.data[j];

        neigh_map.insert(std::make_pair(h_neighbors.data[i], m));
        }

    m_n_unique_neigh = neigh_map.size();

    n = 0;
    for (auto it = neigh_map.begin(); it != neigh_map.end(); ++it)
        {
        h_unique_neighbors.data[n] = it->first;
        h_adj_mask.data[n] = it->second;
        n++;
        }
    }

/*!
 * \param timestep Current timestep for communication
 */
void mpcd::Communicator::communicate(unsigned int timestep)
    {
    if (m_is_communicating)
        {
        m_exec_conf->msg->warning() << "MPCD communication currently underway, ignoring request" << std::endl;
        return;
        }

    // Guard to prevent recursive triggering of migration
    m_is_communicating = true;

    if (m_prof) m_prof->push("MPCD comm");

    migrateParticles();

    if (m_prof) m_prof->pop();

    m_is_communicating = false;
    }

void mpcd::Communicator::migrateParticles()
    {
    if (m_prof) m_prof->push("migrate");

    // determine local particles that are to be sent to neighboring processors
    // TODO: this should check for "covered" box of the cell list
    const BoxDim& box = m_pdata->getBox();

    unsigned int req_comm_flags = setCommFlags(box);
    while (req_comm_flags)
        {
        // fill the buffers and send in each direction
        for (unsigned int dir=0; dir < 6; dir++)
            {
            unsigned int comm_mask = (1 << static_cast<unsigned char>(dir));

            if (!isCommunicating(static_cast<mpcd::detail::face>(dir)) || !(req_comm_flags & comm_mask)) continue;

            // fill send buffer
            m_mpcd_pdata->removeParticles(m_sendbuf, comm_mask);

            // we receive from the direction opposite to the one we send to
            unsigned int send_neighbor = m_decomposition->getNeighborRank(dir);
            unsigned int recv_neighbor;
            if (dir % 2 == 0)
                recv_neighbor = m_decomposition->getNeighborRank(dir+1);
            else
                recv_neighbor = m_decomposition->getNeighborRank(dir-1);

            // communicate size of the message that will contain the particle data
            unsigned int n_recv_ptls;
            unsigned int n_send_ptls = m_sendbuf.size();
            m_reqs.reserve(2); m_stats.reserve(2);
            MPI_Isend(&n_send_ptls, 1, MPI_UNSIGNED, send_neighbor, 0, m_mpi_comm, &m_reqs[0]);
            MPI_Irecv(&n_recv_ptls, 1, MPI_UNSIGNED, recv_neighbor, 0, m_mpi_comm, &m_reqs[1]);
            MPI_Waitall(2, m_reqs.data(), m_stats.data());

            // Resize receive buffer
            m_recvbuf.resize(n_recv_ptls);

            // exchange particle data
                {
                ArrayHandle<mpcd::detail::pdata_element> h_sendbuf(m_sendbuf, access_location::host, access_mode::read);
                ArrayHandle<mpcd::detail::pdata_element> h_recvbuf(m_recvbuf, access_location::host, access_mode::overwrite);
                MPI_Isend(h_sendbuf.data, n_send_ptls*sizeof(mpcd::detail::pdata_element), MPI_BYTE, send_neighbor, 1, m_mpi_comm, &m_reqs[0]);
                MPI_Irecv(h_recvbuf.data, n_recv_ptls*sizeof(mpcd::detail::pdata_element), MPI_BYTE, recv_neighbor, 1, m_mpi_comm, &m_reqs[1]);
                MPI_Waitall(2, m_reqs.data(), m_stats.data());
                }

            // wrap received particles across a global boundary back into global box
                {
                ArrayHandle<mpcd::detail::pdata_element> h_recvbuf(m_recvbuf, access_location::host, access_mode::readwrite);
                const BoxDim& global_box = m_pdata->getGlobalBox();

                for (unsigned int idx = 0; idx < n_recv_ptls; ++idx)
                    {
                    mpcd::detail::pdata_element& p = h_recvbuf.data[idx];
                    Scalar4& postype = p.pos;
                    int3 image = make_int3(0,0,0);

                    global_box.wrap(postype, image);
                    }
                }

            // fill particle data with wrapped, received particles
            m_mpcd_pdata->addParticles(m_recvbuf, comm_mask);
            } // end dir loop

        req_comm_flags = setCommFlags(box);
        }

    if (m_prof) m_prof->pop();
    }

/*!
 * \param box Bounding box
 *
 * Particles lying outside of \a box have their communication flags set along
 * that face.
 */
unsigned int mpcd::Communicator::setCommFlags(const BoxDim& box)
    {
    // mark all particles which have left the box for sending
    unsigned int N = m_mpcd_pdata->getN();
    ArrayHandle<Scalar4> h_pos(m_mpcd_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_comm_flag(m_mpcd_pdata->getCommFlags(), access_location::host, access_mode::overwrite);

    unsigned int req_comm_flags = 0;
    for (unsigned int idx = 0; idx < N; ++idx)
        {
        const Scalar4& postype = h_pos.data[idx];
        Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
        Scalar3 f = box.makeFraction(pos);

        unsigned int flags = 0;
        if (f.x >= Scalar(1.0)) flags |= static_cast<unsigned int>(mpcd::detail::send_mask::east);
        else if (f.x < Scalar(0.0)) flags |= static_cast<unsigned int>(mpcd::detail::send_mask::west);

        if (f.y >= Scalar(1.0)) flags |= static_cast<unsigned int>(mpcd::detail::send_mask::north);
        else if (f.y < Scalar(0.0)) flags |= static_cast<unsigned int>(mpcd::detail::send_mask::south);

        if (f.z >= Scalar(1.0)) flags |= static_cast<unsigned int>(mpcd::detail::send_mask::up);
        else if (f.z < Scalar(0.0)) flags |= static_cast<unsigned int>(mpcd::detail::send_mask::down);

        req_comm_flags |= flags;
        h_comm_flag.data[idx] = flags;
        }

    MPI_Allreduce(MPI_IN_PLACE, &req_comm_flags, 1, MPI_UNSIGNED, MPI_BOR, m_mpi_comm);
    return req_comm_flags;
    }

/*!
 * \param m Python module to export to
 */
void mpcd::detail::export_Communicator(py::module& m)
    {
    py::class_<mpcd::Communicator, std::shared_ptr<mpcd::Communicator> >(m,"Communicator")
    .def(py::init<std::shared_ptr<mpcd::SystemData> >());
    }
#endif // ENABLE_MPI
