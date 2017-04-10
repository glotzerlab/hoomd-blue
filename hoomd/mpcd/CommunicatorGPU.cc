// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/CommunicatorGPU.cc
 * \brief Implements the mpcd::CommunicatorGPU class
 */

#ifdef ENABLE_MPI
#ifdef ENABLE_CUDA

#include "CommunicatorGPU.h"
#include "CommunicatorGPU.cuh"

#include "hoomd/Profiler.h"

namespace py = pybind11;
#include <algorithm>

/*!
 * \param sysdef System definition the communicator is associated with
 * \param mpcd_sys MPCD system data
 */
mpcd::CommunicatorGPU::CommunicatorGPU(std::shared_ptr<mpcd::SystemData> system_data)
    : Communicator(system_data),
      m_max_stages(1),
      m_num_stages(0),
      m_comm_mask(0),
      m_req_comm_flags(m_exec_conf)
    {
    // initialize communciation stages
    initializeCommunicationStages();

    GPUArray<unsigned int> begin(neigh_max,m_exec_conf);
    m_begin.swap(begin);

    GPUArray<unsigned int> end(neigh_max,m_exec_conf);
    m_end.swap(end);

    // autotuners
    m_flags_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "mpcd_comm_flags", m_exec_conf));
    }

mpcd::CommunicatorGPU::~CommunicatorGPU()
    {
    m_exec_conf->msg->notice(5) << "Destroying MPCD CommunicatorGPU";
    }

void mpcd::CommunicatorGPU::initializeCommunicationStages()
    {
    // sanity check for user input
    if (m_max_stages == 0)
        {
        m_exec_conf->msg->warning()
            << "Maximum number of communication stages needs to be greater than zero. Assuming one."
            << std::endl;
        m_max_stages = 1;
        }

    if (m_max_stages > 3)
        {
        m_exec_conf->msg->warning()
            << "Maximum number of communication stages too large. Assuming three."
            << std::endl;
        m_max_stages = 3;
        }

    // accesss neighbors and adjacency  array
    ArrayHandle<unsigned int> h_adj_mask(m_adj_mask, access_location::host, access_mode::read);

    Index3D di= m_decomposition->getDomainIndexer();

    // number of stages in every communication step
    m_num_stages = 0;

    m_comm_mask.clear();
    m_comm_mask.resize(m_max_stages,0);

    const unsigned int mask_east = 1 << 2 | 1 << 5 | 1 << 8 | 1 << 11
        | 1 << 14 | 1 << 17 | 1 << 20 | 1 << 23 | 1 << 26;
    const unsigned int mask_west = mask_east >> 2;
    const unsigned int mask_north = 1 << 6 | 1 << 7 | 1 << 8 | 1 << 15
        | 1 << 16 | 1 << 17 | 1 << 24 | 1 << 25 | 1 << 26;
    const unsigned int mask_south = mask_north >> 6;
    const unsigned int mask_up = 1 << 18 | 1 << 19 | 1 << 20 | 1 << 21
        | 1 << 22 | 1 << 23 | 1 << 24 | 1 << 25 | 1 << 26;
    const unsigned int mask_down = mask_up >> 18;

    // loop through neighbors to determine the communication stages
    std::vector<unsigned int> neigh_flags(m_n_unique_neigh);
    unsigned int max_stage = 0;
    for (unsigned int ineigh = 0; ineigh < m_n_unique_neigh; ++ineigh)
        {
        int stage = 0;
        int n = -1;

        // determine stage
        if (di.getW() > 1 && (n+1 < (int) m_max_stages)) n++;
        if (h_adj_mask.data[ineigh] & (mask_east | mask_west)) stage = n;
        if (di.getH() > 1 && (n+1 < (int) m_max_stages)) n++;
        if (h_adj_mask.data[ineigh] & (mask_north | mask_south)) stage = n;
        if (di.getD() > 1 && (n+1 < (int) m_max_stages)) n++;
        if (h_adj_mask.data[ineigh] & (mask_up | mask_down)) stage = n;

        assert(stage >= 0);
        assert(n >= 0);

        unsigned int mask = 0;
        if (h_adj_mask.data[ineigh] & mask_east)
            mask |= static_cast<unsigned int>(mpcd::detail::send_mask::east);
        if (h_adj_mask.data[ineigh] & mask_west)
            mask |= static_cast<unsigned int>(mpcd::detail::send_mask::west);
        if (h_adj_mask.data[ineigh] & mask_north)
            mask |= static_cast<unsigned int>(mpcd::detail::send_mask::north);
        if (h_adj_mask.data[ineigh] & mask_south)
            mask |= static_cast<unsigned int>(mpcd::detail::send_mask::south);
        if (h_adj_mask.data[ineigh] & mask_up)
            mask |= static_cast<unsigned int>(mpcd::detail::send_mask::up);
        if (h_adj_mask.data[ineigh] & mask_down)
            mask |= static_cast<unsigned int>(mpcd::detail::send_mask::down);

        neigh_flags[ineigh] = mask;

        // set communication flags for stage
        m_comm_mask[stage] |= mask;

        if (stage > (int)max_stage) max_stage = stage;
        }

    // number of communication stages
    m_num_stages = max_stage + 1;

    // every direction occurs in one and only one stages
    // number of communications per stage is constant or decreases with stage number
    for (unsigned int istage = 0; istage < m_num_stages; ++istage)
        for (unsigned int jstage = istage+1; jstage < m_num_stages; ++jstage)
            m_comm_mask[jstage] &= ~m_comm_mask[istage];

    // initialize stages array
    m_stages.resize(m_n_unique_neigh,-1);

    // assign stages to unique neighbors
    for (unsigned int i= 0; i < m_n_unique_neigh; i++)
        {
        for (unsigned int istage = 0; istage < m_num_stages; ++istage)
            {
            // compare adjacency masks of neighbors to the mask for this stage
            if ((neigh_flags[i] & m_comm_mask[istage]) == neigh_flags[i])
                {
                m_stages[i] = istage;
                break; // associate neighbor with stage of lowest index
                }
            }
        }

    m_exec_conf->msg->notice(4) << "MPCD ComunicatorGPU: Using " << m_num_stages
        << " communication stage(s)." << std::endl;
    }

namespace mpcd
{
namespace detail
{
//! Functor to select a particle for migration
struct get_migrate_key : public std::unary_function<const unsigned int, unsigned int>
    {
    const uint3 my_pos;      //!< My domain decomposition position
    const Index3D di;        //!< Domain indexer
    const unsigned int mask; //!< Mask of allowed directions
    const unsigned int *h_cart_ranks; //!< Rank lookup table

    //! Constructor
    /*!
     * \param _my_pos Domain decomposition position
     * \param _di Domain indexer
     * \param _mask Mask of allowed directions
     * \param _h_cart_ranks Rank lookup table
     */
    get_migrate_key(const uint3 _my_pos, const Index3D _di, const unsigned int _mask,
        const unsigned int *_h_cart_ranks)
        : my_pos(_my_pos), di(_di), mask(_mask), h_cart_ranks(_h_cart_ranks)
        { }

    //! Generate key for a sent particle
    /*!
     * \param flags Requested communication flags
     */
    unsigned int operator()(const unsigned int flags)
        {
        int ix, iy, iz;
        ix = iy = iz = 0;

        if ((flags & static_cast<unsigned int>(mpcd::detail::send_mask::east)) &&
            (mask & static_cast<unsigned int>(mpcd::detail::send_mask::east)))
            ix = 1;
        else if ((flags & static_cast<unsigned int>(mpcd::detail::send_mask::west)) &&
                 (mask & static_cast<unsigned int>(mpcd::detail::send_mask::west)))
            ix = -1;

        if ((flags & static_cast<unsigned int>(mpcd::detail::send_mask::north)) &&
            (mask & static_cast<unsigned int>(mpcd::detail::send_mask::north)))
            iy = 1;
        else if ((flags & static_cast<unsigned int>(mpcd::detail::send_mask::south)) &&
                 (mask & static_cast<unsigned int>(mpcd::detail::send_mask::south)))
            iy = -1;

        if ((flags & static_cast<unsigned int>(mpcd::detail::send_mask::up)) &&
            (mask & static_cast<unsigned int>(mpcd::detail::send_mask::up)))
            iz = 1;
        else if ((flags & static_cast<unsigned int>(mpcd::detail::send_mask::down)) &&
                 (mask & static_cast<unsigned int>(mpcd::detail::send_mask::down)))
            iz = -1;

        // sanity check: particle has to be sent somewhere
        assert(ix || iy || iz);

        int i = my_pos.x;
        int j = my_pos.y;
        int k = my_pos.z;

        i += ix;
        if (i == (int)di.getW())
            i = 0;
        else if (i < 0)
            i += di.getW();

        j += iy;
        if (j == (int)di.getH())
            j = 0;
        else if (j < 0)
            j += di.getH();

        k += iz;
        if (k == (int)di.getD())
            k = 0;
        else if (k < 0)
            k += di.getD();

        return h_cart_ranks[di(i,j,k)];
        }
     };
} // end namespace detail
} // end namespace mpcd

void mpcd::CommunicatorGPU::migrateParticles()
    {
    if (m_prof) m_prof->push("migrate");

    // reserve per neighbor memory
    m_n_send_ptls.reserve(m_n_unique_neigh);
    m_n_recv_ptls.reserve(m_n_unique_neigh);
    m_offsets.reserve(m_n_unique_neigh);

    // determine local particles that are to be sent to neighboring processors
    const BoxDim& box = m_mpcd_sys->getCellList()->getCoverageBox();
    setCommFlags(box);

    for (unsigned int stage = 0; stage < m_num_stages; stage++)
        {
        const unsigned int comm_mask = m_comm_mask[stage];

        // fill send buffer
        if (m_prof) m_prof->push(m_exec_conf,"pack");
        m_mpcd_pdata->removeParticlesGPU(m_sendbuf, comm_mask);
        if (m_prof) m_prof->pop(m_exec_conf);

        if (m_prof) m_prof->push("sort");
        // pack the buffers for each neighbor rank in this stage
            {
            ArrayHandle<mpcd::detail::pdata_element> h_sendbuf(m_sendbuf, access_location::host, access_mode::readwrite);
            ArrayHandle<unsigned int> h_begin(m_begin, access_location::host, access_mode::overwrite);
            ArrayHandle<unsigned int> h_end(m_end, access_location::host, access_mode::overwrite);
            ArrayHandle<unsigned int> h_unique_neighbors(m_unique_neighbors, access_location::host, access_mode::read);

            ArrayHandle<unsigned int> h_cart_ranks(m_decomposition->getCartRanks(), access_location::host, access_mode::read);
            std::multimap<unsigned int,mpcd::detail::pdata_element> keys;
            // generate keys
            const uint3 mypos = m_decomposition->getGridPos();
            const Index3D& di = m_decomposition->getDomainIndexer();
            mpcd::detail::get_migrate_key t(mypos, di, m_comm_mask[stage],h_cart_ranks.data);
            for (unsigned int i = 0; i < m_sendbuf.size(); ++i)
                {
                mpcd::detail::pdata_element elem = h_sendbuf.data[i];
                keys.insert(std::pair<unsigned int, mpcd::detail::pdata_element>(t(elem.comm_flag),elem));
                }

            // Find start and end indices
            for (unsigned int i = 0; i < m_n_unique_neigh; ++i)
                {
                auto lower = keys.lower_bound(h_unique_neighbors.data[i]);
                auto upper = keys.upper_bound(h_unique_neighbors.data[i]);
                h_begin.data[i] = std::distance(keys.begin(),lower);
                h_end.data[i] = std::distance(keys.begin(),upper);
                }

            // sort send buffer
            unsigned int i = 0;
            for (auto it = keys.begin(); it != keys.end(); ++it)
                h_sendbuf.data[i++] = it->second;
            }
        if (m_prof) m_prof->pop();

        // communicate total number of particles being sent and received from neighbor ranks
        unsigned int n_recv_tot = 0;
            {
            ArrayHandle<unsigned int> h_begin(m_begin, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_end(m_end, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_unique_neighbors(m_unique_neighbors, access_location::host, access_mode::read);

            // compute send counts
            for (unsigned int ineigh = 0; ineigh < m_n_unique_neigh; ++ineigh)
                m_n_send_ptls[ineigh] = h_end.data[ineigh] - h_begin.data[ineigh];

            // loop over neighbors
            unsigned int nreq = 0;
            m_reqs.reserve(2*m_n_unique_neigh);
            for (unsigned int ineigh = 0; ineigh < m_n_unique_neigh; ++ineigh)
                {
                if (m_stages[ineigh] != (int)stage)
                    {
                    // skip neighbor if not participating in this communication stage
                    m_n_send_ptls[ineigh] = 0;
                    m_n_recv_ptls[ineigh] = 0;
                    continue;
                    }

                // rank of neighbor processor
                unsigned int neighbor = h_unique_neighbors.data[ineigh];

                MPI_Isend(&m_n_send_ptls[ineigh], 1, MPI_UNSIGNED, neighbor, 0, m_mpi_comm, &m_reqs[nreq++]);
                MPI_Irecv(&m_n_recv_ptls[ineigh], 1, MPI_UNSIGNED, neighbor, 0, m_mpi_comm, &m_reqs[nreq++]);
                } // end neighbor loop

            m_stats.reserve(nreq);
            MPI_Waitall(nreq, m_reqs.data(), m_stats.data());

            // sum up receive counts
            for (unsigned int ineigh = 0; ineigh < m_n_unique_neigh; ++ineigh)
                {
                m_offsets[ineigh] = n_recv_tot;
                n_recv_tot += m_n_recv_ptls[ineigh];
                }
            }

        // Resize particles from neighbor ranks
        m_recvbuf.resize(n_recv_tot);
            {
            ArrayHandle<unsigned int> h_begin(m_begin, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_end(m_end, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_unique_neighbors(m_unique_neighbors, access_location::host, access_mode::read);

            ArrayHandle<mpcd::detail::pdata_element> h_sendbuf(m_sendbuf, access_location::host, access_mode::read);
            ArrayHandle<mpcd::detail::pdata_element> h_recvbuf(m_recvbuf, access_location::host, access_mode::overwrite);

            // loop over neighbors
            unsigned int nreq = 0;
            m_reqs.reserve(2*m_n_unique_neigh);
            for (unsigned int ineigh = 0; ineigh < m_n_unique_neigh; ++ineigh)
                {
                // rank of neighbor processor
                unsigned int neighbor = h_unique_neighbors.data[ineigh];

                // exchange particle data
                if (m_n_send_ptls[ineigh])
                    {
                    MPI_Isend(h_sendbuf.data+h_begin.data[ineigh],
                        m_n_send_ptls[ineigh]*sizeof(mpcd::detail::pdata_element),
                        MPI_BYTE,
                        neighbor,
                        1,
                        m_mpi_comm,
                        &m_reqs[nreq++]);
                    }

                if (m_n_recv_ptls[ineigh])
                    {
                    MPI_Irecv(h_recvbuf.data+m_offsets[ineigh],
                        m_n_recv_ptls[ineigh]*sizeof(mpcd::detail::pdata_element),
                        MPI_BYTE,
                        neighbor,
                        1,
                        m_mpi_comm,
                        &m_reqs[nreq++]);
                    }
                }

            m_stats.reserve(2*m_n_unique_neigh);
            MPI_Waitall(nreq, &m_reqs.front(), &m_stats.front());
            }

        // wrap received particles through the global boundary
        if (m_prof) m_prof->push(m_exec_conf, "wrap");
            {
            ArrayHandle<mpcd::detail::pdata_element> d_recvbuf(m_recvbuf, access_location::device, access_mode::readwrite);
            const BoxDim wrap_box = getWrapBox(box);
            mpcd::gpu::wrap_particles(n_recv_tot,
                                      d_recvbuf.data,
                                      wrap_box);
            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }
        if (m_prof) m_prof->pop(m_exec_conf);

        // fill particle data with received particles
        if (m_prof) m_prof->push(m_exec_conf, "unpack");
        m_mpcd_pdata->addParticlesGPU(m_recvbuf, comm_mask);
        if (m_prof) m_prof->pop(m_exec_conf);

        } // end communication stage

    if (m_prof) m_prof->pop();
    }

/*!
 * \param box Bounding box
 *
 * Particles lying outside of \a box have their communication flags set along
 * that face.
 */
void mpcd::CommunicatorGPU::setCommFlags(const BoxDim& box)
    {
    if (m_prof) m_prof->push(m_exec_conf, "comm flags");

    // mark all particles which have left the box for sending
        {
        ArrayHandle<unsigned int> d_comm_flag(m_mpcd_pdata->getCommFlags(), access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_pos(m_mpcd_pdata->getPositions(), access_location::device, access_mode::read);

        m_flags_tuner->begin();
        mpcd::gpu::stage_particles(d_comm_flag.data,
                                   d_pos.data,
                                   m_mpcd_pdata->getN(),
                                   box,
                                   m_flags_tuner->getParam());
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_flags_tuner->end();
        }

    if (m_prof) m_prof->pop(m_exec_conf);
    }

/*!
 * \param m Python module to export to
 */
void mpcd::detail::export_CommunicatorGPU(py::module& m)
    {
    py::class_<mpcd::CommunicatorGPU, std::shared_ptr<mpcd::CommunicatorGPU> >(m,"CommunicatorGPU",py::base<Communicator>())
        .def(py::init<std::shared_ptr<mpcd::SystemData> >())
        .def("setMaxStages",&mpcd::CommunicatorGPU::setMaxStages);
    }

#endif // ENABLE_CUDA
#endif // ENABLE_MPI
