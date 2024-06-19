// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/CommunicatorGPU.cc
 * \brief Implements the mpcd::CommunicatorGPU class
 */

#ifdef ENABLE_MPI
#ifdef ENABLE_HIP

#include "CommunicatorGPU.h"
#include "CommunicatorGPU.cuh"

#include <algorithm>

namespace hoomd
    {
/*!
 * \param sysdef System definition the communicator is associated with
 * \param mpcd_sys MPCD system data
 */
mpcd::CommunicatorGPU::CommunicatorGPU(std::shared_ptr<SystemDefinition> sysdef)
    : Communicator(sysdef), m_max_stages(1), m_num_stages(0), m_comm_mask(0),
      m_tmp_keys(m_exec_conf)
    {
    // initialize communication stages
    initializeCommunicationStages();

    GPUArray<unsigned int> neigh_send(neigh_max, m_exec_conf);
    m_neigh_send.swap(neigh_send);

    GPUArray<unsigned int> num_send(neigh_max, m_exec_conf);
    m_num_send.swap(num_send);

    // autotuners
    m_flags_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                         m_exec_conf,
                                         "mpcd_comm_flags"));
    m_autotuners.push_back(m_flags_tuner);
    }

mpcd::CommunicatorGPU::~CommunicatorGPU() { }

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
            << "Maximum number of communication stages too large. Assuming three." << std::endl;
        m_max_stages = 3;
        }

    // access neighbors and adjacency  array
    ArrayHandle<unsigned int> h_adj_mask(m_adj_mask, access_location::host, access_mode::read);

    Index3D di = m_decomposition->getDomainIndexer();

    // number of stages in every communication step
    m_num_stages = 0;

    m_comm_mask.clear();
    m_comm_mask.resize(m_max_stages, 0);

    const unsigned int mask_east
        = 1 << 2 | 1 << 5 | 1 << 8 | 1 << 11 | 1 << 14 | 1 << 17 | 1 << 20 | 1 << 23 | 1 << 26;
    const unsigned int mask_west = mask_east >> 2;
    const unsigned int mask_north
        = 1 << 6 | 1 << 7 | 1 << 8 | 1 << 15 | 1 << 16 | 1 << 17 | 1 << 24 | 1 << 25 | 1 << 26;
    const unsigned int mask_south = mask_north >> 6;
    const unsigned int mask_up
        = 1 << 18 | 1 << 19 | 1 << 20 | 1 << 21 | 1 << 22 | 1 << 23 | 1 << 24 | 1 << 25 | 1 << 26;
    const unsigned int mask_down = mask_up >> 18;

    // loop through neighbors to determine the communication stages
    std::vector<unsigned int> neigh_flags(m_n_unique_neigh);
    unsigned int max_stage = 0;
    for (unsigned int ineigh = 0; ineigh < m_n_unique_neigh; ++ineigh)
        {
        int stage = 0;
        int n = -1;

        // determine stage
        if (di.getW() > 1 && (n + 1 < (int)m_max_stages))
            n++;
        if (h_adj_mask.data[ineigh] & (mask_east | mask_west))
            stage = n;
        if (di.getH() > 1 && (n + 1 < (int)m_max_stages))
            n++;
        if (h_adj_mask.data[ineigh] & (mask_north | mask_south))
            stage = n;
        if (di.getD() > 1 && (n + 1 < (int)m_max_stages))
            n++;
        if (h_adj_mask.data[ineigh] & (mask_up | mask_down))
            stage = n;

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

        if (stage > (int)max_stage)
            max_stage = stage;
        }

    // number of communication stages
    m_num_stages = max_stage + 1;

    // every direction occurs in one and only one stages
    // number of communications per stage is constant or decreases with stage number
    for (unsigned int istage = 0; istage < m_num_stages; ++istage)
        for (unsigned int jstage = istage + 1; jstage < m_num_stages; ++jstage)
            m_comm_mask[jstage] &= ~m_comm_mask[istage];

    // initialize stages array
    m_stages.resize(m_n_unique_neigh, -1);

    // assign stages to unique neighbors
    for (unsigned int i = 0; i < m_n_unique_neigh; i++)
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

    m_exec_conf->msg->notice(4) << "MPCD CommunicatorGPU: Using " << m_num_stages
                                << " communication stage(s)." << std::endl;
    }

void mpcd::CommunicatorGPU::migrateParticles(uint64_t timestep)
    {
    if (m_mpcd_pdata->getNVirtual() > 0)
        {
        m_exec_conf->msg->warning()
            << "MPCD communication with virtual particles set is not supported, removing them."
            << std::endl;
        m_mpcd_pdata->removeVirtualParticles();
        }

    // reserve per neighbor memory
    m_n_send_ptls.resize(m_n_unique_neigh);
    m_n_recv_ptls.resize(m_n_unique_neigh);
    m_offsets.resize(m_n_unique_neigh);

    // determine local particles that are to be sent to neighboring processors
    const BoxDim box = m_cl->getCoverageBox();
    setCommFlags(box);

    for (unsigned int stage = 0; stage < m_num_stages; stage++)
        {
        const unsigned int comm_mask = m_comm_mask[stage];

        // fill send buffer
        m_mpcd_pdata->removeParticlesGPU(m_sendbuf, comm_mask, timestep);

        // pack the buffers for each neighbor rank in this stage
        std::fill(m_n_send_ptls.begin(), m_n_send_ptls.end(), 0);
        if (m_sendbuf.size() > 0)
            {
            m_tmp_keys.resize(m_sendbuf.size());

            // sort the send buffer on the gpu
            unsigned int num_send_neigh(0);
                {
                ArrayHandle<mpcd::detail::pdata_element> d_sendbuf(m_sendbuf,
                                                                   access_location::device,
                                                                   access_mode::readwrite);
                ArrayHandle<unsigned int> d_neigh_send(m_neigh_send,
                                                       access_location::device,
                                                       access_mode::overwrite);
                ArrayHandle<unsigned int> d_num_send(m_num_send,
                                                     access_location::device,
                                                     access_mode::overwrite);
                ArrayHandle<unsigned int> d_tmp_keys(m_tmp_keys,
                                                     access_location::device,
                                                     access_mode::overwrite);
                ArrayHandle<unsigned int> d_cart_ranks(m_decomposition->getCartRanks(),
                                                       access_location::device,
                                                       access_mode::read);

                num_send_neigh = (unsigned int)mpcd::gpu::sort_comm_send_buffer(
                    d_sendbuf.data,
                    d_neigh_send.data,
                    d_num_send.data,
                    d_tmp_keys.data,
                    m_decomposition->getGridPos(),
                    m_decomposition->getDomainIndexer(),
                    m_comm_mask[stage],
                    d_cart_ranks.data,
                    (unsigned int)(m_sendbuf.size()));
                }

            // fill the number of particles to send for each neighbor
            ArrayHandle<unsigned int> h_neigh_send(m_neigh_send,
                                                   access_location::host,
                                                   access_mode::read);
            ArrayHandle<unsigned int> h_num_send(m_num_send,
                                                 access_location::host,
                                                 access_mode::read);
            ArrayHandle<unsigned int> h_unique_neighbors(m_unique_neighbors,
                                                         access_location::host,
                                                         access_mode::read);
            for (unsigned int i = 0; i < num_send_neigh; ++i)
                {
                const unsigned int neigh = m_unique_neigh_map.find(h_neigh_send.data[i])->second;
                m_n_send_ptls[neigh] = h_num_send.data[i];
                }
            }

        // communicate total number of particles being sent and received from neighbor ranks
        unsigned int n_recv_tot = 0;
            {
            // loop over neighbors
            ArrayHandle<unsigned int> h_unique_neighbors(m_unique_neighbors,
                                                         access_location::host,
                                                         access_mode::read);
            unsigned int nreq = 0;
            m_reqs.resize(2 * m_n_unique_neigh);
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

                MPI_Isend(&m_n_send_ptls[ineigh],
                          1,
                          MPI_UNSIGNED,
                          neighbor,
                          0,
                          m_mpi_comm,
                          &m_reqs[nreq++]);
                MPI_Irecv(&m_n_recv_ptls[ineigh],
                          1,
                          MPI_UNSIGNED,
                          neighbor,
                          0,
                          m_mpi_comm,
                          &m_reqs[nreq++]);
                } // end neighbor loop
            MPI_Waitall(nreq, m_reqs.data(), MPI_STATUSES_IGNORE);

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
            ArrayHandle<unsigned int> h_unique_neighbors(m_unique_neighbors,
                                                         access_location::host,
                                                         access_mode::read);
            ArrayHandle<mpcd::detail::pdata_element> h_sendbuf(m_sendbuf,
                                                               access_location::host,
                                                               access_mode::read);
            ArrayHandle<mpcd::detail::pdata_element> h_recvbuf(m_recvbuf,
                                                               access_location::host,
                                                               access_mode::overwrite);

            // loop over neighbors
            unsigned int nreq = 0;
            m_reqs.resize(2 * m_n_unique_neigh);
            unsigned int sendidx = 0;
            for (unsigned int ineigh = 0; ineigh < m_n_unique_neigh; ++ineigh)
                {
                // rank of neighbor processor
                unsigned int neighbor = h_unique_neighbors.data[ineigh];

                // exchange particle data
                if (m_n_send_ptls[ineigh])
                    {
                    MPI_Isend(h_sendbuf.data + sendidx,
                              m_n_send_ptls[ineigh],
                              m_pdata_element,
                              neighbor,
                              1,
                              m_mpi_comm,
                              &m_reqs[nreq++]);

                    // increment the send index by the amount just transferred
                    sendidx += m_n_send_ptls[ineigh];
                    }

                if (m_n_recv_ptls[ineigh])
                    {
                    MPI_Irecv(h_recvbuf.data + m_offsets[ineigh],
                              m_n_recv_ptls[ineigh],
                              m_pdata_element,
                              neighbor,
                              1,
                              m_mpi_comm,
                              &m_reqs[nreq++]);
                    }
                }

            MPI_Waitall(nreq, m_reqs.data(), MPI_STATUSES_IGNORE);
            }

            // wrap received particles through the global boundary
            {
            ArrayHandle<mpcd::detail::pdata_element> d_recvbuf(m_recvbuf,
                                                               access_location::device,
                                                               access_mode::readwrite);
            const BoxDim wrap_box = getWrapBox(box);
            mpcd::gpu::wrap_particles(n_recv_tot, d_recvbuf.data, wrap_box);
            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

        // fill particle data with received particles
        m_mpcd_pdata->addParticlesGPU(m_recvbuf, comm_mask, timestep);

        } // end communication stage
    }

/*!
 * \param box Bounding box
 *
 * Particles lying outside of \a box have their communication flags set along
 * that face.
 */
void mpcd::CommunicatorGPU::setCommFlags(const BoxDim& box)
    {
    ArrayHandle<unsigned int> d_comm_flag(m_mpcd_pdata->getCommFlags(),
                                          access_location::device,
                                          access_mode::overwrite);
    ArrayHandle<Scalar4> d_pos(m_mpcd_pdata->getPositions(),
                               access_location::device,
                               access_mode::read);

    m_flags_tuner->begin();
    mpcd::gpu::stage_particles(d_comm_flag.data,
                               d_pos.data,
                               m_mpcd_pdata->getN(),
                               box,
                               m_flags_tuner->getParam()[0]);
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_flags_tuner->end();
    }

namespace mpcd
    {
namespace detail
    {
/*!
 * \param m Python module to export to
 */
void export_CommunicatorGPU(pybind11::module& m)
    {
    pybind11::class_<mpcd::CommunicatorGPU,
                     mpcd::Communicator,
                     std::shared_ptr<mpcd::CommunicatorGPU>>(m, "CommunicatorGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
        .def("setMaxStages", &mpcd::CommunicatorGPU::setMaxStages);
    }
    } // namespace detail
    } // namespace mpcd
    } // end namespace hoomd

#endif // ENABLE_HIP
#endif // ENABLE_MPI
