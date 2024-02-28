// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file CommunicatorGPU.cc
    \brief Implements the CommunicatorGPU class
*/

#ifdef ENABLE_MPI
#ifdef ENABLE_HIP

#include "CommunicatorGPU.h"
#include "System.h"

#include <algorithm>

namespace hoomd
    {
//! Constructor
CommunicatorGPU::CommunicatorGPU(std::shared_ptr<SystemDefinition> sysdef,
                                 std::shared_ptr<DomainDecomposition> decomposition)
    : Communicator(sysdef, decomposition), m_max_stages(1), m_num_stages(0), m_comm_mask(0),
      m_bond_comm(*this, m_sysdef->getBondData()), m_angle_comm(*this, m_sysdef->getAngleData()),
      m_dihedral_comm(*this, m_sysdef->getDihedralData()),
      m_improper_comm(*this, m_sysdef->getImproperData()),
      m_constraint_comm(*this, m_sysdef->getConstraintData()),
      m_pair_comm(*this, m_sysdef->getPairData()), m_meshbond_comm(*this),
      m_meshtriangle_comm(*this)
    {
    if (m_exec_conf->allConcurrentManagedAccess())
        {
        // inform the user to use a cuda-aware MPI
        m_exec_conf->msg->notice(2)
            << "Using unified memory with MPI. Make sure to enable CUDA-awareness in your MPI."
            << std::endl;
        }

    // allocate memory
    allocateBuffers();

    // initialize communication stages
    initializeCommunicationStages();

    // create cuda event
    hipEventCreateWithFlags(&m_event, hipEventDisableTiming);
    }

//! Destructor
CommunicatorGPU::~CommunicatorGPU()
    {
    m_exec_conf->msg->notice(5) << "Destroying CommunicatorGPU";
    hipEventDestroy(m_event);
    }

void CommunicatorGPU::updateMeshDefinition()
    {
    Communicator::updateMeshDefinition();
    m_meshbond_comm.addGroupData(m_meshdef->getMeshBondData());
    m_meshtriangle_comm.addGroupData(m_meshdef->getMeshTriangleData());
    }

void CommunicatorGPU::allocateBuffers()
    {
    /*
     * Particle migration
     */
    GlobalVector<detail::pdata_element> gpu_sendbuf(m_exec_conf);
    m_gpu_sendbuf.swap(gpu_sendbuf);

    GlobalVector<detail::pdata_element> gpu_recvbuf(m_exec_conf);
    m_gpu_recvbuf.swap(gpu_recvbuf);

    // Communication flags for every particle sent
    GlobalVector<unsigned int> comm_flags(m_exec_conf);
    m_comm_flags.swap(comm_flags);

    // Key for every particle sent
    GlobalVector<unsigned int> send_keys(m_exec_conf);
    m_send_keys.swap(send_keys);

    /*
     * Ghost communication
     */

    GlobalVector<unsigned int> tag_ghost_sendbuf(m_exec_conf);
    m_tag_ghost_sendbuf.swap(tag_ghost_sendbuf);

    GlobalVector<unsigned int> tag_ghost_recvbuf(m_exec_conf);
    m_tag_ghost_recvbuf.swap(tag_ghost_recvbuf);

    GlobalVector<Scalar4> pos_ghost_sendbuf(m_exec_conf);
    m_pos_ghost_sendbuf.swap(pos_ghost_sendbuf);

    GlobalVector<Scalar4> pos_ghost_recvbuf(m_exec_conf);
    m_pos_ghost_recvbuf.swap(pos_ghost_recvbuf);

    GlobalVector<Scalar4> vel_ghost_sendbuf(m_exec_conf);
    m_vel_ghost_sendbuf.swap(vel_ghost_sendbuf);

    GlobalVector<Scalar4> vel_ghost_recvbuf(m_exec_conf);
    m_vel_ghost_recvbuf.swap(vel_ghost_recvbuf);

    GlobalVector<Scalar> charge_ghost_sendbuf(m_exec_conf);
    m_charge_ghost_sendbuf.swap(charge_ghost_sendbuf);

    GlobalVector<Scalar> charge_ghost_recvbuf(m_exec_conf);
    m_charge_ghost_recvbuf.swap(charge_ghost_recvbuf);

    GlobalVector<unsigned int> body_ghost_sendbuf(m_exec_conf);
    m_body_ghost_sendbuf.swap(body_ghost_sendbuf);

    GlobalVector<unsigned int> body_ghost_recvbuf(m_exec_conf);
    m_body_ghost_recvbuf.swap(body_ghost_recvbuf);

    GlobalVector<int3> image_ghost_sendbuf(m_exec_conf);
    m_image_ghost_sendbuf.swap(image_ghost_sendbuf);

    GlobalVector<int3> image_ghost_recvbuf(m_exec_conf);
    m_image_ghost_recvbuf.swap(image_ghost_recvbuf);

    GlobalVector<Scalar> diameter_ghost_sendbuf(m_exec_conf);
    m_diameter_ghost_sendbuf.swap(diameter_ghost_sendbuf);

    GlobalVector<Scalar> diameter_ghost_recvbuf(m_exec_conf);
    m_diameter_ghost_recvbuf.swap(diameter_ghost_recvbuf);

    GlobalVector<Scalar4> orientation_ghost_sendbuf(m_exec_conf);
    m_orientation_ghost_sendbuf.swap(orientation_ghost_sendbuf);

    GlobalVector<Scalar4> orientation_ghost_recvbuf(m_exec_conf);
    m_orientation_ghost_recvbuf.swap(orientation_ghost_recvbuf);

    GlobalVector<Scalar4> netforce_ghost_sendbuf(m_exec_conf);
    m_netforce_ghost_sendbuf.swap(netforce_ghost_sendbuf);

    GlobalVector<Scalar4> netforce_ghost_recvbuf(m_exec_conf);
    m_netforce_ghost_recvbuf.swap(netforce_ghost_recvbuf);

    GlobalVector<Scalar4> nettorque_ghost_sendbuf(m_exec_conf);
    m_nettorque_ghost_sendbuf.swap(nettorque_ghost_sendbuf);

    GlobalVector<Scalar4> nettorque_ghost_recvbuf(m_exec_conf);
    m_nettorque_ghost_recvbuf.swap(nettorque_ghost_recvbuf);

    GlobalVector<Scalar> netvirial_ghost_sendbuf(m_exec_conf);
    m_netvirial_ghost_sendbuf.swap(netvirial_ghost_sendbuf);

    GlobalVector<Scalar> netvirial_ghost_recvbuf(m_exec_conf);
    m_netvirial_ghost_recvbuf.swap(netvirial_ghost_recvbuf);

    GlobalVector<unsigned int> ghost_begin(m_exec_conf);
    m_ghost_begin.swap(ghost_begin);

    GlobalVector<unsigned int> ghost_end(m_exec_conf);
    m_ghost_end.swap(ghost_end);

    GlobalVector<unsigned int> ghost_plan(m_exec_conf);
    m_ghost_plan.swap(ghost_plan);

    GlobalVector<uint2> ghost_idx_adj(m_exec_conf);
    m_ghost_idx_adj.swap(ghost_idx_adj);

    GlobalVector<unsigned int> ghost_neigh(m_exec_conf);
    m_ghost_neigh.swap(ghost_neigh);

    GlobalVector<unsigned int> neigh_counts(m_exec_conf);
    m_neigh_counts.swap(neigh_counts);

    GlobalVector<unsigned int> scan(m_exec_conf);
    m_scan.swap(scan);
    }

void CommunicatorGPU::initializeCommunicationStages()
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
            mask |= send_east;
        if (h_adj_mask.data[ineigh] & mask_west)
            mask |= send_west;
        if (h_adj_mask.data[ineigh] & mask_north)
            mask |= send_north;
        if (h_adj_mask.data[ineigh] & mask_south)
            mask |= send_south;
        if (h_adj_mask.data[ineigh] & mask_up)
            mask |= send_up;
        if (h_adj_mask.data[ineigh] & mask_down)
            mask |= send_down;

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

    // access unique neighbors
    ArrayHandle<unsigned int> h_unique_neighbors(m_unique_neighbors,
                                                 access_location::host,
                                                 access_mode::read);

    // initialize stages array
    m_stages.resize(m_n_unique_neigh, -1);

    // assign stages to unique neighbors
    for (unsigned int i = 0; i < m_n_unique_neigh; i++)
        for (unsigned int istage = 0; istage < m_num_stages; ++istage)
            // compare adjacency masks of neighbors to the mask for this stage
            if ((neigh_flags[i] & m_comm_mask[istage]) == neigh_flags[i])
                {
                m_stages[i] = istage;
                break; // associate neighbor with stage of lowest index
                }

    m_exec_conf->msg->notice(4) << "CommunicatorGPU: Using " << m_num_stages
                                << " communication stage(s)." << std::endl;
    }

//! Select a particle for migration
struct get_migrate_key
    {
    const uint3 my_pos;               //!< My domain decomposition position
    const Index3D di;                 //!< Domain indexer
    const unsigned int mask;          //!< Mask of allowed directions
    const unsigned int* h_cart_ranks; //!< Rank lookup table

    //! Constructor
    /*!
     */
    get_migrate_key(const uint3 _my_pos,
                    const Index3D _di,
                    const unsigned int _mask,
                    const unsigned int* _h_cart_ranks)
        : my_pos(_my_pos), di(_di), mask(_mask), h_cart_ranks(_h_cart_ranks)
        {
        }

    //! Generate key for a sent particle
    unsigned int operator()(const unsigned int flags)
        {
        int ix, iy, iz;
        ix = iy = iz = 0;

        if ((flags & Communicator::send_east) && (mask & Communicator::send_east))
            ix = 1;
        else if ((flags & Communicator::send_west) && (mask & Communicator::send_west))
            ix = -1;

        if ((flags & Communicator::send_north) && (mask & Communicator::send_north))
            iy = 1;
        else if ((flags & Communicator::send_south) && (mask & Communicator::send_south))
            iy = -1;

        if ((flags & Communicator::send_up) && (mask & Communicator::send_up))
            iz = 1;
        else if ((flags & Communicator::send_down) && (mask & Communicator::send_down))
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

        return h_cart_ranks[di(i, j, k)];
        }
    };

//! Constructor
template<class group_data>
CommunicatorGPU::GroupCommunicatorGPU<group_data>::GroupCommunicatorGPU(
    CommunicatorGPU& gpu_comm,
    std::shared_ptr<group_data> gdata)
    : m_gpu_comm(gpu_comm), m_exec_conf(m_gpu_comm.m_exec_conf), m_gdata(gdata),
      m_ghost_group_begin(m_exec_conf), m_ghost_group_end(m_exec_conf),
      m_ghost_group_idx_adj(m_exec_conf), m_ghost_group_neigh(m_exec_conf),
      m_ghost_group_plan(m_exec_conf), m_neigh_counts(m_exec_conf), m_ghost_scan(m_exec_conf)
    {
    GlobalVector<unsigned int> rank_mask(m_exec_conf);
    m_rank_mask.swap(rank_mask);

    GlobalVector<unsigned int> scan(m_exec_conf);
    m_scan.swap(scan);

    GlobalVector<unsigned int> marked_groups(m_exec_conf);
    m_marked_groups.swap(marked_groups);

    GlobalVector<rank_element_t> ranks_out(m_exec_conf);
    m_ranks_out.swap(ranks_out);

    GlobalVector<rank_element_t> ranks_sendbuf(m_exec_conf);
    m_ranks_sendbuf.swap(ranks_sendbuf);

    GlobalVector<rank_element_t> ranks_recvbuf(m_exec_conf);
    m_ranks_recvbuf.swap(ranks_recvbuf);

    GlobalVector<group_element_t> groups_out(m_exec_conf);
    m_groups_out.swap(groups_out);

    GlobalVector<unsigned int> rank_mask_out(m_exec_conf);
    m_rank_mask_out.swap(rank_mask_out);

    GlobalVector<group_element_t> groups_sendbuf(m_exec_conf);
    m_groups_sendbuf.swap(groups_sendbuf);

    GlobalVector<group_element_t> groups_recvbuf(m_exec_conf);
    m_groups_recvbuf.swap(groups_recvbuf);

    GlobalVector<group_element_t> groups_in(m_exec_conf);
    m_groups_in.swap(groups_in);

    // the size of the bit field must be larger or equal the group size
    assert(sizeof(unsigned int) * 8 >= group_data::size);
    }

template<class group_data>
CommunicatorGPU::GroupCommunicatorGPU<group_data>::GroupCommunicatorGPU(CommunicatorGPU& gpu_comm)
    : m_gpu_comm(gpu_comm), m_exec_conf(m_gpu_comm.m_exec_conf), m_gdata(NULL),
      m_ghost_group_begin(m_exec_conf), m_ghost_group_end(m_exec_conf),
      m_ghost_group_idx_adj(m_exec_conf), m_ghost_group_neigh(m_exec_conf),
      m_ghost_group_plan(m_exec_conf), m_neigh_counts(m_exec_conf), m_ghost_scan(m_exec_conf)
    {
    GlobalVector<unsigned int> rank_mask(m_exec_conf);
    m_rank_mask.swap(rank_mask);

    GlobalVector<unsigned int> scan(m_exec_conf);
    m_scan.swap(scan);

    GlobalVector<unsigned int> marked_groups(m_exec_conf);
    m_marked_groups.swap(marked_groups);

    GlobalVector<rank_element_t> ranks_out(m_exec_conf);
    m_ranks_out.swap(ranks_out);

    GlobalVector<rank_element_t> ranks_sendbuf(m_exec_conf);
    m_ranks_sendbuf.swap(ranks_sendbuf);

    GlobalVector<rank_element_t> ranks_recvbuf(m_exec_conf);
    m_ranks_recvbuf.swap(ranks_recvbuf);

    GlobalVector<group_element_t> groups_out(m_exec_conf);
    m_groups_out.swap(groups_out);

    GlobalVector<unsigned int> rank_mask_out(m_exec_conf);
    m_rank_mask_out.swap(rank_mask_out);

    GlobalVector<group_element_t> groups_sendbuf(m_exec_conf);
    m_groups_sendbuf.swap(groups_sendbuf);

    GlobalVector<group_element_t> groups_recvbuf(m_exec_conf);
    m_groups_recvbuf.swap(groups_recvbuf);

    GlobalVector<group_element_t> groups_in(m_exec_conf);
    m_groups_in.swap(groups_in);
    }

template<class group_data>
void CommunicatorGPU::GroupCommunicatorGPU<group_data>::addGroupData(
    std::shared_ptr<group_data> gdata)
    {
    m_gdata = gdata;

    // the size of the bit field must be larger or equal the group size
    assert(sizeof(unsigned int) * 8 >= group_data::size);
    }

//! Migrate groups
template<class group_data>
void CommunicatorGPU::GroupCommunicatorGPU<group_data>::migrateGroups(bool incomplete,
                                                                      bool local_multiple)
    {
    if (m_gdata->getNGlobal())
        {
        m_exec_conf->msg->notice(7)
            << "GroupCommunicator<" << m_gdata->getName() << ">: migrate" << std::endl;

            {
            // Reset reverse lookup tags of old ghost groups
            ArrayHandle<unsigned int> d_group_rtag(m_gdata->getRTags(),
                                                   access_location::device,
                                                   access_mode::readwrite);
            ArrayHandle<unsigned int> d_group_tag(m_gdata->getTags(),
                                                  access_location::device,
                                                  access_mode::read);

            gpu_reset_rtags(m_gdata->getNGhosts(),
                            d_group_tag.data + m_gdata->getN(),
                            d_group_rtag.data);

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

        // remove ghost groups
        m_gdata->removeAllGhostGroups();

        // resize bitmasks
        m_rank_mask.resize(m_gdata->getN());

        // resize temporary array
        m_scan.resize(m_gdata->getN());
        m_marked_groups.resize(m_gdata->getN());

        unsigned int n_out_ranks;
            {
            ArrayHandle<unsigned int> d_comm_flags(m_gpu_comm.m_pdata->getCommFlags(),
                                                   access_location::device,
                                                   access_mode::read);
            ArrayHandle<typename group_data::members_t> d_members(m_gdata->getMembersArray(),
                                                                  access_location::device,
                                                                  access_mode::read);
            ArrayHandle<typename group_data::ranks_t> d_group_ranks(m_gdata->getRanksArray(),
                                                                    access_location::device,
                                                                    access_mode::readwrite);
            ArrayHandle<unsigned int> d_rank_mask(m_rank_mask,
                                                  access_location::device,
                                                  access_mode::overwrite);
            ArrayHandle<unsigned int> d_rtag(m_gpu_comm.m_pdata->getRTags(),
                                             access_location::device,
                                             access_mode::read);
            ArrayHandle<unsigned int> d_marked_groups(m_marked_groups,
                                                      access_location::device,
                                                      access_mode::overwrite);
            ArrayHandle<unsigned int> d_scan(m_scan,
                                             access_location::device,
                                             access_mode::overwrite);

            std::shared_ptr<DomainDecomposition> decomposition
                = m_gpu_comm.m_pdata->getDomainDecomposition();
            ArrayHandle<unsigned int> d_cart_ranks(decomposition->getCartRanks(),
                                                   access_location::device,
                                                   access_mode::read);

            Index3D di = decomposition->getDomainIndexer();
            uint3 my_pos = decomposition->getGridPos();

            // mark groups that have members leaving this domain
            gpu_mark_groups<group_data::size>(m_gpu_comm.m_pdata->getN(),
                                              d_comm_flags.data,
                                              m_gdata->getN(),
                                              d_members.data,
                                              d_group_ranks.data,
                                              d_rank_mask.data,
                                              d_rtag.data,
                                              d_marked_groups.data,
                                              d_scan.data,
                                              n_out_ranks,
                                              di,
                                              my_pos,
                                              d_cart_ranks.data,
                                              incomplete,
                                              m_exec_conf->getCachedAllocator());

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

        // resize output array
        m_ranks_out.resize(n_out_ranks);

        unsigned int n_out_groups;
            {
            ArrayHandle<unsigned int> d_group_tag(m_gdata->getTags(),
                                                  access_location::device,
                                                  access_mode::read);
            ArrayHandle<typename group_data::members_t> d_members(m_gdata->getMembersArray(),
                                                                  access_location::device,
                                                                  access_mode::read);
            ArrayHandle<typename group_data::ranks_t> d_group_ranks(m_gdata->getRanksArray(),
                                                                    access_location::device,
                                                                    access_mode::read);
            ArrayHandle<unsigned int> d_rank_mask(m_rank_mask,
                                                  access_location::device,
                                                  access_mode::readwrite);

            ArrayHandle<unsigned int> d_rtag(m_gpu_comm.m_pdata->getRTags(),
                                             access_location::device,
                                             access_mode::read);
            ArrayHandle<unsigned int> d_comm_flags(m_gpu_comm.m_pdata->getCommFlags(),
                                                   access_location::device,
                                                   access_mode::readwrite);

            ArrayHandle<rank_element_t> d_ranks_out(m_ranks_out,
                                                    access_location::device,
                                                    access_mode::overwrite);

            ArrayHandle<unsigned int> d_marked_groups(m_marked_groups,
                                                      access_location::device,
                                                      access_mode::overwrite);
            ArrayHandle<unsigned int> d_scan(m_scan,
                                             access_location::device,
                                             access_mode::readwrite);

            // scatter groups into output arrays according to scan result (d_scan), determine send
            // groups and scan
            gpu_scatter_ranks_and_mark_send_groups<group_data::size>(
                m_gdata->getN(),
                d_group_tag.data,
                d_group_ranks.data,
                d_rank_mask.data,
                d_members.data,
                d_rtag.data,
                d_comm_flags.data,
                d_marked_groups.data,
                d_scan.data,
                n_out_groups,
                d_ranks_out.data,
                m_exec_conf->getCachedAllocator());

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

        // fill host send buffers on host
        unsigned int my_rank = m_exec_conf->getRank();

        typedef std::multimap<unsigned int, rank_element_t> map_t;
        map_t send_map;

        unsigned int group_size = group_data::size;

            {
            // access output buffers
            ArrayHandle<rank_element_t> h_ranks_out(m_ranks_out,
                                                    access_location::host,
                                                    access_mode::read);
            ArrayHandle<unsigned int> h_unique_neighbors(m_gpu_comm.m_unique_neighbors,
                                                         access_location::host,
                                                         access_mode::read);

            for (unsigned int i = 0; i < n_out_ranks; ++i)
                {
                rank_element_t el = h_ranks_out.data[i];
                typename group_data::ranks_t r = el.ranks;
                unsigned int mask = el.mask;

                if (incomplete)
                    // in initialization, send to all neighbors
                    for (unsigned int ineigh = 0; ineigh < m_gpu_comm.m_n_unique_neigh; ++ineigh)
                        send_map.insert(std::make_pair(h_unique_neighbors.data[ineigh], el));
                else
                    // send to other ranks owning the bonded group
                    for (unsigned int j = 0; j < group_size; ++j)
                        {
                        unsigned int rank = r.idx[j];
                        bool updated = mask & (1 << j);
                        // send out to ranks different from ours
                        if (rank != my_rank && !updated)
                            send_map.insert(std::make_pair(rank, el));
                        }
                }
            }

        // resize send buffers
        m_ranks_sendbuf.resize(send_map.size());

            {
            // access send buffers
            ArrayHandle<rank_element_t> h_ranks_sendbuf(m_ranks_sendbuf,
                                                        access_location::host,
                                                        access_mode::overwrite);

            // output send data sorted by rank
            unsigned int n = 0;
            for (typename map_t::iterator it = send_map.begin(); it != send_map.end(); ++it)
                {
                h_ranks_sendbuf.data[n] = it->second;
                n++;
                }

            ArrayHandle<unsigned int> h_unique_neighbors(m_gpu_comm.m_unique_neighbors,
                                                         access_location::host,
                                                         access_mode::read);
            ArrayHandle<unsigned int> h_begin(m_gpu_comm.m_begin,
                                              access_location::host,
                                              access_mode::overwrite);
            ArrayHandle<unsigned int> h_end(m_gpu_comm.m_end,
                                            access_location::host,
                                            access_mode::overwrite);

            // Find start and end indices
            for (unsigned int i = 0; i < m_gpu_comm.m_n_unique_neigh; ++i)
                {
                typename map_t::iterator lower = send_map.lower_bound(h_unique_neighbors.data[i]);
                typename map_t::iterator upper = send_map.upper_bound(h_unique_neighbors.data[i]);
                h_begin.data[i] = (unsigned int)(std::distance(send_map.begin(), lower));
                h_end.data[i] = (unsigned int)(std::distance(send_map.begin(), upper));
                }
            }

        /*
         * communicate rank information (phase 1)
         */
        unsigned int n_send_groups[m_gpu_comm.m_n_unique_neigh];
        unsigned int n_recv_groups[m_gpu_comm.m_n_unique_neigh];
        unsigned int offs[m_gpu_comm.m_n_unique_neigh];
        unsigned int n_recv_tot = 0;

            {
            ArrayHandle<unsigned int> h_begin(m_gpu_comm.m_begin,
                                              access_location::host,
                                              access_mode::read);
            ArrayHandle<unsigned int> h_end(m_gpu_comm.m_end,
                                            access_location::host,
                                            access_mode::read);
            ArrayHandle<unsigned int> h_unique_neighbors(m_gpu_comm.m_unique_neighbors,
                                                         access_location::host,
                                                         access_mode::read);

            unsigned int send_bytes = 0;
            unsigned int recv_bytes = 0;

            // compute send counts
            for (unsigned int ineigh = 0; ineigh < m_gpu_comm.m_n_unique_neigh; ineigh++)
                n_send_groups[ineigh] = h_end.data[ineigh] - h_begin.data[ineigh];

            MPI_Request req[2 * m_gpu_comm.m_n_unique_neigh];
            MPI_Status stat[2 * m_gpu_comm.m_n_unique_neigh];

            unsigned int nreq = 0;

            // loop over neighbors
            for (unsigned int ineigh = 0; ineigh < m_gpu_comm.m_n_unique_neigh; ineigh++)
                {
                // rank of neighbor processor
                unsigned int neighbor = h_unique_neighbors.data[ineigh];

                MPI_Isend(&n_send_groups[ineigh],
                          1,
                          MPI_UNSIGNED,
                          neighbor,
                          0,
                          m_gpu_comm.m_mpi_comm,
                          &req[nreq++]);
                MPI_Irecv(&n_recv_groups[ineigh],
                          1,
                          MPI_UNSIGNED,
                          neighbor,
                          0,
                          m_gpu_comm.m_mpi_comm,
                          &req[nreq++]);
                send_bytes += (unsigned int)sizeof(unsigned int);
                recv_bytes += (unsigned int)sizeof(unsigned int);
                } // end neighbor loop

            MPI_Waitall(nreq, req, stat);

            // sum up receive counts
            for (unsigned int ineigh = 0; ineigh < m_gpu_comm.m_n_unique_neigh; ineigh++)
                {
                if (ineigh == 0)
                    offs[ineigh] = 0;
                else
                    offs[ineigh] = offs[ineigh - 1] + n_recv_groups[ineigh - 1];

                n_recv_tot += n_recv_groups[ineigh];
                }
            }

        // Resize receive buffer
        m_ranks_recvbuf.resize(n_recv_tot);

            {
            ArrayHandle<unsigned int> h_begin(m_gpu_comm.m_begin,
                                              access_location::host,
                                              access_mode::read);
            ArrayHandle<unsigned int> h_end(m_gpu_comm.m_end,
                                            access_location::host,
                                            access_mode::read);
            ArrayHandle<unsigned int> h_unique_neighbors(m_gpu_comm.m_unique_neighbors,
                                                         access_location::host,
                                                         access_mode::read);
            ArrayHandle<rank_element_t> ranks_sendbuf_handle(m_ranks_sendbuf,
                                                             access_location::host,
                                                             access_mode::read);
            ArrayHandle<rank_element_t> ranks_recvbuf_handle(m_ranks_recvbuf,
                                                             access_location::host,
                                                             access_mode::overwrite);

            std::vector<MPI_Request> reqs;
            MPI_Request req;

            unsigned int send_bytes = 0;
            unsigned int recv_bytes = 0;

            // loop over neighbors
            for (unsigned int ineigh = 0; ineigh < m_gpu_comm.m_n_unique_neigh; ineigh++)
                {
                // rank of neighbor processor
                unsigned int neighbor = h_unique_neighbors.data[ineigh];

                // exchange particle data
                if (n_send_groups[ineigh])
                    {
                    MPI_Isend(ranks_sendbuf_handle.data + h_begin.data[ineigh],
                              int(n_send_groups[ineigh] * sizeof(rank_element_t)),
                              MPI_BYTE,
                              neighbor,
                              1,
                              m_gpu_comm.m_mpi_comm,
                              &req);
                    reqs.push_back(req);
                    }
                send_bytes += (unsigned int)(n_send_groups[ineigh] * sizeof(rank_element_t));

                if (n_recv_groups[ineigh])
                    {
                    MPI_Irecv(ranks_recvbuf_handle.data + offs[ineigh],
                              int(n_recv_groups[ineigh] * sizeof(rank_element_t)),
                              MPI_BYTE,
                              neighbor,
                              1,
                              m_gpu_comm.m_mpi_comm,
                              &req);
                    reqs.push_back(req);
                    }
                recv_bytes += (unsigned int)(n_recv_groups[ineigh] * sizeof(rank_element_t));
                }

            std::vector<MPI_Status> stats(reqs.size());
            MPI_Waitall((unsigned int)reqs.size(), &reqs.front(), &stats.front());
            }

            {
            // access receive buffers
            ArrayHandle<rank_element_t> d_ranks_recvbuf(m_ranks_recvbuf,
                                                        access_location::device,
                                                        access_mode::read);
            ArrayHandle<typename group_data::ranks_t> d_group_ranks(m_gdata->getRanksArray(),
                                                                    access_location::device,
                                                                    access_mode::readwrite);
            ArrayHandle<unsigned int> d_group_rtag(m_gdata->getRTags(),
                                                   access_location::device,
                                                   access_mode::read);

            // update local rank information
            gpu_update_ranks_table<group_data::size>(m_gdata->getN(),
                                                     d_group_ranks.data,
                                                     d_group_rtag.data,
                                                     n_recv_tot,
                                                     d_ranks_recvbuf.data);
            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

        // resize output buffer
        m_groups_out.resize(n_out_groups);
        m_rank_mask_out.resize(n_out_groups);

            {
            ArrayHandle<typename group_data::members_t> d_groups(m_gdata->getMembersArray(),
                                                                 access_location::device,
                                                                 access_mode::read);
            ArrayHandle<typeval_t> d_group_typeval(m_gdata->getTypeValArray(),
                                                   access_location::device,
                                                   access_mode::read);
            ArrayHandle<unsigned int> d_group_tag(m_gdata->getTags(),
                                                  access_location::device,
                                                  access_mode::read);
            ArrayHandle<unsigned int> d_group_rtag(m_gdata->getRTags(),
                                                   access_location::device,
                                                   access_mode::readwrite);
            ArrayHandle<typename group_data::ranks_t> d_group_ranks(m_gdata->getRanksArray(),
                                                                    access_location::device,
                                                                    access_mode::read);
            ArrayHandle<unsigned int> d_rank_mask(m_rank_mask,
                                                  access_location::device,
                                                  access_mode::readwrite);
            ArrayHandle<unsigned int> d_scan(m_scan, access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_marked_groups(m_marked_groups,
                                                      access_location::device,
                                                      access_mode::overwrite);
            ArrayHandle<group_element_t> d_groups_out(m_groups_out,
                                                      access_location::device,
                                                      access_mode::overwrite);
            ArrayHandle<unsigned int> d_rank_mask_out(m_rank_mask_out,
                                                      access_location::device,
                                                      access_mode::overwrite);
            ArrayHandle<unsigned int> d_rtag(m_gpu_comm.m_pdata->getRTags(),
                                             access_location::device,
                                             access_mode::read);
            ArrayHandle<unsigned int> d_comm_flags(m_gpu_comm.m_pdata->getCommFlags(),
                                                   access_location::device,
                                                   access_mode::read);

            // scatter groups to be sent into output buffer, mark groups that have no local members
            // for removal
            gpu_scatter_and_mark_groups_for_removal<group_data::size>(m_gdata->getN(),
                                                                      d_groups.data,
                                                                      d_group_typeval.data,
                                                                      d_group_tag.data,
                                                                      d_group_rtag.data,
                                                                      d_group_ranks.data,
                                                                      d_rank_mask.data,
                                                                      d_rtag.data,
                                                                      d_comm_flags.data,
                                                                      m_exec_conf->getRank(),
                                                                      d_scan.data,
                                                                      d_marked_groups.data,
                                                                      d_groups_out.data,
                                                                      d_rank_mask_out.data,
                                                                      local_multiple);
            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

        unsigned int new_ngroups;
            {
            // access primary arrays to read from
            ArrayHandle<typename group_data::members_t> d_groups(m_gdata->getMembersArray(),
                                                                 access_location::device,
                                                                 access_mode::read);
            ArrayHandle<typeval_t> d_group_typeval(m_gdata->getTypeValArray(),
                                                   access_location::device,
                                                   access_mode::read);
            ArrayHandle<unsigned int> d_group_tag(m_gdata->getTags(),
                                                  access_location::device,
                                                  access_mode::read);
            ArrayHandle<typename group_data::ranks_t> d_group_ranks(m_gdata->getRanksArray(),
                                                                    access_location::device,
                                                                    access_mode::read);

            // access alternate arrays to write to
            ArrayHandle<typename group_data::members_t> d_groups_alt(m_gdata->getAltMembersArray(),
                                                                     access_location::device,
                                                                     access_mode::overwrite);
            ArrayHandle<typeval_t> d_group_typeval_alt(m_gdata->getAltTypeValArray(),
                                                       access_location::device,
                                                       access_mode::overwrite);
            ArrayHandle<unsigned int> d_group_tag_alt(m_gdata->getAltTags(),
                                                      access_location::device,
                                                      access_mode::overwrite);
            ArrayHandle<typename group_data::ranks_t> d_group_ranks_alt(m_gdata->getAltRanksArray(),
                                                                        access_location::device,
                                                                        access_mode::overwrite);

            ArrayHandle<unsigned int> d_scan(m_scan,
                                             access_location::device,
                                             access_mode::overwrite);
            ArrayHandle<unsigned int> d_marked_groups(m_marked_groups,
                                                      access_location::device,
                                                      access_mode::read);

            // access rtags
            ArrayHandle<unsigned int> d_group_rtag(m_gdata->getRTags(),
                                                   access_location::device,
                                                   access_mode::readwrite);

            unsigned int ngroups = m_gdata->getN();

            // remove groups from local table
            gpu_remove_groups(ngroups,
                              d_groups.data,
                              d_groups_alt.data,
                              d_group_typeval.data,
                              d_group_typeval_alt.data,
                              d_group_tag.data,
                              d_group_tag_alt.data,
                              d_group_ranks.data,
                              d_group_ranks_alt.data,
                              d_group_rtag.data,
                              new_ngroups,
                              d_marked_groups.data,
                              d_scan.data,
                              m_exec_conf->getCachedAllocator());
            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

        // make alternate arrays current
        m_gdata->swapMemberArrays();
        m_gdata->swapTypeArrays();
        m_gdata->swapTagArrays();
        m_gdata->swapRankArrays();

        // resize group arrays
        assert(new_ngroups <= m_gdata->getN());
        m_gdata->removeGroups(m_gdata->getN() - new_ngroups);
        assert(m_gdata->getN() == new_ngroups);

        // fill host send buffers on host
        typedef std::multimap<unsigned int, group_element_t> group_map_t;
        group_map_t group_send_map;
            {
            // access output buffers
            ArrayHandle<group_element_t> h_groups_out(m_groups_out,
                                                      access_location::host,
                                                      access_mode::read);
            ArrayHandle<unsigned int> h_rank_mask_out(m_rank_mask_out,
                                                      access_location::host,
                                                      access_mode::read);

            for (unsigned int i = 0; i < n_out_groups; ++i)
                {
                group_element_t el = h_groups_out.data[i];
                typename group_data::ranks_t ranks = el.ranks;

                for (unsigned int j = 0; j < group_size; ++j)
                    {
                    unsigned int rank = ranks.idx[j];
                    // are we sending to this rank?
                    if (h_rank_mask_out.data[i] & (1 << j))
                        group_send_map.insert(std::make_pair(rank, el));
                    }
                }
            }

        // resize send buffers
        m_groups_sendbuf.resize(group_send_map.size());

            {
            // access send buffers
            ArrayHandle<group_element_t> h_groups_sendbuf(m_groups_sendbuf,
                                                          access_location::host,
                                                          access_mode::overwrite);

            // output send data sorted by rank
            unsigned int n = 0;
            for (typename group_map_t::iterator it = group_send_map.begin();
                 it != group_send_map.end();
                 ++it)
                {
                h_groups_sendbuf.data[n] = it->second;
                n++;
                }

            ArrayHandle<unsigned int> h_unique_neighbors(m_gpu_comm.m_unique_neighbors,
                                                         access_location::host,
                                                         access_mode::read);
            ArrayHandle<unsigned int> h_begin(m_gpu_comm.m_begin,
                                              access_location::host,
                                              access_mode::overwrite);
            ArrayHandle<unsigned int> h_end(m_gpu_comm.m_end,
                                            access_location::host,
                                            access_mode::overwrite);

            // Find start and end indices
            for (unsigned int i = 0; i < m_gpu_comm.m_n_unique_neigh; ++i)
                {
                typename group_map_t::iterator lower
                    = group_send_map.lower_bound(h_unique_neighbors.data[i]);
                typename group_map_t::iterator upper
                    = group_send_map.upper_bound(h_unique_neighbors.data[i]);
                h_begin.data[i] = (unsigned int)std::distance(group_send_map.begin(), lower);
                h_end.data[i] = (unsigned int)std::distance(group_send_map.begin(), upper);
                }
            }

        /*
         * communicate groups (phase 2)
         */

        n_recv_tot = 0;
            {
            ArrayHandle<unsigned int> h_begin(m_gpu_comm.m_begin,
                                              access_location::host,
                                              access_mode::read);
            ArrayHandle<unsigned int> h_end(m_gpu_comm.m_end,
                                            access_location::host,
                                            access_mode::read);
            ArrayHandle<unsigned int> h_unique_neighbors(m_gpu_comm.m_unique_neighbors,
                                                         access_location::host,
                                                         access_mode::read);

            unsigned int send_bytes = 0;
            unsigned int recv_bytes = 0;

            // compute send counts
            for (unsigned int ineigh = 0; ineigh < m_gpu_comm.m_n_unique_neigh; ineigh++)
                n_send_groups[ineigh] = h_end.data[ineigh] - h_begin.data[ineigh];

            MPI_Request req[2 * m_gpu_comm.m_n_unique_neigh];
            MPI_Status stat[2 * m_gpu_comm.m_n_unique_neigh];

            unsigned int nreq = 0;

            // loop over neighbors
            for (unsigned int ineigh = 0; ineigh < m_gpu_comm.m_n_unique_neigh; ineigh++)
                {
                // rank of neighbor processor
                unsigned int neighbor = h_unique_neighbors.data[ineigh];

                MPI_Isend(&n_send_groups[ineigh],
                          1,
                          MPI_UNSIGNED,
                          neighbor,
                          0,
                          m_gpu_comm.m_mpi_comm,
                          &req[nreq++]);
                MPI_Irecv(&n_recv_groups[ineigh],
                          1,
                          MPI_UNSIGNED,
                          neighbor,
                          0,
                          m_gpu_comm.m_mpi_comm,
                          &req[nreq++]);
                send_bytes += (unsigned int)sizeof(unsigned int);
                recv_bytes += (unsigned int)sizeof(unsigned int);
                } // end neighbor loop

            MPI_Waitall(nreq, req, stat);

            // sum up receive counts
            for (unsigned int ineigh = 0; ineigh < m_gpu_comm.m_n_unique_neigh; ineigh++)
                {
                if (ineigh == 0)
                    offs[ineigh] = 0;
                else
                    offs[ineigh] = offs[ineigh - 1] + n_recv_groups[ineigh - 1];

                n_recv_tot += n_recv_groups[ineigh];
                }
            }

        // Resize receive buffer
        m_groups_recvbuf.resize(n_recv_tot);

            {
            ArrayHandle<unsigned int> h_begin(m_gpu_comm.m_begin,
                                              access_location::host,
                                              access_mode::read);
            ArrayHandle<unsigned int> h_end(m_gpu_comm.m_end,
                                            access_location::host,
                                            access_mode::read);
            ArrayHandle<unsigned int> h_unique_neighbors(m_gpu_comm.m_unique_neighbors,
                                                         access_location::host,
                                                         access_mode::read);
            ArrayHandle<group_element_t> groups_sendbuf_handle(m_groups_sendbuf,
                                                               access_location::host,
                                                               access_mode::read);
            ArrayHandle<group_element_t> groups_recvbuf_handle(m_groups_recvbuf,
                                                               access_location::host,
                                                               access_mode::overwrite);

            std::vector<MPI_Request> reqs;
            MPI_Request req;

            unsigned int send_bytes = 0;
            unsigned int recv_bytes = 0;

            // loop over neighbors
            for (unsigned int ineigh = 0; ineigh < m_gpu_comm.m_n_unique_neigh; ineigh++)
                {
                // rank of neighbor processor
                unsigned int neighbor = h_unique_neighbors.data[ineigh];

                // exchange particle data
                if (n_send_groups[ineigh])
                    {
                    MPI_Isend(groups_sendbuf_handle.data + h_begin.data[ineigh],
                              int(n_send_groups[ineigh] * sizeof(group_element_t)),
                              MPI_BYTE,
                              neighbor,
                              1,
                              m_gpu_comm.m_mpi_comm,
                              &req);
                    reqs.push_back(req);
                    }
                send_bytes += (unsigned int)(n_send_groups[ineigh] * sizeof(group_element_t));

                if (n_recv_groups[ineigh])
                    {
                    MPI_Irecv(groups_recvbuf_handle.data + offs[ineigh],
                              int(n_recv_groups[ineigh] * sizeof(group_element_t)),
                              MPI_BYTE,
                              neighbor,
                              1,
                              m_gpu_comm.m_mpi_comm,
                              &req);
                    reqs.push_back(req);
                    }
                recv_bytes += (unsigned int)(n_recv_groups[ineigh] * sizeof(group_element_t));
                }

            std::vector<MPI_Status> stats(reqs.size());
            MPI_Waitall((unsigned int)reqs.size(), &reqs.front(), &stats.front());
            }

        unsigned int n_recv_unique = 0;
            {
            ArrayHandle<group_element_t> h_groups_recvbuf(m_groups_recvbuf,
                                                          access_location::host,
                                                          access_mode::read);

            // use a std::map, i.e. single-key, to filter out duplicate groups in input buffer
            typedef std::map<unsigned int, group_element_t> recv_map_t;
            recv_map_t recv_map;

            for (unsigned int recv_idx = 0; recv_idx < n_recv_tot; recv_idx++)
                {
                group_element_t el = h_groups_recvbuf.data[recv_idx];
                unsigned int tag = el.group_tag;
                recv_map.insert(std::make_pair(tag, el));
                }

            // resize input array of unique groups
            m_groups_in.resize(recv_map.size());

            // write out unique groups
            ArrayHandle<group_element_t> h_groups_in(m_groups_in,
                                                     access_location::host,
                                                     access_mode::overwrite);
            for (typename recv_map_t::iterator it = recv_map.begin(); it != recv_map.end(); ++it)
                h_groups_in.data[n_recv_unique++] = it->second;
            assert(n_recv_unique == recv_map.size());
            }

        unsigned int old_ngroups = m_gdata->getN();

        // resize group arrays to accommodate additional groups (there can still be duplicates with
        // local groups)
        m_gdata->addGroups(n_recv_unique);

            {
            ArrayHandle<group_element_t> d_groups_in(m_groups_in,
                                                     access_location::device,
                                                     access_mode::read);
            ArrayHandle<typename group_data::members_t> d_groups(m_gdata->getMembersArray(),
                                                                 access_location::device,
                                                                 access_mode::readwrite);
            ArrayHandle<typeval_t> d_group_typeval(m_gdata->getTypeValArray(),
                                                   access_location::device,
                                                   access_mode::readwrite);
            ArrayHandle<unsigned int> d_group_tag(m_gdata->getTags(),
                                                  access_location::device,
                                                  access_mode::readwrite);
            ArrayHandle<typename group_data::ranks_t> d_group_ranks(m_gdata->getRanksArray(),
                                                                    access_location::device,
                                                                    access_mode::readwrite);
            ArrayHandle<unsigned int> d_group_rtag(m_gdata->getRTags(),
                                                   access_location::device,
                                                   access_mode::readwrite);

            // get temp buffer
            ScopedAllocation<unsigned int> d_marked_groups(m_exec_conf->getCachedAllocator(),
                                                           n_recv_unique);
            ScopedAllocation<unsigned int> d_tmp(m_exec_conf->getCachedAllocator(), n_recv_unique);

            // add new groups, updating groups that are already present locally
            gpu_add_groups(old_ngroups,
                           n_recv_unique,
                           d_groups_in.data,
                           d_groups.data,
                           d_group_typeval.data,
                           d_group_tag.data,
                           d_group_ranks.data,
                           d_group_rtag.data,
                           new_ngroups,
                           d_marked_groups.data,
                           d_tmp.data,
                           local_multiple,
                           m_exec_conf->getRank(),
                           m_exec_conf->getCachedAllocator());
            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

        // remove duplicate groups
        m_gdata->removeGroups(old_ngroups + n_recv_unique - new_ngroups);
        }
    }

template<class group_data>
void CommunicatorGPU::GroupCommunicatorGPU<group_data>::exchangeGhostGroups(
    const GlobalVector<unsigned int>& plans)
    {
    if (m_gdata->getNGlobal())
        {
        m_exec_conf->msg->notice(7)
            << "CommunicatorGPU: ghost " << group_data::getName() << " exchange " << std::endl;

        std::vector<std::vector<unsigned int>> n_send_ghost_groups;
        std::vector<std::vector<unsigned int>> n_recv_ghost_groups;
        std::vector<std::vector<unsigned int>> ghost_group_offs;

        std::vector<unsigned int> n_send_ghost_groups_tot;
        std::vector<unsigned int> n_recv_ghost_groups_tot;

        // resize arrays
        n_send_ghost_groups.resize(m_gpu_comm.m_num_stages);
        n_recv_ghost_groups.resize(m_gpu_comm.m_num_stages);

        for (unsigned int istage = 0; istage < m_gpu_comm.m_num_stages; ++istage)
            {
            n_send_ghost_groups[istage].resize(m_gpu_comm.m_n_unique_neigh);
            n_recv_ghost_groups[istage].resize(m_gpu_comm.m_n_unique_neigh);
            }

        n_send_ghost_groups_tot.resize(m_gpu_comm.m_num_stages);
        n_recv_ghost_groups_tot.resize(m_gpu_comm.m_num_stages);
        ghost_group_offs.resize(m_gpu_comm.m_num_stages);
        for (unsigned int istage = 0; istage < m_gpu_comm.m_num_stages; ++istage)
            ghost_group_offs[istage].resize(m_gpu_comm.m_n_unique_neigh);

        m_ghost_group_begin.resize(m_gpu_comm.m_n_unique_neigh * m_gpu_comm.m_num_stages);
        m_ghost_group_end.resize(m_gpu_comm.m_n_unique_neigh * m_gpu_comm.m_num_stages);

        std::vector<unsigned int> idx_offs;
        idx_offs.resize(m_gpu_comm.m_num_stages);

        // make room for plans
        m_ghost_group_plan.resize(m_gdata->getN());

            {
            // compute plans for all local groups
            ArrayHandle<typename group_data::members_t> d_groups(m_gdata->getMembersArray(),
                                                                 access_location::device,
                                                                 access_mode::read);
            ArrayHandle<unsigned int> d_ghost_group_plan(m_ghost_group_plan,
                                                         access_location::device,
                                                         access_mode::overwrite);
            ArrayHandle<unsigned int> d_rtag(m_gpu_comm.m_pdata->getRTags(),
                                             access_location::device,
                                             access_mode::read);
            ArrayHandle<unsigned int> d_plans(plans, access_location::device, access_mode::read);

            gpu_make_ghost_group_exchange_plan<group_data::size>(d_ghost_group_plan.data,
                                                                 d_groups.data,
                                                                 m_gdata->getN(),
                                                                 d_rtag.data,
                                                                 d_plans.data,
                                                                 m_gpu_comm.m_pdata->getN());

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

        // main communication loop
        for (unsigned int stage = 0; stage < m_gpu_comm.m_num_stages; stage++)
            {
            // resize temporary number of neighbors array
            m_neigh_counts.resize(m_gdata->getN() + m_gdata->getNGhosts());
            m_ghost_scan.resize(m_gdata->getN() + m_gdata->getNGhosts());

                {
                ArrayHandle<unsigned int> d_ghost_group_plan(m_ghost_group_plan,
                                                             access_location::device,
                                                             access_mode::read);
                ArrayHandle<unsigned int> d_adj_mask(m_gpu_comm.m_adj_mask,
                                                     access_location::device,
                                                     access_mode::read);
                ArrayHandle<unsigned int> d_neigh_counts(m_neigh_counts,
                                                         access_location::device,
                                                         access_mode::overwrite);
                ArrayHandle<unsigned int> d_ghost_scan(m_ghost_scan,
                                                       access_location::device,
                                                       access_mode::overwrite);

                // count number of neighbors (total and per particle) the ghost ptls are sent to
                n_send_ghost_groups_tot[stage]
                    = gpu_exchange_ghosts_count_neighbors(m_gdata->getN() + m_gdata->getNGhosts(),
                                                          d_ghost_group_plan.data,
                                                          d_adj_mask.data,
                                                          d_neigh_counts.data,
                                                          d_ghost_scan.data,
                                                          m_gpu_comm.m_n_unique_neigh,
                                                          m_exec_conf->getCachedAllocator());

                if (m_exec_conf->isCUDAErrorCheckingEnabled())
                    CHECK_CUDA_ERROR();
                }

            // compute offset into ghost idx list
            idx_offs[stage] = 0;
            for (unsigned int i = 0; i < stage; ++i)
                idx_offs[stage] += n_send_ghost_groups_tot[i];

            // compute maximum send buf size
            unsigned int n_max = 0;
            for (unsigned int istage = 0; istage <= stage; ++istage)
                if (n_send_ghost_groups_tot[istage] > n_max)
                    n_max = n_send_ghost_groups_tot[istage];

            // make room for ghost indices and neighbor ranks
            m_ghost_group_idx_adj.resize(idx_offs[stage] + n_send_ghost_groups_tot[stage]);
            m_ghost_group_neigh.resize(idx_offs[stage] + n_send_ghost_groups_tot[stage]);

            // resize send buffer
            m_groups_sendbuf.resize(n_max);

                {
                ArrayHandle<unsigned int> d_ghost_group_plan(m_ghost_group_plan,
                                                             access_location::device,
                                                             access_mode::read);
                ArrayHandle<unsigned int> d_adj_mask(m_gpu_comm.m_adj_mask,
                                                     access_location::device,
                                                     access_mode::read);
                ArrayHandle<unsigned int> d_neigh_counts(m_neigh_counts,
                                                         access_location::device,
                                                         access_mode::read);
                ArrayHandle<unsigned int> d_ghost_scan(m_ghost_scan,
                                                       access_location::device,
                                                       access_mode::read);
                ArrayHandle<unsigned int> d_unique_neighbors(m_gpu_comm.m_unique_neighbors,
                                                             access_location::device,
                                                             access_mode::read);

                ArrayHandle<unsigned int> d_group_tag(m_gdata->getTags(),
                                                      access_location::device,
                                                      access_mode::read);

                ArrayHandle<uint2> d_ghost_group_idx_adj(m_ghost_group_idx_adj,
                                                         access_location::device,
                                                         access_mode::overwrite);
                ArrayHandle<unsigned int> d_ghost_group_neigh(m_ghost_group_neigh,
                                                              access_location::device,
                                                              access_mode::overwrite);
                ArrayHandle<unsigned int> d_ghost_group_begin(m_ghost_group_begin,
                                                              access_location::device,
                                                              access_mode::overwrite);
                ArrayHandle<unsigned int> d_ghost_group_end(m_ghost_group_end,
                                                            access_location::device,
                                                            access_mode::overwrite);

                //! Fill ghost send list and compute start and end indices per unique neighbor in
                //! list
                gpu_exchange_ghosts_make_indices(
                    m_gdata->getN() + m_gdata->getNGhosts(),
                    d_ghost_group_plan.data,
                    d_group_tag.data,
                    d_adj_mask.data,
                    d_unique_neighbors.data,
                    d_neigh_counts.data,
                    d_ghost_scan.data,
                    d_ghost_group_idx_adj.data + idx_offs[stage],
                    d_ghost_group_neigh.data + idx_offs[stage],
                    d_ghost_group_begin.data + stage * m_gpu_comm.m_n_unique_neigh,
                    d_ghost_group_end.data + stage * m_gpu_comm.m_n_unique_neigh,
                    m_gpu_comm.m_n_unique_neigh,
                    n_send_ghost_groups_tot[stage],
                    m_gpu_comm.m_comm_mask[stage],
                    m_exec_conf->getCachedAllocator());

                if (m_exec_conf->isCUDAErrorCheckingEnabled())
                    CHECK_CUDA_ERROR();
                }

                {
                // access group data
                ArrayHandle<unsigned int> d_group_tag(m_gdata->getTags(),
                                                      access_location::device,
                                                      access_mode::read);
                ArrayHandle<typename group_data::members_t> d_groups(m_gdata->getMembersArray(),
                                                                     access_location::device,
                                                                     access_mode::read);
                ArrayHandle<typeval_t> d_group_typeval(m_gdata->getTypeValArray(),
                                                       access_location::device,
                                                       access_mode::read);
                ArrayHandle<typename group_data::ranks_t> d_group_ranks(m_gdata->getRanksArray(),
                                                                        access_location::device,
                                                                        access_mode::read);

                // access ghost send indices
                ArrayHandle<uint2> d_ghost_group_idx_adj(m_ghost_group_idx_adj,
                                                         access_location::device,
                                                         access_mode::read);

                // access output buffers
                ArrayHandle<group_element_t> d_groups_sendbuf(m_groups_sendbuf,
                                                              access_location::device,
                                                              access_mode::overwrite);

                // Pack ghosts into send buffers
                gpu_exchange_ghost_groups_pack(n_send_ghost_groups_tot[stage],
                                               d_ghost_group_idx_adj.data + idx_offs[stage],
                                               d_group_tag.data,
                                               d_groups.data,
                                               d_group_typeval.data,
                                               d_group_ranks.data,
                                               d_groups_sendbuf.data);

                if (m_exec_conf->isCUDAErrorCheckingEnabled())
                    CHECK_CUDA_ERROR();
                }

            /*
             * Ghost particle communication
             */
            n_recv_ghost_groups_tot[stage] = 0;

            unsigned int send_bytes = 0;
            unsigned int recv_bytes = 0;

                {
                ArrayHandle<unsigned int> h_ghost_group_begin(m_ghost_group_begin,
                                                              access_location::host,
                                                              access_mode::read);
                ArrayHandle<unsigned int> h_ghost_group_end(m_ghost_group_end,
                                                            access_location::host,
                                                            access_mode::read);
                ArrayHandle<unsigned int> h_unique_neighbors(m_gpu_comm.m_unique_neighbors,
                                                             access_location::host,
                                                             access_mode::read);

                // compute send counts
                for (unsigned int ineigh = 0; ineigh < m_gpu_comm.m_n_unique_neigh; ineigh++)
                    n_send_ghost_groups[stage][ineigh]
                        = h_ghost_group_end.data[ineigh + stage * m_gpu_comm.m_n_unique_neigh]
                          - h_ghost_group_begin.data[ineigh + stage * m_gpu_comm.m_n_unique_neigh];

                MPI_Request req[2 * m_gpu_comm.m_n_unique_neigh];
                MPI_Status stat[2 * m_gpu_comm.m_n_unique_neigh];

                unsigned int nreq = 0;

                // loop over neighbors
                for (unsigned int ineigh = 0; ineigh < m_gpu_comm.m_n_unique_neigh; ineigh++)
                    {
                    if (m_gpu_comm.m_stages[ineigh] != (int)stage)
                        {
                        // skip neighbor if not participating in this communication stage
                        n_send_ghost_groups[stage][ineigh] = 0;
                        n_recv_ghost_groups[stage][ineigh] = 0;
                        continue;
                        }

                    // rank of neighbor processor
                    unsigned int neighbor = h_unique_neighbors.data[ineigh];

                    MPI_Isend(&n_send_ghost_groups[stage][ineigh],
                              1,
                              MPI_UNSIGNED,
                              neighbor,
                              0,
                              m_gpu_comm.m_mpi_comm,
                              &req[nreq++]);
                    MPI_Irecv(&n_recv_ghost_groups[stage][ineigh],
                              1,
                              MPI_UNSIGNED,
                              neighbor,
                              0,
                              m_gpu_comm.m_mpi_comm,
                              &req[nreq++]);

                    send_bytes += (unsigned int)sizeof(unsigned int);
                    recv_bytes += (unsigned int)sizeof(unsigned int);
                    }

                MPI_Waitall(nreq, req, stat);

                // total up receive counts
                for (unsigned int ineigh = 0; ineigh < m_gpu_comm.m_n_unique_neigh; ineigh++)
                    {
                    if (ineigh == 0)
                        ghost_group_offs[stage][ineigh] = 0;
                    else
                        ghost_group_offs[stage][ineigh] = ghost_group_offs[stage][ineigh - 1]
                                                          + n_recv_ghost_groups[stage][ineigh - 1];

                    n_recv_ghost_groups_tot[stage] += n_recv_ghost_groups[stage][ineigh];
                    }
                }

            n_max = 0;
            // compute maximum number of received ghosts
            for (unsigned int istage = 0; istage <= stage; ++istage)
                if (n_recv_ghost_groups_tot[istage] > n_max)
                    n_max = n_recv_ghost_groups_tot[istage];

            m_groups_recvbuf.resize(n_max);

            // first ghost group index
            unsigned int first_idx = m_gdata->getN() + m_gdata->getNGhosts();

                {
                unsigned int offs = 0;
                // recv buffer
                ArrayHandle<group_element_t> h_groups_recvbuf(m_groups_recvbuf,
                                                              access_location::host,
                                                              access_mode::overwrite);

                // send buffers
                ArrayHandle<group_element_t> h_groups_sendbuf(m_groups_sendbuf,
                                                              access_location::host,
                                                              access_mode::read);

                ArrayHandle<unsigned int> h_unique_neighbors(m_gpu_comm.m_unique_neighbors,
                                                             access_location::host,
                                                             access_mode::read);
                ArrayHandle<unsigned int> h_ghost_group_begin(m_ghost_group_begin,
                                                              access_location::host,
                                                              access_mode::read);
                ArrayHandle<unsigned int> h_ghost_group_end(m_ghost_group_end,
                                                            access_location::host,
                                                            access_mode::read);

                std::vector<MPI_Request> reqs;
                MPI_Request req;

                // loop over neighbors
                for (unsigned int ineigh = 0; ineigh < m_gpu_comm.m_n_unique_neigh; ineigh++)
                    {
                    // rank of neighbor processor
                    unsigned int neighbor = h_unique_neighbors.data[ineigh];

                    // when sending/receiving 0 groups, the send/recv buffer may be uninitialized
                    if (n_send_ghost_groups[stage][ineigh])
                        {
                        MPI_Isend(h_groups_sendbuf.data
                                      + h_ghost_group_begin
                                            .data[ineigh + stage * m_gpu_comm.m_n_unique_neigh],
                                  int(n_send_ghost_groups[stage][ineigh] * sizeof(group_element_t)),
                                  MPI_BYTE,
                                  neighbor,
                                  1,
                                  m_gpu_comm.m_mpi_comm,
                                  &req);
                        reqs.push_back(req);
                        }
                    send_bytes += (unsigned int)(n_send_ghost_groups[stage][ineigh]
                                                 * sizeof(group_element_t));
                    if (n_recv_ghost_groups[stage][ineigh])
                        {
                        MPI_Irecv(h_groups_recvbuf.data + ghost_group_offs[stage][ineigh] + offs,
                                  int(n_recv_ghost_groups[stage][ineigh] * sizeof(group_element_t)),
                                  MPI_BYTE,
                                  neighbor,
                                  1,
                                  m_gpu_comm.m_mpi_comm,
                                  &req);
                        reqs.push_back(req);
                        }
                    recv_bytes += (unsigned int)(n_recv_ghost_groups[stage][ineigh]
                                                 * sizeof(group_element_t));
                    }

                std::vector<MPI_Status> stats(reqs.size());
                MPI_Waitall((unsigned int)reqs.size(), &reqs.front(), &stats.front());
                } // end ArrayHandle scope

            unsigned int old_n_ghost = m_gdata->getNGhosts();

            // update number of ghost particles
            m_gdata->addGhostGroups(n_recv_ghost_groups_tot[stage]);

            unsigned int n_keep = 0;
                {
                // access receive buffers
                ArrayHandle<group_element_t> d_groups_recvbuf(m_groups_recvbuf,
                                                              access_location::device,
                                                              access_mode::read);
                // access group data
                ArrayHandle<unsigned int> d_group_tag(m_gdata->getTags(),
                                                      access_location::device,
                                                      access_mode::readwrite);
                ArrayHandle<typename group_data::members_t> d_groups(m_gdata->getMembersArray(),
                                                                     access_location::device,
                                                                     access_mode::readwrite);
                ArrayHandle<typeval_t> d_group_typeval(m_gdata->getTypeValArray(),
                                                       access_location::device,
                                                       access_mode::readwrite);
                ArrayHandle<typename group_data::ranks_t> d_group_ranks(m_gdata->getRanksArray(),
                                                                        access_location::device,
                                                                        access_mode::readwrite);
                ArrayHandle<unsigned int> d_group_rtag(m_gdata->getRTags(),
                                                       access_location::device,
                                                       access_mode::read);
                ArrayHandle<unsigned int> d_rtag(m_gpu_comm.m_pdata->getRTags(),
                                                 access_location::device,
                                                 access_mode::read);

                CachedAllocator& alloc = m_exec_conf->getCachedAllocator();
                ScopedAllocation<unsigned int> d_keep(alloc, n_recv_ghost_groups_tot[stage]);
                ScopedAllocation<unsigned int> d_scan(alloc, n_recv_ghost_groups_tot[stage]);

                // copy recv buf into group data, omitting duplicates and groups with nonlocal ptls
                gpu_exchange_ghost_groups_copy_buf<group_data::size>(
                    n_recv_ghost_groups_tot[stage],
                    d_groups_recvbuf.data,
                    d_group_tag.data + first_idx,
                    d_groups.data + first_idx,
                    d_group_typeval.data + first_idx,
                    d_group_ranks.data + first_idx,
                    d_keep.data,
                    d_scan.data,
                    d_group_rtag.data,
                    d_rtag.data,
                    m_gpu_comm.m_pdata->getN() + m_gpu_comm.m_pdata->getNGhosts(),
                    n_keep,
                    m_exec_conf->getCachedAllocator());

                if (m_exec_conf->isCUDAErrorCheckingEnabled())
                    CHECK_CUDA_ERROR();
                }

            // update ghost group number
            m_gdata->removeAllGhostGroups();
            m_gdata->addGhostGroups(old_n_ghost + n_keep);

                {
                ArrayHandle<unsigned int> d_group_tag(m_gdata->getTags(),
                                                      access_location::device,
                                                      access_mode::read);
                ArrayHandle<unsigned int> d_group_rtag(m_gdata->getRTags(),
                                                       access_location::device,
                                                       access_mode::readwrite);

                // update reverse-lookup table
                gpu_compute_ghost_rtags(first_idx,
                                        n_keep,
                                        d_group_tag.data + first_idx,
                                        d_group_rtag.data);
                if (m_exec_conf->isCUDAErrorCheckingEnabled())
                    CHECK_CUDA_ERROR();
                }
            } // end main communication loop

        // notify subscribers that group order has changed
        m_gdata->notifyGroupReorder();
        }
    }

//! Mark ghost particles
template<class group_data>
void CommunicatorGPU::GroupCommunicatorGPU<group_data>::markGhostParticles(
    const GlobalVector<unsigned int>& plans,
    unsigned int mask)
    {
    if (m_gdata->getNGlobal())
        {
        m_exec_conf->msg->notice(7) << "GroupCommunicator<" << m_gdata->getName()
                                    << ">: find incomplete groups" << std::endl;

        ArrayHandle<typename group_data::members_t> d_groups(m_gdata->getMembersArray(),
                                                             access_location::device,
                                                             access_mode::read);
        ArrayHandle<typename group_data::ranks_t> d_group_ranks(m_gdata->getRanksArray(),
                                                                access_location::device,
                                                                access_mode::read);
        ArrayHandle<unsigned int> d_rtag(m_gpu_comm.m_pdata->getRTags(),
                                         access_location::device,
                                         access_mode::read);
        ArrayHandle<Scalar4> d_pos(m_gpu_comm.m_pdata->getPositions(),
                                   access_location::device,
                                   access_mode::read);
        ArrayHandle<unsigned int> d_plan(plans, access_location::device, access_mode::readwrite);

        std::shared_ptr<DomainDecomposition> decomposition
            = m_gpu_comm.m_pdata->getDomainDecomposition();
        ArrayHandle<unsigned int> d_cart_ranks_inv(decomposition->getInverseCartRanks(),
                                                   access_location::device,
                                                   access_mode::read);
        Index3D di = decomposition->getDomainIndexer();
        uint3 my_pos = decomposition->getGridPos();

        gpu_mark_bonded_ghosts<group_data::size>(m_gdata->getN(),
                                                 d_groups.data,
                                                 d_group_ranks.data,
                                                 d_pos.data,
                                                 m_gpu_comm.m_pdata->getBox(),
                                                 d_rtag.data,
                                                 d_plan.data,
                                                 di,
                                                 my_pos,
                                                 d_cart_ranks_inv.data,
                                                 m_exec_conf->getRank(),
                                                 mask);
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }
    }

//! Transfer particles between neighboring domains
void CommunicatorGPU::migrateParticles()
    {
    m_exec_conf->msg->notice(7) << "CommunicatorGPU: migrate particles" << std::endl;

    updateGhostWidth();

    // check if simulation box is sufficiently large for domain decomposition
    checkBoxSize();

    // remove ghost particles from system
    m_pdata->removeAllGhostParticles();

    // main communication loop
    for (unsigned int stage = 0; stage < m_num_stages; stage++)
        {
            {
            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(),
                                       access_location::device,
                                       access_mode::read);
            ArrayHandle<unsigned int> d_comm_flag(m_pdata->getCommFlags(),
                                                  access_location::device,
                                                  access_mode::readwrite);

            assert(stage < m_comm_mask.size());

            // mark all particles which have left the box for sending (rtag=NOT_LOCAL)
            gpu_stage_particles(m_pdata->getN(),
                                d_pos.data,
                                d_comm_flag.data,
                                m_pdata->getBox(),
                                m_comm_mask[stage]);

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

        /*
         * Bonded group communication, determine groups to be sent
         */
        // Bonds
        m_bond_comm.migrateGroups(m_bonds_changed, true);
        m_bonds_changed = false;

        // Special pairs
        m_pair_comm.migrateGroups(m_pairs_changed, true);
        m_pairs_changed = false;

        // Angles
        m_angle_comm.migrateGroups(m_angles_changed, true);
        m_angles_changed = false;

        // Dihedrals
        m_dihedral_comm.migrateGroups(m_dihedrals_changed, true);
        m_dihedrals_changed = false;

        // Impropers
        m_improper_comm.migrateGroups(m_impropers_changed, true);
        m_impropers_changed = false;

        // Constraints
        m_constraint_comm.migrateGroups(m_constraints_changed, true);
        m_constraints_changed = false;

        // MeshBonds
        if (m_meshdef)
            {
            m_meshbond_comm.migrateGroups(m_meshbonds_changed, true);
            m_meshbonds_changed = false;

            m_meshtriangle_comm.migrateGroups(m_meshtriangles_changed, true);
            m_meshtriangles_changed = false;
            }

        // fill send buffer
        m_pdata->removeParticlesGPU(m_gpu_sendbuf, m_comm_flags);

        const Index3D& di = m_decomposition->getDomainIndexer();
        // determine local particles that are to be sent to neighboring processors and fill send
        // buffer
        uint3 mypos = m_decomposition->getGridPos();

            {
            // resize keys
            m_send_keys.resize(m_gpu_sendbuf.size());

            ArrayHandle<detail::pdata_element> d_gpu_sendbuf(m_gpu_sendbuf,
                                                             access_location::device,
                                                             access_mode::readwrite);
            ArrayHandle<unsigned int> d_send_keys(m_send_keys,
                                                  access_location::device,
                                                  access_mode::overwrite);
            ArrayHandle<unsigned int> d_begin(m_begin,
                                              access_location::device,
                                              access_mode::overwrite);
            ArrayHandle<unsigned int> d_end(m_end, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_unique_neighbors(m_unique_neighbors,
                                                         access_location::device,
                                                         access_mode::read);
            ArrayHandle<unsigned int> d_comm_flags(m_comm_flags,
                                                   access_location::device,
                                                   access_mode::read);

            ArrayHandle<unsigned int> d_cart_ranks(m_decomposition->getCartRanks(),
                                                   access_location::device,
                                                   access_mode::read);

            // get temporary buffers
            size_t nsend = m_gpu_sendbuf.size();
            CachedAllocator& alloc = m_exec_conf->getCachedAllocator();
            ScopedAllocation<detail::pdata_element> d_in_copy(alloc, nsend);
            ScopedAllocation<unsigned int> d_tmp(alloc, nsend);

            gpu_sort_migrating_particles(m_gpu_sendbuf.size(),
                                         d_gpu_sendbuf.data,
                                         d_comm_flags.data,
                                         di,
                                         mypos,
                                         d_cart_ranks.data,
                                         d_send_keys.data,
                                         d_begin.data,
                                         d_end.data,
                                         d_unique_neighbors.data,
                                         m_n_unique_neigh,
                                         m_comm_mask[stage],
                                         d_tmp.data,
                                         d_in_copy.data,
                                         m_exec_conf->getCachedAllocator());

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

        unsigned int n_send_ptls[m_n_unique_neigh];
        unsigned int n_recv_ptls[m_n_unique_neigh];
        unsigned int offs[m_n_unique_neigh];
        unsigned int n_recv_tot = 0;

            {
            ArrayHandle<unsigned int> h_begin(m_begin, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_end(m_end, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_unique_neighbors(m_unique_neighbors,
                                                         access_location::host,
                                                         access_mode::read);

            unsigned int send_bytes = 0;
            unsigned int recv_bytes = 0;

            // compute send counts
            for (unsigned int ineigh = 0; ineigh < m_n_unique_neigh; ineigh++)
                n_send_ptls[ineigh] = h_end.data[ineigh] - h_begin.data[ineigh];

            MPI_Request req[2 * m_n_unique_neigh];
            MPI_Status stat[2 * m_n_unique_neigh];

            unsigned int nreq = 0;

            // loop over neighbors
            for (unsigned int ineigh = 0; ineigh < m_n_unique_neigh; ineigh++)
                {
                if (m_stages[ineigh] != (int)stage)
                    {
                    // skip neighbor if not participating in this communication stage
                    n_send_ptls[ineigh] = 0;
                    n_recv_ptls[ineigh] = 0;
                    continue;
                    }

                // rank of neighbor processor
                unsigned int neighbor = h_unique_neighbors.data[ineigh];

                MPI_Isend(&n_send_ptls[ineigh],
                          1,
                          MPI_UNSIGNED,
                          neighbor,
                          0,
                          m_mpi_comm,
                          &req[nreq++]);
                MPI_Irecv(&n_recv_ptls[ineigh],
                          1,
                          MPI_UNSIGNED,
                          neighbor,
                          0,
                          m_mpi_comm,
                          &req[nreq++]);
                send_bytes += (unsigned int)sizeof(unsigned int);
                recv_bytes += (unsigned int)sizeof(unsigned int);
                } // end neighbor loop

            MPI_Waitall(nreq, req, stat);

            // sum up receive counts
            for (unsigned int ineigh = 0; ineigh < m_n_unique_neigh; ineigh++)
                {
                if (ineigh == 0)
                    offs[ineigh] = 0;
                else
                    offs[ineigh] = offs[ineigh - 1] + n_recv_ptls[ineigh - 1];

                n_recv_tot += n_recv_ptls[ineigh];
                }
            }

        // Resize receive buffer
        m_gpu_recvbuf.resize(n_recv_tot);

            {
            ArrayHandle<unsigned int> h_begin(m_begin, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_end(m_end, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_unique_neighbors(m_unique_neighbors,
                                                         access_location::host,
                                                         access_mode::read);
            ArrayHandle<detail::pdata_element> gpu_sendbuf_handle(m_gpu_sendbuf,
                                                                  access_location::host,
                                                                  access_mode::read);
            ArrayHandle<detail::pdata_element> gpu_recvbuf_handle(m_gpu_recvbuf,
                                                                  access_location::host,
                                                                  access_mode::overwrite);
            std::vector<MPI_Request> reqs;
            MPI_Request req;

            unsigned int send_bytes = 0;
            unsigned int recv_bytes = 0;

            // loop over neighbors
            for (unsigned int ineigh = 0; ineigh < m_n_unique_neigh; ineigh++)
                {
                // rank of neighbor processor
                unsigned int neighbor = h_unique_neighbors.data[ineigh];

                // exchange particle data
                if (n_send_ptls[ineigh])
                    {
                    MPI_Isend(gpu_sendbuf_handle.data + h_begin.data[ineigh],
                              n_send_ptls[ineigh],
                              m_mpi_pdata_element,
                              neighbor,
                              1,
                              m_mpi_comm,
                              &req);
                    reqs.push_back(req);
                    }
                send_bytes += (unsigned int)(n_send_ptls[ineigh] * sizeof(detail::pdata_element));

                if (n_recv_ptls[ineigh])
                    {
                    MPI_Irecv(gpu_recvbuf_handle.data + offs[ineigh],
                              n_recv_ptls[ineigh],
                              m_mpi_pdata_element,
                              neighbor,
                              1,
                              m_mpi_comm,
                              &req);
                    reqs.push_back(req);
                    }
                recv_bytes += (unsigned int)(n_recv_ptls[ineigh] * sizeof(detail::pdata_element));
                }

            std::vector<MPI_Status> stats(reqs.size());
            MPI_Waitall((unsigned int)(reqs.size()), &reqs.front(), &stats.front());
            }

            {
            ArrayHandle<detail::pdata_element> d_gpu_recvbuf(m_gpu_recvbuf,
                                                             access_location::device,
                                                             access_mode::readwrite);
            const BoxDim shifted_box = getShiftedBox();

            // Apply boundary conditions
            gpu_wrap_particles(n_recv_tot, d_gpu_recvbuf.data, shifted_box);
            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

        // remove particles that were sent and fill particle data with received particles
        m_pdata->addParticlesGPU(m_gpu_recvbuf);

        } // end communication stage
    }

void CommunicatorGPU::removeGhostParticleTags()
    {
    if (m_last_flags[comm_flag::tag])
        {
        m_exec_conf->msg->notice(9)
            << "CommunicatorGPU: removing " << m_ghosts_added << " ghost particles " << std::endl;

        // Reset reverse lookup tags of old ghost atoms
        ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(),
                                         access_location::device,
                                         access_mode::readwrite);
        ArrayHandle<unsigned int> d_tag(m_pdata->getTags(),
                                        access_location::device,
                                        access_mode::read);

        gpu_reset_rtags(m_ghosts_added, d_tag.data + m_pdata->getN(), d_rtag.data);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    m_ghosts_added = 0;
    }

//! Build a ghost particle list, exchange ghost particle data with neighboring processors
void CommunicatorGPU::exchangeGhosts()
    {
    CommFlags current_flags = getFlags();
    if (current_flags[comm_flag::reverse_net_force] && this->m_exec_conf->isCUDAEnabled())
        {
        throw std::runtime_error(
            "Communication error: Reverse force communication is not enabled on the GPU.");
        }

    // check if simulation box is sufficiently large for domain decomposition
    checkBoxSize();

    m_exec_conf->msg->notice(7) << "CommunicatorGPU: ghost exchange" << std::endl;

    // update the subscribed ghost layer width
    updateGhostWidth();

    // resize arrays
    m_n_send_ghosts.resize(m_num_stages);
    m_n_recv_ghosts.resize(m_num_stages);

    for (unsigned int istage = 0; istage < m_num_stages; ++istage)
        {
        m_n_send_ghosts[istage].resize(m_n_unique_neigh);
        m_n_recv_ghosts[istage].resize(m_n_unique_neigh);
        }

    m_n_send_ghosts_tot.resize(m_num_stages);
    m_n_recv_ghosts_tot.resize(m_num_stages);
    m_ghost_offs.resize(m_num_stages);
    for (unsigned int istage = 0; istage < m_num_stages; ++istage)
        m_ghost_offs[istage].resize(m_n_unique_neigh);

    m_ghost_begin.resize(m_n_unique_neigh * m_num_stages);
    m_ghost_end.resize(m_n_unique_neigh * m_num_stages);

    m_idx_offs.resize(m_num_stages);

    // get requested ghost fields
    CommFlags flags = getFlags();

    // main communication loop
    for (unsigned int stage = 0; stage < m_num_stages; stage++)
        {
        // make room for plans
        m_ghost_plan.resize(m_pdata->getN() + m_pdata->getNGhosts());

            {
            // compute plans for all particles, including already received ghosts
            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(),
                                       access_location::device,
                                       access_mode::read);
            ArrayHandle<unsigned int> d_body(m_pdata->getBodies(),
                                             access_location::device,
                                             access_mode::read);
            ArrayHandle<unsigned int> d_ghost_plan(m_ghost_plan,
                                                   access_location::device,
                                                   access_mode::overwrite);

            ArrayHandle<Scalar> d_r_ghost(m_r_ghost, access_location::device, access_mode::read);
            ArrayHandle<Scalar> d_r_ghost_body(m_r_ghost_body,
                                               access_location::device,
                                               access_mode::read);

            gpu_make_ghost_exchange_plan(d_ghost_plan.data,
                                         m_pdata->getN() + m_pdata->getNGhosts(),
                                         d_pos.data,
                                         d_body.data,
                                         m_pdata->getBox(),
                                         d_r_ghost.data,
                                         d_r_ghost_body.data,
                                         m_r_ghost_max,
                                         m_pdata->getNTypes(),
                                         m_comm_mask[stage]);

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

        // mark particles that are members of incomplete of bonded groups as ghost

        // bonds
        m_bond_comm.markGhostParticles(m_ghost_plan, m_comm_mask[stage]);

        // special pairs
        m_pair_comm.markGhostParticles(m_ghost_plan, m_comm_mask[stage]);

        // angles
        m_angle_comm.markGhostParticles(m_ghost_plan, m_comm_mask[stage]);

        // dihedrals
        m_dihedral_comm.markGhostParticles(m_ghost_plan, m_comm_mask[stage]);

        // impropers
        m_improper_comm.markGhostParticles(m_ghost_plan, m_comm_mask[stage]);

        // constraints
        m_constraint_comm.markGhostParticles(m_ghost_plan, m_comm_mask[stage]);

        if (m_meshdef)
            {
            m_meshbond_comm.markGhostParticles(m_ghost_plan, m_comm_mask[stage]);

            m_meshtriangle_comm.markGhostParticles(m_ghost_plan, m_comm_mask[stage]);
            }

        // resize temporary number of neighbors array
        m_neigh_counts.resize(m_pdata->getN() + m_pdata->getNGhosts());
        m_scan.resize(m_pdata->getN() + m_pdata->getNGhosts());

            {
            ArrayHandle<unsigned int> d_ghost_plan(m_ghost_plan,
                                                   access_location::device,
                                                   access_mode::read);
            ArrayHandle<unsigned int> d_adj_mask(m_adj_mask,
                                                 access_location::device,
                                                 access_mode::read);
            ArrayHandle<unsigned int> d_neigh_counts(m_neigh_counts,
                                                     access_location::device,
                                                     access_mode::overwrite);
            ArrayHandle<unsigned int> d_scan(m_scan,
                                             access_location::device,
                                             access_mode::overwrite);

            // count number of neighbors (total and per particle) the ghost ptls are sent to
            m_n_send_ghosts_tot[stage]
                = gpu_exchange_ghosts_count_neighbors(m_pdata->getN() + m_pdata->getNGhosts(),
                                                      d_ghost_plan.data,
                                                      d_adj_mask.data,
                                                      d_neigh_counts.data,
                                                      d_scan.data,
                                                      m_n_unique_neigh,
                                                      m_exec_conf->getCachedAllocator());

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

        // compute offset into ghost idx list
        m_idx_offs[stage] = 0;
        for (unsigned int i = 0; i < stage; ++i)
            m_idx_offs[stage] += m_n_send_ghosts_tot[i];

        // compute maximum send buf size
        unsigned int n_max = 0;
        for (unsigned int istage = 0; istage <= stage; ++istage)
            if (m_n_send_ghosts_tot[istage] > n_max)
                n_max = m_n_send_ghosts_tot[istage];

        // make room for ghost indices and neighbor ranks
        m_ghost_idx_adj.resize(m_idx_offs[stage] + m_n_send_ghosts_tot[stage]);
        m_ghost_neigh.resize(m_idx_offs[stage] + m_n_send_ghosts_tot[stage]);

        if (flags[comm_flag::tag])
            m_tag_ghost_sendbuf.resize(n_max);
        if (flags[comm_flag::position])
            m_pos_ghost_sendbuf.resize(n_max);
        if (flags[comm_flag::velocity])
            m_vel_ghost_sendbuf.resize(n_max);
        if (flags[comm_flag::charge])
            m_charge_ghost_sendbuf.resize(n_max);
        if (flags[comm_flag::body])
            m_body_ghost_sendbuf.resize(n_max);
        if (flags[comm_flag::image])
            m_image_ghost_sendbuf.resize(n_max);
        if (flags[comm_flag::diameter])
            m_diameter_ghost_sendbuf.resize(n_max);
        if (flags[comm_flag::orientation])
            {
            m_orientation_ghost_sendbuf.resize(n_max);
            }

            {
            ArrayHandle<unsigned int> d_ghost_plan(m_ghost_plan,
                                                   access_location::device,
                                                   access_mode::read);
            ArrayHandle<unsigned int> d_adj_mask(m_adj_mask,
                                                 access_location::device,
                                                 access_mode::read);
            ArrayHandle<unsigned int> d_neigh_counts(m_neigh_counts,
                                                     access_location::device,
                                                     access_mode::read);
            ArrayHandle<unsigned int> d_scan(m_scan, access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_unique_neighbors(m_unique_neighbors,
                                                         access_location::device,
                                                         access_mode::read);

            ArrayHandle<unsigned int> d_tag(m_pdata->getTags(),
                                            access_location::device,
                                            access_mode::read);

            ArrayHandle<uint2> d_ghost_idx_adj(m_ghost_idx_adj,
                                               access_location::device,
                                               access_mode::overwrite);
            ArrayHandle<unsigned int> d_ghost_neigh(m_ghost_neigh,
                                                    access_location::device,
                                                    access_mode::overwrite);
            ArrayHandle<unsigned int> d_ghost_begin(m_ghost_begin,
                                                    access_location::device,
                                                    access_mode::overwrite);
            ArrayHandle<unsigned int> d_ghost_end(m_ghost_end,
                                                  access_location::device,
                                                  access_mode::overwrite);

            //! Fill ghost send list and compute start and end indices per unique neighbor in list
            gpu_exchange_ghosts_make_indices(m_pdata->getN() + m_pdata->getNGhosts(),
                                             d_ghost_plan.data,
                                             d_tag.data,
                                             d_adj_mask.data,
                                             d_unique_neighbors.data,
                                             d_neigh_counts.data,
                                             d_scan.data,
                                             d_ghost_idx_adj.data + m_idx_offs[stage],
                                             d_ghost_neigh.data + m_idx_offs[stage],
                                             d_ghost_begin.data + stage * m_n_unique_neigh,
                                             d_ghost_end.data + stage * m_n_unique_neigh,
                                             m_n_unique_neigh,
                                             m_n_send_ghosts_tot[stage],
                                             m_comm_mask[stage],
                                             m_exec_conf->getCachedAllocator());

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

            {
            // access particle data
            ArrayHandle<unsigned int> d_tag(m_pdata->getTags(),
                                            access_location::device,
                                            access_mode::read);
            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(),
                                       access_location::device,
                                       access_mode::read);
            ArrayHandle<int3> d_image(m_pdata->getImages(),
                                      access_location::device,
                                      access_mode::read);
            ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(),
                                       access_location::device,
                                       access_mode::read);
            ArrayHandle<Scalar> d_charge(m_pdata->getCharges(),
                                         access_location::device,
                                         access_mode::read);
            ArrayHandle<unsigned int> d_body(m_pdata->getBodies(),
                                             access_location::device,
                                             access_mode::read);
            ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(),
                                           access_location::device,
                                           access_mode::read);
            ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(),
                                               access_location::device,
                                               access_mode::read);
            ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(),
                                             access_location::device,
                                             access_mode::read);

            // access ghost send indices
            ArrayHandle<uint2> d_ghost_idx_adj(m_ghost_idx_adj,
                                               access_location::device,
                                               access_mode::read);

            // access output buffers
            ArrayHandle<unsigned int> d_tag_ghost_sendbuf(m_tag_ghost_sendbuf,
                                                          access_location::device,
                                                          access_mode::overwrite);
            ArrayHandle<Scalar4> d_pos_ghost_sendbuf(m_pos_ghost_sendbuf,
                                                     access_location::device,
                                                     access_mode::overwrite);
            ArrayHandle<Scalar4> d_vel_ghost_sendbuf(m_vel_ghost_sendbuf,
                                                     access_location::device,
                                                     access_mode::overwrite);
            ArrayHandle<Scalar> d_charge_ghost_sendbuf(m_charge_ghost_sendbuf,
                                                       access_location::device,
                                                       access_mode::overwrite);
            ArrayHandle<unsigned int> d_body_ghost_sendbuf(m_body_ghost_sendbuf,
                                                           access_location::device,
                                                           access_mode::overwrite);
            ArrayHandle<int3> d_image_ghost_sendbuf(m_image_ghost_sendbuf,
                                                    access_location::device,
                                                    access_mode::overwrite);
            ArrayHandle<Scalar> d_diameter_ghost_sendbuf(m_diameter_ghost_sendbuf,
                                                         access_location::device,
                                                         access_mode::overwrite);
            ArrayHandle<Scalar4> d_orientation_ghost_sendbuf(m_orientation_ghost_sendbuf,
                                                             access_location::device,
                                                             access_mode::overwrite);

            const BoxDim global_box = m_pdata->getGlobalBox();
            const Index3D& di = m_pdata->getDomainDecomposition()->getDomainIndexer();
            uint3 my_pos = m_pdata->getDomainDecomposition()->getGridPos();

            // Pack ghosts into send buffers
            gpu_exchange_ghosts_pack(m_n_send_ghosts_tot[stage],
                                     d_ghost_idx_adj.data + m_idx_offs[stage],
                                     d_tag.data,
                                     d_pos.data,
                                     d_image.data,
                                     d_vel.data,
                                     d_charge.data,
                                     d_diameter.data,
                                     d_body.data,
                                     d_orientation.data,
                                     d_tag_ghost_sendbuf.data,
                                     d_pos_ghost_sendbuf.data,
                                     d_vel_ghost_sendbuf.data,
                                     d_charge_ghost_sendbuf.data,
                                     d_diameter_ghost_sendbuf.data,
                                     d_body_ghost_sendbuf.data,
                                     d_image_ghost_sendbuf.data,
                                     d_orientation_ghost_sendbuf.data,
                                     flags[comm_flag::tag],
                                     flags[comm_flag::position],
                                     flags[comm_flag::velocity],
                                     flags[comm_flag::charge],
                                     flags[comm_flag::diameter],
                                     flags[comm_flag::body],
                                     flags[comm_flag::image],
                                     flags[comm_flag::orientation],
                                     di,
                                     my_pos,
                                     global_box);

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

        /*
         * Ghost particle communication
         */
        m_n_recv_ghosts_tot[stage] = 0;

        unsigned int send_bytes = 0;
        unsigned int recv_bytes = 0;

            {
            ArrayHandle<unsigned int> h_ghost_begin(m_ghost_begin,
                                                    access_location::host,
                                                    access_mode::read);
            ArrayHandle<unsigned int> h_ghost_end(m_ghost_end,
                                                  access_location::host,
                                                  access_mode::read);
            ArrayHandle<unsigned int> h_unique_neighbors(m_unique_neighbors,
                                                         access_location::host,
                                                         access_mode::read);

            // compute send counts
            for (unsigned int ineigh = 0; ineigh < m_n_unique_neigh; ineigh++)
                m_n_send_ghosts[stage][ineigh]
                    = h_ghost_end.data[ineigh + stage * m_n_unique_neigh]
                      - h_ghost_begin.data[ineigh + stage * m_n_unique_neigh];

            MPI_Request req[2 * m_n_unique_neigh];
            MPI_Status stat[2 * m_n_unique_neigh];

            unsigned int nreq = 0;

            // loop over neighbors
            for (unsigned int ineigh = 0; ineigh < m_n_unique_neigh; ineigh++)
                {
                if (m_stages[ineigh] != (int)stage)
                    {
                    // skip neighbor if not participating in this communication stage
                    m_n_send_ghosts[stage][ineigh] = 0;
                    m_n_recv_ghosts[stage][ineigh] = 0;
                    continue;
                    }

                // rank of neighbor processor
                unsigned int neighbor = h_unique_neighbors.data[ineigh];

                MPI_Isend(&m_n_send_ghosts[stage][ineigh],
                          1,
                          MPI_UNSIGNED,
                          neighbor,
                          0,
                          m_mpi_comm,
                          &req[nreq++]);
                MPI_Irecv(&m_n_recv_ghosts[stage][ineigh],
                          1,
                          MPI_UNSIGNED,
                          neighbor,
                          0,
                          m_mpi_comm,
                          &req[nreq++]);

                send_bytes += (unsigned int)sizeof(unsigned int);
                recv_bytes += (unsigned int)sizeof(unsigned int);
                }

            MPI_Waitall(nreq, req, stat);

            // total up receive counts
            for (unsigned int ineigh = 0; ineigh < m_n_unique_neigh; ineigh++)
                {
                if (ineigh == 0)
                    m_ghost_offs[stage][ineigh] = 0;
                else
                    m_ghost_offs[stage][ineigh]
                        = m_ghost_offs[stage][ineigh - 1] + m_n_recv_ghosts[stage][ineigh - 1];

                m_n_recv_ghosts_tot[stage] += m_n_recv_ghosts[stage][ineigh];
                }
            }

        n_max = 0;
        // compute maximum number of received ghosts
        for (unsigned int istage = 0; istage <= stage; ++istage)
            if (m_n_recv_ghosts_tot[istage] > n_max)
                n_max = m_n_recv_ghosts_tot[istage];

        if (flags[comm_flag::tag])
            m_tag_ghost_recvbuf.resize(n_max);
        if (flags[comm_flag::position])
            m_pos_ghost_recvbuf.resize(n_max);
        if (flags[comm_flag::velocity])
            m_vel_ghost_recvbuf.resize(n_max);
        if (flags[comm_flag::charge])
            m_charge_ghost_recvbuf.resize(n_max);
        if (flags[comm_flag::body])
            m_body_ghost_recvbuf.resize(n_max);
        if (flags[comm_flag::image])
            m_image_ghost_recvbuf.resize(n_max);
        if (flags[comm_flag::diameter])
            m_diameter_ghost_recvbuf.resize(n_max);
        if (flags[comm_flag::orientation])
            m_orientation_ghost_recvbuf.resize(n_max);

        // first ghost ptl index
        unsigned int first_idx = m_pdata->getN() + m_pdata->getNGhosts();

        // update number of ghost particles
        m_pdata->addGhostParticles(m_n_recv_ghosts_tot[stage]);

            {
            unsigned int offs = 0;
            // recv buffers
            ArrayHandleAsync<unsigned int> tag_ghost_recvbuf_handle(m_tag_ghost_recvbuf,
                                                                    access_location::host,
                                                                    access_mode::overwrite);
            ArrayHandleAsync<Scalar4> pos_ghost_recvbuf_handle(m_pos_ghost_recvbuf,
                                                               access_location::host,
                                                               access_mode::overwrite);
            ArrayHandleAsync<Scalar4> vel_ghost_recvbuf_handle(m_vel_ghost_recvbuf,
                                                               access_location::host,
                                                               access_mode::overwrite);
            ArrayHandleAsync<Scalar> charge_ghost_recvbuf_handle(m_charge_ghost_recvbuf,
                                                                 access_location::host,
                                                                 access_mode::overwrite);
            ArrayHandleAsync<unsigned int> body_ghost_recvbuf_handle(m_body_ghost_recvbuf,
                                                                     access_location::host,
                                                                     access_mode::overwrite);
            ArrayHandleAsync<int3> image_ghost_recvbuf_handle(m_image_ghost_recvbuf,
                                                              access_location::host,
                                                              access_mode::overwrite);
            ArrayHandleAsync<Scalar> diameter_ghost_recvbuf_handle(m_diameter_ghost_recvbuf,
                                                                   access_location::host,
                                                                   access_mode::overwrite);
            ArrayHandleAsync<Scalar4> orientation_ghost_recvbuf_handle(m_orientation_ghost_recvbuf,
                                                                       access_location::host,
                                                                       access_mode::overwrite);
            // send buffers
            ArrayHandleAsync<unsigned int> tag_ghost_sendbuf_handle(m_tag_ghost_sendbuf,
                                                                    access_location::host,
                                                                    access_mode::read);
            ArrayHandleAsync<Scalar4> pos_ghost_sendbuf_handle(m_pos_ghost_sendbuf,
                                                               access_location::host,
                                                               access_mode::read);
            ArrayHandleAsync<Scalar4> vel_ghost_sendbuf_handle(m_vel_ghost_sendbuf,
                                                               access_location::host,
                                                               access_mode::read);
            ArrayHandleAsync<Scalar> charge_ghost_sendbuf_handle(m_charge_ghost_sendbuf,
                                                                 access_location::host,
                                                                 access_mode::read);
            ArrayHandleAsync<unsigned int> body_ghost_sendbuf_handle(m_body_ghost_sendbuf,
                                                                     access_location::host,
                                                                     access_mode::read);
            ArrayHandleAsync<int3> image_ghost_sendbuf_handle(m_image_ghost_sendbuf,
                                                              access_location::host,
                                                              access_mode::read);
            ArrayHandleAsync<Scalar> diameter_ghost_sendbuf_handle(m_diameter_ghost_sendbuf,
                                                                   access_location::host,
                                                                   access_mode::read);
            ArrayHandleAsync<Scalar4> orientation_ghost_sendbuf_handle(m_orientation_ghost_sendbuf,
                                                                       access_location::host,
                                                                       access_mode::read);

            // lump together into one synchronization call
            hipDeviceSynchronize();

            ArrayHandle<unsigned int> h_unique_neighbors(m_unique_neighbors,
                                                         access_location::host,
                                                         access_mode::read);
            ArrayHandle<unsigned int> h_ghost_begin(m_ghost_begin,
                                                    access_location::host,
                                                    access_mode::read);
            ArrayHandle<unsigned int> h_ghost_end(m_ghost_end,
                                                  access_location::host,
                                                  access_mode::read);

            std::vector<MPI_Request> reqs;
            MPI_Request req;

            // loop over neighbors
            for (unsigned int ineigh = 0; ineigh < m_n_unique_neigh; ineigh++)
                {
                // rank of neighbor processor
                unsigned int neighbor = h_unique_neighbors.data[ineigh];

                if (flags[comm_flag::tag])
                    {
                    // when sending/receiving 0 ptls, the send/recv buffer may be uninitialized
                    if (m_n_send_ghosts[stage][ineigh])
                        {
                        MPI_Isend(tag_ghost_sendbuf_handle.data
                                      + h_ghost_begin.data[ineigh + stage * m_n_unique_neigh],
                                  int(m_n_send_ghosts[stage][ineigh] * sizeof(unsigned int)),
                                  MPI_BYTE,
                                  neighbor,
                                  1,
                                  m_mpi_comm,
                                  &req);
                        reqs.push_back(req);
                        }
                    send_bytes
                        += (unsigned int)(m_n_send_ghosts[stage][ineigh] * sizeof(unsigned int));
                    if (m_n_recv_ghosts[stage][ineigh])
                        {
                        MPI_Irecv(tag_ghost_recvbuf_handle.data + m_ghost_offs[stage][ineigh]
                                      + offs,
                                  int(m_n_recv_ghosts[stage][ineigh] * sizeof(unsigned int)),
                                  MPI_BYTE,
                                  neighbor,
                                  1,
                                  m_mpi_comm,
                                  &req);
                        reqs.push_back(req);
                        }
                    recv_bytes
                        += (unsigned int)(m_n_recv_ghosts[stage][ineigh] * sizeof(unsigned int));
                    }

                if (flags[comm_flag::position])
                    {
                    MPI_Request req;
                    if (m_n_send_ghosts[stage][ineigh])
                        {
                        MPI_Isend(pos_ghost_sendbuf_handle.data
                                      + h_ghost_begin.data[ineigh + stage * m_n_unique_neigh],
                                  int(m_n_send_ghosts[stage][ineigh] * sizeof(Scalar4)),
                                  MPI_BYTE,
                                  neighbor,
                                  2,
                                  m_mpi_comm,
                                  &req);
                        reqs.push_back(req);
                        }
                    send_bytes += (unsigned int)(m_n_send_ghosts[stage][ineigh] * sizeof(Scalar4));
                    if (m_n_recv_ghosts[stage][ineigh])
                        {
                        MPI_Irecv(pos_ghost_recvbuf_handle.data + m_ghost_offs[stage][ineigh]
                                      + offs,
                                  int(m_n_recv_ghosts[stage][ineigh] * sizeof(Scalar4)),
                                  MPI_BYTE,
                                  neighbor,
                                  2,
                                  m_mpi_comm,
                                  &req);
                        reqs.push_back(req);
                        }
                    recv_bytes
                        += (unsigned int)(m_n_recv_ghosts[stage][ineigh] * sizeof(unsigned int));
                    }

                if (flags[comm_flag::velocity])
                    {
                    if (m_n_send_ghosts[stage][ineigh])
                        {
                        MPI_Isend(vel_ghost_sendbuf_handle.data
                                      + h_ghost_begin.data[ineigh + stage * m_n_unique_neigh],
                                  int(m_n_send_ghosts[stage][ineigh] * sizeof(Scalar4)),
                                  MPI_BYTE,
                                  neighbor,
                                  3,
                                  m_mpi_comm,
                                  &req);
                        reqs.push_back(req);
                        }
                    send_bytes += (unsigned int)(m_n_send_ghosts[stage][ineigh] * sizeof(Scalar4));
                    if (m_n_recv_ghosts[stage][ineigh])
                        {
                        MPI_Irecv(vel_ghost_recvbuf_handle.data + m_ghost_offs[stage][ineigh]
                                      + offs,
                                  int(m_n_recv_ghosts[stage][ineigh] * sizeof(Scalar4)),
                                  MPI_BYTE,
                                  neighbor,
                                  3,
                                  m_mpi_comm,
                                  &req);
                        reqs.push_back(req);
                        }
                    recv_bytes += (unsigned int)(m_n_recv_ghosts[stage][ineigh] * sizeof(Scalar4));
                    }

                if (flags[comm_flag::charge])
                    {
                    if (m_n_send_ghosts[stage][ineigh])
                        {
                        MPI_Isend(charge_ghost_sendbuf_handle.data
                                      + h_ghost_begin.data[ineigh + stage * m_n_unique_neigh],
                                  int(m_n_send_ghosts[stage][ineigh] * sizeof(Scalar)),
                                  MPI_BYTE,
                                  neighbor,
                                  4,
                                  m_mpi_comm,
                                  &req);
                        reqs.push_back(req);
                        }
                    send_bytes += (unsigned int)(m_n_send_ghosts[stage][ineigh] * sizeof(Scalar));
                    if (m_n_recv_ghosts[stage][ineigh])
                        {
                        MPI_Irecv(charge_ghost_recvbuf_handle.data + m_ghost_offs[stage][ineigh]
                                      + offs,
                                  int(m_n_recv_ghosts[stage][ineigh] * sizeof(Scalar)),
                                  MPI_BYTE,
                                  neighbor,
                                  4,
                                  m_mpi_comm,
                                  &req);
                        reqs.push_back(req);
                        }
                    recv_bytes += (unsigned int)(m_n_recv_ghosts[stage][ineigh] * sizeof(Scalar));
                    }

                if (flags[comm_flag::diameter])
                    {
                    if (m_n_send_ghosts[stage][ineigh])
                        {
                        MPI_Isend(diameter_ghost_sendbuf_handle.data
                                      + h_ghost_begin.data[ineigh + stage * m_n_unique_neigh],
                                  int(m_n_send_ghosts[stage][ineigh] * sizeof(Scalar)),
                                  MPI_BYTE,
                                  neighbor,
                                  5,
                                  m_mpi_comm,
                                  &req);
                        reqs.push_back(req);
                        }
                    send_bytes += (unsigned int)(m_n_send_ghosts[stage][ineigh] * sizeof(Scalar));
                    if (m_n_recv_ghosts[stage][ineigh])
                        {
                        MPI_Irecv(diameter_ghost_recvbuf_handle.data + m_ghost_offs[stage][ineigh]
                                      + offs,
                                  int(m_n_recv_ghosts[stage][ineigh] * sizeof(Scalar)),
                                  MPI_BYTE,
                                  neighbor,
                                  5,
                                  m_mpi_comm,
                                  &req);
                        reqs.push_back(req);
                        }
                    recv_bytes += (unsigned int)(m_n_recv_ghosts[stage][ineigh] * sizeof(Scalar));
                    }

                if (flags[comm_flag::orientation])
                    {
                    if (m_n_send_ghosts[stage][ineigh])
                        {
                        MPI_Isend(orientation_ghost_sendbuf_handle.data
                                      + h_ghost_begin.data[ineigh + stage * m_n_unique_neigh],
                                  int(m_n_send_ghosts[stage][ineigh] * sizeof(Scalar4)),
                                  MPI_BYTE,
                                  neighbor,
                                  6,
                                  m_mpi_comm,
                                  &req);
                        reqs.push_back(req);
                        }
                    send_bytes += (unsigned int)(m_n_send_ghosts[stage][ineigh] * sizeof(Scalar4));
                    if (m_n_recv_ghosts[stage][ineigh])
                        {
                        MPI_Irecv(orientation_ghost_recvbuf_handle.data
                                      + m_ghost_offs[stage][ineigh] + offs,
                                  int(m_n_recv_ghosts[stage][ineigh] * sizeof(Scalar4)),
                                  MPI_BYTE,
                                  neighbor,
                                  6,
                                  m_mpi_comm,
                                  &req);
                        reqs.push_back(req);
                        }
                    recv_bytes += (unsigned int)(m_n_recv_ghosts[stage][ineigh] * sizeof(Scalar4));
                    }

                if (flags[comm_flag::body])
                    {
                    if (m_n_send_ghosts[stage][ineigh])
                        {
                        MPI_Isend(body_ghost_sendbuf_handle.data
                                      + h_ghost_begin.data[ineigh + stage * m_n_unique_neigh],
                                  int(m_n_send_ghosts[stage][ineigh] * sizeof(unsigned int)),
                                  MPI_BYTE,
                                  neighbor,
                                  7,
                                  m_mpi_comm,
                                  &req);
                        reqs.push_back(req);
                        }
                    send_bytes
                        += (unsigned int)(m_n_send_ghosts[stage][ineigh] * sizeof(unsigned int));
                    if (m_n_recv_ghosts[stage][ineigh])
                        {
                        MPI_Irecv(body_ghost_recvbuf_handle.data + m_ghost_offs[stage][ineigh]
                                      + offs,
                                  int(m_n_recv_ghosts[stage][ineigh] * sizeof(unsigned int)),
                                  MPI_BYTE,
                                  neighbor,
                                  7,
                                  m_mpi_comm,
                                  &req);
                        reqs.push_back(req);
                        }
                    recv_bytes
                        += (unsigned int)(m_n_recv_ghosts[stage][ineigh] * sizeof(unsigned int));
                    }

                if (flags[comm_flag::image])
                    {
                    if (m_n_send_ghosts[stage][ineigh])
                        {
                        MPI_Isend(image_ghost_sendbuf_handle.data
                                      + h_ghost_begin.data[ineigh + stage * m_n_unique_neigh],
                                  int(m_n_send_ghosts[stage][ineigh] * sizeof(int3)),
                                  MPI_BYTE,
                                  neighbor,
                                  8,
                                  m_mpi_comm,
                                  &req);
                        reqs.push_back(req);
                        }
                    send_bytes += (unsigned int)(m_n_send_ghosts[stage][ineigh] * sizeof(int3));
                    if (m_n_recv_ghosts[stage][ineigh])
                        {
                        MPI_Irecv(image_ghost_recvbuf_handle.data + m_ghost_offs[stage][ineigh]
                                      + offs,
                                  int(m_n_recv_ghosts[stage][ineigh] * sizeof(int3)),
                                  MPI_BYTE,
                                  neighbor,
                                  8,
                                  m_mpi_comm,
                                  &req);
                        reqs.push_back(req);
                        }
                    recv_bytes += (unsigned int)(m_n_recv_ghosts[stage][ineigh] * sizeof(int3));
                    }
                } // end neighbor loop

            std::vector<MPI_Status> stats(reqs.size());
            MPI_Waitall((unsigned int)reqs.size(), &reqs.front(), &stats.front());
            } // end ArrayHandle scope
            // only unpack in non-CUDA MPI builds
            {
            // access receive buffers
            ArrayHandle<unsigned int> d_tag_ghost_recvbuf(m_tag_ghost_recvbuf,
                                                          access_location::device,
                                                          access_mode::read);
            ArrayHandle<Scalar4> d_pos_ghost_recvbuf(m_pos_ghost_recvbuf,
                                                     access_location::device,
                                                     access_mode::read);
            ArrayHandle<Scalar4> d_vel_ghost_recvbuf(m_vel_ghost_recvbuf,
                                                     access_location::device,
                                                     access_mode::read);
            ArrayHandle<Scalar> d_charge_ghost_recvbuf(m_charge_ghost_recvbuf,
                                                       access_location::device,
                                                       access_mode::read);
            ArrayHandle<unsigned int> d_body_ghost_recvbuf(m_body_ghost_recvbuf,
                                                           access_location::device,
                                                           access_mode::read);
            ArrayHandle<int3> d_image_ghost_recvbuf(m_image_ghost_recvbuf,
                                                    access_location::device,
                                                    access_mode::read);
            ArrayHandle<Scalar> d_diameter_ghost_recvbuf(m_diameter_ghost_recvbuf,
                                                         access_location::device,
                                                         access_mode::read);
            ArrayHandle<Scalar4> d_orientation_ghost_recvbuf(m_orientation_ghost_recvbuf,
                                                             access_location::device,
                                                             access_mode::read);
            // access particle data
            ArrayHandle<unsigned int> d_tag(m_pdata->getTags(),
                                            access_location::device,
                                            access_mode::readwrite);
            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(),
                                       access_location::device,
                                       access_mode::readwrite);
            ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(),
                                       access_location::device,
                                       access_mode::readwrite);
            ArrayHandle<Scalar> d_charge(m_pdata->getCharges(),
                                         access_location::device,
                                         access_mode::readwrite);
            ArrayHandle<unsigned int> d_body(m_pdata->getBodies(),
                                             access_location::device,
                                             access_mode::readwrite);
            ArrayHandle<int3> d_image(m_pdata->getImages(),
                                      access_location::device,
                                      access_mode::readwrite);
            ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(),
                                           access_location::device,
                                           access_mode::readwrite);
            ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(),
                                               access_location::device,
                                               access_mode::readwrite);

            // copy recv buf into particle data
            gpu_exchange_ghosts_copy_buf(m_n_recv_ghosts_tot[stage],
                                         d_tag_ghost_recvbuf.data,
                                         d_pos_ghost_recvbuf.data,
                                         d_vel_ghost_recvbuf.data,
                                         d_charge_ghost_recvbuf.data,
                                         d_diameter_ghost_recvbuf.data,
                                         d_body_ghost_recvbuf.data,
                                         d_image_ghost_recvbuf.data,
                                         d_orientation_ghost_recvbuf.data,
                                         d_tag.data + first_idx,
                                         d_pos.data + first_idx,
                                         d_vel.data + first_idx,
                                         d_charge.data + first_idx,
                                         d_diameter.data + first_idx,
                                         d_body.data + first_idx,
                                         d_image.data + first_idx,
                                         d_orientation.data + first_idx,
                                         flags[comm_flag::tag],
                                         flags[comm_flag::position],
                                         flags[comm_flag::velocity],
                                         flags[comm_flag::charge],
                                         flags[comm_flag::diameter],
                                         flags[comm_flag::body],
                                         flags[comm_flag::image],
                                         flags[comm_flag::orientation]);

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }
        if (flags[comm_flag::tag])
            {
            ArrayHandle<unsigned int> d_tag(m_pdata->getTags(),
                                            access_location::device,
                                            access_mode::read);
            ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(),
                                             access_location::device,
                                             access_mode::readwrite);

            // update reverse-lookup table
            gpu_compute_ghost_rtags(first_idx,
                                    m_n_recv_ghosts_tot[stage],
                                    d_tag.data + first_idx,
                                    d_rtag.data);
            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }
        } // end main communication loop

    m_ghosts_added = m_pdata->getNGhosts();

    // exchange ghost constraints along with ghost particles
    m_constraint_comm.exchangeGhostGroups(m_ghost_plan);

    m_last_flags = flags;
    }

//! Perform ghosts update
void CommunicatorGPU::beginUpdateGhosts(uint64_t timestep)
    {
    m_exec_conf->msg->notice(7) << "CommunicatorGPU: ghost update" << std::endl;

    CommFlags flags = getFlags();

    // main communication loop
    for (unsigned int stage = 0; stage < m_num_stages; ++stage)
        {
            {
            // access particle data
            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(),
                                       access_location::device,
                                       access_mode::read);
            ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(),
                                       access_location::device,
                                       access_mode::read);
            ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(),
                                               access_location::device,
                                               access_mode::read);

            // access ghost send indices
            ArrayHandle<uint2> d_ghost_idx_adj(m_ghost_idx_adj,
                                               access_location::device,
                                               access_mode::read);

            // access output buffers
            ArrayHandle<unsigned int> d_tag_ghost_sendbuf(m_tag_ghost_sendbuf,
                                                          access_location::device,
                                                          access_mode::overwrite);
            ArrayHandle<Scalar4> d_pos_ghost_sendbuf(m_pos_ghost_sendbuf,
                                                     access_location::device,
                                                     access_mode::overwrite);
            ArrayHandle<Scalar4> d_vel_ghost_sendbuf(m_vel_ghost_sendbuf,
                                                     access_location::device,
                                                     access_mode::overwrite);
            ArrayHandle<Scalar4> d_orientation_ghost_sendbuf(m_orientation_ghost_sendbuf,
                                                             access_location::device,
                                                             access_mode::overwrite);

            const BoxDim global_box = m_pdata->getGlobalBox();
            const Index3D& di = m_pdata->getDomainDecomposition()->getDomainIndexer();
            uint3 my_pos = m_pdata->getDomainDecomposition()->getGridPos();

            // Pack ghosts into send buffers
            gpu_exchange_ghosts_pack(m_n_send_ghosts_tot[stage],
                                     d_ghost_idx_adj.data + m_idx_offs[stage],
                                     NULL,
                                     d_pos.data,
                                     NULL,
                                     d_vel.data,
                                     NULL,
                                     NULL,
                                     NULL,
                                     d_orientation.data,
                                     NULL,
                                     d_pos_ghost_sendbuf.data,
                                     d_vel_ghost_sendbuf.data,
                                     NULL,
                                     NULL,
                                     NULL,
                                     NULL,
                                     d_orientation_ghost_sendbuf.data,
                                     false,
                                     flags[comm_flag::position],
                                     flags[comm_flag::velocity],
                                     false,
                                     false,
                                     false,
                                     false,
                                     flags[comm_flag::orientation],
                                     di,
                                     my_pos,
                                     global_box);

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

        /*
         * Ghost particle communication
         */

        // first ghost ptl index
        unsigned int first_idx = m_pdata->getN();

        // total up ghosts received thus far
        for (unsigned int istage = 0; istage < stage; ++istage)
            {
            first_idx += m_n_recv_ghosts_tot[istage];
            }

            {
            unsigned int offs = 0;
            // access particle data
            // recv buffers
            ArrayHandle<Scalar4> pos_ghost_recvbuf_handle(m_pos_ghost_recvbuf,
                                                          access_location::host,
                                                          access_mode::overwrite);
            ArrayHandle<Scalar4> vel_ghost_recvbuf_handle(m_vel_ghost_recvbuf,
                                                          access_location::host,
                                                          access_mode::overwrite);
            ArrayHandle<Scalar4> orientation_ghost_recvbuf_handle(m_orientation_ghost_recvbuf,
                                                                  access_location::host,
                                                                  access_mode::overwrite);

            // send buffers
            ArrayHandleAsync<Scalar4> pos_ghost_sendbuf_handle(m_pos_ghost_sendbuf,
                                                               access_location::host,
                                                               access_mode::read);
            ArrayHandleAsync<Scalar4> vel_ghost_sendbuf_handle(m_vel_ghost_sendbuf,
                                                               access_location::host,
                                                               access_mode::read);
            ArrayHandleAsync<Scalar4> orientation_ghost_sendbuf_handle(m_orientation_ghost_sendbuf,
                                                                       access_location::host,
                                                                       access_mode::read);

            ArrayHandleAsync<unsigned int> h_unique_neighbors(m_unique_neighbors,
                                                              access_location::host,
                                                              access_mode::read);
            ArrayHandleAsync<unsigned int> h_ghost_begin(m_ghost_begin,
                                                         access_location::host,
                                                         access_mode::read);

            // lump together into one synchronization call
            hipEventRecord(m_event);
            hipEventSynchronize(m_event);

            // access send buffers
            m_reqs.clear();
            MPI_Request req;

            unsigned int send_bytes = 0;
            unsigned int recv_bytes = 0;

            // loop over neighbors
            for (unsigned int ineigh = 0; ineigh < m_n_unique_neigh; ineigh++)
                {
                // rank of neighbor processor
                unsigned int neighbor = h_unique_neighbors.data[ineigh];

                if (flags[comm_flag::position])
                    {
                    if (m_n_send_ghosts[stage][ineigh])
                        {
                        MPI_Isend(pos_ghost_sendbuf_handle.data
                                      + h_ghost_begin.data[ineigh + stage * m_n_unique_neigh],
                                  int(m_n_send_ghosts[stage][ineigh] * sizeof(Scalar4)),
                                  MPI_BYTE,
                                  neighbor,
                                  2,
                                  m_mpi_comm,
                                  &req);
                        m_reqs.push_back(req);
                        }
                    send_bytes += (unsigned int)(m_n_send_ghosts[stage][ineigh] * sizeof(Scalar4));

                    if (m_n_recv_ghosts[stage][ineigh])
                        {
                        MPI_Irecv(pos_ghost_recvbuf_handle.data + m_ghost_offs[stage][ineigh]
                                      + offs,
                                  int(m_n_recv_ghosts[stage][ineigh] * sizeof(Scalar4)),
                                  MPI_BYTE,
                                  neighbor,
                                  2,
                                  m_mpi_comm,
                                  &req);
                        m_reqs.push_back(req);
                        }
                    recv_bytes += (unsigned int)(m_n_recv_ghosts[stage][ineigh] * sizeof(Scalar4));
                    }

                if (flags[comm_flag::velocity])
                    {
                    if (m_n_send_ghosts[stage][ineigh])
                        {
                        MPI_Isend(vel_ghost_sendbuf_handle.data
                                      + h_ghost_begin.data[ineigh + stage * m_n_unique_neigh],
                                  int(m_n_send_ghosts[stage][ineigh] * sizeof(Scalar4)),
                                  MPI_BYTE,
                                  neighbor,
                                  3,
                                  m_mpi_comm,
                                  &req);
                        m_reqs.push_back(req);
                        }
                    send_bytes += (unsigned int)(m_n_send_ghosts[stage][ineigh] * sizeof(Scalar4));

                    if (m_n_recv_ghosts[stage][ineigh])
                        {
                        MPI_Irecv(vel_ghost_recvbuf_handle.data + m_ghost_offs[stage][ineigh]
                                      + offs,
                                  int(m_n_recv_ghosts[stage][ineigh] * sizeof(Scalar4)),
                                  MPI_BYTE,
                                  neighbor,
                                  3,
                                  m_mpi_comm,
                                  &req);
                        m_reqs.push_back(req);
                        }
                    recv_bytes += (unsigned int)(m_n_recv_ghosts[stage][ineigh] * sizeof(Scalar4));
                    }

                if (flags[comm_flag::orientation])
                    {
                    if (m_n_send_ghosts[stage][ineigh])
                        {
                        MPI_Isend(orientation_ghost_sendbuf_handle.data
                                      + h_ghost_begin.data[ineigh + stage * m_n_unique_neigh],
                                  int(m_n_send_ghosts[stage][ineigh] * sizeof(Scalar4)),
                                  MPI_BYTE,
                                  neighbor,
                                  6,
                                  m_mpi_comm,
                                  &req);
                        m_reqs.push_back(req);
                        }
                    send_bytes += (unsigned int)(m_n_send_ghosts[stage][ineigh] * sizeof(Scalar4));

                    if (m_n_recv_ghosts[stage][ineigh])
                        {
                        MPI_Irecv(orientation_ghost_recvbuf_handle.data
                                      + m_ghost_offs[stage][ineigh] + offs,
                                  int(m_n_recv_ghosts[stage][ineigh] * sizeof(Scalar4)),
                                  MPI_BYTE,
                                  neighbor,
                                  6,
                                  m_mpi_comm,
                                  &req);
                        m_reqs.push_back(req);
                        }
                    recv_bytes += (unsigned int)(m_n_recv_ghosts[stage][ineigh] * sizeof(Scalar4));
                    }
                } // end neighbor loop

            if (m_num_stages == 1)
                {
                // use non-blocking MPI
                m_comm_pending = true;
                }
            else
                {
                // complete communication
                std::vector<MPI_Status> stats(m_reqs.size());
                MPI_Waitall((unsigned int)m_reqs.size(), &m_reqs.front(), &stats.front());
                }
            } // end ArrayHandle scope

        if (!m_comm_pending)
            {
                // only unpack in non-CUDA MPI builds
                {
                // access receive buffers
                ArrayHandle<Scalar4> d_pos_ghost_recvbuf(m_pos_ghost_recvbuf,
                                                         access_location::device,
                                                         access_mode::read);
                ArrayHandle<Scalar4> d_vel_ghost_recvbuf(m_vel_ghost_recvbuf,
                                                         access_location::device,
                                                         access_mode::read);
                ArrayHandle<Scalar4> d_orientation_ghost_recvbuf(m_orientation_ghost_recvbuf,
                                                                 access_location::device,
                                                                 access_mode::read);
                // access particle data
                ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(),
                                           access_location::device,
                                           access_mode::readwrite);
                ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(),
                                           access_location::device,
                                           access_mode::readwrite);
                ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(),
                                                   access_location::device,
                                                   access_mode::readwrite);

                // copy recv buf into particle data
                gpu_exchange_ghosts_copy_buf(m_n_recv_ghosts_tot[stage],
                                             NULL,
                                             d_pos_ghost_recvbuf.data,
                                             d_vel_ghost_recvbuf.data,
                                             NULL,
                                             NULL,
                                             NULL,
                                             NULL,
                                             d_orientation_ghost_recvbuf.data,
                                             NULL,
                                             d_pos.data + first_idx,
                                             d_vel.data + first_idx,
                                             NULL,
                                             NULL,
                                             NULL,
                                             NULL,
                                             d_orientation.data + first_idx,
                                             false,
                                             flags[comm_flag::position],
                                             flags[comm_flag::velocity],
                                             false,
                                             false,
                                             false,
                                             false,
                                             flags[comm_flag::orientation]);

                if (m_exec_conf->isCUDAErrorCheckingEnabled())
                    CHECK_CUDA_ERROR();
                }
            }
        } // end main communication loop
    }

/*! Finish ghost update
 *
 * \param timestep The time step
 */
void CommunicatorGPU::finishUpdateGhosts(uint64_t timestep)
    {
    if (m_comm_pending)
        {
        m_comm_pending = false;

        // complete communication
        std::vector<MPI_Status> stats(m_reqs.size());
        MPI_Waitall((unsigned int)m_reqs.size(), &m_reqs.front(), &stats.front());

        // only unpack in non-CUDA-MPI builds
        assert(m_num_stages == 1);
        unsigned int stage = 0;
        unsigned int first_idx = m_pdata->getN();
        CommFlags flags = m_last_flags;

            {
            // access receive buffers
            ArrayHandle<Scalar4> d_pos_ghost_recvbuf(m_pos_ghost_recvbuf,
                                                     access_location::device,
                                                     access_mode::read);
            ArrayHandle<Scalar4> d_vel_ghost_recvbuf(m_vel_ghost_recvbuf,
                                                     access_location::device,
                                                     access_mode::read);
            ArrayHandle<Scalar4> d_orientation_ghost_recvbuf(m_orientation_ghost_recvbuf,
                                                             access_location::device,
                                                             access_mode::read);
            // access particle data
            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(),
                                       access_location::device,
                                       access_mode::readwrite);
            ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(),
                                       access_location::device,
                                       access_mode::readwrite);
            ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(),
                                               access_location::device,
                                               access_mode::readwrite);

            // copy recv buf into particle data
            gpu_exchange_ghosts_copy_buf(m_n_recv_ghosts_tot[stage],
                                         NULL,
                                         d_pos_ghost_recvbuf.data,
                                         d_vel_ghost_recvbuf.data,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL,
                                         d_orientation_ghost_recvbuf.data,
                                         NULL,
                                         d_pos.data + first_idx,
                                         d_vel.data + first_idx,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL,
                                         d_orientation.data + first_idx,
                                         false,
                                         flags[comm_flag::position],
                                         flags[comm_flag::velocity],
                                         false,
                                         false,
                                         false,
                                         false,
                                         flags[comm_flag::orientation]);

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }
        }
    }

//! Perform ghosts update
void CommunicatorGPU::updateNetForce(uint64_t timestep)
    {
    CommFlags flags = getFlags();
    if (!flags[comm_flag::net_force] && !flags[comm_flag::net_torque]
        && !flags[comm_flag::net_virial])
        return;

    std::ostringstream oss;
    oss << "CommunicatorGPU: update net ";
    if (flags[comm_flag::net_force])
        {
        oss << "force ";
        }
    if (flags[comm_flag::net_torque])
        {
        oss << "torque ";
        }
    if (flags[comm_flag::net_virial])
        {
        oss << "virial";
        }

    m_exec_conf->msg->notice(7) << oss.str() << std::endl;

    // main communication loop
    for (unsigned int stage = 0; stage < m_num_stages; ++stage)
        {
        // compute maximum send buf size
        unsigned int n_max = 0;
        for (unsigned int istage = 0; istage <= stage; ++istage)
            if (m_n_send_ghosts_tot[istage] > n_max)
                n_max = m_n_send_ghosts_tot[istage];

        m_netforce_ghost_sendbuf.resize(n_max);

        if (flags[comm_flag::net_torque])
            {
            m_nettorque_ghost_sendbuf.resize(n_max);
            }

        if (flags[comm_flag::net_virial])
            {
            m_netvirial_ghost_sendbuf.resize(6 * n_max);
            }

            {
            // access particle data
            ArrayHandle<Scalar4> d_netforce(m_pdata->getNetForce(),
                                            access_location::device,
                                            access_mode::read);

            // access ghost send indices
            ArrayHandle<uint2> d_ghost_idx_adj(m_ghost_idx_adj,
                                               access_location::device,
                                               access_mode::read);

            // access output buffers
            ArrayHandle<Scalar4> d_netforce_ghost_sendbuf(m_netforce_ghost_sendbuf,
                                                          access_location::device,
                                                          access_mode::overwrite);

            // Pack ghosts into send buffers
            gpu_exchange_ghosts_pack_netforce(m_n_send_ghosts_tot[stage],
                                              d_ghost_idx_adj.data + m_idx_offs[stage],
                                              d_netforce.data,
                                              d_netforce_ghost_sendbuf.data);

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

        if (flags[comm_flag::net_torque])
            {
            // access particle data
            ArrayHandle<Scalar4> d_nettorque(m_pdata->getNetTorqueArray(),
                                             access_location::device,
                                             access_mode::read);

            // access ghost send indices
            ArrayHandle<uint2> d_ghost_idx_adj(m_ghost_idx_adj,
                                               access_location::device,
                                               access_mode::read);

            // access output buffers
            ArrayHandle<Scalar4> d_nettorque_ghost_sendbuf(m_nettorque_ghost_sendbuf,
                                                           access_location::device,
                                                           access_mode::overwrite);

            // Pack ghosts into send buffers
            gpu_exchange_ghosts_pack_netforce(m_n_send_ghosts_tot[stage],
                                              d_ghost_idx_adj.data + m_idx_offs[stage],
                                              d_nettorque.data,
                                              d_nettorque_ghost_sendbuf.data);

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

        if (flags[comm_flag::net_virial])
            {
            // access particle data
            ArrayHandle<Scalar> d_netvirial(m_pdata->getNetVirial(),
                                            access_location::device,
                                            access_mode::read);

            // access ghost send indices
            ArrayHandle<uint2> d_ghost_idx_adj(m_ghost_idx_adj,
                                               access_location::device,
                                               access_mode::read);

            // access output buffers
            ArrayHandle<Scalar> d_netvirial_ghost_sendbuf(m_netvirial_ghost_sendbuf,
                                                          access_location::device,
                                                          access_mode::overwrite);

            // Pack ghosts into send buffers
            gpu_exchange_ghosts_pack_netvirial(m_n_send_ghosts_tot[stage],
                                               d_ghost_idx_adj.data + m_idx_offs[stage],
                                               d_netvirial.data,
                                               d_netvirial_ghost_sendbuf.data,
                                               (unsigned int)m_pdata->getNetVirial().getPitch());

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

        /*
         * Ghost particle communication
         */

        n_max = 0;
        // compute maximum number of received ghosts
        for (unsigned int istage = 0; istage <= stage; ++istage)
            if (m_n_recv_ghosts_tot[istage] > n_max)
                n_max = m_n_recv_ghosts_tot[istage];

        m_netforce_ghost_recvbuf.resize(n_max);

        if (flags[comm_flag::net_torque])
            {
            m_nettorque_ghost_recvbuf.resize(n_max);
            }

        if (flags[comm_flag::net_virial])
            {
            m_netvirial_ghost_recvbuf.resize(6 * n_max);
            }

        // first ghost ptl index
        unsigned int first_idx = m_pdata->getN();

        // total up ghosts received thus far
        for (unsigned int istage = 0; istage < stage; ++istage)
            {
            first_idx += m_n_recv_ghosts_tot[istage];
            }

            {
            unsigned int offs = 0;
            // recv buffer
            ArrayHandle<Scalar4> h_netforce_ghost_recvbuf(m_netforce_ghost_recvbuf,
                                                          access_location::host,
                                                          access_mode::overwrite);
            ArrayHandle<Scalar4> h_nettorque_ghost_recvbuf(m_nettorque_ghost_recvbuf,
                                                           access_location::host,
                                                           access_mode::overwrite);
            ArrayHandle<Scalar> h_netvirial_ghost_recvbuf(m_netvirial_ghost_recvbuf,
                                                          access_location::host,
                                                          access_mode::overwrite);

            // send buffer
            ArrayHandle<Scalar4> h_netforce_ghost_sendbuf(m_netforce_ghost_sendbuf,
                                                          access_location::host,
                                                          access_mode::read);
            ArrayHandle<Scalar4> h_nettorque_ghost_sendbuf(m_nettorque_ghost_sendbuf,
                                                           access_location::host,
                                                           access_mode::read);
            ArrayHandle<Scalar> h_netvirial_ghost_sendbuf(m_netvirial_ghost_sendbuf,
                                                          access_location::host,
                                                          access_mode::read);

            ArrayHandleAsync<unsigned int> h_unique_neighbors(m_unique_neighbors,
                                                              access_location::host,
                                                              access_mode::read);
            ArrayHandleAsync<unsigned int> h_ghost_begin(m_ghost_begin,
                                                         access_location::host,
                                                         access_mode::read);

            // access send buffers
            m_reqs.clear();
            MPI_Request req;

            unsigned int send_bytes = 0;
            unsigned int recv_bytes = 0;

            // loop over neighbors
            for (unsigned int ineigh = 0; ineigh < m_n_unique_neigh; ineigh++)
                {
                // rank of neighbor processor
                unsigned int neighbor = h_unique_neighbors.data[ineigh];

                if (m_n_send_ghosts[stage][ineigh])
                    {
                    MPI_Isend(h_netforce_ghost_sendbuf.data
                                  + h_ghost_begin.data[ineigh + stage * m_n_unique_neigh],
                              int(m_n_send_ghosts[stage][ineigh] * sizeof(Scalar4)),
                              MPI_BYTE,
                              neighbor,
                              2,
                              m_mpi_comm,
                              &req);
                    m_reqs.push_back(req);
                    }
                send_bytes += (unsigned int)(m_n_send_ghosts[stage][ineigh] * sizeof(Scalar4));

                if (m_n_recv_ghosts[stage][ineigh])
                    {
                    MPI_Irecv(h_netforce_ghost_recvbuf.data + m_ghost_offs[stage][ineigh] + offs,
                              int(m_n_recv_ghosts[stage][ineigh] * sizeof(Scalar4)),
                              MPI_BYTE,
                              neighbor,
                              2,
                              m_mpi_comm,
                              &req);
                    m_reqs.push_back(req);
                    }
                recv_bytes += (unsigned int)(m_n_recv_ghosts[stage][ineigh] * sizeof(Scalar4));

                if (flags[comm_flag::net_torque])
                    {
                    if (m_n_send_ghosts[stage][ineigh])
                        {
                        MPI_Isend(h_nettorque_ghost_sendbuf.data
                                      + h_ghost_begin.data[ineigh + stage * m_n_unique_neigh],
                                  int(m_n_send_ghosts[stage][ineigh] * sizeof(Scalar4)),
                                  MPI_BYTE,
                                  neighbor,
                                  3,
                                  m_mpi_comm,
                                  &req);
                        m_reqs.push_back(req);
                        }
                    send_bytes += (unsigned int)(m_n_send_ghosts[stage][ineigh] * sizeof(Scalar4));

                    if (m_n_recv_ghosts[stage][ineigh])
                        {
                        MPI_Irecv(h_nettorque_ghost_recvbuf.data + m_ghost_offs[stage][ineigh]
                                      + offs,
                                  int(m_n_recv_ghosts[stage][ineigh] * sizeof(Scalar4)),
                                  MPI_BYTE,
                                  neighbor,
                                  3,
                                  m_mpi_comm,
                                  &req);
                        m_reqs.push_back(req);
                        }
                    recv_bytes += (unsigned int)(m_n_recv_ghosts[stage][ineigh] * sizeof(Scalar4));
                    }

                if (flags[comm_flag::net_virial])
                    {
                    if (m_n_send_ghosts[stage][ineigh])
                        {
                        MPI_Isend(h_netvirial_ghost_sendbuf.data
                                      + 6 * h_ghost_begin.data[ineigh + stage * m_n_unique_neigh],
                                  int(6 * m_n_send_ghosts[stage][ineigh] * sizeof(Scalar)),
                                  MPI_BYTE,
                                  neighbor,
                                  4,
                                  m_mpi_comm,
                                  &req);
                        m_reqs.push_back(req);
                        }
                    send_bytes
                        += (unsigned int)(6 * m_n_send_ghosts[stage][ineigh] * sizeof(Scalar));

                    if (m_n_recv_ghosts[stage][ineigh])
                        {
                        MPI_Irecv(h_netvirial_ghost_recvbuf.data
                                      + 6 * (m_ghost_offs[stage][ineigh] + offs),
                                  int(6 * m_n_recv_ghosts[stage][ineigh] * sizeof(Scalar)),
                                  MPI_BYTE,
                                  neighbor,
                                  4,
                                  m_mpi_comm,
                                  &req);
                        m_reqs.push_back(req);
                        }
                    recv_bytes
                        += (unsigned int)(6 * m_n_recv_ghosts[stage][ineigh] * sizeof(Scalar));
                    }
                }

            // complete communication
            std::vector<MPI_Status> stats(m_reqs.size());
            MPI_Waitall((unsigned int)m_reqs.size(), &m_reqs.front(), &stats.front());
            } // end ArrayHandle scope

            {
            // access receive buffers
            ArrayHandle<Scalar4> d_netforce_ghost_recvbuf(m_netforce_ghost_recvbuf,
                                                          access_location::device,
                                                          access_mode::read);

            // access particle data
            ArrayHandle<Scalar4> d_netforce(m_pdata->getNetForce(),
                                            access_location::device,
                                            access_mode::readwrite);

            // copy recv buf into particle data
            gpu_exchange_ghosts_copy_netforce_buf(m_n_recv_ghosts_tot[stage],
                                                  d_netforce_ghost_recvbuf.data,
                                                  d_netforce.data + first_idx);

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

        if (flags[comm_flag::net_torque])
            {
            // access receive buffers
            ArrayHandle<Scalar4> d_nettorque_ghost_recvbuf(m_nettorque_ghost_recvbuf,
                                                           access_location::device,
                                                           access_mode::read);

            // access particle data
            ArrayHandle<Scalar4> d_nettorque(m_pdata->getNetTorqueArray(),
                                             access_location::device,
                                             access_mode::readwrite);

            // copy recv buf into particle data
            gpu_exchange_ghosts_copy_netforce_buf(m_n_recv_ghosts_tot[stage],
                                                  d_nettorque_ghost_recvbuf.data,
                                                  d_nettorque.data + first_idx);

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

        if (flags[comm_flag::net_virial])
            {
            // access receive buffers
            ArrayHandle<Scalar> d_netvirial_ghost_recvbuf(m_netvirial_ghost_recvbuf,
                                                          access_location::device,
                                                          access_mode::read);

            // access particle data
            ArrayHandle<Scalar> d_netvirial(m_pdata->getNetVirial(),
                                            access_location::device,
                                            access_mode::readwrite);

            // copy recv buf into particle data
            gpu_exchange_ghosts_copy_netvirial_buf(
                m_n_recv_ghosts_tot[stage],
                d_netvirial_ghost_recvbuf.data,
                d_netvirial.data + first_idx,
                (unsigned int)m_pdata->getNetVirial().getPitch());

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }
        } // end main communication loop
    }

namespace detail
    {
//! Export CommunicatorGPU class to python
void export_CommunicatorGPU(pybind11::module& m)
    {
    pybind11::class_<CommunicatorGPU, Communicator, std::shared_ptr<CommunicatorGPU>>(
        m,
        "CommunicatorGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<DomainDecomposition>>())
        .def("setMaxStages", &CommunicatorGPU::setMaxStages);
    }
    } // end namespace detail

    } // end namespace hoomd

#endif // ENABLE_HIP
#endif // ENABLE_MPI
