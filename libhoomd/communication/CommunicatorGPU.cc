/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: jglaser

/*! \file CommunicatorGPU.cc
    \brief Implements the CommunicatorGPU class
*/

#ifdef ENABLE_MPI
#ifdef ENABLE_CUDA

#include "CommunicatorGPU.h"
#include "Profiler.h"
#include "System.h"

#include <boost/python.hpp>
#include <algorithm>
#include <functional>

using namespace boost::python;

//! Constructor
CommunicatorGPU::CommunicatorGPU(boost::shared_ptr<SystemDefinition> sysdef,
                                 boost::shared_ptr<DomainDecomposition> decomposition)
    : Communicator(sysdef, decomposition),
      m_nneigh(0),
      m_n_unique_neigh(0),
      m_max_stages(1),
      m_num_stages(0),
      m_comm_mask(0),
      m_bond_comm(*this, m_sysdef->getBondData()),
      m_angle_comm(*this, m_sysdef->getAngleData()),
      m_dihedral_comm(*this, m_sysdef->getDihedralData()),
      m_improper_comm(*this, m_sysdef->getImproperData()),
      m_last_flags(0)
    {
    // find out if this is a 1D decomposition
    unsigned int d = 0;
    if (decomposition->getDomainIndexer().getW() > 1) d++;
    if (decomposition->getDomainIndexer().getH() > 1) d++;
    if (decomposition->getDomainIndexer().getD() > 1) d++;

    assert(d>=1);
    #ifdef ENABLE_MPI_CUDA
    // print a warning if we are using a higher than linear dimensionality for the processor grid
    // and CUDA-MPI interop is enabled (the latency of a send/recv call is lower if not using CUDA-MPI)
    if (d > 1)
        {
        m_exec_conf->msg->notice(2) << "The processor grid has dimensionality " << d << " > 1 and CUDA-MPI support" << std::endl;
        m_exec_conf->msg->notice(2) << "is enabled. For optimal performance, disable CUDA-MPI support" << std::endl;
        m_exec_conf->msg->notice(2) << "(-D ENABLE_MPI_CUDA=0)." << std::endl;
        }
    #endif

    // allocate memory
    allocateBuffers();

    // initialize data
    initializeNeighborArrays();

    // initialize communciation stages
    initializeCommunicationStages();

    // Initialize cache configuration
    gpu_communicator_initialize_cache_config();

    // create at ModernGPU context
    m_mgpu_context = mgpu::CreateCudaDeviceAttachStream(0);
    }

//! Destructor
CommunicatorGPU::~CommunicatorGPU()
    {
    m_exec_conf->msg->notice(5) << "Destroying CommunicatorGPU";
    }

void CommunicatorGPU::allocateBuffers()
    {
    // using mapped-pinned memory avoids unnecessary memcpy's as buffers grow
    bool mapped = true;

    #ifdef ENABLE_MPI_CUDA
    // store data on device if using CUDA-aware MPI
    mapped = false;
    #endif

    /*
     * Particle migration
     */
    GPUVector<pdata_element> gpu_sendbuf(m_exec_conf,mapped);
    m_gpu_sendbuf.swap(gpu_sendbuf);

    GPUVector<pdata_element> gpu_recvbuf(m_exec_conf,mapped);
    m_gpu_recvbuf.swap(gpu_recvbuf);

    // Communciation flags for every particle sent
    GPUVector<unsigned int> comm_flags(m_exec_conf);
    m_comm_flags.swap(comm_flags);

    // Key for every particle sent
    GPUVector<unsigned int> send_keys(m_exec_conf);
    m_send_keys.swap(send_keys);

    GPUArray<unsigned int> neighbors(NEIGH_MAX,m_exec_conf);
    m_neighbors.swap(neighbors);

    GPUArray<unsigned int> unique_neighbors(NEIGH_MAX,m_exec_conf);
    m_unique_neighbors.swap(unique_neighbors);

    // neighbor masks
    GPUArray<unsigned int> adj_mask(NEIGH_MAX, m_exec_conf);
    m_adj_mask.swap(adj_mask);

    GPUArray<unsigned int> begin(NEIGH_MAX,m_exec_conf,true);
    m_begin.swap(begin);

    GPUArray<unsigned int> end(NEIGH_MAX,m_exec_conf,true);
    m_end.swap(end);

    /*
     * Ghost communication
     */

    GPUVector<unsigned int> tag_ghost_sendbuf(m_exec_conf,mapped);
    m_tag_ghost_sendbuf.swap(tag_ghost_sendbuf);

    GPUVector<unsigned int> tag_ghost_recvbuf(m_exec_conf,mapped);
    m_tag_ghost_recvbuf.swap(tag_ghost_recvbuf);

    GPUVector<Scalar4> pos_ghost_sendbuf(m_exec_conf,mapped);
    m_pos_ghost_sendbuf.swap(pos_ghost_sendbuf);

    GPUVector<Scalar4> pos_ghost_recvbuf(m_exec_conf,mapped);
    m_pos_ghost_recvbuf.swap(pos_ghost_recvbuf);

    GPUVector<Scalar4> vel_ghost_sendbuf(m_exec_conf,mapped);
    m_vel_ghost_sendbuf.swap(vel_ghost_sendbuf);

    GPUVector<Scalar4> vel_ghost_recvbuf(m_exec_conf,mapped);
    m_vel_ghost_recvbuf.swap(vel_ghost_recvbuf);

    GPUVector<Scalar> charge_ghost_sendbuf(m_exec_conf,mapped);
    m_charge_ghost_sendbuf.swap(charge_ghost_sendbuf);

    GPUVector<Scalar> charge_ghost_recvbuf(m_exec_conf,mapped);
    m_charge_ghost_recvbuf.swap(charge_ghost_recvbuf);

    GPUVector<Scalar> diameter_ghost_sendbuf(m_exec_conf,mapped);
    m_diameter_ghost_sendbuf.swap(diameter_ghost_sendbuf);

    GPUVector<Scalar> diameter_ghost_recvbuf(m_exec_conf,mapped);
    m_diameter_ghost_recvbuf.swap(diameter_ghost_recvbuf);

    GPUVector<Scalar4> orientation_ghost_sendbuf(m_exec_conf,mapped);
    m_orientation_ghost_sendbuf.swap(orientation_ghost_sendbuf);

    GPUVector<Scalar4> orientation_ghost_recvbuf(m_exec_conf,mapped);
    m_orientation_ghost_recvbuf.swap(orientation_ghost_recvbuf);

    GPUVector<unsigned int> ghost_begin(m_exec_conf,true);
    m_ghost_begin.swap(ghost_begin);

    GPUVector<unsigned int> ghost_end(m_exec_conf,true);
    m_ghost_end.swap(ghost_end);

    GPUVector<unsigned int> ghost_plan(m_exec_conf);
    m_ghost_plan.swap(ghost_plan);

    GPUVector<unsigned int> ghost_idx(m_exec_conf);
    m_ghost_idx.swap(ghost_idx);

    GPUVector<unsigned int> ghost_neigh(m_exec_conf);
    m_ghost_neigh.swap(ghost_neigh);

    GPUVector<unsigned int> neigh_counts(m_exec_conf);
    m_neigh_counts.swap(neigh_counts);
    }

void CommunicatorGPU::initializeNeighborArrays()
    {
    Index3D di= m_decomposition->getDomainIndexer();

    uint3 mypos = di.getTriple(m_exec_conf->getRank());
    int l = mypos.x;
    int m = mypos.y;
    int n = mypos.z;

    ArrayHandle<unsigned int> h_neighbors(m_neighbors, access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_adj_mask(m_adj_mask, access_location::host, access_mode::overwrite);

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

                unsigned int neighbor = di(i,j,k);
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

        // std::map inserts the same key only once
        neigh_map.insert(std::make_pair(h_neighbors.data[i], m));
        }

    m_n_unique_neigh = neigh_map.size();

    n = 0;
    for (std::map<unsigned int, unsigned int>::iterator it = neigh_map.begin(); it != neigh_map.end(); ++it)
        {
        h_unique_neighbors.data[n] = it->first;
        h_adj_mask.data[n] = it->second;
        n++;
        }
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
            << "Maximum number of communication stages too large. Assuming three."
            << std::endl;
        m_max_stages = 3;
        }

    // accesss neighbors and adjacency  array
    ArrayHandle<unsigned int> h_adj_mask(m_adj_mask, access_location::host, access_mode::read);

    Index3D di= m_decomposition->getDomainIndexer();

    // number of stages in every communication step
    m_num_stages = 0;

    m_comm_mask.resize(m_max_stages);

    #if 0
    // loop through neighbors to determine the communication stages
    unsigned int max_stage = 0;
    for (unsigned int ineigh = 0; ineigh < m_n_unique_neigh; ++ineigh)
        {
        int stage = 0;
        int n = -1;

        // determine stage
        if (di.getW() > 1 && (n+1 < (int) m_max_stages)) n++;
        if (h_adj_mask.data[ineigh] & (send_east | send_west)) stage = n;
        if (di.getH() > 1 && (n+1 < (int) m_max_stages)) n++;
        if (h_adj_mask.data[ineigh] & (send_north | send_south)) stage = n;
        if (di.getD() > 1 && (n+1 < (int) m_max_stages)) n++;
        if (h_adj_mask.data[ineigh] & (send_up | send_down)) stage = n;

        assert(stage >= 0);
        assert(n >= 0);

        // set communication flags for stage
        m_comm_mask[stage] |= h_adj_mask.data[ineigh];

        if (stage > (int)max_stage) max_stage = stage;
        }

    // number of communication stages
    m_num_stages = max_stage + 1;

    // every direction occurs in one and only one stages
    // number of communications per stage is constant or decreases with stage number
    for (unsigned int istage = 0; istage < m_num_stages; ++istage)
        for (unsigned int jstage = istage+1; jstage < m_num_stages; ++jstage)
            m_comm_mask[jstage] &= ~m_comm_mask[istage];

    // access unique neighbors
    ArrayHandle<unsigned int> h_unique_neighbors(m_unique_neighbors, access_location::host, access_mode::read);

    // initialize stages array
    m_stages.resize(m_n_unique_neigh,-1);

    // assign stages to unique neighbors
    for (unsigned int i= 0; i < m_n_unique_neigh; i++)
        for (unsigned int istage = 0; istage < m_num_stages; ++istage)
            // compare adjacency masks of neighbors to mask for this stage
            if ((h_adj_mask.data[i] & m_comm_mask[istage]) == h_adj_mask.data[i])
                {
                m_stages[i] = istage;
                break; // associate neighbor with stage of lowest index
                }
    #else
    m_comm_mask[0] = 0;
    if (di.getW() > 1) m_comm_mask[0] |= (Communicator::send_east | Communicator::send_west);
    if (di.getH() > 1) m_comm_mask[0] |= (Communicator::send_north| Communicator::send_south);
    if (di.getD() > 1) m_comm_mask[0] |= (Communicator::send_up | Communicator::send_down);
    m_stages.resize(m_n_unique_neigh,0);
    m_num_stages = 1;
    #endif
    m_exec_conf->msg->notice(5) << "ComunicatorGPU: " << m_num_stages << " communication stages." << std::endl;
    }

//! Select a particle for migration
struct get_migrate_key : public std::unary_function<const unsigned int, unsigned int >
    {
    const uint3 my_pos;      //!< My domain decomposition position
    const Index3D di;        //!< Domain indexer
    const unsigned int mask; //!< Mask of allowed directions

    //! Constructor
    /*!
     */
    get_migrate_key(const uint3 _my_pos, const Index3D _di, const unsigned int _mask)
        : my_pos(_my_pos), di(_di), mask(_mask)
        { }

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

        return di(i,j,k);
        }

     };

//! Constructor
template<class group_data>
CommunicatorGPU::GroupCommunicatorGPU<group_data>::GroupCommunicatorGPU(CommunicatorGPU& gpu_comm, boost::shared_ptr<group_data> gdata)
    : m_gpu_comm(gpu_comm), m_exec_conf(m_gpu_comm.m_exec_conf), m_gdata(gdata)
    {
    // accelerate copying of data for host MPI
    #ifdef ENABLE_MPI_CUDA
    bool mapped = false;
    #else
    bool mapped = true;
    #endif

    GPUVector<unsigned int> rank_mask(m_gpu_comm.m_exec_conf);
    m_rank_mask.swap(rank_mask);

    GPUVector<unsigned int> scratch(m_gpu_comm.m_exec_conf);
    m_scan.swap(scratch);

    GPUVector<rank_element_t> ranks_out(m_gpu_comm.m_exec_conf,mapped);
    m_ranks_out.swap(ranks_out);

    GPUVector<rank_element_t> ranks_sendbuf(m_gpu_comm.m_exec_conf,mapped);
    m_ranks_sendbuf.swap(ranks_sendbuf);

    GPUVector<rank_element_t> ranks_recvbuf(m_gpu_comm.m_exec_conf,mapped);
    m_ranks_recvbuf.swap(ranks_recvbuf);

    GPUVector<group_element_t> groups_out(m_gpu_comm.m_exec_conf,mapped);
    m_groups_out.swap(groups_out);

    GPUVector<unsigned int> rank_mask_out(m_gpu_comm.m_exec_conf,mapped);
    m_rank_mask_out.swap(rank_mask_out);

    GPUVector<group_element_t> groups_sendbuf(m_gpu_comm.m_exec_conf,mapped);
    m_groups_sendbuf.swap(groups_sendbuf);

    GPUVector<group_element_t> groups_recvbuf(m_gpu_comm.m_exec_conf,mapped);
    m_groups_recvbuf.swap(groups_sendbuf);

    GPUVector<group_element_t> groups_in(m_gpu_comm.m_exec_conf, mapped);
    m_groups_in.swap(groups_in);

    // the size of the bit field must be larger or equal the group size
    assert(sizeof(unsigned int)*8 >= group_data::size);
    }

//! Migrate groups
template<class group_data>
void CommunicatorGPU::GroupCommunicatorGPU<group_data>::migrateGroups(bool incomplete)
    {
    if (m_gdata->getNGlobal())
        {
        if (m_gpu_comm.m_prof) m_gpu_comm.m_prof->push(m_exec_conf, m_gdata->getName());

        // resize bitmasks
        m_rank_mask.resize(m_gdata->getN());

        // resize temporary arry
        m_scan.resize(m_gdata->getN());

        unsigned int n_out_ranks;
            {
            ArrayHandle<unsigned int> d_comm_flags(m_gpu_comm.m_pdata->getCommFlags(), access_location::device, access_mode::read);
            ArrayHandle<typename group_data::members_t> d_members(m_gdata->getMembersArray(), access_location::device, access_mode::read);
            ArrayHandle<typename group_data::ranks_t> d_group_ranks(m_gdata->getRanksArray(), access_location::device, access_mode::readwrite);
            ArrayHandle<unsigned int> d_rank_mask(m_rank_mask, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_rtag(m_gpu_comm.m_pdata->getRTags(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_scan(m_scan, access_location::device, access_mode::overwrite);

            Index3D di = m_gpu_comm.m_pdata->getDomainDecomposition()->getDomainIndexer();
            uint3 my_pos = di.getTriple(m_gpu_comm.m_exec_conf->getRank());

            // mark groups that have members leaving this domain
            gpu_mark_groups<group_data::size>(
                m_gpu_comm.m_pdata->getN(),
                d_comm_flags.data,
                m_gdata->getN(),
                d_members.data,
                d_group_ranks.data,
                d_rank_mask.data,
                d_rtag.data,
                d_scan.data,
                n_out_ranks,
                di,
                my_pos,
                incomplete,
                m_gpu_comm.m_mgpu_context);

            if (m_gpu_comm.m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
            }

        // resize output array
        m_ranks_out.resize(n_out_ranks);

        unsigned int n_out_groups;
            {
            ArrayHandle<unsigned int> d_group_tag(m_gdata->getTags(), access_location::device, access_mode::read);
            ArrayHandle<typename group_data::members_t> d_members(m_gdata->getMembersArray(), access_location::device, access_mode::read);
            ArrayHandle<typename group_data::ranks_t> d_group_ranks(m_gdata->getRanksArray(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_rank_mask(m_rank_mask, access_location::device, access_mode::readwrite);

            ArrayHandle<unsigned int> d_rtag(m_gpu_comm.m_pdata->getRTags(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_comm_flags(m_gpu_comm.m_pdata->getCommFlags(), access_location::device, access_mode::readwrite);

            ArrayHandle<rank_element_t> d_ranks_out(m_ranks_out, access_location::device, access_mode::overwrite);

            ArrayHandle<unsigned int> d_scan(m_scan, access_location::device, access_mode::readwrite);

            // scatter groups into output arrays according to scan result (d_scan), determine send groups and scan
            gpu_scatter_ranks_and_mark_send_groups<group_data::size>(
                m_gdata->getN(),
                d_group_tag.data,
                d_group_ranks.data,
                d_rank_mask.data,
                d_members.data,
                d_rtag.data,
                d_comm_flags.data,
                d_scan.data,
                n_out_groups,
                d_ranks_out.data,
                m_gpu_comm.m_mgpu_context);

            if (m_gpu_comm.m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
            }

        #ifdef ENABLE_MPI_CUDA
        #else
        // fill host send buffers on host
        unsigned int my_rank = m_gpu_comm.m_exec_conf->getRank();

        typedef std::multimap<unsigned int, rank_element_t> map_t;
        map_t send_map;

            {
            // access output buffers
            ArrayHandle<rank_element_t> h_ranks_out(m_ranks_out, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_unique_neighbors(m_gpu_comm.m_unique_neighbors, access_location:: host, access_mode::read);

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
                    for (unsigned int j = 0; j < group_data::size; ++j)
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
            ArrayHandle<rank_element_t> h_ranks_sendbuf(m_ranks_sendbuf, access_location::host, access_mode::overwrite);

            // output send data sorted by rank
            unsigned int n = 0;
            for (typename map_t::iterator it = send_map.begin(); it != send_map.end(); ++it)
                {
                h_ranks_sendbuf.data[n] = it->second;
                n++;
                }

            ArrayHandle<unsigned int> h_unique_neighbors(m_gpu_comm.m_unique_neighbors, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_begin(m_gpu_comm.m_begin, access_location::host, access_mode::overwrite);
            ArrayHandle<unsigned int> h_end(m_gpu_comm.m_end, access_location::host, access_mode::overwrite);

            // Find start and end indices
            for (unsigned int i = 0; i < m_gpu_comm.m_n_unique_neigh; ++i)
                {
                typename map_t::iterator lower = send_map.lower_bound(h_unique_neighbors.data[i]);
                typename map_t::iterator upper = send_map.upper_bound(h_unique_neighbors.data[i]);
                h_begin.data[i] = std::distance(send_map.begin(),lower);
                h_end.data[i] = std::distance(send_map.begin(),upper);
                }
            }
        #endif

        /*
         * communicate rank information (phase 1)
         */
        unsigned int n_send_groups[m_gpu_comm.m_n_unique_neigh];
        unsigned int n_recv_groups[m_gpu_comm.m_n_unique_neigh];
        unsigned int offs[m_gpu_comm.m_n_unique_neigh];
        unsigned int n_recv_tot = 0;

            {
            ArrayHandle<unsigned int> h_begin(m_gpu_comm.m_begin, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_end(m_gpu_comm.m_end, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_unique_neighbors(m_gpu_comm.m_unique_neighbors, access_location::host, access_mode::read);

            unsigned int send_bytes = 0;
            unsigned int recv_bytes = 0;
            if (m_gpu_comm.m_prof) m_gpu_comm.m_prof->push(m_exec_conf, "MPI send/recv");

            // compute send counts
            for (unsigned int ineigh = 0; ineigh < m_gpu_comm.m_n_unique_neigh; ineigh++)
                n_send_groups[ineigh] = h_end.data[ineigh] - h_begin.data[ineigh];

            MPI_Request req[2*m_gpu_comm.m_n_unique_neigh];
            MPI_Status stat[2*m_gpu_comm.m_n_unique_neigh];

            unsigned int nreq = 0;

            // loop over neighbors
            for (unsigned int ineigh = 0; ineigh < m_gpu_comm.m_n_unique_neigh; ineigh++)
                {
                // rank of neighbor processor
                unsigned int neighbor = h_unique_neighbors.data[ineigh];

                MPI_Isend(&n_send_groups[ineigh], 1, MPI_UNSIGNED, neighbor, 0, m_gpu_comm.m_mpi_comm, & req[nreq++]);
                MPI_Irecv(&n_recv_groups[ineigh], 1, MPI_UNSIGNED, neighbor, 0, m_gpu_comm.m_mpi_comm, & req[nreq++]);
                send_bytes += sizeof(unsigned int);
                recv_bytes += sizeof(unsigned int);
                } // end neighbor loop

            MPI_Waitall(nreq, req, stat);

            // sum up receive counts
            for (unsigned int ineigh = 0; ineigh < m_gpu_comm.m_n_unique_neigh; ineigh++)
                {
                if (ineigh == 0)
                    offs[ineigh] = 0;
                else
                    offs[ineigh] = offs[ineigh-1] + n_recv_groups[ineigh-1];

                n_recv_tot += n_recv_groups[ineigh];
                }

            if (m_gpu_comm.m_prof) m_gpu_comm.m_prof->pop(m_exec_conf,0,send_bytes+recv_bytes);
            }

        // Resize receive buffer
        m_ranks_recvbuf.resize(n_recv_tot);

            {
            ArrayHandle<unsigned int> h_begin(m_gpu_comm.m_begin, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_end(m_gpu_comm.m_end, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_unique_neighbors(m_gpu_comm.m_unique_neighbors, access_location::host, access_mode::read);

            if (m_gpu_comm.m_prof) m_gpu_comm.m_prof->push(m_exec_conf,"MPI send/recv");

            #ifdef ENABLE_MPI_CUDA
            ArrayHandle<rank_element_t> ranks_sendbuf_handle(m_ranks_sendbuf, access_location::device, access_mode::read);
            ArrayHandle<rank_element_t> ranks_recvbuf_handle(m_ranks_recvbuf, access_location::device, access_mode::overwrite);
            #else
            ArrayHandle<rank_element_t> ranks_sendbuf_handle(m_ranks_sendbuf, access_location::host, access_mode::read);
            ArrayHandle<rank_element_t> ranks_recvbuf_handle(m_ranks_recvbuf, access_location::host, access_mode::overwrite);
            #endif

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
                    MPI_Isend(ranks_sendbuf_handle.data+h_begin.data[ineigh],
                        n_send_groups[ineigh]*sizeof(rank_element_t),
                        MPI_BYTE,
                        neighbor,
                        1,
                        m_gpu_comm.m_mpi_comm,
                        &req);
                    reqs.push_back(req);
                    }
                send_bytes+= n_send_groups[ineigh]*sizeof(rank_element_t);

                if (n_recv_groups[ineigh])
                    {
                    MPI_Irecv(ranks_recvbuf_handle.data+offs[ineigh],
                        n_recv_groups[ineigh]*sizeof(rank_element_t),
                        MPI_BYTE,
                        neighbor,
                        1,
                        m_gpu_comm.m_mpi_comm,
                        &req);
                    reqs.push_back(req);
                    }
                recv_bytes += n_recv_groups[ineigh]*sizeof(rank_element_t);
                }

            std::vector<MPI_Status> stats(reqs.size());
            MPI_Waitall(reqs.size(), &reqs.front(), &stats.front());

            if (m_gpu_comm.m_prof) m_gpu_comm.m_prof->pop(m_exec_conf,0,send_bytes+recv_bytes);
            }

            {
            // access receive buffers
            ArrayHandle<rank_element_t> d_ranks_recvbuf(m_ranks_recvbuf, access_location::device, access_mode::read);
            ArrayHandle<typename group_data::ranks_t> d_group_ranks(m_gdata->getRanksArray(), access_location::device, access_mode::readwrite);
            ArrayHandle<unsigned int> d_group_rtag(m_gdata->getRTags(), access_location::device, access_mode::read);

            // update local rank information
            gpu_update_ranks_table<group_data::size>(
                m_gdata->getN(),
                d_group_ranks.data,
                d_group_rtag.data,
                n_recv_tot,
                d_ranks_recvbuf.data);
            if (m_gpu_comm.m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
            }

        // resize output buffer
        m_groups_out.resize(n_out_groups);
        m_rank_mask_out.resize(n_out_groups);

            {
            ArrayHandle<typename group_data::members_t> d_groups(m_gdata->getMembersArray(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_group_type(m_gdata->getTypesArray(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_group_tag(m_gdata->getTags(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_group_rtag(m_gdata->getRTags(), access_location::device, access_mode::readwrite);
            ArrayHandle<typename group_data::ranks_t> d_group_ranks(m_gdata->getRanksArray(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_rank_mask(m_rank_mask, access_location::device, access_mode::readwrite);
            ArrayHandle<unsigned int> d_scan(m_scan, access_location::device, access_mode::readwrite);
            ArrayHandle<group_element_t> d_groups_out(m_groups_out, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_rank_mask_out(m_rank_mask_out, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_rtag(m_gpu_comm.m_pdata->getRTags(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_comm_flags(m_gpu_comm.m_pdata->getCommFlags(), access_location::device, access_mode::read);

            // scatter groups to be sent into output buffer, mark groups that have no local members for removal
            gpu_scatter_and_mark_groups_for_removal<group_data::size>(
                m_gdata->getN(),
                d_groups.data,
                d_group_type.data,
                d_group_tag.data,
                d_group_rtag.data,
                d_group_ranks.data,
                d_rank_mask.data,
                d_rtag.data,
                d_comm_flags.data,
                m_exec_conf->getRank(),
                d_scan.data,
                d_groups_out.data,
                d_rank_mask_out.data);
            if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
            }

        unsigned int new_ngroups;
            {
            // access primary arrays to read from
            ArrayHandle<typename group_data::members_t> d_groups(m_gdata->getMembersArray(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_group_type(m_gdata->getTypesArray(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_group_tag(m_gdata->getTags(), access_location::device, access_mode::read);
            ArrayHandle<typename group_data::ranks_t> d_group_ranks(m_gdata->getRanksArray(), access_location::device, access_mode::read);

            // access alternate arrays to write to
            ArrayHandle<typename group_data::members_t> d_groups_alt(m_gdata->getAltMembersArray(), access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_group_type_alt(m_gdata->getAltTypesArray(), access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_group_tag_alt(m_gdata->getAltTags(), access_location::device, access_mode::overwrite);
            ArrayHandle<typename group_data::ranks_t> d_group_ranks_alt(m_gdata->getAltRanksArray(), access_location::device, access_mode::overwrite);

            ArrayHandle<unsigned int> d_scan(m_scan, access_location::device, access_mode::readwrite);

            // access rtags
            ArrayHandle<unsigned int> d_group_rtag(m_gdata->getRTags(), access_location::device, access_mode::readwrite);

            unsigned int ngroups = m_gdata->getN();

            // remove groups from local table
            gpu_remove_groups(ngroups,
                d_groups.data,
                d_groups_alt.data,
                d_group_type.data,
                d_group_type_alt.data,
                d_group_tag.data,
                d_group_tag_alt.data,
                d_group_ranks.data,
                d_group_ranks_alt.data,
                d_group_rtag.data,
                new_ngroups,
                d_scan.data,
                m_gpu_comm.m_mgpu_context);
            if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
            }

        // resize alternate arrays to number of groups
        GPUVector<typename group_data::members_t>& alt_groups_array = m_gdata->getAltMembersArray();
        GPUVector<unsigned int>& alt_group_type_array = m_gdata->getAltTypesArray();
        GPUVector<unsigned int>& alt_group_tag_array = m_gdata->getAltTags();
        GPUVector<typename group_data::ranks_t>& alt_group_ranks_array = m_gdata->getAltRanksArray();

        assert(new_ngroups <= m_gdata->getN());
        alt_groups_array.resize(new_ngroups);
        alt_group_type_array.resize(new_ngroups);
        alt_group_tag_array.resize(new_ngroups);
        alt_group_ranks_array.resize(new_ngroups);

        // make alternate arrays current
        m_gdata->swapMemberArrays();
        m_gdata->swapTypeArrays();
        m_gdata->swapTagArrays();
        m_gdata->swapRankArrays();

        #ifdef ENABLE_MPI_CUDA
        #else
        // fill host send buffers on host
        typedef std::multimap<unsigned int, group_element_t> group_map_t;
        group_map_t group_send_map;

            {
            // access output buffers
            ArrayHandle<group_element_t> h_groups_out(m_groups_out, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_rank_mask_out(m_rank_mask_out, access_location::host, access_mode::read);

            for (unsigned int i = 0; i < n_out_groups; ++i)
                {
                group_element_t el = h_groups_out.data[i];
                typename group_data::ranks_t ranks = el.ranks;

                for (unsigned int j = 0; j < group_data::size; ++j)
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
            ArrayHandle<group_element_t> h_groups_sendbuf(m_groups_sendbuf, access_location::host, access_mode::overwrite);

            // output send data sorted by rank
            unsigned int n = 0;
            for (typename group_map_t::iterator it = group_send_map.begin(); it != group_send_map.end(); ++it)
                {
                h_groups_sendbuf.data[n] = it->second;
                n++;
                }

            ArrayHandle<unsigned int> h_unique_neighbors(m_gpu_comm.m_unique_neighbors, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_begin(m_gpu_comm.m_begin, access_location::host, access_mode::overwrite);
            ArrayHandle<unsigned int> h_end(m_gpu_comm.m_end, access_location::host, access_mode::overwrite);

            // Find start and end indices
            for (unsigned int i = 0; i < m_gpu_comm.m_n_unique_neigh; ++i)
                {
                typename group_map_t::iterator lower = group_send_map.lower_bound(h_unique_neighbors.data[i]);
                typename group_map_t::iterator upper = group_send_map.upper_bound(h_unique_neighbors.data[i]);
                h_begin.data[i] = std::distance(group_send_map.begin(),lower);
                h_end.data[i] = std::distance(group_send_map.begin(),upper);
                }
            }
        #endif

        /*
         * communicate groups (phase 2)
         */

       n_recv_tot = 0;
            {
            ArrayHandle<unsigned int> h_begin(m_gpu_comm.m_begin, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_end(m_gpu_comm.m_end, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_unique_neighbors(m_gpu_comm.m_unique_neighbors, access_location::host, access_mode::read);

            unsigned int send_bytes = 0;
            unsigned int recv_bytes = 0;
            if (m_gpu_comm.m_prof) m_gpu_comm.m_prof->push(m_exec_conf, "MPI send/recv");

            // compute send counts
            for (unsigned int ineigh = 0; ineigh < m_gpu_comm.m_n_unique_neigh; ineigh++)
                n_send_groups[ineigh] = h_end.data[ineigh] - h_begin.data[ineigh];

            MPI_Request req[2*m_gpu_comm.m_n_unique_neigh];
            MPI_Status stat[2*m_gpu_comm.m_n_unique_neigh];

            unsigned int nreq = 0;

            // loop over neighbors
            for (unsigned int ineigh = 0; ineigh < m_gpu_comm.m_n_unique_neigh; ineigh++)
                {
                // rank of neighbor processor
                unsigned int neighbor = h_unique_neighbors.data[ineigh];

                MPI_Isend(&n_send_groups[ineigh], 1, MPI_UNSIGNED, neighbor, 0, m_gpu_comm.m_mpi_comm, & req[nreq++]);
                MPI_Irecv(&n_recv_groups[ineigh], 1, MPI_UNSIGNED, neighbor, 0, m_gpu_comm.m_mpi_comm, & req[nreq++]);
                send_bytes += sizeof(unsigned int);
                recv_bytes += sizeof(unsigned int);
                } // end neighbor loop

            MPI_Waitall(nreq, req, stat);

            // sum up receive counts
            for (unsigned int ineigh = 0; ineigh < m_gpu_comm.m_n_unique_neigh; ineigh++)
                {
                if (ineigh == 0)
                    offs[ineigh] = 0;
                else
                    offs[ineigh] = offs[ineigh-1] + n_recv_groups[ineigh-1];

                n_recv_tot += n_recv_groups[ineigh];
                }

            if (m_gpu_comm.m_prof) m_gpu_comm.m_prof->pop(m_exec_conf,0,send_bytes+recv_bytes);
            }

        // Resize receive buffer
        m_groups_recvbuf.resize(n_recv_tot);

            {
            ArrayHandle<unsigned int> h_begin(m_gpu_comm.m_begin, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_end(m_gpu_comm.m_end, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_unique_neighbors(m_gpu_comm.m_unique_neighbors, access_location::host, access_mode::read);

            if (m_gpu_comm.m_prof) m_gpu_comm.m_prof->push(m_exec_conf,"MPI send/recv");

            #ifdef ENABLE_MPI_CUDA
            ArrayHandle<group_element_t> groups_sendbuf_handle(m_groups_sendbuf, access_location::device, access_mode::read);
            ArrayHandle<group_element_t> groups_recvbuf_handle(m_groups_recvbuf, access_location::device, access_mode::overwrite);
            #else
            ArrayHandle<group_element_t> groups_sendbuf_handle(m_groups_sendbuf, access_location::host, access_mode::read);
            ArrayHandle<group_element_t> groups_recvbuf_handle(m_groups_recvbuf, access_location::host, access_mode::overwrite);
            #endif

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
                    MPI_Isend(groups_sendbuf_handle.data+h_begin.data[ineigh],
                        n_send_groups[ineigh]*sizeof(group_element_t),
                        MPI_BYTE,
                        neighbor,
                        1,
                        m_gpu_comm.m_mpi_comm,
                        &req);
                    reqs.push_back(req);
                    }
                send_bytes+= n_send_groups[ineigh]*sizeof(group_element_t);

                if (n_recv_groups[ineigh])
                    {
                    MPI_Irecv(groups_recvbuf_handle.data+offs[ineigh],
                        n_recv_groups[ineigh]*sizeof(group_element_t),
                        MPI_BYTE,
                        neighbor,
                        1,
                        m_gpu_comm.m_mpi_comm,
                        &req);
                    reqs.push_back(req);
                    }
                recv_bytes += n_recv_groups[ineigh]*sizeof(group_element_t);
                }

            std::vector<MPI_Status> stats(reqs.size());
            MPI_Waitall(reqs.size(), &reqs.front(), &stats.front());

            if (m_gpu_comm.m_prof) m_gpu_comm.m_prof->pop(m_exec_conf,0,send_bytes+recv_bytes);
            }

        unsigned int n_recv_unique = 0;
        #ifdef ENABLE_MPI_CUDA
        #else
            {
            ArrayHandle<group_element_t> h_groups_recvbuf(m_groups_recvbuf, access_location::host, access_mode::read);

            // use a std::map, i.e. single-key, to filter out duplicate groups in input buffer
            typedef std::map<unsigned int, group_element_t> recv_map_t;
            recv_map_t recv_map;

            for (unsigned int recv_idx = 0; recv_idx < n_recv_tot; recv_idx++)
                {
                group_element_t el = h_groups_recvbuf.data[recv_idx];
                unsigned int tag= el.group_tag;
                recv_map.insert(std::make_pair(tag, el));
                }

            // resize input array of unique groups
            m_groups_in.resize(recv_map.size());

            // write out unique groups
            ArrayHandle<group_element_t> h_groups_in(m_groups_in, access_location::host, access_mode::overwrite);
            for (typename recv_map_t::iterator it = recv_map.begin(); it != recv_map.end(); ++it)
                h_groups_in.data[n_recv_unique++] = it->second;
            }
        #endif

        unsigned int old_ngroups = m_gdata->getN();
        new_ngroups = old_ngroups + n_recv_unique;

        // resize group arrays to accomodate additional groups (there can still be duplicates with local groups)
        GPUVector<typename group_data::members_t>& groups_array = m_gdata->getMembersArray();
        GPUVector<unsigned int>& group_type_array = m_gdata->getTypesArray();
        GPUVector<unsigned int>& group_tag_array = m_gdata->getTags();
        GPUVector<typename group_data::ranks_t>& group_ranks_array = m_gdata->getRanksArray();

        groups_array.resize(new_ngroups);
        group_type_array.resize(new_ngroups);
        group_tag_array.resize(new_ngroups);
        group_ranks_array.resize(new_ngroups);

            {
            ArrayHandle<group_element_t> d_groups_in(m_groups_in, access_location::device, access_mode::read);
            ArrayHandle<typename group_data::members_t> d_groups(m_gdata->getMembersArray(), access_location::device, access_mode::readwrite);
            ArrayHandle<unsigned int> d_group_type(m_gdata->getTypesArray(), access_location::device, access_mode::readwrite);
            ArrayHandle<unsigned int> d_group_tag(m_gdata->getTags(), access_location::device, access_mode::readwrite);
            ArrayHandle<typename group_data::ranks_t> d_group_ranks(m_gdata->getRanksArray(), access_location::device, access_mode::readwrite);
            ArrayHandle<unsigned int> d_group_rtag(m_gdata->getRTags(), access_location::device, access_mode::readwrite);

            // get temp buffer
            ScopedAllocation<unsigned int> d_tmp(m_exec_conf->getCachedAllocator(), n_recv_unique);

            // add new groups, updating groups that are already present locally
            gpu_add_groups(old_ngroups,
                n_recv_unique,
                d_groups_in.data,
                d_groups.data,
                d_group_type.data,
                d_group_tag.data,
                d_group_ranks.data,
                d_group_rtag.data,
                new_ngroups,
                d_tmp.data,
                m_gpu_comm.m_mgpu_context);
            if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
            }

        // resize arrays to final size
        groups_array.resize(new_ngroups);
        group_type_array.resize(new_ngroups);
        group_tag_array.resize(new_ngroups);
        group_ranks_array.resize(new_ngroups);

        // indicate that group table has changed
        m_gdata->setDirty();

        if (m_gpu_comm.m_prof) m_gpu_comm.m_prof->pop(m_exec_conf);
        }
    }


//! Mark ghost particles
template<class group_data>
void CommunicatorGPU::GroupCommunicatorGPU<group_data>::markGhostParticles(
    const GPUArray<unsigned int>& plans,
    unsigned int mask)
    {
    if (m_gdata->getNGlobal())
        {
        ArrayHandle<typename group_data::members_t> d_groups(m_gdata->getMembersArray(), access_location::device, access_mode::read);
        ArrayHandle<typename group_data::ranks_t> d_group_ranks(m_gdata->getRanksArray(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_rtag(m_gpu_comm.m_pdata->getRTags(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_plan(plans, access_location::device, access_mode::readwrite);

        Index3D di = m_gpu_comm.m_pdata->getDomainDecomposition()->getDomainIndexer();
        uint3 my_pos = di.getTriple(m_exec_conf->getRank());

        gpu_mark_bonded_ghosts<group_data::size>(
            m_gdata->getN(),
            d_groups.data,
            d_group_ranks.data,
            d_rtag.data,
            d_plan.data,
            di,
            my_pos,
            mask);
        if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
        }
    }

//! Transfer particles between neighboring domains
void CommunicatorGPU::migrateParticles()
    {
    if (m_prof)
        m_prof->push(m_exec_conf,"comm_migrate");

    m_exec_conf->msg->notice(7) << "CommunicatorGPU: migrate particles" << std::endl;

    if (m_last_flags[comm_flag::tag])
        {
        // Reset reverse lookup tags of old ghost atoms
        ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::readwrite);
        ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);

        gpu_reset_rtags(m_pdata->getNGhosts(),
                        d_tag.data + m_pdata->getN(),
                        d_rtag.data);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    // reset ghost particle number
    m_pdata->removeAllGhostParticles();

    // main communication loop
    for (unsigned int stage = 0; stage < m_num_stages; stage++)
        {
            {
            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_comm_flag(m_pdata->getCommFlags(), access_location::device, access_mode::readwrite);

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
        m_bond_comm.migrateGroups(m_bonds_changed);
        m_bonds_changed = false;

        // Angles
        m_angle_comm.migrateGroups(m_angles_changed);
        m_angles_changed = false;

        // Dihedrals
        m_dihedral_comm.migrateGroups(m_dihedrals_changed);
        m_dihedrals_changed = false;

        // Dihedrals
        m_improper_comm.migrateGroups(m_impropers_changed);
        m_impropers_changed = false;

        // fill send buffer
        m_pdata->removeParticlesGPU(m_gpu_sendbuf, m_comm_flags);

        const Index3D& di = m_decomposition->getDomainIndexer();
        // determine local particles that are to be sent to neighboring processors and fill send buffer
        uint3 mypos = di.getTriple(m_exec_conf->getRank());

        /* We need some better heuristics to decide whether to take the GPU or CPU code path */
        #ifdef ENABLE_MPI_CUDA
            {
            // resize keys
            m_send_keys.resize(m_gpu_sendbuf.size());

            ArrayHandle<pdata_element> d_gpu_sendbuf(m_gpu_sendbuf, access_location::device, access_mode::readwrite);
            ArrayHandle<unsigned int> d_send_keys(m_send_keys, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_begin(m_begin, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_end(m_end, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_unique_neighbors(m_unique_neighbors, access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_comm_flags(m_comm_flags, access_location::device, access_mode::read);

            // get temporary buffers
            unsigned int nsend = m_gpu_sendbuf.size();
            CachedAllocator& alloc = m_exec_conf->getCachedAllocator();
            ScopedAllocation<pdata_element> d_in_copy(alloc, nsend);
            ScopedAllocation<unsigned int> d_tmp(alloc, nsend);

            gpu_sort_migrating_particles(m_gpu_sendbuf.size(),
                       d_gpu_sendbuf.data,
                       d_comm_flags.data,
                       di,
                       mypos,
                       m_pdata->getBox(),
                       d_send_keys.data,
                       d_begin.data,
                       d_end.data,
                       d_unique_neighbors.data,
                       m_n_unique_neigh,
                       m_comm_mask[stage],
                       m_mgpu_context,
                       d_tmp,
                       d_in_copy);

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }
        #else
            {
            ArrayHandle<pdata_element> h_gpu_sendbuf(m_gpu_sendbuf, access_location::host, access_mode::readwrite);
            ArrayHandle<unsigned int> h_begin(m_begin, access_location::host, access_mode::overwrite);
            ArrayHandle<unsigned int> h_end(m_end, access_location::host, access_mode::overwrite);
            ArrayHandle<unsigned int> h_unique_neighbors(m_unique_neighbors, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_comm_flags(m_comm_flags, access_location::host, access_mode::read);

            typedef std::multimap<unsigned int,pdata_element> key_t;
            key_t keys;

            // generate keys
            get_migrate_key t(mypos, di, m_comm_mask[stage]);
            for (unsigned int i = 0; i < m_comm_flags.size(); ++i)
                keys.insert(std::pair<unsigned int, pdata_element>(t(h_comm_flags.data[i]),h_gpu_sendbuf.data[i]));

            // Find start and end indices
            for (unsigned int i = 0; i < m_n_unique_neigh; ++i)
                {
                key_t::iterator lower = keys.lower_bound(h_unique_neighbors.data[i]);
                key_t::iterator upper = keys.upper_bound(h_unique_neighbors.data[i]);
                h_begin.data[i] = std::distance(keys.begin(),lower);
                h_end.data[i] = std::distance(keys.begin(),upper);
                }

            // sort send buffer
            unsigned int i = 0;
            for (key_t::iterator it = keys.begin(); it != keys.end(); ++it)
                h_gpu_sendbuf.data[i++] = it->second;
            }
        #endif

        unsigned int n_send_ptls[m_n_unique_neigh];
        unsigned int n_recv_ptls[m_n_unique_neigh];
        unsigned int offs[m_n_unique_neigh];
        unsigned int n_recv_tot = 0;

            {
            ArrayHandle<unsigned int> h_begin(m_begin, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_end(m_end, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_unique_neighbors(m_unique_neighbors, access_location::host, access_mode::read);

            unsigned int send_bytes = 0;
            unsigned int recv_bytes = 0;
            if (m_prof) m_prof->push(m_exec_conf, "MPI send/recv");

            // compute send counts
            for (unsigned int ineigh = 0; ineigh < m_n_unique_neigh; ineigh++)
                n_send_ptls[ineigh] = h_end.data[ineigh] - h_begin.data[ineigh];

            MPI_Request req[2*m_n_unique_neigh];
            MPI_Status stat[2*m_n_unique_neigh];

            unsigned int nreq = 0;

            // loop over neighbors
            for (unsigned int ineigh = 0; ineigh < m_n_unique_neigh; ineigh++)
                {
                if (m_stages[ineigh] != (int) stage)
                    {
                    // skip neighbor if not participating in this communication stage
                    n_send_ptls[ineigh] = 0;
                    n_recv_ptls[ineigh] = 0;
                    continue;
                    }

                // rank of neighbor processor
                unsigned int neighbor = h_unique_neighbors.data[ineigh];

                MPI_Isend(&n_send_ptls[ineigh], 1, MPI_UNSIGNED, neighbor, 0, m_mpi_comm, & req[nreq++]);
                MPI_Irecv(&n_recv_ptls[ineigh], 1, MPI_UNSIGNED, neighbor, 0, m_mpi_comm, & req[nreq++]);
                send_bytes += sizeof(unsigned int);
                recv_bytes += sizeof(unsigned int);
                } // end neighbor loop

            MPI_Waitall(nreq, req, stat);

            // sum up receive counts
            for (unsigned int ineigh = 0; ineigh < m_n_unique_neigh; ineigh++)
                {
                if (ineigh == 0)
                    offs[ineigh] = 0;
                else
                    offs[ineigh] = offs[ineigh-1] + n_recv_ptls[ineigh-1];

                n_recv_tot += n_recv_ptls[ineigh];
                }

            if (m_prof) m_prof->pop(m_exec_conf,0,send_bytes+recv_bytes);
            }

        // Resize receive buffer
        m_gpu_recvbuf.resize(n_recv_tot);

            {
            ArrayHandle<unsigned int> h_begin(m_begin, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_end(m_end, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_unique_neighbors(m_unique_neighbors, access_location::host, access_mode::read);

            if (m_prof) m_prof->push(m_exec_conf,"MPI send/recv");

            #ifdef ENABLE_MPI_CUDA
            ArrayHandle<pdata_element> gpu_sendbuf_handle(m_gpu_sendbuf, access_location::device, access_mode::read);
            ArrayHandle<pdata_element> gpu_recvbuf_handle(m_gpu_recvbuf, access_location::device, access_mode::overwrite);
            #else
            ArrayHandle<pdata_element> gpu_sendbuf_handle(m_gpu_sendbuf, access_location::host, access_mode::read);
            ArrayHandle<pdata_element> gpu_recvbuf_handle(m_gpu_recvbuf, access_location::host, access_mode::overwrite);
            #endif

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
                    MPI_Isend(gpu_sendbuf_handle.data+h_begin.data[ineigh],
                        n_send_ptls[ineigh]*sizeof(pdata_element),
                        MPI_BYTE,
                        neighbor,
                        1,
                        m_mpi_comm,
                        &req);
                    reqs.push_back(req);
                    }
                send_bytes+= n_send_ptls[ineigh]*sizeof(pdata_element);

                if (n_recv_ptls[ineigh])
                    {
                    MPI_Irecv(gpu_recvbuf_handle.data+offs[ineigh],
                        n_recv_ptls[ineigh]*sizeof(pdata_element),
                        MPI_BYTE,
                        neighbor,
                        1,
                        m_mpi_comm,
                        &req);
                    reqs.push_back(req);
                    }
                recv_bytes += n_recv_ptls[ineigh]*sizeof(pdata_element);
                }

            std::vector<MPI_Status> stats(reqs.size());
            MPI_Waitall(reqs.size(), &reqs.front(), &stats.front());

            if (m_prof) m_prof->pop(m_exec_conf,0,send_bytes+recv_bytes);
            }

            {
            ArrayHandle<pdata_element> d_gpu_recvbuf(m_gpu_recvbuf, access_location::device, access_mode::readwrite);
            const BoxDim shifted_box = getShiftedBox();

            // Apply boundary conditions
            gpu_wrap_particles(n_recv_tot,
                               d_gpu_recvbuf.data,
                               shifted_box);
            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

        // remove particles that were sent and fill particle data with received particles
        m_pdata->addParticlesGPU(m_gpu_recvbuf);

        } // end communication stage

    if (m_prof) m_prof->pop(m_exec_conf);
    }

//! Build a ghost particle list, exchange ghost particle data with neighboring processors
void CommunicatorGPU::exchangeGhosts()
    {
    if (m_prof) m_prof->push(m_exec_conf, "comm_ghost_exch");

    m_exec_conf->msg->notice(7) << "CommunicatorGPU: ghost exchange" << std::endl;

    // the ghost layer must be at_least m_r_ghost wide along every lattice direction
    Scalar3 ghost_fraction = m_r_ghost/m_pdata->getBox().getNearestPlaneDistance();

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

    m_ghost_begin.resize(m_n_unique_neigh*m_num_stages);
    m_ghost_end.resize(m_n_unique_neigh*m_num_stages);

    m_idx_offs.resize(m_num_stages);

    // get requested ghost fields
    CommFlags flags = getFlags();

    // main communication loop
    for (unsigned int stage = 0; stage < m_num_stages; stage++)
        {
        // make room for plans
        m_ghost_plan.resize(m_pdata->getN()+m_pdata->getNGhosts());

            {
            // compute plans for all particles, including already received ghosts
            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_ghost_plan(m_ghost_plan, access_location::device, access_mode::overwrite);

            gpu_make_ghost_exchange_plan(d_ghost_plan.data,
                                         m_pdata->getN()+m_pdata->getNGhosts(),
                                         d_pos.data,
                                         m_pdata->getBox(),
                                         ghost_fraction,
                                         m_comm_mask[stage]);

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

        // mark particles that are members of incomplete of bonded groups as ghost

        // bonds
        m_bond_comm.markGhostParticles(m_ghost_plan,m_comm_mask[stage]);

        // angles
        m_angle_comm.markGhostParticles(m_ghost_plan,m_comm_mask[stage]);

        // dihedrals
        m_dihedral_comm.markGhostParticles(m_ghost_plan,m_comm_mask[stage]);

        // impropers
        m_improper_comm.markGhostParticles(m_ghost_plan,m_comm_mask[stage]);

        // resize temporary number of neighbors array
        m_neigh_counts.resize(m_pdata->getN()+m_pdata->getNGhosts());

            {
            ArrayHandle<unsigned int> d_ghost_plan(m_ghost_plan, access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_adj_mask(m_adj_mask, access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_neigh_counts(m_neigh_counts, access_location::device, access_mode::overwrite);

            // count number of neighbors (total and per particle) the ghost ptls are sent to
            m_n_send_ghosts_tot[stage] =
                gpu_exchange_ghosts_count_neighbors(
                    m_pdata->getN()+m_pdata->getNGhosts(),
                    d_ghost_plan.data,
                    d_adj_mask.data,
                    d_neigh_counts.data,
                    m_n_unique_neigh,
                    m_mgpu_context);

            if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
            }

        // compute offset into ghost idx list
        m_idx_offs[stage] = 0;
        for (unsigned int i = 0; i < stage; ++i)
            m_idx_offs[stage] +=  m_n_send_ghosts_tot[i];

        // compute maximum send buf size
        unsigned int n_max = 0;
        for (unsigned int istage = 0; istage <= stage; ++istage)
            if (m_n_send_ghosts_tot[istage] > n_max) n_max = m_n_send_ghosts_tot[istage];

        // make room for ghost indices and neighbor ranks
        m_ghost_idx.resize(m_idx_offs[stage] + m_n_send_ghosts_tot[stage]);
        m_ghost_neigh.resize(m_idx_offs[stage] + m_n_send_ghosts_tot[stage]);

        if (flags[comm_flag::tag]) m_tag_ghost_sendbuf.resize(n_max);
        if (flags[comm_flag::position]) m_pos_ghost_sendbuf.resize(n_max);
        if (flags[comm_flag::velocity]) m_vel_ghost_sendbuf.resize(n_max);
        if (flags[comm_flag::charge]) m_charge_ghost_sendbuf.resize(n_max);
        if (flags[comm_flag::diameter]) m_diameter_ghost_sendbuf.resize(n_max);
        if (flags[comm_flag::orientation]) m_orientation_ghost_sendbuf.resize(n_max);

            {
            ArrayHandle<unsigned int> d_ghost_plan(m_ghost_plan, access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_adj_mask(m_adj_mask, access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_neigh_counts(m_neigh_counts, access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_unique_neighbors(m_unique_neighbors, access_location::device, access_mode::read);

            ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);

            ArrayHandle<unsigned int> d_ghost_idx(m_ghost_idx, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_ghost_neigh(m_ghost_neigh, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_ghost_begin(m_ghost_begin, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_ghost_end(m_ghost_end, access_location::device, access_mode::overwrite);

            //! Fill ghost send list and compute start and end indices per unique neighbor in list
            gpu_exchange_ghosts_make_indices(
                m_pdata->getN() + m_pdata->getNGhosts(),
                d_ghost_plan.data,
                d_tag.data,
                d_adj_mask.data,
                d_unique_neighbors.data,
                d_neigh_counts.data,
                d_ghost_idx.data + m_idx_offs[stage],
                d_ghost_neigh.data + m_idx_offs[stage],
                d_ghost_begin.data + stage*m_n_unique_neigh,
                d_ghost_end.data + stage*m_n_unique_neigh,
                m_n_unique_neigh,
                m_n_send_ghosts_tot[stage],
                m_comm_mask[stage],
                m_mgpu_context);

            if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
            }

            {
            // access particle data
            ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);
            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
            ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::read);
            ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::read);
            ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::read);
            ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::read);

            // access ghost send indices
            ArrayHandle<unsigned int> d_ghost_idx(m_ghost_idx, access_location::device, access_mode::read);

            // access output buffers
            ArrayHandle<unsigned int> d_tag_ghost_sendbuf(m_tag_ghost_sendbuf, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar4> d_pos_ghost_sendbuf(m_pos_ghost_sendbuf, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar4> d_vel_ghost_sendbuf(m_vel_ghost_sendbuf, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar> d_charge_ghost_sendbuf(m_charge_ghost_sendbuf, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar> d_diameter_ghost_sendbuf(m_diameter_ghost_sendbuf, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar4> d_orientation_ghost_sendbuf(m_orientation_ghost_sendbuf, access_location::device, access_mode::overwrite);

            // Pack ghosts into send buffers
            gpu_exchange_ghosts_pack(
                m_n_send_ghosts_tot[stage],
                d_ghost_idx.data + m_idx_offs[stage],
                d_tag.data,
                d_pos.data,
                d_vel.data,
                d_charge.data,
                d_diameter.data,
                d_orientation.data,
                d_tag_ghost_sendbuf.data,
                d_pos_ghost_sendbuf.data,
                d_vel_ghost_sendbuf.data,
                d_charge_ghost_sendbuf.data,
                d_diameter_ghost_sendbuf.data,
                d_orientation_ghost_sendbuf.data,
                flags[comm_flag::tag],
                flags[comm_flag::position],
                flags[comm_flag::velocity],
                flags[comm_flag::charge],
                flags[comm_flag::diameter],
                flags[comm_flag::orientation]);

            if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
            }

        /*
         * Ghost particle communication
         */
        m_n_recv_ghosts_tot[stage] = 0;

        unsigned int send_bytes = 0;
        unsigned int recv_bytes = 0;

            {
            ArrayHandle<unsigned int> h_ghost_begin(m_ghost_begin, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_ghost_end(m_ghost_end, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_unique_neighbors(m_unique_neighbors, access_location::host, access_mode::read);

            if (m_prof) m_prof->push(m_exec_conf, "MPI send/recv");

            // compute send counts
            for (unsigned int ineigh = 0; ineigh < m_n_unique_neigh; ineigh++)
                m_n_send_ghosts[stage][ineigh] = h_ghost_end.data[ineigh+stage*m_n_unique_neigh]
                    - h_ghost_begin.data[ineigh+stage*m_n_unique_neigh];

            MPI_Request req[2*m_n_unique_neigh];
            MPI_Status stat[2*m_n_unique_neigh];

            unsigned int nreq = 0;

            // loop over neighbors
            for (unsigned int ineigh = 0; ineigh < m_n_unique_neigh; ineigh++)
                {
                if (m_stages[ineigh] != (int) stage)
                    {
                    // skip neighbor if not participating in this communication stage
                    m_n_send_ghosts[stage][ineigh] = 0;
                    m_n_recv_ghosts[stage][ineigh] = 0;
                    continue;
                    }

                // rank of neighbor processor
                unsigned int neighbor = h_unique_neighbors.data[ineigh];

                MPI_Isend(&m_n_send_ghosts[stage][ineigh], 1, MPI_UNSIGNED, neighbor, 0, m_mpi_comm, & req[nreq++]);
                MPI_Irecv(&m_n_recv_ghosts[stage][ineigh], 1, MPI_UNSIGNED, neighbor, 0, m_mpi_comm, & req[nreq++]);

                send_bytes += sizeof(unsigned int);
                recv_bytes += sizeof(unsigned int);
                }

            MPI_Waitall(nreq, req, stat);

            // total up receive counts
            for (unsigned int ineigh = 0; ineigh < m_n_unique_neigh; ineigh++)
                {
                if (ineigh == 0)
                    m_ghost_offs[stage][ineigh] = 0;
                else
                    m_ghost_offs[stage][ineigh] = m_ghost_offs[stage][ineigh-1] + m_n_recv_ghosts[stage][ineigh-1];

                m_n_recv_ghosts_tot[stage] += m_n_recv_ghosts[stage][ineigh];
                }

            if (m_prof) m_prof->pop(m_exec_conf,0,send_bytes+recv_bytes);
            }

        n_max = 0;
        // compute maximum number of received ghosts
        for (unsigned int istage = 0; istage <= stage; ++istage)
            if (m_n_recv_ghosts_tot[istage] > n_max) n_max = m_n_recv_ghosts_tot[istage];

        if (flags[comm_flag::tag]) m_tag_ghost_recvbuf.resize(n_max);
        if (flags[comm_flag::position]) m_pos_ghost_recvbuf.resize(n_max);
        if (flags[comm_flag::velocity]) m_vel_ghost_recvbuf.resize(n_max);
        if (flags[comm_flag::charge]) m_charge_ghost_recvbuf.resize(n_max);
        if (flags[comm_flag::diameter]) m_diameter_ghost_recvbuf.resize(n_max);
        if (flags[comm_flag::orientation]) m_orientation_ghost_recvbuf.resize(n_max);

            {
            #ifdef ENABLE_MPI_CUDA
            // recv buffers
            ArrayHandle<unsigned int> tag_ghost_recvbuf_handle(m_tag_ghost_recvbuf, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar4> pos_ghost_recvbuf_handle(m_pos_ghost_recvbuf, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar4> vel_ghost_recvbuf_handle(m_vel_ghost_recvbuf, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar> charge_ghost_recvbuf_handle(m_charge_ghost_recvbuf, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar> diameter_ghost_recvbuf_handle(m_diameter_ghost_recvbuf, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar4> orientation_ghost_recvbuf_handle(m_orientation_ghost_recvbuf, access_location::device, access_mode::overwrite);

            // send buffers
            ArrayHandle<unsigned int> tag_ghost_sendbuf_handle(m_tag_ghost_sendbuf, access_location::device, access_mode::read);
            ArrayHandle<Scalar4> pos_ghost_sendbuf_handle(m_pos_ghost_sendbuf, access_location::device, access_mode::read);
            ArrayHandle<Scalar4> vel_ghost_sendbuf_handle(m_vel_ghost_sendbuf, access_location::device, access_mode::read);
            ArrayHandle<Scalar> charge_ghost_sendbuf_handle(m_charge_ghost_sendbuf, access_location::device, access_mode::read);
            ArrayHandle<Scalar> diameter_ghost_sendbuf_handle(m_diameter_ghost_sendbuf, access_location::device, access_mode::read);
            ArrayHandle<Scalar4> orientation_ghost_sendbuf_handle(m_orientation_ghost_sendbuf, access_location::device, access_mode::read);
            #else
            // recv buffers
            ArrayHandleAsync<unsigned int> tag_ghost_recvbuf_handle(m_tag_ghost_recvbuf, access_location::host, access_mode::overwrite);
            ArrayHandleAsync<Scalar4> pos_ghost_recvbuf_handle(m_pos_ghost_recvbuf, access_location::host, access_mode::overwrite);
            ArrayHandleAsync<Scalar4> vel_ghost_recvbuf_handle(m_vel_ghost_recvbuf, access_location::host, access_mode::overwrite);
            ArrayHandleAsync<Scalar> charge_ghost_recvbuf_handle(m_charge_ghost_recvbuf, access_location::host, access_mode::overwrite);
            ArrayHandleAsync<Scalar> diameter_ghost_recvbuf_handle(m_diameter_ghost_recvbuf, access_location::host, access_mode::overwrite);
            ArrayHandleAsync<Scalar4> orientation_ghost_recvbuf_handle(m_orientation_ghost_recvbuf, access_location::host, access_mode::overwrite);
            // send buffers
            ArrayHandleAsync<unsigned int> tag_ghost_sendbuf_handle(m_tag_ghost_sendbuf, access_location::host, access_mode::read);
            ArrayHandleAsync<Scalar4> pos_ghost_sendbuf_handle(m_pos_ghost_sendbuf, access_location::host, access_mode::read);
            ArrayHandleAsync<Scalar4> vel_ghost_sendbuf_handle(m_vel_ghost_sendbuf, access_location::host, access_mode::read);
            ArrayHandleAsync<Scalar> charge_ghost_sendbuf_handle(m_charge_ghost_sendbuf, access_location::host, access_mode::read);
            ArrayHandleAsync<Scalar> diameter_ghost_sendbuf_handle(m_diameter_ghost_sendbuf, access_location::host, access_mode::read);
            ArrayHandleAsync<Scalar4> orientation_ghost_sendbuf_handle(m_orientation_ghost_sendbuf, access_location::host, access_mode::read);

            // lump together into one synchronization call
            cudaDeviceSynchronize();
            #endif

            ArrayHandle<unsigned int> h_unique_neighbors(m_unique_neighbors, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_ghost_begin(m_ghost_begin, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_ghost_end(m_ghost_end, access_location::host, access_mode::read);

            if (m_prof) m_prof->push(m_exec_conf, "MPI send/recv");

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
                        MPI_Isend(tag_ghost_sendbuf_handle.data+h_ghost_begin.data[ineigh+stage*m_n_unique_neigh],
                            m_n_send_ghosts[stage][ineigh]*sizeof(unsigned int),
                            MPI_BYTE,
                            neighbor,
                            1,
                            m_mpi_comm,
                            &req);
                        reqs.push_back(req);
                        }
                    send_bytes += m_n_send_ghosts[stage][ineigh]*sizeof(unsigned int);
                    if (m_n_recv_ghosts[stage][ineigh])
                        {
                        MPI_Irecv(tag_ghost_recvbuf_handle.data + m_ghost_offs[stage][ineigh],
                            m_n_recv_ghosts[stage][ineigh]*sizeof(unsigned int),
                            MPI_BYTE,
                            neighbor,
                            1,
                            m_mpi_comm,
                            &req);
                        reqs.push_back(req);
                        }
                    recv_bytes += m_n_recv_ghosts[stage][ineigh]*sizeof(unsigned int);
                    }

                if (flags[comm_flag::position])
                    {
                    MPI_Request req;
                    if (m_n_send_ghosts[stage][ineigh])
                        {
                        MPI_Isend(pos_ghost_sendbuf_handle.data+h_ghost_begin.data[ineigh + stage*m_n_unique_neigh],
                            m_n_send_ghosts[stage][ineigh]*sizeof(Scalar4),
                            MPI_BYTE,
                            neighbor,
                            2,
                            m_mpi_comm,
                            &req);
                        reqs.push_back(req);
                        }
                    send_bytes += m_n_send_ghosts[stage][ineigh]*sizeof(Scalar4);
                    if (m_n_recv_ghosts[stage][ineigh])
                        {
                        MPI_Irecv(pos_ghost_recvbuf_handle.data + m_ghost_offs[stage][ineigh],
                            m_n_recv_ghosts[stage][ineigh]*sizeof(Scalar4),
                            MPI_BYTE,
                            neighbor,
                            2,
                            m_mpi_comm,
                            &req);
                        reqs.push_back(req);
                        }
                    recv_bytes += m_n_recv_ghosts[stage][ineigh]*sizeof(unsigned int);
                    }

                if (flags[comm_flag::velocity])
                    {
                    if (m_n_send_ghosts[stage][ineigh])
                        {
                        MPI_Isend(vel_ghost_sendbuf_handle.data+h_ghost_begin.data[ineigh + stage*m_n_unique_neigh],
                            m_n_send_ghosts[stage][ineigh]*sizeof(Scalar4),
                            MPI_BYTE,
                            neighbor,
                            3,
                            m_mpi_comm,
                            &req);
                        reqs.push_back(req);
                        }
                    send_bytes += m_n_send_ghosts[stage][ineigh]*sizeof(Scalar4);
                    if (m_n_recv_ghosts[stage][ineigh])
                        {
                        MPI_Irecv(vel_ghost_recvbuf_handle.data + m_ghost_offs[stage][ineigh],
                            m_n_recv_ghosts[stage][ineigh]*sizeof(Scalar4),
                            MPI_BYTE,
                            neighbor,
                            3,
                            m_mpi_comm,
                            &req);
                        reqs.push_back(req);
                        }
                    recv_bytes += m_n_recv_ghosts[stage][ineigh]*sizeof(Scalar4);
                    }

                if (flags[comm_flag::charge])
                    {
                    if (m_n_send_ghosts[stage][ineigh])
                        {
                        MPI_Isend(charge_ghost_sendbuf_handle.data+h_ghost_begin.data[ineigh + stage*m_n_unique_neigh],
                            m_n_send_ghosts[stage][ineigh]*sizeof(Scalar),
                            MPI_BYTE,
                            neighbor,
                            4,
                            m_mpi_comm,
                            &req);
                        reqs.push_back(req);
                        }
                    send_bytes += m_n_send_ghosts[stage][ineigh]*sizeof(Scalar);
                    if (m_n_recv_ghosts[stage][ineigh])
                        {
                        MPI_Irecv(charge_ghost_recvbuf_handle.data + m_ghost_offs[stage][ineigh],
                            m_n_recv_ghosts[stage][ineigh]*sizeof(Scalar),
                            MPI_BYTE,
                            neighbor,
                            4,
                            m_mpi_comm,
                            &req);
                        reqs.push_back(req);
                        }
                    recv_bytes += m_n_recv_ghosts[stage][ineigh]*sizeof(Scalar);
                    }

                if (flags[comm_flag::diameter])
                    {
                    if (m_n_send_ghosts[stage][ineigh])
                        {
                        MPI_Isend(diameter_ghost_sendbuf_handle.data+h_ghost_begin.data[ineigh + stage*m_n_unique_neigh],
                            m_n_send_ghosts[stage][ineigh]*sizeof(Scalar),
                            MPI_BYTE,
                            neighbor,
                            5,
                            m_mpi_comm,
                            &req);
                        reqs.push_back(req);
                        }
                    send_bytes += m_n_send_ghosts[stage][ineigh]*sizeof(Scalar);
                    if (m_n_recv_ghosts[stage][ineigh])
                        {
                        MPI_Irecv(diameter_ghost_recvbuf_handle.data + m_ghost_offs[stage][ineigh],
                            m_n_recv_ghosts[stage][ineigh]*sizeof(Scalar),
                            MPI_BYTE,
                            neighbor,
                            5,
                            m_mpi_comm,
                            &req);
                        reqs.push_back(req);
                        }
                    recv_bytes += m_n_recv_ghosts[stage][ineigh]*sizeof(Scalar);
                    }

                if (flags[comm_flag::orientation])
                    {
                    if (m_n_send_ghosts[stage][ineigh])
                        {
                        MPI_Isend(orientation_ghost_sendbuf_handle.data+h_ghost_begin.data[ineigh + stage*m_n_unique_neigh],
                            m_n_send_ghosts[stage][ineigh]*sizeof(Scalar4),
                            MPI_BYTE,
                            neighbor,
                            6,
                            m_mpi_comm,
                            &req);
                        reqs.push_back(req);
                        }
                    send_bytes += m_n_send_ghosts[stage][ineigh]*sizeof(Scalar4);
                    if (m_n_recv_ghosts[stage][ineigh])
                        {
                        MPI_Irecv(orientation_ghost_recvbuf_handle.data + m_ghost_offs[stage][ineigh],
                            m_n_recv_ghosts[stage][ineigh]*sizeof(Scalar4),
                            MPI_BYTE,
                            neighbor,
                            6,
                            m_mpi_comm,
                            &req);
                        reqs.push_back(req);
                        }
                    recv_bytes += m_n_recv_ghosts[stage][ineigh]*sizeof(Scalar4);
                    }
                } // end neighbor loop

            std::vector<MPI_Status> stats(reqs.size());
            MPI_Waitall(reqs.size(), &reqs.front(), &stats.front());

            if (m_prof) m_prof->pop(m_exec_conf,0,send_bytes+recv_bytes);
            } // end ArrayHandle scope

        // first ghost ptl index
        unsigned int first_idx = m_pdata->getN()+m_pdata->getNGhosts();

        // update number of ghost particles
        m_pdata->addGhostParticles(m_n_recv_ghosts_tot[stage]);

            {
            // access receive buffers
            ArrayHandle<unsigned int> d_tag_ghost_recvbuf(m_tag_ghost_recvbuf, access_location::device, access_mode::read);
            ArrayHandle<Scalar4> d_pos_ghost_recvbuf(m_pos_ghost_recvbuf, access_location::device, access_mode::read);
            ArrayHandle<Scalar4> d_vel_ghost_recvbuf(m_vel_ghost_recvbuf, access_location::device, access_mode::read);
            ArrayHandle<Scalar> d_charge_ghost_recvbuf(m_charge_ghost_recvbuf, access_location::device, access_mode::read);
            ArrayHandle<Scalar> d_diameter_ghost_recvbuf(m_diameter_ghost_recvbuf, access_location::device, access_mode::read);
            ArrayHandle<Scalar4> d_orientation_ghost_recvbuf(m_orientation_ghost_recvbuf, access_location::device, access_mode::read);
            // access particle data
            ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::readwrite);

            // copy recv buf into particle data
            gpu_exchange_ghosts_copy_buf(
                m_n_recv_ghosts_tot[stage],
                d_tag_ghost_recvbuf.data,
                d_pos_ghost_recvbuf.data,
                d_vel_ghost_recvbuf.data,
                d_charge_ghost_recvbuf.data,
                d_diameter_ghost_recvbuf.data,
                d_orientation_ghost_recvbuf.data,
                d_tag.data + first_idx,
                d_pos.data + first_idx,
                d_vel.data + first_idx,
                d_charge.data + first_idx,
                d_diameter.data + first_idx,
                d_orientation.data + first_idx,
                flags[comm_flag::tag],
                flags[comm_flag::position],
                flags[comm_flag::velocity],
                flags[comm_flag::charge],
                flags[comm_flag::diameter],
                flags[comm_flag::orientation]);

            if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
            }

        if (flags[comm_flag::position])
            {
            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);

            // shifted global box
            const BoxDim& shifted_box = getShiftedBox();

            // wrap received ghosts
            gpu_wrap_ghosts(m_n_recv_ghosts_tot[stage],
                d_pos.data+first_idx,
                shifted_box);

            if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
            }

        if (flags[comm_flag::tag])
            {
            ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::readwrite);

            // update reverse-lookup table
            gpu_compute_ghost_rtags(first_idx,
                m_n_recv_ghosts_tot[stage],
                d_tag.data + first_idx,
                d_rtag.data);
            if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
            }
        } // end main communication loop

    m_last_flags = flags;

    if (m_prof) m_prof->pop(m_exec_conf);

    // we have updated ghost particles, so notify subscribers about this
    m_pdata->notifyGhostParticleNumberChange();
    }

//! Perform ghosts update
void CommunicatorGPU::updateGhosts(unsigned int timestep)
    {
    m_exec_conf->msg->notice(7) << "CommunicatorGPU: ghost update" << std::endl;

    if (m_prof) m_prof->push(m_exec_conf, "comm_ghost_update");

    CommFlags flags = getFlags();

    // main communication loop
    for (unsigned int stage = 0; stage < m_num_stages; ++stage)
        {
            {
            // access particle data
            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
            ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::read);
            ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::read);

            // access ghost send indices
            ArrayHandle<unsigned int> d_ghost_idx(m_ghost_idx, access_location::device, access_mode::read);

            // access output buffers
            ArrayHandle<unsigned int> d_tag_ghost_sendbuf(m_tag_ghost_sendbuf, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar4> d_pos_ghost_sendbuf(m_pos_ghost_sendbuf, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar4> d_vel_ghost_sendbuf(m_vel_ghost_sendbuf, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar4> d_orientation_ghost_sendbuf(m_orientation_ghost_sendbuf, access_location::device, access_mode::overwrite);

            // Pack ghosts into send buffers
            gpu_exchange_ghosts_pack(
                m_n_send_ghosts_tot[stage],
                d_ghost_idx.data + m_idx_offs[stage],
                NULL,
                d_pos.data,
                d_vel.data,
                NULL,
                NULL,
                d_orientation.data,
                NULL,
                d_pos_ghost_sendbuf.data,
                d_vel_ghost_sendbuf.data,
                NULL,
                NULL,
                d_orientation_ghost_sendbuf.data,
                false,
                flags[comm_flag::position],
                flags[comm_flag::velocity],
                false,
                false,
                flags[comm_flag::orientation]);

            if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
            }

        /*
         * Ghost particle communication
         */

            {
            // access particle data
            #ifdef ENABLE_MPI_CUDA
            // recv buffers
            ArrayHandle<Scalar4> pos_ghost_recvbuf_handle(m_pos_ghost_recvbuf, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar4> vel_ghost_recvbuf_handle(m_vel_ghost_recvbuf, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar4> orientation_ghost_recvbuf_handle(m_orientation_ghost_recvbuf, access_location::device, access_mode::overwrite);

            // send buffers
            ArrayHandle<Scalar4> pos_ghost_sendbuf_handle(m_pos_ghost_sendbuf, access_location::device, access_mode::read);
            ArrayHandle<Scalar4> vel_ghost_sendbuf_handle(m_vel_ghost_sendbuf, access_location::device, access_mode::read);
            ArrayHandle<Scalar4> orientation_ghost_sendbuf_handle(m_orientation_ghost_sendbuf, access_location::device, access_mode::read);

            ArrayHandle<unsigned int> h_unique_neighbors(m_unique_neighbors, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_ghost_begin(m_ghost_begin, access_location::host, access_mode::read);
            #else
            // recv buffers
            ArrayHandleAsync<Scalar4> pos_ghost_recvbuf_handle(m_pos_ghost_recvbuf, access_location::host, access_mode::overwrite);
            ArrayHandleAsync<Scalar4> vel_ghost_recvbuf_handle(m_vel_ghost_recvbuf, access_location::host, access_mode::overwrite);
            ArrayHandleAsync<Scalar4> orientation_ghost_recvbuf_handle(m_orientation_ghost_recvbuf, access_location::host, access_mode::overwrite);

            // send buffers
            ArrayHandleAsync<Scalar4> pos_ghost_sendbuf_handle(m_pos_ghost_sendbuf, access_location::host, access_mode::read);
            ArrayHandleAsync<Scalar4> vel_ghost_sendbuf_handle(m_vel_ghost_sendbuf, access_location::host, access_mode::read);
            ArrayHandleAsync<Scalar4> orientation_ghost_sendbuf_handle(m_orientation_ghost_sendbuf, access_location::host, access_mode::read);

            ArrayHandleAsync<unsigned int> h_unique_neighbors(m_unique_neighbors, access_location::host, access_mode::read);
            ArrayHandleAsync<unsigned int> h_ghost_begin(m_ghost_begin, access_location::host, access_mode::read);
 
            // lump together into one synchronization call
            cudaDeviceSynchronize();
            #endif

            // access send buffers
            if (m_prof) m_prof->push(m_exec_conf, "MPI send/recv");

            std::vector<MPI_Request> reqs;
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
                    // when sending/receiving 0 ptls, the send/recv buffer may be uninitialized
                    if (m_n_send_ghosts[stage][ineigh])
                        {
                        MPI_Isend(pos_ghost_sendbuf_handle.data+h_ghost_begin.data[ineigh + stage*m_n_unique_neigh],
                            m_n_send_ghosts[stage][ineigh]*sizeof(Scalar4),
                            MPI_BYTE,
                            neighbor,
                            2,
                            m_mpi_comm,
                            &req);
                        reqs.push_back(req);
                        }
                    send_bytes += m_n_send_ghosts[stage][ineigh]*sizeof(Scalar4);

                    if (m_n_recv_ghosts[stage][ineigh])
                        {
                        MPI_Irecv(pos_ghost_recvbuf_handle.data + m_ghost_offs[stage][ineigh],
                            m_n_recv_ghosts[stage][ineigh]*sizeof(Scalar4),
                            MPI_BYTE,
                            neighbor,
                            2,
                            m_mpi_comm,
                            &req);
                        reqs.push_back(req);
                        }
                    recv_bytes += m_n_recv_ghosts[stage][ineigh]*sizeof(Scalar4);
                    }

                if (flags[comm_flag::velocity])
                    {
                    if (m_n_send_ghosts[stage][ineigh])
                        {
                        MPI_Isend(vel_ghost_sendbuf_handle.data+h_ghost_begin.data[ineigh + stage*m_n_unique_neigh],
                            m_n_send_ghosts[stage][ineigh]*sizeof(Scalar4),
                            MPI_BYTE,
                            neighbor,
                            3,
                            m_mpi_comm,
                            &req);
                        reqs.push_back(req);
                        }
                    send_bytes += m_n_send_ghosts[stage][ineigh]*sizeof(Scalar4);

                    if (m_n_recv_ghosts[stage][ineigh])
                        {
                        MPI_Irecv(vel_ghost_recvbuf_handle.data + m_ghost_offs[stage][ineigh],
                            m_n_recv_ghosts[stage][ineigh]*sizeof(Scalar4),
                            MPI_BYTE,
                            neighbor,
                            3,
                            m_mpi_comm,
                            &req);
                        reqs.push_back(req);
                        }
                    recv_bytes += m_n_recv_ghosts[stage][ineigh]*sizeof(Scalar4);
                    }

                if (flags[comm_flag::orientation])
                    {
                    if (m_n_send_ghosts[stage][ineigh])
                        {
                        MPI_Isend(orientation_ghost_sendbuf_handle.data+h_ghost_begin.data[ineigh + stage*m_n_unique_neigh],
                            m_n_send_ghosts[stage][ineigh]*sizeof(Scalar4),
                            MPI_BYTE,
                            neighbor,
                            6,
                            m_mpi_comm,
                            &req);
                        reqs.push_back(req);
                        }
                    send_bytes += m_n_send_ghosts[stage][ineigh]*sizeof(Scalar4);

                    if (m_n_recv_ghosts[stage][ineigh])
                        {
                        MPI_Irecv(orientation_ghost_recvbuf_handle.data + m_ghost_offs[stage][ineigh],
                            m_n_recv_ghosts[stage][ineigh]*sizeof(Scalar4),
                            MPI_BYTE,
                            neighbor,
                            6,
                            m_mpi_comm,
                            &req);
                        reqs.push_back(req);
                        }
                    recv_bytes += m_n_recv_ghosts[stage][ineigh]*sizeof(Scalar4);
                    }
                } // end neighbor loop

            std::vector<MPI_Status> stats(reqs.size());
            MPI_Waitall(reqs.size(), &reqs.front(), &stats.front());

            if (m_prof) m_prof->pop(m_exec_conf,0,send_bytes+recv_bytes);
            } // end ArrayHandle scope

            // first ghost ptl index
            unsigned int first_idx = m_pdata->getN();

            // total up ghosts received thus far
            for (unsigned int istage = 0; istage < stage; ++istage)
                first_idx += m_n_recv_ghosts_tot[istage];

            {
            // access receive buffers
            ArrayHandle<Scalar4> d_pos_ghost_recvbuf(m_pos_ghost_recvbuf, access_location::device, access_mode::read);
            ArrayHandle<Scalar4> d_vel_ghost_recvbuf(m_vel_ghost_recvbuf, access_location::device, access_mode::read);
            ArrayHandle<Scalar4> d_orientation_ghost_recvbuf(m_orientation_ghost_recvbuf, access_location::device, access_mode::read);
            // access particle data
            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::readwrite);

            // copy recv buf into particle data
            gpu_exchange_ghosts_copy_buf(
                m_n_recv_ghosts_tot[stage],
                NULL,
                d_pos_ghost_recvbuf.data,
                d_vel_ghost_recvbuf.data,
                NULL,
                NULL,
                d_orientation_ghost_recvbuf.data,
                NULL,
                d_pos.data + first_idx,
                d_vel.data + first_idx,
                NULL,
                NULL,
                d_orientation.data + first_idx,
                false,
                flags[comm_flag::position],
                flags[comm_flag::velocity],
                false,
                false,
                flags[comm_flag::orientation]);

            if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
            }

        if (flags[comm_flag::position])
            {
            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);

            // global box
            const BoxDim& shifted_box = getShiftedBox();

            // wrap received ghosts
            gpu_wrap_ghosts(m_n_recv_ghosts_tot[stage],
                d_pos.data+first_idx,
                shifted_box);

            if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
            }
        } // end main communication loop

    if (m_prof) m_prof->pop(m_exec_conf);
    }


//! Export CommunicatorGPU class to python
void export_CommunicatorGPU()
    {
    class_<CommunicatorGPU, bases<Communicator>, boost::shared_ptr<CommunicatorGPU>, boost::noncopyable>("CommunicatorGPU",
           init<boost::shared_ptr<SystemDefinition>,
                boost::shared_ptr<DomainDecomposition> >())
            .def("setMaxStages",&CommunicatorGPU::setMaxStages)
    ;
    }

#endif // ENABLE_CUDA
#endif // ENABLE_MPI
