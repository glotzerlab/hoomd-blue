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
    int dev;
    cudaGetDevice(&dev); // this is a kludge until we figure out how to attach MGPU to an existing context
    m_mgpu_context = mgpu::CreateCudaDevice(dev);
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

    GPUVector<unsigned int> neigh_counts(m_exec_conf);
    m_neigh_counts.swap(neigh_counts);

    // Allocate buffers for bond migration
    GPUVector<bond_element> gpu_bond_sendbuf(m_exec_conf,mapped);
    m_gpu_bond_sendbuf.swap(gpu_bond_sendbuf);

    GPUVector<bond_element> gpu_bond_recvbuf(m_exec_conf,mapped);
    m_gpu_bond_recvbuf.swap(gpu_bond_recvbuf);
    }

void CommunicatorGPU::initializeNeighborArrays()
    {
    Index3D di= m_decomposition->getDomainIndexer();

    // determine allowed communication directions
    unsigned int allowed_directions = 0;
    if (di.getW() > 1) allowed_directions |= send_east | send_west;
    if (di.getH() > 1) allowed_directions |= send_north | send_south;
    if (di.getD() > 1) allowed_directions |= send_up | send_down;

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

        for (int iy=-1; iy <= 1; iy++)
            {
            int j = iy + m;

            if (j == (int)di.getH())
                j = 0;
            else if (j < 0)
                j += di.getH();


            for (int iz=-1; iz <= 1; iz++)
                {
                int k = iz + n;

                if (k == (int)di.getD())
                    k = 0;
                else if (k < 0)
                    k += di.getD();

                unsigned int mask = 0;
                if (ix == -1)
                    mask |= send_west;
                else if (ix == 1)
                    mask |= send_east;

                if (iy == -1)
                    mask |= send_south;
                else if (iy == 1)
                    mask |= send_north;

                if (iz == -1)
                    mask |= send_down;
                else if (iz == 1)
                    mask |= send_up;

                // skip neighbor if we are not communicating
                if (!mask || ! ((mask & allowed_directions) == mask)) continue;

                unsigned int neighbor = di(i,j,k);
                h_neighbors.data[m_nneigh] = neighbor;
                h_adj_mask.data[m_nneigh] = mask;
                m_nneigh++;
                }
            }
        }

    ArrayHandle<unsigned int> h_unique_neighbors(m_unique_neighbors, access_location::host, access_mode::overwrite);

    // filter neighbors
    std::copy(h_neighbors.data, h_neighbors.data + m_nneigh, h_unique_neighbors.data);
    std::sort(h_unique_neighbors.data, h_unique_neighbors.data + m_nneigh);

    // remove duplicates
    m_n_unique_neigh = std::unique(h_unique_neighbors.data, h_unique_neighbors.data + m_nneigh) - h_unique_neighbors.data;
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
    ArrayHandle<unsigned int> h_neighbors(m_neighbors, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_adj_mask(m_adj_mask, access_location::host, access_mode::read);

    Index3D di= m_decomposition->getDomainIndexer();

    // number of stages in every communication step
    m_num_stages = 0;

    m_comm_mask.resize(m_max_stages);

    // loop through neighbors to determine the communication stages
    unsigned int max_stage = 0;
    for (unsigned int ineigh = 0; ineigh < m_nneigh; ++ineigh)
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
            // compare adjacency masks of (non-unique) neighbors to mask for this stage
            for (unsigned int j = 0; j < m_nneigh; ++j)
                if (h_unique_neighbors.data[i] == h_neighbors.data[j]
                    && (h_adj_mask.data[j] &m_comm_mask[istage]) == h_adj_mask.data[j])
                    {
                    m_stages[i] = istage;
                    break; // associate neighbor with stage of lowest index
                    }

    m_exec_conf->msg->notice(5) << "ComunicatorGPU: " << m_num_stages << " communication stages." << std::endl;
    }

//! Select a particle for migration
struct get_migrate_key : public std::unary_function<const pdata_element, std::pair<unsigned int, pdata_element> >
    {
    const BoxDim box;        //!< Local simulation box dimensions
    const uint3 my_pos;      //!< My domain decomposition position
    const Index3D di;        //!< Domain indexer
    const unsigned int mask; //!< Mask of allowed directions

    //! Constructor
    /*!
     */
    get_migrate_key(const BoxDim & _box, const uint3 _my_pos, const Index3D _di, const unsigned int _mask)
        : box(_box), my_pos(_my_pos), di(_di), mask(_mask)
        { }

    //! Generate key for a sent particle
    std::pair<unsigned int, pdata_element> operator()(const pdata_element p)
        {
        Scalar4 postype = p.pos;
        Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
        Scalar3 f = box.makeFraction(pos);

        int ix, iy, iz;
        ix = iy = iz = 0;

        // we allow for a tolerance, large enough so we don't loose particles
        const Scalar tol(1e-5);
        if (f.x >= Scalar(1.0)-tol && (mask & Communicator::send_east))
            ix = 1;
        else if (f.x < tol && (mask & Communicator::send_west))
            ix = -1;

        if (f.y >= Scalar(1.0)-tol && (mask & Communicator::send_north))
            iy = 1;
        else if (f.y < tol && (mask & Communicator::send_south))
            iy = -1;

        if (f.z >= Scalar(1.0)-tol && (mask & Communicator::send_up))
            iz = 1;
        else if (f.z < tol && (mask & Communicator::send_down))
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

        return std::pair<unsigned int,pdata_element>(di(i,j,k),p);
        }

     };

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
            ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::readwrite);

            assert(stage < m_comm_mask.size());

            // mark all particles which have left the box for sending (rtag=NOT_LOCAL)
            gpu_stage_particles(m_pdata->getN(),
                d_pos.data,
                d_tag.data,
                d_rtag.data,
                m_pdata->getBox(),
                m_comm_mask[stage],
                m_cached_alloc);

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();

            }

        // fill send buffer
        m_pdata->removeParticlesGPU(m_gpu_sendbuf);

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

            gpu_sort_migrating_particles(m_gpu_sendbuf.size(),
                       d_gpu_sendbuf.data,
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
                       m_cached_alloc);

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }
        #else
            {
            ArrayHandle<pdata_element> h_gpu_sendbuf(m_gpu_sendbuf, access_location::host, access_mode::readwrite);
            ArrayHandle<unsigned int> h_begin(m_begin, access_location::host, access_mode::overwrite);
            ArrayHandle<unsigned int> h_end(m_end, access_location::host, access_mode::overwrite);
            ArrayHandle<unsigned int> h_unique_neighbors(m_unique_neighbors, access_location::host, access_mode::read);

            typedef std::multimap<unsigned int,pdata_element> key_t;
            key_t keys;

            // generate keys
            std::transform(h_gpu_sendbuf.data, h_gpu_sendbuf.data + m_gpu_sendbuf.size(), std::inserter(keys,keys.begin()),
                get_migrate_key(m_pdata->getBox(), mypos, di, m_comm_mask[stage]));

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

    #if 0
            /*
             * Communicate bonds
             */

            /*
             * Select bonds for sending
             */
            boost::shared_ptr<BondData> bdata = m_sysdef->getBondData();

            if (bdata->getNumBondsGlobal())
                {
                // Access reverse-lookup table for particle tags
                ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::read);

                // Access bond data
                ArrayHandle<uint2> d_bonds(bdata->getBondTable(), access_location::device, access_mode::read);
                ArrayHandle<unsigned int> d_bond_rtag(bdata->getBondRTags(), access_location::device, access_mode::readwrite);
                ArrayHandle<unsigned int> d_bond_tag(bdata->getBondTags(), access_location::device, access_mode::read);

                // select bonds for migration
                gpu_select_bonds(bdata->getNumBonds(),
                                 d_bonds.data,
                                 d_bond_tag.data,
                                 d_bond_rtag.data,
                                 d_rtag.data,
                                 m_cached_alloc);
                if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
                }

            if (bdata->getNumBondsGlobal())
                {
                // fill send buffer for bond data
                bdata->retrieveBondsGPU(m_gpu_bond_sendbuf);

                unsigned int n_send_bonds = m_gpu_bond_sendbuf.size();
                unsigned int n_recv_bonds;

                if (m_prof) m_prof->push(m_exec_conf, "MPI send/recv");

                // exchange size of messages
                MPI_Isend(&n_send_bonds, 1, MPI_UNSIGNED, send_neighbor, 0, m_mpi_comm, & reqs[0]);
                MPI_Irecv(&n_recv_bonds, 1, MPI_UNSIGNED, recv_neighbor, 0, m_mpi_comm, & reqs[1]);
                MPI_Waitall(2, reqs, status);

                if (m_prof) m_prof->pop(m_exec_conf);

                m_gpu_bond_recvbuf.resize(n_recv_bonds);

                    {
                    if (m_prof) m_prof->push(m_exec_conf, "MPI send/recv");

                    // exchange bond data
                    #ifdef ENABLE_MPI_CUDA
                    ArrayHandle<bond_element> bond_sendbuf_handle(m_gpu_bond_sendbuf, access_location::device, access_mode::read);
                    ArrayHandle<bond_element> bond_recvbuf_handle(m_gpu_bond_recvbuf, access_location::device, access_mode::overwrite);
                    #else
                    ArrayHandle<bond_element> bond_sendbuf_handle(m_gpu_bond_sendbuf, access_location::host, access_mode::read);
                    ArrayHandle<bond_element> bond_recvbuf_handle(m_gpu_bond_recvbuf, access_location::host, access_mode::overwrite);
                    #endif

                    MPI_Isend(bond_sendbuf_handle.data,
                              n_send_bonds*sizeof(bond_element),
                              MPI_BYTE,
                              send_neighbor,
                              1,
                              m_mpi_comm,
                              & reqs[0]);
                    MPI_Irecv(bond_recvbuf_handle.data,
                              n_recv_bonds*sizeof(bond_element),
                              MPI_BYTE,
                              recv_neighbor,
                              1,
                              m_mpi_comm,
                              & reqs[1]);
                    MPI_Waitall(2, reqs, status);

                    if (m_prof) m_prof->pop(m_exec_conf);
                    }

                // unpack data and remove bonds that have left the domain
                bdata->addRemoveBondsGPU(m_gpu_bond_recvbuf);
                } // end bond communication

            } // end dir loop
    #endif
        } // end communication stage

    if (m_prof) m_prof->pop(m_exec_conf);
    }

//! Build a ghost particle list, exchange ghost particle data with neighboring processors
void CommunicatorGPU::exchangeGhosts()
    {
    if (m_prof) m_prof->push(m_exec_conf, "comm_ghost_exch");

    m_exec_conf->msg->notice(7) << "CommunicatorGPU: ghost exchange" << std::endl;

#if 0
    /*
     * Mark particles that are part of incomplete bonds for sending
     */
    boost::shared_ptr<BondData> bdata = m_sysdef->getBondData();

    if (bdata->getNumBondsGlobal())
        {
        // Send incomplete bond member to the nearest plane in all directions
        const GPUVector<uint2>& btable = bdata->getBondTable();
        ArrayHandle<uint2> d_btable(btable, access_location::device, access_mode::read);
        ArrayHandle<unsigned char> d_plan(m_plan, access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::read);

        gpu_mark_particles_in_incomplete_bonds(d_btable.data,
                                               d_plan.data,
                                               d_pos.data,
                                               d_rtag.data,
                                               m_pdata->getN(),
                                               bdata->getNumBonds(),
                                               m_pdata->getBox());

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }
#endif

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
                                         m_comm_mask[stage],
                                         m_cached_alloc);

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

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
                    m_nneigh,
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

        // make room for ghost indices
        m_ghost_idx.resize(m_idx_offs[stage] + m_n_send_ghosts_tot[stage]);

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
            ArrayHandle<unsigned int> d_neighbors(m_neighbors, access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_unique_neighbors(m_unique_neighbors, access_location::device, access_mode::read);

            ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);

            ArrayHandle<unsigned int> d_ghost_idx(m_ghost_idx, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_ghost_begin(m_ghost_begin, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_ghost_end(m_ghost_end, access_location::device, access_mode::overwrite);

            //! Fill ghost send list and compute start and end indices per unique neighbor in list
            gpu_exchange_ghosts_make_indices(
                m_pdata->getN() + m_pdata->getNGhosts(),
                d_ghost_plan.data,
                d_tag.data,
                d_adj_mask.data,
                d_neighbors.data,
                d_unique_neighbors.data,
                d_neigh_counts.data,
                d_ghost_idx.data + m_idx_offs[stage],
                d_ghost_begin.data + stage*m_n_unique_neigh,
                d_ghost_end.data + stage*m_n_unique_neigh,
                m_nneigh,
                m_n_unique_neigh,
                m_n_send_ghosts_tot[stage],
                m_comm_mask[stage],
                m_mgpu_context,
                m_cached_alloc);

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
            ArrayHandle<unsigned int> tag_ghost_recvbuf_handle(m_tag_ghost_recvbuf, access_location::host, access_mode::overwrite);
            ArrayHandle<Scalar4> pos_ghost_recvbuf_handle(m_pos_ghost_recvbuf, access_location::host, access_mode::overwrite);
            ArrayHandle<Scalar4> vel_ghost_recvbuf_handle(m_vel_ghost_recvbuf, access_location::host, access_mode::overwrite);
            ArrayHandle<Scalar> charge_ghost_recvbuf_handle(m_charge_ghost_recvbuf, access_location::host, access_mode::overwrite);
            ArrayHandle<Scalar> diameter_ghost_recvbuf_handle(m_diameter_ghost_recvbuf, access_location::host, access_mode::overwrite);
            ArrayHandle<Scalar4> orientation_ghost_recvbuf_handle(m_orientation_ghost_recvbuf, access_location::host, access_mode::overwrite);
            // send buffers
            ArrayHandle<unsigned int> tag_ghost_sendbuf_handle(m_tag_ghost_sendbuf, access_location::host, access_mode::read);
            ArrayHandle<Scalar4> pos_ghost_sendbuf_handle(m_pos_ghost_sendbuf, access_location::host, access_mode::read);
            ArrayHandle<Scalar4> vel_ghost_sendbuf_handle(m_vel_ghost_sendbuf, access_location::host, access_mode::read);
            ArrayHandle<Scalar> charge_ghost_sendbuf_handle(m_charge_ghost_sendbuf, access_location::host, access_mode::read);
            ArrayHandle<Scalar> diameter_ghost_sendbuf_handle(m_diameter_ghost_sendbuf, access_location::host, access_mode::read);
            ArrayHandle<Scalar4> orientation_ghost_sendbuf_handle(m_orientation_ghost_sendbuf, access_location::host, access_mode::read);
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
            #else
            // recv buffers
            ArrayHandle<Scalar4> pos_ghost_recvbuf_handle(m_pos_ghost_recvbuf, access_location::host, access_mode::overwrite);
            ArrayHandle<Scalar4> vel_ghost_recvbuf_handle(m_vel_ghost_recvbuf, access_location::host, access_mode::overwrite);
            ArrayHandle<Scalar4> orientation_ghost_recvbuf_handle(m_orientation_ghost_recvbuf, access_location::host, access_mode::overwrite);

            // send buffers
            ArrayHandle<Scalar4> pos_ghost_sendbuf_handle(m_pos_ghost_sendbuf, access_location::host, access_mode::read);
            ArrayHandle<Scalar4> vel_ghost_sendbuf_handle(m_vel_ghost_sendbuf, access_location::host, access_mode::read);
            ArrayHandle<Scalar4> orientation_ghost_sendbuf_handle(m_orientation_ghost_sendbuf, access_location::host, access_mode::read);

            #endif
            ArrayHandle<unsigned int> h_unique_neighbors(m_unique_neighbors, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_ghost_begin(m_ghost_begin, access_location::host, access_mode::read);

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

            // shifted global box
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
    ;
    }

#endif // ENABLE_CUDA
#endif // ENABLE_MPI
