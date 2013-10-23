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
#include "CommunicatorGPU.cuh"
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
      m_n_recv_ghosts_tot(0)
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

    GPUArray<unsigned int> begin(NEIGH_MAX,m_exec_conf);
    m_begin.swap(begin);

    GPUArray<unsigned int> end(NEIGH_MAX,m_exec_conf);
    m_end.swap(end);


    /*
     * Ghost communication
     */

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

    GPUArray<unsigned int> ghost_begin(NEIGH_MAX, m_exec_conf);
    m_ghost_begin.swap(ghost_begin);

    GPUArray<unsigned int> ghost_end(NEIGH_MAX, m_exec_conf);
    m_ghost_end.swap(ghost_end);

    GPUVector<unsigned int> ghost_plan(m_exec_conf);
    m_ghost_plan.swap(ghost_plan);

    GPUVector<unsigned int> ghost_tag(m_exec_conf);
    m_ghost_tag.swap(ghost_tag);

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
    uint3 mypos = di.getTriple(m_exec_conf->getRank());
    int l = mypos.x;
    int m = mypos.y;
    int n = mypos.z;

    ArrayHandle<unsigned int> h_neighbors(m_neighbors, access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_adj_mask(m_adj_mask, access_location::host, access_mode::overwrite);

    // loop over neighbors
    for (int ix=-1; ix <= 1; ix++)
        {
        if (ix && di.getW() == 1) continue;
        int i = ix + l;
        if (i == (int)di.getW())
            i = 0;
        else if (i < 0)
            i += di.getW();

        for (int iy=-1; iy <= 1; iy++)
            {
            if (iy && di.getH() == 1) continue;
            int j = iy + m;

            if (j == (int)di.getH())
                j = 0;
            else if (j < 0)
                j += di.getH();


            for (int iz=-1; iz <= 1; iz++)
                {
                if (iz && (int)di.getD() == 1) continue;
                int k = iz + n;

                if (k == (int)di.getD())
                    k = 0;
                else if (k < 0)
                    k += di.getD();

                unsigned int neighbor = di(i,j,k);
                if (neighbor == m_exec_conf->getRank()) continue;

                h_neighbors.data[m_nneigh] = neighbor;

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

//! Select a particle for migration
struct get_migrate_key : public std::unary_function<const pdata_element, std::pair<unsigned int, pdata_element> >
    {
    const BoxDim box;       //!< Local simulation box dimensions
    const uint3 my_pos;     //!< My domain decomposition position
    const Index3D di;             //!< Domain indexer

    //! Constructor
    /*!
     */
    get_migrate_key(const BoxDim & _box, const uint3 _my_pos, const Index3D _di)
        : box(_box), my_pos(_my_pos), di(_di)
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
        if (f.x >= Scalar(1.0)-tol)
            ix = 1;
        else if (f.x < tol)
            ix = -1;

        if (f.y >= Scalar(1.0)-tol)
            iy = 1;
        else if (f.y < tol)
            iy = -1;

        if (f.z >= Scalar(1.0)-tol)
            iz = 1;
        else if (f.z < tol)
            iz = -1;

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

    CommFlags flags = getFlags();
    if (flags[comm_flag::tag])
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

        {
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::readwrite);

        // mark all particles which have left the box for sending (rtag=NOT_LOCAL)
        gpu_stage_particles(m_pdata->getN(),
            d_pos.data,
            d_tag.data,
            d_rtag.data,
            m_pdata->getBox(),
            m_cached_alloc);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        }

    // fill send buffer
    m_pdata->removeParticlesGPU(m_gpu_sendbuf);

    const Index3D& di = m_decomposition->getDomainIndexer();
    // determine local particles that are to be sent to neighboring processors and fill send buffer
    uint3 mypos = di.getTriple(m_exec_conf->getRank());

#if 0
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
            get_migrate_key(m_pdata->getBox(), mypos, di));

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

        if (m_prof) m_prof->push(m_exec_conf, "MPI send/recv");

        // compute send counts
        for (unsigned int ineigh = 0; ineigh < m_n_unique_neigh; ineigh++)
            n_send_ptls[ineigh] = h_end.data[ineigh] - h_begin.data[ineigh];

        MPI_Request req[2*m_n_unique_neigh];
        MPI_Status stat[2*m_n_unique_neigh];

        // loop over neighbors
        for (unsigned int ineigh = 0; ineigh < m_n_unique_neigh; ineigh++)
            {
            // rank of neighbor processor
            unsigned int neighbor = h_unique_neighbors.data[ineigh];

            MPI_Isend(&n_send_ptls[ineigh], 1, MPI_UNSIGNED, neighbor, 0, m_mpi_comm, & req[2*ineigh]);
            MPI_Irecv(&n_recv_ptls[ineigh], 1, MPI_UNSIGNED, neighbor, 0, m_mpi_comm, & req[2*ineigh+1]);
            } // end neighbor loop

        MPI_Waitall(2*m_n_unique_neigh, req, stat);

        // sum up receive counts
        for (unsigned int ineigh = 0; ineigh < m_n_unique_neigh; ineigh++)
            {
            if (ineigh == 0)
                offs[ineigh] = 0;
            else
                offs[ineigh] = offs[ineigh-1] + n_recv_ptls[ineigh-1];

            n_recv_tot += n_recv_ptls[ineigh];
            }

        if (m_prof) m_prof->pop(m_exec_conf);
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
            }

        std::vector<MPI_Status> stats(reqs.size());
        MPI_Waitall(reqs.size(), &reqs.front(), &stats.front());

        if (m_prof) m_prof->pop(m_exec_conf);
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

    m_ghost_plan.resize(m_pdata->getN());

        {
        // compute plans
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_ghost_plan(m_ghost_plan, access_location::device, access_mode::overwrite);

        gpu_make_ghost_exchange_plan(d_ghost_plan.data,
                                     m_pdata->getN(),
                                     d_pos.data,
                                     m_pdata->getBox(),
                                     ghost_fraction,
                                     m_cached_alloc);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    // get requested ghost fields
    CommFlags flags = getFlags();

    // number of sent ghost particles
    m_n_send_ghosts_tot = 0;

    // resize temporary number of neighbors array
    m_neigh_counts.resize(m_pdata->getN());

        {
        ArrayHandle<unsigned int> d_ghost_plan(m_ghost_plan, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_adj_mask(m_adj_mask, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_neigh_counts(m_neigh_counts, access_location::device, access_mode::overwrite);

        // count number of neighbors (total and per particle) the ghost ptls are sent to
        m_n_send_ghosts_tot = gpu_exchange_ghosts_count_neighbors(
            m_pdata->getN(),
            d_ghost_plan.data,
            d_adj_mask.data,
            d_neigh_counts.data,
            m_nneigh,
            m_cached_alloc);

        if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
        }

    // resize output buffers
    m_ghost_tag.resize(m_n_send_ghosts_tot);

    if (flags[comm_flag::position]) m_pos_ghost_sendbuf.resize(m_n_send_ghosts_tot);
    if (flags[comm_flag::velocity]) m_vel_ghost_sendbuf.resize(m_n_send_ghosts_tot);
    if (flags[comm_flag::charge]) m_charge_ghost_sendbuf.resize(m_n_send_ghosts_tot);
    if (flags[comm_flag::diameter]) m_diameter_ghost_sendbuf.resize(m_n_send_ghosts_tot);
    if (flags[comm_flag::orientation]) m_orientation_ghost_sendbuf.resize(m_n_send_ghosts_tot);

        {
        ArrayHandle<unsigned int> d_ghost_plan(m_ghost_plan, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_adj_mask(m_adj_mask, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_neigh_counts(m_neigh_counts, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_neighbors(m_neighbors, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_unique_neighbors(m_unique_neighbors, access_location::device, access_mode::read);

        ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);

        ArrayHandle<unsigned int> d_ghost_tag(m_ghost_tag, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_ghost_begin(m_ghost_begin, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_ghost_end(m_ghost_end, access_location::device, access_mode::overwrite);

        //! Fill ghost send list and compute start and end indices per neighbor in list
        gpu_exchange_ghosts_make_indices(
            m_pdata->getN(),
            d_ghost_plan.data,
            d_tag.data,
            d_adj_mask.data,
            d_neighbors.data,
            d_unique_neighbors.data,
            d_neigh_counts.data,
            d_ghost_tag.data,
            d_ghost_begin.data,
            d_ghost_end.data,
            m_nneigh,
            m_n_unique_neigh,
            m_n_send_ghosts_tot,
            m_cached_alloc);
        if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
        }

        {
        // access particle data
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::read);
        ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::read);
        ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::read);

        // access ghost send indices
        ArrayHandle<unsigned int> d_ghost_tag(m_ghost_tag, access_location::device, access_mode::read);

        // access output buffers
        ArrayHandle<Scalar4> d_pos_ghost_sendbuf(m_pos_ghost_sendbuf, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_vel_ghost_sendbuf(m_vel_ghost_sendbuf, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar> d_charge_ghost_sendbuf(m_charge_ghost_sendbuf, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar> d_diameter_ghost_sendbuf(m_diameter_ghost_sendbuf, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_orientation_ghost_sendbuf(m_orientation_ghost_sendbuf, access_location::device, access_mode::overwrite);

        // Pack ghosts into send buffers
        gpu_exchange_ghosts_pack(
            m_n_send_ghosts_tot,
            d_ghost_tag.data,
            d_rtag.data,
            d_pos.data,
            d_vel.data,
            d_charge.data,
            d_diameter.data,
            d_orientation.data,
            d_pos_ghost_sendbuf.data,
            d_vel_ghost_sendbuf.data,
            d_charge_ghost_sendbuf.data,
            d_diameter_ghost_sendbuf.data,
            d_orientation_ghost_sendbuf.data,
            flags[comm_flag::position],
            flags[comm_flag::velocity],
            flags[comm_flag::charge],
            flags[comm_flag::diameter],
            flags[comm_flag::orientation],
            m_cached_alloc);

        if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
        }

    /*
     * Ghost particle communication
     */
    m_n_recv_ghosts_tot = 0;

        {
        ArrayHandle<unsigned int> h_ghost_begin(m_ghost_begin, access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_ghost_end(m_ghost_end, access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_unique_neighbors(m_unique_neighbors, access_location::host, access_mode::read);

        if (m_prof) m_prof->push(m_exec_conf, "MPI send/recv");

        // compute send counts
        for (unsigned int ineigh = 0; ineigh < m_n_unique_neigh; ineigh++)
            m_n_send_ghosts[ineigh] = h_ghost_end.data[ineigh] - h_ghost_begin.data[ineigh];

        MPI_Request req[2*m_n_unique_neigh];
        MPI_Status stat[2*m_n_unique_neigh];

        // loop over neighbors
        for (unsigned int ineigh = 0; ineigh < m_n_unique_neigh; ineigh++)
            {
            // rank of neighbor processor
            unsigned int neighbor = h_unique_neighbors.data[ineigh];

            MPI_Isend(&m_n_send_ghosts[ineigh], 1, MPI_UNSIGNED, neighbor, 0, m_mpi_comm, & req[2*ineigh]);
            MPI_Irecv(&m_n_recv_ghosts[ineigh], 1, MPI_UNSIGNED, neighbor, 0, m_mpi_comm, & req[2*ineigh+1]);
            }

        MPI_Waitall(2*m_n_unique_neigh, req, stat);

        // total up receive counts
        for (unsigned int ineigh = 0; ineigh < m_n_unique_neigh; ineigh++)
            {
            if (ineigh == 0)
                m_ghost_offs[ineigh] = 0;
            else
                m_ghost_offs[ineigh] = m_ghost_offs[ineigh-1] + m_n_recv_ghosts[ineigh-1];

            m_n_recv_ghosts_tot += m_n_recv_ghosts[ineigh];
            }

        if (m_prof) m_prof->pop(m_exec_conf);
        }

    // update number of ghost particles
    m_pdata->addGhostParticles(m_n_recv_ghosts_tot);

    if (flags[comm_flag::tag]) m_tag_ghost_recvbuf.resize(m_n_recv_ghosts_tot);
    if (flags[comm_flag::position]) m_pos_ghost_recvbuf.resize(m_n_recv_ghosts_tot);
    if (flags[comm_flag::velocity]) m_vel_ghost_recvbuf.resize(m_n_recv_ghosts_tot);
    if (flags[comm_flag::charge]) m_charge_ghost_recvbuf.resize(m_n_recv_ghosts_tot);
    if (flags[comm_flag::diameter]) m_diameter_ghost_recvbuf.resize(m_n_recv_ghosts_tot);
    if (flags[comm_flag::orientation]) m_orientation_ghost_recvbuf.resize(m_n_recv_ghosts_tot);

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
        ArrayHandle<unsigned int> ghost_tag_handle(m_ghost_tag, access_location::device, access_mode::read);
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
        ArrayHandle<unsigned int> ghost_tag_handle(m_ghost_tag, access_location::host, access_mode::read);
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
                if (m_n_send_ghosts[ineigh])
                    {
                    MPI_Isend(ghost_tag_handle.data+h_ghost_begin.data[ineigh],
                        m_n_send_ghosts[ineigh]*sizeof(unsigned int),
                        MPI_BYTE,
                        neighbor,
                        1,
                        m_mpi_comm,
                        &req);
                    reqs.push_back(req);
                    }
                if (m_n_recv_ghosts[ineigh])
                    {
                    MPI_Irecv(tag_ghost_recvbuf_handle.data + m_ghost_offs[ineigh],
                        m_n_recv_ghosts[ineigh]*sizeof(unsigned int),
                        MPI_BYTE,
                        neighbor,
                        1,
                        m_mpi_comm,
                        &req);
                    reqs.push_back(req);
                    }
                }

            if (flags[comm_flag::position])
                {
                MPI_Request req;
                if (m_n_send_ghosts[ineigh])
                    {
                    MPI_Isend(pos_ghost_sendbuf_handle.data+h_ghost_begin.data[ineigh],
                        m_n_send_ghosts[ineigh]*sizeof(Scalar4),
                        MPI_BYTE,
                        neighbor,
                        2,
                        m_mpi_comm,
                        &req);
                    reqs.push_back(req);
                    }
                if (m_n_recv_ghosts[ineigh])
                    {
                    MPI_Irecv(pos_ghost_recvbuf_handle.data + m_ghost_offs[ineigh],
                        m_n_recv_ghosts[ineigh]*sizeof(Scalar4),
                        MPI_BYTE,
                        neighbor,
                        2,
                        m_mpi_comm,
                        &req);
                    reqs.push_back(req);
                    }
                }

            if (flags[comm_flag::velocity])
                {
                if (m_n_send_ghosts[ineigh])
                    {
                    MPI_Isend(vel_ghost_sendbuf_handle.data+h_ghost_begin.data[ineigh],
                        m_n_send_ghosts[ineigh]*sizeof(Scalar4),
                        MPI_BYTE,
                        neighbor,
                        3,
                        m_mpi_comm,
                        &req);
                    reqs.push_back(req);
                    }
                if (m_n_recv_ghosts[ineigh])
                    {
                    MPI_Irecv(vel_ghost_recvbuf_handle.data + m_ghost_offs[ineigh],
                        m_n_recv_ghosts[ineigh]*sizeof(Scalar4),
                        MPI_BYTE,
                        neighbor,
                        3,
                        m_mpi_comm,
                        &req);
                    reqs.push_back(req);
                    }
                }

            if (flags[comm_flag::charge])
                {
                if (m_n_send_ghosts[ineigh])
                    {
                    MPI_Isend(charge_ghost_sendbuf_handle.data+h_ghost_begin.data[ineigh],
                        m_n_send_ghosts[ineigh]*sizeof(Scalar),
                        MPI_BYTE,
                        neighbor,
                        4,
                        m_mpi_comm,
                        &req);
                    reqs.push_back(req);
                    }
                if (m_n_recv_ghosts[ineigh])
                    {
                    MPI_Irecv(charge_ghost_recvbuf_handle.data + m_ghost_offs[ineigh],
                        m_n_recv_ghosts[ineigh]*sizeof(Scalar),
                        MPI_BYTE,
                        neighbor,
                        4,
                        m_mpi_comm,
                        &req);
                    reqs.push_back(req);
                    }
                }

            if (flags[comm_flag::diameter])
                {
                if (m_n_send_ghosts[ineigh])
                    {
                    MPI_Isend(diameter_ghost_sendbuf_handle.data+h_ghost_begin.data[ineigh],
                        m_n_send_ghosts[ineigh]*sizeof(Scalar),
                        MPI_BYTE,
                        neighbor,
                        5,
                        m_mpi_comm,
                        &req);
                    reqs.push_back(req);
                    }
                if (m_n_recv_ghosts[ineigh])
                    {
                    MPI_Irecv(diameter_ghost_recvbuf_handle.data + m_ghost_offs[ineigh],
                        m_n_recv_ghosts[ineigh]*sizeof(Scalar),
                        MPI_BYTE,
                        neighbor,
                        5,
                        m_mpi_comm,
                        &req);
                    reqs.push_back(req);
                    }
                }

            if (flags[comm_flag::orientation])
                {
                if (m_n_send_ghosts[ineigh])
                    {
                    MPI_Isend(orientation_ghost_sendbuf_handle.data+h_ghost_begin.data[ineigh],
                        m_n_send_ghosts[ineigh]*sizeof(Scalar4),
                        MPI_BYTE,
                        neighbor,
                        6,
                        m_mpi_comm,
                        &req);
                    reqs.push_back(req);
                    }
                if (m_n_recv_ghosts[ineigh])
                    {
                    MPI_Irecv(orientation_ghost_recvbuf_handle.data + m_ghost_offs[ineigh],
                        m_n_recv_ghosts[ineigh]*sizeof(Scalar4),
                        MPI_BYTE,
                        neighbor,
                        6,
                        m_mpi_comm,
                        &req);
                    reqs.push_back(req);
                    }
                }
            } // end neighbor loop

        std::vector<MPI_Status> stats(reqs.size());
        MPI_Waitall(reqs.size(), &reqs.front(), &stats.front());

        if (m_prof) m_prof->pop(m_exec_conf);
        } // end ArrayHandle scope

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

        // first ghost ptl index
        unsigned int first_idx = m_pdata->getN();

        // copy recv buf into particle data
        gpu_exchange_ghosts_copy_buf(
            m_n_recv_ghosts_tot,
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
            flags[comm_flag::orientation],
            m_cached_alloc);
        if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
        }

    if (flags[comm_flag::position])
        {
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);

        // shifted global box
        const BoxDim& shifted_box = getShiftedBox();

        // wrap received ghosts
        gpu_wrap_ghosts(m_n_recv_ghosts_tot,
            d_pos.data+m_pdata->getN(),
            shifted_box);

        if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
        }

    if (flags[comm_flag::tag])
        {
        ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::readwrite);

        // update reverse-lookup table
        gpu_compute_ghost_rtags(m_pdata->getN(),
            m_n_recv_ghosts_tot,
            d_tag.data + m_pdata->getN(),
            d_rtag.data);
        if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
        }

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

        {
        // access particle data
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::read);

        // access ghost send indices
        ArrayHandle<unsigned int> d_ghost_tag(m_ghost_tag, access_location::device, access_mode::read);

        // access output buffers
        ArrayHandle<Scalar4> d_pos_ghost_sendbuf(m_pos_ghost_sendbuf, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_vel_ghost_sendbuf(m_vel_ghost_sendbuf, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_orientation_ghost_sendbuf(m_orientation_ghost_sendbuf, access_location::device, access_mode::overwrite);

        // Pack ghosts into send buffers
        gpu_exchange_ghosts_pack(
            m_n_send_ghosts_tot,
            d_ghost_tag.data,
            d_rtag.data,
            d_pos.data,
            d_vel.data,
            NULL,
            NULL,
            d_orientation.data,
            d_pos_ghost_sendbuf.data,
            d_vel_ghost_sendbuf.data,
            NULL,
            NULL,
            d_orientation_ghost_sendbuf.data,
            flags[comm_flag::position],
            flags[comm_flag::velocity],
            false,
            false,
            flags[comm_flag::orientation],
            m_cached_alloc);

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

        // loop over neighbors
        for (unsigned int ineigh = 0; ineigh < m_n_unique_neigh; ineigh++)
            {
            // rank of neighbor processor
            unsigned int neighbor = h_unique_neighbors.data[ineigh];

            if (flags[comm_flag::position])
                {
                // when sending/receiving 0 ptls, the send/recv buffer may be uninitialized
                if (m_n_send_ghosts[ineigh])
                    {
                    MPI_Isend(pos_ghost_sendbuf_handle.data+h_ghost_begin.data[ineigh],
                        m_n_send_ghosts[ineigh]*sizeof(Scalar4),
                        MPI_BYTE,
                        neighbor,
                        2,
                        m_mpi_comm,
                        &req);
                    reqs.push_back(req);
                    }
                if (m_n_recv_ghosts[ineigh])
                    {
                    MPI_Irecv(pos_ghost_recvbuf_handle.data + m_ghost_offs[ineigh],
                        m_n_recv_ghosts[ineigh]*sizeof(Scalar4),
                        MPI_BYTE,
                        neighbor,
                        2,
                        m_mpi_comm,
                        &req);
                    reqs.push_back(req);
                    }
                }

            if (flags[comm_flag::velocity])
                {
                if (m_n_send_ghosts[ineigh])
                    {
                    MPI_Isend(vel_ghost_sendbuf_handle.data+h_ghost_begin.data[ineigh],
                        m_n_send_ghosts[ineigh]*sizeof(Scalar4),
                        MPI_BYTE,
                        neighbor,
                        3,
                        m_mpi_comm,
                        &req);
                    reqs.push_back(req);
                    }
                if (m_n_recv_ghosts[ineigh])
                    {
                    MPI_Irecv(vel_ghost_recvbuf_handle.data + m_ghost_offs[ineigh],
                        m_n_recv_ghosts[ineigh]*sizeof(Scalar4),
                        MPI_BYTE,
                        neighbor,
                        3,
                        m_mpi_comm,
                        &req);
                    reqs.push_back(req);
                    }
                }

            if (flags[comm_flag::orientation])
                {
                if (m_n_send_ghosts[ineigh])
                    {
                    MPI_Isend(orientation_ghost_sendbuf_handle.data+h_ghost_begin.data[ineigh],
                        m_n_send_ghosts[ineigh]*sizeof(Scalar4),
                        MPI_BYTE,
                        neighbor,
                        6,
                        m_mpi_comm,
                        &req);
                    reqs.push_back(req);
                    }
                if (m_n_recv_ghosts[ineigh])
                    {
                    MPI_Irecv(orientation_ghost_recvbuf_handle.data + m_ghost_offs[ineigh],
                        m_n_recv_ghosts[ineigh]*sizeof(Scalar4),
                        MPI_BYTE,
                        neighbor,
                        6,
                        m_mpi_comm,
                        &req);
                    reqs.push_back(req);
                    }
                }
            } // end neighbor loop

        std::vector<MPI_Status> stats(reqs.size());
        MPI_Waitall(reqs.size(), &reqs.front(), &stats.front());

        if (m_prof) m_prof->pop(m_exec_conf);
        } // end ArrayHandle scope

        {
        // access receive buffers
        ArrayHandle<Scalar4> d_pos_ghost_recvbuf(m_pos_ghost_recvbuf, access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_vel_ghost_recvbuf(m_vel_ghost_recvbuf, access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_orientation_ghost_recvbuf(m_orientation_ghost_recvbuf, access_location::device, access_mode::read);
        // access particle data
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::readwrite);

        // first ghost ptl index
        unsigned int first_idx = m_pdata->getN();

        // copy recv buf into particle data
        gpu_exchange_ghosts_copy_buf(
            m_n_recv_ghosts_tot,
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
            flags[comm_flag::orientation],
            m_cached_alloc);

        if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
        }

    if (flags[comm_flag::position])
        {
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);

        // shifted global box
        const BoxDim& shifted_box = getShiftedBox();

        // wrap received ghosts
        gpu_wrap_ghosts(m_n_recv_ghosts_tot,
            d_pos.data+m_pdata->getN(),
            shifted_box);
        if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
        }

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
