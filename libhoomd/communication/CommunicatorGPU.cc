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
using namespace boost::python;

//! Constructor
CommunicatorGPU::CommunicatorGPU(boost::shared_ptr<SystemDefinition> sysdef,
                                 boost::shared_ptr<DomainDecomposition> decomposition)
    : Communicator(sysdef, decomposition),
      m_max_copy_ghosts_face(0),
      m_max_copy_ghosts_edge(0),
      m_max_copy_ghosts_corner(0),
      m_max_recv_ghosts(0),
      m_buffers_allocated(false)
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

    // allocate temporary GPU buffers and set some global properties
    unsigned int is_communicating[6];
    for (unsigned int face=0; face<6; ++face)
        is_communicating[face] = isCommunicating(face) ? 1 : 0;

    gpu_allocate_tmp_storage(is_communicating,
                             m_is_at_boundary,
                             corner_plan_lookup,
                             edge_plan_lookup,
                             face_plan_lookup);

    GPUFlags<unsigned int> condition(m_exec_conf);
    m_condition.swap(condition);

#ifdef MPI3
    // create group corresponding to communicator
    MPI_Comm_group(m_mpi_comm, &m_comm_group);

    // create one-sided communication windows
    for (unsigned int i = 0; i < 6; ++i)
        MPIX_Win_create_dynamic(MPI_INFO_NULL, m_mpi_comm, &m_win_face[i]);

    for (unsigned int i = 0; i < 12; ++i)
        MPIX_Win_create_dynamic(MPI_INFO_NULL, m_mpi_comm, &m_win_edge[i]);

    MPIX_Win_create_dynamic(MPI_INFO_NULL, m_mpi_comm, &m_win_local);
#endif
    }

//! Destructor
CommunicatorGPU::~CommunicatorGPU()
    {
    m_exec_conf->msg->notice(5) << "Destroying CommunicatorGPU";

    gpu_deallocate_tmp_storage();
    }

//! Perform ghosts update
void CommunicatorGPU::updateGhosts(unsigned int timestep)
    {
    m_exec_conf->msg->notice(7) << "CommunicatorGPU: ghost update" << std::endl;

    if (m_prof) m_prof->push(m_exec_conf, "comm_ghost_update");

    unsigned int n_tot_local_ghosts = 0;

    CommFlags flags = getFlags();

        {
        ArrayHandle<unsigned int> h_n_local_ghosts_face(m_n_local_ghosts_face, access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_n_local_ghosts_edge(m_n_local_ghosts_edge, access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_n_local_ghosts_corner(m_n_local_ghosts_corner, access_location::host, access_mode::read);

        for (unsigned int i = 0; i < 12; ++i)
            n_tot_local_ghosts += h_n_local_ghosts_edge.data[i];
        for (unsigned int i = 0; i < 6; ++i)
            n_tot_local_ghosts += h_n_local_ghosts_face.data[i];
        for (unsigned int i = 0; i < 8; ++i)
            n_tot_local_ghosts += h_n_local_ghosts_corner.data[i];
        }


        {
        // call update ghosts kernel
        ArrayHandle<unsigned int> d_n_local_ghosts_face(m_n_local_ghosts_face, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_n_local_ghosts_edge(m_n_local_ghosts_edge, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_n_local_ghosts_corner(m_n_local_ghosts_corner, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_ghost_idx_face(m_ghost_idx_face, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_ghost_idx_edge(m_ghost_idx_edge, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_ghost_idx_corner(m_ghost_idx_corner, access_location::device, access_mode::read);

        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::read);

        ArrayHandle<char> d_corner_update_buf(m_corner_update_buf, access_location::device, access_mode::overwrite);
        ArrayHandle<char> d_edge_update_buf(m_edge_update_buf, access_location::device, access_mode::overwrite);
        ArrayHandle<char> d_face_update_buf(m_face_update_buf, access_location::device, access_mode::overwrite);


        gpu_update_ghosts_pack(n_tot_local_ghosts,
                            d_ghost_idx_face.data,
                            m_ghost_idx_face.getPitch(),
                            d_ghost_idx_edge.data,
                            m_ghost_idx_edge.getPitch(),
                            d_ghost_idx_corner.data,
                            m_ghost_idx_corner.getPitch(),
                            d_pos.data,
                            d_vel.data,
                            d_orientation.data,
                            d_corner_update_buf.data,
                            m_corner_update_buf.getPitch(),
                            d_edge_update_buf.data,
                            m_edge_update_buf.getPitch(),
                            d_face_update_buf.data,
                            m_face_update_buf.getPitch(),
                            d_n_local_ghosts_corner.data,
                            d_n_local_ghosts_edge.data,
                            d_n_local_ghosts_face.data,
                            ghost_update_element_size(),
                            flags[comm_flag::position] ? 1 : 0,
                            flags[comm_flag::velocity] ? 1 : 0,
                            flags[comm_flag::orientation] ? 1 : 0);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    unsigned int n_copy_ghosts_face[6];
    unsigned int n_copy_ghosts_edge[12];
    unsigned int n_copy_ghosts_corner[8];

        {
        ArrayHandle<unsigned int> h_n_local_ghosts_face(m_n_local_ghosts_face, access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_n_local_ghosts_edge(m_n_local_ghosts_edge, access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_n_local_ghosts_corner(m_n_local_ghosts_corner, access_location::host, access_mode::read);

        for (unsigned int i = 0; i < 12; ++i)
            n_copy_ghosts_edge[i] = h_n_local_ghosts_edge.data[i];
        for (unsigned int i = 0; i < 6; ++i)
            n_copy_ghosts_face[i] = h_n_local_ghosts_face.data[i];
        for (unsigned int i = 0; i < 8; ++i)
            n_copy_ghosts_corner[i] = h_n_local_ghosts_corner.data[i];
        }

    unsigned int n_tot_recv_ghosts_local = 0;

        {
        #ifdef ENABLE_MPI_CUDA
        ArrayHandle<char> corner_update_buf_handle(m_corner_update_buf, access_location::device, access_mode::read);
        ArrayHandle<char> edge_update_buf_handle(m_edge_update_buf, access_location::device, access_mode::readwrite);
        ArrayHandle<char> face_update_buf_handle(m_face_update_buf, access_location::device, access_mode::readwrite);
        ArrayHandle<char> update_recv_buf_handle(m_update_recv_buf, access_location::device, access_mode::overwrite);
        #else
        ArrayHandle<char> corner_update_buf_handle(m_corner_update_buf, access_location::host, access_mode::read);
        ArrayHandle<char> edge_update_buf_handle(m_edge_update_buf, access_location::host, access_mode::readwrite);
        ArrayHandle<char> face_update_buf_handle(m_face_update_buf, access_location::host, access_mode::readwrite);
        ArrayHandle<char> update_recv_buf_handle(m_update_recv_buf, access_location::host, access_mode::overwrite);
        #endif


        for (unsigned int face = 0; face < 6; ++face)
            {
            if (! isCommunicating(face)) continue;

            ArrayHandle<unsigned int> h_n_recv_ghosts_face(m_n_recv_ghosts_face, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_n_recv_ghosts_edge(m_n_recv_ghosts_edge, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_n_recv_ghosts_local(m_n_recv_ghosts_local, access_location::host, access_mode::read);

            communicateStepTwo(face,
                               corner_update_buf_handle.data,
                               edge_update_buf_handle.data,
                               face_update_buf_handle.data,
                               m_corner_update_buf.getPitch(),
                               m_edge_update_buf.getPitch(),
                               m_face_update_buf.getPitch(),
                               update_recv_buf_handle.data,
                               n_copy_ghosts_corner,
                               n_copy_ghosts_edge,
                               n_copy_ghosts_face,
                               m_update_recv_buf.getNumElements(),
                               n_tot_recv_ghosts_local,
                               ghost_update_element_size(),
                               false);
            // update send buffer sizes
            for (unsigned int i = 0; i < 12; ++i)
                n_copy_ghosts_edge[i] += h_n_recv_ghosts_edge.data[face*12+i];
            for (unsigned int i = 0; i < 6; ++i)
                n_copy_ghosts_face[i] += h_n_recv_ghosts_face.data[face*6+i];

            n_tot_recv_ghosts_local += h_n_recv_ghosts_local.data[face];
            } // end communication loop
        }

        {
        // unpack ghost data
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::readwrite);

        ArrayHandle<unsigned int> d_n_local_ghosts_face(m_n_local_ghosts_face, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_n_local_ghosts_edge(m_n_local_ghosts_edge, access_location::device, access_mode::read);

        ArrayHandle<unsigned int> d_n_recv_ghosts_face(m_n_recv_ghosts_face, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_n_recv_ghosts_edge(m_n_recv_ghosts_edge, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_n_recv_ghosts_local(m_n_recv_ghosts_local, access_location::device, access_mode::read);

        ArrayHandle<char> d_edge_update_buf(m_edge_update_buf, access_location::device, access_mode::read);
        ArrayHandle<char> d_face_update_buf(m_face_update_buf, access_location::device, access_mode::read);
        ArrayHandle<char> d_update_recv_buf(m_update_recv_buf, access_location::device, access_mode::read);

        // get the updated shifted global box
        const BoxDim shifted_box = getShiftedBox();

        // unpack particles
        gpu_update_ghosts_unpack(m_pdata->getN(),
                                 m_n_tot_recv_ghosts,
                                 d_n_local_ghosts_face.data,
                                 d_n_local_ghosts_edge.data,
                                 m_n_tot_recv_ghosts_local,
                                 d_n_recv_ghosts_local.data,
                                 d_n_recv_ghosts_face.data,
                                 d_n_recv_ghosts_edge.data,
                                 d_face_update_buf.data,
                                 m_face_update_buf.getPitch(),
                                 d_edge_update_buf.data,
                                 m_edge_update_buf.getPitch(),
                                 d_update_recv_buf.data,
                                 d_pos.data,
                                 d_vel.data,
                                 d_orientation.data,
                                 shifted_box,
                                 ghost_update_element_size(),
                                 flags[comm_flag::position] ? 1 : 0,
                                 flags[comm_flag::velocity] ? 1 : 0,
                                 flags[comm_flag::orientation] ? 1 : 0);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    if (m_prof) m_prof->pop(m_exec_conf);
    }

void CommunicatorGPU::allocateBuffers()
    {
    // using mapped-pinned memory avoids unnecessary memcpy's as buffers grow
    bool mapped = true;

    #ifdef ENABLE_MPI_CUDA
    // store data on device if using CUDA-aware MPI
    mapped = false;
    #endif

    // Allocate buffers for particle migration
    GPUVector<pdata_element> gpu_sendbuf(m_exec_conf,mapped);
    m_gpu_sendbuf.swap(gpu_sendbuf);

    GPUVector<pdata_element> gpu_recvbuf(m_exec_conf,mapped);
    m_gpu_recvbuf.swap(gpu_recvbuf);

    // Allocate buffers for bond migration
    GPUVector<bond_element> gpu_bond_sendbuf(m_exec_conf,mapped);
    m_gpu_bond_sendbuf.swap(gpu_bond_sendbuf);

    GPUVector<bond_element> gpu_bond_recvbuf(m_exec_conf,mapped);
    m_gpu_bond_recvbuf.swap(gpu_bond_recvbuf);


    /*
     * initial size of particle send buffers = max of avg. number of ptls in skin layer in any direction
     */
    const BoxDim& box = m_pdata->getBox();
    Scalar3 L = box.getNearestPlaneDistance();

    /*
     * initial size of ghost send buffers = max of avg number of ptls in ghost layer in every direction
     */
    unsigned int maxx = (unsigned int)((Scalar)m_pdata->getN()*m_r_ghost/L.x);
    unsigned int maxy = (unsigned int)((Scalar)m_pdata->getN()*m_r_ghost/L.y);
    unsigned int maxz = (unsigned int)((Scalar)m_pdata->getN()*m_r_ghost/L.z);

    m_max_copy_ghosts_face = 1;
    m_max_copy_ghosts_face = m_max_copy_ghosts_face > maxx ? m_max_copy_ghosts_face : maxx;
    m_max_copy_ghosts_face = m_max_copy_ghosts_face > maxy ? m_max_copy_ghosts_face : maxy;
    m_max_copy_ghosts_face = m_max_copy_ghosts_face > maxz ? m_max_copy_ghosts_face : maxz;

    size_t exch_sz = ghost_exchange_element_size();
    size_t updt_sz = ghost_update_element_size();
    #ifdef ENABLE_MPI_CUDA
    GPUArray<char> face_ghosts_buf(exch_sz*m_max_copy_ghosts_face, 6, m_exec_conf);
    #else
    GPUBufferMapped<char> face_ghosts_buf(exch_sz*m_max_copy_ghosts_face, 6, m_exec_conf);
    #endif
    m_face_ghosts_buf.swap(face_ghosts_buf);

    #if defined(ENABLE_MPI_CUDA)
    GPUArray<char> face_update_buf(updt_sz*m_max_copy_ghosts_face, 6, m_exec_conf);
    #else
    GPUBufferMapped<char> face_update_buf(updt_sz*m_max_copy_ghosts_face, 6, m_exec_conf);
    #endif
    m_face_update_buf.swap(face_update_buf);

    unsigned int maxxy = (unsigned int)((Scalar)m_pdata->getN()*m_r_ghost*m_r_ghost/L.x/L.y);
    unsigned int maxxz = (unsigned int)((Scalar)m_pdata->getN()*m_r_ghost*m_r_ghost/L.x/L.z);
    unsigned int maxyz = (unsigned int)((Scalar)m_pdata->getN()*m_r_ghost*m_r_ghost/L.y/L.z);

    m_max_copy_ghosts_edge = 1;
    m_max_copy_ghosts_edge = m_max_copy_ghosts_edge > maxxy ? m_max_copy_ghosts_edge : maxxy;
    m_max_copy_ghosts_edge = m_max_copy_ghosts_edge > maxxz ? m_max_copy_ghosts_edge : maxxz;
    m_max_copy_ghosts_edge = m_max_copy_ghosts_edge > maxyz ? m_max_copy_ghosts_edge : maxyz;

    #ifdef ENABLE_MPI_CUDA
    GPUArray<char> edge_ghosts_buf(ghost_exchange_element_size()*m_max_copy_ghosts_edge, 12, m_exec_conf);
    #else
    GPUBufferMapped<char> edge_ghosts_buf(exch_sz*m_max_copy_ghosts_edge, 12, m_exec_conf);
    #endif
    m_edge_ghosts_buf.swap(edge_ghosts_buf);

    #ifdef ENABLE_MPI_CUDA
    GPUArray<char> edge_update_buf(updt_sz*m_max_copy_ghosts_edge, 12, m_exec_conf);
    #else
    GPUBufferMapped<char> edge_update_buf(updt_sz*m_max_copy_ghosts_edge, 12, m_exec_conf);
    #endif
    m_edge_update_buf.swap(edge_update_buf);

    unsigned int maxxyz = (unsigned int)((Scalar)m_pdata->getN()*m_r_ghost*m_r_ghost*m_r_ghost/L.x/L.y/L.z);
    m_max_copy_ghosts_corner = maxxyz > 1 ? maxxyz : 1;

    #ifdef ENABLE_MPI_CUDA
    GPUArray<char> corner_ghosts_buf(exch_sz*m_max_copy_ghosts_corner, 8, m_exec_conf);
    #else
    GPUBufferMapped<char> corner_ghosts_buf(exch_sz*m_max_copy_ghosts_corner, 8, m_exec_conf);
    #endif
    m_corner_ghosts_buf.swap(corner_ghosts_buf);

    #ifdef ENABLE_MPI_CUDA
    GPUArray<char> corner_update_buf(updt_sz*m_max_copy_ghosts_corner, 8, m_exec_conf);
    #else
    GPUBufferMapped<char> corner_update_buf(updt_sz*m_max_copy_ghosts_corner, 8, m_exec_conf);
    #endif
    m_corner_update_buf.swap(corner_update_buf);

    m_max_recv_ghosts = m_max_copy_ghosts_face*6;
    #ifdef ENABLE_MPI_CUDA
    GPUArray<char> ghost_recv_buf(exch_sz*m_max_recv_ghosts, m_exec_conf);
    #else
    GPUBufferMapped<char> ghost_recv_buf(exch_sz*m_max_recv_ghosts, m_exec_conf);
    #endif
    m_ghosts_recv_buf.swap(ghost_recv_buf);

    #ifdef ENABLE_MPI_CUDA
    GPUArray<char> update_recv_buf(updt_sz*m_max_recv_ghosts,m_exec_conf);
    #else
    GPUBufferMapped<char> update_recv_buf(updt_sz*m_max_recv_ghosts,m_exec_conf);
    #endif
    m_update_recv_buf.swap(update_recv_buf);

    // allocate ghost index lists
    GPUArray<unsigned int> ghost_idx_face(m_max_copy_ghosts_face, 6, m_exec_conf);
    m_ghost_idx_face.swap(ghost_idx_face);

    GPUArray<unsigned int> ghost_idx_edge(m_max_copy_ghosts_edge, 12, m_exec_conf);
    m_ghost_idx_edge.swap(ghost_idx_edge);

    GPUArray<unsigned int> ghost_idx_corner(m_max_copy_ghosts_corner, 8, m_exec_conf);
    m_ghost_idx_corner.swap(ghost_idx_corner);

    GPUArray<unsigned int> n_local_ghosts_face(6, m_exec_conf);
    m_n_local_ghosts_face.swap(n_local_ghosts_face);
    GPUArray<unsigned int> n_local_ghosts_edge(12, m_exec_conf);
    m_n_local_ghosts_edge.swap(n_local_ghosts_edge);
    GPUArray<unsigned int> n_local_ghosts_corner(8, m_exec_conf);
    m_n_local_ghosts_corner.swap(n_local_ghosts_corner);

    GPUArray<unsigned int> n_recv_ghosts_local(6, m_exec_conf);
    m_n_recv_ghosts_local.swap(n_recv_ghosts_local);
    GPUArray<unsigned int> n_recv_ghosts_edge(6*12, m_exec_conf);
    m_n_recv_ghosts_edge.swap(n_recv_ghosts_edge);
    GPUArray<unsigned int> n_recv_ghosts_face(6*6, m_exec_conf);
    m_n_recv_ghosts_face.swap(n_recv_ghosts_face);

    m_buffers_allocated = true;
    }

//! Transfer particles between neighboring domains
void CommunicatorGPU::migrateParticles()
    {
    if (m_prof)
        m_prof->push(m_exec_conf,"comm_migrate");

    m_exec_conf->msg->notice(7) << "CommunicatorGPU: migrate particles" << std::endl;

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

    // initialization of buffers
    if (! m_buffers_allocated)
        allocateBuffers();

    // determine local particles that are to be sent to neighboring processors and fill send buffer
    for (unsigned int dir=0; dir < 6; dir++)
        {
        if (! isCommunicating(dir) ) continue;

            {
            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::readwrite);

            // mark all particles which have left the box for sending (rtag=NOT_LOCAL)
            gpu_stage_particles(m_pdata->getN(),
                d_pos.data,
                d_tag.data,
                d_rtag.data,
                dir,
                m_pdata->getBox(),
                m_cached_alloc);

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

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

        // fill send buffer
        m_pdata->removeParticlesGPU(m_gpu_sendbuf);

        // rank of processor to which we send the data
        unsigned int send_neighbor = m_decomposition->getNeighborRank(dir);

        // we receive from the direction opposite to the one we send to
        unsigned int recv_neighbor;
        if (dir % 2 == 0)
            recv_neighbor = m_decomposition->getNeighborRank(dir+1);
        else
            recv_neighbor = m_decomposition->getNeighborRank(dir-1);

        unsigned int n_recv_ptls;

        // communicate size of the message that will contain the particle data
        MPI_Request reqs[2];
        MPI_Status status[2];

        unsigned int n_send_ptls = m_gpu_sendbuf.size();

        if (m_prof) m_prof->push(m_exec_conf, "MPI send/recv");

        MPI_Isend(&n_send_ptls, 1, MPI_UNSIGNED, send_neighbor, 0, m_mpi_comm, & reqs[0]);
        MPI_Irecv(&n_recv_ptls, 1, MPI_UNSIGNED, recv_neighbor, 0, m_mpi_comm, & reqs[1]);
        MPI_Waitall(2, reqs, status);

        if (m_prof) m_prof->pop(m_exec_conf);

        // Resize receive buffer
        m_gpu_recvbuf.resize(n_recv_ptls);

            {
            if (m_prof) m_prof->push(m_exec_conf,"MPI send/recv");

            #ifdef ENABLE_MPI_CUDA
            ArrayHandle<pdata_element> gpu_sendbuf_handle(m_gpu_sendbuf, access_location::device, access_mode::read);
            ArrayHandle<pdata_element> gpu_recvbuf_handle(m_gpu_recvbuf, access_location::device, access_mode::overwrite);
            #else
            ArrayHandle<pdata_element> gpu_sendbuf_handle(m_gpu_sendbuf, access_location::host, access_mode::read);
            ArrayHandle<pdata_element> gpu_recvbuf_handle(m_gpu_recvbuf, access_location::host, access_mode::overwrite);
            #endif

            // exchange particle data
            MPI_Isend(gpu_sendbuf_handle.data, n_send_ptls*sizeof(pdata_element), MPI_BYTE, send_neighbor, 1, m_mpi_comm, & reqs[0]);
            MPI_Irecv(gpu_recvbuf_handle.data, n_recv_ptls*sizeof(pdata_element), MPI_BYTE, recv_neighbor, 1, m_mpi_comm, & reqs[1]);
            MPI_Waitall(2, reqs, status);

            if (m_prof) m_prof->pop(m_exec_conf);
            }


            {
            ArrayHandle<pdata_element> d_gpu_recvbuf(m_gpu_recvbuf, access_location::device, access_mode::readwrite);
            const BoxDim shifted_box = getShiftedBox();

            // Apply boundary conditions
            gpu_wrap_particles(n_recv_ptls,
                               d_gpu_recvbuf.data,
                               shifted_box);
            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

        // remove particles that were sent and fill particle data with received particles
        m_pdata->addParticlesGPU(m_gpu_recvbuf);

        /*
         * Communicate bonds
         */

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

    if (m_prof) m_prof->pop(m_exec_conf);

    }

//! Build a ghost particle list, exchange ghost particle data with neighboring processors
void CommunicatorGPU::exchangeGhosts()
    {
    if (m_prof) m_prof->push(m_exec_conf, "comm_ghost_exch");

    m_exec_conf->msg->notice(7) << "CommunicatorGPU: ghost exchange" << std::endl;

        {
        // resize and reset plans
        if (m_plan.size() < m_pdata->getN())
            m_plan.resize(m_pdata->getN());

        ArrayHandle<unsigned char> d_plan(m_plan, access_location::device, access_mode::overwrite);
        cudaMemsetAsync(d_plan.data, 0, sizeof(unsigned char)*m_pdata->getN());
        }

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


    /*
     * Mark non-bonded atoms for sending
     */
    // the ghost layer must be at_least m_r_ghost wide along every lattice direction
    Scalar3 ghost_fraction = m_r_ghost/m_pdata->getBox().getNearestPlaneDistance();

        {
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
        ArrayHandle<unsigned char> d_plan(m_plan, access_location::device, access_mode::readwrite);

        gpu_make_nonbonded_exchange_plan(d_plan.data,
                                         m_pdata->getN(),
                                         d_pos.data,
                                         m_pdata->getBox(),
                                         ghost_fraction);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    // initialization of buffers
    if (! m_buffers_allocated)
        allocateBuffers();

    unsigned int n_copy_ghosts_corner[8];
    unsigned int n_copy_ghosts_edge[12];
    unsigned int n_copy_ghosts_face[6];

    unsigned int condition;
    size_t exch_sz = ghost_exchange_element_size();
    size_t updt_sz = ghost_update_element_size();
    do {
        // resize buffers if necessary
        if (m_corner_ghosts_buf.getPitch() < m_max_copy_ghosts_corner*exch_sz)
            m_corner_ghosts_buf.resize(m_max_copy_ghosts_corner*exch_sz, 8);

        if (m_edge_ghosts_buf.getPitch() < m_max_copy_ghosts_edge*exch_sz)
            m_edge_ghosts_buf.resize(m_max_copy_ghosts_edge*exch_sz, 12);

        if (m_face_ghosts_buf.getPitch() < m_max_copy_ghosts_face*exch_sz)
            m_face_ghosts_buf.resize(m_max_copy_ghosts_face*exch_sz, 6);

        if (m_corner_update_buf.getPitch() < m_max_copy_ghosts_corner*updt_sz)
            m_corner_update_buf.resize(m_max_copy_ghosts_corner*updt_sz, 8);

        if (m_edge_update_buf.getPitch() < m_max_copy_ghosts_edge*updt_sz)
            m_edge_update_buf.resize(m_max_copy_ghosts_edge*updt_sz, 12);

        if (m_face_update_buf.getPitch() < m_max_copy_ghosts_face*updt_sz)
            m_face_update_buf.resize(m_max_copy_ghosts_face*updt_sz, 6);

        if (m_ghost_idx_face.getPitch() < m_max_copy_ghosts_face)
            m_ghost_idx_face.resize(m_max_copy_ghosts_face, 6);
        if (m_ghost_idx_edge.getPitch() < m_max_copy_ghosts_edge)
            m_ghost_idx_edge.resize(m_max_copy_ghosts_edge, 12);
        if (m_ghost_idx_corner.getPitch() < m_max_copy_ghosts_corner)
            m_ghost_idx_corner.resize(m_max_copy_ghosts_corner, 8);

        m_condition.resetFlags(0);

            {
            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
            ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::read);
            ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::read);
            ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::read);
            ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);
            ArrayHandle<unsigned char> d_plan(m_plan, access_location::device, access_mode::read);

            ArrayHandle<unsigned int> d_ghost_idx_face(m_ghost_idx_face, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_ghost_idx_edge(m_ghost_idx_edge, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_ghost_idx_corner(m_ghost_idx_corner, access_location::device, access_mode::overwrite);

            ArrayHandle<unsigned int> d_n_local_ghosts_face(m_n_local_ghosts_face, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_n_local_ghosts_edge(m_n_local_ghosts_edge, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_n_local_ghosts_corner(m_n_local_ghosts_corner, access_location::device, access_mode::overwrite);

            ArrayHandle<char> d_corner_ghosts_buf(m_corner_ghosts_buf, access_location::device, access_mode::overwrite);
            ArrayHandle<char> d_edge_ghosts_buf(m_edge_ghosts_buf, access_location::device, access_mode::overwrite);
            ArrayHandle<char> d_face_ghosts_buf(m_face_ghosts_buf, access_location::device, access_mode::overwrite);

            CommFlags flags = getFlags();

            gpu_exchange_ghosts(m_pdata->getN(),
                                d_plan.data,
                                d_tag.data,
                                d_ghost_idx_face.data,
                                m_ghost_idx_face.getPitch(),
                                d_ghost_idx_edge.data,
                                m_ghost_idx_edge.getPitch(),
                                d_ghost_idx_corner.data,
                                m_ghost_idx_corner.getPitch(),
                                d_pos.data,
                                d_charge.data,
                                d_diameter.data,
                                d_vel.data,
                                d_orientation.data,
                                d_corner_ghosts_buf.data,
                                m_corner_ghosts_buf.getPitch(),
                                d_edge_ghosts_buf.data,
                                m_edge_ghosts_buf.getPitch(),
                                d_face_ghosts_buf.data,
                                m_face_ghosts_buf.getPitch(),
                                d_n_local_ghosts_corner.data,
                                d_n_local_ghosts_edge.data,
                                d_n_local_ghosts_face.data,
                                m_max_copy_ghosts_corner,
                                m_max_copy_ghosts_edge,
                                m_max_copy_ghosts_face,
                                m_condition.getDeviceFlags(),
                                exch_sz,
                                flags[comm_flag::position] ? 1 : 0,
                                flags[comm_flag::velocity] ? 1 : 0,
                                flags[comm_flag::charge] ? 1 : 0,
                                flags[comm_flag::diameter] ? 1 : 0,
                                flags[comm_flag::orientation] ? 1 : 0
                                );

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

            {
            ArrayHandleAsync<unsigned int> h_n_local_ghosts_face(m_n_local_ghosts_face, access_location::host, access_mode::read);
            ArrayHandleAsync<unsigned int> h_n_local_ghosts_edge(m_n_local_ghosts_edge, access_location::host, access_mode::read);
            ArrayHandleAsync<unsigned int> h_n_local_ghosts_corner(m_n_local_ghosts_corner, access_location::host, access_mode::read);

            // synchronize with GPU
            cudaDeviceSynchronize();

            for (unsigned int i = 0; i < 6; ++i)
                n_copy_ghosts_face[i] = h_n_local_ghosts_face.data[i];
            for (unsigned int i = 0; i < 12; ++i)
                n_copy_ghosts_edge[i] = h_n_local_ghosts_edge.data[i];
            for (unsigned int i = 0; i < 8; ++i)
                n_copy_ghosts_corner[i] = h_n_local_ghosts_corner.data[i];
            }

        condition = m_condition.readFlags();
        if (condition & 1)
            {
            // overflow of face copy buf
            unsigned int new_size = 1;
            for (unsigned int i = 0; i < 6; ++i)
                if (n_copy_ghosts_face[i] > new_size) new_size = n_copy_ghosts_face[i];
            while (m_max_copy_ghosts_face < new_size)
                m_max_copy_ghosts_face = ((unsigned int)(((float)m_max_copy_ghosts_face)*m_resize_factor))+1;
            }
        if (condition & 2)
            {
            // overflow of edge copy buf
            unsigned int new_size = 1;
            for (unsigned int i = 0; i < 12; ++i)
                if (n_copy_ghosts_edge[i] > new_size) new_size = n_copy_ghosts_edge[i];
            while (m_max_copy_ghosts_edge < new_size)
                m_max_copy_ghosts_edge = ((unsigned int)(((float)m_max_copy_ghosts_edge)*m_resize_factor))+1;
            }
        if (condition & 4)
            {
            // overflow of corner copy buf
            unsigned int new_size = 1;
            for (unsigned int i = 0; i < 8; ++i)
                if (n_copy_ghosts_corner[i] > new_size) new_size = n_copy_ghosts_corner[i];
            while (m_max_copy_ghosts_corner < new_size)
                m_max_copy_ghosts_corner = ((unsigned int)(((float)m_max_copy_ghosts_corner)*m_resize_factor))+1;

            }

        if (condition & 8)
            {
            m_exec_conf->msg->error() << "Invalid particle plan." << std::endl;
            throw std::runtime_error("Error during communication.");
            }
        } while (condition);


    // store number of local particles we are sending as ghosts, for later counting purposes
    unsigned int n_tot_local_ghosts = 0;

    for (unsigned int i = 0; i < 6; ++i)
        n_tot_local_ghosts += n_copy_ghosts_face[i];

    for (unsigned int i = 0; i < 12; ++i)
        n_tot_local_ghosts += n_copy_ghosts_edge[i];

    for (unsigned int i = 0; i < 8; ++i)
        n_tot_local_ghosts += n_copy_ghosts_corner[i];

    /*
     * Fill send buffers, exchange particles according to plans
     */

    // Number of ghosts we received that are not forwarded to other boxes
    m_n_tot_recv_ghosts_local = 0;

    for (unsigned int i = 0; i < NFACE*NCORNER; ++i)
        m_remote_send_corner[i] = 0;
    for (unsigned int i = 0; i < NFACE*NEDGE; ++i)
        m_remote_send_edge[i] = 0;
    for (unsigned int i = 0; i < NFACE; ++i)
        m_remote_send_face[i] = 0;

        {
        ArrayHandle<unsigned int> h_n_recv_ghosts_face(m_n_recv_ghosts_face, access_location::host, access_mode::overwrite);
        ArrayHandle<unsigned int> h_n_recv_ghosts_edge(m_n_recv_ghosts_edge, access_location::host, access_mode::overwrite);
        ArrayHandle<unsigned int> h_n_recv_ghosts_local(m_n_recv_ghosts_local, access_location::host, access_mode::overwrite);

        /*
         * begin communication loop
         */
        for (unsigned int dir = 0; dir < 6; ++dir)
            {
            // reset all received ghost counters
            for (unsigned int i = 0; i < 6; ++i)
                h_n_recv_ghosts_face.data[6*dir+i] = 0;
            for (unsigned int i = 0; i < 12; ++i)
                h_n_recv_ghosts_edge.data[12*dir+i] = 0;
            h_n_recv_ghosts_local.data[dir] = 0;

            if (! isCommunicating(dir) ) continue;

            unsigned int max_n_recv_edge = 0;
            unsigned int max_n_recv_face = 0;

            // exchange message sizes
            communicateStepOne(dir,
                               n_copy_ghosts_corner,
                               n_copy_ghosts_edge,
                               n_copy_ghosts_face,
                               &h_n_recv_ghosts_face.data[6*dir],
                               &h_n_recv_ghosts_edge.data[12*dir],
                               &h_n_recv_ghosts_local.data[dir],
                               false
                               );

            unsigned int max_n_copy_edge = 0;
            unsigned int max_n_copy_face = 0;

            // resize buffers as necessary
            for (unsigned int i = 0; i < 12; ++i)
                {
                if (h_n_recv_ghosts_edge.data[12*dir+i] > max_n_recv_edge)
                    max_n_recv_edge = h_n_recv_ghosts_edge.data[12*dir+i];
                if (n_copy_ghosts_edge[i] > max_n_copy_edge)
                    max_n_copy_edge = n_copy_ghosts_edge[i];
                }


            if (max_n_recv_edge + max_n_copy_edge > m_max_copy_ghosts_edge)
                {
                unsigned int new_size = 1;
                while (new_size < max_n_recv_edge + max_n_copy_edge)
                    new_size = ((unsigned int)(((float)new_size)*m_resize_factor))+1;
                m_max_copy_ghosts_edge = new_size;

                m_edge_ghosts_buf.resize(m_max_copy_ghosts_edge*exch_sz, 12);
                m_edge_update_buf.resize(m_max_copy_ghosts_edge*updt_sz, 12);
                }

            for (unsigned int i = 0; i < 6; ++i)
                {
                if (h_n_recv_ghosts_face.data[6*dir+i] > max_n_recv_face)
                    max_n_recv_face = h_n_recv_ghosts_face.data[6*dir+i];
                if (n_copy_ghosts_face[i] > max_n_copy_face)
                    max_n_copy_face = n_copy_ghosts_face[i];
                }

            if (max_n_recv_face + max_n_copy_face > m_max_copy_ghosts_face)
                {
                unsigned int new_size = 1;
                while (new_size < max_n_recv_face + max_n_copy_face)
                    new_size = ((unsigned int)(((float)new_size)*m_resize_factor))+1;
                m_max_copy_ghosts_face = new_size;

                m_face_ghosts_buf.resize(m_max_copy_ghosts_face*exch_sz, 6);
                m_face_update_buf.resize(m_max_copy_ghosts_face*updt_sz, 6);
                }

            if (m_ghosts_recv_buf.getNumElements() < (m_n_tot_recv_ghosts_local + h_n_recv_ghosts_local.data[dir])*exch_sz)
                {
                unsigned int new_size =1;
                while (new_size < m_n_tot_recv_ghosts_local + h_n_recv_ghosts_local.data[dir])
                    new_size = ((unsigned int)(((float)new_size)*m_resize_factor))+1;
                m_ghosts_recv_buf.resize(new_size*exch_sz);
                }

            if (m_update_recv_buf.getNumElements() < (m_n_tot_recv_ghosts_local + h_n_recv_ghosts_local.data[dir])*updt_sz)
                {
                unsigned int new_size =1;
                while (new_size < m_n_tot_recv_ghosts_local + h_n_recv_ghosts_local.data[dir])
                    new_size = ((unsigned int)(((float)new_size)*m_resize_factor))+1;
                m_update_recv_buf.resize(new_size*updt_sz);
                }

            unsigned int cpitch = m_corner_ghosts_buf.getPitch();
            unsigned int epitch = m_edge_ghosts_buf.getPitch();
            unsigned int fpitch = m_face_ghosts_buf.getPitch();

            #ifdef ENABLE_MPI_CUDA
            ArrayHandle<char> corner_ghosts_buf_handle(m_corner_ghosts_buf, access_location::device, access_mode::read);
            ArrayHandle<char> edge_ghosts_buf_handle(m_edge_ghosts_buf, access_location::device, access_mode::readwrite);
            ArrayHandle<char> face_ghosts_buf_handle(m_face_ghosts_buf, access_location::device, access_mode::readwrite);
            ArrayHandle<char> ghosts_recv_buf_handle(m_ghosts_recv_buf, access_location::device, access_mode::readwrite);
            #else
            ArrayHandle<char> corner_ghosts_buf_handle(m_corner_ghosts_buf, access_location::host, access_mode::read);
            ArrayHandle<char> edge_ghosts_buf_handle(m_edge_ghosts_buf, access_location::host, access_mode::readwrite);
            ArrayHandle<char> face_ghosts_buf_handle(m_face_ghosts_buf, access_location::host, access_mode::readwrite);
            ArrayHandle<char> ghosts_recv_buf_handle(m_ghosts_recv_buf, access_location::host, access_mode::readwrite);
            #endif

            communicateStepTwo(dir,
                               corner_ghosts_buf_handle.data,
                               edge_ghosts_buf_handle.data,
                               face_ghosts_buf_handle.data,
                               cpitch,
                               epitch,
                               fpitch,
                               ghosts_recv_buf_handle.data,
                               n_copy_ghosts_corner,
                               n_copy_ghosts_edge,
                               n_copy_ghosts_face,
                               m_ghosts_recv_buf.getNumElements(),
                               m_n_tot_recv_ghosts_local,
                               exch_sz,
                               false);

            // update buffer sizes
            for (unsigned int i = 0; i < 12; ++i)
                n_copy_ghosts_edge[i] += h_n_recv_ghosts_edge.data[12*dir+i];

            for (unsigned int i = 0; i < 6; ++i)
                n_copy_ghosts_face[i] += h_n_recv_ghosts_face.data[6*dir+i];

            m_n_tot_recv_ghosts_local += h_n_recv_ghosts_local.data[dir];
            } // end communication loop
        }

        {
        // calculate number of forwarded particles for every face and edge
        ArrayHandle<unsigned int> h_n_local_ghosts_face(m_n_local_ghosts_face, access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_n_local_ghosts_edge(m_n_local_ghosts_edge, access_location::host, access_mode::read);

        for (unsigned int i = 0; i < 6; ++i)
            m_n_forward_ghosts_face[i] = n_copy_ghosts_face[i] - h_n_local_ghosts_face.data[i];

        for (unsigned int i = 0; i < 12; ++i)
            m_n_forward_ghosts_edge[i] = n_copy_ghosts_edge[i] - h_n_local_ghosts_edge.data[i];
        }


    // total up number of received ghosts
    m_n_tot_recv_ghosts = m_n_tot_recv_ghosts_local;
    for (unsigned int i = 0; i < 6; ++i)
        m_n_tot_recv_ghosts += m_n_forward_ghosts_face[i];
    for (unsigned int i = 0; i < 12; ++i)
        m_n_tot_recv_ghosts += m_n_forward_ghosts_edge[i];

    // update number of ghost particles
    m_pdata->addGhostParticles(m_n_tot_recv_ghosts);

        {
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::readwrite);
        ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::readwrite);
        ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::readwrite);

        ArrayHandle<unsigned int> d_n_local_ghosts_face(m_n_local_ghosts_face, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_n_local_ghosts_edge(m_n_local_ghosts_edge, access_location::device, access_mode::read);

        ArrayHandle<unsigned int> d_n_recv_ghosts_face(m_n_recv_ghosts_face, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_n_recv_ghosts_edge(m_n_recv_ghosts_edge, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_n_recv_ghosts_local(m_n_recv_ghosts_local, access_location::device, access_mode::read);

        ArrayHandle<char> d_face_ghosts_buf(m_face_ghosts_buf, access_location::device, access_mode::read);
        ArrayHandle<char> d_edge_ghosts_buf(m_edge_ghosts_buf, access_location::device, access_mode::read);
        ArrayHandle<char> d_corner_ghosts_buf(m_corner_ghosts_buf, access_location::device, access_mode::read);
        ArrayHandle<char> d_ghosts_recv_buf(m_ghosts_recv_buf, access_location::device, access_mode::read);

        // get the updated shifted global box
        const BoxDim shifted_box = getShiftedBox();

        CommFlags flags = getFlags();

        gpu_exchange_ghosts_unpack(m_pdata->getN(),
                                   m_n_tot_recv_ghosts,
                                   d_n_local_ghosts_face.data,
                                   d_n_local_ghosts_edge.data,
                                   m_n_tot_recv_ghosts_local,
                                   d_n_recv_ghosts_local.data,
                                   d_n_recv_ghosts_face.data,
                                   d_n_recv_ghosts_edge.data,
                                   d_face_ghosts_buf.data,
                                   m_face_ghosts_buf.getPitch(),
                                   d_edge_ghosts_buf.data,
                                   m_edge_ghosts_buf.getPitch(),
                                   d_ghosts_recv_buf.data,
                                   d_pos.data,
                                   d_charge.data,
                                   d_diameter.data,
                                   d_vel.data,
                                   d_orientation.data,
                                   d_tag.data,
                                   d_rtag.data,
                                   shifted_box,
                                   exch_sz,
                                   flags[comm_flag::position] ? 1 : 0,
                                   flags[comm_flag::velocity] ? 1 : 0,
                                   flags[comm_flag::charge] ? 1 : 0,
                                   flags[comm_flag::diameter] ? 1 : 0,
                                   flags[comm_flag::orientation] ? 1 : 0
                                   );

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    if (m_prof) m_prof->pop(m_exec_conf);

    // we have updated ghost particles, so inform ParticleData about this
    m_pdata->notifyGhostParticleNumberChange();
    }

void CommunicatorGPU::communicateStepOne(unsigned int cur_face,
                                        unsigned int *n_send_ptls_corner,
                                        unsigned int *n_send_ptls_edge,
                                        unsigned int *n_send_ptls_face,
                                        unsigned int *n_recv_ptls_face,
                                        unsigned int *n_recv_ptls_edge,
                                        unsigned int *n_recv_ptls_local,
                                        bool unique_destination)
    {

    // communicate size of the messages that will contain the particle data
    MPI_Request reqs[6];
    MPI_Status status[6];
    unsigned int send_neighbor = m_decomposition->getNeighborRank(cur_face);

    // we receive from the direction opposite to the one we send to
    unsigned int recv_neighbor;
    if (cur_face % 2 == 0)
        recv_neighbor = m_decomposition->getNeighborRank(cur_face+1);
    else
        recv_neighbor = m_decomposition->getNeighborRank(cur_face-1);


    unsigned int n_remote_recv_ptls_edge[12];
    unsigned int n_remote_recv_ptls_face[6];
    unsigned int n_remote_recv_ptls_local;

    for (unsigned int i = 0; i < 12; ++i)
        n_remote_recv_ptls_edge[i] = 0;
    for (unsigned int i = 0; i < 6; ++i)
        n_remote_recv_ptls_face[i] = 0;
    n_remote_recv_ptls_local = 0;

    unsigned int nptl;

    #ifndef MPI3
    MPI_Request send_req, recv_req;
    MPI_Status send_status, recv_status;

    unsigned int tag = 0;
    #endif

    #ifndef MPI3
    MPI_Isend(&n_send_ptls_corner[0], NCORNER, MPI_INT, send_neighbor, tag, m_mpi_comm, &send_req);
    MPI_Irecv(&m_remote_send_corner[NCORNER*cur_face], NCORNER, MPI_INT, recv_neighbor, tag, m_mpi_comm, &recv_req);
    MPI_Wait(&send_req,&send_status);
    MPI_Wait(&recv_req,&recv_status);
    tag++;
    #endif

    // calculate total message sizes
    for (unsigned int corner_i = 0; corner_i < 8; ++corner_i)
        {
        // only send corner particle through face if face touches corner
        nptl = n_send_ptls_corner[corner_i];

        for (unsigned int edge_j = 0; edge_j < 12; ++edge_j)
            if (getRoutingTable().m_route_corner_edge[cur_face][corner_i][edge_j])
                n_remote_recv_ptls_edge[edge_j] += nptl;

        // In particle migration, a particle gets sent to exactly one box
        if (unique_destination) continue;

        for (unsigned int face_j = 0; face_j < 6; ++face_j)
            if (getRoutingTable().m_route_corner_face[cur_face][corner_i][face_j])
                n_remote_recv_ptls_face[face_j] += nptl;

        if (getRoutingTable().m_route_corner_local[cur_face][corner_i])
            n_remote_recv_ptls_local += nptl;
        }

    #ifndef MPI3
    MPI_Isend(&n_send_ptls_edge[0], NEDGE, MPI_INT, send_neighbor, tag, m_mpi_comm, &send_req);
    MPI_Irecv(&m_remote_send_edge[NEDGE*cur_face], NEDGE, MPI_INT, recv_neighbor, tag, m_mpi_comm, &recv_req);
    MPI_Wait(&send_req,&send_status);
    MPI_Wait(&recv_req,&recv_status);
    tag++;
    #endif

    for (unsigned int edge_i = 0; edge_i < 12; ++edge_i)
        {
        nptl = n_send_ptls_edge[edge_i];

        for (unsigned int face_j = 0; face_j < 6; ++face_j)
            if (getRoutingTable().m_route_edge_face[cur_face][edge_i][face_j])
                n_remote_recv_ptls_face[face_j] += nptl;

        if (unique_destination) continue;

        if (getRoutingTable().m_route_edge_local[cur_face][edge_i])
            n_remote_recv_ptls_local += nptl;
        }

    #ifndef MPI3
    MPI_Isend(&n_send_ptls_face[cur_face], 1, MPI_INT, send_neighbor, tag, m_mpi_comm, &send_req);
    MPI_Irecv(&m_remote_send_face[cur_face], 1, MPI_INT, recv_neighbor, tag, m_mpi_comm, &recv_req);
    MPI_Wait(&send_req,&send_status);
    MPI_Wait(&recv_req,&recv_status);
    tag++;
    #endif

    nptl = n_send_ptls_face[cur_face];
    if (getRoutingTable().m_route_face_local[cur_face]) n_remote_recv_ptls_local += nptl;

    // exchange message sizes
    MPI_Isend(n_remote_recv_ptls_edge,
              sizeof(unsigned int)*12,
              MPI_BYTE,
              send_neighbor,
              tag,
              m_mpi_comm,
              & reqs[0]);
    MPI_Isend(n_remote_recv_ptls_face,
              sizeof(unsigned int)*6,
              MPI_BYTE,
              send_neighbor,
              tag+1,
              m_mpi_comm,
              &reqs[1]);
    MPI_Isend(&n_remote_recv_ptls_local,
              sizeof(unsigned int),
              MPI_BYTE,
              send_neighbor,
              tag+2,
              m_mpi_comm,
              &reqs[2]);

    MPI_Irecv(n_recv_ptls_edge,
              12*sizeof(unsigned int),
              MPI_BYTE,
              recv_neighbor,
              tag+0,
              m_mpi_comm,
              &reqs[3]);
    MPI_Irecv(n_recv_ptls_face,
              6*sizeof(unsigned int),
              MPI_BYTE,
              recv_neighbor,
              tag+1,
              m_mpi_comm,
              & reqs[4]);
    MPI_Irecv(n_recv_ptls_local,
              sizeof(unsigned int),
              MPI_BYTE,
              recv_neighbor,
              tag+2,
              m_mpi_comm,
              & reqs[5]);

    MPI_Waitall(6, reqs, status);
    }

void CommunicatorGPU::communicateStepTwo(unsigned int cur_face,
                                char *corner_send_buf,
                                char *edge_send_buf,
                                char *face_send_buf,
                                const unsigned int cpitch,
                                const unsigned int epitch,
                                const unsigned int fpitch,
                                char *local_recv_buf,
                                const unsigned int *n_send_ptls_corner,
                                const unsigned int *n_send_ptls_edge,
                                const unsigned int *n_send_ptls_face,
                                const unsigned int local_recv_buf_size,
                                const unsigned int n_tot_recv_ptls_local,
                                const unsigned int element_size,
                                bool unique_destination)
    {
    int send_neighbor = m_decomposition->getNeighborRank(cur_face);
    int recv_neighbor;
    if (cur_face % 2 == 0)
        recv_neighbor = m_decomposition->getNeighborRank(cur_face+1);
    else
        recv_neighbor = m_decomposition->getNeighborRank(cur_face-1);

    #ifdef MPI3
    // create groups for sending and receiving data
    MPI_Group send_group, recv_group;
    MPI_Group_incl(m_comm_group, 1, &send_neighbor, &send_group);
    MPI_Group_incl(m_comm_group, 1, &recv_neighbor, &recv_group);

    MPI_Aint face_recv_buf_local[6], face_recv_buf_remote[6];
    MPI_Aint edge_recv_buf_local[12], edge_recv_buf_remote[12];
    MPI_Aint local_recv_buf_local, local_recv_buf_remote;


    // attach dynamic one-sided communication windows
    unsigned int offset;
    void *ptr;

    for (unsigned int i = 0; i < 12; ++i)
        {
        offset = n_send_ptls_edge[i]*element_size;
        ptr = edge_send_buf + i*epitch + offset;
        MPI_Get_address(ptr, &edge_recv_buf_local[i]);
        MPIX_Win_attach(m_win_edge[i], ptr, epitch-offset);
        }

    for (unsigned int i = 0; i < 6; ++i)
        {
        offset = n_send_ptls_face[i]*element_size;
        ptr = face_send_buf+i*fpitch + offset;
        MPI_Get_address(ptr, &face_recv_buf_local[i]);
        MPIX_Win_attach(m_win_face[i], ptr, fpitch-offset);
        }

    offset = n_tot_recv_ptls_local*element_size;
    ptr = local_recv_buf + offset;
    MPI_Get_address(ptr, &local_recv_buf_local);
    MPIX_Win_attach(m_win_local, ptr, local_recv_buf_size-offset);

    // communicate buffer addresses to sender
    MPI_Request req[6];
    MPI_Status status[6];

    MPI_Isend(edge_recv_buf_local, 12, MPI_AINT, recv_neighbor, 0, m_mpi_comm, &req[0]);
    MPI_Isend(face_recv_buf_local, 6, MPI_AINT, recv_neighbor, 1, m_mpi_comm, &req[1]);
    MPI_Isend(&local_recv_buf_local, 1 , MPI_AINT, recv_neighbor, 2, m_mpi_comm, &req[2]);

    MPI_Irecv(edge_recv_buf_remote, 12, MPI_AINT, send_neighbor, 0, m_mpi_comm, &req[3]);
    MPI_Irecv(face_recv_buf_remote, 6, MPI_AINT, send_neighbor, 1, m_mpi_comm, &req[4]);
    MPI_Irecv(&local_recv_buf_remote, 1, MPI_AINT, send_neighbor, 2, m_mpi_comm, &req[5]);
    MPI_Waitall(6,req,status);

    // synchronize
    for (unsigned int i = 0; i < 6; ++i)
        MPI_Win_post(recv_group, 0, m_win_face[i]);
    for (unsigned int i = 0; i < 12; ++i)
        MPI_Win_post(recv_group, 0, m_win_edge[i]);
    MPI_Win_post(recv_group, 0, m_win_local);

    for (unsigned int i = 0; i < 6; ++i)
        MPI_Win_start(send_group, 0, m_win_face[i]);
    for (unsigned int i = 0; i < 12; ++i)
        MPI_Win_start(send_group, 0, m_win_edge[i]);
    MPI_Win_start(send_group, 0, m_win_local);
    #else
    // There are (up to) 26 neighborings domains. In one communication step, we are sending to
    // buffers for at most half the domains = 13 buffers max
    MPI_Request send_req[13], recv_req[13];
    unsigned int recv_req_count = 0;
    unsigned int send_req_count = 0;
    unsigned int recv_nptl;
    unsigned int offset;
    unsigned int tag = 0;
    #endif

    unsigned int nptl;
    void *data;

    #ifdef MPI3
    unsigned int foffset[6];
    unsigned int eoffset[12];
    unsigned int loffset;

    for (unsigned int i = 0; i < 12; ++i)
        eoffset[i] = 0;
    for (unsigned int i = 0; i < 6; ++i)
        foffset[i] = 0;
    loffset = 0;
    #else
    unsigned int recv_eoffset[12];
    unsigned int recv_foffset[6];
    unsigned int recv_loffset;

    for (unsigned int i = 0; i < 12; ++i)
        recv_eoffset[i] = 0;
    for (unsigned int i = 0; i < 6; ++i)
        recv_foffset[i] = 0;
    recv_loffset = 0;
    #endif

    for (unsigned int corner_i = 0; corner_i < 8; ++corner_i)
        {
        nptl = n_send_ptls_corner[corner_i];
        data = corner_send_buf+corner_i*cpitch;

        #ifndef MPI3
        recv_nptl = m_remote_send_corner[cur_face*8+corner_i];
        #endif

        for (unsigned int edge_j = 0; edge_j < 12; ++edge_j)
            if (getRoutingTable().m_route_corner_edge[cur_face][corner_i][edge_j])
                {
                // send a corner particle to an edge send buffer in the neighboring box
                #ifdef MPI3
                MPI_Put(data,
                        nptl*element_size,
                        MPI_BYTE,
                        send_neighbor,
                        edge_recv_buf_remote[edge_j]+eoffset[edge_j]*element_size,
                        nptl*element_size,
                        MPI_BYTE,
                        m_win_edge[edge_j]);
                eoffset[edge_j] += nptl;
                #else
                offset = (n_send_ptls_edge[edge_j] + recv_eoffset[edge_j])*element_size;
                bool send = (nptl > 0);
                bool recv = (recv_nptl > 0);
                if (send)
                    MPI_Isend(data,
                          nptl*element_size,
                          MPI_BYTE,
                          send_neighbor,
                          tag,
                          m_mpi_comm,
                          &send_req[send_req_count++]);
                if (recv)
                    MPI_Irecv(edge_send_buf + edge_j * epitch + offset,
                          recv_nptl*element_size,
                          MPI_BYTE,
                          recv_neighbor,
                          tag,
                          m_mpi_comm,
                          &recv_req[recv_req_count++]);
                tag++;
                recv_eoffset[edge_j] += recv_nptl;
                #endif
                }

        // If we are only sending to a single destination box, do not replicate particles
        if (unique_destination) continue;

        for (unsigned int face_j = 0; face_j < 6; ++face_j)
            if (getRoutingTable().m_route_corner_face[cur_face][corner_i][face_j])
                {
                // send a corner particle to a face send buffer in the neighboring box

                #ifdef MPI3
                MPI_Put(data,
                        nptl*element_size,
                        MPI_BYTE,
                        send_neighbor,
                        face_recv_buf_remote[face_j]+foffset[face_j]*element_size,
                        nptl*element_size,
                        MPI_BYTE,
                        m_win_face[face_j]);
                foffset[face_j] += nptl;
                #else
                offset = (n_send_ptls_face[face_j] + recv_foffset[face_j])*element_size;
                bool send = (nptl > 0);
                bool recv = (recv_nptl > 0);
                if (send)
                    MPI_Isend(data,
                          nptl*element_size,
                          MPI_BYTE,
                          send_neighbor,
                          tag,
                          m_mpi_comm,
                          &send_req[send_req_count++]);
                if (recv)
                    MPI_Irecv(face_send_buf + face_j * fpitch + offset,
                          recv_nptl*element_size,
                          MPI_BYTE,
                          recv_neighbor,
                          tag,
                          m_mpi_comm,
                          &recv_req[recv_req_count++]);
                tag++;
                recv_foffset[face_j] += recv_nptl;
                #endif
                }

        if (getRoutingTable().m_route_corner_local[cur_face][corner_i])
            {
            // send a corner particle directly to the neighboring bo
            #ifdef MPI3
            MPI_Put(data,
                    nptl*element_size,
                    MPI_BYTE,
                    send_neighbor,
                    local_recv_buf_remote+loffset*element_size,
                    nptl*element_size,
                    MPI_BYTE,
                    m_win_local);
            loffset += nptl;
            #else
            offset = (n_tot_recv_ptls_local + recv_loffset)*element_size;
            bool send = (nptl > 0);
            bool recv = (recv_nptl > 0);
            if (send)
                MPI_Isend(data,
                      nptl*element_size,
                      MPI_BYTE,
                      send_neighbor,
                      tag,
                      m_mpi_comm,
                      &send_req[send_req_count++]);
            if (recv)
                MPI_Irecv(local_recv_buf + offset,
                      recv_nptl*element_size,
                      MPI_BYTE,
                      recv_neighbor,
                      tag,
                      m_mpi_comm,
                      &recv_req[recv_req_count++]);
            tag++;
            recv_loffset += recv_nptl;
            #endif
            }
        }

    for (unsigned int edge_i = 0; edge_i < 12; ++edge_i)
        {
        nptl = n_send_ptls_edge[edge_i];
        data = edge_send_buf+edge_i*epitch;

        #ifndef MPI3
        recv_nptl = m_remote_send_edge[cur_face*12+edge_i];
        #endif

        for (unsigned int face_j = 0; face_j < 6; ++face_j)
            if (getRoutingTable().m_route_edge_face[cur_face][edge_i][face_j])
                {
                // send an edge particle to a face send buffer in the neighboring box
                #ifdef MPI3
                MPI_Put(data,
                        nptl*element_size,
                        MPI_BYTE,
                        send_neighbor,
                        face_recv_buf_remote[face_j]+foffset[face_j]*element_size,
                        nptl*element_size,
                        MPI_BYTE,
                        m_win_face[face_j]);
                foffset[face_j] += nptl;
                #else
                offset = (n_send_ptls_face[face_j] + recv_foffset[face_j])*element_size;
                bool send = (nptl > 0);
                bool recv = (recv_nptl > 0);
                if (send)
                    MPI_Isend(data,
                          nptl*element_size,
                          MPI_BYTE,
                          send_neighbor,
                          tag,
                          m_mpi_comm,
                          &send_req[send_req_count++]);
                if (recv)
                    MPI_Irecv(face_send_buf + face_j * fpitch + offset,
                          recv_nptl*element_size,
                          MPI_BYTE,
                          recv_neighbor,
                          tag,
                          m_mpi_comm,
                          &recv_req[recv_req_count++]);
                tag++;
                recv_foffset[face_j] += recv_nptl;
                #endif
                }

        if (unique_destination) continue;

        if (getRoutingTable().m_route_edge_local[cur_face][edge_i])
            {
            // send directly to neighboring box
            #ifdef MPI3
            MPI_Put(data,
                    nptl*element_size,
                    MPI_BYTE,
                    send_neighbor,
                    local_recv_buf_remote+loffset*element_size,
                    nptl*element_size,
                    MPI_BYTE,
                    m_win_local);
            loffset += nptl;
            #else
            offset = (n_tot_recv_ptls_local + recv_loffset)*element_size;
            bool send = (nptl > 0);
            bool recv = (recv_nptl > 0);
            if (send)
                MPI_Isend(data,
                      nptl*element_size,
                      MPI_BYTE,
                      send_neighbor,
                      tag,
                      m_mpi_comm,
                      &send_req[send_req_count++]);
            if (recv)
                MPI_Irecv(local_recv_buf + offset,
                      recv_nptl*element_size,
                      MPI_BYTE,
                      recv_neighbor,
                      tag,
                      m_mpi_comm,
                      &recv_req[recv_req_count++]);
            tag++;
            recv_loffset += recv_nptl;
            #endif
            }
        }

    if (getRoutingTable().m_route_face_local[cur_face])
        {
        nptl = n_send_ptls_face[cur_face];
        data = face_send_buf+cur_face*fpitch;

        #ifndef MPI3
        recv_nptl = m_remote_send_face[cur_face];
        #endif

        #ifdef MPI3
        MPI_Put(data,
                nptl*element_size,
                MPI_BYTE,
                send_neighbor,
                local_recv_buf_remote+loffset*element_size,
                nptl*element_size,
                MPI_BYTE,
                m_win_local);
        loffset += nptl;
        #else
        offset = (n_tot_recv_ptls_local + recv_loffset)*element_size;
        bool send = (nptl > 0);
        bool recv = (recv_nptl > 0);
        if (send)
            MPI_Isend(data,
                  nptl*element_size,
                  MPI_BYTE,
                  send_neighbor,
                  tag,
                  m_mpi_comm,
                  &send_req[send_req_count++]);
        if (recv)
            MPI_Irecv(local_recv_buf + offset,
                  recv_nptl*element_size,
                  MPI_BYTE,
                  recv_neighbor,
                  tag,
                  m_mpi_comm,
                  &recv_req[recv_req_count++]);
        tag++;
        recv_loffset += recv_nptl;
        #endif
        }

    #ifdef MPI3
    // synchronize
    for (unsigned int i = 0; i < 12; ++i)
        MPI_Win_complete(m_win_edge[i]);
    for (unsigned int i = 0; i < 6; ++i)
        MPI_Win_complete(m_win_face[i]);
    MPI_Win_complete(m_win_local);

    for (unsigned int i = 0; i < 12; ++i)
        MPI_Win_wait(m_win_edge[i]);
    for (unsigned int i = 0; i < 6; ++i)
        MPI_Win_wait(m_win_face[i]);
    MPI_Win_wait(m_win_local);

    // detach shared memory windows
    for (unsigned int i = 0; i < 12; ++i)
        MPIX_Win_detach(m_win_edge[i], edge_send_buf+i*epitch+n_send_ptls_edge[i]*element_size);

    for (unsigned int i = 0; i < 6; ++i)
        MPIX_Win_detach(m_win_face[i], face_send_buf+i*fpitch+n_send_ptls_face[i]*element_size);

    MPIX_Win_detach(m_win_local, local_recv_buf+n_tot_recv_ptls_local*element_size);
    #else
    MPI_Status send_status[13], recv_status[13];

    MPI_Waitall(send_req_count, send_req, send_status);
    MPI_Waitall(recv_req_count, recv_req, recv_status);
    #endif
    }

size_t CommunicatorGPU::ghost_exchange_element_size()
    {
    // Compute size of exchange data element (properly alignned according
    // to CUDA requirements)
    size_t sz = 0;
    CommFlags flags = getFlags();

    size_t max = 0;
    size_t s;
    if (flags[comm_flag::position])
        {
        size_t s = sizeof(Scalar4);
        sz += s;
        if (s > max) max =s;
        }
    if (flags[comm_flag::velocity])
        {
        s = sizeof(Scalar4);
        sz += s;
        if (s > max) max =s;
        }
    if (flags[comm_flag::orientation])
        {
        s = sizeof(Scalar4);
        sz += s;
        if (s > max) max = s;
        }
    if (flags[comm_flag::charge])
        {
        s = sizeof(Scalar);
        sz += s;
        if (s > max) max = s;
        }
    if (flags[comm_flag::diameter])
        {
        s = sizeof(Scalar);
        sz += s;
        if (s > max) max = s;
        }

    s = sizeof(unsigned int);
    sz += s;
    if (s > max) max =s;

    // Alignment
    if (sz % max) sz =((sz/max)+1)*max;

    return sz;
    }

size_t CommunicatorGPU::ghost_update_element_size()
    {
    // Compute size of update data element
    size_t sz = 0;
    CommFlags flags = getFlags();

    size_t max =0;
    size_t s;
    // only pos, vel, orientation (ignore charge and diameter)
    if (flags[comm_flag::position])
        {
        size_t s = sizeof(Scalar4);
        sz += s;
        if (s > max) max =s;
        }
    if (flags[comm_flag::velocity])
        {
        s = sizeof(Scalar4);
        sz += s;
        if (s > max) max =s;
        }
    if (flags[comm_flag::orientation])
        {
        s = sizeof(Scalar4);
        sz += s;
        if (s > max) max = s;
        }

    // Alignment
    if (max && sz % max) sz =((sz/max)+1)*max;

    return sz;
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
