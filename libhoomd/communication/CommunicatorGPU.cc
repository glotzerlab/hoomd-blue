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
ghost_gpu_thread::ghost_gpu_thread(boost::shared_ptr<const ExecutionConfiguration> exec_conf)
    : m_exec_conf(exec_conf),
      h_pos_copybuf(NULL),
      h_pos_recvbuf(NULL),
      m_size_copy_buf(0),
      m_size_recv_buf(0)
    {
    }

//! Destructor
ghost_gpu_thread::~ghost_gpu_thread()
    {
    if (h_pos_copybuf) cudaFreeHost(h_pos_copybuf);
    if (h_pos_recvbuf) cudaFreeHost(h_pos_recvbuf);
    }

//! Main routine of ghost update worker thread
void ghost_gpu_thread::operator()(WorkQueue<ghost_gpu_thread_params>& queue, boost::barrier& barrier)
    {
    #ifdef VTRACE
    // initialize device context for thresd
    if (m_exec_conf->isCUDAEnabled())
         cudaFree(0);
    #endif

    // request GPU thread id
    m_thread_id = m_exec_conf->requestGPUThreadId();

    bool done = false;
    while (! done)
        {
        try
            {
            ghost_gpu_thread_params params = queue.wait_and_pop();
            update_ghosts(params);

            // synchronize with host thread
            barrier.wait();
            }
        catch(boost::thread_interrupted const)
            {
            done = true;
            }
        }

    }

void ghost_gpu_thread::update_ghosts(ghost_gpu_thread_params& params)
    {
    unsigned int num_tot_recv_ghosts = 0; // total number of ghosts received

    MPI_Request reqs[4];
    MPI_Status status[4];

    for (unsigned int dir = 0; dir < 6; dir +=2)
        {

        if (! params.is_communicating[dir]) continue;

        cudaStream_t stream = m_exec_conf->getThreadStream(m_thread_id);

        // Pack send data for direction dir and dir+1 (opposite) simultaneously.
        // We assume that they are independent, i.e. a send in direction dir+1
        // does not contain any ghosts received from that direction previously.
        gpu_copy_ghosts(params.num_copy_ghosts[dir],
                        params.num_copy_ghosts[dir+1],
                        params.d_pos_data, 
                        params.d_copy_ghosts[dir],
                        params.d_copy_ghosts[dir+1],
                        params.d_pos_copybuf,
                        params.d_pos_copybuf + params.num_copy_ghosts[dir],
                        dir,
                        params.is_at_boundary,
                        params.box,
                        stream);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        unsigned int send_neighbor = params.decomposition->getNeighborRank(dir);
        unsigned int recv_neighbor = params.decomposition->getNeighborRank(dir+1);

        unsigned int start_idx = params.N + num_tot_recv_ghosts;
        num_tot_recv_ghosts += params.num_recv_ghosts[dir] + params.num_recv_ghosts[dir+1];

        unsigned int max_copybuf = (params.num_copy_ghosts[dir] + params.num_copy_ghosts[dir+1])*sizeof(Scalar4);
        if (m_size_copy_buf < max_copybuf)
            {
            unsigned int new_size = 1;
            while (new_size < max_copybuf) new_size*=2;
            if (h_pos_copybuf) cudaFreeHost(h_pos_copybuf);
            cudaMallocHost(&h_pos_copybuf, new_size);
            m_size_copy_buf = new_size;
            }

        unsigned int max_recvbuf = (params.num_recv_ghosts[dir] + params.num_recv_ghosts[dir+1])*sizeof(Scalar4);
        if (m_size_recv_buf < max_recvbuf)
            {
            unsigned int new_size = 1;
            while (new_size < max_recvbuf) new_size*=2;
            if (h_pos_recvbuf) cudaFreeHost(h_pos_recvbuf);
            cudaMallocHost(&h_pos_recvbuf, new_size);
            m_size_recv_buf = new_size;
            }

        // exchange particle data, write directly to the particle data arrays
        cudaMemcpyAsync(h_pos_copybuf, params.d_pos_copybuf, (params.num_copy_ghosts[dir]+params.num_copy_ghosts[dir+1])*sizeof(Scalar4), cudaMemcpyDeviceToHost, stream);

        // we have posted our first CUDA operations, now let the other threads continue
        m_exec_conf->releaseThreads();

        // wait for copy to finish
        cudaEvent_t ev = m_exec_conf->getThreadEvent(m_thread_id);
        cudaEventRecord(ev, stream);
        cudaEventSynchronize(ev);

        MPI_Isend(h_pos_copybuf, params.num_copy_ghosts[dir]*sizeof(Scalar4), MPI_BYTE, send_neighbor, dir, m_exec_conf->getMPICommunicator(), &reqs[0]);
        MPI_Irecv(h_pos_recvbuf, params.num_recv_ghosts[dir]*sizeof(Scalar4), MPI_BYTE, recv_neighbor, dir, m_exec_conf->getMPICommunicator(), &reqs[1]);

        MPI_Isend(h_pos_copybuf+params.num_copy_ghosts[dir], params.num_copy_ghosts[dir+1]*sizeof(Scalar4), MPI_BYTE, recv_neighbor, dir, m_exec_conf->getMPICommunicator(), &reqs[2]);
        MPI_Irecv(h_pos_recvbuf+params.num_recv_ghosts[dir], params.num_recv_ghosts[dir+1]*sizeof(Scalar4), MPI_BYTE, send_neighbor, dir, m_exec_conf->getMPICommunicator(), &reqs[3]);

        MPI_Waitall(4, reqs, status);

        cudaMemcpyAsync(params.d_pos_data + start_idx, h_pos_recvbuf, (params.num_recv_ghosts[dir]+params.num_recv_ghosts[dir+1])*sizeof(Scalar4), cudaMemcpyHostToDevice, stream);
        } // end dir loop

    } 

//! Constructor
CommunicatorGPU::CommunicatorGPU(boost::shared_ptr<SystemDefinition> sysdef,
                                 boost::shared_ptr<DomainDecomposition> decomposition)
    : Communicator(sysdef, decomposition), m_remove_mask(m_exec_conf),
      m_buffers_allocated(false),
      m_resize_factor(9.f/8.f),
      m_barrier(2)
    { 
    m_exec_conf->msg->notice(5) << "Constructing CommunicatorGPU" << std::endl;
    // allocate temporary GPU buffers
    gpu_allocate_tmp_storage();

    // create a worker thread for ghost updates
    m_worker_thread = boost::thread(ghost_gpu_thread(m_exec_conf), boost::ref(m_work_queue), boost::ref(m_barrier));

    GPUFlags<unsigned int> condition(m_exec_conf);
    m_condition.swap(condition);
    }

//! Destructor
CommunicatorGPU::~CommunicatorGPU()
    {
    m_exec_conf->msg->notice(5) << "Destroying CommunicatorGPU";
    gpu_deallocate_tmp_storage();

    // finish worker thread
    m_worker_thread.interrupt();
    m_worker_thread.join();
    }

//! Start ghosts communication
/*! This is the multi-threaded version.
 */
void CommunicatorGPU::startGhostsUpdate(unsigned int timestep)
    {
    if (timestep < m_next_ghost_update)
        return;

    if (m_prof)
        m_prof->push("copy_ghosts");

    // fill thread parameters
    for (unsigned int i = 0; i < 6; ++i)
        {
        m_copy_ghosts_data[i] = m_copy_ghosts[i].acquire(access_location::device, access_mode::read);
        m_communication_dir[i] = isCommunicating(i);
        }

    // lock positions array against writing
    Scalar4 *d_pos_data = m_pdata->getPositions().acquire(access_location::device, access_mode::readwrite_shared);

    Scalar4 *d_pos_copybuf_data = m_pos_copybuf.acquire(access_location::device, access_mode::overwrite);

    // we want to proceed with communication quickly, so partly block scheduling of other CUDA kernels
    m_exec_conf->blockThreads();

    // post the parameters to the worker thread
    m_work_queue.push(ghost_gpu_thread_params(
         m_decomposition,
         m_communication_dir,
         m_is_at_boundary,
         m_pdata->getN(),
         m_num_copy_ghosts,
         m_num_recv_ghosts,
         m_copy_ghosts_data,
         d_pos_data,
         d_pos_copybuf_data,
         m_pdata->getGlobalBox(),
         m_exec_conf));

    if (m_prof) m_prof->pop();
    }

//! Finish ghost communication
void CommunicatorGPU::finishGhostsUpdate(unsigned int timestep)
    {
    if (timestep < m_next_ghost_update)
        return;

    // wait for worker thread to finish task
    if (m_prof) m_prof->push("copy_ghosts");
    m_barrier.wait();

    if (m_prof) m_prof->pop();

    // release locked arrays
    for (unsigned int i = 0; i < 6; ++i)
        m_copy_ghosts[i].release();

    m_pdata->getPositions().release();
    m_pos_copybuf.release();
    }

void CommunicatorGPU::allocateBuffers()
    {
    // initial size = max of avg. number of ptls in ghost layer in any direction
    const BoxDim& box = m_pdata->getBox();
    Scalar3 L = box.getL();

    unsigned int maxx = m_pdata->getN()*m_r_buff/L.x;
    unsigned int maxy = m_pdata->getN()*m_r_buff/L.y;
    unsigned int maxz = m_pdata->getN()*m_r_buff/L.z;

    m_max_send_ptls_face = 1;
    m_max_send_ptls_face = m_max_send_ptls_face > maxx ? m_max_send_ptls_face : maxx;
    m_max_send_ptls_face = m_max_send_ptls_face > maxy ? m_max_send_ptls_face : maxy;
    m_max_send_ptls_face = m_max_send_ptls_face > maxz ? m_max_send_ptls_face : maxz;

    GPUArray<char> face_send_buf(gpu_pdata_element_size()*m_max_send_ptls_face, 6, m_exec_conf);
    m_face_send_buf.swap(face_send_buf);

    unsigned int maxxy = m_pdata->getN()*m_r_buff*m_r_buff/L.x/L.y;
    unsigned int maxxz = m_pdata->getN()*m_r_buff*m_r_buff/L.x/L.z;
    unsigned int maxyz = m_pdata->getN()*m_r_buff*m_r_buff/L.y/L.z;

    m_max_send_ptls_edge = 1;
    m_max_send_ptls_edge = m_max_send_ptls_edge > maxxy ? m_max_send_ptls_edge : maxxy;
    m_max_send_ptls_edge = m_max_send_ptls_edge > maxxz ? m_max_send_ptls_edge : maxxz;
    m_max_send_ptls_edge = m_max_send_ptls_edge > maxyz ? m_max_send_ptls_edge : maxyz;

    GPUArray<char> edge_send_buf(gpu_pdata_element_size()*m_max_send_ptls_edge, 12, m_exec_conf);
    m_edge_send_buf.swap(edge_send_buf);

    unsigned maxxyz = m_pdata->getN()*m_r_buff*m_r_buff*m_r_buff/L.x/L.y/L.z;
    m_max_send_ptls_corner = maxxyz > 1 ? maxxyz : 1;

    GPUArray<char> send_buf_corner(gpu_pdata_element_size()*m_max_send_ptls_corner, 8, m_exec_conf);
    m_corner_send_buf.swap(send_buf_corner);

    GPUArray<char> recv_buf(gpu_pdata_element_size()*m_max_send_ptls_face, m_exec_conf);
    m_recv_buf.swap(recv_buf);
    
    m_buffers_allocated = true;
    }

//! Transfer particles between neighboring domains
void CommunicatorGPU::migrateAtoms()
    {
    if (m_prof)
        m_prof->push("migrate_particles");

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

    unsigned int n_send_ptls_face[6];
    unsigned int n_send_ptls_edge[12];
    unsigned int n_send_ptls_corner[8];
    unsigned int n_remove_ptls;

    if (m_remove_mask.getNumElements() < m_pdata->getN())
        m_remove_mask.resize(m_pdata->getN());

    unsigned int condition;
    do
        {
        if (m_corner_send_buf.getPitch() < m_max_send_ptls_corner*gpu_pdata_element_size())
            m_corner_send_buf.resize(m_max_send_ptls_corner*gpu_pdata_element_size(), 8);

        if (m_edge_send_buf.getPitch() < m_max_send_ptls_edge*gpu_pdata_element_size())
            m_edge_send_buf.resize(m_max_send_ptls_edge*gpu_pdata_element_size(), 12);

        if (m_face_send_buf.getPitch() < m_max_send_ptls_face*gpu_pdata_element_size())
            m_face_send_buf.resize(m_max_send_ptls_face*gpu_pdata_element_size(), 6);

        m_condition.resetFlags(0);

            {
            // remove all particles from our domain that are going to be sent in the current direction

            ArrayHandle<unsigned char> d_remove_mask(m_remove_mask, access_location::device, access_mode::readwrite);

            ArrayHandle<char> d_corner_buf(m_corner_send_buf, access_location::device, access_mode::overwrite);
            ArrayHandle<char> d_edge_buf(m_edge_send_buf, access_location::device, access_mode::overwrite);
            ArrayHandle<char> d_face_buf(m_face_send_buf, access_location::device, access_mode::overwrite);

            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
            ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::read);
            ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::read);
            ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::read);
            ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::read);
            ArrayHandle<int3> d_image(m_pdata->getImages(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_body(m_pdata->getBodies(), access_location::device, access_mode::read);
            ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::readwrite);


            // Stage particle data for sending, wrap particles
            gpu_migrate_select_particles(m_pdata->getN(),
                                   d_pos.data,
                                   d_vel.data,
                                   d_accel.data,
                                   d_image.data,
                                   d_charge.data,
                                   d_diameter.data,
                                   d_body.data,
                                   d_orientation.data,
                                   d_tag.data,
                                   d_rtag.data,
                                   n_send_ptls_corner,
                                   n_send_ptls_edge,
                                   n_send_ptls_face,
                                   n_remove_ptls,
                                   m_max_send_ptls_corner,
                                   m_max_send_ptls_edge,
                                   m_max_send_ptls_face,
                                   d_remove_mask.data,
                                   d_corner_buf.data,
                                   m_corner_send_buf.getPitch(),
                                   d_edge_buf.data,
                                   m_edge_send_buf.getPitch(),
                                   d_face_buf.data,
                                   m_face_send_buf.getPitch(),
                                   m_pdata->getBox(),
                                   m_pdata->getGlobalBox(),
                                   m_is_at_boundary,
                                   m_condition.getDeviceFlags());

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                        CHECK_CUDA_ERROR();

            }

        condition = m_condition.readFlags();
        if (condition & 1)
            {
            // set new maximum size for corner send buffers
            unsigned int new_size = 1;
            for (unsigned int i = 0; i < 8; ++i)
                if (n_send_ptls_corner[i] > new_size) new_size = n_send_ptls_corner[i];
            while (m_max_send_ptls_corner < new_size)
                m_max_send_ptls_corner = ceilf((float)m_max_send_ptls_corner * m_resize_factor);
            }
        if (condition & 2)
            {
            // set new maximum size for edge send buffers
            unsigned int new_size = 1;
            for (unsigned int i = 0; i < 12; ++i)
                if (n_send_ptls_edge[i] > new_size) new_size = n_send_ptls_edge[i];
            while (m_max_send_ptls_edge < new_size)
                m_max_send_ptls_edge = ceilf((float)m_max_send_ptls_edge*m_resize_factor);
            }
        if (condition & 4)
            {
            // set new maximum size for face send buffers
            unsigned int new_size = 1;
            for (unsigned int i = 0; i < 6; ++i)
                if (n_send_ptls_edge[i] > new_size) new_size = n_send_ptls_edge[i];
            while (m_max_send_ptls_face < new_size)
                m_max_send_ptls_face = ceilf((float)m_max_send_ptls_face*m_resize_factor);
            }

        assert((condition & 8) == 0);
        }
    while (condition);

    unsigned int n_tot_recv_ptls = 0;

    for (unsigned int dir=0; dir < 6; dir++)
        {

        if (! isCommunicating(dir) ) continue;

        unsigned int send_neighbor = m_decomposition->getNeighborRank(dir);

        // we receive from the direction opposite to the one we send to
        unsigned int recv_neighbor;
        if (dir % 2 == 0)
            recv_neighbor = m_decomposition->getNeighborRank(dir+1);
        else
            recv_neighbor = m_decomposition->getNeighborRank(dir-1);

        unsigned int n_recv_ptls_edge[12];
        unsigned int n_recv_ptls_face[6];
        unsigned int n_recv_ptls = 0;
        for (unsigned int i = 0; i < 12; ++i)
            n_recv_ptls_edge[i] = 0;

        for (unsigned int i = 0; i < 6; ++i)
            n_recv_ptls_face[i] = 0;

        unsigned int max_n_recv_edge = 0;
        unsigned int max_n_recv_face = 0;

        // communicate size of the messages that will contain the particle data
        MPI_Request reqs[18];
        MPI_Status status[18];
        unsigned int nreq=0;

        // Send message sizes
        if (dir == 0)
            {
            MPI_Isend(&n_send_ptls_corner[corner_east_north_up], sizeof(unsigned int), MPI_BYTE, send_neighbor, 0, m_mpi_comm, & reqs[nreq++]);
            MPI_Isend(&n_send_ptls_corner[corner_east_north_down], sizeof(unsigned int), MPI_BYTE, send_neighbor, 1, m_mpi_comm, & reqs[nreq++]);
            MPI_Isend(&n_send_ptls_corner[corner_east_south_up], sizeof(unsigned int), MPI_BYTE, send_neighbor, 2, m_mpi_comm, & reqs[nreq++]);
            MPI_Isend(&n_send_ptls_corner[corner_east_south_down], sizeof(unsigned int), MPI_BYTE, send_neighbor, 3, m_mpi_comm, & reqs[nreq++]);

            MPI_Isend(&n_send_ptls_edge[edge_east_north], sizeof(unsigned int), MPI_BYTE, send_neighbor, 4, m_mpi_comm, &reqs[nreq++]);
            MPI_Isend(&n_send_ptls_edge[edge_east_south], sizeof(unsigned int), MPI_BYTE, send_neighbor, 5, m_mpi_comm, &reqs[nreq++]);
            MPI_Isend(&n_send_ptls_edge[edge_east_up], sizeof(unsigned int), MPI_BYTE, send_neighbor, 6, m_mpi_comm, &reqs[nreq++]);
            MPI_Isend(&n_send_ptls_edge[edge_east_down], sizeof(unsigned int), MPI_BYTE, send_neighbor, 7, m_mpi_comm, &reqs[nreq++]);

            MPI_Isend(&n_send_ptls_face[face_east], sizeof(unsigned int), MPI_BYTE, send_neighbor, 8, m_mpi_comm, &reqs[nreq++]);
            }
        else if (dir == 1)
            {
            MPI_Isend(&n_send_ptls_corner[corner_west_north_up], sizeof(unsigned int), MPI_BYTE, send_neighbor, 0, m_mpi_comm, & reqs[nreq++]);
            MPI_Isend(&n_send_ptls_corner[corner_west_north_down], sizeof(unsigned int), MPI_BYTE, send_neighbor, 1, m_mpi_comm, & reqs[nreq++]);
            MPI_Isend(&n_send_ptls_corner[corner_west_south_up], sizeof(unsigned int), MPI_BYTE, send_neighbor, 2, m_mpi_comm, & reqs[nreq++]);
            MPI_Isend(&n_send_ptls_corner[corner_west_south_down], sizeof(unsigned int), MPI_BYTE, send_neighbor, 3, m_mpi_comm, & reqs[nreq++]);

            MPI_Isend(&n_send_ptls_edge[edge_west_north], sizeof(unsigned int), MPI_BYTE, send_neighbor, 4, m_mpi_comm, &reqs[nreq++]);
            MPI_Isend(&n_send_ptls_edge[edge_west_south], sizeof(unsigned int), MPI_BYTE, send_neighbor, 5, m_mpi_comm, &reqs[nreq++]);
            MPI_Isend(&n_send_ptls_edge[edge_west_up], sizeof(unsigned int), MPI_BYTE, send_neighbor, 6, m_mpi_comm, &reqs[nreq++]);
            MPI_Isend(&n_send_ptls_edge[edge_west_down], sizeof(unsigned int), MPI_BYTE, send_neighbor, 7, m_mpi_comm, &reqs[nreq++]);
            MPI_Isend(&n_send_ptls_face[face_west], sizeof(unsigned int), MPI_BYTE, send_neighbor, 8, m_mpi_comm, &reqs[nreq++]);
            }
        else if (dir == 2)
            {
            MPI_Isend(&n_send_ptls_edge[edge_north_up], sizeof(unsigned int), MPI_BYTE, send_neighbor, 6, m_mpi_comm, &reqs[nreq++]);
            MPI_Isend(&n_send_ptls_edge[edge_north_down], sizeof(unsigned int), MPI_BYTE, send_neighbor, 7, m_mpi_comm, &reqs[nreq++]);
            MPI_Isend(&n_send_ptls_face[face_north], sizeof(unsigned int), MPI_BYTE, send_neighbor, 8, m_mpi_comm, &reqs[nreq++]);
            }
        else if (dir == 3)
            {
            MPI_Isend(&n_send_ptls_edge[edge_south_up], sizeof(unsigned int), MPI_BYTE, send_neighbor, 6, m_mpi_comm, &reqs[nreq++]);
            MPI_Isend(&n_send_ptls_edge[edge_south_down], sizeof(unsigned int), MPI_BYTE, send_neighbor, 7, m_mpi_comm, &reqs[nreq++]);
            MPI_Isend(&n_send_ptls_face[face_south], sizeof(unsigned int), MPI_BYTE, send_neighbor, 8, m_mpi_comm, &reqs[nreq++]);
            } 
        else if (dir == 4)
            {
            MPI_Isend(&n_send_ptls_face[face_up], sizeof(unsigned int), MPI_BYTE, send_neighbor, 8, m_mpi_comm, &reqs[nreq++]);
            }
        else if (dir == 5)
            {
            MPI_Isend(&n_send_ptls_face[face_down], sizeof(unsigned int), MPI_BYTE, send_neighbor, 8, m_mpi_comm, &reqs[nreq++]);
            }
       
       if (dir < 2)
            {
            MPI_Irecv(&n_recv_ptls_edge[edge_north_up], sizeof(unsigned int), MPI_BYTE, recv_neighbor, 0, m_mpi_comm, &reqs[nreq++]);
            MPI_Irecv(&n_recv_ptls_edge[edge_north_down], sizeof(unsigned int), MPI_BYTE, recv_neighbor, 1, m_mpi_comm, & reqs[nreq++]);
            MPI_Irecv(&n_recv_ptls_edge[edge_south_up], sizeof(unsigned int), MPI_BYTE, recv_neighbor, 2, m_mpi_comm, & reqs[nreq++]);
            MPI_Irecv(&n_recv_ptls_edge[edge_south_down], sizeof(unsigned int), MPI_BYTE, recv_neighbor, 3, m_mpi_comm, & reqs[nreq++]);
            MPI_Irecv(&n_recv_ptls_face[face_north], sizeof(unsigned int), MPI_BYTE, recv_neighbor, 4, m_mpi_comm, & reqs[nreq++]);
            MPI_Irecv(&n_recv_ptls_face[face_south], sizeof(unsigned int), MPI_BYTE, recv_neighbor, 5, m_mpi_comm, & reqs[nreq++]);
            }

        if (dir < 4)
            {
            MPI_Irecv(&n_recv_ptls_face[face_up], sizeof(unsigned int), MPI_BYTE, recv_neighbor, 6, m_mpi_comm, & reqs[nreq++]);
            MPI_Irecv(&n_recv_ptls_face[face_down], sizeof(unsigned int), MPI_BYTE, recv_neighbor, 7, m_mpi_comm, & reqs[nreq++]);
            }

        MPI_Irecv(&n_recv_ptls, sizeof(unsigned int), MPI_BYTE, recv_neighbor, 8, m_mpi_comm, &reqs[nreq++]);

        MPI_Waitall(nreq, reqs, status);

        unsigned int max_n_send_edge = 0;
        unsigned int max_n_send_face = 0;
        // resize buffers as necessary
        for (unsigned int i = 0; i < 12; ++i)
            {
            if (n_recv_ptls_edge[i] > max_n_recv_edge)
                max_n_recv_edge = n_recv_ptls_edge[i];
            if (n_send_ptls_edge[i] > max_n_send_edge)
                max_n_send_edge = n_send_ptls_edge[i];
            }


        if (max_n_recv_edge + max_n_send_edge > m_max_send_ptls_edge)
            {
            unsigned int new_size = 1;
            while (new_size < max_n_recv_edge + max_n_send_edge) new_size = ceilf((float)new_size* m_resize_factor);
            m_max_send_ptls_edge = new_size;

            m_edge_send_buf.resize(m_max_send_ptls_edge*gpu_pdata_element_size(), 12);
            }

        for (unsigned int i = 0; i < 6; ++i)
            {
            if (n_recv_ptls_face[i] > max_n_recv_face)
                max_n_recv_face = n_recv_ptls_face[i];
            if (n_send_ptls_face[i] > max_n_send_face)
                max_n_send_face = n_send_ptls_face[i];
            }

        if (max_n_recv_face + max_n_send_face > m_max_send_ptls_face)
            {
            unsigned int new_size = 1;
            while (new_size < max_n_recv_face + max_n_send_face) new_size = ceilf((float) new_size * m_resize_factor);
            m_max_send_ptls_face = new_size;

            m_face_send_buf.resize(m_max_send_ptls_face*gpu_pdata_element_size(), 6);
            }

        if (m_recv_buf.getNumElements() < (n_tot_recv_ptls + n_recv_ptls)*gpu_pdata_element_size())
            {
            unsigned int new_size =1;
            while (new_size < n_tot_recv_ptls + n_recv_ptls) new_size = ceilf((float) new_size * m_resize_factor);
            m_recv_buf.resize(new_size*gpu_pdata_element_size());
            }
          

        // exchange particle data
        nreq = 0;
#ifdef ENABLE_MPI_CUDA
        ArrayHandle<char> corner_send_buf(m_corner_send_buf, access_location::device, access_mode::read);
        ArrayHandle<char> edge_send_buf(m_edge_send_buf, access_location::device, access_mode::readwrite);
        ArrayHandle<char> face_send_buf(m_face_send_buf, access_location::device, access_mode::readwrite);
#else
        ArrayHandle<char> corner_send_buf(m_corner_send_buf, access_location::host, access_mode::read);
        ArrayHandle<char> edge_send_buf(m_edge_send_buf, access_location::host, access_mode::readwrite);
        ArrayHandle<char> face_send_buf(m_face_send_buf, access_location::host, access_mode::readwrite);
#endif
        unsigned int cpitch = m_corner_send_buf.getPitch();
        unsigned int epitch = m_edge_send_buf.getPitch();
        unsigned int fpitch = m_face_send_buf.getPitch();

        if (dir == 0)
            {
            MPI_Isend(corner_send_buf.data+corner_east_north_up*cpitch, n_send_ptls_corner[corner_east_north_up]*gpu_pdata_element_size(), MPI_BYTE, send_neighbor, 0, m_mpi_comm, & reqs[nreq++]);
            MPI_Isend(corner_send_buf.data+corner_east_north_down*cpitch, n_send_ptls_corner[corner_east_north_down]*gpu_pdata_element_size(), MPI_BYTE, send_neighbor, 1, m_mpi_comm, & reqs[nreq++]);
            MPI_Isend(corner_send_buf.data+corner_east_south_up*cpitch, n_send_ptls_corner[corner_east_south_up]*gpu_pdata_element_size(), MPI_BYTE, send_neighbor, 2, m_mpi_comm, & reqs[nreq++]);
            MPI_Isend(corner_send_buf.data+corner_east_south_down*cpitch, n_send_ptls_corner[corner_east_south_down]*gpu_pdata_element_size(), MPI_BYTE, send_neighbor, 3, m_mpi_comm, & reqs[nreq++]);

            MPI_Isend(edge_send_buf.data+edge_east_north*epitch, n_send_ptls_edge[edge_east_north]*gpu_pdata_element_size(), MPI_BYTE, send_neighbor, 4, m_mpi_comm, &reqs[nreq++]);
            MPI_Isend(edge_send_buf.data+edge_east_south*epitch, n_send_ptls_edge[edge_east_south]*gpu_pdata_element_size(), MPI_BYTE, send_neighbor, 5, m_mpi_comm, &reqs[nreq++]);
            MPI_Isend(edge_send_buf.data+edge_east_up*epitch, n_send_ptls_edge[edge_east_up]*gpu_pdata_element_size(), MPI_BYTE, send_neighbor, 6, m_mpi_comm, &reqs[nreq++]);
            MPI_Isend(edge_send_buf.data+edge_east_down*epitch, n_send_ptls_edge[edge_east_down]*gpu_pdata_element_size(), MPI_BYTE, send_neighbor, 7, m_mpi_comm, &reqs[nreq++]);

            MPI_Isend(face_send_buf.data+face_east*fpitch, n_send_ptls_face[face_east]*gpu_pdata_element_size(), MPI_BYTE, send_neighbor, 8, m_mpi_comm, &reqs[nreq++]);
            }
        else if (dir == 1)
            {
            MPI_Isend(corner_send_buf.data+corner_west_north_up*cpitch, n_send_ptls_corner[corner_west_north_up]*gpu_pdata_element_size(), MPI_BYTE, send_neighbor, 0, m_mpi_comm, & reqs[nreq++]);
            MPI_Isend(corner_send_buf.data+corner_west_north_down*cpitch, n_send_ptls_corner[corner_west_north_down]*gpu_pdata_element_size(), MPI_BYTE, send_neighbor, 1, m_mpi_comm, & reqs[nreq++]);
            MPI_Isend(corner_send_buf.data+corner_west_south_up*cpitch, n_send_ptls_corner[corner_west_south_up]*gpu_pdata_element_size(), MPI_BYTE, send_neighbor, 2, m_mpi_comm, & reqs[nreq++]);
            MPI_Isend(corner_send_buf.data+corner_west_south_down*cpitch, n_send_ptls_corner[corner_west_south_down]*gpu_pdata_element_size(), MPI_BYTE, send_neighbor, 3, m_mpi_comm, & reqs[nreq++]);

            MPI_Isend(edge_send_buf.data+edge_west_north*epitch, n_send_ptls_edge[edge_west_north]*gpu_pdata_element_size(), MPI_BYTE, send_neighbor, 4, m_mpi_comm, &reqs[nreq++]);
            MPI_Isend(edge_send_buf.data+edge_west_south*epitch, n_send_ptls_edge[edge_west_south]*gpu_pdata_element_size(), MPI_BYTE, send_neighbor, 5, m_mpi_comm, &reqs[nreq++]);
            MPI_Isend(edge_send_buf.data+edge_west_up*epitch, n_send_ptls_edge[edge_west_up]*gpu_pdata_element_size(), MPI_BYTE, send_neighbor, 6, m_mpi_comm, &reqs[nreq++]);
            MPI_Isend(edge_send_buf.data+edge_west_down*epitch, n_send_ptls_edge[edge_west_down]*gpu_pdata_element_size(), MPI_BYTE, send_neighbor, 7, m_mpi_comm, &reqs[nreq++]);

            MPI_Isend(face_send_buf.data+face_west*fpitch, n_send_ptls_face[face_west]*gpu_pdata_element_size(), MPI_BYTE, send_neighbor, 8, m_mpi_comm, &reqs[nreq++]);
            }
        else if (dir == 2)
            {
            MPI_Isend(edge_send_buf.data+edge_north_up*epitch, n_send_ptls_edge[edge_north_up]*gpu_pdata_element_size(), MPI_BYTE, send_neighbor, 6, m_mpi_comm, &reqs[nreq++]);
            MPI_Isend(edge_send_buf.data+edge_north_down*epitch, n_send_ptls_edge[edge_north_down]*gpu_pdata_element_size(), MPI_BYTE, send_neighbor, 7, m_mpi_comm, &reqs[nreq++]);

            MPI_Isend(face_send_buf.data+face_north*fpitch, n_send_ptls_face[face_north]*gpu_pdata_element_size(), MPI_BYTE, send_neighbor, 8, m_mpi_comm, &reqs[nreq++]);
            }
        else if (dir == 3)
            {
            MPI_Isend(edge_send_buf.data+edge_south_up*epitch, n_send_ptls_edge[edge_south_up]*gpu_pdata_element_size(), MPI_BYTE, send_neighbor, 6, m_mpi_comm, &reqs[nreq++]);
            MPI_Isend(edge_send_buf.data+edge_south_down*epitch, n_send_ptls_edge[edge_south_down]*gpu_pdata_element_size(), MPI_BYTE, send_neighbor, 7, m_mpi_comm, &reqs[nreq++]);

            MPI_Isend(face_send_buf.data+face_south*fpitch, n_send_ptls_face[face_south]*gpu_pdata_element_size(), MPI_BYTE, send_neighbor, 8, m_mpi_comm, &reqs[nreq++]);
            } 
        else if (dir == 4)
            {
            MPI_Isend(face_send_buf.data+face_up*fpitch, n_send_ptls_face[face_up]*gpu_pdata_element_size(), MPI_BYTE, send_neighbor, 8, m_mpi_comm, &reqs[nreq++]);
            }
        else if (dir == 5)
            {
            MPI_Isend(face_send_buf.data+face_down*fpitch, n_send_ptls_face[face_down]*gpu_pdata_element_size(), MPI_BYTE, send_neighbor, 8, m_mpi_comm, &reqs[nreq++]);
            }

#ifdef ENABLE_MPI_CUDA
        ArrayHandle<char> recv_buf(m_recv_buf, access_location::device, access_mode::readwrite);
#else
        ArrayHandle<char> recv_buf(m_recv_buf, access_location::host, access_mode::readwrite);
#endif

        if (dir < 2)
            {
            MPI_Irecv(edge_send_buf.data+edge_north_up*epitch+n_send_ptls_edge[edge_north_up]*gpu_pdata_element_size(), n_recv_ptls_edge[edge_north_up]*gpu_pdata_element_size(), MPI_BYTE, recv_neighbor, 0, m_mpi_comm, &reqs[nreq++]);
            MPI_Irecv(edge_send_buf.data+edge_north_down*epitch+n_send_ptls_edge[edge_north_down]*gpu_pdata_element_size(), n_recv_ptls_edge[edge_north_down]*gpu_pdata_element_size(), MPI_BYTE, recv_neighbor, 1, m_mpi_comm, &reqs[nreq++]);
            MPI_Irecv(edge_send_buf.data+edge_south_up*epitch+n_send_ptls_edge[edge_south_up]*gpu_pdata_element_size(), n_recv_ptls_edge[edge_south_up]*gpu_pdata_element_size(), MPI_BYTE, recv_neighbor, 2, m_mpi_comm, &reqs[nreq++]);
            MPI_Irecv(edge_send_buf.data+edge_south_down*epitch+n_send_ptls_edge[edge_south_down]*gpu_pdata_element_size(), n_recv_ptls_edge[edge_south_down]*gpu_pdata_element_size(), MPI_BYTE, recv_neighbor, 3, m_mpi_comm, &reqs[nreq++]);

            MPI_Irecv(face_send_buf.data+face_north*fpitch+n_send_ptls_face[face_north]*gpu_pdata_element_size(), n_recv_ptls_face[face_north]*gpu_pdata_element_size(), MPI_BYTE, recv_neighbor, 4, m_mpi_comm, & reqs[nreq++]);
            MPI_Irecv(face_send_buf.data+face_south*fpitch+n_send_ptls_face[face_south]*gpu_pdata_element_size(), n_recv_ptls_face[face_south]*gpu_pdata_element_size(), MPI_BYTE, recv_neighbor, 5, m_mpi_comm, & reqs[nreq++]);
            }

        if (dir < 4)
            {
            MPI_Irecv(face_send_buf.data+face_up*fpitch+n_send_ptls_face[face_up]*gpu_pdata_element_size(), n_recv_ptls_face[face_up]*gpu_pdata_element_size(), MPI_BYTE, recv_neighbor, 6, m_mpi_comm, & reqs[nreq++]);
            MPI_Irecv(face_send_buf.data+face_down*fpitch+n_send_ptls_face[face_down]*gpu_pdata_element_size(), n_recv_ptls_face[face_down]*gpu_pdata_element_size(), MPI_BYTE, recv_neighbor, 7, m_mpi_comm, & reqs[nreq++]);
            }

        MPI_Irecv(recv_buf.data+n_tot_recv_ptls*gpu_pdata_element_size(), n_recv_ptls*gpu_pdata_element_size(), MPI_BYTE, recv_neighbor, 8, m_mpi_comm, &reqs[nreq++]);

        MPI_Waitall(nreq, reqs, status);

        // update buffer sizes
        for (unsigned int i = 0; i < 12; ++i)
            n_send_ptls_edge[i] += n_recv_ptls_edge[i];

        for (unsigned int i = 0; i < 6; ++i)
            n_send_ptls_face[i] += n_recv_ptls_face[i];

        n_tot_recv_ptls += n_recv_ptls;

        } // end dir loop

    unsigned int old_nparticles = m_pdata->getN();

    // allocate memory for particles that will be received
    m_pdata->addParticles(n_tot_recv_ptls);

        {
        // Finally insert new particles into array and remove the ones that are to be deleted
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::readwrite);
        ArrayHandle<int3> d_image(m_pdata->getImages(), access_location::device, access_mode::readwrite);
        ArrayHandle<unsigned int> d_body(m_pdata->getBodies(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::readwrite);
        ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::readwrite);
        ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::readwrite);

        ArrayHandle<unsigned char> d_remove_mask(m_remove_mask, access_location::device, access_mode::read);

        ArrayHandle<char> d_recv_buf(m_recv_buf, access_location::device, access_mode::read);

        gpu_migrate_fill_particle_arrays(old_nparticles,
                               n_tot_recv_ptls,
                               n_remove_ptls,
                               d_remove_mask.data,
                               d_recv_buf.data,
                               d_pos.data,
                               d_vel.data,
                               d_accel.data,
                               d_image.data,
                               d_charge.data,
                               d_diameter.data,
                               d_body.data,
                               d_orientation.data,
                               d_tag.data,
                               d_rtag.data);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }
   
    m_pdata->removeParticles(n_remove_ptls);

    // notify ParticleData that addition / removal of particles is complete
    m_pdata->notifyParticleSort();

    if (m_prof)
        m_prof->pop();
    }

//! Build a ghost particle list, exchange ghost particle data with neighboring processors
void CommunicatorGPU::exchangeGhosts()
    {
    if (m_prof)
        m_prof->push("exchange_ghosts");

    assert(m_r_ghost < (m_pdata->getBox().getL().x));
    assert(m_r_ghost < (m_pdata->getBox().getL().y));
    assert(m_r_ghost < (m_pdata->getBox().getL().z));

    // reset plans
    m_plan.clear();

    // resize plans
    if (m_plan.size() < m_pdata->getN())
        m_plan.resize(m_pdata->getN());

    /*
     * Mark particles that are part of incomplete bonds for sending
     */
    boost::shared_ptr<BondData> bdata = m_sysdef->getBondData();

    if (bdata->getNumBonds())
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
        {
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
        ArrayHandle<unsigned char> d_plan(m_plan, access_location::device, access_mode::readwrite);

        gpu_make_nonbonded_exchange_plan(d_plan.data,
                                         m_pdata->getN(),
                                         d_pos.data,
                                         m_pdata->getBox(),
                                         m_r_ghost);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    /*
     * Fill send buffers, exchange particles according to plans
     */

    MPI_Request reqs[24];
    MPI_Status status[24];
    unsigned int start_idx = 0;
    unsigned int max_copy_ghosts = 0;
    for (unsigned int dir = 0; dir < 6; dir ++)
        {
        if (! isCommunicating(dir) ) continue;

        if (dir % 2 == 0)
            {
            m_num_copy_ghosts[dir] = m_num_copy_ghosts[dir+1] = 0;

            // resize array of ghost particle indices to copy 
            max_copy_ghosts = m_pdata->getN() + m_pdata->getNGhosts();
            if (m_copy_ghosts[dir].size() < max_copy_ghosts)
                {
                m_copy_ghosts[dir].resize(max_copy_ghosts);
                m_copy_ghosts[dir+1].resize(max_copy_ghosts);
                }
            
            // resize buffers
            if (m_pos_copybuf.size() < 2*max_copy_ghosts)
                {
                m_pos_copybuf.resize(2*max_copy_ghosts);
                m_charge_copybuf.resize(2*max_copy_ghosts);
                m_diameter_copybuf.resize(2*max_copy_ghosts);
                m_plan_copybuf.resize(2*max_copy_ghosts);
                m_tag_copybuf.resize(2*max_copy_ghosts);
                }

            // Fill send buffer
            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
            ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::read);
            ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);
            ArrayHandle<unsigned char> d_plan(m_plan, access_location::device, access_mode::read);

            ArrayHandle<unsigned int> d_copy_ghosts(m_copy_ghosts[dir], access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_copy_ghosts_r(m_copy_ghosts[dir+1], access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar4> d_pos_copybuf(m_pos_copybuf, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar> d_charge_copybuf(m_charge_copybuf, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar> d_diameter_copybuf(m_diameter_copybuf, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned char> d_plan_copybuf(m_plan_copybuf, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_tag_copybuf(m_tag_copybuf, access_location::device, access_mode::overwrite);

            gpu_exchange_ghosts(m_pdata->getN()+m_pdata->getNGhosts() ,
                                d_plan.data,
                                d_copy_ghosts.data,
                                d_copy_ghosts_r.data,
                                d_pos.data,
                                d_pos_copybuf.data,
                                d_pos_copybuf.data + max_copy_ghosts,
                                d_charge.data,
                                d_charge_copybuf.data,
                                d_charge_copybuf.data + max_copy_ghosts,
                                d_diameter.data,
                                d_diameter_copybuf.data,
                                d_diameter_copybuf.data + max_copy_ghosts,
                                d_plan_copybuf.data,
                                d_plan_copybuf.data + max_copy_ghosts,
                                d_tag.data,
                                d_tag_copybuf.data,
                                d_tag_copybuf.data + max_copy_ghosts,
                                m_num_copy_ghosts[dir],
                                m_num_copy_ghosts[dir+1],
                                dir,
                                m_is_at_boundary,
                                m_pdata->getGlobalBox());

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

        unsigned int send_neighbor = m_decomposition->getNeighborRank(dir);

        // we receive from the direction opposite to the one we send to
        unsigned int recv_neighbor;
        if (dir % 2 == 0)
            recv_neighbor = m_decomposition->getNeighborRank(dir+1);
        else
            recv_neighbor = m_decomposition->getNeighborRank(dir-1);

        unsigned int shift = (dir % 2 ) ? 10 : 0;

        unsigned int offset_in_pdata = (dir % 2) ? m_num_recv_ghosts[dir-1] : 0;
        unsigned int offset_in_copybuf = (dir % 2) ? max_copy_ghosts : 0;

        if (dir % 2 == 0 )
            {
            if (m_prof)
                m_prof->push("MPI send/recv");


            // communicate size of the message that will contain the particle data
            MPI_Isend(&m_num_copy_ghosts[dir], sizeof(unsigned int), MPI_BYTE, send_neighbor, 0, m_mpi_comm, & reqs[0]);
            MPI_Irecv(&m_num_recv_ghosts[dir], sizeof(unsigned int), MPI_BYTE, recv_neighbor, 0, m_mpi_comm, & reqs[1]);
            // reverse send & receive neighbor for opposite direction
            MPI_Isend(&m_num_copy_ghosts[dir+1], sizeof(unsigned int), MPI_BYTE, recv_neighbor, 0, m_mpi_comm, & reqs[2]);
            MPI_Irecv(&m_num_recv_ghosts[dir+1], sizeof(unsigned int), MPI_BYTE, send_neighbor, 0, m_mpi_comm, & reqs[3]);
            MPI_Waitall(4, reqs, status);

            if (m_prof)
                m_prof->pop();

            // append ghosts at the end of particle data array
            start_idx = m_pdata->getN() + m_pdata->getNGhosts();

            // accommodate new ghost particles
            m_pdata->addGhostParticles(m_num_recv_ghosts[dir]+m_num_recv_ghosts[dir+1]);

            // resize plan array (and clear new plans)
            m_plan.resize(m_pdata->getN() + m_pdata->getNGhosts());
            }

        // exchange particle data, write directly to the particle data arrays
        if (m_prof)
            m_prof->push("MPI send/recv");

#ifdef ENABLE_MPI_CUDA
            {
            ArrayHandle<Scalar4> d_pos_copybuf(m_pos_copybuf, access_location::device, access_mode::read);
            ArrayHandle<Scalar> d_charge_copybuf(m_charge_copybuf, access_location::device, access_mode::read);
            ArrayHandle<Scalar> d_diameter_copybuf(m_diameter_copybuf, access_location::device, access_mode::read);
            ArrayHandle<unsigned char> d_plan_copybuf(m_plan_copybuf, access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_tag_copybuf(m_tag_copybuf, access_location::device, access_mode::read);

            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::readwrite);
            ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::readwrite);
            ArrayHandle<unsigned char> d_plan(m_plan, access_location::device, access_mode::readwrite);

            MPI_Isend(d_plan_copybuf.data + offset_in_copybuf, m_num_copy_ghosts[dir]*sizeof(unsigned char), MPI_BYTE, send_neighbor, 2+shift, m_mpi_comm, &reqs[4+shift]);
            MPI_Irecv(d_plan.data + start_idx + offset_in_pdata, m_num_recv_ghosts[dir]*sizeof(unsigned char), MPI_BYTE, recv_neighbor, 2+shift, m_mpi_comm, &reqs[5+shift]);

            MPI_Isend(d_pos_copybuf.data + offset_in_copybuf, m_num_copy_ghosts[dir]*sizeof(Scalar4), MPI_BYTE, send_neighbor, 3+shift, m_mpi_comm, &reqs[6+shift]);
            MPI_Irecv(d_pos.data + start_idx + offset_in_pdata, m_num_recv_ghosts[dir]*sizeof(Scalar4), MPI_BYTE, recv_neighbor, 3+shift, m_mpi_comm, &reqs[7+shift]);

            MPI_Isend(d_tag_copybuf.data + offset_in_copybuf, m_num_copy_ghosts[dir]*sizeof(unsigned int), MPI_BYTE, send_neighbor, 4+shift, m_mpi_comm, &reqs[8+shift]);
            MPI_Irecv(d_tag.data + start_idx + offset_in_pdata, m_num_recv_ghosts[dir]*sizeof(unsigned int), MPI_BYTE, recv_neighbor, 4+shift, m_mpi_comm, &reqs[9+shift]);

            MPI_Isend(d_charge_copybuf.data + offset_in_copybuf, m_num_copy_ghosts[dir]*sizeof(Scalar), MPI_BYTE, send_neighbor, 5+shift, m_mpi_comm, &reqs[10+shift]);
            MPI_Irecv(d_charge.data + start_idx + offset_in_pdata, m_num_recv_ghosts[dir]*sizeof(Scalar), MPI_BYTE, recv_neighbor, 5+shift, m_mpi_comm, &reqs[11+shift]);

            MPI_Isend(d_diameter_copybuf.data + offset_in_copybuf, m_num_copy_ghosts[dir]*sizeof(Scalar), MPI_BYTE, send_neighbor, 6+shift, m_mpi_comm, &reqs[12+shift]);
            MPI_Irecv(d_diameter.data + start_idx + offset_in_pdata, m_num_recv_ghosts[dir]*sizeof(Scalar), MPI_BYTE, recv_neighbor, 6+shift, m_mpi_comm, &reqs[13+shift]);
            }
#else
            {
            ArrayHandle<Scalar4> h_pos_copybuf(m_pos_copybuf, access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_charge_copybuf(m_charge_copybuf, access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_diameter_copybuf(m_diameter_copybuf, access_location::host, access_mode::read);
            ArrayHandle<unsigned char> h_plan_copybuf(m_plan_copybuf, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_tag_copybuf(m_tag_copybuf, access_location::host, access_mode::read);

            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::readwrite);
            ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::readwrite);
            ArrayHandle<unsigned char> h_plan(m_plan, access_location::host, access_mode::readwrite);

            MPI_Isend(h_plan_copybuf.data + offset_in_copybuf, m_num_copy_ghosts[dir]*sizeof(unsigned char), MPI_BYTE, send_neighbor, 2+shift, m_mpi_comm, &reqs[4+shift]);
            MPI_Irecv(h_plan.data + start_idx + offset_in_pdata, m_num_recv_ghosts[dir]*sizeof(unsigned char), MPI_BYTE, recv_neighbor, 2+shift, m_mpi_comm, &reqs[5+shift]);

            MPI_Isend(h_pos_copybuf.data + offset_in_copybuf, m_num_copy_ghosts[dir]*sizeof(Scalar4), MPI_BYTE, send_neighbor, 3+shift, m_mpi_comm, &reqs[6+shift]);
            MPI_Irecv(h_pos.data + start_idx + offset_in_pdata, m_num_recv_ghosts[dir]*sizeof(Scalar4), MPI_BYTE, recv_neighbor, 3+shift, m_mpi_comm, &reqs[7+shift]);

            MPI_Isend(h_tag_copybuf.data + offset_in_copybuf, m_num_copy_ghosts[dir]*sizeof(unsigned int), MPI_BYTE, send_neighbor, 4+shift, m_mpi_comm, &reqs[8+shift]);
            MPI_Irecv(h_tag.data + start_idx + offset_in_pdata, m_num_recv_ghosts[dir]*sizeof(unsigned int), MPI_BYTE, recv_neighbor, 4+shift, m_mpi_comm, &reqs[9+shift]);

            MPI_Isend(h_charge_copybuf.data + offset_in_copybuf, m_num_copy_ghosts[dir]*sizeof(Scalar), MPI_BYTE, send_neighbor, 5+shift, m_mpi_comm, &reqs[10+shift]);
            MPI_Irecv(h_charge.data + start_idx + offset_in_pdata, m_num_recv_ghosts[dir]*sizeof(Scalar), MPI_BYTE, recv_neighbor, 5+shift, m_mpi_comm, &reqs[11+shift]);

            MPI_Isend(h_diameter_copybuf.data + offset_in_copybuf, m_num_copy_ghosts[dir]*sizeof(Scalar), MPI_BYTE, send_neighbor, 6+shift, m_mpi_comm, &reqs[12+shift]);
            MPI_Irecv(h_diameter.data + start_idx + offset_in_pdata, m_num_recv_ghosts[dir]*sizeof(Scalar), MPI_BYTE, recv_neighbor, 6+shift, m_mpi_comm, &reqs[13+shift]);
            }
#endif

        if (dir%2) MPI_Waitall(20, reqs+4, status+4);
                
        if (m_prof)
            m_prof->pop();

        }

        {
        ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::readwrite);

        gpu_update_rtag(m_pdata->getNGhosts(), m_pdata->getN(), d_tag.data+ m_pdata->getN(), d_rtag.data);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        }

    // we have updated ghost particles, so inform ParticleData about this
    m_pdata->notifyGhostParticleNumberChange();

    if (m_prof)
        m_prof->pop();
    }

//! Update ghost particle positions
void CommunicatorGPU::copyGhosts()
    {
    // we have a current m_copy_ghosts list which contain the indices of particles
    // to send to neighboring processors
    if (m_prof)
        m_prof->push("copy_ghosts");

    unsigned int num_tot_recv_ghosts = 0; // total number of ghosts received


    MPI_Request reqs[4];
    MPI_Status status[4];
    unsigned int max_copy_ghosts = 0;
    for (unsigned int i = 0; i < 3; i++)
        {
        unsigned int n = m_num_copy_ghosts[2*i]+m_num_copy_ghosts[2*i+1];
        if (n > max_copy_ghosts)
            max_copy_ghosts = n;
        }
    
    if (max_copy_ghosts > m_pos_copybuf.size())
        m_pos_copybuf.resize(max_copy_ghosts);

    for (unsigned int dir = 0; dir < 6; dir ++)
        {

        if (! isCommunicating(dir) ) continue;

        unsigned int offset = (dir % 2) ? m_num_copy_ghosts[dir-1] : 0;

        // Pack send data for direction dir and dir+1 (opposite) simultaneously.
        // We assume that they are independent, i.e. a send in direction dir+1
        // does not contain any ghosts received from that direction previously.
        if ((dir % 2) == 0)
            {

            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
            ArrayHandle<Scalar4> d_pos_copybuf(m_pos_copybuf, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_copy_ghosts(m_copy_ghosts[dir], access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_copy_ghosts_r(m_copy_ghosts[dir+1], access_location::device, access_mode::read);

            gpu_copy_ghosts(m_num_copy_ghosts[dir],
                            m_num_copy_ghosts[dir+1],
                            d_pos.data, 
                            d_copy_ghosts.data,
                            d_copy_ghosts_r.data,
                            d_pos_copybuf.data,
                            d_pos_copybuf.data + m_num_copy_ghosts[dir],
                            dir,
                            m_is_at_boundary,
                            m_pdata->getGlobalBox(),
                            0);

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

        unsigned int send_neighbor = m_decomposition->getNeighborRank(dir);

        // we receive from the direction opposite to the one we send to
        unsigned int recv_neighbor;
        if (dir % 2 == 0)
            recv_neighbor = m_decomposition->getNeighborRank(dir+1);
        else
            recv_neighbor = m_decomposition->getNeighborRank(dir-1);

        unsigned int start_idx;

        if (m_prof && (dir%2 == 0))
            m_prof->push("MPI send/recv");


        start_idx = m_pdata->getN() + num_tot_recv_ghosts;

        num_tot_recv_ghosts += m_num_recv_ghosts[dir];

        unsigned int shift = (dir % 2) ? 2 : 0;
#ifdef ENABLE_MPI_CUDA
            {
            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar4> d_pos_copybuf(m_pos_copybuf, access_location::device, access_mode::read);

            // exchange particle data, write directly to the particle data arrays
            MPI_Isend(d_pos_copybuf.data + offset, m_num_copy_ghosts[dir]*sizeof(Scalar4), MPI_BYTE, send_neighbor, dir, m_mpi_comm, &reqs[0+shift]);
            MPI_Irecv(d_pos.data + start_idx, m_num_recv_ghosts[dir]*sizeof(Scalar4), MPI_BYTE, recv_neighbor, dir, m_mpi_comm, &reqs[1+shift]);
            }
#else
            {
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_pos_copybuf(m_pos_copybuf, access_location::host, access_mode::read);

            // exchange particle data, write directly to the particle data arrays
            MPI_Isend(h_pos_copybuf.data + offset, m_num_copy_ghosts[dir]*sizeof(Scalar4), MPI_BYTE, send_neighbor, dir, m_mpi_comm, &reqs[0+shift]);
            MPI_Irecv(h_pos.data + start_idx, m_num_recv_ghosts[dir]*sizeof(Scalar4), MPI_BYTE, recv_neighbor, dir, m_mpi_comm, &reqs[1+shift]);
            }
#endif

        if (dir %2)
            {
            MPI_Waitall(4, reqs, status);

            if (m_prof)
                {
                unsigned int n = m_num_recv_ghosts[dir-1]+m_num_copy_ghosts[dir-1]+m_num_recv_ghosts[dir]+m_num_copy_ghosts[dir];
                m_prof->pop(0, n*sizeof(Scalar4));
                }
            }

        } // end dir loop

        if (m_prof)
            m_prof->pop();
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
