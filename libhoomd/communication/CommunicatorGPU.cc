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

//! This is a lookup from corner to plan
unsigned int corner_plan_lookup[] = { send_east | send_north | send_up,
                                      send_east | send_north | send_down,
                                      send_east | send_south | send_up,
                                      send_east | send_south | send_down,
                                      send_west | send_north | send_up,
                                      send_west | send_north | send_down,
                                      send_west | send_south | send_up,
                                      send_west | send_south | send_down};

//! Lookup from edge to plan
unsigned int edge_plan_lookup[] = { send_east | send_north,
                                    send_east | send_south,
                                    send_east | send_up,
                                    send_east | send_down,
                                    send_west | send_north,
                                    send_west | send_south,
                                    send_west | send_up,
                                    send_west | send_down,
                                    send_north | send_up,
                                    send_north | send_down,
                                    send_south | send_up,
                                    send_south | send_down };

//! Lookup from face to plan
unsigned int face_plan_lookup[] = { send_east,
                                    send_west,
                                    send_north,
                                    send_south,
                                    send_up,
                                    send_down };



//! Constructor
ghost_gpu_thread::ghost_gpu_thread(boost::shared_ptr<const ExecutionConfiguration> exec_conf,
                                   CommunicatorGPU *communicator)
    : m_exec_conf(exec_conf),
      m_communicator(communicator),
      m_recv_buf_size(0),
      m_face_update_buf_size(0),
      m_edge_update_buf_size(0),
      m_corner_update_buf_size(0),
      m_buffers_allocated(false)
    { }

//! Destructor
ghost_gpu_thread::~ghost_gpu_thread()
    {
    if (m_buffers_allocated)
        {
        m_exec_conf->useContext();
        cudaFreeHost(h_face_update_buf);
        cudaFreeHost(h_edge_update_buf);
        cudaFreeHost(h_corner_update_buf);
        cudaFreeHost(h_recv_buf);
        m_exec_conf->releaseContext();
        }
    }

//! Main routine of ghost update worker thread
void ghost_gpu_thread::operator()(WorkQueue<ghost_gpu_thread_params>& queue, boost::barrier& barrier)
    {
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
    unsigned int n_tot_local_ghosts = 0;

    unsigned int n_copy_ghosts_face[6];
    unsigned int n_copy_ghosts_edge[12];
    unsigned int n_copy_ghosts_corner[8];

        {
        ArrayHandle<unsigned int> h_n_local_ghosts_face(params.n_local_ghosts_face, access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_n_local_ghosts_edge(params.n_local_ghosts_edge, access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_n_local_ghosts_corner(params.n_local_ghosts_corner, access_location::host, access_mode::read);

        for (unsigned int i = 0; i < 12; ++i)
            {
            n_copy_ghosts_edge[i] = h_n_local_ghosts_edge.data[i];
            n_tot_local_ghosts += h_n_local_ghosts_face.data[i];
            }
        for (unsigned int i = 0; i < 6; ++i)
            {
            n_copy_ghosts_face[i] = h_n_local_ghosts_face.data[i];
            n_tot_local_ghosts += h_n_local_ghosts_edge.data[i];
            }
        for (unsigned int i = 0; i < 8; ++i)
            {
            n_copy_ghosts_corner[i] = h_n_local_ghosts_corner.data[i];
            n_tot_local_ghosts += h_n_local_ghosts_corner.data[i];
            }
        }

        {
        ArrayHandle<unsigned int> d_n_local_ghosts_face(params.n_local_ghosts_face, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_n_local_ghosts_edge(params.n_local_ghosts_edge, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_n_local_ghosts_corner(params.n_local_ghosts_corner, access_location::device, access_mode::read);

 
        m_exec_conf->useContext();
        gpu_update_ghosts_pack(n_tot_local_ghosts,
                            params.ghost_idx_face_handle,
                            params.ghost_idx_face_pitch,
                            params.ghost_idx_edge_handle, 
                            params.ghost_idx_edge_pitch,
                            params.ghost_idx_corner_handle,
                            params.ghost_idx_corner_pitch,
                            params.pos_handle,
                            params.corner_update_buf_handle,
                            params.corner_update_buf_pitch, 
                            params.edge_update_buf_handle,
                            params.edge_update_buf_pitch,
                            params.face_update_buf_handle,
                            params.face_update_buf_pitch,
                            d_n_local_ghosts_corner.data,
                            d_n_local_ghosts_edge.data,
                            d_n_local_ghosts_face.data,
                            m_exec_conf->getThreadStream(m_thread_id));

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        m_exec_conf->releaseContext();
        }

    unsigned int n_tot_recv_ghosts_local = 0;

    // allocate or resize host buffers if necessary
    unsigned int new_size;
    new_size = 6*params.face_update_buf_pitch;
    if (! m_buffers_allocated || m_face_update_buf_size != new_size)
        {
        m_exec_conf->useContext();
        if (m_buffers_allocated) cudaFreeHost(h_face_update_buf);
        cudaHostAlloc(&h_face_update_buf, new_size, cudaHostAllocDefault);
        m_exec_conf->releaseContext();
        m_face_update_buf_size = new_size;
        }

    new_size = 12*params.edge_update_buf_pitch;
    if (! m_buffers_allocated || m_edge_update_buf_size != new_size)
        {
        m_exec_conf->useContext();
        if (m_buffers_allocated) cudaFreeHost(h_edge_update_buf);
        cudaHostAlloc(&h_edge_update_buf, new_size, cudaHostAllocDefault);
        m_exec_conf->releaseContext();
        m_edge_update_buf_size = new_size;
        }

    new_size = 8*params.corner_update_buf_pitch;
    if (! m_buffers_allocated || m_corner_update_buf_size != new_size)
        {
        m_exec_conf->useContext();
        if (m_buffers_allocated) cudaFreeHost(h_corner_update_buf);
        cudaHostAlloc(&h_corner_update_buf, new_size, cudaHostAllocDefault);
        m_exec_conf->releaseContext();
        m_corner_update_buf_size = new_size;
        }

    new_size = params.recv_ghosts_local_size;
    if (! m_buffers_allocated || m_recv_buf_size != new_size)
        {
        m_exec_conf->useContext();
        if (m_buffers_allocated) cudaFreeHost(h_recv_buf);
        cudaHostAlloc(&h_recv_buf, new_size, cudaHostAllocDefault);
        m_exec_conf->releaseContext();
        m_recv_buf_size = new_size;
        }

    m_buffers_allocated = true;

        {
        // copy data from device to host
        cudaStream_t stream = m_exec_conf->getThreadStream(m_thread_id);
        cudaEvent_t ev = m_exec_conf->getThreadEvent(m_thread_id);

        ArrayHandle<unsigned int> h_n_local_ghosts_face(params.n_local_ghosts_face, access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_n_local_ghosts_edge(params.n_local_ghosts_edge, access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_n_local_ghosts_corner(params.n_local_ghosts_corner, access_location::host, access_mode::read);

        m_exec_conf->useContext();

        unsigned int fpitch = params.face_update_buf_pitch;
        unsigned int epitch = params.edge_update_buf_pitch;
        unsigned int cpitch = params.corner_update_buf_pitch;

        for (unsigned int i = 0; i < 6; ++i)
            if (h_n_local_ghosts_face.data[i])
                cudaMemcpyAsync(h_face_update_buf+i*fpitch, params.face_update_buf_handle + i*fpitch,
                    h_n_local_ghosts_face.data[i]*gpu_update_element_size(), cudaMemcpyDeviceToHost,stream);

        for (unsigned int i = 0; i < 12; ++i)
            if (h_n_local_ghosts_edge.data[i])
                cudaMemcpyAsync(h_edge_update_buf+i*epitch, params.edge_update_buf_handle + i*epitch,
                    h_n_local_ghosts_edge.data[i]*gpu_update_element_size(), cudaMemcpyDeviceToHost,stream);

        for (unsigned int i = 0; i < 8; ++i)
            if (h_n_local_ghosts_corner.data[i])
                cudaMemcpyAsync(h_corner_update_buf+i*cpitch, params.corner_update_buf_handle + i*cpitch,
                    h_n_local_ghosts_corner.data[i]*gpu_update_element_size(), cudaMemcpyDeviceToHost,stream);

        // wait for D->H copy to finish
        cudaEventRecord(ev, stream);
        cudaEventSynchronize(ev);
        m_exec_conf->releaseContext();
        }

    for (unsigned int face = 0; face < 6; ++face)
        {
        if (! m_communicator->isCommunicating(face)) continue;

        ArrayHandle<unsigned int> h_n_recv_ghosts_face(params.n_recv_ghosts_face, access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_n_recv_ghosts_edge(params.n_recv_ghosts_edge, access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_n_recv_ghosts_local(params.n_recv_ghosts_local, access_location::host, access_mode::read);

        m_communicator->communicateStepTwo(face,
                                           h_corner_update_buf,
                                           h_edge_update_buf,
                                           h_face_update_buf,
                                           params.corner_update_buf_pitch,
                                           params.edge_update_buf_pitch,
                                           params.face_update_buf_pitch,
                                           h_recv_buf,
                                           n_copy_ghosts_corner,
                                           n_copy_ghosts_edge,
                                           n_copy_ghosts_face,
                                           params.recv_ghosts_local_size,
                                           n_tot_recv_ghosts_local,
                                           gpu_update_element_size(),
                                           false);
        // update send buffer sizes
        for (unsigned int i = 0; i < 12; ++i)
            n_copy_ghosts_edge[i] += h_n_recv_ghosts_edge.data[face*12+i];
        for (unsigned int i = 0; i < 6; ++i)
            n_copy_ghosts_face[i] += h_n_recv_ghosts_face.data[face*6+i];

        n_tot_recv_ghosts_local += h_n_recv_ghosts_local.data[face];
        } // end dir loop

    unsigned int n_forward_ghosts_face[6];
    unsigned int n_forward_ghosts_edge[12];
        {
        ArrayHandle<unsigned int> h_n_local_ghosts_face(params.n_local_ghosts_face, access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_n_local_ghosts_edge(params.n_local_ghosts_edge, access_location::host, access_mode::read);

        for (unsigned int i = 0; i < 6; ++i)
            n_forward_ghosts_face[i] = n_copy_ghosts_face[i] - h_n_local_ghosts_face.data[i];

        for (unsigned int i = 0; i < 12; ++i)
            n_forward_ghosts_edge[i] = n_copy_ghosts_edge[i] - h_n_local_ghosts_edge.data[i];
        }

    // total up number of received ghosts
    unsigned int n_tot_recv_ghosts = n_tot_recv_ghosts_local;
    for (unsigned int i = 0; i < 6; ++i)
        n_tot_recv_ghosts += n_forward_ghosts_face[i];
    for (unsigned int i = 0; i < 12; ++i)
        n_tot_recv_ghosts += n_forward_ghosts_edge[i];

        {
        ArrayHandle<unsigned int> h_n_local_ghosts_face(params.n_local_ghosts_face, access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_n_local_ghosts_edge(params.n_local_ghosts_edge, access_location::host, access_mode::read);

        cudaStream_t stream = m_exec_conf->getThreadStream(m_thread_id);
        unsigned int fpitch = params.face_update_buf_pitch;
        unsigned int epitch = params.edge_update_buf_pitch;

        // copy data back host->device as needed
        m_exec_conf->useContext();

        unsigned int element_size = gpu_update_element_size();
        for (unsigned int i = 0; i < 6; ++i)
            if (n_forward_ghosts_face[i])
                cudaMemcpyAsync(params.face_update_buf_handle + h_n_local_ghosts_face.data[i]*element_size + i * fpitch,
                           h_face_update_buf + h_n_local_ghosts_face.data[i]*element_size + i*fpitch,
                           n_forward_ghosts_face[i]*element_size,
                           cudaMemcpyHostToDevice,
                           stream);

        for (unsigned int i = 0; i < 12; ++i)
            if (n_forward_ghosts_edge[i])
                cudaMemcpyAsync(params.edge_update_buf_handle + h_n_local_ghosts_edge.data[i]*element_size + i * epitch,
                           h_edge_update_buf + h_n_local_ghosts_edge.data[i]*element_size + i*epitch,
                           n_forward_ghosts_edge[i]*element_size,
                           cudaMemcpyHostToDevice,
                           stream);

        if (n_tot_recv_ghosts_local)
            cudaMemcpyAsync(params.update_recv_buf_handle,
                       h_recv_buf,
                       n_tot_recv_ghosts_local*element_size,
                       cudaMemcpyHostToDevice,
                       stream);

        m_exec_conf->releaseContext(); 
        }

        {
        ArrayHandle<unsigned int> d_n_local_ghosts_face(params.n_local_ghosts_face, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_n_local_ghosts_edge(params.n_local_ghosts_edge, access_location::device, access_mode::read);

        ArrayHandle<unsigned int> d_n_recv_ghosts_face(params.n_recv_ghosts_face, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_n_recv_ghosts_edge(params.n_recv_ghosts_edge, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_n_recv_ghosts_local(params.n_recv_ghosts_local, access_location::device, access_mode::read);

        // unpack particles
        m_exec_conf->useContext();
        gpu_update_ghosts_unpack(params.N,
                                 n_tot_recv_ghosts,
                                 d_n_local_ghosts_face.data,
                                 d_n_local_ghosts_edge.data,
                                 n_tot_recv_ghosts_local,
                                 d_n_recv_ghosts_local.data,
                                 d_n_recv_ghosts_face.data,
                                 d_n_recv_ghosts_edge.data,
                                 params.face_update_buf_handle,
                                 params.face_update_buf_pitch,
                                 params.edge_update_buf_handle,
                                 params.edge_update_buf_pitch,
                                 params.update_recv_buf_handle,
                                 params.pos_handle,
                                 params.d_ghost_plan,
                                 params.global_box,
                                 m_exec_conf->getThreadStream(m_thread_id));

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_exec_conf->releaseContext();
        }
     
    } 

//! Constructor
CommunicatorGPU::CommunicatorGPU(boost::shared_ptr<SystemDefinition> sysdef,
                                 boost::shared_ptr<DomainDecomposition> decomposition)
    : Communicator(sysdef, decomposition), m_remove_mask(m_exec_conf),
      m_buffers_allocated(false),
      m_resize_factor(9.f/8.f),
      m_thread_created(false),
      m_barrier(2)
    { 
    m_exec_conf->msg->notice(5) << "Constructing CommunicatorGPU" << std::endl;

    // allocate temporary GPU buffers and set some global properties
    unsigned int is_communicating[6];
    for (unsigned int face=0; face<6; ++face)
        is_communicating[face] = isCommunicating(face) ? 1 : 0;

    m_exec_conf->useContext();
    gpu_allocate_tmp_storage(is_communicating,m_is_at_boundary);

    cudaEventCreate(&m_event);
    m_exec_conf->releaseContext();

    GPUFlags<unsigned int> condition(m_exec_conf);
    m_condition.swap(condition);

    // create group corresponding to communicator
    MPI_Comm_group(m_mpi_comm, &m_comm_group);

    // create one-sided communication windows
    for (unsigned int i = 0; i < 6; ++i)
        MPIX_Win_create_dynamic(MPI_INFO_NULL, m_mpi_comm, &m_win_face[i]);

    for (unsigned int i = 0; i < 12; ++i)
        MPIX_Win_create_dynamic(MPI_INFO_NULL, m_mpi_comm, &m_win_edge[i]);

    MPIX_Win_create_dynamic(MPI_INFO_NULL, m_mpi_comm, &m_win_local);
    }

//! Destructor
CommunicatorGPU::~CommunicatorGPU()
    {
    m_exec_conf->msg->notice(5) << "Destroying CommunicatorGPU";

    if (m_buffers_allocated)
        deallocateBuffers();

    m_exec_conf->useContext();
    gpu_deallocate_tmp_storage();
    
    cudaEventDestroy(m_event);
    m_exec_conf->releaseContext();

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

    // create a worker thread for ghost updates
    if (! m_thread_created)
        {
        m_worker_thread = boost::thread(ghost_gpu_thread(m_exec_conf,
                                                         this),
                                        boost::ref(m_work_queue), boost::ref(m_barrier));
        m_thread_created = true;
        }

    // fill thread parameters
    ghost_gpu_thread_params params(
        m_ghost_idx_face.acquire(access_location::device, access_mode::read),
        m_ghost_idx_face.getPitch(),
        m_ghost_idx_edge.acquire(access_location::device, access_mode::read),
        m_ghost_idx_edge.getPitch(),
        m_ghost_idx_corner.acquire(access_location::device, access_mode::read),
        m_ghost_idx_corner.getPitch(),
        m_corner_update_buf.acquire(access_location::device, access_mode::overwrite),
        m_corner_update_buf.getPitch(),
        m_edge_update_buf.acquire(access_location::device, access_mode::overwrite),
        m_edge_update_buf.getPitch(),
        m_face_update_buf.acquire(access_location::device, access_mode::overwrite),
        m_face_update_buf.getPitch(),
        m_update_recv_buf.acquire(access_location::device, access_mode::overwrite),
        m_ghost_plan.acquire(access_location::device, access_mode::read),
        m_pdata->getN(), 
        m_update_recv_buf.getNumElements(),
        m_n_recv_ghosts_edge,
        m_n_recv_ghosts_face,
        m_n_recv_ghosts_local,
        m_n_local_ghosts_corner,
        m_n_local_ghosts_edge,
        m_n_local_ghosts_face,
        m_pdata->getPositions().acquire(access_location::device, access_mode::readwrite_shared),
        m_pdata->getGlobalBox());

    // post the parameters to the worker thread
    m_work_queue.push(ghost_gpu_thread_params(params));

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
    m_ghost_idx_face.release();
    m_face_update_buf.release();
    m_ghost_idx_edge.release();
    m_edge_update_buf.release();
    m_ghost_idx_corner.release();
    m_corner_update_buf.release();

    m_ghost_plan.release();

    m_update_recv_buf.release();
    m_pdata->getPositions().release();
    }

void CommunicatorGPU::allocateBuffers()
    {
    /*
     * initial size of particle send buffers = max of avg. number of ptls in skin layer in any direction
     */ 
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
   
    /*
     * initial size of ghost send buffers = max of avg number of ptls in ghost layer in every direction
     */ 
    maxx = m_pdata->getN()*m_r_ghost/L.x;
    maxy = m_pdata->getN()*m_r_ghost/L.y;
    maxz = m_pdata->getN()*m_r_ghost/L.z;

    m_max_copy_ghosts_face = 1;
    m_max_copy_ghosts_face = m_max_copy_ghosts_face > maxx ? m_max_copy_ghosts_face : maxx;
    m_max_copy_ghosts_face = m_max_copy_ghosts_face > maxy ? m_max_copy_ghosts_face : maxy;
    m_max_copy_ghosts_face = m_max_copy_ghosts_face > maxz ? m_max_copy_ghosts_face : maxz;

    GPUArray<char> face_ghosts_buf(gpu_ghost_element_size()*m_max_copy_ghosts_face, 6, m_exec_conf);
    m_face_ghosts_buf.swap(face_ghosts_buf);

    GPUArray<char> face_update_buf(gpu_update_element_size()*m_max_copy_ghosts_face, 6, m_exec_conf);
    m_face_update_buf.swap(face_update_buf);

    maxxy = m_pdata->getN()*m_r_ghost*m_r_ghost/L.x/L.y;
    maxxz = m_pdata->getN()*m_r_ghost*m_r_ghost/L.x/L.z;
    maxyz = m_pdata->getN()*m_r_ghost*m_r_ghost/L.y/L.z;

    m_max_copy_ghosts_edge = 1;
    m_max_copy_ghosts_edge = m_max_copy_ghosts_edge > maxxy ? m_max_copy_ghosts_edge : maxxy;
    m_max_copy_ghosts_edge = m_max_copy_ghosts_edge > maxxz ? m_max_copy_ghosts_edge : maxxz;
    m_max_copy_ghosts_edge = m_max_copy_ghosts_edge > maxyz ? m_max_copy_ghosts_edge : maxyz;

    GPUArray<char> edge_ghosts_buf(gpu_ghost_element_size()*m_max_copy_ghosts_edge, 12, m_exec_conf);
    m_edge_ghosts_buf.swap(edge_ghosts_buf);

    GPUArray<char> edge_update_buf(gpu_update_element_size()*m_max_copy_ghosts_edge, 12, m_exec_conf);
    m_edge_update_buf.swap(edge_update_buf);

    maxxyz = m_pdata->getN()*m_r_ghost*m_r_ghost*m_r_ghost/L.x/L.y/L.z;
    m_max_copy_ghosts_corner = maxxyz > 1 ? maxxyz : 1;

    GPUArray<char> corner_ghosts_buf(gpu_ghost_element_size()*m_max_copy_ghosts_corner, 8, m_exec_conf);
    m_corner_ghosts_buf.swap(corner_ghosts_buf);

    GPUArray<char> corner_update_buf(gpu_update_element_size()*m_max_copy_ghosts_corner, 8, m_exec_conf);
    m_corner_update_buf.swap(corner_update_buf);

    m_max_recv_ghosts = m_max_copy_ghosts_face;
    GPUArray<char> ghost_recv_buf(gpu_ghost_element_size()*m_max_recv_ghosts, m_exec_conf);
    m_ghosts_recv_buf.swap(ghost_recv_buf);

    GPUArray<char> update_recv_buf(gpu_update_element_size()*m_max_recv_ghosts,m_exec_conf);
    m_update_recv_buf.swap(update_recv_buf);

    // allocate ghost index lists
    GPUArray<unsigned int> ghost_idx_face(m_max_copy_ghosts_face, 6, m_exec_conf);
    m_ghost_idx_face.swap(ghost_idx_face);

    GPUArray<unsigned int> ghost_idx_edge(m_max_copy_ghosts_edge, 12, m_exec_conf);
    m_ghost_idx_edge.swap(ghost_idx_edge);
    
    GPUArray<unsigned int> ghost_idx_corner(m_max_copy_ghosts_corner, 8, m_exec_conf);
    m_ghost_idx_corner.swap(ghost_idx_corner);

    // allocate ghost plan buffer
    GPUArray<unsigned int> ghost_plan(m_max_copy_ghosts_face*6, m_exec_conf);
    m_ghost_plan.swap(ghost_plan);

    // allocate mirror host buffers
    m_exec_conf->useContext();
    cudaHostAlloc(&h_face_ghosts_buf, m_face_ghosts_buf.getNumElements(), cudaHostAllocDefault);
    cudaHostAlloc(&h_edge_ghosts_buf, m_edge_ghosts_buf.getNumElements(), cudaHostAllocDefault);
    cudaHostAlloc(&h_corner_ghosts_buf, m_corner_ghosts_buf.getNumElements(), cudaHostAllocDefault);
    cudaHostAlloc(&h_ghosts_recv_buf, m_ghosts_recv_buf.getNumElements(), cudaHostAllocDefault);

    cudaHostAlloc(&h_corner_send_buf, m_corner_send_buf.getNumElements(), cudaHostAllocDefault);
    cudaHostAlloc(&h_edge_send_buf, m_edge_send_buf.getNumElements(), cudaHostAllocDefault);
    cudaHostAlloc(&h_face_send_buf, m_face_send_buf.getNumElements(), cudaHostAllocDefault);
    cudaHostAlloc(&h_recv_buf, m_recv_buf.getNumElements(), cudaHostAllocDefault);
    m_exec_conf->releaseContext();

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

    GPUArray<unsigned int> n_send_ptls_face(6, m_exec_conf);
    m_n_send_ptls_face.swap(n_send_ptls_face);
    GPUArray<unsigned int> n_send_ptls_edge(12, m_exec_conf);
    m_n_send_ptls_edge.swap(n_send_ptls_edge);
    GPUArray<unsigned int> n_send_ptls_corner(8, m_exec_conf);
    m_n_send_ptls_corner.swap(n_send_ptls_corner);

    m_buffers_allocated = true;
    }

void CommunicatorGPU::deallocateBuffers()
    {
    m_exec_conf->useContext();
    cudaFreeHost(h_ghosts_recv_buf);
    cudaFreeHost(h_corner_ghosts_buf);
    cudaFreeHost(h_edge_ghosts_buf);
    cudaFreeHost(h_face_ghosts_buf);

    cudaFreeHost(h_recv_buf);
    cudaFreeHost(h_corner_send_buf);
    cudaFreeHost(h_edge_send_buf);
    cudaFreeHost(h_face_send_buf);
    m_exec_conf->releaseContext();
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

        m_exec_conf->useContext();
        gpu_reset_rtags(m_pdata->getNGhosts(),
                        d_tag.data + m_pdata->getN(),
                        d_rtag.data);
        m_exec_conf->releaseContext();

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    // reset ghost particle number
    m_pdata->removeAllGhostParticles();

    // initialization of buffers
    if (! m_buffers_allocated)
        allocateBuffers();

    if (m_remove_mask.getNumElements() < m_pdata->getN())
        {
        unsigned int new_size = 1;
        while (new_size < m_pdata->getN())
                new_size = ceilf((float)new_size*m_resize_factor);
 
        m_remove_mask.resize(new_size);
        }

    unsigned int n_send_ptls_face[6];
    unsigned int n_send_ptls_edge[12];
    unsigned int n_send_ptls_corner[8];


    unsigned int condition;
    do
        {
        if (m_corner_send_buf.getPitch() < m_max_send_ptls_corner*gpu_pdata_element_size())
            {
            unsigned int old_pitch = m_corner_send_buf.getPitch();
            m_corner_send_buf.resize(m_max_send_ptls_corner*gpu_pdata_element_size(), 8);

            m_exec_conf->useContext();
            char *h_tmp;
            cudaHostAlloc(&h_tmp, m_corner_send_buf.getNumElements(), cudaHostAllocDefault);
            for (unsigned int i = 0; i < 8; ++i)
                memcpy(h_tmp+i*m_corner_send_buf.getPitch(), h_corner_send_buf+i*old_pitch,  old_pitch);
            cudaFreeHost(h_corner_send_buf);
            h_corner_send_buf = h_tmp;
            m_exec_conf->releaseContext();
            }

        if (m_edge_send_buf.getPitch() < m_max_send_ptls_edge*gpu_pdata_element_size())
            {
            unsigned int old_pitch = m_edge_send_buf.getPitch();
            m_edge_send_buf.resize(m_max_send_ptls_edge*gpu_pdata_element_size(), 12);

            m_exec_conf->useContext();
            char *h_tmp;
            cudaHostAlloc(&h_tmp, m_edge_send_buf.getNumElements(), cudaHostAllocDefault);
            for (unsigned int i = 0; i < 12; ++i)
                memcpy(h_tmp+i*m_edge_send_buf.getPitch(), h_edge_send_buf+i*old_pitch,  old_pitch);
            cudaFreeHost(h_edge_send_buf);
            h_edge_send_buf = h_tmp;
            m_exec_conf->releaseContext();
            } 

        if (m_face_send_buf.getPitch() < m_max_send_ptls_face*gpu_pdata_element_size())
            {
            unsigned int old_pitch = m_face_send_buf.getPitch();
            m_face_send_buf.resize(m_max_send_ptls_face*gpu_pdata_element_size(), 6);

            m_exec_conf->useContext();
            char *h_tmp;
            cudaHostAlloc(&h_tmp, m_face_send_buf.getNumElements(), cudaHostAllocDefault);
            for (unsigned int i = 0; i < 6; ++i)
                memcpy(h_tmp+i*m_face_send_buf.getPitch(), h_face_send_buf+i*old_pitch,  old_pitch);
            cudaFreeHost(h_face_send_buf);
            h_face_send_buf = h_tmp;
            m_exec_conf->releaseContext();
            } 

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

            ArrayHandle<unsigned int> d_n_send_ptls_corner(m_n_send_ptls_corner, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_n_send_ptls_edge(m_n_send_ptls_edge, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_n_send_ptls_face(m_n_send_ptls_face, access_location::device, access_mode::overwrite);

            // Stage particle data for sending, wrap particles
            m_exec_conf->useContext();
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
                                   d_n_send_ptls_corner.data,
                                   d_n_send_ptls_edge.data,
                                   d_n_send_ptls_face.data,
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
                                   m_condition.getDeviceFlags());

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                        CHECK_CUDA_ERROR();
            m_exec_conf->releaseContext();
            }

            {
            // read back numbers of sent particles
            ArrayHandleAsync<unsigned int> h_n_send_ptls_face(m_n_send_ptls_face, access_location::host, access_mode::read);
            ArrayHandleAsync<unsigned int> h_n_send_ptls_edge(m_n_send_ptls_edge, access_location::host, access_mode::read);
            ArrayHandleAsync<unsigned int> h_n_send_ptls_corner(m_n_send_ptls_corner, access_location::host, access_mode::read);

            m_exec_conf->useContext();
            cudaEventRecord(m_event, 0);
            cudaEventSynchronize(m_event);
            m_exec_conf->releaseContext();

            for (unsigned int i = 0; i < 6; ++i)
                n_send_ptls_face[i] = h_n_send_ptls_face.data[i];

            for (unsigned int i = 0; i < 12; ++i)
                n_send_ptls_edge[i] = h_n_send_ptls_edge.data[i];

            for (unsigned int i = 0; i < 8; ++i)
                n_send_ptls_corner[i] = h_n_send_ptls_corner.data[i];
            }


        condition = m_condition.readFlags();
        if (condition & 1)
            {
            // set new maximum size for face send buffers
            unsigned int new_size = 1;
            for (unsigned int i = 0; i < 6; ++i)
                if (n_send_ptls_face[i] > new_size) new_size = n_send_ptls_face[i];
            while (m_max_send_ptls_face < new_size)
                m_max_send_ptls_face = ceilf((float)m_max_send_ptls_face*m_resize_factor);
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
            // set new maximum size for corner send buffers
            unsigned int new_size = 1;
            for (unsigned int i = 0; i < 8; ++i)
                if (n_send_ptls_corner[i] > new_size) new_size = n_send_ptls_corner[i];
            while (m_max_send_ptls_corner < new_size)
                m_max_send_ptls_corner = ceilf((float)m_max_send_ptls_corner * m_resize_factor);
            }

        if (condition & 8)
            {
            m_exec_conf->msg->error() << "Invalid particle plan." << std::endl;
            throw std::runtime_error("Error during communication.");
            }
        }
    while (condition);


    // total up number of sent particles
    unsigned int n_remove_ptls = 0;
    for (unsigned int i = 0; i < 6; ++i)
        n_remove_ptls += n_send_ptls_face[i];
    for (unsigned int i = 0; i < 12; ++i)
        n_remove_ptls += n_send_ptls_edge[i];
    for (unsigned int i = 0; i < 8; ++i)
        n_remove_ptls += n_send_ptls_corner[i];

        {
        if (m_prof) m_prof->push("D->H");
        // copy send buffer data to host buffers only as needed
        ArrayHandle<char> d_face_send_buf(m_face_send_buf, access_location::device, access_mode::read);
        ArrayHandle<char> d_edge_send_buf(m_edge_send_buf, access_location::device, access_mode::read);
        ArrayHandle<char> d_corner_send_buf(m_corner_send_buf, access_location::device, access_mode::read);

        unsigned int fpitch = m_face_send_buf.getPitch();
        unsigned int epitch = m_edge_send_buf.getPitch();
        unsigned int cpitch = m_corner_send_buf.getPitch();

        m_exec_conf->useContext();

        for (unsigned int i = 0; i < 6; ++i)
            if (n_send_ptls_face[i])
                cudaMemcpyAsync(h_face_send_buf+i*fpitch, d_face_send_buf.data + i*fpitch,
                    n_send_ptls_face[i]*gpu_pdata_element_size(), cudaMemcpyDeviceToHost,0);

         for (unsigned int i = 0; i < 12; ++i)
            if (n_send_ptls_edge[i])
                cudaMemcpyAsync(h_edge_send_buf+i*epitch, d_edge_send_buf.data + i*epitch,
                    n_send_ptls_edge[i]*gpu_pdata_element_size(), cudaMemcpyDeviceToHost,0);

         for (unsigned int i = 0; i < 8; ++i)
            if (n_send_ptls_corner[i])
                cudaMemcpyAsync(h_corner_send_buf+i*cpitch, d_corner_send_buf.data + i*cpitch,
                    n_send_ptls_corner[i]*gpu_pdata_element_size(), cudaMemcpyDeviceToHost,0);

        cudaEventRecord(m_event, 0);
        cudaEventSynchronize(m_event);

        m_exec_conf->releaseContext();
        if (m_prof) m_prof->pop();
        }


    unsigned int n_tot_recv_ptls = 0;

    for (unsigned int dir=0; dir < 6; dir++)
        {

        if (! isCommunicating(dir) ) continue;

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
        communicateStepOne(dir,
                           n_send_ptls_corner,
                           n_send_ptls_edge,
                           n_send_ptls_face,
                           n_recv_ptls_face,
                           n_recv_ptls_edge,
                           &n_recv_ptls,
                           true);

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

            unsigned int old_pitch = m_edge_send_buf.getPitch();
            m_edge_send_buf.resize(m_max_send_ptls_edge*gpu_pdata_element_size(), 12);

            m_exec_conf->useContext();
            char *h_tmp;
            cudaHostAlloc(&h_tmp, m_edge_send_buf.getNumElements(), cudaHostAllocDefault);
            for (unsigned int i = 0; i < 12; ++i)
                memcpy(h_tmp+i*m_edge_send_buf.getPitch(), h_edge_send_buf+i*old_pitch,  old_pitch);
            cudaFreeHost(h_edge_send_buf);
            h_edge_send_buf = h_tmp;
            m_exec_conf->releaseContext();
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

            unsigned int old_pitch = m_face_send_buf.getPitch();
            m_face_send_buf.resize(m_max_send_ptls_face*gpu_pdata_element_size(), 6);

            m_exec_conf->useContext();
            char *h_tmp;
            cudaHostAlloc(&h_tmp, m_face_send_buf.getNumElements(), cudaHostAllocDefault);
            for (unsigned int i = 0; i < 6; ++i)
                memcpy(h_tmp+i*m_face_send_buf.getPitch(), h_face_send_buf+i*old_pitch,  old_pitch);
            cudaFreeHost(h_face_send_buf);
            h_face_send_buf = h_tmp;
            m_exec_conf->releaseContext();
            }

        if (m_recv_buf.getNumElements() < (n_tot_recv_ptls + n_recv_ptls)*gpu_pdata_element_size())
            {
            unsigned int new_size =1;
            while (new_size < n_tot_recv_ptls + n_recv_ptls) new_size = ceilf((float) new_size * m_resize_factor);

            unsigned int old_size = m_recv_buf.getNumElements();
            m_recv_buf.resize(new_size*gpu_pdata_element_size());

            m_exec_conf->useContext();
            char *h_tmp;
            cudaHostAlloc(&h_tmp, m_recv_buf.getNumElements(), cudaHostAllocDefault);
            memcpy(h_tmp, h_recv_buf, old_size);
            cudaFreeHost(h_recv_buf);
            h_recv_buf = h_tmp;
            m_exec_conf->releaseContext();
            }
          

        unsigned int cpitch = m_corner_send_buf.getPitch();
        unsigned int epitch = m_edge_send_buf.getPitch();
        unsigned int fpitch = m_face_send_buf.getPitch();

        communicateStepTwo(dir,
                           h_corner_send_buf,
                           h_edge_send_buf,
                           h_face_send_buf,
                           cpitch,
                           epitch,
                           fpitch,
                           h_recv_buf,
                           n_send_ptls_corner,
                           n_send_ptls_edge,
                           n_send_ptls_face,
                           m_recv_buf.getNumElements(),
                           n_tot_recv_ptls,
                           gpu_pdata_element_size(),
                           true);

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
        if (m_prof) m_prof->push("H->D");
        // copy back received particle data to device as necessary
        ArrayHandle<char> d_recv_buf(m_recv_buf, access_location::device, access_mode::overwrite);

        m_exec_conf->useContext();
        if (n_tot_recv_ptls)
            cudaMemcpyAsync(d_recv_buf.data,
                       h_recv_buf,
                       n_tot_recv_ptls*gpu_pdata_element_size(),
                       cudaMemcpyHostToDevice,0);

        m_exec_conf->releaseContext();
        if (m_prof) m_prof->pop();
        }

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

        m_exec_conf->useContext();
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
        m_exec_conf->releaseContext();
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

    if (m_prof)
        m_prof->push("GPU plan");

    if (bdata->getNumBonds())
        {
        // Send incomplete bond member to the nearest plane in all directions
        const GPUVector<uint2>& btable = bdata->getBondTable();
        ArrayHandle<uint2> d_btable(btable, access_location::device, access_mode::read);
        ArrayHandle<unsigned char> d_plan(m_plan, access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::read);

        m_exec_conf->useContext();
        gpu_mark_particles_in_incomplete_bonds(d_btable.data,
                                               d_plan.data,
                                               d_pos.data,
                                               d_rtag.data,
                                               m_pdata->getN(),
                                               bdata->getNumBonds(),
                                               m_pdata->getBox());

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_exec_conf->releaseContext();
        }


    /*
     * Mark non-bonded atoms for sending
     */
        {
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
        ArrayHandle<unsigned char> d_plan(m_plan, access_location::device, access_mode::readwrite);

        m_exec_conf->useContext();
        gpu_make_nonbonded_exchange_plan(d_plan.data,
                                         m_pdata->getN(),
                                         d_pos.data,
                                         m_pdata->getBox(),
                                         m_r_ghost);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_exec_conf->releaseContext();
        }

    if (m_prof) m_prof->pop();

    // initialization of buffers
    if (! m_buffers_allocated)
        allocateBuffers();

    unsigned int n_copy_ghosts_corner[8];
    unsigned int n_copy_ghosts_edge[12];
    unsigned int n_copy_ghosts_face[6];

    unsigned int condition;
    do {
        // resize buffers if necessary 
        if (m_corner_ghosts_buf.getPitch() < m_max_copy_ghosts_corner*gpu_ghost_element_size())
            {
            unsigned int old_pitch = m_corner_ghosts_buf.getPitch();
            m_corner_ghosts_buf.resize(m_max_copy_ghosts_corner*gpu_ghost_element_size(), 8);

            m_exec_conf->useContext();
            char *h_tmp;
            cudaHostAlloc(&h_tmp, m_corner_ghosts_buf.getNumElements(), cudaHostAllocDefault);
            for (unsigned int i = 0; i < 8; ++i)
                memcpy(h_tmp+i*m_corner_ghosts_buf.getPitch(), h_corner_ghosts_buf+i*old_pitch,  old_pitch);
            cudaFreeHost(h_corner_ghosts_buf);
            h_corner_ghosts_buf = h_tmp;
            m_exec_conf->releaseContext();
            }

        if (m_edge_ghosts_buf.getPitch() < m_max_copy_ghosts_edge*gpu_ghost_element_size())
            {
            unsigned int old_pitch = m_corner_ghosts_buf.getPitch();
            m_edge_ghosts_buf.resize(m_max_copy_ghosts_edge*gpu_ghost_element_size(), 12);

            m_exec_conf->useContext();
            char *h_tmp;
            cudaHostAlloc(&h_tmp, m_edge_ghosts_buf.getNumElements(), cudaHostAllocDefault);
            for (unsigned int i = 0; i < 12; ++i)
                memcpy(h_tmp+i*m_edge_ghosts_buf.getPitch(), h_edge_ghosts_buf+i*old_pitch,  old_pitch);
            cudaFreeHost(h_edge_ghosts_buf);
            h_edge_ghosts_buf = h_tmp;
            m_exec_conf->releaseContext();
            } 

        if (m_face_ghosts_buf.getPitch() < m_max_copy_ghosts_face*gpu_ghost_element_size())
            {
            unsigned int old_pitch = m_face_ghosts_buf.getPitch();

            m_face_ghosts_buf.resize(m_max_copy_ghosts_face*gpu_ghost_element_size(), 6);

            m_exec_conf->useContext();
            char *h_tmp;
            cudaHostAlloc(&h_tmp, m_face_ghosts_buf.getNumElements(), cudaHostAllocDefault);
            for (unsigned int i = 0; i < 6; ++i)
                memcpy(h_tmp+i*m_face_ghosts_buf.getPitch(), h_face_ghosts_buf+i*old_pitch,  old_pitch);
            cudaFreeHost(h_face_ghosts_buf);
            h_face_ghosts_buf = h_tmp;
            m_exec_conf->releaseContext();
            }
            

        if (m_corner_update_buf.getPitch() < m_max_copy_ghosts_corner*gpu_update_element_size())
            m_corner_update_buf.resize(m_max_copy_ghosts_corner*gpu_update_element_size(), 8);
        if (m_edge_update_buf.getPitch() < m_max_copy_ghosts_edge*gpu_update_element_size())
            m_edge_update_buf.resize(m_max_copy_ghosts_edge*gpu_update_element_size(), 12);
        if (m_face_update_buf.getPitch() < m_max_copy_ghosts_face*gpu_update_element_size())
            m_face_update_buf.resize(m_max_copy_ghosts_face*gpu_update_element_size(), 6);

        if (m_ghost_idx_face.getPitch() < m_max_copy_ghosts_face)
            m_ghost_idx_face.resize(m_max_copy_ghosts_face, 6);
        if (m_ghost_idx_edge.getPitch() < m_max_copy_ghosts_edge)
            m_ghost_idx_edge.resize(m_max_copy_ghosts_edge, 12);
        if (m_ghost_idx_corner.getPitch() < m_max_copy_ghosts_corner)
            m_ghost_idx_corner.resize(m_max_copy_ghosts_corner, 8);

        m_condition.resetFlags(0);

            {
            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
            ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::read);
            ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);
            ArrayHandle<unsigned char> d_plan(m_plan, access_location::device, access_mode::read);

            ArrayHandle<char> d_corner_ghosts_buf(m_corner_ghosts_buf, access_location::device, access_mode::overwrite);
            ArrayHandle<char> d_edge_ghosts_buf(m_edge_ghosts_buf, access_location::device, access_mode::overwrite);
            ArrayHandle<char> d_face_ghosts_buf(m_face_ghosts_buf, access_location::device, access_mode::overwrite);

            ArrayHandle<unsigned int> d_ghost_idx_face(m_ghost_idx_face, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_ghost_idx_edge(m_ghost_idx_edge, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_ghost_idx_corner(m_ghost_idx_corner, access_location::device, access_mode::overwrite);

            ArrayHandle<unsigned int> d_n_local_ghosts_face(m_n_local_ghosts_face, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_n_local_ghosts_edge(m_n_local_ghosts_edge, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_n_local_ghosts_corner(m_n_local_ghosts_corner, access_location::device, access_mode::overwrite);

            if (m_prof) m_prof->push("GPU pack");
            m_exec_conf->useContext();
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
                                m_condition.getDeviceFlags());

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();

            m_exec_conf->releaseContext();
            if (m_prof) m_prof->pop();
            }

            {
            ArrayHandleAsync<unsigned int> h_n_local_ghosts_face(m_n_local_ghosts_face, access_location::host, access_mode::read);
            ArrayHandleAsync<unsigned int> h_n_local_ghosts_edge(m_n_local_ghosts_edge, access_location::host, access_mode::read);
            ArrayHandleAsync<unsigned int> h_n_local_ghosts_corner(m_n_local_ghosts_corner, access_location::host, access_mode::read);

            m_exec_conf->useContext();
            cudaEventRecord(m_event, 0);
            cudaEventSynchronize(m_event);
            m_exec_conf->releaseContext();

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
                m_max_copy_ghosts_face = ceilf((float)m_max_copy_ghosts_face*m_resize_factor);
            }
        if (condition & 2)
            {
            // overflow of edge copy buf
            unsigned int new_size = 1;
            for (unsigned int i = 0; i < 12; ++i)
                if (n_copy_ghosts_edge[i] > new_size) new_size = n_copy_ghosts_edge[i];
            while (m_max_copy_ghosts_edge < new_size)
                m_max_copy_ghosts_edge = ceilf((float)m_max_copy_ghosts_edge*m_resize_factor);
            }
        if (condition & 4)
            {
            // overflow of corner copy buf
            unsigned int new_size = 1;
            for (unsigned int i = 0; i < 8; ++i)
                if (n_copy_ghosts_corner[i] > new_size) new_size = n_copy_ghosts_corner[i];
            while (m_max_copy_ghosts_corner < new_size)
                m_max_copy_ghosts_corner = ceilf((float)m_max_copy_ghosts_corner * m_resize_factor);

            }

        if (condition & 8)
            {
            m_exec_conf->msg->error() << "Invalid particle plan." << std::endl;
            throw std::runtime_error("Error during communication.");
            }
        } while (condition);


    // store number of local particles we are sending as ghosts, for later counting purposes
    unsigned int n_forward_ghosts_face[6];
    unsigned int n_forward_ghosts_edge[12];

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

        {
        if (m_prof) m_prof->push("D->H");
        // copy send buffer data to host buffers only as needed
        ArrayHandle<char> d_face_ghosts_buf(m_face_ghosts_buf, access_location::device, access_mode::read);
        ArrayHandle<char> d_edge_ghosts_buf(m_edge_ghosts_buf, access_location::device, access_mode::read);
        ArrayHandle<char> d_corner_ghosts_buf(m_corner_ghosts_buf, access_location::device, access_mode::read);

        unsigned int fpitch = m_face_ghosts_buf.getPitch();
        unsigned int epitch = m_edge_ghosts_buf.getPitch();
        unsigned int cpitch = m_corner_ghosts_buf.getPitch();

        m_exec_conf->useContext();
 
        for (unsigned int i = 0; i < 6; ++i)
            if (n_copy_ghosts_face[i])
                cudaMemcpyAsync(h_face_ghosts_buf+i*fpitch, d_face_ghosts_buf.data + i*fpitch,
                    n_copy_ghosts_face[i]*gpu_ghost_element_size(), cudaMemcpyDeviceToHost,0);

         for (unsigned int i = 0; i < 12; ++i)
            if (n_copy_ghosts_edge[i])
                cudaMemcpyAsync(h_edge_ghosts_buf+i*epitch, d_edge_ghosts_buf.data + i*epitch,
                    n_copy_ghosts_edge[i]*gpu_ghost_element_size(), cudaMemcpyDeviceToHost,0);

         for (unsigned int i = 0; i < 8; ++i)
            if (n_copy_ghosts_corner[i])
                cudaMemcpyAsync(h_corner_ghosts_buf+i*cpitch, d_corner_ghosts_buf.data + i*cpitch,
                    n_copy_ghosts_corner[i]*gpu_ghost_element_size(), cudaMemcpyDeviceToHost,0);

        cudaEventRecord(m_event, 0);
        cudaEventSynchronize(m_event);

        m_exec_conf->releaseContext();
        if (m_prof) m_prof->pop();
        }

    // Number of ghosts we received that are not forwarded to other boxes
    unsigned int n_tot_recv_ghosts_local = 0;

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

            if (m_prof) m_prof->push("MPI");

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
            
            if (m_prof) m_prof->pop();

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
                while (new_size < max_n_recv_edge + max_n_copy_edge) new_size = ceilf((float)new_size* m_resize_factor);
                m_max_copy_ghosts_edge = new_size;

                unsigned int old_pitch = m_edge_ghosts_buf.getPitch();

                m_edge_ghosts_buf.resize(m_max_copy_ghosts_edge*gpu_ghost_element_size(), 12);
                m_edge_update_buf.resize(m_max_copy_ghosts_edge*gpu_update_element_size(), 12);


                char *h_tmp;
                m_exec_conf->useContext();
                cudaHostAlloc(&h_tmp, m_edge_ghosts_buf.getNumElements(), cudaHostAllocDefault);
                for (unsigned int i = 0; i < 12; ++i)
                    memcpy(h_tmp+i*m_edge_ghosts_buf.getPitch(), h_edge_ghosts_buf+i*old_pitch,  old_pitch);
                cudaFreeHost(h_edge_ghosts_buf);
                h_edge_ghosts_buf = h_tmp;
                m_exec_conf->releaseContext();
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
                while (new_size < max_n_recv_face + max_n_copy_face) new_size = ceilf((float) new_size * m_resize_factor);
                m_max_copy_ghosts_face = new_size;

                unsigned int old_pitch = m_face_ghosts_buf.getPitch();

                m_face_ghosts_buf.resize(m_max_copy_ghosts_face*gpu_ghost_element_size(), 6);
                m_face_update_buf.resize(m_max_copy_ghosts_face*gpu_update_element_size(), 6);

                char *h_tmp;
                m_exec_conf->useContext();
                cudaHostAlloc(&h_tmp, m_face_ghosts_buf.getNumElements(), cudaHostAllocDefault);
                for (unsigned int i = 0; i < 6; ++i)
                    memcpy(h_tmp+i*m_face_ghosts_buf.getPitch(), h_face_ghosts_buf+i*old_pitch,  old_pitch);
                cudaFreeHost(h_face_ghosts_buf);
                h_face_ghosts_buf = h_tmp;
                m_exec_conf->releaseContext();
                }

            if (m_ghosts_recv_buf.getNumElements() < (n_tot_recv_ghosts_local + h_n_recv_ghosts_local.data[dir])*gpu_ghost_element_size())
                {
                unsigned int old_size = m_ghosts_recv_buf.getNumElements();
                unsigned int new_size =1;
                while (new_size < n_tot_recv_ghosts_local + h_n_recv_ghosts_local.data[dir])
                    new_size = ceilf((float) new_size * m_resize_factor);
                m_ghosts_recv_buf.resize(new_size*gpu_ghost_element_size());

                char *h_tmp;
                m_exec_conf->useContext();
                cudaHostAlloc(&h_tmp, new_size*gpu_ghost_element_size(), cudaHostAllocDefault);
                memcpy(h_tmp, h_ghosts_recv_buf, old_size);
                cudaFreeHost(h_ghosts_recv_buf);
                h_ghosts_recv_buf = h_tmp;
                m_exec_conf->releaseContext();
                }

            if (m_update_recv_buf.getNumElements() < (n_tot_recv_ghosts_local + h_n_recv_ghosts_local.data[dir])*gpu_update_element_size())
                {
                unsigned int new_size =1;
                while (new_size < n_tot_recv_ghosts_local + h_n_recv_ghosts_local.data[dir])
                    new_size = ceilf((float) new_size * m_resize_factor);
                m_update_recv_buf.resize(new_size*gpu_update_element_size());
                }

            unsigned int cpitch = m_corner_ghosts_buf.getPitch();
            unsigned int epitch = m_edge_ghosts_buf.getPitch();
            unsigned int fpitch = m_face_ghosts_buf.getPitch();
     
            if (m_prof) m_prof->push("MPI");

            communicateStepTwo(dir,
                               h_corner_ghosts_buf,
                               h_edge_ghosts_buf,
                               h_face_ghosts_buf,
                               cpitch,
                               epitch,
                               fpitch,
                               h_ghosts_recv_buf,
                               n_copy_ghosts_corner,
                               n_copy_ghosts_edge,
                               n_copy_ghosts_face,
                               m_ghosts_recv_buf.getNumElements(),
                               n_tot_recv_ghosts_local, 
                               gpu_ghost_element_size(),
                               false);

            if (m_prof) m_prof->pop();

            // update buffer sizes
            for (unsigned int i = 0; i < 12; ++i)
                n_copy_ghosts_edge[i] += h_n_recv_ghosts_edge.data[12*dir+i];

            for (unsigned int i = 0; i < 6; ++i)
                n_copy_ghosts_face[i] += h_n_recv_ghosts_face.data[6*dir+i];

            n_tot_recv_ghosts_local += h_n_recv_ghosts_local.data[dir];
            } // end communication loop
        }

        {
        // calculate number of forwarded particles for every face and edge
        ArrayHandle<unsigned int> h_n_local_ghosts_face(m_n_local_ghosts_face, access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_n_local_ghosts_edge(m_n_local_ghosts_edge, access_location::host, access_mode::read);

        for (unsigned int i = 0; i < 6; ++i)
            n_forward_ghosts_face[i] = n_copy_ghosts_face[i] - h_n_local_ghosts_face.data[i];

        for (unsigned int i = 0; i < 12; ++i)
            n_forward_ghosts_edge[i] = n_copy_ghosts_edge[i] - h_n_local_ghosts_edge.data[i];
        }


    // total up number of received ghosts
    unsigned int n_tot_recv_ghosts = n_tot_recv_ghosts_local;
    for (unsigned int i = 0; i < 6; ++i)
        n_tot_recv_ghosts += n_forward_ghosts_face[i];
    for (unsigned int i = 0; i < 12; ++i)
        n_tot_recv_ghosts += n_forward_ghosts_edge[i];

    // resize plan array if necessary
    if (m_ghost_plan.getNumElements() < n_tot_recv_ghosts)
        {
        unsigned int new_size = m_ghost_plan.getNumElements();
        while (new_size < n_tot_recv_ghosts) new_size = ceilf((float)new_size*m_resize_factor);
        m_ghost_plan.resize(new_size);
        }

    // update number of ghost particles
    m_pdata->addGhostParticles(n_tot_recv_ghosts);


        {
        if (m_prof) m_prof->push("H->D");
        // copy back received ghost data to device as necessary
        ArrayHandle<char> d_face_ghosts_buf(m_face_ghosts_buf, access_location::device, access_mode::readwrite);
        ArrayHandle<char> d_edge_ghosts_buf(m_edge_ghosts_buf, access_location::device, access_mode::readwrite);
        ArrayHandle<char> d_ghosts_recv_buf(m_ghosts_recv_buf, access_location::device, access_mode::overwrite);

        ArrayHandle<unsigned int> h_n_local_ghosts_face(m_n_local_ghosts_face, access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_n_local_ghosts_edge(m_n_local_ghosts_edge, access_location::host, access_mode::read);

        unsigned int fpitch = m_face_ghosts_buf.getPitch();
        unsigned int epitch = m_edge_ghosts_buf.getPitch();

        m_exec_conf->useContext();

        for (unsigned int i = 0; i < 6; ++i)
            if (n_forward_ghosts_face[i])
                cudaMemcpyAsync(d_face_ghosts_buf.data + h_n_local_ghosts_face.data[i]*gpu_ghost_element_size() + i * fpitch,
                           h_face_ghosts_buf + h_n_local_ghosts_face.data[i]*gpu_ghost_element_size() + i*fpitch,
                           n_forward_ghosts_face[i]*gpu_ghost_element_size(),
                           cudaMemcpyHostToDevice,0);

        for (unsigned int i = 0; i < 12; ++i)
            if (n_forward_ghosts_edge[i])
                cudaMemcpyAsync(d_edge_ghosts_buf.data + h_n_local_ghosts_edge.data[i]*gpu_ghost_element_size() + i * epitch,
                           h_edge_ghosts_buf + h_n_local_ghosts_edge.data[i]*gpu_ghost_element_size() + i * epitch,
                           n_forward_ghosts_edge[i]*gpu_ghost_element_size(),
                           cudaMemcpyHostToDevice,0);

        if (n_tot_recv_ghosts_local)
            cudaMemcpyAsync(d_ghosts_recv_buf.data,
                       h_ghosts_recv_buf,
                       n_tot_recv_ghosts_local*gpu_ghost_element_size(),
                       cudaMemcpyHostToDevice,0);

        m_exec_conf->releaseContext();
        if (m_prof) m_prof->pop();
        }

        {
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::readwrite);
        ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::readwrite);
        ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::readwrite);

        ArrayHandle<char> d_face_ghosts(m_face_ghosts_buf, access_location::device, access_mode::read);
        ArrayHandle<char> d_edge_ghosts(m_edge_ghosts_buf, access_location::device, access_mode::read);
        ArrayHandle<char> d_recv_ghosts(m_ghosts_recv_buf, access_location::device, access_mode::read);

        ArrayHandle<unsigned int> d_ghost_plan(m_ghost_plan, access_location::device, access_mode::overwrite);

        ArrayHandle<unsigned int> d_n_local_ghosts_face(m_n_local_ghosts_face, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_n_local_ghosts_edge(m_n_local_ghosts_edge, access_location::device, access_mode::read);

        ArrayHandleAsync<unsigned int> d_n_recv_ghosts_face(m_n_recv_ghosts_face, access_location::device, access_mode::read);
        ArrayHandleAsync<unsigned int> d_n_recv_ghosts_edge(m_n_recv_ghosts_edge, access_location::device, access_mode::read);
        ArrayHandleAsync<unsigned int> d_n_recv_ghosts_local(m_n_recv_ghosts_local, access_location::device, access_mode::read);

        if (m_prof) m_prof->push("GPU unpack");
        m_exec_conf->useContext();
        gpu_exchange_ghosts_unpack(m_pdata->getN(),
                                     n_tot_recv_ghosts,
                                     d_n_local_ghosts_face.data,
                                     d_n_local_ghosts_edge.data,
                                     n_tot_recv_ghosts_local,
                                     d_n_recv_ghosts_local.data,
                                     d_n_recv_ghosts_face.data,
                                     d_n_recv_ghosts_edge.data,
                                     d_face_ghosts.data,
                                     m_face_ghosts_buf.getPitch(),
                                     d_edge_ghosts.data,
                                     m_edge_ghosts_buf.getPitch(),
                                     d_recv_ghosts.data,
                                     d_pos.data,
                                     d_charge.data,
                                     d_diameter.data,
                                     d_tag.data,
                                     d_rtag.data,
                                     d_ghost_plan.data,
                                     m_pdata->getGlobalBox());

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_exec_conf->releaseContext();
        if (m_prof) m_prof->pop();

        }

    // we have updated ghost particles, so inform ParticleData about this
    m_pdata->notifyGhostParticleNumberChange();

    if (m_prof)
        m_prof->pop();
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
    unsigned int n_remote_recv_ptls_edge[12];
    unsigned int n_remote_recv_ptls_face[6];
    unsigned int n_remote_recv_ptls_local;

    for (unsigned int i = 0; i < 12; ++i)
        n_remote_recv_ptls_edge[i] = 0;
    for (unsigned int i = 0; i < 6; ++i)
        n_remote_recv_ptls_face[i] = 0;
    n_remote_recv_ptls_local = 0;

    unsigned int nptl;

    // calculate total message sizes
    for (unsigned int corner_i = 0; corner_i < 8; ++corner_i)
        {
        bool sent = false;
        unsigned int plan = corner_plan_lookup[corner_i];

        // only send corner particle through face if face touches corner
        if (!((face_plan_lookup[cur_face] & plan) == face_plan_lookup[cur_face])) continue;

        nptl = n_send_ptls_corner[corner_i];

        for (unsigned int edge_j = 0; edge_j < 12; ++edge_j)
            if ((edge_plan_lookup[edge_j] & plan) == edge_plan_lookup[edge_j])
                {
                // if this edge buffer is or has already been emptied in this or previous communication steps, don't add to it
                bool active = true;
                for (unsigned int face_k = 0; face_k < 6; ++face_k)
                    if (face_k <= cur_face && (edge_plan_lookup[edge_j] & face_plan_lookup[face_k])) active = false;
                if (! active) continue;

                // send a corner particle to an edge send buffer in the neighboring box
                n_remote_recv_ptls_edge[edge_j] += nptl;

                sent = true;
                break;
                }
            
        // If we are only sending to one destination box, corner ptls can only be sent by the neighboring box as an edge particle
        if (unique_destination || sent)
            continue;

        // do not place particle in a buffer where it would be sent back to ourselves
        unsigned int next_face = cur_face + ((cur_face % 2) ? 1 : 2);

        for (unsigned int face_j = next_face; face_j < 6; ++face_j)
            if ((face_plan_lookup[face_j] & plan) == face_plan_lookup[face_j])
                {
                // send a corner particle to a face send buffer in the neighboring box
                n_remote_recv_ptls_face[face_j] += nptl;
                sent = true;
                break;
                }

        if (sent) continue;

        if (plan & face_plan_lookup[cur_face])
            n_remote_recv_ptls_local += nptl;
        }
            
    for (unsigned int edge_i = 0; edge_i < 12; ++edge_i)
        {
        bool sent = false;
        unsigned int plan = edge_plan_lookup[edge_i];

        // only send edge particle through face if face touches edge
        if (! ((face_plan_lookup[cur_face] & plan) == face_plan_lookup[cur_face])) continue;

        nptl = n_send_ptls_edge[edge_i];

        // do not place particle in a buffer where it would be sent back to ourselves
        unsigned int next_face = cur_face + ((cur_face % 2) ? 1 : 2);

        for (unsigned int face_j = next_face; face_j < 6; ++face_j)
            if ((face_plan_lookup[face_j] & plan) == face_plan_lookup[face_j])
                {
                // send a corner particle to a face send buffer in the neighboring box
                n_remote_recv_ptls_face[face_j] += nptl;

                sent = true;
                break;
                }

        if (unique_destination || sent) continue;

        if (plan & face_plan_lookup[cur_face])
            n_remote_recv_ptls_local += nptl;
        } 

    for (unsigned int face_i = 0; face_i < 6; ++face_i)
        {
        unsigned int plan = face_plan_lookup[face_i];

        // only send through face if this is the current sending direction
        if (! ((face_plan_lookup[cur_face] & plan) == face_plan_lookup[cur_face])) continue;

        n_remote_recv_ptls_local += n_send_ptls_face[face_i];
        }

 
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

    MPI_Isend(n_remote_recv_ptls_edge, sizeof(unsigned int)*12, MPI_BYTE, send_neighbor, 0, m_mpi_comm, & reqs[0]);
    MPI_Isend(n_remote_recv_ptls_face, sizeof(unsigned int)*6, MPI_BYTE, send_neighbor, 1, m_mpi_comm, &reqs[1]);
    MPI_Isend(&n_remote_recv_ptls_local, sizeof(unsigned int), MPI_BYTE, send_neighbor, 2, m_mpi_comm, &reqs[2]);

    MPI_Irecv(n_recv_ptls_edge, 12*sizeof(unsigned int), MPI_BYTE, recv_neighbor, 0, m_mpi_comm, &reqs[3]);
    MPI_Irecv(n_recv_ptls_face, 6*sizeof(unsigned int), MPI_BYTE, recv_neighbor, 1, m_mpi_comm, & reqs[4]);
    MPI_Irecv(n_recv_ptls_local, sizeof(unsigned int), MPI_BYTE, recv_neighbor, 2, m_mpi_comm, & reqs[5]);

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

    unsigned int foffset[6];
    unsigned int eoffset[12];
    unsigned int loffset;

    for (unsigned int i = 0; i < 12; ++i)
        eoffset[i] = 0;

    for (unsigned int i = 0; i < 6; ++i)
        foffset[i] = 0;

    loffset = 0;

    unsigned int nptl;
    void *data;

    for (unsigned int corner_i = 0; corner_i < 8; ++corner_i)
        {
        bool sent = false;
        unsigned int plan = corner_plan_lookup[corner_i];

        // only send corner particle through face if face touches corner
        if (! ((face_plan_lookup[cur_face] & plan) == face_plan_lookup[cur_face])) continue;

        nptl = n_send_ptls_corner[corner_i];
        data = corner_send_buf+corner_i*cpitch;

        for (unsigned int edge_j = 0; edge_j < 12; ++edge_j)
            if ((edge_plan_lookup[edge_j] & plan) == edge_plan_lookup[edge_j])
                {
                // if this edge buffer is or has already been emptied in this or previous communication steps, don't add to it
                bool active = true;
                for (unsigned int face_k = 0; face_k < 6; ++face_k)
                    if (face_k <= cur_face && (edge_plan_lookup[edge_j] & face_plan_lookup[face_k])) active = false;
                if (! active) continue;

                // send a corner particle to an edge send buffer in the neighboring box
                MPI_Put(data,
                        nptl*element_size,
                        MPI_BYTE,
                        send_neighbor,
                        edge_recv_buf_remote[edge_j]+eoffset[edge_j]*element_size,
                        nptl*element_size,
                        MPI_BYTE,
                        m_win_edge[edge_j]);
                eoffset[edge_j] += nptl;

                sent = true;
                break;
                }
            
        // If we are only sending to one destination box, corner ptls can only be sent by the neighboring box as an edge particle
        if (unique_destination || sent)
            continue;

        // do not place particle in a buffer where it would be sent back to ourselves
        unsigned int next_face = cur_face + ((cur_face % 2) ? 1 : 2);

        for (unsigned int face_j = next_face; face_j < 6; ++face_j)
            if ((face_plan_lookup[face_j] & plan) == face_plan_lookup[face_j])
                {
                // send a corner particle to a face send buffer in the neighboring box
                MPI_Put(data,
                        nptl*element_size,
                        MPI_BYTE,
                        send_neighbor,
                        face_recv_buf_remote[face_j]+foffset[face_j]*element_size,
                        nptl*element_size,
                        MPI_BYTE,
                        m_win_face[face_j]);
                foffset[face_j] += nptl;
                sent = true;
                break;
                }

        if (sent) continue;

        if (plan & face_plan_lookup[cur_face])
            {
            // send a corner particle directly to the neighboring bo
            MPI_Put(data,
                    nptl*element_size,
                    MPI_BYTE,
                    send_neighbor,
                    local_recv_buf_remote+loffset*element_size,
                    nptl*element_size,
                    MPI_BYTE,
                    m_win_local);
            loffset += nptl;
            }
        }
            
    for (unsigned int edge_i = 0; edge_i < 12; ++edge_i)
        {
        bool sent = false;
        unsigned int plan = edge_plan_lookup[edge_i];

        // only send edge particle through face if face touches edge
        if (! ((face_plan_lookup[cur_face] & plan) == face_plan_lookup[cur_face])) continue;

        nptl = n_send_ptls_edge[edge_i];
        data = edge_send_buf+edge_i*epitch;
 
        // do not place particle in a buffer where it would be sent back to ourselves
        unsigned int next_face = cur_face + ((cur_face % 2) ? 1 : 2);

        for (unsigned int face_j = next_face; face_j < 6; ++face_j)
            if ((face_plan_lookup[face_j] & plan) == face_plan_lookup[face_j])
                {
                // send a corner particle to a face send buffer in the neighboring box
                MPI_Put(data,
                        nptl*element_size,
                        MPI_BYTE,
                        send_neighbor,
                        face_recv_buf_remote[face_j]+foffset[face_j]*element_size,
                        nptl*element_size,
                        MPI_BYTE,
                        m_win_face[face_j]);
                foffset[face_j] += nptl;

                sent = true;
                break;
                }

        if (unique_destination || sent) continue;

        if (plan & face_plan_lookup[cur_face])
            {
            // send directly to neighboring box
            MPI_Put(data,
                    nptl*element_size,
                    MPI_BYTE,
                    send_neighbor,
                    local_recv_buf_remote+loffset*element_size,
                    nptl*element_size,
                    MPI_BYTE,
                    m_win_local);
            loffset += nptl;
            }
        } 

    for (unsigned int face_i = 0; face_i < 6; ++face_i)
        {
        unsigned int plan = face_plan_lookup[face_i];

        // only send through face if this is the current sending direction
        if (! ((face_plan_lookup[cur_face] & plan) == face_plan_lookup[cur_face])) continue;

        nptl = n_send_ptls_face[face_i];
        data = face_send_buf+face_i*fpitch;

        MPI_Put(data,
                nptl*element_size,
                MPI_BYTE,
                send_neighbor,
                local_recv_buf_remote+loffset*element_size,
                nptl*element_size,
                MPI_BYTE,
                m_win_local);
        loffset += nptl;
        }

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
