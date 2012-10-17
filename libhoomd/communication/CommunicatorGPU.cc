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
ghost_gpu_thread::ghost_gpu_thread(boost::shared_ptr<const ExecutionConfiguration> exec_conf,
                                   boost::shared_ptr<CommunicatorGPU> communicator)
    : m_exec_conf(exec_conf),
      m_communicator(communicator)
    { }

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
    m_exec_conf->useContext();
    gpu_update_ghosts_pack(params.n_tot_recv_ghosts,
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
                        params.n_copy_ghosts_corner,
                        params.n_copy_ghosts_edge,
                        params.n_copy_ghosts_face,
                        params.is_at_boundary,
                        params.global_box,
                        m_exec_conf->getThreadStream(m_thread_id));

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    m_exec_conf->releaseContext();

    unsigned int n_tot_recv_ghosts_local = 0;

    for (unsigned int face = 0; face < 6; ++face)
        {

        if (! params.is_communicating[face]) continue;

        m_communicator->communicateStepTwo(face,
                                           params.corner_update_buf_handle,
                                           params.edge_update_buf_handle,
                                           params.face_update_buf_handle,
                                           params.corner_update_buf_pitch,
                                           params.edge_update_buf_pitch,
                                           params.face_update_buf_pitch,
                                           params.update_recv_buf_handle,
                                           params.n_copy_ghosts_corner,
                                           params.n_copy_ghosts_edge,
                                           params.n_copy_ghosts_face,
                                           params.n_recv_ghosts_edge[face],
                                           params.n_recv_ghosts_face[face],
                                           params.n_recv_ghosts_local[face],
                                           n_tot_recv_ghosts_local,
                                           gpu_update_element_size());

        n_tot_recv_ghosts_local += params.n_recv_ghosts_local[face];
        } // end dir loop

    unsigned int n_forward_ghosts_face[6];
    unsigned int n_forward_ghosts_edge[12];
    for (unsigned int i = 0; i < 6; ++i)
        n_forward_ghosts_face[i] = params.n_copy_ghosts_face[i] - params.n_local_ghosts_face[i];

    for (unsigned int i = 0; i < 12; ++i)
        n_forward_ghosts_edge[i] = params.n_copy_ghosts_edge[i] - params.n_local_ghosts_edge[i];


    // unpack particles
    m_exec_conf->useContext();
    gpu_update_ghosts_unpack(params.N,
                             params.n_tot_recv_ghosts,
                             params.n_local_ghosts_face,
                             params.n_local_ghosts_edge,
                             n_forward_ghosts_face,
                             n_forward_ghosts_edge,
                             n_tot_recv_ghosts_local,
                             params.face_update_buf_handle,
                             params.face_update_buf_pitch,
                             params.edge_update_buf_handle,
                             params.edge_update_buf_pitch,
                             params.update_recv_buf_handle,
                             params.pos_handle,
                             m_exec_conf->getThreadStream(m_thread_id));

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_exec_conf->releaseContext();
 
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
    // allocate temporary GPU buffers
    m_exec_conf->useContext();
    gpu_allocate_tmp_storage();
    m_exec_conf->releaseContext();

    GPUFlags<unsigned int> condition(m_exec_conf);
    m_condition.swap(condition);

    m_n_recv_ghosts_face.resize(6);
    m_n_recv_ghosts_edge.resize(6);
    m_n_recv_ghosts_local.resize(6);
    for (unsigned int i = 0; i < 6; ++i)
        {
        m_n_recv_ghosts_face[i] = new unsigned int[6];
        m_n_recv_ghosts_edge[i] = new unsigned int[12];
        }
    }

//! Destructor
CommunicatorGPU::~CommunicatorGPU()
    {
    m_exec_conf->msg->notice(5) << "Destroying CommunicatorGPU";

    for (unsigned int i = 0; i < 6; ++i)
        {
        delete m_n_recv_ghosts_face[i];
        delete m_n_recv_ghosts_edge[i];
        }
 
    m_exec_conf->useContext();

    gpu_deallocate_tmp_storage();

    // finish worker thread
    m_worker_thread.interrupt();
    m_worker_thread.join();

    m_exec_conf->releaseContext();
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
                                                         boost::shared_ptr<CommunicatorGPU>(this)),
                                        boost::ref(m_work_queue), boost::ref(m_barrier));
        m_thread_created = true;
        }

    bool is_communicating[6];
    for (unsigned int i = 0; i < 6; ++i)
        {
        is_communicating[i] = isCommunicating(i);
        }


    // fill thread parameters
    ghost_gpu_thread_params params(
        m_ghost_idx_face.acquire(access_location::device, access_mode::read),
        m_ghost_idx_face.getPitch(),
        m_ghost_idx_edge.acquire(access_location::device, access_mode::read),
        m_ghost_idx_edge.getPitch(),
        m_ghost_idx_corner.acquire(access_location::device, access_mode::read),
        m_ghost_idx_corner.getPitch(),
        m_corner_update_buf.acquire(access_location::device, access_mode::readwrite),
        m_corner_update_buf.getPitch(),
        m_edge_update_buf.acquire(access_location::device, access_mode::readwrite),
        m_edge_update_buf.getPitch(),
        m_face_update_buf.acquire(access_location::device, access_mode::readwrite),
        m_face_update_buf.getPitch(),
        m_update_recv_buf.acquire(access_location::device, access_mode::overwrite),
        is_communicating,
        m_is_at_boundary,
        m_pdata->getN(), 
        m_n_recv_ghosts_local,
        m_n_recv_ghosts_face,
        m_n_recv_ghosts_edge,
        m_n_copy_ghosts_corner,
        m_n_copy_ghosts_edge,
        m_n_copy_ghosts_face,
        m_n_tot_recv_ghosts,
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

    GPUArray<char> ghost_recv_buf(gpu_ghost_element_size()*m_max_copy_ghosts_face, m_exec_conf);
    m_ghosts_recv_buf.swap(ghost_recv_buf);

    GPUArray<char> update_recv_buf(gpu_update_element_size()*m_max_copy_ghosts_face, m_exec_conf);
    m_update_recv_buf.swap(update_recv_buf);

    // reallocate ghost index lists
    GPUArray<unsigned int> ghost_idx_face(m_max_copy_ghosts_face, 6, m_exec_conf);
    m_ghost_idx_face.swap(ghost_idx_face);

    GPUArray<unsigned int> ghost_idx_edge(m_max_copy_ghosts_edge, 12, m_exec_conf);
    m_ghost_idx_edge.swap(ghost_idx_edge);
    
    GPUArray<unsigned int> ghost_idx_corner(m_max_copy_ghosts_corner, 8, m_exec_conf);
    m_ghost_idx_corner.swap(ghost_idx_corner);

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
            m_exec_conf->releaseContext();
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

        if (condition & 8)
            {
            m_exec_conf->msg->error() << "Invalid particle plan." << std::endl;
            throw std::runtime_error("Error during communication.");
            }
        }
    while (condition);

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
                           &n_recv_ptls);

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
        #ifdef ENABLE_MPI_CUDAA
        ArrayHandle<char> corner_send_buf_handle(m_corner_send_buf, access_location::device, access_mode::read);
        ArrayHandle<char> edge_send_buf_handle(m_edge_send_buf, access_location::device, access_mode::readwrite);
        ArrayHandle<char> face_send_buf_handle(m_face_send_buf, access_location::device, access_mode::readwrite);
        ArrayHandle<char> recv_buf_handle(m_recv_buf, access_location::device, access_mode::readwrite);
        #else
        ArrayHandle<char> corner_send_buf_handle(m_corner_send_buf, access_location::host, access_mode::read);
        ArrayHandle<char> edge_send_buf_handle(m_edge_send_buf, access_location::host, access_mode::readwrite);
        ArrayHandle<char> face_send_buf_handle(m_face_send_buf, access_location::host, access_mode::readwrite);
        ArrayHandle<char> recv_buf_handle(m_recv_buf, access_location::host, access_mode::readwrite);
        #endif

        unsigned int cpitch = m_corner_send_buf.getPitch();
        unsigned int epitch = m_edge_send_buf.getPitch();
        unsigned int fpitch = m_face_send_buf.getPitch();

        communicateStepTwo(dir,
                           corner_send_buf_handle.data,
                           edge_send_buf_handle.data,
                           face_send_buf_handle.data,
                           cpitch,
                           epitch,
                           fpitch,
                           recv_buf_handle.data,
                           n_send_ptls_corner,
                           n_send_ptls_edge,
                           n_send_ptls_face,
                           n_recv_ptls_edge,
                           n_recv_ptls_face,
                           n_recv_ptls,
                           n_tot_recv_ptls,
                           gpu_pdata_element_size());

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

    // initialization of buffers
    if (! m_buffers_allocated)
        allocateBuffers();

    unsigned int condition;
    do {
        // resize buffers if necessary 
        if (m_corner_ghosts_buf.getPitch() < m_max_copy_ghosts_corner*gpu_ghost_element_size())
            m_corner_ghosts_buf.resize(m_max_copy_ghosts_corner*gpu_ghost_element_size(), 8);

        if (m_corner_update_buf.getPitch() < m_max_copy_ghosts_corner*gpu_update_element_size())
            m_corner_update_buf.resize(m_max_copy_ghosts_corner*gpu_update_element_size(), 8);

        if (m_edge_ghosts_buf.getPitch() < m_max_copy_ghosts_edge*gpu_ghost_element_size())
            m_edge_ghosts_buf.resize(m_max_copy_ghosts_edge*gpu_ghost_element_size(), 12);

        if (m_edge_update_buf.getPitch() < m_max_copy_ghosts_edge*gpu_update_element_size())
            m_edge_update_buf.resize(m_max_copy_ghosts_edge*gpu_update_element_size(), 12);

        if (m_face_ghosts_buf.getPitch() < m_max_copy_ghosts_face*gpu_ghost_element_size())
            m_face_ghosts_buf.resize(m_max_copy_ghosts_face*gpu_ghost_element_size(), 6);

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
                                m_n_copy_ghosts_corner,
                                m_n_copy_ghosts_edge,
                                m_n_copy_ghosts_face,
                                m_max_copy_ghosts_corner,
                                m_max_copy_ghosts_edge,
                                m_max_copy_ghosts_face,
                                m_is_at_boundary,
                                m_pdata->getGlobalBox(),
                                m_condition.getDeviceFlags());

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();

            m_exec_conf->releaseContext();
            }

        condition = m_condition.readFlags();
        if (condition & 1)
            {
            // overflow of corner copy buf
            unsigned int new_size = 1;
            for (unsigned int i = 0; i < 8; ++i)
                if (m_n_copy_ghosts_corner[i] > new_size) new_size = m_n_copy_ghosts_corner[i];
            while (m_max_copy_ghosts_corner < new_size)
                m_max_copy_ghosts_corner = ceilf((float)m_max_copy_ghosts_corner * m_resize_factor);
            }
        if (condition & 2)
            {
            // overflow of edge copy buf
            unsigned int new_size = 1;
            for (unsigned int i = 0; i < 12; ++i)
                if (m_n_copy_ghosts_edge[i] > new_size) new_size = m_n_copy_ghosts_edge[i];
            while (m_max_copy_ghosts_edge < new_size)
                m_max_copy_ghosts_edge = ceilf((float)m_max_copy_ghosts_edge*m_resize_factor);
            }
        if (condition & 4)
            {
            // overflow of face copy buf
            unsigned int new_size = 1;
            for (unsigned int i = 0; i < 6; ++i)
                if (m_n_copy_ghosts_edge[i] > new_size) new_size = m_n_copy_ghosts_edge[i];
            while (m_max_copy_ghosts_face < new_size)
                m_max_copy_ghosts_face = ceilf((float)m_max_copy_ghosts_face*m_resize_factor);
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

    for (unsigned int i = 0; i < 6; ++i)
        m_n_local_ghosts_face[i] = m_n_copy_ghosts_face[i];

    for (unsigned int i = 0; i < 12; ++i)
        m_n_local_ghosts_edge[i] = m_n_copy_ghosts_edge[i];


    /*
     * Fill send buffers, exchange particles according to plans
     */

    // Number of ghosts we received that are not forwarded to other boxes
    unsigned int n_tot_recv_ghosts_local = 0;

    for (unsigned int dir = 0; dir < 6; ++dir)
        {
        if (! isCommunicating(dir) ) continue;

        unsigned int max_n_recv_edge = 0;
        unsigned int max_n_recv_face = 0;

        // exchange message sizes
        communicateStepOne(dir,
                           m_n_copy_ghosts_corner,
                           m_n_copy_ghosts_edge,
                           m_n_copy_ghosts_face,
                           m_n_recv_ghosts_face[dir],
                           m_n_recv_ghosts_edge[dir],
                           &m_n_recv_ghosts_local[dir]
                           );

        unsigned int max_n_copy_edge = 0;
        unsigned int max_n_copy_face = 0;

        // resize buffers as necessary
        for (unsigned int i = 0; i < 12; ++i)
            {
            if (m_n_recv_ghosts_edge[dir][i] > max_n_recv_edge)
                max_n_recv_edge = m_n_recv_ghosts_edge[dir][i];
            if (m_n_copy_ghosts_edge[i] > max_n_copy_edge)
                max_n_copy_edge = m_n_copy_ghosts_edge[i];
            }


        if (max_n_recv_edge + max_n_copy_edge > m_max_copy_ghosts_edge)
            {
            unsigned int new_size = 1;
            while (new_size < max_n_recv_edge + max_n_copy_edge) new_size = ceilf((float)new_size* m_resize_factor);
            m_max_copy_ghosts_edge = new_size;

            m_edge_ghosts_buf.resize(m_max_copy_ghosts_edge*gpu_ghost_element_size(), 12);
            m_edge_update_buf.resize(m_max_copy_ghosts_edge*gpu_update_element_size(), 12);
            }

        for (unsigned int i = 0; i < 6; ++i)
            {
            if (m_n_recv_ghosts_face[dir][i] > max_n_recv_face)
                max_n_recv_face = m_n_recv_ghosts_face[dir][i];
            if (m_n_copy_ghosts_face[i] > max_n_copy_face)
                max_n_copy_face = m_n_copy_ghosts_face[i];
            }

        if (max_n_recv_face + max_n_copy_face > m_max_copy_ghosts_face)
            {
            unsigned int new_size = 1;
            while (new_size < max_n_recv_face + max_n_copy_face) new_size = ceilf((float) new_size * m_resize_factor);
            m_max_copy_ghosts_face = new_size;

            m_face_ghosts_buf.resize(m_max_copy_ghosts_face*gpu_ghost_element_size(), 6);
            m_face_update_buf.resize(m_max_copy_ghosts_face*gpu_update_element_size(), 6);
            }

        if (m_ghosts_recv_buf.getNumElements() < (n_tot_recv_ghosts_local + m_n_recv_ghosts_local[dir])*gpu_ghost_element_size())
            {
            unsigned int new_size =1;
            while (new_size < n_tot_recv_ghosts_local + m_n_recv_ghosts_local[dir])
                new_size = ceilf((float) new_size * m_resize_factor);
            m_ghosts_recv_buf.resize(new_size*gpu_ghost_element_size());
            }

        if (m_update_recv_buf.getNumElements() < (n_tot_recv_ghosts_local + m_n_recv_ghosts_local[dir])*gpu_update_element_size())
            {
            unsigned int new_size =1;
            while (new_size < n_tot_recv_ghosts_local + m_n_recv_ghosts_local[dir])
                new_size = ceilf((float) new_size * m_resize_factor);
            m_update_recv_buf.resize(new_size*gpu_update_element_size());
            }
          
        // exchange ghost particle data
        #ifdef ENABLE_MPI_CUDAA
        ArrayHandle<char> corner_ghosts_buf_handle(m_corner_ghosts_buf, access_location::device, access_mode::read);
        ArrayHandle<char> edge_ghosts_buf_handle(m_edge_ghosts_buf, access_location::device, access_mode::readwrite);
        ArrayHandle<char> face_ghosts_buf_handle(m_face_ghosts_buf, access_location::device, access_mode::readwrite);
        ArrayHandle<char> ghosts_recv_buf_handle(m_recv_buf, access_location::device, access_mode::readwrite);
        #else
        ArrayHandle<char> corner_ghosts_buf_handle(m_corner_ghosts_buf, access_location::host, access_mode::read);
        ArrayHandle<char> edge_ghosts_buf_handle(m_edge_ghosts_buf, access_location::host, access_mode::readwrite);
        ArrayHandle<char> face_ghosts_buf_handle(m_face_ghosts_buf, access_location::host, access_mode::readwrite);
        ArrayHandle<char> ghosts_recv_buf_handle(m_ghosts_recv_buf, access_location::host, access_mode::readwrite);
        #endif
        unsigned int cpitch = m_corner_ghosts_buf.getPitch();
        unsigned int epitch = m_edge_ghosts_buf.getPitch();
        unsigned int fpitch = m_face_ghosts_buf.getPitch();

        communicateStepTwo(dir,
                           corner_ghosts_buf_handle.data,
                           edge_ghosts_buf_handle.data,
                           face_ghosts_buf_handle.data,
                           cpitch,
                           epitch,
                           fpitch,
                           ghosts_recv_buf_handle.data,
                           m_n_copy_ghosts_corner,
                           m_n_copy_ghosts_edge,
                           m_n_copy_ghosts_face,
                           m_n_recv_ghosts_edge[dir],
                           m_n_recv_ghosts_face[dir],
                           m_n_recv_ghosts_local[dir],
                           n_tot_recv_ghosts_local, 
                           gpu_ghost_element_size());

        // update buffer sizes
        for (unsigned int i = 0; i < 12; ++i)
            m_n_copy_ghosts_edge[i] += m_n_recv_ghosts_edge[dir][i];

        for (unsigned int i = 0; i < 6; ++i)
            m_n_copy_ghosts_face[i] += m_n_recv_ghosts_face[dir][i];

        n_tot_recv_ghosts_local += m_n_recv_ghosts_local[dir];
        }

    // calculate number of forwarded particles for every face and edge
    for (unsigned int i = 0; i < 6; ++i)
        n_forward_ghosts_face[i] = m_n_copy_ghosts_face[i] - m_n_local_ghosts_face[i];

    for (unsigned int i = 0; i < 12; ++i)
        n_forward_ghosts_edge[i] = m_n_copy_ghosts_edge[i] - m_n_local_ghosts_edge[i];

    // total up number of received ghosts
    m_n_tot_recv_ghosts = n_tot_recv_ghosts_local;
    for (unsigned int i = 0; i < 6; ++i)
        m_n_tot_recv_ghosts += n_forward_ghosts_face[i];
    for (unsigned int i = 0; i < 12; ++i)
        m_n_tot_recv_ghosts += n_forward_ghosts_edge[i];

    // update number of ghost particles
    m_pdata->addGhostParticles(m_n_tot_recv_ghosts);

        {
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::readwrite);
        ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::readwrite);
        ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::readwrite);

        ArrayHandle<char> d_face_ghosts(m_face_ghosts_buf, access_location::device, access_mode::read);
        ArrayHandle<char> d_edge_ghosts(m_edge_ghosts_buf, access_location::device, access_mode::read);
        ArrayHandle<char> d_recv_ghosts(m_ghosts_recv_buf, access_location::device, access_mode::read);

        m_exec_conf->useContext();
        gpu_exchange_ghosts_unpack(m_pdata->getN(),
                                     m_n_tot_recv_ghosts,
                                     m_n_local_ghosts_face,
                                     m_n_local_ghosts_edge,
                                     n_forward_ghosts_face,
                                     n_forward_ghosts_edge,
                                     n_tot_recv_ghosts_local,
                                     d_face_ghosts.data,
                                     m_face_ghosts_buf.getPitch(),
                                     d_edge_ghosts.data,
                                     m_edge_ghosts_buf.getPitch(),
                                     d_recv_ghosts.data,
                                     d_pos.data,
                                     d_charge.data,
                                     d_diameter.data,
                                     d_tag.data,
                                     d_rtag.data);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_exec_conf->releaseContext();

        }

    // we have updated ghost particles, so inform ParticleData about this
    m_pdata->notifyGhostParticleNumberChange();

    if (m_prof)
        m_prof->pop();
    }

void CommunicatorGPU::communicateStepOne(unsigned int dir,
                                        unsigned int *n_send_ptls_corner,
                                        unsigned int *n_send_ptls_edge,
                                        unsigned int *n_send_ptls_face,
                                        unsigned int *n_recv_ptls_face,
                                        unsigned int *n_recv_ptls_edge,
                                        unsigned int *n_recv_ptls_local)
    {
    // communicate size of the messages that will contain the particle data
    MPI_Request reqs[18];
    MPI_Status status[18];
    unsigned int nreq=0;

    unsigned int send_neighbor = m_decomposition->getNeighborRank(dir);

    // we receive from the direction opposite to the one we send to
    unsigned int recv_neighbor;
    if (dir % 2 == 0)
        recv_neighbor = m_decomposition->getNeighborRank(dir+1);
    else
        recv_neighbor = m_decomposition->getNeighborRank(dir-1);


    if (dir == face_east)
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
    else if (dir == face_west)
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
    else if (dir == face_north)
        {
        MPI_Isend(&n_send_ptls_edge[edge_north_up], sizeof(unsigned int), MPI_BYTE, send_neighbor, 6, m_mpi_comm, &reqs[nreq++]);
        MPI_Isend(&n_send_ptls_edge[edge_north_down], sizeof(unsigned int), MPI_BYTE, send_neighbor, 7, m_mpi_comm, &reqs[nreq++]);
        MPI_Isend(&n_send_ptls_face[face_north], sizeof(unsigned int), MPI_BYTE, send_neighbor, 8, m_mpi_comm, &reqs[nreq++]);
        }
    else if (dir == face_south)
        {
        MPI_Isend(&n_send_ptls_edge[edge_south_up], sizeof(unsigned int), MPI_BYTE, send_neighbor, 6, m_mpi_comm, &reqs[nreq++]);
        MPI_Isend(&n_send_ptls_edge[edge_south_down], sizeof(unsigned int), MPI_BYTE, send_neighbor, 7, m_mpi_comm, &reqs[nreq++]);
        MPI_Isend(&n_send_ptls_face[face_south], sizeof(unsigned int), MPI_BYTE, send_neighbor, 8, m_mpi_comm, &reqs[nreq++]);
        } 
    else if (dir == face_up)
        {
        MPI_Isend(&n_send_ptls_face[face_up], sizeof(unsigned int), MPI_BYTE, send_neighbor, 8, m_mpi_comm, &reqs[nreq++]);
        }
    else if (dir == face_down)
        {
        MPI_Isend(&n_send_ptls_face[face_down], sizeof(unsigned int), MPI_BYTE, send_neighbor, 8, m_mpi_comm, &reqs[nreq++]);
        }
  
    // receive message sizes
    if (dir == face_east || dir == face_west)
        {
        MPI_Irecv(&n_recv_ptls_edge[edge_north_up], sizeof(unsigned int), MPI_BYTE, recv_neighbor, 0, m_mpi_comm, &reqs[nreq++]);
        MPI_Irecv(&n_recv_ptls_edge[edge_north_down], sizeof(unsigned int), MPI_BYTE, recv_neighbor, 1, m_mpi_comm, & reqs[nreq++]);
        MPI_Irecv(&n_recv_ptls_edge[edge_south_up], sizeof(unsigned int), MPI_BYTE, recv_neighbor, 2, m_mpi_comm, & reqs[nreq++]);
        MPI_Irecv(&n_recv_ptls_edge[edge_south_down], sizeof(unsigned int), MPI_BYTE, recv_neighbor, 3, m_mpi_comm, & reqs[nreq++]);
        MPI_Irecv(&n_recv_ptls_face[face_north], sizeof(unsigned int), MPI_BYTE, recv_neighbor, 4, m_mpi_comm, & reqs[nreq++]);
        MPI_Irecv(&n_recv_ptls_face[face_south], sizeof(unsigned int), MPI_BYTE, recv_neighbor, 5, m_mpi_comm, & reqs[nreq++]);
        }

    if (dir == face_east || dir == face_west || dir == face_north || dir == face_south)
        {
        MPI_Irecv(&n_recv_ptls_face[face_up], sizeof(unsigned int), MPI_BYTE, recv_neighbor, 6, m_mpi_comm, & reqs[nreq++]);
        MPI_Irecv(&n_recv_ptls_face[face_down], sizeof(unsigned int), MPI_BYTE, recv_neighbor, 7, m_mpi_comm, & reqs[nreq++]);
        }

    MPI_Irecv(n_recv_ptls_local, sizeof(unsigned int), MPI_BYTE, recv_neighbor, 8, m_mpi_comm, &reqs[nreq++]);

    MPI_Waitall(nreq, reqs, status);
    }

void CommunicatorGPU::communicateStepTwo(unsigned int face,
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
                                const unsigned int *n_recv_ptls_edge,
                                const unsigned int *n_recv_ptls_face,
                                const unsigned int n_recv_ptls_local,
                                const unsigned int n_tot_recv_ptls_local,
                                const unsigned int element_size)
    {
    MPI_Request reqs[18];
    MPI_Status status[18];
    unsigned int nreq=0;

    unsigned int send_neighbor = m_decomposition->getNeighborRank(face);

    // we receive from the direction opposite to the one we send to
    unsigned int recv_neighbor;
    if (face % 2 == 0)
        recv_neighbor = m_decomposition->getNeighborRank(face+1);
    else
        recv_neighbor = m_decomposition->getNeighborRank(face-1);

    if (face == face_east)
        {
        MPI_Isend(corner_send_buf+corner_east_north_up*cpitch,
                  n_send_ptls_corner[corner_east_north_up]*element_size,
                  MPI_BYTE, send_neighbor, 0, m_mpi_comm, & reqs[nreq++]);
        MPI_Isend(corner_send_buf+corner_east_north_down*cpitch,
                  n_send_ptls_corner[corner_east_north_down]*element_size,
                  MPI_BYTE, send_neighbor, 1, m_mpi_comm, & reqs[nreq++]);
        MPI_Isend(corner_send_buf+corner_east_south_up*cpitch,
                  n_send_ptls_corner[corner_east_south_up]*element_size,
                  MPI_BYTE, send_neighbor, 2, m_mpi_comm, & reqs[nreq++]);
        MPI_Isend(corner_send_buf+corner_east_south_down*cpitch,
                  n_send_ptls_corner[corner_east_south_down]*element_size,
                  MPI_BYTE, send_neighbor, 3, m_mpi_comm, & reqs[nreq++]);

        MPI_Isend(edge_send_buf+edge_east_north*epitch,
                  n_send_ptls_edge[edge_east_north]*element_size,
                  MPI_BYTE, send_neighbor, 4, m_mpi_comm, &reqs[nreq++]);
        MPI_Isend(edge_send_buf+edge_east_south*epitch,
                  n_send_ptls_edge[edge_east_south]*element_size,
                  MPI_BYTE, send_neighbor, 5, m_mpi_comm, &reqs[nreq++]);
        MPI_Isend(edge_send_buf+edge_east_up*epitch,
                  n_send_ptls_edge[edge_east_up]*element_size,
                  MPI_BYTE, send_neighbor, 6, m_mpi_comm, &reqs[nreq++]);
        MPI_Isend(edge_send_buf+edge_east_down*epitch,
                  n_send_ptls_edge[edge_east_down]*element_size,
                  MPI_BYTE, send_neighbor, 7, m_mpi_comm, &reqs[nreq++]);

       }
    else if (face == face_west)
        {
        MPI_Isend(corner_send_buf+corner_west_north_up*cpitch,
                  n_send_ptls_corner[corner_west_north_up]*element_size,
                  MPI_BYTE, send_neighbor, 0, m_mpi_comm, & reqs[nreq++]);
        MPI_Isend(corner_send_buf+corner_west_north_down*cpitch,
                  n_send_ptls_corner[corner_west_north_down]*element_size,
                  MPI_BYTE, send_neighbor, 1, m_mpi_comm, & reqs[nreq++]);
        MPI_Isend(corner_send_buf+corner_west_south_up*cpitch,
                  n_send_ptls_corner[corner_west_south_up]*element_size,
                  MPI_BYTE, send_neighbor, 2, m_mpi_comm, & reqs[nreq++]);
        MPI_Isend(corner_send_buf+corner_west_south_down*cpitch,
                  n_send_ptls_corner[corner_west_south_down]*element_size,
                  MPI_BYTE, send_neighbor, 3, m_mpi_comm, & reqs[nreq++]);

        MPI_Isend(edge_send_buf+edge_west_north*epitch,
                  n_send_ptls_edge[edge_west_north]*element_size,
                  MPI_BYTE, send_neighbor, 4, m_mpi_comm, &reqs[nreq++]);
        MPI_Isend(edge_send_buf+edge_west_south*epitch,
                  n_send_ptls_edge[edge_west_south]*element_size,
                  MPI_BYTE, send_neighbor, 5, m_mpi_comm, &reqs[nreq++]);
        MPI_Isend(edge_send_buf+edge_west_up*epitch,
                  n_send_ptls_edge[edge_west_up]*element_size,
                  MPI_BYTE, send_neighbor, 6, m_mpi_comm, &reqs[nreq++]);
        MPI_Isend(edge_send_buf+edge_west_down*epitch,
                  n_send_ptls_edge[edge_west_down]*element_size,
                  MPI_BYTE, send_neighbor, 7, m_mpi_comm, &reqs[nreq++]);
        }
    else if (face == face_north)
        {
        MPI_Isend(edge_send_buf+edge_north_up*epitch,
                  n_send_ptls_edge[edge_north_up]*element_size,
                  MPI_BYTE, send_neighbor, 6, m_mpi_comm, &reqs[nreq++]);
        MPI_Isend(edge_send_buf+edge_north_down*epitch,
                  n_send_ptls_edge[edge_north_down]*element_size,
                  MPI_BYTE, send_neighbor, 7, m_mpi_comm, &reqs[nreq++]);
        }
    else if (face == face_south)
        {
        MPI_Isend(edge_send_buf+edge_south_up*epitch,
                  n_send_ptls_edge[edge_south_up]*element_size,
                  MPI_BYTE, send_neighbor, 6, m_mpi_comm, &reqs[nreq++]);
        MPI_Isend(edge_send_buf+edge_south_down*epitch,
                  n_send_ptls_edge[edge_south_down]*element_size,
                  MPI_BYTE, send_neighbor, 7, m_mpi_comm, &reqs[nreq++]);
        } 

    MPI_Isend(face_send_buf+face*fpitch,
              n_send_ptls_face[face]*element_size,
              MPI_BYTE, send_neighbor, 8, m_mpi_comm, &reqs[nreq++]);
    
    // receive particle data
    if (face == face_east || face == face_west)
        {
        MPI_Irecv(edge_send_buf+edge_north_up*epitch+n_send_ptls_edge[edge_north_up]*element_size,
                  n_recv_ptls_edge[edge_north_up]*element_size,
                  MPI_BYTE, recv_neighbor, 0, m_mpi_comm, &reqs[nreq++]);
        MPI_Irecv(edge_send_buf+edge_north_down*epitch+n_send_ptls_edge[edge_north_down]*element_size,
                  n_recv_ptls_edge[edge_north_down]*element_size,
                  MPI_BYTE, recv_neighbor, 1, m_mpi_comm, &reqs[nreq++]);
        MPI_Irecv(edge_send_buf+edge_south_up*epitch+n_send_ptls_edge[edge_south_up]*element_size,
                  n_recv_ptls_edge[edge_south_up]*element_size,
                  MPI_BYTE, recv_neighbor, 2, m_mpi_comm, &reqs[nreq++]);
        MPI_Irecv(edge_send_buf+edge_south_down*epitch+n_send_ptls_edge[edge_south_down]*element_size,
                  n_recv_ptls_edge[edge_south_down]*element_size,
                  MPI_BYTE, recv_neighbor, 3, m_mpi_comm, &reqs[nreq++]);

        MPI_Irecv(face_send_buf+face_north*fpitch+n_send_ptls_face[face_north]*element_size,
                  n_recv_ptls_face[face_north]*element_size,
                  MPI_BYTE, recv_neighbor, 4, m_mpi_comm, & reqs[nreq++]);
        MPI_Irecv(face_send_buf+face_south*fpitch+n_send_ptls_face[face_south]*element_size,
                  n_recv_ptls_face[face_south]*element_size,
                  MPI_BYTE, recv_neighbor, 5, m_mpi_comm, & reqs[nreq++]);
        }

    if (face == face_east || face == face_west || face == face_north || face == face_south)
        {
        MPI_Irecv(face_send_buf+face_up*fpitch+n_send_ptls_face[face_up]*element_size,
                  n_recv_ptls_face[face_up]*element_size,
                  MPI_BYTE, recv_neighbor, 6, m_mpi_comm, & reqs[nreq++]);
        MPI_Irecv(face_send_buf+face_down*fpitch+n_send_ptls_face[face_down]*element_size,
                  n_recv_ptls_face[face_down]*element_size,
                  MPI_BYTE, recv_neighbor, 7, m_mpi_comm, & reqs[nreq++]);
        }

    MPI_Irecv(local_recv_buf+n_tot_recv_ptls_local*element_size,
              n_recv_ptls_local*element_size,
              MPI_BYTE, recv_neighbor, 8, m_mpi_comm, &reqs[nreq++]);

    MPI_Waitall(nreq, reqs, status);
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
