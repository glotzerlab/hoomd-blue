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

//#define MPI3 // define if the MPI implementation supports MPI3 one-sided communications
//#define MPI_CUDA_UPDATE // define if the ghosts update should be carried out using the MPI-CUDA implementation

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
      m_communicator(communicator)
    { }

//! Destructor
ghost_gpu_thread::~ghost_gpu_thread()
    {
    }

//! Main routine of ghost update worker thread
void ghost_gpu_thread::operator()(WorkQueue<ghost_gpu_thread_params>& queue, boost::barrier& barrier)
    {
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
    unsigned int n_copy_ghosts_face[6];
    unsigned int n_copy_ghosts_edge[12];
    unsigned int n_copy_ghosts_corner[8];

        {
        ArrayHandle<unsigned int> h_n_local_ghosts_face(params.n_local_ghosts_face, access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_n_local_ghosts_edge(params.n_local_ghosts_edge, access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_n_local_ghosts_corner(params.n_local_ghosts_corner, access_location::host, access_mode::read);

        for (unsigned int i = 0; i < 12; ++i)
            n_copy_ghosts_edge[i] = h_n_local_ghosts_edge.data[i];
        for (unsigned int i = 0; i < 6; ++i)
            n_copy_ghosts_face[i] = h_n_local_ghosts_face.data[i];
        for (unsigned int i = 0; i < 8; ++i)
            n_copy_ghosts_corner[i] = h_n_local_ghosts_corner.data[i];
        }

    unsigned int n_tot_recv_ghosts_local = 0;

    for (unsigned int face = 0; face < 6; ++face)
        {
        if (! m_communicator->isCommunicating(face)) continue;

        ArrayHandle<unsigned int> h_n_recv_ghosts_face(params.n_recv_ghosts_face, access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_n_recv_ghosts_edge(params.n_recv_ghosts_edge, access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_n_recv_ghosts_local(params.n_recv_ghosts_local, access_location::host, access_mode::read);

        m_communicator->communicateStepTwo(face,
                                           params.corner_update_buf_handle,
                                           params.edge_update_buf_handle,
                                           params.face_update_buf_handle,
                                           params.corner_update_buf_pitch,
                                           params.edge_update_buf_pitch,
                                           params.face_update_buf_pitch,
                                           params.update_recv_buf_handle,
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
        } // end communication loop

    } 

//! Constructor
CommunicatorGPU::CommunicatorGPU(boost::shared_ptr<SystemDefinition> sysdef,
                                 boost::shared_ptr<DomainDecomposition> decomposition)
    : Communicator(sysdef, decomposition), m_remove_mask(m_exec_conf),
      m_max_send_ptls_face(0),
      m_max_send_ptls_edge(0),
      m_max_send_ptls_corner(0),
      m_max_send_bonds_face(0),
      m_max_send_bonds_edge(0),
      m_max_send_bonds_corner(0),
      m_max_copy_ghosts_face(0),
      m_max_copy_ghosts_edge(0),
      m_max_copy_ghosts_corner(0),
      m_max_recv_ghosts(0),
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

    gpu_allocate_tmp_storage(is_communicating,m_is_at_boundary);

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
    
    if (m_thread_created)
        {
        // finish worker thread
        m_worker_thread.interrupt();
        m_worker_thread.join();
        }

    }

//! Start ghosts communication
/*! This is the multi-threaded version.
 */
void CommunicatorGPU::startGhostsUpdate(unsigned int timestep)
    {
    if (timestep < m_next_ghost_update)
        return;

    m_exec_conf->msg->notice(7) << "CommunicatorGPU: ghost update" << std::endl;

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

        unsigned int n_tot_local_ghosts = 0;
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

        gpu_update_ghosts_pack(n_tot_local_ghosts,
                            d_ghost_idx_face.data,
                            m_ghost_idx_face.getPitch(),
                            d_ghost_idx_edge.data,
                            m_ghost_idx_edge.getPitch(),
                            d_ghost_idx_corner.data,
                            m_ghost_idx_corner.getPitch(),
                            d_pos.data,
                            m_corner_update_buf.getDevicePointer(),
                            m_corner_update_buf.getPitch(),
                            m_edge_update_buf.getDevicePointer(),
                            m_edge_update_buf.getPitch(),
                            m_face_update_buf.getDevicePointer(),
                            m_face_update_buf.getPitch(),
                            d_n_local_ghosts_corner.data,
                            d_n_local_ghosts_edge.data,
                            d_n_local_ghosts_face.data);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    // fill thread parameters
    ghost_gpu_thread_params params(
        m_corner_update_buf.getHostPointer(),
        m_corner_update_buf.getPitch(),
        m_edge_update_buf.getHostPointer(),
        m_edge_update_buf.getPitch(),
        m_face_update_buf.getHostPointer(),
        m_face_update_buf.getPitch(),
        m_update_recv_buf.getHostPointer(),
        m_update_recv_buf.getNumElements(),
        m_n_recv_ghosts_edge,
        m_n_recv_ghosts_face,
        m_n_recv_ghosts_local,
        m_n_local_ghosts_corner,
        m_n_local_ghosts_edge,
        m_n_local_ghosts_face);

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

        {
        // unpack ghost data
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);

        ArrayHandle<unsigned int> d_ghost_plan(m_ghost_plan, access_location::device, access_mode::read);

        ArrayHandle<unsigned int> d_n_local_ghosts_face(m_n_local_ghosts_face, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_n_local_ghosts_edge(m_n_local_ghosts_edge, access_location::device, access_mode::read);

        ArrayHandle<unsigned int> d_n_recv_ghosts_face(m_n_recv_ghosts_face, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_n_recv_ghosts_edge(m_n_recv_ghosts_edge, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_n_recv_ghosts_local(m_n_recv_ghosts_local, access_location::device, access_mode::read);

        // unpack particles
        gpu_update_ghosts_unpack(m_pdata->getN(),
                                 m_n_tot_recv_ghosts,
                                 d_n_local_ghosts_face.data,
                                 d_n_local_ghosts_edge.data,
                                 m_n_tot_recv_ghosts_local,
                                 d_n_recv_ghosts_local.data,
                                 d_n_recv_ghosts_face.data,
                                 d_n_recv_ghosts_edge.data,
                                 m_face_update_buf.getDevicePointer(),
                                 m_face_update_buf.getPitch(),
                                 m_edge_update_buf.getDevicePointer(),
                                 m_edge_update_buf.getPitch(),
                                 m_update_recv_buf.getDevicePointer(),
                                 d_pos.data,
                                 d_ghost_plan.data,
                                 m_pdata->getGlobalBox());

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    if (m_prof) m_prof->pop();
    }

void CommunicatorGPU::allocateBuffers()
    {
    /*
     * initial size of particle send buffers = max of avg. number of ptls in skin layer in any direction
     */ 
    const BoxDim& box = m_pdata->getBox();
    Scalar3 L = box.getL();

    unsigned int maxx = (unsigned int)((Scalar)m_pdata->getN()*m_r_buff/L.x);
    unsigned int maxy = (unsigned int)((Scalar)m_pdata->getN()*m_r_buff/L.y);
    unsigned int maxz = (unsigned int)((Scalar)m_pdata->getN()*m_r_buff/L.z);

    m_max_send_ptls_face = 1;
    m_max_send_ptls_face = m_max_send_ptls_face > maxx ? m_max_send_ptls_face : maxx;
    m_max_send_ptls_face = m_max_send_ptls_face > maxy ? m_max_send_ptls_face : maxy;
    m_max_send_ptls_face = m_max_send_ptls_face > maxz ? m_max_send_ptls_face : maxz;

    GPUBuffer<char> face_send_buf(gpu_pdata_element_size()*m_max_send_ptls_face, 6, m_exec_conf);
    m_face_send_buf.swap(face_send_buf);

    unsigned int maxxy = (unsigned int)((Scalar)m_pdata->getN()*m_r_buff*m_r_buff/L.x/L.y);
    unsigned int maxxz = (unsigned int)((Scalar)m_pdata->getN()*m_r_buff*m_r_buff/L.x/L.z);
    unsigned int maxyz = (unsigned int)((Scalar)m_pdata->getN()*m_r_buff*m_r_buff/L.y/L.z);
    
    m_max_send_ptls_edge = 1;
    m_max_send_ptls_edge = m_max_send_ptls_edge > maxxy ? m_max_send_ptls_edge : maxxy;
    m_max_send_ptls_edge = m_max_send_ptls_edge > maxxz ? m_max_send_ptls_edge : maxxz;
    m_max_send_ptls_edge = m_max_send_ptls_edge > maxyz ? m_max_send_ptls_edge : maxyz;

    GPUBuffer<char> edge_send_buf(gpu_pdata_element_size()*m_max_send_ptls_edge, 12, m_exec_conf);
    m_edge_send_buf.swap(edge_send_buf);

    unsigned maxxyz = (unsigned int)((Scalar)m_pdata->getN()*m_r_buff*m_r_buff*m_r_buff/L.x/L.y/L.z);
    m_max_send_ptls_corner = maxxyz > 1 ? maxxyz : 1;

    GPUBuffer<char> send_buf_corner(gpu_pdata_element_size()*m_max_send_ptls_corner, 8, m_exec_conf);
    m_corner_send_buf.swap(send_buf_corner);

    GPUBuffer<char> recv_buf(gpu_pdata_element_size()*m_max_send_ptls_face*6, m_exec_conf);
    m_recv_buf.swap(recv_buf);
   
    /*
     * initial size of ghost send buffers = max of avg number of ptls in ghost layer in every direction
     */ 
    maxx = (unsigned int)((Scalar)m_pdata->getN()*m_r_ghost/L.x);
    maxx = (unsigned int)((Scalar)m_pdata->getN()*m_r_ghost/L.y);
    maxx = (unsigned int)((Scalar)m_pdata->getN()*m_r_ghost/L.z);

    m_max_copy_ghosts_face = 1;
    m_max_copy_ghosts_face = m_max_copy_ghosts_face > maxx ? m_max_copy_ghosts_face : maxx;
    m_max_copy_ghosts_face = m_max_copy_ghosts_face > maxy ? m_max_copy_ghosts_face : maxy;
    m_max_copy_ghosts_face = m_max_copy_ghosts_face > maxz ? m_max_copy_ghosts_face : maxz;

    GPUBuffer<char> face_ghosts_buf(gpu_ghost_element_size()*m_max_copy_ghosts_face, 6, m_exec_conf);
    m_face_ghosts_buf.swap(face_ghosts_buf);

    GPUBuffer<char> face_update_buf(gpu_update_element_size()*m_max_copy_ghosts_face, 6, m_exec_conf);
    m_face_update_buf.swap(face_update_buf);

    maxxy = (unsigned int)((Scalar)m_pdata->getN()*m_r_ghost*m_r_ghost/L.x/L.y);
    maxxz = (unsigned int)((Scalar)m_pdata->getN()*m_r_ghost*m_r_ghost/L.x/L.z);
    maxyz = (unsigned int)((Scalar)m_pdata->getN()*m_r_ghost*m_r_ghost/L.y/L.z);

    m_max_copy_ghosts_edge = 1;
    m_max_copy_ghosts_edge = m_max_copy_ghosts_edge > maxxy ? m_max_copy_ghosts_edge : maxxy;
    m_max_copy_ghosts_edge = m_max_copy_ghosts_edge > maxxz ? m_max_copy_ghosts_edge : maxxz;
    m_max_copy_ghosts_edge = m_max_copy_ghosts_edge > maxyz ? m_max_copy_ghosts_edge : maxyz;

    GPUBuffer<char> edge_ghosts_buf(gpu_ghost_element_size()*m_max_copy_ghosts_edge, 12, m_exec_conf);
    m_edge_ghosts_buf.swap(edge_ghosts_buf);

    GPUBuffer<char> edge_update_buf(gpu_update_element_size()*m_max_copy_ghosts_edge, 12, m_exec_conf);
    m_edge_update_buf.swap(edge_update_buf);

    maxxyz = (unsigned int)((Scalar)m_pdata->getN()*m_r_ghost*m_r_ghost*m_r_ghost/L.x/L.y/L.z);
    m_max_copy_ghosts_corner = maxxyz > 1 ? maxxyz : 1;

    GPUBuffer<char> corner_ghosts_buf(gpu_ghost_element_size()*m_max_copy_ghosts_corner, 8, m_exec_conf);
    m_corner_ghosts_buf.swap(corner_ghosts_buf);

    GPUBuffer<char> corner_update_buf(gpu_update_element_size()*m_max_copy_ghosts_corner, 8, m_exec_conf);
    m_corner_update_buf.swap(corner_update_buf);

    m_max_recv_ghosts = m_max_copy_ghosts_face*6;
    GPUBuffer<char> ghost_recv_buf(gpu_ghost_element_size()*m_max_recv_ghosts, m_exec_conf);
    m_ghosts_recv_buf.swap(ghost_recv_buf);

    GPUBuffer<char> update_recv_buf(gpu_update_element_size()*m_max_recv_ghosts,m_exec_conf);
    m_update_recv_buf.swap(update_recv_buf);

    // buffer for particle plans
    GPUArray<unsigned int> ptl_plan(m_pdata->getN(), m_exec_conf);
    m_ptl_plan.swap(ptl_plan);

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

    /*
     * Initialize send buffers for bonds only if needed
     */
    boost::shared_ptr<BondData> bdata = m_sysdef->getBondData();

    if (bdata->getNumBondsGlobal())
        {
        // estimate initial numbers of bonds from numbers of particles
        m_max_send_bonds_corner = m_max_send_ptls_corner;
        m_max_send_bonds_edge = m_max_send_ptls_edge;
        m_max_send_bonds_face = m_max_send_ptls_face;

        GPUBuffer<bond_element> bond_corner_send_buf(m_max_send_bonds_corner,8, m_exec_conf);
        m_bond_corner_send_buf.swap(bond_corner_send_buf);

        GPUBuffer<bond_element> bond_edge_send_buf(m_max_send_bonds_edge,12, m_exec_conf);
        m_bond_edge_send_buf.swap(bond_edge_send_buf);

        GPUBuffer<bond_element> bond_face_send_buf(m_max_send_bonds_face,6, m_exec_conf);
        m_bond_face_send_buf.swap(bond_face_send_buf);

        GPUBuffer<bond_element> bond_recv_buf(m_max_send_ptls_face*6, m_exec_conf);
        m_bond_recv_buf.swap(bond_recv_buf);

        // mask for bonds, indicating if they will be removed
        GPUArray<unsigned char> bond_remove_mask(bdata->getNumBonds(), m_exec_conf);
        m_bond_remove_mask.swap(bond_remove_mask);

        // counters for number of sent bonds
        GPUArray<unsigned int> n_send_bonds_face(6, m_exec_conf);
        m_n_send_bonds_face.swap(n_send_bonds_face);
        GPUArray<unsigned int> n_send_bonds_edge(12, m_exec_conf);
        m_n_send_bonds_edge.swap(n_send_bonds_edge);
        GPUArray<unsigned int> n_send_bonds_corner(8, m_exec_conf);
        m_n_send_bonds_corner.swap(n_send_bonds_corner);

        // number of removed bonds
        GPUFlags<unsigned int> n_remove_bonds(m_exec_conf);
        m_n_remove_bonds.swap(n_remove_bonds);
        }

    m_buffers_allocated = true;
    }

//! Transfer particles between neighboring domains
void CommunicatorGPU::migrateAtoms()
    {
    if (m_prof)
        m_prof->push("migrate_particles");

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

    if (m_remove_mask.getNumElements() < m_pdata->getN())
        {
        unsigned int new_size = 1;
        while (new_size < m_pdata->getN())
                new_size = ((unsigned int)(((float)new_size)*m_resize_factor))+1;
 
        m_remove_mask.resize(new_size);
        }

    if (m_ptl_plan.getNumElements() < m_pdata->getN())
        {
        unsigned int new_size = 1;
        while (new_size < m_pdata->getN())
                new_size = ((unsigned int)(((float)new_size)*m_resize_factor))+1;
 
        m_ptl_plan.resize(new_size);
        }


    unsigned int n_send_ptls_face[6];
    unsigned int n_send_ptls_edge[12];
    unsigned int n_send_ptls_corner[8];

    /*
     * Select particles for sending
     */
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

            ArrayHandle<unsigned char> d_remove_mask(m_remove_mask, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_ptl_plan(m_ptl_plan, access_location::device, access_mode::overwrite);

            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
            ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::read);
            ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::read);
            ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::read);
            ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::read);
            ArrayHandle<int3> d_image(m_pdata->getImages(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_body(m_pdata->getBodies(), access_location::device, access_mode::read);
            ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);

            ArrayHandle<unsigned int> d_n_send_ptls_corner(m_n_send_ptls_corner, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_n_send_ptls_edge(m_n_send_ptls_edge, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_n_send_ptls_face(m_n_send_ptls_face, access_location::device, access_mode::overwrite);

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
                                   d_n_send_ptls_corner.data,
                                   d_n_send_ptls_edge.data,
                                   d_n_send_ptls_face.data,
                                   m_max_send_ptls_corner,
                                   m_max_send_ptls_edge,
                                   m_max_send_ptls_face,
                                   d_remove_mask.data,
                                   d_ptl_plan.data,
                                   m_corner_send_buf.getDevicePointer(),
                                   m_corner_send_buf.getPitch(),
                                   m_edge_send_buf.getDevicePointer(),
                                   m_edge_send_buf.getPitch(),
                                   m_face_send_buf.getDevicePointer(),
                                   m_face_send_buf.getPitch(),
                                   m_pdata->getBox(),
                                   m_pdata->getGlobalBox(),
                                   m_condition.getDeviceFlags());

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                        CHECK_CUDA_ERROR();
            }

            {
            // read back numbers of sent particles
            ArrayHandleAsync<unsigned int> h_n_send_ptls_face(m_n_send_ptls_face, access_location::host, access_mode::read);
            ArrayHandleAsync<unsigned int> h_n_send_ptls_edge(m_n_send_ptls_edge, access_location::host, access_mode::read);
            ArrayHandleAsync<unsigned int> h_n_send_ptls_corner(m_n_send_ptls_corner, access_location::host, access_mode::read);

            // synchronize with GPU
            cudaDeviceSynchronize();
 
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
                m_max_send_ptls_face = ((unsigned int)(((float)m_max_send_ptls_face)*m_resize_factor))+1;
            }
        if (condition & 2)
            {
            // set new maximum size for edge send buffers
            unsigned int new_size = 1;
            for (unsigned int i = 0; i < 12; ++i)
                if (n_send_ptls_edge[i] > new_size) new_size = n_send_ptls_edge[i];
            while (m_max_send_ptls_edge < new_size)
                m_max_send_ptls_edge = ((unsigned int)(((float)m_max_send_ptls_edge)*m_resize_factor))+1;
            }
        if (condition & 4)
            {
            // set new maximum size for corner send buffers
            unsigned int new_size = 1;
            for (unsigned int i = 0; i < 8; ++i)
                if (n_send_ptls_corner[i] > new_size) new_size = n_send_ptls_corner[i];
            while (m_max_send_ptls_corner < new_size)
                m_max_send_ptls_corner = ((unsigned int)(((float)m_max_send_ptls_corner)*m_resize_factor))+1;
            }

        if (condition & 8)
            {
            m_exec_conf->msg->error() << "Invalid particle plan." << std::endl;
            throw std::runtime_error("Error during communication.");
            }
        }
    while (condition);

    /*
     * Select bonds for sending
     */
    boost::shared_ptr<BondData> bdata = m_sysdef->getBondData();

    unsigned int n_send_bonds_face[6];
    unsigned int n_send_bonds_edge[12];
    unsigned int n_send_bonds_corner[8];

    for (unsigned int i = 0; i < 6; ++i)
        n_send_bonds_face[i] = 0;
    for (unsigned int i = 0; i < 12; ++i)
        n_send_bonds_edge[i] = 0;
    for (unsigned int i = 0; i < 8; ++i)
        n_send_bonds_corner[i] = 0;

    if (bdata->getNumBondsGlobal())
        {
        // resize mask for bonds to be removed, if necessary
        if (m_bond_remove_mask.getNumElements() < bdata->getNumBonds())
            {
            unsigned int new_size = 1;
            while (new_size < bdata->getNumBonds())
                    new_size = ((unsigned int)(((float)new_size)*m_resize_factor))+1;
     
            m_bond_remove_mask.resize(new_size);
            }

        do
            {
            if (m_bond_corner_send_buf.getPitch() < m_max_send_bonds_corner)
                m_bond_corner_send_buf.resize(m_max_send_bonds_corner, 8);

            if (m_bond_edge_send_buf.getPitch() < m_max_send_bonds_edge)
                m_bond_edge_send_buf.resize(m_max_send_bonds_edge, 12);

            if (m_bond_face_send_buf.getPitch() < m_max_send_bonds_face)
                m_bond_face_send_buf.resize(m_max_send_bonds_face, 6);

            
                {
                ArrayHandle<uint2> d_bonds(bdata->getBondTable(), access_location::device, access_mode::read);
                ArrayHandle<unsigned int> d_bond_type(bdata->getBondTypes(), access_location::device, access_mode::read);
                ArrayHandle<unsigned int> d_bond_tag(bdata->getBondTags(), access_location::device, access_mode::read);
                ArrayHandle<unsigned int> d_bond_rtag(bdata->getBondRTags(), access_location::device, access_mode::readwrite);

                ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::read);
                ArrayHandle<unsigned int> d_ptl_plan(m_ptl_plan, access_location::device, access_mode::read);
                ArrayHandle<unsigned char> d_bond_remove_mask(m_bond_remove_mask, access_location::device, access_mode::overwrite);
                ArrayHandle<unsigned int> d_n_send_bonds_face(m_n_send_bonds_face, access_location::device, access_mode::overwrite);
                ArrayHandle<unsigned int> d_n_send_bonds_edge(m_n_send_bonds_edge, access_location::device, access_mode::overwrite);
                ArrayHandle<unsigned int> d_n_send_bonds_corner(m_n_send_bonds_corner, access_location::device, access_mode::overwrite);
                gpu_send_bonds(bdata->getNumBonds(),
                               m_pdata->getN(),
                               d_bonds.data,
                               d_bond_type.data,
                               d_bond_tag.data,
                               d_bond_rtag.data,
                               d_rtag.data,
                               d_ptl_plan.data,
                               d_bond_remove_mask.data, 
                               m_bond_face_send_buf.getDevicePointer(),
                               m_bond_face_send_buf.getPitch(),
                               m_bond_edge_send_buf.getDevicePointer(),
                               m_bond_edge_send_buf.getPitch(),
                               m_bond_corner_send_buf.getDevicePointer(),
                               m_bond_edge_send_buf.getPitch(),
                               d_n_send_bonds_face.data,   
                               d_n_send_bonds_edge.data,   
                               d_n_send_bonds_corner.data,   
                               m_max_send_bonds_face,
                               m_max_send_bonds_edge,
                               m_max_send_bonds_corner,
                               m_n_remove_bonds.getDeviceFlags(),
                               m_condition.getDeviceFlags());

                if (m_exec_conf->isCUDAErrorCheckingEnabled())
                    CHECK_CUDA_ERROR();
                }

                {
                // read back numbers of sent bonds
                ArrayHandleAsync<unsigned int> h_n_send_bonds_face(m_n_send_bonds_face, access_location::host, access_mode::read);
                ArrayHandleAsync<unsigned int> h_n_send_bonds_edge(m_n_send_bonds_edge, access_location::host, access_mode::read);
                ArrayHandleAsync<unsigned int> h_n_send_bonds_corner(m_n_send_bonds_corner, access_location::host, access_mode::read);

                // synchronize with GPU
                cudaDeviceSynchronize();
     
                for (unsigned int i = 0; i < 6; ++i)
                    n_send_bonds_face[i] = h_n_send_bonds_face.data[i];

                for (unsigned int i = 0; i < 12; ++i)
                    n_send_bonds_edge[i] = h_n_send_bonds_edge.data[i];

                for (unsigned int i = 0; i < 8; ++i)
                    n_send_bonds_corner[i] = h_n_send_bonds_corner.data[i];
                }

            condition = m_condition.readFlags();
            if (condition & 1)
                {
                // set new maximum size for face send buffers
                unsigned int new_size = 1;
                for (unsigned int i = 0; i < 6; ++i)
                    if (n_send_bonds_face[i] > new_size) new_size = n_send_bonds_face[i];
                while (m_max_send_bonds_face < new_size)
                    m_max_send_bonds_face = ((unsigned int)(((float)m_max_send_bonds_face)*m_resize_factor))+1;
                }
            if (condition & 2)
                {
                // set new maximum size for edge send buffers
                unsigned int new_size = 1;
                for (unsigned int i = 0; i < 12; ++i)
                    if (n_send_bonds_edge[i] > new_size) new_size = n_send_bonds_edge[i];
                while (m_max_send_bonds_edge < new_size)
                    m_max_send_bonds_edge = ((unsigned int)(((float)m_max_send_bonds_edge)*m_resize_factor))+1;
                }
            if (condition & 4)
                {
                // set new maximum size for corner send buffers
                unsigned int new_size = 1;
                for (unsigned int i = 0; i < 8; ++i)
                    if (n_send_bonds_corner[i] > new_size) new_size = n_send_bonds_corner[i];
                while (m_max_send_bonds_corner < new_size)
                    m_max_send_bonds_corner = ((unsigned int)(((float)m_max_send_bonds_corner)*m_resize_factor))+1;
                }

            if (condition & 8)
                {
                m_exec_conf->msg->error() << "Invalid particle plan." << std::endl;
                throw std::runtime_error("Error during bond communication.");
                }
            } // end do
        while (condition);
        }

        {
        // reset reverse-lookup tag of sent particles
        ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::readwrite);
        ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);
        ArrayHandle<unsigned char> d_remove_mask(m_remove_mask, access_location::device, access_mode::read);

        gpu_reset_rtag_by_mask(m_pdata->getN(),
                               d_rtag.data,
                               d_tag.data,
                               d_remove_mask.data);
        }

    /*
     * Communicate particles
     */

    // total up number of sent particles
    unsigned int n_remove_ptls = 0;
    for (unsigned int i = 0; i < 6; ++i)
        n_remove_ptls += n_send_ptls_face[i];
    for (unsigned int i = 0; i < 12; ++i)
        n_remove_ptls += n_send_ptls_edge[i];
    for (unsigned int i = 0; i < 8; ++i)
        n_remove_ptls += n_send_ptls_corner[i];

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
            while (new_size < max_n_recv_edge + max_n_send_edge)
                new_size = ((unsigned int)(((float)new_size)*m_resize_factor))+1;
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
            while (new_size < max_n_recv_face + max_n_send_face)
                new_size = ((unsigned int)(((float)new_size)*m_resize_factor))+1;
            m_max_send_ptls_face = new_size;
            m_face_send_buf.resize(m_max_send_ptls_face*gpu_pdata_element_size(), 6);
            }

        if (m_recv_buf.getNumElements() < (n_tot_recv_ptls + n_recv_ptls)*gpu_pdata_element_size())
            {
            unsigned int new_size =1;
            while (new_size < n_tot_recv_ptls + n_recv_ptls)
                new_size = ((unsigned int)(((float)new_size)*m_resize_factor))+1;
            m_recv_buf.resize(new_size*gpu_pdata_element_size());
            }
          

        unsigned int cpitch = m_corner_send_buf.getPitch();
        unsigned int epitch = m_edge_send_buf.getPitch();
        unsigned int fpitch = m_face_send_buf.getPitch();

        #ifdef ENABLE_MPI_CUDA
        char *corner_send_buf_handle = m_corner_send_buf.getDevicePointer();
        char *edge_send_buf_handle = m_edge_send_buf.getDevicePointer();
        char *face_send_buf_handle = m_face_send_buf.getDevicePointer();
        char *recv_buf_handle = m_recv_buf.getDevicePointer();
        #else
        char *corner_send_buf_handle = m_corner_send_buf.getHostPointer();
        char *edge_send_buf_handle = m_edge_send_buf.getHostPointer();
        char *face_send_buf_handle = m_face_send_buf.getHostPointer();
        char *recv_buf_handle = m_recv_buf.getHostPointer();
        #endif

        communicateStepTwo(dir,
                           corner_send_buf_handle,
                           edge_send_buf_handle,
                           face_send_buf_handle,
                           cpitch,
                           epitch,
                           fpitch,
                           recv_buf_handle,
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

        gpu_migrate_fill_particle_arrays(old_nparticles,
                               n_tot_recv_ptls,
                               n_remove_ptls,
                               d_remove_mask.data,
                               m_recv_buf.getDevicePointer(),
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

    /*
     * Communicate bonds
     */

    if (bdata->getNumBondsGlobal())
        {
        unsigned int n_tot_recv_bonds = 0;

        for (unsigned int dir=0; dir < 6; dir++)
            {

            if (! isCommunicating(dir) ) continue;

            unsigned int n_recv_bonds_edge[12];
            unsigned int n_recv_bonds_face[6];
            unsigned int n_recv_bonds = 0;
            for (unsigned int i = 0; i < 12; ++i)
                n_recv_bonds_edge[i] = 0;

            for (unsigned int i = 0; i < 6; ++i)
                n_recv_bonds_face[i] = 0;

            unsigned int max_n_recv_edge = 0;
            unsigned int max_n_recv_face = 0;

            // communicate size of the messages that will contain the particle data
            communicateStepOne(dir,
                               n_send_bonds_corner,
                               n_send_bonds_edge,
                               n_send_bonds_face,
                               n_recv_bonds_face,
                               n_recv_bonds_edge,
                               &n_recv_bonds,
                               true);

            unsigned int max_n_send_edge = 0;
            unsigned int max_n_send_face = 0;
            // resize buffers as necessary
            for (unsigned int i = 0; i < 12; ++i)
                {
                if (n_recv_bonds_edge[i] > max_n_recv_edge)
                    max_n_recv_edge = n_recv_bonds_edge[i];
                if (n_send_bonds_edge[i] > max_n_send_edge)
                    max_n_send_edge = n_send_bonds_edge[i];
                }


            if (max_n_recv_edge + max_n_send_edge > m_max_send_bonds_edge)
                {
                unsigned int new_size = 1;
                while (new_size < max_n_recv_edge + max_n_send_edge)
                    new_size = ((unsigned int)(((float)new_size)*m_resize_factor))+1;
                m_max_send_bonds_edge = new_size;

                m_bond_edge_send_buf.resize(m_max_send_bonds_edge, 12);
                }

            for (unsigned int i = 0; i < 6; ++i)
                {
                if (n_recv_bonds_face[i] > max_n_recv_face)
                    max_n_recv_face = n_recv_bonds_face[i];
                if (n_send_bonds_face[i] > max_n_send_face)
                    max_n_send_face = n_send_bonds_face[i];
                }

            if (max_n_recv_face + max_n_send_face > m_max_send_bonds_face)
                {
                unsigned int new_size = 1;
                while (new_size < max_n_recv_face + max_n_send_face)
                    new_size = ((unsigned int)(((float)new_size)*m_resize_factor))+1;
                m_max_send_bonds_face = new_size;
                m_bond_face_send_buf.resize(m_max_send_bonds_face*gpu_pdata_element_size(), 6);
                }

            if (m_recv_buf.getNumElements() < (n_tot_recv_bonds + n_recv_bonds))
                {
                unsigned int new_size =1;
                while (new_size < n_tot_recv_bonds + n_recv_bonds)
                    new_size = ((unsigned int)(((float)new_size)*m_resize_factor))+1;
                m_recv_buf.resize(new_size);
                }
              

            unsigned int cpitch = m_bond_corner_send_buf.getPitch();
            unsigned int epitch = m_bond_edge_send_buf.getPitch();
            unsigned int fpitch = m_bond_face_send_buf.getPitch();

            #ifdef ENABLE_MPI_CUDA
            char *corner_send_buf_handle = (char *) m_bond_corner_send_buf.getDevicePointer();
            char *edge_send_buf_handle = (char *) m_bond_edge_send_buf.getDevicePointer();
            char *face_send_buf_handle = (char *) m_bond_face_send_buf.getDevicePointer();
            char *recv_buf_handle = (char *) m_bond_recv_buf.getDevicePointer();
            #else
            char *corner_send_buf_handle = (char *) m_bond_corner_send_buf.getHostPointer();
            char *edge_send_buf_handle = (char *) m_bond_edge_send_buf.getHostPointer();
            char *face_send_buf_handle = (char *) m_bond_face_send_buf.getHostPointer();
            char *recv_buf_handle = (char *) m_bond_recv_buf.getHostPointer();
            #endif

            communicateStepTwo(dir,
                               corner_send_buf_handle,
                               edge_send_buf_handle,
                               face_send_buf_handle,
                               cpitch*sizeof(bond_element),
                               epitch*sizeof(bond_element),
                               fpitch*sizeof(bond_element),
                               recv_buf_handle,
                               n_send_bonds_corner,
                               n_send_bonds_edge,
                               n_send_bonds_face,
                               m_bond_recv_buf.getNumElements(),
                               n_tot_recv_bonds,
                               sizeof(bond_element),
                               true);

            // update buffer sizes
            for (unsigned int i = 0; i < 12; ++i)
                n_send_bonds_edge[i] += n_recv_bonds_edge[i];

            for (unsigned int i = 0; i < 6; ++i)
                n_send_bonds_face[i] += n_recv_bonds_face[i];

            n_tot_recv_bonds += n_recv_bonds;

            } // end dir loop

        bdata->unpackRemoveBonds(n_tot_recv_bonds,
                                 m_n_remove_bonds.readFlags(),
                                 m_bond_recv_buf,
                                 m_bond_remove_mask);
        } 

 
    if (m_prof)
        m_prof->pop();

    // notify ParticleData that addition / removal of particles is complete
    m_pdata->notifyParticleSort();
    }

//! Build a ghost particle list, exchange ghost particle data with neighboring processors
void CommunicatorGPU::exchangeGhosts()
    {
    if (m_prof)
        m_prof->push("exchange_ghosts");

    m_exec_conf->msg->notice(7) << "CommunicatorGPU: ghost exchange" << std::endl;
    assert(m_r_ghost < (m_pdata->getBox().getL().x));
    assert(m_r_ghost < (m_pdata->getBox().getL().y));
    assert(m_r_ghost < (m_pdata->getBox().getL().z));

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
            m_corner_ghosts_buf.resize(m_max_copy_ghosts_corner*gpu_ghost_element_size(), 8);

        if (m_edge_ghosts_buf.getPitch() < m_max_copy_ghosts_edge*gpu_ghost_element_size())
            m_edge_ghosts_buf.resize(m_max_copy_ghosts_edge*gpu_ghost_element_size(), 12);

        if (m_face_ghosts_buf.getPitch() < m_max_copy_ghosts_face*gpu_ghost_element_size())
            m_face_ghosts_buf.resize(m_max_copy_ghosts_face*gpu_ghost_element_size(), 6);

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

            ArrayHandle<unsigned int> d_ghost_idx_face(m_ghost_idx_face, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_ghost_idx_edge(m_ghost_idx_edge, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_ghost_idx_corner(m_ghost_idx_corner, access_location::device, access_mode::overwrite);

            ArrayHandle<unsigned int> d_n_local_ghosts_face(m_n_local_ghosts_face, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_n_local_ghosts_edge(m_n_local_ghosts_edge, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_n_local_ghosts_corner(m_n_local_ghosts_corner, access_location::device, access_mode::overwrite);

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
                                m_corner_ghosts_buf.getDevicePointer(),
                                m_corner_ghosts_buf.getPitch(),
                                m_edge_ghosts_buf.getDevicePointer(),
                                m_edge_ghosts_buf.getPitch(),
                                m_face_ghosts_buf.getDevicePointer(),
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
                
                m_edge_ghosts_buf.resize(m_max_copy_ghosts_edge*gpu_ghost_element_size(), 12);
                m_edge_update_buf.resize(m_max_copy_ghosts_edge*gpu_update_element_size(), 12);
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

                m_face_ghosts_buf.resize(m_max_copy_ghosts_face*gpu_ghost_element_size(), 6);
                m_face_update_buf.resize(m_max_copy_ghosts_face*gpu_update_element_size(), 6);
                }

            if (m_ghosts_recv_buf.getNumElements() < (m_n_tot_recv_ghosts_local + h_n_recv_ghosts_local.data[dir])*gpu_ghost_element_size())
                {
                unsigned int new_size =1;
                while (new_size < m_n_tot_recv_ghosts_local + h_n_recv_ghosts_local.data[dir])
                    new_size = ((unsigned int)(((float)new_size)*m_resize_factor))+1;
                m_ghosts_recv_buf.resize(new_size*gpu_ghost_element_size());
                }

            if (m_update_recv_buf.getNumElements() < (m_n_tot_recv_ghosts_local + h_n_recv_ghosts_local.data[dir])*gpu_update_element_size())
                {
                unsigned int new_size =1;
                while (new_size < m_n_tot_recv_ghosts_local + h_n_recv_ghosts_local.data[dir])
                    new_size = ((unsigned int)(((float)new_size)*m_resize_factor))+1;
                m_update_recv_buf.resize(new_size*gpu_update_element_size());
                }

            unsigned int cpitch = m_corner_ghosts_buf.getPitch();
            unsigned int epitch = m_edge_ghosts_buf.getPitch();
            unsigned int fpitch = m_face_ghosts_buf.getPitch();

            #ifdef ENABLE_MPI_CUDA
            char *corner_ghosts_buf_handle = m_corner_ghosts_buf.getDevicePointer();
            char *edge_ghosts_buf_handle = m_edge_ghosts_buf.getDevicePointer();
            char *face_ghosts_buf_handle = m_face_ghosts_buf.getDevicePointer();
            char *ghosts_recv_buf_handle = m_ghosts_recv_buf.getDevicePointer();
            #else
            char *corner_ghosts_buf_handle = m_corner_ghosts_buf.getHostPointer();
            char *edge_ghosts_buf_handle = m_edge_ghosts_buf.getHostPointer();
            char *face_ghosts_buf_handle = m_face_ghosts_buf.getHostPointer();
            char *ghosts_recv_buf_handle = m_ghosts_recv_buf.getHostPointer();
            #endif

            communicateStepTwo(dir,
                               corner_ghosts_buf_handle,
                               edge_ghosts_buf_handle,
                               face_ghosts_buf_handle,
                               cpitch,
                               epitch,
                               fpitch,
                               ghosts_recv_buf_handle,
                               n_copy_ghosts_corner,
                               n_copy_ghosts_edge,
                               n_copy_ghosts_face,
                               m_ghosts_recv_buf.getNumElements(),
                               m_n_tot_recv_ghosts_local, 
                               gpu_ghost_element_size(),
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

    // resize plan array if necessary
    if (m_ghost_plan.getNumElements() < m_n_tot_recv_ghosts)
        {
        unsigned int new_size = m_ghost_plan.getNumElements();
        while (new_size < m_n_tot_recv_ghosts)
            new_size = ((unsigned int)(((float)new_size)*m_resize_factor))+1;
        m_ghost_plan.resize(new_size);
        }

    // update number of ghost particles
    m_pdata->addGhostParticles(m_n_tot_recv_ghosts);

        {
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::readwrite);
        ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::readwrite);
        ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::readwrite);

        ArrayHandle<unsigned int> d_ghost_plan(m_ghost_plan, access_location::device, access_mode::overwrite);

        ArrayHandle<unsigned int> d_n_local_ghosts_face(m_n_local_ghosts_face, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_n_local_ghosts_edge(m_n_local_ghosts_edge, access_location::device, access_mode::read);

        ArrayHandle<unsigned int> d_n_recv_ghosts_face(m_n_recv_ghosts_face, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_n_recv_ghosts_edge(m_n_recv_ghosts_edge, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_n_recv_ghosts_local(m_n_recv_ghosts_local, access_location::device, access_mode::read);

        gpu_exchange_ghosts_unpack(m_pdata->getN(),
                                     m_n_tot_recv_ghosts,
                                     d_n_local_ghosts_face.data,
                                     d_n_local_ghosts_edge.data,
                                     m_n_tot_recv_ghosts_local,
                                     d_n_recv_ghosts_local.data,
                                     d_n_recv_ghosts_face.data,
                                     d_n_recv_ghosts_edge.data,
                                     m_face_ghosts_buf.getDevicePointer(),
                                     m_face_ghosts_buf.getPitch(),
                                     m_edge_ghosts_buf.getDevicePointer(),
                                     m_edge_ghosts_buf.getPitch(),
                                     m_ghosts_recv_buf.getDevicePointer(),
                                     d_pos.data,
                                     d_charge.data,
                                     d_diameter.data,
                                     d_tag.data,
                                     d_rtag.data,
                                     d_ghost_plan.data,
                                     m_pdata->getGlobalBox());

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    if (m_prof)
        m_prof->pop();

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
 
    // calculate total message sizes
    for (unsigned int corner_i = 0; corner_i < 8; ++corner_i)
        {
        bool sent = false;
        unsigned int plan = corner_plan_lookup[corner_i];

        // only send corner particle through face if face touches corner
        if (!((face_plan_lookup[cur_face] & plan) == face_plan_lookup[cur_face])) continue;

        nptl = n_send_ptls_corner[corner_i];

#ifndef MPI3
        MPI_Isend(&nptl, 1, MPI_INT, send_neighbor, tag, m_mpi_comm, &send_req);
        MPI_Irecv(&m_remote_send_corner[8*cur_face+corner_i], 1, MPI_INT, recv_neighbor, tag, m_mpi_comm, &recv_req);
        MPI_Wait(&send_req,&send_status);
        MPI_Wait(&recv_req,&recv_status);
        tag++;
#endif

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

#ifndef MPI3
        MPI_Isend(&nptl, 1, MPI_INT, send_neighbor, tag, m_mpi_comm, &send_req);
        MPI_Irecv(&m_remote_send_edge[12*cur_face+edge_i], 1, MPI_INT, recv_neighbor, tag, m_mpi_comm, &recv_req);
        MPI_Wait(&send_req,&send_status);
        MPI_Wait(&recv_req,&recv_status);
        tag++;
#endif

        // do not place particle in a buffer where it would be sent back to ourselves
        unsigned int next_face = cur_face + ((cur_face % 2) ? 1 : 2);

        for (unsigned int face_j = next_face; face_j < 6; ++face_j)
            if ((face_plan_lookup[face_j] & plan) == face_plan_lookup[face_j])
                {
                // send an edge particle to a face send buffer in the neighboring box
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

        nptl = n_send_ptls_face[face_i];

#ifndef MPI3
        MPI_Isend(&nptl, 1, MPI_INT, send_neighbor, tag, m_mpi_comm, &send_req);
        MPI_Irecv(&m_remote_send_face[6*cur_face+face_i], 1, MPI_INT, recv_neighbor, tag, m_mpi_comm, &recv_req);
        MPI_Wait(&send_req,&send_status);
        MPI_Wait(&recv_req,&recv_status);
        tag++;
#endif
        n_remote_recv_ptls_local += nptl;
        }

    MPI_Isend(n_remote_recv_ptls_edge, sizeof(unsigned int)*12, MPI_BYTE, send_neighbor, tag, m_mpi_comm, & reqs[0]);
    MPI_Isend(n_remote_recv_ptls_face, sizeof(unsigned int)*6, MPI_BYTE, send_neighbor, tag+1, m_mpi_comm, &reqs[1]);
    MPI_Isend(&n_remote_recv_ptls_local, sizeof(unsigned int), MPI_BYTE, send_neighbor, tag+2, m_mpi_comm, &reqs[2]);

    MPI_Irecv(n_recv_ptls_edge, 12*sizeof(unsigned int), MPI_BYTE, recv_neighbor, tag+0, m_mpi_comm, &reqs[3]);
    MPI_Irecv(n_recv_ptls_face, 6*sizeof(unsigned int), MPI_BYTE, recv_neighbor, tag+1, m_mpi_comm, & reqs[4]);
    MPI_Irecv(n_recv_ptls_local, sizeof(unsigned int), MPI_BYTE, recv_neighbor, tag+2, m_mpi_comm, & reqs[5]);

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
    MPI_Request send_req, recv_req;
    MPI_Status send_status, recv_status;
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
        // indicates whether buffer has been sent to current neighbor
        bool sent = false;
        unsigned int plan = corner_plan_lookup[corner_i];

        // only send corner particle through face if face touches corner
        if (! ((face_plan_lookup[cur_face] & plan) == face_plan_lookup[cur_face])) continue;

        nptl = n_send_ptls_corner[corner_i];
        data = corner_send_buf+corner_i*cpitch;

#ifndef MPI3
        recv_nptl = m_remote_send_corner[cur_face*8+corner_i];
#endif

        for (unsigned int edge_j = 0; edge_j < 12; ++edge_j)
            if ((edge_plan_lookup[edge_j] & plan) == edge_plan_lookup[edge_j])
                {
                // if this edge buffer is or has already been emptied in this or previous communication steps, don't add to it
                bool active = true;
                for (unsigned int face_k = 0; face_k < 6; ++face_k)
                    if (face_k <= cur_face && (edge_plan_lookup[edge_j] & face_plan_lookup[face_k])) active = false;
                if (! active) continue;

#ifdef MPI3
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
                          &send_req);
                if (recv)
                    MPI_Irecv(edge_send_buf + edge_j * epitch + offset,
                          recv_nptl*element_size,
                          MPI_BYTE,
                          recv_neighbor,
                          tag,
                          m_mpi_comm,
                          &recv_req);
                if (send) MPI_Wait(&send_req,&send_status);
                if (recv) MPI_Wait(&recv_req,&recv_status);
                tag++;
                recv_eoffset[edge_j] += recv_nptl;
#endif
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
                          &send_req);
                if (recv)
                    MPI_Irecv(face_send_buf + face_j * fpitch + offset,
                          recv_nptl*element_size,
                          MPI_BYTE,
                          recv_neighbor,
                          tag,
                          m_mpi_comm,
                          &recv_req);
                if (send) MPI_Wait(&send_req,&send_status);
                if (recv) MPI_Wait(&recv_req,&recv_status);
                tag++;
                recv_foffset[face_j] += recv_nptl;
#endif
                sent = true;
                break;
                }

        if (sent) continue;

        if (plan & face_plan_lookup[cur_face])
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
                      &send_req);
            if (recv)
                MPI_Irecv(local_recv_buf + offset,
                      recv_nptl*element_size,
                      MPI_BYTE,
                      recv_neighbor,
                      tag,
                      m_mpi_comm,
                      &recv_req);
            if (send) MPI_Wait(&send_req,&send_status);
            if (recv) MPI_Wait(&recv_req,&recv_status);
            tag++;
            recv_loffset += recv_nptl;
#endif
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

#ifndef MPI3
        recv_nptl = m_remote_send_edge[cur_face*12+edge_i];
#endif

        // do not place particle in a buffer where it would be sent back to ourselves
        unsigned int next_face = cur_face + ((cur_face % 2) ? 1 : 2);

        for (unsigned int face_j = next_face; face_j < 6; ++face_j)
            if ((face_plan_lookup[face_j] & plan) == face_plan_lookup[face_j])
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
                          &send_req);
                if (recv)
                    MPI_Irecv(face_send_buf + face_j * fpitch + offset,
                          recv_nptl*element_size,
                          MPI_BYTE,
                          recv_neighbor,
                          tag,
                          m_mpi_comm,
                          &recv_req);
                if (send) MPI_Wait(&send_req,&send_status);
                if (recv) MPI_Wait(&recv_req,&recv_status);
                tag++;
                recv_foffset[face_j] += recv_nptl;
#endif
                sent = true;
                break;
                }

        if (unique_destination || sent) continue;

        if (plan & face_plan_lookup[cur_face])
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
                      &send_req);
            if (recv)
                MPI_Irecv(local_recv_buf + offset,
                      recv_nptl*element_size,
                      MPI_BYTE,
                      recv_neighbor,
                      tag,
                      m_mpi_comm,
                      &recv_req);
            if (send) MPI_Wait(&send_req,&send_status);
            if (recv) MPI_Wait(&recv_req,&recv_status);
            tag++;
            recv_loffset += recv_nptl;
#endif
            }
        } 

    for (unsigned int face_i = 0; face_i < 6; ++face_i)
        {
        unsigned int plan = face_plan_lookup[face_i];

        // only send through face if this is the current sending direction
        if (! ((face_plan_lookup[cur_face] & plan) == face_plan_lookup[cur_face])) continue;

        nptl = n_send_ptls_face[face_i];
        data = face_send_buf+face_i*fpitch;

#ifndef MPI3
        recv_nptl = m_remote_send_face[cur_face*6+face_i];
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
                  &send_req);
        if (recv)
            MPI_Irecv(local_recv_buf + offset,
                  recv_nptl*element_size,
                  MPI_BYTE,
                  recv_neighbor,
                  tag,
                  m_mpi_comm,
                  &recv_req);
        if (send) MPI_Wait(&send_req,&send_status);
        if (recv) MPI_Wait(&recv_req,&recv_status);
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
#endif
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
