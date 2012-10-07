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

//! Main routine of ghost update worker thread
void ghost_gpu_thread::operator()(WorkQueue<ghost_gpu_thread_params>& queue, boost::barrier& barrier)
    {
    bool done = false;
    while (! done)
        {
        try
            {
            ghost_gpu_thread_params &params = queue.wait_and_pop();
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
    m_exec_conf = params.exec_conf;

    unsigned int num_tot_recv_ghosts = 0; // total number of ghosts received

    MPI_Request reqs[4];
    MPI_Status status[4];

    for (unsigned int dir = 0; dir < 6; dir ++)
        {

        if (! params.is_communicating[dir]) continue;

        unsigned int offset = (dir % 2) ? params.num_copy_ghosts[dir-1] : 0;

        // Pack send data for direction dir and dir+1 (opposite) simultaneously.
        // We assume that they are independent, i.e. a send in direction dir+1
        // does not contain any ghosts received from that direction previously.
        if ((dir % 2) == 0)
            {
            gpu_copy_ghosts(params.num_copy_ghosts[dir],
                            params.num_copy_ghosts[dir+1],
                            params.d_pos_data, 
                            params.d_copy_ghosts[dir],
                            params.d_copy_ghosts[dir+1],
                            params.d_pos_copybuf,
                            params.d_pos_copybuf + params.num_copy_ghosts[dir],
                            dir,
                            params.is_at_boundary,
                            params.box);

            CHECK_CUDA_ERROR();
            }

        unsigned int send_neighbor = params.decomposition->getNeighborRank(dir);

        // we receive from the direction opposite to the one we send to
        unsigned int recv_neighbor;
        if (dir % 2 == 0)
            recv_neighbor = params.decomposition->getNeighborRank(dir+1);
        else
            recv_neighbor = params.decomposition->getNeighborRank(dir-1);

        unsigned int start_idx;

        start_idx = params.N + num_tot_recv_ghosts;

        num_tot_recv_ghosts += params.num_recv_ghosts[dir];

        unsigned int shift = (dir % 2) ? 2 : 0;

        // exchange particle data, write directly to the particle data arrays
        MPI_Isend(params.d_pos_copybuf + offset, params.num_copy_ghosts[dir]*sizeof(Scalar4), MPI_BYTE, send_neighbor, dir, m_exec_conf->getMPICommunicator(), &reqs[0+shift]);
        MPI_Irecv(params.d_pos_data + start_idx, params.num_recv_ghosts[dir]*sizeof(Scalar4), MPI_BYTE, recv_neighbor, dir, m_exec_conf->getMPICommunicator(), &reqs[1+shift]);

        if (dir %2)
            MPI_Waitall(4, reqs, status);

        } // end dir loop

    } 

//! Constructor
CommunicatorGPU::CommunicatorGPU(boost::shared_ptr<SystemDefinition> sysdef,
                                 boost::shared_ptr<DomainDecomposition> decomposition)
    : Communicator(sysdef, decomposition), m_remove_mask(m_exec_conf),
      m_pos_stage(m_exec_conf),
      m_vel_stage(m_exec_conf),
      m_accel_stage(m_exec_conf),
      m_image_stage(m_exec_conf),
      m_charge_stage(m_exec_conf),
      m_diameter_stage(m_exec_conf),
      m_body_stage(m_exec_conf),
      m_orientation_stage(m_exec_conf),
      m_tag_stage(m_exec_conf),
      m_barrier(2)
    { 
    m_exec_conf->msg->notice(5) << "Constructing CommunicatorGPU" << std::endl;
    // allocate temporary GPU buffers
    gpu_allocate_tmp_storage();

    // create a worker thread for ghost updates
    m_worker_thread = boost::thread(ghost_gpu_thread(), boost::ref(m_work_queue), boost::ref(m_barrier));
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

#ifdef ENABLE_MPI_CUDA
//! Start ghosts communication
/*! This is the multi-threaded version.
 */
void CommunicatorGPU::startGhostsUpdate(unsigned int timestep)
    {
    if (timestep < m_next_ghost_update)
        return;

    // fill thread parameters
    for (unsigned int i = 0; i < 6; ++i)
        {
        m_copy_ghosts_data[i] = m_copy_ghosts[i].lock(access_location::device, access_mode::read);
        m_communication_dir[i] = isCommunicating(i);
        }

    // lock positions array against writing
    Scalar4 *d_pos_data = m_pdata->getPositions().lock(access_location::device, access_mode::readwrite);
    Scalar4 *d_pos_copybuf_data = m_pos_copybuf.lock(access_location::device, access_mode::overwrite);

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
    }

//! Finish ghost communication
void CommunicatorGPU::finishGhostsUpdate(unsigned int timestep)
    {
    if (timestep < m_next_ghost_update)
        return;

    // wait for worker thread to finish task
    m_barrier.wait();

    // release locked arrays
    for (unsigned int i = 0; i < 6; ++i)
        m_copy_ghosts[i].unlock();

    m_pdata->getPositions().unlock();
    m_pos_copybuf.unlock();
    }
#endif

//! Transfer particles between neighboring domains
void CommunicatorGPU::migrateAtoms()
    {
    if (m_prof)
        m_prof->push("migrate_atoms");

        {
        // Reset reverse lookup tags of old ghost atoms
        ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::readwrite);
        ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);

        gpu_reset_rtags(m_pdata->getNGhosts(),
                        d_tag.data + m_pdata->getN(),
                        d_rtag.data);

        CHECK_CUDA_ERROR();
        }

    // reset ghost particle number
    m_pdata->removeAllGhostParticles();

    m_remove_mask.clear();

    for (unsigned int dir=0; dir < 6; dir++)
        {
        unsigned int n_send_ptls;

        if (! isCommunicating(dir) ) continue;

        if (m_prof)
            m_prof->push("remove ptls");

        // Reallocate send buffers
        unsigned int max_n = m_pdata->getPositions().getNumElements();
        if (m_pos_stage.size() != max_n);
            {
            m_pos_stage.resize(max_n);
            m_vel_stage.resize(max_n);
            m_accel_stage.resize(max_n);
            m_charge_stage.resize(max_n);
            m_diameter_stage.resize(max_n);
            m_image_stage.resize(max_n);
            m_body_stage.resize(max_n);
            m_orientation_stage.resize(max_n);
            m_tag_stage.resize(max_n);

            // resize mask and set newly allocated flags to zero
            m_remove_mask.resize(max_n);
            }


            {
            // remove all particles from our domain that are going to be sent in the current direction

            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
            ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::read);
            ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::read);
            ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::read);
            ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::read);
            ArrayHandle<int3> d_image(m_pdata->getImages(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_body(m_pdata->getBodies(), access_location::device, access_mode::read);
            ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);

            ArrayHandle<Scalar4> d_pos_stage(m_pos_stage, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar4> d_vel_stage(m_vel_stage, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar3> d_accel_stage(m_accel_stage, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar> d_charge_stage(m_charge_stage, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar> d_diameter_stage(m_diameter_stage, access_location::device, access_mode::overwrite);
            ArrayHandle<int3> d_image_stage(m_image_stage, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_body_stage(m_body_stage, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar4> d_orientation_stage(m_orientation_stage, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_tag_stage(m_tag_stage, access_location::device, access_mode::overwrite);

            ArrayHandle<unsigned char> d_remove_mask(m_remove_mask, access_location::device, access_mode::readwrite);

            // Stage particle data for sending, wrap particles
            gpu_migrate_select_particles(m_pdata->getN(),
                                   n_send_ptls,
                                   d_remove_mask.data,                                
                                   d_pos.data,
                                   d_pos_stage.data,
                                   d_vel.data,
                                   d_vel_stage.data,
                                   d_accel.data,
                                   d_accel_stage.data,
                                   d_image.data,
                                   d_image_stage.data,
                                   d_charge.data,
                                   d_charge_stage.data,
                                   d_diameter.data,
                                   d_diameter_stage.data,
                                   d_body.data,
                                   d_body_stage.data,
                                   d_orientation.data,
                                   d_orientation_stage.data,
                                   d_tag.data,
                                   d_tag_stage.data,
                                   m_pdata->getBox(),
                                   m_pdata->getGlobalBox(),
                                   dir,
                                   m_is_at_boundary);
            CHECK_CUDA_ERROR();
            
            }

        if (m_prof)
	        m_prof->pop();

        unsigned int send_neighbor = m_decomposition->getNeighborRank(dir);

        // we receive from the direction opposite to the one we send to
        unsigned int recv_neighbor;
        if (dir % 2 == 0)
            recv_neighbor = m_decomposition->getNeighborRank(dir+1);
        else
            recv_neighbor = m_decomposition->getNeighborRank(dir-1);

        if (m_prof)
            m_prof->push("MPI send/recv");

        unsigned int n_recv_ptls;
        // communicate size of the message that will contain the particle data
        MPI_Request reqs[20];
        MPI_Status status[20];
        MPI_Isend(&n_send_ptls, sizeof(unsigned int), MPI_BYTE, send_neighbor, 0, m_mpi_comm, & reqs[0]);
        MPI_Irecv(&n_recv_ptls, sizeof(unsigned int), MPI_BYTE, recv_neighbor, 0, m_mpi_comm, & reqs[1]);
        MPI_Waitall(2, reqs, status);

        // start index for atoms to be added
        unsigned int add_idx = m_pdata->getN();

        // allocate memory for particles that will be received
        m_pdata->addParticles(n_recv_ptls);

#ifdef ENABLE_MPI_CUDA
            {
            ArrayHandle<Scalar4> d_pos_stage(m_pos_stage, access_location::device, access_mode::read);
            ArrayHandle<Scalar4> d_vel_stage(m_vel_stage, access_location::device, access_mode::read);
            ArrayHandle<Scalar3> d_accel_stage(m_accel_stage, access_location::device, access_mode::read);
            ArrayHandle<Scalar> d_charge_stage(m_charge_stage, access_location::device, access_mode::read);
            ArrayHandle<Scalar> d_diameter_stage(m_diameter_stage, access_location::device, access_mode::read);
            ArrayHandle<int3> d_image_stage(m_image_stage, access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_tag_stage(m_tag_stage, access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_body_stage(m_body_stage, access_location::device, access_mode::read);
            ArrayHandle<Scalar4> d_orientation_stage(m_orientation_stage, access_location::device, access_mode::read);

            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::readwrite);
            ArrayHandle<int3> d_image(m_pdata->getImages(), access_location::device, access_mode::readwrite);
            ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::readwrite);
            ArrayHandle<unsigned> d_body(m_pdata->getBodies(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::readwrite);

            // exchange actual particle data
            MPI_Isend(d_pos_stage.data, n_send_ptls*sizeof(Scalar4), MPI_BYTE, send_neighbor, 1, m_mpi_comm, &reqs[2]);
            MPI_Irecv(d_pos.data+add_idx, n_recv_ptls*sizeof(Scalar4), MPI_BYTE, recv_neighbor, 1, m_mpi_comm, &reqs[3]);

            MPI_Isend(d_vel_stage.data, n_send_ptls*sizeof(Scalar4), MPI_BYTE, send_neighbor, 2, m_mpi_comm, &reqs[4]);
            MPI_Irecv(d_vel.data+add_idx, n_recv_ptls*sizeof(Scalar4), MPI_BYTE, recv_neighbor, 2, m_mpi_comm, &reqs[5]);

            MPI_Isend(d_accel_stage.data, n_send_ptls*sizeof(Scalar3), MPI_BYTE, send_neighbor, 3, m_mpi_comm, &reqs[6]);
            MPI_Irecv(d_accel.data+add_idx, n_recv_ptls*sizeof(Scalar3), MPI_BYTE, recv_neighbor, 3, m_mpi_comm, &reqs[7]);

            MPI_Isend(d_image_stage.data, n_send_ptls*sizeof(int3), MPI_BYTE, send_neighbor, 4, m_mpi_comm, &reqs[8]);
            MPI_Irecv(d_image.data+add_idx, n_recv_ptls*sizeof(int3), MPI_BYTE, recv_neighbor, 4, m_mpi_comm, &reqs[9]);

            MPI_Isend(d_charge_stage.data, n_send_ptls*sizeof(Scalar), MPI_BYTE, send_neighbor, 5, m_mpi_comm, &reqs[10]);
            MPI_Irecv(d_charge.data+add_idx, n_recv_ptls*sizeof(Scalar), MPI_BYTE, recv_neighbor, 5, m_mpi_comm, &reqs[11]);

            MPI_Isend(d_diameter_stage.data, n_send_ptls*sizeof(Scalar), MPI_BYTE, send_neighbor, 6, m_mpi_comm, &reqs[12]);
            MPI_Irecv(d_diameter.data+add_idx, n_recv_ptls*sizeof(Scalar), MPI_BYTE, recv_neighbor, 6, m_mpi_comm, &reqs[13]);

            MPI_Isend(d_tag_stage.data, n_send_ptls*sizeof(unsigned int), MPI_BYTE, send_neighbor, 7, m_mpi_comm, &reqs[14]);
            MPI_Irecv(d_tag.data+add_idx, n_recv_ptls*sizeof(unsigned int), MPI_BYTE, recv_neighbor, 7, m_mpi_comm, &reqs[15]);

            MPI_Isend(d_body_stage.data, n_send_ptls*sizeof(unsigned int), MPI_BYTE, send_neighbor, 8, m_mpi_comm, &reqs[16]);
            MPI_Irecv(d_body.data+add_idx, n_recv_ptls*sizeof(unsigned int), MPI_BYTE, recv_neighbor, 8, m_mpi_comm, &reqs[17]);

            MPI_Isend(d_orientation_stage.data, n_send_ptls*sizeof(Scalar4), MPI_BYTE, send_neighbor, 9, m_mpi_comm, &reqs[18]);
            MPI_Irecv(d_orientation.data+add_idx, n_recv_ptls*sizeof(Scalar4), MPI_BYTE, recv_neighbor, 9, m_mpi_comm, &reqs[19]);

            MPI_Waitall(18,reqs+2, status+2);
            }

#else
            {
            ArrayHandle<Scalar4> h_pos_stage(m_pos_stage, access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_vel_stage(m_vel_stage, access_location::host, access_mode::read);
            ArrayHandle<Scalar3> h_accel_stage(m_accel_stage, access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_charge_stage(m_charge_stage, access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_diameter_stage(m_diameter_stage, access_location::host, access_mode::read);
            ArrayHandle<int3> h_image_stage(m_image_stage, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_tag_stage(m_tag_stage, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_body_stage(m_body_stage, access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_orientation_stage(m_orientation_stage, access_location::host, access_mode::read);

            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::readwrite);
            ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);
            ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::readwrite);
            ArrayHandle<unsigned> h_body(m_pdata->getBodies(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);

            MPI_Isend(h_pos_stage.data, n_send_ptls*sizeof(Scalar4), MPI_BYTE, send_neighbor, 1, m_mpi_comm, &reqs[2]);
            MPI_Irecv(h_pos.data+add_idx, n_recv_ptls*sizeof(Scalar4), MPI_BYTE, recv_neighbor, 1, m_mpi_comm, &reqs[3]);

            MPI_Isend(h_vel_stage.data, n_send_ptls*sizeof(Scalar4), MPI_BYTE, send_neighbor, 2, m_mpi_comm, &reqs[4]);
            MPI_Irecv(h_vel.data+add_idx, n_recv_ptls*sizeof(Scalar4), MPI_BYTE, recv_neighbor, 2, m_mpi_comm, &reqs[5]);

            MPI_Isend(h_accel_stage.data, n_send_ptls*sizeof(Scalar3), MPI_BYTE, send_neighbor, 3, m_mpi_comm, &reqs[6]);
            MPI_Irecv(h_accel.data+add_idx, n_recv_ptls*sizeof(Scalar3), MPI_BYTE, recv_neighbor, 3, m_mpi_comm, &reqs[7]);

            MPI_Isend(h_image_stage.data, n_send_ptls*sizeof(int3), MPI_BYTE, send_neighbor, 4, m_mpi_comm, &reqs[8]);
            MPI_Irecv(h_image.data+add_idx, n_recv_ptls*sizeof(int3), MPI_BYTE, recv_neighbor, 4, m_mpi_comm, &reqs[9]);

            MPI_Isend(h_charge_stage.data, n_send_ptls*sizeof(Scalar), MPI_BYTE, send_neighbor, 5, m_mpi_comm, &reqs[10]);
            MPI_Irecv(h_charge.data+add_idx, n_recv_ptls*sizeof(Scalar), MPI_BYTE, recv_neighbor, 5, m_mpi_comm, &reqs[11]);

            MPI_Isend(h_diameter_stage.data, n_send_ptls*sizeof(Scalar), MPI_BYTE, send_neighbor, 6, m_mpi_comm, &reqs[12]);
            MPI_Irecv(h_diameter.data+add_idx, n_recv_ptls*sizeof(Scalar), MPI_BYTE, recv_neighbor, 6, m_mpi_comm, &reqs[13]);

            MPI_Isend(h_tag_stage.data, n_send_ptls*sizeof(unsigned int), MPI_BYTE, send_neighbor, 7, m_mpi_comm, &reqs[14]);
            MPI_Irecv(h_tag.data+add_idx, n_recv_ptls*sizeof(unsigned int), MPI_BYTE, recv_neighbor, 7, m_mpi_comm, &reqs[15]);

            MPI_Isend(h_body_stage.data, n_send_ptls*sizeof(unsigned int), MPI_BYTE, send_neighbor, 8, m_mpi_comm, &reqs[16]);
            MPI_Irecv(h_body.data+add_idx, n_recv_ptls*sizeof(unsigned int), MPI_BYTE, recv_neighbor, 8, m_mpi_comm, &reqs[17]);

            MPI_Isend(h_orientation_stage.data, n_send_ptls*sizeof(Scalar4), MPI_BYTE, send_neighbor, 9, m_mpi_comm, &reqs[18]);
            MPI_Irecv(h_orientation.data+add_idx, n_recv_ptls*sizeof(Scalar4), MPI_BYTE, recv_neighbor, 9, m_mpi_comm, &reqs[19]);

            MPI_Waitall(18,reqs+2, status+2);
            }
#endif

        if (m_prof)
            m_prof->pop();

        } // end dir loop


    unsigned int n_remove_ptls;

    // Reallocate particle data buffers
    // it is important to use the actual size of the arrays as arguments,
    // which can be larger than the particle number
    unsigned int max_n = m_pdata->getPositions().getNumElements();
    if (m_pos_stage.size() != max_n);
        {
        m_pos_stage.resize(max_n);
        m_vel_stage.resize(max_n);
        m_accel_stage.resize(max_n);
        m_charge_stage.resize(max_n);
        m_diameter_stage.resize(max_n);
        m_image_stage.resize(max_n);
        m_body_stage.resize(max_n);
        m_orientation_stage.resize(max_n);
        m_tag_stage.resize(max_n);

        // resize mask and set newly allocated flags to zero
        m_remove_mask.resize(max_n);
        }

        {
        // reset rtag of deleted particles
        ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::readwrite);
        ArrayHandle<unsigned char> d_remove_mask(m_remove_mask, access_location::device, access_mode::read);
        gpu_reset_rtags_by_mask(m_pdata->getN(),
                               d_remove_mask.data,
                               d_tag.data,
                               d_rtag.data);
        CHECK_CUDA_ERROR();
        }


        {
        // Final array compaction
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::read);
        ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::read);
        ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::read);
        ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::read);
        ArrayHandle<int3> d_image(m_pdata->getImages(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_body(m_pdata->getBodies(), access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);

        ArrayHandle<Scalar4> d_pos_stage(m_pos_stage, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_vel_stage(m_vel_stage, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar3> d_accel_stage(m_accel_stage, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar> d_charge_stage(m_charge_stage, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar> d_diameter_stage(m_diameter_stage, access_location::device, access_mode::overwrite);
        ArrayHandle<int3> d_image_stage(m_image_stage, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_body_stage(m_body_stage, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_orientation_stage(m_orientation_stage, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_tag_stage(m_tag_stage, access_location::device, access_mode::overwrite);

        ArrayHandle<unsigned char> d_remove_mask(m_remove_mask, access_location::device, access_mode::read);

        gpu_migrate_compact_particles(m_pdata->getN(),
                               d_remove_mask.data,
                               n_remove_ptls,
                               d_pos.data,
                               d_pos_stage.data,
                               d_vel.data,
                               d_vel_stage.data,
                               d_accel.data,
                               d_accel_stage.data,
                               d_image.data,
                               d_image_stage.data,
                               d_charge.data,
                               d_charge_stage.data,
                               d_diameter.data,
                               d_diameter_stage.data,
                               d_body.data,
                               d_body_stage.data,
                               d_orientation.data,
                               d_orientation_stage.data,
                               d_tag.data,
                               d_tag_stage.data);
        CHECK_CUDA_ERROR();

        }
   
    // Swap temporary arrays with particle data arrays
    m_pdata->getPositions().swap(m_pos_stage);
    m_pdata->getVelocities().swap(m_vel_stage);
    m_pdata->getAccelerations().swap(m_accel_stage);
    m_pdata->getImages().swap(m_image_stage);
    m_pdata->getCharges().swap(m_charge_stage);
    m_pdata->getDiameters().swap(m_diameter_stage);
    m_pdata->getBodies().swap(m_body_stage);
    m_pdata->getOrientationArray().swap(m_orientation_stage);
    m_pdata->getTags().swap(m_tag_stage);

    m_pdata->removeParticles(n_remove_ptls);

        {
        // update rtag information
        ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::readwrite);
        gpu_update_rtag(m_pdata->getN(),0, d_tag.data, d_rtag.data);
        CHECK_CUDA_ERROR();
        }

 
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
                            m_pdata->getGlobalBox());

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
