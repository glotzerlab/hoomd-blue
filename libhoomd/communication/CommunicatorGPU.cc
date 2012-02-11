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

#include <boost/mpi.hpp>

//! Define some of our types as fixed-size MPI datatypes for performance optimization
BOOST_IS_MPI_DATATYPE(Scalar4)
BOOST_IS_MPI_DATATYPE(Scalar3)
BOOST_IS_MPI_DATATYPE(uint3)
BOOST_IS_MPI_DATATYPE(int3)

BOOST_CLASS_TRACKING(Scalar4,track_never)

//! Constructor
CommunicatorGPU::CommunicatorGPU(boost::shared_ptr<SystemDefinition> sysdef,
                                 boost::shared_ptr<boost::mpi::communicator> mpi_comm,
                                 std::vector<unsigned int> neighbor_rank,
                                 int3 dim,
                                 const BoxDim& global_box)
    : Communicator(sysdef, mpi_comm, neighbor_rank, dim, global_box)
    {
    // initialize send buffer size with size of particle data element on the GPU
    setPackedSize(gpu_pdata_element_size());

    // allocate temporary GPU buffers
    gpu_allocate_tmp_storage();
    }

//! Destructor
CommunicatorGPU::~CommunicatorGPU()
    {
    gpu_deallocate_tmp_storage();
    }

//! Transfer particles between neighboring domains
void CommunicatorGPU::migrateAtoms()
    {
    if (m_prof)
        m_prof->push("migrate_atoms");

    if (!m_is_allocated)
        allocate();

    unsigned int recv_buf_size[6]; // per-direction size of receive buffer

    for (unsigned int dir=0; dir < 6; dir++)
        {
        char *d_send_buf_end;
        unsigned int n_send_ptls;

        if (getDimension(dir/2) == 1) continue;

        if (m_prof)
            m_prof->push("remove ptls");

        {
        // first remove all particles from our domain that are going to be sent in the current direction

        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::readwrite);
        ArrayHandle<int3> d_image(m_pdata->getImages(), access_location::device, access_mode::readwrite);
        ArrayHandle<unsigned int> d_body(m_pdata->getBodies(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::readwrite);
        ArrayHandle<unsigned int> d_global_tag(m_pdata->getGlobalTags(), access_location::device, access_mode::readwrite);

        /* Reorder particles.
           Particles that stay in our domain come first, followed by the particles that are sent to a
           neighboring processor.
         */
        gpu_migrate_select_particles(m_pdata->getN(),
                               n_send_ptls,
                               d_pos.data,
                               d_vel.data,
                               d_accel.data,
                               d_image.data,
                               d_charge.data,
                               d_diameter.data,
                               d_body.data,
                               d_orientation.data,
                               d_global_tag.data,
                               m_pdata->getBoxGPU(),
                               dir);

        m_pdata->removeParticles(n_send_ptls);

        if (m_prof)
            m_prof->pop();

        }

        // scan all atom positions and fill the send buffers with those that have left the domain boundaries
        if (m_prof)
             m_prof->push("pack");

        // Check if send buffer is large enough and resize if necessary
        if (n_send_ptls*m_packed_size > m_sendbuf[dir].getNumElements())
            {
            unsigned int new_size = m_sendbuf[dir].getNumElements();
            while (new_size < n_send_ptls * m_packed_size) new_size *= 2;
            m_sendbuf[dir].resize(new_size);
            }

        unsigned int send_buf_size;

            {
            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::readwrite);
            ArrayHandle<int3> d_image(m_pdata->getImages(), access_location::device, access_mode::readwrite);
            ArrayHandle<unsigned int> d_body(m_pdata->getBodies(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::readwrite);
            ArrayHandle<unsigned int> d_global_tag(m_pdata->getGlobalTags(), access_location::device, access_mode::readwrite);

            ArrayHandle<char> d_sendbuf(m_sendbuf[dir], access_location::device, access_mode::overwrite);

            // the particles that are going to be sent have been moved to the end of the particle data
            unsigned int send_begin = m_pdata->getN();

            // pack send buf
            gpu_migrate_pack_send_buffer(n_send_ptls,
                                         d_pos.data + send_begin,
                                         d_vel.data + send_begin,
                                         d_accel.data + send_begin,
                                         d_image.data + send_begin,
                                         d_charge.data + send_begin,
                                         d_diameter.data + send_begin,
                                         d_body.data + send_begin,
                                         d_orientation.data + send_begin,
                                         d_global_tag.data + send_begin,
                                         d_sendbuf.data,
                                         d_send_buf_end);

            send_buf_size = d_send_buf_end - d_sendbuf.data;
            }

        if (m_prof)
            m_prof->pop();


        unsigned int send_neighbor = m_neighbors[dir];

        // we receive from the direction opposite to the one we send to
        unsigned int recv_neighbor;
        if (dir % 2 == 0)
            recv_neighbor = m_neighbors[dir+1];
        else
            recv_neighbor = m_neighbors[dir-1];

        if (m_prof)
            m_prof->push("MPI send/recv");

        // communicate size of the message that will contain the particle data
        boost::mpi::request reqs[2];
        reqs[0] = m_mpi_comm->isend(send_neighbor,0,send_buf_size);
        reqs[1] = m_mpi_comm->irecv(recv_neighbor,0,recv_buf_size[dir]);
        boost::mpi::wait_all(reqs,reqs+2);

        // Check if receive buffer is large enough and resize if necessary
        if (recv_buf_size[dir] > m_recvbuf[dir].getNumElements())
            {
            unsigned int new_size = m_recvbuf[dir].getNumElements();
            while (new_size < recv_buf_size[dir]) new_size *= 2;
            m_recvbuf[dir].resize(new_size);
            }

#ifdef ENABLE_MPI_CUDA
            {
            ArrayHandle<char> d_sendbuf(m_sendbuf[dir], access_location::device, access_mode::read);
            ArrayHandle<char> d_recvbuf(m_recvbuf[dir], access_location::device, access_mode::overwrite);

            // exchange actual particle data
            reqs[0] = m_mpi_comm->isend(send_neighbor,1,d_sendbuf.data,send_buf_size);
            reqs[1] = m_mpi_comm->irecv(recv_neighbor,1,d_recvbuf.data,recv_buf_size[dir]);
            boost::mpi::wait_all(reqs,reqs+2);
            }

#else
            {
            ArrayHandle<char> h_sendbuf(m_sendbuf[dir], access_location::host, access_mode::read);
            ArrayHandle<char> h_recvbuf(m_recvbuf[dir], access_location::host, access_mode::overwrite);
            // exchange actual particle data
            reqs[0] = m_mpi_comm->isend(send_neighbor,2,h_sendbuf.data,send_buf_size);
            reqs[1] = m_mpi_comm->irecv(recv_neighbor,2,h_recvbuf.data,recv_buf_size[dir]);
            boost::mpi::wait_all(reqs,reqs+2);
            }
#endif
       if (m_prof)
          m_prof->pop();

       unsigned int n_recv_ptl;

            {
            m_global_box_gpu.xlo = m_global_box.xlo;
            m_global_box_gpu.xhi = m_global_box.xhi;
            m_global_box_gpu.ylo = m_global_box.ylo;
            m_global_box_gpu.yhi = m_global_box.yhi;
            m_global_box_gpu.zlo = m_global_box.zlo;
            m_global_box_gpu.zhi = m_global_box.zhi;

            ArrayHandle<char> d_recvbuf(m_recvbuf[dir], access_location::device, access_mode::readwrite);
            gpu_migrate_wrap_received_particles(d_recvbuf.data,
                                                d_recvbuf.data+recv_buf_size[dir],
                                                n_recv_ptl,
                                                m_global_box_gpu,
                                                dir);
            }

            {
            // start index for atoms to be added
            unsigned int add_idx = m_pdata->getN();

            // allocate memory for particles that have been received
            m_pdata->addParticles(n_recv_ptl);

            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::readwrite);
            ArrayHandle<int3> d_image(m_pdata->getImages(), access_location::device, access_mode::readwrite);
            ArrayHandle<unsigned int> d_body(m_pdata->getBodies(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::readwrite);
            ArrayHandle<unsigned int> d_global_tag(m_pdata->getGlobalTags(), access_location::device, access_mode::readwrite);

            ArrayHandle<char> d_recvbuf(m_recvbuf[dir], access_location::device, access_mode::read);

            gpu_migrate_add_particles(  d_recvbuf.data,
                                        d_recvbuf.data+recv_buf_size[dir],
                                        d_pos.data + add_idx,
                                        d_vel.data + add_idx,
                                        d_accel.data + add_idx,
                                        d_image.data + add_idx,
                                        d_charge.data + add_idx,
                                        d_diameter.data + add_idx,
                                        d_body.data + add_idx,
                                        d_orientation.data + add_idx,
                                        d_global_tag.data + add_idx,
                                        m_pdata->getBoxGPU());

            }

        } // end dir loop


        {
        // update rtag information
        ArrayHandle<unsigned int> d_global_tag(m_pdata->getGlobalTags(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_global_rtag(m_pdata->getGlobalRTags(), access_location::device, access_mode::readwrite);
        gpu_update_rtag(m_pdata->getN(),0, d_global_tag.data, d_global_rtag.data);
        }

    // check that total particle number is conserved
    unsigned int N;
    reduce(*m_mpi_comm,m_pdata->getN(), N, std::plus<unsigned int>(), 0);
    if (m_mpi_comm->rank() == 0 && N != m_pdata->getNGlobal())
        {
        cerr << endl << "***Error! Global number of particles has changed unexpectedly (" <<
                N << " != " << m_pdata->getNGlobal() << ")." << endl << endl;
        throw runtime_error("Error in MPI communication.");
        }

    // notify ParticleData that addition / removal of particles is complete
    if (m_prof)
        m_prof->push("group update");

    m_pdata->notifyParticleNumberChange();

    if (m_prof)
        m_prof->pop();

    if (m_prof)
        m_prof->pop();
    }

//! build ghost particle list, copy ghost particle data
void CommunicatorGPU::exchangeGhosts(Scalar r_ghost)
    {
    if (m_prof)
        m_prof->push("exchange_ghosts");

    m_r_ghost = r_ghost;

    // we have a current list of atoms inside this box
    // find all local atoms within a distance r_ghost from the boundary and store them in m_copy_ghosts

    // first reset number of ghost particles
    m_pdata->removeAllGhostParticles();

    for (unsigned int dir = 0; dir < 6; dir ++)
        {
        unsigned int dim = getDimension(dir/2);
        if (dim == 1) continue;

        m_num_copy_ghosts[dir] = 0;


        // scan all atom positions if they are within r_ghost from a neighbor, fill send buffer

            {
            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
            ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::read);
            ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_global_tag(m_pdata->getGlobalTags(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_global_rtag(m_pdata->getGlobalRTags(), access_location::device, access_mode::read);

            ArrayHandle<unsigned int> d_copy_ghosts(m_copy_ghosts[dir], access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar4> d_pos_copybuf(m_pos_copybuf[dir], access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar> d_charge_copybuf(m_charge_copybuf[dir], access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar> d_diameter_copybuf(m_diameter_copybuf[dir], access_location::device, access_mode::overwrite);

            gpu_make_exchange_ghost_list(m_pdata->getN(),
                                         dir,
                                         d_pos.data,
                                         d_global_tag.data,
                                         d_copy_ghosts.data,
                                         m_num_copy_ghosts[dir],
                                         m_pdata->getBoxGPU(),
                                         m_r_ghost);

            gpu_exchange_ghosts(m_num_copy_ghosts[dir],
                                d_copy_ghosts.data,
                                d_global_rtag.data,
                                d_pos.data,
                                d_pos_copybuf.data,
                                d_charge.data,
                                d_charge_copybuf.data,
                                d_diameter.data,
                                d_diameter_copybuf.data);
            }

        // resize array of ghost particle ids to copy if necessary
        if (m_pdata->getN() + m_pdata->getNGhosts() > m_max_copy_ghosts[dir])
            {
            while (m_pdata->getN() + m_pdata->getNGhosts() > m_max_copy_ghosts[dir]) m_max_copy_ghosts[dir] *= 2;

            m_copy_ghosts[dir].resize(m_max_copy_ghosts[dir]);
            m_pos_copybuf[dir].resize(m_max_copy_ghosts[dir]);
            m_charge_copybuf[dir].resize(m_max_copy_ghosts[dir]);
            m_diameter_copybuf[dir].resize(m_max_copy_ghosts[dir]);
            }

            {
            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
            ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::read);
            ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_global_tag(m_pdata->getGlobalTags(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_global_rtag(m_pdata->getGlobalRTags(), access_location::device, access_mode::read);

            ArrayHandle<unsigned int> d_copy_ghosts(m_copy_ghosts[dir], access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar4> d_pos_copybuf(m_pos_copybuf[dir], access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar> d_charge_copybuf(m_charge_copybuf[dir], access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar> d_diameter_copybuf(m_diameter_copybuf[dir], access_location::device, access_mode::readwrite);

            unsigned int num_forward_ghosts;
            gpu_make_exchange_ghost_list(m_pdata->getNGhosts(),
                                         dir,
                                         d_pos.data + m_pdata->getN(),
                                         d_global_tag.data + m_pdata->getN(),
                                         d_copy_ghosts.data + m_num_copy_ghosts[dir],
                                         num_forward_ghosts,
                                         m_pdata->getBoxGPU(),
                                         m_r_ghost);

            gpu_exchange_ghosts(num_forward_ghosts,
                                d_copy_ghosts.data + m_num_copy_ghosts[dir],
                                d_global_rtag.data,
                                d_pos.data,
                                d_pos_copybuf.data + m_num_copy_ghosts[dir],
                                d_charge.data,
                                d_charge_copybuf.data + m_num_copy_ghosts[dir],
                                d_diameter.data,
                                d_diameter_copybuf.data + m_num_copy_ghosts[dir]);

            m_num_copy_ghosts[dir] += num_forward_ghosts;
            }

        unsigned int send_neighbor = m_neighbors[dir];

        // we receive from the direction opposite to the one we send to
        unsigned int recv_neighbor;
        if (dir % 2 == 0)
            recv_neighbor = m_neighbors[dir+1];
        else
            recv_neighbor = m_neighbors[dir-1];


        if (m_prof)
            m_prof->push("MPI send/recv");

        // communicate size of the message that will contain the particle data
        boost::mpi::request reqs[10];
        reqs[0] = m_mpi_comm->isend(send_neighbor,0,m_num_copy_ghosts[dir]);
        reqs[1] = m_mpi_comm->irecv(recv_neighbor,0,m_num_recv_ghosts[dir]);
        boost::mpi::wait_all(reqs,reqs+2);

        if (m_prof)
            m_prof->pop();

        // append ghosts at the end of particle data array
        unsigned int start_idx = m_pdata->getN() + m_pdata->getNGhosts();

        // accommodate new ghost particles
        m_pdata->addGhostParticles(m_num_recv_ghosts[dir]);

        // exchange particle data, write directly to the particle data arrays
        if (m_prof)
            m_prof->push("MPI send/recv");

#ifdef ENABLE_MPI_CUDA
            {
            ArrayHandle<unsigned int> d_copy_ghosts(m_copy_ghosts[dir], access_location::device, access_mode::read);
            ArrayHandle<Scalar4> d_pos_copybuf(m_pos_copybuf[dir], access_location::device, access_mode::read);
            ArrayHandle<Scalar> d_charge_copybuf(m_charge_copybuf[dir], access_location::device, access_mode::read);
            ArrayHandle<Scalar> d_diameter_copybuf(m_diameter_copybuf[dir], access_location::device, access_mode::read);

            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::readwrite);
            ArrayHandle<unsigned int> d_global_tag(m_pdata->getGlobalTags(), access_location::device, access_mode::readwrite);

            reqs[2] = m_mpi_comm->isend(send_neighbor,1,d_pos_copybuf.data, m_num_copy_ghosts[dir]);
            reqs[3] = m_mpi_comm->irecv(recv_neighbor,1,d_pos.data + start_idx, m_num_recv_ghosts[dir]);

            reqs[4] = m_mpi_comm->isend(send_neighbor,2,d_copy_ghosts.data, m_num_copy_ghosts[dir]);
            reqs[5] = m_mpi_comm->irecv(recv_neighbor,2,d_global_tag.data + start_idx, m_num_recv_ghosts[dir]);

            reqs[6] = m_mpi_comm->isend(send_neighbor,3,d_charge_copybuf.data, m_num_copy_ghosts[dir]);
            reqs[7] = m_mpi_comm->irecv(recv_neighbor,3,d_charge.data + start_idx, m_num_recv_ghosts[dir]);

            reqs[8] = m_mpi_comm->isend(send_neighbor,4,d_diameter_copybuf.data, m_num_copy_ghosts[dir]);
            reqs[9] = m_mpi_comm->irecv(recv_neighbor,4,d_diameter.data + start_idx, m_num_recv_ghosts[dir]);

            boost::mpi::wait_all(reqs+2,reqs+10);
            }
#else
            {
            ArrayHandle<unsigned int> h_copy_ghosts(m_copy_ghosts[dir], access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_pos_copybuf(m_pos_copybuf[dir], access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_charge_copybuf(m_charge_copybuf[dir], access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_diameter_copybuf(m_diameter_copybuf[dir], access_location::host, access_mode::read);

            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::readwrite);
            ArrayHandle<unsigned int> h_global_tag(m_pdata->getGlobalTags(), access_location::host, access_mode::readwrite);

            reqs[2] = m_mpi_comm->isend(send_neighbor,1,h_pos_copybuf.data, m_num_copy_ghosts[dir]);
            reqs[3] = m_mpi_comm->irecv(recv_neighbor,1,h_pos.data + start_idx, m_num_recv_ghosts[dir]);

            reqs[4] = m_mpi_comm->isend(send_neighbor,2,h_copy_ghosts.data, m_num_copy_ghosts[dir]);
            reqs[5] = m_mpi_comm->irecv(recv_neighbor,2,h_global_tag.data + start_idx, m_num_recv_ghosts[dir]);

            reqs[6] = m_mpi_comm->isend(send_neighbor,3,h_charge_copybuf.data, m_num_copy_ghosts[dir]);
            reqs[7] = m_mpi_comm->irecv(recv_neighbor,3,h_charge.data + start_idx, m_num_recv_ghosts[dir]);

            reqs[8] = m_mpi_comm->isend(send_neighbor,4,h_diameter_copybuf.data, m_num_copy_ghosts[dir]);
            reqs[9] = m_mpi_comm->irecv(recv_neighbor,4,h_diameter.data + start_idx, m_num_recv_ghosts[dir]);

            boost::mpi::wait_all(reqs+2,reqs+10);
            }
#endif
        if (m_prof)
            m_prof->pop();

            {
            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);

            m_global_box_gpu.xlo = m_global_box.xlo;
            m_global_box_gpu.xhi = m_global_box.xhi;
            m_global_box_gpu.ylo = m_global_box.ylo;
            m_global_box_gpu.yhi = m_global_box.yhi;
            m_global_box_gpu.zlo = m_global_box.zlo;
            m_global_box_gpu.zhi = m_global_box.zhi;

            gpu_wrap_ghost_particles(dir,
                              m_num_recv_ghosts[dir],
                              d_pos.data + start_idx,
                              m_global_box_gpu,
                              m_r_ghost);

            ArrayHandle<unsigned int> d_global_tag(m_pdata->getGlobalTags(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_global_rtag(m_pdata->getGlobalRTags(), access_location::device, access_mode::readwrite);

            gpu_update_rtag(m_num_recv_ghosts[dir], start_idx, d_global_tag.data + start_idx, d_global_rtag.data);

            }
        }

    if (m_prof)
        m_prof->pop();
    }

//! update positions of ghost particles
void CommunicatorGPU::copyGhosts()
    {
    // we have a current m_copy_ghosts list which contain the indices of particles
    // to send to neighboring processors
    if (m_prof)
        m_prof->push("copy_ghosts");

    unsigned int num_tot_recv_ghosts = 0; // total number of ghosts received


    for (unsigned int dir = 0; dir < 6; dir ++)
        {

        unsigned int dim = getDimension(dir/2);
        if (dim == 1) continue;

            {
            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
            ArrayHandle<Scalar4> d_pos_copybuf(m_pos_copybuf[dir], access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_copy_ghosts(m_copy_ghosts[dir], access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_rtag(m_pdata->getGlobalRTags(), access_location::device, access_mode::read);

            gpu_copy_ghosts(m_num_copy_ghosts[dir], d_pos.data, d_copy_ghosts.data, d_pos_copybuf.data,d_rtag.data);
            }

        unsigned int send_neighbor = m_neighbors[dir];

        // we receive from the direction opposite to the one we send to
        unsigned int recv_neighbor;
        if (dir % 2 == 0)
            recv_neighbor = m_neighbors[dir+1];
        else
            recv_neighbor = m_neighbors[dir-1];

        unsigned int start_idx;
        {
        if (m_prof)
            m_prof->push("MPI send/recv");


        start_idx = m_pdata->getN() + num_tot_recv_ghosts;

        num_tot_recv_ghosts += m_num_recv_ghosts[dir];

        boost::mpi::request reqs[2];
#ifdef ENABLE_MPI_CUDA
            {
            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar4> d_pos_copybuf(m_pos_copybuf[dir], access_location::device, access_mode::read);

            // exchange particle data, write directly to the particle data arrays
            reqs[0] = m_mpi_comm->isend(send_neighbor,0,d_pos_copybuf.data, m_num_copy_ghosts[dir]);
            reqs[1] = m_mpi_comm->irecv(recv_neighbor,0,d_pos.data + start_idx, m_num_recv_ghosts[dir]);

            boost::mpi::wait_all(reqs,reqs+2);
            }
#else
            {
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_pos_copybuf(m_pos_copybuf[dir], access_location::host, access_mode::read);

            // exchange particle data, write directly to the particle data arrays
            reqs[0] = m_mpi_comm->isend(send_neighbor,0,h_pos_copybuf.data, m_num_copy_ghosts[dir]);
            reqs[1] = m_mpi_comm->irecv(recv_neighbor,0,h_pos.data + start_idx, m_num_recv_ghosts[dir]);

            boost::mpi::wait_all(reqs,reqs+2);
            }
#endif

        if (m_prof)
            m_prof->pop(0, (m_num_recv_ghosts[dir]+m_num_copy_ghosts[dir])*sizeof(Scalar4));
        }

        if (m_prof)
            m_prof->push("particle wrap");

            {
            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);

            m_global_box_gpu.xlo = m_global_box.xlo;
            m_global_box_gpu.xhi = m_global_box.xhi;
            m_global_box_gpu.ylo = m_global_box.ylo;
            m_global_box_gpu.yhi = m_global_box.yhi;
            m_global_box_gpu.zlo = m_global_box.zlo;
            m_global_box_gpu.zhi = m_global_box.zhi;

            gpu_wrap_ghost_particles(dir,
                           m_num_recv_ghosts[dir],
                           d_pos.data + start_idx,
                           m_global_box_gpu,
                           m_r_ghost);
             }

        if (m_prof)
            m_prof->pop();
        } // end dir loop

        if (m_prof)
            m_prof->pop();
    }

#endif // ENABLE_CUDA
#endif // ENABLE_MPI
