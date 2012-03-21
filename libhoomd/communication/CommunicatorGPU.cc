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

#include <boost/mpi.hpp>
#include <boost/python.hpp>
using namespace boost::python;

//! Define some of our types as fixed-size MPI datatypes for performance optimization
BOOST_IS_MPI_DATATYPE(Scalar4)
BOOST_IS_MPI_DATATYPE(Scalar3)
BOOST_IS_MPI_DATATYPE(int3)

BOOST_CLASS_TRACKING(Scalar4,track_never)

//! Constructor
CommunicatorGPU::CommunicatorGPU(boost::shared_ptr<SystemDefinition> sysdef,
                                 boost::shared_ptr<boost::mpi::communicator> mpi_comm,
                                 const DomainDecomposition decomposition)
    : Communicator(sysdef, mpi_comm, decomposition)
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

        {
        // Reset reverse lookup tags of old ghost atoms
        ArrayHandle<unsigned int> d_global_rtag(m_pdata->getGlobalRTags(), access_location::device, access_mode::readwrite);
        ArrayHandle<unsigned int> d_global_tag(m_pdata->getGlobalTags(), access_location::device, access_mode::read);

        gpu_reset_rtags(m_pdata->getNGhosts(),
                        d_global_tag.data + m_pdata->getN(),
                        d_global_rtag.data);
        }

    // reset ghost particle number
    m_pdata->removeAllGhostParticles();

    unsigned int recv_buf_size; // size of receive buffer

    for (unsigned int dir=0; dir < 6; dir++)
        {
        char *d_send_buf_end;
        unsigned int n_send_ptls;

        if (getDimension(dir/2) == 1) continue;

        if (m_prof)
            m_prof->push("remove ptls");

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
            ArrayHandle<unsigned int> d_global_tag(m_pdata->getGlobalTags(), access_location::device, access_mode::read);

            ArrayHandle<Scalar4> d_pos_tmp(m_pos_tmp, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar4> d_vel_tmp(m_vel_tmp, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar3> d_accel_tmp(m_accel_tmp, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar> d_charge_tmp(m_charge_tmp, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar> d_diameter_tmp(m_diameter_tmp, access_location::device, access_mode::overwrite);
            ArrayHandle<int3> d_image_tmp(m_image_tmp, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_body_tmp(m_body_tmp, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar4> d_orientation_tmp(m_orientation_tmp, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_tag_tmp(m_tag_tmp, access_location::device, access_mode::overwrite);

            /* Reorder particles.
               Particles that stay in our domain come first, followed by the particles that are sent to a
               neighboring processor.
             */
            gpu_migrate_select_particles(m_pdata->getN(),
                                   n_send_ptls,
                                   d_pos.data,
                                   d_pos_tmp.data,
                                   d_vel.data,
                                   d_vel_tmp.data,
                                   d_accel.data,
                                   d_accel_tmp.data,
                                   d_image.data,
                                   d_image_tmp.data,
                                   d_charge.data,
                                   d_charge_tmp.data,
                                   d_diameter.data,
                                   d_diameter_tmp.data,
                                   d_body.data,
                                   d_body_tmp.data,
                                   d_orientation.data,
                                   d_orientation_tmp.data,
                                   d_global_tag.data,
                                   d_tag_tmp.data,
                                   m_pdata->getBoxGPU(),
                                   dir);
            }

        // Swap temporary arrays with particle data arrays
        m_pdata->getPositions().swap(m_pos_tmp);
        m_pdata->getVelocities().swap(m_vel_tmp);
        m_pdata->getAccelerations().swap(m_accel_tmp);
        m_pdata->getImages().swap(m_image_tmp);
        m_pdata->getCharges().swap(m_charge_tmp);
        m_pdata->getDiameters().swap(m_diameter_tmp);
        m_pdata->getBodies().swap(m_body_tmp);
        m_pdata->getOrientationArray().swap(m_orientation_tmp);
        m_pdata->getGlobalTags().swap(m_tag_tmp);

        if (m_prof)
	        m_prof->pop();

        // Update number of particles in system
        m_pdata->removeParticles(n_send_ptls);


            {
            // Reset reverse lookup tags of removed particles
            ArrayHandle<unsigned int> d_global_rtag(m_pdata->getGlobalRTags(), access_location::device, access_mode::readwrite);
            ArrayHandle<unsigned int> d_global_tag(m_pdata->getGlobalTags(), access_location::device, access_mode::read);

            gpu_reset_rtags(n_send_ptls,
                            d_global_tag.data + m_pdata->getN(),
                            d_global_rtag.data);

            }

        // scan all atom positions and fill the send buffers with those that have left the domain boundaries,
        // and remove those particles from the local domain

        // Resize send buffer
        m_sendbuf.resize(n_send_ptls*m_packed_size);

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

            ArrayHandle<char> d_sendbuf(m_sendbuf, access_location::device, access_mode::overwrite);

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

        unsigned int send_neighbor = m_decomposition.neighbors[dir];

        // we receive from the direction opposite to the one we send to
        unsigned int recv_neighbor;
        if (dir % 2 == 0)
            recv_neighbor = m_decomposition.neighbors[dir+1];
        else
            recv_neighbor = m_decomposition.neighbors[dir-1];

        if (m_prof)
            m_prof->push("MPI send/recv");

        // communicate size of the message that will contain the particle data
        boost::mpi::request reqs[2];
        reqs[0] = m_mpi_comm->isend(send_neighbor,0,send_buf_size);
        reqs[1] = m_mpi_comm->irecv(recv_neighbor,0,recv_buf_size);
        boost::mpi::wait_all(reqs,reqs+2);

        // Resize receive buffer
        m_recvbuf.resize(recv_buf_size);

#ifdef ENABLE_MPI_CUDA
            {
            ArrayHandle<char> d_sendbuf(m_sendbuf, access_location::device, access_mode::read);
            ArrayHandle<char> d_recvbuf(m_recvbuf, access_location::device, access_mode::overwrite);

            // exchange actual particle data
            reqs[0] = m_mpi_comm->isend(send_neighbor,1,d_sendbuf.data,send_buf_size);
            reqs[1] = m_mpi_comm->irecv(recv_neighbor,1,d_recvbuf.data,recv_buf_size);
            boost::mpi::wait_all(reqs,reqs+2);
            }

#else
            {
            ArrayHandle<char> h_sendbuf(m_sendbuf, access_location::host, access_mode::read);
            ArrayHandle<char> h_recvbuf(m_recvbuf, access_location::host, access_mode::overwrite);
            // exchange actual particle data
            reqs[0] = m_mpi_comm->isend(send_neighbor,2,h_sendbuf.data,send_buf_size);
            reqs[1] = m_mpi_comm->irecv(recv_neighbor,2,h_recvbuf.data,recv_buf_size);
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

            ArrayHandle<char> d_recvbuf(m_recvbuf, access_location::device, access_mode::readwrite);
            gpu_migrate_wrap_received_particles(d_recvbuf.data,
                                                d_recvbuf.data+recv_buf_size,
                                                n_recv_ptl,
                                                m_global_box_gpu,
                                                dir,
                                                m_decomposition.is_at_boundary);
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

            ArrayHandle<char> d_recvbuf(m_recvbuf, access_location::device, access_mode::read);

            gpu_migrate_add_particles(  d_recvbuf.data,
                                        d_recvbuf.data+recv_buf_size,
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

#ifndef NDEBUG
    // check that total particle number is conserved
    unsigned int N;
    reduce(*m_mpi_comm,m_pdata->getN(), N, std::plus<unsigned int>(), 0);
    if (m_mpi_comm->rank() == 0 && N != m_pdata->getNGlobal())
        {
        cerr << endl << "***Error! Global number of particles has changed unexpectedly (" <<
                N << " != " << m_pdata->getNGlobal() << ")." << endl << endl;
        throw runtime_error("Error in MPI communication.");
        }
#endif

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

    assert(m_r_ghost < (m_pdata->getBox().xhi - m_pdata->getBox().xlo));
    assert(m_r_ghost < (m_pdata->getBox().yhi - m_pdata->getBox().ylo));
    assert(m_r_ghost < (m_pdata->getBox().zhi - m_pdata->getBox().zlo));

    // reset plans
    m_plan.clear();

    // resize plans
    m_plan.resize(m_pdata->getN());

    /*
     * Mark particles that are part of incomplete bonds for sending
     */
    boost::shared_ptr<BondData> bdata = m_sysdef->getBondData();

    if (bdata->getNumBonds())
        {
        // Send incomplete bond member to the nearest plane in all directions
        const GPUArray<uint2>& btable = bdata->getGPUBondList();
        ArrayHandle<uint2> d_btable(btable, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_n_bonds(bdata->getNBondsArray(), access_location::device, access_mode::read);
        ArrayHandle<unsigned char> d_plan(m_plan, access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);

        gpu_mark_particles_in_incomplete_bonds(d_btable.data,
                                               btable.getPitch(),
                                               d_n_bonds.data,
                                               d_plan.data,
                                               d_pos.data,
                                               m_pdata->getBoxGPU(),
                                               m_pdata->getN());
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
                                         m_pdata->getBoxGPU(),
                                         m_r_ghost);
        }

    /*
     * Fill send buffers, exchange particles according to plans
     */

    // resize buffers
    m_plan_copybuf.resize(m_pdata->getN());
    m_pos_copybuf.resize(m_pdata->getN());
    m_charge_copybuf.resize(m_pdata->getN());
    m_diameter_copybuf.resize(m_pdata->getN());

    for (unsigned int dir = 0; dir < 6; dir ++)
        {
        unsigned int dim = getDimension(dir/2);
        if (dim == 1) continue;

        m_num_copy_ghosts[dir] = 0;

        // resize array of ghost particle tags to copy 
        unsigned int max_copy_ghosts = m_pdata->getN() + m_pdata->getNGhosts();
        m_copy_ghosts[dir].resize(max_copy_ghosts);
        
        // resize buffers
        m_pos_copybuf.resize(max_copy_ghosts);
        m_charge_copybuf.resize(max_copy_ghosts);
        m_diameter_copybuf.resize(max_copy_ghosts);
        m_plan_copybuf.resize(max_copy_ghosts);


            {
            // Fill send buffer
            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
            ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::read);
            ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_global_tag(m_pdata->getGlobalTags(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_global_rtag(m_pdata->getGlobalRTags(), access_location::device, access_mode::read);
            ArrayHandle<unsigned char> d_plan(m_plan, access_location::device, access_mode::read);

            ArrayHandle<unsigned int> d_copy_ghosts(m_copy_ghosts[dir], access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar4> d_pos_copybuf(m_pos_copybuf, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar> d_charge_copybuf(m_charge_copybuf, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar> d_diameter_copybuf(m_diameter_copybuf, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned char> d_plan_copybuf(m_plan_copybuf, access_location::device, access_mode::overwrite);

            gpu_make_exchange_ghost_list(m_pdata->getN()+m_pdata->getNGhosts(),
                                         dir,
                                         d_plan.data,
                                         d_global_tag.data,
                                         d_copy_ghosts.data,
                                         m_num_copy_ghosts[dir]);

            gpu_exchange_ghosts(m_num_copy_ghosts[dir],
                                d_copy_ghosts.data,
                                d_global_rtag.data,
                                d_pos.data,
                                d_pos_copybuf.data,
                                d_charge.data,
                                d_charge_copybuf.data,
                                d_diameter.data,
                                d_diameter_copybuf.data,
                                d_plan.data,
                                d_plan_copybuf.data);
            }

        unsigned int send_neighbor = m_decomposition.neighbors[dir];

        // we receive from the direction opposite to the one we send to
        unsigned int recv_neighbor;
        if (dir % 2 == 0)
            recv_neighbor = m_decomposition.neighbors[dir+1];
        else
            recv_neighbor = m_decomposition.neighbors[dir-1];


        if (m_prof)
            m_prof->push("MPI send/recv");

        // communicate size of the message that will contain the particle data
        boost::mpi::request reqs[12];
        reqs[0] = m_mpi_comm->isend(send_neighbor,0,m_num_copy_ghosts[dir]);
        reqs[1] = m_mpi_comm->irecv(recv_neighbor,0,m_num_recv_ghosts[dir]);
        boost::mpi::wait_all(reqs,reqs+2);

        if (m_prof)
            m_prof->pop();

        // append ghosts at the end of particle data array
        unsigned int start_idx = m_pdata->getN() + m_pdata->getNGhosts();

        // accommodate new ghost particles
        m_pdata->addGhostParticles(m_num_recv_ghosts[dir]);

        // resize plan array
        m_plan.resize(m_pdata->getN() + m_pdata->getNGhosts());

        // exchange particle data, write directly to the particle data arrays
        if (m_prof)
            m_prof->push("MPI send/recv");

#ifdef ENABLE_MPI_CUDA
            {
            ArrayHandle<unsigned int> d_copy_ghosts(m_copy_ghosts[dir], access_location::device, access_mode::read);
            ArrayHandle<Scalar4> d_pos_copybuf(m_pos_copybuf, access_location::device, access_mode::read);
            ArrayHandle<Scalar> d_charge_copybuf(m_charge_copybuf, access_location::device, access_mode::read);
            ArrayHandle<Scalar> d_diameter_copybuf(m_diameter_copybuf, access_location::device, access_mode::read);
            ArrayHandle<unsigned char> d_plan_copybuf(m_plan_copybuf, access_location::device, access_mode::read);

            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::readwrite);
            ArrayHandle<unsigned int> d_global_tag(m_pdata->getGlobalTags(), access_location::device, access_mode::readwrite);
            ArrayHandle<unsigned char> d_plan(m_plan, access_location::device, access_mode::readwrite);

            reqs[2] = m_mpi_comm->isend(send_neighbor,1,d_plan_copybuf.data, m_num_copy_ghosts[dir]);
            reqs[3] = m_mpi_comm->irecv(recv_neighbor,1,d_plan.data + start_idx, m_num_recv_ghosts[dir]);

            reqs[4] = m_mpi_comm->isend(send_neighbor,2,d_pos_copybuf.data, m_num_copy_ghosts[dir]);
            reqs[5] = m_mpi_comm->irecv(recv_neighbor,2,d_pos.data + start_idx, m_num_recv_ghosts[dir]);

            reqs[6] = m_mpi_comm->isend(send_neighbor,3,d_copy_ghosts.data, m_num_copy_ghosts[dir]);
            reqs[7] = m_mpi_comm->irecv(recv_neighbor,3,d_global_tag.data + start_idx, m_num_recv_ghosts[dir]);

            reqs[8] = m_mpi_comm->isend(send_neighbor,4,d_charge_copybuf.data, m_num_copy_ghosts[dir]);
            reqs[9] = m_mpi_comm->irecv(recv_neighbor,4,d_charge.data + start_idx, m_num_recv_ghosts[dir]);

            reqs[10] = m_mpi_comm->isend(send_neighbor,5,d_diameter_copybuf.data, m_num_copy_ghosts[dir]);
            reqs[11] = m_mpi_comm->irecv(recv_neighbor,5,d_diameter.data + start_idx, m_num_recv_ghosts[dir]);

            boost::mpi::wait_all(reqs+2,reqs+12);
            }
#else
            {
            ArrayHandle<unsigned int> h_copy_ghosts(m_copy_ghosts[dir], access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_pos_copybuf(m_pos_copybuf, access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_charge_copybuf(m_charge_copybuf, access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_diameter_copybuf(m_diameter_copybuf, access_location::host, access_mode::read);
            ArrayHandle<unsigned char> h_plan_copybuf(m_plan_copybuf, access_location::host, access_mode::read);

            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::readwrite);
            ArrayHandle<unsigned int> h_global_tag(m_pdata->getGlobalTags(), access_location::host, access_mode::readwrite);
            ArrayHandle<unsigned char> h_plan(m_plan, access_location::host, access_mode::readwrite);

            reqs[2] = m_mpi_comm->isend(send_neighbor,1,h_plan_copybuf.data, m_num_copy_ghosts[dir]);
            reqs[3] = m_mpi_comm->irecv(recv_neighbor,1,h_plan.data + start_idx, m_num_recv_ghosts[dir]);

            reqs[4] = m_mpi_comm->isend(send_neighbor,2,h_pos_copybuf.data, m_num_copy_ghosts[dir]);
            reqs[5] = m_mpi_comm->irecv(recv_neighbor,2,h_pos.data + start_idx, m_num_recv_ghosts[dir]);

            reqs[6] = m_mpi_comm->isend(send_neighbor,3,h_copy_ghosts.data, m_num_copy_ghosts[dir]);
            reqs[7] = m_mpi_comm->irecv(recv_neighbor,3,h_global_tag.data + start_idx, m_num_recv_ghosts[dir]);

            reqs[8] = m_mpi_comm->isend(send_neighbor,4,h_charge_copybuf.data, m_num_copy_ghosts[dir]);
            reqs[9] = m_mpi_comm->irecv(recv_neighbor,4,h_charge.data + start_idx, m_num_recv_ghosts[dir]);

            reqs[10] = m_mpi_comm->isend(send_neighbor,5,h_diameter_copybuf.data, m_num_copy_ghosts[dir]);
            reqs[11] = m_mpi_comm->irecv(recv_neighbor,5,h_diameter.data + start_idx, m_num_recv_ghosts[dir]);

            boost::mpi::wait_all(reqs+2,reqs+12);
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
                              m_r_ghost,
                              m_decomposition.is_at_boundary);

            ArrayHandle<unsigned int> d_global_tag(m_pdata->getGlobalTags(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_global_rtag(m_pdata->getGlobalRTags(), access_location::device, access_mode::readwrite);

            gpu_update_rtag(m_num_recv_ghosts[dir], start_idx, d_global_tag.data + start_idx, d_global_rtag.data);

            }
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


    for (unsigned int dir = 0; dir < 6; dir ++)
        {

        unsigned int dim = getDimension(dir/2);
        if (dim == 1) continue;

            {
            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
            ArrayHandle<Scalar4> d_pos_copybuf(m_pos_copybuf, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_copy_ghosts(m_copy_ghosts[dir], access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_rtag(m_pdata->getGlobalRTags(), access_location::device, access_mode::read);

            gpu_copy_ghosts(m_num_copy_ghosts[dir], d_pos.data, d_copy_ghosts.data, d_pos_copybuf.data,d_rtag.data);
            }

        unsigned int send_neighbor = m_decomposition.neighbors[dir];

        // we receive from the direction opposite to the one we send to
        unsigned int recv_neighbor;
        if (dir % 2 == 0)
            recv_neighbor = m_decomposition.neighbors[dir+1];
        else
            recv_neighbor = m_decomposition.neighbors[dir-1];

        unsigned int start_idx;

        if (m_prof)
            m_prof->push("MPI send/recv");


        start_idx = m_pdata->getN() + num_tot_recv_ghosts;

        num_tot_recv_ghosts += m_num_recv_ghosts[dir];

        boost::mpi::request reqs[2];
#ifdef ENABLE_MPI_CUDA
            {
            ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar4> d_pos_copybuf(m_pos_copybuf, access_location::device, access_mode::read);

            // exchange particle data, write directly to the particle data arrays
            reqs[0] = m_mpi_comm->isend(send_neighbor,0,d_pos_copybuf.data, m_num_copy_ghosts[dir]);
            reqs[1] = m_mpi_comm->irecv(recv_neighbor,0,d_pos.data + start_idx, m_num_recv_ghosts[dir]);

            boost::mpi::wait_all(reqs,reqs+2);
            }
#else
            {
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_pos_copybuf(m_pos_copybuf, access_location::host, access_mode::read);

            // exchange particle data, write directly to the particle data arrays
            reqs[0] = m_mpi_comm->isend(send_neighbor,0,h_pos_copybuf.data, m_num_copy_ghosts[dir]);
            reqs[1] = m_mpi_comm->irecv(recv_neighbor,0,h_pos.data + start_idx, m_num_recv_ghosts[dir]);

            boost::mpi::wait_all(reqs,reqs+2);
            }
#endif

        if (m_prof)
            m_prof->pop(0, (m_num_recv_ghosts[dir]+m_num_copy_ghosts[dir])*sizeof(Scalar4));

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
                           m_r_ghost,
                           m_decomposition.is_at_boundary);
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
                boost::shared_ptr<boost::mpi::communicator>,
                const DomainDecomposition>())
    ;
    }

#endif // ENABLE_CUDA
#endif // ENABLE_MPI
