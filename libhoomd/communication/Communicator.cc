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

/*! \file Communicator.cc
    \brief Implements the Communicator class
*/

#ifdef ENABLE_MPI

#include "Communicator.h"
#include "System.h"

#include <boost/mpi.hpp>
#include <boost/python.hpp>

using namespace boost::python;

//! Define some of our types as fixed-size MPI datatypes for performance optimization
BOOST_IS_MPI_DATATYPE(Scalar4)
BOOST_IS_MPI_DATATYPE(Scalar3)
BOOST_IS_MPI_DATATYPE(uint3)
BOOST_IS_MPI_DATATYPE(int3)

//! Constructor
Communicator::Communicator(boost::shared_ptr<SystemDefinition> sysdef,
                           boost::shared_ptr<boost::mpi::communicator> mpi_comm,
                           std::vector<unsigned int> neighbor_rank,
                           int3 dim,
                           const BoxDim& global_box)
          : m_sysdef(sysdef),
            m_pdata(sysdef->getParticleData()),
            exec_conf(m_pdata->getExecConf()),
            m_mpi_comm(mpi_comm),
            m_dim(dim),
            m_global_box(global_box),
            m_is_allocated(false),
            m_r_ghost(Scalar(0.0))
    {
    // initialize array of neighbor processor ids
    if (neighbor_rank.size() != 6)
        {
        //! Set the rank of a neighbor processors of this simulation box in
        cerr << endl << "***Error! Invalid number of neighbor processor ranks supplied (" << neighbor_rank.size() << " != 6)."  << endl
                     << "          One processor rank per direction is required."
                     << endl << endl;
        throw runtime_error("Error initializing MPI communication.");
        }

    for (unsigned int dir = 0; dir < 6; dir++)
        m_neighbors[dir] = neighbor_rank[dir];

    m_packed_size = sizeof(pdata_element);

    allocate();
    }

//! Allocate internal buffers
void Communicator::allocate()
    {
    // the size of the data element may be different between CPU and GPU. It is just
    // used for allocation of the buffers
    unsigned int buf_size = m_pdata->getN() * m_packed_size;

    for (int dir = 0; dir < 6; dir++)
        {
        GPUArray<char> sendbuf(buf_size, exec_conf);
        m_sendbuf[dir].swap(sendbuf);
        }

    // the number of particles in this domain may not be sufficient for the receive buffer buf_size
    for (int dir = 0; dir < 6; dir++)
        {
        GPUArray<char> recvbuf(buf_size, exec_conf);
        m_recvbuf[dir].swap(recvbuf);
        }

    GPUArray<unsigned int> delete_buf(m_pdata->getN(), exec_conf);
    m_delete_buf.swap(delete_buf);

    for (unsigned int dir = 0; dir < 6; dir++)
        {
        m_max_copy_ghosts[dir] = m_pdata->getN(); // initial value
        GPUArray<unsigned int> copy_ghosts(m_max_copy_ghosts[dir], exec_conf);
        m_copy_ghosts[dir].swap(copy_ghosts);

        GPUArray<Scalar4> pos_copybuf(m_max_copy_ghosts[dir], exec_conf);
        m_pos_copybuf[dir].swap(pos_copybuf);

        GPUArray<Scalar> charge_copybuf(m_max_copy_ghosts[dir], exec_conf);
        m_charge_copybuf[dir].swap(charge_copybuf);

        GPUArray<Scalar> diameter_copybuf(m_max_copy_ghosts[dir], exec_conf);
        m_diameter_copybuf[dir].swap(diameter_copybuf);

        }

    m_is_allocated = true;
    }

//! Transfer particles between neighboring domains
void Communicator::migrateAtoms()
    {
    if (m_prof)
        m_prof->push("migrate_atoms");

    if (! m_is_allocated)
        allocate();

    // get box dimensions
    const BoxDim& box = m_pdata->getBox();

    unsigned int n_delete_ptls = 0;

    unsigned int num_recv_particles[6]; // per-direction number of particles received

    unsigned int num_send_particles[6]; // number of particles to send in positive direction

    // first step: determine particles that are going to be deleted
    bool send_x = getDimension(0) > 1;
    bool send_y = getDimension(1) > 1;
    bool send_z = getDimension(2) > 1;

    if (send_x || send_y || send_z) // trivial check if we are sending to someone at all
        {
        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_delete_buf(m_delete_buf, access_location::host, access_mode::readwrite);
        for (unsigned int idx = 0; idx < m_pdata->getN(); idx++)
            {
            Scalar4 pos = h_pos.data[idx];
            // determine whether particle has left the boundaries
            if ((send_x && pos.x >= box.xhi) || // send east
                (send_x && pos.x < box.xlo)  || // send west
                (send_y && pos.y >= box.yhi) || // send north
                (send_y && pos.y < box.ylo)  || // send south
                (send_z && pos.z >= box.zhi) || // send up
                (send_z && pos.z < box.zlo))    // send down
                {
                // add to delete buffer
                h_delete_buf.data[n_delete_ptls++] = idx;
                }
            }
        }

    // second step: determine local particles that are to be sent to neighboring processors and fill send buffer
    for (unsigned int dir=0; dir < 6; dir++)
        {
        num_send_particles[dir] = 0;

        if (getDimension(dir/2) == 1) continue;

            if (m_prof)
                m_prof->push("pack");

            {
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::read);
            ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
            ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_body(m_pdata->getBodies(), access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_global_tag(m_pdata->getGlobalTags(), access_location::host, access_mode::read);

            // FIXME: need to resize send buffer if necessary
            ArrayHandle<char> h_sendbuf(m_sendbuf[dir], access_location::host, access_mode::overwrite);


            for (unsigned int idx = 0; idx < m_pdata->getN(); idx++)
                {
                Scalar4 pos = h_pos.data[idx];

                // determine whether particle has left the boundaries
                if ((dir == 0 && pos.x >= box.xhi) || // send east
                    (dir == 1 && pos.x < box.xlo)  || // send west
                    (dir == 2 && pos.y >= box.yhi) || // send north
                    (dir == 3 && pos.y < box.ylo)  || // send south
                    (dir == 4 && pos.z >= box.zhi) || // send up
                    (dir == 5 && pos.z < box.zlo)) // send down
                    {
                    // pack particle data
                    pdata_element p;
                    p.pos = h_pos.data[idx];
                    p.vel = h_vel.data[idx];
                    p.accel = h_accel.data[idx];
                    p.charge = h_charge.data[idx];
                    p.diameter = h_diameter.data[idx];
                    p.image = h_image.data[idx];
                    p.body = h_body.data[idx];
                    p.orientation = h_orientation.data[idx];
                    p.global_tag = h_global_tag.data[idx];

                    ( (pdata_element *) h_sendbuf.data)[num_send_particles[dir]++] = p;
                    }
                }
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
            m_prof->push("forward ptls");

        // go through received data and determine particles that need to included in the next send and add them
        // to the message buffer
        for (unsigned int dirj = 0; dirj < dir ; dirj++)
            {
            unsigned int dimj = getDimension(dirj/2);
            if (dimj == 1) continue;
                {
                ArrayHandle<char> h_sendbuf(m_sendbuf[dir], access_location::host, access_mode::readwrite);
                ArrayHandle<char> h_recvbuf(m_recvbuf[dirj], access_location::host, access_mode::read);

                for (unsigned int idx = 0; idx < num_recv_particles[dirj]; idx++)
                    {
                    pdata_element &p = ((pdata_element *) h_recvbuf.data)[idx];
                    Scalar4 pos = p.pos;

                    if ((dir==2 && (pos.y >= box.yhi)) || // send north
                        (dir==3 && (pos.y < box.ylo)) ||  // send south
                        (dir==4 && (pos.z >= box.zhi)) || // send up
                        (dir==5 && (pos.z < box.zlo)))    // send down
                        {
                        // send with next message
                        pdata_element &p_send = ((pdata_element *) h_sendbuf.data)[num_send_particles[dir]++];
                        p_send = p;
                        }
                    }
                }
            }

        if (m_prof)
            m_prof->pop();

        if (m_prof)
            m_prof->push("MPI send/recv");

        // communicate size of the message that will contain the particle data
        boost::mpi::request reqs[2];
        reqs[0] = m_mpi_comm->isend(send_neighbor,0,num_send_particles[dir]);
        reqs[1] = m_mpi_comm->irecv(recv_neighbor,0,num_recv_particles[dir]);
        boost::mpi::wait_all(reqs,reqs+2);

            {
            ArrayHandle<char> h_sendbuf(m_sendbuf[dir], access_location::host, access_mode::read);
            ArrayHandle<char> h_recvbuf(m_recvbuf[dir], access_location::host, access_mode::overwrite);
            // exchange actual particle data
            reqs[0] = m_mpi_comm->isend(send_neighbor,1,h_sendbuf.data,num_send_particles[dir]*m_packed_size);
            reqs[1] = m_mpi_comm->irecv(recv_neighbor,1,h_recvbuf.data,num_recv_particles[dir]*m_packed_size);
            boost::mpi::wait_all(reqs,reqs+2);
            }

       if (m_prof)
          m_prof->pop();

            {
            ArrayHandle<char> h_recvbuf(m_recvbuf[dir], access_location::host, access_mode::readwrite);
            for (unsigned int idx = 0; idx < num_recv_particles[dir]; idx++)
                {
                pdata_element& p = ((pdata_element *) h_recvbuf.data)[idx];
                Scalar4& pos = p.pos;
                int3& image = p.image;

                // wrap received particles across a global boundary back into global box
                if (dir == 0 && pos.x >= m_global_box.xhi)
                    {
                    pos.x -= m_global_box.xhi - m_global_box.xlo;
                    image.x++;
                    }
                else if (dir == 1 && pos.x < m_global_box.xlo)
                    {
                    pos.x += m_global_box.xhi - m_global_box.xlo;
                    image.x--;
                    }

                if (dir == 2 && pos.y >= m_global_box.yhi)
                    {
                    pos.y -= m_global_box.yhi - m_global_box.ylo;
                    image.y++;
                    }
                else if (dir == 3 && pos.y < m_global_box.ylo)
                    {
                    pos.y += m_global_box.yhi - m_global_box.ylo;
                    image.y--;
                    }

                if (dir == 4 && pos.z >= m_global_box.zhi)
                    {
                    pos.z -= m_global_box.zhi - m_global_box.zlo;
                    image.z++;
                    }
                else if (dir == 5 && pos.z < m_global_box.zlo)
                    {
                    pos.z += m_global_box.zhi - m_global_box.zlo;
                    image.z--;
                    }

                assert( ((dir==0 || dir ==1) && m_global_box.xlo <= pos.x && pos.x < m_global_box.xhi) ||
                        ((dir==2 || dir ==3) && m_global_box.ylo <= pos.y && pos.y < m_global_box.yhi) ||
                        ((dir==4 || dir ==5) && m_global_box.zlo <= pos.z && pos.z < m_global_box.zhi ));
                }
            }

        } // end dir loop


        {
        // remove deleted atoms
        /* this algorithm removes the particles with indices in h_delete_buf from the particle data arrays
         * and fills any holes created in this way by moving the last particle up to the location of the deleted particle.
         *
         * NOTE: The present algorithm does not preserve the local sorted order. This may be OK on the CPU, though.
         */
        ArrayHandle<unsigned int> h_delete_buf(m_delete_buf, access_location::host, access_mode::read);

        ArrayHandle< Scalar4 > h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
        ArrayHandle< Scalar4 > h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
        ArrayHandle< Scalar3 > h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::readwrite);
        ArrayHandle< int3 > h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);
        ArrayHandle< Scalar > h_charge(m_pdata->getCharges(), access_location::host, access_mode::readwrite);
        ArrayHandle< Scalar > h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::readwrite);
        ArrayHandle< unsigned int > h_body(m_pdata->getBodies(), access_location::host, access_mode::readwrite);
        ArrayHandle< Scalar4 > h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);
        ArrayHandle< unsigned int > h_global_tag(m_pdata->getGlobalTags(), access_location::host, access_mode::read);
        ArrayHandle< unsigned int > h_global_rtag(m_pdata->getGlobalRTags(), access_location::host, access_mode::read);

        for (unsigned int i = 0; i < n_delete_ptls; i++)
            {
            unsigned int idx = h_delete_buf.data[i];
            assert(idx < m_pdata->getN());
            unsigned int last_idx = m_pdata->getN() - 1;

            // move the last particle up to the hole created in the particle data arrays
            h_pos.data[idx] = h_pos.data[last_idx];
            h_vel.data[idx] = h_vel.data[last_idx];
            h_accel.data[idx] = h_accel.data[last_idx];
            h_image.data[idx] = h_image.data[last_idx];
            h_charge.data[idx] = h_charge.data[last_idx];
            h_diameter.data[idx] = h_diameter.data[last_idx];
            h_body.data[idx] = h_body.data[last_idx];
            h_global_tag.data[idx] = h_global_tag.data[last_idx];

            // update global tag reverse lookup for moved particle
            assert(h_global_rtag.data[h_global_tag.data[last_idx]] == last_idx);
            h_global_rtag.data[h_global_tag.data[last_idx]] = idx;

            // also need to change subsequent references in the delete buffer
            for (unsigned int j=i+1; j < n_delete_ptls; j++)
                if (h_delete_buf.data[j] == last_idx) h_delete_buf.data[j] = idx;

            // update number of particles in system
            m_pdata->removeParticles(1);
            }
       }


    // finally, add particles to box

    // first step: count how many particles will be added to this simulation box
    unsigned int num_add_particles = 0;

    for (int dir=0; dir < 6; dir++)
        {
        unsigned int dim = getDimension(dir/2);
        if (dim == 1) continue;

            {
            ArrayHandle<char> h_recvbuf(m_recvbuf[dir], access_location::host, access_mode::read);
            for (unsigned int idx = 0; idx < num_recv_particles[dir]; idx++)
                {
                pdata_element& p = ((pdata_element *) h_recvbuf.data)[idx];
                Scalar4 pos = p.pos;

                // check if particle lies in this box
                if ((box.xlo <= pos.x  && pos.x < box.xhi) &&
                    (box.ylo <= pos.y  && pos.y < box.yhi) &&
                    (box.zlo <= pos.z  && pos.z < box.zhi))
                    {
                    num_add_particles++;
                    }
                }
            }
        }

    if (num_add_particles)
        {

        // start index for atoms to be added
        unsigned int add_idx = m_pdata->getN();

        // add particles that have migrated to this domain
        m_pdata->addParticles(num_add_particles);

        // go through receive buffers and update local particle data
        for (int dir=0; dir < 6; dir++)
            {
            unsigned int dim = getDimension(dir/2);
            if (dim == 1) continue;

                {
                ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
                ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
                ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::readwrite);
                ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::readwrite);
                ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::readwrite);
                ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);
                ArrayHandle<unsigned int> h_body(m_pdata->getBodies(), access_location::host, access_mode::readwrite);
                ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);
                ArrayHandle<unsigned int> h_global_tag(m_pdata->getGlobalTags(), access_location::host, access_mode::readwrite);
                ArrayHandle<unsigned int> h_global_rtag(m_pdata->getGlobalRTags(), access_location::host, access_mode::readwrite);


                ArrayHandle<char> h_recvbuf(m_recvbuf[dir], access_location::host, access_mode::read);
                for (unsigned int idx = 0; idx < num_recv_particles[dir]; idx++)
                    {
                    pdata_element& p =  ((pdata_element *) h_recvbuf.data)[idx];
                    const Scalar4& pos = p.pos;
                    if ((box.xlo <= pos.x  && pos.x < box.xhi) &&
                        (box.ylo <= pos.y  && pos.y < box.yhi) &&
                        (box.zlo <= pos.z  && pos.z < box.zhi))
                        {
                        // copy over particle coordinates to domain
                        h_pos.data[add_idx] = p.pos;
                        h_vel.data[add_idx] = p.vel;
                        h_accel.data[add_idx] = p.accel;
                        h_charge.data[add_idx] = p.charge;
                        h_diameter.data[add_idx] = p.diameter;
                        h_image.data[add_idx] = p.image;
                        h_body.data[add_idx] = p.body;
                        h_orientation.data[add_idx] = p.orientation;
                        h_global_tag.data[add_idx] = p.global_tag;
                        h_global_rtag.data[p.global_tag] = add_idx;

                        // update information that the particle is now local to this processor
                        add_idx++;
                        }
                    }
                }
            }
        }

    // check that global number of particles is conserved
    unsigned int N;
    reduce(*m_mpi_comm,m_pdata->getN(), N, std::plus<unsigned int>(), 0);
    if (m_mpi_comm->rank() == 0 && N != m_pdata->getNGlobal())
        {
        cerr << endl << "***Error! Global number of particles has changed unexpectedly." << endl << endl;
        throw runtime_error("Error in MPI communication.");
        }


    // notify ParticleData that addition / removal of particles is complete
    if (m_prof)
        m_prof->push("group update");

    if (n_delete_ptls || num_add_particles)
        m_pdata->notifyParticleNumberChange();

    if (m_prof)
        m_prof->pop();

    if (m_prof)
        m_prof->pop();
    }

//! build ghost particle list, copy ghost particle data
void Communicator::exchangeGhosts(Scalar r_ghost)
    {
    if (m_prof)
        m_prof->push("exchange_ghosts");

    const BoxDim& box = m_pdata->getBox();

    assert(r_ghost < (box.xhi - box.xlo));
    assert(r_ghost < (box.yhi - box.ylo));
    assert(r_ghost < (box.zhi - box.zlo));

    m_r_ghost = r_ghost;

    // we have a current list of atoms inside this box
    // find all local atoms within a distance r_ghost from the boundary and store them in m_copy_ghosts

    // first clear all ghost particles
    m_pdata->removeAllGhostParticles();

    for (unsigned int dir = 0; dir < 6; dir ++)
        {
        unsigned int dim = getDimension(dir/2);
        if (dim == 1) continue;

        m_num_copy_ghosts[dir] = 0;


        // scan all atom positions if they are within r_ghost from a neighbor, fill send buffer

            {
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_global_tag(m_pdata->getGlobalTags(), access_location::host, access_mode::read);

            ArrayHandle<unsigned int> h_copy_ghosts(m_copy_ghosts[dir], access_location::host, access_mode::overwrite);
            ArrayHandle<Scalar4> h_pos_copybuf(m_pos_copybuf[dir], access_location::host, access_mode::overwrite);
            ArrayHandle<Scalar> h_charge_copybuf(m_charge_copybuf[dir], access_location::host, access_mode::overwrite);
            ArrayHandle<Scalar> h_diameter_copybuf(m_diameter_copybuf[dir], access_location::host, access_mode::overwrite);

            for (unsigned int idx = 0; idx < m_pdata->getN(); idx++)
                {
                Scalar4 pos = h_pos.data[idx];

                if ((dir==0 && (pos.x >= box.xhi - r_ghost)) ||                      // send east
                    (dir==1 && (pos.x < box.xlo + r_ghost) && (pos.x >= box.xlo)) || // send west
                    (dir==2 && (pos.y >= box.yhi - r_ghost)) ||                      // send north
                    (dir==3 && (pos.y < box.ylo + r_ghost) && (pos.y >= box.ylo)) || // send south
                    (dir==4 && (pos.z >= box.zhi - r_ghost)) ||                      // send up
                    (dir==5 && (pos.z < box.zlo + r_ghost) && (pos.z >= box.zlo)))   // send down
                    {
                    // send with next message
                    h_pos_copybuf.data[m_num_copy_ghosts[dir]] = pos;
                    h_charge_copybuf.data[m_num_copy_ghosts[dir]] = h_charge.data[idx];
                    h_diameter_copybuf.data[m_num_copy_ghosts[dir]] = h_diameter.data[idx];

                    h_copy_ghosts.data[m_num_copy_ghosts[dir]] = h_global_tag.data[idx];
                    m_num_copy_ghosts[dir]++;
                    }
                }
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
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_global_tag(m_pdata->getGlobalTags(), access_location::host, access_mode::read);

            ArrayHandle<unsigned int> h_copy_ghosts(m_copy_ghosts[dir], access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_pos_copybuf(m_pos_copybuf[dir], access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar> h_charge_copybuf(m_charge_copybuf[dir], access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar> h_diameter_copybuf(m_diameter_copybuf[dir], access_location::host, access_mode::readwrite);

            // scan all ghost particles if they are within r_ghost from a neighbor, fill send buffer
            for (unsigned int idx = m_pdata->getN(); idx < m_pdata->getN() + m_pdata->getNGhosts(); idx++)
                {
                Scalar4 pos = h_pos.data[idx];

                if ((dir==0 && (pos.x >= box.xhi - r_ghost)) ||                      // send east
                    (dir==1 && (pos.x < box.xlo + r_ghost) && (pos.x >= box.xlo)) || // send west
                    (dir==2 && (pos.y >= box.yhi - r_ghost)) ||                      // send north
                    (dir==3 && (pos.y < box.ylo + r_ghost) && (pos.y >= box.ylo)) || // send south
                    (dir==4 && (pos.z >= box.zhi - r_ghost)) ||                      // send up
                    (dir==5 && (pos.z < box.zlo + r_ghost) && (pos.z >= box.zlo)))   // send down
                    {
                    // send with next message
                    h_pos_copybuf.data[m_num_copy_ghosts[dir]] = pos;
                    h_charge_copybuf.data[m_num_copy_ghosts[dir]] = h_charge.data[idx];
                    h_diameter_copybuf.data[m_num_copy_ghosts[dir]] = h_diameter.data[idx];
                    h_copy_ghosts.data[m_num_copy_ghosts[dir]] = h_global_tag.data[idx];
                    m_num_copy_ghosts[dir]++;
                    }

                 }
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
        if (m_prof)
            m_prof->pop();

            {
            ArrayHandle<unsigned int> h_global_tag(m_pdata->getGlobalTags(), access_location::host, access_mode::readwrite);
            ArrayHandle<unsigned int> h_global_rtag(m_pdata->getGlobalRTags(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
            for (unsigned int idx = start_idx; idx < start_idx + m_num_recv_ghosts[dir]; idx++)
                {
                Scalar4& pos = h_pos.data[idx];

                // wrap particles received across a global boundary back into global box
                if (dir==0 && pos.x >= m_global_box.xhi - r_ghost)
                    pos.x -= m_global_box.xhi - m_global_box.xlo;
                else if (dir==1 && pos.x < m_global_box.xlo + r_ghost)
                    pos.x += m_global_box.xhi - m_global_box.xlo;
                else if (dir==2 && pos.y >= m_global_box.yhi - r_ghost)
                    pos.y -= m_global_box.yhi - m_global_box.ylo;
                else if (dir==3 && pos.y < m_global_box.ylo + r_ghost)
                    pos.y += m_global_box.yhi - m_global_box.ylo;
                else if (dir==4 && pos.z >= m_global_box.zhi - r_ghost)
                    pos.z -= m_global_box.zhi - m_global_box.zlo;
                else if (dir==5 && pos.z < m_global_box.zlo + r_ghost)
                    pos.z += m_global_box.zhi - m_global_box.zlo;

                h_global_rtag.data[h_global_tag.data[idx]] = idx;
                }
            }
        }

    if (m_prof)
        m_prof->pop();
    }

//! update positions of ghost particles
void Communicator::copyGhosts()
    {
    // we have a current m_copy_ghosts liss which contain the indices of particles
    // to send to neighboring processors

    if (m_prof)
        m_prof->push("copy_ghosts");

    // update data in these arrays

    unsigned int num_tot_recv_ghosts = 0; // total number of ghosts received

    for (unsigned int dir = 0; dir < 6; dir ++)
        {

        unsigned int dim = getDimension(dir/2);
        if (dim == 1) continue;

            {
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);

            ArrayHandle<Scalar4> h_pos_copybuf(m_pos_copybuf[dir], access_location::host, access_mode::overwrite);

            ArrayHandle<unsigned int> h_copy_ghosts(m_copy_ghosts[dir], access_location::host, access_mode::read);

            if (m_prof)
                m_prof->push("fetch ptls");

            // copy positions of ghost particles
            for (unsigned int ghost_idx = 0; ghost_idx < m_num_copy_ghosts[dir]; ghost_idx++)
                {
                unsigned int idx = m_pdata->getGlobalRTag(h_copy_ghosts.data[ghost_idx]);

                // copy position, charge and diameter into send buffer
                h_pos_copybuf.data[ghost_idx] = h_pos.data[idx];
                }

            if (m_prof)
                m_prof->pop();
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


            {
            boost::mpi::request reqs[2];

            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_pos_copybuf(m_pos_copybuf[dir], access_location::host, access_mode::read);

            // exchange particle data, write directly to the particle data arrays
            reqs[0] = m_mpi_comm->isend(send_neighbor,1,h_pos_copybuf.data, m_num_copy_ghosts[dir]);
            reqs[1] = m_mpi_comm->irecv(recv_neighbor,1,h_pos.data + start_idx, m_num_recv_ghosts[dir]);
            boost::mpi::wait_all(reqs,reqs+2);
            }


        if (m_prof)
            m_prof->pop(0, (m_num_recv_ghosts[dir]+m_num_copy_ghosts[dir])*sizeof(Scalar4));
        }

        if (m_prof)
            m_prof->push("particle wrap");

             {
             ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);

             for (unsigned int idx = start_idx; idx < start_idx + m_num_recv_ghosts[dir]; idx++)
                 {
                 Scalar4& pos = h_pos.data[idx];

                 // wrap particles received across a global boundary back into global box
                 if (dir==0 && pos.x >= m_global_box.xhi - m_r_ghost)
                     pos.x -= m_global_box.xhi - m_global_box.xlo;
                 else if (dir==1 && pos.x < m_global_box.xlo + m_r_ghost)
                     pos.x += m_global_box.xhi - m_global_box.xlo;
                 else if (dir==2 && pos.y >= m_global_box.yhi - m_r_ghost)
                     pos.y -= m_global_box.yhi - m_global_box.ylo;
                 else if (dir==3 && pos.y < m_global_box.ylo + m_r_ghost)
                     pos.y += m_global_box.yhi - m_global_box.ylo;
                 else if (dir==4 && pos.z >= m_global_box.zhi - m_r_ghost)
                     pos.z -= m_global_box.zhi - m_global_box.zlo;
                 else if (dir==5 && pos.z < m_global_box.zlo + m_r_ghost)
                     pos.z += m_global_box.zhi - m_global_box.zlo;
                 }
            }

        if (m_prof)
            m_prof->pop();
        } // end dir loop

        if (m_prof)
            m_prof->pop();
    }

//! Export Communicator class to python
void export_Communicator()
    {
    class_<Communicator, boost::shared_ptr<Communicator>, boost::noncopyable>("Communicator",
           init<boost::shared_ptr<SystemDefinition>,
                boost::shared_ptr<boost::mpi::communicator>,
                std::vector<unsigned int>,
                int3,
                const BoxDim &>())
    ;
    }
#endif // ENABLE_MPI
