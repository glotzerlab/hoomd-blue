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
#include <algorithm>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

using namespace boost::python;

//! Define some of our types as fixed-size MPI datatypes for performance optimization
BOOST_IS_MPI_DATATYPE(Scalar4)
BOOST_IS_MPI_DATATYPE(Scalar3)
BOOST_IS_MPI_DATATYPE(uint3)
BOOST_IS_MPI_DATATYPE(int3)


//! Select a particle for migration
struct select_particle_migrate : public std::unary_function<const unsigned int&, bool>
    {
    const float xlo;        //!< Lower x boundary
    const float xhi;        //!< Upper x boundary
    const float ylo;        //!< Lower y boundary
    const float yhi;        //!< Upper y boundary
    const float zlo;        //!< Lower z boundary
    const float zhi;        //!< Upper z boundary
    const unsigned int dir; //!< Direction to send particles to
    const Scalar4 *h_pos;   //!< Array of particle positions


    //! Constructor
    /*!
     */
    select_particle_migrate(const float _xlo,
                            const float _xhi,
                            const float _ylo,
                            const float _yhi,
                            const float _zlo,
                            const float _zhi,
                            const unsigned int _dir,
                            const Scalar4 *_h_pos)
        : xlo(_xlo), xhi(_xhi), ylo(_ylo), yhi(_yhi), zlo(_zlo), zhi(_zhi), dir(_dir), h_pos(_h_pos)
        {
        }

    //! Select a particle
    /*! t particle data to consider for sending
     * \return true if particle stays in the box
     */
    __host__ __device__ bool operator()(const unsigned int& idx)
        {
        const Scalar4& pos = h_pos[idx];
        // we return true if the particle stays in our box,
        // false otherwise
        return !((dir == 0 && pos.x >= xhi) ||  // send east
                (dir == 1 && pos.x < xlo)  ||  // send west
                (dir == 2 && pos.y >= yhi) ||  // send north
                (dir == 3 && pos.y < ylo)  ||  // send south
                (dir == 4 && pos.z >= zhi) ||  // send up
                (dir == 5 && pos.z < zlo ));   // send down
        }

     };



//! Constructor
Communicator::Communicator(boost::shared_ptr<SystemDefinition> sysdef,
                           boost::shared_ptr<boost::mpi::communicator> mpi_comm,
                           std::vector<unsigned int> neighbor_rank,
                           std::vector<bool> is_at_boundary,
                           uint3 dim,
                           const BoxDim global_box)
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
    assert(neighbor_rank.size() == 6);
    assert(is_at_boundary.size() == 6);

    for (unsigned int dir = 0; dir < 6; dir++)
        {
        m_neighbors[dir] = neighbor_rank[dir];
        m_is_at_boundary[dir] = is_at_boundary[dir];
        }

    m_packed_size = sizeof(pdata_element);

    allocate();

    for (unsigned int dir = 0; dir < 6; dir ++)
        {
        m_num_copy_ghosts[dir] = 0;
        m_num_recv_ghosts[dir] = 0;
        }
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

    // wipe out reverse-lookup tag -> idx for old ghost atoms
        {
        ArrayHandle<unsigned int> h_global_tag(m_pdata->getGlobalTags(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_global_rtag(m_pdata->getGlobalRTags(), access_location::host, access_mode::readwrite);
        for (unsigned int i = 0; i < m_pdata->getNGhosts(); i++)
            {
            unsigned int idx = m_pdata->getN() + i;
            h_global_rtag.data[h_global_tag.data[idx]] = NOT_LOCAL;
            }
        }

    //  reset ghost particle number
    m_pdata->removeAllGhostParticles();


    // get box dimensions
    const BoxDim& box = m_pdata->getBox();

    // determine local particles that are to be sent to neighboring processors and fill send buffer
    for (unsigned int dir=0; dir < 6; dir++)
        {

        if (getDimension(dir/2) == 1) continue;

            if (m_prof)
                m_prof->push("remove ptls");

        unsigned int n_send_ptls;
            {
            // first remove all particles from our domain that are going to be sent in the current direction
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::readwrite);
            ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);
            ArrayHandle<unsigned int> h_body(m_pdata->getBodies(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);
            ArrayHandle<unsigned int> h_global_tag(m_pdata->getGlobalTags(), access_location::host, access_mode::readwrite);

            /* Reorder particles.
               Particles that stay in our domain come first, followed by the particles that are sent to a
               neighboring processor.
             */

            // Fill key vector with indices 0...N-1
            std::vector<unsigned int> sort_keys(m_pdata->getN());
            for (unsigned int i = 0; i < m_pdata->getN(); i++)
                sort_keys[i] = i;

            // partition the keys according to the particle positions corresponding to the indices
            std::vector<unsigned int>::iterator sort_keys_middle;
            sort_keys_middle = std::stable_partition(sort_keys.begin(),
                                                 sort_keys.begin() + m_pdata->getN(),
                                                 select_particle_migrate(box.xlo, box.xhi, box.ylo, box.yhi, box.zlo, box.zhi, dir,
                                                                        h_pos.data));

            n_send_ptls = (sort_keys.begin() + m_pdata->getN()) - sort_keys_middle;

            // reorder the particle data
            Scalar4 *scal4_tmp = new Scalar4[m_pdata->getN()];
            Scalar3 *scal3_tmp = new Scalar3[m_pdata->getN()];
            Scalar *scal_tmp = new Scalar[m_pdata->getN()];
            unsigned int *uint_tmp = new unsigned int[m_pdata->getN()];
            int3 *int3_tmp = new int3[m_pdata->getN()];

            for (unsigned int i = 0; i < m_pdata->getN(); i++)
                scal4_tmp[i] = h_pos.data[sort_keys[i]];
            for (unsigned int i = 0; i < m_pdata->getN(); i++)
                h_pos.data[i] = scal4_tmp[i];

            for (unsigned int i = 0; i < m_pdata->getN(); i++)
                scal4_tmp[i] = h_vel.data[sort_keys[i]];
            for (unsigned int i = 0; i < m_pdata->getN(); i++)
                h_vel.data[i] = scal4_tmp[i];

            for (unsigned int i = 0; i < m_pdata->getN(); i++)
                scal3_tmp[i] = h_accel.data[sort_keys[i]];
            for (unsigned int i = 0; i < m_pdata->getN(); i++)
                h_accel.data[i] = scal3_tmp[i];

            for (unsigned int i = 0; i < m_pdata->getN(); i++)
                scal_tmp[i] = h_charge.data[sort_keys[i]];
            for (unsigned int i = 0; i < m_pdata->getN(); i++)
                h_charge.data[i] = scal_tmp[i];

            for (unsigned int i = 0; i < m_pdata->getN(); i++)
                scal_tmp[i] = h_diameter.data[sort_keys[i]];
            for (unsigned int i = 0; i < m_pdata->getN(); i++)
                h_diameter.data[i] = scal_tmp[i];

            for (unsigned int i = 0; i < m_pdata->getN(); i++)
                int3_tmp[i] = h_image.data[sort_keys[i]];
            for (unsigned int i = 0; i < m_pdata->getN(); i++)
                h_image.data[i] = int3_tmp[i];

            for (unsigned int i = 0; i < m_pdata->getN(); i++)
                scal4_tmp[i] = h_orientation.data[sort_keys[i]];
            for (unsigned int i = 0; i < m_pdata->getN(); i++)
                h_orientation.data[i] = scal4_tmp[i];

            for (unsigned int i = 0; i < m_pdata->getN(); i++)
                uint_tmp[i] = h_body.data[sort_keys[i]];
            for (unsigned int i = 0; i < m_pdata->getN(); i++)
                h_body.data[i] = uint_tmp[i];

            for (unsigned int i = 0; i < m_pdata->getN(); i++)
                uint_tmp[i] = h_global_tag.data[sort_keys[i]];
            for (unsigned int i = 0; i < m_pdata->getN(); i++)
                h_global_tag.data[i] = uint_tmp[i];

            delete[] scal4_tmp;
            delete[] scal3_tmp;
            delete[] scal_tmp;
            delete[] int3_tmp;
            delete[] uint_tmp;
            }

        // remove particles from local data that are being sent
        m_pdata->removeParticles(n_send_ptls);

            {
            // update reverse lookup tags
            ArrayHandle<unsigned int> h_global_tag(m_pdata->getGlobalTags(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_global_rtag(m_pdata->getGlobalRTags(), access_location::host, access_mode::readwrite);
            for (unsigned int idx = 0; idx < m_pdata->getN(); idx++)
                {
                h_global_rtag.data[h_global_tag.data[idx]] = idx;
                }
            }


            // resize send buffer if necessary
            if (n_send_ptls*m_packed_size > m_sendbuf[dir].getNumElements())
                {
                unsigned int new_size = m_sendbuf[dir].getNumElements();
                while (new_size < n_send_ptls * m_packed_size) new_size *= 2;
                m_sendbuf[dir].resize(new_size);
                }

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

            ArrayHandle<unsigned int> h_global_rtag(m_pdata->getGlobalRTags(), access_location::host, access_mode::readwrite);

            ArrayHandle<char> h_sendbuf(m_sendbuf[dir], access_location::host, access_mode::overwrite);

            for (unsigned int i = 0;  i<  n_send_ptls; i++)
                {
                unsigned int idx = m_pdata->getN() + i;

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

                // Reset the global rtag for the particle we are sending to indicate it is no longer local
                assert(h_global_rtag.data[h_global_tag.data[idx]] < m_pdata->getN() + n_send_ptls);
                h_global_rtag.data[h_global_tag.data[idx]] = NOT_LOCAL;

                ( (pdata_element *) h_sendbuf.data)[i] = p;
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
            m_prof->push("MPI send/recv");

        unsigned int n_recv_ptls;

        // communicate size of the message that will contain the particle data
        boost::mpi::request reqs[2];
        reqs[0] = m_mpi_comm->isend(send_neighbor,0,n_send_ptls);
        reqs[1] = m_mpi_comm->irecv(recv_neighbor,0,n_recv_ptls);
        boost::mpi::wait_all(reqs,reqs+2);

            {
            ArrayHandle<char> h_sendbuf(m_sendbuf[dir], access_location::host, access_mode::read);
            ArrayHandle<char> h_recvbuf(m_recvbuf[dir], access_location::host, access_mode::overwrite);
            // exchange actual particle data
            reqs[0] = m_mpi_comm->isend(send_neighbor,1,h_sendbuf.data,n_send_ptls*m_packed_size);
            reqs[1] = m_mpi_comm->irecv(recv_neighbor,1,h_recvbuf.data,n_recv_ptls*m_packed_size);
            boost::mpi::wait_all(reqs,reqs+2);
            }

       if (m_prof)
          m_prof->pop();

            {
            // wrap received particles across a global boundary back into global box
            ArrayHandle<char> h_recvbuf(m_recvbuf[dir], access_location::host, access_mode::readwrite);
            for (unsigned int idx = 0; idx < n_recv_ptls; idx++)
                {
                pdata_element& p = ((pdata_element *) h_recvbuf.data)[idx];
                Scalar4& pos = p.pos;
                int3& image = p.image;

                if (dir == 0 && m_is_at_boundary[1])
                    {
                    pos.x -= m_global_box.xhi - m_global_box.xlo;
                    image.x++;
                    }
                else if (dir == 1 && m_is_at_boundary[0])
                    {
                    pos.x += m_global_box.xhi - m_global_box.xlo;
                    image.x--;
                    }

                if (dir == 2 && m_is_at_boundary[3])
                    {
                    pos.y -= m_global_box.yhi - m_global_box.ylo;
                    image.y++;
                    }
                else if (dir == 3 && m_is_at_boundary[2])
                    {
                    pos.y += m_global_box.yhi - m_global_box.ylo;
                    image.y--;
                    }

                if (dir == 4 && m_is_at_boundary[5])
                    {
                    pos.z -= m_global_box.zhi - m_global_box.zlo;
                    image.z++;
                    }
                else if (dir == 5 && m_is_at_boundary[4])
                    {
                    pos.z += m_global_box.zhi - m_global_box.zlo;
                    image.z--;
                    }

                assert( ((dir==0 || dir ==1) && m_global_box.xlo-1e-3 <= pos.x && pos.x < m_global_box.xhi+1e-3) ||
                        ((dir==2 || dir ==3) && m_global_box.ylo-1e-3 <= pos.y && pos.y < m_global_box.yhi+1e-3) ||
                        ((dir==4 || dir ==5) && m_global_box.zlo-1e-3 <= pos.z && pos.z < m_global_box.zhi+1e-3 ));
                }
            }

        // start index for atoms to be added
        unsigned int add_idx = m_pdata->getN();

        // allocate memory for received particles
        m_pdata->addParticles(n_recv_ptls);

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
            for (unsigned int i = 0; i < n_recv_ptls; i++)
                {
                pdata_element& p =  ((pdata_element *) h_recvbuf.data)[i];

                // copy particle coordinates to domain
                h_pos.data[add_idx] = p.pos;
                h_vel.data[add_idx] = p.vel;
                h_accel.data[add_idx] = p.accel;
                h_charge.data[add_idx] = p.charge;
                h_diameter.data[add_idx] = p.diameter;
                h_image.data[add_idx] = p.image;
                h_body.data[add_idx] = p.body;
                h_orientation.data[add_idx] = p.orientation;
                h_global_tag.data[add_idx] = p.global_tag;

                assert(h_global_rtag.data[h_global_tag.data[add_idx]] == NOT_LOCAL);
                h_global_rtag.data[h_global_tag.data[add_idx]] = add_idx;
                add_idx++;
                }
            }
        } // end dir loop

    // check that global number of particles is conserved
    unsigned int N;
    reduce(*m_mpi_comm,m_pdata->getN(), N, std::plus<unsigned int>(), 0);
    if (m_mpi_comm->rank() == 0 && N != m_pdata->getNGlobal())
        {
        cerr << endl << "***Error! Global number of particles has changed unexpectedly." << endl << endl;
        throw runtime_error("Error in MPI communication.");
        }


    // notify ParticleData that addition / removal of particles is complete
    m_pdata->notifyLocalParticleNumChange();

    if (m_prof)
        m_prof->pop();
    }

//! Build ghost particle list, exchange ghost particle data
void Communicator::exchangeGhosts(Scalar r_ghost)
    {

    if (m_prof)
        m_prof->push("exchange_ghosts");

    const BoxDim& box = m_pdata->getBox();

    assert(r_ghost < (box.xhi - box.xlo));
    assert(r_ghost < (box.yhi - box.ylo));
    assert(r_ghost < (box.zhi - box.zlo));

    m_r_ghost = r_ghost;

    for (unsigned int dir = 0; dir < 6; dir ++)
        {
        unsigned int dim = getDimension(dir/2);
        if (dim == 1) continue;

        m_num_copy_ghosts[dir] = 0;


        // scan all local atom positions if they are within r_ghost from a neighbor, fill send buffer

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

                if ((dir==0 && (pos.x >= box.xhi - r_ghost)) ||  // send east
                    (dir==1 && (pos.x < box.xlo + r_ghost)) ||   // send west
                    (dir==2 && (pos.y >= box.yhi - r_ghost)) ||  // send north
                    (dir==3 && (pos.y < box.ylo + r_ghost)) ||   // send south
                    (dir==4 && (pos.z >= box.zhi - r_ghost)) ||  // send up
                    (dir==5 && (pos.z < box.zlo + r_ghost)))     // send down
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
            // add extra check that we are not sending back particles in the opposite direction from
            // which we have received them
            unsigned int omit_ghosts = ((dir % 2) == 1) ? m_num_recv_ghosts[dir-1] : 0;
            for (unsigned int idx = m_pdata->getN(); idx < m_pdata->getN() + m_pdata->getNGhosts() - omit_ghosts; idx++)
                {
                Scalar4 pos = h_pos.data[idx];

                if ((dir==0 && (pos.x >= box.xhi - r_ghost)) || // send east
                    (dir==1 && (pos.x < box.xlo + r_ghost)) ||  // send west
                    (dir==2 && (pos.y >= box.yhi - r_ghost)) || // send north
                    (dir==3 && (pos.y < box.ylo + r_ghost)) ||  // send south
                    (dir==4 && (pos.z >= box.zhi - r_ghost)) || // send up
                    (dir==5 && (pos.z < box.zlo + r_ghost)))    // send down
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
            ArrayHandle<unsigned int> h_global_tag(m_pdata->getGlobalTags(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_global_rtag(m_pdata->getGlobalRTags(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
            for (unsigned int idx = start_idx; idx < start_idx + m_num_recv_ghosts[dir]; idx++)
                {
                Scalar4& pos = h_pos.data[idx];

                // wrap particles received across a global boundary back into global box
                if (dir==0 && m_is_at_boundary[1])
                    pos.x -= m_global_box.xhi - m_global_box.xlo;
                else if (dir==1 && m_is_at_boundary[0])
                    pos.x += m_global_box.xhi - m_global_box.xlo;
                else if (dir==2 && m_is_at_boundary[3])
                    pos.y -= m_global_box.yhi - m_global_box.ylo;
                else if (dir==3 && m_is_at_boundary[2])
                    pos.y += m_global_box.yhi - m_global_box.ylo;
                else if (dir==4 && m_is_at_boundary[5])
                    pos.z -= m_global_box.zhi - m_global_box.zlo;
                else if (dir==5 && m_is_at_boundary[4])
                    pos.z += m_global_box.zhi - m_global_box.zlo;

                assert ( (dir == 0 && pos.x < box.xlo+1e-3 && pos.x >= box.xlo-r_ghost-1e-3) ||
                         (dir == 1 && pos.x >= box.xhi-1e-3 && pos.x < box.xhi+r_ghost+1e-3) ||
                         (dir == 2 && pos.y < box.ylo+1e-3 && pos.y >= box.ylo-r_ghost-1e-3) ||
                         (dir == 3 && pos.y >= box.yhi-1e-3 && pos.y < box.yhi+r_ghost+1e-3) ||
                         (dir == 4 && pos.z < box.zlo+1e-3 && pos.z >= box.zlo-r_ghost-1e-3) ||
                         (dir == 5 && pos.z >= box.zhi-1e-3 && pos.z < box.zhi+r_ghost+1e-3) );


                assert(h_global_rtag.data[h_global_tag.data[idx]] == NOT_LOCAL);
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
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_pos_copybuf(m_pos_copybuf[dir], access_location::host, access_mode::overwrite);
            ArrayHandle<unsigned int> h_copy_ghosts(m_copy_ghosts[dir], access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_global_rtag(m_pdata->getGlobalRTags(), access_location::host, access_mode::read);

            // copy positions of ghost particles
            for (unsigned int ghost_idx = 0; ghost_idx < m_num_copy_ghosts[dir]; ghost_idx++)
                {
                unsigned int idx = h_global_rtag.data[h_copy_ghosts.data[ghost_idx]];

                assert(idx < m_pdata->getN() + m_pdata->getNGhosts());

                // copy position into send buffer
                h_pos_copybuf.data[ghost_idx] = h_pos.data[idx];
                }
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

             {
             ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);

             for (unsigned int idx = start_idx; idx < start_idx + m_num_recv_ghosts[dir]; idx++)
                 {
                 Scalar4& pos = h_pos.data[idx];

                 // wrap particles received across a global boundary back into global box
                 if (dir==0 && m_is_at_boundary[1] )
                     pos.x -= m_global_box.xhi - m_global_box.xlo;
                 else if (dir==1 && m_is_at_boundary[0])
                     pos.x += m_global_box.xhi - m_global_box.xlo;
                 else if (dir==2 && m_is_at_boundary[3])
                     pos.y -= m_global_box.yhi - m_global_box.ylo;
                 else if (dir==3 && m_is_at_boundary[2])
                     pos.y += m_global_box.yhi - m_global_box.ylo;
                 else if (dir==4 && m_is_at_boundary[5])
                     pos.z -= m_global_box.zhi - m_global_box.zlo;
                 else if (dir==5 && m_is_at_boundary[4])
                     pos.z += m_global_box.zhi - m_global_box.zlo;

                 }
            }

        } // end dir loop

        if (m_prof)
            m_prof->pop();
    }

//! Export Communicator class to python
void export_Communicator()
    {
     class_< std::vector<bool> >("std_vector_bool")
    .def(vector_indexing_suite<std::vector<bool> >());

    class_<Communicator, boost::shared_ptr<Communicator>, boost::noncopyable>("Communicator",
           init<boost::shared_ptr<SystemDefinition>,
                boost::shared_ptr<boost::mpi::communicator>,
                std::vector<unsigned int>,
                std::vector<bool>,
                uint3,
                const BoxDim >())
    ;
    }
#endif // ENABLE_MPI
