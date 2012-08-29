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

#include <boost/bind.hpp>
#include <boost/python.hpp>
#include <algorithm>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

using namespace boost::python;

//! Select a particle for migration
struct select_particle_migrate : public std::unary_function<const unsigned int&, bool>
    {
    Scalar3 lo;             //!< Lower local box boundary
    Scalar3 hi;             //!< Upper local box boundary
    const unsigned int dir; //!< Direction to send particles to
    const Scalar4 *h_pos;   //!< Array of particle positions


    //! Constructor
    /*!
     */
    select_particle_migrate(const BoxDim & _box,
                            const unsigned int _dir,
                            const Scalar4 *_h_pos)
        : dir(_dir), h_pos(_h_pos)
        {
        lo = _box.getLo();
        hi = _box.getHi();
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
        return !((dir == 0 && pos.x >= hi.x) ||  // send east
                (dir == 1 && pos.x < lo.x)  ||  // send west
                (dir == 2 && pos.y >= hi.y) ||  // send north
                (dir == 3 && pos.y < lo.y)  ||  // send south
                (dir == 4 && pos.z >= hi.z) ||  // send up
                (dir == 5 && pos.z < lo.z ));   // send down
        }

     };



//! Constructor
Communicator::Communicator(boost::shared_ptr<SystemDefinition> sysdef,
                           boost::shared_ptr<DomainDecomposition> decomposition)
          : m_sysdef(sysdef),
            m_pdata(sysdef->getParticleData()),
            m_exec_conf(m_pdata->getExecConf()),
            m_mpi_comm(m_exec_conf->getMPICommunicator()),
            m_decomposition(decomposition),
            m_is_communicating(false),
            m_force_migrate(false),
            m_sendbuf(m_exec_conf),
            m_recvbuf(m_exec_conf),
            m_pos_copybuf(m_exec_conf),
            m_charge_copybuf(m_exec_conf),
            m_diameter_copybuf(m_exec_conf),
            m_plan_copybuf(m_exec_conf),
            m_tag_copybuf(m_exec_conf),
            m_r_ghost(Scalar(0.0)),
            m_plan(m_exec_conf)
    {
    // initialize array of neighbor processor ids
    assert(m_mpi_comm);
    assert(m_decomposition);

    m_exec_conf->msg->notice(5) << "Constructing Communicator" << endl;

    for (unsigned int dir = 0; dir < 6; dir ++)
        {
        m_is_at_boundary[dir] = m_decomposition->isAtBoundary(dir);
        }

    m_packed_size = sizeof(pdata_element);
 
    for (unsigned int dir = 0; dir < 6; dir ++)
        {
        GPUVector<unsigned int> copy_ghosts(m_exec_conf);
        m_copy_ghosts[dir].swap(copy_ghosts);
        m_num_copy_ghosts[dir] = 0;
        m_num_recv_ghosts[dir] = 0;
        }
    }

//! Destructor
Communicator::~Communicator()
    {
    m_exec_conf->msg->notice(5) << "Destroying Communicator";
    }

//! Interface to the communication methods.
void Communicator::communicate(unsigned int timestep)
    {
    // Guard to prevent recursive triggering of migration
    m_is_communicating = true;

    // Check if migration of particles is requested
    if (m_force_migrate || m_migrate_requests(timestep))
        {
        m_force_migrate = false;

        // If so, migrate atoms
        migrateAtoms();

        // Construct ghost send lists, exchange ghost atom data
        exchangeGhosts();
        }
    else
        {
        // only update ghost atom coordinates
        copyGhosts();
        }

    m_is_communicating = false;
    }

//! Transfer particles between neighboring domains
void Communicator::migrateAtoms()
    {
    if (m_prof)
        m_prof->push("migrate_atoms");

    // wipe out reverse-lookup tag -> idx for old ghost atoms
        {
        ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::readwrite);
        for (unsigned int i = 0; i < m_pdata->getNGhosts(); i++)
            {
            unsigned int idx = m_pdata->getN() + i;
            h_rtag.data[h_tag.data[idx]] = NOT_LOCAL;
            }
        }

    //  reset ghost particle number
    m_pdata->removeAllGhostParticles();


    // get box dimensions
    const BoxDim& box = m_pdata->getBox();

    // determine local particles that are to be sent to neighboring processors and fill send buffer
    for (unsigned int dir=0; dir < 6; dir++)
        {
        if (! isCommunicating(dir) ) continue;

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
            ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::readwrite);

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
                                                 select_particle_migrate(box, dir, h_pos.data));

            n_send_ptls = (sort_keys.begin() + m_pdata->getN()) - sort_keys_middle;

            // reorder the particle data
            if (scal4_tmp.size() < m_pdata->getN())
                {
                scal4_tmp.resize(m_pdata->getN());
                scal3_tmp.resize(m_pdata->getN());
                scal_tmp.resize(m_pdata->getN());
                uint_tmp.resize(m_pdata->getN());
                int3_tmp.resize(m_pdata->getN());
                }

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
                uint_tmp[i] = h_tag.data[sort_keys[i]];
            for (unsigned int i = 0; i < m_pdata->getN(); i++)
                h_tag.data[i] = uint_tmp[i];
            }

        // remove particles from local data that are being sent
        m_pdata->removeParticles(n_send_ptls);

            {
            // update reverse lookup tags
            ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::readwrite);
            for (unsigned int idx = 0; idx < m_pdata->getN(); idx++)
                {
                h_rtag.data[h_tag.data[idx]] = idx;
                }
            }


        // resize send buffer
        m_sendbuf.resize(n_send_ptls*m_packed_size);

            {
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::read);
            ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
            ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_body(m_pdata->getBodies(), access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

            ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::readwrite);

            ArrayHandle<char> h_sendbuf(m_sendbuf, access_location::host, access_mode::overwrite);

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
                p.tag = h_tag.data[idx];

                // Reset the global rtag for the particle we are sending to indicate it is no longer local
                assert(h_rtag.data[h_tag.data[idx]] < m_pdata->getN() + n_send_ptls);
                h_rtag.data[h_tag.data[idx]] = NOT_LOCAL;

                ( (pdata_element *) h_sendbuf.data)[i] = p;
                }
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
        MPI_Request reqs[2];
        MPI_Status status[2];

        MPI_Isend(&n_send_ptls, sizeof(unsigned int), MPI_BYTE, send_neighbor, 0, m_mpi_comm, & reqs[0]);
        MPI_Irecv(&n_recv_ptls, sizeof(unsigned int), MPI_BYTE, recv_neighbor, 0, m_mpi_comm, & reqs[1]);
        MPI_Waitall(2, reqs, status);

        // Resize receive buffer 
        m_recvbuf.resize(n_recv_ptls*m_packed_size);

            {
            ArrayHandle<char> h_sendbuf(m_sendbuf, access_location::host, access_mode::read);
            ArrayHandle<char> h_recvbuf(m_recvbuf, access_location::host, access_mode::overwrite);
            // exchange actual particle data
            MPI_Isend(h_sendbuf.data, n_send_ptls*m_packed_size, MPI_BYTE, send_neighbor, 1, m_mpi_comm, & reqs[0]);
            MPI_Irecv(h_recvbuf.data, n_recv_ptls*m_packed_size, MPI_BYTE, recv_neighbor, 1, m_mpi_comm, & reqs[1]);
            MPI_Waitall(2, reqs, status);
            }

        if (m_prof)
            m_prof->pop();

        const BoxDim& global_box = m_pdata->getGlobalBox(); 
            {
            // wrap received particles across a global boundary back into global box
            Scalar3 L = global_box.getL();
            ArrayHandle<char> h_recvbuf(m_recvbuf, access_location::host, access_mode::readwrite);
            for (unsigned int idx = 0; idx < n_recv_ptls; idx++)
                {
                pdata_element& p = ((pdata_element *) h_recvbuf.data)[idx];
                Scalar4& postype = p.pos;
                int3& image = p.image;

                if (dir==0 && m_is_at_boundary[1])
                    {
                    postype.x -= L.x;
                    image.x++;
                    }
                else if (dir==1 && m_is_at_boundary[0])
                    {
                    postype.x += L.x;
                    image.x--;
                    }
                else if (dir==2 && m_is_at_boundary[3])
                    {
                    postype.y -= L.y;
                    image.y++;
                    }
                else if (dir==3 && m_is_at_boundary[2])
                    {
                    postype.y += L.y;
                    image.y--;
                    }
                else if (dir==4 && m_is_at_boundary[5])
                    {
                    postype.z -= L.z;
                    image.z++;
                    }
                else if (dir==5 && m_is_at_boundary[4])
                    {
                    postype.z += L.z;
                    image.z--;
                    }
     
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
            ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::readwrite);
            ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::readwrite);

            ArrayHandle<char> h_recvbuf(m_recvbuf, access_location::host, access_mode::read);
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
                h_tag.data[add_idx] = p.tag;

                assert(h_rtag.data[h_tag.data[add_idx]] == NOT_LOCAL);
                h_rtag.data[h_tag.data[add_idx]] = add_idx;
                add_idx++;
                }
            }
        } // end dir loop

    // notify ParticleData that addition / removal of particles is complete
    m_pdata->notifyParticleSort();
 
    if (m_prof)
        m_prof->pop();
    }

//! Build ghost particle list, exchange ghost particle data
void Communicator::exchangeGhosts()
    {
    if (m_prof)
        m_prof->push("exchange_ghosts");

    const BoxDim& box = m_pdata->getBox();

    // Sending ghosts proceeds in two stages:
    // Stage 1: mark ghost atoms for sending (for covalently bonded particles, and non-bonded interactions)
    //          construct plans (= itineraries for ghost particles)
    // Stage 2: fill send buffers, exchange ghosts according to plans (sending the plan along with the particle)

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
        const GPUVector<uint2>& btable = bdata->getBondTable();
        ArrayHandle<uint2> h_btable(btable, access_location::host, access_mode::read);
        ArrayHandle<unsigned char> h_plan(m_plan, access_location::host, access_mode::readwrite);
        ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

        unsigned nbonds = bdata->getNumBonds();
        unsigned int N = m_pdata->getN();
        Scalar3 lo = box.getLo();
        Scalar3 L2 = box.getL()/Scalar(2.0);
        for (unsigned int bond_idx = 0; bond_idx < nbonds; bond_idx++)
            {
            uint2 bond = h_btable.data[bond_idx];

            unsigned int tag1 = bond.x;
            unsigned int tag2 = bond.y;
            unsigned int idx1 = h_rtag.data[tag1];
            unsigned int idx2 = h_rtag.data[tag2];
       
            if ((idx1 >= N) && (idx2 < N))
                {
                Scalar4 pos = h_pos.data[idx2];
                h_plan.data[idx2] |= (pos.x > lo.x + L2.x) ? send_east : send_west;
                h_plan.data[idx2] |= (pos.y > lo.y + L2.y) ? send_north : send_south;
                h_plan.data[idx2] |= (pos.z > lo.z + L2.z) ? send_up : send_down;

                }
            else if ((idx1 < N) && (idx2 >= N))
                {
                Scalar4 pos = h_pos.data[idx1];
                h_plan.data[idx1] |= (pos.x > lo.x + L2.x) ? send_east : send_west;
                h_plan.data[idx1] |= (pos.y > lo.y + L2.y) ? send_north : send_south;
                h_plan.data[idx1] |= (pos.z > lo.z + L2.z) ? send_up : send_down;
                } 
            }
        }


    /*
     * Mark non-bonded atoms for sending
     */
    Scalar3 lo = box.getLo();
    Scalar3 hi = box.getHi();

        {
        // scan all local atom positions if they are within r_ghost from a neighbor
        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<unsigned char> h_plan(m_plan, access_location::host, access_mode::readwrite);

        for (unsigned int idx = 0; idx < m_pdata->getN(); idx++)
            {
            Scalar4 pos = h_pos.data[idx];

            if (pos.x >= hi.x  - m_r_ghost)
                h_plan.data[idx] |= send_east;

            if (pos.x < lo.x + m_r_ghost)
                h_plan.data[idx] |= send_west;

            if (pos.y >= hi.y - m_r_ghost)
                h_plan.data[idx] |= send_north;

            if (pos.y < lo.y + m_r_ghost)
                h_plan.data[idx] |= send_south;

            if (pos.z >= hi.z - m_r_ghost)
                h_plan.data[idx] |= send_up;

            if (pos.z < lo.z + m_r_ghost)
                h_plan.data[idx] |= send_down;
            }
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
        if (! isCommunicating(dir) ) continue;

        m_num_copy_ghosts[dir] = 0;

        // resize array of ghost particle tags
        unsigned int max_copy_ghosts = m_pdata->getN() + m_pdata->getNGhosts();
        m_copy_ghosts[dir].resize(max_copy_ghosts);

        // resize buffers
        m_plan_copybuf.resize(max_copy_ghosts);
        m_pos_copybuf.resize(max_copy_ghosts);
        m_charge_copybuf.resize(max_copy_ghosts);
        m_diameter_copybuf.resize(max_copy_ghosts);

      
            {
            // Fill send buffer
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
            ArrayHandle<unsigned char>  h_plan(m_plan, access_location::host, access_mode::read);

            ArrayHandle<unsigned int> h_copy_ghosts(m_copy_ghosts[dir], access_location::host, access_mode::overwrite);
            ArrayHandle<unsigned char> h_plan_copybuf(m_plan_copybuf, access_location::host, access_mode::overwrite);
            ArrayHandle<Scalar4> h_pos_copybuf(m_pos_copybuf, access_location::host, access_mode::overwrite);
            ArrayHandle<Scalar> h_charge_copybuf(m_charge_copybuf, access_location::host, access_mode::overwrite);
            ArrayHandle<Scalar> h_diameter_copybuf(m_diameter_copybuf, access_location::host, access_mode::overwrite);

            for (unsigned int idx = 0; idx < m_pdata->getN() + m_pdata->getNGhosts(); idx++)
                {

                if (h_plan.data[idx] & (1 << dir))
                    {
                    // send with next message
                    h_pos_copybuf.data[m_num_copy_ghosts[dir]] = h_pos.data[idx];
                    h_charge_copybuf.data[m_num_copy_ghosts[dir]] = h_charge.data[idx];
                    h_diameter_copybuf.data[m_num_copy_ghosts[dir]] = h_diameter.data[idx];
                    h_plan_copybuf.data[m_num_copy_ghosts[dir]] = h_plan.data[idx];

                    h_copy_ghosts.data[m_num_copy_ghosts[dir]] = h_tag.data[idx];
                    m_num_copy_ghosts[dir]++;
                    }
                }
            }
        unsigned int send_neighbor = m_decomposition->getNeighborRank(dir);

        // we receive from the direction opposite to the one we send to
        unsigned int recv_neighbor;
        if (dir % 2 == 0)
            recv_neighbor = m_decomposition->getNeighborRank(dir+1);
        else
            recv_neighbor = m_decomposition->getNeighborRank(dir-1);

        if (m_prof)
            m_prof->push("MPI send/recv");

        // communicate size of the message that will contain the particle data
        MPI_Request reqs[12];
        MPI_Status status[12];

        MPI_Isend(&m_num_copy_ghosts[dir], sizeof(unsigned int), MPI_BYTE, send_neighbor, 0, m_mpi_comm, &reqs[0]); 
        MPI_Irecv(&m_num_recv_ghosts[dir], sizeof(unsigned int), MPI_BYTE, recv_neighbor, 0, m_mpi_comm, &reqs[1]);
        MPI_Waitall(2, reqs, status);

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

            {
            ArrayHandle<unsigned int> h_copy_ghosts(m_copy_ghosts[dir], access_location::host, access_mode::read);
            ArrayHandle<unsigned char> h_plan_copybuf(m_plan_copybuf, access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_pos_copybuf(m_pos_copybuf, access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_charge_copybuf(m_charge_copybuf, access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_diameter_copybuf(m_diameter_copybuf, access_location::host, access_mode::read);

            ArrayHandle<unsigned char> h_plan(m_plan, access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::readwrite);
            ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::readwrite);

            MPI_Isend(h_plan_copybuf.data, m_num_copy_ghosts[dir]*sizeof(unsigned char), MPI_BYTE, send_neighbor, 1, m_mpi_comm, &reqs[2]);
            MPI_Irecv(h_plan.data + start_idx, m_num_recv_ghosts[dir]*sizeof(unsigned char), MPI_BYTE, recv_neighbor, 1, m_mpi_comm, &reqs[3]);

            MPI_Isend(h_pos_copybuf.data, m_num_copy_ghosts[dir]*sizeof(Scalar4), MPI_BYTE, send_neighbor, 2, m_mpi_comm, &reqs[4]);
            MPI_Irecv(h_pos.data + start_idx, m_num_recv_ghosts[dir]*sizeof(Scalar4), MPI_BYTE, recv_neighbor, 2, m_mpi_comm, &reqs[5]);

            MPI_Isend(h_copy_ghosts.data, m_num_copy_ghosts[dir]*sizeof(unsigned int), MPI_BYTE, send_neighbor, 3, m_mpi_comm, &reqs[6]);
            MPI_Irecv(h_tag.data + start_idx, m_num_recv_ghosts[dir]*sizeof(unsigned int), MPI_BYTE, recv_neighbor, 3, m_mpi_comm, &reqs[7]);

            MPI_Isend(h_charge_copybuf.data, m_num_copy_ghosts[dir]*sizeof(Scalar), MPI_BYTE, send_neighbor, 4, m_mpi_comm, &reqs[8]);
            MPI_Irecv(h_charge.data + start_idx, m_num_recv_ghosts[dir]*sizeof(Scalar), MPI_BYTE, recv_neighbor, 4, m_mpi_comm, &reqs[9]);

            MPI_Isend(h_diameter_copybuf.data, m_num_copy_ghosts[dir]*sizeof(Scalar), MPI_BYTE, send_neighbor, 5, m_mpi_comm, &reqs[10]);
            MPI_Irecv(h_diameter.data + start_idx, m_num_recv_ghosts[dir]*sizeof(Scalar), MPI_BYTE, recv_neighbor, 5, m_mpi_comm, &reqs[11]);
            
            MPI_Waitall(10, reqs+2, status+2);
            }

        if (m_prof)
            m_prof->pop();

        Scalar3 L = m_pdata->getGlobalBox().getL();


            {
            ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
            for (unsigned int idx = start_idx; idx < start_idx + m_num_recv_ghosts[dir]; idx++)
                {
                Scalar4& pos = h_pos.data[idx];

                // wrap particles received across a global boundary 
                // we are not actually folding back particles into the global box, but into the ghost layer
                if (dir==0 && m_is_at_boundary[1])
                    pos.x -= L.x;
                else if (dir==1 && m_is_at_boundary[0])
                    pos.x += L.x;
                else if (dir==2 && m_is_at_boundary[3])
                    pos.y -= L.y;
                else if (dir==3 && m_is_at_boundary[2])
                    pos.y += L.y;
                else if (dir==4 && m_is_at_boundary[5])
                    pos.z -= L.z;
                else if (dir==5 && m_is_at_boundary[4])
                    pos.z += L.z;


                // set reverse-lookup tag -> idx
                assert(h_rtag.data[h_tag.data[idx]] == NOT_LOCAL);
                h_rtag.data[h_tag.data[idx]] = idx;
                }
            }
        } // end dir loop

    // we have updated ghost particles, so inform ParticleData about this
    m_pdata->notifyGhostParticleNumberChange();

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
        if (! isCommunicating(dir) ) continue;

            {
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_pos_copybuf(m_pos_copybuf, access_location::host, access_mode::overwrite);
            ArrayHandle<unsigned int> h_copy_ghosts(m_copy_ghosts[dir], access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

            // copy positions of ghost particles
            for (unsigned int ghost_idx = 0; ghost_idx < m_num_copy_ghosts[dir]; ghost_idx++)
                {
                unsigned int idx = h_rtag.data[h_copy_ghosts.data[ghost_idx]];

                assert(idx < m_pdata->getN() + m_pdata->getNGhosts());

                // copy position into send buffer
                h_pos_copybuf.data[ghost_idx] = h_pos.data[idx];
                }
            }
        unsigned int send_neighbor = m_decomposition->getNeighborRank(dir);

        // we receive from the direction opposite to the one we send to
        unsigned int recv_neighbor;
        if (dir % 2 == 0)
            recv_neighbor = m_decomposition->getNeighborRank(dir+1);
        else
            recv_neighbor = m_decomposition->getNeighborRank(dir-1);


        unsigned int start_idx;
        {
        if (m_prof)
            m_prof->push("MPI send/recv");


        start_idx = m_pdata->getN() + num_tot_recv_ghosts;

        num_tot_recv_ghosts += m_num_recv_ghosts[dir];

            {
            MPI_Request reqs[2];
            MPI_Status status[2];

            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_pos_copybuf(m_pos_copybuf, access_location::host, access_mode::read);

            // exchange particle data, write directly to the particle data arrays
            MPI_Isend(h_pos_copybuf.data, m_num_copy_ghosts[dir]*sizeof(Scalar4), MPI_BYTE, send_neighbor, 1, m_mpi_comm, &reqs[0]);
            MPI_Irecv(h_pos.data + start_idx, m_num_recv_ghosts[dir]*sizeof(Scalar4), MPI_BYTE, recv_neighbor, 1, m_mpi_comm, &reqs[1]);
            MPI_Waitall(2, reqs, status);
            }


        if (m_prof)
            m_prof->pop(0, (m_num_recv_ghosts[dir]+m_num_copy_ghosts[dir])*sizeof(Scalar4));
        }

        Scalar3 L= m_pdata->getGlobalBox().getL();

            {
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);

            for (unsigned int idx = start_idx; idx < start_idx + m_num_recv_ghosts[dir]; idx++)
                {
                Scalar4& pos = h_pos.data[idx];

                // wrap particles received across a global boundary 
                // we are not actually folding back particles into the global box, but into the ghost layer
                if (dir==0 && m_is_at_boundary[1])
                    pos.x -= L.x;
                else if (dir==1 && m_is_at_boundary[0])
                    pos.x += L.x;
                else if (dir==2 && m_is_at_boundary[3])
                    pos.y -= L.y;
                else if (dir==3 && m_is_at_boundary[2])
                    pos.y += L.y;
                else if (dir==4 && m_is_at_boundary[5])
                    pos.z -= L.z;
                else if (dir==5 && m_is_at_boundary[4])
                    pos.z += L.z;
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
                boost::shared_ptr<DomainDecomposition> >())
    ;
    }
#endif // ENABLE_MPI
