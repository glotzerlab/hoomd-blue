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
struct select_particle_migrate : public std::unary_function<const unsigned int, bool>
    {
    const BoxDim& box;      //!< Local simulation box dimensions
    const unsigned int dir; //!< Direction to send particles to
    const Scalar4 *h_pos;   //!< Array of particle positions


    //! Constructor
    /*!
     */
    select_particle_migrate(const BoxDim & _box,
                            const unsigned int _dir,
                            const Scalar4 *_h_pos)
        : box(_box), dir(_dir), h_pos(_h_pos)
        { }

    //! Select a particle
    /*! t particle data to consider for sending
     * \return true if particle stays in the box
     */
    bool operator()(const unsigned int idx)
        {
        const Scalar4& postype = h_pos[idx];
        Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
        Scalar3 f = box.makeFraction(pos);

        // return true if the particle stays leaves the box
        return ((dir == 0 && f.x >= Scalar(1.0)) ||  // send east
                (dir == 1 && f.x < Scalar(0.0))  ||  // send west
                (dir == 2 && f.y >= Scalar(1.0)) ||  // send north
                (dir == 3 && f.y < Scalar(0.0))  ||  // send south
                (dir == 4 && f.z >= Scalar(1.0)) ||  // send up
                (dir == 5 && f.z < Scalar(0.0) ));   // send down
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
            m_pos_copybuf(m_exec_conf),
            m_charge_copybuf(m_exec_conf),
            m_diameter_copybuf(m_exec_conf),
            m_velocity_copybuf(m_exec_conf),
            m_orientation_copybuf(m_exec_conf),
            m_plan_copybuf(m_exec_conf),
            m_tag_copybuf(m_exec_conf),
            m_r_ghost(Scalar(0.0)),
            m_r_buff(Scalar(0.0)),
            m_resize_factor(9.f/8.f),
            m_plan(m_exec_conf),
            m_is_first_step(true)
    {
    // initialize array of neighbor processor ids
    assert(m_mpi_comm);
    assert(m_decomposition);

    m_exec_conf->msg->notice(5) << "Constructing Communicator" << endl;

    for (unsigned int dir = 0; dir < 6; dir ++)
        {
        m_is_at_boundary[dir] = m_decomposition->isAtBoundary(dir) ? 1 : 0;
        }

    for (unsigned int dir = 0; dir < 6; dir ++)
        {
        GPUVector<unsigned int> copy_ghosts(m_exec_conf);
        m_copy_ghosts[dir].swap(copy_ghosts);
        m_num_copy_ghosts[dir] = 0;
        m_num_recv_ghosts[dir] = 0;
        }

    // connect to particle sort signal
    m_sort_connection = m_pdata->connectParticleSort(boost::bind(&Communicator::forceMigrate, this));

    /*
     * Bonded group communication
     */
    m_bonds_changed = true;
    m_bond_connection = m_sysdef->getBondData()->connectGroupNumChange(boost::bind(&Communicator::setBondsChanged, this));

    m_angles_changed = true;
    m_angle_connection = m_sysdef->getAngleData()->connectGroupNumChange(boost::bind(&Communicator::setAnglesChanged, this));

    m_dihedrals_changed = true;
    m_dihedral_connection = m_sysdef->getDihedralData()->connectGroupNumChange(boost::bind(&Communicator::setDihedralsChanged, this));

    m_impropers_changed = true;
    m_improper_connection = m_sysdef->getImproperData()->connectGroupNumChange(boost::bind(&Communicator::setImpropersChanged, this));
    }

//! Destructor
Communicator::~Communicator()
    {
    m_exec_conf->msg->notice(5) << "Destroying Communicator";
    m_sort_connection.disconnect();
    m_bond_connection.disconnect();
    m_angle_connection.disconnect();
    }

//! Interface to the communication methods.
void Communicator::communicate(unsigned int timestep)
    {
    // Guard to prevent recursive triggering of migration
    m_is_communicating = true;

    // update ghost communication flags
    m_flags = m_requested_flags(timestep);

    // Check if migration of particles is requested
    if (m_force_migrate || m_migrate_requests(timestep) || m_is_first_step)
        {
        m_force_migrate = false;
        m_is_first_step = false;

        // If so, migrate atoms
        migrateParticles();

        // Construct ghost send lists, exchange ghost atom data
        exchangeGhosts();
        }
    else
        {
        // just update ghost positions
        updateGhosts(timestep);
        }

    m_is_communicating = false;
    }

//! Transfer particles between neighboring domains
void Communicator::migrateParticles()
    {
    if (m_prof)
        m_prof->push("comm_migrate");

    m_exec_conf->msg->notice(7) << "Communicator: migrate particles" << std::endl;

        {
        // wipe out reverse-lookup tag -> idx for old ghost atoms
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

            {
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_comm_flag(m_pdata->getCommFlags(), access_location::host, access_mode::readwrite);

            // mark all particles which have left the box for sending (rtag=NOT_LOCAL)
            unsigned int N = m_pdata->getN();
            select_particle_migrate pred(box, dir, h_pos.data);

            for (unsigned int idx = 0; idx < N; ++idx)
                if (pred(idx)) h_comm_flag.data[idx] = 1; // currently just set to unity
            }

        boost::shared_ptr<BondData> bdata(m_sysdef->getBondData());

        #if 0
        if (bdata->getNumBondsGlobal())
            {
            /*
             * Select bonds for sending.
             */
            ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

            ArrayHandle<uint2> h_bonds(bdata->getBondTable(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_bond_tag(bdata->getBondTags(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_bond_rtag(bdata->getBondRTags(), access_location::host, access_mode::readwrite);

            unsigned int num_bonds = bdata->getNumBonds();
            for (unsigned int bond_idx = 0; bond_idx < num_bonds; ++bond_idx)
                {
                uint2 bond = h_bonds.data[bond_idx];

                assert(bond.x < m_pdata->getNGlobal());
                assert(bond.y < m_pdata->getNGlobal());

                unsigned int rtag_a = h_rtag.data[bond.x];
                unsigned int rtag_b = h_rtag.data[bond.y];

                unsigned int bond_tag = h_bond_tag.data[bond_idx];
                assert(bond_tag < bdata->getNumBondsGlobal());

                // number of particles that remain local
                unsigned num_local = 2;
                if (rtag_a == NOT_LOCAL || rtag_a == STAGED) num_local--;
                if (rtag_b == NOT_LOCAL || rtag_b == STAGED) num_local--;

                // number of particles that leave the domain
                unsigned int num_leave = 0;
                if (rtag_a == STAGED) num_leave++;
                if (rtag_b == STAGED) num_leave++;

                // if no particle leaves, do nothing
                if (!num_leave) continue;

                // if the bond has no local particles anymore, send and remove it
                if (!num_local)
                    h_bond_rtag.data[bond_tag] = BOND_STAGED;
                else
                    // otherwise, the bond is split
                    h_bond_rtag.data[bond_tag] = BOND_SPLIT;
                }
            }
        #endif

        // fill send buffer
        std::vector<unsigned int> comm_flag_out; // not currently used
        m_pdata->removeParticles(m_sendbuf, comm_flag_out);

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

        unsigned int n_send_ptls = m_sendbuf.size();

        MPI_Isend(&n_send_ptls, 1, MPI_UNSIGNED, send_neighbor, 0, m_mpi_comm, & reqs[0]);
        MPI_Irecv(&n_recv_ptls, 1, MPI_UNSIGNED, recv_neighbor, 0, m_mpi_comm, & reqs[1]);
        MPI_Waitall(2, reqs, status);

        // Resize receive buffer
        m_recvbuf.resize(n_recv_ptls);

        // exchange particle data
        MPI_Isend(&m_sendbuf.front(), n_send_ptls*sizeof(pdata_element), MPI_BYTE, send_neighbor, 1, m_mpi_comm, & reqs[0]);
        MPI_Irecv(&m_recvbuf.front(), n_recv_ptls*sizeof(pdata_element), MPI_BYTE, recv_neighbor, 1, m_mpi_comm, & reqs[1]);
        MPI_Waitall(2, reqs, status);

        if (m_prof)
            m_prof->pop();

        const BoxDim shifted_box = getShiftedBox();

        // wrap received particles across a global boundary back into global box
        for (unsigned int idx = 0; idx < n_recv_ptls; idx++)
            {
            pdata_element& p = m_recvbuf[idx];
            Scalar4& postype = p.pos;
            int3& image = p.image;

            shifted_box.wrap(postype, image);
            }

        // remove particles that were sent and fill particle data with received particles
        m_pdata->addParticles(m_recvbuf);

        #if 0
        /*
         *  Bond communication
         */
        if (bdata->getNumBondsGlobal())
            {
            // fill bond send buffer
            bdata->retrieveBonds(m_bond_send_buf);

            unsigned int n_recv_bonds;
            unsigned int n_send_bonds = m_bond_send_buf.size();

            // exchange size of messages
            MPI_Isend(&n_send_bonds, sizeof(unsigned int), MPI_BYTE, send_neighbor, 0, m_mpi_comm, & reqs[0]);
            MPI_Irecv(&n_recv_bonds, sizeof(unsigned int), MPI_BYTE, recv_neighbor, 0, m_mpi_comm, & reqs[1]);
            MPI_Waitall(2, reqs, status);

            // resize recv buffer
            m_bond_recv_buf.resize(n_recv_bonds);

            // exchange bond data
            MPI_Isend(&m_bond_send_buf.front(),
                      n_send_bonds*sizeof(bond_element),
                      MPI_BYTE,
                      send_neighbor,
                      1,
                      m_mpi_comm,
                      & reqs[0]);
            MPI_Irecv(&m_bond_recv_buf.front(),
                      n_recv_bonds*sizeof(bond_element),
                      MPI_BYTE,
                      recv_neighbor,
                      1,
                      m_mpi_comm,
                      & reqs[1]);
            MPI_Waitall(2, reqs, status);

            // unpack data
            bdata->addRemoveBonds(m_bond_recv_buf);

            } // end bond communication
            #endif
        } // end dir loop

    if (m_prof)
        m_prof->pop();
    }

//! Build ghost particle list, exchange ghost particle data
void Communicator::exchangeGhosts()
    {
    if (m_prof)
        m_prof->push("comm_ghost_exch");

    m_exec_conf->msg->notice(7) << "Communicator: exchange ghosts" << std::endl;

    const BoxDim& box = m_pdata->getBox();

    // Sending ghosts proceeds in two stages:
    // Stage 1: mark ghost atoms for sending (for covalently bonded particles, and non-bonded interactions)
    //          construct plans (= itineraries for ghost particles)
    // Stage 2: fill send buffers, exchange ghosts according to plans (sending the plan along with the particle)

    // resize and reset plans
    m_plan.resize(m_pdata->getN());

        {
        ArrayHandle<unsigned char> h_plan(m_plan, access_location::host, access_mode::readwrite);

        for (unsigned int i = 0; i < m_pdata->getN(); ++i)
            h_plan.data[i] = 0;
        }

    /*
     * Mark particles that are part of incomplete bonds for sending
     */
    boost::shared_ptr<BondData> bdata = m_sysdef->getBondData();

    /*
     * Mark non-bonded atoms for sending
     */

    // the ghost layer must be at_least m_r_ghost wide along every lattice direction
    Scalar3 ghost_fraction = m_r_ghost/box.getNearestPlaneDistance();
        {
        // scan all local atom positions if they are within r_ghost from a neighbor
        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<unsigned char> h_plan(m_plan, access_location::host, access_mode::readwrite);

        for (unsigned int idx = 0; idx < m_pdata->getN(); idx++)
            {
            Scalar4 postype = h_pos.data[idx];
            Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);

            Scalar3 f = box.makeFraction(pos);
            if (f.x >= Scalar(1.0) - ghost_fraction.x)
                h_plan.data[idx] |= send_east;

            if (f.x < ghost_fraction.x)
                h_plan.data[idx] |= send_west;

            if (f.y >= Scalar(1.0) - ghost_fraction.y)
                h_plan.data[idx] |= send_north;

            if (f.y < ghost_fraction.y)
                h_plan.data[idx] |= send_south;

            if (f.z >= Scalar(1.0) - ghost_fraction.z)
                h_plan.data[idx] |= send_up;

            if (f.z < ghost_fraction.z)
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
    m_velocity_copybuf.resize(m_pdata->getN());
    m_orientation_copybuf.resize(m_pdata->getN());

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
        m_velocity_copybuf.resize(max_copy_ghosts);
        m_orientation_copybuf.resize(max_copy_ghosts);


            {
            // we fill all fields, but send only those that are requested by the CommFlags bitset
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
            ArrayHandle<unsigned char>  h_plan(m_plan, access_location::host, access_mode::read);

            ArrayHandle<unsigned int> h_copy_ghosts(m_copy_ghosts[dir], access_location::host, access_mode::overwrite);
            ArrayHandle<unsigned char> h_plan_copybuf(m_plan_copybuf, access_location::host, access_mode::overwrite);
            ArrayHandle<Scalar4> h_pos_copybuf(m_pos_copybuf, access_location::host, access_mode::overwrite);
            ArrayHandle<Scalar> h_charge_copybuf(m_charge_copybuf, access_location::host, access_mode::overwrite);
            ArrayHandle<Scalar> h_diameter_copybuf(m_diameter_copybuf, access_location::host, access_mode::overwrite);
            ArrayHandle<Scalar4> h_velocity_copybuf(m_velocity_copybuf, access_location::host, access_mode::overwrite);
            ArrayHandle<Scalar4> h_orientation_copybuf(m_orientation_copybuf, access_location::host, access_mode::overwrite);

            for (unsigned int idx = 0; idx < m_pdata->getN() + m_pdata->getNGhosts(); idx++)
                {

                if (h_plan.data[idx] & (1 << dir))
                    {
                    // send with next message
                    h_pos_copybuf.data[m_num_copy_ghosts[dir]] = h_pos.data[idx];
                    h_charge_copybuf.data[m_num_copy_ghosts[dir]] = h_charge.data[idx];
                    h_diameter_copybuf.data[m_num_copy_ghosts[dir]] = h_diameter.data[idx];
                    h_velocity_copybuf.data[m_num_copy_ghosts[dir]] = h_vel.data[idx];
                    h_orientation_copybuf.data[m_num_copy_ghosts[dir]] = h_orientation.data[idx];
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
        MPI_Request reqs[14];
        MPI_Status status[14];

        MPI_Isend(&m_num_copy_ghosts[dir],
            sizeof(unsigned int),
            MPI_BYTE,
            send_neighbor,
            0,
            m_mpi_comm,
            &reqs[0]);
        MPI_Irecv(&m_num_recv_ghosts[dir],
            sizeof(unsigned int),
            MPI_BYTE,
            recv_neighbor,
            0,
            m_mpi_comm,
            &reqs[1]);
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
            ArrayHandle<Scalar4> h_velocity_copybuf(m_velocity_copybuf, access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_orientation_copybuf(m_orientation_copybuf, access_location::host, access_mode::read);

            ArrayHandle<unsigned char> h_plan(m_plan, access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);
            ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::readwrite);

            unsigned int nreq = 0;

            MPI_Isend(h_plan_copybuf.data,
                m_num_copy_ghosts[dir]*sizeof(unsigned char),
                MPI_BYTE,
                send_neighbor,
                1,
                m_mpi_comm,
                &reqs[nreq++]);
            MPI_Irecv(h_plan.data + start_idx,
                m_num_recv_ghosts[dir]*sizeof(unsigned char),
                MPI_BYTE,
                recv_neighbor,
                1,
                m_mpi_comm,
                &reqs[nreq++]);

            MPI_Isend(h_copy_ghosts.data,
                m_num_copy_ghosts[dir]*sizeof(unsigned int),
                MPI_BYTE,
                send_neighbor,
                2,
                m_mpi_comm,
                &reqs[nreq++]);
            MPI_Irecv(h_tag.data + start_idx,
                m_num_recv_ghosts[dir]*sizeof(unsigned int),
                MPI_BYTE,
                recv_neighbor,
                2,
                m_mpi_comm,
                &reqs[nreq++]);

            CommFlags flags = getFlags();

            if (flags[comm_flag::position])
                {
                MPI_Isend(h_pos_copybuf.data,
                    m_num_copy_ghosts[dir]*sizeof(Scalar4),
                    MPI_BYTE,
                    send_neighbor,
                    3,
                    m_mpi_comm,
                    &reqs[nreq++]);
                MPI_Irecv(h_pos.data + start_idx,
                    m_num_recv_ghosts[dir]*sizeof(Scalar4),
                    MPI_BYTE,
                    recv_neighbor,
                    3,
                    m_mpi_comm,
                    &reqs[nreq++]);
                }

            if (flags[comm_flag::charge])
                {
                MPI_Isend(h_charge_copybuf.data,
                    m_num_copy_ghosts[dir]*sizeof(Scalar),
                    MPI_BYTE,
                    send_neighbor,
                    4,
                    m_mpi_comm,
                    &reqs[nreq++]);
                MPI_Irecv(h_charge.data + start_idx,
                    m_num_recv_ghosts[dir]*sizeof(Scalar),
                    MPI_BYTE,
                    recv_neighbor,
                    4,
                    m_mpi_comm,
                    &reqs[nreq++]);
                }

            if (flags[comm_flag::diameter])
                {
                MPI_Isend(h_diameter_copybuf.data,
                    m_num_copy_ghosts[dir]*sizeof(Scalar),
                    MPI_BYTE,
                    send_neighbor,
                    5,
                    m_mpi_comm,
                    &reqs[nreq++]);
                MPI_Irecv(h_diameter.data + start_idx,
                    m_num_recv_ghosts[dir]*sizeof(Scalar),
                    MPI_BYTE,
                    recv_neighbor,
                    5,
                    m_mpi_comm,
                    &reqs[nreq++]);
                }

            if (flags[comm_flag::velocity])
                {
                MPI_Isend(h_velocity_copybuf.data,
                    m_num_copy_ghosts[dir]*sizeof(Scalar4),
                    MPI_BYTE,
                    send_neighbor,
                    6,
                    m_mpi_comm,
                    &reqs[nreq++]);
                MPI_Irecv(h_vel.data + start_idx,
                    m_num_recv_ghosts[dir]*sizeof(Scalar4),
                    MPI_BYTE,
                    recv_neighbor,
                    6,
                    m_mpi_comm,
                    &reqs[nreq++]);
                }


            if (flags[comm_flag::orientation])
                {
                MPI_Isend(h_orientation_copybuf.data,
                    m_num_copy_ghosts[dir]*sizeof(Scalar4),
                    MPI_BYTE,
                    send_neighbor,
                    7,
                    m_mpi_comm,
                    &reqs[nreq++]);
                MPI_Irecv(h_orientation.data + start_idx,
                    m_num_recv_ghosts[dir]*sizeof(Scalar4),
                    MPI_BYTE,
                    recv_neighbor,
                    7,
                    m_mpi_comm,
                    &reqs[nreq++]);
                }

            MPI_Waitall(nreq, reqs, status);
            }

        if (m_prof)
            m_prof->pop();

        // wrap particle positions
        CommFlags flags = getFlags();
        if (flags[comm_flag::position])
            {
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);

            const BoxDim shifted_box = getShiftedBox();

            for (unsigned int idx = start_idx; idx < start_idx + m_num_recv_ghosts[dir]; idx++)
                {
                Scalar4& pos = h_pos.data[idx];

                // wrap particles received across a global boundary
                int3 img = make_int3(0,0,0);
                shifted_box.wrap(pos,img);
                }
            }

            {
            // set reverse-lookup tag -> idx
            ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::readwrite);

            for (unsigned int idx = start_idx; idx < start_idx + m_num_recv_ghosts[dir]; idx++)
                {
                assert(h_tag.data[idx] <= m_pdata->getNGlobal());
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
void Communicator::updateGhosts(unsigned int timestep)
    {
    // we have a current m_copy_ghosts liss which contain the indices of particles
    // to send to neighboring processors
    if (m_prof)
        m_prof->push("comm_ghost_update");

    m_exec_conf->msg->notice(7) << "Communicator: update ghosts" << std::endl;

    // update data in these arrays

    unsigned int num_tot_recv_ghosts = 0; // total number of ghosts received

    for (unsigned int dir = 0; dir < 6; dir ++)
        {
        if (! isCommunicating(dir) ) continue;

        CommFlags flags = getFlags();

        if (flags[comm_flag::position])
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

        if (flags[comm_flag::velocity])
            {
            ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_velocity_copybuf(m_velocity_copybuf, access_location::host, access_mode::overwrite);
            ArrayHandle<unsigned int> h_copy_ghosts(m_copy_ghosts[dir], access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

            // copy velocity of ghost particles
            for (unsigned int ghost_idx = 0; ghost_idx < m_num_copy_ghosts[dir]; ghost_idx++)
                {
                unsigned int idx = h_rtag.data[h_copy_ghosts.data[ghost_idx]];

                assert(idx < m_pdata->getN() + m_pdata->getNGhosts());

                // copy velocityition into send buffer
                h_velocity_copybuf.data[ghost_idx] = h_vel.data[idx];
                }
            }

        if (flags[comm_flag::orientation])
            {
            ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_orientation_copybuf(m_orientation_copybuf, access_location::host, access_mode::overwrite);
            ArrayHandle<unsigned int> h_copy_ghosts(m_copy_ghosts[dir], access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

            // copy orientation of ghost particles
            for (unsigned int ghost_idx = 0; ghost_idx < m_num_copy_ghosts[dir]; ghost_idx++)
                {
                unsigned int idx = h_rtag.data[h_copy_ghosts.data[ghost_idx]];

                assert(idx < m_pdata->getN() + m_pdata->getNGhosts());

                // copy orientation into send buffer
                h_orientation_copybuf.data[ghost_idx] = h_orientation.data[idx];
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

        if (m_prof)
            m_prof->push("MPI send/recv");

        start_idx = m_pdata->getN() + num_tot_recv_ghosts;

        num_tot_recv_ghosts += m_num_recv_ghosts[dir];

        size_t sz = 0;
        // only non-permanent fields (position, velocity, orientation) need to be considered here
        // charge and diameter are not updated during a run
        if (flags[comm_flag::position])
            {
            MPI_Request reqs[2];
            MPI_Status status[2];

            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_pos_copybuf(m_pos_copybuf, access_location::host, access_mode::read);

            // exchange particle data, write directly to the particle data arrays
            MPI_Isend(h_pos_copybuf.data, m_num_copy_ghosts[dir]*sizeof(Scalar4), MPI_BYTE, send_neighbor, 1, m_mpi_comm, &reqs[0]);
            MPI_Irecv(h_pos.data + start_idx, m_num_recv_ghosts[dir]*sizeof(Scalar4), MPI_BYTE, recv_neighbor, 1, m_mpi_comm, &reqs[1]);
            MPI_Waitall(2, reqs, status);

            sz += sizeof(Scalar4);
            }

        if (flags[comm_flag::velocity])
            {
            MPI_Request reqs[2];
            MPI_Status status[2];

            ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_vel_copybuf(m_velocity_copybuf, access_location::host, access_mode::read);

            // exchange particle data, write directly to the particle data arrays
            MPI_Isend(h_vel_copybuf.data, m_num_copy_ghosts[dir]*sizeof(Scalar4), MPI_BYTE, send_neighbor, 2, m_mpi_comm, &reqs[0]);
            MPI_Irecv(h_vel.data + start_idx, m_num_recv_ghosts[dir]*sizeof(Scalar4), MPI_BYTE, recv_neighbor, 2, m_mpi_comm, &reqs[1]);
            MPI_Waitall(2, reqs, status);

            sz += sizeof(Scalar4);
            }

        if (flags[comm_flag::orientation])
            {
            MPI_Request reqs[2];
            MPI_Status status[2];

            ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_orientation_copybuf(m_orientation_copybuf, access_location::host, access_mode::read);

            // exchange particle data, write directly to the particle data arrays
            MPI_Isend(h_orientation_copybuf.data, m_num_copy_ghosts[dir]*sizeof(Scalar4), MPI_BYTE, send_neighbor, 3, m_mpi_comm, &reqs[0]);
            MPI_Irecv(h_orientation.data + start_idx, m_num_recv_ghosts[dir]*sizeof(Scalar4), MPI_BYTE, recv_neighbor, 3, m_mpi_comm, &reqs[1]);
            MPI_Waitall(2, reqs, status);

            sz += sizeof(Scalar4);
            }

        if (m_prof)
            m_prof->pop(0, (m_num_recv_ghosts[dir]+m_num_copy_ghosts[dir])*sz);


        // wrap particle positions (only if copying positions)
        if (flags[comm_flag::position])
            {
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);

            const BoxDim shifted_box = getShiftedBox();
            for (unsigned int idx = start_idx; idx < start_idx + m_num_recv_ghosts[dir]; idx++)
                {
                Scalar4& pos = h_pos.data[idx];

                // wrap particles received across a global boundary
                int3 img = make_int3(0,0,0);
                shifted_box.wrap(pos, img);
                }
            }

        } // end dir loop

        if (m_prof)
            m_prof->pop();
    }

const BoxDim Communicator::getShiftedBox() const
    {
    // construct the shifted global box for applying global boundary conditions
    BoxDim shifted_box = m_pdata->getGlobalBox();
    Scalar3 f= make_scalar3(0.5,0.5,0.5);

    Scalar3 r_ghost=m_r_ghost*make_scalar3(1.0,1.0,1.0);
    Scalar3 shift = (m_pdata->getBox().getNearestPlaneDistance()-r_ghost)/
        shifted_box.getNearestPlaneDistance();

    Scalar tol = 0.0001;
    shift += tol*make_scalar3(1.0,1.0,1.0);
    for (unsigned int dir = 0; dir < 6; dir ++)
        {
        if (m_decomposition->isAtBoundary(dir) &&  isCommunicating(dir))
            {
            if (dir == face_east)
                f.x += shift.x;
            else if (dir == face_west)
                f.x -= shift.x;
            else if (dir == face_north)
                f.y += shift.y;
            else if (dir == face_south)
                f.y -= shift.y;
            else if (dir == face_up)
                f.z += shift.z;
            else if (dir == face_down)
                f.z -= shift.z;
            }
        }
    Scalar3 dx = shifted_box.makeCoordinates(f);
    Scalar3 lo = shifted_box.getLo();
    Scalar3 hi = shifted_box.getHi();
    lo += dx;
    hi += dx;
    shifted_box.setLoHi(lo, hi);

    // only apply global boundary conditions along the communication directions
    uchar3 periodic = make_uchar3(0,0,0);

    periodic.x = isCommunicating(face_east) ? 1 : 0;
    periodic.y = isCommunicating(face_north) ? 1 : 0;
    periodic.z = isCommunicating(face_up) ? 1 : 0;

    shifted_box.setPeriodic(periodic);

    return shifted_box;
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
