/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2014 The Regents of
the University of Michigan All rights reserved.

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

/*! \file Communicator.h
    \brief Defines the Communicator class
*/

#ifdef ENABLE_MPI

#ifndef __COMMUNICATOR_H__
#define __COMMUNICATOR_H__

#define NCORNER 8
#define NEDGE 12
#define NFACE 6

#include "HOOMDMath.h"
#include "GPUArray.h"
#include "GPUVector.h"
#include "ParticleData.h"
#include "BondedGroupData.h"
#include "DomainDecomposition.h"

#include <boost/shared_ptr.hpp>
#include <boost/signals2.hpp>

#include "Autotuner.h"

/*! \ingroup hoomd_lib
    @{
*/

/*! \defgroup communication Communication
    \brief All classes that are related to communication via MPI.

    \details See \ref page_dev_info for more information
*/

/*! @}
*/

//! Forward declarations for some classes
class SystemDefinition;
class Profiler;
class BoxDim;
class ParticleData;

// in 3d, there are 27 neighbors max.
#define NEIGH_MAX 27

//! Optional flags to enable communication of certain ParticleData fields for ghost particles
struct comm_flag
    {
    //! The enum
    enum Enum
        {
        tag,         //! Bit id in CommFlags for particle tags
        position,    //! Bit id in CommFlags for particle positions
        charge,      //! Bit id in CommFlags for particle charge
        diameter,    //! Bit id in CommFlags for particle diameter
        velocity,    //! Bit id in CommFlags for particle velocity
        orientation  //! Bit id in CommFlags for particle orientation
        };
    };

//! Bitset to determine required ghost communication fields
typedef std::bitset<32> CommFlags;

//! Perform a logical or operation on the return values of several signals
struct migrate_logical_or
    {
    //! This is needed by boost::signals2
    typedef bool result_type;

    //! Combine return values using logical or
    /*! \param first First return value
        \param last Last return value
     */
    template<typename InputIterator>
    bool operator()(InputIterator first, InputIterator last) const
        {
        if (first == last)
            return false;

        bool return_value = *first++;
        while (first != last)
            {
            if (*first++)
                return_value = true;
            }

        return return_value;
        }
    };

//! Perform a bitwise or operation on the return values of several signals
struct comm_flags_bitwise_or
    {
    //! This is needed by boost::signals
    typedef CommFlags result_type;

    //! Combine return values using logical or
    /*! \param first First return value
        \param last Last return value
     */
    template<typename InputIterator>
    CommFlags operator()(InputIterator first, InputIterator last) const
        {
        if (first == last) return CommFlags(0);

        CommFlags return_value(0);
        while (first != last) return_value |= *first++;

        return return_value;
        }
    };

//! A compact storage for rank information
template<typename ranks_t>
struct rank_element
    {
    ranks_t ranks;
    unsigned int mask;
    unsigned int tag;
    };


//! <b>Class that handles MPI communication</b>
/*! This class implements the communication algorithms that are used in parallel simulations.
 * In the communication pattern used here, every processor exchanges particle data with its
 * six next neighbors lying along the three spatial axes. The original implementation of the communication scheme is
 * described in \cite Plimpton1995.
 *
 * The communication scheme consists of three stages.
 *
 *
 * -# <b> First stage</b>: Atom migration (migrateParticles())
 * <br> Atoms that have left the current domain boundary are deleted from the current processor, exchanged with neighboring
 * processors, and particles received from neighboring processors are added to the system.
 * Atoms are exchanged only infrequently, i.e. only in those steps where the neighbor list needs to be rebuilt.
 * In all other time steps, the processor that currently owns the particle continues to update its position, even if the particle
 * has moved outside the domain boundaries. It is guaruanteed that the processor can correctly calculate the forces
 * on its particles, by maintaining a current list of so-called 'ghost particle' positions.
 *
 * -# <b> Second stage</b>: Ghost exchange (exchangeGhosts())
 * <br>A list of local ghost atoms is constructed. These are atoms that are found within a distance of \f$ r_{\mathrm{cut}} + r_{\mathrm{buff}} \f$
 * from a neighboring processor's boundary (see also NeighborList). The neighboring processors need this information in order
 * to calculate the forces on their local atoms correctly. After construction of the ghost particle list, the positions of these atoms are communicated
 * to the neighboring processors.
 *
 * -# <b> Third stage</b>: Update of ghost positions (updateGhosts())
 * <br> If it is not necessary to renew the list of ghost particles (i.e. when no particle in the global system has moved more than a
 * distance \f$ r_{\mathrm{buff}}/2 \f$), we use the current ghost particle list to update the ghost positions on the neighboring
 * processors.
 *
 * Stages \b one and \b two are performed before every neighbor list build, stage \b three is executed in all other steps (before the calculation
 * of forces).
 *
 * <b>Implementation details:</b>
 *
 * In every stage, particles are subsequently exchanged in six directions:
 *
 * -# send particles to the east, receive from the west
 * -# send particles to the west, receive from the east
 * -# send particles to the north, receive from the south
 * -# send particles to the south, receive from the north
 * -# send particles upwards, receive particles from below
 * -# send particles downwards, receive particles from above.
 *
 * After every step, particles already received from a neighbor in one of the previous steps are added
 * to the list of particles considered for sending. E.g. a particle that is migrating to the processor
 * north-east of the present one is first sent to the processor in the east. This processor then forwards it
 * to its northern neighbor.
 *
 * In stage one, by deleting particles immediately after sending, the processor that sends the particle transfers
 * ownership of the particle to its neighboring processor. In this way it is guaranteed that the decision about where
 * to send the particle to is always made by one and only one processor. This ensure that the total number of particles remains
 * constant (no particle can get lost).
 *
 * In stage two and three, ghost atoms received from a neighboring processor are always included in the local
 * ghost atom lists, and they maybe replicated to more neighboring processors by the communication pattern
 * described above.
 * \ingroup communication
 */
class Communicator
    {
    public:
        //! Constructor
        /*! \param sysdef system definition the communicator is associated with
         *  \param decomposition Information about the decomposition of the global simulation domain
         */
        Communicator(boost::shared_ptr<SystemDefinition> sysdef,
                     boost::shared_ptr<DomainDecomposition> decomposition);
        virtual ~Communicator();


        //! \name accessor methods
        //@{

        //! Set the profiler.
        /*! \param prof Profiler to use with this class
         */
        void setProfiler(boost::shared_ptr<Profiler> prof)
            {
            m_prof = prof;
            }

        //! Subscribe to list of functions that determine when the particles are migrated
        /*! This method keeps track of all functions that may request particle migration.
         * \return A connection to the present class
         */
        boost::signals2::connection addMigrateRequest(const boost::function<bool (unsigned int timestep)>& subscriber)
            {
            return m_migrate_requests.connect(subscriber);
            }

        //! Subscribe to list of functions that determine the communication flags
        /*! This method keeps track of all functions that may request communication flags
         * \return A connection to the present class
         */
        boost::signals2::connection addCommFlagsRequest(const boost::function<CommFlags (unsigned int timestep)>& subscriber)
            {
            return m_requested_flags.connect(subscriber);
            }

        //! Subscribe to list of call-backs for additional communication
        /*!
         * Good candidates for functions to be called after finishing the ghost update step
         * are functions that involve all-to-all synchronization or similar expensive
         * communication that can be overlapped with computation.
         *
         * \param subscriber The callback
         * \returns a connection to this class
         */
        boost::signals2::connection addCommunicationCallback(
            const boost::function<void (unsigned int timestep)>& subscriber)
            {
            return m_comm_callbacks.connect(subscriber);
            }

        //! Subscribe to list of call-backs for overlapping computation
        boost::signals2::connection addLocalComputeCallback(
            const boost::function<void (unsigned int timestep)>& subscriber)
            {
            return m_local_compute_callbacks.connect(subscriber);
            }

        //! Subscribe to list of *optional* call-backs for computation using ghost particles
        /*!
         * Subscribe to a list of call-backs that precompute quantities using information about ghost particles
         * before awaiting the result of the particle migration check. Pre-computation must be entirely *optional* for
         * the subscribing class. When the signal is triggered the class may pre-compute quantities
         * under the assumption that no particle migration will occur. Since the result of the
         * particle migration check is in general available only *after* the signal has been triggered,
         * the class must *not* rely on this assumption. Plus, triggering of the signal is not guaruanteed
         * when particle migration does occur.
         *
         * Methods subscribed to the compute callback signal are those that improve performance by
         * overlapping computation with all-to-all MPI synchronization and communication callbacks.
         * For this optimization to work, subscribing methods should NOT synchronize the GPU execution stream.
         *
         * \note Triggering of the signal before or after MPI synchronization is dependent upon runtime (auto-) tuning.
         *
         * \note Subscribers are called only after updated ghost information is available
         *       but BEFORE particle migration
         *
         * \param subscriber The callback
         * \returns a connection to this class
         */
        boost::signals2::connection addComputeCallback(
            const boost::function<void (unsigned int timestep)>& subscriber)
            {
            return m_compute_callbacks.connect(subscriber);
            }


        //! Set width of ghost layer
        /*! \param ghost_width The width of the ghost layer
         */
        void setGhostLayerWidth(Scalar ghost_width)
            {
            assert(ghost_width > 0);
            m_r_ghost = ghost_width;
            }

        //! Set skin layer width
        /*! \param r_buff The width of the skin buffer
         */
        void setRBuff(Scalar r_buff)
            {
            assert(r_buff > 0);

            m_r_buff = r_buff;
            }

        //! Return current skin layer width
        Scalar getRBuff()
            {
            return m_r_buff;
            }

        //! Get the ghost communication flags
        CommFlags getFlags() { return m_flags; }

        //! Set the ghost communication flags
        /*! \note Flags will be available after the next call to communicate().
         */
        void setFlags(const CommFlags& flags) { m_flags = flags; }

        //@}

        //! \name communication methods
        //@{

        /*! Interface to the communication methods.
         * This method is supposed to be called every time step and automatically performs all necessary
         * communication steps.
         */
        void communicate(unsigned int timestep);

        //@}

        //! Force particle migration
        void forceMigrate()
            {
            // prevent recursive force particle migration
            if (! m_is_communicating)
                m_force_migrate = true;
            }

        /*! Exchange positions of ghost particles
         * Using the previously constructed ghost exchange lists, ghost positions are updated on the
         * neighboring processors.
         *
         * This routine uses non-blocking MPI communication, to make it possible to overlap
         * additional computation or communication during the update substep. To complete
         * the communication, call finishUpdateGhosts()
         *
         * \param timestep The time step
         *
         * \pre The ghost exchange list has been constructed in a previous time step, using exchangeGhosts().
         * \post The ghost positions on the neighboring processors are current
         */
        virtual void beginUpdateGhosts(unsigned int timestep);

        /*! Finish ghost update
         *
         * \param timestep The time step
         */
        virtual void finishUpdateGhosts(unsigned int timestep)
            {
            // the base class implementation is currently empty
            m_comm_pending = false;
            }

        /*! This methods finds all the particles that are no longer inside the domain
         * boundaries and transfers them to neighboring processors.
         *
         * Particles sent to a neighbor are deleted from the local particle data.
         * Particles received from a neighbor in one of the six communication steps
         * are added to the local particle data, and are also considered for forwarding to a neighbor
         * in the subseqent communication steps.
         *
         * \post Every particle on every processor can be found inside the local domain boundaries.
         */
        virtual void migrateParticles();

        /*! Particles that are within r_ghost from a neighboring domain's boundary are exchanged with the
         * processor that is responsible for it. Only information needed for calulating the forces (i.e.
         * particle position, type, charge and diameter) is exchanged.
         *
         * \post A list of ghost atom tags has been constructed which can be used for updating the
         *       the ghost positions, until a new list is constructed. Ghost particle positions on the
         *       neighboring processors are current.
         */
        virtual void exchangeGhosts();

        //! \name Enumerations
        //@{

        //! Enumeration of the faces of the simulation box
        /*! Their order determines the communication pattern, these must be three pairs
            of opposite directions.
         */
        enum faceEnum
            {
            face_east = 0,
            face_west,
            face_north,
            face_south,
            face_up,
            face_down
            };

        //! The flags used for indicating the itinerary of a ghost particle
        enum Enum
            {
            send_east = 1,
            send_west = 2,
            send_north = 4,
            send_south = 8,
            send_up = 16,
            send_down = 32
            };

        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs

            Derived classes should override this to set the parameters of their autotuners.
        */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            if (m_tuner_precompute)
                {
                m_tuner_precompute->setPeriod(period);
                m_tuner_precompute->setEnabled(enable);
                }
            }

        //@}
    protected:
        //! Helper class to perform the communication tasks related to bonded groups
        template<class group_data>
        class GroupCommunicator
            {
            public:
                typedef struct rank_element<typename group_data::ranks_t> rank_element_t;
                typedef typename group_data::packed_t group_element_t;

                //! Constructor
                GroupCommunicator(Communicator& comm, boost::shared_ptr<group_data> gdata);

                //! Migrate groups
                /*! \param incomplete If true, mark all groups that have non-local members and update local
                 *         member rank information. Otherwise, mark only groups flagged for communication
                 *         in particle data
                 *
                 * A group is marked for sending by setting its rtag to GROUP_NOT_LOCAL, and by updating
                 * the rank information with the destination ranks (or the local ranks if incomplete=true)
                 */
                void migrateGroups(bool incomplete);

                //! Mark ghost particles
                /* All particles that need to be sent as ghosts because they are members
                 * of incomplete groups are marked, and destination ranks are compute accordingly.
                 *
                 * \param plans Array of particle plans to write to
                 * \param mask Mask for allowed sending directions
                 */
                void markGhostParticles(const GPUArray<unsigned int>& plans, unsigned int mask);

            private:
                Communicator& m_comm;                            //!< The outer class
                boost::shared_ptr<const ExecutionConfiguration> m_exec_conf; //< The execution configuration
                boost::shared_ptr<group_data> m_gdata;           //!< The group data

                std::vector<rank_element_t> m_ranks_sendbuf;     //!< Send buffer for rank elements
                std::vector<rank_element_t> m_ranks_recvbuf;     //!< Receive buffer for rank elements

                std::vector<typename group_data::packed_t> m_groups_sendbuf;     //!< Send buffer for group elements
                std::vector<typename group_data::packed_t> m_groups_recvbuf;     //!< Receive buffer for group elements
            };

        //! Returns true if we are communicating particles along a given direction
        /*! \param dir Direction to return dimensions for
         */
        bool isCommunicating(unsigned int dir) const
            {
            assert(dir < 6);
            const Index3D& di = m_decomposition->getDomainIndexer();

            bool res = true;

            if ((dir==0 || dir == 1) && di.getW() == 1)
                res = false;
            if ((dir==2 || dir == 3) && di.getH() == 1)
                res = false;
            if ((dir==4 || dir == 5) && di.getD() == 1)
                res = false;

            return res;
            }

        //! Helper function to update the shifted box for ghost particle PBC
        const BoxDim getShiftedBox() const;

        boost::shared_ptr<SystemDefinition> m_sysdef;                 //!< System definition
        boost::shared_ptr<ParticleData> m_pdata;                      //!< Particle data
        boost::shared_ptr<const ExecutionConfiguration> m_exec_conf;  //!< Execution configuration
        const MPI_Comm m_mpi_comm; //!< MPI communciator
        boost::shared_ptr<DomainDecomposition> m_decomposition;       //!< Domain decomposition information
        boost::shared_ptr<Profiler> m_prof;                           //!< Profiler

        bool m_is_communicating;               //!< Whether we are currently communicating
        bool m_force_migrate;                  //!< True if particle migration is forced

        unsigned int m_is_at_boundary[6];      //!< Array of flags indicating whether this box lies at a global boundary

        GPUArray<unsigned int> m_neighbors;            //!< Neighbor ranks
        GPUArray<unsigned int> m_unique_neighbors;     //!< Neighbor ranks w/duplicates removed
        GPUArray<unsigned int> m_adj_mask;             //!< Adjacency mask for every neighbor
        unsigned int m_nneigh;                         //!< Number of neighbors
        unsigned int m_n_unique_neigh;                 //!< Number of unique neighbors
        GPUArray<unsigned int> m_begin;                //!< Begin index for every neighbor in send buf
        GPUArray<unsigned int> m_end;                  //!< End index for every neighbor in send buf

        GPUVector<Scalar4> m_pos_copybuf;         //!< Buffer for particle positions to be copied
        GPUVector<Scalar> m_charge_copybuf;       //!< Buffer for particle charges to be copied
        GPUVector<Scalar> m_diameter_copybuf;     //!< Buffer for particle diameters to be copied
        GPUVector<Scalar4> m_velocity_copybuf;    //!< Buffer for particle velocities to be copied
        GPUVector<Scalar4> m_orientation_copybuf; //!< Buffer for particle orientation to be copied
        GPUVector<unsigned int> m_plan_copybuf;  //!< Buffer for particle plans
        GPUVector<unsigned int> m_tag_copybuf;    //!< Buffer for particle tags

        GPUVector<unsigned int> m_copy_ghosts[6]; //!< Per-direction list of indices of particles to send as ghosts
        unsigned int m_num_copy_ghosts[6];       //!< Number of local particles that are sent to neighboring processors
        unsigned int m_num_recv_ghosts[6];       //!< Number of ghosts received per direction

        BoxDim m_global_box;                     //!< Global simulation box
        Scalar m_r_ghost;                        //!< Width of ghost layer
        Scalar m_r_buff;                         //!< Width of skin layer

        GPUVector<unsigned int> m_plan;          //!< Array of per-direction flags that determine the sending route

        boost::signals2::signal<bool(unsigned int timestep), migrate_logical_or>
            m_migrate_requests; //!< List of functions that may request particle migration

        boost::signals2::signal<CommFlags(unsigned int timestep), comm_flags_bitwise_or>
            m_requested_flags;  //!< List of functions that may request ghost communication flags

        boost::signals2::signal<void (unsigned int timestep)>
            m_local_compute_callbacks;   //!< List of functions that can be overlapped with communication

        boost::signals2::signal<void (unsigned int timestep)>
            m_compute_callbacks;   //!< List of functions that are called after ghost communication

        boost::signals2::signal<void (unsigned int timestep)>
            m_comm_callbacks;   //!< List of functions that are called after the compute callbacks

        boost::scoped_ptr<Autotuner> m_tuner_precompute; //!< Autotuner for precomputation of quantites

        CommFlags m_flags;                       //!< The ghost communication flags
        CommFlags m_last_flags;                       //!< Flags of last ghost exchange

        bool m_comm_pending;                     //!< If true, a communication is in process
        std::vector<MPI_Request> m_reqs;         //!< List of pending MPI requests

        /* Bonds communication */
        bool m_bonds_changed;                          //!< True if bond information needs to be refreshed
        boost::signals2::connection m_bond_connection; //!< Connection to BondData addition/removal of bonds signal
        void setBondsChanged()
            {
            m_bonds_changed = true;
            }

        /* Angles communication */
        bool m_angles_changed;                          //!< True if angle information needs to be refreshed
        boost::signals2::connection m_angle_connection; //!< Connection to AngleData addition/removal of angles signal
        void setAnglesChanged()
            {
            m_angles_changed = true;
            }

        /* Dihedrals communication */
        bool m_dihedrals_changed;                          //!< True if dihedral information needs to be refreshed
        boost::signals2::connection m_dihedral_connection; //!< Connection to DihedralData addition/removal of dihedrals signal
        void setDihedralsChanged()
            {
            m_dihedrals_changed = true;
            }

        /* Impropers communication */
        bool m_impropers_changed;                          //!< True if improper information needs to be refreshed
        boost::signals2::connection m_improper_connection; //!< Connection to ImproperData addition/removal of impropers signal
        void setImpropersChanged()
            {
            m_impropers_changed = true;
            }

    private:
        std::vector<pdata_element> m_sendbuf;  //!< Buffer for particles that are sent
        std::vector<pdata_element> m_recvbuf;  //!< Buffer for particles that are received

        /* Communication of bonded groups */
        GroupCommunicator<BondData> m_bond_comm;    //!< Communication helper for bonds
        friend class GroupCommunicator<BondData>;

        GroupCommunicator<AngleData> m_angle_comm;  //!< Communication helper for angles
        friend class GroupCommunicator<AngleData>;

        GroupCommunicator<DihedralData> m_dihedral_comm;  //!< Communication helper for dihedrals
        friend class GroupCommunicator<DihedralData>;

        GroupCommunicator<ImproperData> m_improper_comm;  //!< Communication helper for impropers
        friend class GroupCommunicator<ImproperData>;

        bool m_is_first_step;                    //!< True if no communication has yet occured

        //! Connection to the signal notifying when particles are resorted
        boost::signals2::connection m_sort_connection;

        //! Helper function to initialize adjacency arrays
        void initializeNeighborArrays();
    };


//! Declaration of python export function
void export_Communicator();

#endif // __COMMUNICATOR_H__
#endif // ENABLE_MPI
