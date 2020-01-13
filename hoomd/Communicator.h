// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


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
#include "GlobalArray.h"
#include "GPUVector.h"
#include "ParticleData.h"
#include "BondedGroupData.h"
#include "DomainDecomposition.h"

#include <memory>
#include <hoomd/extern/nano-signal-slot/nano_signal_slot.hpp>

#ifndef NVCC
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#endif

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

//! a define to indicate API requirements
#define HOOMD_COMM_GHOST_LAYER_WIDTH_REQUEST

//! Forward declarations for some classes
class SystemDefinition;
class Profiler;
struct BoxDim;
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
        orientation, //! Bit id in CommFlags for particle orientation
        body,        //! Bit id in CommFlags for particle body id
        image,       //! Bit id in CommFlags for particle image
        net_force,   //! Communicate net force
        reverse_net_force,   //! Communicate net force on ghost particles. Added by Vyas
        net_torque,  //! Communicate net torque
        net_virial   //! Communicate net virial
        };
    };

//! Bitset to determine required ghost communication fields
typedef std::bitset<32> CommFlags;

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
 * has moved outside the domain boundaries. It is guaranteed that the processor can correctly calculate the forces
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
class PYBIND11_EXPORT Communicator
    {
    public:
        //! Constructor
        /*! \param sysdef system definition the communicator is associated with
         *  \param decomposition Information about the decomposition of the global simulation domain
         */
        Communicator(std::shared_ptr<SystemDefinition> sysdef,
                     std::shared_ptr<DomainDecomposition> decomposition);
        virtual ~Communicator();


        //! \name accessor methods
        //@{

        //! Set the profiler.
        /*! \param prof Profiler to use with this class
         */
        void setProfiler(std::shared_ptr<Profiler> prof)
            {
            m_prof = prof;
            }

        //! Subscribe to list of functions that determine when the particles are migrated
        /*! This method keeps track of all functions that may request particle migration.
         * \return A Nano::Signal object reference to be used for connect and disconnect calls.
         */
        Nano::Signal<bool(unsigned int timestep)>& getMigrateSignal()
            {
            return m_migrate_requests;
            }

        //! Subscribe to list of functions that request a minimum ghost layer width
        /*! This method keeps track of all functions that request a minimum ghost layer width
         * The actual ghost layer width is chosen from the max over the inputs
         * \return A connection to the present class
         */
        Nano::Signal<Scalar (unsigned int)>& getGhostLayerWidthRequestSignal()
            {
            return m_ghost_layer_width_requests;
            }

        //! Subscribe to list of functions that request a minimum extra ghost layer width (added to the maximum ghost layer)
        /*! This method keeps track of all functions that request a minimum ghost layer width
         * The actual ghost layer width is chosen from the max over the inputs
         * \return A connection to the present class
         */
        Nano::Signal<Scalar (unsigned int)>& getExtraGhostLayerWidthRequestSignal()
            {
            return m_extra_ghost_layer_width_requests;
            }


        //! Subscribe to list of functions that determine the communication flags
        /*! This method keeps track of all functions that may request communication flags
         * \return A connection to the present class
         */
        Nano::Signal<CommFlags (unsigned int timestep)>& getCommFlagsRequestSignal()
            {
            return m_requested_flags;
            }


        //! Subscribe to list of call-backs for ghost communication
        /*!
         * A subscribing function is passed a reference to the ghost plans array
         * which it can then update
         *
         * \param subscriber The callback
         * \returns a connection to this class
         */
        Nano::Signal<void (const GlobalArray<unsigned int> &)>& getCommunicationCallbackSignal()
            {
            return m_comm_callbacks;
            }

        //! Subscribe to list of *optional* call-backs for computation using ghost particles
        /*!
         * Subscribe to a list of call-backs that precompute quantities using information about ghost particles
         * before awaiting the result of the particle migration check.
         *
         * \param subscriber The callback
         * \return A Nano::Signal object reference to be used for connect and disconnect calls.
         */
        Nano::Signal<void (unsigned int timestep)>& getComputeCallbackSignal()
            {
            return m_compute_callbacks;
            }

        //! Get the ghost communication flags
        CommFlags getFlags() { return m_flags; }

        //! Get the number of unique neighbors
        unsigned int getNUniqueNeighbors() const
            {
            return m_n_unique_neigh;
            }

        //! Get the array of unique neighbors
        const GlobalArray<unsigned int>& getUniqueNeighbors() const
            {
            return m_unique_neighbors;
            }

        //! Get the current ghost layer width array
        const GlobalArray<Scalar>& getGhostLayerWidth() const
            {
            return m_r_ghost;
            }

        //! Get the current maximum ghost layer width
        Scalar getGhostLayerMaxWidth() const
            {
            return m_r_ghost_max + m_r_extra_ghost_max;
            }

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

        /*! Communicate the net particle force
         * \parm timestep The time step
         */
        virtual void updateNetForce(unsigned int timestep);

        /*! This methods finds all the particles that are no longer inside the domain
         * boundaries and transfers them to neighboring processors.
         *
         * Particles sent to a neighbor are deleted from the local particle data.
         * Particles received from a neighbor in one of the six communication steps
         * are added to the local particle data, and are also considered for forwarding to a neighbor
         * in the subsequent communication steps.
         *
         * \post Every particle on every processor can be found inside the local domain boundaries.
         */
        virtual void migrateParticles();

        /*! Particles that are within r_ghost from a neighboring domain's boundary are exchanged with the
         * processor that is responsible for it. Only information needed for calculating the forces (i.e.
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

        //@}

        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs

            Derived classes should override this to set the parameters of their autotuners.
        */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            { }

    protected:
        //! Helper class to perform the communication tasks related to bonded groups
        template<class group_data>
        class GroupCommunicator
            {
            public:
                typedef struct rank_element<typename group_data::ranks_t> rank_element_t;
                typedef typename group_data::packed_t group_element_t;

                //! Constructor
                GroupCommunicator(Communicator& comm, std::shared_ptr<group_data> gdata);

                //! Migrate groups
                /*! \param incomplete If true, mark all groups that have non-local members and update local
                 *         member rank information. Otherwise, mark only groups flagged for communication
                 *         in particle data
                 *  \param local_multiple If true, a group may be split across several ranks
                 * A group is marked for sending by setting its rtag to GROUP_NOT_LOCAL, and by updating
                 * the rank information with the destination ranks (or the local ranks if incomplete=true)
                 */
                void migrateGroups(bool incomplete, bool local_multiple);

                //! Mark ghost particles
                /* All particles that need to be sent as ghosts because they are members
                 * of incomplete groups are marked, and destination ranks are compute accordingly.
                 *
                 * \param plans Array of particle plans to write to
                 * \param mask Mask for allowed sending directions
                 */
                void markGhostParticles(const GlobalVector<unsigned int>& plans, unsigned int mask);

                //! Copy 'ghost groups' between domains
                /*! Both members of a ghost group are inside the ghost layer
                 *
                 * \param plans The ghost particle send directions determined by Communicator
                 * \param mask Mask for allowed sending directions
                 */
                void exchangeGhostGroups(const GlobalArray<unsigned int>& plans, unsigned int mask);

            private:
                Communicator& m_comm;                            //!< The outer class
                std::shared_ptr<const ExecutionConfiguration> m_exec_conf; //< The execution configuration
                std::shared_ptr<group_data> m_gdata;           //!< The group data

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

        std::shared_ptr<SystemDefinition> m_sysdef;                 //!< System definition
        std::shared_ptr<ParticleData> m_pdata;                      //!< Particle data
        std::shared_ptr<const ExecutionConfiguration> m_exec_conf;  //!< Execution configuration
        const MPI_Comm m_mpi_comm; //!< MPI communicator
        std::shared_ptr<DomainDecomposition> m_decomposition;       //!< Domain decomposition information
        std::shared_ptr<Profiler> m_prof;                           //!< Profiler

        bool m_is_communicating;               //!< Whether we are currently communicating
        bool m_force_migrate;                  //!< True if particle migration is forced

        unsigned int m_is_at_boundary[6];      //!< Array of flags indicating whether this box lies at a global boundary

        GlobalArray<unsigned int> m_neighbors;            //!< Neighbor ranks
        GlobalArray<unsigned int> m_unique_neighbors;     //!< Neighbor ranks w/duplicates removed
        GlobalArray<unsigned int> m_adj_mask;             //!< Adjacency mask for every neighbor
        unsigned int m_nneigh;                         //!< Number of neighbors
        unsigned int m_n_unique_neigh;                 //!< Number of unique neighbors
        GlobalArray<unsigned int> m_begin;                //!< Begin index for every neighbor in send buf
        GlobalArray<unsigned int> m_end;                  //!< End index for every neighbor in send buf

        GlobalVector<Scalar4> m_pos_copybuf;         //!< Buffer for particle positions to be copied
        GlobalVector<Scalar> m_charge_copybuf;       //!< Buffer for particle charges to be copied
        GlobalVector<Scalar> m_diameter_copybuf;     //!< Buffer for particle diameters to be copied
        GlobalVector<unsigned int> m_body_copybuf;   //!< Buffer for particle body ids to be copied
        GlobalVector<int3> m_image_copybuf;          //!< Buffer for particle body ids to be copied
        GlobalVector<Scalar4> m_velocity_copybuf;    //!< Buffer for particle velocities to be copied
        GlobalVector<Scalar4> m_orientation_copybuf; //!< Buffer for particle orientation to be copied
        GlobalVector<unsigned int> m_plan_copybuf;  //!< Buffer for particle plans
        GlobalVector<unsigned int> m_tag_copybuf;    //!< Buffer for particle tags
        GlobalVector<Scalar4> m_netforce_copybuf;    //!< Buffer for net force
        GlobalVector<Scalar4> m_nettorque_copybuf;   //!< Buffer for net torque
        GlobalVector<Scalar> m_netvirial_copybuf;   //!< Buffer for net virial
        GlobalVector<Scalar> m_netvirial_recvbuf;   //!< Buffer for net virial (receive)

        GlobalVector<unsigned int> m_copy_ghosts[6]; //!< Per-direction list of indices of particles to send as ghosts
        unsigned int m_num_copy_ghosts[6];       //!< Number of local particles that are sent to neighboring processors
        unsigned int m_num_recv_ghosts[6];       //!< Number of ghosts received per direction

        GlobalVector<unsigned int> m_plan;          //!< Array of per-direction flags that determine the sending route

        // Variables needed for sending ghost particles backwards
        GlobalVector<unsigned int> m_plan_reverse;          //!< Array of flags that determine the reverse sending route for ghosts
        GlobalVector<unsigned int> m_tag_reverse;          //!< Array of flags that determine which ghost particles are being sent back. This has no analog normally because particles actually store their tags, but in this case we don't want to so we have to make a vector. This vector corresponds to the m_copy_ghosts_reverse copybuf (m_copy_ghosts writes directly to m_pdata->getTags())
        GlobalVector<unsigned int> m_copy_ghosts_reverse[6]; //!< Per-direction list of indices of particles to send back as ghosts. Copy buffer for m_tag_reverse
        GlobalVector<unsigned int> m_plan_reverse_copybuf[6];   //!< Per-direction buffer for reverse particle plans. Copy buffer for m_plan_reverse
        unsigned int m_num_copy_local_ghosts_reverse[6];       //!< Number of ghost particles in local domain that may need forwarding. Size of m_plan_reverse and m_tag_reverse
        unsigned int m_num_recv_local_ghosts_reverse[6];       //!< Number of ghost particles in local domain that may need forwarding. Receive buffer corresponding to m_num_copy_local_ghosts_reverse

        // This is for forwarding ghost particles if they traverse multiple MPI decomposition domains. They are stored separately from the main dataset to avoid double counting, so instead of a main dataset and a copybuf there is a copybuf and a receive buffer
        unsigned int m_num_forward_ghosts_reverse[6];       //!< Number of ghost particles forwarded to this domain that may need forwarding to neighboring processors. Size of m_recv_tag_reverse and m_recv_tag_reverse
        unsigned int m_num_recv_forward_ghosts_reverse[6];       //!< Number of reverse ghosts received per direction. Receive buffer corresponding to m_num_forward_ghosts_reverse
        GlobalVector<unsigned int> m_forward_ghosts_reverse[6];    //!< Indicates the index in the forwarded ghosts array containing a given particle in the received array

        // Variables for sending forces in reverse
        GlobalVector<Scalar4> m_netforce_reverse_copybuf;            //!< Buffer for reverse net force from ghosts
        GlobalVector<Scalar4> m_netforce_reverse_recvbuf;            //!< Buffer for the reverse net force. Receive buffer for m_netforce_reverse_copybuf

        BoxDim m_global_box;                     //!< Global simulation box
        GlobalArray<Scalar> m_r_ghost;              //!< Width of ghost layer
        GlobalArray<Scalar> m_r_ghost_body;         //!< Extra ghost width for rigid bodies
        Scalar m_r_ghost_max;                    //!< Maximum ghost layer width
        Scalar m_r_extra_ghost_max;              //!< Maximum extra ghost layer width

        unsigned int m_ghosts_added;             //!< Number of ghosts added
        bool m_has_ghost_particles;              //!< True if we have a current copy of ghost particles

        MPI_Datatype m_mpi_pdata_element;        //!< A datatype for the (non-packed) pdata_element struct

        //! Update the ghost width array
        void updateGhostWidth();

        Nano::Signal<bool(unsigned int timestep)>
            m_migrate_requests; //!< List of functions that may request particle migration

        Nano::Signal<CommFlags(unsigned int timestep) >
            m_requested_flags;  //!< List of functions that may request ghost communication flags

        Nano::Signal<Scalar(unsigned int type) >
            m_ghost_layer_width_requests;  //!< List of functions that request a minimum ghost layer width

        Nano::Signal<Scalar(unsigned int type) >
            m_extra_ghost_layer_width_requests;  //!< List of functions that request an extra ghost layer width

        Nano::Signal<void (unsigned int timestep)>
            m_compute_callbacks;   //!< List of functions that are called after ghost communication

        Nano::Signal<void (const GlobalArray<unsigned int>& )>
            m_comm_callbacks;   //!< List of functions that are called after the compute callbacks

        CommFlags m_flags;                       //!< The ghost communication flags
        CommFlags m_last_flags;                       //!< Flags of last ghost exchange

        bool m_comm_pending;                     //!< If true, a communication is in process
        std::vector<MPI_Request> m_reqs; //!< Container for all MPI communication requests
        std::vector<MPI_Status> m_stats; //!< Container for all MPI communication statuses

        /* Bonds communication */
        bool m_bonds_changed;                          //!< True if bond information needs to be refreshed
        void setBondsChanged()
            {
            m_bonds_changed = true;
            }

        /* Angles communication */
        bool m_angles_changed;                          //!< True if angle information needs to be refreshed
        void setAnglesChanged()
            {
            m_angles_changed = true;
            }

        /* Dihedrals communication */
        bool m_dihedrals_changed;                          //!< True if dihedral information needs to be refreshed
        void setDihedralsChanged()
            {
            m_dihedrals_changed = true;
            }

        /* Impropers communication */
        bool m_impropers_changed;                          //!< True if improper information needs to be refreshed
        void setImpropersChanged()
            {
            m_impropers_changed = true;
            }

        /* Constraints communication */
        bool m_constraints_changed;                          //!< True if constraint information needs to be refreshed
        void setConstraintsChanged()
            {
            m_constraints_changed = true;
            }

        /* Pairs communication */
        bool m_pairs_changed;                          //!< True if pair information needs to be refreshed
        void setPairsChanged()
            {
            m_pairs_changed = true;
            }

        //! Remove tags of ghost particles
        virtual void removeGhostParticleTags();

        // check if box is sufficiently large for communication
        void checkBoxSize()
            {
            Scalar3 L= m_pdata->getBox().getNearestPlaneDistance();
            const Index3D& di = m_decomposition->getDomainIndexer();

            Scalar r_ghost_max = getGhostLayerMaxWidth();
            if ((r_ghost_max >= L.x/Scalar(2.0) && di.getW() > 1) ||
                (r_ghost_max >= L.y/Scalar(2.0) && di.getH() > 1) ||
                (r_ghost_max >= L.z/Scalar(2.0) && di.getD() > 1))
                {
                m_exec_conf->msg->error() << "Simulation box too small for domain decomposition." << std::endl;
                throw std::runtime_error("Error during communication");
                }
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

        GroupCommunicator<ConstraintData> m_constraint_comm; //!< Communicator helper for constraints
        friend class GroupCommunicator<ConstraintData>;

        /* Communication of bonded groups */
        GroupCommunicator<PairData> m_pair_comm;    //!< Communication helper for special pairs
        friend class GroupCommunicator<PairData>;

        //! Reallocate the ghost layer width arrays when number of types change
        void slotNumTypesChanged()
            {
            // skip the reallocation if the number of types does not change
            // this keeps old parameters when restoring a snapshot
            // it will result in invalid coefficients if the snapshot has a different type id -> name mapping
            if (m_pdata->getNTypes() == m_r_ghost.getNumElements())
                return;

            GlobalArray<Scalar> r_ghost(m_pdata->getNTypes(), m_exec_conf);
            m_r_ghost.swap(r_ghost);

            GlobalArray<Scalar> r_ghost_body(m_pdata->getNTypes(), m_exec_conf);
            m_r_ghost_body.swap(r_ghost_body);
            }

        //! Helper function to initialize adjacency arrays
        void initializeNeighborArrays();

        //! Method that is called when ghost particles are requested to be removed
        void slotGhostParticlesRemoved()
            {
            removeGhostParticleTags();
            m_has_ghost_particles = false;
            }

    };


//! Declaration of python export function
void export_Communicator(pybind11::module& m);

#endif // __COMMUNICATOR_H__
#endif // ENABLE_MPI
