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

/*! \file Communicator.h
    \brief Defines the Communicator class
*/

#ifdef ENABLE_MPI

#ifndef __COMMUNICATOR_H__
#define __COMMUNICATOR_H__

#include "HOOMDMath.h"
#include "GPUArray.h"
#include "GPUVector.h"
#include "ParticleData.h"
#include "BondData.h"
#include "DomainDecomposition.h"

#include <boost/shared_ptr.hpp>
#include <boost/signals.hpp>

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

//! Structure to store packed particle data
struct pdata_element
    {
    Scalar4 pos;               //!< Position
    Scalar4 vel;               //!< Velocity
    Scalar3 accel;             //!< Acceleration
    Scalar charge;             //!< Charge
    Scalar diameter;           //!< Diameter
    int3 image;               //!< Image
    unsigned int body;        //!< Body id
    Scalar4 orientation;       //!< Orientation
    unsigned int tag;  //!< global tag
    };

//! Perform a logical or operation on the return values of several signals
struct migrate_logical_or
    {
    typedef bool result_type;

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

//! <b>Class that handles MPI communication</b>
/*! This class implements the communication algorithms that are used in parallel simulations.
 * In the communication pattern used here, every processor exchanges particle data with its
 * six next neighbors lying along the three spatial axes. The original implementation of the communication scheme is
 * described in \cite Plimpton1995.
 *
 * The communication scheme consists of three stages.
 *
 *
 * -# <b> First stage</b>: Atom migration (migrateAtoms())
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
 * -# <b> Third stage</b>: Update of ghost positions (copyGhosts())
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
        boost::signals::connection addMigrateRequest(const boost::function<bool (unsigned int timestep)>& subscriber)
            {
            return m_migrate_requests.connect(subscriber);
            }

        //! Set width of ghost layer
        /*! \param ghost_width The width of the ghost layer
         */
        void setGhostLayerWidth(Scalar ghost_width)
            {
            assert(ghost_width > 0);
            assert(ghost_width < m_pdata->getBox().getL().x);
            assert(ghost_width < m_pdata->getBox().getL().y);
            assert(ghost_width < m_pdata->getBox().getL().z);
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
            
        //@}

        //! \name communication methods
        //@{

        /*! Interface to the communication methods.
         * This method is supposed to be called every time step and automatically performs all necessary
         * communication steps.
         */
        void communicate(unsigned int timestep);

        /*! Start ghost communication.
         * Ghost-communication can be multi-threaded, if so this method spawns the corresponding thread
         */
        virtual void startGhostsUpdate(unsigned int timestep);

        /*! Finish ghost communication.
         */
        virtual void finishGhostsUpdate(unsigned int timestep);

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
         * \pre The ghost exchange list has been constructed in a previous time step, using exchangeGhosts().
         * \post The ghost positions on the neighboring processors are current
         */
        virtual void copyGhosts();

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
        virtual void migrateAtoms();

        /*! Particles that are within r_ghost from a neighboring domain's boundary are exchanged with the
         * processor that is responsible for it. Only information needed for calulating the forces (i.e.
         * particle position, type, charge and diameter) is exchanged.
         *
         * \post A list of ghost atom tags has been constructed which can be used for updating the
         *       the ghost positions, until a new list is constructed. Ghost particle positions on the
         *       neighboring processors are current.
         */
        virtual void exchangeGhosts();

    protected:
        //! Set size of a packed data element
        /*! \param size size of data element (in bytes)
         */
        void setPackedSize(unsigned int size)
            {
            assert(size > 0);
            m_packed_size = size;
            }

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

        //! Returns true if we are communicating particles along a given direction
        /*! \param dir Direction to return dimensions for
         */
        bool isCommunicating(unsigned int dir)
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

        boost::shared_ptr<SystemDefinition> m_sysdef;                 //!< System definition
        boost::shared_ptr<ParticleData> m_pdata;                      //!< Particle data
        boost::shared_ptr<const ExecutionConfiguration> m_exec_conf;  //!< Execution configuration
        const MPI_Comm m_mpi_comm; //!< MPI communciator
        boost::shared_ptr<DomainDecomposition> m_decomposition;       //!< Domain decomposition information
        boost::shared_ptr<Profiler> m_prof;                           //!< Profiler

        bool m_is_communicating;               //!< Whether we are currently communicating
        bool m_force_migrate;                  //!< True if particle migration is forced

        unsigned int m_is_at_boundary[6];      //!< Array of flags indicating whether this box lies at a global boundary

        GPUVector<char> m_sendbuf;             //!< Buffer for particles that are sent
        GPUVector<char> m_recvbuf;             //!< Buffer for particles that are received
        GPUBuffer<bond_element> m_bond_send_buf;//!< Buffer for bonds that are sent
        GPUBuffer<bond_element> m_bond_recv_buf;//!< Buffer for bonds that are received
        GPUArray<unsigned int> m_bond_remove_mask; //!< Per-bond flag (1= remove, 0= keep)
        GPUVector<Scalar4> m_pos_copybuf;         //!< Buffer for particle positions to be copied
        GPUVector<Scalar> m_charge_copybuf;       //!< Buffer for particle charges to be copied
        GPUVector<Scalar> m_diameter_copybuf;     //!< Buffer for particle diameters to be copied
        GPUVector<unsigned char> m_plan_copybuf;  //!< Buffer for particle plans
        GPUVector<unsigned int> m_tag_copybuf;    //!< Buffer for particle tags

        GPUVector<unsigned int> m_copy_ghosts[6]; //!< Per-direction list of indices of particles to send as ghosts
        unsigned int m_num_copy_ghosts[6];       //!< Number of local particles that are sent to neighboring processors
        unsigned int m_num_recv_ghosts[6];       //!< Number of ghosts received per direction

        BoxDim m_global_box;                     //!< Global simulation box
        unsigned int m_packed_size;              //!< Size of packed particle data element in bytes
        Scalar m_r_ghost;                        //!< Width of ghost layer
        Scalar m_r_buff;                         //!< Width of skin layer
        const float m_resize_factor;                //!< Factor used for amortized array resizing

        GPUVector<unsigned char> m_plan;         //!< Array of per-direction flags that determine the sending route

        boost::signal<bool(unsigned int timestep), migrate_logical_or>
            m_migrate_requests; //!< List of functions that may request particle migration

        std::vector<Scalar4> scal4_tmp;          //!< Temporary list used to apply the sort order to the particle data
        std::vector<Scalar3> scal3_tmp;          //!< Temporary list used to apply the sort order to the particle data
        std::vector<Scalar> scal_tmp;            //!< Temporary list used to apply the sort order to the particle data
        std::vector<unsigned int> uint_tmp;      //!< Temporary list used to apply the sort order to the particle data
        std::vector<int3> int3_tmp;              //!< Temporary list used to apply the sort order to the particle data

        unsigned int m_next_ghost_update;        //!< Timestep in which the ghosts are next updated
        bool m_is_first_step;                    //!< True if no communication has yet occured
    };

//! Declaration of python export function
void export_Communicator();

#endif // __COMMUNICATOR_H__
#endif // ENABLE_MPI
