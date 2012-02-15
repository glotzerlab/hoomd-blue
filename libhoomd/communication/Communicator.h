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
#include "MPIInitializer.h"

#include <boost/shared_ptr.hpp>

//! Forward declarations
namespace boost
    {
    namespace mpi
        {
        class communicator;
        }
    }

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
    float4 pos;               //!< Position
    float4 vel;               //!< Velocity
    float3 accel;             //!< Acceleration
    float charge;             //!< Charge
    float diameter;           //!< Diameter
    int3 image;               //!< Image
    unsigned int body;        //!< Body id
    float4 orientation;       //!< Orientation
    unsigned int global_tag;  //!< global tag
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
         *  \param mpi_comm the underlying MPI communicator
         *  \param neighbor_rank list of neighbor processor ranks
         *  \param dim Dimensions of global simulation box (number of boxes along every axis)
         *  \param global_box Dimensions global simulation box
         */
        Communicator(boost::shared_ptr<SystemDefinition> sysdef,
                     boost::shared_ptr<boost::mpi::communicator> mpi_comm,
                     std::vector<unsigned int> neighbor_rank,
                     std::vector<bool> is_at_boundary,
                     uint3 dim,
                     const BoxDim global_box);

        //! \name accessor methods
        //@{

        //! Get the underlying MPI communicator
        /*! \return Boost MPI communicator
         */
        const boost::shared_ptr<const boost::mpi::communicator> getMPICommunicator()
            {
            return m_mpi_comm;
            }

        //! Set the profiler.
        /*! \param prof Profiler to use with this class
         */
        void setProfiler(boost::shared_ptr<Profiler> prof)
            {
            m_prof = prof;
            }

        //! Get the dimensions of the global simulation box
        /*! \param dir Direction to return dimensions for
         *  \return Number of simulation boxes along the specified direction
         */
        unsigned int getDimension(unsigned int dir)
            {
            assert(dir < 3);
            switch(dir)
                {
                case 0:
                    return m_dim.x;
                    break;
                case 1:
                    return m_dim.y;
                    break;
                case 2:
                    return m_dim.z;
                    break;
                }

            return 0; // we should never arrive here
            }

        //@}

        //! \name communication methods
        //@{

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
         *
         * \param r_ghost Width of ghost layer
         */
        virtual void exchangeGhosts(Scalar r_ghost);

        /*! Exchange positions of ghost particles
         * Using the previously constructed ghost exchange lists, ghost positions are updated on the
         * neighboring processors.
         *
         * \pre The ghost exchange list has been constructed in a previous time step, using exchangeGhosts().
         * \post The ghost positions on the neighboring processors are current
         */
        virtual void copyGhosts();

        //@}

    protected:

        //! Set size of a packed data element
        /*! \param size size of data element (in bytes)
         */
        void setPackedSize(unsigned int size)
            {
            assert(size > 0);
            m_packed_size = size;
            }

        //! Helper function to allocate internal buffers
        void allocate();

        GPUArray<unsigned int> m_delete_buf;     //!< Buffer of particle indices that are going to be deleted

        GPUArray<char> m_sendbuf[6];             //!< Per-direction buffer for particles that are sent
        GPUArray<char> m_recvbuf[6];             //!< Per-direction buffer for particles that are received

        GPUArray<Scalar4> m_pos_copybuf[6];      //!< Per-direction send buffer for particle positions to be copied
        GPUArray<Scalar> m_charge_copybuf[6];    //!< Per-direction send buffer for particle charges to be copied
        GPUArray<Scalar> m_diameter_copybuf[6];  //!< Per-direction send buffer for particle diameters to be copied

        GPUArray<unsigned int> m_copy_ghosts[6]; //!< Per-direction list of indices of particles to send as ghosts to neighboring processors

        unsigned int m_num_copy_ghosts[6];       //!< Number of local particles that are sent to neighboring processors
        unsigned int m_num_recv_ghosts[6];       //!< Number of ghosts received per direction


        unsigned int m_max_copy_ghosts[6];       //!< Max size of m_copy_ghosts array
        unsigned int m_max_ghost_copybuf;        //!< Max  size of ghost particle data buffer

        unsigned int m_neighbors[6];             //!< MPI rank of neighbor domain  in every direction
        bool m_is_at_boundary[6];                //!< Per-direction flas to indicate whether the box is at a a boundary

        boost::shared_ptr<SystemDefinition> m_sysdef;              //!< System definition
        boost::shared_ptr<ParticleData> m_pdata;                   //!< Particle data
        boost::shared_ptr<const ExecutionConfiguration> exec_conf; //!< Execution configuration
        boost::shared_ptr<const boost::mpi::communicator> m_mpi_comm; //!< MPI communciator
        boost::shared_ptr<Profiler> m_prof;                        //!< Profiler

        const uint3 m_dim;                        //!< Dimensions of global simulation box (number of boxes along every axis)
        BoxDim m_global_box;                     //!< Global simulation box
        unsigned int m_packed_size;              //!< Size of packed particle data element in bytes
        bool m_is_allocated;                     //!< True if internal buffers have been allocated
        Scalar m_r_ghost;                        //!< Width of ghost layer

    };

//! Declaration of python export function
void export_Communicator();

#endif // __COMMUNICATOR_H__
#endif // ENABLE_MPI
