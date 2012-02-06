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

/*! \defgroup communication MPI communication
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
struct PackedParticleData
    {
    public:
        //! constructor
        //! \param data pointer to the data
        PackedParticleData(char *data)
         : m_data(data)
            {
            }

        //! get particle position by reference
        //! \param idx index of particle in array
        inline Scalar4& pos(unsigned int idx)
            {
            return *(Scalar4 *)(m_data + m_size*idx);
            }

        //! get particle velocity by reference
        //! \param idx index of particle in array
        inline Scalar4& vel(unsigned int idx)
            {
            return *(Scalar4 *) (m_data + m_size*idx + sizeof(Scalar4) );
            }

        //! get particle acceleration by reference
        //! \param idx index of particle in array
        inline Scalar3& accel(unsigned int idx)
            {
            return *(Scalar3 *) (m_data + m_size*idx + 2*sizeof(Scalar4));
            }
        //! get particle charge by reference
        //! \param idx index of particle in array
        inline Scalar& charge(unsigned int idx)
            {
            return *(Scalar *) (m_data + m_size*idx + 2*sizeof(Scalar4) + sizeof(Scalar3));
            }

        //! get particle diameter by reference
        //! \param idx index of particle in array
        inline Scalar& diameter(unsigned int idx)
            {
            return *(Scalar *) (m_data + m_size*idx + 2*sizeof(Scalar4) + sizeof(Scalar3) + sizeof(Scalar));
            }

        //! get particle image by reference
        //! \param idx index of particle in array
        inline int3& image(unsigned int idx)
            {
            return *(int3 *) (m_data + m_size*idx + 2*sizeof(Scalar4) + sizeof(Scalar3) + 2*sizeof(Scalar));
            }
        //! get particle body id by reference
        //! \param idx index of particle in array
        inline unsigned int& body(unsigned int idx)
            {
            return *(unsigned int *) (m_data + m_size*idx + 2*sizeof(Scalar4) + sizeof(Scalar3) + 2*sizeof(Scalar) + sizeof(int3));
            }

        //! get the particle orientation by reference
        inline Scalar4& orientation(unsigned int idx)
            {
            return *(Scalar4 *) (m_data + m_size*idx + 2*sizeof(Scalar4) + sizeof(Scalar3) + 2*sizeof(Scalar) + sizeof(int3) + sizeof(unsigned int));
            }

        //! get the particle global tag by reference
        inline unsigned int& global_tag(unsigned int idx)
            {
            return *(unsigned int*) (m_data + m_size*idx + 2*sizeof(Scalar4) + sizeof(Scalar3) + 2*sizeof(Scalar) + sizeof(int3) + sizeof(unsigned int)+sizeof(Scalar4));
            }

        //! get size of per-particle data record (in number of bytes)
        static unsigned int getSize()
            {
            return m_size;
            }

    private:
        char *m_data;               //!< the data array
        static const unsigned int m_size = (2*sizeof(Scalar4)+sizeof(Scalar3)+2*sizeof(Scalar)+sizeof(int3)+sizeof(unsigned int)+sizeof(Scalar4)+sizeof(unsigned int));  //!< size of a data record per particle

    };


//! Class that handles MPI communication
class Communicator
    {
    public:
        //! Constructor
        /*! \param sysdef system definition the communicator is associated with
         *  \param mpi_comm the underlying MPI communicator
         *  \param mpi_init the MPI initializer the system was initialized with
         */
        Communicator(boost::shared_ptr<SystemDefinition> sysdef,
                     boost::shared_ptr<boost::mpi::communicator> mpi_comm,
                     boost::shared_ptr<MPIInitializer> mpi_init);

        //! \name accessor methods
        //@{

        //! Get the underlying MPI communicator
        /*! \return the boost MPI communicator
         */
        const boost::shared_ptr<const boost::mpi::communicator> getMPICommunicator()
            {
            return m_mpi_comm;
            }

        //! Set the profiler to use
        /*! \param prof Profiler to use with this class
         */
        void setProfiler(boost::shared_ptr<Profiler> prof)
            {
            m_prof = prof;
            }

        //! Get the dimensions of the global simulation box
        /*! \param dir direction to return dimensions for
         *  \return number of simulation boxes along the specified direction
         */
        unsigned int getDimension(unsigned int dir)
            {
            return m_mpi_init->getDimension(dir);
            }

        //@}

        //! \name communication methods
        //@{

        //! transfer particles between domains
        virtual void migrateAtoms();

        //! build a ghost particle list, copy ghost particle data to neighboring domains
        /*! \param width of ghost layer
         */
        virtual void exchangeGhosts(Scalar r_ghost);

        //! update ghost particle positions
        virtual void copyGhosts();

        //@}

    protected:

        //! Helper function to allocate internal buffers
        void allocate();

        GPUArray<unsigned int> m_delete_buf;     //!< buffer of particle indices that are going to be deleted

        GPUArray<char> m_sendbuf[6];             //!< per-direction buffer for particles that are sent
        GPUArray<char> m_recvbuf[6];             //!< per-direction buffer for particles that are received

        GPUArray<Scalar4> m_pos_copybuf[6];      //!< per-direction send buffer for particle positions to be copied
        GPUArray<Scalar> m_charge_copybuf[6];    //!< per-direction send buffer for particle charges to be copied
        GPUArray<Scalar> m_diameter_copybuf[6];  //!< per-direction send buffer for particle diameters to be copied

        GPUArray<unsigned int> m_copy_ghosts[6]; //!< per-direction list of indices of particles to send as ghosts to neighboring processors

        unsigned int m_num_copy_ghosts[6];       //!< number of local particles that are sent to neighboring processors
        unsigned int m_num_recv_ghosts[6];       //!< number of ghosts received per direction


        unsigned int m_max_copy_ghosts[6];       //!< max size of m_copy_ghosts array
        unsigned int m_max_ghost_copybuf;        //!< max  size of ghost particle data buffer

        unsigned int m_neighbors[6];             //!< MPI rank of neighbor domain  in every direction

        boost::shared_ptr<SystemDefinition> m_sysdef;              //!< system definitino
        boost::shared_ptr<ParticleData> m_pdata;                   //!< particle data
        boost::shared_ptr<const ExecutionConfiguration> exec_conf; //!< execution configuration
        boost::shared_ptr<const boost::mpi::communicator> m_mpi_comm; //!< MPI communciator
        boost::shared_ptr<MPIInitializer> m_mpi_init;              //!< MPI initializer
        boost::shared_ptr<Profiler> m_prof;                        //!< Profiler

        const BoxDim& m_global_box;              //!< global simulation box
        unsigned int m_packed_size;              //!< size of packed particle data element in bytes
        bool m_is_allocated;                     //!< true if internal buffers have been allocated
        Scalar m_r_ghost;                        //!< width of ghost layer

    };
#endif // __COMMUNICATOR_H__
#endif // ENABLE_MPI
