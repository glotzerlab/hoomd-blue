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

/*! \file MPIInitializer.h
    \brief Defines the MPInitializer class
*/

#ifndef __MPI_INITIALIZER_H__
#define __MPI_INITIALIZER_H__

#ifdef ENABLE_MPI
#include "HOOMDMath.h"
#include "ParticleData.h"


namespace boost
   {
   namespace mpi
       {
       //! Forward declaration
       class communicator;
       }

    //! Serialization functions for some of our data types
    namespace serialization
        {
        //! serialization of uint3
        template<class Archive>
        void serialize(Archive & ar, uint3 & u, const unsigned int version)
            {
            ar & u.x;
            ar & u.y;
            ar & u.z;
            }

        //! serialization of BoxDim
        template<class Archive>
        void serialize(Archive & ar, BoxDim & box, const unsigned int version)
            {
            ar & box.xlo;
            ar & box.xhi;
            ar & box.ylo;
            ar & box.yhi;
            ar & box.zlo;
            ar & box.zhi;
            }

        //! Serialization of Scalar3
        template<class Archive>
        void serialize(Archive & ar, Scalar3 & s, const unsigned int version)
            {
            ar & s.x;
            ar & s.y;
            ar & s.z;
            }

        //! Serialization of Scalar4
        template<class Archive>
        void serialize(Archive & ar, Scalar4 & s, const unsigned int version)
            {
            ar & s.x;
            ar & s.y;
            ar & s.z;
            ar & s.w;
            }

        //! Serialization of int3
        template<class Archive>
        void serialize(Archive & ar, int3 & i, const unsigned int version)
            {
            ar & i.x;
            ar & i.y;
            ar & i.z;
            }

        }
    }

//! Forward definitions
class SystemDefinition;
class ParticleData;

/*! \ingroup communication
*/

//! Class that initializes every processor using spatial domain-decomposition
/*! This class is used to set up the initial state of the particle data on every processor.
 *  The global simulation box is sub-divided into domains which are assigned to
 *  individual processors. The criterium whether a particle belongs into a certain box is
 *  based upon its position.
 *
 *  There are two main methods defined by this class: scatter() and gather().
 *
 *  The scatter() method distributes ParticleData from the processor with rank \b root to the other processors,
 *
 *  The gather() method recombines all ParticleData of the processors on the processor with rank \b root.
 *
 *  <b>Implementation details</b>
 *
 *  To achieve an optimal domain decomposition (i.e. minimal communication costs), the global domain is sub-divided
 *  such as to minimize surface area between domains, while utilizing all processors in the MPI communicator.
 *
 *  The initialization of the domain decomposition scheme is performed in the constructor.
 */
class MPIInitializer
    {
    public:
        //! Constructor
        /*! \param sysdef System definition of the local system this initializer acts upon
         * \param comm MPI communicator to use to initialize the sub-domains
         * \param root Rank of processor to perform the domain decomposition on
         * \param nx Requested number of domains along the x direction (0 == choose default)
         * \param ny Requested number of domains along the y direction (0 == choose default)
         * \param nz Requested number of domains along the z direction (0 == choose default)
         */
        MPIInitializer(boost::shared_ptr<SystemDefinition> sysdef,
                       boost::shared_ptr<boost::mpi::communicator> comm,
                       unsigned int root,
                       unsigned int nx = 0,
                       unsigned int ny = 0,
                       unsigned int nz = 0);
 
        //! Distribute particles onto processors
        /*! \param root Rank of the processor to distribute particles from
         */
        void scatter(unsigned int root);
 
        //! Gather particle data from all processors into a snapshot on a single processor
        /*! \param root Rank of processor to gather particle data on
         *  \param global_snapshot Snapshot to collect particle data in
         */
        void gatherSnapshot(SnapshotParticleData &global_snapshot, unsigned int root);
 
        //! Calculate MPI ranks of neighboring domain
        /*! \param dir neighbor direction to calculate rank for<br>
        *  dir =<br>
        *        0 <-> east <br>
        *        1 <-> west <br>
        *        2 <-> north <br>
        *        3 <-> south <br>
        *        4 <-> up <br>
        *        5 <-> down <br>
        *
        *  \return rank of neighbor in the specified direction
        */
        virtual unsigned int getNeighborRank(unsigned int dir);
 
        //! Get the global simulation box
        /*! \return Dimensions of the global simulation box
        */
        virtual const BoxDim getGlobalBox()
        {
        return m_global_box;
        }
 
        //! Get the number of simulation boxes along a certain direction
        /*! \param dir Direction (\b dir = 0, 1, 2)
         * \return Number of boxes along the specified direction
         */
        virtual unsigned int getDimension(unsigned int dir) const;
 
        //! Determine whether this box shares a boundary with the global simulation box
        /*! \param dir Direction (0 <= \b dir < 6)
         */
        bool isAtBoundary(unsigned int dir) const;
 
    private:
        unsigned int m_N;              //!< Number of particles on this processor
        unsigned int m_nglobal;        //!< Global number of particles

        unsigned int m_rank;                             //!< Rank of this processor
        std::vector<BoxDim> m_box_proc;                  //!< Box dimensions of every processor
        std::vector<uint3> m_grid_pos_proc;              //!< Grid position of every processor
        unsigned m_num_particle_types;                   //!< Number of particle types
        std::vector<std::string> m_type_mapping;         //!< Number of particle types
 
 
        Scalar m_Lx;         //!< Length of this box in x direction
        Scalar m_Ly;         //!< Length of this box in y direction
        Scalar m_Lz;         //!< Length of this box in z direction
 
        uint3  m_grid_pos;   //!< This processor's position in the grid
 
        unsigned int m_nx;   //!< Grid dimensions in x direction
        unsigned int m_ny;   //!< Grid dimensions in y direction
        unsigned int m_nz;   //!< Grid dimensions in z direction
 
        boost::shared_ptr<SystemDefinition> m_sysdef; //!< Definition of the local simulation
        boost::shared_ptr<ParticleData> m_pdata;      //!< Local particle data
        boost::shared_ptr<boost::mpi::communicator> m_mpi_comm; //!< MPI communicator
 
        BoxDim m_global_box;                             //!< Global simulation box
        BoxDim m_box;                                    //!< Dimensions of this box
 
        //! Find a domain decomposition with given parameters
        bool findDecomposition(unsigned int& nx, unsigned int& ny, unsigned int& nz);
    };

//! Declare function that exports MPIInitializer to python
void export_MPIInitializer();

#endif // __MPI_INITIALIZER_H
#endif // ENABLE_MPI
