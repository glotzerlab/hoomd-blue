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

/*! \file DomainDecomposition.h
    \brief Defines the DomainDecomposition class
*/

#ifndef __DOMAIN_DECOMPOSITION_H__
#define __DOMAIN_DECOMPOSITION_H__

#include "HOOMDMath.h"
#include "Index1D.h"
#include "BoxDim.h"
#include "ExecutionConfiguration.h"
#include "GPUArray.h"

/*! \ingroup communication
*/

//! Class that initializes and holds information about the domain decomposition

//! Class that initializes every processor using spatial domain-decomposition
/*! This class is used to divide the global simulation box into sub-domains and to assign a box to every processor.
 *
 *  <b>Implementation details</b>
 *
 *  To achieve an optimal domain decomposition (i.e. minimal communication costs), the global domain is sub-divided
 *  such as to minimize surface area between domains, while utilizing all processors in the MPI communicator.
 *
 *  The initialization of the domain decomposition scheme is performed in the constructor.
 */
class DomainDecomposition
    {
#ifdef ENABLE_MPI
    public:
        //! Constructor
        /*! \param exec_conf The execution configuration
         * \param L Box lengths of global box to sub-divide
         * \param nx Requested number of domains along the x direction (0 == choose default)
         * \param ny Requested number of domains along the y direction (0 == choose default)
         * \param nz Requested number of domains along the z direction (0 == choose default)
         */
        DomainDecomposition(boost::shared_ptr<ExecutionConfiguration> exec_conf,
                       Scalar3 L,
                       unsigned int nx = 0,
                       unsigned int ny = 0,
                       unsigned int nz = 0);

        //! Calculate MPI ranks of neighboring domain.
        unsigned int getNeighborRank(unsigned int dir) const;

        //! Get domain indexer
        const Index3D& getDomainIndexer() const
            {
            return m_index;
            }

        //! Get the cartesian ranks lookup table (linear cartesian index -> rank)
        const GPUArray<unsigned int>& getCartRanks() const
            {
            return m_cart_ranks;
            }

        //! Get the inverse lookup table (rank -> linear cartesian index)
        const GPUArray<unsigned int>& getInverseCartRanks() const
            {
            return m_cart_ranks_inv;
            }


        //! Get the grid position of this rank
        uint3 getGridPos() const
            {
            return m_grid_pos;
            }

        //! Determines whether the local box shares a boundary with the global box
        bool isAtBoundary(unsigned int dir) const;

        //! Get the dimensions of the local simulation box
        const BoxDim calculateLocalBox(const BoxDim& global_box);

        //! Get the rank for a particle to be placed
        /*! \param pos Particle position
         * \returns the rank of the processor that should receive the particle
         */
        unsigned int placeParticle(const BoxDim& global_box, Scalar3 pos);

    private:
        unsigned int m_nx;           //!< Number of processors along the x-axis
        unsigned int m_ny;           //!< Number of processors along the y-axis
        unsigned int m_nz;           //!< Number of processors along the z-axis

        uint3 m_grid_pos;            //!< Position of this domain in the grid
        Index3D m_index;             //!< Index to the 3D processor grid
        Index3D m_node_grid;         //!< Indexer of the grid of nodes
        Index3D m_intra_node_grid;   //!< The grid in every node

        std::set<std::string> m_nodes; //!< List of nodes
        std::multimap<std::string, unsigned int> m_node_map; //!< Map of ranks per node
        unsigned int m_max_n_node;   //!< Maximum number of ranks on a node
        bool m_twolevel;             //!< Whether we use a two-level decomposition

        GPUArray<unsigned int> m_cart_ranks; //!< A lookup-table to map the cartesian grid index onto ranks
        GPUArray<unsigned int> m_cart_ranks_inv; //!< Inverse permutation of grid index lookup table

        //! Find a domain decomposition with given parameters
        bool findDecomposition(unsigned int nranks, Scalar3 L, unsigned int& nx, unsigned int& ny, unsigned int& nz);

        //! Helper method to group ranks by nodes
        void findCommonNodes();

        //! Helper method to initialize the two-level decomposition
        void initializeTwoLevel();

        boost::shared_ptr<ExecutionConfiguration> m_exec_conf; //!< The execution configuration
        const MPI_Comm m_mpi_comm; //!< MPI communicator
#endif // ENABLE_MPI
   };

#ifdef ENABLE_MPI
//! Export the domain decomposition information
void export_DomainDecomposition();
#endif

#endif // __DOMAIN_DECOMPOSITION_H
