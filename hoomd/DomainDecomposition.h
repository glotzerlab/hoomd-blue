// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


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
#include "GlobalArray.h"

#include <set>
#include <vector>

#ifndef NVCC
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#endif


/*! \ingroup communication
*/

//! Class that initializes every processor using spatial domain-decomposition
/*! This class is used to divide the global simulation box into sub-domains and to assign a box to every processor.
 *
 *  <b>Implementation details</b>
 *
 *  One way to perform a domain decomposition is to subdivide the box into equal sized widths along each dimension.
 *  To achieve an optimal domain decomposition (i.e. minimal communication costs), the global domain is sub-divided
 *  such as to minimize surface area between domains, while utilizing all processors in the MPI communicator.
 *
 *  Alternatively, unequal sized cuts can be taken. This is advantageous for simulations with non-homogeneous
 *  particle distributions, e.g., a vapor-liquid interface. The user can specify N-1 of the fractions at construction
 *  time, provided that the specified fractions must create a grid commensurate with the number of ranks available.
 *  The final rank with is chosen so that the total box is covered. If the specified number of
 *  ranks does not match the number that is available, behavior is reverted to the normal default with
 *  uniform cuts along each dimension.
 *
 *  The initialization of the domain decomposition scheme is performed in the constructor.
 */
class PYBIND11_EXPORT DomainDecomposition
    {
#ifdef ENABLE_MPI
    public:
        //! Constructor
        /*! \param exec_conf The execution configuration
         * \param L Box lengths of global box to sub-divide
         * \param nx Requested number of domains along the x direction (0 == choose default)
         * \param ny Requested number of domains along the y direction (0 == choose default)
         * \param nz Requested number of domains along the z direction (0 == choose default)
         * \param twolevel If true, attempt two level decomposition (default == false)
         */
        DomainDecomposition(std::shared_ptr<ExecutionConfiguration> exec_conf,
                       Scalar3 L,
                       unsigned int nx = 0,
                       unsigned int ny = 0,
                       unsigned int nz = 0,
                       bool twolevel = false);

        //! Constructor for fixed fractions
        DomainDecomposition(std::shared_ptr<ExecutionConfiguration> exec_conf,
                            Scalar3 L,
                            const std::vector<Scalar>& fxs,
                            const std::vector<Scalar>& fys,
                            const std::vector<Scalar>& fzs);

        //! Calculate MPI ranks of neighboring domain.
        unsigned int getNeighborRank(unsigned int dir) const;

        //! Get domain indexer
        const Index3D& getDomainIndexer() const
            {
            return m_index;
            }

        //! Get the cartesian ranks lookup table (linear cartesian index -> rank)
        const GlobalArray<unsigned int>& getCartRanks() const
            {
            return m_cart_ranks;
            }

        //! Get the inverse lookup table (rank -> linear cartesian index)
        const GlobalArray<unsigned int>& getInverseCartRanks() const
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

        //! Get the cumulative box fraction at a specific rank index
        /*!
         * \param dir Direction (0=x, 1=y, 2=z) to get fraction
         * \param idx The rank index to get the cumulative fraction below (0 to N+1)
         * \returns Cumulative fraction of global box length below rank at \a idx
         */
        Scalar getCumulativeFraction(unsigned int dir, unsigned int idx) const
            {
            if (dir == 0)
                {
                assert(idx >= 0 && idx < m_nx +1);
                return m_cum_frac_x[idx];
                }
            else if (dir == 1)
                {
                assert(idx >= 0 && idx < m_ny+1);
                return m_cum_frac_y[idx];
                }
            else if (dir == 2)
                {
                assert(idx >= 0 && idx < m_nz+1);
                return m_cum_frac_z[idx];
                }
            else
                {
                m_exec_conf->msg->error() << "comm: requested direction does not exist" << std::endl;
                throw std::runtime_error("comm: requested direction does not exist");
                }
            }

        //! Get the cumulative box fractions along each dimension
        /*!
         * \param dir Direction (0=x, 1=y, 2=z) to get fraction
         * \returns Array of cumulative fractions of global box length below rank
         */
        std::vector<Scalar> getCumulativeFractions(unsigned int dir) const
            {
            if (dir == 0) return m_cum_frac_x;
            else if (dir == 1) return m_cum_frac_y;
            else if (dir == 2) return m_cum_frac_z;
            else
                {
                m_exec_conf->msg->error() << "comm: requested direction does not exist" << std::endl;
                throw std::runtime_error("comm: requested direction does not exist");
                }
            }

        //! Collectively set the cumulative fractions along a dimension from a given rank
        void setCumulativeFractions(unsigned int dir, const std::vector<Scalar>& cum_frac, unsigned int root);

        //! Get the dimensions of the local simulation box
        const BoxDim calculateLocalBox(const BoxDim& global_box);

        //! Get the rank for a particle to be placed
        unsigned int placeParticle(const BoxDim& global_box, Scalar3 pos, const unsigned int *cart_ranks);

        //! Get the number of grid cells in each dimension.
        uint3 getGridSize(void)const{return make_uint3(m_nx,m_ny,m_nz);}
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

        GlobalArray<unsigned int> m_cart_ranks; //!< A lookup-table to map the cartesian grid index onto ranks
        GlobalArray<unsigned int> m_cart_ranks_inv; //!< Inverse permutation of grid index lookup table

        //! Find a domain decomposition with given parameters
        bool findDecomposition(unsigned int nranks, Scalar3 L,
            unsigned int& nx, unsigned int& ny, unsigned int& nz);

        //! Find a two-level decomposition of the global grid
        void subdivide(unsigned int n_node_ranks, Scalar3 L,
            unsigned int nx, unsigned int ny, unsigned int nz,
            unsigned int& nx_intra, unsigned int &ny_intra, unsigned int& nz_intra);

        //! Helper method to group ranks by nodes
        void findCommonNodes();

        //! Helper method to initialize the two-level decomposition
        void initializeTwoLevel();

        //! Helper method to perform common grid initialization tasks in constructors
        void initializeDomainGrid(Scalar3 L,
                                  unsigned int nx,
                                  unsigned int ny,
                                  unsigned int nz,
                                  bool twolevel);

        //! Helper function to perform partial sums on fractional domain widths
        void initializeCumulativeFractions(const std::vector<Scalar>& fxs,
                                           const std::vector<Scalar>& fys,
                                           const std::vector<Scalar>& fzs);

        std::shared_ptr<ExecutionConfiguration> m_exec_conf; //!< The execution configuration
        const MPI_Comm m_mpi_comm; //!< MPI communicator

        std::vector<Scalar> m_cum_frac_x;   //!< Cumulative fractions in x below cut plane index
        std::vector<Scalar> m_cum_frac_y;   //!< Cumulative fractions in y below cut plane index
        std::vector<Scalar> m_cum_frac_z;   //!< Cumulative fractions in z below cut plane index
#endif // ENABLE_MPI
   };

#ifdef ENABLE_MPI
//! Export the domain decomposition information
void export_DomainDecomposition(pybind11::module& m);
#endif

#endif // __DOMAIN_DECOMPOSITION_H
