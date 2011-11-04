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

* All publications based on HOOMD-blue, including any reports or published
results obtained, in whole or in part, with HOOMD-blue, will acknowledge its use
according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website
at: http://codeblue.umich.edu/hoomd-blue/.

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

// Maintainer: joaander

#include <boost/shared_ptr.hpp>
#include <boost/signals.hpp>
#include "GPUArray.h"

#include "Index1D.h"
#include "Compute.h"

/*! \file CellList.h
    \brief Declares the CellList class
*/

#ifndef __CELLLIST_H__
#define __CELLLIST_H__

//! Computes a cell list from the particles in the system
/*! \b Overview:
    Cell lists are useful data structures when working with locality queries on particles. The most notable useage of
    cell lists in HOOMD is as an auxilliary data structure used when building the neighbor list. The data layout design
    decisions for CellList were made to optimize the performance of the neighbor list build as deteremined by
    microbenchmarking. However, CellList is written as generally as possible so that it can be used throughout the code
    in other locations where a cell list is needed.
    
    A cell list defines a set of cuboid cells that completely covers the simulation box. A single nominal width is given
    which is then rounded up to fit an integral number of cells across each dimension. A given particle belongs in one
    and only one cell, the one that contains the particle's x,y,z position.
    
    <b>Data storage:</b>
    
    All data is stored in GPUArrays for access on the host and device.
    
     - The \c cell_size array lists the number of members in each cell.
     - The \c xyzf array contains Scalar4 elements each of which holds the x,y,z coordinates of the particle and a flag.
       That flag can optionally be the particle index or its charge.
     - The \c tdb array contains Scalar4 elements each of which holds the type, diameter, and body of the particle.
       It is only computed is requested to reduce the computation time when it is not needed.
     - The cell_adj array lists indices of adjacent cells. A specified radius (3,5,7,...) of cells is included in the
       list.
    
    A given cell cuboid with x,y,z indices of i,j,k has a unique cell index. This index can be obtained from the Index3D
    object returned by getCellIndexer()
    \code
    Index3D cell_indexer = cl->getCellIndexer();
    cidx = cell_indexer(i,j,k);
    \endcode
    
    The other arrays are 2D (or 4D if you include i,j,k) in nature. They can be indexed with the appropriate Index2D
    from getCellListIndexer() or getCellAdjIndexer(). 
    \code
    Index2D cell_list_indexer = cl->getCellListIndexer();
    Index2D cell_adj_indexer = cl->getCellAdjIndexer();
    \endcode
    
     - <code>cell_size[cidx]</code> is the number of particles in the cell with index \c cidx
     - \c xyzf is Ncells x Nmax and <code>xyzf[cell_list_indexer(offset,cidx)]</code> is the data stored for particle
       \c offset in cell \c cidx (\c offset can vary from 0 to <code>cell_size[cidx]-1</code>)
     - \c tbd is structured identically to \c xyzf
     - <code>cell_adj[cell_adj_indexer(offset,cidx)]</code> is the cell index for neighboring cell \c offset to \c cidx.
       \c offset can vary from 0 to (radius*2+1)^3-1 (typically 26 with radius 1)
     
    <b>Parameters:</b>
     - \c width - minimum width of a cell in any x,y,z direction
     - \c radius - integer radius of cells to generate in \c cell_adj (1,2,3,4,...)
     - \c max_cells - maximum number of cells to allocate
     
    After a set call is made to adjust a parameter, changes do not take effect until the next call to compute().

    <b>Overvlow and error flag handling:</b>
    For easy support of derived GPU classes to implement overvlow detection and error handling, all error flags are
    stored in the GPUArray \a d_conditions.
     - 0: Maximum cell size (implementations are free to write to this element only in overflow conditions if they
          choose.)
     - 1: Set to non-zero if any particle has nan coordinates
     - 2: Set to non-zero if any particle is outside of the addressable bins

    Condition flags are to be set during the computeCellList() call and will be checked by compute() which will then 
    take the appropriate action. If possible, flags 1 and 2 should be set to the index of the particle causing the
    flag plus 1.
*/
class CellList : public Compute
    {
    public:
        //! Construct a cell list
        CellList(boost::shared_ptr<SystemDefinition> sysdef);
        
        virtual ~CellList();

        //! \name Set parameters
        // @{

        //! Set the minimum cell width in any dimension
        void setNominalWidth(Scalar width)
            {
            m_nominal_width = width;
            m_params_changed = true;
            }
        
        //! Set the radius of cells to include in the adjacency list
        void setRadius(unsigned int radius)
            {
            m_radius = radius;
            m_params_changed = true;
            }
        
        //! Set the maximum number of cells to allocate
        void setMaxCells(unsigned int max_cells)
            {
            m_max_cells = max_cells;
            m_params_changed = true;
            }
        
        //! Specify if the TDB cell list is to be computed
        void setComputeTDB(bool compute_tdb)
            {
            m_compute_tdb = compute_tdb;
            m_params_changed = true;
            }
        
        //! Specify that the flag is to be filled with the particle charge
        void setFlagCharge()
            {
            m_flag_charge = true;
            m_params_changed = true;
            }
        
        //! Specify that the flag is to be the particle index (encoded as an integer in the float variable)
        void setFlagIndex()
            {
            m_flag_charge = false;
            m_params_changed = true;
            }
        
        //! Notification of a particle resort
        void slotParticlesSorted()
            {
            m_particles_sorted = true;
            }
        
        //! Notification of a box size change
        void slotBoxChanged()
            {
            m_box_changed = true;
            }

        // @}
        //! \name Get properties
        // @{

        //! Get the actual width of the cells
        const Scalar3& getWidth() const
            {
            return m_width;
            }
        
        //! Get the dimensions of the cell list
        const uint3& getDim() const
            {
            return m_dim;
            }
        
        //! Get an indexer to identify cell indices
        const Index3D& getCellIndexer() const
            {
            return m_cell_indexer;
            }
        
        //! Get an indexer to index into the cell lists
        const Index2D& getCellListIndexer() const
            {
            return m_cell_list_indexer;
            }
        
        //! Get an indexer to index into the adjacency list
        const Index2D& getCellAdjIndexer() const
            {
            return m_cell_adj_indexer;
            }
        
        //! Get number of memory slots allocated for each cell
        const unsigned int getNmax() const
            {
            return m_Nmax;
            }
        
        // @}
        //! \name Get data
        // @{
        
        //! Get the array of cell sizes
        const GPUArray<unsigned int>& getCellSizeArray() const
            {
            return m_cell_size;
            }
        
        //! Get the adjacency list
        const GPUArray<unsigned int>& getCellAdjArray() const
            {
            return m_cell_adj;
            }
        
        //! Get the cell list containing x,y,z,flag
        const GPUArray<Scalar4>& getXYZFArray() const
            {
            return m_xyzf;
            }
        
        //! Get the cell list containting t,d,b
        const GPUArray<Scalar4>& getTDBArray() const
            {
            return m_tdb;
            }
            
        //! Compute the cell list given the current particle positions
        void compute(unsigned int timestep);
        
        //! Benchmark the computation
        double benchmark(unsigned int num_iters);
        
        // @}
        
    protected:
        // user specified parameters
        Scalar m_nominal_width;      //!< Minimum width of cell in any direction
        unsigned int m_radius;       //!< Radius of adjacency bins to list
        unsigned int m_max_cells;    //!< Maximum number of cells to allocate
        bool m_compute_tdb;          //!< true if the tdb list should be computed
        bool m_flag_charge;          //!< true if the flag should be set to the charge, it will be index otherwise
        bool m_params_changed;       //!< Set to true when parameters are changed
        bool m_particles_sorted;     //!< Set to true when the particles have been sorted
        bool m_box_changed;          //!< Set to ttrue when the box size has changed
        
        // parameters determined by initialize
        Scalar3 m_width;             //!< Actual width
        uint3 m_dim;                 //!< Current dimensions
        Index3D m_cell_indexer;      //!< Indexes cells from i,j,k
        Index2D m_cell_list_indexer; //!< Indexes elements in the cell list
        Index2D m_cell_adj_indexer;  //!< Indexes elements in the cell adjacency list
        unsigned int m_Nmax;         //!< Numer of spaces reserved for particles in each cell
        
        // values computed by compute()
        GPUArray<unsigned int> m_cell_size;  //!< Number of members in each cell
        GPUArray<unsigned int> m_cell_adj;   //!< Cell adjacency list
        GPUArray<Scalar4> m_xyzf;            //!< Cell list with position and flags
        GPUArray<Scalar4> m_tdb;             //!< Cell list with type,diameter,body
        GPUArray<unsigned int> m_conditions; //!< Condition flags set during the computeCellList() call
        
        boost::signals::connection m_sort_connection;        //!< Connection to the ParticleData sort signal
        boost::signals::connection m_boxchange_connection;   //!< Connection to the ParticleData box size change signal
        
        //! Computes what the dimensions should me
        uint3 computeDimensions();
        
        //! Initialize width and indexers, allocates memory
        void initializeAll();
        
        //! Initialize width
        void initializeWidth();
        
        //! Initialize indexers and allocate memory
        void initializeMemory();
        
        //! Initializes values in the cell_adj array
        void initializeCellAdj();
        
        //! Compute the cell list
        virtual void computeCellList();

        //! Check the status of the conditions
        bool checkConditions();

        //! Resets the condition status
        void resetConditions();
    };

//! Export the CellList class to python
void export_CellList();

#endif

