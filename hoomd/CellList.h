// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "GlobalArray.h"
#include "HOOMDMath.h"

#include "Compute.h"
#include "Index1D.h"

#include <hoomd/extern/nano-signal-slot/nano_signal_slot.hpp>
#include <memory>

/*! \file CellList.h
    \brief Declares the CellList class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __CELLLIST_H__
#define __CELLLIST_H__

namespace hoomd
    {
//! Computes a cell list from the particles in the system
/*! \b Overview:
    Cell lists are useful data structures when working with locality queries on particles. The most
   notable usage of cell lists in HOOMD is as an auxiliary data structure used when building the
   neighbor list. The data layout design decisions for CellList were made to optimize the
   performance of the neighbor list build as determined by microbenchmarking. However, CellList is
   written as generally as possible so that it can be used throughout the code in other locations
   where a cell list is needed.

    A cell list defines a set of cuboid cells that completely covers the simulation box. A single
   nominal width is given which is then rounded up to fit an integral number of cells across each
   dimension. A given particle belongs in one and only one cell, the one that contains the
   particle's x,y,z position.

    <b>Data storage:</b>

    All data is stored in GlobalArrays for access on the host and device, and concurrent access
   between GPUs.

     - The \c cell_size array lists the number of members in each cell.
     - The \c xyzf array contains Scalar4 elements each of which holds the x,y,z coordinates of the
   particle and a flag. That flag can optionally be the particle index, its charge, or its type.
     - The \c type_body array contains uint2 elements each of which holds the type, and body
   of the particle. It is only computed is requested to reduce the computation time when it is not
   needed.
     - The \c orientation array contains Scalar4 elements listing the orientation of each particle.
       It is only computed is requested to reduce the computation time when it is not needed.
     - The \c idx array contains unsigned int elements listing the index of each particle. It is
   useful when xyzf is set to hold type. It is only computed is requested to reduce the computation
   time when it is not needed.
     - The cell_adj array lists indices of adjacent cells. A specified radius (3,5,7,...) of cells
   is included in the list.

    A given cell cuboid with x,y,z indices of i,j,k has a unique cell index. This index can be
   obtained from the Index3D object returned by getCellIndexer() \code Index3D cell_indexer =
   cl->getCellIndexer(); cidx = cell_indexer(i,j,k); \endcode

    The other arrays are 2D (or 4D if you include i,j,k) in nature. They can be indexed with the
   appropriate Index2D from getCellListIndexer() or getCellAdjIndexer(). \code Index2D
   cell_list_indexer = cl->getCellListIndexer(); Index2D cell_adj_indexer = cl->getCellAdjIndexer();
    \endcode

     - <code>cell_size[cidx]</code> is the number of particles in the cell with index \c cidx
     - \c xyzf is Ncells x Nmax and <code>xyzf[cell_list_indexer(offset,cidx)]</code> is the data
   stored for particle \c offset in cell \c cidx (\c offset can vary from 0 to
   <code>cell_size[cidx]-1</code>)
     - \c tbd, idx, and orientation is structured identically to \c xyzf
     - <code>cell_adj[cell_adj_indexer(offset,cidx)]</code> is the cell index for neighboring cell
   \c offset to \c cidx. \c offset can vary from 0 to (radius*2+1)^3-1 (typically 26 with radius 1)

    <b>Parameters:</b>
     - \c width - minimum width of a cell in any x,y,z direction
     - \c radius - integer radius of cells to generate in \c cell_adj (1,2,3,4,...)
     - \c multiple - Round down to the nearest multiple number of cells in each direction (only
   applied to cells inside the domain, not the ghost cells).

    After a set call is made to adjust a parameter, changes do not take effect until the next call
   to compute().

    <b>Overflow and error flag handling:</b>
    For easy support of derived GPU classes to implement overflow detection and error handling, all
   error flags are stored in the GlobalArray \a d_conditions.
     - 0: Maximum cell size (implementations are free to write to this element only in overflow
   conditions if they choose.)
     - 1: Set to non-zero if any particle has nan coordinates
     - 2: Set to non-zero if any particle is outside of the addressable bins

    Condition flags are to be set during the computeCellList() call and will be checked by compute()
   which will then take the appropriate action. If possible, flags 1 and 2 should be set to the
   index of the particle causing the flag plus 1.
*/
class PYBIND11_EXPORT CellList : public Compute
    {
    public:
    //! Construct a cell list
    CellList(std::shared_ptr<SystemDefinition> sysdef);

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

    //! Specify if the XYZ,flag cell list is to be computed
    void setComputeXYZF(bool compute_xyzf)
        {
        m_compute_xyzf = compute_xyzf;
        m_params_changed = true;
        }

    //! Specify if the TypeBody cell list is to be computed
    void setComputeTypeBody(bool compute_type_body)
        {
        m_compute_type_body = compute_type_body;
        m_params_changed = true;
        }

    //! Specify if the orientation cell list is to be computed
    void setComputeOrientation(bool compute_orientation)
        {
        m_compute_orientation = compute_orientation;
        m_params_changed = true;
        }

    //! Specify if the index cell list is to be computed
    void setComputeIdx(bool compute_idx)
        {
        m_compute_idx = compute_idx;
        m_params_changed = true;
        }

    //! Specify that the flag is to be filled with the particle charge
    void setFlagCharge()
        {
        m_flag_charge = true;
        m_flag_type = false;
        m_params_changed = true;
        }

    //! Specify that the flag is to be filled with the particle type
    void setFlagType()
        {
        m_flag_charge = false;
        m_flag_type = true;
        m_params_changed = true;
        }

    //! Specify that the flag is to be the particle index (encoded as an integer in the Scalar
    //! variable)
    void setFlagIndex()
        {
        m_flag_charge = false;
        m_flag_type = false;
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

    //! Set the multiple value
    void setMultiple(unsigned int multiple)
        {
        if (multiple != 0)
            m_multiple = multiple;
        else
            m_multiple = 1;
        }

    //! Set the sort flag
    void setSortCellList(bool sort)
        {
        m_sort_cell_list = sort;
        m_params_changed = true;
        }

    /// Get whether the cell list is sorted
    bool getSortCellList()
        {
        return m_sort_cell_list;
        }

    //! Set the flag to compute the cell adjacency list
    void setComputeAdjList(bool compute_adj_list)
        {
        m_compute_adj_list = compute_adj_list;
        m_params_changed = true;
        }

    //! Request a multi-GPU cell list
    virtual void setPerDevice(bool per_device)
        {
        // base class does nothing
        }

    //! Return true if we maintain a cell list per device
    virtual bool getPerDevice() const
        {
        // base class doesn't support GPU
        return false;
        }

    // @}
    //! \name Get properties
    // @{

    //! Get the nominal width of the cells
    Scalar getNominalWidth() const
        {
        return m_nominal_width;
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

    //! Get width of ghost cells
    const Scalar3 getGhostWidth() const
        {
        return m_ghost_width;
        }

    //! Get the actual cell width that was computed (includes ghost layer)
    const Scalar3 getCellWidth() const
        {
        return m_actual_width;
        }

    // @}
    //! \name Get data
    // @{

    //! Get the array of cell sizes
    const GlobalArray<unsigned int>& getCellSizeArray() const
        {
        return m_cell_size;
        }

    //! Get the array of cell sizes (per device)
    virtual const GlobalArray<unsigned int>& getCellSizeArrayPerDevice() const
        {
        throw std::runtime_error("Per-device cell size array not available in base class.\n");
        }

    //! Get the adjacency list
    const GlobalArray<unsigned int>& getCellAdjArray() const
        {
        if (!m_compute_adj_list)
            {
            throw std::runtime_error("Cell adjacency list not available");
            }
        return m_cell_adj;
        }

    //! Get the cell list containing x,y,z,flag
    const GlobalArray<Scalar4>& getXYZFArray() const
        {
        return m_xyzf;
        }

    //! Get the cell list containing t,b
    const GlobalArray<uint2>& getTypeBodyArray() const
        {
        return m_type_body;
        }

    //! Get the cell list containing orientation
    const GlobalArray<Scalar4>& getOrientationArray() const
        {
        return m_orientation;
        }

    //! Get the cell list containing index
    const GlobalArray<unsigned int>& getIndexArray() const
        {
        return m_idx;
        }

    //! Get the cell list containing index (per device)
    virtual const GlobalArray<unsigned int>& getIndexArrayPerDevice() const
        {
        // base class returns an empty array
        throw std::runtime_error("Per-device cell index array not available in base class.\n");
        }

    //! Compute the cell list given the current particle positions
    void compute(uint64_t timestep);

    // @}

    /*! \param func Function to call when the cell width changes
        \return Connection to manage the signal/slot connection
        Calls are performed by using nano_signal_slot. The function passed in
        \a func will be called every time the CellList is notified of a change in the cell width
        \note If the caller class is destroyed, it needs to disconnect the signal connection
        via \b con.disconnect where \b con is the return value of this function.
    */
    Nano::Signal<void()>& getCellWidthChangeSignal()
        {
        return m_width_change;
        }

    protected:
    // user specified parameters
    Scalar m_nominal_width;     //!< Minimum width of cell in any direction
    unsigned int m_radius;      //!< Radius of adjacency bins to list
    bool m_compute_xyzf;        //!< true if the xyzf list should be computed
    bool m_compute_type_body;   //!< true if the TypeBody list should be computed
    bool m_compute_orientation; //!< true if the orientation list should be computed
    bool m_compute_idx;         //!< true if the idx list should be computed
    bool m_flag_charge;      //!< true if the flag should be set to the charge, it will be index (or
                             //!< type) otherwise
    bool m_flag_type;        //!< true if the flag should be set to type, it will be index otherwise
    bool m_params_changed;   //!< Set to true when parameters are changed
    bool m_particles_sorted; //!< Set to true when the particles have been sorted
    bool m_box_changed;      //!< Set to true when the box size has changed
    unsigned int m_multiple; //!< Round cell dimensions down to a multiple of this value

    // parameters determined by initialize
    uint3 m_dim;                 //!< Current dimensions
    Index3D m_cell_indexer;      //!< Indexes cells from i,j,k
    Index2D m_cell_list_indexer; //!< Indexes elements in the cell list
    Index2D m_cell_adj_indexer;  //!< Indexes elements in the cell adjacency list
    unsigned int m_Nmax;         //!< Numer of spaces reserved for particles in each cell
    Scalar3 m_actual_width;      //!< Actual width of a cell in each direction
    Scalar3 m_ghost_width;       //!< Width of ghost layer sized for (on one side only)

    // values computed by compute()
    GlobalArray<unsigned int> m_cell_size; //!< Number of members in each cell
    GlobalArray<unsigned int> m_cell_adj;  //!< Cell adjacency list
    GlobalArray<Scalar4> m_xyzf;           //!< Cell list with position and flags
    GlobalArray<uint2> m_type_body;        //!< Cell list with type,body
    GlobalArray<Scalar4> m_orientation;    //!< Cell list with orientation
    GlobalArray<unsigned int> m_idx;       //!< Cell list with index
    GlobalArray<uint3> m_conditions; //!< Condition flags set during the computeCellList() call

    bool m_sort_cell_list;   //!< If true, sort cell list
    bool m_compute_adj_list; //!< If true, compute the cell adjacency lists

#ifdef ENABLE_MPI
    /// The system's communicator.
    std::shared_ptr<Communicator> m_comm;
#endif

    //! Computes what the dimensions should me
    uint3 computeDimensions();

    //! Initialize width and indexers, allocates memory
    void initializeAll();

    //! Initialize width
    void initializeWidth();

    //! Initialize indexers and allocate memory
    virtual void initializeMemory();

    //! Initializes values in the cell_adj array
    void initializeCellAdj();

    //! Compute the cell list
    virtual void computeCellList();

    //! Check the status of the conditions
    bool checkConditions();

    //! Reads back the conditions
    virtual uint3 readConditions();

    //! Resets the condition status
    virtual void resetConditions();

    Nano::Signal<void()> m_width_change; //!< Signal that is triggered when the cell width changes
    };

namespace detail
    {
//! Export the CellList class to python
#ifndef __HIPCC__
void export_CellList(pybind11::module& m);
#endif
    } // end namespace detail

    } // end namespace hoomd
#endif
