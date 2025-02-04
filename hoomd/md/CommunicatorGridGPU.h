// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __COMMUNICATOR_GRID_GPU_H__
#define __COMMUNICATOR_GRID_GPU_H__

#ifdef ENABLE_HIP
#include "CommunicatorGrid.h"

namespace hoomd
    {
namespace md
    {
#ifdef ENABLE_MPI
/*! Class to communicate the boundary layer of a regular grid (GPU version)
 */
template<typename T> class CommunicatorGridGPU : public CommunicatorGrid<T>
    {
    public:
    //! Constructor
    CommunicatorGridGPU(std::shared_ptr<SystemDefinition> sysdef,
                        uint3 dim,
                        uint3 embed,
                        uint3 offset,
                        bool add_outer_layer_to_inner);

    //! Communicate grid
    virtual void communicate(const GlobalArray<T>& grid);

    protected:
    unsigned int m_n_unique_recv_cells; //!< Number of unique receiving cells

    //! Initialize grid communication
    virtual void initGridCommGPU();

    private:
    GlobalArray<unsigned int>
        m_cell_recv; //!< Array of per-cell receive elements (multiple possible)
    GlobalArray<unsigned int> m_cell_recv_begin; //!< Begin of recv indices per cell
    GlobalArray<unsigned int> m_cell_recv_end;   //!< End of recv indices per cell
    };

    } // end namespace md
    } // end namespace hoomd

#endif // ENABLE_MPI
#endif // __COMMUNICATOR_GRID_GPU_H__
#endif
