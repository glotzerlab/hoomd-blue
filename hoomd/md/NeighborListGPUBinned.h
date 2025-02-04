// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "NeighborListGPU.h"
#include "hoomd/Autotuner.h"
#include "hoomd/CellListGPU.h"

/*! \file NeighborListGPUBinned.h
    \brief Declares the NeighborListGPUBinned class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __NEIGHBORLISTGPUBINNED_H__
#define __NEIGHBORLISTGPUBINNED_H__

namespace hoomd
    {
namespace md
    {
//! Neighbor list build on the GPU
/*! Implements the O(N) neighbor list build on the GPU using a cell list.

    GPU kernel methods are defined in NeighborListGPUBinned.cuh and defined in
   NeighborListGPUBinned.cu.

    \ingroup computes
*/
class PYBIND11_EXPORT NeighborListGPUBinned : public NeighborListGPU
    {
    public:
    //! Constructs the compute
    NeighborListGPUBinned(std::shared_ptr<SystemDefinition> sysdef, Scalar r_buff);

    //! Destructor
    virtual ~NeighborListGPUBinned();

    /// Notify NeighborList that a r_cut matrix value has changed
    virtual void notifyRCutMatrixChange()
        {
        m_update_cell_size = true;
        NeighborListGPU::notifyRCutMatrixChange();
        }

    /// Make the neighborlist deterministic
    void setDeterministic(bool deterministic)
        {
        m_cl->setSortCellList(deterministic);
        }

    /// Get the deterministic flag
    bool getDeterministic()
        {
        return m_cl->getSortCellList();
        }

    /// Get the dimensions of the cell list
    const uint3& getDim() const
        {
        return m_cl->getDim();
        }

    /// Get number of memory slots allocated for each cell
    const unsigned int getNmax() const
        {
        return m_cl->getNmax();
        }

    /// Start autotuning kernel launch parameters
    virtual void startAutotuning()
        {
        NeighborListGPU::startAutotuning();

        // Start autotuning the cell list.
        m_cl->startAutotuning();
        }

    /// Check if autotuning is complete.
    virtual bool isAutotuningComplete()
        {
        bool result = NeighborListGPU::isAutotuningComplete();
        result = result && m_cl->isAutotuningComplete();
        return result;
        }

    protected:
    std::shared_ptr<CellList> m_cl; //!< The cell list
    bool m_use_index;               //!< True for indirect lookup of particle data via index

    /// Track when the cell size needs to be updated
    bool m_update_cell_size = true;

    std::shared_ptr<Autotuner<2>> m_tuner; //!< Autotuner for block size and threads per particle

    //! Builds the neighbor list
    virtual void buildNlist(uint64_t timestep);
    };

    } // end namespace md
    } // end namespace hoomd

#endif
