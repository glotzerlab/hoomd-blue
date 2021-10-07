// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: joaander

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

    //! Set the autotuner period
    void setTuningParam(unsigned int param)
        {
        m_param = param;
        }

    //! Set autotuner parameters
    /*! \param enable Enable/disable autotuning
        \param period period (approximate) in time steps when returning occurs
    */
    virtual void setAutotunerParams(bool enable, unsigned int period)
        {
        NeighborListGPU::setAutotunerParams(enable, period);
        m_tuner->setPeriod(period / 10);
        m_tuner->setEnabled(enable);
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

    protected:
    std::shared_ptr<CellList> m_cl; //!< The cell list
    unsigned int m_block_size;      //!< Block size to execute on the GPU
    unsigned int m_param;           //!< Kernel tuning parameter
    bool m_use_index;               //!< True for indirect lookup of particle data via index

    /// Track when the cell size needs to be updated
    bool m_update_cell_size = true;

    std::unique_ptr<Autotuner> m_tuner; //!< Autotuner for block size and threads per particle

    //! Builds the neighbor list
    virtual void buildNlist(uint64_t timestep);
    };

namespace detail
    {
//! Exports NeighborListGPUBinned to python
void export_NeighborListGPUBinned(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
