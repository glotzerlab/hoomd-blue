// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "Autotuner.h"
#include "CellList.h"

/*! \file CellListGPU.h
    \brief Declares the CellListGPU class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __CELLLISTGPU_H__
#define __CELLLISTGPU_H__

namespace hoomd
    {
//! Computes a cell list from the particles in the system on the GPU
/*! Calls GPU functions in CellListGPU.cuh and CellListGPU.cu
    \sa CellList
    \ingroup computes
*/
class PYBIND11_EXPORT CellListGPU : public CellList
    {
    public:
    //! Construct a cell list
    CellListGPU(std::shared_ptr<SystemDefinition> sysdef);

    virtual ~CellListGPU() {};

    //! Request a multi-GPU cell list
    virtual void setPerDevice(bool per_device)
        {
        if (per_device && !this->m_exec_conf->allConcurrentManagedAccess())
            throw std::runtime_error("Per-device cell list only supported with unified memory.");

        m_per_device = per_device;
        m_params_changed = true;
        }

    //! Return true if we maintain a cell list per device
    virtual bool getPerDevice() const
        {
        return m_per_device;
        }

    //! Get the cell list containing index (per device)
    virtual const GlobalArray<unsigned int>& getIndexArrayPerDevice() const
        {
        return m_idx_scratch;
        }

    //! Get the array of cell sizes (per device)
    virtual const GlobalArray<unsigned int>& getCellSizeArrayPerDevice() const
        {
        return m_cell_size_scratch;
        }

    protected:
    GlobalArray<unsigned int>
        m_cell_size_scratch; //!< Number of members in each cell, one list per GPU
    GlobalArray<unsigned int> m_cell_adj_scratch; //!< Cell adjacency list, one list per GPU
    GlobalArray<Scalar4> m_xyzf_scratch;    //!< Cell list with position and flags, one list per GPU
    GlobalArray<uint2> m_type_body_scratch; //!< Cell list with type,body, one list per GPU
    GlobalArray<Scalar4> m_orientation_scratch; //!< Cell list with orientation, one list per GPU
    GlobalArray<unsigned int> m_idx_scratch;    //!< Cell list with index, one list per GPU

    bool m_per_device; //!< True if we maintain a per-GPU cell list

    //! Compute the cell list
    virtual void computeCellList();

    // Initialize GPU-specific data storage
    virtual void initializeMemory();

    //! Combine the per-device cell lists
    virtual void combineCellLists();

    /// Autotune block sizes for main kernel.
    std::shared_ptr<Autotuner<1>> m_tuner;

    /// Autotune block sizes for combination kernel.
    std::shared_ptr<Autotuner<1>> m_tuner_combine;
    };

namespace detail
    {
//! Exports CellListGPU to python
void export_CellListGPU(pybind11::module& m);
    } // end namespace detail

    } // end namespace hoomd
#endif
