// Copyright (c) 2009-2025 The Regents of the University of Michigan.
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

    virtual ~CellListGPU() { };

    protected:
    //! Compute the cell list
    virtual void computeCellList();

    /// Autotune block sizes for main kernel.
    std::shared_ptr<Autotuner<1>> m_tuner;
    };

namespace detail
    {
//! Exports CellListGPU to python
void export_CellListGPU(pybind11::module& m);
    } // end namespace detail

    } // end namespace hoomd
#endif
