// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "CellList.h"
#include "Autotuner.h"

/*! \file CellListGPU.h
    \brief Declares the CellListGPU class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __CELLLISTGPU_H__
#define __CELLLISTGPU_H__

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

        virtual ~CellListGPU();

        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs
        */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            CellList::setAutotunerParams(enable, period);
            m_tuner->setPeriod(period/10);
            m_tuner->setEnabled(enable);
            }

    protected:
        //! Reallocate when maximum particle number changes
        virtual void reallocateMaxN();

        //! update GPU mapping
        virtual void updateGPUMapping();

        GlobalArray<unsigned int> m_cell_size_scratch;  //!< Number of members in each cell, one list per GPU
        GlobalArray<unsigned int> m_cell_adj_scratch;   //!< Cell adjacency list, one list per GPU
        GlobalArray<Scalar4> m_xyzf_scratch;            //!< Cell list with position and flags, one list per GPU
        GlobalArray<Scalar4> m_tdb_scratch;             //!< Cell list with type,diameter,body, one list per GPU
        GlobalArray<Scalar4> m_orientation_scratch;     //!< Cell list with orientation, one list per GPU
        GlobalArray<unsigned int> m_idx_scratch;        //!< Cell list with index, one list per GPU
        GlobalArray<unsigned int> m_particle_cli;       //!< Lookup of cell list position per particle

        //! Compute the cell list
        virtual void computeCellList();

        // Initialize GPU-specific data storage
        virtual void initializeMemory();

        std::unique_ptr<Autotuner> m_tuner;         //!< Autotuner for block size
        std::unique_ptr<Autotuner> m_tuner_combine; //!< Autotuner for block size of combine cell lists kernel
    };

//! Exports CellListGPU to python
void export_CellListGPU(pybind11::module& m);

#endif
