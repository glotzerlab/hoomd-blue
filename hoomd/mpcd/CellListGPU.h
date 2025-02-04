// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/CellListGPU.h
 * \brief Declaration of mpcd::CellListGPU
 */

#ifndef MPCD_CELL_LIST_GPU_H_
#define MPCD_CELL_LIST_GPU_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "CellList.h"
#include "hoomd/Autotuner.h"

namespace hoomd
    {
namespace mpcd
    {
//! Computes the MPCD cell list on the GPU
class PYBIND11_EXPORT CellListGPU : public mpcd::CellList
    {
    public:
    //! Constructor
    CellListGPU(std::shared_ptr<SystemDefinition> sysdef, Scalar cell_size, bool shift);

    virtual ~CellListGPU();

    protected:
    //! Compute the cell list of particles on the GPU
    virtual void buildCellList();

    //! Callback to sort cell list on the GPU when particle data is sorted
    virtual void sort(uint64_t timestep,
                      const GPUArray<unsigned int>& order,
                      const GPUArray<unsigned int>& rorder);

#ifdef ENABLE_MPI
    //! Determine if embedded particles require migration on the gpu
    virtual bool needsEmbedMigrate(uint64_t timestep);
    GPUFlags<unsigned int> m_migrate_flag; //!< Flag to signal migration is needed
#endif                                     // ENABLE_MPI

    private:
    /// Autotuner for the cell list calculation.
    std::shared_ptr<Autotuner<1>> m_tuner_cell;

    /// Autotuner for sorting the cell list.
    std::shared_ptr<Autotuner<1>> m_tuner_sort;
#ifdef ENABLE_MPI
    /// Autotuner for checking embedded migration.
    std::shared_ptr<Autotuner<1>> m_tuner_embed_migrate;
#endif // ENABLE_MPI
    };
    } // end namespace mpcd
    } // end namespace hoomd
#endif // MPCD_CELL_LIST_GPU_H_
