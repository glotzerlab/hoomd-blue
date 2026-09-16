// Copyright (c) 2009-2026 The Regents of the University of Michigan.
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
#include "CellListGPU.cuh"
#include "hoomd/Autotuner.h"

namespace hoomd
    {
namespace mpcd
    {
//! Computes the MPCD cell list on the GPU
class PYBIND11_EXPORT CellListGPU : public mpcd::CellList
    {
    public:
    //! Constructor by size (deprecated)
    CellListGPU(std::shared_ptr<SystemDefinition> sysdef, Scalar cell_size, bool shift);

    //! Constructor by dimension
    CellListGPU(std::shared_ptr<SystemDefinition> sysdef, const uint3& global_cell_dim, bool shift);

    virtual ~CellListGPU();

    protected:
    //! Compute the cell list of particles on the GPU
    void buildCellList() override;

    //! Do final cell property calculation
    void finishComputeProperties() override;

    //! Compute the net properties of all the cells
    void computeNetProperties() override;

#ifdef ENABLE_MPI
    //! Determine if embedded particles require migration on the gpu
    virtual bool needsEmbedMigrate(uint64_t timestep);
    GPUFlags<unsigned int> m_migrate_flag; //!< Flag to signal migration is needed
#endif                                     // ENABLE_MPI

    private:
    GPUVector<mpcd::detail::cell_thermo_element>
        m_tmp_thermo; //!< Temporary array for holding cell data
    GPUFlags<mpcd::detail::cell_thermo_element> m_reduced; //!< Flags to hold reduced sum

    /// Autotuner for the cell list calculation.
    std::shared_ptr<Autotuner<1>> m_tuner_cell;

    /// Autotuner for finishing cell property calculation.
    std::shared_ptr<Autotuner<1>> m_tuner_property;

    /// Autotuner for the net property calculation.
    std::shared_ptr<Autotuner<1>> m_tuner_net;

#ifdef ENABLE_MPI
    /// Autotuner for checking embedded migration.
    std::shared_ptr<Autotuner<1>> m_tuner_embed_migrate;
#endif // ENABLE_MPI
    };
    } // end namespace mpcd
    } // end namespace hoomd
#endif // MPCD_CELL_LIST_GPU_H_
