// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/CellListGPU.h
 * \brief Declaration of mpcd::CellListGPU
 */

#ifndef MPCD_CELL_LIST_GPU_H_
#define MPCD_CELL_LIST_GPU_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "CellList.h"
#include "hoomd/Autotuner.h"

namespace mpcd
{

//! Computes the MPCD cell list on the GPU
class PYBIND11_EXPORT CellListGPU : public mpcd::CellList
    {
    public:
    //! Constructor
        CellListGPU(std::shared_ptr<SystemDefinition> sysdef,
                    std::shared_ptr<mpcd::ParticleData> mpcd_pdata);

        virtual ~CellListGPU();

        //! Set autotuner parameters
        /*!
         * \param enable Enable/disable autotuning
         * \param period period (approximate) in time steps when returning occurs
         */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            mpcd::CellList::setAutotunerParams(enable, period);

            m_tuner_cell->setPeriod(period); m_tuner_cell->setEnabled(enable);
            m_tuner_sort->setPeriod(period); m_tuner_sort->setEnabled(enable);
            #ifdef ENABLE_MPI
            m_tuner_embed_migrate->setPeriod(period); m_tuner_embed_migrate->setEnabled(enable);
            #endif // ENABLE_MPI
            }

    protected:
        //! Compute the cell list of particles on the GPU
        virtual void buildCellList();

        //! Callback to sort cell list on the GPU when particle data is sorted
        virtual void sort(unsigned int timestep,
                          const GPUArray<unsigned int>& order,
                          const GPUArray<unsigned int>& rorder);

        #ifdef ENABLE_MPI
        //! Determine if embedded particles require migration on the gpu
        virtual bool needsEmbedMigrate(unsigned int timestep);
        GPUFlags<unsigned int> m_migrate_flag;  //!< Flag to signal migration is needed
        #endif // ENABLE_MPI

    private:
        std::unique_ptr<Autotuner> m_tuner_cell;    //!< Autotuner for the cell list calculation
        std::unique_ptr<Autotuner> m_tuner_sort;    //!< Autotuner for sorting the cell list
        #ifdef ENABLE_MPI
        std::unique_ptr<Autotuner> m_tuner_embed_migrate;   //!< Autotuner for checking embedded migration
        #endif // ENABLE_MPI
    };

namespace detail
{
//! Export the CellListGPU class to python
void export_CellListGPU(pybind11::module& m);
} // end namespace detail

} // end namespace mpcd

#endif // MPCD_CELL_LIST_GPU_H_
