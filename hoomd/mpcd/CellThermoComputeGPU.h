// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/CellThermoComputeGPU.h
 * \brief Declaration of mpcd::CellThermoComputeGPU
 */

#ifndef MPCD_CELL_THERMO_COMPUTE_GPU_H_
#define MPCD_CELL_THERMO_COMPUTE_GPU_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "CellThermoComputeGPU.cuh"
#include "CellThermoCompute.h"

#include "hoomd/Autotuner.h"

// pybind11
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

namespace mpcd
{
//! Computes the cell (thermodynamic) properties on the GPU
class CellThermoComputeGPU : public mpcd::CellThermoCompute
    {
    public:
        //! Constructor
        CellThermoComputeGPU(std::shared_ptr<mpcd::SystemData> sysdata,
                             const std::string& suffix = std::string(""));

        //! Destructor
        virtual ~CellThermoComputeGPU();

        //! Set autotuner parameters
        /*!
         * \param enable Enable/disable autotuning
         * \param period period (approximate) in time steps when returning occurs
         */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            mpcd::CellThermoCompute::setAutotunerParams(enable, period);

            m_begin_tuner->setEnabled(enable);
            m_begin_tuner->setPeriod(period);

            m_end_tuner->setEnabled(enable);
            m_end_tuner->setPeriod(period);

            m_stage_tuner->setEnabled(enable);
            m_stage_tuner->setPeriod(period);
            }

    protected:
        //! Compute the cell properties
        virtual void computeCellProperties();

        //! Compute the net properties from the cell properties
        virtual void computeNetProperties();

    private:
        std::unique_ptr<Autotuner> m_begin_tuner;   //!< Tuner for cell begin kernel
        std::unique_ptr<Autotuner> m_end_tuner;     //!< Tuner for cell end kernel
        std::unique_ptr<Autotuner> m_stage_tuner;   //!< Tuner for staging net property compute

        GPUVector<mpcd::detail::cell_thermo_element> m_tmp_thermo;  //!< Temporary array for holding cell data
        GPUFlags<mpcd::detail::cell_thermo_element> m_reduced;      //!< Flags to hold reduced sum
    };

namespace detail
{
//! Export the CellThermoComputeGPU class to python
void export_CellThermoComputeGPU(pybind11::module& m);
} // end namespace detail
} // end namespace mpcd

#endif // MPCD_CELL_THERMO_COMPUTE_GPU_H_
