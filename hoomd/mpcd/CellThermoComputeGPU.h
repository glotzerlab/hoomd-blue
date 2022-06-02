// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/CellThermoComputeGPU.h
 * \brief Declaration of mpcd::CellThermoComputeGPU
 */

#ifndef MPCD_CELL_THERMO_COMPUTE_GPU_H_
#define MPCD_CELL_THERMO_COMPUTE_GPU_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "CellThermoCompute.h"
#include "CellThermoComputeGPU.cuh"

#include "hoomd/Autotuner.h"

// pybind11
#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace mpcd
    {
//! Computes the cell (thermodynamic) properties on the GPU
class PYBIND11_EXPORT CellThermoComputeGPU : public mpcd::CellThermoCompute
    {
    public:
    //! Constructor
    CellThermoComputeGPU(std::shared_ptr<mpcd::SystemData> sysdata);

    //! Destructor
    virtual ~CellThermoComputeGPU();

    protected:
#ifdef ENABLE_MPI
    //! Begin the calculation of outer cell properties on the GPU
    virtual void beginOuterCellProperties();

    //! Finish the calculation of outer cell properties on the GPU
    virtual void finishOuterCellProperties();
#endif // ENABLE_MPI

    //! Calculate the inner cell properties on the GPU
    virtual void calcInnerCellProperties();

    //! Compute the net properties from the cell properties
    virtual void computeNetProperties();

    private:
    std::unique_ptr<Autotuner> m_begin_tuner; //!< Tuner for cell begin kernel
    std::unique_ptr<Autotuner> m_end_tuner;   //!< Tuner for cell end kernel
    std::unique_ptr<Autotuner> m_inner_tuner; //!< Tuner for inner cell compute kernel
    std::unique_ptr<Autotuner> m_stage_tuner; //!< Tuner for staging net property compute

    GPUVector<mpcd::detail::cell_thermo_element>
        m_tmp_thermo; //!< Temporary array for holding cell data
    GPUFlags<mpcd::detail::cell_thermo_element> m_reduced; //!< Flags to hold reduced sum
    };

namespace detail
    {
//! Export the CellThermoComputeGPU class to python
void export_CellThermoComputeGPU(pybind11::module& m);
    } // end namespace detail
    } // end namespace mpcd
    } // end namespace hoomd

#endif // MPCD_CELL_THERMO_COMPUTE_GPU_H_
