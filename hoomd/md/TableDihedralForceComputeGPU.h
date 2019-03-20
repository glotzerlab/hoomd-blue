// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: phillicl

#include "TableDihedralForceCompute.h"
#include "TableDihedralForceGPU.cuh"
#include "hoomd/Autotuner.h"

/*! \file TableDihedralForceComputeGPU.h
    \brief Declares the TableDihedralForceComputeGPU class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __TABLEDIHEDRALFORCECOMPUTEGPU_H__
#define __TABLEDIHEDRALFORCECOMPUTEGPU_H__

//! Compute table based bond potentials on the GPU
/*! Calculates exactly the same thing as TableDihedralForceCompute, but on the GPU

    The GPU kernel for calculating this can be found in TableDihedralForceComputeGPU.cu/
    \ingroup computes
*/
class PYBIND11_EXPORT TableDihedralForceComputeGPU : public TableDihedralForceCompute
    {
    public:
        //! Constructs the compute
        TableDihedralForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                          unsigned int table_width,
                          const std::string& log_suffix="");

        //! Destructor
        virtual ~TableDihedralForceComputeGPU() { }

        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs
        */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            TableDihedralForceCompute::setAutotunerParams(enable, period);
            m_tuner->setPeriod(period);
            m_tuner->setEnabled(enable);
            }

    private:
        std::unique_ptr<Autotuner> m_tuner; //!< Autotuner for block size
        GPUArray<unsigned int> m_flags;       //!< Flags set during the kernel execution
        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

//! Exports the TableDihedralForceComputeGPU class to python
void export_TableDihedralForceComputeGPU(pybind11::module& m);

#endif
