// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "TablePotential.h"
#include "TablePotentialGPU.cuh"
#include "hoomd/Autotuner.h"

/*! \file TablePotentialGPU.h
    \brief Declares the TablePotentialGPU class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __TABLEPOTENTIALGPU_H__
#define __TABLEPOTENTIALGPU_H__

//! Compute table based potentials on the GPU
/*! Calculates exactly the same thing as TablePotential, but on the GPU

    The GPU kernel for calculating this can be found in TablePotentialGPU.cu/
    \ingroup computes
*/
class PYBIND11_EXPORT TablePotentialGPU : public TablePotential
    {
    public:
        //! Constructs the compute
        TablePotentialGPU(std::shared_ptr<SystemDefinition> sysdef,
                          std::shared_ptr<NeighborList> nlist,
                          unsigned int table_width,
                          const std::string& log_suffix="");

        //! Destructor
        virtual ~TablePotentialGPU() { }

        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs
        */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            TablePotential::setAutotunerParams(enable, period);
            m_tuner->setPeriod(period);
            m_tuner->setEnabled(enable);
            }

    private:
        std::unique_ptr<Autotuner> m_tuner; //!< Autotuner for block size

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

//! Exports the TablePotentialGPU class to python
void export_TablePotentialGPU(pybind11::module& m);

#endif
