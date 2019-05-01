// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "OneDConstraint.h"
#include "hoomd/Autotuner.h"

/*! \file OneDConstraintGPU.h
    \brief Declares a class for computing sphere constraint forces on the GPU
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __ONE_D_CONSTRAINT_GPU_H__
#define __ONE_D_CONSTRAINT_GPU_H__

//! Applys a constraint force to keep a group of particles on a sphere on the GPU
/*! \ingroup computes
*/
class PYBIND11_EXPORT OneDConstraintGPU : public OneDConstraint
    {
    public:
        //! Constructs the compute
        OneDConstraintGPU(std::shared_ptr<SystemDefinition> sysdef,
                          std::shared_ptr<ParticleGroup> group,
                          Scalar3 constraint_vec);

        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs
        */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            OneDConstraint::setAutotunerParams(enable, period);
            m_tuner->setPeriod(period);
            m_tuner->setEnabled(enable);
            }

    protected:
        unsigned int m_block_size;  //!< block size to execute on the GPU

        std::unique_ptr<Autotuner> m_tuner; //!< Autotuner for block size

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

//! Exports the OneDConstraintGPU class to python
void export_OneDConstraintGPU(pybind11::module& m);

#endif
