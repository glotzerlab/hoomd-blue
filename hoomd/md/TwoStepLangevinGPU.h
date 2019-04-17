// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "TwoStepLangevin.h"
#include "hoomd/Autotuner.h"

#ifndef __TWO_STEP_LANGEVIN_GPU_H__
#define __TWO_STEP_LANGEVIN_GPU_H__

/*! \file TwoStepLangevinGPU.h
    \brief Declares the TwoStepLangevinGPU class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

//! Implements Langevin dynamics on the GPU
/*! GPU accelerated version of TwoStepLangevin

    \ingroup updaters
*/
class PYBIND11_EXPORT TwoStepLangevinGPU : public TwoStepLangevin
    {
    public:
        //! Constructs the integration method and associates it with the system
        TwoStepLangevinGPU(std::shared_ptr<SystemDefinition> sysdef,
                           std::shared_ptr<ParticleGroup> group,
                           std::shared_ptr<Variant> T,
                           unsigned int seed,
                           bool use_lambda,
                           Scalar lambda,
                           bool noiseless_t,
                           bool noiseless_r,
                           const std::string& suffix = std::string(""));
        virtual ~TwoStepLangevinGPU() {};

        //! Performs the first step of the integration
        virtual void integrateStepOne(unsigned int timestep);

        //! Performs the second step of the integration
        virtual void integrateStepTwo(unsigned int timestep);

        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs
        */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            TwoStepLangevin::setAutotunerParams(enable, period);
            m_tuner_one->setPeriod(period);
            m_tuner_one->setEnabled(enable);
            m_tuner_angular_one->setPeriod(period);
            m_tuner_angular_one->setEnabled(enable);
            }

    protected:
        unsigned int m_block_size;               //!< block size for partial sum memory
        unsigned int m_num_blocks;               //!< number of memory blocks reserved for partial sum memory
        GPUArray<Scalar> m_partial_sum1;         //!< memory space for partial sum over bd energy transfers
        GPUArray<Scalar> m_sum;                  //!< memory space for sum over bd energy transfers

        std::unique_ptr<Autotuner> m_tuner_one; //!< Autotuner for block size (step one kernel)
        std::unique_ptr<Autotuner> m_tuner_angular_one; //!< Autotuner for block size (angular step one kernel)
    };

//! Exports the TwoStepLangevinGPU class to python
void export_TwoStepLangevinGPU(pybind11::module& m);

#endif // #ifndef __TWO_STEP_LANGEVIN_GPU_H__
