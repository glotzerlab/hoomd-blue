// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "TwoStepNVE.h"

#ifndef __TWO_STEP_NVE_GPU_H__
#define __TWO_STEP_NVE_GPU_H__

/*! \file TwoStepNVEGPU.h
    \brief Declares the TwoStepNVEGPU class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#include "hoomd/Autotuner.h"

//! Integrates part of the system forward in two steps in the NVE ensemble on the GPU
/*! Implements velocity-verlet NVE integration through the IntegrationMethodTwoStep interface, runs on the GPU

    \ingroup updaters
*/
class PYBIND11_EXPORT TwoStepNVEGPU : public TwoStepNVE
    {
    public:
        //! Constructs the integration method and associates it with the system
        TwoStepNVEGPU(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<ParticleGroup> group);
        virtual ~TwoStepNVEGPU() {};

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
            TwoStepNVE::setAutotunerParams(enable, period);
            m_tuner_one->setPeriod(period);
            m_tuner_one->setEnabled(enable);
            m_tuner_two->setPeriod(period);
            m_tuner_two->setEnabled(enable);
            m_tuner_angular_one->setPeriod(period);
            m_tuner_angular_one->setEnabled(enable);
            m_tuner_angular_two->setPeriod(period);
            m_tuner_angular_two->setEnabled(enable);
            }

    private:
        std::unique_ptr<Autotuner> m_tuner_one; //!< Autotuner for block size (step one kernel)
        std::unique_ptr<Autotuner> m_tuner_two; //!< Autotuner for block size (step two kernel)
        std::unique_ptr<Autotuner> m_tuner_angular_one; //!< Autotuner for block size (angular step one kernel)
        std::unique_ptr<Autotuner> m_tuner_angular_two; //!< Autotuner for block size (angular step two kernel)
    };

//! Exports the TwoStepNVEGPU class to python
void export_TwoStepNVEGPU(pybind11::module& m);

#endif // #ifndef __TWO_STEP_NVE_GPU_H__
