// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

#include "TwoStepNVTMTK.h"

#ifndef __TWO_STEP_NVT_MTK_GPU_H__
#define __TWO_STEP_NVT_MTK_GPU_H__

/*! \file TwoStepNVTMTKGPU.h
    \brief Declares the TwoStepNVTMTKGPU class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#include "hoomd/Autotuner.h"

//! Integrates part of the system forward in two steps in the NVT ensemble on the GPU
/*! Implements Nose-Hoover NVT integration through the IntegrationMethodTwoStep interface, runs on the GPU

    In order to compute efficiently and limit the number of kernel launches integrateStepOne() performs a first
    pass reduction on the sum of m*v^2 and stores the partial reductions. A second kernel is then launched to reduce
    those to a final \a sum2K, which is a scalar but stored in a GPUArray for convenience.

    \ingroup updaters
*/
class PYBIND11_EXPORT TwoStepNVTMTKGPU : public TwoStepNVTMTK
    {
    public:
        //! Constructs the integration method and associates it with the system
        TwoStepNVTMTKGPU(std::shared_ptr<SystemDefinition> sysdef,
                      std::shared_ptr<ParticleGroup> group,
                      std::shared_ptr<ComputeThermo> thermo,
                      Scalar tau,
                      std::shared_ptr<Variant> T,
                      const std::string& suffix = std::string(""));
        virtual ~TwoStepNVTMTKGPU() {};

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
            TwoStepNVTMTK::setAutotunerParams(enable, period);
            m_tuner_one->setPeriod(period);
            m_tuner_one->setEnabled(enable);
            m_tuner_two->setPeriod(period);
            m_tuner_two->setEnabled(enable);
            m_tuner_angular_one->setPeriod(period);
            m_tuner_angular_one->setEnabled(enable);
            m_tuner_angular_two->setPeriod(period);
            m_tuner_angular_two->setEnabled(enable);
            }

    protected:
        std::unique_ptr<Autotuner> m_tuner_one; //!< Autotuner for block size (step one kernel)
        std::unique_ptr<Autotuner> m_tuner_two; //!< Autotuner for block size (step two kernel)
        std::unique_ptr<Autotuner> m_tuner_angular_one; //!< Autotuner_angular for block size (angular step one kernel)
        std::unique_ptr<Autotuner> m_tuner_angular_two; //!< Autotuner_angular for block size (angular step two kernel)
    };

//! Exports the TwoStepNVTMTKGPU class to python
void export_TwoStepNVTMTKGPU(pybind11::module& m);

#endif // #ifndef __TWO_STEP_NVT_MTK_GPU_H__
