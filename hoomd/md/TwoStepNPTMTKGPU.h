// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

#include "TwoStepNPTMTK.h"
#include "hoomd/Variant.h"
#include "hoomd/ComputeThermo.h"
#include "hoomd/Autotuner.h"

#include <memory>


#ifndef __TWO_STEP_NPT_MTK_GPU_H__
#define __TWO_STEP_NPT_MTK_GPU_H__

/*! \file TwoStepNPTMTKGPU.h
    \brief Declares the TwoStepNPTMTKGPU class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

//! Integrates part of the system forward in two steps in the NPT ensemble
/*! This is a version of TwoStepNPTMTK that runs on the GPU.
 *
    \ingroup updaters
*/
class PYBIND11_EXPORT TwoStepNPTMTKGPU : public TwoStepNPTMTK
    {
    public:
        //! Constructs the integration method and associates it with the system
        TwoStepNPTMTKGPU(std::shared_ptr<SystemDefinition> sysdef,
                   std::shared_ptr<ParticleGroup> group,
                   std::shared_ptr<ComputeThermo> thermo_group,
                   std::shared_ptr<ComputeThermo> thermo_group_t,
                   Scalar tau,
                   Scalar tauP,
                   std::shared_ptr<Variant> T,
                   pybind11::list S,
                   couplingMode couple,
                   unsigned int flags,
                   const bool nph=false);

        virtual ~TwoStepNPTMTKGPU();

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
            TwoStepNPTMTK::setAutotunerParams(enable, period);
            m_tuner_one->setPeriod(period);
            m_tuner_one->setEnabled(enable);
            m_tuner_two->setPeriod(period);
            m_tuner_two->setEnabled(enable);
            m_tuner_wrap->setPeriod(period);
            m_tuner_wrap->setEnabled(enable);
            m_tuner_rescale->setPeriod(period);
            m_tuner_rescale->setEnabled(enable);
            m_tuner_angular_one->setPeriod(period);
            m_tuner_angular_one->setEnabled(enable);
            m_tuner_angular_two->setPeriod(period);
            m_tuner_angular_two->setEnabled(enable);
            }

    protected:
        std::unique_ptr<Autotuner> m_tuner_one; //!< Autotuner for block size (step one kernel)
        std::unique_ptr<Autotuner> m_tuner_two; //!< Autotuner for block size (step two kernel)
        std::unique_ptr<Autotuner> m_tuner_wrap; //!< Autotuner for wrapping particle positions
        std::unique_ptr<Autotuner> m_tuner_rescale; //!< Autotuner for thermostat rescaling
        std::unique_ptr<Autotuner> m_tuner_angular_one; //!< Autotuner for angular step one
        std::unique_ptr<Autotuner> m_tuner_angular_two; //!< Autotuner for angular step two
    };

//! Exports the TwoStepNPTMTKGPU class to python
void export_TwoStepNPTMTKGPU(pybind11::module& m);

#endif // #ifndef __TWO_STEP_NPT_MTK_GPU_H__
