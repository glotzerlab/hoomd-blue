// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "TwoStepRATTLELangevin.h"
#include "hoomd/Autotuner.h"
#include "EvaluatorConstraintManifold.h"

#ifndef __TWO_STEP_RATTLE_LANGEVIN_GPU_H__
#define __TWO_STEP_RATTLE_LANGEVIN_GPU_H__

/*! \file TwoStepRATTLELangevinGPU.h
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
class PYBIND11_EXPORT TwoStepRATTLELangevinGPU : public TwoStepRATTLELangevin
    {
    public:
        //! Constructs the integration method and associates it with the system
        TwoStepRATTLELangevinGPU(std::shared_ptr<SystemDefinition> sysdef,
                           std::shared_ptr<ParticleGroup> group,
                     	   std::shared_ptr<Manifold> manifold,
                           std::shared_ptr<Variant> T,
                           unsigned int seed,
                           Scalar eta = 0.000001,
                           const std::string& suffix = std::string(""));
        virtual ~TwoStepRATTLELangevinGPU() {};

        //! Performs the first step of the integration
        virtual void integrateStepOne(unsigned int timestep);

        //! Performs the second step of the integration
        virtual void integrateStepTwo(unsigned int timestep);

        //! Includes the RATTLE forces to the virial/net force
        virtual void IncludeRATTLEForce(unsigned int timestep);

        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs
        */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            TwoStepRATTLELangevin::setAutotunerParams(enable, period);
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

	EvaluatorConstraintManifold m_manifoldGPU;
    };

//! Exports the TwoStepLangevinGPU class to python
void export_TwoStepRATTLELangevinGPU(pybind11::module& m);

#endif // #ifndef __TWO_STEP_LANGEVIN_GPU_H__
