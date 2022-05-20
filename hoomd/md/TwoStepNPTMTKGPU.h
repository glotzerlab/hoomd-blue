// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ComputeThermo.h"
#include "TwoStepNPTMTK.h"
#include "hoomd/Autotuner.h"
#include "hoomd/Variant.h"

#include <memory>

#ifndef __TWO_STEP_NPT_MTK_GPU_H__
#define __TWO_STEP_NPT_MTK_GPU_H__

/*! \file TwoStepNPTMTKGPU.h
    \brief Declares the TwoStepNPTMTKGPU class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {
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
                     Scalar tauS,
                     std::shared_ptr<Variant> T,
                     const std::vector<std::shared_ptr<Variant>>& S,
                     const std::string& couple,
                     const std::vector<bool>& flags,
                     const bool nph);

    virtual ~TwoStepNPTMTKGPU();

    //! Performs the first step of the integration
    virtual void integrateStepOne(uint64_t timestep);

    //! Performs the second step of the integration
    virtual void integrateStepTwo(uint64_t timestep);

    /// Start autotuning kernel launch parameters
    virtual void startAutotuning()
        {
        TwoStepNPTMTK::startAutotuning();
        m_tuner_one->startScan();
        m_tuner_two->startScan();
        m_tuner_wrap->startScan();
        m_tuner_rescale->startScan();
        m_tuner_angular_one->startScan();
        m_tuner_angular_two->startScan();
        }

    protected:
    std::unique_ptr<Autotuner> m_tuner_one;         //!< Autotuner for block size (step one kernel)
    std::unique_ptr<Autotuner> m_tuner_two;         //!< Autotuner for block size (step two kernel)
    std::unique_ptr<Autotuner> m_tuner_wrap;        //!< Autotuner for wrapping particle positions
    std::unique_ptr<Autotuner> m_tuner_rescale;     //!< Autotuner for thermostat rescaling
    std::unique_ptr<Autotuner> m_tuner_angular_one; //!< Autotuner for angular step one
    std::unique_ptr<Autotuner> m_tuner_angular_two; //!< Autotuner for angular step two
    };

    } // end namespace md
    } // end namespace hoomd

#endif // #ifndef __TWO_STEP_NPT_MTK_GPU_H__
