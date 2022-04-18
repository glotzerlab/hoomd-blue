// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ActiveForceCompute.h"
#include "hoomd/Autotuner.h"

/*! \file ActiveForceComputeGPU.h
    \brief Declares a class for computing active forces on the GPU
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __ACTIVEFORCECOMPUTE_GPU_H__
#define __ACTIVEFORCECOMPUTE_GPU_H__

namespace hoomd
    {
namespace md
    {
//! Adds an active force to a number of particles on the GPU
/*! \ingroup computes
 */
class PYBIND11_EXPORT ActiveForceComputeGPU : public ActiveForceCompute
    {
    public:
    //! Constructs the compute
    ActiveForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                          std::shared_ptr<ParticleGroup> group);

    //! Set autotuner parameters
    /*! \param enable Enable/disable autotuning
        \param period period (approximate) in time steps when returning occurs
    */
    virtual void setAutotunerParams(bool enable, unsigned int period)
        {
        ActiveForceCompute::setAutotunerParams(enable, period);
        m_tuner_force->setPeriod(period);
        m_tuner_force->setEnabled(enable);
        m_tuner_diffusion->setPeriod(period);
        m_tuner_diffusion->setEnabled(enable);
        }

    protected:
    std::unique_ptr<Autotuner> m_tuner_force;     //!< Autotuner for block size (force kernel)
    std::unique_ptr<Autotuner> m_tuner_diffusion; //!< Autotuner for block size (diff kernel)

    //! Set forces for particles
    virtual void setForces();

    //! Orientational diffusion for spherical particles
    virtual void rotationalDiffusion(Scalar rotational_diffusion, uint64_t timestep);
    };

    } // end namespace md
    } // end namespace hoomd
#endif
