// Copyright (c) 2009-2024 The Regents of the University of Michigan.
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

    protected:
    std::shared_ptr<Autotuner<1>> m_tuner_force;     //!< Autotuner for block size (force kernel)
    std::shared_ptr<Autotuner<1>> m_tuner_diffusion; //!< Autotuner for block size (diff kernel)

    //! Set forces for particles
    virtual void setForces();

    //! Orientational diffusion for spherical particles
    virtual void rotationalDiffusion(Scalar rotational_diffusion, uint64_t timestep);
    };

    } // end namespace md
    } // end namespace hoomd
#endif
