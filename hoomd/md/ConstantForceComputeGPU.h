// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ConstantForceCompute.h"
#include "hoomd/Autotuner.h"

/*! \file ConstantForceComputeGPU.h
    \brief Declares a class for computing constant forces on the GPU
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __CONSTANTFORCECOMPUTE_GPU_H__
#define __CONSTANTFORCECOMPUTE_GPU_H__

namespace hoomd
    {
namespace md
    {
//! Adds a constant force to a number of particles on the GPU
/*! \ingroup computes
 */
class PYBIND11_EXPORT ConstantForceComputeGPU : public ConstantForceCompute
    {
    public:
    //! Constructs the compute
    ConstantForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                            std::shared_ptr<ParticleGroup> group);

    protected:
    std::shared_ptr<Autotuner<1>> m_tuner; //!< Autotuner for block size (force kernel)

    //! Set forces for particles
    virtual void setForces();
    };

    } // end namespace md
    } // end namespace hoomd
#endif
