// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "WallDistanceForceCompute.h"
#include "hoomd/Autotuner.h"

/*! \file WallDistanceForceComputeGPU.h
    \brief Declares a class for computing constant forces on the GPU
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __WALLDISTANCEFORCECOMPUTE_GPU_H__
#define __WALLDISTANCEFORCECOMPUTE_GPU_H__

namespace hoomd
    {
namespace md
    {
//! Adds a constant force to a number of particles on the GPU
/*! \ingroup computes
 */
class PYBIND11_EXPORT WallDistanceForceComputeGPU : public WallDistanceForceCompute
    {
    public:
    //! Constructs the compute
    WallDistanceForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                            std::shared_ptr<ParticleGroup> group,
			    Scalar k,
			    Scalar R);

    protected:
    std::shared_ptr<Autotuner<1>> m_tuner; //!< Autotuner for block size (force kernel)

    //! Set forces for particles
    virtual void setForces();
    };

    } // end namespace md
    } // end namespace hoomd
#endif
