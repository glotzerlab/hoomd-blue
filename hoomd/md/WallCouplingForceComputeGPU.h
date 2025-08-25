// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "WallCouplingForceCompute.h"
#include "hoomd/Autotuner.h"

/*! \file WallCouplingForceComputeGPU.h
    \brief Declares a class for computing constant forces on the GPU
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __WALLCOUPLINGFORCECOMPUTE_GPU_H__
#define __WALLCOUPLINGFORCECOMPUTE_GPU_H__

namespace hoomd
    {
namespace md
    {
//! Adds a constant force to a number of particles on the GPU
/*! \ingroup computes
 */
class PYBIND11_EXPORT WallCouplingForceComputeGPU : public WallCouplingForceCompute
    {
    public:
    //! Constructs the compute
    WallCouplingForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                            std::shared_ptr<ParticleGroup> group,
			    Scalar radial_force,
			    Scalar tagential_force,
			    Scalar lift,
			    Scalar R);

    protected:
    std::shared_ptr<Autotuner<1>> m_tuner; //!< Autotuner for block size (force kernel)

    //! Set forces for particles
    virtual void setForces();
    };

    } // end namespace md
    } // end namespace hoomd
#endif
