// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "CollectiveActiveForceCompute.h"
#include "hoomd/Autotuner.h"

/*! \file CollectiveActiveForceComputeGPU.h
    \brief Declares a class for computing collective active forces on the GPU
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __COLLECTIVEACTIVEFORCECOMPUTE_GPU_H__
#define __COLLECTIVEACTIVEFORCECOMPUTE_GPU_H__

namespace hoomd
    {
namespace md
    {
//! Adds an active force whose vector is shared by all particles of the same type on the GPU
/*! Only setForces runs on the GPU, as a kernel over the group members. The rotational diffusion
    inherited from CollectiveActiveForceCompute stays on the host: it draws one random number
    per particle type, which is far too little work for a kernel, and the per-type vectors are
    copied to the device automatically when the next set of forces is computed.

    \ingroup computes
 */
class PYBIND11_EXPORT CollectiveActiveForceComputeGPU : public CollectiveActiveForceCompute
    {
    public:
    //! Constructs the compute
    CollectiveActiveForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                                    std::shared_ptr<ParticleGroup> group);

    protected:
    //! Set the forces from the per-type active force vectors
    virtual void setForces();

    std::shared_ptr<Autotuner<1>> m_tuner_force; //!< Autotuner for block size (force kernel)
    };

namespace detail
    {
//! Exports CollectiveActiveForceComputeGPU to python
void export_CollectiveActiveForceComputeGPU(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
#endif
