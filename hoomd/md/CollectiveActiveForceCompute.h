// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ActiveForceCompute.h"

/*! \file CollectiveActiveForceCompute.h
    \brief Declares a class for computing active forces that share a common direction per type
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __COLLECTIVEACTIVEFORCECOMPUTE_H__
#define __COLLECTIVEACTIVEFORCECOMPUTE_H__

namespace hoomd
    {
namespace md
    {
//! Adds an active force whose vector is shared by all particles of the same type
/*! \ingroup computes
 */
class PYBIND11_EXPORT CollectiveActiveForceCompute : public ActiveForceCompute
    {
    public:
    //! Constructs the compute
    CollectiveActiveForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                                 std::shared_ptr<ParticleGroup> group);

    //! Destructor
    virtual ~CollectiveActiveForceCompute();

    protected:
    //! Set the forces from the per-type active force vectors
    virtual void setForces();

    //! Diffuse the active force direction of each particle type with a nonzero active force
    virtual void rotationalDiffusion(Scalar rotational_diffusion, uint64_t timestep);
    };

namespace detail
    {
//! Exports CollectiveActiveForceCompute to python
void export_CollectiveActiveForceCompute(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
#endif
