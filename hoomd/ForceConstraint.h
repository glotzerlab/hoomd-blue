// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ForceCompute.h"

#include <memory>

/*! \file ForceConstraint.h
    \brief Declares a base class for computing constraint
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __ForceConstraint_H__
#define __ForceConstraint_H__

namespace hoomd
    {
//! Base class for all constraint forces
/*! See Integrator for detailed documentation on constraint force implementation.
    \ingroup computes
*/
class PYBIND11_EXPORT ForceConstraint : public ForceCompute
    {
    public:
    //! Constructs the compute
    ForceConstraint(std::shared_ptr<SystemDefinition> sysdef);

    //! Return the number of DOF removed from a group by this constraint
    /*! The base class ForceConstraint returns 0, derived classes should override
        @param query The group over which to compute the removed degrees of freedom
    */
    virtual Scalar getNDOFRemoved(std::shared_ptr<ParticleGroup> query)
        {
        return 0;
        }

    protected:
    //! Compute the forces
    virtual void computeForces(uint64_t timestep);
    };

namespace detail
    {
//! Exports the ForceConstraint to python
void export_ForceConstraint(pybind11::module& m);
    } // end namespace detail

    } // end namespace hoomd
#endif
