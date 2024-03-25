// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ForceConstraint.h"

using namespace std;

/*! \file ForceConstraint.cc
    \brief Contains code for the ForceConstraint class
*/

namespace hoomd
    {
/*! \param sysdef SystemDefinition containing the ParticleData to compute forces on
 */
ForceConstraint::ForceConstraint(std::shared_ptr<SystemDefinition> sysdef)
    : ForceCompute(sysdef) { }

/*! Does nothing in the base class
    \param timestep Current timestep
*/
void ForceConstraint::computeForces(uint64_t timestep) { }

namespace detail
    {
void export_ForceConstraint(pybind11::module& m)
    {
    pybind11::class_<ForceConstraint, ForceCompute, std::shared_ptr<ForceConstraint>>(
        m,
        "ForceConstraint")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>());
    }
    } // end namespace detail

    } // end namespace hoomd
