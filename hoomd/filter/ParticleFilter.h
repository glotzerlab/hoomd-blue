// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include "../SystemDefinition.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <vector>

namespace hoomd
    {
/// Utility class to select particles based on given conditions
/** \b Overview

    In order to flexibly specify the particles that belong to a given
    ParticleGroup, it will take a ParticleFilter as a parameter in its
    constructor. The selector will provide a true/false membership test that
    will be applied to each particle tag, selecting those that belong in the
    group. As it is specified via a virtual class, the group definition can be
    expanded to include any conceivable selection criteria.

    <b>Implementation details</b> So that any range of selection
    criteria can be applied (e.g. particles with mass > 2.0, or all particles
    bonded to particle j, ...) the selector will get a shared pointer to the
    SystemDefinition on construction, along with any parameters to specify the
    selection criteria. Then, calling getSelectedTags() will return a list
    of particle tags meeting the criteria.

    In MPI simulations, getSelectedTags() should return only tags on the local
    rank.

    The base class getSelectedTags() method returns an empty vector.
*/
class PYBIND11_EXPORT ParticleFilter
    {
    public:
    /// constructs a base ParticleFilter (does nothing)
    ParticleFilter() {};
    virtual ~ParticleFilter() { }

    /** Test if a particle meets the selection criteria.
     *  The base case returns an empty vector.
     */
    virtual std::vector<unsigned int>
    getSelectedTags(std::shared_ptr<SystemDefinition> sysdef) const
        {
        return std::vector<unsigned int>();
        }
    };

    } // end namespace hoomd
