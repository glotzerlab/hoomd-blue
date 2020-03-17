#ifndef __PARTICLE_FILTER_H__
#define __PARTICLE_FILTER_H__

#include "../SystemDefinition.h"
#include <vector>
#include <pybind11/pybind11.h>
#include <memory>

//! Utility class to select particles based on given conditions
/*! \b Overview

    In order to flexibly specify the particles that belong to a given
    ParticleGroup, it will simple take a ParticleFilter as a parameter in its
    constructor. The selector will provide a true/false membership test that
    will be applied to each particle tag, selecting those that belong in the
    group. As it is specified via a virtual class, the group definition can be
    expanded to include any conceivable selection criteria.

    <b>Implementation details</b> So that an infinite range of selection
    criteria can be applied (i.e. particles with mass > 2.0, or all particles
    bonded to particle j, ...) the selector will get a reference to the
    SystemDefinition on construction, along with any parameters to specify the
    selection criteria. Then, a simple getSelectedTags() call will return a list
    of particle tags meeting the criteria.

    In parallel simulations, getSelectedTags() should return only local tags.

    The base class getSelectedTags() method will simply return an empty list.
    selection semantics.
*/
class PYBIND11_EXPORT ParticleFilter
    {
    public:
        /// constructs a base ParticleFilter (does nothing)
        ParticleFilter() {};
        virtual ~ParticleFilter() {}

        /// Test if a particle meets the selection criteria
        /// base case does nothing
        virtual std::vector<unsigned int> getSelectedTags(
                std::shared_ptr<SystemDefinition> sysdef) const
            {
            return std::vector<unsigned int>();
            }
    };
#endif
