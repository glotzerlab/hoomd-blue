#pragma once

#include "ParticleFilter.h"

namespace hoomd
    {
//! Select all particles
class PYBIND11_EXPORT ParticleFilterNull : public ParticleFilter
    {
    public:
    /// Constructs the selector
    ParticleFilterNull() : ParticleFilter() {};
    virtual ~ParticleFilterNull() { }

    /** Args:
     *  sysdef: the System Definition
     *
     *  Returns:
     *  an empty list
     */
    virtual std::vector<unsigned int>
    getSelectedTags(std::shared_ptr<SystemDefinition> sysdef) const
        {
        std::vector<unsigned int> member_tags;
        return member_tags;
        }
    };

    } // end namespace hoomd
