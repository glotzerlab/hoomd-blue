// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __PARTICLE_FILTER_ALL_H__
#define __PARTICLE_FILTER_ALL_H__

#include "ParticleFilter.h"

namespace hoomd
    {
//! Select all particles
class PYBIND11_EXPORT ParticleFilterAll : public ParticleFilter
    {
    public:
    /// Constructs the selector
    ParticleFilterAll() : ParticleFilter() { };
    virtual ~ParticleFilterAll() { }

    /** Args:
     *  sysdef: the System Definition
     *
     *  Returns:
     *  all particles in the local rank
     */
    virtual std::vector<unsigned int>
    getSelectedTags(std::shared_ptr<SystemDefinition> sysdef) const
        {
        const auto pdata = sysdef->getParticleData();

        // loop through local particles and select those that match selection
        // criterion
        const ArrayHandle<unsigned int> h_tag(pdata->getTags(),
                                              access_location::host,
                                              access_mode::read);

        const auto N = pdata->getN();
        std::vector<unsigned int> member_tags(N);
        std::copy_n(h_tag.data, N, member_tags.begin());
        return member_tags;
        }
    };

    } // end namespace hoomd
#endif
