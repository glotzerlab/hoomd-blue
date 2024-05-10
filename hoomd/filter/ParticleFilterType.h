// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __PARTICLE_FILTER_TYPE_H__
#define __PARTICLE_FILTER_TYPE_H__

#include "ParticleFilter.h"
#include <pybind11/stl.h>
#include <string>
#include <unordered_set>

namespace hoomd
    {
//! Select particles based on their type
class PYBIND11_EXPORT ParticleFilterType : public ParticleFilter
    {
    public:
    /** Constructs the selector
     *  Args:
     *  included_types: set of type string values to include
     */
    ParticleFilterType(std::unordered_set<std::string> included_types)
        : ParticleFilter(), m_types(included_types)
        {
        }

    virtual ~ParticleFilterType() { }

    /** Test if a particle meets the selection criteria
     *  sysdef: system definition to find tags for
     *
     *  Returns:
     *  tags of all rank local particles of types in m_types
     */
    virtual std::vector<unsigned int>
    getSelectedTags(std::shared_ptr<SystemDefinition> sysdef) const
        {
        const auto pdata = sysdef->getParticleData();
        // loop through local particles and select those that match
        // selection criterion
        const ArrayHandle<unsigned int> h_tag(pdata->getTags(),
                                              access_location::host,
                                              access_mode::read);
        const ArrayHandle<Scalar4> h_postype(pdata->getPositions(),
                                             access_location::host,
                                             access_mode::read);

        // Get types as unsigned ints
        std::unordered_set<unsigned int> types(m_types.size());
        for (auto type_str : m_types)
            {
            types.insert(pdata->getTypeByName(type_str));
            }

        // Add correctly typed particles to vector
        const auto N = pdata->getN();
        std::vector<unsigned int> member_tags(N);
        auto tag_it = member_tags.begin();
        for (unsigned int idx = 0; idx < N; ++idx)
            {
            unsigned int typ = __scalar_as_int(h_postype.data[idx].w);
            if (types.count(typ))
                {
                *tag_it = h_tag.data[idx];
                tag_it++;
                }
            }
        member_tags.resize(tag_it - member_tags.begin());
        return member_tags;
        }

    protected:
    std::unordered_set<std::string> m_types; ///< Set of types to select
    };

    } // end namespace hoomd
#endif
