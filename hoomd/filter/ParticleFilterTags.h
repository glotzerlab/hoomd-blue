// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __PARTICLE_FILTER_TAGS_H__
#define __PARTICLE_FILTER_TAGS_H__

#include "ParticleFilter.h"
#include <pybind11/numpy.h>

namespace hoomd
    {
/// Select particles based on their tag
class PYBIND11_EXPORT ParticleFilterTags : public ParticleFilter
    {
    public:
    /** Args:
     *  tags: std::vector of tags to select
     */
    ParticleFilterTags(std::vector<unsigned int> tags) : ParticleFilter(), m_tags(tags) { }

    /** Args:
     *  tags: pybind11::array of tags to select
     */
    ParticleFilterTags(
        pybind11::array_t<unsigned int, pybind11::array::c_style | pybind11::array::forcecast> tags)
        : ParticleFilter()
        {
        unsigned int* tags_ptr = (unsigned int*)tags.data();
        m_tags.assign(tags_ptr, tags_ptr + tags.size());
        }

    virtual ~ParticleFilterTags() { }

    /** Args:
     *  sysdef System Definition
     *
     *  Returns:
     *  m_tags
     */
    virtual std::vector<unsigned int>
    getSelectedTags(std::shared_ptr<SystemDefinition> sysdef) const
        {
        return m_tags;
        }

    protected:
    std::vector<unsigned int> m_tags; //< Tags to use for filter
    };

    } // end namespace hoomd
#endif
