// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __PARTICLE_FILTER_CUSTOM_H__
#define __PARTICLE_FILTER_CUSTOM_H__

#include "ParticleFilter.h"

namespace hoomd
    {
class PYBIND11_EXPORT ParticleFilterCustom : public ParticleFilter
    {
    public:
    /// constructs a custom ParticleFilter
    ParticleFilterCustom(pybind11::object py_filter, pybind11::object state)
        : m_py_filter(py_filter), m_state(state) {};

    virtual ~ParticleFilterCustom() { }

    /** Test if a particle meets the selection criteria.
     *  Uses the composed Python custom particle filter to get a list of
     *  tags to use.
     */
    virtual std::vector<unsigned int>
    getSelectedTags(std::shared_ptr<SystemDefinition> sysdef) const
        {
        pybind11::array_t<unsigned int, pybind11::array::c_style | pybind11::array::forcecast> tags(
            m_py_filter(m_state));
        unsigned int* tags_ptr = (unsigned int*)tags.data();
        return std::vector<unsigned int>(tags_ptr, tags_ptr + tags.size());
        }

    protected:
    pybind11::object m_py_filter; /// Python hoomd.filter.CustomFilter object
    pybind11::object m_state;     /// Python hoomd.State object
    };

    } // end namespace hoomd
#endif // __PARTICLE_FILTER_CUSTOM_H__
