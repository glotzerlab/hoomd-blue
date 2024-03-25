// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __PARTICLE_FILTER_UNION_H__
#define __PARTICLE_FILTER_UNION_H__

#include "ParticleFilter.h"
#include <algorithm>

namespace hoomd
    {
class PYBIND11_EXPORT ParticleFilterUnion : public ParticleFilter
    {
    public:
    /** Constructs the selector
     *  Args:
     *  f: first filter
     *  g: second filter
     */
    ParticleFilterUnion(std::shared_ptr<ParticleFilter> f, std::shared_ptr<ParticleFilter> g)
        : ParticleFilter(), m_f(f), m_g(g)
        {
        }

    virtual ~ParticleFilterUnion() { }

    /** Test if a particle meets the selection criteria
     *  sysdef: the System Definition
     *
     *  Returns:
     *  all rank local particles that are in either filter m_f or filter
     *  m_g
     */
    virtual std::vector<unsigned int>
    getSelectedTags(std::shared_ptr<SystemDefinition> sysdef) const
        {
        // Get tags for f() and g() as sets
        auto X = m_f->getSelectedTags(sysdef);
        std::sort(X.begin(), X.end());

        auto Y = m_g->getSelectedTags(sysdef);
        std::sort(Y.begin(), Y.end());

        // Create vector and get union
        auto tags = std::vector<unsigned int>(X.size() + Y.size());
        auto it = std::set_union(X.begin(), X.end(), Y.begin(), Y.end(), tags.begin());
        tags.resize(it - tags.begin());
        return tags;
        }

    protected:
    std::shared_ptr<ParticleFilter> m_f;
    std::shared_ptr<ParticleFilter> m_g;
    };

    } // end namespace hoomd
#endif
