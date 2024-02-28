// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __PARTICLE_FILTER_INTERSECTION_H__
#define __PARTICLE_FILTER_INTERSECTION_H__

#include "ParticleFilter.h"
#include <algorithm>

namespace hoomd
    {
/// Represents the intersection of two filters: f and g.
class PYBIND11_EXPORT ParticleFilterIntersection : public ParticleFilter
    {
    public:
    /** Constructs the selector
     *  Args:
     *  f: first filter
     *  g: second filter
     */
    ParticleFilterIntersection(std::shared_ptr<ParticleFilter> f, std::shared_ptr<ParticleFilter> g)
        : ParticleFilter(), m_f(f), m_g(g)
        {
        }

    virtual ~ParticleFilterIntersection() { }

    /** Test if a particle meets the selection criteria
     *  Args:
     *  sysdef: the System Definition
     *
     *  Returns:
     *  all rank local particles that are in filter m_f and filter
     *  m_g
     */
    virtual std::vector<unsigned int>
    getSelectedTags(std::shared_ptr<SystemDefinition> sysdef) const
        {
        // Get vectors of tags from f and g
        auto X = m_f->getSelectedTags(sysdef);
        std::sort(X.begin(), X.end());

        auto Y = m_g->getSelectedTags(sysdef);
        std::sort(Y.begin(), Y.end());

        // Create vector and get intersection
        auto tags = std::vector<unsigned int>(std::min(X.size(), Y.size()));
        auto it = std::set_intersection(X.begin(), X.end(), Y.begin(), Y.end(), tags.begin());
        tags.resize(it - tags.begin());
        return tags;
        }

    protected:
    std::shared_ptr<ParticleFilter> m_f;
    std::shared_ptr<ParticleFilter> m_g;
    };

    } // end namespace hoomd
#endif
