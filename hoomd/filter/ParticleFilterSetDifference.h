// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __PARTICLE_FILTER_SET_DIFFERENCE_H__
#define __PARTICLE_FILTER_SET_DIFFERENCE_H__

#include "ParticleFilter.h"
#include <algorithm>

namespace hoomd
    {
/// Takes the set difference of two other filters
class PYBIND11_EXPORT ParticleFilterSetDifference : public ParticleFilter
    {
    public:
    /** Constructs the selector
     *  Args:
     *  f: first filter
     *  g: second filter
     */
    ParticleFilterSetDifference(std::shared_ptr<ParticleFilter> f,
                                std::shared_ptr<ParticleFilter> g)
        : ParticleFilter(), m_f(f), m_g(g)
        {
        }

    virtual ~ParticleFilterSetDifference() { }

    /** Test if a particle meets the selection criteria
     *  Args:
     *  sysdef: the System Definition
     *
     *  Returns:
     *  all rank local particles that are in filter m_f but not in
     *  filter m_g
     */
    virtual std::vector<unsigned int>
    getSelectedTags(std::shared_ptr<SystemDefinition> sysdef) const
        {
        // Get tags for f() and g()
        auto X = m_f->getSelectedTags(sysdef);
        std::sort(X.begin(), X.end());

        auto Y = m_g->getSelectedTags(sysdef);
        std::sort(Y.begin(), Y.end());

        // Create vector and get set difference
        auto tags = std::vector<unsigned int>(X.size());
        auto it = std::set_difference(X.begin(), X.end(), Y.begin(), Y.end(), tags.begin());
        tags.resize(it - tags.begin());
        return tags;
        }

    protected:
    std::shared_ptr<ParticleFilter> m_f;
    std::shared_ptr<ParticleFilter> m_g;
    };

    } // end namespace hoomd
#endif
