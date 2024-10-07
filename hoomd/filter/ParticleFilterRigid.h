// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __PARTICLE_FILTER_RIGID_H__
#define __PARTICLE_FILTER_RIGID_H__

#include "../SystemDefinition.h"
#include "ParticleFilter.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <string>
#include <vector>

/// Utility class to select particles based on given conditions
/** \b Overview
    Filters particles based on them being rigid body centers or not.
*/

namespace hoomd
    {
enum class RigidBodySelection
    {
    NONE = 0,
    CENTERS = 1,
    CONSTITUENT = 2,
    FREE = 4
    };

constexpr enum RigidBodySelection operator|(const enum RigidBodySelection flag1,
                                            const enum RigidBodySelection flag2)
    {
    return static_cast<enum RigidBodySelection>(static_cast<unsigned int>(flag1)
                                                | static_cast<unsigned int>(flag2));
    }

constexpr enum RigidBodySelection operator&(const enum RigidBodySelection flag1,
                                            const enum RigidBodySelection flag2)
    {
    return static_cast<enum RigidBodySelection>(static_cast<unsigned int>(flag1)
                                                & static_cast<unsigned int>(flag2));
    }

constexpr bool toBool(const enum RigidBodySelection& flag)
    {
    return static_cast<unsigned int>(flag) > 0;
    }

class PYBIND11_EXPORT ParticleFilterRigid : public ParticleFilter
    {
    public:
    /// constructs a ParticleFilterRigid
    ParticleFilterRigid(RigidBodySelection flag) : m_current_selection {flag} { };

    ParticleFilterRigid(pybind11::tuple flags) : m_current_selection(RigidBodySelection::NONE)
        {
        for (size_t i = 0; i < pybind11::len(flags); ++i)
            {
            auto flag = flags[i].cast<std::string>();
            if (flag == "center")
                {
                m_current_selection = m_current_selection | RigidBodySelection::CENTERS;
                }
            else if (flag == "constituent")
                {
                m_current_selection = m_current_selection | RigidBodySelection::CONSTITUENT;
                }
            else if (flag == "free")
                {
                m_current_selection = m_current_selection | RigidBodySelection::FREE;
                }
            }
        }

    virtual ~ParticleFilterRigid() { }

    /** Test if a particle meets the selection criteria.
     *  The base case returns an empty vector.
     */
    virtual std::vector<unsigned int>
    getSelectedTags(std::shared_ptr<SystemDefinition> sysdef) const
        {
        auto pdata = sysdef->getParticleData();

        ArrayHandle<unsigned int> h_tag(pdata->getTags(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_body(pdata->getBodies(),
                                         access_location::host,
                                         access_mode::read);

        std::vector<unsigned int> member_tags;
        for (unsigned int idx = 0; idx < pdata->getN(); ++idx)
            {
            unsigned int tag = h_tag.data[idx];

            // get position of particle
            unsigned int body = h_body.data[idx];

            // see if it matches the criteria
            bool include_particle = false;
            if (toBool(m_current_selection & RigidBodySelection::CENTERS))
                {
                include_particle = include_particle || (tag == body);
                }
            if (toBool(m_current_selection & RigidBodySelection::CONSTITUENT))
                {
                include_particle = include_particle || (body < MIN_FLOPPY && body != tag);
                }
            if (toBool(m_current_selection & RigidBodySelection::FREE))
                {
                include_particle = include_particle || (body == NO_BODY);
                }

            if (include_particle)
                {
                member_tags.push_back(tag);
                }
            }
        return member_tags;
        }

    private:
    /// Current selection of particles to chose from rigid body center, constituent particles,
    /// and free bodies.
    RigidBodySelection m_current_selection;
    };

    } // end namespace hoomd
#endif
