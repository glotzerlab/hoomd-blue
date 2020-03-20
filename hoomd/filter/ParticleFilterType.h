#ifndef __PARTICLE_FILTER_TYPE_H__
#define __PARTICLE_FILTER_TYPE_H__

#include "ParticleFilter.h"
#include <unordered_set>
#include <string>
#include <pybind11/stl.h>

//! Select particles based on their type
class PYBIND11_EXPORT ParticleFilterType : public ParticleFilter
    {
    public:
        /** Constructs the selector
         *  Args:
         *  included_types: set of type string values to include
        */
        ParticleFilterType(std::unordered_set<std::string> included_types)
            : ParticleFilter(), m_types(included_types) {}

        virtual ~ParticleFilterType() {}

        /** Test if a particle meets the selection criteria
         *  sysdef: system definition to find tags for
         *
         *  Returns:
         *  tags of all rank local particles of types in m_types
        */
        virtual std::vector<unsigned int> getSelectedTags(
                std::shared_ptr<SystemDefinition> sysdef) const
            {
            auto pdata = sysdef->getParticleData();
            // loop through local particles and select those that match
            // selection criterion
            ArrayHandle<unsigned int> h_tag(pdata->getTags(),
                                            access_location::host,
                                            access_mode::read);
            ArrayHandle<Scalar4> h_postype(pdata->getPositions(),
                                           access_location::host,
                                           access_mode::read);

            // Get types as unsigned ints
            std::unordered_set<unsigned int> types(m_types.size());
            for (auto type_str: m_types)
                {
                types.insert(pdata->getTypeByName(type_str));
                }

            // Add correctly typed particles to vector
            std::vector<unsigned int> member_tags;
            for (unsigned int idx = 0; idx < pdata->getN(); ++idx)
                {
                unsigned int tag = h_tag.data[idx];
                unsigned int typ = __scalar_as_int(h_postype.data[idx].w);
                if (types.count(typ))
                    member_tags.push_back(tag);
                }
            return member_tags;
            }

    protected:
        std::unordered_set<std::string> m_types;   ///< Set of types to select
    };
#endif
