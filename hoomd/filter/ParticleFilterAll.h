#ifndef __PARTICLE_FILTER_ALL_H__
#define __PARTICLE_FILTER_ALL_H__

#include "ParticleFilter.h"

//! Select all particles
class PYBIND11_EXPORT ParticleFilterAll : public ParticleFilter
    {
    public:
        //! Constructs the selector
        ParticleFilterAll() : ParticleFilter() {};
        virtual ~ParticleFilterAll() {}

        /*! \param sysdef the System Definition
            \returns all particles in the local rank
        */
        virtual std::vector<unsigned int> getSelectedTags(
                std::shared_ptr<SystemDefinition> sysdef) const
        {
        std::vector<unsigned int> member_tags;
        auto pdata = sysdef->getParticleData();

        // loop through local particles and select those that match selection
        // criterion
        ArrayHandle<unsigned int> h_tag(pdata->getTags(),
                                        access_location::host,
                                        access_mode::read);

        for (unsigned int idx = 0; idx < pdata->getN(); ++idx)
            {
            unsigned int tag = h_tag.data[idx];
            member_tags.push_back(tag);
            }
        return member_tags;
        }

    };
#endif
