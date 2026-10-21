// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __NVCC__
#include "hoomd/ParticleData.h"
#include "hoomd/SystemDefinition.h"
#endif // __NVCC__

#ifdef __NVCC__
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif // __NVCC__

namespace hoomd
    {
class ParticleLoader
    {
    public:
    static constexpr bool use_particle_group = true;

#ifndef __NVCC__
    ParticleLoader(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<ParticleGroup> group)
        : m_pdata(sysdef->getParticleData()), m_group(group)
        {
        }
    const auto& getPositions() const
        {
        return m_pdata->getPositions();
        }
    const auto& getVelocities() const
        {
        return m_pdata->getVelocities();
        }
    const auto& getTags() const
        {
        return m_pdata->getTags();
        }
    auto getN() const
        {
        return m_group->getNumMembers();
        }
    const auto& getGroup() const
        {
        return m_group;
        }
#endif // __NVCC__

    class IndexReader
        {
        public:
        HOSTDEVICE IndexReader() : m_group_members(nullptr) { }
        HOSTDEVICE IndexReader(const unsigned int* group_members) : m_group_members(group_members)
            {
            }

        HOSTDEVICE unsigned int operator()(unsigned int idx) const
            {
            return m_group_members[idx];
            }

        private:
        const unsigned int* m_group_members;
        };

    class VelocityMassReader
        {
        public:
        HOSTDEVICE VelocityMassReader(const Scalar4* vel) : m_vel(vel) { }

        HOSTDEVICE void read(Scalar3& velocity, Scalar& mass, unsigned int idx) const
            {
            const Scalar4 vel = m_vel[idx];
            velocity.x = vel.x;
            velocity.y = vel.y;
            velocity.z = vel.z;
            mass = vel.w;
            }

        private:
        const Scalar4* m_vel;
        };

    VelocityMassReader makeVelocityMassReader(const Scalar4* vel) const
        {
        return VelocityMassReader(vel);
        }

#ifndef __NVCC__
    private:
    std::shared_ptr<ParticleData> m_pdata;  //!< Particle data
    std::shared_ptr<ParticleGroup> m_group; //!< Group member indexes
#endif                                      // __NVCC__
    };
    } // namespace hoomd
#undef HOSTDEVICE
