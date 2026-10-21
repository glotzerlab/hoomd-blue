// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __NVCC__
#include "hoomd/SystemDefinition.h"
#include "hoomd/mpcd/ParticleData.h"
#endif // __NVCC__

#ifdef __NVCC__
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif // __NVCC__

namespace hoomd
    {
namespace mpcd
    {
class ParticleLoader
    {
    public:
    static constexpr bool use_particle_group = false;

#ifndef __NVCC__
    ParticleLoader(std::shared_ptr<SystemDefinition> sysdef)
        : m_mpcd_pdata(sysdef->getMPCDParticleData())
        {
        }
    const auto& getPositions() const
        {
        return m_mpcd_pdata->getPositions();
        }
    const auto& getVelocities() const
        {
        return m_mpcd_pdata->getVelocities();
        }
    const auto& getTags() const
        {
        return m_mpcd_pdata->getTags();
        }
    auto getN() const
        {
        return m_mpcd_pdata->getN();
        }
#endif // __NVCC__

    class IndexReader
        {
        public:
        HOSTDEVICE IndexReader() { }

        HOSTDEVICE unsigned int operator()(unsigned int idx) const
            {
            return idx;
            }
        };

    class VelocityMassReader
        {
        public:
        HOSTDEVICE VelocityMassReader(const Scalar4* velcell, Scalar mass)
            : m_velcell(velcell), m_mass(mass)
            {
            }

        HOSTDEVICE void read(Scalar3& velocity, Scalar& mass, unsigned int idx) const
            {
            const Scalar4 velcell = m_velcell[idx];
            velocity.x = velcell.x;
            velocity.y = velcell.y;
            velocity.z = velcell.z;
            mass = m_mass;
            }

        private:
        const Scalar4* m_velcell;
        Scalar m_mass;
        };

#ifndef __NVCC__
    VelocityMassReader makeVelocityMassReader(const Scalar4* velcell) const
        {
        return VelocityMassReader(velcell, m_mpcd_pdata->getMass());
        }

    private:
    std::shared_ptr<mpcd::ParticleData> m_mpcd_pdata; //!< MPCD particle data
#endif                                                // __NVCC__
    };
    } // namespace mpcd
    } // namespace hoomd
#undef HOSTDEVICE
