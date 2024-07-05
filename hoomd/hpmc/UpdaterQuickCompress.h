// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// inclusion guard
#pragma once

#include <cmath>
#include <hoomd/RandomNumbers.h>
#include <hoomd/Updater.h>
#include <hoomd/Variant.h>
#include <hoomd/VectorVariant.h>

#include "IntegratorHPMC.h"

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace hpmc
    {
/** Quick compression algorithm

    Quickly compress HPMC systems to a (time-varying) target box volume. The quick compression
   algorithm performs random box moves to compress the box and *temporarily* allows overlaps. Local
   trial moves remove these overlaps over time.
*/
class UpdaterQuickCompress : public Updater
    {
    public:
    /** Constructor

        @param sysdef System definition
        @param mc HPMC integrator object
        @param max_overlaps_per_particle Maximum number of overlaps allowed per particle
        @param min_scale The minimum scale factor to use when scaling the box parameters
        @param target_box The target box
    */
    UpdaterQuickCompress(std::shared_ptr<SystemDefinition> sysdef,
                         std::shared_ptr<Trigger> trigger,
                         std::shared_ptr<IntegratorHPMC> mc,
                         double max_overlaps_per_particle,
                         double min_scale,
                         std::shared_ptr<VectorVariantBox> target_box);

    /// Destructor
    virtual ~UpdaterQuickCompress();

    /** Handle MaxParticleNumberChange signal

        Resize the m_pos_backup array
    */
    void slotMaxNChange()
        {
        unsigned int MaxN = m_pdata->getMaxN();
        m_pos_backup.resize(MaxN);
        }

    /** Take one timestep forward

        @param timestep timestep at which update is being evaluated
    */
    virtual void update(uint64_t timestep);

    /// Get whether unsafe (overlap-allowed) resizes are enabled
    double getAllowUnsafeResize()
        {
        return m_allow_unsafe_resize;
        }

    /// Set whether unsafe (overlap-allowed) resizes are enabled
    void setAllowUnsafeResize(bool allow_unsafe_resize)
        {
        m_allow_unsafe_resize = allow_unsafe_resize;
        }

    /// Get the maximum number of overlaps allowed per particle
    double getMaxOverlapsPerParticle()
        {
        return m_max_overlaps_per_particle;
        }

    /// Set the maximum number of overlaps allowed per particle
    void setMaxOverlapsPerParticle(double max_overlaps_per_particle)
        {
        m_max_overlaps_per_particle = max_overlaps_per_particle;
        }

    /// Get the minimum scale factor
    double getMinScale()
        {
        return m_min_scale;
        }

    /// Set the minimum scale factor
    void setMinScale(double min_scale)
        {
        if (min_scale <= 0 || min_scale >= 1.0)
            {
            throw std::domain_error("min_scale must be in the range (0,1)");
            }
        m_min_scale = min_scale;
        }

    /// Get the target box
    std::shared_ptr<VectorVariantBox> getTargetBox()
        {
        return m_target_box;
        }

    /// Set the target box
    void setTargetBox(std::shared_ptr<VectorVariantBox> target_box)
        {
        m_target_box = target_box;
        }

    /// Return true if the updater is complete and the simulation should end.
    virtual bool isComplete()
        {
        return m_is_complete;
        }

    /// Set the RNG instance
    void setInstance(unsigned int instance)
        {
        m_instance = instance;
        }

    /// Get the RNG instance
    unsigned int getInstance()
        {
        return m_instance;
        }

    private:
    /// HPMC integrator object
    std::shared_ptr<IntegratorHPMC> m_mc;

    /// Maximum number of overlaps allowed per particle
    double m_max_overlaps_per_particle;

    /// Minimum scale factor to use when scaling box parameters
    double m_min_scale;

    /// The target box dimensions
    std::shared_ptr<VectorVariantBox> m_target_box;

    /// Unique ID for RNG seeding
    unsigned int m_instance = 0;

    /// hold backup copy of particle positions
    GPUArray<Scalar4> m_pos_backup;

    /// Flag whether unsafe box resizes are allowed
    bool m_allow_unsafe_resize = false;

    /// Perform the box scale move
    void performBoxScale(uint64_t timestep, const BoxDim& target_box);

    /// Get the new box to set
    BoxDim getNewBox(uint64_t timestep, const BoxDim& target_box);

    /// Store the last HPMC counters
    hpmc_counters_t m_last_move_counters;

    /// Track whether the compression is complete
    bool m_is_complete = false;
    };

namespace detail
    {
/// Export UpdaterQuickCompress to Python
void export_UpdaterQuickCompress(pybind11::module& m);
    } // end namespace detail
    } // end namespace hpmc
    } // end namespace hoomd
