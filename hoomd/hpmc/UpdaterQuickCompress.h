// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// inclusion guard
#pragma once

#include <cmath>
#include <hoomd/RandomNumbers.h>
#include <hoomd/Updater.h>
#include <hoomd/Variant.h>

#include "IntegratorHPMC.h"

#include <pybind11/pybind11.h>

namespace hpmc
    {
/** Quick compression algorithm

    Quickly compress HPMC systems to a target box volume. The quick compression algorithm performs
    random box moves to compress the box and *temporarily* allows overlaps. Local trial moves
    remove these overlaps over time.
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
        @param seed PRNG seed
    */
    UpdaterQuickCompress(std::shared_ptr<SystemDefinition> sysdef,
                         std::shared_ptr<IntegratorHPMC> mc,
                         double max_overlaps_per_particle,
                         double min_scale,
                         pybind11::object target_box,
                         const unsigned int seed);

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
    virtual void update(unsigned int timestep);

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
    pybind11::object getTargetBox()
        {
        return m_target_box;
        }

    /// Set the target box
    void setTargetBox(pybind11::object target_box)
        {
        m_target_box = target_box;
        }

    unsigned int getSeed()
        {
        return m_seed;
        }

    private:
    /// HPMC integrator object
    std::shared_ptr<IntegratorHPMC> m_mc;

    /// Maximum number of overlaps allowed per particle
    double m_max_overlaps_per_particle;

    /// Minimum scale factor to use when scaling box parameters
    double m_min_scale;

    /// The target box dimensions
    pybind11::object m_target_box;

    /// The RNG seed
    unsigned int m_seed;

    /// hold backup copy of particle positions
    GPUArray<Scalar4> m_pos_backup;

    /// Perform the box scale move
    void performBoxScale(unsigned int timestep);

    /// Get the new box to set
    BoxDim getNewBox(unsigned int timestep);
    };

/// Export UpdaterQuickCompress to Python
void export_UpdaterQuickCompress(pybind11::module& m);

    } // namespace hpmc
