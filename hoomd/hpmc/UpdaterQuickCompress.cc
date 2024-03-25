// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "UpdaterQuickCompress.h"
#include "hoomd/RNGIdentifiers.h"
#include <cmath>

namespace hoomd
    {
namespace hpmc
    {
UpdaterQuickCompress::UpdaterQuickCompress(std::shared_ptr<SystemDefinition> sysdef,
                                           std::shared_ptr<Trigger> trigger,
                                           std::shared_ptr<IntegratorHPMC> mc,
                                           double max_overlaps_per_particle,
                                           double min_scale,
                                           std::shared_ptr<VectorVariantBox> target_box)
    : Updater(sysdef, trigger), m_mc(mc), m_max_overlaps_per_particle(max_overlaps_per_particle),
      m_target_box(target_box)
    {
    m_exec_conf->msg->notice(5) << "Constructing UpdaterQuickCompress" << std::endl;
    setMinScale(min_scale);

    // allocate memory for m_pos_backup
    unsigned int MaxN = m_pdata->getMaxN();
    GPUArray<Scalar4>(MaxN, m_exec_conf).swap(m_pos_backup);

    // Connect to the MaxParticleNumberChange signal
    m_pdata->getMaxParticleNumberChangeSignal()
        .connect<UpdaterQuickCompress, &UpdaterQuickCompress::slotMaxNChange>(this);

    m_last_move_counters = m_mc->getCounters();
    }

UpdaterQuickCompress::~UpdaterQuickCompress()
    {
    m_exec_conf->msg->notice(5) << "Destroying UpdaterQuickCompress" << std::endl;
    m_pdata->getMaxParticleNumberChangeSignal()
        .disconnect<UpdaterQuickCompress, &UpdaterQuickCompress::slotMaxNChange>(this);
    }

void UpdaterQuickCompress::update(uint64_t timestep)
    {
    Updater::update(timestep);
    m_exec_conf->msg->notice(10) << "UpdaterQuickCompress: " << timestep << std::endl;

    // count the number of overlaps in the current configuration
    auto n_overlaps = m_mc->countOverlaps(false);
    BoxDim current_box = m_pdata->getGlobalBox();
    BoxDim target_box = BoxDim((*m_target_box)(timestep));
    if (n_overlaps == 0 && current_box != target_box)
        {
        performBoxScale(timestep, target_box);
        }

    // The compression is complete when we have reached the target box and there are no overlaps.
    if (n_overlaps == 0 && current_box == target_box)
        m_is_complete = true;
    else
        m_is_complete = false;
    }

void UpdaterQuickCompress::performBoxScale(uint64_t timestep, const BoxDim& target_box)
    {
    auto new_box = getNewBox(timestep, target_box);
    auto old_box = m_pdata->getGlobalBox();

    Scalar3 old_origin = m_pdata->getOrigin();

    // Make a backup copy of position data
    unsigned int N_backup = m_pdata->getN();
        {
        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::read);
        ArrayHandle<Scalar4> h_pos_backup(m_pos_backup,
                                          access_location::host,
                                          access_mode::overwrite);
        memcpy(h_pos_backup.data, h_pos.data, sizeof(Scalar4) * N_backup);
        }

    m_mc->attemptBoxResize(timestep, new_box);
    Scalar3 new_origin = m_pdata->getOrigin();
    Scalar3 origin_shift = new_origin - old_origin;

    auto n_overlaps = m_mc->countOverlaps(false);
    if (n_overlaps > m_max_overlaps_per_particle * m_pdata->getNGlobal())
        {
        // the box move generated too many overlaps, undo the move
        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::readwrite);
        ArrayHandle<Scalar4> h_pos_backup(m_pos_backup, access_location::host, access_mode::read);
        unsigned int N = m_pdata->getN();
        assert(N == N_backup);
        memcpy(h_pos.data, h_pos_backup.data, sizeof(Scalar4) * N);
        m_pdata->setGlobalBox(old_box);
        m_pdata->translateOrigin(-origin_shift);

        // we have moved particles, communicate those changes
        m_mc->communicate(false);
        }
    }

/** Adjust a value by the scale.

    Scales the current value toward a target without exceeding the target.

    @param current The current value.
    @param target The target value.
    @param s Scale factor (in the range [0,1)).

    @returns The scaled value.
*/
static inline double scaleLength(double current, double target, double s)
    {
    assert(s <= 1.0);
    if (target < current)
        {
        return std::max(target, current * s);
        }
    else
        {
        return std::min(target, current * 1.0 / s);
        }
    }

static inline double scaleTilt(double current, double target, double s)
    {
    assert(s <= 1.0);
    double scaled_value = current + (1.0 - s) * target;
    if (target < current)
        {
        return std::min(scaled_value, target);
        }
    else
        {
        return std::max(scaled_value, target);
        }
    }

BoxDim UpdaterQuickCompress::getNewBox(uint64_t timestep, const BoxDim& target_box)
    {
    // compute the current MC translate acceptance ratio
    auto current_counters = m_mc->getCounters();
    auto counter_delta = current_counters - m_last_move_counters;
    m_last_move_counters = current_counters;
    double accept_ratio = 1.0;
    if (counter_delta.translate_accept_count > 0)
        {
        accept_ratio
            = double(counter_delta.translate_accept_count)
              / double(counter_delta.translate_accept_count + counter_delta.translate_reject_count);
        }

    // If unsafe box moves are allowed, set min_scale without considering min_move_size.
    // Otherwise, determine the worst case minimum allowable scale factor. The minimum
    // allowable scale factor assumes that the typical accepted trial move shifts
    // particles by the current acceptance ratio times the maximum displacement. Assuming
    // that the particles are all spheres with their circumsphere diameter, set the
    // minimum allowable scale factor so that overlaps of this size can be removed by
    // trial move. The worst case estimate uses the minimum move size and the maximum core
    // diameter. Cap the acceptance ratio at 0.5 to prevent excessive box moves.
    double min_scale;

    if (m_allow_unsafe_resize)
        {
        min_scale = m_min_scale;
        }
    else
        {
        double max_diameter = m_mc->getMaxCoreDiameter();
        double min_move_size = m_mc->getMinTransMoveSize() * std::min(accept_ratio, 0.5);
        min_scale = std::max(m_min_scale, 1.0 - min_move_size / max_diameter);
        }

    // Create a prng instance for this timestep
    hoomd::RandomGenerator rng(
        hoomd::Seed(hoomd::RNGIdentifier::UpdaterQuickCompress, timestep, m_sysdef->getSeed()),
        hoomd::Counter(m_instance));

    // choose a scale randomly between min_scale and 1.0
    hoomd::UniformDistribution<double> uniform(min_scale, 1.0);
    double scale = uniform(rng);

    // construct the scaled box
    BoxDim current_box = m_pdata->getGlobalBox();
    Scalar3 new_L;
    Scalar new_xy, new_xz, new_yz;
    if (m_sysdef->getNDimensions() == 3)
        {
        new_L.x = scaleLength(current_box.getL().x, target_box.getL().x, scale);
        new_L.y = scaleLength(current_box.getL().y, target_box.getL().y, scale);
        new_L.z = scaleLength(current_box.getL().z, target_box.getL().z, scale);
        new_xy = scaleTilt(current_box.getTiltFactorXY(), target_box.getTiltFactorXY(), scale);
        new_xz = scaleTilt(current_box.getTiltFactorXZ(), target_box.getTiltFactorXZ(), scale);
        new_yz = scaleTilt(current_box.getTiltFactorYZ(), target_box.getTiltFactorYZ(), scale);
        }
    else
        {
        new_L.x = scaleLength(current_box.getL().x, target_box.getL().x, scale);
        new_L.y = scaleLength(current_box.getL().y, target_box.getL().y, scale);
        new_xy = scaleTilt(current_box.getTiltFactorXY(), target_box.getTiltFactorXY(), scale);

        // assume that the unused fields in the 2D target box are valid
        new_L.z = target_box.getL().z;
        new_xz = target_box.getTiltFactorXZ();
        new_yz = target_box.getTiltFactorYZ();
        }

    BoxDim new_box = current_box;
    new_box.setL(new_L);
    new_box.setTiltFactors(new_xy, new_xz, new_yz);
    return new_box;
    }

namespace detail
    {
void export_UpdaterQuickCompress(pybind11::module& m)
    {
    pybind11::class_<UpdaterQuickCompress, Updater, std::shared_ptr<UpdaterQuickCompress>>(
        m,
        "UpdaterQuickCompress")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<Trigger>,
                            std::shared_ptr<IntegratorHPMC>,
                            double,
                            double,
                            std::shared_ptr<VectorVariantBox>>())
        .def("isComplete", &UpdaterQuickCompress::isComplete)
        .def_property("max_overlaps_per_particle",
                      &UpdaterQuickCompress::getMaxOverlapsPerParticle,
                      &UpdaterQuickCompress::setMaxOverlapsPerParticle)
        .def_property("min_scale",
                      &UpdaterQuickCompress::getMinScale,
                      &UpdaterQuickCompress::setMinScale)
        .def_property("target_box",
                      &UpdaterQuickCompress::getTargetBox,
                      &UpdaterQuickCompress::setTargetBox)
        .def_property("instance",
                      &UpdaterQuickCompress::getInstance,
                      &UpdaterQuickCompress::setInstance)
        .def_property("allow_unsafe_resize",
                      &UpdaterQuickCompress::getAllowUnsafeResize,
                      &UpdaterQuickCompress::setAllowUnsafeResize);
    }
    } // end namespace detail
    } // end namespace hpmc
    } // end namespace hoomd
