// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "UpdaterQuickCompress.h"
#include "hoomd/RNGIdentifiers.h"

namespace hpmc
    {
UpdaterQuickCompress::UpdaterQuickCompress(std::shared_ptr<SystemDefinition> sysdef,
                                           std::shared_ptr<IntegratorHPMC> mc,
                                           double max_overlaps_per_particle,
                                           double min_scale,
                                           const BoxDim& target_box,
                                           const unsigned int seed)
    : Updater(sysdef), m_mc(mc), m_max_overlaps_per_particle(max_overlaps_per_particle),
      m_target_box(target_box), m_seed(seed)
    {
    m_exec_conf->msg->notice(5) << "Constructing UpdaterQuickCompress" << std::endl;
    setMinScale(min_scale);

// broadcast the seed from rank 0 to all other ranks.
#ifdef ENABLE_MPI
    if (this->m_pdata->getDomainDecomposition())
        bcast(m_seed, 0, this->m_exec_conf->getMPICommunicator());
#endif

    // allocate memory for m_pos_backup
    unsigned int MaxN = m_pdata->getMaxN();
    GPUArray<Scalar4>(MaxN, m_exec_conf).swap(m_pos_backup);

    // Connect to the MaxParticleNumberChange signal
    m_pdata->getMaxParticleNumberChangeSignal()
        .connect<UpdaterQuickCompress, &UpdaterQuickCompress::slotMaxNChange>(this);
    }

UpdaterQuickCompress::~UpdaterQuickCompress()
    {
    m_exec_conf->msg->notice(5) << "Destroying UpdaterQuickCompress" << std::endl;
    m_pdata->getMaxParticleNumberChangeSignal()
        .disconnect<UpdaterQuickCompress, &UpdaterQuickCompress::slotMaxNChange>(this);
    }

void UpdaterQuickCompress::update(unsigned int timestep)
    {
    if (m_prof)
        m_prof->push("UpdaterQuickCompress");
    m_exec_conf->msg->notice(10) << "UpdaterQuickCompress: " << timestep << std::endl;

    // count the number of overlaps in the current configuration
    auto n_overlaps = m_mc->countOverlaps(false);
    BoxDim curBox = m_pdata->getGlobalBox();

    if (n_overlaps == 0 && curBox != m_target_box)
        {
        performBoxScale(timestep);
        }

    if (m_prof)
        m_prof->pop();

    // TODO: Enable the ability to flag that the compression is completed
    // if (n_overlaps == 0 && curBox == m_target_box)
    //     return true;
    // else
    //     return false;
    }

void UpdaterQuickCompress::performBoxScale(unsigned int timestep)
    {
    BoxDim new_box = getNewBox(timestep);

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

    auto n_overlaps = m_mc->countOverlaps(false);
    if (n_overlaps > m_max_overlaps_per_particle * m_pdata->getN())
        {
        // the box move generated too many overlaps, undo the move
        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::readwrite);
        ArrayHandle<Scalar4> h_pos_backup(m_pos_backup, access_location::host, access_mode::read);
        unsigned int N = m_pdata->getN();
        assert(N == N_backup);
        memcpy(h_pos.data, h_pos_backup.data, sizeof(Scalar4) * N);
        }
    }

/** Adjust a value by the scale.

    Scales the current value toward a target without exceeding the target.

    @param current The current value.
    @param target The target value.
    @param s Scale factor (in the range [0,1)).

    @returns The scaled value.
*/
static inline double scaleValue(double current, double target, double s)
    {
    assert(s <= 1.0);
    if (target < current)
        {
        return std::min(target, current * s);
        }
    else
        {
        return std::max(target, current * 1.0 / s);
        }
    }

BoxDim UpdaterQuickCompress::getNewBox(unsigned int timestep)
    {
    // Create a prng instance for this timestep
    hoomd::RandomGenerator rng(hoomd::RNGIdentifier::UpdaterQuickCompress, m_seed, timestep);

    // Determine the minimum allowable scale factor for each type in the simulation. The minimum
    // allowable scale factor assumes that the typical accepted trial move shifts particles 1/4 of
    // the maximum displacement. Assuming that the particles are all spheres with their circumsphere
    // diameter, set the minimum allowable scale factor so that overlaps of this size can be removed
    // by trial move. Max a conservative estimate using the minimum move size and the maximum core
    // diameter.
    double max_diameter = m_mc->getMaxCoreDiameter();
    double min_move_size = m_mc->getMinTransMoveSize() / 4.0;
    double min_scale = std::max(m_min_scale, 1.0 - min_move_size / max_diameter);

    // choose a scale randomly between min_scale and 1.0
    hoomd::UniformDistribution<double> uniform(min_scale, 1.0);
    double scale = uniform(rng);

    // construct the scaled box
    BoxDim current_box = m_pdata->getGlobalBox();
    Scalar3 new_L;
    new_L.x = scaleValue(current_box.getL().x, m_target_box.getL().x, scale);
    new_L.y = scaleValue(current_box.getL().y, m_target_box.getL().y, scale);
    new_L.z = scaleValue(current_box.getL().z, m_target_box.getL().z, scale);
    Scalar new_xy
        = scaleValue(current_box.getTiltFactorXY(), m_target_box.getTiltFactorXY(), scale);
    Scalar new_xz
        = scaleValue(current_box.getTiltFactorXZ(), m_target_box.getTiltFactorXZ(), scale);
    Scalar new_yz
        = scaleValue(current_box.getTiltFactorYZ(), m_target_box.getTiltFactorYZ(), scale);

    BoxDim new_box = current_box;
    new_box.setL(new_L);
    new_box.setTiltFactors(new_xy, new_xz, new_yz);
    return new_box;
    }

void export_UpdaterQuickCompress(pybind11::module& m)
    {
    pybind11::class_<UpdaterQuickCompress, Updater, std::shared_ptr<UpdaterQuickCompress>>(
        m,
        "UpdaterQuickCompress")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<IntegratorHPMC>,
                            double,
                            double,
                            const BoxDim&,
                            const unsigned int>())
        .def_property("max_overlaps_per_particle",
                      &UpdaterQuickCompress::getMaxOverlapsPerParticle,
                      &UpdaterQuickCompress::setMaxOverlapsPerParticle)
        .def_property("min_scale",
                      &UpdaterQuickCompress::getMinScale,
                      &UpdaterQuickCompress::setMinScale)
        .def_property("target_box",
                      &UpdaterQuickCompress::getTargetBox,
                      &UpdaterQuickCompress::setTargetBox)
        .def_property_readonly("seed", &UpdaterQuickCompress::getSeed);
    }

    } // end namespace hpmc
