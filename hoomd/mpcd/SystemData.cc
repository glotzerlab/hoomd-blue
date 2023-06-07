// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/SystemData.cc
 * \brief Defines the mpcd::SystemData class
 */

#include "SystemData.h"
#ifdef ENABLE_HIP
#include "CellListGPU.h"
#endif // ENABLE_HIP

namespace hoomd
    {
/*!
 * \param sysdef System definition
 * \param particles MPCD ParticleData
 */
mpcd::SystemData::SystemData(std::shared_ptr<hoomd::SystemDefinition> sysdef,
                             std::shared_ptr<mpcd::ParticleData> particles)
    : m_sysdef(sysdef), m_particles(particles),
      m_global_box(std::make_shared<const BoxDim>(m_sysdef->getParticleData()->getGlobalBox()))
    {
// Generate one companion cell list for the system
/*
 * There is limited overhead to automatically creating a cell list (it is not sized
 * until first compute), so we always make one.
 */
#ifdef ENABLE_HIP
    if (m_sysdef->getParticleData()->getExecConf()->isCUDAEnabled())
        {
        m_cl = std::make_shared<mpcd::CellListGPU>(m_sysdef, m_particles);
        }
    else
#endif // ENABLE_HIP
        {
        m_cl = std::make_shared<mpcd::CellList>(m_sysdef, m_particles);
        }

    // connect to box change signal to enforce constant box dim in MPCD
    m_sysdef->getParticleData()
        ->getBoxChangeSignal()
        .connect<mpcd::SystemData, &mpcd::SystemData::checkBox>(this);

    // check that the MPCD box matches the HOOMD box
    checkBox();
    }

/*!
 * \param snapshot MPCD system snapshot to initialize from
 */
mpcd::SystemData::SystemData(std::shared_ptr<hoomd::SystemDefinition> sysdef,
                             std::shared_ptr<mpcd::ParticleDataSnapshot> snapshot)
    : m_sysdef(sysdef)
    {
    auto pdata = m_sysdef->getParticleData();
    m_global_box = std::make_shared<const BoxDim>(pdata->getGlobalBox());
    m_particles = std::shared_ptr<mpcd::ParticleData>(
#ifdef ENABLE_MPI
        new mpcd::ParticleData(snapshot,
                               m_global_box,
                               pdata->getExecConf(),
                               pdata->getDomainDecomposition())
#else
        new mpcd::ParticleData(snapshot, m_global_box, pdata->getExecConf())
#endif
    );

// Generate one companion cell list for the system
/*
 * There is limited overhead to automatically creating a cell list (it is not sized
 * until first compute), so we always make one.
 */
#ifdef ENABLE_HIP
    if (pdata->getExecConf()->isCUDAEnabled())
        {
        m_cl = std::make_shared<mpcd::CellListGPU>(m_sysdef, m_particles);
        }
    else
#endif // ENABLE_HIP
        {
        m_cl = std::make_shared<mpcd::CellList>(m_sysdef, m_particles);
        }

    // connect to box change signal to enforce constant box dim in MPCD
    m_sysdef->getParticleData()
        ->getBoxChangeSignal()
        .connect<mpcd::SystemData, &mpcd::SystemData::checkBox>(this);

    // check that the MPCD box matches the HOOMD box
    checkBox();
    }

mpcd::SystemData::~SystemData()
    {
    m_sysdef->getParticleData()
        ->getBoxChangeSignal()
        .disconnect<mpcd::SystemData, &mpcd::SystemData::checkBox>(this);
    }

std::shared_ptr<mpcd::ParticleDataSnapshot> mpcd::SystemData::takeParticleSnapshot()
    {
    std::shared_ptr<mpcd::ParticleDataSnapshot> snapshot;
    m_particles->takeSnapshot(snapshot, m_global_box);
    return snapshot;
    }

/*!
 * \param snapshot HOOMD system snapshot to initialize from
 */
void mpcd::SystemData::initializeFromParticleSnapshot(
    std::shared_ptr<mpcd::ParticleDataSnapshot> snapshot)
    {
    m_particles->initializeFromSnapshot(snapshot, m_global_box);
    }

/*!
 * \param m Python module to export to
 */
void mpcd::detail::export_SystemData(pybind11::module& m)
    {
    pybind11::class_<mpcd::SystemData, std::shared_ptr<mpcd::SystemData>>(m, "SystemData")
        .def(pybind11::init<std::shared_ptr<hoomd::SystemDefinition>,
                            std::shared_ptr<mpcd::ParticleData>>())
        .def(pybind11::init<std::shared_ptr<hoomd::SystemDefinition>,
                            std::shared_ptr<mpcd::ParticleDataSnapshot>>())
        .def("getParticleData", &mpcd::SystemData::getParticleData)
        .def("getCellList", &mpcd::SystemData::getCellList)
        .def("takeParticleSnapshot", &mpcd::SystemData::takeParticleSnapshot)
        .def("initializeFromParticleSnapshot", &mpcd::SystemData::initializeFromParticleSnapshot);
    }

    } // end namespace hoomd
