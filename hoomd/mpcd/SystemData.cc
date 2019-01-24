// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/SystemData.cc
 * \brief Defines the mpcd::SystemData class
 */

#include "SystemData.h"
#ifdef ENABLE_CUDA
#include "CellListGPU.h"
#endif // ENABLE_CUDA

/*!
 * \param sysdef System definition
 * \param particles MPCD ParticleData
 */
mpcd::SystemData::SystemData(std::shared_ptr<::SystemDefinition> sysdef,
                             std::shared_ptr<mpcd::ParticleData> particles)
    : m_sysdef(sysdef),
      m_particles(particles),
      m_global_box(m_sysdef->getParticleData()->getGlobalBox())
    {
    // Generate one companion cell list for the system
    /*
     * There is limited overhead to automatically creating a cell list (it is not sized
     * until first compute), so we always make one.
     */
    #ifdef ENABLE_CUDA
    if (m_sysdef->getParticleData()->getExecConf()->isCUDAEnabled())
        {
        m_cl = std::make_shared<mpcd::CellListGPU>(m_sysdef, m_particles);
        }
    else
    #endif // ENABLE_CUDA
        {
        m_cl = std::make_shared<mpcd::CellList>(m_sysdef, m_particles);
        }

    // connect to box change signal to enforce constant box dim in MPCD
    m_sysdef->getParticleData()->getBoxChangeSignal().connect<mpcd::SystemData, &mpcd::SystemData::checkBox>(this);

    // check that the MPCD box matches the HOOMD box
    checkBox();
    }

/*!
 * \param snapshot MPCD system snapshot to initialize from
 */
mpcd::SystemData::SystemData(std::shared_ptr<mpcd::SystemDataSnapshot> snapshot)
    : m_sysdef(snapshot->getSystemDefinition()), m_global_box(snapshot->getGlobalBox())
    {
    m_particles = std::shared_ptr<mpcd::ParticleData>(new mpcd::ParticleData(snapshot->particles,
                                                                             m_global_box,
                                                                             snapshot->getExecutionConfiguration(),
                                                                             snapshot->getDomainDecomposition()));

    // Generate one companion cell list for the system
    /*
     * There is limited overhead to automatically creating a cell list (it is not sized
     * until first compute), so we always make one.
     */
    #ifdef ENABLE_CUDA
    if (snapshot->getExecutionConfiguration()->isCUDAEnabled())
        {
        m_cl = std::make_shared<mpcd::CellListGPU>(m_sysdef, m_particles);
        }
    else
    #endif // ENABLE_CUDA
        {
        m_cl = std::make_shared<mpcd::CellList>(m_sysdef, m_particles);
        }

    // connect to box change signal to enforce constant box dim in MPCD
    m_sysdef->getParticleData()->getBoxChangeSignal().connect<mpcd::SystemData, &mpcd::SystemData::checkBox>(this);

    // check that the MPCD box matches the HOOMD box
    checkBox();
    }

mpcd::SystemData::~SystemData()
    {
    m_sysdef->getParticleData()->getBoxChangeSignal().disconnect<mpcd::SystemData, &mpcd::SystemData::checkBox>(this);
    }

//! Take a snapshot of the system
/*!
 * \param particles True if particle data should be saved
 */
std::shared_ptr<mpcd::SystemDataSnapshot> mpcd::SystemData::takeSnapshot(bool particles)
    {
    auto snap = std::make_shared<mpcd::SystemDataSnapshot>(m_sysdef);
    m_particles->takeSnapshot(snap->particles, m_global_box);

    return snap;
    }

//! (Re-)initialize the system from a snapshot
/*
 * \param snapshot MPCD system snapshot to initialize from
 */
void mpcd::SystemData::initializeFromSnapshot(std::shared_ptr<mpcd::SystemDataSnapshot> snapshot)
    {
    m_particles->initializeFromSnapshot(snapshot->particles, m_global_box);
    }

/*!
 * \param m Python module to export to
 */
void mpcd::detail::export_SystemData(pybind11::module& m)
    {
    namespace py = pybind11;

    py::class_<mpcd::SystemData, std::shared_ptr<mpcd::SystemData> >(m,"SystemData")
    .def(py::init<std::shared_ptr<::SystemDefinition>,std::shared_ptr<mpcd::ParticleData> >())
    .def(py::init<std::shared_ptr<mpcd::SystemDataSnapshot> >())
    .def("getParticleData", &mpcd::SystemData::getParticleData)
    .def("getCellList", &mpcd::SystemData::getCellList)
    .def("takeSnapshot", &mpcd::SystemData::takeSnapshot)
    .def("initializeFromSnapshot", &mpcd::SystemData::initializeFromSnapshot);
    }
