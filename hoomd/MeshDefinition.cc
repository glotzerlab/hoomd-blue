// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: joaander

/*! \file MeshDefinition.cc
    \brief Defines MeshDefinition
*/

#include "MeshDefinition.h"

#ifdef ENABLE_MPI
#include "Communicator.h"
#endif

namespace py = pybind11;

using namespace std;

/*! \post All shared pointers contained in MeshDefinition are NULL
 */
MeshDefinition::MeshDefinition() { }

/*! \param N Number of particles to allocate
    \param box Initial box particles are in
    \param n_types Number of particle types to set
    \param n_bond_types Number of bond types to create
    \param n_angle_types Number of angle types to create
    \param n_dihedral_types Number of dihedral types to create
    \param n_improper_types Number of improper types to create
    \param n_triangle_types Number of triangle types to create
    \param exec_conf The ExecutionConfiguration HOOMD is to be run on

    Creating MeshDefinition with this constructor results in
     - ParticleData constructed with the arguments \a N, \a box, \a n_types, and \a exec_conf->
     - BondData constructed with the arguments \a n_bond_types
     - All other data structures are default constructed.
*/
MeshDefinition::MeshDefinition(std::shared_ptr<ParticleData> pdata, unsigned int n_triangle_types)
    {
    m_triangle_data
        = std::shared_ptr<TriangleData>(new TriangleData(pdata, n_triangle_types));
    m_meshtriangle_data
        = std::shared_ptr<MeshTriangleData>(new MeshTriangleData(pdata, n_triangle_types));
    m_meshbond_data
        = std::shared_ptr<MeshBondData>(new MeshBondData(pdata, n_triangle_types));

    m_triangle_change = false;
    m_mesh_change = false;
    m_mesh_energy = 0;
    m_mesh_energy_old = 0;
    }

/*! Evaluates the snapshot and initializes the respective *Data classes using
   its contents (box dimensions and sub-snapshots)
    \param snapshot Snapshot to use
    \param exec_conf Execution configuration to run on
    \param decomposition (optional) The domain decomposition layout
*/
MeshDefinition::MeshDefinition(std::shared_ptr<ParticleData> pdata, TriangleData::Snapshot snapshot)
    {

    m_triangle_data
        = std::shared_ptr<TriangleData>(new TriangleData(pdata, snapshot));

    m_meshtriangle_data
        = std::shared_ptr<MeshTriangleData>(new MeshTriangleData(pdata, snapshot));

    m_meshbond_data
        = std::shared_ptr<MeshBondData>(new MeshBondData(pdata, snapshot));

    m_mesh_energy = 0;
    m_mesh_energy_old = 0;
    }

//! Re-initialize the system from a snapshot
std::shared_ptr<TriangleData> MeshDefinition::getTriangleData()
    {
    if(m_mesh_change)
        {
        TriangleData::Snapshot snapshot;
        m_meshtriangle_data->takeSnapshot(snapshot);
        m_triangle_data = std::shared_ptr<TriangleData>(new TriangleData(m_particle_data, snapshot));
        m_mesh_change = false;
        }
    m_triangle_change = true;
    return m_triangle_data;
    }

//! Re-initialize the system from a snapshot
void MeshDefinition::checkMeshData()
    {
    if(m_triangle_change)
        {
        TriangleData::Snapshot snapshot;
        m_triangle_data->takeSnapshot(snapshot);
        m_meshtriangle_data = std::shared_ptr<MeshTriangleData>(new MeshTriangleData(m_particle_data, snapshot));
        m_meshbond_data = std::shared_ptr<MeshBondData>(new MeshBondData(m_particle_data, snapshot));
        m_triangle_change = false;
        }
    }

void export_MeshDefinition(py::module& m)
    {
    py::class_<MeshDefinition, std::shared_ptr<MeshDefinition>>(m, "MeshDefinition")
        .def(py::init<>())
        .def(py::init<std::shared_ptr<ParticleData>,
                      unsigned int>())
        .def(py::init<std::shared_ptr<ParticleData>,
                      TriangleData::Snapshot>())
        .def("getTriangleData", &MeshDefinition::getTriangleData)
        .def("getMeshTriangleData", &MeshDefinition::getMeshTriangleData)
        .def("getMeshBondData", &MeshDefinition::getMeshBondData)
        .def("checkMeshData", &MeshDefinition::checkMeshData)
    }
