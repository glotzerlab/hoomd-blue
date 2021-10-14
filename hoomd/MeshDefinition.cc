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
    \param n_triangle_types Number of triangle types to create

*/
MeshDefinition::MeshDefinition(std::shared_ptr<ParticleData> pdata)
    {

    m_particle_data = pdata;
    m_data_changed = false;
    m_meshtriangle_data
        = std::shared_ptr<MeshTriangleData>(new MeshTriangleData(m_particle_data, 1));
    m_meshbond_data
        = std::shared_ptr<MeshBondData>(new MeshBondData(m_particle_data, 1));

    m_mesh_energy = 0;
    m_mesh_energy_old = 0;
    }

//! Re-initialize the system from a snapshot
void MeshDefinition::updateTriangleData()
    {
    if(m_data_changed)
         {
	 m_meshtriangle_data->takeSnapshot(triangle_data);
	 m_data_changed=false;
	 }
    }

//! Re-initialize the system from a snapshot
void MeshDefinition::updateMeshData()
    {
    m_meshtriangle_data = std::shared_ptr<MeshTriangleData>(new MeshTriangleData(m_particle_data, triangle_data));
    m_meshbond_data = std::shared_ptr<MeshBondData>(new MeshBondData(m_particle_data, triangle_data));
    }

void export_MeshDefinition(py::module& m)
    {
    py::class_<MeshDefinition, std::shared_ptr<MeshDefinition>>(m, "MeshDefinition")
        .def(py::init<>())
        .def(py::init<std::shared_ptr<ParticleData> >())
        .def("getMeshTriangleData", &MeshDefinition::getMeshTriangleData)
        .def("getMeshBondData", &MeshDefinition::getMeshBondData)
        .def("updateTriangleData", &MeshDefinition::updateTriangleData)
        .def("updateMeshData", &MeshDefinition::updateMeshData)
        .def_readonly("triangles", &MeshDefinition::triangle_data)
        .def_readonly("mesh_energy", &MeshDefinition::m_mesh_energy);
    }
