/*! \file MeshDefinition.cc
    \brief Defines MeshDefinition
*/

#include "MeshDefinition.h"

#ifdef ENABLE_MPI
#include "Communicator.h"
#endif

using namespace std;

namespace hoomd
    {
/*! \post All shared pointers contained in MeshDefinition are NULL
 */
MeshDefinition::MeshDefinition() { }

/*! \param sysdef Simulation system
 */
MeshDefinition::MeshDefinition(std::shared_ptr<SystemDefinition> sysdef)
    : m_sysdef(sysdef), m_meshbond_data(std::shared_ptr<MeshBondData>(
                            new MeshBondData(m_sysdef->getParticleData(), 1))),
      m_meshtriangle_data(
          std::shared_ptr<MeshTriangleData>(new MeshTriangleData(m_sysdef->getParticleData(), 1))),
      m_mesh_energy(0), m_data_changed(false)

    {
    }

//! Bond array getter
BondData::Snapshot MeshDefinition::getBondData()
    {
    BondData::Snapshot bond_data;
    m_meshbond_data->takeSnapshot(bond_data);
    return bond_data;
    }

//! Update triangle data to make it accessible for python
void MeshDefinition::updateTriangleData()
    {
    if (m_data_changed)
        {
        m_meshtriangle_data->takeSnapshot(triangle_data);
        m_data_changed = false;
        }
    }

//! Update data from snapshot
void MeshDefinition::updateMeshData()
    {
    m_meshtriangle_data = std::shared_ptr<MeshTriangleData>(
        new MeshTriangleData(m_sysdef->getParticleData(), triangle_data));
    m_meshbond_data = std::shared_ptr<MeshBondData>(
        new MeshBondData(m_sysdef->getParticleData(), triangle_data));
    }

//! Update data from snapshot
void MeshDefinition::MeshDataChange()
    {
    m_data_changed = true;
    }

namespace detail
    {
void export_MeshDefinition(pybind11::module& m)
    {
    pybind11::class_<MeshDefinition, std::shared_ptr<MeshDefinition>>(m, "MeshDefinition")
        .def(pybind11::init<>())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
        .def("getMeshTriangleData", &MeshDefinition::getMeshTriangleData)
        .def("getMeshBondData", &MeshDefinition::getMeshBondData)
        .def("getBondData", &MeshDefinition::getBondData)
        .def("updateTriangleData", &MeshDefinition::updateTriangleData)
        .def("updateMeshData", &MeshDefinition::updateMeshData)
        .def("getEnergy", &MeshDefinition::getEnergy)
        .def_readonly("triangles", &MeshDefinition::triangle_data)
        .def_property_readonly("types", &MeshDefinition::getTypes)
        .def_property_readonly("size", &MeshDefinition::getSize);
    }

    } // end namespace detail

    } // end namespace hoomd
