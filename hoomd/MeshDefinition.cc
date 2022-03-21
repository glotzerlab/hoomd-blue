// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

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
      m_mesh_energy(0)

    {
    }

void MeshDefinition::setTypes(pybind11::list types)
    {
    m_meshbond_data->setTypeName(0, pybind11::cast<string>(types[0]));
    m_meshtriangle_data->setTypeName(0, pybind11::cast<string>(types[0]));
    }

//! Bond array getter
BondData::Snapshot MeshDefinition::getBondData()
    {
    BondData::Snapshot bond_data;
    m_meshbond_data->takeSnapshot(bond_data);
    return bond_data;
    }

//! Triangle array getter
TriangleData::Snapshot MeshDefinition::getTriangleData()
    {
    TriangleData::Snapshot triangle_data;
	std::cout << "Schon da?" << std::endl;
    m_meshtriangle_data->takeSnapshot(triangle_data);
	std::cout << "Schon weg?" << std::endl;
    return triangle_data;
    }

//! Triangle array setter
void MeshDefinition::setTriangleData(pybind11::array_t<int> triangles)
    {

	std::cout << "Wo sind ma denn hier?" << std::endl;
    TriangleData::Snapshot triangle_data = getTriangleData();
    //TriangleData::Snapshot triangle_data;
	std::cout << "Schon da?" << std::endl;
    pybind11::buffer_info buf = triangles.request();
	std::cout << "Fast am Ende" << std::endl;
    int* ptr = static_cast<int*>(buf.ptr);
	std::cout << "Weiter" << std::endl;
    size_t len_triang = len(triangles);
	std::cout << "Oha" << std::endl;
    triangle_data.resize(static_cast<unsigned int>(len_triang));
	std::cout << "Jetzt aber" << std::endl;
    TriangleData::members_t triangle_new;


    for (size_t i = 0; i < len_triang; i++)
        {
        triangle_new.tag[0] = ptr[i * 3];
        triangle_new.tag[1] = ptr[i * 3 + 1];
        triangle_new.tag[2] = ptr[i * 3 + 2];
        triangle_data.groups[i] = triangle_new;
        }


    m_meshtriangle_data = std::shared_ptr<MeshTriangleData>(
        new MeshTriangleData(m_sysdef->getParticleData(), triangle_data));
    m_meshbond_data = std::shared_ptr<MeshBondData>(
        new MeshBondData(m_sysdef->getParticleData(), triangle_data));

	std::cout << "Ende!" << std::endl;
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
        .def("getTriangleData", &MeshDefinition::getTriangleData)
        .def("setTriangleData", &MeshDefinition::setTriangleData)
        .def("setTypes", &MeshDefinition::setTypes)
        .def("getEnergy", &MeshDefinition::getEnergy)
        .def_property_readonly("types", &MeshDefinition::getTypes)
        .def_property_readonly("size", &MeshDefinition::getSize)
#ifdef ENABLE_MPI
        .def("setCommunicator", &MeshDefinition::setCommunicator)
#endif
	;
    }

    } // end namespace detail

    } // end namespace hoomd
