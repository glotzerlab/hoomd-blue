// Copyright (c) 2009-2024 The Regents of the University of Michigan.
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
MeshDefinition::MeshDefinition(std::shared_ptr<SystemDefinition> sysdef, unsigned int n_types)
    : m_sysdef(sysdef), m_meshbond_data(std::shared_ptr<MeshBondData>(
                            new MeshBondData(m_sysdef->getParticleData(), n_types))),
      m_meshtriangle_data(
          std::shared_ptr<TriangleData>(new TriangleData(m_sysdef->getParticleData(), n_types)))

    {
    // allocate the max number of neighbors per type allowed
    GlobalArray<unsigned int> globalN(n_types, m_sysdef->getParticleData()->getExecConf());
    m_globalN.swap(globalN);
    TAG_ALLOCATION(m_globalN);
    }

void MeshDefinition::setTypes(pybind11::list types)
    {
    m_globalN.resize(len(types));
    for (unsigned int i = 0; i < len(types); i++)
        {
        m_meshbond_data->setTypeName(i, types[i].cast<string>());
        m_meshtriangle_data->setTypeName(i, types[i].cast<string>());
        }
    }

//! Bond array getter
BondData::Snapshot MeshDefinition::getBondData()
    {
    BondData::Snapshot bond_data;
    m_meshbond_data->takeSnapshot(bond_data);
#ifdef ENABLE_MPI
    bond_data.bcast(0, m_sysdef->getParticleData()->getExecConf()->getMPICommunicator());
#endif
    return bond_data;
    }

//! Triangle array getter
TriangleData::Snapshot MeshDefinition::getTriangleData()
    {
    TriangleData::Snapshot triangle_data;
    m_meshtriangle_data->takeSnapshot(triangle_data);
#ifdef ENABLE_MPI
    triangle_data.bcast(0, m_sysdef->getParticleData()->getExecConf()->getMPICommunicator());
#endif
    return triangle_data;
    }

//! Triangle array getter
pybind11::object MeshDefinition::getTriangulationData()
    {
    pybind11::dict triangulation;

    TriangleData::Snapshot triangle_data = getTriangleData();

    unsigned int len_triang = triangle_data.getSize();

    std::vector<size_t> dims {len_triang, 3};
    auto triangles = pybind11::array_t<unsigned int>(dims);
    int* ptr1 = static_cast<int*>(triangles.request().ptr);

    auto typeids = pybind11::array_t<unsigned int>(len_triang);
    int* ptr2 = static_cast<int*>(typeids.request().ptr);

    for (size_t i = 0; i < len_triang; i++)
        {
        ptr1[i * 3] = triangle_data.groups[i].tag[0];
        ptr1[i * 3 + 1] = triangle_data.groups[i].tag[1];
        ptr1[i * 3 + 2] = triangle_data.groups[i].tag[2];
        ptr2[i] = triangle_data.type_id[i];
        }

    triangulation["type_ids"] = typeids;
    triangulation["triangles"] = triangles;

    return triangulation;
    }

//! Triangle array setter
void MeshDefinition::setTriangulationData(pybind11::dict triangulation)
    {
    pybind11::array_t<int> triangles = triangulation["triangles"].cast<pybind11::array_t<int>>();
    pybind11::array_t<int> type_ids = triangulation["type_ids"].cast<pybind11::array_t<int>>();

    TriangleData::Snapshot triangle_data = getTriangleData();
    const int* ptr1 = static_cast<const int*>(triangles.request().ptr);

    const int* ptr2 = static_cast<const int*>(type_ids.request().ptr);

    size_t len_triang = len(triangles);
    triangle_data.resize(static_cast<unsigned int>(len_triang));
    TriangleData::members_t triangle_new;

    ArrayHandle<unsigned int> h_globalN(m_globalN, access_location::host, access_mode::overwrite);

    for (unsigned int i = 0; i < m_meshtriangle_data->getNTypes(); i++)
        h_globalN.data[i] = 0;

    for (size_t i = 0; i < len_triang; i++)
        {
        triangle_new.tag[0] = ptr1[i * 3];
        triangle_new.tag[1] = ptr1[i * 3 + 1];
        triangle_new.tag[2] = ptr1[i * 3 + 2];
        triangle_data.groups[i] = triangle_new;
        triangle_data.type_id[i] = ptr2[i];

        h_globalN.data[triangle_data.type_id[i]] += 1;
        }

    m_meshtriangle_data = std::shared_ptr<TriangleData>(
        new TriangleData(m_sysdef->getParticleData(), triangle_data));
    m_meshbond_data = std::shared_ptr<MeshBondData>(
        new MeshBondData(m_sysdef->getParticleData(), triangle_data));

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        auto comm_weak = m_sysdef->getCommunicator();
        assert(comm_weak.lock());
        auto comm = comm_weak.lock();

        // register this class with the communicator
        comm->updateMeshDefinition();
        }
#endif
    }

namespace detail
    {
void export_MeshDefinition(pybind11::module& m)
    {
    pybind11::class_<MeshDefinition, std::shared_ptr<MeshDefinition>>(m, "MeshDefinition")
        .def(pybind11::init<>())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, unsigned int>())
        .def("getMeshTriangleData", &MeshDefinition::getMeshTriangleData)
        .def("getMeshBondData", &MeshDefinition::getMeshBondData)
        .def("getBondData", &MeshDefinition::getBondData)
        .def("setTypes", &MeshDefinition::setTypes)
        .def("getSize", &MeshDefinition::getSize)
        .def_property("triangulation",
                      &MeshDefinition::getTriangulationData,
                      &MeshDefinition::setTriangulationData)
        .def_property_readonly("types", &MeshDefinition::getTypes)
#ifdef ENABLE_MPI
        .def("setTriangulation", &MeshDefinition::setTriangulationData)
#endif
        ;
    }

    } // end namespace detail

    } // end namespace hoomd
