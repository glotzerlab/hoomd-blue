// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: joaander

/*! \file MeshData.cc
    \brief Defines MeshData
*/

#include "MeshData.h"

#include "SnapshotSystemData.h"

#ifdef ENABLE_MPI
#include "Communicator.h"
#endif

namespace py = pybind11;

using namespace std;

/*! \param N Number of particles to allocate
    \param box Initial box particles are in
    \param n_types Number of particle types to set
    \param n_bond_types Number of bond types to create
    \param n_angle_types Number of angle types to create
    \param n_dihedral_types Number of dihedral types to create
    \param n_improper_types Number of improper types to create
    \param exec_conf The ExecutionConfiguration HOOMD is to be run on

    Creating MeshData with this constructor results in
     - ParticleData constructed with the arguments \a N, \a box, \a n_types, and \a exec_conf->
     - BondData constructed with the arguments \a n_bond_types
     - All other data structures are default constructed.
*/
MeshData::MeshData(std::shared_ptr<ParticleData> pdata,
                                   unsigned int n_triangle_types)
    {
    m_meshtriangle_data
        = std::shared_ptr<MeshTriangleData>(new MeshTriangleData(pdata, n_triangle_types));

    m_meshbond_data
        = std::shared_ptr<MeshBondData>(new MeshBondData(pdata, n_triangle_types));

    }

/*! Evaluates the snapshot and initializes the respective *Data classes using
   its contents (box dimensions and sub-snapshots)
    \param snapshot Snapshot to use
    \param exec_conf Execution configuration to run on
    \param decomposition (optional) The domain decomposition layout
*/
MeshData::MeshData(std::shared_ptr<ParticleData> pdata,
		TriangleData::Snapshot snapshot)
    {
    m_meshtriangle_data
        = std::shared_ptr<MeshTriangleData>(new MeshTriangleData(pdata, snapshot));

    m_meshbond_data
        = std::shared_ptr<MeshBondData>(new MeshBondData(pdata, snapshot));
    }


/*! \param particles True if particle data should be saved
 *  \param bonds True if bond data should be saved
 *  \param angles True if angle data should be saved
 *  \param dihedrals True if dihedral data should be saved
 *  \param impropers True if improper data should be saved
 *  \param constraints True if constraint data should be saved
 *  \param integrators True if integrator data should be saved
 *  \param pairs True if pair data should be saved
 */
template<class Real> void MeshData::takeSnapshot(std::shared_ptr<SnapshotSystemData<Real>> snap)
    {
    m_meshtriangle_data->takeSnapshot(snap->triangle_data);
    }

//! Re-initialize the system from a snapshot
void MeshData::initializeFromSnapshot( TriangleData::Snapshot snapshot)
    {
    m_meshtriangle_data->initializeFromSnapshot(snapshot);
    m_meshbond_data->initializeFromSnapshot(snapshot);

    }

std::shared_ptr<TriangleData> MeshData::getTriangleData()
        {
        std::shared_ptr<TriangleData> triangle_data;
        return triangle_data;
        }


template void MeshData::takeSnapshot<float>(std::shared_ptr<SnapshotSystemData<float>> snap);
template void MeshData::takeSnapshot<double>(std::shared_ptr<SnapshotSystemData<double>> snap);

void export_MeshData(py::module& m)
    {
    py::class_<MeshData, std::shared_ptr<MeshData>>(m, "MeshData")
        .def(py::init<std::shared_ptr<ParticleData>, unsigned int>())
        .def(py::init<std::shared_ptr<ParticleData>, TriangleData::Snapshot>())
        .def("takeSnapshot_float", &MeshData::takeSnapshot<float>)
        .def("takeSnapshot_double", &MeshData::takeSnapshot<double>)
        .def("initializeFromSnapshot", &MeshData::initializeFromSnapshot)
        .def("getTriangleData", &MeshData::getTriangleData)
        .def("getMeshTriangleData", &MeshData::getMeshTriangleData)
        .def("getMeshBondData", &MeshData::getMeshBondData);
    }
