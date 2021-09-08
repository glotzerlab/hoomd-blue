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
    triangle_data
        = std::shared_ptr<TriangleData>(new TriangleData(pdata, n_triangle_types));
    m_meshtriangle_data
        = std::shared_ptr<MeshTriangleData>(new MeshTriangleData(pdata, n_triangle_types));
    }

/*! Evaluates the snapshot and initializes the respective *Data classes using
   its contents (box dimensions and sub-snapshots)
    \param snapshot Snapshot to use
    \param exec_conf Execution configuration to run on
    \param decomposition (optional) The domain decomposition layout
*/
template<class Real>
MeshData::MeshData(std::shared_ptr<ParticleData> pdata,
		std::shared_ptr<SnapshotSystemData<Real>> snapshot)
    {

    triangle_data = std::shared_ptr<TriangleData>(new TriangleData(pdata, snapshot->triangle_data));

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
    triangle_data->takeSnapshot(snap->triangle_data);
    }

//! Re-initialize the system from a snapshot
template<class Real>
void MeshData::initializeFromSnapshot(std::shared_ptr<SnapshotSystemData<Real>> snapshot)
    {

    triangle_data->initializeFromSnapshot(snapshot->triangle_data);
    }

// instantiate both float and double methods
template MeshData::MeshData(std::shared_ptr<ParticleData> pdata,
		                            std::shared_ptr<SnapshotSystemData<float>> snapshot);

template void MeshData::takeSnapshot<float>(std::shared_ptr<SnapshotSystemData<float>> snap);
template void MeshData::initializeFromSnapshot<float>(
    std::shared_ptr<SnapshotSystemData<float>> snapshot);

template MeshData::MeshData(std::shared_ptr<ParticleData> pdata,
		                            std::shared_ptr<SnapshotSystemData<double>> snapshot);

template void MeshData::takeSnapshot<double>(std::shared_ptr<SnapshotSystemData<double>> snap);
template void MeshData::initializeFromSnapshot<double>(
    std::shared_ptr<SnapshotSystemData<double>> snapshot);

void export_MeshData(py::module& m)
    {
    py::class_<MeshData, std::shared_ptr<MeshData>>(m, "MeshData")
        .def(py::init<std::shared_ptr<ParticleData>, unsigned int>())
        .def(py::init<std::shared_ptr<ParticleData>,
		      std::shared_ptr<SnapshotSystemData<float>>>())
        .def(py::init<std::shared_ptr<ParticleData>,
		      std::shared_ptr<SnapshotSystemData<double>>>())
        .def("takeSnapshot_float", &MeshData::takeSnapshot<float>)
        .def("takeSnapshot_double", &MeshData::takeSnapshot<double>)
        .def("initializeFromSnapshot", &MeshData::initializeFromSnapshot<float>)
        .def("initializeFromSnapshot", &MeshData::initializeFromSnapshot<double>)
        .def("getTriangleData", &MeshData::getTriangleData);
    }
