// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

/*! \file SnapshotSystemData.cc
    \brief Implements SnapshotSystemData related functions
 */

#include "SnapshotSystemData.h"
#include <pybind11/pybind11.h>
namespace py = pybind11;

template <class Real>
void SnapshotSystemData<Real>::replicate(unsigned int nx, unsigned int ny, unsigned int nz)
    {
    assert(nx > 0);
    assert(ny > 0);
    assert(nz > 0);

    // Update global box
    BoxDim old_box = global_box;
    Scalar3 L = global_box.getL();
    L.x *= (Scalar) nx;
    L.y *= (Scalar) ny;
    L.z *= (Scalar) nz;
    global_box.setL(L);

    unsigned int old_n = particle_data.size;
    unsigned int n = nx * ny *nz;

    // replicate snapshots
    particle_data.replicate(nx, ny, nz, old_box, global_box);
    bond_data.replicate(n,old_n);
    angle_data.replicate(n,old_n);
    dihedral_data.replicate(n,old_n);
    improper_data.replicate(n,old_n);
    constraint_data.replicate(n,old_n);
    pair_data.replicate(n,old_n);
    }

template <class Real>
void SnapshotSystemData<Real>::broadcast_box(std::shared_ptr<MPIConfiguration> mpi_conf)
    {
    #ifdef ENABLE_MPI
    if (mpi_conf->getNRanks() > 1)
        {
        bcast(global_box, 0, mpi_conf->getCommunicator());
        bcast(dimensions, 0, mpi_conf->getCommunicator());
        }
    #endif
    }

template <class Real>
void SnapshotSystemData<Real>::broadcast(unsigned int root, std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    #ifdef ENABLE_MPI
    if (exec_conf->getNRanks() > 1)
        {
        bcast(global_box, root, exec_conf->getMPICommunicator());
        bcast(dimensions, root, exec_conf->getMPICommunicator());

        particle_data.bcast(root, exec_conf->getMPICommunicator());
        bcast(map, root, exec_conf->getMPICommunicator());
        bond_data.bcast(root, exec_conf->getMPICommunicator());
        angle_data.bcast(root, exec_conf->getMPICommunicator());
        dihedral_data.bcast(root, exec_conf->getMPICommunicator());
        improper_data.bcast(root, exec_conf->getMPICommunicator());
        constraint_data.bcast(root, exec_conf->getMPICommunicator());
        pair_data.bcast(root, exec_conf->getMPICommunicator());
        }
    #endif
    }

template <class Real>
void SnapshotSystemData<Real>::broadcast_all(unsigned int root, std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    #ifdef ENABLE_MPI
    MPI_Comm hoomd_world = exec_conf->getHOOMDWorldMPICommunicator();
    int n_ranks;
    MPI_Comm_size(hoomd_world, &n_ranks);
    if (n_ranks > 0)
        {
        bcast(global_box, root, hoomd_world);
        bcast(dimensions, root, hoomd_world);

        particle_data.bcast(root, hoomd_world);
        bcast(map, root, hoomd_world);

        bond_data.bcast(root, hoomd_world);
        angle_data.bcast(root, hoomd_world);
        dihedral_data.bcast(root, hoomd_world);
        improper_data.bcast(root, hoomd_world);
        constraint_data.bcast(root, hoomd_world);
        pair_data.bcast(root, hoomd_world);
        }
    #endif
    }

// instantiate both float and double snapshots
template struct PYBIND11_EXPORT SnapshotSystemData<float>;
template struct PYBIND11_EXPORT SnapshotSystemData<double>;

void export_SnapshotSystemData(py::module& m)
    {
    py::class_<SnapshotSystemData<float>, std::shared_ptr< SnapshotSystemData<float> > >(m,"SnapshotSystemData_float")
    .def(py::init<>())
    .def_readwrite("_dimensions", &SnapshotSystemData<float>::dimensions)
    .def_readwrite("_global_box", &SnapshotSystemData<float>::global_box)
    .def_readonly("particles", &SnapshotSystemData<float>::particle_data)
    .def_readonly("bonds", &SnapshotSystemData<float>::bond_data)
    .def_readonly("angles", &SnapshotSystemData<float>::angle_data)
    .def_readonly("dihedrals", &SnapshotSystemData<float>::dihedral_data)
    .def_readonly("impropers", &SnapshotSystemData<float>::improper_data)
    .def_readonly("constraints", &SnapshotSystemData<float>::constraint_data)
    .def_readonly("pairs", &SnapshotSystemData<float>::pair_data)
    .def("replicate", &SnapshotSystemData<float>::replicate)
    .def("_broadcast_box", &SnapshotSystemData<float>::broadcast_box)
    .def("_broadcast", &SnapshotSystemData<float>::broadcast)
    .def("_broadcast_all", &SnapshotSystemData<float>::broadcast_all)
    ;

    py::class_<SnapshotSystemData<double>, std::shared_ptr< SnapshotSystemData<double> > >(m,"SnapshotSystemData_double")
    .def(py::init<>())
    .def_readwrite("_dimensions", &SnapshotSystemData<double>::dimensions)
    .def_readwrite("_global_box", &SnapshotSystemData<double>::global_box)
    .def_readonly("particles", &SnapshotSystemData<double>::particle_data)
    .def_readonly("bonds", &SnapshotSystemData<double>::bond_data)
    .def_readonly("angles", &SnapshotSystemData<double>::angle_data)
    .def_readonly("dihedrals", &SnapshotSystemData<double>::dihedral_data)
    .def_readonly("impropers", &SnapshotSystemData<double>::improper_data)
    .def_readonly("constraints", &SnapshotSystemData<double>::constraint_data)
    .def_readonly("pairs", &SnapshotSystemData<double>::pair_data)
    .def("replicate", &SnapshotSystemData<double>::replicate)
    .def("_broadcast_box", &SnapshotSystemData<double>::broadcast_box)
    .def("_broadcast", &SnapshotSystemData<double>::broadcast)
    .def("_broadcast_all", &SnapshotSystemData<double>::broadcast_all)
    ;
    }
