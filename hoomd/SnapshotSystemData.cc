// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

/*! \file SnapshotSystemData.cc
    \brief Implements SnapshotSystemData related functions
 */

#include "SnapshotSystemData.h"
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
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
    if (has_particle_data)
        particle_data.replicate(nx, ny, nz, old_box, global_box);
    if (has_bond_data)
        bond_data.replicate(n,old_n);
    if (has_angle_data)
        angle_data.replicate(n,old_n);
    if (has_dihedral_data)
        dihedral_data.replicate(n,old_n);
    if (has_improper_data)
        improper_data.replicate(n,old_n);
    if (has_constraint_data)
        constraint_data.replicate(n,old_n);
    if (has_pair_data)
        pair_data.replicate(n,old_n);
    }

template <class Real>
void SnapshotSystemData<Real>::broadcast_box(std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    #ifdef ENABLE_MPI
    if (exec_conf->getNRanks() > 1)
        {
        bcast(global_box, 0, exec_conf->getMPICommunicator());
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
        bcast(has_particle_data, root, exec_conf->getMPICommunicator());
        bcast(has_bond_data, root, exec_conf->getMPICommunicator());
        bcast(has_angle_data, root, exec_conf->getMPICommunicator());
        bcast(has_dihedral_data, root, exec_conf->getMPICommunicator());
        bcast(has_improper_data, root, exec_conf->getMPICommunicator());
        bcast(has_constraint_data, root, exec_conf->getMPICommunicator());
        bcast(has_pair_data, root, exec_conf->getMPICommunicator());
        bcast(has_integrator_data, root, exec_conf->getMPICommunicator());

        if (has_particle_data)
            {
            particle_data.bcast(root, exec_conf->getMPICommunicator());
            bcast(map, root, exec_conf->getMPICommunicator());
            }
        if (has_bond_data) bond_data.bcast(root, exec_conf->getMPICommunicator());
        if (has_angle_data) angle_data.bcast(root, exec_conf->getMPICommunicator());
        if (has_dihedral_data) dihedral_data.bcast(root, exec_conf->getMPICommunicator());
        if (has_improper_data) improper_data.bcast(root, exec_conf->getMPICommunicator());
        if (has_constraint_data) constraint_data.bcast(root, exec_conf->getMPICommunicator());
        if (has_pair_data) pair_data.bcast(root, exec_conf->getMPICommunicator());
        if (has_integrator_data) bcast(integrator_data, root, exec_conf->getMPICommunicator());
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
        bcast(has_particle_data, root, hoomd_world);
        bcast(has_bond_data, root, hoomd_world);
        bcast(has_angle_data, root, hoomd_world);
        bcast(has_dihedral_data, root, hoomd_world);
        bcast(has_improper_data, root, hoomd_world);
        bcast(has_constraint_data, root, hoomd_world);
        bcast(has_pair_data, root, hoomd_world);
        bcast(has_integrator_data, root, hoomd_world);

        if (has_particle_data)
            {
            particle_data.bcast(root, hoomd_world);
            bcast(map, root, hoomd_world);
            }
        if (has_bond_data) bond_data.bcast(root, hoomd_world);
        if (has_angle_data) angle_data.bcast(root, hoomd_world);
        if (has_dihedral_data) dihedral_data.bcast(root, hoomd_world);
        if (has_improper_data) improper_data.bcast(root, hoomd_world);
        if (has_constraint_data) constraint_data.bcast(root, hoomd_world);
        if (has_pair_data) pair_data.bcast(root, hoomd_world);
        if (has_integrator_data) bcast(integrator_data, root, hoomd_world);
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
    .def_readonly("has_particle_data", &SnapshotSystemData<float>::has_particle_data)
    .def_readonly("has_bond_data", &SnapshotSystemData<float>::has_bond_data)
    .def_readonly("has_angle_data", &SnapshotSystemData<float>::has_angle_data)
    .def_readonly("has_dihedral_data", &SnapshotSystemData<float>::has_dihedral_data)
    .def_readonly("has_improper_data", &SnapshotSystemData<float>::has_improper_data)
    .def_readonly("has_constraint_data", &SnapshotSystemData<float>::has_constraint_data)
    .def_readonly("has_pair_data", &SnapshotSystemData<float>::has_pair_data)
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
    .def_readonly("has_particle_data", &SnapshotSystemData<double>::has_particle_data)
    .def_readonly("has_bond_data", &SnapshotSystemData<double>::has_bond_data)
    .def_readonly("has_angle_data", &SnapshotSystemData<double>::has_angle_data)
    .def_readonly("has_dihedral_data", &SnapshotSystemData<double>::has_dihedral_data)
    .def_readonly("has_improper_data", &SnapshotSystemData<double>::has_improper_data)
    .def_readonly("has_constraint_data", &SnapshotSystemData<double>::has_constraint_data)
    .def_readonly("has_pair_data", &SnapshotSystemData<double>::has_pair_data)
    .def("replicate", &SnapshotSystemData<double>::replicate)
    .def("_broadcast_box", &SnapshotSystemData<double>::broadcast_box)
    .def("_broadcast", &SnapshotSystemData<double>::broadcast)
    .def("_broadcast_all", &SnapshotSystemData<double>::broadcast_all)
    ;
    }
