// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

/*! \file SnapshotSystemData.cc
    \brief Implements SnapshotSystemData related functions
 */

#include "SnapshotSystemData.h"
#include <boost/python.hpp>

using namespace boost::python;

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
    }

template <class Real>
void SnapshotSystemData<Real>::broadcast(boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    #ifdef ENABLE_MPI
    if (exec_conf->getNRanks() > 1)
        {
        bcast(global_box, 0, exec_conf->getMPICommunicator());
        }
    #endif
    }

// instantiate both float and double snapshots
template struct SnapshotSystemData<float>;
template struct SnapshotSystemData<double>;

void export_SnapshotSystemData()
    {
    class_<SnapshotSystemData<float>, boost::shared_ptr< SnapshotSystemData<float> > >("SnapshotSystemData_float")
    .def(init<>())
    .def_readwrite("_dimensions", &SnapshotSystemData<float>::dimensions)
    .def_readwrite("_global_box", &SnapshotSystemData<float>::global_box)
    .def_readwrite("particles", &SnapshotSystemData<float>::particle_data)
    .def_readwrite("bonds", &SnapshotSystemData<float>::bond_data)
    .def_readwrite("angles", &SnapshotSystemData<float>::angle_data)
    .def_readwrite("dihedrals", &SnapshotSystemData<float>::dihedral_data)
    .def_readwrite("impropers", &SnapshotSystemData<float>::improper_data)
    .def_readwrite("constraints", &SnapshotSystemData<float>::constraint_data)
    .def("replicate", &SnapshotSystemData<float>::replicate)
    .def("_broadcast", &SnapshotSystemData<float>::broadcast)
    ;

    implicitly_convertible<boost::shared_ptr< SnapshotSystemData<float> >, boost::shared_ptr< const SnapshotSystemData<float> > >();

    class_<SnapshotSystemData<double>, boost::shared_ptr< SnapshotSystemData<double> > >("SnapshotSystemData_double")
    .def(init<>())
    .def_readwrite("_dimensions", &SnapshotSystemData<double>::dimensions)
    .def_readwrite("_global_box", &SnapshotSystemData<double>::global_box)
    .def_readwrite("particles", &SnapshotSystemData<double>::particle_data)
    .def_readwrite("bonds", &SnapshotSystemData<double>::bond_data)
    .def_readwrite("angles", &SnapshotSystemData<double>::angle_data)
    .def_readwrite("dihedrals", &SnapshotSystemData<double>::dihedral_data)
    .def_readwrite("impropers", &SnapshotSystemData<double>::improper_data)
    .def_readwrite("constraints", &SnapshotSystemData<double>::constraint_data)
    .def("replicate", &SnapshotSystemData<double>::replicate)
    .def("_broadcast", &SnapshotSystemData<double>::broadcast)
    ;

    implicitly_convertible<boost::shared_ptr< SnapshotSystemData<double> >, boost::shared_ptr< const SnapshotSystemData<double> > >();
    }
