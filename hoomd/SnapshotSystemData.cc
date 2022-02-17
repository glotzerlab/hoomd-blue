// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file SnapshotSystemData.cc
    \brief Implements SnapshotSystemData related functions
 */

#include "SnapshotSystemData.h"
#include <pybind11/pybind11.h>

namespace hoomd
    {
template<class Real>
void SnapshotSystemData<Real>::replicate(unsigned int nx, unsigned int ny, unsigned int nz)
    {
    assert(nx > 0);
    assert(ny > 0);
    assert(nz > 0);

    // Update global box
    BoxDim old_box = *global_box;
    Scalar3 L = old_box.getL();
    L.x *= (Scalar)nx;
    L.y *= (Scalar)ny;
    L.z *= (Scalar)nz;
    global_box->setL(L);

    unsigned int old_n = particle_data.size;
    unsigned int n = nx * ny * nz;

    // replicate snapshots
    particle_data.replicate(nx, ny, nz, old_box, *global_box);
    bond_data.replicate(n, old_n);
    angle_data.replicate(n, old_n);
    dihedral_data.replicate(n, old_n);
    improper_data.replicate(n, old_n);
    constraint_data.replicate(n, old_n);
    pair_data.replicate(n, old_n);
    }

template<class Real> void SnapshotSystemData<Real>::wrap()
    {
    for (unsigned int i = 0; i < particle_data.size; i++)
        {
        auto const frac = global_box->makeFraction(particle_data.pos[i]);
        auto modulus_positive
            = [](Real x) { return std::fmod(std::fmod(x, Real(1.0)) + Real(1.0), Real(1.0)); };
        auto const wrapped = vec3<Real>(modulus_positive(static_cast<Real>(frac.x)),
                                        modulus_positive(static_cast<Real>(frac.y)),
                                        modulus_positive(static_cast<Real>(frac.z)));
        particle_data.pos[i] = global_box->makeCoordinates(wrapped);
        auto const img = make_int3(static_cast<int>(std::floor(frac.x)),
                                   static_cast<int>(std::floor(frac.y)),
                                   static_cast<int>(std::floor(frac.z)));
        particle_data.image[i] += img;
        }
    }

template<class Real>
void SnapshotSystemData<Real>::broadcast_box(std::shared_ptr<MPIConfiguration> mpi_conf)
    {
#ifdef ENABLE_MPI
    if (mpi_conf->getNRanks() > 1)
        {
        auto box = *global_box;
        bcast(box, 0, mpi_conf->getCommunicator());
        global_box = std::make_shared<BoxDim>(box);
        bcast(dimensions, 0, mpi_conf->getCommunicator());
        }
#endif
    }

template<class Real>
void SnapshotSystemData<Real>::broadcast(unsigned int root,
                                         std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
#ifdef ENABLE_MPI
    auto communicator = exec_conf->getMPICommunicator();
    broadcast_box(exec_conf->getMPIConfig());
    if (exec_conf->getNRanks() > 1)
        {
        particle_data.bcast(root, communicator);
        bcast(map, root, communicator);
        bond_data.bcast(root, communicator);
        angle_data.bcast(root, communicator);
        dihedral_data.bcast(root, communicator);
        improper_data.bcast(root, communicator);
        constraint_data.bcast(root, communicator);
        pair_data.bcast(root, communicator);
        }
#endif
    }

template<class Real>
void SnapshotSystemData<Real>::broadcast_all(unsigned int root,
                                             std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
#ifdef ENABLE_MPI
    MPI_Comm hoomd_world = exec_conf->getHOOMDWorldMPICommunicator();
    int n_ranks;
    MPI_Comm_size(hoomd_world, &n_ranks);
    broadcast_box(exec_conf->getMPIConfig());
    if (n_ranks > 0)
        {
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

namespace detail
    {
void export_SnapshotSystemData(pybind11::module& m)
    {
    pybind11::class_<SnapshotSystemData<float>, std::shared_ptr<SnapshotSystemData<float>>>(
        m,
        "SnapshotSystemData_float")
        .def(pybind11::init<>())
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
        .def("wrap", &SnapshotSystemData<float>::wrap)
        .def("_broadcast_box", &SnapshotSystemData<float>::broadcast_box)
        .def("_broadcast", &SnapshotSystemData<float>::broadcast)
        .def("_broadcast_all", &SnapshotSystemData<float>::broadcast_all);

    pybind11::class_<SnapshotSystemData<double>, std::shared_ptr<SnapshotSystemData<double>>>(
        m,
        "SnapshotSystemData_double")
        .def(pybind11::init<>())
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
        .def("wrap", &SnapshotSystemData<double>::wrap)
        .def("_broadcast_box", &SnapshotSystemData<double>::broadcast_box)
        .def("_broadcast", &SnapshotSystemData<double>::broadcast)
        .def("_broadcast_all", &SnapshotSystemData<double>::broadcast_all);
    }

    } // end namespace detail

    } // end namespace hoomd
