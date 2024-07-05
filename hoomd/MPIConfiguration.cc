// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "MPIConfiguration.h"

#ifdef ENABLE_MPI
#include "HOOMDMPI.h"
#endif

#include <iostream>
#include <sstream>
#include <stdexcept>

/*! \file MPIConfiguration.cc
    \brief Defines MPIConfiguration and related classes
*/

namespace hoomd
    {
//! Default constructor
MPIConfiguration::MPIConfiguration(
#ifdef ENABLE_MPI
    MPI_Comm hoomd_world
#endif
    )
    : m_rank(0), m_n_rank(1)
    {
#ifdef ENABLE_MPI
    m_mpi_comm = m_hoomd_world = hoomd_world;

    // use all ranks in a single partition
    int size;
    MPI_Comm_size(m_hoomd_world, &size);
    m_n_rank = size;

    int rank;
    MPI_Comm_rank(m_mpi_comm, &rank);
    m_rank = rank;
#endif
    }

void MPIConfiguration::splitPartitions(unsigned int nrank)
    {
#ifdef ENABLE_MPI
    int num_total_ranks;
    MPI_Comm_size(m_hoomd_world, &num_total_ranks);

    unsigned int partition = 0;
    m_n_rank = nrank;

    if (m_n_rank == 0)
        throw std::runtime_error("--nrank setting has to be > 0");

    int rank;
    MPI_Comm_rank(m_hoomd_world, &rank);

    if (num_total_ranks % m_n_rank != 0)
        throw std::runtime_error("Invalid setting --nrank.");

    partition = rank / m_n_rank;

    // Split the communicator
    MPI_Comm new_comm;
    MPI_Comm_split(m_hoomd_world, partition, rank, &new_comm);

    // update communicator
    m_mpi_comm = new_comm;

    MPI_Comm_rank(m_mpi_comm, &rank);
    m_rank = rank;
#endif
    }

unsigned int MPIConfiguration::getNRanks() const
    {
#ifdef ENABLE_MPI
    int size;
    MPI_Comm_size(m_mpi_comm, &size);
    return size;
#else
    return 1;
#endif
    }

namespace detail
    {
void export_MPIConfiguration(pybind11::module& m)
    {
    pybind11::class_<MPIConfiguration, std::shared_ptr<MPIConfiguration>> mpiconfiguration(
        m,
        "MPIConfiguration");
    mpiconfiguration.def(pybind11::init<>())
        .def("splitPartitions", &MPIConfiguration::splitPartitions)
        .def("getPartition", &MPIConfiguration::getPartition)
        .def("getNPartitions", &MPIConfiguration::getNPartitions)
        .def("getNRanks", &MPIConfiguration::getNRanks)
        .def("getRank", &MPIConfiguration::getRank)
        .def("barrier", &MPIConfiguration::barrier)
        .def("getNRanksGlobal", &MPIConfiguration::getNRanksGlobal)
        .def("getRankGlobal", &MPIConfiguration::getRankGlobal)
        .def("getWalltime", &MPIConfiguration::getWalltime)
#ifdef ENABLE_MPI
        .def_static("_make_mpi_conf_mpi_comm",
                    [](pybind11::object mpi_comm) -> std::shared_ptr<MPIConfiguration>
                    {
                        MPI_Comm* comm = (MPI_Comm*)PyLong_AsVoidPtr(mpi_comm.ptr());
                        return std::make_shared<MPIConfiguration>(*comm);
                    })
#endif
        ;
    }

    } // end namespace detail

    } // end namespace hoomd
