// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "MPIConfiguration.h"

#include "VectorMath.h"

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

        // create scalar2 data type for MPI
        {
        int blocklengths[] = {1, 1};
        MPI_Datatype types[] = {MPI_HOOMD_SCALAR, MPI_HOOMD_SCALAR};
        MPI_Aint offsets[] = {offsetof(Scalar2, x), offsetof(Scalar2, y)};

        MPI_Datatype tmp;
        MPI_Type_create_struct(2, blocklengths, offsets, types, &tmp);
        MPI_Type_create_resized(tmp, 0, sizeof(Scalar2), &m_mpi_scalar2);
        MPI_Type_commit(&m_mpi_scalar2);
        }

        // create scalar3 data type for MPI
        {
        int blocklengths[] = {1, 1, 1};
        MPI_Datatype types[] = {MPI_HOOMD_SCALAR, MPI_HOOMD_SCALAR, MPI_HOOMD_SCALAR};
        MPI_Aint offsets[] = {offsetof(Scalar3, x), offsetof(Scalar3, y), offsetof(Scalar3, z)};

        MPI_Datatype tmp;
        MPI_Type_create_struct(3, blocklengths, offsets, types, &tmp);
        MPI_Type_create_resized(tmp, 0, sizeof(Scalar3), &m_mpi_scalar3);
        MPI_Type_commit(&m_mpi_scalar3);
        }

        // create vec3<Scalar> data type for MPI
        {
        int blocklengths[] = {1, 1, 1};
        MPI_Datatype types[] = {MPI_HOOMD_SCALAR, MPI_HOOMD_SCALAR, MPI_HOOMD_SCALAR};
        MPI_Aint offsets[]
            = {offsetof(vec3<Scalar>, x), offsetof(vec3<Scalar>, y), offsetof(vec3<Scalar>, z)};

        MPI_Datatype tmp;
        MPI_Type_create_struct(3, blocklengths, offsets, types, &tmp);
        MPI_Type_create_resized(tmp, 0, sizeof(vec3<Scalar>), &m_mpi_vec3_scalar);
        MPI_Type_commit(&m_mpi_vec3_scalar);
        }

        // create scalar4 data type for MPI
        {
        int blocklengths[] = {1, 1, 1, 1};
        MPI_Datatype types[]
            = {MPI_HOOMD_SCALAR, MPI_HOOMD_SCALAR, MPI_HOOMD_SCALAR, MPI_HOOMD_SCALAR};
        MPI_Aint offsets[] = {offsetof(Scalar4, x),
                              offsetof(Scalar4, y),
                              offsetof(Scalar4, z),
                              offsetof(Scalar4, w)};

        MPI_Datatype tmp;
        MPI_Type_create_struct(4, blocklengths, offsets, types, &tmp);
        MPI_Type_create_resized(tmp, 0, sizeof(Scalar4), &m_mpi_scalar4);
        MPI_Type_commit(&m_mpi_scalar4);
        }
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

#ifdef ENABLE_MPI
MPI_Datatype MPIConfiguration::getScalar2Datatype() const
    {
    return m_mpi_scalar2;
    }

MPI_Datatype MPIConfiguration::getScalar3Datatype() const
    {
    return m_mpi_scalar3;
    }

MPI_Datatype MPIConfiguration::getVec3ScalarDatatype() const
    {
    return m_mpi_vec3_scalar;
    }

MPI_Datatype MPIConfiguration::getScalar4Datatype() const
    {
    return m_mpi_scalar4;
    }
#endif // ENABLE_MPI

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
