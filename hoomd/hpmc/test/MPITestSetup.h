// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


#ifdef ENABLE_MPI

#include <mpi.h>

//! Enable CUDA MPI if using MVAPICH2
char env_str[] = "MV2_USE_CUDA=1";

//! Fixture to setup and tear down MPI
struct MPISetup
    {
    //! Setup
    MPISetup()
        {
        int argc = boost::unit_test::framework::master_test_suite().argc;
        char **argv = boost::unit_test::framework::master_test_suite().argv;

        MPI_Init(&argc, &argv);
        }

    //! Cleanup
    ~MPISetup()
        {
        MPI_Finalize();
        }

    };

BOOST_GLOBAL_FIXTURE( MPISetup );

#endif //ENABLE_MPI
