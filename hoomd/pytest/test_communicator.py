# Copyright (c) 2009-2025 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
import pytest
import time
import importlib.util

skip_mpi4py = (
    not hoomd.version.mpi_enabled or importlib.util.find_spec("mpi4py") is None
)
if not skip_mpi4py:
    from mpi4py import MPI
else:
    MPI = None

skip_mpi4py = pytest.mark.skipif(skip_mpi4py, reason="mpi4py could not be imported.")


def test_communicator_methods():
    """Ensure that communicator methods may be called.

    Ensuring that these methods act appropriately is not possible in an
    automated testing environment. These have been checked with manually
    run test scripts. Here, just test that the methods can be called.
    """
    communicator = hoomd.communicator.Communicator()

    communicator.barrier()
    communicator.barrier_all()
    with communicator.localize_abort():
        pass


def test_communicator_ranks():
    """Check that the ranks and num_ranks properties function."""
    communicator = hoomd.communicator.Communicator()
    assert communicator.num_ranks >= 1
    assert communicator.rank < communicator.num_ranks
    assert communicator.partition == 0

    if MPI is not None:
        mpi_communicator = MPI.COMM_WORLD
        assert communicator.rank == mpi_communicator.Get_rank()
        assert communicator.num_ranks == mpi_communicator.Get_size()


def test_communicator_partition():
    """Check that communicators can be partitioned."""
    world_communicator = hoomd.communicator.Communicator()

    if world_communicator.num_ranks == 2:
        communicator = hoomd.communicator.Communicator(ranks_per_partition=1)
        assert communicator.num_partitions == 2
        assert communicator.num_ranks == 1
        assert communicator.rank == 0
        assert communicator.partition < communicator.num_partitions

        if MPI is not None:
            mpi_communicator = MPI.COMM_WORLD
            assert communicator.partition == mpi_communicator.Get_rank()


def test_commuicator_walltime():
    """Check that Communicator.walltime functions."""
    ref_time = 1 / 16
    c = hoomd.communicator.Communicator()
    time.sleep(ref_time)
    t = c.walltime

    assert t >= ref_time


@skip_mpi4py
@pytest.mark.skipif(not hoomd.version.mpi_enabled, reason="This test requires MPI")
def test_communicator_mpi4py():
    """Check that Communicator can be initialized with mpi4py."""
    world_communicator = hoomd.communicator.Communicator()
    communicator = hoomd.communicator.Communicator(mpi_comm=MPI.COMM_WORLD)
    assert world_communicator.num_ranks == communicator.num_ranks
    assert world_communicator.rank == communicator.rank
