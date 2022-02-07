# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
import pytest
import time


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


def test_communicator_ranks_with_mpi4py():
    """Check that the ranks are set correctly with mpi4py."""
    mpi4py = pytest.importorskip("mpi4py")
    mpi4py.MPI = pytest.importorskip("mpi4py.MPI")

    communicator = hoomd.communicator.Communicator()

    mpi_communicator = mpi4py.MPI.COMM_WORLD
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


def test_communicator_partition_with_mpi4py():
    """Check that communicators are partitioned correctly with mpi4py."""
    mpi4py = pytest.importorskip("mpi4py")
    mpi4py.MPI = pytest.importorskip("mpi4py.MPI")

    world_communicator = hoomd.communicator.Communicator()

    if world_communicator.num_ranks == 2:
        communicator = hoomd.communicator.Communicator(ranks_per_partition=1)

        mpi_communicator = mpi4py.MPI.COMM_WORLD
        assert communicator.partition == mpi_communicator.Get_rank()


def test_commuicator_walltime():
    """Check that Communicator.walltime functions."""
    ref_time = 1 / 16
    c = hoomd.communicator.Communicator()
    time.sleep(ref_time)
    t = c.walltime

    assert t >= ref_time


@pytest.mark.skipif(not hoomd.version.mpi_enabled,
                    reason='This test requires MPI')
def test_communicator_mpi4py():
    """Check that Communicator can be initialized with mpi4py."""
    mpi4py = pytest.importorskip("mpi4py")
    mpi4py.MPI = pytest.importorskip("mpi4py.MPI")

    world_communicator = hoomd.communicator.Communicator()
    communicator = hoomd.communicator.Communicator(
        mpi_comm=mpi4py.MPI.COMM_WORLD)
    assert world_communicator.num_ranks == communicator.num_ranks
    assert world_communicator.rank == communicator.rank
