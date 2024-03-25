# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""MPI communicator.

When compiled without MPI support, `Communicator` acts as if there is one MPI
rank and 1 partition. To use MPI, :doc:`compile HOOMD-blue <building>` with the
option ``ENABLE_MPI=on`` and use the appropriate MPI launcher to launch Python.
Then the `Communicator` class will configure and query MPI ranks and partitions.
By default, `Communicator` starts with the ``MPI_COMM_WOLRD`` MPI communicator,
and the communicator is not available for user scripts.

`Communicator` also accepts MPI communicators from ``mpi4py``. Use this to
implement workflows with multiple simulations that communicate using ``mpi4py``
calls in user code (e.g. genetic algorithms, umbrella sampling).

See Also:
    :doc:`tutorial/03-Parallel-Simulations-With-MPI/00-index`

    :doc:`tutorial/05-Organizing-and-Executing-Simulations/00-index`

.. invisible-code-block: python

    simulation = hoomd.util.make_example_simulation()
"""

from hoomd import _hoomd
import hoomd

import contextlib


class Communicator(object):
    """MPI communicator.

    Args:
        mpi_comm: Accepts an mpi4py communicator. Use this argument to perform
          many independent hoomd simulations where you communicate between those
          simulations using mpi4py.
        ranks_per_partition (int): (MPI) Number of ranks to include in a
          partition.


    The `Communicator` class initializes MPI communications for a
    `hoomd.Simulation` and exposes rank and partition information to the user as
    properties. To use MPI, launch your Python script with an MPI launcher (e.g.
    ``mpirun`` or ``mpiexec``). By default, `Communicator` uses all ranks
    provided by the launcher ``num_launch_ranks`` for a single
    `hoomd.Simulation` object which decomposes the state onto that many domains.

    Set ``ranks_per_partition`` to an integer to partition launched ranks into
    ``num_launch_ranks / ranks_per_partition`` communicators, each with their
    own `partition` index. Use this to perform many simulations in parallel, for
    example by using `partition` as an index into an array of state points to
    execute.

    .. rubric:: Examples:

    .. code-block:: python

        communicator = hoomd.communicator.Communicator()

    .. code-block:: python

        communicator = simulation.device.communicator
    """

    def __init__(self, mpi_comm=None, ranks_per_partition=None):

        # check ranks_per_partition
        if ranks_per_partition is not None:
            if not hoomd.version.mpi_enabled:
                raise RuntimeError(
                    "The ranks_per_partition option is only available in MPI.\n"
                )

        mpi_available = hoomd.version.mpi_enabled

        self.cpp_mpi_conf = None

        # create the specified configuration
        if mpi_comm is None:
            self.cpp_mpi_conf = _hoomd.MPIConfiguration()
        else:
            if not mpi_available:
                raise RuntimeError("mpi_comm is not supported in serial builds")

            handled = False

            # pass in pointer to MPI_Comm object provided by mpi4py
            try:
                import mpi4py
                if isinstance(mpi_comm, mpi4py.MPI.Comm):
                    addr = mpi4py.MPI._addressof(mpi_comm)
                    self.cpp_mpi_conf = \
                        _hoomd.MPIConfiguration._make_mpi_conf_mpi_comm(addr)
                    handled = True
            except ImportError:
                # silently ignore when mpi4py is missing
                pass

            # undocumented case: handle plain integers as pointers to MPI_Comm
            # objects
            if not handled and isinstance(mpi_comm, int):
                self.cpp_mpi_conf = \
                    _hoomd.MPIConfiguration._make_mpi_conf_mpi_comm(mpi_comm)
                handled = True

            if not handled:
                raise RuntimeError(
                    "Invalid mpi_comm object: {}".format(mpi_comm))

        if ranks_per_partition is not None:
            # check validity
            if (self.cpp_mpi_conf.getNRanksGlobal() % ranks_per_partition):
                raise RuntimeError('Total number of ranks is not a multiple of '
                                   'ranks_per_partition.')

            # split the communicator into partitions
            self.cpp_mpi_conf.splitPartitions(ranks_per_partition)

    @property
    def num_ranks(self):
        """int: The number of ranks in this partition.

        When initialized with ``ranks_per_partition=None``, `num_ranks` is equal
        to the ``num_launch_ranks`` set by the MPI launcher. When using
        partitions, `num_ranks` is equal to ``ranks_per_partition``.

        Note:
            Returns 1 in builds with ENABLE_MPI=off.

        .. rubric:: Example:

        .. code-block:: python

            num_ranks = communicator.num_ranks
        """
        if hoomd.version.mpi_enabled:
            return self.cpp_mpi_conf.getNRanks()
        else:
            return 1

    @property
    def rank(self):
        """int: The current rank within the partition.

        Note:
            Returns 0 in builds with ENABLE_MPI=off.

        .. rubric:: Example:

        .. code-block:: python

            rank = communicator.rank
        """
        if hoomd.version.mpi_enabled:
            return self.cpp_mpi_conf.getRank()
        else:
            return 0

    @property
    def num_partitions(self):
        """int: The number of partitions in this execution.

        Create partitions with the ``ranks_per_partition`` argument on
        initialization. Then, the number of partitions is
        ``num_launch_ranks / ranks_per_partition``.

        Note:
            Returns 1 in builds with ENABLE_MPI=off.

        .. rubric:: Example:

        .. code-block:: python

            num_partitions = communicator.num_partitions
        """
        if hoomd.version.mpi_enabled:
            return self.cpp_mpi_conf.getNPartitions()
        else:
            return 1

    @property
    def partition(self):
        """int: The current partition.

        Note:
            Returns 0 in builds with ENABLE_MPI=off.

        .. rubric:: Example:

        .. code-block:: python

            partition = communicator.partition
        """
        if hoomd.version.mpi_enabled:
            return self.cpp_mpi_conf.getPartition()
        else:
            return 0

    def barrier_all(self):
        """Perform a MPI barrier synchronization across all ranks.

        Note:
            Does nothing in builds with ENABLE_MPI=off.

        .. rubric:: Example:

        .. code-block:: python

            communicator.barrier_all()
        """
        if hoomd.version.mpi_enabled:
            _hoomd.mpi_barrier_world()

    def barrier(self):
        """Perform a barrier synchronization across all ranks in the partition.

        Note:
            Does nothing in builds with ENABLE_MPI=off.

        .. rubric:: Example:

        .. code-block:: python

            communicator.barrier()
        """
        if hoomd.version.mpi_enabled:
            self.cpp_mpi_conf.barrier()

    @contextlib.contextmanager
    def localize_abort(self):
        """Localize MPI_Abort to this partition.

        HOOMD calls ``MPI_Abort`` to tear down all running MPI processes
        whenever there is an uncaught exception. By default, this will abort the
        entire MPI execution. When using partitions, an uncaught exception on
        one partition will therefore abort all of them.

        Use the return value of :py:meth:`localize_abort()` as a context manager
        to tell HOOMD that all operations within the context will use only
        that MPI communicator so that an uncaught exception in one partition
        will only abort that partition and leave the others running.

        .. rubric:: Example:

        .. code-block:: python

            with communicator.localize_abort():
                simulation.run(1_000)
        """
        global _current_communicator
        prev = _current_communicator

        _current_communicator = self
        yield None
        _current_communicator = prev

    @property
    def walltime(self):
        """Wall clock time since creating the `Communicator` [seconds].

        `walltime` returns the same value on each rank in the current partition.

        .. rubric:: Example:

        .. code-block:: python

            walltime = communicator.walltime
        """
        return self.cpp_mpi_conf.getWalltime()


# store the "current" communicator to be used for MPI_Abort calls. This defaults
# to the world communicator, but users can opt in to a more specific
# communicator using the Device.localize_abort context manager
_current_communicator = Communicator()
