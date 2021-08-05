# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""Code to support unit and validation tests.

``conftest`` is not part of HOOMD-blue's public API.
"""

import pickle
import pytest
import hoomd
import atexit
import os
import numpy
import itertools
import math
import warnings
from hoomd.snapshot import Snapshot
from hoomd import Simulation

pytest_plugins = ("hoomd.pytest_plugin_validate",)

devices = [hoomd.device.CPU]
if (hoomd.device.GPU.is_available()
        and len(hoomd.device.GPU.get_available_devices()) > 0):

    if os.environ.get('_HOOMD_SKIP_CPU_TESTS_WHEN_GPUS_PRESENT_') is not None:
        devices.pop(0)

    devices.append(hoomd.device.GPU)


@pytest.fixture(scope='session', params=devices)
def device(request):
    """Parameterized Device fixture.

    Tests that use `device` will run once on the CPU and once on the GPU. The
    device object is session scoped to avoid device creation overhead when
    running tests.
    """
    d = request.param()

    # enable GPU error checking
    if isinstance(d, hoomd.device.GPU):
        d.gpu_error_checking = True

    return d


@pytest.fixture(scope='session')
def simulation_factory(device):
    """Make a Simulation object from a snapshot.

    TODO: duck type this to allow it to create state from GSD files as well
    """

    def make_simulation(snapshot=None, domain_decomposition=None):
        sim = Simulation(device)

        # reduce sorter grid to avoid Hilbert curve overhead in unit tests
        for tuner in sim.operations.tuners:
            if isinstance(tuner, hoomd.tune.ParticleSorter):
                tuner.grid = 8

        if (snapshot is not None):
            if domain_decomposition is None:
                sim.create_state_from_snapshot(snapshot)
            else:
                sim.create_state_from_snapshot(snapshot, domain_decomposition)
        return sim

    return make_simulation


@pytest.fixture(scope='session')
def two_particle_snapshot_factory(device):
    """Make a snapshot with two particles."""

    def make_snapshot(particle_types=['A'], dimensions=3, d=1, L=20):
        """Make the snapshot.

        Args:
            particle_types: List of particle type names
            dimensions: Number of dimensions (2 or 3)
            d: Distance apart to place particles
            L: Box length

        The two particles are placed at (-d/2, 0, 0) and (d/2,0,0). When,
        dimensions==3, the box is L by L by L. When dimensions==2, the box is
        L by L by 0.
        """
        s = Snapshot(device.communicator)
        N = 2

        if s.communicator.rank == 0:
            box = [L, L, L, 0, 0, 0]
            if dimensions == 2:
                box[2] = 0
            s.configuration.box = box
            s.particles.N = N
            # shift particle positions slightly in z so MPI tests pass
            s.particles.position[:] = [[-d / 2, 0, .1], [d / 2, 0, .1]]
            s.particles.types = particle_types
            if dimensions == 2:
                box[2] = 0
                s.particles.position[:] = [[-d / 2, 0.1, 0], [d / 2, 0.1, 0]]

        return s

    return make_snapshot


@pytest.fixture(scope='session')
def lattice_snapshot_factory(device):
    """Make a snapshot with particles on a cubic/square lattice."""

    def make_snapshot(particle_types=['A'], dimensions=3, a=1, n=7, r=0):
        """Make the snapshot.

        Args:
            particle_types: List of particle type names
            dimensions: Number of dimensions (2 or 3)
            a: Lattice constant
            n: Number of particles along each box edge
            r: Fraction of `a` to randomly perturb particles

        Place particles on a simple cubic (dimensions==3) or square
        (dimensions==2) lattice. The box is cubic (or square) with a side length
        of `n * a`.

        Set `r` to randomly perturb particles a small amount off their lattice
        positions. This is useful in MD simulation testing so that forces do not
        cancel out by symmetry.
        """
        s = Snapshot(device.communicator)

        if s.communicator.rank == 0:
            box = [n * a, n * a, n * a, 0, 0, 0]
            if dimensions == 2:
                box[2] = 0
            s.configuration.box = box

            s.particles.N = n**dimensions
            s.particles.types = particle_types

            # create the lattice
            if n > 0:
                range_ = numpy.arange(-n / 2, n / 2)
                if dimensions == 2:
                    pos = list(itertools.product(range_, range_, [0]))
                else:
                    pos = list(itertools.product(range_, repeat=3))
                pos = numpy.array(pos) * a
                pos[:, 0] += a / 2
                pos[:, 1] += a / 2
                if dimensions == 3:
                    pos[:, 2] += a / 2

                # perturb the positions
                if r > 0:
                    shift = numpy.random.uniform(-r, r, size=(s.particles.N, 3))
                    if dimensions == 2:
                        shift[:, 2] = 0
                    pos += shift

                s.particles.position[:] = pos

        return s

    return make_snapshot


@pytest.fixture(scope='session')
def fcc_snapshot_factory(device):
    """Make a snapshot with particles in a fcc structure."""

    def make_snapshot(particle_types=['A'], a=1, n=7, r=0):
        """Make a snapshot with particles in a fcc structure.

        Args:
            particle_types: List of particle type names
            a: Lattice constant
            n: Number of unit cells along each box edge
            r: Amount to randomly perturb particles in x,y,z

        Place particles in a fcc structure. The box is cubic with a side length
        of ``n * a``. There will be ``4 * n**3`` particles in the snapshot.
        """
        s = Snapshot(device.communicator)

        if s.communicator.rank == 0:
            # make one unit cell
            s.configuration.box = [a, a, a, 0, 0, 0]
            s.particles.N = 4
            s.particles.types = particle_types
            s.particles.position[:] = [
                [0, 0, 0],
                [0, a / 2, a / 2],
                [a / 2, 0, a / 2],
                [a / 2, a / 2, 0],
            ]
            # and replicate it
            s.replicate(n, n, n)

        # perturb the positions
        if r > 0:
            shift = numpy.random.uniform(-r, r, size=(s.particles.N, 3))
            s.particles.position[:] += shift

        return s

    return make_snapshot


@pytest.fixture(autouse=True)
def skip_mpi(request):
    """Skip tests marked ``serial`` when running with MPI."""
    if request.node.get_closest_marker('serial'):
        if 'device' in request.fixturenames:
            if request.getfixturevalue('device').communicator.num_ranks > 1:
                pytest.skip('Test does not support MPI execution')
        else:
            raise ValueError('skip_mpi requires the *device* fixture')


@pytest.fixture(autouse=True)
def only_gpu(request):
    """Skip CPU tests marked ``gpu``."""
    if request.node.get_closest_marker('gpu'):
        if 'device' in request.fixturenames:
            if not isinstance(request.getfixturevalue('device'),
                              hoomd.device.GPU):
                pytest.skip('Test is run only on GPU(s).')
        else:
            raise ValueError('only_gpu requires the *device* fixture')


@pytest.fixture(autouse=True)
def only_cpu(request):
    """Skip GPU tests marked ``cpu``."""
    if request.node.get_closest_marker('cpu'):
        if 'device' in request.fixturenames:
            if not isinstance(request.getfixturevalue('device'),
                              hoomd.device.CPU):
                pytest.skip('Test is run only on CPU(s).')
        else:
            raise ValueError('only_cpu requires the *device* fixture')


@pytest.fixture(scope='function', autouse=True)
def numpy_random_seed():
    """Seed the numpy random number generator.

    Automatically reset the numpy random seed at the start of each function
    for reproducible tests.
    """
    numpy.random.seed(42)


def pytest_configure(config):
    """Add markers to pytest configuration."""
    config.addinivalue_line(
        "markers",
        "serial: Tests that will not execute with more than 1 MPI process")
    config.addinivalue_line("markers",
                            "gpu: Tests that should only run on the gpu.")
    config.addinivalue_line(
        "markers",
        "cupy_optional: tests that should pass with and without CuPy.")
    config.addinivalue_line("markers", "cpu: Tests that only run on the CPU.")
    config.addinivalue_line("markers", "gpu: Tests that only run on the GPU.")


def abort(exitstatus):
    """Call MPI_Abort when pytest tests fail."""
    # get a default mpi communicator
    communicator = hoomd.communicator.Communicator()
    # abort the deadlocked ranks
    hoomd._hoomd.abort_mpi(communicator.cpp_mpi_conf, exitstatus)


def pytest_sessionfinish(session, exitstatus):
    """Finalize pytest session.

    MPI tests may fail on one rank but not others. To prevent deadlocks in these
    situations, this code calls ``MPI_Abort`` when pytest is exiting with a
    non-zero exit code. **pytest** should be run with the ``-x`` option so that
    it exits on the first error.
    """
    if exitstatus != 0 and hoomd.version.mpi_enabled:
        atexit.register(abort, exitstatus)


def logging_check(cls, expected_namespace, expected_loggables):
    """Function for testing object logging specification.

    Args:
        cls (object): The loggable class to test for the correct logging
            specfication.
        expected_namespace (tuple[str]): A tuple of strings that indicate the
            expected namespace minus the class name.
        expected_loggables (dict[str, dict[str, Any]]): A dict with string keys
            representing the expected loggable quantities. If the value for a
            key is ``None`` then, only check for the existence of the loggable
            quantity. Otherwise, the inner `dict` should consist of some
            combination of the keys ``default`` and ``category`` indicating the
            expected value of each for the loggable.
    """
    # Check namespace
    assert all(log_quantity.namespace == expected_namespace + (cls.__name__,)
               for log_quantity in cls._export_dict.values())

    # Check specific loggables
    def check_loggable(cls, name, properties):
        assert name in cls._export_dict
        if properties is None:
            return None
        log_quantity = cls._export_dict[name]
        for name, prop in properties.items():
            assert getattr(log_quantity, name) == prop

    for name, properties in expected_loggables.items():
        check_loggable(cls, name, properties)


def pickling_check(instance):
    """Test that an instance can be pickled and unpickled."""
    pkled_instance = pickle.loads(pickle.dumps(instance))
    assert instance == pkled_instance


def operation_pickling_check(instance, sim):
    """Test that an operation can be pickled and unpickled."""
    pickling_check(instance)
    sim.operations += instance
    sim.run(0)
    pickling_check(instance)


class BlockAverage:
    """Block average method for estimating standard deviation of the mean.

    Args:
        data: List of values
    """

    def __init__(self, data):
        # round down to the nearest power of 2
        N = 2**int(math.log(len(data)) / math.log(2))
        if N != len(data):
            warnings.warn(
                "Ignoring some data. Data array should be a power of 2.")

        block_sizes = []
        block_mean = []
        block_variance = []

        # take means of blocks and the mean/variance of all blocks, growing
        # blocks by factors of 2
        block_size = 1
        while block_size <= N // 8:
            num_blocks = N // block_size
            block_data = numpy.zeros(num_blocks)

            for i in range(0, num_blocks):
                start = i * block_size
                end = start + block_size
                block_data[i] = numpy.mean(data[start:end])

            block_mean.append(numpy.mean(block_data))
            block_variance.append(numpy.var(block_data) / (num_blocks - 1))

            block_sizes.append(block_size)
            block_size *= 2

        self._block_mean = numpy.array(block_mean)
        self._block_variance = numpy.array(block_variance)
        self._block_sizes = numpy.array(block_sizes)
        self.data = numpy.array(data)

        # check for a plateau in the relative error before the last data point
        block_relative_error = numpy.sqrt(self._block_variance) / numpy.fabs(
            self._block_mean)
        relative_error_derivative = (numpy.diff(block_relative_error)
                                     / numpy.diff(self._block_sizes))
        if numpy.all(relative_error_derivative > 0):
            warnings.warn("Block averaging failed to plateau, run longer")

    def get_hierarchical_errors(self):
        """Get details on the hierarchical errors."""
        return (self._block_sizes, self._block_mean, self._block_variance)

    @property
    def standard_deviation(self):
        """float: The error estimate on the mean."""
        if numpy.all(self.data == self.data[0]):
            return 0

        return numpy.sqrt(numpy.max(self._block_variance))

    @property
    def mean(self):
        """float: The mean."""
        return self._block_mean[-1]

    @property
    def relative_error(self):
        """float: The relative error."""
        return self.standard_deviation / numpy.fabs(self.mean)

    def assert_close(self,
                     reference_mean,
                     reference_deviation,
                     z=6,
                     max_relative_error=0.02):
        """Assert that the distribution is constent with a given reference.

        Also assert that the relative error of the distribution is small.
        Otherwise, test runs with massive fluctuations would likely lead to
        passing tests.

        Args:
            reference_mean: Known good mean value
            reference_deviation: Standard deviation of the known good value
            z: Number of standard deviations
            max_relative_error: Maximum relative error to allow
        """
        sample_mean = self.mean
        sample_deviation = self.standard_deviation

        assert sample_deviation / sample_mean <= max_relative_error

        # compare if 0 is within the confidence interval around the difference
        # of the means
        deviation_diff = ((sample_deviation**2
                           + reference_deviation**2)**(1 / 2.))
        mean_diff = math.fabs(sample_mean - reference_mean)
        deviation_allowed = z * deviation_diff
        assert mean_diff <= deviation_allowed


class ListWriter(hoomd.custom.Action):
    """Log a single quantity to a list.

    On each triggered timestep, access the given attribute and add the value
    to `data`.

    Args:
        operation: Operation to log
        attribute: Name of the attribute to log

    Attributes:
        data (list): Saved data
    """

    def __init__(self, operation, attribute):
        self._operation = operation
        self._attribute = attribute
        self.data = []

    def act(self, timestep):
        """Add the attribute value to the list."""
        self.data.append(getattr(self._operation, self._attribute))
