import pytest
import hoomd
import atexit
import numpy
import itertools
from hoomd.snapshot import Snapshot
from hoomd import Simulation

devices = [hoomd.device.CPU]
if hoomd.device.GPU.is_available():
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


@pytest.fixture(scope='session', params=devices)
def device_class(request):
    """Parameterized Device class fixture.

    Use the `device_class` fixture in tests that need to pass parameters to the
    device creation.
    """
    return request.param


@pytest.fixture(scope='session')
def device_cpu():
    """CPU only device fixture.

    Use this fixture when a test only executes on the CPU.

    TODO: This might be better implemented as a skip on the GPU fixture, like
    skip_mpi... Then the device fixture would work well with the factories
    below even for CPU only tests. Same goes for device_gpu.
    """
    return hoomd.device.CPU()


@pytest.fixture(scope='session')
def device_gpu():
    if hoomd.device.GPU.is_available():
        return hoomd.device.GPU()
    else:
        pytest.skip("GPU support not available")


@pytest.fixture(scope='session')
def simulation_factory(device):
    """Make a Simulation object from a snapshot.

    TODO: duck type this to allow it to create state from GSD files as well
    """

    def make_simulation(snapshot):
        sim = Simulation(device)

        # reduce sorter grid to avoid Hilbert curve overhead in unit tests
        for tuner in sim.operations.tuners:
            if isinstance(tuner, hoomd.tune.ParticleSorter):
                tuner.grid = 8

        sim.create_state_from_snapshot(snapshot)
        return sim

    return make_simulation


@pytest.fixture(scope='session')
def two_particle_snapshot_factory(device):
    """Make a snapshot with two particles.

    Args:
        particle_types: List of particle type names
        dimensions: Number of dimensions (2 or 3)
        d: Distance apart to place particles
        L: Box length

    The two particles are placed at (-d/2, 0, 0) and (d/2,0,0). When,
    dimensions==3, the box is L by L by L. When dimensions==2, the box is L by L
    by 1.
    """

    def make_snapshot(particle_types=['A'], dimensions=3, d=1, L=20):
        s = Snapshot(device.comm)
        N = 2

        if s.exists:
            box = [L, L, L, 0, 0, 0]
            if dimensions == 2:
                box[2] = 1
            s.configuration.box = box
            s.configuration.dimensions = dimensions
            s.particles.N = N
            s.particles.position[:] = [[-d / 2, 0, 0], [d / 2, 0, 0]]
            s.particles.types = particle_types

        return s

    return make_snapshot


@pytest.fixture(scope='session')
def lattice_snapshot_factory(device):
    """Make a snapshot with particles on a cubic/square lattice.

    Args:
        particle_types: List of particle type names
        dimensions: Number of dimensions (2 or 3)
        a: Lattice constant
        n: Number of particles along each box edge
        r: Fraction of `a` to randomly perturb particles

    Place particles on a simple cubic (dimensions==3) or square (dimensions==2)
    lattice. The box is cubic (or square) with a side length of `n * a`.

    Set `r` to randomly perturb particles a small amount off their lattice
    positions. This is useful in MD simulation testing so that forces do not
    cancel out by symmetry.
    """

    def make_snapshot(particle_types=['A'], dimensions=3, a=1, n=7, r=0):
        s = Snapshot(device.comm)

        if s.exists:
            box = [n * a, n * a, n * a, 0, 0, 0]
            if dimensions == 2:
                box[2] = 1
            s.configuration.box = box
            s.configuration.dimensions = dimensions

            s.particles.N = n**dimensions
            s.particles.types = particle_types

            # create the lattice
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


@pytest.fixture(autouse=True)
def skip_mpi(request):
    if request.node.get_closest_marker('serial'):
        if 'device' in request.fixturenames:
            if request.getfixturevalue('device').comm.num_ranks > 1:
                pytest.skip('Test does not support MPI execution')
        else:
            raise ValueError('skip_mpi requires the *device* fixture')


@pytest.fixture(autouse=True)
def only_gpu(request):
    if request.node.get_closest_marker('gpu'):
        if 'device' in request.fixturenames:
            if request.getfixturevalue('device').mode != 'gpu':
                pytest.skip('Test is run on GPU(s).')
        else:
            raise ValueError('only_gpu requires the *device* fixture')


@pytest.fixture(scope='function', autouse=True)
def numpy_random_seed():
    """Seed the numpy random number generator.

    Automatically reset the numpy random seed at the start of each function
    for reproducible tests.
    """
    numpy.random.seed(42)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "serial: Tests that will not execute with more than 1 MPI process")
    config.addinivalue_line(
        "markers",
        "validation: Long running tests that validate simulation output")
    config.addinivalue_line(
        "markers",
        "gpu: Tests that should only run on the gpu.")
    config.addinivalue_line(
        "markers",
        "cupy_optional: tests that should pass with and without CuPy.")


def abort(exitstatus):
    # get a default mpi communicator
    communicator = hoomd.comm.Communicator()
    # abort the deadlocked ranks
    hoomd._hoomd.abort_mpi(communicator.cpp_mpi_conf, exitstatus)


def pytest_sessionfinish(session, exitstatus):
    """ Finalize pytest session

    MPI tests may fail on one rank but not others. To prevent deadlocks in these
    situations, this code calls ``MPI_Abort`` when pytest is exiting with a
    non-zero exit code. **pytest** should be run with the ``-x`` option so that
    it exits on the first error.
    """

    if exitstatus != 0 and hoomd._hoomd.is_MPI_available():
        atexit.register(abort, exitstatus)
