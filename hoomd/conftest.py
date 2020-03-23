import pytest
import hoomd
import atexit
import numpy
import hoomd.hpmc
from hoomd.snapshot import Snapshot
from hoomd import Simulation


devices = [hoomd.device.CPU]
if hoomd.device.GPU.is_available():
    devices.append(hoomd.device.GPU)


@pytest.fixture(scope='session',
                params=devices)
def device(request):
    return request.param()


@pytest.fixture(scope='session',
                params=devices)
def device_class(request):
    return request.param


@pytest.fixture(scope='session')
def device_cpu():
    return hoomd.device.CPU()


@pytest.fixture(scope='session')
def device_gpu():
    if hoomd.device.GPU.is_available():
        return hoomd.device.GPU()
    else:
        pytest.skip("GPU support not available")


@pytest.fixture(scope='session')
def dummy_simulation_factory(device):
    def make_simulation(particle_types=['A']):
        s = Snapshot(device.comm)
        N = 10

        if s.exists:
            s.configuration.box = [20, 20, 20, 0, 0, 0]

            s.particles.N = N
            s.particles.position[:] = numpy.random.uniform(-10, 10, size=(N, 3))
            s.particles.types = particle_types

        sim = Simulation(device)

        # reduce sorter grid to avoid Hilbert curve overhead in unit tests
        for tuner in sim.operations.tuners:
            if isinstance(tuner, hoomd.tuner.ParticleSorter):
                tuner.grid = 8

        sim.create_state_from_snapshot(s)
        return sim
    return make_simulation


@pytest.fixture(scope='session')
def lattice_simulation_factory(device):
    def make_simulation(particle_types=['A'], dimensions=3, l=20, n=7, a=3):
        n_list = []
        if isinstance(n, list):
            n = tuple(n)
        elif not isinstance(n, tuple):
            for i in range(dimensions):
                n_list.append(n)
            n = tuple(n_list)

        box_list = [l, l, l, 0, 0, 0]
        bounds = []
        for n_val in n:
            bound = (n_val - 1) * a * 0.5
            bound_up = bound + (a / 1000)
            if bound == 0:
                bound_up = a
            bounds.append((bound, bound_up))
        s = Snapshot(device.comm)
        if s.exists:
            s.configuration.box = box_list

            s.particles.N = 1
            for num_particles in n:
                s.particles.N *= num_particles

            i = 0
            for x in numpy.arange(-bounds[0][0], bounds[0][1], a):
                for y in numpy.arange(-bounds[1][0], bounds[1][1], a):
                    if dimensions == 3:
                        for z in numpy.arange(-bounds[2][0], bounds[2][1], a):
                            s.particles.position[i] = [x, y, z]
                            i += 1
                    elif dimensions == 2:
                        s.particles.position[i] = [x, y, 0]
                        i += 1

                s.particles.types = particle_types

        sim = Simulation(device)

        # reduce sorter grid to avoid Hilbert curve overhead in unit tests
        for tuner in sim.operations.tuners:
            if isinstance(tuner, hoomd.tuner.ParticleSorter):
                tuner.grid = 8

        sim.create_state_from_snapshot(s)
        return sim
    return make_simulation


@pytest.fixture(autouse=True)
def skip_mpi(request):
    if request.node.get_closest_marker('serial'):
        if 'device' in request.fixturenames:
            if request.getfixturevalue('device').comm.num_ranks > 1:
                pytest.skip('Test does not support MPI execution')
        else:
            raise ValueError('skip_mpi requires the *device* fixture')


def pytest_configure(config):
    config.addinivalue_line("markers", "serial: Tests that will not execute with more than 1 MPI process")
    config.addinivalue_line("markers", "validation: Long running tests that validate simulation output")


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
