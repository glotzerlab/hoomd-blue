import pytest
import hoomd
import atexit
import numpy
from hoomd.snapshot import Snapshot
from hoomd.simulation import Simulation
import numpy as np

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

        center_distance = 0.5 * l - 1

        s = Snapshot(device.comm)
        if s.exists:
            s.configuration.box = box_list

            s.particles.N = 1
            for num_particles in n:
                s.particles.N *= num_particles

            if n == (2, 1):
                s.particles.position[0] = [0, 0, 0]
                s.particles.position[1] = [0, a, 0]
                s.particles.types = particle_types
            else:
                i = 0
                for x in numpy.linspace(-center_distance,
                                        center_distance,
                                        n[0]):
                    for y in numpy.linspace(-center_distance,
                                            center_distance,
                                            n[1]):
                        if dimensions == 3:
                            for z in numpy.linspace(-center_distance,
                                                    center_distance,
                                                    n[2]):
                                s.particles.position[i] = [x, y, z]
                                i += 1
                        elif dimensions == 2:
                            s.particles.position[i] = [x, y, 0]
                            i += 1

                s.particles.types = particle_types

        sim = Simulation(device)
        sim.create_state_from_snapshot(s)
        return sim
    return make_simulation


@pytest.fixture(autouse=True)
def skip_mpi(request, device):
    if request.node.get_closest_marker('serial'):
        if device.comm.num_ranks > 1:
            pytest.skip('Test does not support MPI execution')


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
