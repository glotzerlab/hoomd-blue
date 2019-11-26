import pytest
import hoomd
import atexit
import numpy
from hoomd.snapshot import Snapshot
from hoomd.simulation import Simulation

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
