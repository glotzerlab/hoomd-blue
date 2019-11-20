import pytest
import hoomd

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

@pytest.fixture(autouse=True)
def skip_mpi(request, device):
    if request.node.get_closest_marker('serial'):
        if device.comm.num_ranks > 1:
            pytest.skip('Test does not support MPI execution')


def pytest_configure(config):
    config.addinivalue_line("markers", "serial: Tests that will not execute with more than 1 MPI process")
    config.addinivalue_line("markers", "validation: Long running tests that validate simulation output")
