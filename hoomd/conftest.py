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
