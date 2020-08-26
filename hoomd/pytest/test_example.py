import hoomd
import pytest

""" This script demonstrates how to test HOOMD classes with pytest
"""

def test_typical(device):
    """ Typical hoomd test execute on both the CPU and GPU. This example test uses the ``device`` fixture to provide
    the execution device object. It will be called once with a CPU device and once with a GPU device. These device
    objects are re-used across all tests for efficiency, do not modify them.
    """

    print(device.mode)
    assert True

@pytest.mark.serial
def test_serial(device):
    """ Some tests will not run in MPI. Skip these with the `serial` mark.
    """
    assert True

@pytest.mark.cpu
def test_cpu_only(device):
    """ Some tests need a device but only operate correctly on the CPU.

    Use the ``cpu`` mark to skip these tests on the GPU.
    """

    assert device.mode == 'cpu'

@pytest.mark.gpu
def test_gpu_only(device):
    """ Some tests need a device but only operate correctly on the GPU.

    Use the ``gpu`` mark to skip these tests on the GPU.
    """

    assert device.mode == 'gpu'

def test_python_only():
    """ Python only tests operate in pure python without a device or simulation context
    """

    assert 2*2 == 4

def test_create_own_device(device_class):
    """ Some tests require that they instantiate their own device (i.e. need to specify MPI nrank options). Use the
    ``device_class`` fixture to run these on both the CPU and GPU. Use this fixture only when necessary due to the
    overhead of device creation.
    """

    device = device_class(notice_level = 1)
    assert device.mode == 'gpu' or device.mode == 'cpu'

