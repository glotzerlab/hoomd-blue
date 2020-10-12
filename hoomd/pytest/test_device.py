import hoomd
import pytest

@pytest.mark.gpu
def test_gpu_profile(device):

    print(device)

    with device.enable_profiling():
        pass


def _assert_common_properties(dev, notice_level, msg_file, num_cpu_threads):
    """Assert the properties common to all devices are correct."""
    assert dev.notice_level == notice_level
    assert dev.msg_file == msg_file
    if hoomd.version.tbb_enabled:
        assert dev.num_cpu_threads == num_cpu_threads
    else:
        assert dev.num_cpu_threads == 1
    assert type(dev.communicator) == hoomd.comm.Communicator


def test_common_properties():
    # test default params
    dev = hoomd.device.auto_select()
    _assert_common_properties(dev, 2, None, 1)

    # make sure we can set those properties
    dev.notice_level = 3
    dev.msg_file = "example.txt"
    dev.num_cpu_threads = 5
    _assert_common_properties(dev, 3, "example.txt", 5)

    # now make a device with non-default arguments
    dev = hoomd.device.auto_select(msg_file="example2.txt", notice_level=10)
    _assert_common_properties(dev, notice_level=10, msg_file="example2.txt", num_cpu_threads=10)

    # shared_msg_file_stuff
    if hoomd.version.mpi_enabled:
        dev2 = hoomd.device.auto_select(shared_msg_file="shared.txt")
    else:
        with pytest.raises(RuntimeError):
            dev2 = hoomd.device.auto_select(shared_msg_file="shared.txt")


def _assert_gpu_properties(dev, mem_traceback, gpu_error_checking):
    """Assert properties specific to GPU objects are correct."""
    assert dev.memory_traceback == mem_traceback
    assert dev.gpu_error_checking == gpu_error_checking


@pytest.mark.gpu
def test_gpu_specific_properties(device):
    # assert the defaults are right
    _assert_gpu_properties(device, False, True)

    # make sure we can set the properties
    device.memory_traceback = True
    device.gpu_error_checking = False
    device.memory_traceback
    #_assert_gpu_properties(device, True, False)


def test_cpu_build_specifics():
    if hoomd.version.gpu_enabled == True:
        pytest.skip("Don't run CPU-build specific tests when GPU is available")
    assert hoomd.device.GPU.is_available() == False
    assert type(hoomd.device.auto_select()) == hoomd.device.CPU
