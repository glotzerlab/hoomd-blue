# Copyright (c) 2009-2025 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
import pytest


@pytest.mark.gpu
def test_gpu_profile(device):
    print(device)

    with device.enable_profiling():
        pass


def _assert_common_properties(dev, notice_level, message_filename):
    """Assert the properties common to all devices are correct."""
    assert dev.notice_level == notice_level
    assert dev.message_filename == message_filename
    assert type(dev.communicator) is hoomd.communicator.Communicator


def test_common_properties(device, tmp_path):
    # test default params, don't assert default
    _assert_common_properties(device, 2, None)

    # make sure we can set those properties
    device.notice_level = 3
    device.message_filename = str(tmp_path / "example.txt")
    _assert_common_properties(device, 3, str(tmp_path / "example.txt"))

    # now make a device with non-default arguments
    device_type = type(device)
    dev = device_type(message_filename=str(tmp_path / "example2.txt"), notice_level=10)
    _assert_common_properties(
        dev, notice_level=10, message_filename=str(tmp_path / "example2.txt")
    )


@pytest.mark.gpu
def test_gpu_specific_properties(device):
    # assert the defaults are right
    assert device.gpu_error_checking

    # make sure we can set the properties
    device.gpu_error_checking = False
    assert not device.gpu_error_checking

    # make sure we can give a GPU id
    hoomd.device.GPU(gpu_id=0)

    c = device.compute_capability
    assert type(c) is tuple
    assert len(c) == 2
    assert c[0] > 0
    assert c[1] >= 0


@pytest.mark.gpu
def test_other_gpu_specifics(device):
    # make sure GPU is available and auto-select gives a GPU
    assert hoomd.device.GPU.is_available()
    assert isinstance(hoomd.device.auto_select(), hoomd.device.GPU)

    # make sure we can still make a CPU
    hoomd.device.CPU()


def _assert_list_str(values):
    """Asserts the input is a list of strings."""
    assert type(values) is list
    if len(values) > 0:
        assert type(values[0]) is str


@pytest.mark.gpu
def test_query_hardware_info(device):
    reasons = device.get_unavailable_device_reasons()
    _assert_list_str(reasons)

    devices = device.get_available_devices()
    _assert_list_str(devices)


def test_cpu_build_specifics():
    if hoomd.version.gpu_enabled:
        pytest.skip("Don't run CPU-build specific tests when GPU is available")
    assert not hoomd.device.GPU.is_available()
    assert isinstance(hoomd.device.auto_select(), hoomd.device.CPU)


def test_device_notice(device, tmp_path):
    # Message file declared. Should output in specified file.
    device.notice_level = 4
    device.message_filename = str(tmp_path / "str_message")
    msg = "This message should output."
    device.notice(msg)

    if device.communicator.rank == 0:
        with open(device.message_filename) as fh:
            assert fh.read() == msg + "\n"

    # Test notice with a message that is not a string.
    device.message_filename = str(tmp_path / "int_message")
    msg = 123456
    device.notice(msg)

    if device.communicator.rank == 0:
        with open(device.message_filename) as fh:
            assert fh.read() == str(msg) + "\n"

    # Test the level argument.
    device.message_filename = str(tmp_path / "empty_notice")
    msg = "This message should not output."
    device.notice(msg, level=5)

    if device.communicator.rank == 0:
        with open(device.message_filename) as fh:
            assert fh.read() == ""


def test_noticefile(device, tmp_path):
    # Message file declared. Should output in specified file.
    device.message_filename = str(tmp_path / "str_message")
    msg = "This message should output.\n"
    device.notice_level = 4
    notice_file = hoomd.device.NoticeFile(device)
    notice_file.write(msg)

    if device.communicator.rank == 0:
        with open(device.message_filename) as fh:
            assert fh.read() == str(msg)

    # Test notice with a message that is not a string.
    msg = 123456
    device.message_filename = str(tmp_path / "int_message")
    with pytest.raises(TypeError):
        notice_file.write(msg)

    # Test the level argument
    msg = "This message should not output.\n"
    device.message_filename = str(tmp_path / "empty_notice")
    notice_file = hoomd.device.NoticeFile(device, level=5)
    notice_file.write(msg)

    if device.communicator.rank == 0:
        with open(device.message_filename) as fh:
            assert fh.read() == ""
