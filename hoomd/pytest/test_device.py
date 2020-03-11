def test_device():
    assert True


def test_gpu_profile(device_gpu):
    device_gpu.profile_start()
    device_gpu.profile_stop()
