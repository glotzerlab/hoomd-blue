def test_device():
    assert True


def test_gpu_profile(device_gpu):
    with device_gpu.enable_profiling():
        pass
