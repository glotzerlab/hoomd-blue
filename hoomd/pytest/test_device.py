import pytest

@pytest.mark.gpu
def test_gpu_profile(device):

    print(device)

    with device.enable_profiling():
        pass
