import mlx.core as mx
import pytest

def test_mlx_defaults_to_the_cpu_device_in_tests():
    #unit parity tests must not touch the GPU, which another run may hold
    assert mx.default_device() == mx.cpu

@pytest.mark.gpu
def test_mlx_uses_the_gpu_under_the_gpu_marker():
    assert mx.default_device() == mx.gpu
