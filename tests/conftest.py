import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'mlx_af2'))  #jax_units and the capture scripts

INTEGRATION_ENV = 'BINDCRAFT_MLX_INTEGRATION'

def pytest_collection_modifyitems(config, items):
    if os.environ.get(INTEGRATION_ENV) == '1':
        for item in items:
            if item.get_closest_marker('gpu'):
                item.fixturenames.insert(0, 'gpu_session')
        return
    skip = pytest.mark.skip(reason=f'needs {INTEGRATION_ENV}=1 (captured references, real weights or the GPU)')
    for item in items:
        if item.get_closest_marker('integration') or item.get_closest_marker('gpu'):
            item.add_marker(skip)

@pytest.fixture(scope='session')
def gpu_session():
    #one hold for the whole session; a held lock is a failure, never a skip, so a busy GPU is not mistaken for a green run
    from bindcraft.af.mlx_af2.gpu_lock import hold
    try:
        with hold():
            yield
    except SystemExit as busy:
        pytest.fail(str(busy), pytrace=False)

@pytest.fixture(scope='session')
def refs_dir():
    path = os.path.expanduser(os.environ.get('BINDCRAFT_MLX_REFS', '~/.cache/bindcraft_mlx_refs'))
    if not os.path.isdir(path):
        pytest.fail(f'no references at {path}; run scripts/mlx_af2/capture_refs.py first', pytrace=False)
    return path

@pytest.fixture(scope='session')
def params_dir():
    path = os.path.expanduser(os.environ.get('BINDCRAFT_AF2_PARAMS') or '~/.cache/bindcraft/alphafold')
    if not os.path.isdir(path):
        pytest.fail(f'no AlphaFold parameters at {path}; run bindcraft fetch-weights', pytrace=False)
    return path

@pytest.fixture(autouse=True)
def mlx_device(request):
    #unit parity tests run MLX on the CPU so they never touch the GPU, which another run may hold; only gpu-marked tests, which hold the lock, get the GPU
    import mlx.core as mx
    mx.set_default_device(mx.gpu if request.node.get_closest_marker('gpu') else mx.cpu)
    yield
