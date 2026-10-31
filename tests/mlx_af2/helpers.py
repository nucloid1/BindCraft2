import numpy as np

def rel_err(got, want) -> float:
    got, want = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
    return float(np.linalg.norm(got - want) / (np.linalg.norm(want) + 1e-30))

def cosine(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))

def assert_live(array, name: str = 'array') -> None:
    #a reference that is all zeros, constant or non-finite would let any wrong port pass
    array = np.asarray(array, dtype=np.float64)
    assert np.isfinite(array).all(), f'{name} has non-finite values'
    assert array.size and array.std() > 0, f'{name} is constant ({array.ravel()[0] if array.size else "empty"})'

def load_ref(case: str, variant: str, index: int = 0) -> dict:
    #one captured call of the JAX path (scripts/mlx_af2/capture_refs.py) as a dict of numpy arrays, with the metadata parsed under 'meta'
    import json
    import os
    path = os.path.join(os.path.expanduser(os.environ.get('BINDCRAFT_MLX_REFS', '~/.cache/bindcraft_mlx_refs')), case, variant, f'call{index}.npz')
    data = np.load(path, allow_pickle=False)
    arrays = {name: data[name] for name in data.files if name != 'meta'}
    arrays['meta'] = json.loads(str(data['meta']))
    return arrays
