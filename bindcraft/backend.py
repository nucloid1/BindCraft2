import importlib.util
import os
import platform
import sys

BACKENDS = ('jax', 'mlx')

def apple_silicon() -> bool:
    return sys.platform == 'darwin' and platform.machine() == 'arm64'

def design_model_class() -> type:
    if resolve_backend() == 'mlx':
        from bindcraft.af.mlx_af2.predictor import MlxAlphaFoldDesignModel
        return MlxAlphaFoldDesignModel
    from bindcraft.af2 import AlphaFoldDesignModel
    return AlphaFoldDesignModel

def resolve_backend() -> str:
    backend = os.environ.get('BINDCRAFT_BACKEND', 'jax')
    if backend not in BACKENDS:
        raise ValueError(f'BINDCRAFT_BACKEND={backend!r} names no backend; use one of {", ".join(BACKENDS)}')
    if backend == 'mlx' and not apple_silicon():
        raise ValueError('the mlx backend runs AlphaFold on the Apple GPU and needs an Apple silicon Mac; leave BINDCRAFT_BACKEND unset (jax) elsewhere')
    if backend == 'mlx' and importlib.util.find_spec('mlx') is None:
        raise ValueError("BINDCRAFT_BACKEND=mlx needs the mlx package, which is not installed; install it with: pip install 'bindcraft[mlx]'")
    return backend
