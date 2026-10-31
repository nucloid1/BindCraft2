import os
import platform
import sys

BACKENDS = ('jax', 'mlx')

def apple_silicon() -> bool:
    return sys.platform == 'darwin' and platform.machine() == 'arm64'

def resolve_backend() -> str:
    backend = os.environ.get('BINDCRAFT_BACKEND', 'jax')
    if backend not in BACKENDS:
        raise ValueError(f'BINDCRAFT_BACKEND={backend!r} names no backend; use one of {", ".join(BACKENDS)}')
    if backend == 'mlx' and not apple_silicon():
        raise ValueError('the mlx backend runs AlphaFold on the Apple GPU and needs an Apple silicon Mac; leave BINDCRAFT_BACKEND unset (jax) elsewhere')
    return backend
