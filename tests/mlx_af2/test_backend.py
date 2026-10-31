import os
import subprocess
import sys

import pytest

from bindcraft import backend
from bindcraft.backend import resolve_backend

DROPPED_ENV = ('JAX_PLATFORMS', 'JAX_PLATFORM_NAME', 'BINDCRAFT_BACKEND')

def platform_after_import(**extra_env) -> str:
    env = {name: value for name, value in os.environ.items() if name not in DROPPED_ENV}
    env.update(extra_env)
    result = subprocess.run([sys.executable, '-I', '-B', '-c', "import bindcraft, os; print(os.environ.get('JAX_PLATFORMS', ''))"], env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()

def test_default_backend_is_jax(monkeypatch):
    monkeypatch.delenv('BINDCRAFT_BACKEND', raising=False)
    assert resolve_backend() == 'jax'

def test_unknown_backend_names_the_accepted_values(monkeypatch):
    monkeypatch.setenv('BINDCRAFT_BACKEND', 'tpu')
    with pytest.raises(ValueError, match=r"'tpu'.*jax.*mlx"):
        resolve_backend()

def test_mlx_backend_is_refused_off_apple_silicon(monkeypatch):
    monkeypatch.setenv('BINDCRAFT_BACKEND', 'mlx')
    monkeypatch.setattr(backend, 'apple_silicon', lambda: False)
    with pytest.raises(ValueError, match='Apple silicon'):
        resolve_backend()

def test_mlx_backend_defaults_jax_to_cpu():
    assert platform_after_import(BINDCRAFT_BACKEND='mlx') == 'cpu'

def test_explicit_platform_wins():
    assert platform_after_import(BINDCRAFT_BACKEND='mlx', JAX_PLATFORMS='metal') == 'metal'

def test_jax_backend_leaves_platform_logic_alone():
    import importlib.util
    on_metal = sys.platform == 'darwin' and os.uname().machine == 'arm64' and importlib.util.find_spec('jax_plugins.metal') is not None
    assert platform_after_import() == ('metal' if on_metal else '')
