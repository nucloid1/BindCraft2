"""Run one vendored haiku module on its own, under hk.transform, with weights sliced from the real npz: the per-module reference every MLX port is judged against.
Usage: run_unit(build, scope, name, inputs, layer=0), where build(model_config, global_config) returns the module (constructed with name=name) and scope is the module's path in the npz."""
import contextlib
import copy
import os

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from bindcraft.af.alphafold.model import config as af_config, data as af_data, utils as af_utils
from bindcraft.af.mlx_af2.weights import PREFIX, STACK_MARKERS

MODEL = 'model_1_multimer_v3'
_loaded: dict = {}

def params_dir() -> str:
    return os.path.expanduser(os.environ.get('BINDCRAFT_AF2_PARAMS') or '~/.cache/bindcraft/alphafold')

def jax_params(model_name: str = MODEL) -> dict:
    if model_name not in _loaded:
        _loaded[model_name] = af_data.get_model_haiku_params(model_name=model_name, data_dir=params_dir(), fuse=True)
    return _loaded[model_name]

def configs(model_name: str = MODEL, bfloat16: bool = False, subbatch_size=None, attention_backend: str = 'stock'):
    config = copy.deepcopy(af_config.model_config(model_name)).model
    config.global_config.bfloat16, config.global_config.subbatch_size, config.global_config.attention_backend = bfloat16, subbatch_size, attention_backend
    return config, config.global_config

def reference_params(scope: str, name: str, layer: int | None = None, model_name: str = MODEL, keep_layers: int | None = None) -> dict:
    root = f'{PREFIX}{scope}'
    found = {}
    for key, tensors in jax_params(model_name).items():
        if key == root or key.startswith(root + '/'):
            stacked = any(marker in key for marker in STACK_MARKERS)
            take = (lambda array: jnp.asarray(np.asarray(array)[layer])) if stacked and layer is not None else (lambda array: jnp.asarray(np.asarray(array)[:keep_layers])) if stacked and keep_layers is not None else (lambda array: array)  #keep_layers: the first k layers of every stack, for a model cut down to k blocks
            found[name + key[len(root):]] = {tensor: take(array) for tensor, array in tensors.items()}
    assert found, f'no parameters under {scope}'
    return found

def run_unit(build, scope: str, name: str, inputs: list, layer: int | None = None, bfloat16: bool = False, subbatch_size=None, attention_backend: str = 'stock', model_name: str = MODEL, rng=None, adjust=None, keep_layers: int | None = None, outer_bfloat16_context: bool = True):
    config, global_config = configs(model_name, bfloat16, subbatch_size, attention_backend)
    if adjust is not None:
        adjust(config)  #e.g. fewer blocks, templates off
    params = reference_params(scope, name, layer, model_name, keep_layers)

    def apply(*arguments):
        with af_utils.bfloat16_context() if bfloat16 and outer_bfloat16_context else contextlib.nullcontext():  #modules that enter the context themselves (EmbeddingsAndEvoformer) must not be wrapped in it twice
            return build(config, global_config)(*arguments)

    result = hk.transform(apply).apply(params, jax.random.PRNGKey(0) if rng is None else rng, *[jax.tree.map(jnp.asarray, value) for value in inputs])  #inputs may be pytrees (dicts of activations); None stays None
    return jax.tree.map(np.asarray, result)
