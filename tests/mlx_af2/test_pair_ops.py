import mlx.core as mx
import numpy as np
import pytest

import cases
import jax_units
from bindcraft.af.alphafold.model import modules as af_modules
from bindcraft.af.mlx_af2 import pair_ops
from bindcraft.af.mlx_af2.weights import load_params
from helpers import assert_live, rel_err

EVO = 'evoformer/evoformer_iteration'
N = 12

@pytest.fixture(scope='module')
def params():
    return load_params(jax_units.MODEL, cases.params_path(), dtype=mx.float32)

def evoformer_config(name):
    return jax_units.configs()[0].embeddings_and_evoformer.evoformer[name]

def pair_inputs(seed, valid=N):
    rng = np.random.default_rng(seed)
    act = rng.standard_normal((N, N, 128)).astype(np.float32)
    mask = np.zeros((N, N), np.float32)
    mask[:valid, :valid] = 1.0
    return act, mask

@pytest.mark.integration
@pytest.mark.parametrize('layer', (0, 47))
@pytest.mark.parametrize('operation', ('outgoing', 'incoming'))
def test_triangle_multiplication_matches(params, operation, layer):
    name = f'triangle_multiplication_{operation}'
    act, mask = pair_inputs(0 if operation == 'outgoing' else 1, valid=N - 2)
    want = jax_units.run_unit(lambda c, g: af_modules.TriangleMultiplication(c.embeddings_and_evoformer.evoformer[name], g, name=name), f'{EVO}/{name}', name, [act, mask], layer=layer)
    assert_live(want)
    got = np.asarray(pair_ops.triangle_multiplication(params.scope(f'{EVO}/{name}').layer(layer), mx.array(act), mx.array(mask), evoformer_config(name).to_dict()))
    assert rel_err(got, want) < 1e-5

def test_the_config_uses_the_fused_projection_path():
    for name in ('triangle_multiplication_outgoing', 'triangle_multiplication_incoming'):
        assert evoformer_config(name).fuse_projection_weights, name  #the port implements the fused path only, which is what the fused weights serve

@pytest.mark.integration
@pytest.mark.parametrize('layer', (0, 47))
@pytest.mark.parametrize('node', ('starting', 'ending'))
def test_triangle_attention_matches(params, node, layer):
    name = f'triangle_attention_{node}_node'
    act, mask = pair_inputs(2 if node == 'starting' else 3, valid=N - 3)
    want = jax_units.run_unit(lambda c, g: af_modules.TriangleAttention(c.embeddings_and_evoformer.evoformer[name], g, name=name), f'{EVO}/{name}', name, [act, mask], layer=layer)
    assert_live(want)
    got = np.asarray(pair_ops.triangle_attention(params.scope(f'{EVO}/{name}').layer(layer), mx.array(act), mx.array(mask), evoformer_config(name).to_dict()))
    assert rel_err(got, want) < 1e-5

@pytest.mark.integration
def test_triangle_attention_subbatch_equals_unbatched(params):
    name = 'triangle_attention_starting_node'
    act, mask = pair_inputs(4, valid=N - 1)
    p, cfg = params.scope(f'{EVO}/{name}').layer(5), evoformer_config(name).to_dict()
    plain = np.asarray(pair_ops.triangle_attention(p, mx.array(act), mx.array(mask), cfg))
    assert rel_err(np.asarray(pair_ops.triangle_attention(p, mx.array(act), mx.array(mask), cfg, subbatch_size=5)), plain) < 1e-6

@pytest.mark.integration
def test_fused_halves_equal_the_unfused_algorithm():
    #an independent numpy run of the unfused algorithm (separate left and right projections and gates, as AlphaFold 2 defines it) on the two halves of the fused weights, against the port
    import os
    raw = np.load(os.path.join(cases.params_path(), f'params_{jax_units.MODEL}.npz'))
    base = 'alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing'
    t = lambda scope, name: raw[f'{base}/{scope}//{name}'][3]
    assert t('projection', 'weights').shape == (128, 256)  #the multimer v3 file is stored fused already: one projection of 2 x 128 channels
    half = lambda scope, name, side: np.split(t(scope, name), 2, axis=-1)[side]
    sigmoid = lambda v: 1.0 / (1.0 + np.exp(-v))
    def norm(x, scope):
        mean = x.mean(-1, keepdims=True)
        return (x - mean) / np.sqrt(((x - mean) ** 2).mean(-1, keepdims=True) + 1e-5) * t(scope, 'scale') + t(scope, 'offset')
    act, mask = pair_inputs(7, valid=N - 2)
    x = norm(act, 'left_norm_input')
    side = lambda s: mask[..., None] * (x @ half('projection', 'weights', s) + half('projection', 'bias', s)) * sigmoid(x @ half('gate', 'weights', s) + half('gate', 'bias', s))
    out = norm(np.einsum('ikc,jkc->ijc', side(0), side(1)), 'center_norm') @ t('output_projection', 'weights') + t('output_projection', 'bias')
    want = out * sigmoid(x @ t('gating_linear', 'weights') + t('gating_linear', 'bias'))
    p = load_params(jax_units.MODEL, cases.params_path(), dtype=mx.float32).scope(f'{EVO}/triangle_multiplication_outgoing').layer(3)
    got = np.asarray(pair_ops.triangle_multiplication(p, mx.array(act), mx.array(mask), evoformer_config('triangle_multiplication_outgoing').to_dict()))
    assert rel_err(got, want) < 1e-4  #the numpy norms use the exact variance, the port the fast one the JAX model uses

@pytest.mark.integration
@pytest.mark.parametrize('operation', ('triangle_multiplication_outgoing', 'triangle_attention_starting_node', 'triangle_attention_ending_node'))
def test_masked_positions_do_not_leak(params, operation):
    valid = N - 3
    act, mask = pair_inputs(8, valid=valid)
    other = act.copy()
    other[valid:, :], other[:, valid:] = 99.0 * np.random.default_rng(9).standard_normal((N - valid, N, 128)), 99.0 * np.random.default_rng(10).standard_normal((N, N - valid, 128))
    op = pair_ops.triangle_multiplication if 'multiplication' in operation else pair_ops.triangle_attention
    p, cfg = params.scope(f'{EVO}/{operation}').layer(2), evoformer_config(operation).to_dict()
    first = np.asarray(op(p, mx.array(act), mx.array(mask), cfg))[:valid, :valid]
    second = np.asarray(op(p, mx.array(other), mx.array(mask), cfg))[:valid, :valid]
    assert_live(first)
    assert rel_err(second, first) < 1e-6
