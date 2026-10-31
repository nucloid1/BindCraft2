import mlx.core as mx
import numpy as np
import pytest

import cases
import jax_units
from bindcraft.af.alphafold.model import common_modules, modules as af_modules
from bindcraft.af.mlx_af2 import layers
from bindcraft.af.mlx_af2.weights import Params, Store, load_params
from helpers import assert_live, rel_err

EVO = 'evoformer/evoformer_iteration'

@pytest.fixture(scope='module')
def params():
    return load_params(jax_units.MODEL, cases.params_path(), dtype=mx.float32)

def evoformer_config(config, name):
    return config.embeddings_and_evoformer.evoformer[name]

def normal(shape, seed, scale=1.0):
    return (scale * np.random.default_rng(seed).standard_normal(shape)).astype(np.float32)

@pytest.mark.integration
def test_linear_matches_haiku(params):
    x = normal((4, 7, 256), 0)
    scope = f'{EVO}/msa_transition/transition1'
    want = jax_units.run_unit(lambda c, g: common_modules.Linear(1024, initializer='relu', name='transition1'), scope, 'transition1', [x], layer=0)
    assert_live(want)
    got = np.asarray(layers.linear(params.scope(scope).layer(0), mx.array(x)))
    assert rel_err(got, want) < 1e-5

def test_linear_with_two_input_and_two_output_dims_matches_einsum():
    weights, bias, x = normal((3, 4, 5, 6), 1), normal((5, 6), 2), normal((2, 7, 3, 4), 3)
    store = Store({'w/weights': weights, 'w/bias': bias}, {'w/weights': ['a'], 'w/bias': ['b']}, mx.float32, 'synthetic')
    got = np.asarray(layers.linear(Params(store).scope('w'), mx.array(x), num_input_dims=2))
    assert got.shape == (2, 7, 5, 6) and rel_err(got, np.einsum('...ab,abhi->...hi', x, weights) + bias) < 1e-6

def test_linear_without_a_bias_has_none_to_add():
    weights, x = normal((4, 5), 4), normal((3, 4), 5)
    store = Store({'w/weights': weights}, {'w/weights': ['a']}, mx.float32, 'synthetic')
    assert rel_err(np.asarray(layers.linear(Params(store).scope('w'), mx.array(x))), x @ weights) < 1e-6

@pytest.mark.integration
def test_layer_norm_matches_haiku(params):
    x = normal((4, 7, 256), 6, scale=3.0) + 1.5
    scope = f'{EVO}/msa_transition/input_layer_norm'
    want = jax_units.run_unit(lambda c, g: common_modules.LayerNorm(axis=[-1], create_scale=True, create_offset=True, name='input_layer_norm'), scope, 'input_layer_norm', [x], layer=0)
    assert rel_err(np.asarray(layers.layer_norm(params.scope(scope).layer(0), mx.array(x))), want) < 1e-5

@pytest.mark.integration
def test_layer_norm_parameters_stay_float32_in_bf16_mode(monkeypatch):
    low = load_params(jax_units.MODEL, cases.params_path(), dtype=mx.bfloat16)
    requested = []
    original = Params.get
    monkeypatch.setattr(Params, 'get', lambda self, path, dtype=None: requested.append(dtype) or original(self, path, dtype))
    x = mx.array(normal((4, 7, 256), 7)).astype(mx.bfloat16)
    out = layers.layer_norm(low.scope(f'{EVO}/msa_transition/input_layer_norm').layer(0), x)
    assert out.dtype == mx.bfloat16 and requested == [mx.float32, mx.float32]  #as the repo's LayerNorm: computed in fp32 with fp32 parameters, cast back

def attention_inputs(rows, columns, seed):
    q = normal((rows, columns, 256), seed)
    mask = (np.random.default_rng(seed + 1).random((rows, columns)) > 0.2).astype(np.float32)
    mask[:, 0] = 1.0
    return q, (1e9 * (mask - 1.0))[:, None, None, :].astype(np.float32)

@pytest.mark.integration
def test_attention_with_gating_and_bias_matches(params):
    q, bias = attention_inputs(5, 9, 10)
    nonbatched = normal((8, 9, 9), 12)
    scope = f'{EVO}/msa_row_attention_with_pair_bias/attention'
    cfg = evoformer_config(jax_units.configs()[0], 'msa_row_attention_with_pair_bias')
    want = jax_units.run_unit(lambda c, g: af_modules.Attention(evoformer_config(c, 'msa_row_attention_with_pair_bias'), g, 256, name='attention'), scope, 'attention', [q, q, bias, nonbatched], layer=0)
    assert_live(want)
    got = np.asarray(layers.attention(params.scope(scope).layer(0), mx.array(q), mx.array(q), mx.array(bias), mx.array(nonbatched), cfg.to_dict()))
    assert rel_err(got, want) < 1e-5

@pytest.mark.integration
def test_attention_without_nonbatched_bias_and_masked_keys_matches(params):
    q, bias = attention_inputs(6, 8, 20)
    scope = f'{EVO}/msa_column_attention/attention'
    cfg = evoformer_config(jax_units.configs()[0], 'msa_column_attention')
    want = jax_units.run_unit(lambda c, g: af_modules.Attention(evoformer_config(c, 'msa_column_attention'), g, 256, name='attention'), scope, 'attention', [q, q, bias, None], layer=0)
    got = np.asarray(layers.attention(params.scope(scope).layer(0), mx.array(q), mx.array(q), mx.array(bias), None, cfg.to_dict()))
    assert rel_err(got, want) < 1e-5

@pytest.mark.integration
def test_attention_subbatch_and_query_chunks_equal_unbatched(params):
    q, bias = attention_inputs(7, 11, 30)
    nonbatched = normal((8, 11, 11), 32)
    p = params.scope(f'{EVO}/msa_row_attention_with_pair_bias/attention').layer(0)
    cfg = evoformer_config(jax_units.configs()[0], 'msa_row_attention_with_pair_bias').to_dict()
    args = (mx.array(q), mx.array(q), mx.array(bias), mx.array(nonbatched), cfg)
    plain = np.asarray(layers.attention(p, *args, chunk_size=0))
    assert rel_err(np.asarray(layers.attention(p, *args, subbatch_size=3, chunk_size=0)), plain) < 1e-6  #the remainder shard (7 = 3 + 3 + 1) included
    assert rel_err(np.asarray(layers.attention(p, *args, chunk_size=4)), plain) < 1e-6  #queries in chunks of 4 (11 = 4 + 4 + 3)

@pytest.mark.integration
def test_global_attention_matches(params):
    rows, columns = 6, 5
    q = normal((columns, rows, 64), 40)
    mask = (np.random.default_rng(41).random((columns, rows, 1)) > 0.3).astype(np.float32)
    mask[:, 0] = 1.0
    scope = 'evoformer/extra_msa_stack/msa_column_global_attention/attention'
    want = jax_units.run_unit(lambda c, g: af_modules.GlobalAttention(c.embeddings_and_evoformer.evoformer.msa_column_attention, g, 64, name='attention'), scope, 'attention', [q, q, mask, np.zeros((columns, 1, rows), np.float32)], layer=0)  #the last argument, bias, is dead in the JAX module
    assert_live(want)
    cfg = evoformer_config(jax_units.configs()[0], 'msa_column_attention').to_dict()  #the extra-MSA stack's global attention is built from the column-attention config
    got = np.asarray(layers.global_attention(params.scope(scope).layer(0), mx.array(q), mx.array(q), mx.array(mask), cfg))
    assert rel_err(got, want) < 1e-5

@pytest.mark.integration
def test_transition_matches_and_subbatch_equals_unbatched(params):
    x, mask = normal((6, 10, 256), 50), np.ones((6, 10), np.float32)
    scope = f'{EVO}/msa_transition'
    cfg = evoformer_config(jax_units.configs()[0], 'msa_transition')
    want = jax_units.run_unit(lambda c, g: af_modules.Transition(evoformer_config(c, 'msa_transition'), g, name='msa_transition'), scope, 'msa_transition', [x, mask], layer=0)
    p = params.scope(scope).layer(0)
    got = np.asarray(layers.transition(p, mx.array(x), mx.array(mask), cfg.to_dict()))
    assert rel_err(got, want) < 1e-5
    assert rel_err(np.asarray(layers.transition(p, mx.array(x), mx.array(mask), cfg.to_dict(), subbatch_size=4)), got) < 1e-6
