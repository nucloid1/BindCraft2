import mlx.core as mx
import numpy as np
import pytest

import cases
import jax_units
from bindcraft.af.alphafold.model import modules as af_modules
from bindcraft.af.mlx_af2 import msa_ops
from bindcraft.af.mlx_af2.weights import load_params
from helpers import assert_live, rel_err

EVO, EXTRA = 'evoformer/evoformer_iteration', 'evoformer/extra_msa_stack'
N = 11

@pytest.fixture(scope='module')
def params():
    return load_params(jax_units.MODEL, cases.params_path(), dtype=mx.float32)

def evoformer_config(name):
    return jax_units.configs()[0].embeddings_and_evoformer.evoformer[name]

def msa_inputs(rows, channels, seed, masked_columns=0):
    rng = np.random.default_rng(seed)
    act = rng.standard_normal((rows, N, channels)).astype(np.float32)
    pair = rng.standard_normal((N, N, 128)).astype(np.float32)
    mask = np.ones((rows, N), np.float32)
    if rows > 1:
        mask[rng.random((rows, N)) > 0.8] = 0.0
        mask[0] = 1.0
    if masked_columns:
        mask[:, N - masked_columns:] = 0.0
    return act, mask, pair

ROWS = pytest.mark.parametrize('rows', (5, 1))  #five rows, and the one row a design has

@pytest.mark.integration
@ROWS
@pytest.mark.parametrize('layer', (0, 47))
def test_row_attention_with_pair_bias_matches(params, rows, layer):
    name = 'msa_row_attention_with_pair_bias'
    act, mask, pair = msa_inputs(rows, 256, 0)
    want = jax_units.run_unit(lambda c, g: af_modules.MSARowAttentionWithPairBias(c.embeddings_and_evoformer.evoformer[name], g, name=name), f'{EVO}/{name}', name, [act, mask, pair], layer=layer)
    assert_live(want)
    got = np.asarray(msa_ops.msa_row_attention_with_pair_bias(params.scope(f'{EVO}/{name}').layer(layer), mx.array(act), mx.array(mask), mx.array(pair), evoformer_config(name).to_dict()))
    assert rel_err(got, want) < 1e-5

@pytest.mark.integration
@ROWS
@pytest.mark.parametrize('layer', (0, 47))
def test_column_attention_matches(params, rows, layer):
    name = 'msa_column_attention'
    act, mask, _ = msa_inputs(rows, 256, 1)
    want = jax_units.run_unit(lambda c, g: af_modules.MSAColumnAttention(c.embeddings_and_evoformer.evoformer[name], g, name=name), f'{EVO}/{name}', name, [act, mask], layer=layer)
    assert_live(want)
    got = np.asarray(msa_ops.msa_column_attention(params.scope(f'{EVO}/{name}').layer(layer), mx.array(act), mx.array(mask), evoformer_config(name).to_dict()))
    assert rel_err(got, want) < 1e-5

@pytest.mark.integration
@ROWS
@pytest.mark.parametrize('layer', (0, 2))
def test_column_global_attention_matches_at_the_extra_msa_width(params, rows, layer):
    name = 'msa_column_global_attention'
    act, mask, _ = msa_inputs(rows, 64, 2)
    want = jax_units.run_unit(lambda c, g: af_modules.MSAColumnGlobalAttention(c.embeddings_and_evoformer.evoformer.msa_column_attention, g, name=name), f'{EXTRA}/{name}', name, [act, mask], layer=layer)
    assert_live(want)
    got = np.asarray(msa_ops.msa_column_global_attention(params.scope(f'{EXTRA}/{name}').layer(layer), mx.array(act), mx.array(mask), evoformer_config('msa_column_attention').to_dict()))
    assert rel_err(got, want) < 1e-5

@pytest.mark.integration
def test_the_last_extra_msa_global_attention_is_exactly_zero_in_the_checkpoint(params):
    #the trained fourth block has all-zero output weights, bias and gating, so the module is a no-op there: both implementations must return exact zeros, not near zeros
    name = 'msa_column_global_attention'
    act, mask, _ = msa_inputs(5, 64, 2)
    want = jax_units.run_unit(lambda c, g: af_modules.MSAColumnGlobalAttention(c.embeddings_and_evoformer.evoformer.msa_column_attention, g, name=name), f'{EXTRA}/{name}', name, [act, mask], layer=3)
    got = np.asarray(msa_ops.msa_column_global_attention(params.scope(f'{EXTRA}/{name}').layer(3), mx.array(act), mx.array(mask), evoformer_config('msa_column_attention').to_dict()))
    assert not want.any() and not got.any()

@pytest.mark.integration
@ROWS
@pytest.mark.parametrize('stack, channels, layer', ((EVO, 256, 0), (EVO, 256, 47), (EXTRA, 64, 0)))
def test_outer_product_mean_matches(params, rows, stack, channels, layer):
    act, mask, _ = msa_inputs(rows, channels, 3)
    want = jax_units.run_unit(lambda c, g: af_modules.OuterProductMean(c.embeddings_and_evoformer.evoformer.outer_product_mean, g, 128, name='outer_product_mean'), f'{stack}/outer_product_mean', 'outer_product_mean', [act, mask], layer=layer)
    assert_live(want)
    got = np.asarray(msa_ops.outer_product_mean(params.scope(f'{stack}/outer_product_mean').layer(layer), mx.array(act), mx.array(mask), evoformer_config('outer_product_mean').to_dict()))
    assert got.shape == (N, N, 128) and rel_err(got, want) < 1e-5

@pytest.mark.integration
def test_outer_product_mean_chunks_equal_unchunked(params):
    act, mask, _ = msa_inputs(4, 256, 4)
    p, cfg = params.scope(f'{EVO}/outer_product_mean').layer(2), evoformer_config('outer_product_mean').to_dict()
    whole = np.asarray(msa_ops.outer_product_mean(p, mx.array(act), mx.array(mask), cfg, chunk_size=None))
    assert rel_err(np.asarray(msa_ops.outer_product_mean(p, mx.array(act), mx.array(mask), cfg, chunk_size=4)), whole) < 1e-6  #11 residues = 4 + 4 + 3

@pytest.mark.integration
@pytest.mark.parametrize('operation', ('row', 'column', 'global'))
def test_masked_columns_do_not_leak(params, operation):
    valid = N - 3
    act, mask, pair = msa_inputs(4, 64 if operation == 'global' else 256, 5, masked_columns=3)
    other = act.copy()
    other[:, valid:] = 99.0 * np.random.default_rng(6).standard_normal(other[:, valid:].shape)
    pair_other = pair.copy()
    pair_other[valid:, :], pair_other[:, valid:] = 50.0, 50.0
    cfg = evoformer_config('msa_row_attention_with_pair_bias' if operation == 'row' else 'msa_column_attention').to_dict()
    if operation == 'row':
        p = params.scope(f'{EVO}/msa_row_attention_with_pair_bias').layer(1)
        run = lambda a, q: msa_ops.msa_row_attention_with_pair_bias(p, mx.array(a), mx.array(mask), mx.array(q), cfg)
    elif operation == 'column':
        p = params.scope(f'{EVO}/msa_column_attention').layer(1)
        run = lambda a, q: msa_ops.msa_column_attention(p, mx.array(a), mx.array(mask), cfg)
    else:
        p = params.scope(f'{EXTRA}/msa_column_global_attention').layer(1)
        run = lambda a, q: msa_ops.msa_column_global_attention(p, mx.array(a), mx.array(mask), cfg)
    first, second = np.asarray(run(act, pair))[:, :valid], np.asarray(run(other, pair_other))[:, :valid]
    assert_live(first)
    assert rel_err(second, first) < 1e-6
