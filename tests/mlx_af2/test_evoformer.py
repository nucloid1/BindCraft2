import jax
import jax.numpy as jnp
import mlx.core as mx
import numpy as np
import pytest

import cases
import jax_units
from bindcraft.af.alphafold.model import layer_stack, modules as af_modules, prng
from bindcraft.af.mlx_af2 import evoformer
from bindcraft.af.mlx_af2.weights import load_params
from helpers import assert_live, rel_err

MAIN, EXTRA = 'evoformer/evoformer_iteration', 'evoformer/extra_msa_stack'
ROWS, N = 3, 10

@pytest.fixture(scope='module')
def params():
    return load_params(jax_units.MODEL, cases.params_path(), dtype=mx.float32)

def evoformer_config():
    return jax_units.configs()[0].embeddings_and_evoformer.evoformer

def block_inputs(msa_channels, seed, masked_residues=2):
    rng = np.random.default_rng(seed)
    msa = (0.5 * rng.standard_normal((ROWS, N, msa_channels))).astype(np.float32)
    pair = (0.5 * rng.standard_normal((N, N, 128))).astype(np.float32)
    residues = np.ones(N, np.float32)
    residues[N - masked_residues:] = 0.0
    msa_mask = np.ones((ROWS, N), np.float32) * residues
    msa_mask[1, 2] = 0.0
    return msa, pair, msa_mask, np.outer(residues, residues).astype(np.float32)

def reference_block(is_extra, scope, name, layer, msa, pair, msa_mask, pair_mask):
    return jax_units.run_unit(lambda c, g: af_modules.EvoformerIteration(c.embeddings_and_evoformer.evoformer, g, is_extra_msa=is_extra, name=name), scope, name,
                              [{'msa': msa, 'pair': pair}, {'msa': msa_mask, 'pair': pair_mask}, jnp.asarray(False)], layer=layer)

def reference_stack(is_extra, scope, name, num_block, msa, pair, msa_mask, pair_mask):
    def build(c, g):
        def run(activations, masks, key):
            iteration = af_modules.EvoformerIteration(c.embeddings_and_evoformer.evoformer, g, is_extra_msa=is_extra, name=name)
            def block(x):
                act, safe_key = x
                safe_key, sub = safe_key.split()
                return iteration(activations=act, masks=masks, use_dropout=jnp.asarray(False), safe_key=sub), safe_key
            return layer_stack.layer_stack(num_block)(block)((activations, prng.SafeKey(key)))[0]
        return run
    return jax_units.run_unit(build, scope, name, [{'msa': msa, 'pair': pair}, {'msa': msa_mask, 'pair': pair_mask}, jax.random.PRNGKey(1)])

@pytest.mark.integration
@pytest.mark.parametrize('layer', (0, 47))
def test_iteration_matches_a_single_block(params, layer):
    msa, pair, msa_mask, pair_mask = block_inputs(256, layer)
    want = reference_block(False, MAIN, 'evoformer_iteration', layer, msa, pair, msa_mask, pair_mask)
    assert_live(want['msa']) and assert_live(want['pair'])
    got_msa, got_pair = evoformer.evoformer_iteration(params.scope(MAIN).layer(layer), mx.array(msa), mx.array(pair), mx.array(msa_mask), mx.array(pair_mask), evoformer_config().to_dict())
    assert rel_err(np.asarray(got_msa), want['msa']) < 1e-5 and rel_err(np.asarray(got_pair), want['pair']) < 1e-5

@pytest.mark.integration
@pytest.mark.parametrize('layer', (0, 3))
def test_extra_msa_iteration_matches_a_single_block(params, layer):
    msa, pair, msa_mask, pair_mask = block_inputs(64, 10 + layer)
    want = reference_block(True, EXTRA, 'extra_msa_stack', layer, msa, pair, msa_mask, pair_mask)
    got_msa, got_pair = evoformer.evoformer_iteration(params.scope(EXTRA).layer(layer), mx.array(msa), mx.array(pair), mx.array(msa_mask), mx.array(pair_mask), evoformer_config().to_dict(), is_extra=True)
    assert rel_err(np.asarray(got_msa), want['msa']) < 1e-5 and rel_err(np.asarray(got_pair), want['pair']) < 1e-5

@pytest.mark.integration
def test_stack_of_48_matches(params):
    msa, pair, msa_mask, pair_mask = block_inputs(256, 20)
    want = reference_stack(False, MAIN, 'evoformer_iteration', 48, msa, pair, msa_mask, pair_mask)
    assert_live(want['msa']) and assert_live(want['pair'])
    got_msa, got_pair = evoformer.evoformer_stack(params.scope(MAIN), mx.array(msa), mx.array(pair), mx.array(msa_mask), mx.array(pair_mask), evoformer_config().to_dict(), num_block=48)
    assert rel_err(np.asarray(got_msa), want['msa']) < 1e-3 and rel_err(np.asarray(got_pair), want['pair']) < 1e-3

@pytest.mark.integration
def test_extra_msa_stack_of_4_matches(params):
    msa, pair, msa_mask, pair_mask = block_inputs(64, 30)
    want = reference_stack(True, EXTRA, 'extra_msa_stack', 4, msa, pair, msa_mask, pair_mask)
    got_msa, got_pair = evoformer.evoformer_stack(params.scope(EXTRA), mx.array(msa), mx.array(pair), mx.array(msa_mask), mx.array(pair_mask), evoformer_config().to_dict(), num_block=4, is_extra=True)
    assert rel_err(np.asarray(got_msa), want['msa']) < 1e-4 and rel_err(np.asarray(got_pair), want['pair']) < 1e-4

@pytest.mark.integration
def test_chunked_equals_unchunked(params):
    msa, pair, msa_mask, pair_mask = (mx.array(a) for a in block_inputs(256, 40))
    args = (params.scope(MAIN).layer(5), msa, pair, msa_mask, pair_mask, evoformer_config().to_dict())
    whole = evoformer.evoformer_iteration(*args)
    chunked = evoformer.evoformer_iteration(*args, subbatch_size=4)  #10 residues = 4 + 4 + 2, 3 rows = 3
    assert rel_err(np.asarray(chunked[0]), np.asarray(whole[0])) < 1e-5 and rel_err(np.asarray(chunked[1]), np.asarray(whole[1])) < 1e-5

@pytest.mark.integration
def test_remat_gradient_equals_plain_gradient(params):
    msa, pair, msa_mask, pair_mask = (mx.array(a) for a in block_inputs(256, 50))
    cfg = evoformer_config().to_dict()
    def loss(msa_in, pair_in, remat):
        out_msa, out_pair = evoformer.evoformer_stack(params.scope(MAIN), msa_in, pair_in, msa_mask, pair_mask, cfg, num_block=3, remat=remat)
        return mx.sum(out_msa * out_msa) * 1e-3 + mx.sum(out_pair * out_pair) * 1e-3
    plain = mx.grad(lambda m, p: loss(m, p, False), argnums=(0, 1))(msa, pair)
    remat = mx.grad(lambda m, p: loss(m, p, True), argnums=(0, 1))(msa, pair)
    for got, want in zip(remat, plain):
        assert_live(np.asarray(want)) and rel_err(np.asarray(got), np.asarray(want)) < 1e-5

def test_dropout_residual_is_shared_along_the_broadcast_axis_and_keeps_the_expected_fraction():
    update = mx.ones((64, 64, 128))
    key = mx.random.key(0)
    per_row = np.asarray(evoformer.dropout_update(update, {'dropout_rate': 0.25, 'orientation': 'per_row', 'shared_dropout': True}, key))
    per_column = np.asarray(evoformer.dropout_update(update, {'dropout_rate': 0.25, 'orientation': 'per_column', 'shared_dropout': True}, key))
    assert (per_row == per_row[:1]).all() and not (per_row == per_row[:, :1]).all()  #per_row: one mask shared along axis 0, varying along axis 1
    assert (per_column == per_column[:, :1]).all() and not (per_column == per_column[:1]).all()  #per_column: shared along axis 1
    assert abs((per_row > 0).mean() - 0.75) < 0.05 and np.allclose(per_row[per_row > 0], 1 / 0.75)  #kept entries are scaled by 1 / keep rate
    assert np.asarray(evoformer.dropout_update(update, {'dropout_rate': 0.0, 'orientation': 'per_row', 'shared_dropout': True}, key)).tolist() == np.asarray(update).tolist()  #rate zero is the identity
    assert evoformer.dropout_update(update, {'dropout_rate': 0.25, 'orientation': 'per_row', 'shared_dropout': True}, None) is update  #no key: dropout off

@pytest.mark.integration
def test_dropout_off_is_deterministic_and_on_is_keyed(params):
    msa, pair, msa_mask, pair_mask = (mx.array(a) for a in block_inputs(256, 60))
    args = (params.scope(MAIN).layer(1), msa, pair, msa_mask, pair_mask, evoformer_config().to_dict())
    off_a, off_b = evoformer.evoformer_iteration(*args), evoformer.evoformer_iteration(*args)
    assert np.array_equal(np.asarray(off_a[1]), np.asarray(off_b[1]))
    on_a = evoformer.evoformer_iteration(*args, dropout_key=mx.random.key(1))
    on_b = evoformer.evoformer_iteration(*args, dropout_key=mx.random.key(1))
    on_c = evoformer.evoformer_iteration(*args, dropout_key=mx.random.key(2))
    assert np.array_equal(np.asarray(on_a[1]), np.asarray(on_b[1])) and not np.array_equal(np.asarray(on_a[1]), np.asarray(on_c[1])) and not np.array_equal(np.asarray(on_a[1]), np.asarray(off_a[1]))
