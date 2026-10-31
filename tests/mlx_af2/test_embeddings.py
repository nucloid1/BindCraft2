import jax.numpy as jnp
import mlx.core as mx
import numpy as np
import pytest

import cases
import jax_units
from bindcraft.af.alphafold.model import modules as af_modules, modules_multimer
from bindcraft.af.mlx_af2 import embeddings
from bindcraft.af.mlx_af2.weights import load_params
from helpers import assert_live, load_ref, rel_err

@pytest.fixture(scope='module')
def params():
    return load_params(jax_units.MODEL, cases.params_path(), dtype=mx.float32)

def embedding_config():
    return jax_units.configs()[0].embeddings_and_evoformer

def cut_to_one_block(config):
    config.embeddings_and_evoformer.evoformer_num_block = 1
    config.embeddings_and_evoformer.extra_msa_stack_num_block = 1
    config.embeddings_and_evoformer.template.enabled = False

def chain_features(seed, lengths=(7, 5, 6), cyclic=False):
    rng = np.random.default_rng(seed)
    asym = np.repeat(np.arange(len(lengths)), lengths).astype(np.int32)
    n = asym.size
    residue_index = np.concatenate([np.arange(length) for length in lengths]).astype(np.int32)
    features = {'asym_id': asym, 'entity_id': np.minimum(asym, 1).astype(np.int32), 'sym_id': np.where(asym == 0, 0, asym - 1).astype(np.int32), 'residue_index': residue_index}
    if cyclic:
        features['offset'] = rng.integers(-40, 40, (n, n)).astype(np.int32)  #a cyclic peptide's offsets, wide enough to hit the clipping
    return features

@pytest.mark.integration
@pytest.mark.parametrize('cyclic', (False, True))
def test_relative_position_encoding_matches_haiku(params, cyclic):
    features = chain_features(1, cyclic=cyclic)
    want = jax_units.run_unit(lambda c, g: modules_multimer.EmbeddingsAndEvoformer(c.embeddings_and_evoformer, g, name='evoformer')._relative_encoding, 'evoformer/~_relative_encoding/position_activations', 'evoformer/~_relative_encoding/position_activations', [features])
    got = np.asarray(embeddings.relative_encoding(params.scope('evoformer'), {k: mx.array(v) for k, v in features.items()}, embedding_config().to_dict(), mx.float32))
    assert want.shape == got.shape == (18, 18, 128) and assert_live(want) is None and rel_err(got, want) < 1e-6

def test_relative_encoding_one_hots_cover_same_chain_and_different_chain():
    features = {k: mx.array(v) for k, v in chain_features(2).items()}
    cfg = embedding_config().to_dict()
    k, m = cfg['max_relative_idx'], cfg['max_relative_chain']
    rel = embeddings.relative_features(features, cfg, mx.float32)
    assert rel.shape == (18, 18, (2 * k + 2) + 1 + (2 * m + 2))
    assert np.asarray(rel[..., :2 * k + 2]).sum(-1).min() == 1 and np.asarray(rel[..., :2 * k + 2]).sum(-1).max() == 1  #exactly one position bin per pair
    assert np.asarray(rel[0, 10, :2 * k + 2])[2 * k + 1] == 1  #residues 0 and 10 are in different chains: the extra bin

@pytest.mark.integration
def test_recycle_distogram_matches_the_jax_helpers():
    ref = load_ref('small', 'fp32', 0)
    aatype, positions = ref['in/aatype'], ref['out/structure_module/final_atom_positions']
    cfg = embedding_config().to_dict()
    want = np.asarray(af_modules.dgram_from_positions(af_modules.pseudo_beta_fn(jnp.asarray(aatype), jnp.asarray(positions), None), **cfg['prev_pos']))
    got = np.asarray(embeddings.recycle_distogram(mx.array(aatype), mx.array(positions), cfg['prev_pos'], mx.float32))
    assert want.shape == got.shape and assert_live(want) is None and np.array_equal(got, want)

def real_batch(prev_scale):
    ref = load_ref('small', 'fp32', 0)
    features = cases.model_inputs('small')
    length = features['aatype'].shape[0]
    rng = np.random.default_rng(3)
    prev = {'prev_msa_first_row': (prev_scale * rng.standard_normal((length, 256))).astype(np.float32), 'prev_pair': (prev_scale * rng.standard_normal((length, length, 128))).astype(np.float32),
            'prev_pos': (ref['out/structure_module/final_atom_positions'] if prev_scale else np.zeros((length, 37, 3), np.float32)).astype(np.float32)}
    return {k: np.asarray(v) for k, v in features.items() if k != 'prev'}, prev

@pytest.mark.integration
@pytest.mark.parametrize('prev_scale', (0.0, 0.5))
def test_embeddings_and_one_block_stacks_match(params, prev_scale):
    features, prev = real_batch(prev_scale)
    want = jax_units.run_unit(lambda c, g: modules_multimer.EmbeddingsAndEvoformer(c.embeddings_and_evoformer, g, name='evoformer'), 'evoformer', 'evoformer', [{**features, **prev}], adjust=cut_to_one_block, keep_layers=1)
    got = embeddings.embeddings_and_evoformer(params.scope('evoformer'), {k: mx.array(v) for k, v in features.items()}, {k: mx.array(v) for k, v in prev.items()}, embedding_config().to_dict(), dtype=mx.float32, num_block=1, extra_num_block=1)
    assert set(got) == {'single', 'pair', 'msa', 'msa_first_row'} and set(want) >= set(got)
    for name in got:
        assert_live(want[name])
        assert rel_err(np.asarray(got[name]), want[name]) < 1e-4, name

@pytest.mark.integration
def test_extra_msa_features_match_the_jax_function():
    rng = np.random.default_rng(5)
    batch = {'extra_msa': rng.integers(0, 23, (2, 9)).astype(np.int32), 'extra_deletion_value': (3 * rng.random((2, 9))).astype(np.float32), 'extra_msa_mask': np.ones((2, 9), np.float32)}
    want, want_mask = modules_multimer.create_extra_msa_feature({k: jnp.asarray(v) for k, v in batch.items()}, 2)
    got, got_mask = embeddings.extra_msa_features({k: mx.array(v) for k, v in batch.items()}, 2)
    assert rel_err(np.asarray(got), np.asarray(want)) < 1e-6 and np.array_equal(np.asarray(got_mask), np.asarray(want_mask)) and got.shape[-1] == 25
