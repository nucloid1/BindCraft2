import haiku as hk
import jax.numpy as jnp
import mlx.core as mx
import numpy as np
import pytest

import cases
import jax_units
from bindcraft import af2
from bindcraft.af.alphafold.model import all_atom_multimer, folding_multimer, geometry as jgeo, modules_multimer
from bindcraft.af.mlx_af2 import all_atom, embeddings, geometry, templates
from bindcraft.af.mlx_af2.weights import load_params
from helpers import assert_live, load_ref, rel_err

EVO = 'evoformer'

@pytest.fixture(scope='module')
def params():
    return load_params(jax_units.MODEL, cases.params_path(), dtype=mx.float32)

def template_config():
    return jax_units.configs()[0].embeddings_and_evoformer.template

def real_template_batch():
    ref = load_ref('small', 'fp32', 0)
    return ref, {name: ref[f'in/{name}'] for name in ('template_aatype', 'template_all_atom_positions', 'template_all_atom_mask')}

def test_dihedral_angle_matches_jax():
    rng = np.random.default_rng(1)
    points = [rng.standard_normal((3, 9)).astype(np.float32) * 3 for _ in range(4)]
    got = geometry.dihedral_angle(*[geometry.Vec3(*[mx.array(c) for c in p]) for p in points])
    want = jgeo.dihedral_angle(*[jgeo.Vec3Array(*[jnp.asarray(c) for c in p]) for p in points])
    assert rel_err(np.asarray(got), np.asarray(want)) < 1e-6

@pytest.mark.integration
def test_chi_angles_and_mask_match_jax_on_the_real_target():
    ref, batch = real_template_batch()
    positions, mask, aatype = batch['template_all_atom_positions'][0], batch['template_all_atom_mask'][0], batch['template_aatype'][0]
    want_angles, want_mask = all_atom_multimer.compute_chi_angles(jgeo.Vec3Array.from_array(jnp.asarray(positions)), jnp.asarray(mask), jnp.asarray(aatype))
    got_angles, got_mask = all_atom.compute_chi_angles(mx.array(positions), mx.array(mask), mx.array(aatype))
    assert np.asarray(want_mask).sum() > 100 and np.array_equal(np.asarray(got_mask), np.asarray(want_mask))
    live = np.asarray(want_mask) > 0
    assert rel_err(np.asarray(got_angles)[live], np.asarray(want_angles)[live]) < 1e-5

@pytest.mark.integration
def test_backbone_affine_matches_jax():
    _, batch = real_template_batch()
    positions, mask, aatype = batch['template_all_atom_positions'][0], batch['template_all_atom_mask'][0], batch['template_aatype'][0]
    want, want_mask = folding_multimer.make_backbone_affine(jgeo.Vec3Array.from_array(jnp.asarray(positions)), jnp.asarray(mask), jnp.asarray(aatype))
    got, got_mask = all_atom.make_backbone_affine(geometry.Vec3.from_array(mx.array(positions)), mx.array(mask), mx.array(aatype))
    assert np.array_equal(np.asarray(got_mask), np.asarray(want_mask)) and rel_err(np.asarray(got.to_array()), np.asarray(want.to_array())) < 1e-5

@pytest.mark.integration
def test_template_embedding_1d_matches(params):
    _, batch = real_template_batch()
    def build(c, g):
        def run(batch):
            with hk.experimental.name_scope(EVO):
                return modules_multimer.template_embedding_1d(batch, c.embeddings_and_evoformer.msa_channel, g)
        return run
    want_act, want_mask = jax_units.run_unit(build, EVO, EVO, [batch])
    got_act, got_mask = templates.template_embedding_1d(params.scope(EVO), {k: mx.array(v) for k, v in batch.items()}, 256, mx.float32)
    assert assert_live(want_act) is None and want_act.shape == tuple(got_act.shape) == (1, 147, 256)
    assert rel_err(np.asarray(got_act), want_act) < 1e-5 and np.array_equal(np.asarray(got_mask), want_mask)

def pair_inputs(ref, interchain):
    rng = np.random.default_rng(2)
    n = ref['in/aatype'].shape[0]
    query = (0.5 * rng.standard_normal((n, n, 128))).astype(np.float32)
    seq_mask = ref['in/seq_mask']
    padding = np.outer(seq_mask, seq_mask).astype(np.float32)
    asym = ref['in/asym_id']
    multichain = (asym[:, None] == asym[None, :]) if interchain else np.ones((n, n), bool)
    return query, padding, multichain

@pytest.mark.integration
@pytest.mark.parametrize('mask_template_interchain', (True, False))
def test_template_pair_embedding_matches(params, mask_template_interchain):
    ref, batch = real_template_batch()
    query, padding, multichain = pair_inputs(ref, mask_template_interchain)
    scope = f'{EVO}/template_embedding'
    def build(c, g):
        module = modules_multimer.TemplateEmbedding(c.embeddings_and_evoformer.template, g, name='template_embedding')
        return lambda q, tb, pm, mm: module(q, tb, pm, mm, jnp.asarray(False))
    want = jax_units.run_unit(build, scope, 'template_embedding', [query, batch, padding, multichain])
    assert_live(want)
    got = np.asarray(templates.template_embedding(params.scope(scope), mx.array(query), {k: mx.array(v) for k, v in batch.items()}, mx.array(padding), mx.array(multichain), template_config().to_dict()))
    assert got.shape == want.shape == (147, 147, 128) and rel_err(got, want) < 1e-4

def filled_template_batch():
    #a template with atoms on BOTH chains: a real design's template has atoms on the target only (the binder's are masked), so there the interchain mask has nothing to remove
    ref, batch = real_template_batch()
    present = np.where(batch['template_all_atom_mask'][0][:, 1] > 0)[0]
    source = present[np.arange(batch['template_aatype'].shape[1]) % len(present)]
    return ref, {name: value[:, source] for name, value in batch.items()}

def run_pair_embedding(params, ref, batch, interchain):
    query, padding, multichain = pair_inputs(ref, interchain)
    scope = f'{EVO}/template_embedding'
    def build(c, g):
        module = modules_multimer.TemplateEmbedding(c.embeddings_and_evoformer.template, g, name='template_embedding')
        return lambda q, tb, pm, mm: module(q, tb, pm, mm, jnp.asarray(False))
    want = jax_units.run_unit(build, scope, 'template_embedding', [query, batch, padding, multichain])
    got = np.asarray(templates.template_embedding(params.scope(scope), mx.array(query), {k: mx.array(v) for k, v in batch.items()}, mx.array(padding), mx.array(multichain), template_config().to_dict()))
    return got, want

@pytest.mark.integration
def test_a_real_design_template_is_unaffected_by_the_interchain_mask(params):
    ref, batch = real_template_batch()
    assert not batch['template_all_atom_mask'][0][:30].any()  #the binder residues carry no template atoms
    on, off = (run_pair_embedding(params, ref, batch, flag)[0] for flag in (True, False))
    assert rel_err(on, off) < 1e-6

@pytest.mark.integration
@pytest.mark.parametrize('mask_template_interchain', (True, False))
def test_template_with_atoms_on_both_chains_matches_and_the_interchain_mask_matters(params, mask_template_interchain):
    ref, batch = filled_template_batch()
    assert batch['template_all_atom_mask'][0][:, 1].all()
    got, want = run_pair_embedding(params, ref, batch, mask_template_interchain)
    assert rel_err(got, want) < 1e-4
    other, _ = run_pair_embedding(params, ref, batch, not mask_template_interchain)
    assert rel_err(got, other) > 1e-3  #masking the cross-chain template features changes the embedding, so this parity covers the mask

@pytest.mark.integration
def test_flexible_target_template_mask_matches_the_jax_pipeline(params):
    #target_flexibility removes template atoms from the fixed target at evenly spaced residues; the embedding must follow the changed mask exactly as the JAX one does
    a = cases.front_arguments('small')
    features = af2.target_template_features(jnp.argmax(a['sequence'], -1).astype(jnp.int32), a['atoms'], a['atom_mask'], a['flags'], a['residue_index'], target_flexibility=0.3)
    batch = {name: np.asarray(features[name]) for name in ('template_aatype', 'template_all_atom_positions', 'template_all_atom_mask')}
    rigid_batch = {name: np.asarray(cases.model_inputs('small')[name]) for name in batch}
    assert not np.array_equal(batch['template_all_atom_mask'], rigid_batch['template_all_atom_mask'])  #the flexible mask differs from the rigid one
    ref = load_ref('small', 'fp32', 0)
    query, padding, multichain = pair_inputs(ref, True)
    scope = f'{EVO}/template_embedding'
    def build(c, g):
        module = modules_multimer.TemplateEmbedding(c.embeddings_and_evoformer.template, g, name='template_embedding')
        return lambda q, tb, pm, mm: module(q, tb, pm, mm, jnp.asarray(False))
    want = jax_units.run_unit(build, scope, 'template_embedding', [query, batch, padding, multichain])
    got = np.asarray(templates.template_embedding(params.scope(scope), mx.array(query), {k: mx.array(v) for k, v in batch.items()}, mx.array(padding), mx.array(multichain), template_config().to_dict()))
    assert rel_err(got, want) < 1e-4

def cut_templates_to_one_block(config):
    config.embeddings_and_evoformer.evoformer_num_block = 1
    config.embeddings_and_evoformer.extra_msa_stack_num_block = 1
    config.embeddings_and_evoformer.template.template_pair_stack.num_block = 1

@pytest.mark.integration
def test_the_whole_front_end_with_templates_matches_with_one_block_stacks(params):
    ref = load_ref('small', 'fp32', 0)
    features = {k: np.asarray(v) for k, v in cases.model_inputs('small').items() if k != 'prev'}
    rng = np.random.default_rng(3)
    n = features['aatype'].shape[0]
    prev = {'prev_msa_first_row': (0.5 * rng.standard_normal((n, 256))).astype(np.float32), 'prev_pair': (0.5 * rng.standard_normal((n, n, 128))).astype(np.float32), 'prev_pos': ref['out/structure_module/final_atom_positions']}
    want = jax_units.run_unit(lambda c, g: modules_multimer.EmbeddingsAndEvoformer(c.embeddings_and_evoformer, g, name=EVO), EVO, EVO, [{**features, **prev}], adjust=cut_templates_to_one_block, keep_layers=1)
    cfg = jax_units.configs()[0].embeddings_and_evoformer.to_dict()
    cfg['template']['template_pair_stack']['num_block'] = 1
    mlx_features = {k: mx.array(v) for k, v in features.items()}
    hook = templates.template_hook(params.scope(EVO), mlx_features, cfg, mx.float32)
    got = embeddings.embeddings_and_evoformer(params.scope(EVO), mlx_features, {k: mx.array(v) for k, v in prev.items()}, cfg, dtype=mx.float32, num_block=1, extra_num_block=1, template_embedding=hook)
    assert got['msa'].shape[0] == 1 and want['msa'].shape[0] == 1  #the template rows are cropped from the returned MSA
    for name in got:
        assert_live(want[name])
        assert rel_err(np.asarray(got[name]), want[name]) < 1e-4, name
