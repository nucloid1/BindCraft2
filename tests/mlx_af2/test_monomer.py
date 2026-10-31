"""The AlphaFold monomer (pTM) models on MLX, judged against the vendored JAX modules and the captured seam references (capture_refs.py --monomer): the torsion angles of the template rows, the template
embedding, the embeddings with one-block stacks, and the trunk representations of whole models (fp32). model_1_ptm and model_2_ptm embed templates and their torsion angles, model_3_ptm has no templates
(config.CONFIG_DIFFS)."""
import jax
import jax.numpy as jnp
import mlx.core as mx
import numpy as np
import pytest

import cases
import jax_units
from bindcraft.af.alphafold.model import all_atom as af_all_atom, modules as af_modules
from bindcraft.af.mlx_af2 import all_atom, monomer
from bindcraft.af.mlx_af2.weights import load_params
from helpers import assert_live, load_ref, rel_err

PTM_WITH_TEMPLATES = ('model_1_ptm', 'model_2_ptm')
PTM = PTM_WITH_TEMPLATES + ('model_3_ptm',)
_params: dict = {}

def params(model_name):
    if model_name not in _params:
        _params[model_name] = load_params(model_name, cases.params_path(), dtype=mx.float32)
    return _params[model_name]

def config(model_name='model_1_ptm'):
    return jax_units.configs(model_name)[0]

def ref_features(ref):
    return {name[3:]: mx.array(value) for name, value in ref.items() if name.startswith('in/')}

def random_structure(seed, templates=2, residues=20):
    #atom37 coordinates with whole residues and single atoms masked, residue types 0 to 21 (the unknown types included): the torsion code must agree where atoms are missing too
    rng = np.random.default_rng(seed)
    aatype = rng.integers(0, 22, (templates, residues)).astype(np.int32)
    positions = (3.0 * rng.standard_normal((templates, residues, 37, 3))).astype(np.float32)
    mask = (rng.random((templates, residues, 37)) > 0.15).astype(np.float32)
    mask[:, 3] = 0.0
    return aatype, positions * mask[..., None], mask

@pytest.mark.integration
@pytest.mark.parametrize('placeholder', (False, True))
@pytest.mark.parametrize('source', ('real', 'random'))
def test_torsion_angles_match_the_jax_function(source, placeholder):
    if source == 'real':
        ref = load_ref('small', 'fp32__model_1_ptm', 0)
        aatype, positions, mask = ref['in/template_aatype'], ref['in/template_all_atom_positions'], ref['in/template_all_atom_mask']
    else:
        aatype, positions, mask = random_structure(11)
    want = af_all_atom.atom37_to_torsion_angles(jnp.asarray(aatype), jnp.asarray(positions), jnp.asarray(mask), placeholder_for_undefined=placeholder)
    got = all_atom.atom37_to_torsion_angles(mx.array(aatype), mx.array(positions), mx.array(mask), placeholder_for_undefined=placeholder)
    assert set(got) == set(want) == {'torsion_angles_sin_cos', 'alt_torsion_angles_sin_cos', 'torsion_angles_mask'}
    for name in want:
        assert_live(np.asarray(want[name]), name)
        assert got[name].shape == want[name].shape and rel_err(np.asarray(got[name]), np.asarray(want[name])) < 1e-5, name

def template_inputs(ref, tile_over_chains):
    #the captured template (the target's atoms only); tile_over_chains puts the target's atoms on every residue, so the interchain mask has something to mask
    features = {name[3:]: np.asarray(value) for name, value in ref.items() if name.startswith('in/')}
    if tile_over_chains:
        present = np.where(features['template_all_atom_mask'][0].any(-1))[0]
        take = present[np.arange(features['template_aatype'].shape[1]) % len(present)]
        for name in ('template_aatype', 'template_all_atom_positions', 'template_all_atom_mask'):
            features[name] = features[name][:, take]
        beta, beta_mask = af_modules.pseudo_beta_fn(jnp.asarray(features['template_aatype']), jnp.asarray(features['template_all_atom_positions']), jnp.asarray(features['template_all_atom_mask']))
        features['template_pseudo_beta'], features['template_pseudo_beta_mask'] = np.asarray(beta), np.asarray(beta_mask)
    return features

@pytest.mark.integration
@pytest.mark.parametrize('tile_over_chains, mask_interchain', ((False, True), (True, True), (True, False)))
def test_monomer_template_embedding_matches(tile_over_chains, mask_interchain):
    ref = load_ref('small', 'fp32__model_1_ptm', 0)
    features = template_inputs(ref, tile_over_chains)
    length = features['aatype'].shape[0]
    rng = np.random.default_rng(4)
    query = (0.5 * rng.standard_normal((length, length, 128))).astype(np.float32)
    mask_2d = (features['seq_mask'][:, None] * features['seq_mask'][None, :]).astype(np.float32)
    same_chain = features['asym_id'][:, None] == features['asym_id'][None, :]
    multichain = np.where(mask_interchain, same_chain, True)
    template_batch = {name: features[name] for name in features if name.startswith('template_')}
    want = jax_units.run_unit(lambda c, g: af_modules.TemplateEmbedding(c.embeddings_and_evoformer.template, g), 'evoformer/template_embedding', 'template_embedding',
                              [query, template_batch, mask_2d, multichain, False], model_name='model_1_ptm')
    got = monomer.template_embedding(params('model_1_ptm').scope('evoformer/template_embedding'), mx.array(query), {name: mx.array(value) for name, value in template_batch.items()}, mx.array(mask_2d),
                                     mx.array(multichain), config().to_dict()['embeddings_and_evoformer']['template'])
    assert_live(want)
    assert got.shape == want.shape and rel_err(np.asarray(got), want) < 1e-4, rel_err(np.asarray(got), want)

def test_the_interchain_mask_changes_the_template_embedding_when_templates_span_chains():
    #control for the test above: with the target's atoms on every residue the mask is not a no-op, so the (True, False) case pins it
    ref = load_ref('small', 'fp32__model_1_ptm', 0)
    features = template_inputs(ref, True)
    length = features['aatype'].shape[0]
    query = (0.5 * np.random.default_rng(4).standard_normal((length, length, 128))).astype(np.float32)
    mask_2d = (features['seq_mask'][:, None] * features['seq_mask'][None, :]).astype(np.float32)
    same_chain = features['asym_id'][:, None] == features['asym_id'][None, :]
    template_batch = {name: mx.array(features[name]) for name in features if name.startswith('template_')}
    cfg = config().to_dict()['embeddings_and_evoformer']['template']
    scope = params('model_1_ptm').scope('evoformer/template_embedding')
    intra, everything = (monomer.template_embedding(scope, mx.array(query), template_batch, mx.array(mask_2d), mx.array(mask), cfg) for mask in (same_chain, np.ones_like(same_chain)))
    assert rel_err(np.asarray(intra), np.asarray(everything)) > 1e-3

def cut_to_one_block(model_config, templates):
    model_config.global_config.use_dgram = False  #the repo's runner sets this (AlphaFoldDesignModel._alphafold_runner); the vendored monomer module reads it
    model_config.embeddings_and_evoformer.evoformer_num_block = 1
    model_config.embeddings_and_evoformer.extra_msa_stack_num_block = 1
    model_config.embeddings_and_evoformer.template.template_pair_stack.num_block = 1
    model_config.embeddings_and_evoformer.template.enabled = templates

def real_batch(model_name, prev_scale):
    ref = load_ref('small', f'fp32__{model_name}', 0)
    features = {name[3:]: np.asarray(value) for name, value in ref.items() if name.startswith('in/')}
    length = features['aatype'].shape[0]
    rng = np.random.default_rng(3)
    prev = {'prev_msa_first_row': (prev_scale * rng.standard_normal((length, 256))).astype(np.float32), 'prev_pair': (prev_scale * rng.standard_normal((length, length, 128))).astype(np.float32),
            'prev_pos': (ref['out/structure_module/final_atom_positions'] if prev_scale else np.zeros((length, 37, 3), np.float32)).astype(np.float32)}
    return features, prev

@pytest.mark.integration
@pytest.mark.parametrize('model_name, templates, prev_scale', (('model_1_ptm', True, 0.5), ('model_1_ptm', False, 0.0), ('model_3_ptm', False, 0.5)))
def test_monomer_embeddings_match(model_name, templates, prev_scale):
    #the whole front end (embeddings, template pair rows and torsion rows, extra MSA stack, main stack, single activations) with one-block stacks, real features and a nonzero previous state
    features, prev = real_batch(model_name, prev_scale)
    want = jax_units.run_unit(lambda c, g: af_modules.EmbeddingsAndEvoformer(c.embeddings_and_evoformer, g, name='evoformer'), 'evoformer', 'evoformer', [{**features, **prev}],
                              adjust=lambda c: cut_to_one_block(c, templates), keep_layers=1, model_name=model_name)
    cfg = config(model_name).to_dict()
    cfg['embeddings_and_evoformer']['template']['enabled'] = templates
    cfg['embeddings_and_evoformer']['template']['template_pair_stack']['num_block'] = 1
    got = monomer.embeddings_and_evoformer(params(model_name).scope('evoformer'), {name: mx.array(value) for name, value in features.items()}, {name: mx.array(value) for name, value in prev.items()},
                                           cfg['embeddings_and_evoformer'], dtype=mx.float32, num_block=1, extra_num_block=1)
    assert set(got) == {'single', 'pair', 'msa', 'msa_first_row'} and set(want) >= set(got)
    for name in got:
        assert_live(want[name])
        assert got[name].shape == want[name].shape and rel_err(np.asarray(got[name]), want[name]) < 1e-4, name
    if templates:
        assert want['msa'].shape[0] == features['msa_feat'].shape[0]  #the template rows are cropped from the returned MSA

def jit_torsion_rows(ref):
    #the vendored function as XLA's jit runs it, which is how the seam reference was captured; see the test below for what that changes
    arrays = [jnp.asarray(ref[f'in/{name}']) for name in ('template_aatype', 'template_all_atom_positions', 'template_all_atom_mask')]
    return jax.jit(lambda a, p, m: af_all_atom.atom37_to_torsion_angles(a, p, m, placeholder_for_undefined=False))(*arrays)

@pytest.mark.integration
@pytest.mark.parametrize('case', ('small', 'medium'))
@pytest.mark.parametrize('model_name', PTM_WITH_TEMPLATES)
def test_torsion_angles_disagree_with_jit_only_where_they_are_undefined(model_name, case):
    #The undefined angles (mask 0) of a degenerate frame are rounding residue: the pre-omega of the first target residue sits after residues at zero positions, so its three points are collinear and
    #the epsilon-regularised normalisation turns whatever residue the arithmetic leaves into a unit vector. Eagerly JAX gets (0, 0) and so does MLX; XLA's jit left (0.7455, 0.6664) in the captured
    #reference. Every DEFINED angle agrees to 1e-5 under jit, and the disagreement is exactly the undefined entries listed here.
    ref = load_ref(case, f'fp32__{model_name}', 1)
    jitted = jit_torsion_rows(ref)
    got = all_atom.atom37_to_torsion_angles(mx.array(ref['in/template_aatype']), mx.array(ref['in/template_all_atom_positions']), mx.array(ref['in/template_all_atom_mask']))
    difference = np.abs(np.asarray(got['torsion_angles_sin_cos']) - np.asarray(jitted['torsion_angles_sin_cos'])).max(-1)
    defined = np.asarray(jitted['torsion_angles_mask']) > 0
    assert difference[defined].max() < 1e-5, 'a defined torsion angle differs'
    assert (difference[~defined] > 1e-5).sum() <= 2, 'more undefined entries differ than the chain starts of the template'

@pytest.mark.integration
@pytest.mark.parametrize('case', ('small', 'medium'))
@pytest.mark.parametrize('model_name', PTM)
def test_monomer_trunk_representations_match(model_name, case, monkeypatch):
    ref = load_ref(case, f'fp32__{model_name}', 1)  #no recycle: the previous state is zeros
    assert ref['meta']['num_recycle'] == 0 and ref['meta']['model_name'] == model_name
    cfg = config(model_name).to_dict()
    if monomer.uses_templates(cfg):  #the reference's own torsion rows, so the comparison covers everything except the rounding residue of the undefined angles (the test above)
        rows = {name: mx.array(np.asarray(value)) for name, value in jit_torsion_rows(ref).items()}
        monkeypatch.setattr(all_atom, 'atom37_to_torsion_angles', lambda *arguments, **keywords: rows)
    model = monomer.MonomerModel(model_name, params(model_name), cfg, mx.float32)
    features = ref_features(ref)
    got = model.trunk(features, model.zero_prev(features))
    for name in ('single', 'pair', 'msa', 'msa_first_row'):
        want = ref[f'out/representations/{name}']
        assert_live(want, name)
        assert got[name].shape == want.shape and rel_err(np.asarray(got[name]), want) < 1e-3, (name, rel_err(np.asarray(got[name]), want))

def test_both_ptm_models_and_their_config_diffs():
    for model_name in PTM_WITH_TEMPLATES:
        template = config(model_name).to_dict()['embeddings_and_evoformer']['template']
        assert template['enabled'] and template['embed_torsion_angles'], model_name
    template = config('model_3_ptm').to_dict()['embeddings_and_evoformer']['template']
    assert not template['enabled'], 'model_3_ptm has no templates'
    assert monomer.uses_templates(config('model_1_ptm').to_dict()) and not monomer.uses_templates(config('model_3_ptm').to_dict())
    with pytest.raises(NotImplementedError, match='use_template_unit_vector'):
        unit = config('model_1_ptm').to_dict()
        unit['embeddings_and_evoformer']['template']['use_template_unit_vector'] = True
        monomer.check_supported(unit)
