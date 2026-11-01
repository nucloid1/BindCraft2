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

# ----------------------------------------------------------------------------------------------------------------------------------------------------- Task 19: structure module, heads, model
import os

from bindcraft.af.alphafold.model import folding, prng, quat_affine as af_quat
from bindcraft.af.mlx_af2 import monomer_structure
from bindcraft.af.mlx_af2.outputs import OUTPUT_KEYS
from helpers import cosine

STRUCTURE_SCOPE = 'structure_module/fold_iteration'

def random_affine(seed, n=24):
    #unit quaternions and translations in the structure module's units (angstroms / position_scale)
    rng = np.random.default_rng(seed)
    quaternion = rng.standard_normal((n, 4)).astype(np.float32)
    quaternion /= np.linalg.norm(quaternion, axis=-1, keepdims=True)
    return np.concatenate([quaternion, (0.5 * rng.standard_normal((n, 3))).astype(np.float32)], axis=-1)

def test_quaternion_affine_operations_match_quat_affine():
    rng = np.random.default_rng(2)
    tensor = random_affine(0)
    update = (0.3 * rng.standard_normal((24, 6))).astype(np.float32)
    points = [(2.0 * rng.standard_normal((24, 5))).astype(np.float32) for _ in range(3)]
    want = af_quat.QuatAffine.from_tensor(jnp.asarray(tensor))
    got = monomer_structure.Affine.from_tensor(mx.array(tensor))
    assert rel_err(np.asarray(got.rotation), np.stack([np.stack([np.asarray(x) for x in row], -1) for row in want.rotation], -2)) < 1e-6
    for name, mine, theirs in (('pre_compose', got.pre_compose(mx.array(update)), want.pre_compose(jnp.asarray(update))), ('scale_translation', got.scale_translation(10.0), want.scale_translation(10.0))):
        assert rel_err(np.asarray(mine.to_tensor()), np.asarray(theirs.to_tensor())) < 1e-6, name
    for name, mine, theirs in (('apply_to_point', got.apply_to_point([mx.array(x) for x in points], 1), want.apply_to_point([jnp.asarray(x) for x in points], extra_dims=1)),
                               ('invert_point', got.invert_point([mx.array(x) for x in points], 1), want.invert_point([jnp.asarray(x) for x in points], extra_dims=1))):
        for axis in range(3):
            assert rel_err(np.asarray(mine[axis]), np.asarray(theirs[axis])) < 1e-6, (name, axis)

def structure_inputs(seed, n=24):
    rng = np.random.default_rng(seed)
    return {'act': (0.5 * rng.standard_normal((n, 384))).astype(np.float32), 'initial_act': (0.5 * rng.standard_normal((n, 384))).astype(np.float32), 'pair': (0.5 * rng.standard_normal((n, n, 128))).astype(np.float32),
            'mask': (np.arange(n) < n - 2).astype(np.float32)[:, None], 'affine': random_affine(seed, n), 'aatype': rng.integers(0, 20, n).astype(np.int32)}

@pytest.mark.integration
def test_monomer_ipa_and_fold_iteration_match():
    x = structure_inputs(1)
    cfg = config().to_dict()['heads']['structure_module']
    scope = params('model_1_ptm').scope(STRUCTURE_SCOPE)

    def build_ipa(c, g):
        module = folding.InvariantPointAttention(c.heads.structure_module, g)
        return lambda act, pair, mask, affine: module(inputs_1d=act, inputs_2d=pair, mask=mask, affine=af_quat.QuatAffine.from_tensor(affine))

    want = jax_units.run_unit(build_ipa, f'{STRUCTURE_SCOPE}/invariant_point_attention', 'invariant_point_attention', [x['act'], x['pair'], x['mask'], x['affine']], model_name='model_1_ptm')
    got = monomer_structure.invariant_point_attention(scope.scope('invariant_point_attention'), mx.array(x['act']), mx.array(x['pair']), mx.array(x['mask']), monomer_structure.Affine.from_tensor(mx.array(x['affine'])), cfg)
    assert_live(want) is None and rel_err(np.asarray(got), want) < 1e-5, rel_err(np.asarray(got), want)

    def build_fold(c, g):
        module = folding.FoldIteration(c.heads.structure_module, g, name='fold_iteration')
        return lambda act, affine, mask, initial_act, pair, aatype: module({'act': act, 'affine': affine}, sequence_mask=mask, update_affine=True, initial_act=initial_act, use_dropout=jnp.asarray(False),
                                                                             safe_key=prng.SafeKey(jax.random.PRNGKey(0)), static_feat_2d=pair, aatype=aatype)

    (want_act_and_affine, want_out) = jax_units.run_unit(build_fold, STRUCTURE_SCOPE, 'fold_iteration', [x['act'], x['affine'], x['mask'], x['initial_act'], x['pair'], x['aatype']], model_name='model_1_ptm')
    act, affine, sc = monomer_structure.fold_iteration(scope, mx.array(x['act']), monomer_structure.Affine.from_tensor(mx.array(x['affine'])), mx.array(x['mask']), mx.array(x['initial_act']), mx.array(x['pair']),
                                                       mx.array(x['aatype']), cfg)
    assert rel_err(np.asarray(act), want_act_and_affine['act']) < 1e-5 and rel_err(np.asarray(affine.to_tensor()), want_act_and_affine['affine']) < 1e-5
    assert rel_err(np.asarray(sc['angles_sin_cos']), want_out['sc']['angles_sin_cos']) < 1e-5 and rel_err(np.asarray(sc['unnormalized_angles_sin_cos']), want_out['sc']['unnormalized_angles_sin_cos']) < 1e-5
    for axis, name in enumerate('xyz'):
        assert rel_err(np.asarray(getattr(sc['atom_pos'], name)), getattr(want_out['sc']['atom_pos'], name)) < 1e-4, name  #angstroms, from frames ten times the unit
        assert rel_err(np.asarray(sc['backb_to_global'].translation.to_array()[..., axis]), getattr(want_out['sc']['backb_to_global'].trans, name)) < 1e-5, name

def reference_features_and_rows(model_name, case, index, monkeypatch):
    ref = load_ref(case, f'fp32__{model_name}', index)
    if monomer.uses_templates(config(model_name).to_dict()):  #the reference's own jitted torsion rows (see the Task 18 finding)
        rows = {name: mx.array(np.asarray(value)) for name, value in jit_torsion_rows(ref).items()}
        monkeypatch.setattr(all_atom, 'atom37_to_torsion_angles', lambda *arguments, **keywords: rows)
    return ref, ref_features(ref)

@pytest.mark.integration
@pytest.mark.parametrize('case', ('small', 'medium'))
@pytest.mark.parametrize('model_name', PTM_WITH_TEMPLATES)
def test_monomer_final_positions_and_mask_match(model_name, case):
    #the structure module on the CAPTURED representations of the no-recycle call: every output of the head by name. Dead residues (seq_mask 0, the padding) and the atoms a residue type does not
    #have carry values nothing constrains (3e-3 to 1e-2 apart here, while everything live agrees to 1e-6), so the comparisons are on live residues and existing atoms, as in the multimer padding tests.
    ref = load_ref(case, f'fp32__{model_name}', 1)
    features = ref_features(ref)
    live = ref['in/seq_mask'] > 0
    exists14 = ref['in/atom14_atom_exists'][live] > 0
    representations = {name: mx.array(ref[f'out/representations/{name}']) for name in ('single', 'pair')}
    got = monomer_structure.structure_module(params(model_name).scope('structure_module'), representations, features, config(model_name).to_dict()['heads']['structure_module'])
    got = {name: (value if name == 'sidechains' else np.asarray(value)) for name, value in got.items()}
    prefix = 'out/structure_module/'
    assert rel_err(got['traj'][:, live], ref[prefix + 'traj'][:, live]) < 1e-5 and rel_err(got['final_affines'][live], ref[prefix + 'final_affines'][live]) < 1e-5
    assert got['final_atom_mask'].shape == ref[prefix + 'final_atom_mask'].shape and np.array_equal(got['final_atom_mask'], ref[prefix + 'final_atom_mask'])
    assert np.array_equal(got['final_atom14_mask'], ref[prefix + 'final_atom14_mask'])
    assert_live(ref[prefix + 'final_atom_positions'], 'final_atom_positions')
    assert rel_err(got['final_atom_positions'], ref[prefix + 'final_atom_positions']) < 1e-5, rel_err(got['final_atom_positions'], ref[prefix + 'final_atom_positions'])  #the whole array: dead rows are zeroed
    assert rel_err(got['final_atom14_positions'][live][exists14], ref[prefix + 'final_atom14_positions'][live][exists14]) < 1e-5
    assert rel_err(got['act'][live], ref['out/representations/structure_module'][live]) < 1e-5
    for name in ('angles_sin_cos', 'unnormalized_angles_sin_cos'):
        assert rel_err(np.asarray(got['sidechains'][name])[:, live], ref[f'{prefix}sidechains/{name}'][:, live]) < 1e-5, name
    for axis, name in enumerate('xyz'):
        assert rel_err(np.asarray(got['sidechains']['atom_pos'])[:, live][:, exists14, axis], ref[f'{prefix}sidechains/atom_pos/{name}'][:, live][:, exists14]) < 1e-5, name

@pytest.mark.integration
@pytest.mark.parametrize('case', ('small', 'medium'))
@pytest.mark.parametrize('model_name', PTM_WITH_TEMPLATES)
def test_monomer_full_forward_all_output_keys(model_name, case, monkeypatch):
    ref, features = reference_features_and_rows(model_name, case, 0, monkeypatch)
    assert ref['meta']['num_recycle'] == 1
    model = monomer.MonomerModel(model_name, params(model_name), config(model_name).to_dict(), mx.float32)
    got, prev = model.forward(features, mx.array(np.asarray(ref['meta']['key'], np.uint32)), 1)
    assert set(got) == set(OUTPUT_KEYS)
    live = ref['in/seq_mask'] > 0
    per_residue = ('predicted_lddt/logits', 'experimentally_resolved/logits')  #read off the structure module's activations, which nothing constrains for the dead (padding) residues
    for name in OUTPUT_KEYS:
        want = ref[f'out/{name}']
        assert got[name].shape == want.shape, name
        assert_live(want, name)
        error = rel_err(np.asarray(got[name])[live] if name in per_residue else np.asarray(got[name]), want[live] if name in per_residue else want)
        print(model_name, case, name, f'{error:.2e}')
        assert error < 1e-3, (name, error)
    assert set(prev) == {'prev_msa_first_row', 'prev_pair', 'prev_pos'}

@pytest.mark.integration
@pytest.mark.parametrize('case', ('small', 'medium'))
@pytest.mark.parametrize('model_name', PTM_WITH_TEMPLATES)
def test_monomer_bf16_is_no_worse_than_three_times_the_jax_bf16_error(model_name, case):
    #A loose gate on purpose. The JAX-CPU bf16 model is flattered by XLA's excess precision (checkpoints.md), so a strict MLX bf16 model sits 1.3x to 1.5x above it on the pair-derived outputs, which
    #are stable and are gated at three times the JAX error. Positions and pLDDT are chaotic (a medium-case JAX bf16 ensemble spans 3x or more, and one JAX draw is the low end of its own range), so
    #for those the gate is a ceiling for a gross error, 0.25. The monomer ops are the multimer's (strict-ensemble gate there) and fp32 parity is 1e-7. Errors are against the fp32 reference.
    fp32, bf16 = load_ref(case, f'fp32__{model_name}', 0), load_ref(case, f'bf16__{model_name}', 0)
    model = monomer.MonomerModel(model_name, load_params(model_name, cases.params_path(), dtype=mx.bfloat16), config(model_name).to_dict(), mx.bfloat16)
    got, _ = model.forward(ref_features(bf16), mx.array(np.asarray(bf16['meta']['key'], np.uint32)), 1)
    chaotic = ('structure_module/final_atom_positions', 'predicted_lddt/logits')
    for name in OUTPUT_KEYS:
        if name.endswith(('final_atom_mask', 'breaks')):
            continue
        jax_error, mlx_error = rel_err(bf16[f'out/{name}'], fp32[f'out/{name}']), rel_err(np.asarray(got[name]), fp32[f'out/{name}'])
        print(model_name, case, name, f'mlx {mlx_error:.3g} jax {jax_error:.3g}')
        assert mlx_error <= (0.25 if name in chaotic else 3 * max(jax_error, 1e-3)), (name, mlx_error, jax_error)

def test_monomer_models_have_no_gradient_by_name():
    model = monomer.MonomerModel('model_1_ptm', params('model_1_ptm'), config('model_1_ptm').to_dict(), mx.float32)
    with pytest.raises(NotImplementedError, match='model_1_ptm'):
        model.vjp({}, mx.array(np.asarray([0, 1], np.uint32)), 1, {}, ('msa_feat',))
    with pytest.raises(NotImplementedError, match='initial_atom_pos'):
        model.forward({'initial_atom_pos': mx.zeros((3, 37, 3)), 'aatype': mx.zeros((3,), mx.int32)}, mx.array(np.asarray([0, 1], np.uint32)), 0)

def test_default_validation_model_builds_under_mlx(monkeypatch):
    #a default-settings campaign builds a multimer design model and a monomer validation model through the repo's own factory; neither is refused (no worker starts: a fake remote)
    from bindcraft import af2
    from bindcraft.af.mlx_af2 import predictor
    from bindcraft.backend import design_model_class
    from test_predictor import FakeRemote
    FakeRemote.instance = None
    monkeypatch.setenv('BINDCRAFT_BACKEND', 'mlx')
    monkeypatch.setenv(predictor.DEVICE_ENV, 'cpu')
    monkeypatch.setattr(predictor, 'RemoteModel', FakeRemote)
    monkeypatch.setattr(predictor, 'session', None)
    monkeypatch.setattr(predictor.preflight, 'checks', lambda: None)  #the disk is not what is under test
    assert design_model_class() is predictor.MlxAlphaFoldDesignModel
    arguments = dict(data_dir=cases.params_path(), max_cache_size=16, dropout=False, length_bucket_size=32)
    design = design_model_class()(presets=af2.MULTIMER_POOL[:3], num_recycle=1, **arguments)
    validation = design_model_class()(presets=af2.MONOMER_POOL, num_recycle=3, **arguments)
    assert design.remote is validation.remote is FakeRemote.instance
    assert FakeRemote.instance.loaded == list(af2.MULTIMER_POOL[:3]) + list(af2.MONOMER_POOL)
