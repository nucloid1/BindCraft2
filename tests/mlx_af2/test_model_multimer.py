import json
import os

import jax
import mlx.core as mx
import numpy as np
import pytest

import cases
import jax_units
from bindcraft.af.mlx_af2 import heads, model_multimer
from bindcraft.af.mlx_af2.outputs import OUTPUT_KEYS
from bindcraft.af.mlx_af2.weights import load_params
from helpers import assert_live, cosine, load_ref, rel_err

NOT_PORTED = ('masked_msa_head',)  #nothing reads the masked-MSA head's output: the loss and the metrics never touch it

def model_config():
    return jax_units.configs()[0].to_dict()

def build_model(dtype=mx.float32):
    return model_multimer.MultimerModel(jax_units.MODEL, load_params(jax_units.MODEL, cases.params_path(), dtype=dtype), model_config(), dtype)

def features_from(ref):
    features = {name[3:]: mx.array(value) for name, value in ref.items() if name.startswith('in/') and not name.startswith('in/prev')}
    return features

def run(model, ref, **overrides):
    features = {**features_from(ref), **overrides.pop('features', {})}
    key = np.asarray(ref['meta']['key'], np.uint32)
    outputs, prev = model.forward(features, mx.array(key), overrides.pop('num_recycle', ref['meta']['num_recycle']), **overrides)
    return {name: np.asarray(value) for name, value in outputs.items()}, prev

def assert_matches(got, ref, gate=1e-3):
    assert set(got) == set(OUTPUT_KEYS)
    for name in OUTPUT_KEYS:
        want = ref[f'out/{name}']
        assert got[name].shape == want.shape, name
        assert_live(want, name)
        assert rel_err(got[name], want) < gate, (name, rel_err(got[name], want))

@pytest.fixture(scope='module')
def model():
    return build_model()

@pytest.mark.integration
def test_heads_match_the_captured_outputs_from_the_captured_representations():
    ref = load_ref('small', 'fp32', 0)
    params = load_params(jax_units.MODEL, cases.params_path(), dtype=mx.float32)
    cfg = model_config()['heads']
    reps = {name: mx.array(ref[f'out/representations/{name}']) for name in ('single', 'pair', 'structure_module')}
    out = {'predicted_lddt': heads.predicted_lddt(params.scope('predicted_lddt_head'), reps, cfg['predicted_lddt']), 'predicted_aligned_error': heads.predicted_aligned_error(params.scope('predicted_aligned_error_head'), reps, cfg['predicted_aligned_error']),
           'distogram': heads.distogram(params.scope('distogram_head'), reps, cfg['distogram']), 'experimentally_resolved': heads.experimentally_resolved(params.scope('experimentally_resolved_head'), reps, cfg['experimentally_resolved'])}
    for head, name in (('predicted_lddt', 'logits'), ('predicted_aligned_error', 'logits'), ('predicted_aligned_error', 'breaks'), ('distogram', 'logits'), ('experimentally_resolved', 'logits')):
        assert rel_err(np.asarray(out[head][name]), ref[f'out/{head}/{name}']) < 1e-5, (head, name)
    assert np.array_equal(np.asarray(out['distogram']['logits']), np.asarray(out['distogram']['logits']).swapaxes(0, 1))  #the distogram is symmetric

@pytest.mark.integration
@pytest.mark.parametrize('case', ('small', 'medium'))
def test_full_forward_matches_every_output_key(model, case):
    ref = load_ref(case, 'fp32', 0)
    got, prev = run(model, ref)
    assert_matches(got, ref)
    pos, want = got['structure_module/final_atom_positions'], ref['out/structure_module/final_atom_positions']
    mask = ref['out/structure_module/final_atom_mask'] > 0
    #atom positions are judged against the JAX model's own sensitivity: how far its atoms move when its input is nudged by float32 rounding (1e-7 relative), recorded by capture_refs.py --noise-floor
    floor = json.load(open(os.path.join(os.path.expanduser(os.environ.get('BINDCRAFT_MLX_REFS', '~/.cache/bindcraft_mlx_refs')), 'spread.json')))[case]['position_noise_floor']
    deviation = np.linalg.norm(pos - want, axis=-1)[mask]
    assert deviation.max() <= max(3 * floor['max_angstrom'], 0.005), (deviation.max(), floor)
    assert deviation.mean() <= max(3 * floor['mean_angstrom'], 5e-4), (deviation.mean(), floor)
    assert set(prev) == {'prev_pos', 'prev_msa_first_row', 'prev_pair'} and rel_err(np.asarray(prev['prev_pair']), ref['out/prev/prev_pair']) < 1e-3

@pytest.mark.integration
def test_the_loaded_parameters_are_all_read(model):
    ref = load_ref('small', 'fp32', 0)
    run(model, ref)
    unused = [name for name in model.params.unused() if not name.startswith(NOT_PORTED)]
    assert not unused, unused[:6]

@pytest.mark.integration
@pytest.mark.parametrize('index, num_recycle', ((1, 0), (0, 1), (2, 3)))
def test_recycles_0_1_and_3_match(model, index, num_recycle):
    ref = load_ref('small', 'fp32', index)
    assert ref['meta']['num_recycle'] == num_recycle
    got, _ = run(model, ref)
    assert_matches(got, ref)

@pytest.mark.integration
def test_large_complex_initialization_updates_atom_pos_between_recycles(model):
    ref = load_ref('small', 'fp32', 3)
    assert ref['meta']['large_complex_initialization'] and 'in/initial_atom_pos' in ref
    got, _ = run(model, ref)
    assert_matches(got, ref)
    without = {name: value for name, value in features_from(ref).items() if name != 'initial_atom_pos'}
    other, _ = model.forward(without, mx.array(np.asarray(ref['meta']['key'], np.uint32)), 1)
    assert rel_err(np.asarray(other['structure_module/final_atom_positions']), got['structure_module/final_atom_positions']) > 1e-3  #the initial positions matter

@pytest.mark.integration
def test_padding_does_not_change_valid_positions_or_confidences(model):
    #valid means seq_mask > 0: the unpadded complex already holds two dead residues (the binder's own padding to 32), whose outputs nothing constrains
    ref = load_ref('small', 'fp32', 0)
    length = ref['in/aatype'].shape[0]
    live = ref['in/seq_mask'] > 0
    assert (~live).sum() == 2 and live[:length].sum() == length - 2
    padded = {name: mx.array(np.asarray(value)) for name, value in cases.model_inputs('small', extra_padding=32).items() if name != 'prev'}
    assert padded['aatype'].shape[0] == length + 32
    out, _ = model.forward(padded, mx.array(np.asarray(ref['meta']['key'], np.uint32)), 1)
    out = {name: np.asarray(value) for name, value in out.items()}
    pairs = np.ix_(live, live)
    assert rel_err(out['structure_module/final_atom_positions'][:length][live], ref['out/structure_module/final_atom_positions'][live]) < 1e-4
    assert rel_err(out['predicted_lddt/logits'][:length][live], ref['out/predicted_lddt/logits'][live]) < 1e-4
    assert rel_err(out['predicted_aligned_error/logits'][:length, :length][pairs], ref['out/predicted_aligned_error/logits'][pairs]) < 1e-4
    assert rel_err(out['distogram/logits'][:length, :length][pairs], ref['out/distogram/logits'][pairs]) < 1e-4
    assert rel_err(out['experimentally_resolved/logits'][:length][live], ref['out/experimentally_resolved/logits'][live]) < 1e-4
    assert not out['structure_module/final_atom_mask'][length:].any() and not out['structure_module/final_atom_positions'][length:].any()  #the added dead residues carry nothing

def spread_entry(case):
    path = os.path.join(os.path.expanduser(os.environ.get('BINDCRAFT_MLX_REFS', '~/.cache/bindcraft_mlx_refs')), 'spread.json')
    return json.load(open(path))[case]

@pytest.mark.integration
@pytest.mark.parametrize('case', ('small', 'medium'))
def test_bf16_within_the_jax_spread(case):
    #The spread of the shipped JAX bf16 model against fp32 is a RANGE of rounding realizations (capture_refs.py --bf16-ensemble --strict-rounding), because these outputs are chaotic: positions feed pLDDT,
    #and one draw is a noisy estimate of the noise itself. A correct port is one more draw, so it can exceed any single draw; a systematic problem shows as a large average ratio. Gates: the geometric mean
    #over the five float outputs of (MLX error / mean of the JAX ensemble) is at most 1.5, and no output exceeds three times the ensemble's maximum.
    #The reference is the STRICT-rounding ensemble: XLA's CPU compiler (xla_allow_excess_precision, on by default) keeps float32 between fused bfloat16 ops, so the default JAX-CPU band is more accurate
    #than bfloat16 arithmetic is (with the flag off JAX reproduces MLX to 1% at every depth of the real 48-block stack, docs/mlx_backend/checkpoints.md). It is printed, not gated.
    entry = spread_entry(case)
    assert 'bf16_ensemble_strict' in entry, 'run capture_refs.py --bf16-ensemble --strict-rounding first'
    assert entry['bf16_ensemble_strict']['allow_excess_precision'] is False
    ensemble = entry['bf16_ensemble_strict']['errors']
    default_band = entry['bf16_ensemble']['errors']
    ref = load_ref(case, 'fp32', 0)
    got, _ = run(build_model(mx.bfloat16), ref)
    keys = [name for name in OUTPUT_KEYS if max(ensemble[name]) > 0]
    errors = {name: rel_err(got[name], ref[f'out/{name}']) for name in keys}
    ratios = {name: errors[name] / np.mean(ensemble[name]) for name in keys}
    print(case, {name.split('/')[-2] + '/' + name.split('/')[-1]: f'mlx {errors[name]:.3g}, jax strict {min(ensemble[name]):.3g}..{max(ensemble[name]):.3g}, jax default {min(default_band[name]):.3g}..{max(default_band[name]):.3g}' for name in keys})
    geometric_mean = float(np.exp(np.mean(np.log(list(ratios.values())))))
    assert geometric_mean <= 1.5, (geometric_mean, ratios)
    for name in keys:
        assert errors[name] <= 3 * max(ensemble[name]), (name, errors[name], ensemble[name])
    #the two outputs read straight from the pair representation have a JAX band under 1% wide (positions and pLDDT are chaotic, these are not), so a ratio is exact there: the port measures 0.99 to 1.01
    #and a 10% margin catches a systematic bf16 error in the pair stack, which the same output judged against the flattered default band (ratios 1.28 to 1.56) would have flagged
    for name in ('predicted_aligned_error/logits', 'distogram/logits'):
        assert ratios[name] <= 1.10, (name, ratios[name], errors[name], ensemble[name])

@pytest.mark.integration
def test_dropout_flag_is_keyed_and_reproducible(model):
    ref = load_ref('small', 'fp32', 0)
    features = {**features_from(ref), 'use_dropout': mx.array(True)}
    def logits(seed):
        out, _ = model.forward(features, mx.array(np.asarray([0, seed], np.uint32)), 1)
        return np.asarray(out['predicted_lddt/logits'])
    first, again, other = logits(1), logits(1), logits(2)
    assert np.array_equal(first, again) and not np.array_equal(first, other)
    off, _ = model.forward({**features, 'use_dropout': mx.array(False)}, mx.array(np.asarray([0, 1], np.uint32)), 1)
    assert not np.array_equal(first, np.asarray(off['predicted_lddt/logits']))
    #the spread over keys is a statistic, compared with the JAX model's over its own keys, never pointwise
    mlx_spread = np.std([logits(seed) for seed in range(1, 5)], axis=0).mean()
    assert mlx_spread > 0
    assert 0.5 < mlx_spread / jax_dropout_spread(ref) < 2.0

def jax_dropout_spread(ref):
    #the mean over elements of the standard deviation of the pLDDT logits across four dropout keys, in the vendored JAX model with the same dropout rates
    import capture_refs
    from bindcraft.af.alphafold.model import data as af_data
    from bindcraft.af2 import recycled_alphafold_outputs
    runner = capture_refs.build_runner(jax_units.MODEL, False, 'chunked', None)
    params = af_data.get_model_haiku_params(model_name=jax_units.MODEL, data_dir=cases.params_path(), fuse=True)
    features = cases.model_inputs('small', dropout=True)
    forward = jax.jit(lambda key: recycled_alphafold_outputs(runner, params, key, features, 1)['predicted_lddt']['logits'])
    return float(np.std([np.asarray(forward(jax.random.PRNGKey(seed))) for seed in range(1, 5)], axis=0).mean())
