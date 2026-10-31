import json
from pathlib import Path

import numpy as np
import pytest

import cases
from bindcraft.af.mlx_af2.outputs import OUTPUT_KEYS, nest, output_shapes
from helpers import assert_live, cosine, rel_err
from seam_support import StandInModel, jax_stand_in

CASE_NAMES = ('small', 'medium')
VARIANTS = ('fp32', 'bf16')
LIVE_INPUTS = ('msa_feat', 'target_feat', 'all_atom_positions')

def load_call(refs_dir, case, variant, index=0):
    path = Path(refs_dir) / case / variant / f'call{index}.npz'
    assert path.exists(), f'{path} is missing; run scripts/mlx_af2/capture_refs.py'
    data = np.load(path, allow_pickle=False)
    return {name: data[name] for name in data.files}, json.loads(str(data['meta']))

@pytest.mark.integration
@pytest.mark.parametrize('case', CASE_NAMES)
@pytest.mark.parametrize('variant', VARIANTS)
def test_refs_are_live_and_complete(refs_dir, case, variant):
    arrays, meta = load_call(refs_dir, case, variant)
    for name in OUTPUT_KEYS:
        assert f'out/{name}' in arrays, name
        assert_live(arrays[f'out/{name}'], f'out/{name}')
    for name in LIVE_INPUTS:
        assert_live(arrays[f'in/{name}'], f'in/{name}')
    float_keys = [name for name in OUTPUT_KEYS if name not in ('structure_module/final_atom_mask', 'predicted_aligned_error/breaks')]
    for name in float_keys:
        assert_live(arrays[f'cot/{name}'], f'cot/{name}')
    for name in ('msa_feat', 'target_feat'):
        assert arrays[f'grad/{name}'].shape == arrays[f'in/{name}'].shape
        assert_live(arrays[f'grad/{name}'], f'grad/{name}')
    assert meta['num_recycle'] == 1 and meta['use_dropout'] is False and meta['variant'] == variant

@pytest.mark.integration
@pytest.mark.parametrize('case', CASE_NAMES)
def test_variants_agree_on_shapes_and_the_dtype_contract_is_the_jax_path(refs_dir, case):
    for variant in VARIANTS:
        arrays, meta = load_call(refs_dir, case, variant)
        for name, (shape, dtype) in output_shapes(meta['length']).items():
            assert arrays[f'out/{name}'].shape == shape, (variant, name)
            assert meta['out_dtypes'][name] == dtype, (variant, name, meta['out_dtypes'][name])

@pytest.mark.integration
@pytest.mark.parametrize('case', CASE_NAMES)
def test_spread_is_recorded_and_nonzero(refs_dir, case):
    spread = json.loads((Path(refs_dir) / 'spread.json').read_text())[case]
    assert set(spread['outputs']) == set(OUTPUT_KEYS)
    for name in ('predicted_lddt/logits', 'distogram/logits', 'structure_module/final_atom_positions'):
        assert 0.0 < spread['outputs'][name] < 1.0, name  #bf16 differs from fp32, and by something sane
    for name in ('msa_feat', 'target_feat'):
        assert 0.5 < spread['gradient_cosine'][name] < 1.0, name
    arrays32, _ = load_call(refs_dir, case, 'fp32')
    arrays16, _ = load_call(refs_dir, case, 'bf16')
    assert abs(spread['outputs']['predicted_lddt/logits'] - rel_err(arrays16['out/predicted_lddt/logits'], arrays32['out/predicted_lddt/logits'])) < 1e-9
    assert abs(spread['gradient_cosine']['target_feat'] - cosine(arrays16['grad/target_feat'], arrays32['grad/target_feat'])) < 1e-9

class AccessRecorder(dict):
    #a nested dict that logs every key the tail asks for
    def __init__(self, source: dict, path: str = '', log: set | None = None):
        super().__init__({name: AccessRecorder(value, f'{path}{name}/', log if log is not None else set()) if isinstance(value, dict) else value for name, value in source.items()})
        self.path, self.log = path, log if log is not None else set()

    def __getitem__(self, name):
        self.log.add(f'{self.path}{name}')
        return super().__getitem__(name)

    def __contains__(self, name):
        self.log.add(f'{self.path}{name}')
        return super().__contains__(name)

def test_the_tail_reads_nothing_beyond_the_output_keys():
    log: set = set()

    class Recording(StandInModel):
        def _model_outputs(self, alphafold_runner, model_parameters, key, model_inputs):
            return AccessRecorder(jax_stand_in(model_parameters, model_inputs), log=log)

    _, states, _, losses = cases.design_inputs('small')
    model = cases.build_model(Recording)
    model.predict(states, model=cases.MULTIMER_MODEL)
    model.sequence_gradients(states, losses, model=cases.MULTIMER_MODEL)
    leaves = {name for name in log if name in OUTPUT_KEYS}
    assert leaves == set(OUTPUT_KEYS), set(OUTPUT_KEYS) - leaves  #every output key is read, none outside the contract is
    for name in log:
        assert any(key == name or key.startswith(f'{name}/') for key in OUTPUT_KEYS), name

def test_the_recorder_notices_a_missing_key():
    #a control: the same run without the positions must fail, so the test above is not vacuous
    class Missing(StandInModel):
        def _model_outputs(self, alphafold_runner, model_parameters, key, model_inputs):
            outputs = jax_stand_in(model_parameters, model_inputs)
            del outputs['structure_module']
            return outputs

    _, states, _, _ = cases.design_inputs('small')
    with pytest.raises(KeyError):
        cases.build_model(Missing).predict(states, model=cases.MULTIMER_MODEL)

def test_a_head_that_is_missing_from_the_outputs_is_dropped_silently_not_reported():
    #the tail guards each confidence head with `in`, so a backend that left one out would lose that metric without an error: the bridge must always return every OUTPUT_KEY
    class NoDistogram(StandInModel):
        def _model_outputs(self, alphafold_runner, model_parameters, key, model_inputs):
            outputs = jax_stand_in(model_parameters, model_inputs)
            del outputs['distogram']
            return outputs

    _, states, _, _ = cases.design_inputs('small')
    predictions = cases.build_model(NoDistogram).predict(states, model=cases.MULTIMER_MODEL)
    metrics = next(iter(predictions.values())).metrics
    assert 'distogram' not in metrics and 'plddt' in metrics
