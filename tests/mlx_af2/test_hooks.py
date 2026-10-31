import inspect
import os
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest

import cases
from bindcraft import campaign
from bindcraft.af2 import AlphaFoldDesignModel
from bindcraft.backend import design_model_class

class HookCalled(Exception):
    pass

def test_design_model_class_defaults_to_the_jax_class(monkeypatch):
    monkeypatch.delenv('BINDCRAFT_BACKEND', raising=False)
    assert design_model_class() is AlphaFoldDesignModel

def test_campaign_builds_the_design_model_through_design_model_class(monkeypatch):
    built = []

    class Recorder:
        def __init__(self, **arguments):
            built.append(arguments)

    monkeypatch.setattr(campaign, 'design_model_class', lambda: Recorder)
    plan = SimpleNamespace(subbatch_size='auto', attention_backend='auto', use_cueq=False, use_remat=True, length_bucket_size=32, multi_chain_binders=(), target_pad_length=0, design_settings=SimpleNamespace(binder=SimpleNamespace(amino_acid_bias=None)))
    model = campaign.campaign_design_model(plan, (cases.MULTIMER_MODEL,), 'weights', {})
    assert isinstance(model, Recorder) and built[0]['presets'] == (cases.MULTIMER_MODEL,) and built[0]['data_dir'] == 'weights'

def test_campaign_never_builds_the_jax_class_directly():
    assert 'AlphaFoldDesignModel(' not in inspect.getsource(campaign)  #the validation model's builder must go through design_model_class too

@pytest.mark.integration
def test_load_parameters_hook_is_used_by_the_constructor():
    class Recording(AlphaFoldDesignModel):
        loaded = []

        def _load_parameters(self, model_name, data_dir):
            Recording.loaded.append((model_name, data_dir))
            return super()._load_parameters(model_name, data_dir)

    model = cases.build_model(Recording)
    assert Recording.loaded == [(cases.MULTIMER_MODEL, cases.params_path())] and model.model_parameters[cases.MULTIMER_MODEL] is not None

@pytest.mark.integration
def test_model_outputs_hook_is_called_when_predicting():
    _, states, _, _ = cases.design_inputs('small')
    model = cases.build_model()
    seen = []

    def hook(alphafold_runner, model_parameters, key, model_inputs):
        seen.append(sorted(model_inputs))
        raise HookCalled

    model._model_outputs = hook
    with pytest.raises(HookCalled):
        model.predict(states, model=cases.MULTIMER_MODEL)
    assert 'msa_feat' in seen[0] and 'target_feat' in seen[0]

@pytest.mark.integration
def test_model_outputs_hook_is_called_when_taking_gradients():
    _, states, _, losses = cases.design_inputs('small')
    model = cases.build_model()
    seen = []

    def hook(alphafold_runner, model_parameters, key, model_inputs):
        seen.append(jax.tree.structure(model_parameters) == jax.tree.structure(model.model_parameters[cases.MULTIMER_MODEL]))  #the per-model runtime argument reaches the hook (as tracers under jit)
        raise HookCalled

    model._model_outputs = hook
    with pytest.raises(HookCalled):
        model.sequence_gradients(states, losses, model=cases.MULTIMER_MODEL)
    assert seen == [True]

@pytest.mark.integration
def test_jax_path_is_bit_identical_after_hooks(refs_dir):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'mlx_af2'))
    import capture_golden_jax
    golden = np.load(Path(refs_dir) / 'golden_jax_small.npz')
    now = capture_golden_jax.capture('small')
    assert sorted(now) == sorted(golden.files)
    differing = [name for name in now if not np.array_equal(now[name], golden[name])]
    assert not differing, f'the refactor changed the JAX path: {differing[:5]}'
