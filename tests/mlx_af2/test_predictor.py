"""MlxAlphaFoldDesignModel against the repo's JAX path on real design states: the same predictions and sequence gradients (fp32 on both sides, dropout off, small and pair cases), the runtime model
handle, one worker and one lock for the design and validation models, and the failure rules (a dead worker names its cause and stops, a memory failure is never mistaken for a JAX out-of-memory).
The worker runs on the CPU device here (set before it starts) and the disk preflight is not under test: no GPU is touched and no lock is taken except in the test that fakes the remote."""
import os
import signal
import time
from multiprocessing import shared_memory

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import cases
import capture_refs
from bindcraft.af.mlx_af2 import bridge, gpu_lock, predictor
from bindcraft.af.mlx_af2.worker import RemoteModel, WorkerError
from bindcraft.af2 import AlphaFoldDesignModel, alphafold_input_features, is_out_of_memory, prepare_design_sequence_features
from helpers import cosine, rel_err

MODEL = cases.MULTIMER_MODEL
FIVE = tuple(f'model_{index}_multimer_v3' for index in range(1, 6))

class Fp32JaxModel(AlphaFoldDesignModel):
    #the repo's own JAX path with bfloat16 off: the reference an fp32 MLX model is judged against
    def _alphafold_runner(self, model_family, subbatch_size, attention_backend=None):
        attention_backend = attention_backend or self.attention_backend
        key = 'fp32', model_family, subbatch_size, attention_backend
        if key not in self.alphafold_runners:
            self.alphafold_runners[key] = capture_refs.build_runner(self.model_family_models[model_family], False, attention_backend, subbatch_size)
        return self.alphafold_runners[key]

class Fp32MlxModel(predictor.MlxAlphaFoldDesignModel):
    dtype = 'fp32'

@pytest.fixture(scope='module', autouse=True)
def cpu_worker_without_preflight():
    #module scope on purpose: the fixtures below start the worker while they build their models, before the per-test device fixture of conftest runs
    patch = pytest.MonkeyPatch()
    patch.setenv(predictor.DEVICE_ENV, 'gpu' if os.environ.get('BINDCRAFT_MLX_TEST_DEVICE') == 'gpu' else 'cpu')
    patch.setattr(predictor.preflight, 'checks', lambda: None)
    patch.setattr(predictor, 'session', None)
    RemoteModel.instances.clear()
    yield
    for remote in list(RemoteModel.instances.values()):
        remote.close()
    RemoteModel.instances.clear()
    patch.undo()

def build_pair(case, models):
    return cases.build_model(Fp32JaxModel, case=case, models=models), cases.build_model(Fp32MlxModel, case=case, models=models)

@pytest.fixture(scope='module')
def small():
    yield build_pair('small', FIVE)

@pytest.fixture(scope='module')
def pair():
    yield build_pair('pair', (MODEL,))

def gradients_and_loss(model, case, name=MODEL):
    _, states, _, losses = cases.design_inputs(case)
    _, gradients, loss = model.sequence_gradients(states, losses, model=name)
    return gradients, float(loss)

def assert_gradients_match(got, want, got_loss, want_loss):
    assert set(got) == set(want) and got
    for chain in want:
        assert np.abs(np.asarray(want[chain])).max() > 0, chain
        print(f'{chain}: cosine {cosine(got[chain], want[chain]):.6f}, relative error {rel_err(got[chain], want[chain]):.3e}')
        assert cosine(got[chain], want[chain]) >= 0.999, chain
    assert abs(got_loss - want_loss) <= 1e-3 * max(1.0, abs(want_loss)), (got_loss, want_loss)

@pytest.mark.integration
def test_predict_matches_the_jax_path(small):
    jax_model, mlx_model = small
    _, states, _, _ = cases.design_inputs('small')
    want, got = jax_model.predict(states, model=MODEL), mlx_model.predict(states, model=MODEL)
    assert set(got) == set(want) and want
    for state in want:
        assert set(got[state].metrics) == set(want[state].metrics)
        for name, value in want[state].metrics.items():
            assert rel_err(got[state].metrics[name], value) < 1e-3, (state, name, rel_err(got[state].metrics[name], value))
        assert set(got[state].protein_complex) == set(want[state].protein_complex)
        for chain, protein in want[state].protein_complex.items():
            print(state, chain, f'positions relative error {rel_err(got[state].protein_complex[chain].atoms, protein.atoms):.3e}')
            assert rel_err(got[state].protein_complex[chain].atoms, protein.atoms) < 1e-3, (state, chain)

@pytest.mark.integration
def test_sequence_gradients_match_the_jax_path(small):
    jax_model, mlx_model = small
    want, want_loss = gradients_and_loss(jax_model, 'small')
    got, got_loss = gradients_and_loss(mlx_model, 'small')
    assert_gradients_match(got, want, got_loss, want_loss)

def test_diff_keys_cover_every_sequence_dependent_float_feature():
    a = cases.front_arguments('small')

    def front(sequence):
        sequence_features, sequence_profile = prepare_design_sequence_features(sequence, a['flags'], jnp.asarray(1.0), jnp.asarray(1.0), jnp.asarray(0.01), jnp.asarray(2.0), None)
        features = alphafold_input_features(sequence_features, sequence_profile, a['atoms'], a['atom_mask'], a['residue_index'], a['asym_id'], a['seq_mask'], a['flags'], jnp.asarray(False), 'direction',
                                            a['entity_id'], 0.0, False)
        return {name: value for name, value in features.items() if name not in bridge.DROPPED_FEATURES}

    tangent = jax.random.normal(jax.random.PRNGKey(7), a['sequence'].shape, a['sequence'].dtype)
    primals, tangents = jax.jvp(front, (a['sequence'],), (tangent,))
    floats = {name for name, value in primals.items() if jnp.issubdtype(value.dtype, jnp.floating)}
    sensitive = {name for name in floats if np.abs(np.asarray(tangents[name])).max() > 0}
    assert sensitive == set(bridge.DIFF_KEYS), sensitive  #a missing key would silently lose its gradient; an extra one would send a pair-sized zero for nothing

@pytest.mark.integration
@pytest.mark.xfail(os.environ.get('BINDCRAFT_MLX_TEST_DEVICE') == 'gpu', strict=False, reason='the second state of this case has a chaotic fp32 gradient: the JAX-CPU reference is 4-5% from an fp64 truth (haiku fast-variance layer norm, E[x^2] - E[x]^2, cancels badly on large-mean pair '
                   'activations; exact variance puts it at 8e-4), and the GPU gradient is the better one (6e-4 from the truth); see docs/mlx_backend/checkpoints.md')
def test_two_states_in_one_call(pair):
    jax_model, mlx_model = pair
    _, states, _, _ = cases.design_inputs('pair')
    assert len(states) == 2
    want, want_loss = gradients_and_loss(jax_model, 'pair')
    got, got_loss = gradients_and_loss(mlx_model, 'pair')
    assert_gradients_match(got, want, got_loss, want_loss)

@pytest.mark.integration
def test_model_is_redrawn_each_step(small):
    #the model is a runtime handle: five models share one traced program per shape, each call runs the model it names, and a model drawn at random (model=None) is the one JAX draws from the same key
    jax_model, mlx_model = small
    traces = []
    real = mlx_model._model_outputs
    mlx_model._model_outputs = lambda *arguments: (traces.append(1), real(*arguments))[1]
    try:
        wanted, gotten = {}, []
        for name in (FIVE[0], FIVE[1], FIVE[0], FIVE[2]):
            if name not in wanted:
                wanted[name] = gradients_and_loss(jax_model, 'small', name)
            got, got_loss = gradients_and_loss(mlx_model, 'small', name)
            gotten.append((name, got))
            assert_gradients_match(got, wanted[name][0], got_loss, wanted[name][1])
        assert len(traces) <= 1, 'a model name was baked into the trace: the shape compiled once per model'  #0 when an earlier test compiled the shape, 1 when this one did; baked-in names would be 3 or more
        by_model = {name: got for name, got in gotten}
        chain = next(iter(by_model[FIVE[0]]))
        for other in (FIVE[1], FIVE[2]):
            assert rel_err(by_model[other][chain], by_model[FIVE[0]][chain]) > 1e-2, other  #the models are different, so is what they return
        draws = {'jax': [], 'mlx': []}
        for label, model in (('jax', jax_model), ('mlx', mlx_model)):
            sample = model._sample_design_model
            model._sample_design_model = lambda sample=sample, label=label: (draws[label].append(sample()), draws[label][-1])[1]
        _, states, _, losses = cases.design_inputs('small')
        _, got, got_loss = mlx_model.sequence_gradients(states, losses)
        _, want, want_loss = jax_model.sequence_gradients(states, losses)
        assert draws['mlx'] == draws['jax'] and len(draws['mlx']) == 1  #same key, same draw
        assert_gradients_match(got, want, float(got_loss), float(want_loss))
    finally:
        mlx_model._model_outputs = real

def test_design_and_validation_models_share_one_worker():
    RemoteModel.instances.clear()
    design = cases.build_model(Fp32MlxModel, models=(MODEL,))
    validation = cases.build_model(Fp32MlxModel, models=(FIVE[1],))
    try:
        assert design.remote is validation.remote and len(RemoteModel.instances) == 1
        assert design.remote.stats()['registered_models'] == 2 and design.remote.pid == validation.remote.pid
    finally:
        design.remote.close()
        RemoteModel.instances.clear()

class FakeRemote:
    #no process, no GPU: records what the predictor asks of the worker
    instance = None

    def __init__(self):
        self.loaded = []

    @classmethod
    def shared(cls, params_dir, cap_gib, dtype):
        cls.instance = cls.instance or cls()
        return cls.instance

    def load(self, name, cfg):
        self.loaded.append(name)
        return len(self.loaded) - 1

@pytest.mark.skipif(os.environ.get('BINDCRAFT_MLX_TEST_DEVICE') == 'gpu', reason='an explicit GPU run already holds the lock for the whole session, so the depth does not start at zero')
def test_design_and_validation_models_take_the_gpu_lock_and_run_the_preflight_once(monkeypatch, tmp_path):
    entries, checks = [], []
    real_hold = gpu_lock.hold
    FakeRemote.instance = None
    monkeypatch.setattr(predictor, 'RemoteModel', FakeRemote)
    monkeypatch.setattr(predictor, 'session', None)
    monkeypatch.setattr(predictor.preflight, 'checks', lambda: checks.append(1))
    monkeypatch.setattr(predictor.gpu_lock, 'hold', lambda: (entries.append(1), real_hold())[1])
    monkeypatch.setenv(predictor.DEVICE_ENV, 'gpu')  #the fake remote never starts a worker, so nothing here touches the GPU
    monkeypatch.setenv(gpu_lock.PATH_ENV, str(tmp_path / 'gpu.lock'))
    monkeypatch.delenv(gpu_lock.OWNER_ENV, raising=False)
    try:
        design = cases.build_model(Fp32MlxModel, models=(MODEL,))
        validation = cases.build_model(Fp32MlxModel, models=(FIVE[1],))
        assert design.remote is validation.remote is FakeRemote.instance and FakeRemote.instance.loaded == [MODEL, FIVE[1]]
        assert len(entries) == 1 and len(checks) == 1  #the second model found the session open
        assert gpu_lock.held_depth == 1 and open(tmp_path / 'gpu.lock').read().strip() == str(os.getpid())
    finally:
        if predictor.session is not None:
            predictor.session.close()
    assert gpu_lock.held_depth == 0

def test_a_cpu_worker_takes_no_gpu_lock(monkeypatch):
    entries = []
    monkeypatch.setattr(predictor, 'RemoteModel', FakeRemote)
    monkeypatch.setattr(predictor, 'session', None)
    monkeypatch.setattr(predictor.gpu_lock, 'hold', lambda: entries.append(1))
    monkeypatch.setenv(predictor.DEVICE_ENV, 'cpu')
    cases.build_model(Fp32MlxModel, models=(MODEL,))
    assert entries == []

@pytest.mark.integration
def test_worker_killed_during_gradients_names_the_cause_and_stops(small):
    _, mlx_model = small
    _, states, _, losses = cases.design_inputs('small')
    mlx_model.sequence_gradients(states, losses, model=MODEL)  #compiled and warm, so the kill lands inside the backward call
    remote = mlx_model.remote
    original = remote.exchange_once

    def kill_before_vjp(kind, request, arrays, replay=False):
        if kind == 'vjp':
            os.kill(remote.pid, signal.SIGKILL)
            time.sleep(0.5)
        return original(kind, request, arrays, replay)

    remote.exchange_once = kill_before_vjp
    try:
        with pytest.raises(Exception) as failure:
            mlx_model.sequence_gradients(states, losses, model=MODEL)
    finally:
        remote.exchange_once = original
    text = str(failure.value)
    assert 'exit code -9' in text and 'memory cap was' in text and 'died during vjp' in text, text
    assert not is_out_of_memory(failure.value)  #so the repo's retry ladder does not rerun it
    for name in remote.last_shm_names:
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=name)
    gradients, _ = gradients_and_loss(mlx_model, 'small')  #the next call starts a new worker; registered models are replayed
    assert gradients

def test_unimplemented_family_is_refused_by_name():
    with pytest.raises(ValueError, match=r"'model_1'"):
        cases.build_model(Fp32MlxModel, models=('model_1',))  #a monomer model without a predicted-aligned-error head; the pTM monomer models are implemented

def test_memory_failures_do_not_trigger_the_jax_ladder():
    remote = RemoteModel('/unused', cap_gib=36.0, dtype='fp32')
    for phrase, kept in (('RESOURCE_EXHAUSTED: Out of memory while trying to allocate 80000000000 bytes', 'resource exhausted'), ('RuntimeError: [metal::malloc] Out Of Memory: attempting to allocate 80000000000 bytes', 'out of memory'),
                         ('kIOGPUCommandBufferCallbackErrorOutOfMemory (out of memory)', 'out of memory')):
        text = remote.worker_error(f'Traceback (most recent call last):\n  File "worker.py", line 1\n{phrase}')
        assert not is_out_of_memory(WorkerError(text)), phrase
        assert 'worker memory cap: 36.0 GiB' in text  #the cap is still named
        assert kept in text.lower().replace('-', ' ').replace('_', ' '), text  #and the cause is still readable
    killed = RemoteModel('/unused', cap_gib=36.0, dtype='fp32')
    assert 'memory cap was 36.0 GiB' in killed.death('during forward') and not is_out_of_memory(WorkerError(killed.death('during forward')))

@pytest.mark.integration
def test_sequence_gradients_run_through_the_repos_shape_logic_unchanged():
    model = cases.build_model(Fp32MlxModel, models=(MODEL,))
    calls = []
    real = model._widest_shard_that_fits
    model._widest_shard_that_fits = lambda *arguments, **keywords: (calls.append(1), real(*arguments, **keywords))[1]
    gradients, _ = gradients_and_loss(model, 'small')
    assert gradients and len(calls) == 1
    assert model.shape_subbatch_size and model.shape_attention_backend  #recorded per shape, as with the JAX path
    assert list(model.shape_subbatch_size.values()) == [None]  #the one rung of the MLX ladder

def test_the_memory_ladder_has_one_rung(monkeypatch):
    #the repo's ladder retries a failed gradient with shards of 256, 128, 64 and then chunked attention; a worker's memory failure is final, so rerunning it would only fail again, slower
    monkeypatch.setattr(predictor, 'RemoteModel', FakeRemote)
    monkeypatch.setenv(predictor.DEVICE_ENV, 'cpu')
    assert cases.build_model(Fp32MlxModel, models=(MODEL,), subbatch_size='auto').subbatch_ladder() == (None,)
    assert cases.build_model(Fp32MlxModel, models=(MODEL,), subbatch_size=128).subbatch_ladder() == (128,)
