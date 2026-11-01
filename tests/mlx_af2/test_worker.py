"""The MLX worker process: the same result as in-process, reused across calls, never imports jax or torch, fails loudly (no silent fallback) with the cause and the cap named, keeps at most five models
resident, and leaves no shared memory behind. Workers here run MLX on the CPU device (conftest) so these tests never touch a GPU another run may hold."""
import os
import re
import signal
import time
import types
from multiprocessing import shared_memory

import mlx.core as mx
import numpy as np
import pytest

import cases
import jax_units
from bindcraft.af.mlx_af2 import model_multimer, worker as worker_module
from bindcraft.af.mlx_af2.outputs import OUTPUT_KEYS
from bindcraft.af.mlx_af2.weights import load_params
from bindcraft.af.mlx_af2.worker import RemoteModel, WorkerError, payload_bytes
from helpers import load_ref, rel_err

MODEL = cases.MULTIMER_MODEL

def model_config(**extra):
    cfg = jax_units.configs()[0].to_dict()
    cfg['global_config']['bfloat16'] = False
    return {**cfg, **extra}

def crop(ref, start, stop):
    #a tiny design for the tests that need a forward but not the full model's cost: every axis of length L is cut to [start, stop)
    length = ref['in/aatype'].shape[0]
    cut = lambda array: array[tuple(slice(start, stop) if dim == length else slice(None) for dim in array.shape)]
    return {name[3:]: cut(value) for name, value in ref.items() if name.startswith('in/')}

def real_features():
    ref = load_ref('small', 'fp32', 0)
    return {name[3:]: value for name, value in ref.items() if name.startswith('in/')}, np.asarray(ref['meta']['key'], np.uint32)

@pytest.fixture(scope='module')
def remote():
    worker = RemoteModel(cases.params_path(), dtype='fp32')
    yield worker
    worker.close()

@pytest.mark.integration
def test_worker_matches_the_in_process_forward(remote):
    features, key = real_features()
    handle = remote.load(MODEL, model_config())
    got, _ = remote.forward(handle, key, features, 1, None)
    model = model_multimer.MultimerModel(MODEL, load_params(MODEL, cases.params_path(), dtype=mx.float32), model_config(), mx.float32)
    want, _ = model.forward({name: mx.array(value) for name, value in features.items()}, mx.array(key), 1)
    assert set(got) == set(OUTPUT_KEYS)
    for name in OUTPUT_KEYS:
        assert got[name].shape == tuple(want[name].shape) and rel_err(got[name], np.asarray(want[name])) < 1e-5, name

@pytest.mark.integration
def test_worker_is_reused_across_calls(remote):
    features = crop(load_ref('small', 'fp32', 0), 30, 42)
    key = np.asarray([0, 1], np.uint32)
    handle = remote.load(MODEL, model_config())
    remote.forward(handle, key, features, 0, None)
    pid = remote.pid
    remote.forward(handle, key, features, 0, None)
    assert remote.pid == pid and remote.last_seconds > 0

@pytest.mark.integration
def test_worker_never_imports_jax_or_torch(remote):
    remote.start()
    assert remote.ready_info['jax_imported'] is False and remote.ready_info['torch_imported'] is False and remote.ready_info['device'] == os.environ['BINDCRAFT_MLX_DEVICE']  #cpu, or gpu in an explicit GPU run (conftest sets it per test)

@pytest.mark.integration
def test_loading_a_model_without_weights_names_the_model(remote):
    with pytest.raises(WorkerError, match='model_9_multimer_v3'):
        remote.load('model_9_multimer_v3', model_config())

@pytest.mark.integration
def test_an_unimplemented_family_is_refused_by_name(remote):
    with pytest.raises(WorkerError, match=r"(?s)does not implement.*model_1\b"):
        remote.load('model_1', model_config())  #the monomer models without a predicted-aligned-error head; the pTM ones are implemented

@pytest.mark.integration
def test_forward_with_an_unknown_handle_names_the_handle(remote):
    features = crop(load_ref('small', 'fp32', 0), 30, 42)
    with pytest.raises(WorkerError, match='model handle 99'):
        remote.forward(99, np.asarray([0, 1], np.uint32), features, 0, None)

@pytest.mark.integration
def test_dead_worker_raises_with_the_exit_code_then_a_new_one_serves_the_next_call():
    worker = RemoteModel(cases.params_path(), dtype='fp32')
    features, key = crop(load_ref('small', 'fp32', 0), 30, 42), np.asarray([0, 1], np.uint32)
    try:
        handle = worker.load(MODEL, model_config())
        worker.forward(handle, key, features, 0, None)
        first_pid = worker.pid
        os.kill(first_pid, signal.SIGKILL)
        time.sleep(0.5)
        with pytest.raises(WorkerError, match='exit code -9'):
            worker.forward(handle, key, features, 0, None)  #the failed call is not retried behind the caller's back
        outputs, _ = worker.forward(handle, key, features, 0, None)  #the registered model was told to the new worker again, so the handle is still valid
        assert worker.pid != first_pid and set(outputs) == set(OUTPUT_KEYS)
    finally:
        worker.close()

@pytest.mark.integration
def test_cap_exceeded_error_names_the_cap_and_does_not_look_like_a_jax_out_of_memory():
    from bindcraft.af2 import is_out_of_memory
    worker = RemoteModel(cases.params_path(), cap_gib=0.3, dtype='fp32')
    features, key = crop(load_ref('small', 'fp32', 0), 30, 42), np.asarray([0, 1], np.uint32)
    try:
        with pytest.raises(WorkerError, match=r'(?s)exit code -9.*0\.3 GiB') as failure:  #killed by the memory guard, not just any error
            handle = worker.load(MODEL, model_config())
            worker.forward(handle, key, features, 0, None)
        assert not is_out_of_memory(failure.value)  #or the repo's JAX-specific fallback would retry what cannot succeed
    finally:
        worker.close()

def test_an_ordinary_worker_error_is_not_mistaken_for_a_memory_kill():
    worker = RemoteModel('unused', cap_gib=0.5)
    ordinary = worker.worker_error('Traceback (most recent call last):\nValueError: bad layout')
    assert re.search('0.5 GiB', ordinary) and not re.search(r'(?s)exit code -9.*0\.5 GiB', ordinary)
    worker.proc = types.SimpleNamespace(wait=lambda timeout=None: -9)
    assert re.search(r'(?s)exit code -9.*0\.5 GiB', worker.death('during forward'))

@pytest.mark.integration
def test_shared_memory_is_unlinked_after_success_and_error(remote):
    features = crop(load_ref('small', 'fp32', 0), 30, 42)
    key = np.asarray([0, 1], np.uint32)
    handle = remote.load(MODEL, model_config())
    remote.forward(handle, key, features, 0, None)
    names = list(remote.last_shm_names)
    with pytest.raises(WorkerError):
        remote.forward(98, key, features, 0, None)
    names += remote.last_shm_names
    assert len(names) >= 3
    for name in names:
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=name)

@pytest.mark.integration
def test_at_most_five_models_stay_resident_and_an_evicted_one_is_read_again():
    worker = RemoteModel(cases.params_path(), dtype='fp32')
    features, key = crop(load_ref('small', 'fp32', 0), 30, 42), np.asarray([0, 1], np.uint32)
    try:
        handles = [worker.load(MODEL, model_config(tag=index)) for index in range(6)]  #six registrations of the same weights
        assert handles == list(range(6))
        first, _ = worker.forward(handles[0], key, features, 0, None)
        for handle in handles[1:]:
            worker.forward(handle, key, features, 0, None)
        stats = worker.stats()
        assert stats['loaded_models'] == 5 and stats['registered_models'] == 6  #the first was evicted by the sixth
        again, _ = worker.forward(handles[0], key, features, 0, None)
        assert all(np.array_equal(first[name], again[name]) for name in OUTPUT_KEYS)  #read again, same result
    finally:
        worker.close()

def test_the_shared_remote_is_one_worker_for_two_models():
    RemoteModel.instances.clear()
    first = RemoteModel.shared(cases.params_path(), None, 'fp32')
    second = RemoteModel.shared(cases.params_path(), None, 'fp32')
    assert first is second and RemoteModel.shared(cases.params_path(), None, 'bf16') is not first
    RemoteModel.instances.clear()

def test_the_payload_of_a_call_is_small_at_300_residues():
    n = 300
    features = {'aatype': np.zeros(n, np.int32), 'residue_index': np.zeros(n, np.int32), 'offset': np.zeros((n, n), np.int32), 'asym_id': np.zeros(n, np.int32), 'entity_id': np.zeros(n, np.int32),
                'sym_id': np.zeros(n, np.int32), 'seq_mask': np.ones(n, np.float32), 'msa_mask': np.ones((1, n), np.float32), 'target_feat': np.zeros((n, 20), np.float32), 'msa_feat': np.zeros((1, n, 49), np.float32),
                'all_atom_positions': np.zeros((n, 37, 3), np.float32), 'all_atom_mask': np.zeros((n, 37), np.float32), 'template_all_atom_positions': np.zeros((1, n, 37, 3), np.float32)}
    assert payload_bytes(features) < 16e6  #the pair-sized logits come back, but the call carries only residue-sized features and the offsets (the 300 x 300 prev is not sent)

@pytest.mark.integration
def test_the_worker_runs_a_monomer_model_forward_only(remote):
    #a pTM monomer model through the real worker process: the same outputs as in-process, and a gradient request is refused by name (the monomer models only validate)
    from bindcraft.af.mlx_af2 import monomer
    ref = load_ref('small', 'fp32__model_1_ptm', 1)
    features = crop(ref, 30, 42)
    key = np.asarray(ref['meta']['key'], np.uint32)
    cfg = jax_units.configs('model_1_ptm')[0].to_dict()
    handle = remote.load('model_1_ptm', cfg)
    got, call_id = remote.forward(handle, key, features, 0, None)
    want, _ = monomer.MonomerModel('model_1_ptm', load_params('model_1_ptm', cases.params_path(), dtype=mx.float32), cfg, mx.float32).forward({name: mx.array(value) for name, value in features.items()}, mx.array(key), 0)
    assert set(got) == set(OUTPUT_KEYS)
    for name in OUTPUT_KEYS:
        assert got[name].shape == tuple(want[name].shape) and rel_err(got[name], np.asarray(want[name])) < 1e-5, name
    with pytest.raises(WorkerError, match=r"(?s)model_1_ptm.*no gradient"):
        remote.vjp(handle, call_id, key, features, 0, None, {'distogram/logits': np.ones((12, 12, 64), np.float32)}, ('msa_feat',))

@pytest.mark.integration
def test_a_stalled_worker_is_killed_and_the_cause_named():
    #a worker that stops answering without dying (a wedged kernel, a deadlock) must not hang the run forever with the GPU lock held: the call has a deadline, the worker is killed, the error says it stalled
    import threading
    worker = RemoteModel(cases.params_path(), dtype='fp32')
    features, key = crop(load_ref('small', 'fp32', 0), 30, 42), np.asarray([0, 1], np.uint32)
    safety = None
    try:
        handle = worker.load(MODEL, model_config())
        worker.forward(handle, key, features, 0, None)  #warm: weights read, graph built, so the deadline below is for a call that would take a second
        worker.call_timeout_s = 4.0
        safety = threading.Timer(90, lambda: os.kill(worker.pid, signal.SIGKILL))  #so a missing deadline fails this test instead of hanging the suite
        safety.start()
        os.kill(worker.pid, signal.SIGSTOP)  #alive, and silent
        started = time.time()
        with pytest.raises(WorkerError, match=r'(?s)stalled.*no reply within 4 s') as failure:
            worker.forward(handle, key, features, 0, None)
        assert time.time() - started < 60
        assert 'out of memory' not in str(failure.value).lower()  #and the repo's retry ladder does not rerun it
        for name in worker.last_shm_names:
            with pytest.raises(FileNotFoundError):
                shared_memory.SharedMemory(name=name)
        safety.cancel()
        outputs, _ = worker.forward(handle, key, features, 0, None)  #the next call starts a new worker and replays the registered models
        assert set(outputs) == set(OUTPUT_KEYS)
    finally:
        if safety is not None:
            safety.cancel()
        worker.close()

def test_the_call_deadline_has_a_default_and_an_environment_override(monkeypatch):
    monkeypatch.delenv(worker_module.CALL_TIMEOUT_ENV, raising=False)
    assert RemoteModel('/unused', dtype='fp32').call_timeout_s == worker_module.DEFAULT_CALL_TIMEOUT_S == 1800.0
    monkeypatch.setenv(worker_module.CALL_TIMEOUT_ENV, '45')
    assert RemoteModel('/unused', dtype='fp32').call_timeout_s == 45.0
    monkeypatch.setenv(worker_module.CALL_TIMEOUT_ENV, '0')
    assert RemoteModel('/unused', dtype='fp32').call_timeout_s is None  #0 turns the deadline off
