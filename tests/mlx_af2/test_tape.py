"""The tape across calls: the design forward call keeps its final pass's block inputs (the extra-MSA and the main stack) so the vjp that follows skips that pass's Evoformer forward. The gradient must be the
same with the tape as without it, at the model and through the worker, with dropout on too; the worker keeps tapes only when the forward asks it to, under a byte budget that evicts the oldest first, and a vjp
whose tape is gone is recomputed, never wrong."""
import mlx.core as mx
import numpy as np
import pytest

import cases
from bindcraft.af.mlx_af2 import worker as worker_module
from bindcraft.af.mlx_af2.outputs import OUTPUT_KEYS
from bindcraft.af.mlx_af2.worker import RemoteModel
from helpers import assert_live, load_ref, rel_err
from test_model_multimer import build_model, features_from
from test_vjp import DIFF, cotangents_from, key_of, random_cotangents
from test_worker import MODEL, crop, model_config

pytestmark = pytest.mark.integration

@pytest.fixture(scope='module')
def model():
    return build_model()

@pytest.mark.parametrize('dropout', (False, True))
def test_vjp_from_the_forward_calls_tapes_equals_the_recomputed_vjp(model, dropout):
    ref = load_ref('small', 'fp32', 0)
    features, key, cotangents, recycles = features_from(ref), key_of(ref), cotangents_from(ref), ref['meta']['num_recycle']
    if dropout:
        features = {**features, 'use_dropout': mx.array(True)}
    tapes = model.new_tapes()
    _, _, before = model.forward_with_state(features, key, recycles, tapes=tapes)
    assert set(tapes) == {'extra_msa_stack', 'evoformer_iteration'} and all(tape.outputs is not None for tape in tapes.values())
    taped = model.vjp(features, key, recycles, cotangents, DIFF, prev=before, tapes=tapes)
    recomputed = model.vjp(features, key, recycles, cotangents, DIFF, prev=before)
    for name in DIFF:
        assert_live(np.asarray(recomputed[name]), name)
        assert rel_err(taped[name], recomputed[name]) < 1e-6, name

@pytest.mark.parametrize('dropout', (False, True))
def test_the_vjp_equals_the_whole_graph_vjp_of_the_final_pass(model, dropout):
    #the oracle: mx.vjp over the whole final pass (the previous implementation). The model's vjp is a chain (tail, main stack, extra stack, front) run outside any transformation so the compiled blocks stay
    #fused; a feature that reached the outputs other than through the front's msa and pair would get a wrong gradient from the chain, and this comparison is what rules that out
    ref = load_ref('small', 'fp32', 0)
    features, key, cotangents, recycles = features_from(ref), key_of(ref), cotangents_from(ref), ref['meta']['num_recycle']
    if dropout:
        features = {**features, 'use_dropout': mx.array(True)}
    _, _, before = model.forward_with_state(features, key, recycles)
    keys = model.dropout_keys(features, key, recycles, None)
    final = model.final_features(features, before, recycles)
    names = [name for name in OUTPUT_KEYS if name in cotangents]

    def whole(*arrays):
        outputs, _ = model.single_pass({**final, **dict(zip(DIFF, arrays))}, before, keys[-1] if keys else None, remat=True)
        flat = {f'{group}/{name}': value for group, tensors in outputs.items() for name, value in tensors.items()}
        return [flat[name].astype(mx.float32) for name in names]

    _, want = mx.vjp(whole, [final[name] for name in DIFF], [cotangents[name] for name in names])
    got = model.vjp(features, key, recycles, cotangents, DIFF, prev=before)
    for name, expected in zip(DIFF, want):
        assert_live(np.asarray(expected), name)
        assert rel_err(got[name], expected) < 1e-6, name

def test_a_tape_makes_the_vjp_skip_the_final_pass_of_both_stacks(model, monkeypatch):
    from bindcraft.af.mlx_af2 import compiled, evoformer
    monkeypatch.setenv(compiled.COMPILE_ENV, '0')  #counts the loop's block calls; a compiled block runs evoformer_iteration only when it traces
    ref = load_ref('small', 'fp32', 0)
    features, key, cotangents = features_from(ref), key_of(ref), cotangents_from(ref)
    tapes = model.new_tapes()
    _, _, before = model.forward_with_state(features, key, 1, tapes=tapes)
    calls, original = {'blocks': 0}, evoformer.evoformer_iteration
    monkeypatch.setattr(evoformer, 'evoformer_iteration', lambda *a, **k: (calls.__setitem__('blocks', calls['blocks'] + 1), original(*a, **k))[1])
    model.vjp(features, key, 1, cotangents, DIFF, prev=before)
    without = calls['blocks']
    calls['blocks'] = 0
    model.vjp(features, key, 1, cotangents, DIFF, prev=before, tapes=tapes)
    blocks = model.cfg['embeddings_and_evoformer']['evoformer_num_block'] + model.cfg['embeddings_and_evoformer']['extra_msa_stack_num_block']
    assert without == 2 * blocks and calls['blocks'] == blocks  #the forward of every block is gone; the recompute inside the reverse loop stays

@pytest.fixture
def remote():
    RemoteModel.instances.clear()
    worker = RemoteModel(cases.params_path(), dtype='fp32')
    yield worker
    worker.close()

def small_call():
    ref = load_ref('small', 'fp32', 0)
    return crop(ref, 30, 42), np.asarray(ref['meta']['key'], np.uint32)

def test_a_retained_forward_makes_the_vjp_a_tape_hit_with_the_same_gradients(remote):
    features, key = small_call()
    handle = remote.load(MODEL, model_config())
    outputs, kept = remote.forward(handle, key, features, 1, None, retain=True)
    _, plain = remote.forward(handle, key, features, 1, None)
    assert remote.stats()['tape_bytes'] > 0
    cotangents = random_cotangents(outputs)
    hit = remote.vjp(handle, kept, key, features, 1, None, cotangents, DIFF)
    miss = remote.vjp(handle, plain, key, features, 1, None, cotangents, DIFF)
    stats = remote.stats()
    assert stats['tape_hits'] == 1 and stats['tape_misses'] == 1 and stats['vjp_cache_hits'] == 2  #both recycle states were cached; only the retained call had a tape
    for name in DIFF:
        assert np.abs(miss[name]).max() > 0
        assert rel_err(hit[name], miss[name]) < 1e-6, name

def test_a_forward_that_does_not_retain_keeps_no_tape(remote):
    features, key = small_call()
    handle = remote.load(MODEL, model_config())
    remote.forward(handle, key, features, 1, None)
    assert remote.stats()['tape_bytes'] == 0

def test_the_tape_budget_keeps_the_newest_and_an_evicted_call_still_gets_its_gradient(monkeypatch):
    features, key = small_call()
    RemoteModel.instances.clear()
    probe = RemoteModel(cases.params_path(), dtype='fp32')
    try:
        handle = probe.load(MODEL, model_config())
        probe.forward(handle, key, features, 1, None, retain=True)
        one = probe.stats()['tape_bytes']
    finally:
        probe.close()
    monkeypatch.setenv(worker_module.TAPE_BUDGET_ENV, repr(1.5 * one / 2 ** 30))  #room for one call's tapes, not two
    RemoteModel.instances.clear()
    remote = RemoteModel(cases.params_path(), dtype='fp32')
    try:
        handle = remote.load(MODEL, model_config())
        outputs, first = remote.forward(handle, key, features, 1, None, retain=True)
        _, second = remote.forward(handle, key, features, 1, None, retain=True)
        assert remote.stats()['tape_bytes'] == one  #the older call's tapes were dropped, its recycle state kept
        cotangents = random_cotangents(outputs)
        old = remote.vjp(handle, first, key, features, 1, None, cotangents, DIFF)
        new = remote.vjp(handle, second, key, features, 1, None, cotangents, DIFF)
        stats = remote.stats()
        assert stats['tape_misses'] == 1 and stats['tape_hits'] == 1 and stats['vjp_cache_hits'] == 2
        for name in DIFF:
            assert np.abs(old[name]).max() > 0
            assert rel_err(old[name], new[name]) < 1e-6, name
    finally:
        remote.close()

def test_the_tape_budget_defaults_to_a_fifth_of_the_cap_and_has_an_override(monkeypatch):
    monkeypatch.delenv(worker_module.TAPE_BUDGET_ENV, raising=False)
    assert worker_module.tape_budget_bytes(36.0) == int(0.2 * 36.0 * 2 ** 30)
    assert worker_module.tape_budget_bytes(None) == int(worker_module.UNCAPPED_TAPE_GIB * 2 ** 30)
    monkeypatch.setenv(worker_module.TAPE_BUDGET_ENV, '0.5')
    assert worker_module.tape_budget_bytes(36.0) == int(0.5 * 2 ** 30)
