"""The vector-Jacobian product of the MLX model: the gradient of the sequence-dependent features (msa_feat, target_feat) for given output cotangents, judged against the captured jax.vjp
(grad/* in the references) in float32. Recycle passes carry no gradient (recycled_alphafold_outputs stop_gradients them). The worker keeps the state before the final pass by call id, so a
backward pass does not repeat the recycle passes, and recomputes it when the id is unknown or evicted."""
import os
import signal
import time

import mlx.core as mx
import numpy as np
import pytest

import cases
from bindcraft.af.mlx_af2 import worker as worker_module
from bindcraft.af.mlx_af2.outputs import OUTPUT_KEYS
from bindcraft.af.mlx_af2.worker import RemoteModel, WorkerError
from helpers import assert_live, cosine, load_ref, rel_err
from test_model_multimer import build_model, features_from
from test_worker import MODEL, crop, model_config

DIFF = ('msa_feat', 'target_feat')

def cotangents_from(ref):
    return {name: mx.array(ref[f'cot/{name}']) for name in OUTPUT_KEYS if f'cot/{name}' in ref}

def key_of(ref):
    return mx.array(np.asarray(ref['meta']['key'], np.uint32))

@pytest.fixture(scope='module')
def model():
    return build_model()

@pytest.mark.integration
@pytest.mark.parametrize('case', ('small', 'medium'))
def test_vjp_matches_jax_vjp(model, case):
    ref = load_ref(case, 'fp32', 0)
    got = model.vjp(features_from(ref), key_of(ref), ref['meta']['num_recycle'], cotangents_from(ref), DIFF)
    assert tuple(got) == DIFF
    for name in DIFF:
        want = ref[f'grad/{name}']
        assert_live(want, f'grad/{name}')
        assert got[name].shape == want.shape and got[name].dtype == mx.float32, name
        print(case, name, f'cosine {cosine(got[name], want):.6f}  relative error {rel_err(got[name], want):.3e}')
        assert cosine(got[name], want) >= 0.999, (name, cosine(got[name], want))

@pytest.mark.integration
def test_cached_and_recomputed_prev_give_identical_gradients(model):
    #the third extra capture of the small case: one recycle and the large-complex initialization, so the final pass starts from initial_atom_pos = the recycle pass's positions
    ref = load_ref('small', 'fp32', 3)
    assert ref['meta']['num_recycle'] == 1 and ref['meta']['large_complex_initialization'] and 'in/initial_atom_pos' in ref
    features, key, cotangents = features_from(ref), key_of(ref), cotangents_from(ref)
    _, _, before = model.forward_with_state(features, key, 1)
    cached = model.vjp(features, key, 1, cotangents, DIFF, prev=before)
    recomputed = model.vjp(features, key, 1, cotangents, DIFF)
    for name in DIFF:
        assert np.array_equal(np.asarray(cached[name]), np.asarray(recomputed[name])), name
        assert rel_err(cached[name], ref[f'grad/{name}']) < 1e-2  #and both are the JAX gradient (cosine is gated in test_vjp_matches_jax_vjp)
    #control: a resume that forgets the initial_atom_pos rewrite computes a different gradient, so the equality above pins the rewrite
    original = model.final_features
    model.final_features = lambda features, prev, num_recycle: features
    try:
        forgetful = model.vjp(features, key, 1, cotangents, DIFF, prev=before)
    finally:
        model.final_features = original
    assert max(rel_err(forgetful[name], recomputed[name]) for name in DIFF) > 1e-3

@pytest.mark.integration
def test_only_requested_keys_are_returned(model):
    ref = load_ref('small', 'fp32', 1)  #no recycle: the cheapest capture
    features, key, cotangents = features_from(ref), key_of(ref), cotangents_from(ref)
    only_msa = model.vjp(features, key, 0, cotangents, ('msa_feat',))
    assert tuple(only_msa) == ('msa_feat',)
    #cotangents for some outputs only: the same gradient as giving the others zero cotangents
    subset = {name: cotangents[name] for name in ('distogram/logits',)}
    zeroed = {name: (value if name in subset else mx.zeros_like(value)) for name, value in cotangents.items()}
    got, want = model.vjp(features, key, 0, subset, DIFF), model.vjp(features, key, 0, zeroed, DIFF)
    for name in DIFF:
        assert rel_err(got[name], want[name]) < 1e-6, name
        assert np.abs(np.asarray(got[name])).max() > 0  #not a vacuous zero

@pytest.mark.integration
def test_no_gradient_flows_through_the_recycle_passes(model):
    #three recycles: the gradient is the final pass's alone and matches JAX's, which stop_gradients the recycle passes
    ref = load_ref('small', 'fp32', 2)
    assert ref['meta']['num_recycle'] == 3
    got = model.vjp(features_from(ref), key_of(ref), 3, cotangents_from(ref), DIFF)
    for name in DIFF:
        assert cosine(got[name], ref[f'grad/{name}']) >= 0.999, name
    #control on one recycle: the same chain with the previous state left differentiable is a different gradient, so the test can tell
    ref = load_ref('small', 'fp32', 0)
    features, key, cotangents = features_from(ref), key_of(ref), cotangents_from(ref)
    names = [name for name in OUTPUT_KEYS if name in cotangents]

    def leaky(msa_feat, target_feat):
        f = {**features, 'msa_feat': msa_feat, 'target_feat': target_feat}
        _, prev = model.single_pass(f, model.zero_prev(f), None, remat=True)
        outputs, _ = model.single_pass(f, prev, None, remat=True)
        flat = {f'{group}/{name}': value for group, tensors in outputs.items() for name, value in tensors.items()}
        return [flat[name].astype(mx.float32) for name in names]

    _, leaked = mx.vjp(leaky, [features['msa_feat'], features['target_feat']], [cotangents[name] for name in names])
    ours = model.vjp(features, key, 1, cotangents, DIFF)
    assert max(rel_err(leaked[index], ours[name]) for index, name in enumerate(DIFF)) > 1e-3

@pytest.fixture
def remote():
    RemoteModel.instances.clear()
    worker = RemoteModel(cases.params_path(), dtype='fp32')
    yield worker
    worker.close()

@pytest.mark.integration
def test_worker_vjp_matches_in_process_and_uses_the_cached_state(remote):
    ref = load_ref('small', 'fp32', 0)
    features = crop(ref, 30, 42)
    key = np.asarray(ref['meta']['key'], np.uint32)
    handle = remote.load(MODEL, model_config())
    outputs, call_id = remote.forward(handle, key, features, 1, None)
    cotangents = {name: np.random.default_rng(index).standard_normal(outputs[name].shape).astype(np.float32) for index, name in enumerate(OUTPUT_KEYS) if name not in ('structure_module/final_atom_mask', 'predicted_aligned_error/breaks')}
    local = build_model().vjp({name: mx.array(value) for name, value in features.items()}, mx.array(key), 1, {name: mx.array(value) for name, value in cotangents.items()}, DIFF)
    got = remote.vjp(handle, call_id, key, features, 1, None, cotangents, DIFF)
    assert tuple(got) == DIFF
    for name in DIFF:
        assert rel_err(got[name], np.asarray(local[name])) < 1e-5, name
    stats = remote.stats()
    assert stats['vjp_cache_hits'] == 1 and stats['vjp_cache_misses'] == 0
    again = remote.vjp(handle, call_id + 1000, key, features, 1, None, cotangents, DIFF)  #an id the worker never issued: the state is recomputed, the gradient is the same
    assert all(np.array_equal(got[name], again[name]) for name in DIFF)
    assert remote.stats()['vjp_cache_misses'] == 1

@pytest.mark.integration
def test_worker_keeps_only_the_last_eight_recycle_states(remote):
    ref = load_ref('small', 'fp32', 0)
    features = crop(ref, 30, 42)
    key = np.asarray(ref['meta']['key'], np.uint32)
    handle = remote.load(MODEL, model_config())
    ids = [remote.forward(handle, key, features, 0, None)[1] for _ in range(worker_module.MAX_CALLS + 1)]
    assert len(set(ids)) == len(ids)
    assert remote.stats()['cached_calls'] == worker_module.MAX_CALLS
    cotangent = {'distogram/logits': np.ones((12, 12, 64), np.float32)}
    first = remote.vjp(handle, ids[0], key, features, 0, None, cotangent, DIFF)  #the oldest was evicted by the ninth: recomputed
    last = remote.vjp(handle, ids[-1], key, features, 0, None, cotangent, DIFF)  #the newest is cached
    stats = remote.stats()
    assert stats['vjp_cache_misses'] == 1 and stats['vjp_cache_hits'] == 1
    assert all(np.array_equal(first[name], last[name]) for name in DIFF)  #the same inputs either way

@pytest.mark.integration
def test_a_call_id_issued_for_other_inputs_is_a_miss_not_a_wrong_gradient(remote):
    ref = load_ref('small', 'fp32', 0)
    features = crop(ref, 30, 42)
    key = np.asarray(ref['meta']['key'], np.uint32)
    handle = remote.load(MODEL, model_config())
    _, call_id = remote.forward(handle, key, features, 1, None)
    cotangent = {'distogram/logits': np.ones((12, 12, 64), np.float32)}
    remote.vjp(handle, call_id, np.asarray([7, 9], np.uint32), features, 1, None, cotangent, DIFF)  #another key
    remote.vjp(handle, call_id, key, features, 0, None, cotangent, DIFF)  #another number of recycles
    other = remote.load(MODEL, model_config(tag='other'))
    remote.vjp(other, call_id, key, features, 1, None, cotangent, DIFF)  #another model
    assert remote.stats()['vjp_cache_misses'] == 3 and remote.stats()['vjp_cache_hits'] == 0
    remote.vjp(handle, call_id, key, features, 1, None, cotangent, DIFF)
    assert remote.stats()['vjp_cache_hits'] == 1

@pytest.mark.integration
def test_call_ids_do_not_repeat_after_a_restart_and_an_old_id_is_a_miss(remote):
    ref = load_ref('small', 'fp32', 0)
    features = crop(ref, 30, 42)
    key = np.asarray(ref['meta']['key'], np.uint32)
    handle = remote.load(MODEL, model_config())
    _, before = remote.forward(handle, key, features, 0, None)
    os.kill(remote.pid, signal.SIGKILL)
    time.sleep(0.5)
    with pytest.raises(WorkerError, match='exit code -9'):
        remote.forward(handle, key, features, 0, None)
    _, after = remote.forward(handle, key, features, 0, None)  #a new worker, whose own count would have started again
    _, later = remote.forward(handle, key, features, 0, None)  #a count that restarts with the worker only collides from the second call on
    assert before + 1 < after < later  #the failed call used an id too
    remote.vjp(handle, before, key, features, 0, None, {'distogram/logits': np.ones((12, 12, 64), np.float32)}, DIFF)  #the backward pass of a call the old worker served
    stats = remote.stats()
    assert stats['vjp_cache_misses'] == 1 and stats['vjp_cache_hits'] == 0
