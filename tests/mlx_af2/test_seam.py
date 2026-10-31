"""The seam, proven before any porting: the jitted loss of the real AlphaFoldDesignModel runs with the model call replaced by the custom_vjp bridge over a stub worker in
another process, and every gradient, state, dtype, error and runtime argument that has to cross it does."""
import os
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import cases
from bindcraft.af.mlx_af2 import bridge
from bindcraft.af.mlx_af2.outputs import OUTPUT_KEYS, output_shapes
from seam_support import BOTH_MODELS, BridgedModel, StandInModel, jax_stand_in
from stub_remote import StubRemote

@pytest.fixture
def remote():
    stub = StubRemote()
    yield stub
    stub.close()

def live(array) -> bool:
    array = np.asarray(array, dtype=np.float64)
    return bool(np.isfinite(array).all() and array.std() > 1e-8)

def relative_error(got, want) -> float:
    got, want = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
    return float(np.linalg.norm(got - want) / (np.linalg.norm(want) + 1e-30))

def gradients_and_loss(model, case):
    _, states, _, losses = cases.design_inputs(case)
    _, gradients, loss = model.sequence_gradients(states, losses, model=cases.MULTIMER_MODEL)
    return gradients, loss

def test_gradient_reaches_the_sequence_logits(remote):
    want, want_loss = gradients_and_loss(cases.build_model(StandInModel), 'small')
    got, got_loss = gradients_and_loss(cases.build_model(BridgedModel, remote=remote), 'small')
    assert set(got) == set(want) and got
    for chain in want:
        assert live(want[chain]) and live(got[chain]), chain
        assert relative_error(got[chain], want[chain]) < 1e-4, chain
    assert abs(float(got_loss) - float(want_loss)) < 1e-4 * max(1.0, abs(float(want_loss)))
    assert [call['op'] for call in remote.calls()] == ['forward', 'vjp']

def test_two_states_in_one_jit_call(remote):
    want, _ = gradients_and_loss(cases.build_model(StandInModel, case='pair'), 'pair')
    model = cases.build_model(BridgedModel, case='pair', remote=remote)
    got, _ = gradients_and_loss(model, 'pair')
    for chain in want:
        assert relative_error(got[chain], want[chain]) < 1e-4, chain
    forwards = [call for call in remote.calls() if call['op'] == 'forward']
    assert len(forwards) == 2 and model.trace_count == 2  #one call per state, each traced once
    assert remote.max_active == 1  #xla may run the two states' callbacks on different threads; the bridge must serialize them, a worker has one pipe

def test_integer_inputs_get_float0_cotangents(remote):
    features = cases.model_inputs('small')
    handle, key = jnp.asarray(1, jnp.int32), jax.random.PRNGKey(0)
    assert any(jnp.issubdtype(value.dtype, jnp.integer) for value in features.values() if hasattr(value, 'dtype'))
    def total(target_feat):
        outputs = bridge.mlx_model_outputs(remote, handle, key, {**features, 'target_feat': target_feat}, 1, None)
        return jnp.sum(outputs['distogram']['logits'] ** 2) + jnp.sum(outputs['predicted_lddt']['logits'])
    gradient = jax.grad(total)(features['target_feat'])
    want = jax.grad(lambda t: jnp.sum(jax_stand_in(handle, {**features, 'target_feat': t})['distogram']['logits'] ** 2) + jnp.sum(jax_stand_in(handle, {**features, 'target_feat': t})['predicted_lddt']['logits']))(features['target_feat'])
    assert gradient.shape == features['target_feat'].shape and live(gradient) and relative_error(gradient, want) < 1e-4

def test_compile_only_never_calls_the_remote(remote):
    _, states, _, losses = cases.design_inputs('small')
    model = cases.build_model(BridgedModel, remote=remote)
    model.predict(states, model=cases.MULTIMER_MODEL, compile_only=True, compile_blocking=False)
    model.sequence_gradients(states, losses, model=cases.MULTIMER_MODEL, compile_only=True)
    assert remote.calls() == [] and model.trace_count >= 2  #both functions were traced and compiled, neither executed

def test_remote_error_surfaces_naming_the_cause(remote):
    _, states, _, losses = cases.design_inputs('small')
    model = cases.build_model(BridgedModel, remote=remote)
    remote.fail_next('worker killed: exit code -9 (memory cap 28.0 GiB)')
    with pytest.raises(Exception) as failure:
        model.sequence_gradients(states, losses, model=cases.MULTIMER_MODEL)
    assert 'exit code -9' in str(failure.value) and '28.0 GiB' in str(failure.value)

def test_key_and_dropout_flag_arrive_as_values(remote):
    _, states, _, _ = cases.design_inputs('small')
    model = cases.build_model(BridgedModel, remote=remote)
    model.key, model.dropout = jax.random.PRNGKey(1), False
    model.predict(states, model=cases.MULTIMER_MODEL)
    model.key, model.dropout = jax.random.PRNGKey(2), True
    model.predict(states, model=cases.MULTIMER_MODEL)
    first, second = remote.calls()
    assert first['key'] == tuple(np.asarray(jax.random.PRNGKey(1)).tolist()) and second['key'] == tuple(np.asarray(jax.random.PRNGKey(2)).tolist())
    assert (first['use_dropout'], second['use_dropout']) == (False, True)
    assert model.trace_count == 1  #the second call reused the compiled function

def test_model_identity_is_a_runtime_argument(remote):
    _, states, _, _ = cases.design_inputs('small')
    model = cases.build_model(BridgedModel, remote=remote, models=BOTH_MODELS)
    first = model.predict(states, model='model_1_multimer_v3')
    second = model.predict(states, model='model_2_multimer_v3')
    assert [call['handle'] for call in remote.calls()] == [0, 1] and model.trace_count == 1
    name = next(iter(first))
    assert not np.allclose(np.asarray(first[name].metrics['plddt']), np.asarray(second[name].metrics['plddt']))

def test_stub_runs_in_another_process(remote):
    assert remote.pid != os.getpid() and remote.process.is_alive()

def test_front_always_starts_from_zero_prev():
    #the bridge does not ship prev to the worker (a pair-sized zero array) and the worker starts its recycles from zeros: pin that this is what the front builds
    for large_complex in (False, True):
        prev = cases.model_inputs('small', large_complex_initialization=large_complex)['prev']
        assert set(prev) == {'prev_msa_first_row', 'prev_pair', 'prev_pos'} and all(float(jnp.abs(value).max()) == 0.0 for value in prev.values())

def test_output_shapes_cover_the_output_keys():
    assert tuple(output_shapes(147)) == OUTPUT_KEYS
