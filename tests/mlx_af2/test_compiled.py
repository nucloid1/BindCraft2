"""Compiled Evoformer blocks (compiled.py): the block's parameters reach one compiled function per stack as a list of arrays, so the 48 layers share one trace per shape. Compiled must equal uncompiled (fp32 tightly,
bf16 within one bf16 ulp) in the forward and in the explicit reverse loop, with and without dropout keys. Compile is on by default on the GPU only, and BINDCRAFT_MLX_COMPILE=0 turns it off."""
import mlx.core as mx
import numpy as np
import pytest

import cases
import jax_units
from bindcraft.af.mlx_af2 import compiled, evoformer
from bindcraft.af.mlx_af2.weights import load_params
from helpers import assert_live, rel_err

pytestmark = pytest.mark.integration

MAIN, EXTRA = 'evoformer/evoformer_iteration', 'evoformer/extra_msa_stack'
ROWS, N = 3, 10

def config():
    return jax_units.configs()[0].embeddings_and_evoformer.evoformer.to_dict()

def inputs(msa_channels, seed, dtype=mx.float32, residues=N):
    rng = np.random.default_rng(seed)
    msa = (0.5 * rng.standard_normal((ROWS, residues, msa_channels))).astype(np.float32)
    pair = (0.5 * rng.standard_normal((residues, residues, 128))).astype(np.float32)
    live = np.ones(residues, np.float32)
    live[-2:] = 0.0
    return tuple(mx.array(array).astype(dtype) for array in (msa, pair, np.ones((ROWS, residues), np.float32) * live, live[:, None] * live[None, :]))

def test_compile_is_on_by_default_on_the_gpu_only_and_the_kill_switch_turns_it_off(monkeypatch):
    monkeypatch.delenv(compiled.COMPILE_ENV, raising=False)
    try:
        mx.set_default_device(mx.cpu)
        assert not compiled.enabled()  #on the CPU, MLX compiles fused kernels with a C++ compiler: seconds per kernel
        mx.set_default_device(mx.gpu)  #only the device setting is read; nothing runs on the GPU here
        assert compiled.enabled()
        monkeypatch.setenv(compiled.COMPILE_ENV, '0')
        assert not compiled.enabled()
    finally:
        mx.set_default_device(mx.cpu)

@pytest.mark.parametrize('scope,is_extra,channels', ((MAIN, False, 256), (EXTRA, True, 64)))
@pytest.mark.parametrize('with_key', (False, True))
@pytest.mark.parametrize('dtype', (mx.float32, mx.bfloat16))  #in bf16 the layer norms read float32 parameters and the linears bf16 ones: the spec must keep the dtype of each read
def test_array_params_serve_exactly_what_the_block_reads(scope, is_extra, channels, with_key, dtype):
    params = load_params(jax_units.MODEL, cases.params_path(), dtype=dtype).scope(scope)
    msa, pair, msa_mask, pair_mask = inputs(channels, 1, dtype)
    key = mx.random.key(3) if with_key else None
    spec = compiled.discover(params, config(), msa, pair, msa_mask, pair_mask, is_extra=is_extra, with_key=with_key)
    layer = compiled.ArrayParams(dict(zip(spec, compiled.layer_weights(params, 2, spec))), params.store.dtype)
    want = evoformer.evoformer_iteration(params.layer(2), msa, pair, msa_mask, pair_mask, config(), dropout_key=key, is_extra=is_extra)
    got = evoformer.evoformer_iteration(layer, msa, pair, msa_mask, pair_mask, config(), dropout_key=key, is_extra=is_extra)
    for g, w in zip(got, want):
        assert g.dtype == w.dtype
        assert_live(np.asarray(w.astype(mx.float32)))
        assert np.array_equal(np.asarray(g.astype(mx.float32)), np.asarray(w.astype(mx.float32)))

def test_a_parameter_outside_the_spec_is_a_key_error_naming_it():
    layer = compiled.ArrayParams({('a/weights', mx.float32): mx.zeros((2,))}, mx.float32)
    with pytest.raises(KeyError, match='b/weights'):
        layer.scope('b').get('weights')
    assert layer.scope('a').has('weights') and not layer.scope('a').has('bias')

def stack_results(scope, is_extra, channels, dtype, key, compile_on, monkeypatch):
    monkeypatch.setenv(compiled.COMPILE_ENV, '1' if compile_on else '0')
    params = load_params(jax_units.MODEL, cases.params_path(), dtype=dtype).scope(scope)
    msa, pair, msa_mask, pair_mask = inputs(channels, 5, dtype)
    run = lambda m, q, remat: evoformer.evoformer_stack(params, m, q, msa_mask, pair_mask, config(), num_block=3, remat=remat, dropout_key=key, is_extra=is_extra)
    forward = run(msa, pair, False)
    rng = np.random.default_rng(6)
    cotangents = [mx.array(rng.standard_normal(array.shape).astype(np.float32)).astype(dtype) for array in forward]
    gradients = mx.vjp(lambda m, q: list(run(m, q, True)), [msa, pair], cotangents)[1]
    mx.eval(forward, gradients)
    return [np.asarray(array.astype(mx.float32)) for array in (*forward, *gradients)]

@pytest.mark.gpu
@pytest.mark.parametrize('dtype,tolerance', ((mx.float32, 1e-5), (mx.bfloat16, 8e-3)))
@pytest.mark.parametrize('scope,is_extra,channels', ((MAIN, False, 256), (EXTRA, True, 64)))
@pytest.mark.parametrize('with_key', (False, True))
def test_compiled_stack_equals_uncompiled_forward_and_reverse_loop(dtype, tolerance, scope, is_extra, channels, with_key, monkeypatch):
    key = mx.random.key(9) if with_key else None
    plain = stack_results(scope, is_extra, channels, dtype, key, False, monkeypatch)
    fused = stack_results(scope, is_extra, channels, dtype, key, True, monkeypatch)
    for got, want in zip(fused, plain):
        assert_live(want)
        assert rel_err(got, want) < tolerance  #one bf16 ulp is 2^-7 = 7.8e-3

@pytest.mark.gpu
def test_one_compiled_function_serves_every_layer_and_a_new_shape_traces_once_more(monkeypatch):
    monkeypatch.setenv(compiled.COMPILE_ENV, '1')
    params = load_params(jax_units.MODEL, cases.params_path(), dtype=mx.bfloat16).scope(MAIN)
    before = dict(compiled.TRACES)
    for residues in (N + 1, N + 1, N + 5):  #shapes no other test in the module traces, so the counts are this test's own
        msa, pair, msa_mask, pair_mask = inputs(256, 7, mx.bfloat16, residues)
        tape = evoformer.Tape()
        mx.eval(evoformer.evoformer_stack(params, msa, pair, msa_mask, pair_mask, config(), num_block=3, remat=False, tape=tape))  #the forward the design call runs
        mx.eval(evoformer.stack_vjp(params, tape, [mx.ones_like(msa), mx.ones_like(pair)], msa_mask, pair_mask, config()))  #the reverse loop the model's vjp calls directly
    new = {name: compiled.TRACES[name] - before.get(name, 0) for name in ('forward', 'backward')}
    assert new == {'forward': 2, 'backward': 2}, new  #3 blocks x 3 runs, but one trace per shape: 11 residues (twice) and 15 residues

@pytest.mark.gpu
def test_the_models_vjp_runs_the_compiled_blocks_outside_any_transformation():
    #MLX runs a compiled function inline, unfused and retraced on every call, when it is called inside another transformation (mx.vjp, mx.grad): the model's vjp must call the stacks' compiled vjp
    #closures directly, so a second vjp at the same shape traces nothing new
    from helpers import load_ref
    from test_model_multimer import build_model, features_from
    from test_vjp import DIFF, cotangents_from, key_of
    model = build_model()
    ref = load_ref('small', 'fp32', 0)
    features, key, cotangents = features_from(ref), key_of(ref), cotangents_from(ref)
    _, _, before = model.forward_with_state(features, key, 1)
    model.vjp(features, key, 1, cotangents, DIFF, prev=before)
    traced = dict(compiled.TRACES)
    model.vjp(features, key, 1, cotangents, DIFF, prev=before)
    assert dict(compiled.TRACES) == traced
