"""The block stacks' explicit reverse loop and the tape a forward call keeps for it (evoformer.evoformer_stack with remat=True and tape=): the gradient must equal autodiff through plain blocks, a tape must make the
vjp skip the stack's forward pass without changing a bit of the result, and a tape that does not belong to the call must be refused. fp32 on the CPU device against the plain-block autodiff oracle."""
import mlx.core as mx
import numpy as np
import pytest

import cases
import jax_units
from bindcraft.af.mlx_af2 import evoformer
from bindcraft.af.mlx_af2.weights import load_params
from helpers import assert_live, rel_err

pytestmark = pytest.mark.integration

MAIN, EXTRA = 'evoformer/evoformer_iteration', 'evoformer/extra_msa_stack'
ROWS, N = 3, 10

@pytest.fixture(scope='module')
def params():
    return load_params(jax_units.MODEL, cases.params_path(), dtype=mx.float32)

def config():
    return jax_units.configs()[0].embeddings_and_evoformer.evoformer.to_dict()

def inputs(msa_channels, seed, residues=N):
    rng = np.random.default_rng(seed)
    msa = (0.5 * rng.standard_normal((ROWS, residues, msa_channels))).astype(np.float32)
    pair = (0.5 * rng.standard_normal((residues, residues, 128))).astype(np.float32)
    live = np.ones(residues, np.float32)
    live[-2:] = 0.0
    msa_mask, pair_mask = np.ones((ROWS, residues), np.float32) * live, live[:, None] * live[None, :]
    return tuple(mx.array(array) for array in (msa, pair, msa_mask, pair_mask))

def loss_of(outputs, only_pair=False):
    msa, pair = outputs
    return mx.sum(pair * pair) * 1e-3 if only_pair else mx.sum(msa * msa) * 1e-3 + mx.sum(pair * pair) * 1e-3

def stack(params, scope, msa, pair, msa_mask, pair_mask, **options):
    return evoformer.evoformer_stack(params.scope(scope), msa, pair, msa_mask, pair_mask, config(), **options)

def gradients(params, scope, tensors, only_pair=False, **options):
    msa, pair, msa_mask, pair_mask = tensors
    return mx.grad(lambda m, q: loss_of(stack(params, scope, m, q, msa_mask, pair_mask, **options), only_pair), argnums=(0, 1))(msa, pair)

def test_explicit_reverse_loop_matches_autodiff_through_plain_blocks(params):
    tensors = inputs(256, 1)
    plain = gradients(params, MAIN, tensors, num_block=3, remat=False)
    explicit = gradients(params, MAIN, tensors, num_block=3, remat=True)
    for got, want in zip(explicit, plain):
        assert_live(np.asarray(want))
        assert rel_err(np.asarray(got), np.asarray(want)) < 1e-5

def test_extra_stack_gradient_is_right_when_only_pair_is_used(params):
    #the extra-MSA stack's MSA output is dropped by its caller: its cotangent is zero, and the pair gradient must still be the plain one
    tensors = inputs(64, 2)
    plain = gradients(params, EXTRA, tensors, only_pair=True, num_block=2, remat=False, is_extra=True)
    explicit = gradients(params, EXTRA, tensors, only_pair=True, num_block=2, remat=True, is_extra=True)
    for got, want in zip(explicit, plain):
        assert_live(np.asarray(want))
        assert rel_err(np.asarray(got), np.asarray(want)) < 1e-5

def test_gradient_with_dropout_keys_matches_plain_blocks_and_dropout_is_live(params):
    tensors, key = inputs(256, 3), mx.random.key(11)
    plain = gradients(params, MAIN, tensors, num_block=3, remat=False, dropout_key=key)
    explicit = gradients(params, MAIN, tensors, num_block=3, remat=True, dropout_key=key)
    without = gradients(params, MAIN, tensors, num_block=3, remat=True)
    for got, want, undropped in zip(explicit, plain, without):
        assert_live(np.asarray(want))
        assert rel_err(np.asarray(got), np.asarray(want)) < 1e-5
        assert rel_err(np.asarray(undropped), np.asarray(want)) > 1e-3  #the keys really change the gradient: the comparison above is not between two undropped runs

def test_tape_records_one_input_per_block_and_the_outputs(params):
    msa, pair, msa_mask, pair_mask = inputs(256, 4)
    tape = evoformer.Tape()
    out_msa, out_pair = stack(params, MAIN, msa, pair, msa_mask, pair_mask, num_block=3, remat=False, tape=tape)
    assert len(tape.inputs) == 3
    assert np.array_equal(np.asarray(tape.inputs[0][0]), np.asarray(msa)) and np.array_equal(np.asarray(tape.inputs[0][1]), np.asarray(pair))
    first_out = evoformer.evoformer_iteration(params.scope(MAIN).layer(0), msa, pair, msa_mask, pair_mask, config())
    assert rel_err(np.asarray(tape.inputs[1][0]), np.asarray(first_out[0])) < 1e-6 and rel_err(np.asarray(tape.inputs[1][1]), np.asarray(first_out[1])) < 1e-6  #block 1 starts where block 0 ended
    assert np.array_equal(np.asarray(tape.outputs[0]), np.asarray(out_msa)) and np.array_equal(np.asarray(tape.outputs[1]), np.asarray(out_pair))
    assert tape.nbytes == sum(array.nbytes for pair_ in tape.inputs for array in pair_) + sum(array.nbytes for array in tape.outputs)

def test_vjp_from_a_tape_equals_the_vjp_without_one(params):
    tensors = inputs(256, 5)
    tape = evoformer.Tape()
    stack(params, MAIN, *tensors, num_block=3, remat=False, tape=tape)
    without = gradients(params, MAIN, tensors, num_block=3, remat=True)
    with_tape = gradients(params, MAIN, tensors, num_block=3, remat=True, tape=tape)
    for got, want in zip(with_tape, without):
        assert_live(np.asarray(want))
        assert rel_err(np.asarray(got), np.asarray(want)) < 1e-6

def test_a_tape_skips_the_forward_pass_of_the_stack(params, monkeypatch):
    tensors, calls, original = inputs(256, 6), {'blocks': 0}, evoformer.evoformer_iteration

    def counting(*args, **kwargs):
        calls['blocks'] += 1
        return original(*args, **kwargs)

    tape = evoformer.Tape()
    stack(params, MAIN, *tensors, num_block=3, remat=False, tape=tape)
    monkeypatch.setattr(evoformer, 'evoformer_iteration', counting)
    gradients(params, MAIN, tensors, num_block=3, remat=True)
    without_tape = calls['blocks']
    calls['blocks'] = 0
    gradients(params, MAIN, tensors, num_block=3, remat=True, tape=tape)
    assert without_tape == 6 and calls['blocks'] == 3  #no tape: the forward (3) and the recompute in the reverse loop (3); with one, only the recompute

def test_a_tape_that_does_not_belong_to_the_call_is_refused(params):
    tensors = inputs(256, 7)
    tape = evoformer.Tape()
    stack(params, MAIN, *tensors, num_block=3, remat=False, tape=tape)
    with pytest.raises(ValueError, match='tape'):
        gradients(params, MAIN, tensors, num_block=2, remat=True, tape=tape)  #recorded for 3 blocks
    other = inputs(256, 8, residues=N + 2)
    with pytest.raises(ValueError, match='tape'):
        gradients(params, MAIN, other, num_block=3, remat=True, tape=tape)  #recorded at 10 residues
    with pytest.raises(ValueError, match='tape'):
        gradients(params, MAIN, tensors, num_block=3, remat=True, tape=evoformer.Tape())  #an empty tape is not a recording
