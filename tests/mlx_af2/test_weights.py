import os

import mlx.core as mx
import numpy as np
import pytest

import cases
from bindcraft.af.mlx_af2.weights import load_params

MODEL = cases.MULTIMER_MODEL

def raw_npz() -> dict:
    data = np.load(os.path.join(cases.params_path(), f'params_{MODEL}.npz'), allow_pickle=False)
    return {name: data[name] for name in data.files}

@pytest.fixture(scope='module')
def params32():
    return load_params(MODEL, cases.params_path(), dtype=mx.float32)

@pytest.mark.integration
def test_every_tensor_is_mapped_once(params32):
    raw = raw_npz()
    assert len(raw) == 330
    sources = [source for name in params32.names() for source in params32.sources(name)]
    assert sorted(sources) == sorted(raw), 'every raw tensor must feed exactly one loaded tensor'

@pytest.mark.integration
def test_fused_arrays_equal_the_jax_loaders(params32):
    from bindcraft.af.alphafold.model import data as af_data
    prefix = 'alphafold/alphafold_iteration/'
    reference = af_data.get_model_haiku_params(model_name=MODEL, data_dir=cases.params_path(), fuse=True)
    flat = {f'{scope}/{name}'.removeprefix(prefix): np.asarray(array) for scope, tensors in reference.items() for name, array in tensors.items()}
    assert sorted(params32.names()) == sorted(flat)
    for name, want in flat.items():
        got = np.asarray(params32.get(name))
        assert got.shape == want.shape and got.dtype == want.dtype and np.array_equal(got, want), name

@pytest.mark.integration
def test_missing_key_names_the_key(params32):
    with pytest.raises(KeyError, match='evoformer/not_a_module/weights'):
        params32.get('evoformer/not_a_module/weights')

def test_unknown_model_is_refused_by_name():
    with pytest.raises(ValueError, match='model_9_multimer_v3'):
        load_params('model_9_multimer_v3', cases.params_path())

@pytest.mark.integration
def test_bf16_cast_on_get_not_on_load(params32):
    low = load_params(MODEL, cases.params_path(), dtype=mx.bfloat16)
    name = 'evoformer/evoformer_iteration/triangle_multiplication_outgoing/gate/weights'
    assert low.get(name).dtype == mx.bfloat16 and params32.get(name).dtype == mx.float32
    assert low.stored(name).dtype == np.float32 and np.array_equal(low.stored(name), np.asarray(params32.get(name)))  #stored as float32, cast when read
    assert np.array_equal(np.asarray(low.get(name).astype(mx.float32)), np.asarray(mx.array(low.stored(name)).astype(mx.bfloat16).astype(mx.float32)))

@pytest.mark.integration
def test_layer_slices_the_stacked_axis(params32):
    scope = params32.scope('evoformer/evoformer_iteration/triangle_multiplication_outgoing')
    stacked = scope.get('gate/weights')
    assert stacked.shape[0] == 48
    assert np.array_equal(np.asarray(scope.layer(3).get('gate/weights')), np.asarray(stacked)[3])
    assert params32.scope('evoformer/extra_msa_stack/triangle_multiplication_outgoing').get('gate/weights').shape[0] == 4
    assert params32.get('distogram_head/half_logits/weights').shape == (128, 64)  #an unstacked tensor is returned whole, even through a layer view
    assert params32.layer(5).get('distogram_head/half_logits/weights').shape == (128, 64)

@pytest.mark.integration
def test_unused_reports_what_was_not_read():
    params = load_params(MODEL, cases.params_path(), dtype=mx.float32)
    assert len(params.unused()) == len(params.names())
    params.get('evoformer/evoformer_iteration/triangle_multiplication_outgoing/gate/weights')
    assert 'evoformer/evoformer_iteration/triangle_multiplication_outgoing/gate/weights' not in params.unused()
    with pytest.raises(AssertionError, match='unused'):
        params.assert_all_used()
