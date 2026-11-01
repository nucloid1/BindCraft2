#The seam: AlphaFold inside the jitted loss becomes a jax.custom_vjp whose forward and backward each call the MLX worker through jax.pure_callback. jax's own autodiff then yields
#forward, loss, backward with the front, the loss and the optimizer untouched. The forward rule asks the worker to keep the final pass's block inputs (a tape) under the call id, so the vjp that follows does not
#repeat the final pass's Evoformer; a plain forward (validation, prediction) keeps none.
import functools
import threading

import jax
import jax.numpy as jnp
import numpy as np

from bindcraft.af.mlx_af2.outputs import OUTPUT_KEYS, nest, output_shapes

#the only float features that depend on the design sequence (checked against a jvp of the front in the predictor tests); every other float feature gets a zero cotangent
DIFF_KEYS = ('msa_feat', 'target_feat')
#the front always builds zeros here (pinned by a test), so the worker starts its recycles from zeros instead of being sent a pair-sized array of them on every call
DROPPED_FEATURES = ('prev',)
#the callbacks of independent states (the complex and the binder alone) are independent ops in the compiled graph, so xla runs them on different threads at once; the worker has one pipe and one GPU, so its calls are serialized here
REMOTE_CALLS = threading.Lock()

def output_spec(length: int) -> dict[str, jax.ShapeDtypeStruct]:
    return {name: jax.ShapeDtypeStruct(shape, np.dtype(dtype)) for name, (shape, dtype) in output_shapes(length).items()}

def raw_key(key: jax.Array) -> jax.Array:
    return jax.random.key_data(key) if jnp.issubdtype(key.dtype, jax.dtypes.prng_key) else key

def float0_zeros(array) -> np.ndarray:
    return np.zeros(np.shape(array), dtype=jax.dtypes.float0)

def mlx_model_outputs(remote, model_handle: jax.Array, key: jax.Array, model_inputs: dict, num_recycle: int, subbatch_size: int | None) -> dict:
    #model_handle is a traced integer: both compile caches are keyed by model family, so which model runs has to be a runtime argument, never a name baked in at trace time
    features = {name: value for name, value in model_inputs.items() if name not in DROPPED_FEATURES}
    spec = output_spec(features['aatype'].shape[0])
    result_spec = tuple(spec[name] for name in OUTPUT_KEYS)
    call_id_spec = jax.ShapeDtypeStruct((), jnp.int32)
    gradient_spec = {name: jax.ShapeDtypeStruct(features[name].shape, features[name].dtype) for name in DIFF_KEYS}

    def numpy_features(arrays: dict) -> dict:
        return {name: np.asarray(value) for name, value in arrays.items()}

    def host_forward(handle, key, arrays, retain=False):
        with REMOTE_CALLS:
            outputs, call_id = remote.forward(int(handle), np.asarray(key), numpy_features(arrays), num_recycle, subbatch_size, retain=retain)
        return tuple(np.asarray(outputs[name], dtype=spec[name].dtype) for name in OUTPUT_KEYS), np.int32(call_id)

    def host_vjp(handle, call_id, key, arrays, cotangents):
        with REMOTE_CALLS:
            gradients = remote.vjp(int(handle), int(call_id), np.asarray(key), numpy_features(arrays), num_recycle, subbatch_size, {name: np.asarray(value) for name, value in zip(OUTPUT_KEYS, cotangents)}, DIFF_KEYS)
        return {name: np.asarray(gradients[name], dtype=gradient_spec[name].dtype) for name in DIFF_KEYS}

    @jax.custom_vjp
    def run(handle, key, arrays):
        outputs, _ = jax.pure_callback(host_forward, (result_spec, call_id_spec), handle, key, arrays)
        return outputs

    def run_forward(handle, key, arrays):
        outputs, call_id = jax.pure_callback(functools.partial(host_forward, retain=True), (result_spec, call_id_spec), handle, key, arrays)
        return outputs, (handle, key, arrays, call_id)

    def run_backward(residuals, cotangents):
        handle, key, arrays, call_id = residuals
        gradients = jax.pure_callback(host_vjp, gradient_spec, handle, call_id, key, arrays, cotangents)
        array_cotangents = {name: gradients[name] if name in gradients else jnp.zeros_like(value) if jnp.issubdtype(value.dtype, jnp.floating) else float0_zeros(value) for name, value in arrays.items()}
        return float0_zeros(handle), float0_zeros(key), array_cotangents

    run.defvjp(run_forward, run_backward)
    return nest(dict(zip(OUTPUT_KEYS, run(model_handle, raw_key(key), features))))
