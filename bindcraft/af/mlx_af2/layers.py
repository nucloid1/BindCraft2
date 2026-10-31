#The model's basic layers as functions of a Params scope, in the arithmetic and dtype flow of the haiku modules they port (common_modules.py, modules.py, accel.py). Arguments follow the
#JAX __call__ minus is_training and safe_key. A weight is cast to the dtype of the activation it meets, as haiku's bfloat16 getter does; norms keep float32 parameters.
import math

import mlx.core as mx

LOGIT_CLIP = 1e8
ATTENTION_CHUNK = 256

def sharded(function, shard_size: int | None, batched: tuple, nonbatched: tuple = ()):
    #mapping.inference_subbatch: the batched arguments are cut along axis 0 into shards of shard_size (the last may be smaller), the outputs joined along axis 0
    if shard_size is None:
        return function(*batched, *nonbatched)
    size = batched[0].shape[0]
    return mx.concatenate([function(*(array[start:start + shard_size] for array in batched), *nonbatched) for start in range(0, size, shard_size)], axis=0)

def linear(p, x: mx.array, *, num_input_dims: int = 1) -> mx.array:
    weights = p.get('weights', x.dtype)
    in_shape, out_shape = weights.shape[:num_input_dims], weights.shape[num_input_dims:]
    lead = x.shape[:x.ndim - num_input_dims]
    out = (x.reshape(*lead, math.prod(in_shape)) @ weights.reshape(math.prod(in_shape), math.prod(out_shape))).reshape(*lead, *out_shape)
    return out + p.get('bias', x.dtype) if p.has('bias') else out

def layer_norm(p, x: mx.array, eps: float = 1e-5, fast_variance: bool = False) -> mx.array:
    #the repo's LayerNorm: bfloat16 input is normalised in float32 with float32 parameters and cast back. fast_variance is haiku's E[x^2] - E[x]^2 clamped at zero, which the triangle
    #multiplication's two norms use (modules._layer_norm); it differs from the exact variance when the mean is large
    low = x.dtype == mx.bfloat16
    x = x.astype(mx.float32) if low else x
    scale, offset = p.get('scale', mx.float32), p.get('offset', mx.float32)
    if fast_variance:
        mean = mx.mean(x, axis=-1, keepdims=True)
        variance = mx.maximum(mx.mean(mx.square(x), axis=-1, keepdims=True) - mx.square(mean), 0.0)
        out = (x - mean) * mx.rsqrt(variance + eps) * scale + offset
    else:
        out = mx.fast.layer_norm(x, scale, offset, eps)
    return out.astype(mx.bfloat16) if low else out

def slice_queries(tensor, start: int, end: int):
    return tensor if tensor is None or tensor.shape[-2] == 1 else tensor[..., start:end, :]

def stock_attend(q, k, v, bias, nonbatched_bias):
    #accel._attend_stock: logits [batch, head, query, key] plus the biases, clipped, softmax over the keys. precise=True: MLX's default softmax on bfloat16 is about 12x less accurate
    #(1.9e-2 against 1.6e-3 relative to the exact result), which made the bf16 model's logits worse than the JAX bf16 model's; float32 inputs are unaffected
    logits = q.transpose(0, 2, 1, 3) @ k.transpose(0, 2, 3, 1) + bias
    if nonbatched_bias is not None:
        logits = logits + nonbatched_bias[None]
    weights = mx.softmax(mx.clip(logits, -LOGIT_CLIP, LOGIT_CLIP), axis=-1, precise=True)
    return (weights @ v.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)

def attend(q, k, v, bias, nonbatched_bias=None, chunk_size: int | None = ATTENTION_CHUNK):
    #accel._attend_chunked: the queries in chunks, the same arithmetic, a smaller logits tensor
    query_count = q.shape[1]
    if not chunk_size or chunk_size >= query_count:
        return stock_attend(q, k, v, bias, nonbatched_bias)
    return mx.concatenate([stock_attend(q[:, start:start + chunk_size], k, v, slice_queries(bias, start, start + chunk_size), slice_queries(nonbatched_bias, start, start + chunk_size))
                           for start in range(0, query_count, chunk_size)], axis=1)

def attention(p, q_data: mx.array, m_data: mx.array, bias: mx.array, nonbatched_bias, cfg: dict, *, subbatch_size: int | None = None, chunk_size: int | None = ATTENTION_CHUNK) -> mx.array:
    if subbatch_size is not None:
        return sharded(lambda q, m, b: attention(p, q, m, b, nonbatched_bias, cfg, chunk_size=chunk_size), subbatch_size, (q_data, m_data, bias))
    dtype, num_head = q_data.dtype, cfg['num_head']
    key_dim, value_dim = cfg.get('key_dim', q_data.shape[-1]) // num_head, cfg.get('value_dim', m_data.shape[-1]) // num_head
    batch, queries, keys = q_data.shape[0], q_data.shape[1], m_data.shape[1]
    q = (q_data @ p.get('query_w', dtype).reshape(q_data.shape[-1], num_head * key_dim)).reshape(batch, queries, num_head, key_dim) * key_dim ** -0.5
    k = (m_data @ p.get('key_w', dtype).reshape(m_data.shape[-1], num_head * key_dim)).reshape(batch, keys, num_head, key_dim)
    v = (m_data @ p.get('value_w', dtype).reshape(m_data.shape[-1], num_head * value_dim)).reshape(batch, keys, num_head, value_dim)
    weighted_avg = attend(q, k, v, bias, nonbatched_bias, chunk_size)
    if cfg['gating']:
        gate = (q_data @ p.get('gating_w', dtype).reshape(q_data.shape[-1], num_head * value_dim)).reshape(batch, queries, num_head, value_dim) + p.get('gating_b', dtype)
        weighted_avg = weighted_avg * mx.sigmoid(gate)
    output_weights = p.get('output_w', dtype)
    return weighted_avg.reshape(batch, queries, num_head * value_dim) @ output_weights.reshape(num_head * value_dim, output_weights.shape[-1]) + p.get('output_b', dtype)

def global_attention(p, q_data: mx.array, m_data: mx.array, q_mask: mx.array, cfg: dict) -> mx.array:
    #GlobalAttention: one averaged query per head attends over the keys (the JAX module's bias argument is dead: the bias is rebuilt from q_mask)
    dtype, num_head = q_data.dtype, cfg['num_head']
    key_dim, value_dim = cfg.get('key_dim', q_data.shape[-1]) // num_head, cfg.get('value_dim', m_data.shape[-1]) // num_head
    batch = q_data.shape[0]
    v = m_data @ p.get('value_w', dtype)
    q_avg = mx.sum(q_mask * q_data, axis=1) / (mx.sum(q_mask, axis=1) + 1e-10)
    q = (q_avg @ p.get('query_w', dtype).reshape(q_data.shape[-1], num_head * key_dim)).reshape(batch, num_head, key_dim) * key_dim ** -0.5
    k = m_data @ p.get('key_w', dtype)
    bias = 1e9 * (q_mask[:, None, :, 0] - 1.0)
    weights = mx.softmax(q @ k.transpose(0, 2, 1) + bias, axis=-1, precise=True)
    weighted_avg = weights @ v
    output_weights = p.get('output_w', dtype)
    flat_output = output_weights.reshape(num_head * value_dim, output_weights.shape[-1])
    if cfg['gating']:
        gate = mx.sigmoid((q_data @ p.get('gating_w', dtype).reshape(q_data.shape[-1], num_head * value_dim)).reshape(batch, q_data.shape[1], num_head, value_dim) + p.get('gating_b', dtype))
        gated = weighted_avg[:, None] * gate
        return gated.reshape(batch, q_data.shape[1], num_head * value_dim) @ flat_output + p.get('output_b', dtype)
    return (weighted_avg.reshape(batch, num_head * value_dim) @ flat_output + p.get('output_b', dtype))[:, None]

def transition(p, act: mx.array, mask: mx.array, cfg: dict, *, subbatch_size: int | None = None) -> mx.array:
    #Transition: layer norm, then a two-layer relu MLP in shards; the JAX module computes a mask it never applies, so mask is unused here too
    act = layer_norm(p.scope('input_layer_norm'), act)
    return sharded(lambda chunk: linear(p.scope('transition2'), mx.maximum(linear(p.scope('transition1'), chunk), 0)), subbatch_size, (act,))
