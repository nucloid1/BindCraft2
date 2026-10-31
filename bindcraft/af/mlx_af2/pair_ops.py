#The pair stack's operations (modules.TriangleMultiplication in its fused-projection form, which the fused weights serve, and modules.TriangleAttention), as functions of a Params scope.
#Both return what the JAX modules return: an update, not the residual sum.
import mlx.core as mx

from bindcraft.af.mlx_af2 import layers

def triangle_multiplication(p, left_act: mx.array, left_mask: mx.array, cfg: dict) -> mx.array:
    channels = cfg['num_intermediate_channel']
    mask = left_mask[..., None]
    act = layers.layer_norm(p.scope('left_norm_input'), left_act, fast_variance=True)
    projected = mask * layers.linear(p.scope('projection'), act)
    projected = projected * mx.sigmoid(layers.linear(p.scope('gate'), act))
    #outgoing 'ikc,jkc->ijc', incoming 'kjc,kic->ijc'; the left half of the channels is a, the right half b
    contracted = mx.einsum(cfg['equation'], projected[:, :, :channels], projected[:, :, channels:])
    out = layers.linear(p.scope('output_projection'), layers.layer_norm(p.scope('center_norm'), contracted, fast_variance=True))
    return out * mx.sigmoid(layers.linear(p.scope('gating_linear'), act))

def triangle_attention(p, pair_act: mx.array, pair_mask: mx.array, cfg: dict, *, subbatch_size: int | None = None, chunk_size: int | None = layers.ATTENTION_CHUNK) -> mx.array:
    #starting node is orientation per_row, ending node per_column (the config says which, there is no separate flag)
    per_column = cfg['orientation'] == 'per_column'
    if per_column:
        pair_act, pair_mask = pair_act.swapaxes(-2, -3), pair_mask.swapaxes(-1, -2)
    bias = (1e9 * (pair_mask - 1.0))[:, None, None, :]
    pair_act = layers.layer_norm(p.scope('query_norm'), pair_act)
    nonbatched_bias = mx.einsum('qkc,ch->hqk', pair_act, p.get('feat_2d_weights', pair_act.dtype))
    out = layers.attention(p.scope('attention'), pair_act, pair_act, bias, nonbatched_bias, cfg, subbatch_size=subbatch_size, chunk_size=chunk_size)
    return out.swapaxes(-2, -3) if per_column else out
