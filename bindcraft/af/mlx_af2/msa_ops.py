#The MSA stack's operations (modules.MSARowAttentionWithPairBias, MSAColumnAttention, MSAColumnGlobalAttention, OuterProductMean) as functions of a Params scope.
#Each returns the update the JAX module returns, not the residual sum.
import mlx.core as mx

from bindcraft.af.mlx_af2 import layers

def msa_row_attention_with_pair_bias(p, msa_act: mx.array, msa_mask: mx.array, pair_act: mx.array, cfg: dict, *, subbatch_size: int | None = None, chunk_size: int | None = layers.ATTENTION_CHUNK) -> mx.array:
    bias = (1e9 * (msa_mask - 1.0))[:, None, None, :]
    msa_act = layers.layer_norm(p.scope('query_norm'), msa_act)
    pair_act = layers.layer_norm(p.scope('feat_2d_norm'), pair_act)
    nonbatched_bias = mx.einsum('qkc,ch->hqk', pair_act, p.get('feat_2d_weights', msa_act.dtype))
    return layers.attention(p.scope('attention'), msa_act, msa_act, bias, nonbatched_bias, cfg, subbatch_size=subbatch_size, chunk_size=chunk_size)

def msa_column_attention(p, msa_act: mx.array, msa_mask: mx.array, cfg: dict, *, subbatch_size: int | None = None, chunk_size: int | None = layers.ATTENTION_CHUNK) -> mx.array:
    msa_act, msa_mask = msa_act.swapaxes(-2, -3), msa_mask.swapaxes(-1, -2)
    bias = (1e9 * (msa_mask - 1.0))[:, None, None, :]
    msa_act = layers.layer_norm(p.scope('query_norm'), msa_act)
    return layers.attention(p.scope('attention'), msa_act, msa_act, bias, None, cfg, subbatch_size=subbatch_size, chunk_size=chunk_size).swapaxes(-2, -3)

def msa_column_global_attention(p, msa_act: mx.array, msa_mask: mx.array, cfg: dict, *, subbatch_size: int | None = None) -> mx.array:
    #the extra-MSA stack's attention: built from the column-attention config; the JAX module also passes a bias that GlobalAttention never reads
    msa_act, msa_mask = msa_act.swapaxes(-2, -3), msa_mask.swapaxes(-1, -2)
    msa_act = layers.layer_norm(p.scope('query_norm'), msa_act)
    out = layers.sharded(lambda act, mask: layers.global_attention(p.scope('attention'), act, act, mask, cfg), subbatch_size, (msa_act, msa_mask[..., None]))
    return out.swapaxes(-2, -3)

def outer_product_mean(p, act: mx.array, mask: mx.array, cfg: dict, *, chunk_size: int | None = ...) -> mx.array:
    #out[b, d, f] = sum over rows a and channels c, e of left[a, b, c] right[a, d, e] output_w[c, e, f], by residue chunks of b; two matmuls instead of the four-index einsum
    chunk_size = cfg['chunk_size'] if chunk_size is ... else chunk_size
    mask3 = mask[..., None]
    act = layers.layer_norm(p.scope('layer_norm_input'), act)
    left = mask3 * layers.linear(p.scope('left_projection'), act)
    right = mask3 * layers.linear(p.scope('right_projection'), act)
    output_w, output_b = p.get('output_w', act.dtype), p.get('output_b', act.dtype)
    rows, residues, left_channels = left.shape
    right_channels, flat_right = right.shape[-1], right.reshape(rows, residues * right.shape[-1])
    flat_w = output_w.reshape(left_channels * right_channels, output_w.shape[-1])

    def chunk(left_chunk):
        size = left_chunk.shape[1]
        pairs = (left_chunk.reshape(rows, size * left_channels).T @ flat_right).reshape(size, left_channels, residues, right_channels)
        return (pairs.transpose(0, 2, 1, 3).reshape(size * residues, left_channels * right_channels) @ flat_w).reshape(size, residues, -1) + output_b

    starts = range(0, residues, chunk_size) if chunk_size else (0,)
    out = mx.concatenate([chunk(left[:, start:start + chunk_size] if chunk_size else left) for start in starts], axis=0)
    return out / (1e-3 + (mask.T @ mask)[..., None])
