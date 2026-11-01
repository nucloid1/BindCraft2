#The multimer model's input embeddings (modules_multimer.EmbeddingsAndEvoformer up to the stacks): the target and MSA features, the recycle embedding of the previous pass, the relative position and
#chain encoding (with the cyclic offsets a design can carry), the extra-MSA features, and the assembly of the extra-MSA stack, the main stack and the single activations.
#dtype is the compute dtype (bfloat16 as shipped, or float32); the casts are where the JAX code makes them.
import math
from dataclasses import dataclass

import mlx.core as mx

from bindcraft.af.alphafold.common import residue_constants
from bindcraft.af.mlx_af2 import evoformer, layers

@dataclass
class Embedded:
    msa_act: mx.array
    pair_act: mx.array
    mask_2d: mx.array
    extra_msa_act: mx.array
    extra_msa_mask: mx.array

def one_hot(index: mx.array, size: int) -> mx.array:
    return (index[..., None] == mx.arange(size)).astype(mx.float32)

def relative_features(features: dict, cfg: dict, dtype) -> mx.array:
    #the encoding before its linear layer: the clipped offset (an extra bin for a different chain), whether two residues share an entity, and the clipped relative chain index
    asym_id = features['asym_id']
    same_chain = asym_id[:, None] == asym_id[None, :]
    offset = features['offset'] if 'offset' in features else features['residue_index'][:, None] - features['residue_index'][None, :]
    reach = cfg['max_relative_idx']
    clipped = mx.clip(offset + reach, 0, 2 * reach)
    if not cfg['use_chain_relative']:
        return one_hot(clipped, 2 * reach + 1).astype(dtype)
    entity_id, sym_id = features['entity_id'], features['sym_id']
    same_entity = entity_id[:, None] == entity_id[None, :]
    chain_reach = cfg['max_relative_chain']
    clipped_chain = mx.clip(sym_id[:, None] - sym_id[None, :] + chain_reach, 0, 2 * chain_reach)
    pieces = [one_hot(mx.where(same_chain, clipped, 2 * reach + 1), 2 * reach + 2), same_entity.astype(mx.float32)[..., None], one_hot(mx.where(same_entity, clipped_chain, 2 * chain_reach + 1), 2 * chain_reach + 2)]
    return mx.concatenate(pieces, axis=-1).astype(dtype)

def relative_encoding(p, features: dict, cfg: dict, dtype) -> mx.array:
    return layers.linear(p.scope('~_relative_encoding/position_activations'), relative_features(features, cfg, dtype))  #haiku scopes a non-transparent method under its own name

def pseudo_beta(aatype: mx.array, positions: mx.array, mask: mx.array | None = None):
    #modules.pseudo_beta_fn: the beta carbon (the alpha carbon for glycine), and its mask when a mask is given
    ca, cb = residue_constants.atom_order['CA'], residue_constants.atom_order['CB']
    is_glycine = aatype == residue_constants.restype_order['G']
    beta = mx.where(is_glycine[..., None], positions[..., ca, :], positions[..., cb, :])
    return beta if mask is None else (beta, mx.where(is_glycine, mask[..., ca], mask[..., cb]).astype(mx.float32))

def dgram_from_positions(positions: mx.array, num_bins: int, min_bin: float, max_bin: float) -> mx.array:
    #modules.dgram_from_positions: one-hot squared distances between every pair, the last bin catching everything beyond max_bin
    lower = mx.square(mx.linspace(min_bin, max_bin, num_bins))
    upper = mx.concatenate([lower[1:], mx.array([1e8], dtype=mx.float32)])
    squared = mx.sum(mx.square(positions[:, None] - positions[None, :]), axis=-1, keepdims=True)
    return (squared > lower).astype(mx.float32) * (squared < upper).astype(mx.float32)

def recycle_distogram(aatype: mx.array, prev_pos: mx.array, prev_pos_cfg: dict, dtype) -> mx.array:
    return dgram_from_positions(pseudo_beta(aatype, prev_pos), prev_pos_cfg['num_bins'], prev_pos_cfg['min_bin'], prev_pos_cfg['max_bin']).astype(dtype)

def extra_msa_features(features: dict, num_extra_msa: int) -> tuple[mx.array, mx.array]:
    extra_msa, deletion = features['extra_msa'][:num_extra_msa], features['extra_deletion_value'][:num_extra_msa]
    has_deletion = mx.clip(deletion, 0.0, 1.0)[..., None]
    deletion_value = (mx.arctan(deletion / 3.0) * (2.0 / math.pi))[..., None]
    return mx.concatenate([one_hot(extra_msa, 23), has_deletion, deletion_value], axis=-1), features['extra_msa_mask'][:num_extra_msa]

def embed_inputs(p, features: dict, prev: dict, cfg: dict, dtype) -> Embedded:
    msa_feat = features['msa_feat'].astype(dtype)
    target_feat = mx.pad(features['target_feat'].astype(dtype), [(0, 0), (0, 1)])
    msa = layers.linear(p.scope('preprocess_1d'), target_feat)[None] + layers.linear(p.scope('preprocess_msa'), msa_feat)
    pair = layers.linear(p.scope('left_single'), target_feat)[:, None] + layers.linear(p.scope('right_single'), target_feat)[None]
    mask_2d = (features['seq_mask'][:, None] * features['seq_mask'][None, :]).astype(dtype)
    if cfg['recycle_pos']:
        pair = pair + layers.linear(p.scope('prev_pos_linear'), recycle_distogram(features['aatype'], prev['prev_pos'], cfg['prev_pos'], dtype))
    if cfg['recycle_features']:
        first_row = layers.layer_norm(p.scope('prev_msa_first_row_norm'), prev['prev_msa_first_row']).astype(dtype)
        msa = mx.concatenate([msa[:1] + first_row[None], msa[1:]], axis=0)
        pair = pair + layers.layer_norm(p.scope('prev_pair_norm'), prev['prev_pair']).astype(dtype)
    if cfg['max_relative_idx']:
        pair = pair + relative_encoding(p, features, cfg, dtype)
    extra_feat, extra_mask = extra_msa_features(features, cfg['num_extra_msa'])
    return Embedded(msa, pair, mask_2d, layers.linear(p.scope('extra_msa_activations'), extra_feat).astype(dtype), extra_mask.astype(dtype))

def extra_msa_stack(p, extra_msa_act: mx.array, pair_act: mx.array, extra_msa_mask: mx.array, pair_mask: mx.array, cfg: dict, *, num_block: int | None = None, dropout_key=None,
                    remat: bool = True, subbatch_size: int | None = None, tape=None) -> mx.array:
    #p is the 'evoformer' scope; returns the updated pair activations (the extra rows are not carried further)
    _, pair = evoformer.evoformer_stack(p.scope('extra_msa_stack'), extra_msa_act, pair_act, extra_msa_mask, pair_mask, cfg['evoformer'], num_block=cfg['extra_msa_stack_num_block'] if num_block is None else num_block,
                                        dropout_key=dropout_key, remat=remat, subbatch_size=subbatch_size, is_extra=True, tape=tape)
    return pair

@dataclass
class Front:
    #the final pass up to the stacks: msa and pair are the only activations that depend on the sequence features (msa_feat, target_feat); the rest are constants of the pass
    msa: mx.array
    pair: mx.array
    msa_mask: mx.array
    mask_2d: mx.array
    extra_msa_act: mx.array
    extra_msa_mask: mx.array
    rows: mx.array | None  #the template rows the main stack's MSA gets, and their mask
    rows_mask: mx.array | None
    extra_key: mx.array | None  #the dropout keys of the extra-MSA stack and the main stack
    main_key: mx.array | None

def front(p, features: dict, prev: dict, cfg: dict, *, dtype, dropout_key=None, template_embedding=None) -> Front:
    #the embeddings and the template embedding. template_embedding, when given, is called as template_embedding(query_pair_act, mask_2d, key) -> (update to the pair activations, extra MSA rows, their mask)
    keys = list(mx.random.split(dropout_key, 3)) if dropout_key is not None else [None] * 3
    embedded = embed_inputs(p, features, prev, cfg, dtype)
    pair, rows, rows_mask = embedded.pair_act, None, None
    if template_embedding is not None:
        update, rows, rows_mask = template_embedding(pair, embedded.mask_2d, keys[0])
        pair = pair + update
    return Front(embedded.msa_act, pair, features['msa_mask'].astype(dtype), embedded.mask_2d, embedded.extra_msa_act, embedded.extra_msa_mask, rows, rows_mask, keys[1], keys[2])

def join(f: Front) -> tuple[mx.array, mx.array]:
    #the main stack's MSA and its mask: the MSA rows, then the template rows
    if f.rows is None:
        return f.msa, f.msa_mask
    return mx.concatenate([f.msa, f.rows], axis=0), mx.concatenate([f.msa_mask, f.rows_mask], axis=0)

def finish(p, msa: mx.array, pair: mx.array, sequences: int) -> dict[str, mx.array]:
    #the representations after the main stack: the single activations from the first MSA row, the MSA without the template rows, float32
    single = layers.linear(p.scope('single_activations'), msa[0])
    outputs = {'single': single, 'pair': pair, 'msa': msa[:sequences], 'msa_first_row': msa[0]}
    return {name: value.astype(mx.float32) if value.dtype == mx.bfloat16 else value for name, value in outputs.items()}

def embeddings_and_evoformer(p, features: dict, prev: dict, cfg: dict, *, dtype, num_block: int | None = None, extra_num_block: int | None = None, dropout_key=None, remat: bool = True,
                             subbatch_size: int | None = None, template_embedding=None, tapes: dict | None = None) -> dict[str, mx.array]:
    #EmbeddingsAndEvoformer.__call__: front, the extra-MSA stack, the main stack, finish.
    #tapes, when given ({'extra_msa_stack': Tape, 'evoformer_iteration': Tape}), are recorded by a forward pass (remat=False) and consumed by a differentiated one (remat=True): evoformer.evoformer_stack
    tapes = tapes or {}
    f = front(p, features, prev, cfg, dtype=dtype, dropout_key=dropout_key, template_embedding=template_embedding)
    pair = extra_msa_stack(p, f.extra_msa_act, f.pair, f.extra_msa_mask, f.mask_2d, cfg, num_block=extra_num_block, dropout_key=f.extra_key, remat=remat, subbatch_size=subbatch_size, tape=tapes.get('extra_msa_stack'))
    msa, msa_mask = join(f)
    msa, pair = evoformer.evoformer_stack(p.scope('evoformer_iteration'), msa, pair, msa_mask, f.mask_2d, cfg['evoformer'], num_block=cfg['evoformer_num_block'] if num_block is None else num_block,
                                          dropout_key=f.main_key, remat=remat, subbatch_size=subbatch_size, tape=tapes.get('evoformer_iteration'))
    return finish(p, msa, pair, f.msa.shape[0])
