#The multimer template embedding (modules_multimer.TemplateEmbedding, SingleTemplateEmbedding, TemplateEmbeddingIteration and template_embedding_1d): the fixed target is a template, whose
#distances, residue types, backbone-frame directions and the query pair activations are embedded, passed through a small pair stack, averaged over templates and projected onto the pair
#activations; and its chi angles give extra MSA rows. The casts to the compute dtype are where the JAX code makes them.
import mlx.core as mx

from bindcraft.af.mlx_af2 import all_atom, embeddings, evoformer, geometry, layers, pair_ops

def template_embedding_1d(p, features: dict, num_channel: int, dtype) -> tuple[mx.array, mx.array]:
    #p is the 'evoformer' scope; returns the template rows [templates, residues, channels] and their mask
    positions, mask, aatype = features['template_all_atom_positions'], features['template_all_atom_mask'], features['template_aatype']
    chi = [all_atom.compute_chi_angles(positions[index], mask[index], aatype[index]) for index in range(aatype.shape[0])]
    chi_angles, chi_mask = mx.stack([angles for angles, _ in chi]), mx.stack([chi_mask for _, chi_mask in chi])
    template_features = mx.concatenate([embeddings.one_hot(aatype, 22), mx.sin(chi_angles) * chi_mask, mx.cos(chi_angles) * chi_mask, chi_mask], axis=-1)
    template_mask = chi_mask[:, :, 0]
    if dtype == mx.bfloat16:
        template_features, template_mask = template_features.astype(mx.bfloat16), template_mask.astype(mx.bfloat16)
    act = mx.maximum(layers.linear(p.scope('template_single_embedding'), template_features), 0)
    return layers.linear(p.scope('template_projection'), act), template_mask

def template_iteration(p, act: mx.array, pair_mask: mx.array, cfg: dict, dropout_key=None) -> mx.array:
    #TemplateEmbeddingIteration: the pair half of an Evoformer block, each update dropped out and added
    keys = iter(list(mx.random.split(dropout_key, 20))[1:]) if dropout_key is not None else iter(lambda: None, 1)
    for name in ('triangle_multiplication_outgoing', 'triangle_multiplication_incoming'):
        act = act + evoformer.dropout_update(pair_ops.triangle_multiplication(p.scope(name), act, pair_mask, cfg[name]), cfg[name], next(keys))
    for name in ('triangle_attention_starting_node', 'triangle_attention_ending_node'):
        act = act + evoformer.dropout_update(pair_ops.triangle_attention(p.scope(name), act, pair_mask, cfg[name]), cfg[name], next(keys))
    return act + evoformer.dropout_update(layers.transition(p.scope('pair_transition'), act, pair_mask, cfg['pair_transition']), cfg['pair_transition'], next(keys))

def single_template_embedding(p, query_embedding: mx.array, template: dict, padding_mask_2d: mx.array, multichain_mask_2d: mx.array, cfg: dict, dropout_key=None, remat: bool = True) -> mx.array:
    dtype = query_embedding.dtype
    positions, mask, aatype = template['template_all_atom_positions'], template['template_all_atom_mask'], template['template_aatype']
    beta, beta_mask = embeddings.pseudo_beta(aatype, positions, mask)
    beta_mask_2d = beta_mask[:, None] * beta_mask[None, :] * multichain_mask_2d.astype(mx.float32)
    dgram = (embeddings.dgram_from_positions(beta, **cfg['dgram_features']) * beta_mask_2d[..., None]).astype(dtype)
    beta_mask_2d = beta_mask_2d.astype(dtype)
    aatype_one_hot = embeddings.one_hot(aatype, 22).astype(dtype)
    to_concat = [(dgram, 1), (beta_mask_2d, 0), (aatype_one_hot[None], 1), (aatype_one_hot[:, None], 1)]
    #the direction between every pair of backbone frames, in each residue's own frame
    rigid, backbone_mask = all_atom.make_backbone_affine(geometry.Vec3.from_array(positions.astype(mx.float32)), mask, aatype)
    unit = rigid[:, None].inverse().apply_to_point(rigid.translation).normalized()
    unit = [unit.x, unit.y, unit.z]
    if dtype == mx.bfloat16:
        unit, backbone_mask = [component.astype(mx.bfloat16) for component in unit], backbone_mask.astype(mx.bfloat16)
    backbone_mask_2d = mx.sqrt(backbone_mask[:, None] * backbone_mask[None, :]) * multichain_mask_2d.astype(backbone_mask.dtype)
    to_concat += [(component * backbone_mask_2d, 0) for component in unit] + [(backbone_mask_2d, 0), (layers.layer_norm(p.scope('query_embedding_norm'), query_embedding), 1)]
    act = 0
    for index, (feature, input_dims) in enumerate(to_concat):
        act = act + layers.linear(p.scope(f'template_pair_embedding_{index}'), feature, num_input_dims=input_dims)
    stack_cfg = cfg['template_pair_stack']
    keys = list(mx.random.split(dropout_key, stack_cfg['num_block'])) if dropout_key is not None else [None] * stack_cfg['num_block']
    for index in range(stack_cfg['num_block']):
        def block(activations, index=index):
            return template_iteration(p.scope('template_embedding_iteration').layer(index), activations, padding_mask_2d, stack_cfg, keys[index])
        act = mx.checkpoint(block)(act) if remat else block(act)
    return layers.layer_norm(p.scope('output_layer_norm'), act)

def template_embedding(p, query_embedding: mx.array, template_batch: dict, padding_mask_2d: mx.array, multichain_mask_2d: mx.array, cfg: dict, *, dropout_key=None, remat: bool = True) -> mx.array:
    #p is the 'evoformer/template_embedding' scope; the templates are embedded one by one, averaged, rectified and projected to the query's channels
    count = template_batch['template_aatype'].shape[0]
    keys = list(mx.random.split(dropout_key, count)) if dropout_key is not None else [None] * count
    total = mx.zeros((query_embedding.shape[0], query_embedding.shape[1], cfg['num_channels']), dtype=query_embedding.dtype)
    for index in range(count):
        total = total + single_template_embedding(p.scope('single_template_embedding'), query_embedding, {name: value[index] for name, value in template_batch.items()}, padding_mask_2d, multichain_mask_2d, cfg, keys[index], remat)
    return layers.linear(p.scope('output_linear'), mx.maximum(total / count, 0))

def template_hook(p, features: dict, cfg: dict, dtype):
    #the template_embedding callable embeddings.embeddings_and_evoformer takes; p is the 'evoformer' scope and cfg the embeddings_and_evoformer config
    template_batch = {name: features[name] for name in ('template_aatype', 'template_all_atom_positions', 'template_all_atom_mask')}
    same_chain = features['asym_id'][:, None] == features['asym_id'][None, :]
    multichain_mask = mx.where(features['mask_template_interchain'], same_chain, mx.ones_like(same_chain))  #intra-chain pairs only, unless interchain templates are allowed

    def hook(query_pair, mask_2d, key):
        update = template_embedding(p.scope('template_embedding'), query_pair, template_batch, mask_2d, multichain_mask, cfg['template'], dropout_key=key)
        rows, rows_mask = template_embedding_1d(p, features, cfg['msa_channel'], dtype)
        return update, rows, rows_mask
    return hook
