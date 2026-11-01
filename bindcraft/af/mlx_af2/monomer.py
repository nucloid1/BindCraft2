#The AlphaFold monomer (pTM) models' trunk (modules.EmbeddingsAndEvoformer, TemplateEmbedding, SingleTemplateEmbedding and TemplatePairStack of the monomer family, as the repo vendors and extends them): the
#embeddings with the previous pass, the template pair embedding and its torsion-angle rows, the extra-MSA stack, the main stack and the single activations. The Evoformer ops, the layers and the stacks are the
#shared ones of the multimer port; what differs is here: the input padding, the relative position encoding without chain terms, the template modules, and the rows the template angles add to the MSA.
#dtype is the compute dtype; the casts are where the JAX code makes them.
import mlx.core as mx

from bindcraft.af.alphafold.common import residue_constants
from bindcraft.af.mlx_af2 import all_atom, embeddings, evoformer, heads, layers, monomer_structure, pair_ops
from bindcraft.af.mlx_af2.model_multimer import MultimerModel

def uses_templates(cfg: dict) -> bool:
    return bool(cfg['embeddings_and_evoformer']['template']['enabled'])

def check_supported(cfg: dict) -> None:
    #what the vendored JAX code cannot run itself (the unit-vector branch refers to names that do not exist) or the repo never switches on is refused by name rather than computed differently
    embedding = cfg['embeddings_and_evoformer']
    if embedding['template']['use_template_unit_vector']:
        raise NotImplementedError('use_template_unit_vector: the monomer template unit vectors are not ported (the JAX branch is not runnable either)')
    if embedding['backprop_dgram'] or embedding['template']['backprop_dgram']:
        raise NotImplementedError('backprop_dgram: the soft distogram of the monomer models is not ported')
    if cfg['global_config'].get('use_dgram'):
        raise NotImplementedError('use_dgram: the repo builds the monomer models with predicted positions as the recycled input')

def embed_inputs(p, features: dict, prev: dict, cfg: dict, dtype) -> tuple[mx.array, mx.array, mx.array]:
    #the MSA and pair activations and the 2D mask before the templates and stacks (EmbeddingsAndEvoformer up to the template embedding); p is the 'evoformer' scope
    msa_feat = features['msa_feat'].astype(dtype)
    target_feat = mx.pad(features['target_feat'].astype(dtype), [(0, 0), (1, 1)])
    msa = layers.linear(p.scope('preprocess_1d'), target_feat)[None] + layers.linear(p.scope('preprocess_msa'), msa_feat)
    pair = layers.linear(p.scope('left_single'), target_feat)[:, None] + layers.linear(p.scope('right_single'), target_feat)[None]
    mask_2d = (features['seq_mask'][:, None] * features['seq_mask'][None, :]).astype(dtype)
    pair = pair + layers.linear(p.scope('prev_pos_linear'), embeddings.recycle_distogram(features['aatype'], prev['prev_pos'], cfg['prev_pos'], dtype))
    if cfg['recycle_features']:
        first_row = layers.layer_norm(p.scope('prev_msa_first_row_norm'), prev['prev_msa_first_row']).astype(dtype)
        msa = mx.concatenate([msa[:1] + first_row[None], msa[1:]], axis=0)
        pair = pair + layers.layer_norm(p.scope('prev_pair_norm'), prev['prev_pair']).astype(dtype)
    if cfg['max_relative_feature']:
        reach = cfg['max_relative_feature']
        offset = features['offset'] if 'offset' in features else features['residue_index'][:, None] - features['residue_index'][None, :]
        rel_pos = embeddings.one_hot(mx.clip(offset + reach, 0, 2 * reach), 2 * reach + 1).astype(dtype)
        pair = pair + layers.linear(p.scope('pair_activiations'), rel_pos)  #(sic: the vendored scope name)
    return msa, pair, mask_2d

def template_pair_stack(p, act: mx.array, pair_mask: mx.array, cfg: dict, dropout_key=None, remat: bool = True) -> mx.array:
    #TemplatePairStack: the pair half of an Evoformer block, in the monomer order (the attentions first), each update dropped out and added; p is the single template embedding's scope
    count = cfg['num_block']
    if not count:
        return act
    stack = p.scope('template_pair_stack/__layer_stack_no_state')
    keys = list(mx.random.split(dropout_key, count)) if dropout_key is not None else [None] * count
    for index in range(count):
        def block(activations, index=index):
            layer = stack.layer(index)
            sub = iter(list(mx.random.split(keys[index], 6))[1:]) if keys[index] is not None else iter(lambda: None, 1)
            for name in ('triangle_attention_starting_node', 'triangle_attention_ending_node'):
                activations = activations + evoformer.dropout_update(pair_ops.triangle_attention(layer.scope(name), activations, pair_mask, cfg[name]), cfg[name], next(sub))
            for name in ('triangle_multiplication_outgoing', 'triangle_multiplication_incoming'):
                activations = activations + evoformer.dropout_update(pair_ops.triangle_multiplication(layer.scope(name), activations, pair_mask, cfg[name]), cfg[name], next(sub))
            return activations + evoformer.dropout_update(layers.transition(layer.scope('pair_transition'), activations, pair_mask, cfg['pair_transition']), cfg['pair_transition'], next(sub))
        act = mx.checkpoint(block)(act) if remat else block(act)
    return act

def single_template_embedding(p, query_embedding: mx.array, template: dict, mask_2d: mx.array, multichain_mask_2d: mx.array, cfg: dict, dropout_key=None, remat: bool = True) -> mx.array:
    #SingleTemplateEmbedding for one template (the template axis already stripped); p is 'evoformer/template_embedding/single_template_embedding'. The unit vectors are zero in the repo's configuration.
    dtype = query_embedding.dtype
    num_res = template['template_aatype'].shape[0]
    num_channels = cfg['template_pair_stack']['triangle_attention_ending_node']['value_dim']
    multichain = multichain_mask_2d.astype(mx.float32)
    beta_mask = template['template_pseudo_beta_mask']
    beta_mask_2d = (beta_mask[:, None] * beta_mask[None, :] * multichain).astype(dtype)
    dgram = (embeddings.dgram_from_positions(template['template_pseudo_beta'], **cfg['dgram_features']) * beta_mask_2d[..., None]).astype(dtype)
    aatype = embeddings.one_hot(template['template_aatype'], 22).astype(dtype)
    to_concat = [dgram, beta_mask_2d[:, :, None], mx.broadcast_to(aatype[None], (num_res, num_res, 22)), mx.broadcast_to(aatype[:, None], (num_res, num_res, 22))]
    n, ca, c = (residue_constants.atom_order[name] for name in ('N', 'CA', 'C'))
    atom_mask = template['template_all_atom_mask']
    backbone_mask = atom_mask[..., n] * atom_mask[..., ca] * atom_mask[..., c]
    backbone_mask_2d = (backbone_mask[:, None] * backbone_mask[None, :] * multichain).astype(dtype)
    to_concat += [mx.zeros((num_res, num_res, 1), dtype=dtype)] * 3 + [backbone_mask_2d[..., None]]
    act = mx.concatenate(to_concat, axis=-1) * backbone_mask_2d[..., None]
    act = layers.linear(p.scope('embedding2d'), act)
    act = template_pair_stack(p, act, mask_2d, cfg['template_pair_stack'], dropout_key, remat)
    return layers.layer_norm(p.scope('output_layer_norm'), act)

def template_embedding(p, query_embedding: mx.array, template_batch: dict, mask_2d: mx.array, multichain_mask_2d: mx.array, cfg: dict, *, dropout_key=None, remat: bool = True) -> mx.array:
    #TemplateEmbedding: every template embedded on its own, then the query attends over the templates at each residue pair; p is the 'evoformer/template_embedding' scope
    count = template_batch['template_aatype'].shape[0]
    num_res, query_channels = query_embedding.shape[0], query_embedding.shape[-1]
    dtype = query_embedding.dtype
    template_mask = template_batch['template_mask'].astype(dtype)
    keys = list(mx.random.split(dropout_key, count)) if dropout_key is not None else [None] * count
    embedded = mx.stack([single_template_embedding(p.scope('single_template_embedding'), query_embedding, {name: value[index] for name, value in template_batch.items()}, mask_2d, multichain_mask_2d, cfg,
                                                   keys[index], remat) for index in range(count)])
    flat_query = query_embedding.reshape(num_res * num_res, 1, query_channels)
    flat_templates = mx.transpose(embedded, (1, 2, 0, 3)).reshape(num_res * num_res, count, embedded.shape[-1])
    bias = 1e9 * (template_mask[None, None, None, :] - 1.0)
    embedding = layers.attention(p.scope('attention'), flat_query, flat_templates, bias, None, cfg['attention']).reshape(num_res, num_res, query_channels)
    return embedding * (mx.sum(template_mask) > 0).astype(embedding.dtype)  #no templates, no update

def template_rows(p, features: dict, dtype, zero_init: bool) -> tuple[mx.array, mx.array]:
    #the rows the template residue types and torsion angles add to the MSA, and their mask (the psi angle's, which depends on the backbone atoms of one residue); p is the 'evoformer' scope
    aatype = features['template_aatype']
    angles = all_atom.atom37_to_torsion_angles(aatype, features['template_all_atom_positions'], features['template_all_atom_mask'], placeholder_for_undefined=not zero_init)
    templates, residues = aatype.shape
    template_features = mx.concatenate([embeddings.one_hot(aatype, 22), angles['torsion_angles_sin_cos'].reshape(templates, residues, 14), angles['alt_torsion_angles_sin_cos'].reshape(templates, residues, 14),
                                        angles['torsion_angles_mask']], axis=-1).astype(dtype)
    act = mx.maximum(layers.linear(p.scope('template_single_embedding'), template_features), 0)
    return layers.linear(p.scope('template_projection'), act), angles['torsion_angles_mask'][:, :, 2].astype(dtype)

def embeddings_and_evoformer(p, features: dict, prev: dict, cfg: dict, *, dtype, zero_init: bool = True, num_block: int | None = None, extra_num_block: int | None = None, dropout_key=None,
                             remat: bool = True, subbatch_size: int | None = None) -> dict[str, mx.array]:
    #EmbeddingsAndEvoformer.__call__ of the monomer family; cfg is the embeddings_and_evoformer config; p the 'evoformer' scope. Returns single, pair, msa (the template rows cropped) and msa_first_row.
    keys = list(mx.random.split(dropout_key, 3)) if dropout_key is not None else [None] * 3
    msa, pair, mask_2d = embed_inputs(p, features, prev, cfg, dtype)
    template = cfg['template']
    if template['enabled']:
        template_batch = {name: value for name, value in features.items() if name.startswith('template_')}
        same_chain = features['asym_id'][:, None] == features['asym_id'][None, :]
        multichain = mx.where(features['mask_template_interchain'], same_chain, mx.ones_like(same_chain))
        pair = pair + template_embedding(p.scope('template_embedding'), pair, template_batch, mask_2d, multichain, template, dropout_key=keys[0], remat=remat)
    if cfg['use_extra_msa']:
        extra_feature = mx.concatenate([embeddings.one_hot(features['extra_msa'], 23), features['extra_has_deletion'][..., None], features['extra_deletion_value'][..., None]], axis=-1)
        extra_act = layers.linear(p.scope('extra_msa_activations'), extra_feature).astype(dtype)
        pair = embeddings.extra_msa_stack(p, extra_act, pair, features['extra_msa_mask'].astype(dtype), mask_2d, cfg, num_block=extra_num_block, dropout_key=keys[1], remat=remat, subbatch_size=subbatch_size)
    msa_mask = features['msa_mask'].astype(dtype)
    if template['enabled'] and template['embed_torsion_angles']:
        rows, rows_mask = template_rows(p, features, dtype, zero_init)
        msa, msa_mask = mx.concatenate([msa, rows], axis=0), mx.concatenate([msa_mask, rows_mask], axis=0)
    if cfg['use_msa']:
        msa, pair = evoformer.evoformer_stack(p.scope('evoformer_iteration'), msa, pair, msa_mask, mask_2d, cfg['evoformer'], num_block=cfg['evoformer_num_block'] if num_block is None else num_block,
                                              dropout_key=keys[2], remat=remat, subbatch_size=subbatch_size)
    outputs = {'single': layers.linear(p.scope('single_activations'), msa[0]), 'pair': pair, 'msa': msa[:features['msa_feat'].shape[0]], 'msa_first_row': msa[0]}
    return {name: value.astype(mx.float32) if value.dtype == mx.bfloat16 else value for name, value in outputs.items()}

class MonomerModel(MultimerModel):
    #the monomer family, forward only (validation): the recycling and forward logic of the multimer model around this family's trunk, structure module and the shared heads.
    #cfg is the effective model config (config.model_config(name).model as a dict, with the parent's overrides applied)
    def __init__(self, model_name: str, params, cfg: dict, dtype=None):
        check_supported(cfg)
        super().__init__(model_name, params, cfg, dtype)

    def trunk(self, features: dict, prev: dict, dropout_key=None, *, subbatch_size: int | None = None, remat: bool = True) -> dict[str, mx.array]:
        return embeddings_and_evoformer(self.params.scope('evoformer'), features, prev, self.cfg['embeddings_and_evoformer'], dtype=self.dtype, zero_init=self.cfg['global_config']['zero_init'],
                                        dropout_key=dropout_key, remat=remat, subbatch_size=subbatch_size)

    def single_pass(self, features: dict, prev: dict, dropout_key=None, *, subbatch_size: int | None = None, remat: bool = True, tapes: dict | None = None) -> tuple[dict, dict]:
        #one AlphaFoldIteration: returns the nested outputs and the state the next pass starts from
        if tapes is not None:
            raise NotImplementedError(f'{self.model_name} is a validation model of the MLX backend (forward only): it keeps no tapes for a gradient')
        cfg, p = self.cfg, self.params
        keys = list(mx.random.split(dropout_key, 2)) if dropout_key is not None else [None, None]
        representations = self.trunk(features, prev, keys[0], subbatch_size=subbatch_size, remat=remat)
        sm = monomer_structure.structure_module(p.scope('structure_module'), representations, features, cfg['heads']['structure_module'], dropout_key=keys[1])
        representations['structure_module'] = sm['act']
        h = cfg['heads']
        outputs = {'structure_module': {'final_atom_positions': sm['final_atom_positions'], 'final_atom_mask': sm['final_atom_mask']},
                   'predicted_lddt': heads.predicted_lddt(p.scope('predicted_lddt_head'), representations, h['predicted_lddt']),
                   'predicted_aligned_error': heads.predicted_aligned_error(p.scope('predicted_aligned_error_head'), representations, h['predicted_aligned_error']),
                   'distogram': heads.distogram(p.scope('distogram_head'), representations, h['distogram']),
                   'experimentally_resolved': heads.experimentally_resolved(p.scope('experimentally_resolved_head'), representations, h['experimentally_resolved'])}
        return outputs, {'prev_pos': sm['final_atom_positions'], 'prev_msa_first_row': representations['msa_first_row'], 'prev_pair': representations['pair']}

    def vjp(self, *arguments, **keywords):
        raise NotImplementedError(f'{self.model_name}: the MLX backend has no gradient for the monomer models (they only validate designs); design with a multimer model')
