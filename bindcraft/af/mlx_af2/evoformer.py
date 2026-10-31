#One Evoformer block and the stacks of them (modules.EvoformerIteration scanned by layer_stack, with hk.remat per block): the order of the operations, the dropout of each update shared along the
#axis its orientation names, and the residual adds. The main stack and the extra-MSA stack differ only in the column operation (attention against global attention) and in their parameters.
import mlx.core as mx

from bindcraft.af.mlx_af2 import layers, msa_ops, pair_ops

def dropout_update(update: mx.array, module_cfg: dict, key):
    #modules.dropout_wrapper: the update is dropped out (the mask shared along axis 0 for per_row modules, axis 1 for per_column, when shared_dropout) and rescaled by the keep rate; no key means no dropout
    rate = module_cfg['dropout_rate']
    if key is None or not rate:
        return update
    shape = list(update.shape)
    if module_cfg['shared_dropout']:
        shape[0 if module_cfg['orientation'] == 'per_row' else 1] = 1
    keep = mx.random.bernoulli(1.0 - rate, shape, key=key)
    return keep * update / (1.0 - rate)

def evoformer_iteration(p, msa_act: mx.array, pair_act: mx.array, msa_mask: mx.array, pair_mask: mx.array, cfg: dict, *, dropout_key=None, is_extra: bool = False, subbatch_size: int | None = None) -> tuple[mx.array, mx.array]:
    keys = iter(list(mx.random.split(dropout_key, 10))[1:]) if dropout_key is not None else iter(lambda: None, 1)  #the nine sub keys the JAX block draws, in the order it draws them

    def outer_product(pair, msa):
        return pair + dropout_update(msa_ops.outer_product_mean(p.scope('outer_product_mean'), msa, msa_mask, cfg['outer_product_mean']), cfg['outer_product_mean'], next(keys))

    if cfg['outer_product_mean']['first']:
        pair_act = outer_product(pair_act, msa_act)
    row = msa_ops.msa_row_attention_with_pair_bias(p.scope('msa_row_attention_with_pair_bias'), msa_act, msa_mask, pair_act, cfg['msa_row_attention_with_pair_bias'], subbatch_size=subbatch_size)
    msa_act = msa_act + dropout_update(row, cfg['msa_row_attention_with_pair_bias'], next(keys))
    if is_extra:
        column = msa_ops.msa_column_global_attention(p.scope('msa_column_global_attention'), msa_act, msa_mask, cfg['msa_column_attention'], subbatch_size=subbatch_size)
    else:
        column = msa_ops.msa_column_attention(p.scope('msa_column_attention'), msa_act, msa_mask, cfg['msa_column_attention'], subbatch_size=subbatch_size)
    msa_act = msa_act + dropout_update(column, cfg['msa_column_attention'], next(keys))
    msa_act = msa_act + dropout_update(layers.transition(p.scope('msa_transition'), msa_act, msa_mask, cfg['msa_transition'], subbatch_size=subbatch_size), cfg['msa_transition'], next(keys))
    if not cfg['outer_product_mean']['first']:
        pair_act = outer_product(pair_act, msa_act)
    for name in ('triangle_multiplication_outgoing', 'triangle_multiplication_incoming'):
        pair_act = pair_act + dropout_update(pair_ops.triangle_multiplication(p.scope(name), pair_act, pair_mask, cfg[name]), cfg[name], next(keys))
    for name in ('triangle_attention_starting_node', 'triangle_attention_ending_node'):
        pair_act = pair_act + dropout_update(pair_ops.triangle_attention(p.scope(name), pair_act, pair_mask, cfg[name], subbatch_size=subbatch_size), cfg[name], next(keys))
    pair_act = pair_act + dropout_update(layers.transition(p.scope('pair_transition'), pair_act, pair_mask, cfg['pair_transition'], subbatch_size=subbatch_size), cfg['pair_transition'], next(keys))
    return msa_act, pair_act

def evoformer_stack(p, msa_act: mx.array, pair_act: mx.array, msa_mask: mx.array, pair_mask: mx.array, cfg: dict, *, num_block: int, dropout_key=None, remat: bool = True,
                    subbatch_size: int | None = None, is_extra: bool = False) -> tuple[mx.array, mx.array]:
    #p is the scope that holds the stacked tensors (evoformer/evoformer_iteration or evoformer/extra_msa_stack); remat recomputes each block in the backward pass instead of keeping its activations
    keys = list(mx.random.split(dropout_key, num_block)) if dropout_key is not None else [None] * num_block
    for index in range(num_block):
        def block(msa, pair, index=index):
            return evoformer_iteration(p.layer(index), msa, pair, msa_mask, pair_mask, cfg, dropout_key=keys[index], is_extra=is_extra, subbatch_size=subbatch_size)
        msa_act, pair_act = mx.checkpoint(block)(msa_act, pair_act) if remat else block(msa_act, pair_act)
    return msa_act, pair_act
