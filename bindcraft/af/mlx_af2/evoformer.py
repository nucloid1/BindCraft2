#One Evoformer block and the stacks of them (modules.EvoformerIteration scanned by layer_stack, with hk.remat per block): the order of the operations, the dropout of each update shared along the
#axis its orientation names, and the residual adds. The main stack and the extra-MSA stack differ only in the column operation (attention against global attention) and in their parameters.
import mlx.core as mx

from bindcraft.af.mlx_af2 import compiled, layers, msa_ops, pair_ops

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

class Tape:
    #what a forward call keeps of one block stack for the vjp that follows it: the input of every block and the stack's outputs. A recorded tape lets the vjp skip the stack's forward pass.
    def __init__(self):
        self.inputs: list[tuple[mx.array, mx.array]] = []
        self.outputs: tuple[mx.array, mx.array] | None = None

    @property
    def nbytes(self) -> int:
        kept = [array for pair in self.inputs for array in pair] + list(self.outputs or ())
        return sum(array.nbytes for array in kept)

    def check(self, num_block: int, msa_act: mx.array, pair_act: mx.array) -> None:
        #a tape belongs to one call: refuse one that was not recorded, or recorded for another depth, shape or dtype (the caller decides whether that is a miss or an error)
        if self.outputs is None or len(self.inputs) != num_block:
            raise ValueError(f'the tape is not a recording of a {num_block}-block stack ({"empty" if self.outputs is None else f"{len(self.inputs)} blocks"})')
        msa_in, pair_in = self.inputs[0]
        if msa_in.shape != msa_act.shape or pair_in.shape != pair_act.shape or msa_in.dtype != msa_act.dtype or pair_in.dtype != pair_act.dtype:
            raise ValueError(f'the tape was recorded for inputs {tuple(msa_in.shape)}/{tuple(pair_in.shape)} {msa_in.dtype}, not {tuple(msa_act.shape)}/{tuple(pair_act.shape)} {msa_act.dtype}')

def evoformer_stack(p, msa_act: mx.array, pair_act: mx.array, msa_mask: mx.array, pair_mask: mx.array, cfg: dict, *, num_block: int, dropout_key=None, remat: bool = True,
                    subbatch_size: int | None = None, is_extra: bool = False, tape: Tape | None = None) -> tuple[mx.array, mx.array]:
    #p is the scope that holds the stacked tensors (evoformer/evoformer_iteration or evoformer/extra_msa_stack).
    #remat=False: a plain loop over the blocks, which records its block inputs and outputs into tape when one is given.
    #remat=True: the stack is one custom function whose vjp is an explicit reverse loop: each block is recomputed from its recorded input and differentiated, so no block keeps its activations. The recorded inputs come from
    #the tape when one is given (the forward pass is then skipped altogether), else from this call's own forward. Gradients flow to msa_act and pair_act ONLY: the masks, the key and the parameters are constants.
    #On the GPU the blocks run compiled (compiled.py): one compiled forward and one compiled vjp closure per stack, the layer's parameters passed in as arrays; the same arithmetic, fused.
    keys = list(mx.random.split(dropout_key, num_block)) if dropout_key is not None else [None] * num_block
    functions = compiled.functions_for(p, cfg, msa_act, pair_act, msa_mask, pair_mask, is_extra=is_extra, with_key=dropout_key is not None, subbatch_size=subbatch_size) if compiled.enabled() else None

    def key_of(index: int) -> tuple:
        return (keys[index],) if keys[index] is not None else ()

    def block(index: int, msa: mx.array, pair: mx.array) -> tuple[mx.array, mx.array]:
        if functions is not None:
            return tuple(functions.forward(compiled.layer_weights(p, index, functions.spec), msa, pair, msa_mask, pair_mask, *key_of(index)))
        return evoformer_iteration(p.layer(index), msa, pair, msa_mask, pair_mask, cfg, dropout_key=keys[index], is_extra=is_extra, subbatch_size=subbatch_size)

    def block_vjp(index: int, msa: mx.array, pair: mx.array, cotangent_msa: mx.array, cotangent_pair: mx.array) -> list[mx.array]:
        #the block recomputed from its input and differentiated: the gradients of its input
        if functions is not None:
            return functions.backward(compiled.layer_weights(p, index, functions.spec), msa, pair, cotangent_msa, cotangent_pair, msa_mask, pair_mask, *key_of(index))
        return mx.vjp(lambda m, q: list(block(index, m, q)), [msa, pair], [cotangent_msa, cotangent_pair])[1]

    if not remat:
        if tape is not None:
            tape.inputs, tape.outputs = [], None
        for index in range(num_block):
            if tape is not None:
                tape.inputs.append((msa_act, pair_act))
            msa_act, pair_act = block(index, msa_act, pair_act)
        if tape is not None:
            tape.outputs = (msa_act, pair_act)
        return msa_act, pair_act
    if tape is not None:
        tape.check(num_block, msa_act, pair_act)
    recorded: list[tuple[mx.array, mx.array]] = []

    @mx.custom_function
    def run(msa: mx.array, pair: mx.array) -> tuple[mx.array, mx.array]:
        if tape is not None:
            return tape.outputs
        recorded.clear()
        for index in range(num_block):
            recorded.append((msa, pair))
            msa, pair = block(index, msa, pair)
        return msa, pair

    @run.vjp
    def run_vjp(primals, cotangents, outputs):
        inputs = tape.inputs if tape is not None else recorded
        cotangent_msa, cotangent_pair = (cotangent.astype(output.dtype) for cotangent, output in zip(cotangents, outputs))
        for index in reversed(range(num_block)):
            cotangent_msa, cotangent_pair = block_vjp(index, *inputs[index], cotangent_msa, cotangent_pair)
        return cotangent_msa, cotangent_pair

    return run(msa_act, pair_act)
