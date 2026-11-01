#The AlphaFold-Multimer model as the design loop uses it: recycle passes without gradient from a zero previous state (the front always builds zeros there), then a final pass; each pass is
#the embeddings and Evoformer, the structure module and the heads (modules_multimer.AlphaFoldIteration). The outputs are the flat names in outputs.OUTPUT_KEYS.
import mlx.core as mx

from bindcraft.af.mlx_af2 import embeddings, evoformer, heads, structure, templates
from bindcraft.af.mlx_af2.outputs import OUTPUT_KEYS

class MultimerModel:
    def __init__(self, model_name: str, params, cfg: dict, dtype=None):
        #cfg is the effective model config (config.model_config(name).model as a dict, with the parent's overrides applied)
        self.model_name, self.params, self.cfg = model_name, params, cfg
        self.dtype = dtype if dtype is not None else (mx.bfloat16 if cfg['global_config']['bfloat16'] else mx.float32)

    def zero_prev(self, features: dict) -> dict:
        length = features['aatype'].shape[0]
        return {'prev_msa_first_row': mx.zeros((length, self.cfg['embeddings_and_evoformer']['msa_channel'])), 'prev_pair': mx.zeros((length, length, self.cfg['embeddings_and_evoformer']['pair_channel'])), 'prev_pos': mx.zeros((length, 37, 3))}

    def new_tapes(self) -> dict:
        #one tape per block stack of the final pass: forward_with_state(tapes=) records them, vjp(tapes=) consumes them and skips that pass's Evoformer forward
        return {'extra_msa_stack': evoformer.Tape(), 'evoformer_iteration': evoformer.Tape()}

    def single_pass(self, features: dict, prev: dict, dropout_key=None, *, subbatch_size: int | None = None, remat: bool = True, tapes: dict | None = None) -> tuple[dict, dict]:
        #one AlphaFoldIteration: returns the nested outputs and the state the next pass starts from
        cfg, p = self.cfg, self.params
        keys = list(mx.random.split(dropout_key, 2)) if dropout_key is not None else [None, None]
        embedding_cfg = cfg['embeddings_and_evoformer']
        hook = templates.template_hook(p.scope('evoformer'), features, embedding_cfg, self.dtype) if embedding_cfg['template']['enabled'] else None
        representations = embeddings.embeddings_and_evoformer(p.scope('evoformer'), features, prev, embedding_cfg, dtype=self.dtype, dropout_key=keys[0], remat=remat, subbatch_size=subbatch_size, template_embedding=hook, tapes=tapes)
        return self.heads_of(representations, features, keys[1])

    def heads_of(self, representations: dict, features: dict, dropout_key=None) -> tuple[dict, dict]:
        #the structure module and the heads on the Evoformer's representations: the nested outputs and the state the next pass starts from
        cfg, p = self.cfg, self.params
        sm = structure.structure_module(p.scope('structure_module'), representations, features, cfg['heads']['structure_module'], dropout_key=dropout_key)
        representations = {**representations, 'structure_module': sm['act']}
        h = cfg['heads']
        outputs = {'structure_module': {'final_atom_positions': sm['final_atom_positions'], 'final_atom_mask': sm['final_atom_mask']},
                   'predicted_lddt': heads.predicted_lddt(p.scope('predicted_lddt_head'), representations, h['predicted_lddt']),
                   'predicted_aligned_error': heads.predicted_aligned_error(p.scope('predicted_aligned_error_head'), representations, h['predicted_aligned_error']),
                   'distogram': heads.distogram(p.scope('distogram_head'), representations, h['distogram']),
                   'experimentally_resolved': heads.experimentally_resolved(p.scope('experimentally_resolved_head'), representations, h['experimentally_resolved'])}
        return outputs, {'prev_pos': sm['final_atom_positions'], 'prev_msa_first_row': representations['msa_first_row'], 'prev_pair': representations['pair']}

    def recycle(self, features: dict, dropout_keys: list | None, num_recycle: int, *, subbatch_size: int | None = None) -> tuple[dict, dict]:
        #the passes before the last: no gradient flows through them, and initial_atom_pos (the large-complex initialization) becomes the previous pass's positions, as recycled_alphafold_outputs does
        prev = self.zero_prev(features)
        for index in range(num_recycle):
            _, prev = self.single_pass(features, prev, dropout_keys[index] if dropout_keys else None, subbatch_size=subbatch_size, remat=False)
            prev = {name: mx.stop_gradient(value) for name, value in prev.items()}
            mx.eval(prev)
            if 'initial_atom_pos' in features:
                features = {**features, 'initial_atom_pos': prev['prev_pos']}
        return features, prev

    def dropout_keys(self, features: dict, key, num_recycle: int, use_dropout: bool | None) -> list | None:
        use_dropout = bool(features.get('use_dropout', False)) if use_dropout is None else use_dropout
        if not use_dropout:
            return None
        seed = (int(key[0]) * 1000003 + int(key[1])) % (2 ** 31 - 1)  #the dropout streams are MLX's own: reproducible per key, not equal to JAX's
        return list(mx.random.split(mx.random.key(seed), num_recycle + 1))

    def final_features(self, features: dict, prev: dict, num_recycle: int) -> dict:
        #recycle() rewrites initial_atom_pos after every pass, so the final pass sees the last recycle pass's positions; a caller that holds the state before the final pass applies the same rewrite
        if num_recycle > 0 and 'initial_atom_pos' in features:
            return {**features, 'initial_atom_pos': prev['prev_pos']}
        return features

    def forward_with_state(self, features: dict, key, num_recycle: int, *, subbatch_size: int | None = None, use_dropout: bool | None = None, tapes: dict | None = None) -> tuple[dict[str, mx.array], dict, dict]:
        #the flat outputs by OUTPUT_KEYS, the state after the final pass, and the state BEFORE it (what vjp resumes from: the recycle passes are not repeated); key is the uint32 pair of a JAX PRNG key.
        #tapes (new_tapes()) are filled with the final pass's block inputs, for a vjp of the same call
        keys = self.dropout_keys(features, key, num_recycle, use_dropout)
        features, before = self.recycle(features, keys, num_recycle, subbatch_size=subbatch_size)
        outputs, after = self.single_pass(features, before, keys[-1] if keys else None, subbatch_size=subbatch_size, remat=False, tapes=tapes)
        flat = {f'{group}/{name}': value for group, tensors in outputs.items() for name, value in tensors.items()}
        return {name: flat[name].astype(mx.float32) for name in OUTPUT_KEYS}, after, before

    def forward(self, features: dict, key, num_recycle: int, *, subbatch_size: int | None = None, use_dropout: bool | None = None) -> tuple[dict[str, mx.array], dict]:
        outputs, after, _ = self.forward_with_state(features, key, num_recycle, subbatch_size=subbatch_size, use_dropout=use_dropout)
        return outputs, after

    def vjp(self, features: dict, key, num_recycle: int, cotangents: dict[str, mx.array], diff_keys: tuple[str, ...], *, prev: dict | None = None, subbatch_size: int | None = None,
            use_dropout: bool | None = None, tapes: dict | None = None) -> dict[str, mx.array]:
        #the gradient of sum(cotangent * output) with respect to the features named in diff_keys, through the final pass only (the recycle passes carry no gradient, as in recycled_alphafold_outputs); prev, the
        #state before the final pass from forward_with_state, skips them. use_dropout follows the features' flag like forward does. The backward of the final pass is a chain run outside any MLX transformation,
        #so the stacks' compiled block vjp closures stay fused (MLX runs a compiled function inline, unfused, under mx.vjp): the vjp of the tail (single activations, structure module, heads) for the main stack's
        #outputs, the main stack's reverse loop, the extra-MSA stack's, then the vjp of the front (embeddings, templates) for the features. The stacks' block inputs come from tapes, recorded by forward_with_state
        #for this same call (the final pass's Evoformer forward is then skipped), or from this call's own forward. Each block is recomputed in the reverse loop instead of kept.
        keys = self.dropout_keys(features, key, num_recycle, use_dropout)
        if prev is None:
            features, prev = self.recycle(features, keys, num_recycle, subbatch_size=subbatch_size)
        else:
            features = self.final_features(features, prev, num_recycle)
        names = [name for name in OUTPUT_KEYS if name in cotangents]
        cfg, evo = self.cfg['embeddings_and_evoformer'], self.params.scope('evoformer')
        pass_keys = list(mx.random.split(keys[-1], 2)) if keys else [None, None]  #as single_pass splits them: the embeddings and stacks, the structure module
        hook = templates.template_hook(evo, features, cfg, self.dtype) if cfg['template']['enabled'] else None

        def front_of(*arrays) -> embeddings.Front:
            return embeddings.front(evo, {**features, **dict(zip(diff_keys, arrays))}, prev, cfg, dtype=self.dtype, dropout_key=pass_keys[0], template_embedding=hook)

        base = front_of(*[features[name] for name in diff_keys])
        msa_in, msa_mask = embeddings.join(base)
        if tapes is None:
            tapes = self.new_tapes()
            pair_in = embeddings.extra_msa_stack(evo, base.extra_msa_act, base.pair, base.extra_msa_mask, base.mask_2d, cfg, dropout_key=base.extra_key, remat=False, subbatch_size=subbatch_size,
                                                 tape=tapes['extra_msa_stack'])
            evoformer.evoformer_stack(evo.scope('evoformer_iteration'), msa_in, pair_in, msa_mask, base.mask_2d, cfg['evoformer'], num_block=cfg['evoformer_num_block'], dropout_key=base.main_key,
                                      remat=False, subbatch_size=subbatch_size, tape=tapes['evoformer_iteration'])
        else:
            tapes['extra_msa_stack'].check(cfg['extra_msa_stack_num_block'], base.extra_msa_act, base.pair)
            tapes['evoformer_iteration'].check(cfg['evoformer_num_block'], msa_in, tapes['extra_msa_stack'].outputs[1])
        sequences = base.msa.shape[0]

        def tail(msa, pair):
            outputs, _ = self.heads_of(embeddings.finish(evo, msa, pair, sequences), features, pass_keys[1])
            flat = {f'{group}/{name}': value for group, tensors in outputs.items() for name, value in tensors.items()}
            return [flat[name].astype(mx.float32) for name in names]

        _, (cotangent_msa, cotangent_pair) = mx.vjp(tail, list(tapes['evoformer_iteration'].outputs), [cotangents[name] for name in names])
        cotangent_msa, cotangent_pair = evoformer.stack_vjp(evo.scope('evoformer_iteration'), tapes['evoformer_iteration'], (cotangent_msa, cotangent_pair), msa_mask, base.mask_2d, cfg['evoformer'],
                                                            dropout_key=base.main_key, subbatch_size=subbatch_size)
        _, cotangent_pair = evoformer.stack_vjp(evo.scope('extra_msa_stack'), tapes['extra_msa_stack'], (mx.zeros_like(tapes['extra_msa_stack'].outputs[0]), cotangent_pair), base.extra_msa_mask, base.mask_2d,
                                                cfg['evoformer'], dropout_key=base.extra_key, subbatch_size=subbatch_size, is_extra=True)
        _, gradients = mx.vjp(lambda *arrays: (lambda f: [f.msa, f.pair])(front_of(*arrays)), [features[name] for name in diff_keys], [cotangent_msa[:sequences], cotangent_pair])
        return dict(zip(diff_keys, gradients))
