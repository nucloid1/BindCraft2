#The AlphaFold-Multimer model as the design loop uses it: recycle passes without gradient from a zero previous state (the front always builds zeros there), then a final pass; each pass is
#the embeddings and Evoformer, the structure module and the heads (modules_multimer.AlphaFoldIteration). The outputs are the flat names in outputs.OUTPUT_KEYS.
import mlx.core as mx

from bindcraft.af.mlx_af2 import embeddings, heads, structure, templates
from bindcraft.af.mlx_af2.outputs import OUTPUT_KEYS

class MultimerModel:
    def __init__(self, model_name: str, params, cfg: dict, dtype=None):
        #cfg is the effective model config (config.model_config(name).model as a dict, with the parent's overrides applied)
        self.model_name, self.params, self.cfg = model_name, params, cfg
        self.dtype = dtype if dtype is not None else (mx.bfloat16 if cfg['global_config']['bfloat16'] else mx.float32)

    def zero_prev(self, features: dict) -> dict:
        length = features['aatype'].shape[0]
        return {'prev_msa_first_row': mx.zeros((length, self.cfg['embeddings_and_evoformer']['msa_channel'])), 'prev_pair': mx.zeros((length, length, self.cfg['embeddings_and_evoformer']['pair_channel'])), 'prev_pos': mx.zeros((length, 37, 3))}

    def single_pass(self, features: dict, prev: dict, dropout_key=None, *, subbatch_size: int | None = None, remat: bool = True) -> tuple[dict, dict]:
        #one AlphaFoldIteration: returns the nested outputs and the state the next pass starts from
        cfg, p = self.cfg, self.params
        keys = list(mx.random.split(dropout_key, 2)) if dropout_key is not None else [None, None]
        embedding_cfg = cfg['embeddings_and_evoformer']
        hook = templates.template_hook(p.scope('evoformer'), features, embedding_cfg, self.dtype) if embedding_cfg['template']['enabled'] else None
        representations = embeddings.embeddings_and_evoformer(p.scope('evoformer'), features, prev, embedding_cfg, dtype=self.dtype, dropout_key=keys[0], remat=remat, subbatch_size=subbatch_size, template_embedding=hook)
        sm = structure.structure_module(p.scope('structure_module'), representations, features, cfg['heads']['structure_module'], dropout_key=keys[1])
        representations['structure_module'] = sm['act']
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

    def forward_with_state(self, features: dict, key, num_recycle: int, *, subbatch_size: int | None = None, use_dropout: bool | None = None) -> tuple[dict[str, mx.array], dict, dict]:
        #the flat outputs by OUTPUT_KEYS, the state after the final pass, and the state BEFORE it (what vjp resumes from: the recycle passes are not repeated); key is the uint32 pair of a JAX PRNG key
        keys = self.dropout_keys(features, key, num_recycle, use_dropout)
        features, before = self.recycle(features, keys, num_recycle, subbatch_size=subbatch_size)
        outputs, after = self.single_pass(features, before, keys[-1] if keys else None, subbatch_size=subbatch_size, remat=False)
        flat = {f'{group}/{name}': value for group, tensors in outputs.items() for name, value in tensors.items()}
        return {name: flat[name].astype(mx.float32) for name in OUTPUT_KEYS}, after, before

    def forward(self, features: dict, key, num_recycle: int, *, subbatch_size: int | None = None, use_dropout: bool | None = None) -> tuple[dict[str, mx.array], dict]:
        outputs, after, _ = self.forward_with_state(features, key, num_recycle, subbatch_size=subbatch_size, use_dropout=use_dropout)
        return outputs, after

    def vjp(self, features: dict, key, num_recycle: int, cotangents: dict[str, mx.array], diff_keys: tuple[str, ...], *, prev: dict | None = None, subbatch_size: int | None = None,
            use_dropout: bool | None = None) -> dict[str, mx.array]:
        #the gradient of sum(cotangent * output) with respect to the features named in diff_keys, through the final pass only (the recycle passes carry no gradient, as in recycled_alphafold_outputs); prev, the
        #state before the final pass from forward_with_state, skips them; each Evoformer block is recomputed in the backward pass instead of kept. use_dropout follows the features' flag like forward does
        keys = self.dropout_keys(features, key, num_recycle, use_dropout)
        if prev is None:
            features, prev = self.recycle(features, keys, num_recycle, subbatch_size=subbatch_size)
        else:
            features = self.final_features(features, prev, num_recycle)
        names = [name for name in OUTPUT_KEYS if name in cotangents]

        def final(*arrays):
            outputs, _ = self.single_pass({**features, **dict(zip(diff_keys, arrays))}, prev, keys[-1] if keys else None, subbatch_size=subbatch_size, remat=True)
            flat = {f'{group}/{name}': value for group, tensors in outputs.items() for name, value in tensors.items()}
            return [flat[name].astype(mx.float32) for name in names]

        _, gradients = mx.vjp(final, [features[name] for name in diff_keys], [cotangents[name] for name in names])
        return dict(zip(diff_keys, gradients))
