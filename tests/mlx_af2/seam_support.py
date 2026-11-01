"""Model subclasses and a remote shared by the seam tests and the checkpoint-0 benchmark."""
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from bindcraft.af.mlx_af2 import bridge
from bindcraft.af.mlx_af2.outputs import nest, output_shapes
from bindcraft.af2 import AlphaFoldDesignModel
from stub_remote import stub_weights

HANDLES = {'model_1_multimer_v3': 0, 'model_2_multimer_v3': 1}
BOTH_MODELS = tuple(HANDLES)

def jax_stand_in(handle, model_inputs) -> dict:
    #the stub's math in plain JAX: the reference the bridge's gradients are compared with
    w = {name: jnp.asarray(value) for name, value in stub_weights().items()}
    scale = 1.0 + 0.5 * handle.astype(jnp.float32)
    t, m = model_inputs['target_feat'], model_inputs['msa_feat'][0]
    delta = 0.05 * (t @ w['pt'] + m @ w['pm']).reshape(t.shape[0], 37, 3)
    u, v, pu, pv = t @ w['dt'], m @ w['dm'], t @ w['pt2'], m @ w['pm2']
    return nest({'structure_module/final_atom_positions': model_inputs['all_atom_positions'] + scale * delta, 'structure_module/final_atom_mask': model_inputs['atom37_atom_exists'].astype(jnp.float32),
                 'predicted_lddt/logits': scale * (m @ w['lm'] + t @ w['lt']), 'predicted_aligned_error/logits': scale * (pu[:, None, :] - pv[None, :, :]),
                 'predicted_aligned_error/breaks': jnp.linspace(0.0, 31.0, 63, dtype=jnp.float32), 'distogram/logits': scale * (u[:, None, :] + v[None, :, :]),
                 'experimentally_resolved/logits': scale * (t @ w['et'])})

class StandInModel(AlphaFoldDesignModel):
    def _load_parameters(self, model_name, data_dir):
        return jnp.asarray(HANDLES[model_name], jnp.int32)

    def _alphafold_runner(self, model_family, subbatch_size, attention_backend=None):
        return SimpleNamespace(subbatch_size=subbatch_size, attention_backend=attention_backend or self.attention_backend)

    def _model_outputs(self, alphafold_runner, model_parameters, key, model_inputs):
        return jax_stand_in(model_parameters, model_inputs)

class BridgedModel(StandInModel):
    def __init__(self, remote, **arguments):
        self.remote, self.trace_count = remote, 0
        super().__init__(**arguments)

    def _model_outputs(self, alphafold_runner, model_parameters, key, model_inputs):
        self.trace_count += 1
        return bridge.mlx_model_outputs(self.remote, model_parameters, key, model_inputs, self.num_recycle, alphafold_runner.subbatch_size)


class InstantRemote:
    #answers from memory with arrays of the right shape and zero gradients, so the JAX side of a step (front, loss, callbacks, array handling) is all that is left to time
    def __init__(self):
        self.outputs, self.rng = {}, np.random.default_rng(0)

    def forward(self, model_handle, key, features, num_recycle, subbatch_size, retain=False):
        length = features['aatype'].shape[0]
        if length not in self.outputs:
            made = {name: (0.1 * self.rng.standard_normal(shape)).astype(dtype) for name, (shape, dtype) in output_shapes(length).items()}
            made['structure_module/final_atom_mask'] = features['atom37_atom_exists'].astype(np.float32)
            made['predicted_aligned_error/breaks'] = np.linspace(0.0, 31.0, 63, dtype=np.float32)
            self.outputs[length] = made
        positions = dict(self.outputs[length])
        positions['structure_module/final_atom_positions'] = features['all_atom_positions'].astype(np.float32)
        return positions, 1

    def vjp(self, model_handle, call_id, key, features, num_recycle, subbatch_size, cotangents, diff_keys):
        return {name: np.zeros_like(features[name]) for name in diff_keys}
