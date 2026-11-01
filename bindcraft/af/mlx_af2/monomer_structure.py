#The monomer models' structure module (folding.py with quat_affine.py): quaternion-and-translation affines, the invariant point attention, the fold iteration (one set of weights, applied num_layer
#times), the sidechains (shared with the multimer port: the same module and parameter names) and the final atom positions. It runs in float32 whatever the compute dtype, as the JAX module does.
#The large-complex initial positions (initial_atom_pos, which needs the quaternion of a rotation matrix) are refused: the repo only builds the monomer models for validation, without them.
import math
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from bindcraft.af.mlx_af2 import geometry, layers, structure

QUAT_TO_ROT = np.zeros((4, 4, 3, 3), dtype=np.float32)
QUAT_TO_ROT[0, 0] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
QUAT_TO_ROT[1, 1] = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
QUAT_TO_ROT[2, 2] = [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]
QUAT_TO_ROT[3, 3] = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
QUAT_TO_ROT[1, 2] = [[0, 2, 0], [2, 0, 0], [0, 0, 0]]
QUAT_TO_ROT[1, 3] = [[0, 0, 2], [0, 0, 0], [2, 0, 0]]
QUAT_TO_ROT[2, 3] = [[0, 0, 0], [0, 0, 2], [0, 2, 0]]
QUAT_TO_ROT[0, 1] = [[0, 0, 0], [0, 0, -2], [0, 2, 0]]
QUAT_TO_ROT[0, 2] = [[0, 0, 2], [0, 0, 0], [-2, 0, 0]]
QUAT_TO_ROT[0, 3] = [[0, -2, 0], [2, 0, 0], [0, 0, 0]]
QUAT_MULTIPLY = np.zeros((4, 4, 4), dtype=np.float32)
QUAT_MULTIPLY[:, :, 0] = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]
QUAT_MULTIPLY[:, :, 1] = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]]
QUAT_MULTIPLY[:, :, 2] = [[0, 0, 1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 0, 0]]
QUAT_MULTIPLY[:, :, 3] = [[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0]]
QUAT_MULTIPLY_BY_VEC = QUAT_MULTIPLY[:, 1:, :]

def quat_to_rot(quaternion: mx.array) -> mx.array:
    #the rotation matrix [..., 3, 3] of a unit quaternion [..., 4], summed as quat_affine.quat_to_rot sums it
    table = mx.array(QUAT_TO_ROT.reshape(4, 4, 9))
    flat = mx.sum(table * quaternion[..., :, None, None] * quaternion[..., None, :, None], axis=(-3, -2))
    return flat.reshape(*quaternion.shape[:-1], 3, 3)

def quat_multiply_by_vec(quaternion: mx.array, vector: mx.array) -> mx.array:
    return mx.sum(mx.array(QUAT_MULTIPLY_BY_VEC) * quaternion[..., :, None, None] * vector[..., None, :, None], axis=(-3, -2))

def expand(array: mx.array, extra_dims: int) -> mx.array:
    for _ in range(extra_dims):
        array = array[..., None]
    return array

@dataclass
class Affine:
    #quat_affine.QuatAffine: a unit quaternion [N, 4], the translation [N, 3] and the rotation matrix [N, 3, 3] derived from the quaternion
    quaternion: mx.array
    translation: mx.array
    rotation: mx.array

    @classmethod
    def from_tensor(cls, tensor: mx.array) -> 'Affine':
        quaternion = tensor[..., :4]
        return cls(quaternion, tensor[..., 4:7], quat_to_rot(quaternion))

    @classmethod
    def identity(cls, count: int) -> 'Affine':
        quaternion = mx.concatenate([mx.ones((count, 1)), mx.zeros((count, 3))], axis=-1)
        return cls(quaternion, mx.zeros((count, 3)), quat_to_rot(quaternion))

    def to_tensor(self) -> mx.array:
        return mx.concatenate([self.quaternion, self.translation], axis=-1)

    def scale_translation(self, scale: float) -> 'Affine':
        return Affine(self.quaternion, self.translation * scale, self.rotation)

    def pre_compose(self, update: mx.array) -> 'Affine':
        #update: the three imaginary parts of the quaternion update (1, x, y, z) and the translation update in the current frame, [N, 6]
        quaternion = self.quaternion + quat_multiply_by_vec(self.quaternion, update[..., :3])
        x, y, z = update[..., 3], update[..., 4], update[..., 5]
        rotated = [self.rotation[..., row, 0] * x + self.rotation[..., row, 1] * y + self.rotation[..., row, 2] * z for row in range(3)]
        translation = mx.stack([self.translation[..., axis] + rotated[axis] for axis in range(3)], axis=-1)
        quaternion = quaternion / mx.sqrt(mx.sum(mx.square(quaternion), axis=-1, keepdims=True))
        return Affine(quaternion, translation, quat_to_rot(quaternion))

    def apply_to_point(self, point: list, extra_dims: int = 0) -> list:
        #point: x, y, z, each with extra_dims trailing axes the affine does not have
        x, y, z = point
        entry = lambda row, column: expand(self.rotation[..., row, column], extra_dims)
        return [entry(row, 0) * x + entry(row, 1) * y + entry(row, 2) * z + expand(self.translation[..., row], extra_dims) for row in range(3)]

    def invert_point(self, point: list, extra_dims: int = 0) -> list:
        moved = [point[axis] - expand(self.translation[..., axis], extra_dims) for axis in range(3)]
        entry = lambda row, column: expand(self.rotation[..., row, column], extra_dims)
        return [entry(0, column) * moved[0] + entry(1, column) * moved[1] + entry(2, column) * moved[2] for column in range(3)]

    def as_rigid(self) -> geometry.Rigid3:
        return geometry.Rigid3(geometry.Rot3.from_array(self.rotation), geometry.Vec3.from_array(self.translation))

def invariant_point_attention(p, inputs_1d: mx.array, inputs_2d: mx.array, mask: mx.array, affine: Affine, cfg: dict, dist_epsilon: float = 1e-8) -> mx.array:
    #folding.InvariantPointAttention: scalar queries and keys, points in each residue's frame taken to the global frame, a bias from the pair activations; p is the module's scope
    n, heads = inputs_1d.shape[0], cfg['num_head']
    scalar_qk, point_qk, scalar_v, point_v = cfg['num_scalar_qk'], cfg['num_point_qk'], cfg['num_scalar_v'], cfg['num_point_v']
    q_scalar = layers.linear(p.scope('q_scalar'), inputs_1d).reshape(n, heads, scalar_qk)
    kv_scalar = layers.linear(p.scope('kv_scalar'), inputs_1d).reshape(n, heads, scalar_v + scalar_qk)
    k_scalar, v_scalar = kv_scalar[..., :scalar_qk], kv_scalar[..., scalar_qk:]
    q_point = [x.reshape(n, heads, point_qk) for x in affine.apply_to_point(list(mx.split(layers.linear(p.scope('q_point_local'), inputs_1d), 3, axis=-1)), 1)]
    kv_point = [x.reshape(n, heads, point_qk + point_v) for x in affine.apply_to_point(list(mx.split(layers.linear(p.scope('kv_point_local'), inputs_1d), 3, axis=-1)), 1)]
    k_point, v_point = [x[..., :point_qk] for x in kv_point], [x[..., point_qk:] for x in kv_point]
    scalar_weights = math.sqrt(1.0 / (3 * max(scalar_qk, 1) * 1.0))
    point_weights = math.sqrt(1.0 / (3 * (max(point_qk, 1) * 9.0 / 2))) * structure.softplus(p.get('trainable_point_weights', mx.float32))[:, None]
    v_point, q_point, k_point = ([x.swapaxes(-2, -3) for x in group] for group in (v_point, q_point, k_point))
    dist2 = sum(mx.square(qx[:, :, None, :] - kx[:, None, :, :]) for qx, kx in zip(q_point, k_point))
    attn_qk_point = -0.5 * mx.sum(point_weights[:, None, None, :] * dist2, axis=-1)
    v = v_scalar.swapaxes(-2, -3)
    q = (scalar_weights * q_scalar).swapaxes(-2, -3)
    k = k_scalar.swapaxes(-2, -3)
    logits = mx.matmul(q, k.swapaxes(-2, -1)) + attn_qk_point
    logits = logits + math.sqrt(1.0 / 3) * layers.linear(p.scope('attention_2d'), inputs_2d).transpose(2, 0, 1)
    logits = logits - 1e5 * (1.0 - mask * mask.swapaxes(-1, -2))
    attn = mx.softmax(logits, axis=-1)
    result_scalar = mx.matmul(attn, v).swapaxes(-2, -3).reshape(n, heads * scalar_v)
    result_global = [mx.sum(attn[:, :, :, None] * vx[:, None, :, :], axis=-2).swapaxes(-2, -3).reshape(n, heads * point_v) for vx in v_point]
    local = affine.invert_point(result_global, 1)
    over_2d = mx.einsum('hij,ijc->ihc', attn, inputs_2d).reshape(n, -1)
    features = [result_scalar, *local, mx.sqrt(dist_epsilon + mx.square(local[0]) + mx.square(local[1]) + mx.square(local[2])), over_2d]
    return layers.linear(p.scope('output_projection'), mx.concatenate(features, axis=-1))

def fold_iteration(p, act: mx.array, affine: Affine, sequence_mask: mx.array, initial_act: mx.array, static_feat_2d: mx.array, aatype: mx.array, cfg: dict, *, dropout_key=None) -> tuple[mx.array, Affine, dict]:
    #folding.FoldIteration with update_affine: attention, the transition, the affine update and the sidechains of the updated frames; p is the 'structure_module/fold_iteration' scope
    keys = list(mx.random.split(dropout_key, 3))[1:] if dropout_key is not None else [None, None]
    act = act + invariant_point_attention(p.scope('invariant_point_attention'), act, static_feat_2d, sequence_mask, affine, cfg)
    act = layers.layer_norm(p.scope('attention_layer_norm'), structure.plain_dropout(act, cfg['dropout'], keys[0]))
    input_act, count = act, cfg['num_layer_in_transition']
    for index in range(count):
        act = layers.linear(p.scope('transition' if index == 0 else f'transition_{index}'), act)
        if index < count - 1:
            act = mx.maximum(act, 0)
    act = layers.layer_norm(p.scope('transition_layer_norm'), structure.plain_dropout(act + input_act, cfg['dropout'], keys[1]))
    affine = affine.pre_compose(layers.linear(p.scope('affine_update'), act))
    rigid = affine.scale_translation(cfg['position_scale']).as_rigid()
    sc = structure.sidechains(p.scope('rigid_sidechain'), rigid, [act, initial_act], aatype, cfg['sidechain'])
    return act, affine, {**sc, 'backb_to_global': rigid}

def atom14_to_atom37(atom14_positions: mx.array, features: dict) -> mx.array:
    #all_atom.atom14_to_atom37 with the batch's own index and existence tables (zeroed for padded residues by the front)
    index = features['residx_atom37_to_atom14']
    gathered = mx.take_along_axis(atom14_positions, mx.broadcast_to(index[..., None], (*index.shape, 3)), axis=1)
    return gathered * features['atom37_atom_exists'][:, :, None].astype(gathered.dtype)

def structure_module(p, representations: dict, features: dict, cfg: dict, *, dropout_key=None) -> dict:
    #StructureModule.__call__ with generate_affines: the outputs of the JAX head by name, the per-iteration ones stacked on a leading axis of num_layer; p is the 'structure_module' scope
    if 'initial_atom_pos' in features:
        raise NotImplementedError('initial_atom_pos: the large-complex initialisation of the monomer structure module (the quaternion of a rotation matrix) is not ported; the repo builds the monomer models without it')
    sequence_mask = features['seq_mask'][:, None]
    initial_act = layers.layer_norm(p.scope('single_layer_norm'), representations['single'])
    act = layers.linear(p.scope('initial_projection'), initial_act)
    affine = Affine.identity(sequence_mask.shape[0])
    static_feat_2d = layers.layer_norm(p.scope('pair_layer_norm'), representations['pair'])
    keys = list(mx.random.split(dropout_key, cfg['num_layer'])) if dropout_key is not None else [None] * cfg['num_layer']
    tensors, outputs = [], []
    for index in range(cfg['num_layer']):
        act, affine, sc = fold_iteration(p.scope('fold_iteration'), act, affine, sequence_mask, initial_act, static_feat_2d, features['aatype'], cfg, dropout_key=keys[index])
        affine = Affine.from_tensor(affine.to_tensor())  #the carried state is the tensor, as in the scan
        tensors.append(affine.to_tensor())
        outputs.append(sc)
    scale = cfg['position_scale']
    traj = mx.stack(tensors) * mx.array([1.0] * 4 + [scale] * 3)
    stacked = {'angles_sin_cos': mx.stack([sc['angles_sin_cos'] for sc in outputs]), 'unnormalized_angles_sin_cos': mx.stack([sc['unnormalized_angles_sin_cos'] for sc in outputs]),
               'atom_pos': mx.stack([sc['atom_pos'].to_array() for sc in outputs]), 'frames': mx.stack([sc['frames'].to_array() for sc in outputs]),
               'backb_to_global': mx.stack([sc['backb_to_global'].to_array() for sc in outputs])}
    atom14_positions = stacked['atom_pos'][-1]
    return {'traj': traj, 'sidechains': stacked, 'final_atom14_positions': atom14_positions, 'final_atom14_mask': features['atom14_atom_exists'],
            'final_atom_positions': atom14_to_atom37(atom14_positions, features), 'final_atom_mask': features['atom37_atom_exists'], 'final_affines': traj[-1], 'act': act}
