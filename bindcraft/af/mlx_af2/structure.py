#The structure module (folding_multimer.py): the invariant point attention, the fold iteration that updates the activations and the backbone frames, and the eight-iteration backbone loop.
#It runs in float32 whatever the model's compute dtype, as the JAX module does. The sidechains and the atom positions are added below by a later task.
import math

import mlx.core as mx

from bindcraft.af.alphafold.common import residue_constants
from bindcraft.af.mlx_af2 import geometry, layers

def softplus(x: mx.array) -> mx.array:
    return mx.logaddexp(x, mx.zeros_like(x))

def plain_dropout(tensor: mx.array, rate: float, key) -> mx.array:
    #modules.apply_dropout without a broadcast axis; no key means off
    if key is None or not rate:
        return tensor
    return mx.random.bernoulli(1.0 - rate, tensor.shape, key=key) * tensor / (1.0 - rate)

def quat_rigid(p, activations: mx.array) -> geometry.Rigid3:
    #QuatRigid with a three-component rotation (the quaternion's first component fixed at 1) and a translation
    flat = layers.linear(p.scope('rigid'), activations)
    qx, qy, qz = flat[..., 0], flat[..., 1], flat[..., 2]
    return geometry.Rigid3(geometry.Rot3.from_quaternion(mx.ones_like(qx), qx, qy, qz, normalize=True), geometry.Vec3(flat[..., 3], flat[..., 4], flat[..., 5]))

def point_projection(p, activations: mx.array, rigids: geometry.Rigid3, num_head: int, num_points: int) -> geometry.Vec3:
    #points in each residue's local frame, taken to the global frame; the rigid [N] is viewed as [N, 1, 1] against points [N, heads, points]
    local = layers.linear(p.scope('point_projection'), activations)
    x, y, z = mx.split(local, 3, axis=-1)
    return rigids[:, None, None].apply_to_point(geometry.Vec3(x, y, z))

def invariant_point_attention(p, inputs_1d: mx.array, inputs_2d: mx.array, mask: mx.array, rigid: geometry.Rigid3, cfg: dict, dist_epsilon: float = 1e-8) -> mx.array:
    heads, points_qk, points_v = cfg['num_head'], cfg['num_point_qk'], cfg['num_point_v']
    queries = inputs_1d.shape[0]
    point_weights = math.sqrt(1.0 / (max(points_qk, 1) * 9.0 / 2)) * softplus(p.get('trainable_point_weights', mx.float32))
    q_point = point_projection(p.scope('q_point_projection'), inputs_1d, rigid, heads, points_qk)
    k_point = point_projection(p.scope('k_point_projection'), inputs_1d, rigid, heads, points_qk)
    squared = mx.square(q_point.x[:, None] - k_point.x[None]) + mx.square(q_point.y[:, None] - k_point.y[None]) + mx.square(q_point.z[:, None] - k_point.z[None])
    logits = -0.5 * mx.sum(point_weights[:, None] * squared, axis=-1)
    q_scalar = layers.linear(p.scope('q_scalar_projection'), inputs_1d) * math.sqrt(1.0 / max(cfg['num_scalar_qk'], 1))
    k_scalar = layers.linear(p.scope('k_scalar_projection'), inputs_1d)
    logits = logits + mx.einsum('qhc,khc->qkh', q_scalar, k_scalar) + layers.linear(p.scope('attention_2d'), inputs_2d)
    mask_2d = mask * mask.swapaxes(-1, -2)
    logits = (logits - 1e5 * (1.0 - mask_2d[..., None])) * math.sqrt(1.0 / 3)
    attn = mx.softmax(logits, axis=-2)
    result_scalar = mx.einsum('qkh,khc->qhc', attn, layers.linear(p.scope('v_scalar_projection'), inputs_1d))
    v_point = point_projection(p.scope('v_point_projection'), inputs_1d, rigid, heads, points_v)
    flat = lambda array: array.reshape(queries, -1)
    result_global = v_point.map(lambda component: flat(mx.sum(attn[..., None] * component[None], axis=-3)))
    result_local = rigid[..., None].apply_inverse_to_point(result_global)
    features = [flat(result_scalar), result_local.x, result_local.y, result_local.z, result_local.norm(dist_epsilon), flat(mx.einsum('ijh,ijc->ihc', attn, inputs_2d))]
    return layers.linear(p.scope('output_projection'), mx.concatenate(features, axis=-1))

def fold_iteration(p, act: mx.array, rigid: geometry.Rigid3, sequence_mask: mx.array, static_feat_2d: mx.array, cfg: dict, *, update_rigid: bool = True, dropout_key=None) -> tuple[mx.array, geometry.Rigid3]:
    #attention, the three-layer transition and the rigid update (FoldIteration without its sidechain head); dropout only with a key
    keys = list(mx.random.split(dropout_key, 3))[1:] if dropout_key is not None else [None, None]
    act = act + invariant_point_attention(p.scope('invariant_point_attention'), act, static_feat_2d, sequence_mask, rigid, cfg)
    act = layers.layer_norm(p.scope('attention_layer_norm'), plain_dropout(act, cfg['dropout'], keys[0]))
    input_act, layer_count = act, cfg['num_layer_in_transition']
    for index in range(layer_count):
        act = layers.linear(p.scope('transition' if index == 0 else f'transition_{index}'), act)
        if index < layer_count - 1:
            act = mx.maximum(act, 0)
    act = layers.layer_norm(p.scope('transition_layer_norm'), plain_dropout(act + input_act, cfg['dropout'], keys[1]))
    if update_rigid:
        rigid = rigid @ quat_rigid(p.scope('quat_rigid'), act)
    return act, rigid

def backbone_trajectory(p, representations: dict, features: dict, cfg: dict, *, dropout_key=None) -> tuple[mx.array, list[geometry.Rigid3]]:
    #generate_monomer_rigids without the sidechains: returns the final activations and the rigid after each of the iterations, in units of position_scale
    sequence_mask = features['seq_mask'][:, None]
    act = layers.layer_norm(p.scope('single_layer_norm'), representations['single'])
    act = layers.linear(p.scope('initial_projection'), act)
    if 'initial_atom_pos' in features:
        atoms, order = features['initial_atom_pos'].astype(mx.float32), residue_constants.atom_order
        rigid = geometry.make_transform_from_reference(*[geometry.Vec3.from_array(atoms[:, order[name]]) for name in ('N', 'CA', 'C')]).scale_translation(1.0 / cfg['position_scale'])
    else:
        rigid = geometry.Rigid3.identity((sequence_mask.shape[0],))
    static_feat_2d = layers.layer_norm(p.scope('pair_layer_norm'), representations['pair'])
    keys = list(mx.random.split(dropout_key, cfg['num_layer'])) if dropout_key is not None else [None] * cfg['num_layer']
    trajectory = []
    for index in range(cfg['num_layer']):
        act, rigid = fold_iteration(p.scope('fold_iteration'), act, rigid, sequence_mask, static_feat_2d, cfg, dropout_key=keys[index])
        trajectory.append(rigid)
    return act, trajectory
