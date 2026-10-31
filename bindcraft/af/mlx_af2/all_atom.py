#The all-atom helpers of the structure module (all_atom_multimer.py): the per-residue-type tables, the atom14 and atom37 conversions, the torsion angles to rigid group frames, and the frames with
#literature positions to atom14 coordinates. The tables are built here from residue_constants exactly as the JAX module builds them (that module imports jax, which the worker never does).
import mlx.core as mx
import numpy as np

from bindcraft.af.alphafold.common import residue_constants
from bindcraft.af.mlx_af2 import geometry

def _make_chi_atom_indices():
    chi_atom_indices = []
    for residue_name in residue_constants.restypes:
        residue_name = residue_constants.restype_1to3[residue_name]
        atom_indices = [[residue_constants.atom_order[atom] for atom in chi_angle] for chi_angle in residue_constants.chi_angles_atoms[residue_name]]
        atom_indices += [[0, 0, 0, 0]] * (4 - len(atom_indices))
        chi_atom_indices.append(atom_indices)
    chi_atom_indices.append([[0, 0, 0, 0]] * 4)
    return np.array(chi_atom_indices)

def _make_restype_atom37_mask():
    mask = np.zeros([21, 37], dtype=np.float32)
    for restype, letter in enumerate(residue_constants.restypes):
        for atom_name in residue_constants.residue_atoms[residue_constants.restype_1to3[letter]]:
            mask[restype, residue_constants.atom_order[atom_name]] = 1
    return mask

def _make_restype_atom14_mask():
    rows = [[(1.0 if name else 0.0) for name in residue_constants.restype_name_to_atom14_names[residue_constants.restype_1to3[rt]]] for rt in residue_constants.restypes]
    return np.array(rows + [[0.0] * 14], dtype=np.float32)

def _make_restype_atom37_to_atom14():
    rows = []
    for rt in residue_constants.restypes:
        names = residue_constants.restype_name_to_atom14_names[residue_constants.restype_1to3[rt]]
        index = {name: i for i, name in enumerate(names)}
        rows.append([(index[name] if name in index else 0) for name in residue_constants.atom_types])
    return np.array(rows + [[0] * 37], dtype=np.int32)

def _make_restype_atom14_to_atom37():
    rows = []
    for rt in residue_constants.restypes:
        names = residue_constants.restype_name_to_atom14_names[residue_constants.restype_1to3[rt]]
        rows.append([(residue_constants.atom_order[name] if name else 0) for name in names])
    return np.array(rows + [[0] * 14], dtype=np.int32)

def _make_restype_rigidgroup_base_atom37_idx():
    base_atom_names = np.full([21, 8, 3], '', dtype=object)
    base_atom_names[:, 0, :] = ['C', 'CA', 'N']
    base_atom_names[:, 3, :] = ['CA', 'C', 'O']
    for restype, letter in enumerate(residue_constants.restypes):
        resname = residue_constants.restype_1to3[letter]
        for chi_index in range(4):
            if residue_constants.chi_angles_mask[restype][chi_index]:
                base_atom_names[restype, chi_index + 4, :] = residue_constants.chi_angles_atoms[resname][chi_index][1:]
    lookup = residue_constants.atom_order.copy()
    lookup[''] = 0
    return np.vectorize(lambda name: lookup[name])(base_atom_names)

CHI_ATOM_INDICES = _make_chi_atom_indices()
RESTYPE_ATOM14_TO_ATOM37 = _make_restype_atom14_to_atom37()
RESTYPE_ATOM37_TO_ATOM14 = _make_restype_atom37_to_atom14()
RESTYPE_ATOM37_MASK = _make_restype_atom37_mask()
RESTYPE_ATOM14_MASK = _make_restype_atom14_mask()
RESTYPE_RIGIDGROUP_BASE_ATOM37_IDX = _make_restype_rigidgroup_base_atom37_idx()
CHI_ANGLES_MASK = np.array(list(residue_constants.chi_angles_mask) + [[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
_cache: dict = {}

def rows_for(table: np.ndarray, aatype: mx.array, name: str | None = None) -> mx.array:
    #batched_gather(table, aatype) with the indices clipped as the JAX gather clips them
    key = name or id(table)
    if key not in _cache:
        _cache[key] = mx.array(table)
    return mx.take(_cache[key], mx.clip(aatype, 0, table.shape[0] - 1), axis=0)

def atom37_mask(aatype: mx.array) -> mx.array:
    return rows_for(RESTYPE_ATOM37_MASK, aatype, 'atom37_mask')

def atom14_mask(aatype: mx.array) -> mx.array:
    return rows_for(RESTYPE_ATOM14_MASK, aatype, 'atom14_mask')

def atom14_to_atom37(atom14_data: mx.array, aatype: mx.array) -> mx.array:
    index = rows_for(RESTYPE_ATOM37_TO_ATOM14, aatype, 'atom37_to_atom14')
    mask = atom37_mask(aatype)
    if atom14_data.ndim == 2:
        return mx.take_along_axis(atom14_data, index, axis=1) * mask
    return mx.take_along_axis(atom14_data, index[..., None], axis=1) * mask[:, :, None].astype(atom14_data.dtype)

def concat_rigids(parts: list, axis: int = -1) -> geometry.Rigid3:
    components = lambda rigid: list(rigid.rotation.components()) + [rigid.translation.x, rigid.translation.y, rigid.translation.z]
    joined = [mx.concatenate(group, axis=axis) for group in zip(*[components(part) for part in parts])]
    return geometry.Rigid3(geometry.Rot3(*joined[:9]), geometry.Vec3(*joined[9:]))

def torsion_angles_to_frames(aatype: mx.array, backbone_to_global: geometry.Rigid3, torsion_angles_sin_cos: mx.array) -> geometry.Rigid3:
    #the eight rigid group frames of every residue in the global frame; the chi2 to chi4 frames are chained to the previous group
    default = rows_for(residue_constants.restype_rigid_group_default_frame, aatype, 'default_frames')  #(N, 8, 4, 4)
    default_frames = geometry.Rigid3(geometry.Rot3(*[default[..., row, column] for row in range(3) for column in range(3)]), geometry.Vec3(default[..., 0, 3], default[..., 1, 3], default[..., 2, 3]))
    residues = aatype.shape[0]
    sin_angles = mx.concatenate([mx.zeros([residues, 1]), torsion_angles_sin_cos[..., 0]], axis=-1)
    cos_angles = mx.concatenate([mx.ones([residues, 1]), torsion_angles_sin_cos[..., 1]], axis=-1)
    zeros, ones = mx.zeros_like(sin_angles), mx.ones_like(sin_angles)
    rotations = geometry.Rot3(ones, zeros, zeros, zeros, cos_angles, -sin_angles, zeros, sin_angles, cos_angles)
    all_frames = geometry.Rigid3(default_frames.rotation @ rotations, default_frames.translation)
    chi1 = all_frames[:, 4]
    chi2 = chi1 @ all_frames[:, 5]
    chi3 = chi2 @ all_frames[:, 6]
    chi4 = chi3 @ all_frames[:, 7]
    to_backbone = concat_rigids([all_frames[:, 0:5], chi2[:, None], chi3[:, None], chi4[:, None]], axis=-1)
    return backbone_to_global[:, None] @ to_backbone

def frames_and_literature_positions_to_atom14_pos(aatype: mx.array, all_frames_to_global: geometry.Rigid3) -> geometry.Vec3:
    group_index = rows_for(residue_constants.restype_atom14_to_rigid_group, aatype, 'atom14_to_group')  #(N, 14)
    group_mask = (group_index[..., None] == mx.arange(8)).astype(mx.float32)  #(N, 14, 8)
    atoms_to_global = all_frames_to_global.map(lambda component: mx.sum(component[:, None, :] * group_mask, axis=-1))
    literature = geometry.Vec3.from_array(rows_for(residue_constants.restype_atom14_rigid_group_positions, aatype, 'literature_positions'))
    return atoms_to_global.apply_to_point(literature).scale(rows_for(residue_constants.restype_atom14_mask, aatype, 'residue_atom14_mask'))

def make_backbone_affine(positions: geometry.Vec3, mask: mx.array, aatype: mx.array) -> tuple[geometry.Rigid3, mx.array]:
    #the frame of N, CA, C of every residue, and the mask that all three are present
    a, b, c = (residue_constants.atom_order[name] for name in ('N', 'CA', 'C'))
    return geometry.make_transform_from_reference(positions[:, a], positions[:, b], positions[:, c]), (mask[:, a] * mask[:, b] * mask[:, c]).astype(mx.float32)

def compute_chi_angles(positions: mx.array, mask: mx.array, aatype: mx.array) -> tuple[mx.array, mx.array]:
    #the four chi angles of every residue from [N, 37, 3] positions and the mask of chi angles whose atoms are all present
    residues = aatype.shape[0]
    atom_indices = rows_for(CHI_ATOM_INDICES, aatype, 'chi_atom_indices').reshape(residues, 16)
    atoms = geometry.Vec3.from_array(positions).map(lambda component: mx.take_along_axis(component, atom_indices, axis=1).reshape(residues, 4, 4))
    a, b, c, d = (atoms[..., index] for index in range(4))
    present = mx.prod(mx.take_along_axis(mask, atom_indices, axis=1).reshape(residues, 4, 4), axis=-1)
    return geometry.dihedral_angle(a, b, c, d), rows_for(CHI_ANGLES_MASK, aatype, 'chi_angles_mask') * present.astype(mx.float32)

CHI_PI_PERIODIC = np.array(residue_constants.chi_pi_periodic, dtype=np.float32)

def robust_normalize(vectors: mx.array) -> mx.array:
    #r3.vecs_robust_normalize: the epsilon sits inside the square root, so a zero vector stays zero
    return vectors / mx.sqrt(mx.sum(mx.square(vectors), axis=-1, keepdims=True) + 1e-8)

def atom37_to_torsion_angles(aatype: mx.array, positions: mx.array, mask: mx.array, placeholder_for_undefined: bool = False) -> dict[str, mx.array]:
    #the monomer models' all_atom.atom37_to_torsion_angles for [templates, residues] aatype, [templates, residues, 37, 3] positions and [templates, residues, 37] mask: the 7 torsion angles
    #[pre_omega, phi, psi, chi_1 to chi_4] as (sin, cos), the same shifted by pi where the chi's atom naming is ambiguous, and which angles exist. Unlike the multimer path (chi angles only), the angles
    #the mask says are missing are NOT zeroed here (they come out of the degenerate frames) unless placeholder_for_undefined asks for the (1, 0) placeholder
    aatype = mx.minimum(aatype, 20)
    templates, residues = aatype.shape
    previous_positions = mx.concatenate([mx.zeros((templates, 1, 37, 3)), positions[:, :-1]], axis=1)
    previous_mask = mx.concatenate([mx.zeros((templates, 1, 37)), mask[:, :-1]], axis=1)
    pre_omega_positions = mx.concatenate([previous_positions[:, :, 1:3], positions[:, :, 0:2]], axis=-2)  #previous CA, C and this N, CA
    phi_positions = mx.concatenate([previous_positions[:, :, 2:3], positions[:, :, 0:3]], axis=-2)  #previous C and this N, CA, C
    psi_positions = mx.concatenate([positions[:, :, 0:3], positions[:, :, 4:5]], axis=-2)  #this N, CA, C and O
    pre_omega_mask = mx.prod(previous_mask[:, :, 1:3], axis=-1) * mx.prod(mask[:, :, 0:2], axis=-1)
    phi_mask = previous_mask[:, :, 2] * mx.prod(mask[:, :, 0:3], axis=-1)
    psi_mask = mx.prod(mask[:, :, 0:3], axis=-1) * mask[:, :, 4]
    atom_indices = rows_for(CHI_ATOM_INDICES, aatype, 'chi_atom_indices').reshape(templates, residues, 16)
    chi_positions = mx.take_along_axis(positions, mx.broadcast_to(atom_indices[..., None], (templates, residues, 16, 3)), axis=2).reshape(templates, residues, 4, 4, 3)
    chi_atoms_present = mx.prod(mx.take_along_axis(mask, atom_indices, axis=2).reshape(templates, residues, 4, 4), axis=-1)
    chi_mask = rows_for(CHI_ANGLES_MASK, aatype, 'chi_angles_mask') * chi_atoms_present.astype(mx.float32)
    torsion_positions = mx.concatenate([pre_omega_positions[:, :, None], phi_positions[:, :, None], psi_positions[:, :, None], chi_positions], axis=2)  #[T, N, 7, 4, 3]
    torsion_mask = mx.concatenate([pre_omega_mask[..., None], phi_mask[..., None], psi_mask[..., None], chi_mask], axis=2)
    #a frame from the first three atoms (origin the third, the second on the negative x axis, the first in the xy plane) and the fourth atom in it: its y and z are cos and sin of the angle
    point_on_xy_plane, point_on_neg_x, origin, fourth = (torsion_positions[:, :, :, index] for index in range(4))  #atoms 0, 1, 2 and 3 of each torsion
    e0 = robust_normalize(origin - point_on_neg_x)
    in_plane = point_on_xy_plane - origin
    e1 = robust_normalize(in_plane - mx.sum(in_plane * e0, axis=-1, keepdims=True) * e0)
    e2 = mx.stack([e0[..., 1] * e1[..., 2] - e0[..., 2] * e1[..., 1], e0[..., 2] * e1[..., 0] - e0[..., 0] * e1[..., 2], e0[..., 0] * e1[..., 1] - e0[..., 1] * e1[..., 0]], axis=-1)
    relative_y = mx.sum(e1 * fourth, axis=-1) - mx.sum(e1 * origin, axis=-1)
    relative_z = mx.sum(e2 * fourth, axis=-1) - mx.sum(e2 * origin, axis=-1)
    sin_cos = mx.stack([relative_z, relative_y], axis=-1)
    sin_cos = sin_cos / mx.sqrt(mx.sum(mx.square(sin_cos), axis=-1, keepdims=True) + 1e-8)
    sin_cos = sin_cos * mx.array([1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0])[None, None, :, None]  #psi was computed from the oxygen
    ambiguous = rows_for(CHI_PI_PERIODIC, aatype, 'chi_pi_periodic')
    mirror = mx.concatenate([mx.ones((templates, residues, 3)), 1.0 - 2.0 * ambiguous], axis=-1)
    alt_sin_cos = sin_cos * mirror[..., None]
    if placeholder_for_undefined:
        placeholder = mx.stack([mx.ones(sin_cos.shape[:-1]), mx.zeros(sin_cos.shape[:-1])], axis=-1)
        sin_cos = sin_cos * torsion_mask[..., None] + placeholder * (1 - torsion_mask[..., None])
        alt_sin_cos = alt_sin_cos * torsion_mask[..., None] + placeholder * (1 - torsion_mask[..., None])
    return {'torsion_angles_sin_cos': sin_cos, 'alt_torsion_angles_sin_cos': alt_sin_cos, 'torsion_angles_mask': torsion_mask}
