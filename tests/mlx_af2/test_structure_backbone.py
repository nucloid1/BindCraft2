import jax.numpy as jnp
import mlx.core as mx
import numpy as np
import pytest

import cases
import jax_units
from bindcraft.af.alphafold.model import folding_multimer, geometry as jgeo
from bindcraft.af.mlx_af2 import geometry, structure
from bindcraft.af.mlx_af2.weights import load_params
from helpers import assert_live, load_ref, rel_err

SM = 'structure_module'
N = 12

@pytest.fixture(scope='module')
def params():
    return load_params(jax_units.MODEL, cases.params_path(), dtype=mx.float32)

def sm_config():
    return jax_units.configs()[0].heads.structure_module

def random_rigid_array(seed, n=N, scale=5.0):
    rng = np.random.default_rng(seed)
    quaternion = rng.standard_normal((4, n)).astype(np.float32)
    rotation = jgeo.Rot3Array.from_quaternion(*[jnp.asarray(q) for q in quaternion])
    translation = (scale * rng.standard_normal((3, n))).astype(np.float32)
    return np.asarray(jgeo.Rigid3Array(rotation, jgeo.Vec3Array(*[jnp.asarray(t) for t in translation])).to_array())

def to_mlx_rigid(array):
    return geometry.Rigid3.from_array(mx.array(array))

def test_geometry_matches_the_jax_classes():
    a, b = random_rigid_array(1), random_rigid_array(2)
    ja, jb = jgeo.Rigid3Array.from_array(jnp.asarray(a)), jgeo.Rigid3Array.from_array(jnp.asarray(b))
    ma, mb = to_mlx_rigid(a), to_mlx_rigid(b)
    assert rel_err(np.asarray((ma @ mb).to_array()), np.asarray((ja @ jb).to_array())) < 1e-6
    assert rel_err(np.asarray(ma.inverse().to_array()), np.asarray(ja.inverse().to_array())) < 1e-6
    points = np.random.default_rng(3).standard_normal((3, N)).astype(np.float32) * 4
    for apply_mlx, apply_jax in ((lambda r, p: r.apply_to_point(p), lambda r, p: r.apply_to_point(p)), (lambda r, p: r.apply_inverse_to_point(p), lambda r, p: r.apply_inverse_to_point(p))):
        got = apply_mlx(ma, geometry.Vec3(*[mx.array(c) for c in points])).to_array()
        want = apply_jax(ja, jgeo.Vec3Array(*[jnp.asarray(c) for c in points])).to_array()
        assert rel_err(np.asarray(got), np.asarray(want)) < 1e-6
    assert rel_err(np.asarray(ma.scale_translation(0.05).to_array()), np.asarray(ja.scale_translation(0.05).to_array())) < 1e-7

def test_quaternion_and_two_vector_rotations_match_including_tiny_inputs():
    rng = np.random.default_rng(4)
    q = rng.standard_normal((4, N)).astype(np.float32)
    q[:, 0] = 0.0  #a zero quaternion: the epsilon clip decides
    got = geometry.Rot3.from_quaternion(*[mx.array(c) for c in q]).to_array()
    want = jgeo.Rot3Array.from_quaternion(*[jnp.asarray(c) for c in q]).to_array()
    assert rel_err(np.asarray(got), np.asarray(want)) < 1e-6
    e0, e1 = rng.standard_normal((3, N)).astype(np.float32), rng.standard_normal((3, N)).astype(np.float32)
    got = geometry.Rot3.from_two_vectors(geometry.Vec3(*[mx.array(c) for c in e0]), geometry.Vec3(*[mx.array(c) for c in e1])).to_array()
    want = jgeo.Rot3Array.from_two_vectors(jgeo.Vec3Array(*[jnp.asarray(c) for c in e0]), jgeo.Vec3Array(*[jnp.asarray(c) for c in e1])).to_array()
    assert rel_err(np.asarray(got), np.asarray(want)) < 1e-5

def test_vec3_norm_clips_to_epsilon_like_jax():
    tiny = np.zeros((3, 4), np.float32)
    tiny[:, 1] = 1e-9
    tiny[:, 2] = 3.0
    got = geometry.Vec3(*[mx.array(c) for c in tiny])
    want = jgeo.Vec3Array(*[jnp.asarray(c) for c in tiny])
    assert np.allclose(np.asarray(got.norm()), np.asarray(want.norm()), rtol=1e-6) and np.allclose(np.asarray(got.normalized().to_array()), np.asarray(want.normalized().to_array()), rtol=1e-6, atol=1e-9)
    assert np.allclose(np.asarray(got.norm(1e-8)), np.asarray(want.norm(1e-8)), rtol=1e-6)

@pytest.mark.integration
def test_invariant_point_attention_matches(params):
    rng = np.random.default_rng(5)
    inputs_1d, inputs_2d = rng.standard_normal((N, 384)).astype(np.float32), rng.standard_normal((N, N, 128)).astype(np.float32)
    mask = np.ones((N, 1), np.float32)
    mask[N - 2:] = 0.0
    rigid = random_rigid_array(6)
    scope = f'{SM}/fold_iteration/invariant_point_attention'
    def build(c, g):
        module = folding_multimer.InvariantPointAttention(c.heads.structure_module, g, name='invariant_point_attention')
        return lambda a, b, m, r: module(a, b, m, jgeo.Rigid3Array.from_array(r))
    want = jax_units.run_unit(build, scope, 'invariant_point_attention', [inputs_1d, inputs_2d, mask, rigid])
    assert_live(want)
    got = np.asarray(structure.invariant_point_attention(params.scope(scope), mx.array(inputs_1d), mx.array(inputs_2d), mx.array(mask), to_mlx_rigid(rigid), sm_config().to_dict()))
    assert rel_err(got, want) < 1e-5

@pytest.mark.integration
def test_fold_iteration_backbone_matches(params):
    rng = np.random.default_rng(7)
    act, initial_act = rng.standard_normal((N, 384)).astype(np.float32), rng.standard_normal((N, 384)).astype(np.float32)
    static_2d = rng.standard_normal((N, N, 128)).astype(np.float32)
    aatype, mask, rigid = rng.integers(0, 20, N).astype(np.int32), np.ones((N, 1), np.float32), random_rigid_array(8)
    scope = f'{SM}/fold_iteration'
    def build(c, g):
        module = folding_multimer.FoldIteration(c.heads.structure_module, g, name='fold_iteration')
        def run(act, rigid_array, aatype, mask, initial_act, static_2d):
            new, _ = module({'act': act, 'rigid': jgeo.Rigid3Array.from_array(rigid_array)}, aatype, mask, True, initial_act, jnp.asarray(False), static_feat_2d=static_2d)
            return new['act'], new['rigid'].to_array()
        return run
    want_act, want_rigid = jax_units.run_unit(build, scope, 'fold_iteration', [act, rigid, aatype, mask, initial_act, static_2d])
    got_act, got_rigid = structure.fold_iteration(params.scope(scope), mx.array(act), to_mlx_rigid(rigid), mx.array(mask), mx.array(static_2d), sm_config().to_dict())
    assert rel_err(np.asarray(got_act), want_act) < 1e-5 and rel_err(np.asarray(got_rigid.to_array()), want_rigid) < 1e-5

def run_vendored_structure_module(ref, with_initial_positions: bool):
    batch = {'aatype': ref['in/aatype'], 'seq_mask': ref['in/seq_mask'], 'use_dropout': np.asarray(False)}
    if with_initial_positions:
        batch['initial_atom_pos'] = ref['in/initial_atom_pos']
    representations = {'single': ref['out/representations/single'], 'pair': ref['out/representations/pair']}
    build = lambda c, g: (lambda reps, batch: folding_multimer.StructureModule(c.heads.structure_module, g)(reps, batch))
    return jax_units.run_unit(build, SM, SM, [representations, batch])

def mlx_backbone(ref, with_initial_positions: bool):
    features = {'seq_mask': mx.array(ref['in/seq_mask'])}
    if with_initial_positions:
        features['initial_atom_pos'] = mx.array(ref['in/initial_atom_pos'])
    representations = {'single': mx.array(ref['out/representations/single']), 'pair': mx.array(ref['out/representations/pair'])}
    cfg = sm_config().to_dict()
    act, trajectory = structure.backbone_trajectory(load_params(jax_units.MODEL, cases.params_path(), dtype=mx.float32).scope(SM), representations, features, cfg)
    return np.asarray(act), np.stack([np.asarray(r.scale_translation(cfg['position_scale']).to_array()) for r in trajectory])

@pytest.mark.integration
def test_backbone_trajectory_matches_the_captured_run():
    ref = load_ref('small', 'fp32', 0)
    act, got = mlx_backbone(ref, False)
    want = ref['out/structure_module/traj']
    assert got.shape == want.shape == (8, 147, 3, 4) and assert_live(want) is None
    assert rel_err(got, want) < 1e-3 and rel_err(act, ref['out/representations/structure_module']) < 1e-3

@pytest.mark.integration
def test_backbone_with_initial_atom_positions_matches_the_vendored_structure_module():
    #the capture's own trajectory cannot be used here: the final recycle pass starts from the previous pass's positions (recycled_alphafold_outputs rewrites initial_atom_pos), which the capture does not hold
    ref = load_ref('small', 'fp32', 3)
    assert 'in/initial_atom_pos' in ref
    want = run_vendored_structure_module(ref, True)
    act, got = mlx_backbone(ref, True)
    assert rel_err(got, want['traj']) < 1e-4 and rel_err(act, want['act']) < 1e-4
    plain_want, (plain_act, plain_got) = run_vendored_structure_module(ref, False), mlx_backbone(ref, False)
    assert rel_err(got, plain_got) > 1e-2  #the initial positions change the result, so the check is not vacuous
    assert rel_err(plain_got, plain_want['traj']) < 1e-4
