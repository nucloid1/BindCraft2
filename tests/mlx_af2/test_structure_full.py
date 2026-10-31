import jax.numpy as jnp
import mlx.core as mx
import numpy as np
import pytest

import cases
import jax_units
from bindcraft.af.alphafold.model import all_atom_multimer, folding_multimer, geometry as jgeo
from bindcraft.af.mlx_af2 import all_atom, structure
from bindcraft.af.mlx_af2.weights import load_params
from helpers import assert_live, load_ref, rel_err

SM = 'structure_module'

@pytest.fixture(scope='module')
def params():
    return load_params(jax_units.MODEL, cases.params_path(), dtype=mx.float32)

def sm_config():
    return jax_units.configs()[0].heads.structure_module

def test_the_copied_tables_equal_the_jax_modules_tables():
    for name in ('RESTYPE_ATOM37_MASK', 'RESTYPE_ATOM14_MASK', 'RESTYPE_ATOM14_TO_ATOM37', 'RESTYPE_ATOM37_TO_ATOM14', 'CHI_ATOM_INDICES', 'RESTYPE_RIGIDGROUP_BASE_ATOM37_IDX'):
        assert np.array_equal(getattr(all_atom, name), getattr(all_atom_multimer, name)), name

def test_atom14_to_atom37_matches_jax_for_every_residue_type():
    rng = np.random.default_rng(1)
    aatype = np.arange(21).astype(np.int32)
    atom14 = rng.standard_normal((21, 14, 3)).astype(np.float32)
    want = np.asarray(all_atom_multimer.atom14_to_atom37(jnp.asarray(atom14), jnp.asarray(aatype)))
    got = np.asarray(all_atom.atom14_to_atom37(mx.array(atom14), mx.array(aatype)))
    assert assert_live(want) is None and np.array_equal(got, want)
    assert np.array_equal(np.asarray(all_atom.atom37_mask(mx.array(aatype))), np.asarray(all_atom_multimer.get_atom37_mask(jnp.asarray(aatype))))
    assert np.array_equal(np.asarray(all_atom.atom14_mask(mx.array(aatype))), np.asarray(all_atom_multimer.get_atom14_mask(jnp.asarray(aatype))))

@pytest.mark.integration
def test_sidechain_head_matches_the_vendored_one(params):
    rng = np.random.default_rng(2)
    n = 21
    reps = [rng.standard_normal((n, 384)).astype(np.float32) for _ in range(2)]
    aatype = np.arange(n).astype(np.int32) % 20  #every residue type, glycine and proline included
    quaternion = rng.standard_normal((4, n)).astype(np.float32)
    rigid = np.asarray(jgeo.Rigid3Array(jgeo.Rot3Array.from_quaternion(*[jnp.asarray(q) for q in quaternion]), jgeo.Vec3Array(*[jnp.asarray(20 * rng.standard_normal(n).astype(np.float32)) for _ in range(3)])).to_array())
    scope = f'{SM}/fold_iteration/rigid_sidechain'
    def build(c, g):
        module = folding_multimer.MultiRigidSidechain(c.heads.structure_module.sidechain, g, name='rigid_sidechain')
        def run(rigid_array, first, second, aatype):
            out = module(jgeo.Rigid3Array.from_array(rigid_array), [first, second], aatype)
            return {'angles': out['angles_sin_cos'], 'unnormalized': out['unnormalized_angles_sin_cos'], 'atom_pos': out['atom_pos'].to_array(), 'frames': out['frames'].to_array()}
        return run
    want = jax_units.run_unit(build, scope, 'rigid_sidechain', [rigid, reps[0], reps[1], aatype])
    got = structure.sidechains(params.scope(scope), structure.geometry.Rigid3.from_array(mx.array(rigid)), [mx.array(r) for r in reps], mx.array(aatype), sm_config().sidechain.to_dict())
    for name, mlx_name in (('angles', 'angles_sin_cos'), ('unnormalized', 'unnormalized_angles_sin_cos')):
        assert rel_err(np.asarray(got[mlx_name]), want[name]) < 1e-5, name
    assert rel_err(np.asarray(got['atom_pos'].to_array()), want['atom_pos']) < 1e-5 and rel_err(np.asarray(got['frames'].to_array()), want['frames']) < 1e-5

@pytest.mark.integration
def test_structure_module_matches_the_captured_run(params):
    ref = load_ref('small', 'fp32', 0)
    features = {'seq_mask': mx.array(ref['in/seq_mask']), 'aatype': mx.array(ref['in/aatype'])}
    representations = {'single': mx.array(ref['out/representations/single']), 'pair': mx.array(ref['out/representations/pair'])}
    out = structure.structure_module(params.scope(SM), representations, features, sm_config().to_dict())
    pairs = {'final_atom_positions': 'out/structure_module/final_atom_positions', 'final_atom_mask': 'out/structure_module/final_atom_mask', 'final_atom14_positions': 'out/structure_module/final_atom14_positions',
             'final_atom14_mask': 'out/structure_module/final_atom14_mask', 'final_rigids': 'out/structure_module/final_rigids', 'traj': 'out/structure_module/traj'}
    for name, key in pairs.items():
        assert_live(ref[key], key)
        assert rel_err(np.asarray(out[name]), ref[key]) < 1e-3, name
    for name in ('angles_sin_cos', 'unnormalized_angles_sin_cos', 'atom_pos', 'frames'):
        assert rel_err(np.asarray(out['sidechains'][name]), ref[f'out/structure_module/sidechains/{name}']) < 1e-3, name
    assert np.array_equal(np.asarray(out['final_atom_mask']), ref['out/structure_module/final_atom_mask'])  #the mask is exact
    assert np.asarray(out['final_atom_mask']).sum() > 0 and (np.asarray(out['final_atom_positions'])[np.asarray(out['final_atom_mask']) == 0] == 0).all()

@pytest.mark.integration
def test_glycine_and_padded_residues_have_the_vendored_masks_and_zero_positions(params):
    ref = load_ref('small', 'fp32', 0)
    seq_mask = ref['in/seq_mask'].copy()
    seq_mask[-6:] = 0.0  #padded residues
    aatype = ref['in/aatype'].copy()
    aatype[3] = 7  #glycine
    representations = {'single': ref['out/representations/single'], 'pair': ref['out/representations/pair']}
    batch = {'aatype': aatype, 'seq_mask': seq_mask, 'use_dropout': np.asarray(False)}
    build = lambda c, g: (lambda reps, batch: folding_multimer.StructureModule(c.heads.structure_module, g)(reps, batch))
    want = jax_units.run_unit(build, SM, SM, [representations, batch])
    got = structure.structure_module(params.scope(SM), {k: mx.array(v) for k, v in representations.items()}, {'seq_mask': mx.array(seq_mask), 'aatype': mx.array(aatype)}, sm_config().to_dict())
    assert np.array_equal(np.asarray(got['final_atom_mask']), want['final_atom_mask']) and np.array_equal(np.asarray(got['final_atom14_mask']), want['final_atom14_mask'])
    assert not np.asarray(got['final_atom_mask'])[-6:].any() and not np.asarray(got['final_atom_positions'])[-6:].any()
    assert np.asarray(got['final_atom_mask'])[3].sum() == 4  #glycine has four atoms
    assert rel_err(np.asarray(got['final_atom_positions']), want['final_atom_positions']) < 1e-4
