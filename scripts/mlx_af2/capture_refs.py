"""Capture the JAX path's real inputs, outputs, cotangents and gradients at the model-call seam, on the CPU, as the references every MLX parity test is judged against.
For each case and variant (fp32: the runner with bfloat16 off; bf16: the runner exactly as shipped), one call of recycled_alphafold_outputs on the features the repo's own front builds,
its jax.vjp for seeded random cotangents, and the spread of bf16 against fp32. Usage: JAX_PLATFORMS=cpu python scripts/mlx_af2/capture_refs.py [--cases small medium] [--refs PATH]"""
import copy
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
if '--strict-rounding' in sys.argv:  #XLA's CPU compiler keeps float32 between fused bfloat16 ops unless told not to (xla_allow_excess_precision), so the flag has to be set before jax is imported
    os.environ['XLA_FLAGS'] = (os.environ.get('XLA_FLAGS', '') + ' --xla_allow_excess_precision=false').strip()
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tests' / 'mlx_af2'))

import jax
import jax.numpy as jnp
import numpy as np

import cases
from bindcraft.af.alphafold.model import config as af_config, data as af_data, model as af_model
from bindcraft.af.mlx_af2.outputs import OUTPUT_KEYS
from bindcraft.af2 import recycled_alphafold_outputs, resolve_subbatch_size
from helpers import assert_live, cosine, rel_err

NO_COTANGENT = ('structure_module/final_atom_mask', 'predicted_aligned_error/breaks')

def refs_root(argv=None) -> Path:
    argv = sys.argv if argv is None else argv
    if '--refs' in argv:
        return Path(argv[argv.index('--refs') + 1])
    return Path(os.path.expanduser(os.environ.get('BINDCRAFT_MLX_REFS', '~/.cache/bindcraft_mlx_refs')))

def build_runner(model_name: str, bfloat16: bool, attention_backend: str, subbatch_size):
    #the lines of AlphaFoldDesignModel._alphafold_runner, with bfloat16 as the one difference
    config = copy.deepcopy(af_config.model_config(model_name))
    global_config = config.model.global_config
    global_config.use_dgram, global_config.use_remat, global_config.bfloat16 = False, True, bfloat16
    global_config.subbatch_size, global_config.attention_backend, global_config.use_cueq = subbatch_size, attention_backend, False
    config.model.num_recycle = 0
    return af_model.RunModel(config, params=None, use_multimer='multimer' in model_name)

def flatten_floats(nested: dict, prefix: str = '') -> dict:
    flat = {}
    for name, value in nested.items():
        if isinstance(value, dict):
            flat.update(flatten_floats(value, f'{prefix}{name}/'))
        elif jnp.issubdtype(value.dtype, jnp.floating):
            flat[f'{prefix}{name}'] = value
    return flat

def stored(array) -> np.ndarray:
    array = np.asarray(array)
    return array.astype(np.float32) if array.dtype.kind == 'f' else array  #npz cannot hold bfloat16; the original dtype goes in the metadata

def capture_call(case: str, variant: str, num_recycle: int = 1, large_complex_initialization: bool = False, cotangent_seed: int = 0) -> tuple[dict, dict]:
    model = cases.build_model(case=case)
    name = cases.MULTIMER_MODEL
    features = cases.model_inputs(case, dropout=False, large_complex_initialization=large_complex_initialization)
    length = int(features['aatype'].shape[0])
    subbatch_size = resolve_subbatch_size(length, 'auto')
    runner = build_runner(name, variant == 'bf16', model.attention_backend, subbatch_size)
    if variant == 'bf16':
        shipped = model._alphafold_runner(model.model_families[name], subbatch_size)
        assert runner.config.to_dict() == shipped.config.to_dict(), 'the bf16 runner is not the shipped one'
    params = af_data.get_model_haiku_params(model_name=name, data_dir=cases.params_path(), fuse=True)
    key = model.key
    rest = {field: value for field, value in features.items() if field not in ('msa_feat', 'target_feat')}

    @jax.jit
    def forward(msa_feat, target_feat, rest):
        return flatten_floats(recycled_alphafold_outputs(runner, params, key, {**rest, 'msa_feat': msa_feat, 'target_feat': target_feat}, num_recycle))

    start = time.perf_counter()
    outputs, pullback = jax.vjp(lambda m, t: forward(m, t, rest), features['msa_feat'], features['target_feat'])
    rng = np.random.default_rng(cotangent_seed)
    cotangents = {field: (jnp.asarray(rng.standard_normal(value.shape), dtype=value.dtype) if field in OUTPUT_KEYS and field not in NO_COTANGENT else jnp.zeros_like(value)) for field, value in outputs.items()}
    gradient_msa, gradient_target = pullback(cotangents)
    jax.block_until_ready((outputs, gradient_msa, gradient_target))
    seconds = time.perf_counter() - start
    arrays = {f'in/{field}': stored(value) for field, value in features.items() if field != 'prev'}
    arrays.update({f'out/{field}': stored(value) for field, value in outputs.items()})
    arrays.update({f'cot/{field}': stored(cotangents[field]) for field in OUTPUT_KEYS if field not in NO_COTANGENT})
    arrays['grad/msa_feat'], arrays['grad/target_feat'] = stored(gradient_msa), stored(gradient_target)
    meta = {'case': case, 'variant': variant, 'model_name': name, 'num_recycle': num_recycle, 'use_dropout': False, 'large_complex_initialization': large_complex_initialization, 'length': length,
            'subbatch_size': subbatch_size, 'out_dtypes': {field: str(outputs[field].dtype) for field in OUTPUT_KEYS}, 'key': np.asarray(key).tolist(), 'seconds': round(seconds, 1)}
    for field in ('out/' + k for k in OUTPUT_KEYS):
        assert_live(arrays[field], field)
    return arrays, meta

def save(root: Path, case: str, variant: str, index: int, arrays: dict, meta: dict) -> None:
    folder = root / case / variant
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / f'call{index}.npz', meta=np.array(json.dumps(meta)), **arrays)

def record_spread(root: Path, case: str) -> None:
    low, high = (np.load(root / case / variant / 'call0.npz') for variant in ('bf16', 'fp32'))
    entry = {'outputs': {name: rel_err(low[f'out/{name}'], high[f'out/{name}']) for name in OUTPUT_KEYS},
             'gradient_cosine': {name: cosine(low[f'grad/{name}'], high[f'grad/{name}']) for name in ('msa_feat', 'target_feat')}}
    path = root / 'spread.json'
    spread = json.loads(path.read_text()) if path.exists() else {}
    spread[case] = entry
    path.write_text(json.dumps(spread, indent=1))

def position_noise_floor(case: str, epsilon: float = 1e-7, seed: int = 0) -> dict:
    #how far the JAX model's own atoms move when msa_feat is perturbed by a relative epsilon (float32 rounding level): the yardstick for what a port's position error can be called
    model = cases.build_model(case=case)
    features = cases.model_inputs(case)
    length = int(features['aatype'].shape[0])
    runner = build_runner(cases.MULTIMER_MODEL, False, model.attention_backend, resolve_subbatch_size(length, 'auto'))
    params = af_data.get_model_haiku_params(model_name=cases.MULTIMER_MODEL, data_dir=cases.params_path(), fuse=True)
    forward = jax.jit(lambda f: recycled_alphafold_outputs(runner, params, model.key, f, 1)['structure_module'])
    base = forward(features)
    noise = jnp.asarray(np.random.default_rng(seed).standard_normal(features['msa_feat'].shape), jnp.float32)
    moved = forward({**features, 'msa_feat': features['msa_feat'] * (1 + epsilon * noise)})
    live = np.asarray(base['final_atom_mask']) > 0
    deviation = np.linalg.norm(np.asarray(moved['final_atom_positions']) - np.asarray(base['final_atom_positions']), axis=-1)[live]
    return {'epsilon': epsilon, 'max_angstrom': float(deviation.max()), 'mean_angstrom': float(deviation.mean())}

def excess_precision_allowed() -> bool:
    return 'xla_allow_excess_precision=false' not in os.environ.get('XLA_FLAGS', '')

def bf16_ensemble(case: str, count: int = 6, epsilon: float = 4e-3) -> dict:
    #the spread of the shipped bf16 model against fp32 as a RANGE, not one draw: each member nudges msa_feat by a relative epsilon at the bfloat16 rounding scale (2^-8 = 3.9e-3, so the rounding
    #of most values changes; a nudge of 1e-4 changed it for about 3% and gave four near-identical members) and compares the JAX bf16 and fp32 outputs for that same input
    model = cases.build_model(case=case)
    features = cases.model_inputs(case)
    length = int(features['aatype'].shape[0])
    params = af_data.get_model_haiku_params(model_name=cases.MULTIMER_MODEL, data_dir=cases.params_path(), fuse=True)
    forwards = {}
    for label, low in (('fp32', False), ('bf16', True)):
        runner = build_runner(cases.MULTIMER_MODEL, low, model.attention_backend, resolve_subbatch_size(length, 'auto'))
        forwards[label] = jax.jit(lambda f, runner=runner: flatten_floats(recycled_alphafold_outputs(runner, params, model.key, f, 1)))
    errors = {name: [] for name in OUTPUT_KEYS}
    for member in range(count):
        noise = jnp.asarray(np.random.default_rng(100 + member).standard_normal(features['msa_feat'].shape), jnp.float32)
        nudged = {**features, 'msa_feat': features['msa_feat'] * (1 + epsilon * noise)}
        low, high = forwards['bf16'](nudged), forwards['fp32'](nudged)
        for name in OUTPUT_KEYS:
            errors[name].append(rel_err(np.asarray(low[name], np.float32), np.asarray(high[name], np.float32)))
    return {'epsilon': epsilon, 'allow_excess_precision': excess_precision_allowed(), 'errors': errors}

def record_bf16_ensemble(root: Path, case: str, strict: bool = False) -> None:
    #strict (--strict-rounding) rounds to bfloat16 after every op like bfloat16 hardware and like MLX; the default band is flattered by XLA-CPU's excess precision and is kept for comparison
    assert strict == (not excess_precision_allowed()), 'the XLA flag does not match the requested ensemble: pass --strict-rounding on the command line'
    path = root / 'spread.json'
    spread = json.loads(path.read_text())
    spread[case]['bf16_ensemble_strict' if strict else 'bf16_ensemble'] = bf16_ensemble(case)
    path.write_text(json.dumps(spread, indent=1))

def record_noise_floor(root: Path, case: str) -> None:
    path = root / 'spread.json'
    spread = json.loads(path.read_text())
    spread[case]['position_noise_floor'] = position_noise_floor(case)
    path.write_text(json.dumps(spread, indent=1))

def main(argv=None) -> None:
    argv = sys.argv if argv is None else argv
    root = refs_root(argv)
    names = argv[argv.index('--cases') + 1:] if '--cases' in argv else ['small', 'medium']
    names = [name for name in names if not name.startswith('--') and name != str(root)]
    if '--bf16-ensemble' in argv:
        strict = '--strict-rounding' in argv
        for case in names:
            record_bf16_ensemble(root, case, strict)
            ensemble = json.loads((root / 'spread.json').read_text())[case]['bf16_ensemble_strict' if strict else 'bf16_ensemble']['errors']
            print(f'{case}: bf16 vs fp32 over the {"strict-rounding " if strict else ""}ensemble: ' + ', '.join(f'{name.split("/")[-2]}/{name.split("/")[-1]} {min(values):.3g}..{max(values):.3g}' for name, values in ensemble.items() if max(values) > 0), flush=True)
        return
    if '--noise-floor' in argv:
        for case in names:
            record_noise_floor(root, case)
            print(f'{case}: position noise floor {json.loads((root / "spread.json").read_text())[case]["position_noise_floor"]}', flush=True)
        return
    for case in names:
        for variant in ('fp32', 'bf16'):
            arrays, meta = capture_call(case, variant)
            save(root, case, variant, 0, arrays, meta)
            print(f'{case}/{variant}: {meta["length"]} residues, {meta["seconds"]} s, {len(arrays)} arrays', flush=True)
        record_spread(root, case)
        if case == 'small':  #the extra configurations later tasks are judged against: no recycle, three recycles, and the large-complex initial positions
            for index, (num_recycle, large) in enumerate(((0, False), (3, False), (1, True)), start=1):
                arrays, meta = capture_call(case, 'fp32', num_recycle=num_recycle, large_complex_initialization=large)
                save(root, case, 'fp32', index, arrays, meta)
                print(f'{case}/fp32 call{index}: num_recycle={num_recycle} large_complex_initialization={large}, {meta["seconds"]} s', flush=True)

if __name__ == '__main__':
    main()
