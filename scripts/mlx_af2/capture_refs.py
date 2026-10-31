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

def main(argv=None) -> None:
    argv = sys.argv if argv is None else argv
    root = refs_root(argv)
    names = argv[argv.index('--cases') + 1:] if '--cases' in argv else ['small', 'medium']
    names = [name for name in names if not name.startswith('--') and name != str(root)]
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
