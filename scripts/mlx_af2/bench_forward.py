"""Forward-only time and memory (reported, never a stop): the multimer design forward (one recycle) and the monomer validation forward (three recycles, as validation_recycles defaults), warm, bf16, at chosen
residue counts, on the MLX worker or on the JAX path (JAX_PLATFORMS=metal for metaljax). The features are the small case padded with dead residues (lengths above 147) or cropped (below): the shapes set the cost.
Under the shared GPU lock unless --device cpu. Peak memory: the worker's MLX peak for MLX; the process's peak RSS for JAX (a different measure, which does not include the GPU's buffers).
Usage: python scripts/mlx_af2/bench_forward.py --backend mlx|jax --model multimer|monomer --lengths 80 195 259 [--repeats 3] [--device gpu|cpu]"""
import argparse
import json
import os
import resource
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tests' / 'mlx_af2'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

MODELS = {'multimer': ('model_1_multimer_v3', 1), 'monomer': ('model_1_ptm', 3)}

def features_for(family: str, length: int) -> dict:
    import numpy as np

    import cases
    base = int(cases.model_inputs('small', monomer=family == 'monomer')['aatype'].shape[0])
    features = {name: value for name, value in cases.model_inputs('small', extra_padding=max(length - base, 0), monomer=family == 'monomer').items() if name != 'prev'}
    features = {name: np.asarray(value) for name, value in features.items()}
    if length < base:  #a window of the residues: the same shapes as a shorter complex
        total = features['aatype'].shape[0]
        features = {name: value[tuple(slice(0, length) if dim == total else slice(None) for dim in value.shape)] for name, value in features.items()}
    return features

def timed(function, repeats: int) -> float:
    function()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return statistics.median(times)

def bench_mlx(family, lengths, repeats, device):
    import contextlib

    import numpy as np

    import cases
    import jax_units
    from bindcraft.af.mlx_af2 import gpu_lock, preflight
    from bindcraft.af.mlx_af2.worker import RemoteModel

    name, recycles = MODELS[family]
    os.environ['BINDCRAFT_MLX_DEVICE'] = device
    preflight.check_disk()
    rows = []
    with gpu_lock.hold() if device == 'gpu' else contextlib.nullcontext():
        remote = RemoteModel(cases.params_path(), cap_gib=36.0, dtype='bf16')
        try:
            handle = remote.load(name, jax_units.configs(name, True)[0].to_dict())
            key = np.asarray([0, 1], np.uint32)
            for length in lengths:
                features = features_for(family, length)
                seconds = timed(lambda: remote.forward(handle, key, features, recycles, None), repeats)
                peak = remote.stats()
                rows.append({'backend': 'mlx', 'model': name, 'recycles': recycles, 'residues': length, 'forward_s': round(seconds, 3), 'worker_peak_gib': round(peak.get('mlx', 0), 2)})
                print(json.dumps(rows[-1]), flush=True)
        finally:
            remote.close()
    return rows

def bench_jax(family, lengths, repeats, device):
    import contextlib

    import jax
    import jax.numpy as jnp
    import numpy as np

    import cases
    import capture_refs
    from bindcraft.af.alphafold.model import data as af_data
    from bindcraft.af.mlx_af2 import gpu_lock, preflight
    from bindcraft.af2 import recycled_alphafold_outputs, resolve_subbatch_size

    name, recycles = MODELS[family]
    preflight.check_disk()
    rows = []
    on_cpu = device == 'cpu' or os.environ.get('JAX_PLATFORMS', '') == 'cpu'
    with contextlib.nullcontext() if on_cpu else gpu_lock.hold():
        model = cases.build_model(models=(name,))
        params = af_data.get_model_haiku_params(model_name=name, data_dir=cases.params_path(), fuse=True)
        for length in lengths:
            features = {key: jnp.asarray(value) for key, value in features_for(family, length).items()}
            features['prev'] = {'prev_msa_first_row': jnp.zeros((length, 256)), 'prev_pair': jnp.zeros((length, length, 128)), 'prev_pos': jnp.zeros((length, 37, 3))}
            runner = capture_refs.build_runner(name, True, model.attention_backend, resolve_subbatch_size(length, 'auto'))
            forward = jax.jit(lambda f, runner=runner: capture_refs.flatten_floats(recycled_alphafold_outputs(runner, params, model.key, f, recycles)))
            seconds = timed(lambda: jax.block_until_ready(forward(features)), repeats)
            rows.append({'backend': f'jax:{jax.default_backend()}', 'model': name, 'recycles': recycles, 'residues': length, 'forward_s': round(seconds, 3), 'peak_rss_gib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30, 2)})
            print(json.dumps(rows[-1]), flush=True)
    return rows

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--backend', choices=('mlx', 'jax'), required=True)
    parser.add_argument('--model', choices=tuple(MODELS), required=True)
    parser.add_argument('--lengths', type=int, nargs='+', required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--device', choices=('gpu', 'cpu'), default='gpu')
    arguments = parser.parse_args(argv)
    if arguments.backend == 'mlx':
        os.environ.setdefault('JAX_PLATFORMS', 'cpu')  #the front that builds the features is JAX on the CPU
    return (bench_mlx if arguments.backend == 'mlx' else bench_jax)(arguments.model, arguments.lengths, arguments.repeats, arguments.device)

if __name__ == '__main__':
    main()
