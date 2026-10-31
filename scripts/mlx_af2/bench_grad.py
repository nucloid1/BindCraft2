"""Checkpoint 3 (reported, never a stop): the cost of one design step's model work, a forward call and the vjp that follows it, at chosen residue counts, with the peak memory the worker's cap must cover.
The features are the small case padded with dead residues to the requested length (cases.model_inputs(extra_padding=...)): the shapes, not the residues, set the cost.
MLX half (default): the real architecture, a persistent worker process reached through RemoteModel, run under the shared GPU lock (one GPU run at a time) with a memory cap and the disk preflight.
JAX half (--jax): the repo's own JAX path on whatever platform JAX is given, e.g. JAX_PLATFORMS=metal for metaljax (jax.vjp of recycled_alphafold_outputs, compile excluded by a warm-up call).
Usage: python scripts/mlx_af2/bench_grad.py --lengths 250 400 [--dtype bf16|fp32] [--repeats 3] [--cap-gib 36] [--device gpu|cpu] [--jax]   (--device cpu is a smoke test, not a measurement)"""
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

DIFF = ('msa_feat', 'target_feat')
NO_COTANGENT = ('structure_module/final_atom_mask', 'predicted_aligned_error/breaks')

def padded_features(length: int):
    import numpy as np

    import cases
    base = cases.model_inputs('small')
    count = int(base['aatype'].shape[0])
    assert length >= count, f'the small case already has {count} residues'
    features = cases.model_inputs('small', extra_padding=length - count)
    assert int(features['aatype'].shape[0]) == length
    return {name: np.asarray(value) for name, value in features.items() if name != 'prev'}

def timed(function, repeats: int) -> float:
    function()  #warm-up: kernel compilation, weights read, allocator growth
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return statistics.median(times)

def bench_mlx(lengths, dtype: str, repeats: int, cap_gib: float, device: str) -> list[dict]:
    import numpy as np

    import cases
    import jax_units
    from bindcraft.af.mlx_af2 import gpu_lock, preflight
    from bindcraft.af.mlx_af2.outputs import OUTPUT_KEYS, output_shapes
    from bindcraft.af.mlx_af2.worker import RemoteModel

    os.environ['BINDCRAFT_MLX_DEVICE'] = device
    preflight.check_disk()
    rows = []
    with gpu_lock.hold() if device == 'gpu' else __import__('contextlib').nullcontext():
        remote = RemoteModel(cases.params_path(), cap_gib=cap_gib, dtype=dtype)
        try:
            config, _ = jax_units.configs(cases.MULTIMER_MODEL, dtype == 'bf16')
            handle = remote.load(cases.MULTIMER_MODEL, config.to_dict())
            key = np.asarray([0, 1], np.uint32)
            for length in lengths:
                features = padded_features(length)
                rng = np.random.default_rng(0)
                cotangents = {name: rng.standard_normal(shape).astype(np.float32) for name, (shape, _) in output_shapes(length).items() if name not in NO_COTANGENT}
                state = {}

                def forward():
                    state['outputs'], state['call'] = remote.forward(handle, key, features, 1, None)

                def backward():
                    remote.vjp(handle, state['call'], key, features, 1, None, cotangents, DIFF)

                forward_s = timed(forward, repeats)
                forward()  #the call id the timed vjp names
                vjp_s = timed(backward, repeats)
                peak = remote.stats()
                rows.append({'backend': 'mlx', 'device': device, 'dtype': dtype, 'residues': length, 'forward_s': round(forward_s, 3), 'vjp_s': round(vjp_s, 3), 'step_s': round(forward_s + vjp_s, 3),
                             'worker_peak_gib': {name: round(value, 2) for name, value in peak.items() if name in ('mlx', 'rss')}, 'vjp_cache_hits': peak['vjp_cache_hits']})
                print(json.dumps(rows[-1]), flush=True)
        finally:
            remote.close()
    return rows

def bench_jax(lengths, dtype: str, repeats: int) -> list[dict]:
    import jax
    import jax.numpy as jnp
    import numpy as np

    import cases
    import capture_refs
    from bindcraft.af.alphafold.model import data as af_data
    from bindcraft.af2 import recycled_alphafold_outputs, resolve_subbatch_size

    rows = []
    model = cases.build_model()
    params = af_data.get_model_haiku_params(model_name=cases.MULTIMER_MODEL, data_dir=cases.params_path(), fuse=True)
    base = int(cases.model_inputs('small')['aatype'].shape[0])
    for length in lengths:
        features = jax.tree.map(jnp.asarray, cases.model_inputs('small', extra_padding=length - base))  #'prev' is a dict of arrays
        runner = capture_refs.build_runner(cases.MULTIMER_MODEL, dtype == 'bf16', model.attention_backend, resolve_subbatch_size(length, 'auto'))
        rest = {name: value for name, value in features.items() if name not in DIFF}

        @jax.jit
        def forward(msa_feat, target_feat):
            return capture_refs.flatten_floats(recycled_alphafold_outputs(runner, params, model.key, {**rest, 'msa_feat': msa_feat, 'target_feat': target_feat}, 1))

        outputs, pullback = jax.vjp(forward, features['msa_feat'], features['target_feat'])
        rng = np.random.default_rng(0)
        cotangents = {name: (jnp.asarray(rng.standard_normal(value.shape), value.dtype) if name in capture_refs.OUTPUT_KEYS and name not in NO_COTANGENT else jnp.zeros_like(value)) for name, value in outputs.items()}

        def step():
            _, pull = jax.vjp(forward, features['msa_feat'], features['target_feat'])
            jax.block_until_ready(pull(cotangents))

        step_s = timed(step, repeats)
        rows.append({'backend': f'jax:{jax.default_backend()}', 'dtype': dtype, 'residues': length, 'step_s': round(step_s, 3), 'peak_rss_gib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30, 2)})
        print(json.dumps(rows[-1]), flush=True)
    return rows

def main(argv=None) -> list[dict]:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--lengths', type=int, nargs='+', default=[250, 400])
    parser.add_argument('--dtype', choices=('bf16', 'fp32'), default='bf16')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--cap-gib', type=float, default=36.0, help="the worker's memory cap (this machine: about 36 GiB usable)")
    parser.add_argument('--device', choices=('gpu', 'cpu'), default='gpu')
    parser.add_argument('--jax', action='store_true', help='time the JAX path instead (set JAX_PLATFORMS=metal for metaljax)')
    arguments = parser.parse_args(argv)
    if not arguments.jax:
        os.environ.setdefault('JAX_PLATFORMS', 'cpu')  #the front that builds the features is JAX on the CPU; the model is MLX in the worker
    if arguments.jax:
        return bench_jax(arguments.lengths, arguments.dtype, arguments.repeats)
    return bench_mlx(arguments.lengths, arguments.dtype, arguments.repeats, arguments.cap_gib, arguments.device)

if __name__ == '__main__':
    main()
