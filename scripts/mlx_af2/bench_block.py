"""Checkpoint 1 (reported, never a stop): one Evoformer block, forward and forward plus backward, at chosen residue counts, with its memory. One backend per invocation, because MLX has been measured
slower in a process where another framework has done work: the MLX half imports no JAX (the block's config comes from a short CPU subprocess), the JAX half is the repo's vendored haiku block jitted on whatever
platform JAX is given (JAX_PLATFORMS=metal for metaljax; the second call in the process is the timed one, the first compiles). Both run under the shared GPU lock unless --device cpu (a smoke test, not a measurement).
Inputs are the shapes of a real design (2 MSA rows, 256 and 128 channels), random activations of unit-ish scale, bfloat16 like the shipped model, layer 10 of model_1_multimer_v3.
Usage: python scripts/mlx_af2/bench_block.py [--backend mlx|jax] [--lengths 150 250 400] [--dtype bf16|fp32] [--repeats 5] [--device gpu|cpu]"""
import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'tests' / 'mlx_af2'))
sys.path.insert(0, str(HERE))

ROWS, MSA_CHANNELS, PAIR_CHANNELS, LAYER = 2, 256, 128, 10
MODEL = 'model_1_multimer_v3'

def block_inputs(length: int):
    import numpy as np
    rng = np.random.default_rng(0)
    return ((0.5 * rng.standard_normal((ROWS, length, MSA_CHANNELS))).astype(np.float32), (0.5 * rng.standard_normal((length, length, PAIR_CHANNELS))).astype(np.float32),
            np.ones((ROWS, length), np.float32), np.ones((length, length), np.float32),
            rng.standard_normal((ROWS, length, MSA_CHANNELS)).astype(np.float32), rng.standard_normal((length, length, PAIR_CHANNELS)).astype(np.float32))

def timed(function, repeats: int) -> float:
    function()  #warm-up: kernel compilation, allocator growth
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return statistics.median(times)

def evoformer_config() -> dict:
    code = 'import json, jax_units; print(json.dumps(jax_units.configs()[0].to_dict()["embeddings_and_evoformer"]["evoformer"]))'
    result = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True, cwd=HERE, env={**os.environ, 'JAX_PLATFORMS': 'cpu', 'PYTHONPATH': f'{HERE}:{HERE.parents[1]}'}, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])

def bench_mlx(lengths, dtype: str, repeats: int, device: str) -> list[dict]:
    import contextlib

    import mlx.core as mx
    import numpy as np

    from bindcraft.af.mlx_af2 import evoformer, gpu_lock, preflight
    from bindcraft.af.mlx_af2.weights import load_params

    preflight.check_disk()
    cfg = evoformer_config()
    params_dir = os.path.expanduser(os.environ.get('BINDCRAFT_AF2_PARAMS') or '~/.cache/bindcraft/alphafold')
    rows = []
    with gpu_lock.hold() if device == 'gpu' else contextlib.nullcontext():
        mx.set_default_device(mx.gpu if device == 'gpu' else mx.cpu)
        dt = mx.bfloat16 if dtype == 'bf16' else mx.float32
        layer = load_params(MODEL, params_dir, dtype=dt).scope('evoformer/evoformer_iteration').layer(LAYER)
        for length in lengths:
            msa, pair, msa_mask, pair_mask, msa_cotangent, pair_cotangent = (mx.array(array).astype(dt) for array in block_inputs(length))
            mx.eval(msa, pair, msa_mask, pair_mask, msa_cotangent, pair_cotangent)

            def block(m, p):
                return evoformer.evoformer_iteration(layer, m, p, msa_mask, pair_mask, cfg)

            def forward():
                mx.eval(block(msa, pair))

            def forward_backward():
                outputs, gradients = mx.vjp(block, [msa, pair], [msa_cotangent, pair_cotangent])
                mx.eval(outputs, gradients)

            forward_s = timed(forward, repeats)
            mx.reset_peak_memory()
            both_s = timed(forward_backward, repeats)
            rows.append({'backend': 'mlx', 'device': device, 'dtype': dtype, 'residues': length, 'forward_ms': round(1e3 * forward_s, 1), 'forward_backward_ms': round(1e3 * both_s, 1),
                         'peak_gib': round(mx.get_peak_memory() / 2 ** 30, 2)})
            print(json.dumps(rows[-1]), flush=True)
            del msa, pair, msa_mask, pair_mask, msa_cotangent, pair_cotangent
            mx.clear_cache()
    return rows

def bench_jax(lengths, dtype: str, repeats: int, device: str) -> list[dict]:
    import contextlib

    import haiku as hk
    import jax
    import jax.numpy as jnp
    import numpy as np

    import jax_units
    from bindcraft.af.alphafold.model import modules as af_modules
    from bindcraft.af.alphafold.model import utils as af_utils
    from bindcraft.af.mlx_af2 import gpu_lock, preflight

    preflight.check_disk()
    low = dtype == 'bf16'
    on_cpu = device == 'cpu' or os.environ.get('JAX_PLATFORMS', '') == 'cpu'  #decided from the environment: asking JAX which backend it has would start it, on the GPU, before the lock is held
    scope = 'evoformer/evoformer_iteration'
    rows = []
    with contextlib.nullcontext() if on_cpu else gpu_lock.hold():
        config, global_config = jax_units.configs(MODEL, low)

        def apply(activations, masks):
            with af_utils.bfloat16_context() if low else contextlib.nullcontext():
                return af_modules.EvoformerIteration(config.embeddings_and_evoformer.evoformer, global_config, is_extra_msa=False, name='evoformer_iteration')(activations=activations, masks=masks,
                                                                                                                                                         use_dropout=jnp.asarray(False), safe_key=None)

        params = jax_units.reference_params(scope, 'evoformer_iteration', LAYER)
        transformed = hk.transform(apply)
        for length in lengths:
            cast = (lambda array: jnp.asarray(array).astype(jnp.bfloat16)) if low else jnp.asarray
            msa, pair, msa_mask, pair_mask, msa_cotangent, pair_cotangent = (cast(array) for array in block_inputs(length))

            @jax.jit
            def forward(msa, pair):
                return transformed.apply(params, jax.random.PRNGKey(0), {'msa': msa, 'pair': pair}, {'msa': msa_mask, 'pair': pair_mask})

            @jax.jit
            def forward_backward(msa, pair):
                outputs, pullback = jax.vjp(lambda m, p: forward(m, p), msa, pair)
                return outputs, pullback({'msa': msa_cotangent, 'pair': pair_cotangent})

            forward_s = timed(lambda: jax.block_until_ready(forward(msa, pair)), repeats)
            both_s = timed(lambda: jax.block_until_ready(forward_backward(msa, pair)), repeats)
            rows.append({'backend': f'jax:{jax.default_backend()}', 'dtype': dtype, 'residues': length, 'forward_ms': round(1e3 * forward_s, 1), 'forward_backward_ms': round(1e3 * both_s, 1),
                         'peak_rss_gib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30, 2)})
            print(json.dumps(rows[-1]), flush=True)
    return rows

def main(argv=None) -> list[dict]:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--backend', choices=('mlx', 'jax'), default='mlx')
    parser.add_argument('--lengths', type=int, nargs='+', default=[150, 250, 400])
    parser.add_argument('--dtype', choices=('bf16', 'fp32'), default='bf16')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--device', choices=('gpu', 'cpu'), default='gpu', help='cpu is a smoke test; for the JAX half the platform comes from JAX_PLATFORMS')
    arguments = parser.parse_args(argv)
    if arguments.backend == 'mlx':
        return bench_mlx(arguments.lengths, arguments.dtype, arguments.repeats, arguments.device)
    return bench_jax(arguments.lengths, arguments.dtype, arguments.repeats, arguments.device)

if __name__ == '__main__':
    main()
