"""Before/after measurement of the speed work (reported, never a stop), the same code against any tree: the bindcraft package and the test helpers come from TREE (default: this checkout), so an export of an older
commit runs the identical measurement. Through the real worker, bf16, warm medians of --repeats, one process per tree, under the shared GPU lock:
  design   the design step as the bridge drives it: forward (retain=True where the tree supports it) then the vjp of that call; plus the first call at each new length (trace and kernel cost)
  multimer the forward alone, one recycle (no retain: a plain forward)
  monomer  the validation forward, model_1_ptm, three recycles
Usage: TREE=/path/to/tree python scripts/mlx_af2/experiments/before_after.py [--design 250 400] [--multimer 250 400] [--monomer 80 195 259] [--repeats 3]"""
import argparse
import inspect
import json
import os
import statistics
import sys
import time
from pathlib import Path

TREE = Path(os.environ.get('TREE') or Path(__file__).resolve().parents[3])
sys.path[:0] = [str(TREE / 'tests' / 'mlx_af2'), str(TREE / 'scripts' / 'mlx_af2'), str(TREE)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--design', type=int, nargs='*', default=[250, 400])
    parser.add_argument('--multimer', type=int, nargs='*', default=[250, 400])
    parser.add_argument('--monomer', type=int, nargs='*', default=[80, 195, 259])
    parser.add_argument('--repeats', type=int, default=3)
    arguments = parser.parse_args()
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    import numpy as np

    import bench_forward
    import bench_grad
    import cases
    import jax_units
    import bindcraft
    from bindcraft.af.mlx_af2 import gpu_lock, preflight
    from bindcraft.af.mlx_af2.outputs import output_shapes
    from bindcraft.af.mlx_af2.worker import RemoteModel

    assert Path(bindcraft.__file__).resolve().is_relative_to(TREE.resolve()), f'bindcraft came from {bindcraft.__file__}, not {TREE}'
    retain = {'retain': True} if 'retain' in inspect.signature(RemoteModel.forward).parameters else {}
    label = {'tree': TREE.name, 'retain': bool(retain)}
    emit = lambda **row: print(json.dumps({**label, **row}), flush=True)
    key = np.asarray([0, 1], np.uint32)
    preflight.check_disk()
    with gpu_lock.hold():
        os.environ['BINDCRAFT_MLX_DEVICE'] = 'gpu'
        remote = RemoteModel(cases.params_path(), cap_gib=36.0, dtype='bf16')
        try:
            handle = remote.load(cases.MULTIMER_MODEL, jax_units.configs(cases.MULTIMER_MODEL, True)[0].to_dict())
            for length in arguments.design:
                features = bench_grad.padded_features(length)
                rng = np.random.default_rng(0)
                cotangents = {name: rng.standard_normal(shape).astype(np.float32) for name, (shape, _) in output_shapes(length).items() if name not in bench_grad.NO_COTANGENT}
                start = time.perf_counter()
                _, call = remote.forward(handle, key, features, 1, None, **retain)
                first_forward = time.perf_counter() - start
                start = time.perf_counter()
                remote.vjp(handle, call, key, features, 1, None, cotangents, bench_grad.DIFF)
                first_vjp = time.perf_counter() - start
                forwards, vjps = [], []
                for _ in range(arguments.repeats):
                    start = time.perf_counter()
                    _, call = remote.forward(handle, key, features, 1, None, **retain)
                    forwards.append(time.perf_counter() - start)
                    start = time.perf_counter()
                    remote.vjp(handle, call, key, features, 1, None, cotangents, bench_grad.DIFF)
                    vjps.append(time.perf_counter() - start)
                stats = remote.stats()
                emit(kind='design', residues=length, forward_s=round(statistics.median(forwards), 3), vjp_s=round(statistics.median(vjps), 3),
                     step_s=round(statistics.median(forwards) + statistics.median(vjps), 3), first_forward_s=round(first_forward, 3), first_vjp_s=round(first_vjp, 3),
                     worker_peak_gib=round(stats.get('mlx', 0.0), 2), tape_hits=stats.get('tape_hits'), tape_misses=stats.get('tape_misses'))
            for family, lengths in (('multimer', arguments.multimer), ('monomer', arguments.monomer)):
                name, recycles = bench_forward.MODELS[family]
                model = handle if family == 'multimer' else remote.load(name, jax_units.configs(name, True)[0].to_dict())
                for length in lengths:
                    features = bench_forward.features_for(family, length)
                    seconds = bench_forward.timed(lambda: remote.forward(model, key, features, recycles, None), arguments.repeats)
                    emit(kind=f'{family} forward', model=name, recycles=recycles, residues=length, forward_s=round(seconds, 3), worker_peak_gib=round(remote.stats().get('mlx', 0.0), 2))
        finally:
            remote.close()

if __name__ == '__main__':
    main()
