#Probe (not backend code): lead C repeated on the new code: the 400-residue design step in process (forward with tapes, then the vjp from them), MLX allocator cache limit 2 GiB (the worker's) against MLX's default, alternating, 3 rounds
import os, statistics, sys, time, json
sys.path[:0] = [os.path.expanduser('~/BindCraft2/scripts/mlx_af2'), os.path.expanduser('~/BindCraft2/tests/mlx_af2')]
import numpy as np, mlx.core as mx
import profile_model as P
from bindcraft.af.mlx_af2 import gpu_lock, preflight
from bindcraft.af.mlx_af2.model_multimer import MultimerModel
from bindcraft.af.mlx_af2.outputs import output_shapes
from bindcraft.af.mlx_af2.weights import load_params
preflight.check_disk()
features_np, cfg = P.inputs_from_a_jax_process(400)
with gpu_lock.hold():
    mx.set_default_device(mx.gpu)
    default_limit = mx.set_cache_limit(int(2 * 2 ** 30)); mx.set_cache_limit(default_limit)
    model = MultimerModel(P.MODEL, load_params(P.MODEL, os.path.expanduser('~/.cache/bindcraft/alphafold'), dtype=mx.bfloat16), cfg, mx.bfloat16)
    features = {k: mx.array(v) for k, v in features_np.items()}
    key = mx.array(np.asarray([0, 1], np.uint32))
    rng = np.random.default_rng(0)
    cot = {n: mx.array(rng.standard_normal(s).astype(np.float32)) for n, (s, _) in output_shapes(400).items() if n not in ('structure_module/final_atom_mask', 'predicted_aligned_error/breaks')}
    def step():
        tapes = model.new_tapes()
        outputs, _, before = model.forward_with_state(features, key, 1, tapes=tapes); mx.eval(outputs)
        t = time.perf_counter(); mx.eval(*model.vjp(features, key, 1, cot, ('msa_feat', 'target_feat'), prev=before, tapes=tapes).values()); return time.perf_counter() - t
    step()
    for round_ in range(2):
        for label, limit in [(f'{g} GiB', int(g * 2 ** 30)) for g in (2, 4, 9)] + [('default', default_limit)]:
            mx.set_cache_limit(limit); mx.clear_cache(); step(); mx.reset_peak_memory()
            times = [step() for _ in range(2)]
            print(json.dumps({'round': round_ + 1, 'cache_limit': label, 'vjp_s': round(statistics.median(times), 3), 'peak_active_gib': round(mx.get_peak_memory() / 2 ** 30, 2),
                              'cache_after_gib': round(mx.get_cache_memory() / 2 ** 30, 2)}), flush=True)
    print(json.dumps({'default_cache_limit_gib': round(default_limit / 2 ** 30, 1)}))
