"""Probe (not backend code): the production form of the compiled block, with the layer's parameters passed in as a list of arrays (one compiled function serves all 48 layers) and an optional dropout key as an input.
Reported, never a stop: traces per shape (the python body of a compiled function runs once per trace), first-call time per new function and per new shape (compile cost), and warm 48-block sweeps against the current
checkpointed stack: the forward sweep (compiled vs plain), and the reverse sweep that starts from recorded block inputs (compiled per-block vjp closures vs the vjp through mx.checkpoint, which includes its forward).
Main stack only (msa 2 rows x 256 channels, pair 128), bf16, layer parameters of model_1_multimer_v3, as bench_block.py. Under the shared GPU lock.
Usage: python scripts/mlx_af2/experiments/compile_inputs.py [lengths ...]   (default 250 400)"""
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'scripts' / 'mlx_af2'))
sys.path.insert(0, str(REPO / 'tests' / 'mlx_af2'))
import bench_block as B

BLOCKS = 48

def main(lengths):
    import mlx.core as mx

    from bindcraft.af.mlx_af2 import evoformer, gpu_lock, preflight
    from bindcraft.af.mlx_af2.weights import load_params

    class Recorder:
        #serves the real parameters and logs every (name, dtype) the block reads
        def __init__(self, inner, log, prefix=''):
            self.inner, self.log, self.prefix = inner, log, prefix

        def scope(self, prefix):
            return Recorder(self.inner, self.log, f'{self.prefix}{prefix.strip("/")}/')

        def get(self, path, dtype=None):
            dtype = dtype or self.inner.store.dtype
            self.log[(self.prefix + path, dtype)] = None
            return self.inner.get(self.prefix + path, dtype)

        def has(self, path):
            return self.inner.has(self.prefix + path)

    class ArrayParams:
        def __init__(self, arrays, default, prefix=''):
            self.arrays, self.default, self.prefix = arrays, default, prefix

        def scope(self, prefix):
            return ArrayParams(self.arrays, self.default, f'{self.prefix}{prefix.strip("/")}/')

        def get(self, path, dtype=None):
            return self.arrays[(self.prefix + path, dtype or self.default)]

        def has(self, path):
            return any(name == self.prefix + path for name, _ in self.arrays)

    preflight.check_disk()
    cfg = B.evoformer_config()
    params_dir = os.path.expanduser(os.environ.get('BINDCRAFT_AF2_PARAMS') or '~/.cache/bindcraft/alphafold')
    traces = {'fwd': 0, 'bwd': 0}
    emit = lambda **row: print(json.dumps(row), flush=True)
    with gpu_lock.hold():
        mx.set_default_device(mx.gpu)
        stack = load_params(B.MODEL, params_dir, dtype=mx.bfloat16).scope('evoformer/evoformer_iteration')
        default = stack.store.dtype
        for use_key in (False, True):
            #fresh compiled function objects per key mode, so the first call at the first length is a cold compile
            log: dict = {}
            sample = [mx.array(array).astype(mx.bfloat16) for array in B.block_inputs(lengths[0])]
            evoformer.evoformer_iteration(Recorder(stack.layer(0), log), sample[0], sample[1], sample[2], sample[3], cfg, dropout_key=mx.random.key(0) if use_key else None)  #graph only, never evaluated
            spec = list(log)
            weights = [[stack.layer(i).get(name, dtype) for name, dtype in spec] for i in range(BLOCKS)]
            mx.eval(weights)
            emit(stage='discovered', dropout_key=use_key, tensors_per_block=len(spec))

            def block(w, msa, pair, msa_mask, pair_mask, key):
                return list(evoformer.evoformer_iteration(ArrayParams(dict(zip(spec, w)), default), msa, pair, msa_mask, pair_mask, cfg, dropout_key=key))

            def fwd(w, msa, pair, msa_mask, pair_mask, *key):
                traces['fwd'] += 1
                return block(w, msa, pair, msa_mask, pair_mask, key[0] if key else None)

            def bwd(w, msa, pair, ct_msa, ct_pair, msa_mask, pair_mask, *key):
                traces['bwd'] += 1
                return mx.vjp(lambda m, q: block(w, m, q, msa_mask, pair_mask, key[0] if key else None), [msa, pair], [ct_msa, ct_pair])[1]

            fwd_c, bwd_c = mx.compile(fwd), mx.compile(bwd)
            for length in lengths:
                msa, pair, msa_mask, pair_mask, ct_msa, ct_pair = (mx.array(array).astype(mx.bfloat16) for array in B.block_inputs(length))
                keys = list(mx.random.split(mx.random.key(7), BLOCKS)) if use_key else [None] * BLOCKS
                extra = lambda i: (keys[i],) if use_key else ()
                mx.eval(msa, pair, msa_mask, pair_mask, ct_msa, ct_pair, keys if use_key else [])
                label = {'residues': length, 'dropout_key': use_key}

                def forward_sweep():
                    m, q, boundaries = msa, pair, []
                    for i in range(BLOCKS):
                        boundaries.append((m, q))
                        m, q = fwd_c(weights[i], m, q, msa_mask, pair_mask, *extra(i))
                    return m, q, boundaries

                # --- compile cost: the first call of each function at this shape, then the second layer (must not trace again)
                before = dict(traces)
                start = time.perf_counter(); mx.eval(*fwd_c(weights[0], msa, pair, msa_mask, pair_mask, *extra(0))); first_fwd = time.perf_counter() - start
                start = time.perf_counter(); mx.eval(*fwd_c(weights[1], msa, pair, msa_mask, pair_mask, *extra(1))); second_fwd = time.perf_counter() - start
                start = time.perf_counter(); mx.eval(*bwd_c(weights[0], msa, pair, ct_msa, ct_pair, msa_mask, pair_mask, *extra(0))); first_bwd = time.perf_counter() - start
                start = time.perf_counter(); mx.eval(*bwd_c(weights[1], msa, pair, ct_msa, ct_pair, msa_mask, pair_mask, *extra(1))); second_bwd = time.perf_counter() - start
                emit(**label, stage='first calls', fwd_first_s=round(first_fwd, 3), fwd_second_layer_s=round(second_fwd, 3), bwd_first_s=round(first_bwd, 3), bwd_second_layer_s=round(second_bwd, 3),
                     new_fwd_traces=traces['fwd'] - before['fwd'], new_bwd_traces=traces['bwd'] - before['bwd'])

                # --- warm sweeps
                def median(function, repeats=3):
                    function()
                    times = []
                    for _ in range(repeats):
                        start = time.perf_counter(); function(); times.append(time.perf_counter() - start)
                    return statistics.median(times)

                def plain_forward():
                    mx.eval(*evoformer.evoformer_stack(stack, msa, pair, msa_mask, pair_mask, cfg, num_block=BLOCKS, remat=False, dropout_key=mx.random.key(7) if use_key else None))

                def checkpointed_vjp():
                    function = lambda m, q: list(evoformer.evoformer_stack(stack, m, q, msa_mask, pair_mask, cfg, num_block=BLOCKS, remat=True, dropout_key=mx.random.key(7) if use_key else None))
                    mx.eval(*mx.vjp(function, [msa, pair], [ct_msa, ct_pair])[1])

                _, _, boundaries = forward_sweep()
                mx.eval([b for pair_ in boundaries for b in pair_])

                def reverse_sweep():
                    cm, cq = ct_msa, ct_pair
                    for i in reversed(range(BLOCKS)):
                        cm, cq = bwd_c(weights[i], *boundaries[i], cm, cq, msa_mask, pair_mask, *extra(i))
                    mx.eval(cm, cq)
                    return cm, cq

                mx.reset_peak_memory()
                t_plain_fwd, t_compiled_fwd = median(plain_forward), median(lambda: mx.eval(*forward_sweep()[:2]))
                t_checkpointed, t_reverse = median(checkpointed_vjp), median(reverse_sweep)
                emit(**label, stage='sweeps (48 blocks, s)', plain_forward=round(t_plain_fwd, 3), compiled_forward=round(t_compiled_fwd, 3), checkpointed_vjp_with_forward=round(t_checkpointed, 3),
                     compiled_reverse_from_tape=round(t_reverse, 3), compiled_forward_plus_reverse=round(t_compiled_fwd + t_reverse, 3), peak_gib=round(mx.get_peak_memory() / 2 ** 30, 2))

                # --- the gradients agree (both from the same random keys)
                function = lambda m, q: list(evoformer.evoformer_stack(stack, m, q, msa_mask, pair_mask, cfg, num_block=BLOCKS, remat=True, dropout_key=mx.random.key(7) if use_key else None))
                reference = mx.vjp(function, [msa, pair], [ct_msa, ct_pair])[1]
                candidate = reverse_sweep()
                mx.eval(reference, candidate)
                emit(**label, stage='gradient vs checkpointed stack',
                     max_rel_diff=[round(float(mx.max(mx.abs(c.astype(mx.float32) - r.astype(mx.float32))) / mx.max(mx.abs(r.astype(mx.float32)))), 5) for r, c in zip(reference, candidate)])
                mx.clear_cache()
        emit(stage='total traces', **traces)

if __name__ == '__main__':
    main([int(value) for value in sys.argv[1:]] or [250, 400])
