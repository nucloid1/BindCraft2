"""Scratch (not part of the backend): where the 400-residue vjp goes. In one MLX process on the GPU, under the shared lock:
 forward single pass; the whole vjp (as the worker runs it); the vjp of embeddings_and_evoformer alone at depth (1+1), (1+4) and (48+4) blocks, so the per-block cost in situ falls out of the slope.
Knobs: --cache-gib (MLX allocator cache limit; the worker uses 2.0, none = MLX's default), --chunk (layers.ATTENTION_CHUNK, patched into every function that bound it as a default), --length."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'scripts' / 'mlx_af2'))
sys.path.insert(0, str(REPO / 'tests' / 'mlx_af2'))
import profile_model as P

def swap() -> str:
    return subprocess.run(['sysctl', '-n', 'vm.swapusage'], capture_output=True, text=True).stdout.strip()

def patch_chunk(chunk):
    from bindcraft.af.mlx_af2 import layers, msa_ops, pair_ops
    layers.ATTENTION_CHUNK = chunk
    for function in (layers.attend, layers.attention, pair_ops.triangle_attention, msa_ops.msa_row_attention_with_pair_bias, msa_ops.msa_column_attention):
        if function.__kwdefaults__ and 'chunk_size' in function.__kwdefaults__:
            function.__kwdefaults__['chunk_size'] = chunk
        else:
            names = function.__code__.co_varnames[:function.__code__.co_argcount]
            defaults = list(function.__defaults__)
            defaults[len(defaults) - (len(names) - names.index('chunk_size'))] = chunk
            function.__defaults__ = tuple(defaults)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--length', type=int, default=400)
    parser.add_argument('--cache-gib', type=float, default=None)
    parser.add_argument('--chunk', type=int, default=256)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--skip-whole', action='store_true')
    args = parser.parse_args()

    import mlx.core as mx
    import numpy as np

    from bindcraft.af.mlx_af2 import embeddings, gpu_lock, preflight, templates
    from bindcraft.af.mlx_af2.model_multimer import MultimerModel
    from bindcraft.af.mlx_af2.weights import load_params
    preflight.check_disk()
    chunk = None if args.chunk == 0 else args.chunk
    patch_chunk(chunk)
    features_np, cfg = P.inputs_from_a_jax_process(args.length)
    params_dir = os.path.expanduser(os.environ.get('BINDCRAFT_AF2_PARAMS') or '~/.cache/bindcraft/alphafold')
    label = {'length': args.length, 'cache_gib': args.cache_gib, 'chunk': args.chunk}
    emit = lambda stage, value, **more: print(json.dumps({**label, 'stage': stage, 'value': round(value, 3), **more}), flush=True)
    with gpu_lock.hold():
        mx.set_default_device(mx.gpu)
        if args.cache_gib is not None:
            mx.set_cache_limit(int(args.cache_gib * 2 ** 30))
        model = MultimerModel(P.MODEL, load_params(P.MODEL, params_dir, dtype=mx.bfloat16), cfg, mx.bfloat16)
        features = {name: mx.array(value) for name, value in features_np.items()}
        key = mx.array(np.asarray([0, 1], np.uint32))
        evo, embedding_cfg = model.params.scope('evoformer'), cfg['embeddings_and_evoformer']
        print('swap before:', swap(), flush=True)
        mx.reset_peak_memory()
        emit('forward single pass (s)', P.timed(lambda: mx.eval(model.forward(features, key, 0)[0]), args.repeats), peak_gib=round(mx.get_peak_memory() / 2 ** 30, 2))
        if not args.skip_whole:
            outputs, _, before = model.forward_with_state(features, key, 1)
            rng = np.random.default_rng(0)
            cot = {name: mx.array(rng.standard_normal(tuple(outputs[name].shape)).astype(np.float32)) for name in outputs if name not in ('structure_module/final_atom_mask', 'predicted_aligned_error/breaks')}
            mx.reset_peak_memory()
            emit('whole vjp (s)', P.timed(lambda: mx.eval(*model.vjp(features, key, 1, cot, ('msa_feat', 'target_feat'), prev=before).values()), args.repeats), peak_gib=round(mx.get_peak_memory() / 2 ** 30, 2))
            print('swap after whole vjp:', swap(), flush=True)
        prev = model.zero_prev(features)
        hook = templates.template_hook(evo, features, embedding_cfg, mx.bfloat16)
        pair_cot = mx.array(np.random.default_rng(1).standard_normal((args.length, args.length, cfg['embeddings_and_evoformer']['pair_channel'])).astype(np.float32))
        times = {}
        for main_blocks, extra_blocks in ((1, 1), (1, 4), (48, 4)):
            def f(msa_feat):
                out = embeddings.embeddings_and_evoformer(evo, {**features, 'msa_feat': msa_feat}, prev, embedding_cfg, dtype=mx.bfloat16, num_block=main_blocks, extra_num_block=extra_blocks, remat=True, template_embedding=hook)
                return [out['pair']]
            mx.reset_peak_memory()
            times[(main_blocks, extra_blocks)] = P.timed(lambda: mx.eval(*mx.vjp(f, [features['msa_feat']], [pair_cot.astype(mx.float32)])[1]), args.repeats)
            emit(f'evoformer vjp {main_blocks}+{extra_blocks} blocks (s)', times[(main_blocks, extra_blocks)], peak_gib=round(mx.get_peak_memory() / 2 ** 30, 2))
        per_extra = (times[(1, 4)] - times[(1, 1)]) / 3
        per_main = (times[(48, 4)] - times[(1, 4)]) / 47
        emit('per extra-MSA block in situ, vjp (ms)', 1e3 * per_extra)
        emit('per main block in situ, vjp (ms)', 1e3 * per_main)
        print('swap after:', swap(), flush=True)

if __name__ == '__main__':
    main()
