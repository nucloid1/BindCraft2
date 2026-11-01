"""Scratch (not part of the backend): does mx.compile fuse enough of a whole Evoformer block to matter? bf16, warm, layer 10 of model_1_multimer_v3, 2 MSA rows, as bench_block.py.
 base         the shipped block, forward / forward+backward (mx.vjp)
 compiled     mx.compile(block): forward fused; the backward of a compiled function is differentiated from the unfused graph
 vjp-compiled mx.compile of a closure that computes the vjp itself, so the backward's elementwise chains are fused too
Gradients of the compiled closure are checked against base (max abs difference over the largest gradient)."""
import json, os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'scripts' / 'mlx_af2')); sys.path.insert(0, str(REPO / 'tests' / 'mlx_af2'))
import bench_block as B
import profile_block as PB

def main(lengths):
    import mlx.core as mx
    from bindcraft.af.mlx_af2 import evoformer, gpu_lock, preflight
    from bindcraft.af.mlx_af2.weights import load_params
    preflight.check_disk()
    cfg = B.evoformer_config()
    params_dir = os.path.expanduser(os.environ.get('BINDCRAFT_AF2_PARAMS') or '~/.cache/bindcraft/alphafold')
    with gpu_lock.hold():
        mx.set_default_device(mx.gpu)
        layer = load_params(B.MODEL, params_dir, dtype=mx.bfloat16).scope('evoformer/evoformer_iteration').layer(B.LAYER)
        for length in lengths:
            msa, pair, msa_mask, pair_mask, msa_cot, pair_cot = (mx.array(a).astype(mx.bfloat16) for a in B.block_inputs(length))
            mx.eval(msa, pair, msa_mask, pair_mask, msa_cot, pair_cot)
            block = lambda m, q: list(evoformer.evoformer_iteration(layer, m, q, msa_mask, pair_mask, cfg))
            compiled = mx.compile(block)
            def vjp_closure(m, q, cm, cq):
                return mx.vjp(block, [m, q], [cm, cq])[1]
            vjp_compiled = mx.compile(vjp_closure)
            reference = vjp_closure(msa, pair, msa_cot, pair_cot)
            candidate = vjp_compiled(msa, pair, msa_cot, pair_cot)
            mx.eval(reference, candidate)
            diffs = [float(mx.max(mx.abs(c.astype(mx.float32) - r.astype(mx.float32)))) / float(mx.max(mx.abs(r.astype(mx.float32)))) for r, c in zip(reference, candidate)]
            rows = {'base': (PB.timed(lambda: mx.eval(block(msa, pair)), 7), PB.timed(lambda: mx.eval(*vjp_closure(msa, pair, msa_cot, pair_cot)), 7)),
                    'compiled': (PB.timed(lambda: mx.eval(compiled(msa, pair)), 7), PB.timed(lambda: mx.eval(*mx.vjp(compiled, [msa, pair], [msa_cot, pair_cot])[1]), 7)),
                    'vjp-compiled': (None, PB.timed(lambda: mx.eval(*vjp_compiled(msa, pair, msa_cot, pair_cot)), 7))}
            for name, (forward, both) in rows.items():
                print(json.dumps({'residues': length, 'variant': name, 'forward_ms': None if forward is None else round(1e3 * forward, 2), 'forward_backward_ms': round(1e3 * both, 2), 'grad_max_rel_diff_vs_base': [round(d, 5) for d in diffs] if name == 'vjp-compiled' else None}), flush=True)
            mx.clear_cache()

if __name__ == '__main__':
    main([int(v) for v in sys.argv[1:]] or [250, 400])
