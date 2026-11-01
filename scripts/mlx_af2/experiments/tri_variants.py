"""Scratch (not part of the backend): is there speed to take in the triangle multiplication (0.6-0.8x of metaljax in the per-op profile)? Variants, forward and forward+backward, bf16, warm, at chosen lengths:
 base      the shipped function
 compiled  mx.compile of the shipped function (elementwise chains fused)
 fastln    layers.layer_norm forced to mx.fast.layer_norm (EXACT variance: a numerics change, timed only to see what it is worth)
 fastln+c  both
 matmul    the einsum replaced by a batched matmul over channels (a,b transposed to channel-major)
 matmul+c  both
Each variant is checked against base on the same inputs (max abs difference, relative to the output's scale); matmul and compiled must match to bf16 rounding."""
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'scripts' / 'mlx_af2'))
sys.path.insert(0, str(REPO / 'tests' / 'mlx_af2'))
import bench_block as B
import profile_block as PB

def main(lengths):
    import mlx.core as mx

    from bindcraft.af.mlx_af2 import gpu_lock, layers, pair_ops, preflight
    from bindcraft.af.mlx_af2.weights import load_params
    preflight.check_disk()
    cfg = B.evoformer_config()
    params_dir = os.path.expanduser(os.environ.get('BINDCRAFT_AF2_PARAMS') or '~/.cache/bindcraft/alphafold')
    original_layer_norm = layers.layer_norm

    def exact_layer_norm(p, x, eps=1e-5, fast_variance=False):
        return original_layer_norm(p, x, eps, False)

    def matmul_triangle(p, left_act, left_mask, cfg_):
        channels = cfg_['num_intermediate_channel']
        mask = left_mask[..., None]
        act = layers.layer_norm(p.scope('left_norm_input'), left_act, fast_variance=True)
        projected = mask * layers.linear(p.scope('projection'), act)
        projected = projected * mx.sigmoid(layers.linear(p.scope('gate'), act))
        a, b = projected[:, :, :channels], projected[:, :, channels:]
        if cfg_['equation'] == 'ikc,jkc->ijc':  #outgoing
            contracted = mx.matmul(a.transpose(2, 0, 1), b.transpose(2, 1, 0)).transpose(1, 2, 0)
        else:  #incoming 'kjc,kic->ijc'
            contracted = mx.matmul(b.transpose(2, 1, 0), a.transpose(2, 0, 1)).transpose(1, 2, 0)
        out = layers.linear(p.scope('output_projection'), layers.layer_norm(p.scope('center_norm'), contracted, fast_variance=True))
        return out * mx.sigmoid(layers.linear(p.scope('gating_linear'), act))

    with gpu_lock.hold():
        mx.set_default_device(mx.gpu)
        p = load_params(B.MODEL, params_dir, dtype=mx.bfloat16).scope('evoformer/evoformer_iteration').layer(B.LAYER)
        for length in lengths:
            _, pair, _, pair_mask, _, pair_cot = (mx.array(array).astype(mx.bfloat16) for array in B.block_inputs(length))
            mx.eval(pair, pair_mask, pair_cot)
            for name in ('triangle_multiplication_outgoing', 'triangle_multiplication_incoming'):
                scope, config = p.scope(name), cfg[name]
                reference = None
                for variant in ('base', 'compiled', 'fastln', 'fastln+c', 'matmul', 'matmul+c'):
                    layers.layer_norm = exact_layer_norm if variant.startswith('fastln') else original_layer_norm
                    body = matmul_triangle if variant.startswith('matmul') else pair_ops.triangle_multiplication
                    function = (lambda q, b=body: b(scope, q, pair_mask, config))
                    if variant.endswith('c') or variant == 'compiled':
                        function = mx.compile(function)
                    output = function(pair)
                    mx.eval(output)
                    if variant == 'base':
                        reference = output.astype(mx.float32)
                    difference = float(mx.max(mx.abs(output.astype(mx.float32) - reference))) / float(mx.max(mx.abs(reference)))
                    forward = PB.timed(lambda: mx.eval(function(pair)), 7)
                    both = PB.timed(lambda: mx.eval(*mx.vjp(function, [pair], [pair_cot])), 7)
                    print(json.dumps({'op': name, 'residues': length, 'variant': variant, 'forward_ms': round(1e3 * forward, 2), 'forward_backward_ms': round(1e3 * both, 2), 'max_rel_diff_vs_base': round(difference, 5)}), flush=True)
                layers.layer_norm = original_layer_norm
            mx.clear_cache()

if __name__ == '__main__':
    main([int(value) for value in sys.argv[1:]] or [250, 400])
