"""Where the time of one Evoformer block goes (reported, never a stop): every operation of the block (outer product mean, MSA row and column attention, MSA transition, the two triangle multiplications, the two
triangle attentions, the pair transition) timed on its own, forward and forward plus backward, bf16, warm, on the MLX GPU or on the JAX path (JAX_PLATFORMS=metal for metaljax), at chosen residue counts. Inputs are the
shapes of a real design (2 MSA rows, 256 and 128 channels), layer 10 of model_1_multimer_v3, as in bench_block.py. Under the shared GPU lock unless --device cpu (a smoke test, not a measurement).
Usage: python scripts/mlx_af2/profile_block.py --backend mlx|jax [--lengths 250 400] [--repeats 5] [--dtype bf16|fp32] [--device gpu|cpu]"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tests' / 'mlx_af2'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_block as B

#(name, the inputs it takes from (msa, pair), the shape of its output: msa or pair)
OPS = (('outer_product_mean', ('msa',), 'pair'), ('msa_row_attention_with_pair_bias', ('msa', 'pair'), 'msa'), ('msa_column_attention', ('msa',), 'msa'), ('msa_transition', ('msa',), 'msa'),
       ('triangle_multiplication_outgoing', ('pair',), 'pair'), ('triangle_multiplication_incoming', ('pair',), 'pair'), ('triangle_attention_starting_node', ('pair',), 'pair'),
       ('triangle_attention_ending_node', ('pair',), 'pair'), ('pair_transition', ('pair',), 'pair'))

def timed(function, repeats: int) -> float:
    function()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return statistics.median(times)

def profile_mlx(lengths, dtype, repeats, device):
    import contextlib

    import mlx.core as mx

    from bindcraft.af.mlx_af2 import gpu_lock, layers, msa_ops, pair_ops, preflight
    from bindcraft.af.mlx_af2.weights import load_params

    preflight.check_disk()
    cfg = B.evoformer_config()
    params_dir = os.path.expanduser(os.environ.get('BINDCRAFT_AF2_PARAMS') or '~/.cache/bindcraft/alphafold')
    rows = []
    with gpu_lock.hold() if device == 'gpu' else contextlib.nullcontext():
        mx.set_default_device(mx.gpu if device == 'gpu' else mx.cpu)
        dt = mx.bfloat16 if dtype == 'bf16' else mx.float32
        p = load_params(B.MODEL, params_dir, dtype=dt).scope('evoformer/evoformer_iteration').layer(B.LAYER)
        for length in lengths:
            msa, pair, msa_mask, pair_mask, msa_cot, pair_cot = (mx.array(array).astype(dt) for array in B.block_inputs(length))
            mx.eval(msa, pair, msa_mask, pair_mask, msa_cot, pair_cot)
            functions = {
                'outer_product_mean': lambda m: msa_ops.outer_product_mean(p.scope('outer_product_mean'), m, msa_mask, cfg['outer_product_mean']),
                'msa_row_attention_with_pair_bias': lambda m, q: msa_ops.msa_row_attention_with_pair_bias(p.scope('msa_row_attention_with_pair_bias'), m, msa_mask, q, cfg['msa_row_attention_with_pair_bias']),
                'msa_column_attention': lambda m: msa_ops.msa_column_attention(p.scope('msa_column_attention'), m, msa_mask, cfg['msa_column_attention']),
                'msa_transition': lambda m: layers.transition(p.scope('msa_transition'), m, msa_mask, cfg['msa_transition']),
                'triangle_multiplication_outgoing': lambda q: pair_ops.triangle_multiplication(p.scope('triangle_multiplication_outgoing'), q, pair_mask, cfg['triangle_multiplication_outgoing']),
                'triangle_multiplication_incoming': lambda q: pair_ops.triangle_multiplication(p.scope('triangle_multiplication_incoming'), q, pair_mask, cfg['triangle_multiplication_incoming']),
                'triangle_attention_starting_node': lambda q: pair_ops.triangle_attention(p.scope('triangle_attention_starting_node'), q, pair_mask, cfg['triangle_attention_starting_node']),
                'triangle_attention_ending_node': lambda q: pair_ops.triangle_attention(p.scope('triangle_attention_ending_node'), q, pair_mask, cfg['triangle_attention_ending_node']),
                'pair_transition': lambda q: layers.transition(p.scope('pair_transition'), q, pair_mask, cfg['pair_transition'])}
            for name, kinds, out in OPS:
                inputs = [msa if kind == 'msa' else pair for kind in kinds]
                function, cotangent = functions[name], msa_cot if out == 'msa' else pair_cot
                forward = timed(lambda: mx.eval(function(*inputs)), repeats)
                both = timed(lambda: mx.eval(*mx.vjp(function, inputs, [cotangent])), repeats)
                rows.append({'backend': 'mlx', 'op': name, 'residues': length, 'forward_ms': round(1e3 * forward, 2), 'forward_backward_ms': round(1e3 * both, 2)})
                print(json.dumps(rows[-1]), flush=True)
            mx.clear_cache()
    return rows

def profile_jax(lengths, dtype, repeats, device):
    import contextlib

    import haiku as hk
    import jax
    import jax.numpy as jnp

    import jax_units
    from bindcraft.af.alphafold.model import modules as af_modules
    from bindcraft.af.alphafold.model import utils as af_utils
    from bindcraft.af.mlx_af2 import gpu_lock, preflight

    preflight.check_disk()
    low = dtype == 'bf16'
    on_cpu = device == 'cpu' or os.environ.get('JAX_PLATFORMS', '') == 'cpu'
    scope = 'evoformer/evoformer_iteration'
    rows = []
    with contextlib.nullcontext() if on_cpu else gpu_lock.hold():
        config, global_config = jax_units.configs(B.MODEL, low)
        evo = config.embeddings_and_evoformer.evoformer
        modules = {'outer_product_mean': lambda: af_modules.OuterProductMean(config.embeddings_and_evoformer.evoformer.outer_product_mean, global_config, 128, name='outer_product_mean'),
                   'msa_row_attention_with_pair_bias': lambda: af_modules.MSARowAttentionWithPairBias(evo['msa_row_attention_with_pair_bias'], global_config, name='msa_row_attention_with_pair_bias'),
                   'msa_column_attention': lambda: af_modules.MSAColumnAttention(evo['msa_column_attention'], global_config, name='msa_column_attention'),
                   'msa_transition': lambda: af_modules.Transition(evo['msa_transition'], global_config, name='msa_transition'),
                   'triangle_multiplication_outgoing': lambda: af_modules.TriangleMultiplication(evo['triangle_multiplication_outgoing'], global_config, name='triangle_multiplication_outgoing'),
                   'triangle_multiplication_incoming': lambda: af_modules.TriangleMultiplication(evo['triangle_multiplication_incoming'], global_config, name='triangle_multiplication_incoming'),
                   'triangle_attention_starting_node': lambda: af_modules.TriangleAttention(evo['triangle_attention_starting_node'], global_config, name='triangle_attention_starting_node'),
                   'triangle_attention_ending_node': lambda: af_modules.TriangleAttention(evo['triangle_attention_ending_node'], global_config, name='triangle_attention_ending_node'),
                   'pair_transition': lambda: af_modules.Transition(evo['pair_transition'], global_config, name='pair_transition')}
        for length in lengths:
            cast = (lambda array: jnp.asarray(array).astype(jnp.bfloat16)) if low else jnp.asarray
            msa, pair, msa_mask, pair_mask, msa_cot, pair_cot = (cast(array) for array in B.block_inputs(length))
            for name, kinds, out in OPS:
                masks = [msa_mask if kind == 'msa' else pair_mask for kind in kinds[:1]]
                extra = [pair] if kinds == ('msa', 'pair') else []
                params = jax_units.reference_params(f'{scope}/{name}', name, B.LAYER)

                def apply(first, mask, *rest, name=name):
                    with af_utils.bfloat16_context() if low else contextlib.nullcontext():
                        return modules[name]()(first, mask, *rest)

                transformed = hk.transform(apply)
                inputs = [msa if kind == 'msa' else pair for kind in kinds]
                cotangent = msa_cot if out == 'msa' else pair_cot
                if kinds == ('msa', 'pair'):
                    function = lambda m, q, t=transformed, k=masks[0]: t.apply(params, jax.random.PRNGKey(0), m, k, q)
                else:
                    function = lambda x, t=transformed, k=masks[0]: t.apply(params, jax.random.PRNGKey(0), x, k)
                forward_fn = jax.jit(function)
                backward_fn = jax.jit(lambda *arguments, f=function, c=cotangent: jax.vjp(f, *arguments)[1](c))
                forward = timed(lambda: jax.block_until_ready(forward_fn(*inputs)), repeats)
                both = timed(lambda: jax.block_until_ready(backward_fn(*inputs)), repeats)
                rows.append({'backend': f'jax:{jax.default_backend()}', 'op': name, 'residues': length, 'forward_ms': round(1e3 * forward, 2), 'forward_backward_ms': round(1e3 * both, 2)})
                print(json.dumps(rows[-1]), flush=True)
    return rows

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--backend', choices=('mlx', 'jax'), required=True)
    parser.add_argument('--lengths', type=int, nargs='+', default=[250, 400])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--dtype', choices=('bf16', 'fp32'), default='bf16')
    parser.add_argument('--device', choices=('gpu', 'cpu'), default='gpu')
    arguments = parser.parse_args(argv)
    return (profile_mlx if arguments.backend == 'mlx' else profile_jax)(arguments.lengths, arguments.dtype, arguments.repeats, arguments.device)

if __name__ == '__main__':
    main()
