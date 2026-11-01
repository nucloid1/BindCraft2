"""Where the time of one AlphaFold pass goes, stage by stage (reported, never a stop), multimer model, bf16, warm, forward only: the embeddings with the templates, one Evoformer block of the extra-MSA stack and one of
the main stack (from the difference of module variants with 1 and 4 / 1 and 48 blocks), the structure module with the heads (what is left of a whole pass), and the whole pass, on the MLX GPU or on the JAX path
(JAX_PLATFORMS=metal for metaljax). MLX only: the cost of the vjp with the cotangent on all outputs, on the distogram alone (no structure module in the backward) and on the positions alone, and the forward with
the recycle passes. The features are the small case padded with dead residues to --length. Under the shared GPU lock unless --device cpu (a smoke test).
Usage: python scripts/mlx_af2/profile_model.py --backend mlx|jax [--length 250] [--repeats 3] [--device gpu|cpu]"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'tests' / 'mlx_af2'))
sys.path.insert(0, str(HERE))

MODEL = 'model_1_multimer_v3'

def timed(function, repeats: int) -> float:
    function()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return statistics.median(times)

def inputs_from_a_jax_process(length: int):
    #the features and the config come from a short CPU subprocess, so the MLX process stays JAX-free (MLX has been measured slower in a process where JAX has run)
    import numpy as np
    folder = tempfile.mkdtemp()
    code = (f"import json, numpy as np, bench_grad, jax_units; f = bench_grad.padded_features({length}); np.savez('{folder}/f.npz', **f);"
            f"json.dump(jax_units.configs('{MODEL}', True)[0].to_dict(), open('{folder}/c.json', 'w'))")
    subprocess.run([sys.executable, '-c', code], cwd=HERE, env={**os.environ, 'JAX_PLATFORMS': 'cpu', 'PYTHONPATH': f'{HERE}:{HERE.parents[1]}:{HERE.parents[1] / "tests" / "mlx_af2"}'}, check=True, capture_output=True)
    return dict(np.load(f'{folder}/f.npz')), json.load(open(f'{folder}/c.json'))

def profile_mlx(length, repeats, device):
    import contextlib

    import mlx.core as mx
    import numpy as np

    from bindcraft.af.mlx_af2 import embeddings, gpu_lock, preflight, templates
    from bindcraft.af.mlx_af2.model_multimer import MultimerModel
    from bindcraft.af.mlx_af2.weights import load_params

    preflight.check_disk()
    features_np, cfg = inputs_from_a_jax_process(length)
    params_dir = os.path.expanduser(os.environ.get('BINDCRAFT_AF2_PARAMS') or '~/.cache/bindcraft/alphafold')
    rows = []
    emit = lambda stage, ms, note='': (rows.append({'backend': 'mlx', 'residues': length, 'stage': stage, 'ms': round(ms, 1), 'note': note}), print(json.dumps(rows[-1]), flush=True))
    with gpu_lock.hold() if device == 'gpu' else contextlib.nullcontext():
        mx.set_default_device(mx.gpu if device == 'gpu' else mx.cpu)
        model = MultimerModel(MODEL, load_params(MODEL, params_dir, dtype=mx.bfloat16), cfg, mx.bfloat16)
        features = {name: mx.array(value) for name, value in features_np.items()}
        key = mx.array(np.asarray([0, 1], np.uint32))
        embedding_cfg, evo = cfg['embeddings_and_evoformer'], model.params.scope('evoformer')
        prev = model.zero_prev(features)
        hook = templates.template_hook(evo, features, embedding_cfg, mx.bfloat16)

        def variant(main, extra):
            return lambda: mx.eval(embeddings.embeddings_and_evoformer(evo, features, prev, embedding_cfg, dtype=mx.bfloat16, num_block=main, extra_num_block=extra, remat=False, template_embedding=hook))

        t11, t14, t480 = (1e3 * timed(variant(main, extra), repeats) for main, extra in ((1, 1), (1, 4), (48, 4)))
        extra_block, main_block = (t14 - t11) / 3, (t480 - t14) / 47
        embed = t11 - extra_block - main_block
        whole = 1e3 * timed(lambda: mx.eval(model.forward(features, key, 0)[0]), repeats)
        emit('embeddings + templates', embed)
        emit('one extra-MSA block', extra_block, 'x4 in a pass')
        emit('one main Evoformer block', main_block, 'x48 in a pass')
        emit('embeddings and Evoformer (1+1 / 1+4 / 48+4 blocks)', t480, f'{t11:.1f} / {t14:.1f} / {t480:.1f} ms')
        emit('structure module + heads (whole pass minus the above)', whole - t480)
        emit('whole forward pass, no recycle', whole)
        emit('whole forward, one recycle', 1e3 * timed(lambda: mx.eval(model.forward(features, key, 1)[0]), repeats))
        outputs, _, before = model.forward_with_state(features, key, 1)
        rng = np.random.default_rng(0)
        shapes = {name: tuple(outputs[name].shape) for name in ('distogram/logits', 'predicted_lddt/logits', 'predicted_aligned_error/logits', 'structure_module/final_atom_positions', 'experimentally_resolved/logits')}
        cot = {name: mx.array(rng.standard_normal(shape).astype(np.float32)) for name, shape in shapes.items()}
        diff = ('msa_feat', 'target_feat')
        for label, names in (('vjp, cotangent on every output', tuple(shapes)), ('vjp, cotangent on the distogram only (no structure module in the backward)', ('distogram/logits',)),
                             ('vjp, cotangent on the positions only', ('structure_module/final_atom_positions',))):
            chosen = {name: cot[name] for name in names}
            emit(label, 1e3 * timed(lambda: mx.eval(*model.vjp(features, key, 1, chosen, diff, prev=before).values()), repeats), 'final pass recomputed with remat; recycle state cached')
        print(f'peak memory {mx.get_peak_memory() / 2 ** 30:.2f} GiB', flush=True)
    return rows

def profile_jax(length, repeats, device):
    import contextlib

    import haiku as hk
    import jax
    import jax.numpy as jnp
    import numpy as np

    import bench_grad
    import capture_refs
    import cases
    import jax_units
    from bindcraft.af.alphafold.model import data as af_data, modules_multimer
    from bindcraft.af.mlx_af2 import gpu_lock, preflight
    from bindcraft.af2 import recycled_alphafold_outputs, resolve_subbatch_size

    preflight.check_disk()
    rows = []
    emit = lambda stage, ms, note='': (rows.append({'backend': f'jax:{jax.default_backend()}', 'residues': length, 'stage': stage, 'ms': round(ms, 1), 'note': note}), print(json.dumps(rows[-1]), flush=True))
    on_cpu = device == 'cpu' or os.environ.get('JAX_PLATFORMS', '') == 'cpu'
    with contextlib.nullcontext() if on_cpu else gpu_lock.hold():
        features_np = bench_grad.padded_features(length)
        features = {name: jnp.asarray(value) for name, value in features_np.items()}
        prev = {'prev_msa_first_row': jnp.zeros((length, 256)), 'prev_pair': jnp.zeros((length, length, 128)), 'prev_pos': jnp.zeros((length, 37, 3))}

        full_params = jax_units.reference_params('evoformer', 'evoformer', None, MODEL)

        def cut_stacks(main, extra):
            #only the two block stacks are cut (to main and extra blocks); the template stack keeps its own two blocks
            keep = {'evoformer_iteration': main, 'extra_msa_stack': extra}
            return {key: {name: (array[:next(count for marker, count in keep.items() if marker in key)] if any(marker in key for marker in keep) else array) for name, array in tensors.items()} for key, tensors in full_params.items()}

        def variant(main, extra):
            config, global_config = jax_units.configs(MODEL, True)
            config.embeddings_and_evoformer.evoformer_num_block = main
            config.embeddings_and_evoformer.extra_msa_stack_num_block = extra
            params = cut_stacks(main, extra)
            transformed = hk.transform(lambda batch: modules_multimer.EmbeddingsAndEvoformer(config.embeddings_and_evoformer, global_config, name='evoformer')(batch))
            forward = jax.jit(lambda batch: transformed.apply(params, jax.random.PRNGKey(0), batch))
            return lambda: jax.block_until_ready(forward({**features, **prev}))

        t11, t14, t480 = (1e3 * timed(variant(main, extra), repeats) for main, extra in ((1, 1), (1, 4), (48, 4)))
        extra_block, main_block = (t14 - t11) / 3, (t480 - t14) / 47
        embed = t11 - extra_block - main_block
        model = cases.build_model(models=(MODEL,))
        params = af_data.get_model_haiku_params(model_name=MODEL, data_dir=cases.params_path(), fuse=True)
        runner = capture_refs.build_runner(MODEL, True, model.attention_backend, resolve_subbatch_size(length, 'auto'))
        full = {**features, 'prev': prev}
        whole_fn = jax.jit(lambda f: capture_refs.flatten_floats(recycled_alphafold_outputs(runner, params, model.key, f, 0)))
        whole = 1e3 * timed(lambda: jax.block_until_ready(whole_fn(full)), repeats)
        emit('embeddings + templates', embed)
        emit('one extra-MSA block', extra_block, 'x4 in a pass')
        emit('one main Evoformer block', main_block, 'x48 in a pass')
        emit('embeddings and Evoformer (1+1 / 1+4 / 48+4 blocks)', t480, f'{t11:.1f} / {t14:.1f} / {t480:.1f} ms')
        emit('structure module + heads (whole pass minus the above)', whole - t480)
        emit('whole forward pass, no recycle', whole)
    return rows

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--backend', choices=('mlx', 'jax'), required=True)
    parser.add_argument('--length', type=int, default=250)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--device', choices=('gpu', 'cpu'), default='gpu')
    arguments = parser.parse_args(argv)
    return (profile_mlx if arguments.backend == 'mlx' else profile_jax)(arguments.length, arguments.repeats, arguments.device)

if __name__ == '__main__':
    main()
