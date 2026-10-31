"""Checkpoint 0 (CPU only, reported, never a stop): the cost of the JAX side of a design step. The model call is answered instantly from memory, so what is timed is the front, the
loss, its gradient, the callbacks and the array handling, for the real AlphaFoldDesignModel on real design states. Usage: JAX_PLATFORMS=cpu python scripts/mlx_af2/bench_seam.py"""
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tests' / 'mlx_af2'))

import jax

import cases
from bindcraft.af.mlx_af2.outputs import output_shapes
from bindcraft.af2 import protein_state_shapes, pad_design_chains, renamed_protein_states, canonical_state_names, canonical_chain_names
from seam_support import BridgedModel, InstantRemote

def padded_length(states) -> int:
    names = canonical_state_names(states)
    padded = pad_design_chains(renamed_protein_states(states, names, canonical_chain_names(states, names)), 32, 0)
    return max(sum(lengths) for _, _, lengths in protein_state_shapes(padded))

def timed(function, repeats: int = 3) -> tuple[float, float]:
    function()  #compile and first run
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return statistics.median(times), min(times)

def main(case_names=('small', 'medium', 'large')) -> list[dict]:
    rows = []
    for case in case_names:
        _, states, _, losses = cases.design_inputs(case)
        model = cases.build_model(BridgedModel, case=case, remote=InstantRemote())
        length = padded_length(states)
        gradient_step, gradient_best = timed(lambda: jax.block_until_ready(model.sequence_gradients(states, losses, model=cases.MULTIMER_MODEL)[2]))
        predict_step, _ = timed(lambda: jax.block_until_ready(next(iter(model.predict(states, model=cases.MULTIMER_MODEL).values())).protein_complex['binder'].atoms))
        shapes = output_shapes(length)
        payload = sum(4 * int(__import__('numpy').prod(shape)) for shape, _ in shapes.values()) / 1e6
        rows.append({'case': case, 'residues': length, 'gradient_step_ms': round(1e3 * gradient_step), 'gradient_best_ms': round(1e3 * gradient_best), 'predict_ms': round(1e3 * predict_step), 'outputs_mb': round(payload, 1)})
        print(rows[-1], flush=True)
    return rows

if __name__ == '__main__':
    main()
