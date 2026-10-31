# MLX AlphaFold backend: checkpoints

Measured on the M3 Max (macOS 27.0.1). Every number says how it was taken; nothing here stops the work.

## Checkpoint 0: the JAX side of a design step (2026-10-30, CPU only)

The real `AlphaFoldDesignModel` on real design states, with the model call replaced by the `custom_vjp` bridge and answered instantly from memory (`InstantRemote`, zero gradients,
fixed random outputs of the right shapes). What is timed is everything that stays in JAX under this backend: the front, the 900-line loss and its gradient, the callbacks and the
array handling. Median of three warm runs after the compile, JAX on the CPU, `scripts/mlx_af2/bench_seam.py`.

| case | residues (padded) | `sequence_gradients` step | `predict` | model outputs crossing the boundary per state |
|---|---|---|---|---|
| small (hPDL1 + 30) | 147 | 9 ms | 4 ms | 11.2 MB |
| medium (hIL7RA + 60) | 259 | 19 ms | 7 ms | 34.6 MB |
| large (hIL2R + 30) | 419 | 33 ms | 11 ms | 90.3 MB |

What this does not include: the worker's transfer of those outputs and of the same-sized cotangents (measured in Task 15), the MLX compute, ProteinMPNN and the filters (Task 20). The
README reports a six-step trajectory in about 32 s on metaljax for this machine (README-reported, not measured here, case size not stated), roughly 5 s a step; the full metaljax step
time at these three sizes is not measured yet because the shared GPU lock is held by another run (Ethan's RFD4 job). It will be filled in at the checkpoints that need the GPU.

Phase-2 trigger (Global Constraints of the plan: port the CPU-side JAX if it exceeds 20% of the MLX backend's wall time): at 9 to 33 ms a step, the front and the loss are a fraction
of a percent of a step of several seconds (INFERRED from the README's figure), far below the threshold. ProteinMPNN and the filters are judged in Task 20.
