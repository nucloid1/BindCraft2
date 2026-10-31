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

## References and the bf16 spread (2026-10-30)

`scripts/mlx_af2/capture_refs.py` records, for each case and variant, one call of `recycled_alphafold_outputs` on the features the repo's own front builds (JAX on the CPU, dropout off, one
recycle), the `jax.vjp` of every float output for seeded random cotangents, and the spread of the shipped bf16 model against fp32 (refs in `~/.cache/bindcraft_mlx_refs`, not in the repo).
The bf16 runner is checked equal to the one `AlphaFoldDesignModel` builds. Capture time on the CPU: small 34 s fp32 and 50 s bf16, medium 107 s and 141 s.

| output | small: rel. error bf16 vs fp32 | medium |
|---|---|---|
| final atom positions | 0.113 | 0.118 |
| pLDDT logits | 0.032 | 0.021 |
| PAE logits | 0.011 | 0.011 |
| distogram logits | 0.0085 | 0.0083 |
| experimentally-resolved logits | 0.019 | 0.018 |

Gradient cosine, bf16 against fp32, for the sequence-dependent features: `msa_feat` 0.80 (small) and 0.79 (medium), `target_feat` 0.77 and 0.76. These cotangents are random over the
outputs, so a real loss may align better, but the shipped bf16 gradient is demonstrably noisy against fp32, which is the README's warning about bf16 trajectories. The MLX bf16 model is
judged against this spread, never against a tighter figure; the fp32 gates (1e-5 per stage, 1e-3 composite, cosine 0.999) are the exact ones.
