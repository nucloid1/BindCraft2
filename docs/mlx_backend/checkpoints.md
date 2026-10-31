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

## Evoformer stack accuracy (2026-10-30, fp32, MLX on the CPU device against the vendored JAX block chained layer by layer)

Random activations of unit-ish scale (3 MSA rows, 10 residues) through the real weights of `model_1_multimer_v3`; the pair activations grow to an RMS near 3,000 by block 48, so this is not a benign regime.

| blocks | MSA rel. error | pair rel. error |
|---|---|---|
| 1 | 5.7e-7 | 2.4e-7 |
| 12 | 4.3e-7 | 1.5e-7 |
| 48 | 4.6e-7 | 1.1e-7 |

No growth with depth. The plan's gates are 1e-5 per block and 1e-3 for a stack.

## bf16 end to end: what the JAX reference is, and where MLX stands (2026-10-31, CPU, MLX on the CPU device)

**The default JAX-CPU "bf16" model is more accurate than bfloat16 arithmetic.** XLA's CPU compiler runs with `xla_allow_excess_precision=true`: between fused ops it keeps float32 and drops the round
trip through bfloat16. The first end-to-end comparison (MLX against the default JAX bf16 band) showed the pair-derived outputs 1.3x to 1.5x worse than JAX (PAE logits 0.0138 against 0.0106,
distogram 0.0129 against 0.0085). Localizing it, on the REAL input of the 48-block stack (captured from an MLX forward of the small case: 2 MSA rows, 147 residues, pair channels at RMS up to 170
going in), with the same fp32-weight distogram head applied to each pair:

| blocks | pair error: JAX default / JAX strict / MLX | distogram error through the head: JAX default / JAX strict / MLX |
|---|---|---|
| 1 | 3.50e-3 / 3.89e-3 / 3.89e-3 | 3.13e-3 / 4.96e-3 / 4.93e-3 |
| 4 | 4.21e-3 / 4.90e-3 / 4.88e-3 | 3.05e-3 / 3.92e-3 / 3.88e-3 |
| 12 | 8.10e-3 / 9.01e-3 / 9.01e-3 | 5.11e-3 / 6.34e-3 / 6.49e-3 |
| 24 | 9.86e-3 / 1.18e-2 / 1.18e-2 | 5.03e-3 / 8.52e-3 / 8.53e-3 |
| 48 | 1.27e-2 / 1.40e-2 / 1.40e-2 | 8.54e-3 / 1.28e-2 / 1.29e-2 |

"Strict" is JAX with `XLA_FLAGS=--xla_allow_excess_precision=false`, i.e. rounding to bfloat16 after every op as bfloat16 hardware does. MLX equals strict JAX within 1% on the pair at every depth (MLX's
MSA error is lower: 2.26e-2 against 2.50e-2 at 48 blocks). What was ruled out on the way: the heads (the same fp32 head on each trunk's pair reproduces each full-model logit error exactly: the error
is the trunk's), every op at real layer-0 inputs (MLX error / JAX error: outer product 1.00, row attention 0.24, column attention 0.97, MSA transition 1.01, triangle multiplication 0.98 and 0.90, triangle
attention 0.50 and 0.68, pair transition 1.00), the dtypes of every update (bfloat16 on both sides), bf16 casts, adds, sums, means, exp and products (bit-identical between MLX and JAX over 2.8M real values;
sigmoid alone differs, in 25% of values by up to 2 ulp, MLX the more accurate), and a chain of the block's ops with eager bfloat16 residual adds (JAX's own ops 3.888e-3, MLX's 3.893e-3).

The gate (`test_bf16_within_the_jax_spread`) is therefore judged against the strict ensemble (6 members, `capture_refs.py --bf16-ensemble --strict-rounding`), and prints the default band:

| output | MLX small / medium | JAX strict small | JAX strict medium | JAX default small | JAX default medium |
|---|---|---|---|---|---|
| final atom positions | 0.129 / 0.042 | 0.086..0.143 | 0.105..0.223 | 0.046..0.109 | 0.069..0.222 |
| pLDDT logits | 0.090 / 0.013 | 0.030..0.077 | 0.019..0.031 | 0.016..0.031 | 0.015..0.030 |
| PAE logits | 0.0138 / 0.0136 | 0.0136..0.0137 | 0.0137..0.0138 | 0.0105..0.0108 | 0.0106..0.0107 |
| distogram logits | 0.0129 / 0.0129 | 0.0128..0.0129 | 0.0130 | 0.0085 | 0.0082..0.0083 |
| experimentally-resolved logits | 0.017 / 0.015 | 0.018..0.022 | 0.019..0.020 | 0.018..0.020 | 0.018..0.020 |

Gates: geometric-mean ratio over the five outputs <= 1.5, each output <= 3x the strict maximum, and (tight, because their bands are under 1% wide) PAE and distogram ratios <= 1.10. A control: a MLX
layer norm computed in bfloat16 instead of float32 passes the coarse gates (ratios 1.05 on PAE and distogram), so the coarse gates only catch gross bf16 errors; the same PAE ratio against the flattered
default band is 1.30, which the tight gate would reject.

Two unmeasured things. (1) Whether the shipped Apple GPU path (metaljax) rounds like strict JAX or like default JAX-CPU: it needs the GPU lock. MLX on the GPU rounds to bfloat16 at every op, as it does on the
CPU. (2) If more accuracy than strict bf16 is wanted, carrying the residual stream in float32 would give it at a memory and speed cost; nothing of the kind is built. `precise=True` in the softmax makes no
difference on the real stack (pair error 1.399e-2 either way, 48 blocks) and is kept as accuracy insurance for peaked attention; its speed is to be read at Checkpoint 1 and it is the first thing to drop
if it costs more than 2%.

## Checkpoint 1: one Evoformer block, forward and forward plus backward (speed)

NOT TAKEN YET: it needs the GPU for both MLX and the metaljax reference, and the shared lock is held by Ethan's RFD4 job. To be filled in when the lock is free.
