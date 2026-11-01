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

## Checkpoints 1 and 3 on the GPU (2026-10-31, M3 Max, bf16, MLX 0.32.4 against metaljax 0.11.9 on `JAX_PLATFORMS=metal`; reported, never a stop)

One Evoformer block (layer 10, 2 MSA rows, `scripts/mlx_af2/bench_block.py`, JAX jitted with a warm second call, MLX eager, median of 5 to 7):

| residues | forward MLX / JAX | forward + backward MLX / JAX | MLX peak |
|---|---|---|---|
| 150 | 10.0 / 12.2 ms | 27.4 / 27.1 ms | 1.11 GiB |
| 250 | 31.1 / 31.6 ms | 85.9 / 81.9 ms | 2.51 GiB |
| 400 | 100.8 / 96.9 ms | 278.5 / 497.5 ms | 7.25 GiB |

One design step's model work, forward plus the vjp that follows (`scripts/mlx_af2/bench_grad.py`; MLX through the real worker with a 36 GiB cap, JAX the repo's own `jax.vjp` of `recycled_alphafold_outputs`, one recycle):

| residues | MLX forward + vjp | MLX worker peak | JAX Metal step |
|---|---|---|---|
| 250 | 3.4 + 6.5 = 9.9 s | 5.0 GiB | 12.45 s (1.26x) |
| 400 | 11.2 + 24.1 = 35.3 s | 11.9 to 12.1 GiB | 54.1 s (1.53x) |

Corrected 2026-11-01. This table first read 400 residues: 12.1 + 35.2 = 47.3 s against 83.0 s (1.75x), and 250 residues: 13.5 s for JAX (1.37x). Neither reproduced in a quiet session: MLX through the worker gave a 24.1 s vjp twice (35.31 s per step both times; 9.88 s at 250), in process 22.6 to 24.9 s, and JAX gave 54.29 and 53.94 s at 400 and 12.46 and 12.44 s at 250. The cause of the slow session is not established (the single-block and forward rows of the tables above, re-measured in the quiet session, reproduced within about 2%, so only the whole-step runs were affected). Raw rows: `e2e_results/design_step_requery_mlx.jsonl`, `design_step_requery_jax.jsonl`, `quiet_session_block_and_forward.jsonl` and `vjp_400_experiments.log`.

`precise=True` in the softmax costs nothing measurable (identical within noise, under 1%, in alternating runs at 150, 250 and 400 residues), so it stays.

### Speed headroom: what the investigation found (Ethan asked for it 2026-10-31; run 2026-11-01)

Everything here is measured on the M3 Max in bf16, warm, in one quiet session; the scripts are `scripts/mlx_af2/profile_block.py`, `profile_model.py` and `experiments/` (the experiments are scratch probes, not backend code), the raw rows are in `e2e_results/`. This section is the investigation as it was run: leads A, B and C were then built, with the results in "Speed work: built 2026-11-01" below.

**1. The "unexplained 15 s at 400 residues" was a measurement artifact.** The vjp at 400 residues is 24.1 s through the worker (twice) and 22.6 to 24.9 s in process, against about 19.8 s predicted from the blocks. The 35.2 s did not reproduce (see the correction above). `layers.ATTENTION_CHUNK` 512 instead of 256 gains nothing (24.9 against 23.9 s, and 1.2 GiB more memory), so the chunk size is not a lever.

**2. Where the end-to-end gain comes from.** Not from MLX being faster per op: per op, MLX ties or trails (`e2e_results/profile_block_mlx_vs_jax.jsonl`; the two triangle multiplications, about 40% of a block, run at 0.59 to 0.81x of metaljax, the MSA ops and the pair transition are 1.4 to 4x faster, the triangle attentions tie). It comes from what metaljax pays for the same blocks inside the whole-model graph. One bf16 pass at 250 residues (`e2e_results/stage_profile_250.jsonl`):

| | MLX | metaljax |
|---|---|---|
| one main block, in the model (x48) | 31.2 ms | 53.4 ms |
| one extra-MSA block, in the model (x4) | 30.4 ms | 69.8 ms |
| sum of the nine ops of a block, alone (profile_block) | 35.0 ms | 33.1 ms |
| structure module + heads | 13 ms | 55 ms |
| whole pass | 1674 ms | 2874 ms |

The metaljax rows are not all equally firm. Its per-block figures come from the difference of module variants cut to 1+1, 1+4 and 48+4 blocks (built with `jax_units.configs`: stock attention backend, no remat, no subbatch size), and the structure module + heads row is the whole pass minus the 48+4 variant, two graphs configured differently, so the decomposition is not additive (it even gives a negative 25 ms for the embeddings and templates, which `stage_profile_250.jsonl` shows as is). Only the whole-pass and per-main-block metaljax rows are to be relied on; the conclusion that the structure module and heads are a small share holds either way (under 2% of a pass on metaljax, under 1% on MLX).

MLX's blocks cost the same in the model as alone; metaljax's cost 60% more in the model (and 78% more at 400 residues, whole block forward + backward 488 ms against 274 ms for its ops summed, which reproduces). The cause on the metaljax side was not isolated. The structure module and the heads are under 3% of a pass on either side, and the vjp costs the same with the cotangent on the distogram alone (6.21 s) as on every output (6.25 s), so the backward through the structure module is negligible. The vjp is 6.25 s at 250 residues: the final pass forward again (1.67 s), then 52 blocks of recompute plus backward (52 x 80.5 ms = 4.2 s); the arithmetic fits.

**3. Headroom, ranked by what it could save, with its risk.**
- **A. Compile the Evoformer block (largest, moderate risk).** `mx.compile` of a block cuts its forward by 26% at 250 residues (30.9 to 22.7 ms) and 32% at 400 (100.7 to 68.2 ms). Compiling the function does not touch the backward when its vjp is taken afterwards (85.2 against 84.9 ms, 274.2 against 273.8 ms), but compiling a closure that computes the vjp itself fuses the backward too: block forward + backward 85.2 to 66.6 ms (-22%) at 250 and 274.2 to 204.2 ms (-26%) at 400, with gradients identical to the uncompiled ones (relative difference 0.0 in the test). The backend takes the vjp over `mx.checkpoint`ed blocks, which is the case that gains nothing, so using this means writing the stack's backward as an explicit reverse loop over per-block compiled vjp closures. Not measured: the compile time of a new shape (every new binder length is one), and the memory of the compiled graphs. If the closure captures a layer's parameters (as the probe does), each of the 52 blocks is its own trace for every shape; passing the parameters in as inputs would make it one trace per shape, which is the first thing to check before building.
- **B. Do not repeat the final pass's forward in the vjp (second, same restructuring as A).** The design forward call keeps nothing of its final pass, so the vjp redoes it: 1.67 s of the 6.25 s at 250 residues, about 5.5 s of the 24 s at 400 (22 to 27% of the vjp). Keeping each block's inputs from the forward call (52 blocks of one pair tensor, 41 MB at 400 residues, about 2.2 GiB in all; only the latest call needs it, the design loop's vjp follows its forward) lets the vjp start at the backward.
- Projected together from the block timings, NOT measured, and an upper bound: a design step of about 6 s at 250 residues and 19.5 s at 400 against 9.9 and 35.3 s now, which would be about 2.0x and 2.8x faster than metaljax instead of 1.26x and 1.53x. It assumes the isolated-block savings carry over into the model, and the in-place cost does not quite: a main block's vjp costs 402 to 458 ms in place at 400 residues (`e2e_results/vjp_400_experiments.log`) against about 375 ms predicted from the isolated block.
- **C. The worker's allocator cache limit (small, low risk).** The worker caps MLX's allocator cache at 2 GiB; with MLX's default the 400-residue vjp took 22.6 s instead of 23.9 s and the 48+4-block Evoformer vjp 19.9 s instead of 22.5 s (5 to 12%, one run each, to be repeated before it is believed). The peak is 9.7 GiB against a 36 GiB cap.
- **D. Exact-variance layer norms (the open decision below).** Replacing the triangle multiplication's fast-variance chain with `mx.fast.layer_norm` speeds the uncompiled op by 10% (250) and 13% (400) forward + backward, but a compiled fast-variance forward (3.37 ms at 250) already beats the uncompiled exact one (4.57 ms), so with A the speed argument for exact variance disappears and only the accuracy argument remains.
- **Ruled out:** the attention chunk size; writing the triangle multiplication's einsum as channel-major matmuls (identical, 4.82 against 4.83 ms; MLX's einsum already does that); the structure module and heads; the Neural Engine; softmax precision. **Not tested:** `mx.fast.scaled_dot_product_attention` for the triangle and MSA attentions (its MLX vjp is unfused, and attention already ties metaljax per op).

### Speed work: built 2026-11-01 (Ethan: "build the speed work")

Items A and B above, plus C, are built (plan `docs/superpowers/plans/2026-11-01-mlx-speed-work.md`). Numerics are unchanged: compiled and uncompiled agree (fp32 within 1e-5, bf16 within one bf16 ulp, gradients bitwise equal in the probe), and the model's vjp equals the previous whole-graph vjp to 1e-6 in fp32 with dropout on and off.

What was built:
- **Compiled blocks** (`compiled.py`): on the GPU every Evoformer block runs as one compiled forward and, in the backward, one compiled vjp closure. The block's 85 parameter tensors are passed in as a list, so the 48 layers share one trace per shape (a new shape costs about 0.03 s; the cold start of a worker about 1 to 2 s once). `BINDCRAFT_MLX_COMPILE=0` turns it off; on the CPU it is off (MLX builds fused CPU kernels with a C++ compiler, seconds per kernel).
- **An explicit backward chain.** MLX runs a compiled function INLINE, unfused and retraced on every call, when it is called inside another transformation: a compiled block vjp took 64.5 ms called directly and 84.5 ms (85.0 uncompiled) from a custom function's vjp under `mx.vjp`. So `MultimerModel.vjp` no longer differentiates the whole final pass with one `mx.vjp`: it is `mx.vjp` of the tail (single activations, structure module, heads), the main stack's reverse loop (`evoformer.stack_vjp`, called directly), the extra-MSA stack's, then `mx.vjp` of the front (embeddings, templates). Each block is recomputed from its input in the reverse loop, as remat did.
- **Tapes.** The design forward call (the bridge's custom_vjp forward rule asks for it) keeps each block's input; the worker holds them with the call's recycle state under a byte budget (`BINDCRAFT_MLX_TAPE_GIB`, default a fifth of the cap, oldest dropped first) and releases them after the vjp. The vjp then skips the final pass's Evoformer forward. A vjp whose tapes are gone recomputes them; plain forwards (validation, prediction) and the monomer models keep none.
- **Allocator cache** a quarter of the worker's cap (was 2 GiB): at 400 residues the vjp took 12.83 s with 2 GiB, 11.59 s with 4, 11.21 s with 9 and the same unbounded (which grew to 16 GiB). With a 16 GiB cap the 400-residue step still ran without a guard kill (active plus cached peak 12.4 GiB): MLX releases cached buffers before its own limit, 0.9 of the cap.

Before and after, through the real worker, bf16, two alternating rounds per tree (an export of the pre-change commit against the new code, `scripts/mlx_af2/experiments/before_after.py`, `e2e_results/speed_work_before_after.jsonl`; before the allocator cache change):

| | before | after | change |
|---|---|---|---|
| design step (forward + vjp), 250 residues | 9.77 / 10.37 s | 6.86 / 6.34 s | -34% |
| design step, 400 residues | 38.09 / 36.29 s | 22.42 / 21.05 s | -42% |
| vjp alone, 250 / 400 residues | 6.38 / 6.76, 26.06 / 24.61 s | 4.03 / 3.74, 13.70 / 12.99 s | -40% / -47% |
| multimer forward, one recycle, 400 residues | 14.05 / 11.50 s | 8.65 / 8.02 s | -30% |
| monomer validation forward, 3 recycles, 195 / 259 residues | 4.35 / 3.84, 8.69 / 7.88 s | 3.32 / 3.10, 6.48 / 6.03 s | -20% / -22% |
| worker peak (active + cached), 250 / 400 residues | 5.0 to 5.6 / 11.8 GiB | 5.3 to 5.4 / 11.6 to 12.0 GiB | unchanged |

The allocator cache change then took the 400-residue step from 21.7 to 18.7 s (cap 36 GiB, three steps). On the finished code (`e2e_results/speed_work_final.jsonl`): the design step 6.18 s at 250 residues and
18.84 s at 400 (worker peak 6.8 / 16.6 GiB with the allocator cache counted), the multimer forward 2.55 / 7.84 s, the monomer validation forward 0.67 / 3.20 / 6.27 s at 80 / 195 / 259 residues. The investigation's
projection, 6 s and 19.5 s, was an upper bound; the realized 6.2 s and 18.8 s are at it. Against the quiet metaljax step (12.45 s and 54.1 s), the design step is 2.0x faster at 250 residues and 2.9x at 400.
In real campaigns (`e2e.md`) a gradient round went from 6.92 to 4.48 s at 195 residues (1.98x faster than metaljax, was 1.28x), and every trajectory came out the same as before, seed by seed.

Deferred from the review of this work (the ledger has them): the tape budget bounds the tapes held between calls, not one forward's allocation, so two LARGE states near the cap hold up to a fifth of it more than before;
the per-layer weight slices of the compiled blocks are copies (about 150 MB per resident model in bf16); compiled traces are never evicted (about 0.03 s and a little memory per new length).

## The suite on the GPU, and a gradient finding about the shipped model (2026-10-31)

`BINDCRAFT_MLX_INTEGRATION=1 BINDCRAFT_MLX_TEST_DEVICE=gpu pytest tests/mlx_af2` runs every MLX test on the GPU under the shared lock (the switch is off by default; unset, only gpu-marked tests use the GPU).
Result: 182 passed, 9 skipped, 1 expected failure in 7 minutes. The nine skips exist only in that mode and say why (the clean-state lock tests inherit the session lock; the CPU-default assertion;
the lock-count test). Everything else holds on the GPU at the CPU gates: per-op 1e-5, whole-model 1e-3, the vjp cosine 0.999 on small and medium, the predictor's predictions and gradients.

The expected failure, `test_two_states_in_one_call` (the pair case, two states): the second state's fp32 gradient disagrees between the GPU and the JAX-CPU reference by 5.4% (cosine 0.9987; the target chain
0.9966 once padded and summed). The forwards agree to 1e-7. Against an fp64 CPU run of the same graph:

| gradient of the second state | msa_feat error | target_feat error |
|---|---|---|
| CPU fp32, shipped formula (the reference's behaviour) | 5.3e-2 | 4.3e-2 |
| CPU fp32, exact variance in every layer norm | 7.8e-4 | 6.0e-4 |
| GPU fp32 | 6.6e-4 | 5.0e-4 |

A relative 1e-7 nudge to `msa_feat` moves the CPU gradient by 5.4% (cosine 0.9987), the same size as the GPU-to-CPU gap. So this is not a GPU defect: the cause is haiku's "fast variance" layer norm in the
triangle multiplications (E[x^2] - E[x]^2), which cancels badly when the pair activations have large means (their RMS reaches thousands deep in the stack). JAX has it, the port reproduces it on purpose (parity),
and both CPU paths share the resulting 4-5% gradient noise on this state; the GPU's different reduction order happened to land near the truth. The forward is unaffected. In the shipped bf16 model the same
cancellation sits under bf16's own gradient noise (cosine 0.8 against fp32, see the bf16 spread above), so it is not what limits bf16 designs.

Option, not taken: an exact variance in those two norms would make MLX gradients about 60x more accurate than the shipped fp32 ones, at the price of no longer reproducing the JAX formula (the parity gates would
then need the exact-variance JAX module as their reference). It is Ethan's call; it changes numbers, not shapes or speed.

## The monomer trunk, and an arbitrary value in the shipped templates (2026-10-31, fp32, MLX on the CPU device)

The monomer (pTM) trunk (embeddings, the template pair stack and pointwise attention, the template torsion-angle rows, the extra-MSA stack, 48 blocks, the single activations) against the vendored
`EmbeddingsAndEvoformer` run directly: single 2.7e-7, pair 9.1e-8, first MSA row 4.9e-7 at 48 blocks for `model_1_ptm` (templates on), and 4 / 12 / 24 / 48 blocks all at 1e-7; the template embedding alone
2.6e-7 (the three mask configurations, with a control that the interchain mask changes the result when templates span chains); the torsion angles 6e-8 against the vendored function on the real template.

Against the captured SEAM reference (the jitted model) the template models first showed 7.2e-3 on `single` and 1.4e-2 on the first MSA row (small, `model_1_ptm`), 3.8e-3 to 1.1e-2 on the other three
combinations, while the template-free `model_3_ptm` matched at 1e-5. The cause, found by elimination (float64 run of the same graph: MLX within 3e-5 of it, so not MLX rounding; JAX's own sensitivity to a
1e-7 input nudge 2e-7, so not chaos; vendored module run eagerly against the capture: the same 7e-3, so not the port; stock against chunked attention: bit-identical): ONE entry of 1029 in the template's
torsion-angle features. The undefined pre-omega angle (mask 0) of the first target residue, which follows residues at zero positions, is a frame of three collinear points; the epsilon-regularised normalisation
turns the rounding residue of the arithmetic into a unit vector. JAX eagerly gets (0, 0), and so does MLX; XLA's jit gets (0.7455, 0.6664) and that value is in the reference. Injecting the jitted rows
into the MLX trunk brings all four model and case combinations to 2e-7 to 5e-7.

So the shipped monomer models carry one rounding-dependent, undefined input per target chain start (platform- and fusion-dependent, so the Metal path may carry yet another value), worth 1e-3 to 1.4e-2
relative in `single` and the MSA row, 1e-4 to 2e-4 in `pair`. MLX computes the deterministic value (zeros, like every fully masked residue) and the trunk test feeds it the reference's own jitted rows so the
comparison covers everything else; a test pins that the only disagreement between MLX and jit is on undefined angles. Not done, and Ethan's call if wanted: zero every undefined angle in all backends, which
would make the templates deterministic across platforms (it changes the shipped JAX numbers too).

## Checkpoint 1: one Evoformer block, forward and forward plus backward (speed)

Taken 2026-10-31: see "Checkpoints 1 and 3 on the GPU" above.
