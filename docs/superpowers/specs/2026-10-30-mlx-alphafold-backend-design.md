# BindCraft2: an MLX backend for the AlphaFold model

Status: design, awaiting Ethan's approval. Claims are tagged VERIFIED (a command ran and its output was read), INFERRED (code reading or
arithmetic, not measured) or UNKNOWN.

## Intent and why

This fork already runs AlphaFold 2 on the Apple GPU through the third-party `metaljax` JAX plugin (commit b4556f7; the README measures about 3.5x
over the same code on the CPU). Ethan wants the Mac path to run on Metal through MLX instead of through that plugin, for speed and so the Mac path
no longer depends on a plugin that is itself experimental. Earlier MLX ports of ESMFold2, Mol-1 and RFD4-Proteina reached 1.6 to 2.3x over
torch on MPS; none was measured against a JAX Metal path, so the gain here is a hypothesis, not a promise.

## Decided by Ethan

- **Scope: the AlphaFold model only.** Its forward pass and its gradient run in MLX in a worker process. The design loop, the 907-line loss
  module (`bindcraft/loss.py`), the sequence optimizer, ProteinMPNN and the filters stay as they are.
- **MLX is a new backend added beside the existing ones.** CUDA, ROCm, oneAPI and metaljax keep working and are the parity reference.
- **bf16, as today.** AlphaFold runs in bf16 now (`global_config.bfloat16 = True`); the MLX model mirrors it. fp32 is kept as an option for parity
  testing.
- **No stop rule.** Speed is measured and reported at every checkpoint but never stops the work (Ethan wants JAX and the plugin off the Mac path
  regardless).
- **Neural Engine, "if possible" (asked twice).** Not part of phase 1, which is GPU-only. Section "Neural Engine" records what exists, what is
  unknown, and two throwaway spikes that would settle it; they start only on Ethan's separate yes.

Standing rules from earlier work: nothing in a repo this work does not own is modified; one GPU run at a time (the shared lock
`~/.cache/esmfold2_mlx/gpu.lock`); disk under 90% unless Ethan opts in (`..._MIN_FREE_GB`); never a silent fallback to another backend.

## What exists today (the facts the design rests on)

- **Where AlphaFold is called (VERIFIED, `bindcraft/af2.py`).** `AlphaFoldDesignModel` implements `DifferentiableProteinPredictor`
  (`predict(...)` and `sequence_gradients(...)`, `bindcraft/prediction.py`) and is built in `bindcraft/campaign.py:262,265` for the design models and
  the validation models. The vendored model is `bindcraft/af/alphafold/` (11.6k lines of JAX and haiku).
- **The gradient seam (VERIFIED).** `sequence_design_loss` is one `jit(value_and_grad(..., argnums=2))` over three parts: (1) sequence logits to model
  input features (`prepare_design_sequence_features`, `alphafold_input_features`: a one-row `msa_feat` of 49 channels, `target_feat`, template
  features of the fixed target, offsets, chain ids, a dropout flag); (2) the model (`recycled_alphafold_outputs`); (3) metrics and the design loss
  from the outputs. Only part 2 is AlphaFold. Every state of a design (complex, binder alone, extra targets) is one model call with its own shapes.
- **Recycling (VERIFIED).** The model config has `num_recycle = 0`; recycles are run by hand and every pass except the last is under
  `stop_gradient`, so the gradient flows through one pass only. `design_recycles` is 1 and `validation_recycles` 3 by default.
- **What the loss reads from the model (VERIFIED).** `structure_module.final_atom_positions` and `final_atom_mask`, `predicted_lddt.logits`,
  `predicted_aligned_error.logits` and `breaks`, `distogram.logits`, `experimentally_resolved.logits`, and `prev` (the recycle state).
- **The model (VERIFIED, `config.py`).** 48 Evoformer blocks (MSA width 256, pair width 128), a 4-block extra-MSA stack (one extra row), a template
  stack for the fixed target, an 8-layer structure module with invariant point attention, and the heads. Two families are used: `multimer_v3`
  (five models) and the monomer pTM models (`model_1_ptm`, `model_2_ptm`), selected by `select_design_and_validation_models`.
- **Precision and memory switches (VERIFIED).** `bfloat16 = True` (parameters and most compute in bf16 through a haiku `bfloat16_context`, with
  parts of the structure module cast explicitly), `use_remat = True` (per-block rematerialisation), `subbatch_size = auto` (chunking over rows once
  a complex has more than 384 residues). Dropout is on during design (`design_dropout`).
- **Padding (VERIFIED).** Lengths are padded to buckets of 32 and padded residues are masked ("dead" residues), so shapes repeat across designs.
- **The numbers on this machine (VERIFIED, my micro-benchmark, fp32, N=256, pair width 128).** Raw 4096-square matmul: 11.4 TFLOP/s on both JAX
  Metal and MLX (bf16: 12.9 on both). On pair-sized operations MLX is faster: triangle contraction 0.81 vs 1.35 ms, pair projection 8.5 vs 4.3
  TFLOP/s, elementwise gating 0.39 vs 0.88 ms, layer norm 0.91 vs 1.23 ms. These are single operations, not the full model.
- **Time split (INFERRED).** The repository does not import PyRosetta; AlphaFold forward and gradient steps plus the validation refolds are most
  of a design's wall time. Not measured.
- **Expected gain (INFERRED).** About 1.5 to 2x on the AlphaFold compute. UNKNOWN until a block is measured.

## Goals

1. `bindcraft design ...` on an Apple silicon Mac can run with the MLX backend selected, producing designs through the same loop, losses, MPNN
   stage and filters as today.
2. The MLX model's forward outputs and its sequence gradient agree with the repo's own JAX path (fp32 against JAX on the CPU; bf16 within the spread
   JAX's own bf16 has against its fp32).
3. Faster than the current metaljax path, measured and reported honestly at each checkpoint (one block, the forward pass, forward plus gradient,
   a full trajectory, the validation refold set), with memory beside it.

## Non-goals

- The Neural Engine in phase 1 (see below). The loss, ProteinMPNN, the optimizer and the filters in MLX (a possible later phase). Training, new modalities,
  changing any loss or setting. Multi-GPU. Running on Intel Macs. Replacing or removing the JAX paths.

## Anti-goals (a failure even if it technically works)

- Designs that differ from the JAX path for a reason that is not numerics (a changed loss, a dropped template, a different recycle rule).
- Any silent fallback: a model family the MLX backend does not implement, a dead worker or a memory-cap hit must stop the run with a message that
  names the cause, never continue on JAX.
- Breaking or slowing the existing backends; the MLX code must be unreachable unless selected.
- Claiming parity from a single bf16 trajectory (the README records that bf16 trajectories are noisy and not reproducible between backends).

## Shape of the solution

- **A new predictor class** implementing the same protocol as `AlphaFoldDesignModel`, chosen where `campaign.py` builds the model, by an explicit
  backend setting (opt-in at first; making it the Apple default is Ethan's decision after it is verified).
- **MLX in a worker process.** The model (embeddings, template stack, extra-MSA stack, Evoformer, structure module, heads) runs in a persistent
  child process that never imports JAX; arrays cross by shared memory. This is the pattern of the earlier ports (an MLX process that has done
  other CPU-side framework work ran about a third slower in Mol-1's measurements, cured by a process boundary). The parent keeps JAX, forced onto the
  CPU platform on this backend, for the front (sequence to features), the loss and the optimizer; the GPU then belongs to MLX alone, and the metaljax
  plugin is not needed on this path. **My call, and the one to check first:** ProteinMPNN, the filters and the rest of `bindcraft/` are JAX too
  (`proteinmpnn.py`, `MPNN_stage.py`, `filters.py`), so they move to the CPU with the parent, at a cost nobody has measured. It is measured at the
  first checkpoint. A setting can leave the parent on metaljax instead (the plugin then stays a dependency and the two processes share the GPU). The cross-boundary arrays are small (logits and positions of the order of 10 MB per step at 300 residues).
- **The call contract.** `forward(model, features, dropout/rng) -> outputs` and `vjp(model, features, cotangents) -> d(features)`. JAX computes the
  loss from the outputs and its gradient with respect to the outputs (the cotangents); MLX returns the gradient with respect to the float input
  features (`msa_feat`, `target_feat`, ...), which JAX chains into the sequence logits through the front part with `jax.vjp`. Because the cotangents
  depend on the outputs, a gradient step costs a forward pass, then the vjp (which recomputes the forward inside, as the rematerialising JAX path does):
  one extra forward per step (INFERRED, about a quarter of a step). Moving the loss into MLX would remove it; that is the larger scope Ethan
  declined for now.
- **Weights.** A converter turns the haiku `.npz` parameters (with the triangle projections fused as upstream's `fuse=True` does) into MLX files,
  bit for bit, with a manifest that accounts for every tensor, as in the earlier ports. AlphaFold parameters are read from the existing locations
  (`~/.cache/bindcraft/alphafold` has the multimer v3 files, `~/.cache/colabfold/params` the pTM files; `BINDCRAFT_AF2_PARAMS` overrides).
- **Numerics.** Mirror what JAX does: bf16 parameters and compute where the haiku context applies it, fp32 where the structure module casts
  explicitly, fp32 for softmaxes and norms exactly where the JAX code has them. Rematerialisation per block with `mx.checkpoint`, row chunking with
  the same `subbatch_size` rule. Dropout uses MLX's RNG (not bit-identical to JAX's; parity is checked with dropout off).
- **Families (VERIFIED in `settings.py`, `campaign.py`).** Design draws only from the multimer pool and `sequence_gradients` is called only on design
  models, so multimer v3 is the only family that needs a vjp. The monomer pTM family is forward-only (validation) but is needed for any default run:
  with no scaffold and one copy, `resolve_validation_model` picks the monomer pool (`settings.py:590-593`). Order (my call): multimer first, with
  `trajectory_only` or `validation_model: multimer` as the first end-to-end milestone; the monomer forward next, which makes the default run end to end.
  A family not yet implemented is refused by name. Whether validation runs with dropout off is read from `campaign.py` by the builder (not checked here).
- **What the predictor must reproduce (VERIFIED in code).** (1) The JAX path casts predicted atom positions to float16 before the loss
  (`af2.py:281,343`); that cast stays in the JAX tail. (2) The campaign sets `provides_distogram`, `num_recycle`, `target_flexibility` and
  `bigbang_initialization` on the model after construction (`campaign.py:61-75`); each must change what the MLX path does, as the JAX path keys its compile
  cache on them. (3) `sequence_gradients(..., compile_only=True)` is called from a daemon thread to precompile the next length bucket
  (`campaign.py:101-110`); on MLX it must be a warm-up or queue behind the worker's other requests, never a concurrent second request.
- **Failure behaviour.** A dead worker or a cap hit raises an error that names the cause and the cap, exits non-zero and leaves no shared memory behind;
  the run is never continued on another backend.
- **Operations.** The shared GPU lock for the whole run, the preflight (disk, heavy processes), and a worker memory cap, as in the earlier ports.

## Neural Engine

Ethan asked for it "if possible", twice. What I found (this machine is an M3 Max on macOS 27.0.1):

**Three ways to reach it.**
1. **Core ML**, the long-standing public route (coremltools, compute units `CPU_AND_NE`). Inference; fp16. Known platform behaviour, not run here.
2. **Core AI**, Apple's successor. VERIFIED present: `CoreAI.framework` is in the macOS 27 SDK and on the system. Its interface names the Neural Engine as a compute
   unit (`ComputeUnitKind.neuralEngine`, `SpecializationOptions(preferredComputeUnitKind:)`) and has a graph compiler and an inference runtime with a
   Metal-backed compute stream. VERIFIED: none of its 2,377 lines of exported symbols contains gradient, backward, autodiff or train. UNKNOWN: its operation
   coverage, its precision, and whether a hand-written backward graph compiles for the Neural Engine.
3. **Private interfaces.** VERIFIED present: `AppleNeuralEngine.framework` and `ANECompiler.framework`. REPORTED, not run by me, in the open-source project
   `maderix/ANE` (its own README): it compiles hand-written MIL graphs in memory through `_ANEClient`/`_ANECompiler` and runs forward and activation-gradient
   (dx) passes on the Neural Engine, with weight gradients, the optimizer and many elementwise operations on the CPU; models up to 0.6B parameters at 412 ms/step.
   Its own stated limits: tested on M4 only (not M3), macOS 15 or later, 5 to 9% of peak utilization, a compiler leak that forces a restart after about
   119 compiles, fp16 backward passes underflow without loss scaling, attention masks ignored on-chip, undocumented and may break on any macOS update,
   and "does not replace GPU training beyond small research models".

**Correction to what I said earlier.** "Gradients cannot run on the Neural Engine" was too strong. "Inference only" describes Core ML's policy, not the
chip. A backward pass is another graph of matmuls and elementwise operations, and someone has run that on the chip, on an M4, through private
interfaces, for small transformers.

**Why it is still not phase 1 (INFERRED, not measured).**
- Speed: the reported 5 to 9% utilization against GPU matmuls at 93 to 98% of the measured ceiling on this machine means the Neural Engine is unlikely to beat
  the GPU on AlphaFold's matmuls. The case for it is concurrency, and both units share the same memory bandwidth, which AlphaFold's pair activations lean on.
- AlphaFold specifics: 48 blocks, forward and backward, times the length buckets is a lot of compiles against a limit of about 119 per process (a disk cache
  may help; unmeasured). Triangle attention, invariant point attention and the gathers may fall back to the CPU.
- fp16 only for the gradient, with underflow risk; AlphaFold's pair activations and attention logits are where fp16 is most likely to hurt.
- The private route is untested on an M3 Max and can break with an OS update.

**Spike B (private route) was run on 2026-10-30; Spike A (public route) has not been.** Throwaway harness in the session scratchpad, not in this repo; the
`maderix/ANE` bridge was built from source, patched only to print errors and time an in-C loop. Every figure below is MEASURED on this M3 Max (macOS 27.0.1),
fp16 on the Neural Engine against MLX fp16 on the GPU (bf16 is the same to within 3%), same op and shape, GPU lock held. Each Neural Engine result was checked
live and against a numpy fp32 reference using the same fp16-rounded weights. The project's own peak probe was not trusted: it never checks eval success or output,
and its flat 0.45 ms per eval is a dispatch floor.

- **It works.** The private path compiles, loads and runs on an M3 Max (the project documents M4 only). Relative error against fp32 is about 3e-4 for 1x1
  convolutions and the contraction, 3e-3 for layer norm plus gate, 7e-3 through 16 chained layers, on random unit-scale data (not AlphaFold activations).
- **Dispatch floor:** about 0.27 ms per eval. Marginal throughput once launches are amortized: about 6.7 TFLOP/s on 1x1 convolutions.
- **Kernel time at N=256 (ms), Neural Engine with fp16 buffers in and out against MLX fp16 on the GPU:**

  | op | Neural Engine | GPU | GPU faster |
  |---|---|---|---|
  | transition, 128 to 512 to 128, ReLU | 2.84 | 1.86 | 1.5x |
  | 1x1 projection, 128 to 128 | 0.59 | 0.37 | 1.6x |
  | backward-style conv, 512 to 128 | 1.69 | 0.88 | 1.9x |
  | triangle contraction, batch 128 | 1.56 | 0.55 | 2.8x |
  | layer norm, 2 projections, sigmoid gate (fp32 buffers, not rerun) | 4.02 | 1.37 | 2.9x |

  **Correction.** The first run fed the Neural Engine fp32 buffers (twice the bytes of the GPU's fp16) and overstated the gap on the low-arithmetic ops: projection
  3.0x, backward-style conv 3.1x, contraction 3.9x. The compute-bound transition does not change with buffer type (1.5x either way), which also says the Neural
  Engine's dense compute is about 6.0 TFLOP/s against the GPU's 9.2 on this op. The fp16 rerun ran while another job held the GPU, so the Neural Engine figures in
  it may be a few percent pessimistic (the fp32 rerun under the same load was within 12% of the clean first run); the GPU figures are from a clean run. Counting
  the host-side copy of every buffer, as this harness does, the Neural Engine was 2x to 9x slower; a design that keeps tensors on the chip across a whole block would
  not pay that. The ordering held at N=192 and N=384 in the fp32 runs.
- **Concurrency (8 s runs, one repeat each, same op on both units, so noise of a few percent is possible; the Neural Engine ran with fp32 buffers, and a fp16 rerun
  is pending because the GPU lock was taken, so the contraction figure may move; the compute-bound transition figure does not depend on buffer type):**
  - transition: the Neural Engine kept 99% of its solo rate and the GPU 95%. Together 860 ops/s against 532 for the GPU alone, up to 1.6x for work that can be
    split across the two.
  - contraction (memory-bound): the Neural Engine kept 96% and the GPU fell to 78%. Together 1855 ops/s against 1809, about 1.0x: the units compete for memory.
  - INFERRED: a real block mixes both kinds, so the gain from concurrency lies between 1.0x and 1.6x, and only for work that splits into whole independent
    programs, such as another model of the ensemble or validation refolds.
- **Limits found:** each tensor row must be a multiple of 64 bytes (N=150 and 250 failed with "IOSurface smaller than the model expects"; N a multiple of 16 at
  fp32 passes, and the repo's 32-residue length buckets satisfy it); about 125 programs can be loaded at once ("no ANE resources" at 126), while 220 sequential
  compile, load, free cycles succeeded and the temp directory did not grow; fp16 only (the repo's AlphaFold runs in bf16, whose range is wider, so fp16 overflow on real
  pair activations is an open risk); a hand-spelled layer norm did not compile but the native `layer_norm` op did. NOT TESTED: softmax attention with bias, invariant
  point attention, gathers, a whole Evoformer block, and the cost of moving tensors between MLX and the Neural Engine without the copies used here.
- **Not built:** a full Evoformer block, forward and backward, on the Neural Engine. The kernels lose on speed, so it is a decision for Ethan whether it is worth the
  effort at all.

**What the answer could unlock (phase 2 candidates, not started):** validation refolds on the Neural Engine (forward only) while the GPU runs the next
design's gradient steps; or, only if Spike B is good, one slice of the design gradient on it.

## Verification

- **Reference:** the repo's own JAX path, on the CPU in fp32 for parity and on metaljax in bf16 for "today". Captures of real inputs and
  outputs on the CPU (as in the RFD4 port), per stage: embeddings, one Evoformer block, the Evoformer stack, the structure module, the heads, the full
  forward, and the sequence gradient.
- **Gates (fp32, MLX model with `bfloat16 = False`):** per-stage relative error 1e-5 where the stage is a plain composition of matmuls, norms and
  gates; composite (full forward outputs, positions) 1e-3; sequence gradient cosine at least 0.999 against JAX on the same inputs (the README's own
  JAX Metal against CPU figure is 0.9997). Each reference is checked live (never all zeros) and every weight is accounted for.
- **bf16 (the default):** judged against JAX's own spread (JAX bf16 against JAX fp32 on identical inputs): the MLX bf16 outputs and gradient must
  lie within that spread. Trajectory-level checks compare distributions over several seeds (final design loss, pLDDT, ipTM), not single runs.
- **Tests that name the failure modes:** padded and dead residues give identical results at valid positions; dropout off is deterministic; a worker
  that is killed or hits its cap fails loudly with the cap named; every model family and recycle count in the pool; multi-state designs (complex plus
  binder alone); chunking above 384 residues equals the unchunked result.
- **Measured at checkpoints, reported, never a stop:** one block; the model forward at 150, 250 and 400 residues; forward plus gradient; one full
  trajectory; the validation refold set; each beside the current metaljax path (warm, second run in a process, 10% noise floor) with peak memory.

## Risks and unknowns

1. Backward through 48 Evoformer blocks with MLX autograd and `mx.checkpoint` at 250 to 400 residues: time and memory unproven (UNKNOWN).
2. The extra forward per step may eat a fifth to a quarter of the gain (INFERRED).
3. The structure module and the all-atom code are many small operations; MLX could be slower than JAX's jit there until they are compiled or fused
   (UNKNOWN).
4. bf16 fidelity: XLA and MLX may accumulate reductions differently; the yardstick above is how this is judged, and exact agreement is not promised.
5. The template stack for the fixed target and the optional `initial_atom_pos` inputs must both be ported (they are used in design).
6. The monomer family doubles part of the structure-module work.
7. The time share of AlphaFold in a full design is not measured; if it is smaller than assumed the end-to-end gain shrinks.
8. Moving the parent's JAX to the CPU slows ProteinMPNN and the filters by an unmeasured amount (see the parent-device call).

## Left to the builder

The module layout (for example `bindcraft/af/mlx/`), the worker protocol details, the converter format, test organisation, whether and where to use
`mx.compile`, and kernel questions (a hand-written Metal kernel only under the earlier rule: an operation class at least a quarter of the time and
not already near the GPU's matmul ceiling).

## Calls the author made (for Ethan to check)

1. The parent's JAX runs on the CPU on this backend, with a setting to keep it on metaljax, decided by a measurement at the first checkpoint.
2. Multimer first, monomer forward second; `trajectory_only` or `validation_model: multimer` is the first end-to-end milestone.
3. The Neural Engine stays out of phase 1. Spike B (private route) is done and shows the GPU 1.5x to 2.8x faster on every op tested (fp16 buffers), with a concurrency upside of at most 1.6x; Spike A (public route) is not run.
4. The work lives on branch `mlx-backend` in `~/BindCraft2`, off `apple-silicon-metal`, committed locally, nothing pushed.
5. The extra forward pass per gradient step is accepted as the price of keeping the loss in JAX.
